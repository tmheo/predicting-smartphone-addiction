"""Contextualized deep univariate Transformer 구현. (#149, #287)

이 파일은 Kaggle 공개 노트북 "Contextualized Deep Univariate Spline
Transformer" 판본 3의 구조를 참고해 크게 수정한 파생 구현이다.
원문: https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3
원문 소스 SHA-256: c308b69cfeabad223a1e147fa174f78d1ddaccc09991b2075eecaf757f4781a2

이슈 #287의 다층 출력 경로는 Kaggle 공개 노트북 "Multi-Level Deep
Univariate Spline Transformer" 판본 5의 1단계 구조를 참고했다.
원문: https://www.kaggle.com/code/ern711/multi-level-deep-univariate-spline-transformer?scriptVersionId=342998269
원문 소스 SHA-256: 3ae3e15d975892f90875d57e3877ba299ea073c5e019295b58870a09a35793fd

원 공개 노트북 소스에는 Apache License 2.0이 적용된다.
이 저장소의 변경 사항은 다음과 같다.

- 커밋된 outer fold와 ``ModelAdapter`` 경계를 사용한다.
- 표준화, knot와 정확값 어휘를 outer 학습 부분에서만 맞춘다.
- 검증·시험의 미등록값과 결측값에 서로 다른 식별자를 쓴다.
- 목표·빈도 인코딩을 제거하고 exp067 피처 계획을 그대로 받는다.
- M0 조각선형 경로와 매개변수 규모를 맞춘 A0 주기 경로를 한 구현에서 제공한다.
- 결정적 검증 permutation importance와 공통 진입 진단 관측값을 제공한다.
- ``output_path=multilevel_stage1``에서 ADD, UNI, ATTN, FINAL 머리와 행별
  혼합기를 선택하며, 2단계 hypernetwork는 포함하지 않는다.
- 다층 머리는 공유 기반 모형 뒤에 초기화해 ``direct_final`` 대조군과 공유
  매개변수의 초기값 및 자료 순서를 같게 유지한다.

Apache License 2.0 원문은 ``contextualized_spline_transformer.LICENSE``에 있다.
"""

from __future__ import annotations

import contextlib
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from .model import AdapterDiagnostics
from .paired_training_length import pop_parent_balanced_exposure

_NA_ID = 0
_MODES = {"spline", "periodic"}
_OPTIMIZERS = {"adamw", "muon"}
_OUTPUT_PATHS = {"direct_final", "multilevel_stage1"}
_AUX_HEAD_HIDDEN = 128
_AUX_HEAD_DROPOUT = 0.05
_MIXER_HIDDEN = 16
_MIXER_FINAL_BIAS = 2.0
_UNI_AUX_WEIGHT = 0.1
_ATTN_AUX_WEIGHT = 0.1
_MIXER_ENTROPY_WEIGHT = 1e-5

_RAW_CAPACITY = {
    "age": (48, 2),
    "daily_screen_time_hours": (96, 4),
    "social_media_hours": (96, 4),
    "gaming_hours": (80, 3),
    "work_study_hours": (96, 4),
    "sleep_hours": (80, 3),
    "notifications_per_day": (64, 3),
    "app_opens_per_day": (64, 3),
    "weekend_screen_time": (96, 4),
}

_RAW_SPLINE_SPECS = {
    "age": [(8, 16), (16, 24), (19, 32)],
    "daily_screen_time_hours": [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)],
    "social_media_hours": [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)],
    "gaming_hours": [(32, 16), (64, 16), (128, 32), (256, 64)],
    "work_study_hours": [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)],
    "sleep_hours": [(32, 16), (64, 16), (128, 32), (256, 64)],
    "notifications_per_day": [(32, 16), (64, 16), (128, 32), (232, 64)],
    "app_opens_per_day": [(32, 16), (64, 16), (128, 32), (167, 64)],
    "weekend_screen_time": [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)],
}


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))


def _unique_quantile_knots(values: np.ndarray, requested: int) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    unique = np.unique(values)
    if len(unique) == 1:
        value = float(unique[0])
        return np.array([value - 1e-5, value + 1e-5], dtype="float32")
    quantiles = np.linspace(0.0, 1.0, min(requested, len(unique)))
    knots = np.unique(np.quantile(values, quantiles).astype("float32"))
    return knots if len(knots) >= 2 else unique.astype("float32")


def _adaptive_specs(values: np.ndarray) -> list[tuple[int, int]]:
    cardinality = len(np.unique(values))
    if cardinality <= 2:
        return [(2, 16)]
    if cardinality <= 4:
        return [(4, 16)]
    if cardinality <= 16:
        return [(4, 16), (8, 16), (16, 24)]
    if cardinality <= 64:
        return [(8, 16), (16, 16), (32, 24), (64, 32)]
    return [(16, 16), (32, 16), (64, 32), (128, 64)]


def _embedding_dim(cardinality: int) -> int:
    return int(min(32, max(4, round(1.6 * cardinality**0.5))))


class _DynamicGate(nn.Module):
    def __init__(self, outputs: int, hidden: int, initial_scale: float = 0.1) -> None:
        super().__init__()
        self.global_logits = nn.Parameter(torch.zeros(outputs))
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, outputs)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale_raw = nn.Parameter(torch.tensor(float(np.arctanh(initial_scale))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.global_logits.unsqueeze(0) + torch.tanh(
            self.scale_raw
        ) * self.net(x.unsqueeze(1))
        return torch.softmax(logits, dim=1)


class _PiecewiseLinear(nn.Module):
    def __init__(self, knots: torch.Tensor, output_dim: int) -> None:
        super().__init__()
        self.register_buffer("knots", knots.clone())
        self.values = nn.Parameter(torch.randn(len(knots), output_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.knots[0], max=self.knots[-1])
        right = torch.searchsorted(self.knots, x, right=True).clamp(
            1, len(self.knots) - 1
        )
        left = right - 1
        x0, x1 = self.knots[left], self.knots[right]
        alpha = (x - x0) / (x1 - x0).clamp_min(1e-8)
        return self.values[left] + alpha.unsqueeze(1) * (
            self.values[right] - self.values[left]
        )


class _PeriodicLinear(nn.Module):
    """Lookup-Transformer의 PLR을 knot당 매개변수 규모에 맞춘 주기 표현."""

    def __init__(self, basis_size: int, output_dim: int) -> None:
        super().__init__()
        frequencies = max(1, math.ceil(basis_size / 2))
        self.frequency = nn.Parameter(torch.randn(frequencies) * 0.5)
        self.weight = nn.Parameter(
            torch.randn(2 * frequencies, output_dim) / math.sqrt(2 * frequencies)
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2 * math.pi * x.unsqueeze(1) * self.frequency.unsqueeze(0)
        basis = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return basis @ self.weight + self.bias


class _MultiResolutionExpert(nn.Module):
    def __init__(
        self,
        mode: str,
        knots: list[torch.Tensor],
        specs: list[tuple[int, int]],
        output_dim: int,
        gate_hidden: int,
    ) -> None:
        super().__init__()
        paths: list[nn.Module] = []
        for feature_knots, (_, embedding_dim) in zip(knots, specs, strict=True):
            if mode == "spline":
                paths.append(_PiecewiseLinear(feature_knots, embedding_dim))
            else:
                paths.append(_PeriodicLinear(len(feature_knots), embedding_dim))
        self.paths = nn.ModuleList(paths)
        self.projections = nn.ModuleList(
            [nn.Linear(spec[1], output_dim) for spec in specs]
        )
        self.gate = _DynamicGate(len(specs), gate_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [
                projection(path(x))
                for path, projection in zip(self.paths, self.projections)
            ],
            dim=1,
        )
        return (tokens * self.gate(x).unsqueeze(2)).sum(dim=1)


class _TinyMLP(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        hidden = max(16, output_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))


class _RawPath(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, output_dim)
        self.gate = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar = x.unsqueeze(1)
        return self.projection(scalar) * torch.sigmoid(self.gate(scalar))


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(self.value(x) * F.silu(self.gate(x))))


class _LocalBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = _SwiGLU(dim, max(16, round(dim * 2.0)), dropout)
        self.scale_raw = nn.Parameter(torch.tensor(float(np.arctanh(0.2))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.scale_raw) * self.ff(self.norm(x))


class _FeatureEncoder(nn.Module):
    def __init__(
        self,
        mode: str,
        knots: list[torch.Tensor],
        specs: list[tuple[int, int]],
        coarse_knots: torch.Tensor,
        width: int,
        depth: int,
        token_dim: int,
        gate_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.multi = _MultiResolutionExpert(mode, knots, specs, width, gate_hidden)
        coarse_dim = min(32, width)
        self.coarse = nn.Sequential(
            _PiecewiseLinear(coarse_knots, coarse_dim), nn.Linear(coarse_dim, width)
        )
        self.tiny = _TinyMLP(width)
        self.raw = _RawPath(width)
        self.expert_gate = _DynamicGate(4, gate_hidden)
        self.local = nn.ModuleList([_LocalBlock(width, dropout) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        experts = torch.stack(
            [self.multi(x), self.coarse(x), self.tiny(x), self.raw(x)], dim=1
        )
        token = (experts * self.expert_gate(x).unsqueeze(2)).sum(dim=1)
        for block in self.local:
            token = block(token)
        return self.output(token)


class _PreContext(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.output = nn.Linear(hidden, features)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.scale_raw = nn.Parameter(torch.full((features,), float(np.arctanh(0.02))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.output(F.silu(self.input(x))) * torch.tanh(
            self.scale_raw
        ).unsqueeze(0)
        return x + correction


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.attention_scale = nn.Parameter(torch.tensor(0.2))
        self.norm2 = nn.LayerNorm(dim)
        self.ff = _SwiGLU(dim, dim * 2, dropout)
        self.ff_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + torch.tanh(self.attention_scale) * attended
        return x + torch.tanh(self.ff_scale) * self.ff(self.norm2(x))


class _PooledTokenHead(nn.Module):
    """token 묶음의 원소별 평균과 최댓값으로 한 logit을 만든다."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * token_dim),
            nn.Linear(2 * token_dim, _AUX_HEAD_HIDDEN),
            nn.SiLU(),
            nn.Dropout(_AUX_HEAD_DROPOUT),
            nn.Linear(_AUX_HEAD_HIDDEN, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [tokens.mean(dim=1), tokens.max(dim=1).values],
            dim=1,
        )
        return self.net(pooled).squeeze(1)


class _MultiLogitMixer(nn.Module):
    """네 logit과 쌍별 차이로 행마다 다른 볼록 혼합 가중치를 만든다."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, _MIXER_HIDDEN),
            nn.SiLU(),
            nn.Linear(_MIXER_HIDDEN, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[3] = _MIXER_FINAL_BIAS

    def forward(
        self,
        additive_logit: torch.Tensor,
        univariate_logit: torch.Tensor,
        attention_logit: torch.Tensor,
        final_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.stack(
            [additive_logit, univariate_logit, attention_logit, final_logit],
            dim=1,
        )
        differences = torch.stack(
            [
                (additive_logit - univariate_logit).abs(),
                (additive_logit - attention_logit).abs(),
                (additive_logit - final_logit).abs(),
                (univariate_logit - attention_logit).abs(),
                (univariate_logit - final_logit).abs(),
                (attention_logit - final_logit).abs(),
            ],
            dim=1,
        )
        weights = torch.softmax(
            self.net(torch.cat([logits, differences], dim=1)), dim=1
        )
        return (weights * logits).sum(dim=1), weights


class _ContextualizedModel(nn.Module):
    def __init__(
        self,
        mode: str,
        numerical_columns: list[str],
        exact_sizes: list[int],
        feature_knots: list[list[torch.Tensor]],
        feature_specs: list[list[tuple[int, int]]],
        coarse_knots: list[torch.Tensor],
        capacities: list[tuple[int, int]],
        *,
        token_dim: int,
        attention_dim: int,
        attention_heads: int,
        context_hidden: int,
        gate_hidden: int,
        residual_hidden: int,
        dropout: float,
        output_path: str,
    ) -> None:
        super().__init__()
        n_num = len(numerical_columns)
        self.pre_context = _PreContext(n_num, context_hidden)
        self.encoders = nn.ModuleList(
            [
                _FeatureEncoder(
                    mode,
                    feature_knots[index],
                    feature_specs[index],
                    coarse_knots[index],
                    capacities[index][0],
                    capacities[index][1],
                    token_dim,
                    gate_hidden,
                    dropout,
                )
                for index in range(n_num)
            ]
        )
        self.additive_heads = nn.ModuleList(
            [nn.Linear(token_dim, 1) for _ in range(n_num)]
        )
        self.additive_bias = nn.Parameter(torch.zeros(1))
        self.interaction_projection = nn.Linear(token_dim, attention_dim)
        self.feature_identity = nn.Parameter(
            torch.randn(1, n_num, attention_dim) * 0.02
        )
        self.interaction = _AttentionBlock(attention_dim, attention_heads, dropout)
        self.interaction_norm = nn.LayerNorm(attention_dim)

        self.exact_embeddings = nn.ModuleList()
        exact_dim = 0
        for size in exact_sizes:
            dim = _embedding_dim(size)
            self.exact_embeddings.append(nn.Embedding(size, dim))
            exact_dim += dim

        final_dim = (
            n_num + n_num * token_dim + n_num * attention_dim + exact_dim + n_num + 1
        )
        self.final_norm = nn.LayerNorm(final_dim)
        self.final_head = nn.Sequential(
            nn.Linear(final_dim, residual_hidden),
            nn.LayerNorm(residual_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, residual_hidden // 2),
            nn.LayerNorm(residual_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden // 2, 1),
        )
        self.output_path = output_path
        if output_path == "multilevel_stage1":
            self.univariate_head: _PooledTokenHead | None = _PooledTokenHead(token_dim)
            self.attention_head: _PooledTokenHead | None = _PooledTokenHead(
                attention_dim
            )
            self.logit_mixer: _MultiLogitMixer | None = _MultiLogitMixer()
        else:
            self.univariate_head = None
            self.attention_head = None
            self.logit_mixer = None

    def forward(
        self, x_num: torch.Tensor, x_exact: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        contextualized = self.pre_context(x_num)
        tokens = torch.stack(
            [encoder(contextualized[:, i]) for i, encoder in enumerate(self.encoders)],
            dim=1,
        )
        feature_scores = torch.stack(
            [
                head(tokens[:, i]).squeeze(1)
                for i, head in enumerate(self.additive_heads)
            ],
            dim=1,
        )
        additive_logit = self.additive_bias + feature_scores.sum(dim=1)
        interaction = self.interaction_norm(
            self.interaction(
                self.interaction_projection(tokens) + self.feature_identity
            )
        )
        exact = torch.cat(
            [
                embedding(x_exact[:, i])
                for i, embedding in enumerate(self.exact_embeddings)
            ],
            dim=1,
        )
        final_input = torch.cat(
            [
                x_num,
                tokens.flatten(1),
                interaction.flatten(1),
                exact,
                feature_scores,
                torch.sigmoid(additive_logit).unsqueeze(1),
            ],
            dim=1,
        )
        final_input = F.normalize(self.final_norm(final_input), p=2, dim=1)
        base_final_logit = self.final_head(final_input).squeeze(1)
        output = {
            "final_logit": base_final_logit,
            "base_final_logit": base_final_logit,
            "additive_logit": additive_logit,
        }
        if self.output_path == "multilevel_stage1":
            assert self.univariate_head is not None
            assert self.attention_head is not None
            assert self.logit_mixer is not None
            univariate_logit = self.univariate_head(tokens)
            attention_logit = self.attention_head(interaction)
            mixed_logit, mixer_weights = self.logit_mixer(
                additive_logit,
                univariate_logit,
                attention_logit,
                base_final_logit,
            )
            output.update(
                {
                    "final_logit": mixed_logit,
                    "univariate_logit": univariate_logit,
                    "attention_logit": attention_logit,
                    "mixer_weights": mixer_weights,
                }
            )
        return output


def _muon_parameter_names(model: _ContextualizedModel) -> set[str]:
    """예측 출력층을 제외한 선형 변환 행렬 이름을 반환한다."""
    final_output = next(
        module
        for module in reversed(model.final_head)
        if isinstance(module, nn.Linear)
    )
    excluded_module_ids = {
        id(final_output),
        *(id(module) for module in model.additive_heads),
    }
    selected: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and id(module) not in excluded_module_ids:
            selected.add(f"{module_name}.weight")
        if (
            isinstance(module, nn.MultiheadAttention)
            and module.in_proj_weight is not None
        ):
            selected.add(f"{module_name}.in_proj_weight")
    return selected


class ContextualizedSplineTransformerFold:
    """fold 하나의 전처리, 학습, 예측과 중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._paired_exposure = pop_parent_balanced_exposure(params)
        self._exact_cols = list(params.pop("exact_cols"))
        self._mode = str(params.pop("numeric_mode", "spline"))
        if self._mode not in _MODES:
            raise ValueError(
                f"numeric_mode는 {sorted(_MODES)} 중 하나여야 한다: {self._mode!r}"
            )
        self._optimizer_name = str(params.pop("optimizer", "adamw")).lower()
        if self._optimizer_name not in _OPTIMIZERS:
            raise ValueError(
                f"optimizer는 {sorted(_OPTIMIZERS)} 중 하나여야 한다: "
                f"{self._optimizer_name!r}"
            )
        self._output_path = str(params.pop("output_path", "direct_final"))
        if self._output_path not in _OUTPUT_PATHS:
            raise ValueError(
                f"output_path는 {sorted(_OUTPUT_PATHS)} 중 하나여야 한다: "
                f"{self._output_path!r}"
            )
        self._exact_max_card = int(params.pop("exact_max_card", 5000))
        self._token_dim = int(params.pop("token_dim", 64))
        self._attention_dim = int(params.pop("attention_dim", 64))
        self._attention_heads = int(params.pop("attention_heads", 8))
        self._default_width = int(params.pop("default_width", 64))
        self._default_depth = int(params.pop("default_depth", 3))
        self._context_hidden = int(params.pop("context_hidden", 32))
        self._gate_hidden = int(params.pop("gate_hidden", 32))
        self._residual_hidden = int(params.pop("residual_hidden", 384))
        self._epochs = int(params.pop("epochs", 35))
        self._patience = int(params.pop("patience", 7))
        self._batch_size = int(params.pop("batch_size", 4096))
        self._lr = float(params.pop("lr", 7e-4))
        self._weight_decay = float(params.pop("weight_decay", 5e-4))
        self._label_smoothing = float(params.pop("label_smoothing", 0.005))
        self._grad_clip = float(params.pop("grad_clip", 5.0))
        self._additive_weight = float(params.pop("additive_weight", 0.3))
        self._dropout = float(params.pop("dropout", 0.05))
        self._perm_repeats = int(params.pop("perm_repeats", 3))
        if params:
            raise ValueError(
                f"contextualized_spline_transformer가 모르는 params: {sorted(params)}"
            )
        if not self._exact_cols or len(set(self._exact_cols)) != len(self._exact_cols):
            raise ValueError("exact_cols는 중복 없는 비어 있지 않은 목록이어야 한다.")
        if self._attention_dim % self._attention_heads:
            raise ValueError("attention_dim은 attention_heads의 배수여야 한다.")
        if (
            min(
                self._token_dim,
                self._attention_dim,
                self._attention_heads,
                self._default_width,
                self._default_depth,
                self._context_hidden,
                self._gate_hidden,
                self._residual_hidden,
                self._epochs,
                self._patience,
                self._batch_size,
                self._perm_repeats,
            )
            <= 0
        ):
            raise ValueError("모델 크기와 학습 횟수 params는 양수여야 한다.")
        if self._residual_hidden < 2:
            raise ValueError("residual_hidden은 2 이상이어야 한다.")
        if not 0 <= self._dropout < 1:
            raise ValueError("dropout은 0 이상 1 미만이어야 한다.")
        if not 0 <= self._label_smoothing < 1:
            raise ValueError("label_smoothing은 0 이상 1 미만이어야 한다.")
        if self._lr <= 0 or self._weight_decay < 0 or self._additive_weight < 0:
            raise ValueError(
                "lr은 양수이고 weight_decay와 additive_weight는 0 이상이어야 한다."
            )
        if self._grad_clip <= 0 or self._exact_max_card < 1:
            raise ValueError(
                "grad_clip은 양수이고 exact_max_card는 1 이상이어야 한다."
            )

        self._seed = seed
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            torch.set_num_threads(1)
        self._columns: list[str] | None = None
        self._numeric_cols: list[str] = []
        self._numeric_stats: dict[str, tuple[float, float]] = {}
        self._vocabs: dict[str, list[object]] = {}
        self._model: _ContextualizedModel | None = None
        self._validation: tuple[pd.DataFrame, np.ndarray] | None = None
        self._validation_auc: float | None = None
        self._additive_auc: float | None = None
        self._trainable_parameters: int | None = None
        self._training_diagnostics: dict[str, object] | None = None
        self._raw_training_length_selection: int | None = None
        self._multilevel_diagnostics: dict[str, object] = {}

    def _seed_everything(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _values(series: pd.Series) -> pd.Series:
        return (
            series.astype(object)
            if isinstance(series.dtype, pd.CategoricalDtype)
            else series
        )

    def _fit_preprocessing(self, X: pd.DataFrame) -> None:
        self._columns = list(X.columns)
        missing = [column for column in self._exact_cols if column not in self._columns]
        if missing:
            raise ValueError(f"exact_cols가 입력에 없다: {missing}")
        self._numeric_cols = [
            column
            for column in self._columns
            if pd.api.types.is_numeric_dtype(X[column])
        ]
        non_numeric_without_exact = [
            column
            for column in self._columns
            if column not in self._numeric_cols and column not in self._exact_cols
        ]
        if non_numeric_without_exact:
            raise ValueError(
                "수치가 아닌 입력은 exact_cols에 있어야 한다: "
                f"{non_numeric_without_exact}"
            )
        if not self._numeric_cols:
            raise ValueError("수치 입력 열이 하나 이상 필요하다.")

        for column in self._numeric_cols:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype="float64")
            observed = values[np.isfinite(values)]
            if not len(observed):
                raise ValueError(f"학습 fold의 {column}에 유한한 수치가 없다.")
            mean = float(observed.mean())
            std = float(observed.std(ddof=0))
            self._numeric_stats[column] = (mean, std if std > 0 else 1.0)

        for column in self._exact_cols:
            values = self._values(X[column]).dropna()
            vocab = sorted(pd.unique(values).tolist(), key=lambda value: repr(value))
            if len(vocab) > self._exact_max_card:
                raise ValueError(
                    f"exact 컬럼 {column}의 학습 fold 카디널리티 {len(vocab)}이 "
                    f"exact_max_card {self._exact_max_card}를 넘는다."
                )
            self._vocabs[column] = vocab

    def _encode(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if list(X.columns) != self._columns:
            raise AssertionError("인코딩 입력 컬럼이 학습 때와 다르다.")
        numerical = np.zeros((len(X), len(self._numeric_cols)), dtype="float32")
        for index, column in enumerate(self._numeric_cols):
            mean, std = self._numeric_stats[column]
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype="float64")
            standardized = (values - mean) / std
            numerical[:, index] = np.nan_to_num(
                standardized, nan=0.0, posinf=0.0, neginf=0.0
            ).astype("float32")

        exact = np.zeros((len(X), len(self._exact_cols)), dtype="int64")
        for index, column in enumerate(self._exact_cols):
            values = self._values(X[column])
            vocab = self._vocabs[column]
            mapping = {
                value: value_index + 1 for value_index, value in enumerate(vocab)
            }
            unknown_id = len(vocab) + 1
            ids = (
                values.map(mapping)
                .fillna(unknown_id)
                .to_numpy(dtype="int64", copy=True)
            )
            ids[values.isna().to_numpy()] = _NA_ID
            exact[:, index] = ids
        return torch.from_numpy(numerical), torch.from_numpy(exact)

    def _build_model(self, numeric_train: np.ndarray) -> _ContextualizedModel:
        feature_specs: list[list[tuple[int, int]]] = []
        feature_knots: list[list[torch.Tensor]] = []
        coarse_knots: list[torch.Tensor] = []
        capacities: list[tuple[int, int]] = []
        for index, column in enumerate(self._numeric_cols):
            values = numeric_train[:, index]
            specs = _RAW_SPLINE_SPECS.get(column, _adaptive_specs(values))
            feature_specs.append(specs)
            feature_knots.append(
                [
                    torch.tensor(_unique_quantile_knots(values, requested))
                    for requested, _ in specs
                ]
            )
            coarse_count = min(32, max(4, len(np.unique(values))))
            coarse_knots.append(
                torch.tensor(_unique_quantile_knots(values, coarse_count))
            )
            capacities.append(
                _RAW_CAPACITY.get(column, (self._default_width, self._default_depth))
            )
        exact_sizes = [len(self._vocabs[column]) + 2 for column in self._exact_cols]
        return _ContextualizedModel(
            self._mode,
            self._numeric_cols,
            exact_sizes,
            feature_knots,
            feature_specs,
            coarse_knots,
            capacities,
            token_dim=self._token_dim,
            attention_dim=self._attention_dim,
            attention_heads=self._attention_heads,
            context_hidden=self._context_hidden,
            gate_hidden=self._gate_hidden,
            residual_hidden=self._residual_hidden,
            dropout=self._dropout,
            output_path=self._output_path,
        )

    def _initialize_training(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[
        _ContextualizedModel,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.optim.Optimizer,
        torch.Generator,
    ]:
        self._seed_everything()
        if self._paired_exposure is not None:
            self._paired_exposure.validate_training_row_count(len(X))
            preprocessing_frame = X.iloc[
                : self._paired_exposure.original_row_count
            ]
        else:
            preprocessing_frame = X
        self._fit_preprocessing(preprocessing_frame)
        numeric_cpu, exact_cpu = self._encode(X)
        model = self._build_model(
            numeric_cpu[: len(preprocessing_frame)].numpy()
        ).to(self._device)
        self._model = model
        self._trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"[contextualized_spline_transformer] mode={self._mode} "
            f"optimizer={self._optimizer_name} "
            f"output_path={self._output_path} "
            f"parameters={self._trainable_parameters:,}",
            flush=True,
        )

        x_num = numeric_cpu.to(self._device)
        x_exact = exact_cpu.to(self._device)
        target = torch.from_numpy(y.to_numpy(dtype="float32")).to(self._device)
        soft_target = (
            target * (1.0 - self._label_smoothing) + 0.5 * self._label_smoothing
        )
        if self._optimizer_name == "muon":
            from .muon import MuonWithAdamW, hybrid_parameter_groups

            names = _muon_parameter_names(model)
            named_parameters = dict(model.named_parameters())
            parameter_groups = hybrid_parameter_groups(
                [
                    {
                        "params": list(model.parameters()),
                        "weight_decay": self._weight_decay,
                    }
                ],
                [named_parameters[name] for name in sorted(names)],
            )
            optimizer = MuonWithAdamW(
                parameter_groups,
                lr=self._lr,
                weight_decay=self._weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self._lr, weight_decay=self._weight_decay
            )
        generator = torch.Generator(device=self._device).manual_seed(self._seed)
        return model, x_num, x_exact, soft_target, optimizer, generator

    def _train_epoch(
        self,
        model: _ContextualizedModel,
        x_num: torch.Tensor,
        x_exact: torch.Tensor,
        soft_target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
        epoch_index: int,
    ) -> float:
        model.train()
        parent_rows = (
            self._paired_exposure.original_row_count
            if self._paired_exposure is not None
            else len(x_num)
        )
        permutation = torch.randperm(
            parent_rows, generator=generator, device=self._device
        )
        if self._paired_exposure is not None:
            replica = (
                permutation + epoch_index
            ) % self._paired_exposure.row_multiplier
            permutation = permutation + replica * parent_rows
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), self._batch_size):
            rows = permutation[start : start + self._batch_size]
            output = model(x_num[rows], x_exact[rows])
            final_loss = F.binary_cross_entropy_with_logits(
                output["final_logit"], soft_target[rows]
            )
            additive_loss = F.binary_cross_entropy_with_logits(
                output["additive_logit"], soft_target[rows]
            )
            loss = final_loss + self._additive_weight * additive_loss
            if self._output_path == "multilevel_stage1":
                univariate_loss = F.binary_cross_entropy_with_logits(
                    output["univariate_logit"], soft_target[rows]
                )
                attention_loss = F.binary_cross_entropy_with_logits(
                    output["attention_logit"], soft_target[rows]
                )
                mixer_weights = output["mixer_weights"]
                mixer_entropy = -(
                    mixer_weights * torch.log(mixer_weights.clamp_min(1e-8))
                ).sum(dim=1).mean()
                loss = (
                    loss
                    + _UNI_AUX_WEIGHT * univariate_loss
                    + _ATTN_AUX_WEIGHT * attention_loss
                    + _MIXER_ENTROPY_WEIGHT * mixer_entropy
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        return epoch_loss / batches

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        model, x_num, x_exact, soft_target, optimizer, generator = (
            self._initialize_training(X_tr, y_tr)
        )
        numeric_va_cpu, exact_va_cpu = self._encode(X_va)
        x_num_va = numeric_va_cpu.to(self._device)
        x_exact_va = exact_va_cpu.to(self._device)
        y_va_array = y_va.to_numpy(dtype="float64")

        best_auc = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        best_additive_auc = -np.inf
        best_epoch: int | None = None
        stale = 0
        end_epoch = 0
        for epoch in range(1, self._epochs + 1):
            end_epoch = epoch
            epoch_loss = self._train_epoch(
                model, x_num, x_exact, soft_target, optimizer, generator, epoch - 1
            )
            final_logit, additive_logit = self._predict_tensors(x_num_va, x_exact_va)
            final_auc = float(roc_auc_score(y_va_array, final_logit))
            additive_auc = float(roc_auc_score(y_va_array, additive_logit))
            print(
                f"[contextualized_spline_transformer] mode={self._mode} "
                f"epoch={epoch:02d} loss={epoch_loss:.6f} "
                f"add_auc={additive_auc:.6f} final_auc={final_auc:.6f}",
                flush=True,
            )
            if final_auc > best_auc:
                best_auc = final_auc
                best_additive_auc = additive_auc
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= self._patience:
                break

        if best_state is None or best_epoch is None:
            raise RuntimeError("검증 checkpoint를 만들지 못했다.")
        model.load_state_dict(best_state)
        validation_logit, _ = self._predict_tensors(x_num_va, x_exact_va)
        if self._output_path == "multilevel_stage1":
            self._multilevel_diagnostics = self._measure_multilevel_heads(
                x_num_va, x_exact_va, y_va_array
            )
        self._validation = (X_va.copy(), y_va_array)
        self._validation_auc = float(roc_auc_score(y_va_array, validation_logit))
        self._additive_auc = best_additive_auc
        self._training_diagnostics = {
            "initialization_seed": self._seed,
            "numeric_mode": self._mode,
            "optimizer": self._optimizer_name,
            "output_path": self._output_path,
            "configured_epochs": self._epochs,
            "end_epoch": end_epoch,
            "best_epoch": best_epoch,
            "observed_best_epoch": best_epoch,
            "best_validation_auc": float(best_auc),
            "full_fit": False,
            **self._multilevel_diagnostics,
        }
        self._raw_training_length_selection = best_epoch
        return _sigmoid(validation_logit)

    def fit_full(self, X: pd.DataFrame, y: pd.Series, epochs: int) -> None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(
                "Contextualized Spline Transformer 전체 자료 재학습 epoch 수는 "
                "양의 정수여야 한다."
            )
        model, x_num, x_exact, soft_target, optimizer, generator = (
            self._initialize_training(X, y)
        )
        for epoch in range(1, epochs + 1):
            epoch_loss = self._train_epoch(
                model, x_num, x_exact, soft_target, optimizer, generator, epoch - 1
            )
            print(
                f"[contextualized_spline_transformer] mode={self._mode} "
                f"epoch={epoch:02d} loss={epoch_loss:.6f} full_fit=true",
                flush=True,
            )
        self._training_diagnostics = {
            "initialization_seed": self._seed,
            "numeric_mode": self._mode,
            "optimizer": self._optimizer_name,
            "output_path": self._output_path,
            "configured_epochs": self._epochs,
            "end_epoch": epochs,
            "best_epoch": epochs,
            "observed_best_epoch": None,
            "best_validation_auc": None,
            "full_fit": True,
        }
        if self._paired_exposure is not None:
            self._training_diagnostics["parent_balanced_exposure"] = {
                "original_row_count": self._paired_exposure.original_row_count,
                "row_multiplier": self._paired_exposure.row_multiplier,
                "preprocessing_fit_rows": self._paired_exposure.original_row_count,
                "replica_selection": "deterministic_parent_rotation",
            }
        # 전체 자료 재학습에는 검증 선택이 없다. 없는 관측을 지어내지 않는다. (#372)
        self._raw_training_length_selection = None

    def _measure_multilevel_heads(
        self,
        numerical: torch.Tensor,
        exact: torch.Tensor,
        target: np.ndarray,
        chunk: int = 16384,
    ) -> dict[str, object]:
        if self._model is None:
            raise RuntimeError("다층 머리 진단은 fit 뒤에 호출해야 한다.")
        self._model.eval()
        collected: dict[str, list[torch.Tensor]] = {
            "base_final_logit": [],
            "univariate_logit": [],
            "attention_logit": [],
            "mixer_weights": [],
        }
        with torch.no_grad(), contextlib.nullcontext():
            for start in range(0, len(numerical), chunk):
                output = self._model(
                    numerical[start : start + chunk], exact[start : start + chunk]
                )
                for key in collected:
                    collected[key].append(output[key].float().cpu())
        arrays = {
            key: torch.cat(parts).numpy().astype("float64")
            for key, parts in collected.items()
        }
        mixer_mean = arrays["mixer_weights"].mean(axis=0)
        return {
            "best_base_final_auc": float(
                roc_auc_score(target, arrays["base_final_logit"])
            ),
            "best_univariate_auc": float(
                roc_auc_score(target, arrays["univariate_logit"])
            ),
            "best_attention_auc": float(
                roc_auc_score(target, arrays["attention_logit"])
            ),
            "best_mixer_mean_weights": [float(value) for value in mixer_mean],
        }

    def _predict_tensors(
        self, numerical: torch.Tensor, exact: torch.Tensor, chunk: int = 16384
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._model is not None
        self._model.eval()
        final_parts: list[torch.Tensor] = []
        additive_parts: list[torch.Tensor] = []
        with torch.no_grad(), contextlib.nullcontext():
            for start in range(0, len(numerical), chunk):
                output = self._model(
                    numerical[start : start + chunk], exact[start : start + chunk]
                )
                final_parts.append(output["final_logit"])
                additive_parts.append(output["additive_logit"])
        return (
            torch.cat(final_parts).float().cpu().numpy().astype("float64"),
            torch.cat(additive_parts).float().cpu().numpy().astype("float64"),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        numerical, exact = self._encode(X)
        final_logit, _ = self._predict_tensors(
            numerical.to(self._device), exact.to(self._device)
        )
        return _sigmoid(final_logit)

    def importance(self) -> pd.DataFrame:
        if self._validation is None or self._validation_auc is None:
            raise RuntimeError("importance는 fit 뒤에 호출해야 한다.")
        X_va, y_va = self._validation
        gains: list[float] = []
        for column_index, column in enumerate(self._columns or []):
            drops = []
            for repeat in range(self._perm_repeats):
                generator = np.random.default_rng(
                    self._seed * 10007 + column_index * 101 + repeat
                )
                permutation = generator.permutation(len(X_va))
                permuted = X_va.copy()
                permuted[column] = X_va[column].to_numpy()[permutation]
                drops.append(
                    self._validation_auc - roc_auc_score(y_va, self.predict(permuted))
                )
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})

    def training_diagnostics(self) -> dict[str, object]:
        if self._training_diagnostics is None:
            raise RuntimeError("training_diagnostics는 fit 뒤에 호출해야 한다.")
        return self._training_diagnostics

    def raw_training_length_selections(self) -> tuple[int, ...]:
        """검증이 고른 1부터 세는 epoch 횟수 하나. (#372)"""
        if self._raw_training_length_selection is None:
            raise RuntimeError(
                "Contextualized Spline Transformer 원시 epoch 횟수는 "
                "검증 분할로 학습한 뒤에만 읽을 수 있다."
            )
        return (self._raw_training_length_selection,)

    def entry_diagnostics(self) -> AdapterDiagnostics:
        if self._trainable_parameters is None:
            raise RuntimeError("entry_diagnostics는 fit 뒤에 호출해야 한다.")
        return AdapterDiagnostics(
            assertions={
                "preprocessing_training_rows_only": True,
                "validation_labels_excluded_from_preprocessing": True,
                "missing_and_unknown_ids_distinct": True,
            },
            observations={
                "numeric_mode": self._mode,
                "optimizer": self._optimizer_name,
                "output_path": self._output_path,
                "numerical_feature_count": len(self._numeric_cols),
                "exact_feature_count": len(self._exact_cols),
                "trainable_parameters": self._trainable_parameters,
                "best_additive_auc": float(self._additive_auc),
                **self._multilevel_diagnostics,
            },
        )
