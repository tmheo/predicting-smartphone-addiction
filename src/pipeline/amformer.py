"""AMFormer의 논문 수식을 옮긴 독립 PyTorch 구현. (#144)

이 모듈은 AAAI 2024 논문에 적힌 수치·범주 토큰화, 병렬 덧셈·곱셈 주의,
prompt query, top-k 곱셈 경로와 residual feed-forward 구조만을 근거로 작성했다.
GPL-3.0인 공식 구현의 소스나 TALENT의 파생 구현을 복사하거나 번역하지 않는다.

이 티켓의 ``target_mode=mix``는 prompt query와 같은 위치의 입력 토큰을 더해
자료 공통 상호작용과 행별 상호작용을 함께 질의하는 것으로 명시적으로 정의한다.
``token_descent=false``에서는 모든 층의 토큰 수가 같으므로 residual 경계가 유지된다.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .model import AdapterDiagnostics


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))


@dataclass(frozen=True)
class _ColumnSpec:
    name: str
    kind: str
    mean: float | None = None
    scale: float | None = None
    vocabulary: tuple[object, ...] = ()


class _FoldEncoder:
    """학습 fold에서만 수치 통계와 범주 어휘를 맞추는 42-token 입력 경계."""

    def __init__(self, missing_indicator_cols: list[str]) -> None:
        if len(set(missing_indicator_cols)) != len(missing_indicator_cols):
            raise ValueError("missing_indicator_cols에 중복이 있다.")
        self._missing_indicator_cols = list(missing_indicator_cols)
        self.columns: list[str] | None = None
        self.specs: list[_ColumnSpec] = []
        self.numeric_token_positions: list[int] = []
        self.categorical_token_positions: list[int] = []
        self.categorical_cardinalities: list[int] = []
        self.token_names: list[str] = []

    def fit(self, X: pd.DataFrame) -> None:
        self.columns = list(X.columns)
        missing = [name for name in self._missing_indicator_cols if name not in X.columns]
        if missing:
            raise ValueError(f"결측 표시 대상 열이 입력에 없다: {missing}")

        self.specs = []
        self.numeric_token_positions = []
        self.categorical_token_positions = []
        self.categorical_cardinalities = []
        self.token_names = list(self.columns) + [
            f"{name}__missing" for name in self._missing_indicator_cols
        ]
        for position, name in enumerate(self.columns):
            values = X[name]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype(object)
            if pd.api.types.is_numeric_dtype(values):
                observed = values.dropna().to_numpy(dtype="float64")
                if not len(observed):
                    raise ValueError(f"수치 열 {name}의 학습 fold에 관측값이 없다.")
                mean = float(np.mean(observed))
                scale = float(np.std(observed))
                if not np.isfinite(mean) or not np.isfinite(scale):
                    raise ValueError(f"수치 열 {name}의 학습 fold 통계가 유한하지 않다.")
                self.specs.append(
                    _ColumnSpec(
                        name=name,
                        kind="numeric",
                        mean=mean,
                        scale=max(scale, 1e-6),
                    )
                )
                self.numeric_token_positions.append(position)
            else:
                vocabulary = tuple(
                    sorted(pd.unique(values.dropna()), key=lambda value: str(value))
                )
                self.specs.append(
                    _ColumnSpec(name=name, kind="categorical", vocabulary=vocabulary)
                )
                self.categorical_token_positions.append(position)
                # 0은 결측, 마지막은 학습 fold 미등록값이다.
                self.categorical_cardinalities.append(len(vocabulary) + 2)

        start = len(self.columns)
        self.numeric_token_positions.extend(
            range(start, start + len(self._missing_indicator_cols))
        )

    @property
    def token_count(self) -> int:
        return len(self.token_names)

    def transform(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if list(X.columns) != self.columns:
            raise AssertionError("AMFormer 입력 컬럼이 학습 때와 다르다.")
        numeric: list[np.ndarray] = []
        categorical: list[np.ndarray] = []
        for spec in self.specs:
            values = X[spec.name]
            if spec.kind == "numeric":
                raw = values.to_numpy(dtype="float64")
                standardized = (
                    np.where(np.isnan(raw), spec.mean, raw) - spec.mean
                ) / spec.scale
                numeric.append(standardized.astype("float32"))
                continue
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype(object)
            mapping = {value: index + 1 for index, value in enumerate(spec.vocabulary)}
            ids = (
                values.map(mapping)
                .fillna(len(spec.vocabulary) + 1)
                .to_numpy(dtype="int64")
            )
            ids[values.isna().to_numpy()] = 0
            categorical.append(ids)

        for name in self._missing_indicator_cols:
            numeric.append(X[name].isna().to_numpy(dtype="float32"))

        numeric_array = np.column_stack(numeric).astype("float32", copy=False)
        categorical_array = (
            np.column_stack(categorical).astype("int64", copy=False)
            if categorical
            else np.empty((len(X), 0), dtype="int64")
        )
        return torch.from_numpy(numeric_array), torch.from_numpy(categorical_array)


class _ColumnTokenizer(nn.Module):
    """논문의 열별 1-in-d 수치 FC와 범주 embedding lookup."""

    def __init__(
        self,
        token_count: int,
        numeric_positions: list[int],
        categorical_positions: list[int],
        categorical_cardinalities: list[int],
        d_model: int,
    ) -> None:
        super().__init__()
        self._token_count = token_count
        self._numeric_positions = list(numeric_positions)
        self._categorical_positions = list(categorical_positions)
        self.numeric_weight = nn.Parameter(torch.empty(len(numeric_positions), d_model))
        self.numeric_bias = nn.Parameter(torch.zeros(len(numeric_positions), d_model))
        nn.init.normal_(self.numeric_weight, std=0.02)

        offsets = np.concatenate(
            [[0], np.cumsum(categorical_cardinalities, dtype="int64")[:-1]]
        )
        self.register_buffer(
            "categorical_offsets",
            torch.from_numpy(offsets.astype("int64")),
            persistent=False,
        )
        total_categories = int(sum(categorical_cardinalities))
        self.categorical_embedding = (
            nn.Embedding(total_categories, d_model) if total_categories else None
        )
        if self.categorical_embedding is not None:
            nn.init.normal_(self.categorical_embedding.weight, std=0.02)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        categorical_tokens = None
        if self.categorical_embedding is not None:
            categorical_tokens = self.categorical_embedding(
                categorical + self.categorical_offsets.unsqueeze(0)
            )
        tokens: list[torch.Tensor | None] = [None] * self._token_count
        for source, position in enumerate(self._numeric_positions):
            tokens[position] = numeric_tokens[:, source]
        for source, position in enumerate(self._categorical_positions):
            assert categorical_tokens is not None
            tokens[position] = categorical_tokens[:, source]
        if any(token is None for token in tokens):
            raise AssertionError("AMFormer token 위치가 모두 채워지지 않았다.")
        return torch.stack([token for token in tokens if token is not None], dim=1)


class _PromptArithmeticAttention(nn.Module):
    """논문 식 (2)-(4)의 prompt형 덧셈·곱셈 병렬 주의."""

    def __init__(
        self,
        token_count: int,
        d_model: int,
        heads: int,
        prod_top_k: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model {d_model}은 heads {heads}의 배수여야 한다.")
        if not 1 <= prod_top_k <= token_count:
            raise ValueError("prod_num_per_group은 1 이상 token 수 이하여야 한다.")
        self._heads = heads
        self._head_dim = d_model // heads
        self._prod_top_k = prod_top_k
        self.prompt_add = nn.Parameter(torch.randn(1, token_count, d_model) * 0.02)
        self.prompt_prod = nn.Parameter(torch.randn(1, token_count, d_model) * 0.02)
        self.add_q = nn.Linear(d_model, d_model, bias=False)
        self.add_k = nn.Linear(d_model, d_model, bias=False)
        self.add_v = nn.Linear(d_model, d_model, bias=False)
        self.prod_q = nn.Linear(d_model, d_model, bias=False)
        self.prod_k = nn.Linear(d_model, d_model, bias=False)
        self.prod_v = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        # 논문 식 (4): 2N개의 상호작용 후보를 embedding 축마다 N개로 줄인다.
        self.candidate_fusion = nn.Linear(2 * token_count, token_count)

    def _heads_view(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self._heads, self._head_dim).transpose(1, 2)

    def _merge_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, tokens, _ = value.shape
        return value.transpose(1, 2).reshape(batch, tokens, -1)

    def _attention(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        q_projection: nn.Linear,
        k_projection: nn.Linear,
        v_projection: nn.Linear,
        *,
        top_k: int | None,
    ) -> torch.Tensor:
        q = self._heads_view(q_projection(query))
        k = self._heads_view(k_projection(source))
        v = self._heads_view(v_projection(source))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self._head_dim)
        if top_k is None:
            weights = self.attention_dropout(torch.softmax(scores, dim=-1))
            return self._merge_heads(torch.matmul(weights, v))

        selected_scores, selected_indices = torch.topk(scores, top_k, dim=-1)
        weights = self.attention_dropout(torch.softmax(selected_scores, dim=-1))
        batch, heads, queries, _ = selected_indices.shape
        expanded_v = v.unsqueeze(2).expand(batch, heads, queries, -1, self._head_dim)
        selected_v = torch.gather(
            expanded_v,
            3,
            selected_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self._head_dim),
        )
        return self._merge_heads(torch.sum(weights.unsqueeze(-1) * selected_v, dim=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # target_mode=mix: 고정 prompt와 같은 위치의 행별 토큰을 함께 query로 쓴다.
        add_query = self.prompt_add + x
        additive = self._attention(
            add_query, x, self.add_q, self.add_k, self.add_v, top_k=None
        )

        # 논문 식 (2). exp는 float32에서 계산하고 fp16 유한 범위 안으로 제한한다.
        log_x = torch.log(torch.relu(x.float()) + 1e-6).to(dtype=x.dtype)
        prod_query = self.prompt_prod + log_x
        multiplicative_log = self._attention(
            prod_query,
            log_x,
            self.prod_q,
            self.prod_k,
            self.prod_v,
            top_k=self._prod_top_k,
        )
        multiplicative = torch.exp(
            torch.clamp(multiplicative_log.float(), -10.0, 10.0)
        ).to(dtype=x.dtype)

        candidates = torch.cat([additive, multiplicative], dim=1)
        return self.candidate_fusion(candidates.transpose(1, 2)).transpose(1, 2)


class _AMFormerLayer(nn.Module):
    def __init__(
        self,
        token_count: int,
        d_model: int,
        heads: int,
        prod_top_k: int,
        attention_dropout: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.arithmetic = _PromptArithmeticAttention(
            token_count, d_model, heads, prod_top_k, attention_dropout
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.residual_dropout(self.arithmetic(x)))
        return self.norm2(x + self.feed_forward(x))


class _AMFormer(nn.Module):
    def __init__(
        self,
        token_count: int,
        numeric_positions: list[int],
        categorical_positions: list[int],
        categorical_cardinalities: list[int],
        d_model: int,
        heads: int,
        prod_top_ks: list[int],
        attention_dropout: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = _ColumnTokenizer(
            token_count,
            numeric_positions,
            categorical_positions,
            categorical_cardinalities,
            d_model,
        )
        self.layers = nn.ModuleList(
            [
                _AMFormerLayer(
                    token_count,
                    d_model,
                    heads,
                    prod_top_k,
                    attention_dropout,
                    dropout,
                )
                for prod_top_k in prod_top_ks
            ]
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_attention = nn.MultiheadAttention(
            d_model, heads, dropout=attention_dropout, batch_first=True
        )
        self.cls_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(numeric, categorical)
        for layer in self.layers:
            x = layer(x)
        cls = self.cls.expand(len(x), -1, -1)
        pooled, _ = self.cls_attention(cls, x, x, need_weights=False)
        return self.head(self.cls_norm(cls + pooled)[:, 0]).squeeze(-1)


class AMFormerFold:
    """fold 하나의 AMFormer 학습, 예측, 중요도와 진입 관측을 소유한다."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._missing_indicator_cols = list(params.pop("missing_indicator_cols"))
        self._layers = int(params.pop("layers", 3))
        self._requested_d_model = int(params.pop("d_model", 128))
        self._heads = int(params.pop("heads", 8))
        self._groups = list(params.pop("groups", [42, 42, 42]))
        self._prod_top_ks = list(params.pop("prod_num_per_group", [4, 4, 4]))
        self._cluster = params.pop("cluster", True)
        self._target_mode = params.pop("target_mode", "mix")
        self._token_descent = params.pop("token_descent", False)
        self._use_prod = params.pop("use_prod", True)
        self._use_cls_token = params.pop("use_cls_token", True)
        self._attention_dropout = float(params.pop("attention_dropout", 0.2))
        self._dropout = float(params.pop("dropout", 0.1))
        self._epochs = int(params.pop("epochs", 50))
        self._patience = int(params.pop("patience", 5))
        self._batch_size = int(params.pop("batch_size", 512))
        self._eval_batch_size = int(params.pop("eval_batch_size", 8192))
        self._lr = float(params.pop("lr", 1e-4))
        self._weight_decay = float(params.pop("weight_decay", 1e-5))
        self._perm_sample = int(params.pop("perm_sample", 8192))
        self._perm_repeats = int(params.pop("perm_repeats", 1))
        requested_device = params.pop("device", None)
        if params:
            raise ValueError(f"amformer가 모르는 params: {sorted(params)}")
        if self._layers != len(self._groups) or self._layers != len(self._prod_top_ks):
            raise ValueError("layers, groups, prod_num_per_group 길이가 같아야 한다.")
        if self._requested_d_model != 128:
            raise ValueError("AMFormer 기준 구성 d_model은 128이어야 한다.")
        if self._requested_d_model % self._heads:
            raise ValueError("AMFormer d_model은 heads의 배수여야 한다.")
        if not all((self._cluster, self._use_prod, self._use_cls_token)):
            raise ValueError("AMFormer 기준 구성은 cluster/use_prod/use_cls_token=true다.")
        if self._target_mode != "mix" or self._token_descent is not False:
            raise ValueError("AMFormer 기준 구성은 target_mode=mix, token_descent=false다.")
        if min(
            self._epochs,
            self._patience,
            self._batch_size,
            self._eval_batch_size,
            self._perm_sample,
            self._perm_repeats,
        ) <= 0:
            raise ValueError("AMFormer 학습·importance 크기 설정은 양수여야 한다.")
        if requested_device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = str(requested_device)
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("AMFormer device=cuda인데 CUDA를 사용할 수 없다.")
        if self._device == "cpu":
            torch.set_num_threads(1)
        if self._device.startswith("cuda"):
            self._amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self._amp_dtype = torch.float16
        self._seed = seed
        self._encoder = _FoldEncoder(self._missing_indicator_cols)
        self._model: _AMFormer | None = None
        self._importance_X: pd.DataFrame | None = None
        self._importance_y: np.ndarray | None = None
        self._importance_base_auc: float | None = None
        self._diagnostics = AdapterDiagnostics()

    def _autocast(self):
        if self._device.startswith("cuda"):
            return torch.autocast("cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def _new_model(self, d_model: int) -> _AMFormer:
        torch.manual_seed(self._seed)
        if self._device.startswith("cuda"):
            torch.cuda.manual_seed(self._seed)
        return _AMFormer(
            self._encoder.token_count,
            self._encoder.numeric_token_positions,
            self._encoder.categorical_token_positions,
            self._encoder.categorical_cardinalities,
            d_model,
            self._heads,
            self._prod_top_ks,
            self._attention_dropout,
            self._dropout,
        ).to(self._device)

    def _memory_threshold(self) -> tuple[int | None, int | None]:
        if not self._device.startswith("cuda"):
            return None, None
        total = int(torch.cuda.get_device_properties(self._device).total_memory)
        total_gib = total / 1024**3
        threshold_gib = 14 if total_gib < 20 else 20
        return total, threshold_gib * 1024**3

    def _probe(
        self,
        model: _AMFormer,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, object]:
        batch_size = min(self._batch_size, len(target))
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._device)
            torch.cuda.synchronize(self._device)
        started = time.monotonic()
        model.train()
        model.zero_grad(set_to_none=True)
        with self._autocast():
            output = model(numeric[:batch_size], categorical[:batch_size])
            loss = nn.functional.binary_cross_entropy_with_logits(output, target[:batch_size])
        loss.backward()
        if self._device.startswith("cuda"):
            torch.cuda.synchronize(self._device)
        elapsed = time.monotonic() - started
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        finite = bool(torch.isfinite(output).all() and torch.isfinite(loss) and gradients_finite)
        model.zero_grad(set_to_none=True)
        reserved = (
            int(torch.cuda.max_memory_reserved(self._device))
            if self._device.startswith("cuda")
            else None
        )
        total, threshold = self._memory_threshold()
        return {
            "batch_size": batch_size,
            "output_shape": list(output.shape),
            "finite": finite,
            "seconds": float(elapsed),
            "rows_per_second": float(batch_size / elapsed),
            "cuda_max_reserved_bytes": reserved,
            "cuda_device_total_bytes": total,
            "cuda_memory_threshold_bytes": threshold,
            "memory_within_threshold": reserved is None or threshold is None or reserved <= threshold,
        }

    def _prepare_model_with_probe(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[_AMFormer, int, dict[str, object], str | None]:
        fallback_reason = None
        for d_model in (self._requested_d_model, 64):
            model = None
            try:
                model = self._new_model(d_model)
                probe = self._probe(model, numeric, categorical, target)
                if probe["finite"] and probe["memory_within_threshold"]:
                    return model, d_model, probe, fallback_reason
                fallback_reason = (
                    "전진·역전파 수치가 유한하지 않다."
                    if not probe["finite"]
                    else "전진·역전파 최고 GPU 메모리가 장치 한도를 넘었다."
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                fallback_reason = "전진·역전파에서 GPU 메모리가 부족했다."
            if d_model == 64:
                break
            if model is not None:
                del model
            gc.collect()
            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
        raise RuntimeError(f"AMFormer 폭 64 재시도도 진입 진단에 실패했다: {fallback_reason}")

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        self._encoder.fit(X_tr)
        if any(group != self._encoder.token_count for group in self._groups):
            raise ValueError(
                f"AMFormer groups {self._groups}가 실제 token 수 "
                f"{self._encoder.token_count}와 다르다."
            )
        if any(not 1 <= value <= self._encoder.token_count for value in self._prod_top_ks):
            raise ValueError("prod_num_per_group은 1 이상 실제 token 수 이하여야 한다.")

        numeric_tr, categorical_tr = (
            tensor.to(self._device) for tensor in self._encoder.transform(X_tr)
        )
        numeric_va, categorical_va = (
            tensor.to(self._device) for tensor in self._encoder.transform(X_va)
        )
        target_tr = torch.from_numpy(y_tr.to_numpy(dtype="float32")).to(self._device)
        target_va = y_va.to_numpy(dtype="float64")

        model, used_d_model, probe, fallback_reason = self._prepare_model_with_probe(
            numeric_tr, categorical_tr, target_tr
        )
        self._model = model
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        device_type = "cuda" if self._device.startswith("cuda") else "cpu"
        scaler = torch.amp.GradScaler(
            device_type,
            enabled=device_type == "cuda" and self._amp_dtype == torch.float16,
        )
        generator = torch.Generator().manual_seed(self._seed)
        best_auc = -math.inf
        best_epoch = -1
        best_weights = None
        bad_epochs = 0
        epoch_seconds: list[float] = []
        epoch_aucs: list[float] = []
        for epoch in range(self._epochs):
            started = time.monotonic()
            model.train()
            permutation = torch.randperm(len(X_tr), generator=generator).to(self._device)
            for offset in range(0, len(X_tr), self._batch_size):
                rows = permutation[offset : offset + self._batch_size]
                with self._autocast():
                    output = model(numeric_tr[rows], categorical_tr[rows])
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        output, target_tr[rows]
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("AMFormer 학습 손실에 유한하지 않은 값이 생겼다.")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            validation_logit = self._predict_encoded(numeric_va, categorical_va)
            auc = float(roc_auc_score(target_va, validation_logit))
            elapsed = time.monotonic() - started
            epoch_seconds.append(float(elapsed))
            epoch_aucs.append(auc)
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_weights = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(
                f"[amformer] ep{epoch:3d} valAUC={auc:.6f} best={best_auc:.6f} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )
            if bad_epochs >= self._patience:
                break
        assert best_weights is not None
        model.load_state_dict(best_weights)
        validation_logit = self._predict_encoded(numeric_va, categorical_va)

        sample_size = min(self._perm_sample, len(X_va))
        rng = np.random.default_rng(self._seed)
        sample_positions = np.sort(rng.choice(len(X_va), size=sample_size, replace=False))
        self._importance_X = X_va.iloc[sample_positions].copy()
        self._importance_y = target_va[sample_positions]
        self._importance_base_auc = float(
            roc_auc_score(self._importance_y, self.predict(self._importance_X))
        )
        self._diagnostics = AdapterDiagnostics(
            assertions={
                "probe_output_shape": probe["output_shape"] == [probe["batch_size"]],
                "probe_forward_backward_finite": bool(probe["finite"]),
                "probe_memory_within_device_threshold": bool(
                    probe["memory_within_threshold"]
                ),
                "token_count_matches_prompt_groups": all(
                    group == self._encoder.token_count for group in self._groups
                ),
            },
            observations={
                "token_count": self._encoder.token_count,
                "token_names": self._encoder.token_names,
                "prompt_groups": self._groups,
                "prod_num_per_group": self._prod_top_ks,
                "requested_d_model": self._requested_d_model,
                "used_d_model": used_d_model,
                "width_fallback_reason": fallback_reason,
                "model_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "probe": probe,
                "epoch_seconds": epoch_seconds,
                "epoch_validation_aucs": epoch_aucs,
                "best_epoch": best_epoch,
                "best_validation_auc": best_auc,
                "importance_rows": sample_size,
                "importance_repeats": self._perm_repeats,
            },
        )
        return _sigmoid(validation_logit)

    def _predict_encoded(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> np.ndarray:
        assert self._model is not None
        self._model.eval()
        outputs = []
        with torch.no_grad(), self._autocast():
            for offset in range(0, len(numeric), self._eval_batch_size):
                outputs.append(
                    self._model(
                        numeric[offset : offset + self._eval_batch_size],
                        categorical[offset : offset + self._eval_batch_size],
                    )
                )
        return torch.cat(outputs).float().cpu().numpy().astype("float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        numeric, categorical = (
            tensor.to(self._device) for tensor in self._encoder.transform(X)
        )
        return _sigmoid(self._predict_encoded(numeric, categorical))

    def importance(self) -> pd.DataFrame:
        assert self._importance_X is not None
        assert self._importance_y is not None
        assert self._importance_base_auc is not None
        gains = []
        for column_index, column in enumerate(self._importance_X.columns):
            drops = []
            for repeat in range(self._perm_repeats):
                rng = np.random.default_rng(
                    self._seed * 10007 + column_index * 101 + repeat
                )
                shuffled = self._importance_X.copy()
                order = rng.permutation(len(shuffled))
                shuffled[column] = (
                    self._importance_X[column].iloc[order].set_axis(shuffled.index)
                )
                score = roc_auc_score(self._importance_y, self.predict(shuffled))
                drops.append(self._importance_base_auc - score)
            gains.append(float(np.mean(drops)))
        return pd.DataFrame(
            {"feature": list(self._importance_X.columns), "gain": gains}
        )

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics
