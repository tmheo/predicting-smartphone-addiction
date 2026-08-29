"""Lookup-Transformer 학습기의 torch 구현. (#58)

원문(tamerlanomralinov의 S6E8 Lookup-Transformer, LB 0.97041) 아키텍처의 재현:
컬럼마다 토큰 하나를 만들되, 정확값 lookup embedding(합성 데이터의 반복 키)과
rank-gauss 수치 embedding(PLR, 완만한 연속 추세)을 더해 Transformer encoder로
상호작용시키고 CLS 토큰으로 분류한다.

원문과 다른 점(티켓 #58의 충돌 해소):
- fold는 커밋된 artifacts/folds.parquet의 5-fold다(원문은 10/11-fold 혼재).
- 기본 어휘와 rank-gauss 분위 함수는 학습 fold에서만 맞춘다.
  검증·테스트에만 있는 값은 컬럼별 UNK id로 보내 결측(NA id)과 구분한다.
  목표값 비참조 train+test 결합 전처리는 명시적인 설정에서만 허용한다.
- lookup 대상 컬럼은 설정의 lookup_cols로 명시한다. 나머지 컬럼(연속 파생·placebo
  등)은 원문의 파생 토큰처럼 PLR 전용 토큰이 된다.

전처리 기준 집합은 preprocessing_scope로 고른다. 기본 fold_train은 기존 누출 규율을
유지하고, train_test는 목표값을 보지 않는 train+test 결합 어휘와 분위 변환을 쓴다.
validation_selection=final은 검증 점수를 관찰만 하고 고정 epoch 마지막 EMA를 선택한다.

티켓 #360의 값 가리기 증강 분포 형태는 value_dropout_sampler로 고른다. 기본
independent는 셀별 독립 균등 Bernoulli(기존 경로, 수치 무변경)이고, row_mask는
mask_pool(fold_train 또는 test)의 실측 행 마스크를 배치 행마다 하나씩 기증받아
alpha = value_dropout / (기증 열의 셀 단위 평균 결측률)로 채택한다. 풀에 결측이 없는
열은 기증받을 형태가 없으므로 기존 경로의 value_dropout을 그대로 쓴다. 그래서 기대
가림 셀 수가 열마다 보존되고, 바뀌는 것은 가리는 양이 아니라 열 사이 상관과 꼬리의
형태다.

티켓 #127의 fold 내 초기화 평균은 파이프라인 시드와 구분되는
fold_seed_offsets로 설정한다. 파이프라인 시드 s와 offset o의 합을 초기화 시드로
쓰고, 같은 fold에서 학습한 구성원의 확률 예측을 평균해 fold 예측 하나를 만든다.

torch가 필요하므로 model.py의 adapter가 이 모듈을 lazy import한다.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

_NA = 0  # 컬럼별 local id 0은 결측 전용. lookup 컬럼의 마지막 id는 UNK.
_TORCH_INIT_LOCK = threading.Lock()
_OPTIMIZERS = {"adamw", "radam", "nadam", "muon"}
_VALUE_DROPOUT_SAMPLERS = {"independent", "row_mask"}
_MASK_POOLS = {"fold_train", "test"}
_LR_SCHEDULES = {
    "one_cycle",
    "one_cycle_fixed_momentum",
    "warmup_cosine",
    "warmup_linear",
    "warmup_constant",
    "warmup_plateau",
}
_TARGET_INDEPENDENT_LR_SCHEDULES = _LR_SCHEDULES - {"warmup_plateau"}
_WARMUP_FRACTION = 0.15
_START_DIVISOR = 25.0
_FINAL_DIVISOR = 250_000.0
_GRADIENT_CLIP_NORM = 1.0


def _positive_epoch(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}은 1 이상의 정수여야 한다: {value!r}")
    return value


def _validate_epoch_window(
    trajectory_end_epochs: object,
    schedule_horizon_epochs: object,
) -> tuple[int, int]:
    end = _positive_epoch(trajectory_end_epochs, "trajectory_end_epochs")
    horizon = _positive_epoch(schedule_horizon_epochs, "schedule_horizon_epochs")
    if end > horizon:
        raise ValueError(
            "trajectory_end_epochs는 schedule_horizon_epochs보다 클 수 없다: "
            f"{end} > {horizon}"
        )
    return end, horizon


def _validate_completed_epochs(
    completed_epochs: object,
    *,
    trajectory_end_epochs: int,
    schedule_horizon_epochs: int,
) -> tuple[int, ...]:
    if not isinstance(completed_epochs, (list, tuple)) or not completed_epochs:
        raise ValueError("completed_epochs는 비어 있지 않은 양의 정수 목록이어야 한다.")
    points = tuple(
        _positive_epoch(value, "completed_epochs 항목") for value in completed_epochs
    )
    if points != tuple(sorted(set(points))):
        raise ValueError("completed_epochs는 중복 없는 오름차순이어야 한다.")
    if points[-1] > trajectory_end_epochs or points[-1] > schedule_horizon_epochs:
        raise ValueError(
            "completed_epochs는 trajectory 종료와 schedule 지평 이하여야 한다: "
            f"{points[-1]} > {trajectory_end_epochs}/{schedule_horizon_epochs}"
        )
    return points


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    """다른 adapter들과 같은 확률 축으로 돌려준다(AUC는 순위 기반이라 영향 없음)."""
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))


def _value_dropout_mask(
    shape: tuple[int, int],
    *,
    device: str,
    rate: float,
    pool: torch.Tensor | None,
    alpha: float | None,
    donor_columns: torch.Tensor | None,
) -> torch.Tensor:
    """값 가리기 증강에서 가릴 셀을 고른다. (#360)

    pool이 None이면 셀별 독립 균등 Bernoulli(기존 경로)다. pool이 있으면 배치 행마다
    풀에서 실측 행 마스크 하나를 균등 복원 추출해 기증받고, 기증 마스크의 각 셀을
    확률 alpha로 채택한다.

    donor_columns가 참인 열만 기증 마스크를 쓴다. 풀에 결측이 하나도 없는 열은
    기증받을 형태가 없으므로 기존 경로의 rate를 그대로 유지한다. alpha는 caller가
    기증 열 전체의 기대 가림 셀 수를 rate와 같게 맞춘 값이므로, 바뀌는 것은 가리는
    양이 아니라 열 사이 상관과 꼬리의 형태다.
    """
    uniform = torch.rand(shape, device=device)
    if pool is None:
        return uniform < rate
    donor = pool[torch.randint(pool.shape[0], (shape[0],), device=device)]
    probability = torch.where(
        donor_columns, donor.to(uniform.dtype) * alpha, uniform.new_full((), rate)
    )
    return uniform < probability


def _muon_parameter_names(model: nn.Module) -> set[str]:
    """Muon으로 학습할 은닉 행렬 가중치의 이름. (#196)

    Transformer encoder의 2차원 행렬(attention 사영, feed-forward)과 head의
    은닉 Linear만 해당한다. embedding, PLR, cls·pos, 편향, LayerNorm과
    출력층(head의 마지막 Linear)은 AdamW로 남긴다.
    """
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2
        and (name.startswith("tr.") or name == "head.1.weight")
    }


def _create_optimizer(
    name: str,
    parameter_groups: list[dict[str, object]],
    lr: float,
) -> torch.optim.Optimizer:
    """가중치 감쇠 그룹을 보존한 최적화 알고리즘을 만든다.

    muon은 그룹별 algorithm 표시가 붙은 그룹을 요구하고, 나머지 이름은 표시 없는
    Adam 계열 그룹을 받는다.
    """
    if name == "muon":
        from .muon import MuonWithAdamW

        return MuonWithAdamW(parameter_groups, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(parameter_groups, lr=lr)
    if name == "radam":
        return torch.optim.RAdam(
            parameter_groups,
            lr=lr,
            decoupled_weight_decay=True,
        )
    if name == "nadam":
        return torch.optim.NAdam(
            parameter_groups,
            lr=lr,
            decoupled_weight_decay=True,
        )
    raise ValueError(f"optimizer는 {sorted(_OPTIMIZERS)} 중 하나여야 한다: {name!r}")


class _LearningRateController:
    """배치 일정과 검증 AUC 기반 일정을 같은 좁은 계약으로 감싼다."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        name: str,
        max_lr: float,
        total_steps: int,
        group_lr_scales: list[float] | None = None,
    ) -> None:
        if name not in _LR_SCHEDULES:
            raise ValueError(f"lr_schedule은 {sorted(_LR_SCHEDULES)} 중 하나여야 한다: {name!r}")
        if total_steps <= 0:
            raise ValueError("학습률 일정의 total_steps는 양수여야 한다.")
        scales = (
            [1.0] * len(optimizer.param_groups)
            if group_lr_scales is None
            else [float(scale) for scale in group_lr_scales]
        )
        if len(scales) != len(optimizer.param_groups):
            raise ValueError(
                "group_lr_scales의 길이가 param_groups 수와 다르다: "
                f"{len(scales)} != {len(optimizer.param_groups)}"
            )
        if any(scale <= 0 for scale in scales):
            raise ValueError(f"group_lr_scales는 모두 양수여야 한다: {scales}")
        self.optimizer = optimizer
        self.name = name
        # max_lr과 min_lr은 첫 그룹(진단이 읽는 축) 기준값이다. 배율이 붙은 그룹은
        # _set_lr과 OneCycleLR이 그룹별로 곱해 적용한다.
        self.max_lr = max_lr
        self.group_lr_scales = scales
        self.total_steps = total_steps
        self.completed_steps = 0
        self.warmup_steps = max(1, math.ceil(total_steps * _WARMUP_FRACTION))
        self.min_lr = max_lr / _FINAL_DIVISOR
        self._one_cycle = None
        self._plateau = None

        if name in {"one_cycle", "one_cycle_fixed_momentum"}:
            self._one_cycle = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [max_lr * scale for scale in scales],
                total_steps=total_steps,
                pct_start=_WARMUP_FRACTION,
                cycle_momentum=name == "one_cycle",
                div_factor=_START_DIVISOR,
                final_div_factor=_FINAL_DIVISOR / _START_DIVISOR,
            )
        else:
            self._set_lr(max_lr / _START_DIVISOR)
            if name == "warmup_plateau":
                self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.3,
                    patience=1,
                    threshold=0.0,
                    threshold_mode="abs",
                    min_lr=[self.min_lr * scale for scale in scales],
                )

    def _set_lr(self, value: float) -> None:
        for group, scale in zip(self.optimizer.param_groups, self.group_lr_scales):
            group["lr"] = value * scale

    def _scheduled_lr(self, completed_steps: int) -> float:
        if completed_steps <= self.warmup_steps:
            progress = completed_steps / self.warmup_steps
            start = self.max_lr / _START_DIVISOR
            return start + (self.max_lr - start) * progress
        if self.name in {"warmup_constant", "warmup_plateau"}:
            return self.learning_rate
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (completed_steps - self.warmup_steps) / decay_steps)
        if self.name == "warmup_linear":
            return self.max_lr + (self.min_lr - self.max_lr) * progress
        if self.name == "warmup_cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine
        raise AssertionError(f"배치 일정을 계산할 수 없다: {self.name}")

    def step_batch(self) -> None:
        self.completed_steps += 1
        if self._one_cycle is not None:
            self._one_cycle.step()
            return
        if self.completed_steps <= self.warmup_steps:
            self._set_lr(self._scheduled_lr(self.completed_steps))
        elif self.name in {"warmup_cosine", "warmup_linear"}:
            self._set_lr(self._scheduled_lr(self.completed_steps))

    def step_validation(self, auc: float) -> None:
        if self._plateau is not None and self.completed_steps >= self.warmup_steps:
            self._plateau.step(auc)

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def beta1(self) -> float | None:
        betas = self.optimizer.param_groups[0].get("betas")
        return float(betas[0]) if betas is not None else None


class _PLR(nn.Module):
    """Periodic-Linear 수치 embedding(컬럼별 학습 Fourier 주파수). 원문 PLR 그대로."""

    def __init__(self, nfeat: int, k: int, d: int, sigma: float = 0.5) -> None:
        super().__init__()
        self.f = nn.Parameter(torch.randn(nfeat, k) * sigma)
        self.w = nn.Parameter(torch.randn(nfeat, 2 * k, d) / math.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(nfeat, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2 * math.pi * x.unsqueeze(-1) * self.f.unsqueeze(0)
        z = torch.cat([torch.sin(z), torch.cos(z)], -1)
        return torch.einsum("bfk,fkd->bfd", z, self.w) + self.b


class _LookupTransformer(nn.Module):
    """CLS + 컬럼 토큰(lookup embedding + 마스크된 PLR)의 Transformer encoder."""

    def __init__(
        self, totv: int, nfeat: int, d: int, k: int, layers: int, heads: int, drop: float
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(totv, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.plr = _PLR(nfeat, k, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 1 + nfeat, d) * 0.02)
        self.edrop = nn.Dropout(drop)
        enc = nn.TransformerEncoderLayer(
            d, heads, d * 2, drop, activation="gelu", batch_first=True, norm_first=True
        )
        # norm_first=True에서는 PyTorch 중첩 텐서 최적화를 쓸 수 없다.
        # 자동 시도 경고를 피하고 실제로 쓰는 일반 텐서 경로를 명시한다.
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1)
        )

    def forward(
        self, ids: torch.Tensor, num: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        tok = self.emb(ids) + self.plr(num) * (1 - mask).unsqueeze(-1)
        t = torch.cat([self.cls.expand(ids.shape[0], -1, -1), tok], 1) + self.pos
        return self.head(self.tr(self.edrop(t))[:, 0]).squeeze(-1)


class _LookupTransformerMember:
    """초기화 시드 하나에 해당하는 fold 학습·예측·중요도 상태."""

    def __init__(
        self,
        params: dict,
        seed: int,
        device: str | None = None,
        init_barrier: threading.Barrier | None = None,
    ) -> None:
        params = dict(params)
        self._lookup_cols = list(params.pop("lookup_cols"))
        self._lookup_max_card = int(params.pop("lookup_max_card", 5000))
        self._d_model = int(params.pop("d_model", 128))
        self._plr_k = int(params.pop("plr_k", 24))
        self._layers = int(params.pop("layers", 4))
        self._heads = int(params.pop("heads", 8))
        self._dropout = float(params.pop("dropout", 0.1))
        self._epochs = int(params.pop("epochs", 32))
        self._batch_size = int(params.pop("batch_size", 2048))
        self._lr = float(params.pop("lr", 2e-3))
        self._optimizer_name = str(params.pop("optimizer", "adamw")).lower()
        self._muon_lr_multiplier = float(params.pop("muon_lr_multiplier", 1.0))
        self._lr_schedule = str(params.pop("lr_schedule", "one_cycle")).lower()
        self._value_dropout = float(params.pop("value_dropout", 0.10))
        self._value_dropout_sampler = str(
            params.pop("value_dropout_sampler", "independent")
        ).lower()
        self._mask_pool = str(params.pop("mask_pool", "fold_train")).lower()
        self._weight_decay = float(params.pop("weight_decay", 1e-5))
        self._emb_weight_decay = float(params.pop("emb_weight_decay", 3e-4))
        self._ema_decay = float(params.pop("ema_decay", 0.999))
        self._patience = int(params.pop("patience", 5))
        self._perm_repeats = int(params.pop("perm_repeats", 3))
        self._preprocessing_scope = str(
            params.pop("preprocessing_scope", "fold_train")
        ).lower()
        self._validation_selection = str(
            params.pop("validation_selection", "best")
        ).lower()
        if params:
            raise ValueError(f"lookup_transformer가 모르는 params: {sorted(params)}")
        if self._optimizer_name not in _OPTIMIZERS:
            raise ValueError(
                f"optimizer는 {sorted(_OPTIMIZERS)} 중 하나여야 한다: "
                f"{self._optimizer_name!r}"
            )
        if self._muon_lr_multiplier <= 0:
            raise ValueError(
                f"muon_lr_multiplier는 양수여야 한다: {self._muon_lr_multiplier!r}"
            )
        if self._muon_lr_multiplier != 1.0 and self._optimizer_name != "muon":
            raise ValueError(
                "muon_lr_multiplier는 optimizer가 muon일 때만 1이 아닌 값을 가진다: "
                f"{self._optimizer_name!r}"
            )
        if self._lr_schedule not in _LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule은 {sorted(_LR_SCHEDULES)} 중 하나여야 한다: "
                f"{self._lr_schedule!r}"
            )
        if self._preprocessing_scope not in {"fold_train", "train_test"}:
            raise ValueError(
                "preprocessing_scope은 'fold_train' 또는 'train_test'여야 한다: "
                f"{self._preprocessing_scope!r}"
            )
        if self._validation_selection not in {"best", "final"}:
            raise ValueError(
                "validation_selection은 'best' 또는 'final'이어야 한다: "
                f"{self._validation_selection!r}"
            )
        if self._value_dropout_sampler not in _VALUE_DROPOUT_SAMPLERS:
            raise ValueError(
                "value_dropout_sampler는 "
                f"{sorted(_VALUE_DROPOUT_SAMPLERS)} 중 하나여야 한다: "
                f"{self._value_dropout_sampler!r}"
            )
        if self._mask_pool not in _MASK_POOLS:
            raise ValueError(
                f"mask_pool은 {sorted(_MASK_POOLS)} 중 하나여야 한다: {self._mask_pool!r}"
            )
        if self._value_dropout_sampler == "independent" and self._mask_pool != "fold_train":
            raise ValueError(
                "mask_pool은 value_dropout_sampler='row_mask'에서만 뜻을 가진다."
            )
        if self._epochs <= 0 or self._batch_size <= 0 or self._lr <= 0:
            raise ValueError("epochs, batch_size와 lr은 양수여야 한다.")
        if self._d_model % self._heads:
            raise ValueError(f"d_model {self._d_model}은 heads {self._heads}의 배수여야 한다.")
        self._seed = seed
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_barrier = init_barrier
        if self._device == "cpu":
            # macOS에서 lightgbm·xgboost의 libomp와 torch의 libomp가 같이 적재되면
            # CPU 연산의 OpenMP fork가 교착한다. CPU 경로(로컬 테스트)는 단일 스레드로 돈다.
            torch.set_num_threads(1)
        # T4/P100은 bf16이 없어 fp16 + GradScaler, Ampere 이상은 bf16을 쓴다(원문 동일).
        if self._device.startswith("cuda"):
            with torch.cuda.device(self._device):
                self._amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
        else:
            self._amp_dtype = torch.float16
        self._columns: list[str] | None = None
        self._specs: dict[str, tuple] = {}  # 컬럼별 ("lookup", vocab, qt|None) 또는 ("plr", qt)
        self._offsets: np.ndarray | None = None
        self._model: _LookupTransformer | None = None
        self._val: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray] | None = None
        self._val_auc: float | None = None
        self._training_diagnostics: dict[str, object] | None = None
        self._raw_training_length_selection: int | None = None
        self._training_point_states: dict[int, tuple[torch.Tensor, ...]] = {}
        self._training_point_diagnostics: dict[str, object] | None = None
        self._captured_completed_epochs: tuple[int, ...] = ()
        self._trajectory_end_epochs: int | None = None
        self._schedule_horizon_epochs: int | None = None
        self._trajectory_steps_per_epoch: int | None = None
        self._dataset_reference: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def set_dataset_reference(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("Lookup-Transformer 전처리 기준 집합의 train/test 열이 다르다.")
        self._dataset_reference = (X_train, X_test)

    # ---- 인코딩: 어휘·분위 fit 범위는 설정 계약, transform은 어디에나 적용 ----

    def _fit_specs(self, X_tr: pd.DataFrame) -> None:
        from sklearn.preprocessing import QuantileTransformer

        self._columns = list(X_tr.columns)
        missing = [c for c in self._lookup_cols if c not in self._columns]
        if missing:
            raise ValueError(f"lookup_cols가 입력에 없다: {missing}")
        vocab_sizes = []
        if self._preprocessing_scope == "train_test":
            if self._dataset_reference is None:
                raise ValueError(
                    "preprocessing_scope='train_test'에는 train+test 전처리 기준 집합이 필요하다."
                )
            reference_train, reference_test = self._dataset_reference
            if list(reference_train.columns) != self._columns:
                raise ValueError("Lookup-Transformer 입력과 전처리 기준 집합의 열이 다르다.")

        for col in self._columns:
            if self._preprocessing_scope == "train_test":
                reference_train, reference_test = self._dataset_reference
                values = pd.concat(
                    [reference_train[col], reference_test[col]], ignore_index=True
                )
            else:
                values = X_tr[col]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype(object)
            is_numeric = pd.api.types.is_numeric_dtype(values)
            qt = None
            if is_numeric:
                observed = values.dropna().to_numpy(dtype="float64").reshape(-1, 1)
                # subsample을 표본 수보다 크게 둬 분위 추정이 표본추출 없이 결정적이다.
                qt = QuantileTransformer(
                    n_quantiles=min(1000, len(observed)),
                    output_distribution="normal",
                    subsample=2_000_000_000,
                    random_state=self._seed,
                )
                qt.fit(observed)
            if col in self._lookup_cols:
                vocab = sorted(pd.unique(values.dropna()))
                if len(vocab) > self._lookup_max_card:
                    raise ValueError(
                        f"lookup 컬럼 {col}의 학습 fold 카디널리티 {len(vocab)}이 "
                        f"lookup_max_card {self._lookup_max_card}를 넘는다. "
                        "연속 컬럼은 lookup_cols에서 빼 PLR 전용 토큰으로 둔다."
                    )
                self._specs[col] = ("lookup", vocab, qt)
                vocab_sizes.append(len(vocab) + 2)  # NA + 값들 + UNK
            else:
                if not is_numeric:
                    raise ValueError(f"lookup_cols에 없는 컬럼 {col}은 수치여야 한다: PLR 전용 토큰.")
                self._specs[col] = ("plr", qt)
                vocab_sizes.append(1)  # NA 자리만. embedding은 컬럼 상수 역할이다.
        self._offsets = np.concatenate([[0], np.cumsum(vocab_sizes)[:-1]]).astype("int64")
        self._total_vocab = int(sum(vocab_sizes))

    def _encode(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert list(X.columns) == self._columns, "인코딩 입력 컬럼이 학습 때와 다르다."
        n = len(X)
        ids = np.zeros((n, len(self._columns)), dtype="int64")
        num = np.zeros((n, len(self._columns)), dtype="float32")
        mask = np.zeros((n, len(self._columns)), dtype="float32")
        for j, col in enumerate(self._columns):
            values = X[col]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype(object)
            spec = self._specs[col]
            isna = values.isna().to_numpy()
            if spec[0] == "lookup":
                _, vocab, qt = spec
                mapping = {v: i + 1 for i, v in enumerate(vocab)}
                col_ids = np.array(
                    values.map(mapping).fillna(len(vocab) + 1),  # 학습 fold에 없던 값 -> UNK
                    dtype="int64",
                )
                col_ids[isna] = _NA
                ids[:, j] = col_ids
            else:
                qt = spec[1]
            if qt is not None:
                observed = ~isna
                if observed.any():
                    num[observed, j] = (
                        qt.transform(
                            values.to_numpy(dtype="float64")[observed].reshape(-1, 1)
                        )
                        .ravel()
                        .astype("float32")
                    )
                mask[isna, j] = 1.0
            else:
                mask[:, j] = 1.0  # 수치 채널이 없는 범주 컬럼은 lookup embedding만 말한다.
            ids[:, j] += self._offsets[j]
        return torch.from_numpy(ids), torch.from_numpy(num), torch.from_numpy(mask)

    def _missing_matrix(self, X: pd.DataFrame) -> torch.Tensor:
        """행 x 열 실측 결측 지시자. 값 가리기 증강의 기증 마스크 풀 원본이다. (#360)

        인코딩의 mask 채널은 수치 채널이 없는 범주 열에서 항상 1이라 결측 지시자로
        쓸 수 없다. 여기서는 열마다 원자료의 결측만 읽는다.
        """
        assert list(X.columns) == self._columns, "마스크 풀 입력 열이 학습 때와 다르다."
        missing = np.zeros((len(X), len(self._columns)), dtype=bool)
        for j, col in enumerate(self._columns):
            missing[:, j] = X[col].isna().to_numpy()
        return torch.from_numpy(missing)

    # ---- 학습: 분리형 가중치 감쇠 + 학습률 일정 + EMA + 값 dropout ----

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        result = self._fit(X_tr, y_tr, X_va, y_va, epochs=self._epochs)
        assert result is not None
        return result

    def fit_full(self, X: pd.DataFrame, y: pd.Series, epochs: int) -> None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("Lookup-Transformer 전체 자료 재학습 epoch 수는 양의 정수여야 한다.")
        self._fit(X, y, None, None, epochs=epochs)

    def _validate_training_point_path(self) -> None:
        if self._validation_selection != "final":
            raise ValueError(
                "여러 학습 시점 포착은 validation_selection='final'에서만 지원한다."
            )
        if self._lr_schedule not in _TARGET_INDEPENDENT_LR_SCHEDULES:
            raise ValueError(
                "여러 학습 시점 포착의 학습률 일정은 검증 목표값을 참조할 수 없다: "
                f"{self._lr_schedule!r}"
            )

    def _validate_training_trajectory(
        self,
        completed_epochs: object,
        trajectory_end_epochs: object,
        schedule_horizon_epochs: object,
    ) -> tuple[tuple[int, ...], int, int]:
        self._validate_training_point_path()
        trajectory_end, schedule_horizon = _validate_epoch_window(
            trajectory_end_epochs, schedule_horizon_epochs
        )
        points = _validate_completed_epochs(
            completed_epochs,
            trajectory_end_epochs=trajectory_end,
            schedule_horizon_epochs=schedule_horizon,
        )
        return points, trajectory_end, schedule_horizon

    def _fit_training_trajectory(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        *,
        completed_epochs: tuple[int, ...],
        trajectory_end_epochs: int,
        schedule_horizon_epochs: int,
    ) -> None:
        self._fit(
            X_tr,
            y_tr,
            X_va,
            y_va,
            epochs=trajectory_end_epochs,
            schedule_horizon_epochs=schedule_horizon_epochs,
            capture_completed_epochs=completed_epochs,
        )
        if tuple(self._training_point_states) != completed_epochs:
            raise RuntimeError(
                "요청한 Lookup-Transformer 학습 시점을 모두 포착하지 못했다: "
                f"{tuple(self._training_point_states)} != {completed_epochs}"
            )

    def _fit_full_training_point(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        trajectory_end_epochs: int,
        schedule_horizon_epochs: int,
    ) -> None:
        self._fit(
            X,
            y,
            None,
            None,
            epochs=trajectory_end_epochs,
            schedule_horizon_epochs=schedule_horizon_epochs,
        )

    def _select_training_point(self, completed_epoch: object) -> np.ndarray:
        point = _positive_epoch(completed_epoch, "completed_epoch")
        if point not in self._training_point_states:
            raise ValueError(
                f"포착하지 않은 completed_epoch이다: {point}; "
                f"포착값={self._captured_completed_epochs}"
            )
        if self._model is None or self._val is None:
            raise RuntimeError("select_training_point는 검증 궤적 학습 뒤에 호출해야 한다.")
        state = self._training_point_states[point]
        params_list = list(self._model.parameters())
        if len(params_list) != len(state):
            raise RuntimeError("포착한 EMA 상태와 현재 모델의 매개변수 수가 다르다.")
        with torch.no_grad():
            for parameter, weight in zip(params_list, state):
                if parameter.shape != weight.shape:
                    raise RuntimeError("포착한 EMA 상태와 현재 모델의 매개변수 모양이 다르다.")
                parameter.copy_(weight.to(device=parameter.device, dtype=parameter.dtype))

        ids_va, num_va, mask_va, y_va = self._val
        val_logit = self._predict_tensors(ids_va, num_va, mask_va)
        selected_auc = roc_auc_score(y_va, val_logit)
        self._val_auc = selected_auc
        # raw_field는 0부터 세는 종료 위치다. completed_epoch와 정확히 한 칸 차이다.
        self._raw_training_length_selection = point - 1
        self._training_diagnostics = self._diagnostics_for_training_point(
            point, selected_auc
        )
        return _sigmoid(val_logit)

    def _diagnostics_for_training_point(
        self, completed_epoch: int, validation_auc: float
    ) -> dict[str, object]:
        if (
            self._training_point_diagnostics is None
            or self._trajectory_steps_per_epoch is None
            or self._trajectory_end_epochs is None
            or self._schedule_horizon_epochs is None
        ):
            raise RuntimeError("학습 시점 진단을 만들 궤적 정보가 없다.")
        diagnostics = dict(self._training_point_diagnostics)
        all_evaluations = diagnostics["evaluations"]
        assert isinstance(all_evaluations, list)
        evaluations = [
            dict(observation)
            for observation in all_evaluations
            if int(observation["epoch"]) < completed_epoch
        ]
        if evaluations:
            observed_best = max(
                evaluations, key=lambda observation: float(observation["validation_auc"])
            )
            observed_best_epoch: int | None = int(observed_best["epoch"])
            best_validation_auc = float(observed_best["validation_auc"])
        else:
            observed_best_epoch = None
            best_validation_auc = 0.0
        diagnostics.update(
            {
                "evaluations": evaluations,
                "best_epoch": completed_epoch - 1,
                "observed_best_epoch": observed_best_epoch,
                "best_validation_auc": best_validation_auc,
                "end_epoch": completed_epoch - 1,
                "completed_steps": self._trajectory_steps_per_epoch
                * completed_epoch,
                "trajectory_end_epochs": self._trajectory_end_epochs,
                "schedule_horizon_epochs": self._schedule_horizon_epochs,
                "captured_completed_epochs": list(self._captured_completed_epochs),
                "selected_completed_epoch": completed_epoch,
                "selected_validation_auc": float(validation_auc),
                "state_kind": "ema",
            }
        )
        return diagnostics

    def _fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame | None,
        y_va: pd.Series | None,
        *,
        epochs: int,
        schedule_horizon_epochs: int | None = None,
        capture_completed_epochs: tuple[int, ...] = (),
    ) -> np.ndarray | None:
        schedule_horizon = (
            epochs if schedule_horizon_epochs is None else schedule_horizon_epochs
        )
        if schedule_horizon < epochs:
            raise ValueError(
                "schedule_horizon_epochs는 실제 학습 epoch 수 이상이어야 한다: "
                f"{schedule_horizon} < {epochs}"
            )
        training_point_path = schedule_horizon_epochs is not None
        self._training_point_states = {}
        self._training_point_diagnostics = None
        self._captured_completed_epochs = capture_completed_epochs
        self._trajectory_end_epochs = epochs if training_point_path else None
        self._schedule_horizon_epochs = schedule_horizon if training_point_path else None
        self._trajectory_steps_per_epoch = None
        torch.backends.cuda.matmul.allow_tf32 = True
        self._fit_specs(X_tr)
        dev = self._device
        ids_tr, num_tr, mask_tr = (t.to(dev) for t in self._encode(X_tr))
        validation = X_va is not None
        if validation:
            assert y_va is not None
            ids_va, num_va, mask_va = (t.to(dev) for t in self._encode(X_va))
            y_va_np = y_va.to_numpy(dtype="float64")
        else:
            assert y_va is None
            ids_va = num_va = mask_va = None
            y_va_np = None
        y = torch.from_numpy(y_tr.to_numpy(dtype="float32")).to(dev)
        na_ids = torch.from_numpy(self._offsets).to(dev)  # 컬럼별 global NA id

        # 값 가리기 증강의 기증 마스크 풀. row_mask 표본기에서만 만든다. (#360)
        mask_pool: torch.Tensor | None = None
        donor_columns: torch.Tensor | None = None
        mask_pool_rate: float | None = None
        mask_pool_alpha: float | None = None
        mask_pool_donor_columns: int | None = None
        if self._value_dropout > 0 and self._value_dropout_sampler == "row_mask":
            if self._mask_pool == "fold_train":
                pool_frame = X_tr
            else:
                if self._dataset_reference is None:
                    raise ValueError(
                        "mask_pool='test'에는 train+test 전처리 기준 집합이 필요하다."
                    )
                pool_frame = self._dataset_reference[1]
            mask_pool = self._missing_matrix(pool_frame).to(dev)
            # 풀에 결측이 하나도 없는 열은 기증받을 형태가 없다. 그런 열은 기존 경로에
            # 그대로 두고, alpha 정규화도 기증 열에서만 잰다. 이래야 기대 가림 셀 수가
            # 열마다 value_dropout으로 보존돼 바뀌는 것이 형태 하나로 좁혀진다.
            donor_columns = mask_pool.any(dim=0)
            mask_pool_donor_columns = int(donor_columns.sum())
            if mask_pool_donor_columns == 0:
                raise ValueError("기증 마스크 풀에 결측 셀이 없어 값 가리기를 표본할 수 없다.")
            mask_pool_rate = float(
                mask_pool[:, donor_columns].to(torch.float32).mean()
            )
            mask_pool_alpha = min(1.0, self._value_dropout / mask_pool_rate)
            donor_columns = donor_columns.reshape(1, -1)

        # torch의 모델 초기화 난수는 프로세스 전역이므로 여러 GPU thread에서도
        # 초기화만 직렬화한다. 학습은 GPU별 난수 상태를 다시 고정한 뒤 병렬로 돈다.
        with _TORCH_INIT_LOCK:
            torch.manual_seed(self._seed)
            model = _LookupTransformer(
                self._total_vocab,
                len(self._columns),
                self._d_model,
                self._plr_k,
                self._layers,
                self._heads,
                self._dropout,
            ).to(dev)
        self._model = model
        emb_params = [p for n_, p in model.named_parameters() if n_.startswith("emb")]
        if self._optimizer_name == "muon":
            # 은닉 행렬만 Muon으로 보내고 나머지는 기존 AdamW 그룹 그대로다.
            # rest 그룹을 첫 그룹으로 유지해 학습률 진단(param_groups[0])이
            # 다른 optimizer와 같은 축을 읽게 한다.
            muon_names = _muon_parameter_names(model)
            rest = [
                p
                for n_, p in model.named_parameters()
                if not n_.startswith("emb") and n_ not in muon_names
            ]
            muon_params = [
                p for n_, p in model.named_parameters() if n_ in muon_names
            ]
            parameter_groups: list[dict[str, object]] = [
                {
                    "params": rest,
                    "weight_decay": self._weight_decay,
                    "algorithm": "adamw",
                },
                {
                    "params": emb_params,
                    "weight_decay": self._emb_weight_decay,
                    "algorithm": "adamw",
                },
                {
                    "params": muon_params,
                    "weight_decay": self._weight_decay,
                    "algorithm": "muon",
                },
            ]
            # Muon 그룹만 공유 학습률의 배수로 운전할 수 있다. (#385 2단계)
            group_lr_scales = [1.0, 1.0, self._muon_lr_multiplier]
        else:
            rest = [p for n_, p in model.named_parameters() if not n_.startswith("emb")]
            parameter_groups = [
                {"params": rest, "weight_decay": self._weight_decay},
                {"params": emb_params, "weight_decay": self._emb_weight_decay},
            ]
            group_lr_scales = [1.0, 1.0]
        opt = _create_optimizer(self._optimizer_name, parameter_groups, self._lr)
        n_tr = len(X_tr)
        steps_per_epoch = math.ceil(n_tr / self._batch_size)
        steps = steps_per_epoch * schedule_horizon + 10
        if training_point_path:
            self._trajectory_steps_per_epoch = steps_per_epoch
        schedule = _LearningRateController(
            opt,
            name=self._lr_schedule,
            max_lr=self._lr,
            total_steps=steps,
            group_lr_scales=group_lr_scales,
        )
        device_type = "cuda" if dev.startswith("cuda") else "cpu"
        use_scaler = device_type == "cuda" and self._amp_dtype == torch.float16
        scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
        loss_fn = nn.BCEWithLogitsLoss()
        params_list = list(model.parameters())
        ema = [p.detach().clone() for p in params_list]
        g = torch.Generator().manual_seed(self._seed)
        if self._init_barrier is not None:
            # 모든 구성원의 torch.manual_seed 호출이 끝난 다음 각 GPU의 dropout
            # 난수 상태를 해당 초기화 시드로 복원한다. 제한 시간은 실패 시 교착 방지용이다.
            self._init_barrier.wait(timeout=600)
            with torch.cuda.device(dev):
                torch.cuda.manual_seed(self._seed)
            self._init_barrier.wait(timeout=60)

        best_auc, best_weights, best_epoch, bad = 0.0, None, None, 0
        evaluations: list[dict[str, object]] = []
        eval_start = min(5, epochs - 1)
        end_epoch = -1
        for ep in range(epochs):
            end_epoch = ep
            model.train()
            perm = torch.randperm(n_tr, generator=g).to(dev)
            epoch_loss_sum = torch.zeros((), dtype=torch.float32, device=dev)
            epoch_gradient_norm_sum = torch.zeros((), dtype=torch.float32, device=dev)
            epoch_clipped_steps = torch.zeros((), dtype=torch.float32, device=dev)
            epoch_rows = 0
            epoch_steps = 0
            for i in range(0, n_tr, self._batch_size):
                sl = perm[i : i + self._batch_size]
                ids_b, mask_b = ids_tr[sl], mask_tr[sl]
                if self._value_dropout > 0:
                    # 값을 추가로 숨겨 모든 결측 패턴을 학습한다(원문 aug).
                    drop = _value_dropout_mask(
                        ids_b.shape,
                        device=dev,
                        rate=self._value_dropout,
                        pool=mask_pool,
                        alpha=mask_pool_alpha,
                        donor_columns=donor_columns,
                    )
                    ids_b = torch.where(drop, na_ids.expand_as(ids_b), ids_b)
                    mask_b = torch.maximum(mask_b, drop.float())
                with self._autocast():
                    loss = loss_fn(model(ids_b, num_tr[sl], mask_b), y[sl])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    params_list, _GRADIENT_CLIP_NORM
                )
                scaler.step(opt)
                scaler.update()
                schedule.step_batch()
                rows = len(sl)
                epoch_loss_sum += loss.detach().float() * rows
                epoch_gradient_norm_sum += gradient_norm.detach().float()
                epoch_clipped_steps += (
                    gradient_norm.detach() > _GRADIENT_CLIP_NORM
                ).float()
                epoch_rows += rows
                epoch_steps += 1
                with torch.no_grad():
                    torch._foreach_mul_(ema, self._ema_decay)
                    torch._foreach_add_(
                        ema, [p.detach() for p in params_list], alpha=1 - self._ema_decay
                    )
            if validation and ep >= eval_start and (ep % 2 == 1 or ep == epochs - 1):
                assert ids_va is not None and num_va is not None and mask_va is not None
                assert y_va_np is not None
                backup = [p.detach().clone() for p in params_list]
                with torch.no_grad():
                    for p, e in zip(params_list, ema):
                        p.copy_(e)
                auc = roc_auc_score(y_va_np, self._predict_tensors(ids_va, num_va, mask_va))
                if auc > best_auc:
                    best_auc, best_weights, best_epoch, bad = (
                        auc,
                        (
                            [e.clone() for e in ema]
                            if self._validation_selection == "best"
                            else None
                        ),
                        ep,
                        0,
                    )
                else:
                    bad += 1
                lr_before_update = schedule.learning_rate
                beta1_before_update = schedule.beta1
                schedule.step_validation(auc)
                observation = {
                    "epoch": ep,
                    "learning_rate": lr_before_update,
                    "learning_rate_after_validation": schedule.learning_rate,
                    "beta1": beta1_before_update,
                    "training_loss": float((epoch_loss_sum / epoch_rows).item()),
                    "validation_auc": float(auc),
                    "best_epoch": best_epoch,
                    "best_validation_auc": float(best_auc),
                    "gradient_norm_mean": float(
                        (epoch_gradient_norm_sum / epoch_steps).item()
                    ),
                    "gradient_clip_fraction": float(
                        (epoch_clipped_steps / epoch_steps).item()
                    ),
                }
                evaluations.append(observation)
                print(
                    f"[lookup_transformer] seed={self._seed} ep{ep:3d} "
                    f"valAUC={auc:.5f} best={best_auc:.5f} "
                    f"lr={lr_before_update:.8g} beta1={beta1_before_update}",
                    flush=True,
                )
                print(
                    "[lookup_transformer.training] "
                    + json.dumps(observation, ensure_ascii=False, allow_nan=False),
                    flush=True,
                )
                with torch.no_grad():
                    for p, b in zip(params_list, backup):
                        p.copy_(b)
                if self._validation_selection == "best" and bad >= self._patience:
                    break
            completed_epoch = ep + 1
            if completed_epoch in capture_completed_epochs:
                self._training_point_states[completed_epoch] = tuple(
                    weight.detach().cpu().clone() for weight in ema
                )
        observed_best_epoch = best_epoch
        if self._validation_selection == "final":
            selected_weights = ema
            selected_epoch = end_epoch
        elif best_weights is None:
            # 전체 자료 재학습에는 검증 선택이 없으므로 마지막 EMA를 그대로 쓴다.
            # 짧은 검증 학습에서 평가 시점 전에 끝나는 테스트 경로도 같은 방어선이다.
            selected_weights = ema
            selected_epoch = end_epoch
        else:
            selected_weights = best_weights
            selected_epoch = best_epoch
        with torch.no_grad():
            for p, w in zip(params_list, selected_weights):
                p.copy_(w)
        self._training_diagnostics = {
            "initialization_seed": self._seed,
            "device": self._device,
            "preprocessing_scope": self._preprocessing_scope,
            "validation_selection": self._validation_selection,
            "value_dropout": self._value_dropout,
            "value_dropout_sampler": self._value_dropout_sampler,
            "mask_pool": self._mask_pool if mask_pool is not None else None,
            "mask_pool_rows": int(mask_pool.shape[0]) if mask_pool is not None else None,
            "mask_pool_donor_columns": mask_pool_donor_columns,
            "mask_pool_missing_rate": mask_pool_rate,
            "value_dropout_alpha": mask_pool_alpha,
            "optimizer": self._optimizer_name,
            "lr_schedule": self._lr_schedule,
            "max_learning_rate": self._lr,
            "muon_lr_multiplier": self._muon_lr_multiplier,
            "muon_max_learning_rate": (
                self._lr * self._muon_lr_multiplier
                if self._optimizer_name == "muon"
                else None
            ),
            "start_learning_rate": self._lr / _START_DIVISOR,
            "nominal_min_learning_rate": self._lr / _FINAL_DIVISOR,
            "warmup_fraction": _WARMUP_FRACTION,
            "gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "evaluations": evaluations,
            "best_epoch": selected_epoch,
            "observed_best_epoch": observed_best_epoch,
            "best_validation_auc": float(best_auc) if validation else None,
            "end_epoch": end_epoch,
            "completed_steps": schedule.completed_steps,
            "planned_total_steps": steps,
            "full_fit": not validation,
        }
        if training_point_path:
            self._training_diagnostics.update(
                {
                    "trajectory_end_epochs": epochs,
                    "schedule_horizon_epochs": schedule_horizon,
                    "captured_completed_epochs": list(capture_completed_epochs),
                    "selected_completed_epoch": epochs,
                    "state_kind": "ema",
                }
            )
            self._training_point_diagnostics = dict(self._training_diagnostics)
        # 전체 자료 재학습에는 검증 선택이 없다. 없는 관측을 지어내지 않는다. (#372)
        self._raw_training_length_selection = selected_epoch if validation else None
        if not validation:
            return None
        assert ids_va is not None and num_va is not None and mask_va is not None
        assert y_va_np is not None
        val_logit = self._predict_tensors(ids_va, num_va, mask_va)
        self._val = (ids_va, num_va, mask_va, y_va_np)
        self._val_auc = roc_auc_score(y_va_np, val_logit)
        return _sigmoid(val_logit)

    def training_diagnostics(self) -> dict[str, object]:
        if self._training_diagnostics is None:
            raise RuntimeError("training_diagnostics는 fit 뒤에 호출해야 한다.")
        return self._training_diagnostics

    def raw_training_length_selection(self) -> int:
        """검증이 고른 0부터 세는 epoch 위치. 연결부가 관측 학습 길이로 바꾼다. (#372)"""
        if self._raw_training_length_selection is None:
            raise RuntimeError(
                "Lookup-Transformer 원시 epoch 위치는 검증 분할로 학습한 뒤에만 읽을 수 있다."
            )
        return self._raw_training_length_selection

    def _autocast(self):
        if self._device.startswith("cuda"):
            return torch.autocast("cuda", dtype=self._amp_dtype)
        import contextlib

        return contextlib.nullcontext()

    def _predict_tensors(
        self,
        ids: torch.Tensor,
        num: torch.Tensor,
        mask: torch.Tensor,
        chunk: int = 16384,
    ) -> np.ndarray:
        self._model.eval()
        outs = []
        with torch.no_grad(), self._autocast():
            for i in range(0, len(ids), chunk):
                outs.append(
                    self._model(ids[i : i + chunk], num[i : i + chunk], mask[i : i + chunk])
                )
        return torch.cat(outs).float().cpu().numpy().astype("float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ids, num, mask = (t.to(self._device) for t in self._encode(X))
        return _sigmoid(self._predict_tensors(ids, num, mask))

    # ---- 중요도: 검증 fold permutation(AUC 하락 폭)을 gain 축으로 (#97 규약) ----

    def importance(self) -> pd.DataFrame:
        ids, num, mask, y_va = self._val
        base = self._val_auc
        gains = []
        for j in range(len(self._columns)):
            drops = []
            for r in range(self._perm_repeats):
                g = torch.Generator().manual_seed(self._seed * 10007 + j * 101 + r)
                perm = torch.randperm(len(y_va), generator=g).to(self._device)
                ids_p, num_p, mask_p = ids.clone(), num.clone(), mask.clone()
                ids_p[:, j] = ids[perm, j]
                num_p[:, j] = num[perm, j]
                mask_p[:, j] = mask[perm, j]
                drops.append(base - roc_auc_score(y_va, self._predict_tensors(ids_p, num_p, mask_p)))
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})


class LookupTransformerFold:
    """fold 하나의 초기화 평균 학습·예측·중요도 상태. adapter가 소유한다."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        offsets = params.pop("fold_seed_offsets", [0])
        if not isinstance(offsets, (list, tuple)) or not offsets:
            raise ValueError("fold_seed_offsets는 비어 있지 않은 정수 목록이어야 한다.")
        if any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets):
            raise ValueError("fold_seed_offsets는 비어 있지 않은 정수 목록이어야 한다.")
        if len(set(offsets)) != len(offsets):
            raise ValueError(f"fold_seed_offsets에 중복이 있다: {offsets}")

        self._seed = seed
        self._perm_repeats = int(params.get("perm_repeats", 3))
        devices = self._parallel_devices(len(offsets))
        init_barrier = threading.Barrier(len(offsets)) if devices is not None else None
        self._members = [
            _LookupTransformerMember(
                params,
                seed + offset,
                device=devices[index] if devices is not None else None,
                init_barrier=init_barrier,
            )
            for index, offset in enumerate(offsets)
        ]
        self._parallel = devices is not None
        self._columns: list[str] | None = None
        self._val: tuple[pd.DataFrame, np.ndarray] | None = None
        self._val_auc: float | None = None
        self._captured_completed_epochs: tuple[int, ...] = ()
        self._selected_completed_epoch: int | None = None

    def set_dataset_reference(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        """초기화 구성원들이 공유할 목표값 비참조 전처리 기준 집합을 건넨다."""
        for member in self._members:
            member.set_dataset_reference(X_train, X_test)

    @staticmethod
    def _parallel_devices(member_count: int) -> list[str] | None:
        raw = os.environ.get("PIPELINE_FOLD_GPUS", "").strip()
        if not raw:
            return None
        try:
            gpu_ids = [int(part.strip()) for part in raw.split(",")]
        except ValueError as exc:
            raise ValueError("PIPELINE_FOLD_GPUS는 쉼표로 구분한 GPU 번호여야 한다.") from exc
        if len(gpu_ids) != member_count or len(set(gpu_ids)) != member_count:
            raise ValueError(
                f"PIPELINE_FOLD_GPUS는 구성원 {member_count}개와 같은 수의 "
                f"서로 다른 GPU여야 한다: {gpu_ids}"
            )
        if not torch.cuda.is_available() or any(
            gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids
        ):
            raise ValueError(
                f"PIPELINE_FOLD_GPUS가 사용 가능한 CUDA 장치 범위를 벗어난다: {gpu_ids}"
            )
        return [f"cuda:{gpu_id}" for gpu_id in gpu_ids]

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        self._captured_completed_epochs = ()
        self._selected_completed_epoch = None
        self._columns = list(X_va.columns)
        y_va_np = y_va.to_numpy(dtype="float64")
        val_pred = np.zeros(len(X_va), dtype="float64")
        for index, member in enumerate(self._members, start=1):
            print(
                f"[lookup_transformer] member {index}/{len(self._members)} "
                f"seed={member._seed} device={member._device}",
                flush=True,
            )
        if self._parallel:
            with ThreadPoolExecutor(max_workers=len(self._members)) as executor:
                member_preds = list(
                    executor.map(
                        lambda member: member.fit(X_tr, y_tr, X_va, y_va),
                        self._members,
                    )
                )
        else:
            member_preds = [
                member.fit(X_tr, y_tr, X_va, y_va) for member in self._members
            ]
        for member_pred in member_preds:
            val_pred += member_pred / len(self._members)
        self._val = (X_va.copy(), y_va_np)
        self._val_auc = roc_auc_score(y_va_np, val_pred)
        if len(self._members) > 1:
            print(
                f"[lookup_transformer] fold initialization-avg "
                f"valAUC={self._val_auc:.5f}",
                flush=True,
            )
        return val_pred

    def fit_training_trajectory(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        completed_epochs: object,
        trajectory_end_epochs: object,
        schedule_horizon_epochs: object,
    ) -> None:
        """한 학습 궤적에서 사전 고정한 완료 epoch의 EMA 상태를 포착한다."""
        validated = [
            member._validate_training_trajectory(
                completed_epochs,
                trajectory_end_epochs,
                schedule_horizon_epochs,
            )
            for member in self._members
        ]
        points, trajectory_end, schedule_horizon = validated[0]
        if any(result != validated[0] for result in validated[1:]):
            raise RuntimeError("초기화 구성원의 학습 시점 계약이 서로 다르다.")

        self._columns = list(X_va.columns)
        y_va_np = y_va.to_numpy(dtype="float64")
        self._val = (X_va.copy(), y_va_np)
        self._val_auc = None
        self._captured_completed_epochs = points
        self._selected_completed_epoch = None
        for index, member in enumerate(self._members, start=1):
            print(
                f"[lookup_transformer] trajectory member {index}/{len(self._members)} "
                f"seed={member._seed} device={member._device} "
                f"end={trajectory_end} horizon={schedule_horizon} "
                f"points={list(points)}",
                flush=True,
            )
        if self._parallel:
            with ThreadPoolExecutor(max_workers=len(self._members)) as executor:
                list(
                    executor.map(
                        lambda member: member._fit_training_trajectory(
                            X_tr,
                            y_tr,
                            X_va,
                            y_va,
                            completed_epochs=points,
                            trajectory_end_epochs=trajectory_end,
                            schedule_horizon_epochs=schedule_horizon,
                        ),
                        self._members,
                    )
                )
        else:
            for member in self._members:
                member._fit_training_trajectory(
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    completed_epochs=points,
                    trajectory_end_epochs=trajectory_end,
                    schedule_horizon_epochs=schedule_horizon,
                )

        # 학습 직후에도 fold 상태가 포착된 후보 하나를 가리키도록 마지막 후보를 고른다.
        self.select_training_point(points[-1])

    def select_training_point(self, completed_epoch: object) -> np.ndarray:
        """포착한 EMA 상태를 복원하고 그 상태의 검증 확률 예측을 돌려준다."""
        point = _positive_epoch(completed_epoch, "completed_epoch")
        if point not in self._captured_completed_epochs:
            raise ValueError(
                f"포착하지 않은 completed_epoch이다: {point}; "
                f"포착값={self._captured_completed_epochs}"
            )
        if self._val is None:
            raise RuntimeError("select_training_point는 검증 궤적 학습 뒤에 호출해야 한다.")
        if self._parallel:
            with ThreadPoolExecutor(max_workers=len(self._members)) as executor:
                member_predictions = list(
                    executor.map(
                        lambda member: member._select_training_point(point),
                        self._members,
                    )
                )
        else:
            member_predictions = [
                member._select_training_point(point) for member in self._members
            ]

        validation_prediction = np.zeros(len(self._val[1]), dtype="float64")
        for member_prediction in member_predictions:
            validation_prediction += member_prediction / len(self._members)
        self._val_auc = roc_auc_score(self._val[1], validation_prediction)
        self._selected_completed_epoch = point
        if len(self._members) > 1:
            print(
                f"[lookup_transformer] fold initialization-avg "
                f"completed_epoch={point} valAUC={self._val_auc:.5f}",
                flush=True,
            )
        return validation_prediction

    def fit_full(self, X: pd.DataFrame, y: pd.Series, epochs: int) -> None:
        """초기화 구성원 모두를 전체 자료에서 같은 고정 epoch 수로 학습한다."""
        self.fit_full_member_epochs(X, y, (epochs,) * len(self._members))

    def fit_full_member_epochs(
        self, X: pd.DataFrame, y: pd.Series, member_epochs: tuple[int, ...]
    ) -> None:
        """초기화 구성원마다 출처 실행에서 고정한 epoch 수로 학습한다."""
        if len(member_epochs) != len(self._members) or any(
            isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
            for epochs in member_epochs
        ):
            raise ValueError(
                "Lookup-Transformer 구성원별 epoch 수가 초기화 구성원과 맞지 않는다."
            )
        self._captured_completed_epochs = ()
        self._selected_completed_epoch = None
        self._columns = list(X.columns)
        for index, (member, epochs) in enumerate(
            zip(self._members, member_epochs), start=1
        ):
            print(
                f"[lookup_transformer] full member {index}/{len(self._members)} "
                f"seed={member._seed} device={member._device} epochs={epochs}",
                flush=True,
            )
        if self._parallel:
            with ThreadPoolExecutor(max_workers=len(self._members)) as executor:
                list(
                    executor.map(
                        lambda item: item[0].fit_full(X, y, item[1]),
                        zip(self._members, member_epochs),
                    )
                )
        else:
            for member, epochs in zip(self._members, member_epochs):
                member.fit_full(X, y, epochs)

    def fit_full_training_point(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        trajectory_end_epochs: object,
        schedule_horizon_epochs: object,
    ) -> None:
        """선택한 완료 시점에서 끝내되 원래 학습률 일정 지평을 보존한다."""
        trajectory_end, schedule_horizon = _validate_epoch_window(
            trajectory_end_epochs, schedule_horizon_epochs
        )
        for member in self._members:
            member._validate_training_point_path()

        self._captured_completed_epochs = ()
        self._selected_completed_epoch = trajectory_end
        self._columns = list(X.columns)
        for index, member in enumerate(self._members, start=1):
            print(
                f"[lookup_transformer] full training-point member "
                f"{index}/{len(self._members)} seed={member._seed} "
                f"device={member._device} end={trajectory_end} "
                f"horizon={schedule_horizon}",
                flush=True,
            )
        if self._parallel:
            with ThreadPoolExecutor(max_workers=len(self._members)) as executor:
                list(
                    executor.map(
                        lambda member: member._fit_full_training_point(
                            X,
                            y,
                            trajectory_end_epochs=trajectory_end,
                            schedule_horizon_epochs=schedule_horizon,
                        ),
                        self._members,
                    )
                )
        else:
            for member in self._members:
                member._fit_full_training_point(
                    X,
                    y,
                    trajectory_end_epochs=trajectory_end,
                    schedule_horizon_epochs=schedule_horizon,
                )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._parallel:
            with ThreadPoolExecutor(max_workers=len(self._members)) as executor:
                member_predictions = list(
                    executor.map(lambda member: member.predict(X), self._members)
                )
        else:
            member_predictions = [member.predict(X) for member in self._members]

        pred = np.zeros(len(X), dtype="float64")
        for member_prediction in member_predictions:
            pred += member_prediction / len(self._members)
        return pred

    def importance(self) -> pd.DataFrame:
        if len(self._members) == 1:
            return self._members[0].importance()

        X_va, y_va = self._val
        base = self._val_auc
        gains = []
        for j, col in enumerate(self._columns):
            drops = []
            for r in range(self._perm_repeats):
                rng = np.random.default_rng(self._seed * 10007 + j * 101 + r)
                X_p = X_va.copy()
                X_p[col] = X_va[col].take(rng.permutation(len(X_va))).set_axis(X_p.index)
                drops.append(base - roc_auc_score(y_va, self.predict(X_p)))
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})

    def training_diagnostics(self) -> dict[str, object]:
        return {
            "fold_initialization_members": [
                member.training_diagnostics() for member in self._members
            ]
        }

    def raw_training_length_selections(self) -> tuple[int, ...]:
        """초기화 구성원 순서대로 원시 epoch 위치를 전부 돌려준다. (#372)"""
        return tuple(
            member.raw_training_length_selection() for member in self._members
        )
