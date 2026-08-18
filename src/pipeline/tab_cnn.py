"""Kaggle 공개 표 합성곱망을 저장소 규율에 맞게 고친 구현. (#177)

원문: https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction?scriptVersionId=342747549
원문 소스 SHA-256: 2310c4fa1b98230989f8e3bcf3f9661985a2c30df90597786e739cd34321f4dc

원 공개 노트북 소스에는 Apache License 2.0이 적용된다.
이 저장소의 변경 사항은 다음과 같다.

- 커밋된 outer fold와 ``ModelAdapter`` 경계를 사용한다.
- 정확값 빈도와 목표 인코딩을 제거하고 champion 피처 계획만 받는다.
- 범주 어휘와 분위수 변환을 outer 학습 부분에서만 맞춘다.
- fold별 초기화 시드를 실행 시드와 학습 행 식별자에서 결정적으로 파생한다.
- 최대 학습 길이를 진입 진단용 30 epoch로 제한한다.
- 합성곱 M0와 매개변수 규모를 맞춘 완전 연결 A0를 함께 제공한다.
- 검증 fold의 결정적 순열 중요도와 공통 진입 진단 관측값을 제공한다.

Apache License 2.0 원문은 ``contextualized_spline_transformer.LICENSE``에 있다.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import math
import os
import random
import time
from dataclasses import dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from .model import AdapterDiagnostics

_INTERACTION_MODES = {"convolution", "dense"}


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@dataclass(frozen=True)
class _ColumnSpec:
    name: str
    kind: str
    vocabulary: tuple[object, ...] = ()


class _FoldQuantileEncoder:
    """학습 fold에서만 범주 어휘와 열별 경험 누적분포를 맞춘다."""

    def __init__(self, n_quantiles: int, seed: int) -> None:
        self._n_quantiles = n_quantiles
        self._seed = seed
        self.columns: list[str] | None = None
        self.specs: list[_ColumnSpec] = []
        self.quantile: QuantileTransformer | None = None
        self.fit_rows: int | None = None

    @staticmethod
    def _object_values(series: pd.Series) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            return series.astype(object)
        return series

    def fit(self, X: pd.DataFrame) -> None:
        if not len(X):
            raise ValueError("표 합성곱망 전처리에는 학습 행이 필요하다.")
        self.columns = list(X.columns)
        if not self.columns:
            raise ValueError("표 합성곱망 입력 열이 비어 있다.")
        self.specs = []
        for name in self.columns:
            values = X[name]
            if pd.api.types.is_numeric_dtype(values) and not isinstance(
                values.dtype, pd.CategoricalDtype
            ):
                self.specs.append(_ColumnSpec(name=name, kind="numeric"))
                continue
            objects = self._object_values(values)
            vocabulary = tuple(
                sorted(pd.unique(objects.dropna()).tolist(), key=repr)
            )
            self.specs.append(
                _ColumnSpec(name=name, kind="categorical", vocabulary=vocabulary)
            )

        raw = self.raw_transform(X)
        quantiles = min(self._n_quantiles, len(X))
        self.quantile = QuantileTransformer(
            n_quantiles=quantiles,
            output_distribution="normal",
            random_state=self._seed,
            subsample=None,
            copy=True,
        )
        self.quantile.fit(raw)
        self.fit_rows = len(X)

    def raw_transform(self, X: pd.DataFrame) -> np.ndarray:
        if list(X.columns) != self.columns:
            raise AssertionError("표 합성곱망 입력 열이 학습 때와 다르다.")
        output = np.zeros((len(X), len(self.specs)), dtype="float32")
        for column_index, spec in enumerate(self.specs):
            values = X[spec.name]
            if spec.kind == "numeric":
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(
                    dtype="float64"
                )
                output[:, column_index] = np.nan_to_num(
                    numeric, nan=0.0, posinf=0.0, neginf=0.0
                ).astype("float32")
                continue

            objects = self._object_values(values)
            mapping = {
                value: value_index + 1
                for value_index, value in enumerate(spec.vocabulary)
            }
            unknown_id = len(spec.vocabulary) + 1
            ids = objects.map(mapping).fillna(unknown_id).to_numpy(
                dtype="int64", copy=True
            )
            ids[objects.isna().to_numpy()] = 0
            output[:, column_index] = ids.astype("float32")
        return output

    def transform(self, X: pd.DataFrame) -> torch.Tensor:
        if self.quantile is None:
            raise RuntimeError("표 합성곱망 전처리를 먼저 학습해야 한다.")
        transformed = self.quantile.transform(self.raw_transform(X))
        if not np.isfinite(transformed).all():
            raise RuntimeError("분위수 변환 결과에 유한하지 않은 값이 있다.")
        return torch.from_numpy(transformed.astype("float32", copy=False))


class _PiecewiseLinearEmbedding(nn.Module):
    """원문 이름을 보존하되 실제 연산은 학습 가능한 ReLU 경첩 기저다."""

    def __init__(self, features: int, bins: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, bins) * 0.1)
        self.bias = nn.Parameter(torch.zeros(features, bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(
            x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        )


class _PeriodicEmbedding(nn.Module):
    def __init__(self, features: int, output_dim: int, sigma: float) -> None:
        super().__init__()
        frequencies = output_dim // 2
        self.coefficient = nn.Parameter(
            torch.randn(features, frequencies) * sigma
        )
        self.phase = nn.Parameter(torch.zeros(features, frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = (
            2
            * math.pi
            * x.unsqueeze(-1)
            * self.coefficient.unsqueeze(0)
            + self.phase.unsqueeze(0)
        )
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class _SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = channels // reduction
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.Mish(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(self.pool(x).squeeze(-1)).unsqueeze(-1)
        return x * weight


class _ConvolutionInteraction(nn.Module):
    output_dim = 256

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(128)
        self.se = _SEBlock1D(128)
        self.activation = nn.Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = embedding.transpose(1, 2)
        x = self.dropout(self.activation(self.norm1(self.conv1(x))))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.se(x)
        return torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)


def _matched_dense_hidden(features: int, target_parameters: int) -> int:
    """완전 연결 경로의 학습 매개변수를 합성곱 경로에 가장 가깝게 맞춘다."""
    coefficient = 32 * features + 259
    estimate = max(1, round((target_parameters - 768) / coefficient))
    candidates = range(max(1, estimate - 2), estimate + 3)
    return min(
        candidates,
        key=lambda hidden: abs(hidden * coefficient + 768 - target_parameters),
    )


class _DenseInteraction(nn.Module):
    output_dim = 256

    def __init__(self, features: int, target_parameters: int, dropout: float) -> None:
        super().__init__()
        hidden = _matched_dense_hidden(features, target_parameters)
        self.hidden = hidden
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features * 32, hidden),
            nn.BatchNorm1d(hidden),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Mish(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.layers(embedding)


class _MultiSampleDropoutHead(nn.Module):
    def __init__(self, features: int, samples: int, dropout: float) -> None:
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(samples)])
        self.output = nn.Linear(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack(
            [self.output(dropout(x)) for dropout in self.dropouts], dim=0
        )
        return logits.mean(dim=0).squeeze(1)


class _TabCNN(nn.Module):
    def __init__(
        self,
        features: int,
        interaction_mode: str,
        *,
        plr_bins: int,
        periodic_dim: int,
        periodic_sigma: float,
        dropout: float,
        head_samples: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.piecewise = _PiecewiseLinearEmbedding(features, plr_bins)
        self.periodic = _PeriodicEmbedding(features, periodic_dim, periodic_sigma)
        self.projection = nn.Sequential(
            nn.Linear(plr_bins + periodic_dim, 32), nn.Mish()
        )

        reference_parameters = _parameter_count(_ConvolutionInteraction(dropout))
        if interaction_mode == "convolution":
            self.interaction = _ConvolutionInteraction(dropout)
        else:
            self.interaction = _DenseInteraction(
                features, reference_parameters, dropout
            )
        self.reference_interaction_parameters = reference_parameters
        self.interaction_parameters = _parameter_count(self.interaction)

        self.backbone = nn.Sequential(
            nn.Linear(self.interaction.output_dim + features, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
        )
        self.head = _MultiSampleDropoutHead(64, head_samples, head_dropout)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        embedding = self.projection(
            torch.cat([self.piecewise(raw), self.periodic(raw)], dim=-1)
        )
        interacted = self.interaction(embedding)
        return self.head(self.backbone(torch.cat([interacted, raw], dim=1)))


class TabCNNFold:
    """fold 하나의 전처리, M0/A0 학습, 예측과 중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._interaction_mode = str(params.pop("interaction_mode", "convolution"))
        self._plr_bins = int(params.pop("plr_bins", 12))
        self._periodic_dim = int(params.pop("periodic_dim", 12))
        self._periodic_sigma = float(params.pop("periodic_sigma", 2.33))
        self._dropout = float(params.pop("dropout", 0.08))
        self._head_samples = int(params.pop("head_samples", 5))
        self._head_dropout = float(params.pop("head_dropout", 0.2))
        self._epochs = int(params.pop("epochs", 30))
        self._patience = int(params.pop("patience", 15))
        self._batch_size = int(params.pop("batch_size", 128))
        self._eval_batch_size = int(params.pop("eval_batch_size", 8192))
        self._lr = float(params.pop("lr", 0.002))
        self._weight_decay = float(params.pop("weight_decay", 0.015))
        self._label_smoothing = float(params.pop("label_smoothing", 0.01))
        self._scheduler_t0 = int(params.pop("scheduler_t0", 15))
        self._n_quantiles = int(params.pop("n_quantiles", 1000))
        self._perm_sample = int(params.pop("perm_sample", 8192))
        self._perm_repeats = int(params.pop("perm_repeats", 1))
        requested_device = params.pop("device", None)
        if params:
            raise ValueError(f"tab_cnn이 모르는 params: {sorted(params)}")
        if self._interaction_mode not in _INTERACTION_MODES:
            raise ValueError(
                f"interaction_mode는 {sorted(_INTERACTION_MODES)} 중 하나여야 한다."
            )
        if self._periodic_dim < 2 or self._periodic_dim % 2:
            raise ValueError("periodic_dim은 2 이상의 짝수여야 한다.")
        if min(
            self._plr_bins,
            self._head_samples,
            self._epochs,
            self._patience,
            self._batch_size,
            self._eval_batch_size,
            self._scheduler_t0,
            self._n_quantiles,
            self._perm_sample,
            self._perm_repeats,
        ) <= 0:
            raise ValueError("tab_cnn 크기와 학습 횟수 설정은 양수여야 한다.")
        if not 0 <= self._dropout < 1 or not 0 <= self._head_dropout < 1:
            raise ValueError("dropout은 0 이상 1 미만이어야 한다.")
        if not 0 <= self._label_smoothing < 1:
            raise ValueError("label_smoothing은 0 이상 1 미만이어야 한다.")
        if self._lr <= 0 or self._weight_decay < 0 or self._periodic_sigma <= 0:
            raise ValueError("lr·periodic_sigma는 양수이고 weight_decay는 0 이상이어야 한다.")

        self._seed = seed
        if requested_device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = str(requested_device)
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("tab_cnn device=cuda인데 CUDA를 사용할 수 없다.")
        if self._device == "cpu":
            torch.set_num_threads(1)

        self._encoder = _FoldQuantileEncoder(self._n_quantiles, seed)
        self._model: _TabCNN | None = None
        self._importance_X: pd.DataFrame | None = None
        self._importance_y: np.ndarray | None = None
        self._importance_base_auc: float | None = None
        self._diagnostics = AdapterDiagnostics()
        self._fit_seconds: float | None = None
        self._importance_seconds: float | None = None
        self._prediction_calls = 0
        self._all_predictions_finite = True

    def _autocast(self):
        if self._device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _fold_seed(self, index: pd.Index) -> int:
        hashed = pd.util.hash_pandas_object(index, index=False).to_numpy(
            dtype="uint64"
        )
        digest = hashlib.sha256(hashed.tobytes()).digest()
        fold_component = int.from_bytes(digest[:8], "little")
        return int((self._seed * 1_000_003 + fold_component) % (2**31 - 1))

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        fit_started = time.monotonic()
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._device)
            torch.cuda.synchronize(self._device)
        fold_seed = self._fold_seed(X_tr.index)
        self._seed_everything(fold_seed)
        self._encoder.fit(X_tr)
        train = self._encoder.transform(X_tr).to(self._device)
        validation = self._encoder.transform(X_va).to(self._device)
        target = torch.from_numpy(y_tr.to_numpy(dtype="float32")).to(self._device)
        smooth_target = (
            target * (1.0 - self._label_smoothing)
            + 0.5 * self._label_smoothing
        )
        validation_target = y_va.to_numpy(dtype="float64")

        model = _TabCNN(
            train.shape[1],
            self._interaction_mode,
            plr_bins=self._plr_bins,
            periodic_dim=self._periodic_dim,
            periodic_sigma=self._periodic_sigma,
            dropout=self._dropout,
            head_samples=self._head_samples,
            head_dropout=self._head_dropout,
        ).to(self._device)
        self._model = model
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self._scheduler_t0, T_mult=1, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=self._device.startswith("cuda")
        )
        generator = torch.Generator().manual_seed(fold_seed)
        best_auc = -math.inf
        best_epoch = -1
        best_weights: dict[str, torch.Tensor] | None = None
        stale = 0
        epoch_seconds: list[float] = []
        validation_aucs: list[float] = []
        training_losses: list[float] = []

        for epoch in range(1, self._epochs + 1):
            started = time.monotonic()
            model.train()
            permutation = torch.randperm(len(train), generator=generator)
            usable = (len(permutation) // self._batch_size) * self._batch_size
            if usable == 0:
                usable = len(permutation)
            permutation = permutation[:usable]
            loss_sum = 0.0
            batches = 0
            for offset in range(0, usable, self._batch_size):
                rows = permutation[offset : offset + self._batch_size].to(
                    self._device
                )
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logit = model(train[rows])
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logit, smooth_target[rows]
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("tab_cnn 학습 손실에 유한하지 않은 값이 생겼다.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach())
                batches += 1

            validation_logit = self._predict_tensor(validation)
            validation_auc = float(
                roc_auc_score(validation_target, validation_logit)
            )
            scheduler.step()
            elapsed = time.monotonic() - started
            training_loss = loss_sum / batches
            epoch_seconds.append(float(elapsed))
            validation_aucs.append(validation_auc)
            training_losses.append(training_loss)
            print(
                f"[tab_cnn] mode={self._interaction_mode} epoch={epoch:02d} "
                f"loss={training_loss:.6f} val_auc={validation_auc:.6f} "
                f"best={max(best_auc, validation_auc):.6f} seconds={elapsed:.2f}",
                flush=True,
            )
            if validation_auc > best_auc:
                best_auc = validation_auc
                best_epoch = epoch
                best_weights = copy.deepcopy(
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                )
                stale = 0
            else:
                stale += 1
            if stale >= self._patience:
                break

        if best_weights is None:
            raise RuntimeError("tab_cnn 검증 checkpoint를 만들지 못했다.")
        model.load_state_dict(best_weights)
        validation_logit = self._predict_tensor(validation)
        validation_prediction = _sigmoid(validation_logit)

        sample_size = min(self._perm_sample, len(X_va))
        rng = np.random.default_rng(self._seed)
        sample_positions = np.sort(
            rng.choice(len(X_va), size=sample_size, replace=False)
        )
        self._importance_X = X_va.iloc[sample_positions].copy()
        self._importance_y = validation_target[sample_positions]
        self._importance_base_auc = float(
            roc_auc_score(self._importance_y, self.predict(self._importance_X))
        )
        relative_difference = abs(
            model.interaction_parameters - model.reference_interaction_parameters
        ) / model.reference_interaction_parameters
        target_or_frequency = [
            name
            for name in X_tr.columns
            if name.endswith("_te") or name.endswith("_freq")
        ]
        self._diagnostics = AdapterDiagnostics(
            assertions={
                "preprocessing_training_rows_only": self._encoder.fit_rows
                == len(X_tr),
                "validation_labels_excluded_from_preprocessing": True,
                "target_and_frequency_encodings_absent": not target_or_frequency,
                "placebo_feature_present": "placebo_noise" in X_tr.columns,
                "interaction_parameter_scale_matched": relative_difference < 0.05,
            },
            observations={
                "interaction_mode": self._interaction_mode,
                "preprocessing_fit_rows": self._encoder.fit_rows,
                "training_rows": len(X_tr),
                "validation_rows": len(X_va),
                "input_columns": list(X_tr.columns),
                "input_feature_count": len(X_tr.columns),
                "fold_initialization_seed": fold_seed,
                "quantile_count": min(self._n_quantiles, len(X_tr)),
                "quantile_subsample": None,
                "model_parameter_count": _parameter_count(model),
                "reference_convolution_interaction_parameters": (
                    model.reference_interaction_parameters
                ),
                "used_interaction_parameters": model.interaction_parameters,
                "interaction_parameter_relative_difference": relative_difference,
                "epoch_seconds": epoch_seconds,
                "training_losses": training_losses,
                "epoch_validation_aucs": validation_aucs,
                "best_epoch": best_epoch,
                "best_validation_auc": best_auc,
                "importance_rows": sample_size,
                "importance_repeats": self._perm_repeats,
                "prediction_source_dtype": "float32",
                "source_script_version_id": 342747549,
                "source_sha256": (
                    "2310c4fa1b98230989f8e3bcf3f9661985a2c30df90597786e739cd34321f4dc"
                ),
            },
        )
        self._fit_seconds = float(time.monotonic() - fit_started)
        return validation_prediction

    def _predict_tensor(self, encoded: torch.Tensor) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("tab_cnn을 먼저 학습해야 한다.")
        self._model.eval()
        outputs: list[torch.Tensor] = []
        with torch.no_grad(), self._autocast():
            for offset in range(0, len(encoded), self._eval_batch_size):
                outputs.append(
                    self._model(encoded[offset : offset + self._eval_batch_size])
                )
        return torch.cat(outputs).float().cpu().numpy().astype("float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self._encoder.transform(X).to(self._device)
        prediction = _sigmoid(self._predict_tensor(encoded))
        self._prediction_calls += 1
        self._all_predictions_finite = bool(
            self._all_predictions_finite and np.isfinite(prediction).all()
        )
        return prediction

    def importance(self) -> pd.DataFrame:
        if (
            self._importance_X is None
            or self._importance_y is None
            or self._importance_base_auc is None
        ):
            raise RuntimeError("importance는 tab_cnn fit 뒤에 호출해야 한다.")
        importance_started = time.monotonic()
        gains: list[float] = []
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
        result = pd.DataFrame(
            {"feature": list(self._importance_X.columns), "gain": gains}
        )
        self._importance_seconds = float(time.monotonic() - importance_started)
        self._diagnostics.observations.update(
            {
                "importance_feature_count": len(result),
                "importance_values_finite": bool(np.isfinite(result["gain"]).all()),
                "placebo_importance": float(
                    result.loc[result["feature"] == "placebo_noise", "gain"].iloc[0]
                ),
            }
        )
        return result

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics

    def training_diagnostics(self) -> dict[str, object]:
        observations = dict(self._diagnostics.observations)
        observations.update(
            {
                "integrity_assertions": dict(self._diagnostics.assertions),
                "prediction_calls": self._prediction_calls,
                "all_predictions_finite": self._all_predictions_finite,
                "fit_seconds": self._fit_seconds,
                "importance_seconds": self._importance_seconds,
                "fold_adapter_seconds": (
                    None
                    if self._fit_seconds is None or self._importance_seconds is None
                    else self._fit_seconds + self._importance_seconds
                ),
                "cuda_max_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(self._device))
                    if self._device.startswith("cuda")
                    else None
                ),
                "cuda_max_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(self._device))
                    if self._device.startswith("cuda")
                    else None
                ),
                "cuda_device_total_bytes": (
                    int(torch.cuda.get_device_properties(self._device).total_memory)
                    if self._device.startswith("cuda")
                    else None
                ),
            }
        )
        return observations
