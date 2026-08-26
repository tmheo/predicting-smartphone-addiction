"""고정 4 epoch RealMLP의 fold 안전 재현. (#180)

원문: https://www.kaggle.com/code/beicicc/s6e8-fold-safe-realmlp
2026-08-19에 받은 원문 노트북 SHA-256:
``60a0bd05332e8932468d9cc796855013be3c3798344fd75c15c016764eba58ef``

원 공개 노트북 소스에는 Apache License 2.0이 적용된다.
이 저장소의 변경 사항은 다음과 같다.

- 커밋된 outer fold와 ``ModelAdapter`` 경계를 사용한다.
- 파이프라인의 필수 플라시보 열은 원문 파생 없이 수치 입력으로 보존한다.
- 두 초기화의 예측을 fold 안에서 평균하고 파이프라인 시드 평균과 구분한다.
- 검증 목표값을 학습이나 checkpoint 선택에 쓰지 않고 고정 4 epoch 끝 상태를 쓴다.
- 원시 입력 열 기준 순열 중요도와 구조화된 학습 관측을 제공한다.
- 전체 자료 재학습 경로에서도 내부 OOF 목표 인코딩과 고정 학습 길이를 유지한다.

Apache License 2.0 원문은 ``contextualized_spline_transformer.LICENSE``에 있다.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 결정론 모드의 CUDA 행렬곱이 요구하는 프로세스 설정이다.
# PyTorch를 불러오기 전에 고정해야 첫 CuBLAS 호출에도 적용된다.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    KBinsDiscretizer,
    QuantileTransformer,
    TargetEncoder,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from .data import TARGET
from .model import AdapterDiagnostics


SOURCE_NOTEBOOK_URL = "https://www.kaggle.com/code/beicicc/s6e8-fold-safe-realmlp"
SOURCE_NOTEBOOK_SHA256 = (
    "60a0bd05332e8932468d9cc796855013be3c3798344fd75c15c016764eba58ef"
)
SOURCE_KERNEL_ID = 129554888

RAW_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
RAW_NUMERICAL = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
REFERENCE_QNORMAL_SUFFIX = "_reference_qnormal"

_DEFAULTS = {
    "optimizer": "adamw",
    "n_ens": 8,
    "embed_dim": 8,
    "onehot_thresh": 8,
    "hidden_dims": [512, 512, 512],
    "dropout": 0.06,
    "p_drop_sched": "expm4t",
    "pbld_hidden_dim": 20,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 5.0,
    "pbld_lr_factor": 0.093,
    "lr": 0.01,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.3,
    "first_layer_lr_factor": 1.0,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 0.013,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "ema_decay": 0.997875,
    "grad_clip": 1.2,
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale"],
    "fixed_epochs": 4,
    "schedule_epochs": 8,
    "batch_size": 256,
    "eval_batch_size": 10240,
    "n_init_avg": 2,
    "init_seed_stride": 1000,
    "inner_folds": 5,
    "perm_sample": 8192,
    "perm_repeats": 1,
    "reference_qnormal_columns": [],
    "preprocessing_scope": "fold_train",
    "device": "cuda",
    "verbosity": 1,
}


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _validated_completed_epochs(
    completed_epochs: Sequence[int],
    *,
    fixed_epochs: int,
    schedule_epochs: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if isinstance(completed_epochs, (str, bytes)) or not isinstance(
        completed_epochs, Sequence
    ):
        raise ValueError("completed_epochs는 양의 정수 목록이어야 한다.")
    epochs = tuple(completed_epochs)
    if not epochs and not allow_empty:
        raise ValueError("completed_epochs는 비어 있지 않은 양의 정수 목록이어야 한다.")
    if any(
        isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
        for epoch in epochs
    ):
        raise ValueError("completed_epochs는 양의 정수 목록이어야 한다.")
    if epochs != tuple(sorted(set(epochs))):
        raise ValueError("completed_epochs는 중복 없는 오름차순이어야 한다.")
    if epochs and epochs[-1] > min(fixed_epochs, schedule_epochs):
        raise ValueError(
            "completed_epochs는 fixed_epochs와 schedule_epochs 이하여야 한다: "
            f"{epochs[-1]} > {fixed_epochs}/{schedule_epochs}"
        )
    return epochs


class _FoldFeatureEngineer:
    """outer 학습 부분에서만 결측 대체, 어휘와 bin 경계를 맞춘다."""

    BIN_CONFIG = {
        "daily_screen_time_hours": 10,
        "social_media_hours": 10,
    }

    def __init__(
        self,
        reference_qnormal_columns: list[str] | None = None,
        reference_seed: int = 42,
    ) -> None:
        self.reference_qnormal_columns = list(reference_qnormal_columns or [])
        self.reference_qnormal_output_columns = [
            f"{column}{REFERENCE_QNORMAL_SUFFIX}"
            for column in self.reference_qnormal_columns
        ]
        self.reference_seed = reference_seed
        self.reference_qnormal_estimators: dict[str, QuantileTransformer] = {}
        self.reference_qnormal_fit_rows: dict[str, int] = {}
        self.reference_qnormal_reference_rows = 0

    def fit(
        self,
        frame: pd.DataFrame,
        reference_frame: pd.DataFrame | None = None,
    ) -> _FoldFeatureEngineer:
        missing = sorted(set(RAW_CATEGORICAL + RAW_NUMERICAL) - set(frame.columns))
        if missing:
            raise ValueError(f"RealMLP 원문 입력 열이 없다: {missing}")
        collisions = sorted(
            set(self.reference_qnormal_output_columns) & set(frame.columns)
        )
        if collisions:
            raise ValueError(f"RealMLP 기준 집합 값 좌표 열이 입력과 충돌한다: {collisions}")
        unexpected_categorical = [
            column
            for column in frame.columns
            if column not in RAW_CATEGORICAL
            and not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if unexpected_categorical:
            raise ValueError(
                f"RealMLP가 모르는 범주 입력 열이다: {unexpected_categorical}"
            )
        self.input_columns = list(frame.columns)
        self.passthrough_numeric = [
            column
            for column in self.input_columns
            if column not in RAW_CATEGORICAL + RAW_NUMERICAL
        ]
        self.medians = {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in RAW_NUMERICAL + self.passthrough_numeric
        }
        if not all(np.isfinite(value) for value in self.medians.values()):
            raise ValueError("RealMLP 학습 부분에서 중앙값을 정할 수 없는 수치 열이 있다.")

        self.category_maps: dict[str, dict[object, int]] = {}
        self.category_dims_by_name: dict[str, int] = {}
        self.mapped_category_columns: list[str] = []
        for column in RAW_CATEGORICAL:
            values = frame[column].astype("string").fillna("missing").astype(str)
            vocabulary = pd.Index(pd.unique(values))
            self.category_maps[column] = {
                value: index + 1 for index, value in enumerate(vocabulary)
            }
            self.category_dims_by_name[column] = len(vocabulary) + 1
            self.mapped_category_columns.append(column)

        for column in RAW_NUMERICAL:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(
                self.medians[column]
            )
            name = f"{column}_cat_"
            vocabulary = pd.Index(pd.unique(values))
            self.category_maps[name] = {
                value: index + 1 for index, value in enumerate(vocabulary)
            }
            self.category_dims_by_name[name] = len(vocabulary) + 1
            self.mapped_category_columns.append(name)

        self.bin_estimators: dict[str, tuple[str, KBinsDiscretizer]] = {}
        for column, n_bins in self.BIN_CONFIG.items():
            estimator = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            values = pd.to_numeric(frame[column], errors="coerce").fillna(
                self.medians[column]
            )
            estimator.fit(values.to_frame())
            name = f"{column}_{n_bins}_quantile_bin_"
            self.bin_estimators[name] = (column, estimator)
            self.category_dims_by_name[name] = int(estimator.n_bins_[0])

        if self.reference_qnormal_columns:
            if reference_frame is None:
                raise ValueError("RealMLP 기준 집합 값 좌표에는 전처리 기준 집합이 필요하다.")
            missing_reference = sorted(
                set(self.reference_qnormal_columns) - set(reference_frame.columns)
            )
            if missing_reference:
                raise ValueError(
                    "RealMLP 전처리 기준 집합에 원시 수치 열이 없다: "
                    f"{missing_reference}"
                )
            self.reference_qnormal_reference_rows = len(reference_frame)
            for column in self.reference_qnormal_columns:
                values = (
                    pd.to_numeric(reference_frame[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy(dtype="float64")
                )
                if len(values) == 0:
                    raise ValueError(
                        f"RealMLP 전처리 기준 집합에서 관측값이 없는 열이다: {column}"
                    )
                estimator = QuantileTransformer(
                    n_quantiles=min(1000, len(values)),
                    output_distribution="normal",
                    subsample=2_000_000_000,
                    random_state=self.reference_seed,
                )
                estimator.fit(values.reshape(-1, 1))
                self.reference_qnormal_estimators[column] = estimator
                self.reference_qnormal_fit_rows[column] = len(values)

        missing_columns = [f"_miss_{column}" for column in RAW_NUMERICAL]
        numeric_category_columns = [f"{column}_cat_" for column in RAW_NUMERICAL]
        bin_columns = list(self.bin_estimators)
        self.output_cat_cols = sorted(
            RAW_CATEGORICAL
            + missing_columns
            + numeric_category_columns
            + bin_columns
        )
        self.output_columns = sorted(
            self.input_columns
            + self.reference_qnormal_output_columns
            + missing_columns
            + numeric_category_columns
            + bin_columns
        )
        for column in missing_columns:
            self.category_dims_by_name[column] = 2
        self.cat_dims = [
            self.category_dims_by_name[column] for column in self.output_cat_cols
        ]
        self.fit_rows = len(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if list(frame.columns) != self.input_columns:
            raise AssertionError("RealMLP 입력 열이 학습 때와 다르다.")
        output = frame.copy()
        for column in RAW_NUMERICAL:
            values = pd.to_numeric(output[column], errors="coerce")
            output[f"_miss_{column}"] = values.isna().astype("int32")
        for column, estimator in self.reference_qnormal_estimators.items():
            values = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            observed = values.notna().to_numpy()
            coordinates = np.zeros(len(frame), dtype="float64")
            if observed.any():
                coordinates[observed] = estimator.transform(
                    values.loc[values.notna()].to_numpy(dtype="float64").reshape(-1, 1)
                ).ravel()
            output[f"{column}{REFERENCE_QNORMAL_SUFFIX}"] = coordinates
        for column in RAW_CATEGORICAL:
            values = output[column].astype("string").fillna("missing").astype(str)
            output[column] = (
                values.map(self.category_maps[column]).fillna(0).astype("int32")
            )
        for column in RAW_NUMERICAL + self.passthrough_numeric:
            output[column] = (
                pd.to_numeric(output[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.medians[column])
            )
        # 어휘와 bin 경계는 fit의 float64 값으로 만들어졌으므로, float32 형
        # 변환은 매핑과 bin 변환이 끝난 뒤에 해야 정확값 범주가 살아남는다. (#243)
        for column in RAW_NUMERICAL:
            name = f"{column}_cat_"
            output[name] = (
                output[column]
                .map(self.category_maps[name])
                .fillna(0)
                .astype("int32")
            )
        for name, (column, estimator) in self.bin_estimators.items():
            output[name] = (
                estimator.transform(output[[column]]).ravel().astype("int32")
            )
        for column in (
            RAW_NUMERICAL
            + self.passthrough_numeric
            + self.reference_qnormal_output_columns
        ):
            output[column] = output[column].astype("float32")
        output = output.reindex(self.output_columns, axis=1)
        if output.isna().any().any():
            raise RuntimeError("RealMLP fold 전처리 결과에 결측이 남았다.")
        for column, dimension in zip(self.output_cat_cols, self.cat_dims):
            if not output[column].between(0, dimension - 1).all():
                raise RuntimeError(f"RealMLP 범주 코드가 범위를 벗어났다: {column}")
        return output

    def fit_transform(
        self,
        frame: pd.DataFrame,
        reference_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.fit(frame, reference_frame).transform(frame)

    def unknown_value_count(self, transformed: pd.DataFrame) -> int:
        return int(
            sum(
                (transformed[column] == 0).sum()
                for column in self.mapped_category_columns
            )
        )


class _FoldTargetEncoder:
    """outer 학습 행에는 내부 OOF, 나머지 행에는 전체 학습 표를 적용한다."""

    def __init__(self, inner_folds: int, seed: int) -> None:
        self.inner_folds = inner_folds
        self.seed = seed
        self.encoder: TargetEncoder | None = None
        self.target_columns: list[str] = []
        self.output_names: list[str] = []
        self.fit_rows: int | None = None

    def fit_transform(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
        categorical_columns: list[str],
    ) -> pd.DataFrame:
        self.target_columns = [
            column
            for column in categorical_columns
            if not column.endswith("_bin_")
        ]
        self.output_names = [f"_{column}TE" for column in self.target_columns]
        self.encoder = TargetEncoder(
            categories="auto",
            target_type="binary",
            smooth="auto",
            cv=StratifiedKFold(
                n_splits=self.inner_folds,
                shuffle=True,
                random_state=self.seed,
            ),
        )
        values = self.encoder.fit_transform(
            frame[self.target_columns], target.to_numpy(dtype="int64")
        )
        output = frame.copy()
        output[self.output_names] = values
        self.fit_rows = len(frame)
        self._validate(output)
        return output

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.encoder is None:
            raise RuntimeError("RealMLP 목표 인코더를 먼저 학습해야 한다.")
        output = frame.copy()
        output[self.output_names] = self.encoder.transform(
            frame[self.target_columns]
        )
        self._validate(output)
        return output

    def _validate(self, frame: pd.DataFrame) -> None:
        values = frame[self.output_names].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise RuntimeError("RealMLP 목표 인코딩에 유한하지 않은 값이 있다.")


class _NumericalPreprocessor:
    def __init__(self, transformations: list[str]) -> None:
        allowed = {"median_center", "robust_scale", "smooth_clip", "l2_normalize"}
        unknown = sorted(set(transformations) - allowed)
        if unknown:
            raise ValueError(f"RealMLP가 모르는 수치 변환이다: {unknown}")
        self.transformations = list(transformations)

    def fit(self, values: np.ndarray) -> _NumericalPreprocessor:
        if {"median_center", "robust_scale"} & set(self.transformations):
            self.median = np.median(values, axis=0)
            spread = np.quantile(values, 0.75, axis=0) - np.quantile(
                values, 0.25, axis=0
            )
            zero = spread == 0.0
            spread[zero] = 0.5 * (
                values.max(axis=0)[zero] - values.min(axis=0)[zero]
            )
            self.iqr_factors = 1.0 / (spread + 1e-30)
            self.iqr_factors[spread == 0.0] = 0.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = values.copy().astype("float32")
        for transformation in self.transformations:
            if transformation == "median_center":
                output -= self.median[None, :]
            elif transformation == "robust_scale":
                output *= self.iqr_factors[None, :]
            elif transformation == "smooth_clip":
                output = output / np.sqrt(1 + (output / 3) ** 2)
            elif transformation == "l2_normalize":
                norms = np.linalg.norm(output, axis=1, keepdims=True)
                output /= np.where(norms == 0, 1.0, norms)
        return output


class _CategoricalFeatureLayer(nn.Module):
    def __init__(
        self, n_ens: int, cat_dims: list[int], embed_dim: int, onehot_thresh: int
    ) -> None:
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features: list[int] = []
        self.embed_layers = nn.ModuleList()
        self.embed_feature_indices: list[int] = []
        for feature_index, dimension in enumerate(cat_dims):
            if dimension <= onehot_thresh:
                self.onehot_features.append(feature_index)
            else:
                self.embed_layers.append(
                    nn.ModuleList(
                        [nn.Embedding(dimension, embed_dim) for _ in range(n_ens)]
                    )
                )
                self.embed_feature_indices.append(feature_index)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, n_ens, _ = values.shape
        features: list[torch.Tensor] = []
        if self.onehot_features:
            onehot_values = values[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[index] for index in self.onehot_features]
            encoded = torch.zeros(
                batch_size, n_ens, sum(onehot_dims), device=values.device
            )
            start = 0
            for index, dimension in enumerate(onehot_dims):
                positions = onehot_values[:, :, index : index + 1].long()
                encoded.scatter_(2, positions + start, 1.0)
                start += dimension
            features.append(encoded)
        for embeddings, feature_index in zip(
            self.embed_layers, self.embed_feature_indices
        ):
            features.append(
                torch.cat(
                    [
                        embeddings[model_index](
                            values[:, model_index, feature_index : feature_index + 1].long()
                        )
                        for model_index in range(self.n_ens)
                    ],
                    dim=1,
                )
            )
        return torch.cat(features, dim=2)


class _ScalingLayer(nn.Module):
    def __init__(self, n_ens: int, features: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale[None, :, :]


class _NTPLinear(nn.Module):
    def __init__(
        self,
        n_ens: int,
        input_features: int,
        output_features: int,
        *,
        split_weight: bool = False,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        initial_weight = torch.randn(n_ens, input_features, output_features)
        self.weight = (
            None if split_weight else nn.Parameter(initial_weight)
        )
        self.weights = nn.ParameterList(
            [
                nn.Parameter(initial_weight[index].clone())
                for index in range(n_ens)
            ]
            if split_weight
            else []
        )
        self.bias = nn.Parameter(torch.randn(n_ens, output_features))

    def matrix_parameters(self) -> list[nn.Parameter]:
        return list(self.weights) if self.weights else [self.weight]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weight = torch.stack(list(self.weights)) if self.weights else self.weight
        output = torch.einsum("bki,kio->bko", values, weight)
        return output / math.sqrt(self.input_features) + self.bias


class _PBLDEmbedding(nn.Module):
    def __init__(
        self,
        n_ens: int,
        features: int,
        hidden_dim: int,
        output_dim: int,
        frequency_scale: float,
    ) -> None:
        super().__init__()
        self.weight1 = nn.Parameter(
            torch.randn(n_ens, features, hidden_dim) * frequency_scale
        )
        self.bias1 = nn.Parameter(torch.randn(n_ens, features, hidden_dim))
        self.weight2 = nn.Parameter(
            torch.randn(n_ens, features, hidden_dim, output_dim - 1)
            / math.sqrt(hidden_dim)
        )
        self.bias2 = nn.Parameter(
            torch.zeros(n_ens, features, output_dim - 1)
        )
        self.activation = nn.PReLU()
        nn.init.uniform_(self.bias1, -math.pi, math.pi)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        periodic = torch.cos(
            2
            * math.pi
            * (
                values.unsqueeze(-1) * self.weight1.unsqueeze(0)
                + self.bias1.unsqueeze(0)
            )
        )
        transformed = self.activation(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.weight2)
            + self.bias2.unsqueeze(0)
        )
        return torch.cat([values.unsqueeze(-1), transformed], dim=-1).flatten(
            start_dim=2
        )


class _RealMLP(nn.Module):
    def __init__(self, cat_dims: list[int], numerical_features: int, config: dict) -> None:
        super().__init__()
        n_ens = int(config["n_ens"])
        self.n_ens = n_ens
        self.categorical = _CategoricalFeatureLayer(
            n_ens,
            cat_dims,
            int(config["embed_dim"]),
            int(config["onehot_thresh"]),
        )
        self.numerical = _PBLDEmbedding(
            n_ens,
            numerical_features,
            int(config["pbld_hidden_dim"]),
            int(config["pbld_out_dim"]),
            float(config["pbld_freq_scale"]),
        )
        numerical_dim = numerical_features * int(config["pbld_out_dim"])
        categorical_dim = sum(
            dimension
            if dimension <= int(config["onehot_thresh"])
            else int(config["embed_dim"])
            for dimension in cat_dims
        )
        input_dim = numerical_dim + categorical_dim
        layers: list[nn.Module] = [_ScalingLayer(n_ens, input_dim)]
        self.dropout_modules: list[nn.Dropout] = []
        split_hidden_weights = config["optimizer"] == "muon"
        for layer_index, output_dim in enumerate(config["hidden_dims"]):
            linear = _NTPLinear(
                n_ens,
                input_dim,
                int(output_dim),
                split_weight=split_hidden_weights,
            )
            if layer_index == 0:
                self.first_linear = linear
            dropout = nn.Dropout(float(config["dropout"]))
            self.dropout_modules.append(dropout)
            layers.extend([linear, nn.SiLU(), dropout])
            input_dim = int(output_dim)
        self.hidden = nn.Sequential(*layers)
        self.output = _NTPLinear(n_ens, input_dim, 2)

    def forward(
        self, numerical: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        numerical = numerical.unsqueeze(1).expand(-1, self.n_ens, -1)
        categorical = categorical.unsqueeze(1).expand(-1, self.n_ens, -1)
        combined = torch.cat(
            [self.numerical(numerical), self.categorical(categorical)], dim=2
        )
        return self.output(self.hidden(combined)).softmax(dim=2)


def _schedule(value: float, progress: float, name: str, flat_ratio: float) -> float:
    if name == "constant":
        return value
    if name == "cos":
        return value * (math.cos(math.pi * progress) + 1) / 2
    if name == "flat_cos":
        if progress < flat_ratio:
            return value
        position = (progress - flat_ratio) / (1 - flat_ratio)
        return value * (math.cos(math.pi * position) + 1) / 2
    if name == "expm4t":
        return value * math.exp(-4 * progress)
    raise ValueError(f"RealMLP가 모르는 schedule이다: {name!r}")


def _parameter_groups(model: _RealMLP, config: dict) -> list[dict]:
    first_weight_ids = {
        id(parameter) for parameter in model.first_linear.matrix_parameters()
    }
    scale, pbld, first_weight, other_weight, bias = [], [], [], [], []
    for name, parameter in model.named_parameters():
        if "numerical" in name:
            pbld.append(parameter)
        elif "scale" in name:
            scale.append(parameter)
        elif id(parameter) in first_weight_ids:
            first_weight.append(parameter)
        elif "bias" in name:
            bias.append(parameter)
        else:
            other_weight.append(parameter)
    learning_rate = float(config["lr"])
    weight_decay = float(config["weight_decay"])
    return [
        {
            "params": scale,
            "lr": learning_rate * float(config["lr_scale_mult"]),
            "weight_decay": weight_decay * float(config["wd_scale_mult"]),
        },
        {
            "params": pbld,
            "lr": learning_rate * float(config["pbld_lr_factor"]),
            "weight_decay": weight_decay,
        },
        {
            "params": first_weight,
            "lr": learning_rate * float(config["first_layer_lr_factor"]),
            "weight_decay": weight_decay
            * float(config["first_layer_wd_factor"]),
        },
        {
            "params": other_weight,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": bias,
            "lr": learning_rate * float(config["lr_bias_mult"]),
            "weight_decay": weight_decay * float(config["wd_bias_mult"]),
        },
    ]


def _muon_parameters(model: _RealMLP) -> list[nn.Parameter]:
    """내부 앙상블별 RealMLP 은닉 행렬만 반환한다."""
    return [
        parameter
        for module in model.hidden.modules()
        if isinstance(module, _NTPLinear)
        for parameter in module.matrix_parameters()
    ]


def _smoothed_cross_entropy(
    target: torch.Tensor,
    prediction: torch.Tensor,
    smoothing: float,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    classes = prediction.size(1)
    smooth_target = torch.full_like(prediction, smoothing / classes)
    smooth_target.scatter_(
        1, target.unsqueeze(1), 1.0 - smoothing + smoothing / classes
    )
    losses = -(
        smooth_target * torch.log(prediction.clamp(1e-15, 1))
    ).sum(dim=1)
    weights = class_weights[target]
    return (losses * weights).sum() / weights.sum()


class _FixedEpochClassifier:
    """검증 자료를 받지 않고 일정의 앞부분만 고정 길이로 학습한다."""

    def __init__(self, config: dict, seed: int, device: str) -> None:
        self.config = config
        self.seed = seed
        self.device = device
        self.model: _RealMLP | None = None
        self.training_history: list[dict[str, float | int]] = []
        self._captured_ema_states: dict[int, dict[str, torch.Tensor]] = {}
        self.selected_epoch: int | None = None

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False

    def fit(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
        categorical_columns: list[str],
        category_dimensions: list[int],
        *,
        capture_epochs: Sequence[int] = (),
    ) -> None:
        epochs = int(self.config["fixed_epochs"])
        capture_epochs = _validated_completed_epochs(
            capture_epochs,
            fixed_epochs=epochs,
            schedule_epochs=int(self.config["schedule_epochs"]),
            allow_empty=True,
        )
        capture_epoch_set = set(capture_epochs)
        self._captured_ema_states = {}
        self.selected_epoch = None
        self._seed_everything(self.seed)
        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = [
            column for column in frame.columns if column not in categorical_columns
        ]
        self.category_dimensions = list(category_dimensions)
        numerical = frame[self.numerical_columns].to_numpy(dtype="float32")
        categorical = frame[self.categorical_columns].to_numpy(
            dtype="int64", copy=True
        )
        labels = target.to_numpy(dtype="int64", copy=True)
        self.preprocessor = _NumericalPreprocessor(self.config["tfms"]).fit(
            numerical
        )
        numerical = self.preprocessor.transform(numerical)

        class_values = np.unique(labels)
        if not np.array_equal(class_values, np.array([0, 1])):
            raise ValueError("RealMLP 학습 부분에는 두 클래스가 모두 있어야 한다.")
        class_weights = torch.as_tensor(
            compute_class_weight(
                class_weight="balanced", classes=class_values, y=labels
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.model = _RealMLP(
            self.category_dimensions, numerical.shape[1], self.config
        ).to(self.device)
        parameter_groups = _parameter_groups(self.model, self.config)
        for group in parameter_groups:
            group["lr_base"] = group["lr"]
        betas = (float(self.config["mom"]), float(self.config["sq_mom"]))
        if self.config["optimizer"] == "muon":
            from .muon import (
                MuonWithAdamW,
                hybrid_parameter_groups,
            )

            parameter_groups = hybrid_parameter_groups(
                parameter_groups, _muon_parameters(self.model)
            )
            optimizer = MuonWithAdamW(
                parameter_groups,
                lr=float(self.config["lr"]),
                weight_decay=float(self.config["weight_decay"]),
                betas=betas,
            )
        else:
            optimizer = torch.optim.AdamW(parameter_groups, betas=betas)
        numerical_tensor = torch.as_tensor(
            numerical, dtype=torch.float32, device=self.device
        )
        categorical_tensor = torch.as_tensor(
            categorical, dtype=torch.long, device=self.device
        )
        target_tensor = torch.as_tensor(
            labels, dtype=torch.long, device=self.device
        )
        n_ens = int(self.config["n_ens"])
        batch_size = int(self.config["batch_size"])
        total_schedule_steps = int(self.config["schedule_epochs"]) * len(frame)
        train_order = np.arange(len(frame))
        rng = np.random.RandomState(self.seed)
        ema_decay = float(self.config["ema_decay"])
        ema_state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

        self.training_history = []
        for epoch in range(epochs):
            started = time.monotonic()
            weighted_loss = 0.0
            self.model.train()
            for start in range(0, len(frame), batch_size):
                progress = (epoch * len(frame) + start) / total_schedule_steps
                batch_indices = train_order[start : start + batch_size]
                for group in optimizer.param_groups:
                    group["lr"] = _schedule(
                        group["lr_base"],
                        progress,
                        self.config["lr_sched"],
                        float(self.config["flat_ratio"]),
                    )
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(
                    numerical_tensor[batch_indices],
                    categorical_tensor[batch_indices],
                )
                smoothing = _schedule(
                    float(self.config["ls_eps"]),
                    progress,
                    self.config["ls_eps_sched"],
                    float(self.config["flat_ratio"]),
                )
                dropout = _schedule(
                    float(self.config["dropout"]),
                    progress,
                    self.config["p_drop_sched"],
                    float(self.config["flat_ratio"]),
                )
                for module in self.model.dropout_modules:
                    module.p = dropout
                loss = _smoothed_cross_entropy(
                    target_tensor[batch_indices].repeat_interleave(n_ens),
                    prediction.reshape(-1, 2),
                    smoothing,
                    class_weights,
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("RealMLP 학습 손실이 유한하지 않다.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config["grad_clip"])
                )
                optimizer.step()
                weighted_loss += float(loss.detach().cpu()) * len(batch_indices)
                with torch.no_grad():
                    for key, value in self.model.state_dict().items():
                        if torch.is_floating_point(value):
                            ema_state[key].mul_(ema_decay).add_(
                                value.detach(), alpha=1.0 - ema_decay
                            )
                        else:
                            ema_state[key].copy_(value)
            rng.shuffle(train_order)
            self.model.load_state_dict(ema_state, strict=True)
            completed_epoch = epoch + 1
            if completed_epoch in capture_epoch_set:
                self._captured_ema_states[completed_epoch] = {
                    key: value.detach().cpu().clone()
                    for key, value in ema_state.items()
                }
            record = {
                "epoch": completed_epoch,
                "loss": weighted_loss / len(frame),
                "seconds": time.monotonic() - started,
                "smoothing": float(smoothing),
                "dropout": float(dropout),
            }
            self.training_history.append(record)
            if int(self.config["verbosity"]) >= 1:
                print(
                    f"[realmlp] init_seed={self.seed} epoch={epoch + 1}/{epochs} "
                    f"loss={record['loss']:.6f} seconds={record['seconds']:.1f}",
                    flush=True,
                )
        self.selected_epoch = epochs

    @property
    def captured_epochs(self) -> tuple[int, ...]:
        return tuple(sorted(self._captured_ema_states))

    def select_epoch(self, completed_epoch: int) -> None:
        if self.model is None:
            raise RuntimeError("RealMLP를 먼저 학습해야 한다.")
        if (
            isinstance(completed_epoch, bool)
            or not isinstance(completed_epoch, int)
            or completed_epoch < 1
        ):
            raise ValueError("선택할 completed epoch는 양의 정수여야 한다.")
        state = self._captured_ema_states.get(completed_epoch)
        if state is None:
            raise ValueError(
                "포착하지 않은 completed epoch를 선택할 수 없다: "
                f"{completed_epoch} (포착: {list(self.captured_epochs)})"
            )
        self.model.load_state_dict(state, strict=True)
        self.selected_epoch = completed_epoch

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RealMLP를 먼저 학습해야 한다.")
        numerical = self.preprocessor.transform(
            frame[self.numerical_columns].to_numpy(dtype="float32")
        )
        categorical = frame[self.categorical_columns].to_numpy(
            dtype="int64", copy=True
        )
        numerical_tensor = torch.as_tensor(
            numerical, dtype=torch.float32, device=self.device
        )
        categorical_tensor = torch.as_tensor(
            categorical, dtype=torch.long, device=self.device
        )
        self.model.eval()
        batch_size = int(self.config["eval_batch_size"])
        with torch.inference_mode():
            probabilities = np.concatenate(
                [
                    self.model(
                        numerical_tensor[start : start + batch_size],
                        categorical_tensor[start : start + batch_size],
                    )
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                    for start in range(0, len(frame), batch_size)
                ]
            )
        return probabilities

    def to(self, device: str) -> None:
        if self.model is None:
            raise RuntimeError("RealMLP를 먼저 학습해야 한다.")
        self.model.to(device)
        self.device = device


@dataclass(frozen=True)
class _PreparedFold:
    train: pd.DataFrame
    validation: pd.DataFrame | None


class RealMLPFold:
    """fold 전처리, 두 초기화 평균, 예측과 중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        unknown = sorted(set(params) - set(_DEFAULTS))
        if unknown:
            raise ValueError(f"realmlp가 모르는 params: {unknown}")
        self.config = {**_DEFAULTS, **params}
        self.config["optimizer"] = str(self.config["optimizer"]).lower()
        if self.config["optimizer"] not in {"adamw", "muon"}:
            raise ValueError(
                "realmlp optimizer는 ['adamw', 'muon'] 중 하나여야 한다: "
                f"{self.config['optimizer']!r}"
            )
        self.config["hidden_dims"] = list(self.config["hidden_dims"])
        self.config["tfms"] = list(self.config["tfms"])
        self.config["reference_qnormal_columns"] = list(
            self.config["reference_qnormal_columns"]
        )
        self.seed = seed
        self._validate_config()
        self.device = str(self.config["device"])
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("realmlp device=cuda인데 CUDA를 사용할 수 없다.")
        if self.device == "cpu":
            torch.set_num_threads(1)
        self.engineer: _FoldFeatureEngineer | None = None
        self.target_encoder: _FoldTargetEncoder | None = None
        self.models: list[_FixedEpochClassifier] = []
        self.importance_frame: pd.DataFrame | None = None
        self.importance_target: np.ndarray | None = None
        self.importance_base_auc: float | None = None
        self._importance: pd.DataFrame | None = None
        self._diagnostics: dict[str, object] | None = None
        self._raw_training_length_selection: int | None = None
        self._fit_seconds: float | None = None
        self._importance_seconds: float | None = None
        self._prediction_calls = 0
        self._all_predictions_finite = True
        self._dataset_reference: tuple[pd.DataFrame, pd.DataFrame] | None = None
        self._dataset_reference_target_free = True
        self._dataset_reference_train_rows: int | None = None
        self._dataset_reference_test_rows: int | None = None
        self._trajectory_candidate_epochs: tuple[int, ...] = ()
        self._trajectory_validation_predictions: dict[int, np.ndarray] = {}
        self._trajectory_validation_aucs: dict[int, float] = {}
        self._trajectory_member_validation_aucs: dict[int, list[float]] = {}
        self._trajectory_member_records: list[dict[str, object]] = []
        self._trajectory_base_diagnostics: dict[str, object] | None = None

    def _validate_config(self) -> None:
        positive_ints = [
            "n_ens",
            "embed_dim",
            "onehot_thresh",
            "pbld_hidden_dim",
            "pbld_out_dim",
            "fixed_epochs",
            "schedule_epochs",
            "batch_size",
            "eval_batch_size",
            "n_init_avg",
            "init_seed_stride",
            "inner_folds",
            "perm_sample",
            "perm_repeats",
        ]
        for name in positive_ints:
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"realmlp {name}은 양의 정수여야 한다: {value!r}")
        if self.config["fixed_epochs"] > self.config["schedule_epochs"]:
            raise ValueError("realmlp fixed_epochs는 schedule_epochs보다 클 수 없다.")
        if not self.config["hidden_dims"] or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.config["hidden_dims"]
        ):
            raise ValueError("realmlp hidden_dims는 양의 정수 목록이어야 한다.")
        for name in ["dropout", "ls_eps"]:
            if not 0 <= float(self.config[name]) < 1:
                raise ValueError(f"realmlp {name}은 0 이상 1 미만이어야 한다.")
        for name in [
            "pbld_freq_scale",
            "pbld_lr_factor",
            "lr",
            "mom",
            "sq_mom",
            "ema_decay",
            "grad_clip",
        ]:
            if float(self.config[name]) <= 0:
                raise ValueError(f"realmlp {name}은 양수여야 한다.")
        columns = self.config["reference_qnormal_columns"]
        if columns and columns != RAW_NUMERICAL:
            raise ValueError(
                "realmlp reference_qnormal_columns는 원시 수치 9열을 순서대로 "
                f"선언해야 한다: {RAW_NUMERICAL}"
            )
        scope = str(self.config["preprocessing_scope"])
        if scope not in {"fold_train", "train_test"}:
            raise ValueError(
                "realmlp preprocessing_scope은 'fold_train' 또는 'train_test'여야 한다: "
                f"{scope!r}"
            )
        if scope == "train_test" and not columns:
            raise ValueError(
                "realmlp preprocessing_scope='train_test'에는 "
                "reference_qnormal_columns 선언이 필요하다."
            )

    def set_dataset_reference(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        if self.engineer is not None or self.models:
            raise RuntimeError("전처리 기준 집합은 RealMLP 학습 전에 정해야 한다.")
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("RealMLP 전처리 기준 집합의 train/test 열이 다르다.")
        if TARGET in X_train.columns or TARGET in X_test.columns:
            self._dataset_reference_target_free = False
            raise ValueError("RealMLP 전처리 기준 집합은 목표값을 포함할 수 없다.")
        columns = self.config["reference_qnormal_columns"]
        if columns:
            missing = sorted(set(columns) - set(X_train.columns))
            if missing:
                raise ValueError(
                    f"RealMLP 전처리 기준 집합에 원시 수치 열이 없다: {missing}"
                )
            self._dataset_reference = (
                X_train[columns].copy(),
                X_test[columns].copy(),
            )
        self._dataset_reference_train_rows = len(X_train)
        self._dataset_reference_test_rows = len(X_test)

    def _fold_seed(self, index: pd.Index) -> int:
        hashed = pd.util.hash_pandas_object(index, index=False).to_numpy(
            dtype="uint64"
        )
        digest = hashlib.sha256(hashed.tobytes()).digest()
        fold_component = int.from_bytes(digest[:8], "little")
        return int((self.seed * 1_000_003 + fold_component) % (2**31 - 1))

    def _prepare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame | None,
    ) -> _PreparedFold:
        reference_columns = self.config["reference_qnormal_columns"]
        reference_frame = None
        if reference_columns:
            if self.config["preprocessing_scope"] == "fold_train":
                reference_frame = X_train[reference_columns]
            else:
                if self._dataset_reference is None:
                    raise RuntimeError(
                        "preprocessing_scope='train_test'에는 train+test "
                        "전처리 기준 집합이 필요하다."
                    )
                reference_frame = pd.concat(
                    list(self._dataset_reference), ignore_index=True
                )
        self.engineer = _FoldFeatureEngineer(reference_columns, self.seed)
        train = self.engineer.fit_transform(X_train, reference_frame)
        validation = (
            self.engineer.transform(X_validation)
            if X_validation is not None
            else None
        )
        fold_seed = self._fold_seed(X_train.index)
        self.target_encoder = _FoldTargetEncoder(
            int(self.config["inner_folds"]), fold_seed
        )
        train = self.target_encoder.fit_transform(
            train, y_train, self.engineer.output_cat_cols
        )
        if validation is not None:
            validation = self.target_encoder.transform(validation)
        return _PreparedFold(train=train, validation=validation)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> np.ndarray:
        return self._fit_validation(
            X_train,
            y_train,
            X_validation,
            y_validation,
            completed_epochs=(),
        )

    def fit_training_trajectory(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        completed_epochs: Sequence[int],
    ) -> dict[int, np.ndarray]:
        """한 고정 궤적에서 미리 선언한 epoch 종료 EMA 예측을 포착한다."""
        validated_epochs = _validated_completed_epochs(
            completed_epochs,
            fixed_epochs=int(self.config["fixed_epochs"]),
            schedule_epochs=int(self.config["schedule_epochs"]),
        )
        self._fit_validation(
            X_train,
            y_train,
            X_validation,
            y_validation,
            completed_epochs=validated_epochs,
        )
        # 학습 직후에도 공개 예측 상태가 포착된 후보 하나를 가리키게 한다.
        self.select_training_point(validated_epochs[-1])
        return {
            epoch: prediction.copy()
            for epoch, prediction in self._trajectory_validation_predictions.items()
        }

    def _fit_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        *,
        completed_epochs: tuple[int, ...],
    ) -> np.ndarray:
        trajectory_end_epoch = int(self.config["fixed_epochs"])
        trajectory_mode = bool(completed_epochs)
        captured_epochs = (
            tuple(sorted(set(completed_epochs + (trajectory_end_epoch,))))
            if trajectory_mode
            else ()
        )
        self._trajectory_candidate_epochs = completed_epochs
        self._trajectory_validation_predictions = {}
        self._trajectory_validation_aucs = {}
        self._trajectory_member_validation_aucs = {
            epoch: [] for epoch in completed_epochs
        }
        self._trajectory_member_records = []
        self._trajectory_base_diagnostics = None
        started = time.monotonic()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        prepared = self._prepare(X_train, y_train, X_validation)
        assert prepared.validation is not None
        fold_seed = self._fold_seed(X_train.index)
        validation_prediction = np.zeros(len(X_validation), dtype="float64")
        trajectory_predictions = {
            epoch: np.zeros(len(X_validation), dtype="float64")
            for epoch in completed_epochs
        }
        member_records = []
        self.models = []
        for member_index in range(int(self.config["n_init_avg"])):
            initialization_seed = (
                fold_seed + member_index * int(self.config["init_seed_stride"])
            ) % (2**31 - 1)
            model = _FixedEpochClassifier(
                dict(self.config), initialization_seed, self.device
            )
            model.fit(
                prepared.train,
                y_train,
                self.engineer.output_cat_cols,
                self.engineer.cat_dims,
                capture_epochs=captured_epochs,
            )
            member_prediction = model.predict_proba(prepared.validation)[:, 1].astype(
                "float64"
            )
            validation_prediction += member_prediction / int(
                self.config["n_init_avg"]
            )
            member_records.append(
                {
                    "initialization_seed": initialization_seed,
                    "validation_auc": float(
                        roc_auc_score(y_validation, member_prediction)
                    ),
                    "training_history": model.training_history,
                    "parameter_count": _parameter_count(model.model),
                }
            )
            if trajectory_mode:
                for completed_epoch in completed_epochs:
                    if completed_epoch == trajectory_end_epoch:
                        point_prediction = member_prediction
                    else:
                        model.select_epoch(completed_epoch)
                        point_prediction = model.predict_proba(prepared.validation)[
                            :, 1
                        ].astype("float64")
                    trajectory_predictions[completed_epoch] += (
                        point_prediction / int(self.config["n_init_avg"])
                    )
                    self._trajectory_member_validation_aucs[
                        completed_epoch
                    ].append(float(roc_auc_score(y_validation, point_prediction)))
                model.select_epoch(trajectory_end_epoch)
            model.to("cpu")
            self.models.append(model)
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        sample_size = min(int(self.config["perm_sample"]), len(X_validation))
        sample_rng = np.random.default_rng(self.seed)
        sample_positions = np.sort(
            sample_rng.choice(len(X_validation), size=sample_size, replace=False)
        )
        self.importance_frame = X_validation.iloc[sample_positions].copy()
        self.importance_target = y_validation.iloc[sample_positions].to_numpy(
            dtype="float64"
        )
        self.importance_base_auc = float(
            roc_auc_score(self.importance_target, self.predict(self.importance_frame))
        )
        self._fit_seconds = float(time.monotonic() - started)
        cuda_allocated = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.startswith("cuda")
            else None
        )
        cuda_reserved = (
            int(torch.cuda.max_memory_reserved(self.device))
            if self.device.startswith("cuda")
            else None
        )
        self._diagnostics = {
            "source_notebook_url": SOURCE_NOTEBOOK_URL,
            "source_kernel_id": SOURCE_KERNEL_ID,
            "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
            "source_backend": "custom_pytorch_realmlp",
            "pytabkit_estimator_used": False,
            "optimizer": self.config["optimizer"],
            "preprocessing_scope": self.config["preprocessing_scope"],
            "reference_qnormal_columns": list(
                self.engineer.reference_qnormal_output_columns
            ),
            "reference_qnormal_reference_rows": (
                self.engineer.reference_qnormal_reference_rows
            ),
            "reference_qnormal_fit_rows_by_column": dict(
                self.engineer.reference_qnormal_fit_rows
            ),
            "reference_qnormal_random_state": self.seed,
            "reference_qnormal_target_free": self._dataset_reference_target_free,
            "dataset_reference_train_rows": self._dataset_reference_train_rows,
            "dataset_reference_test_rows": self._dataset_reference_test_rows,
            "preprocessing_fit_rows": self.engineer.fit_rows,
            "target_encoding_fit_rows": self.target_encoder.fit_rows,
            "training_rows": len(X_train),
            "validation_rows": len(X_validation),
            "raw_input_columns": list(X_train.columns),
            "raw_input_feature_count": len(X_train.columns),
            "engineered_feature_count": len(prepared.train.columns),
            "categorical_feature_count": len(self.engineer.output_cat_cols),
            "target_encoding_count": len(self.target_encoder.output_names),
            "fold_initialization_seed": fold_seed,
            "fold_initialization_members": member_records,
            "fold_initialization_average_count": int(self.config["n_init_avg"]),
            "internal_ensemble_count": int(self.config["n_ens"]),
            "fixed_epochs": int(self.config["fixed_epochs"]),
            "schedule_horizon_epochs": int(self.config["schedule_epochs"]),
            "validation_selection": "final_fixed_epoch",
            "validation_auc": float(
                roc_auc_score(y_validation, validation_prediction)
            ),
            "validation_unknown_category_values": self.engineer.unknown_value_count(
                self.engineer.transform(X_validation)
            ),
            "cuda_max_allocated_bytes": cuda_allocated,
            "cuda_max_reserved_bytes": cuda_reserved,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        }
        self._raw_training_length_selection = int(self.config["fixed_epochs"])
        if trajectory_mode:
            self._trajectory_validation_predictions = trajectory_predictions
            self._trajectory_validation_aucs = {
                epoch: float(roc_auc_score(y_validation, prediction))
                for epoch, prediction in trajectory_predictions.items()
            }
            self._trajectory_member_records = [dict(record) for record in member_records]
            self._diagnostics.update(
                {
                    "training_trajectory": True,
                    "captured_completed_epochs": list(completed_epochs),
                    "trajectory_end_epochs": trajectory_end_epoch,
                    "completed_epochs": trajectory_end_epoch,
                    "selected_completed_epoch": trajectory_end_epoch,
                    "state_kind": "ema",
                    "selection_rule": "precommitted",
                }
            )
            self._trajectory_base_diagnostics = dict(self._diagnostics)
        return validation_prediction

    def select_training_point(self, completed_epoch: int) -> np.ndarray:
        """포착한 시점의 두 초기화 EMA를 함께 복원하고 검증 예측을 돌려준다."""
        if self._trajectory_base_diagnostics is None:
            raise RuntimeError("fit_training_trajectory를 먼저 호출해야 한다.")
        if (
            isinstance(completed_epoch, bool)
            or not isinstance(completed_epoch, int)
            or completed_epoch < 1
        ):
            raise ValueError("선택할 completed epoch는 양의 정수여야 한다.")
        if completed_epoch not in self._trajectory_candidate_epochs:
            raise ValueError(
                "미리 선언하지 않은 completed epoch를 선택할 수 없다: "
                f"{completed_epoch} (후보: {list(self._trajectory_candidate_epochs)})"
            )
        for model in self.models:
            model.select_epoch(completed_epoch)
        self._raw_training_length_selection = completed_epoch
        self._importance = None
        self._importance_seconds = None
        if self.importance_frame is None or self.importance_target is None:
            raise RuntimeError("학습 궤적의 검증 중요도 표본이 없다.")
        self.importance_base_auc = float(
            roc_auc_score(self.importance_target, self.predict(self.importance_frame))
        )

        member_records = []
        member_aucs = self._trajectory_member_validation_aucs[completed_epoch]
        for member_index, base_record in enumerate(self._trajectory_member_records):
            full_history = list(base_record["training_history"])
            record = dict(base_record)
            record.update(
                {
                    "validation_auc": member_aucs[member_index],
                    "training_history": full_history[:completed_epoch],
                    "trajectory_training_history": full_history,
                    "selected_epoch": completed_epoch,
                }
            )
            member_records.append(record)

        prediction = self._trajectory_validation_predictions[completed_epoch]
        diagnostics = dict(self._trajectory_base_diagnostics)
        diagnostics.update(
            {
                "fold_initialization_members": member_records,
                "fixed_epochs": completed_epoch,
                "completed_epochs": completed_epoch,
                "selected_completed_epoch": completed_epoch,
                "validation_selection": "precommitted_completed_epoch",
                "validation_auc": self._trajectory_validation_aucs[completed_epoch],
            }
        )
        self._diagnostics = diagnostics
        return prediction.copy()

    def fit_full(
        self, X: pd.DataFrame, y: pd.Series, training_budget: int
    ) -> None:
        if (
            isinstance(training_budget, bool)
            or not isinstance(training_budget, int)
            or training_budget < 1
            or training_budget > int(self.config["schedule_epochs"])
        ):
            raise ValueError(
                "RealMLP 전체 자료 고정 epoch는 1 이상 schedule 지평 이하여야 한다."
            )
        self._trajectory_candidate_epochs = ()
        self._trajectory_validation_predictions = {}
        self._trajectory_validation_aucs = {}
        self._trajectory_member_validation_aucs = {}
        self._trajectory_member_records = []
        self._trajectory_base_diagnostics = None
        self.config["fixed_epochs"] = training_budget
        # 전체 자료 재학습의 epoch 수는 관측이 아니라 이미 정해진 예산이다. (#372)
        self._raw_training_length_selection = None
        prepared = self._prepare(X, y, None)
        fold_seed = self._fold_seed(X.index)
        self.models = []
        member_records = []
        for member_index in range(int(self.config["n_init_avg"])):
            initialization_seed = (
                fold_seed + member_index * int(self.config["init_seed_stride"])
            ) % (2**31 - 1)
            model = _FixedEpochClassifier(
                dict(self.config), initialization_seed, self.device
            )
            model.fit(
                prepared.train,
                y,
                self.engineer.output_cat_cols,
                self.engineer.cat_dims,
            )
            member_records.append(
                {
                    "initialization_seed": initialization_seed,
                    "training_history": model.training_history,
                    "parameter_count": _parameter_count(model.model),
                }
            )
            model.to("cpu")
            self.models.append(model)
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        self._diagnostics = {
            "source_notebook_url": SOURCE_NOTEBOOK_URL,
            "source_kernel_id": SOURCE_KERNEL_ID,
            "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
            "source_backend": "custom_pytorch_realmlp",
            "pytabkit_estimator_used": False,
            "optimizer": self.config["optimizer"],
            "preprocessing_scope": self.config["preprocessing_scope"],
            "reference_qnormal_columns": list(
                self.engineer.reference_qnormal_output_columns
            ),
            "reference_qnormal_reference_rows": (
                self.engineer.reference_qnormal_reference_rows
            ),
            "reference_qnormal_fit_rows_by_column": dict(
                self.engineer.reference_qnormal_fit_rows
            ),
            "reference_qnormal_random_state": self.seed,
            "reference_qnormal_target_free": self._dataset_reference_target_free,
            "dataset_reference_train_rows": self._dataset_reference_train_rows,
            "dataset_reference_test_rows": self._dataset_reference_test_rows,
            "full_fit": True,
            "full_training_budget": training_budget,
            "preprocessing_fit_rows": len(X),
            "target_encoding_fit_rows": len(X),
            "training_rows": len(X),
            "validation_rows": 0,
            "fold_initialization_seed": fold_seed,
            "fold_initialization_members": member_records,
            "fold_initialization_average_count": int(self.config["n_init_avg"]),
            "internal_ensemble_count": int(self.config["n_ens"]),
            "schedule_horizon_epochs": int(self.config["schedule_epochs"]),
            "validation_selection": "none",
        }

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.engineer is None or self.target_encoder is None:
            raise RuntimeError("RealMLP를 먼저 학습해야 한다.")
        return self.target_encoder.transform(self.engineer.transform(frame))

    def _predict_prepared(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = np.zeros(len(frame), dtype="float64")
        for model in self.models:
            model.to(self.device)
            prediction += model.predict_proba(frame)[:, 1].astype("float64") / len(
                self.models
            )
            model.to("cpu")
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        self._prediction_calls += 1
        self._all_predictions_finite &= bool(np.isfinite(prediction).all())
        return prediction

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._predict_prepared(self._transform(frame))

    def importance(self) -> pd.DataFrame:
        if self._importance is not None:
            return self._importance.copy()
        if self.importance_frame is None or self.importance_target is None:
            raise RuntimeError("RealMLP 순열 중요도는 검증 fold 학습 뒤에만 계산한다.")
        started = time.monotonic()
        gains = []
        for column_index, column in enumerate(self.importance_frame.columns):
            drops = []
            for repeat in range(int(self.config["perm_repeats"])):
                rng = np.random.default_rng(
                    self.seed * 10007 + column_index * 101 + repeat
                )
                permuted = self.importance_frame.copy()
                permuted[column] = (
                    self.importance_frame[column]
                    .take(rng.permutation(len(self.importance_frame)))
                    .set_axis(permuted.index)
                )
                drops.append(
                    self.importance_base_auc
                    - roc_auc_score(self.importance_target, self.predict(permuted))
                )
            gains.append(float(np.mean(drops)))
        self._importance_seconds = float(time.monotonic() - started)
        self._importance = pd.DataFrame(
            {"feature": list(self.importance_frame.columns), "gain": gains}
        )
        return self._importance.copy()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError("entry_diagnostics는 fit 뒤에 호출해야 한다.")
        assertions = {
            "preprocessing_training_rows_only": self.engineer.fit_rows
            == self._diagnostics["training_rows"],
            "target_encoding_training_rows_only": self.target_encoder.fit_rows
            == self._diagnostics["training_rows"],
            "reference_qnormal_target_free": self._diagnostics[
                "reference_qnormal_target_free"
            ],
            "reference_qnormal_scope_matches_contract": (
                not self.config["reference_qnormal_columns"]
                or self.engineer.reference_qnormal_reference_rows
                == (
                    self._diagnostics["training_rows"]
                    if self.config["preprocessing_scope"] == "fold_train"
                    else self._dataset_reference_train_rows
                    + self._dataset_reference_test_rows
                )
            ),
            "validation_labels_excluded_from_training": True,
            "validation_checkpoint_selection_absent": True,
            "fixed_epoch_state_used": self._diagnostics["validation_selection"]
            in {"final_fixed_epoch", "precommitted_completed_epoch"},
            "placebo_feature_present": "placebo_noise"
            in self._diagnostics["raw_input_columns"],
            "two_initialization_average": self._diagnostics[
                "fold_initialization_average_count"
            ]
            == 2,
        }
        return AdapterDiagnostics(
            assertions=assertions,
            observations=dict(self._diagnostics),
        )

    def raw_training_length_selections(self) -> tuple[int, ...]:
        """설정이 고정한 단일 시점 또는 미리 선언해 선택한 실제 epoch 횟수."""
        if self._raw_training_length_selection is None:
            raise RuntimeError(
                "RealMLP 고정 epoch 횟수는 검증 분할로 학습한 뒤에만 읽을 수 있다."
            )
        return (self._raw_training_length_selection,)

    def training_diagnostics(self) -> dict[str, object]:
        if self._diagnostics is None:
            raise RuntimeError("training_diagnostics는 fit 뒤에 호출해야 한다.")
        diagnostics = dict(self._diagnostics)
        diagnostics.update(
            {
                "prediction_calls": self._prediction_calls,
                "all_predictions_finite": self._all_predictions_finite,
                "importance_values_finite": (
                    None
                    if self._importance is None
                    else bool(np.isfinite(self._importance["gain"]).all())
                ),
                "placebo_importance": (
                    None
                    if self._importance is None
                    else float(
                        self._importance.set_index("feature").loc[
                            "placebo_noise", "gain"
                        ]
                    )
                ),
                "fit_seconds": self._fit_seconds,
                "importance_seconds": self._importance_seconds,
            }
        )
        return diagnostics
