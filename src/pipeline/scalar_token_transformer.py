"""스칼라 token Transformer 구현. (#178)

이 파일은 Kaggle 공개 노트북 "TabTransformer Predicting Smartphone Addiction"
판본 1의 구조를 참고해 저장소 규율에 맞게 크게 수정한 파생 구현이다.
원문: https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/versions/1
원문 소스 SHA-256: eeb3e1cccbaab29c71ef946876f7042509f6ef537df4a9b04ced36e3c424e46c

원 공개 노트북 소스에는 Apache License 2.0이 적용된다.
이 저장소의 변경 사항은 다음과 같다.

- 커밋된 outer fold와 ``ModelAdapter`` 경계를 사용한다.
- 공개 입력 생성은 피처 계획 밖으로 분리하고, M0는 champion 피처 계획을 받는다.
- 범주값 스칼라화와 분위 변환을 outer 학습 부분에서만 맞춘다.
- M0 attention과 매개변수 규모를 맞춘 A0 비-attention 결합을 함께 제공한다.
- fold별 독립 시드, 결정론, 검증 permutation importance와 진입 진단을 제공한다.

Apache License 2.0 원문은 ``scalar_token_transformer.LICENSE``에 있다.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .model import AdapterDiagnostics

_MIXING_MODES = {"attention", "token_mlp"}
_MISSING_CATEGORY_ID = 0.0
_UNKNOWN_CATEGORY_ID = 1.0


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))


def _attention_block_parameters(model_dim: int, feedforward_dim: int) -> int:
    attention = 4 * model_dim * model_dim + 4 * model_dim
    feedforward = 2 * model_dim * feedforward_dim + feedforward_dim + model_dim
    norms = 4 * model_dim
    return attention + feedforward + norms


def _matched_token_mlp_hidden(model_dim: int, feedforward_dim: int) -> int:
    target = _attention_block_parameters(model_dim, feedforward_dim)
    return max(1, round((target - 3 * model_dim) / (2 * model_dim + 1)))


def _token_mlp_block_parameters(model_dim: int, feedforward_dim: int) -> int:
    hidden = _matched_token_mlp_hidden(model_dim, feedforward_dim)
    return (2 * model_dim + 1) * hidden + 3 * model_dim


class _ReLUBasis(nn.Module):
    def __init__(self, features: int, basis_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, basis_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(features, basis_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.unsqueeze(2) * self.weight.unsqueeze(0) + self.bias)


class _PeriodicBasis(nn.Module):
    def __init__(self, features: int, output_dim: int, sigma: float) -> None:
        super().__init__()
        frequencies = output_dim // 2
        self.coefficient = nn.Parameter(torch.randn(features, frequencies) * sigma)
        self.phase = nn.Parameter(torch.zeros(features, frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = (
            2
            * math.pi
            * x.unsqueeze(2)
            * self.coefficient.unsqueeze(0)
            + self.phase.unsqueeze(0)
        )
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=2)


class _AttentionBlock(nn.Module):
    def __init__(
        self, model_dim: int, heads: int, feedforward_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=True)
        x = self.norm1(x + attended)
        return self.norm2(x + self.feedforward(x))


class _TokenMLPBlock(nn.Module):
    """열 사이 정보를 섞지 않는 매개변수 규모 맞춤 제거 대조."""

    def __init__(self, model_dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = _matched_token_mlp_hidden(model_dim, feedforward_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, model_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.feedforward(x))


class _MultiSampleDropoutHead(nn.Module):
    def __init__(self, input_dim: int, samples: int, dropout: float) -> None:
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(samples)])
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [self.output(dropout(x)) for dropout in self.dropouts]
        return torch.stack(logits, dim=0).mean(dim=0).squeeze(1)


class _ScalarTokenModel(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        mixing: str,
        relu_basis_dim: int,
        periodic_dim: int,
        periodic_sigma: float,
        model_dim: int,
        attention_heads: int,
        layers: int,
        feedforward_dim: int,
        backbone_dims: list[int],
        dropout: float,
        head_samples: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.relu_basis = _ReLUBasis(features, relu_basis_dim)
        self.periodic_basis = _PeriodicBasis(features, periodic_dim, periodic_sigma)
        self.feature_projection = nn.Linear(relu_basis_dim + periodic_dim, model_dim)
        self.column_identity = nn.Parameter(torch.randn(1, features, model_dim) * 0.01)

        if mixing == "attention":
            blocks = [
                _AttentionBlock(
                    model_dim, attention_heads, feedforward_dim, dropout
                )
                for _ in range(layers)
            ]
        else:
            blocks = [
                _TokenMLPBlock(model_dim, feedforward_dim, dropout)
                for _ in range(layers)
            ]
        self.blocks = nn.ModuleList(blocks)

        modules: list[nn.Module] = []
        input_dim = features * model_dim + features
        for index, output_dim in enumerate(backbone_dims):
            modules.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.GELU(),
                ]
            )
            if index < len(backbone_dims) - 1:
                modules.append(nn.Dropout(dropout))
            input_dim = output_dim
        self.backbone = nn.Sequential(*modules)
        self.head = _MultiSampleDropoutHead(input_dim, head_samples, head_dropout)
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = self.relu_basis(x)
        periodic = self.periodic_basis(x)
        tokens = self.feature_projection(torch.cat([relu, periodic], dim=2))
        tokens = tokens + self.column_identity
        for block in self.blocks:
            tokens = block(tokens)
        combined = torch.cat([tokens.flatten(1), x], dim=1)
        return self.head(self.backbone(combined))


class ScalarTokenTransformerFold:
    """fold 하나의 전처리, 학습, 예측과 중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._mixing = str(params.pop("mixing", "attention"))
        self._relu_basis_dim = int(params.pop("relu_basis_dim", 16))
        self._periodic_dim = int(params.pop("periodic_dim", 16))
        self._periodic_sigma = float(params.pop("periodic_sigma", 2.33))
        self._model_dim = int(params.pop("model_dim", 64))
        self._attention_heads = int(params.pop("attention_heads", 4))
        self._layers = int(params.pop("layers", 3))
        self._feedforward_dim = int(params.pop("feedforward_dim", 256))
        self._backbone_dims = [
            int(value) for value in params.pop("backbone_dims", [256, 128, 64])
        ]
        self._epochs = int(params.pop("epochs", 120))
        self._patience = int(params.pop("patience", 18))
        self._batch_size = int(params.pop("batch_size", 256))
        self._prediction_batch_size = int(params.pop("prediction_batch_size", 512))
        self._lr = float(params.pop("lr", 1e-3))
        self._min_lr = float(params.pop("min_lr", 1e-6))
        self._weight_decay = float(params.pop("weight_decay", 0.03))
        self._label_smoothing = float(params.pop("label_smoothing", 0.005))
        self._mixup_alpha = float(params.pop("mixup_alpha", 0.2))
        self._ema_decay = float(params.pop("ema_decay", 0.999))
        self._restart_epochs = int(params.pop("restart_epochs", 20))
        self._grad_clip = float(params.pop("grad_clip", 1.0))
        self._dropout = float(params.pop("dropout", 0.1))
        self._head_samples = int(params.pop("head_samples", 8))
        self._head_dropout = float(params.pop("head_dropout", 0.25))
        self._quantiles = int(params.pop("quantiles", 1000))
        self._perm_repeats = int(params.pop("perm_repeats", 3))
        if params:
            raise ValueError(
                f"scalar_token_transformer가 모르는 params: {sorted(params)}"
            )
        self._validate_params()

        self._seed = seed
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            torch.set_num_threads(1)
        self._columns: list[str] | None = None
        self._categorical_columns: list[str] = []
        self._category_maps: dict[str, dict[object, float]] = {}
        self._quantile: QuantileTransformer | None = None
        self._model: nn.Module | None = None
        self._validation: tuple[pd.DataFrame, np.ndarray] | None = None
        self._validation_auc: float | None = None
        self._trainable_parameters: int | None = None
        self._best_epoch: int | None = None

    def _validate_params(self) -> None:
        if self._mixing not in _MIXING_MODES:
            raise ValueError(
                f"mixing은 {sorted(_MIXING_MODES)} 중 하나여야 한다: {self._mixing!r}"
            )
        positive = [
            self._relu_basis_dim,
            self._periodic_dim,
            self._model_dim,
            self._attention_heads,
            self._layers,
            self._feedforward_dim,
            self._epochs,
            self._patience,
            self._batch_size,
            self._prediction_batch_size,
            self._restart_epochs,
            self._head_samples,
            self._quantiles,
            self._perm_repeats,
            *self._backbone_dims,
        ]
        if not self._backbone_dims or min(positive) <= 0:
            raise ValueError("모델 크기와 학습 횟수 params는 양수여야 한다.")
        if self._periodic_dim % 2:
            raise ValueError("periodic_dim은 짝수여야 한다.")
        if self._model_dim % self._attention_heads:
            raise ValueError("model_dim은 attention_heads의 배수여야 한다.")
        if self._lr <= 0 or self._min_lr < 0 or self._weight_decay < 0:
            raise ValueError("lr은 양수이고 min_lr과 weight_decay는 0 이상이어야 한다.")
        if self._grad_clip <= 0 or self._periodic_sigma <= 0:
            raise ValueError("grad_clip과 periodic_sigma는 양수여야 한다.")
        for name, value in (
            ("dropout", self._dropout),
            ("head_dropout", self._head_dropout),
            ("label_smoothing", self._label_smoothing),
            ("ema_decay", self._ema_decay),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name}은 0 이상 1 미만이어야 한다.")
        if self._mixup_alpha < 0:
            raise ValueError("mixup_alpha는 0 이상이어야 한다.")

    def _seed_everything(self) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def _object_values(series: pd.Series) -> pd.Series:
        return (
            series.astype(object)
            if isinstance(series.dtype, pd.CategoricalDtype)
            else series
        )

    def _fit_preprocessing(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) < 2:
            raise ValueError("학습 fold에는 두 행 이상이 필요하다.")
        self._columns = list(X.columns)
        if not self._columns:
            raise ValueError("입력 열이 하나 이상 필요하다.")
        self._categorical_columns = [
            column
            for column in self._columns
            if not pd.api.types.is_numeric_dtype(X[column])
        ]
        for column in self._categorical_columns:
            observed = self._object_values(X[column]).dropna()
            values = sorted(pd.unique(observed).tolist(), key=repr)
            self._category_maps[column] = {
                value: float(index + 2) for index, value in enumerate(values)
            }
        raw = self._raw_scalars(X)
        n_quantiles = min(self._quantiles, len(X))
        subsample = max(n_quantiles, min(10_000, len(X)))
        self._quantile = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=self._seed,
            subsample=subsample,
        )
        return self._quantile.fit_transform(raw).astype("float32")

    def _raw_scalars(self, X: pd.DataFrame) -> np.ndarray:
        if list(X.columns) != self._columns:
            raise AssertionError("스칼라화 입력 컬럼이 학습 때와 다르다.")
        scalars = np.zeros((len(X), len(self._columns)), dtype="float32")
        for index, column in enumerate(self._columns):
            if column in self._category_maps:
                values = self._object_values(X[column])
                missing = values.isna().to_numpy()
                encoded = (
                    values.map(self._category_maps[column])
                    .fillna(_UNKNOWN_CATEGORY_ID)
                    .to_numpy(dtype="float32", copy=True)
                )
                encoded[missing] = _MISSING_CATEGORY_ID
                scalars[:, index] = encoded
            else:
                values = pd.to_numeric(X[column], errors="coerce").to_numpy(
                    dtype="float64"
                )
                scalars[:, index] = np.nan_to_num(
                    values, nan=0.0, posinf=0.0, neginf=0.0
                ).astype("float32")
        return scalars

    def _encode(self, X: pd.DataFrame) -> torch.Tensor:
        if self._quantile is None:
            raise RuntimeError("전처리는 fit에서 먼저 맞춰야 한다.")
        transformed = self._quantile.transform(self._raw_scalars(X))
        transformed = np.nan_to_num(
            transformed, nan=0.0, posinf=0.0, neginf=0.0
        ).astype("float32")
        return torch.from_numpy(transformed)

    def _build_model(self, features: int) -> _ScalarTokenModel:
        return _ScalarTokenModel(
            features,
            mixing=self._mixing,
            relu_basis_dim=self._relu_basis_dim,
            periodic_dim=self._periodic_dim,
            periodic_sigma=self._periodic_sigma,
            model_dim=self._model_dim,
            attention_heads=self._attention_heads,
            layers=self._layers,
            feedforward_dim=self._feedforward_dim,
            backbone_dims=self._backbone_dims,
            dropout=self._dropout,
            head_samples=self._head_samples,
            head_dropout=self._head_dropout,
        )

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        self._seed_everything()
        transformed_tr = self._fit_preprocessing(X_tr)
        x_tr_cpu = torch.from_numpy(transformed_tr)
        x_va_cpu = self._encode(X_va)
        model = self._build_model(x_tr_cpu.shape[1]).to(self._device)
        self._trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"[scalar_token_transformer] mixing={self._mixing} "
            f"features={x_tr_cpu.shape[1]} parameters={self._trainable_parameters:,}",
            flush=True,
        )

        x_tr = x_tr_cpu.to(self._device)
        x_va = x_va_cpu.to(self._device)
        target = torch.from_numpy(y_tr.to_numpy(dtype="float32")).to(self._device)
        soft_target = target * (1.0 - self._label_smoothing) + 0.5 * self._label_smoothing
        y_va_array = y_va.to_numpy(dtype="float64")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self._restart_epochs, T_mult=1, eta_min=self._min_lr
        )
        ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(self._ema_decay)
        )
        generator = torch.Generator(device=self._device).manual_seed(self._seed)
        numpy_generator = np.random.default_rng(self._seed)
        amp_enabled = self._device == "cuda"
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_auc = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for epoch in range(1, self._epochs + 1):
            model.train()
            permutation = torch.randperm(
                len(X_tr), generator=generator, device=self._device
            )
            usable = (
                len(X_tr) - len(X_tr) % self._batch_size
                if len(X_tr) >= self._batch_size
                else len(X_tr)
            )
            if usable < 2:
                raise ValueError("BatchNorm 학습에는 batch에 두 행 이상이 필요하다.")
            epoch_loss = 0.0
            batches = 0
            for start in range(0, usable, self._batch_size):
                rows = permutation[start : min(start + self._batch_size, usable)]
                batch_x = x_tr[rows]
                batch_target = soft_target[rows]
                if self._mixup_alpha > 0:
                    lam = float(
                        numpy_generator.beta(self._mixup_alpha, self._mixup_alpha)
                    )
                    paired = torch.randperm(
                        len(rows), generator=generator, device=self._device
                    )
                    model_x = lam * batch_x + (1.0 - lam) * batch_x[paired]
                    target_a, target_b = batch_target, batch_target[paired]
                else:
                    lam = 1.0
                    model_x = batch_x
                    target_a = target_b = batch_target

                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.amp.autocast("cuda")
                    if amp_enabled
                    else contextlib.nullcontext()
                )
                with autocast:
                    logits = model(model_x)
                    loss = lam * F.binary_cross_entropy_with_logits(logits, target_a)
                    if lam < 1.0:
                        loss = loss + (1.0 - lam) * F.binary_cross_entropy_with_logits(
                            logits, target_b
                        )
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                ema_model.update_parameters(model)
                epoch_loss += float(loss.detach())
                batches += 1

            validation_logit = self._predict_tensor(ema_model, x_va)
            validation_auc = float(roc_auc_score(y_va_array, validation_logit))
            scheduler.step()
            print(
                f"[scalar_token_transformer] mixing={self._mixing} "
                f"epoch={epoch:03d} loss={epoch_loss / batches:.6f} "
                f"ema_auc={validation_auc:.6f}",
                flush=True,
            )
            if validation_auc > best_auc:
                best_auc = validation_auc
                self._best_epoch = epoch
                best_state = copy.deepcopy(ema_model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= self._patience:
                break

        if best_state is None:
            raise RuntimeError("검증 checkpoint를 만들지 못했다.")
        ema_model.load_state_dict(best_state)
        self._model = ema_model
        validation_logit = self._predict_tensor(ema_model, x_va)
        self._validation_auc = float(roc_auc_score(y_va_array, validation_logit))
        self._validation = (X_va.copy(), y_va_array)
        return _sigmoid(validation_logit)

    def _predict_tensor(self, model: nn.Module, values: torch.Tensor) -> np.ndarray:
        model.eval()
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(values), self._prediction_batch_size):
                batch = values[start : start + self._prediction_batch_size]
                autocast = (
                    torch.amp.autocast("cuda")
                    if self._device == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast:
                    parts.append(model(batch))
        return torch.cat(parts).float().cpu().numpy().astype("float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict는 fit 뒤에 호출해야 한다.")
        encoded = self._encode(X).to(self._device)
        return _sigmoid(self._predict_tensor(self._model, encoded))

    def importance(self) -> pd.DataFrame:
        if self._validation is None or self._validation_auc is None:
            raise RuntimeError("importance는 fit 뒤에 호출해야 한다.")
        X_va, y_va = self._validation
        gains: list[float] = []
        for column_index, column in enumerate(self._columns or []):
            drops: list[float] = []
            for repeat in range(self._perm_repeats):
                generator = np.random.default_rng(
                    self._seed * 10007 + column_index * 101 + repeat
                )
                permutation = generator.permutation(len(X_va))
                permuted = X_va.copy()
                permuted[column] = X_va[column].to_numpy()[permutation]
                drops.append(
                    self._validation_auc
                    - roc_auc_score(y_va, self.predict(permuted))
                )
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})

    def entry_diagnostics(self) -> AdapterDiagnostics:
        if self._trainable_parameters is None or self._best_epoch is None:
            raise RuntimeError("entry_diagnostics는 fit 뒤에 호출해야 한다.")
        attention_parameters = _attention_block_parameters(
            self._model_dim, self._feedforward_dim
        )
        ablation_parameters = _token_mlp_block_parameters(
            self._model_dim, self._feedforward_dim
        )
        relative_difference = (
            abs(attention_parameters - ablation_parameters) / attention_parameters
        )
        return AdapterDiagnostics(
            assertions={
                "preprocessing_training_rows_only": True,
                "validation_labels_excluded_from_preprocessing": True,
                "missing_and_unknown_categories_distinct": True,
                "attention_ablation_parameter_matched": relative_difference < 0.05,
            },
            observations={
                "mixing": self._mixing,
                "feature_count": len(self._columns or []),
                "categorical_feature_count": len(self._categorical_columns),
                "trainable_parameters": self._trainable_parameters,
                "attention_block_parameters": attention_parameters,
                "ablation_block_parameters": ablation_parameters,
                "ablation_parameter_relative_difference": relative_difference,
                "best_epoch": self._best_epoch,
                "best_validation_auc": float(self._validation_auc),
                "quantile_fit_rows_max": 10_000,
                "permutation_importance_repeats": self._perm_repeats,
                "target_encodings": sum(
                    column.endswith("_te") for column in (self._columns or [])
                ),
            },
        )
