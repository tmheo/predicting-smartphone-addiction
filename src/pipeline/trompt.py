"""TALENT 판본을 기준으로 한 Trompt fold 학습기. (#145)

Trompt는 행마다 prompt와 열 임베딩으로 특성 중요도를 계산하고, Cell 여섯 개의
출력을 함께 학습한 뒤 평균한다.
구조는 MIT TALENT 커밋 08301d670a7c854bcf3a73298763484ba58eecdb를 기준으로
이 저장소의 고정 outer fold, 관찰과 permutation importance 계약에 맞춰 옮겼다.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

# PyTorch requires this process-level setting before the first CUDA CuBLAS call
# when deterministic algorithms are enabled.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

TALENT_REVISION = "08301d670a7c854bcf3a73298763484ba58eecdb"
GIB = 1024**3


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class _LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, width))
        self.bias = nn.Parameter(torch.empty(n_features, width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class _CategoricalEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], width: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, width) for cardinality in cardinalities]
        )
        for embedding in self.embeddings:
            bound = embedding.weight.shape[1] ** -0.5
            nn.init.uniform_(embedding.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack(
            [embedding(x[:, index]) for index, embedding in enumerate(self.embeddings)],
            dim=1,
        )


class _TromptEmbedding(nn.Module):
    def __init__(
        self, n_numeric: int, categorical_cardinalities: list[int], width: int
    ) -> None:
        super().__init__()
        self.numeric = _LinearEmbeddings(n_numeric, width) if n_numeric else None
        self.categorical = (
            _CategoricalEmbeddings(categorical_cardinalities, width)
            if categorical_cardinalities
            else None
        )
        self.numeric_norm = nn.LayerNorm(width)
        self.categorical_norm = nn.LayerNorm(width)

    def forward(self, x_numeric: Tensor, x_categorical: Tensor | None) -> Tensor:
        parts: list[Tensor] = []
        if self.numeric is not None:
            parts.append(self.numeric_norm(torch.relu(self.numeric(x_numeric))))
        if self.categorical is not None:
            assert x_categorical is not None
            parts.append(self.categorical_norm(self.categorical(x_categorical)))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


class _ImportanceGetter(nn.Module):
    def __init__(self, prompts: int, columns: int, width: int) -> None:
        super().__init__()
        self.column_embedding = nn.Parameter(torch.empty(columns, width))
        self.prompt_embedding = nn.Parameter(torch.empty(prompts, width))
        nn.init.normal_(self.column_embedding, std=0.01)
        nn.init.normal_(self.prompt_embedding, std=0.01)
        self.dense = nn.Linear(2 * width, width)
        self.prompt_norm = nn.LayerNorm(width)
        self.column_norm = nn.LayerNorm(width)

    def forward(self, previous: Tensor) -> Tensor:
        prompts = self.prompt_embedding.unsqueeze(0).expand(len(previous), -1, -1)
        dense = self.dense(torch.cat((self.prompt_norm(prompts), previous), dim=-1))
        dense = dense + prompts + previous
        columns = self.column_norm(self.column_embedding).unsqueeze(0)
        return torch.softmax(dense @ columns.transpose(1, 2), dim=-1)


class _Expander(nn.Module):
    def __init__(self, prompts: int) -> None:
        super().__init__()
        if prompts % 2:
            raise ValueError("Trompt prompt 수는 GroupNorm을 위해 짝수여야 한다.")
        self.linear = nn.Linear(1, prompts)
        self.norm = nn.GroupNorm(2, prompts)

    def forward(self, embedded: Tensor) -> Tensor:
        expanded = torch.relu(self.linear(embedded.unsqueeze(-1)))
        expanded = expanded.permute(0, 3, 1, 2)
        return embedded.unsqueeze(1) + self.norm(expanded)


class _TromptCell(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_cardinalities: list[int],
        prompts: int,
        width: int,
    ) -> None:
        super().__init__()
        columns = n_numeric + len(categorical_cardinalities)
        self.embedding = _TromptEmbedding(n_numeric, categorical_cardinalities, width)
        self.importance = _ImportanceGetter(prompts, columns, width)
        self.expander = _Expander(prompts)

    def forward(
        self, x_numeric: Tensor, x_categorical: Tensor | None, previous: Tensor
    ) -> Tensor:
        expanded = self.expander(self.embedding(x_numeric, x_categorical))
        importance = self.importance(previous)
        return (importance.unsqueeze(-1) * expanded).sum(dim=2)


class _TromptDecoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.prompt_weight = nn.Linear(width, 1)
        self.hidden = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 2)

    def forward(self, prompts: Tensor) -> Tensor:
        weights = torch.softmax(self.prompt_weight(prompts).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * prompts).sum(dim=-2)
        return self.output(self.norm(torch.relu(self.hidden(pooled))))


class _Trompt(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_cardinalities: list[int],
        prompts: int,
        width: int,
        cells: int,
    ) -> None:
        super().__init__()
        self.cells = nn.ModuleList(
            [
                _TromptCell(n_numeric, categorical_cardinalities, prompts, width)
                for _ in range(cells)
            ]
        )
        self.decoder = _TromptDecoder(width)
        self.initial_prompts = nn.Parameter(torch.empty(prompts, width))
        nn.init.normal_(self.initial_prompts, std=0.01)

    def forward_cells(self, x_numeric: Tensor, x_categorical: Tensor | None) -> Tensor:
        prompts = self.initial_prompts.unsqueeze(0).expand(len(x_numeric), -1, -1)
        outputs: list[Tensor] = []
        for cell in self.cells:
            prompts = cell(x_numeric, x_categorical, prompts)
            outputs.append(self.decoder(prompts))
        return torch.stack(outputs, dim=1)

    def forward(self, x_numeric: Tensor, x_categorical: Tensor | None) -> Tensor:
        return self.forward_cells(x_numeric, x_categorical).mean(dim=1)


@dataclass(frozen=True)
class _Encoded:
    numeric: Tensor
    categorical: Tensor | None


class _FoldEncoder:
    def __init__(self) -> None:
        self.columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, int]] = {}

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        return isinstance(
            series.dtype, pd.CategoricalDtype
        ) or not pd.api.types.is_numeric_dtype(series.dtype)

    def fit(self, X: pd.DataFrame) -> None:
        self.columns = list(X.columns)
        self.categorical_columns = [
            column for column in self.columns if self._is_categorical(X[column])
        ]
        self.numeric_columns = [
            column for column in self.columns if column not in self.categorical_columns
        ]
        if not self.numeric_columns:
            raise ValueError("Trompt 기준 구성은 하나 이상의 수치 열이 필요하다.")
        for column in self.numeric_columns:
            values = pd.to_numeric(X[column], errors="raise").astype("float64")
            median = float(values.median())
            if not np.isfinite(median):
                raise ValueError(
                    f"Trompt 학습 fold에서 {column}의 중앙값이 유한하지 않다."
                )
            filled = values.fillna(median)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            self.medians[column] = median
            self.means[column] = mean
            self.scales[column] = scale if np.isfinite(scale) and scale > 0 else 1.0
        for column in self.categorical_columns:
            values = X[column].astype("object")
            known = sorted({str(value) for value in values[values.notna()]})
            self.category_maps[column] = {
                value: index for index, value in enumerate(known)
            }

    def cardinalities(self) -> list[int]:
        return [
            len(self.category_maps[column]) + 2 for column in self.categorical_columns
        ]

    def transform(self, X: pd.DataFrame) -> _Encoded:
        if list(X.columns) != self.columns:
            raise ValueError("Trompt 예측 컬럼이 학습 때와 다르다.")
        numeric = np.empty((len(X), len(self.numeric_columns)), dtype="float32")
        for index, column in enumerate(self.numeric_columns):
            values = pd.to_numeric(X[column], errors="raise").astype("float64")
            values = values.fillna(self.medians[column])
            numeric[:, index] = (
                (values.to_numpy() - self.means[column]) / self.scales[column]
            ).astype("float32")
        categorical: Tensor | None = None
        if self.categorical_columns:
            encoded = np.empty((len(X), len(self.categorical_columns)), dtype="int64")
            for index, column in enumerate(self.categorical_columns):
                mapping = self.category_maps[column]
                missing_code = len(mapping)
                unknown_code = missing_code + 1
                values = X[column].astype("object")
                encoded[:, index] = [
                    missing_code
                    if pd.isna(value)
                    else mapping.get(str(value), unknown_code)
                    for value in values
                ]
            categorical = torch.from_numpy(encoded)
        return _Encoded(torch.from_numpy(numeric), categorical)


class TromptFold:
    """Trompt 기준 구성의 fold 학습, 예측, 중요도와 진입 진단을 소유한다."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._prompts = int(params.pop("prompts", 128))
        self._width = int(params.pop("width", 128))
        self._cells = int(params.pop("cells", 6))
        self._epochs = int(params.pop("epochs", 50))
        self._batch_size = int(params.pop("batch_size", 256))
        self._eval_batch_size = int(params.pop("eval_batch_size", 4096))
        self._lr = float(params.pop("lr", 3e-4))
        self._weight_decay = float(params.pop("weight_decay", 1e-5))
        self._patience = int(params.pop("patience", 5))
        self._perm_sample = int(params.pop("perm_sample", 8192))
        self._perm_repeats = int(params.pop("perm_repeats", 1))
        raw_projected_limit = params.pop("max_projected_5fold_hours", None)
        self._max_projected_5fold_hours = (
            None if raw_projected_limit is None else float(raw_projected_limit)
        )
        self._device_name = params.pop("device", None)
        if params:
            raise ValueError(f"trompt가 모르는 params: {sorted(params)}")
        if self._prompts < 2 or self._prompts % 2:
            raise ValueError("prompts는 2 이상의 짝수여야 한다.")
        if (
            self._max_projected_5fold_hours is not None
            and self._max_projected_5fold_hours <= 0
        ):
            raise ValueError("max_projected_5fold_hours는 양수여야 한다.")
        if (
            min(
                self._width,
                self._cells,
                self._epochs,
                self._batch_size,
                self._eval_batch_size,
                self._patience,
                self._perm_sample,
                self._perm_repeats,
            )
            < 1
        ):
            raise ValueError(
                "Trompt 크기, epoch, batch와 importance 설정은 양수여야 한다."
            )
        self._seed = seed
        self._device = torch.device(
            self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self._device.type == "cpu":
            torch.set_num_threads(1)
        self._encoder = _FoldEncoder()
        self._model: _Trompt | None = None
        self._validation: tuple[pd.DataFrame, np.ndarray] | None = None
        self._epoch_seconds: list[float] = []
        self._training_losses: list[float] = []
        self._validation_aucs: list[float] = []
        self._best_epoch = 0
        self._probe: dict[str, object] = {}
        self._capacity_attempts: list[dict[str, object]] = []
        self._effective_prompts = self._prompts
        self._effective_batch_size = self._batch_size
        self._effective_eval_batch_size = min(
            self._eval_batch_size, self._effective_batch_size
        )
        self._projected_5fold_training_seconds: float | None = None
        self._entry_abort_reason: str | None = None

    def _new_model(self, prompts: int) -> _Trompt:
        return _Trompt(
            len(self._encoder.numeric_columns),
            self._encoder.cardinalities(),
            prompts,
            self._width,
            self._cells,
        ).to(self._device)

    @staticmethod
    def _mean_probabilities(cell_logits: Tensor) -> Tensor:
        return torch.softmax(cell_logits.mean(dim=1), dim=-1)[:, 1]

    def _probe_once(
        self, encoded: _Encoded, labels: Tensor, prompts: int, batch_size: int
    ) -> tuple[dict[str, object], np.ndarray]:
        _seed_everything(self._seed)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._device)
        model = self._new_model(prompts)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        take = min(batch_size, len(labels))
        x_numeric = encoded.numeric[:take].to(self._device)
        x_categorical = (
            None
            if encoded.categorical is None
            else encoded.categorical[:take].to(self._device)
        )
        y = labels[:take].to(self._device)
        model.train()
        cell_logits = model.forward_cells(x_numeric, x_categorical)
        repeated_y = y.repeat_interleave(self._cells)
        losses = nn.functional.cross_entropy(
            cell_logits.reshape(-1, 2), repeated_y, reduction="none"
        ).reshape(take, self._cells)
        loss = losses.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        probabilities = self._mean_probabilities(cell_logits).detach().cpu().numpy()
        peak = (
            int(torch.cuda.max_memory_reserved(self._device))
            if self._device.type == "cuda"
            else None
        )
        result: dict[str, object] = {
            "cell_output_shape": list(cell_logits.shape),
            "cell_losses": [
                float(value) for value in losses.mean(dim=0).detach().cpu()
            ],
            "mean_prediction_shape": list(probabilities.shape),
            "mean_prediction_min": float(probabilities.min()),
            "mean_prediction_max": float(probabilities.max()),
            "loss": float(loss.detach().cpu()),
            "cuda_peak_reserved_bytes": peak,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
        del optimizer, model, cell_logits, losses, loss, x_numeric, x_categorical
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return result, probabilities

    def _capacity_limit(self) -> int | None:
        if self._device.type != "cuda":
            return None
        total = int(torch.cuda.get_device_properties(self._device).total_memory)
        return 14 * GIB if total <= 20 * GIB else 20 * GIB

    def _choose_capacity(self, encoded: _Encoded, labels: Tensor) -> None:
        attempts = [
            (self._prompts, self._batch_size),
            (self._prompts, 128),
            (64, 128),
        ]
        attempts = list(dict.fromkeys(attempts))
        limit = self._capacity_limit()
        for prompts, batch_size in attempts:
            attempt: dict[str, object] = {
                "prompts": prompts,
                "batch_size": batch_size,
                "memory_limit_bytes": limit,
            }
            try:
                first, first_predictions = self._probe_once(
                    encoded, labels, prompts, batch_size
                )
                second, second_predictions = self._probe_once(
                    encoded, labels, prompts, batch_size
                )
            except torch.OutOfMemoryError:
                attempt["status"] = "cuda_oom"
                self._capacity_attempts.append(attempt)
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            max_difference = float(
                np.max(np.abs(first_predictions - second_predictions))
            )
            deterministic = bool(
                np.array_equal(first_predictions, second_predictions)
                and first["cell_losses"] == second["cell_losses"]
            )
            peak = first["cuda_peak_reserved_bytes"]
            within_limit = limit is None or (peak is not None and int(peak) <= limit)
            attempt.update(
                {
                    "status": "pass" if within_limit else "memory_limit",
                    "peak_reserved_bytes": peak,
                    "deterministic": deterministic,
                    "max_prediction_difference": max_difference,
                }
            )
            self._capacity_attempts.append(attempt)
            if not deterministic:
                raise RuntimeError("Trompt 같은 seed 배치 진입 진단이 결정적이지 않다.")
            if within_limit:
                self._effective_prompts = prompts
                self._effective_batch_size = batch_size
                self._effective_eval_batch_size = min(self._eval_batch_size, batch_size)
                self._probe = {
                    **first,
                    "deterministic": deterministic,
                    "max_prediction_difference": max_difference,
                }
                return
        raise RuntimeError(
            "Trompt prompt 64, batch 128도 메모리 진입 관문을 통과하지 못했다."
        )

    def _predict_encoded(self, encoded: _Encoded) -> np.ndarray:
        assert self._model is not None
        self._model.eval()
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(
                0, len(encoded.numeric), self._effective_eval_batch_size
            ):
                stop = start + self._effective_eval_batch_size
                x_numeric = encoded.numeric[start:stop].to(self._device)
                x_categorical = (
                    None
                    if encoded.categorical is None
                    else encoded.categorical[start:stop].to(self._device)
                )
                logits = self._model.forward_cells(x_numeric, x_categorical)
                predictions.append(
                    self._mean_probabilities(logits).cpu().numpy().astype("float64")
                )
        return np.concatenate(predictions)

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        if list(X_tr.columns) != list(X_va.columns):
            raise ValueError("Trompt 학습과 검증 컬럼이 다르다.")
        if len(set(y_tr.unique())) != 2:
            raise ValueError("Trompt는 이진 분류 라벨 두 개가 필요하다.")
        self._encoder.fit(X_tr)
        train = self._encoder.transform(X_tr)
        validation = self._encoder.transform(X_va)
        train_y = torch.from_numpy(y_tr.to_numpy(dtype="int64", copy=True))
        validation_y = y_va.to_numpy(dtype="float64", copy=True)
        self._validation = (X_va.copy(), validation_y)
        self._choose_capacity(train, train_y)

        _seed_everything(self._seed)
        self._model = self._new_model(self._effective_prompts)
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        generator = torch.Generator().manual_seed(self._seed)
        best_auc = -math.inf
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0

        for epoch in range(self._epochs):
            started = time.monotonic()
            self._model.train()
            permutation = torch.randperm(len(train_y), generator=generator)
            loss_sum = 0.0
            rows_seen = 0
            for start in range(0, len(permutation), self._effective_batch_size):
                batch = permutation[start : start + self._effective_batch_size]
                x_numeric = train.numeric[batch].to(self._device)
                x_categorical = (
                    None
                    if train.categorical is None
                    else train.categorical[batch].to(self._device)
                )
                y = train_y[batch].to(self._device)
                cell_logits = self._model.forward_cells(x_numeric, x_categorical)
                loss = nn.functional.cross_entropy(
                    cell_logits.reshape(-1, 2), y.repeat_interleave(self._cells)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(batch)
                rows_seen += len(batch)
            training_loss = loss_sum / rows_seen
            validation_pred = self._predict_encoded(validation)
            auc = float(roc_auc_score(validation_y, validation_pred))
            seconds = float(time.monotonic() - started)
            self._training_losses.append(training_loss)
            self._validation_aucs.append(auc)
            self._epoch_seconds.append(seconds)
            self._projected_5fold_training_seconds = (
                float(np.mean(self._epoch_seconds)) * self._epochs * 5
            )
            if auc > best_auc:
                best_auc = auc
                self._best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self._model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"[trompt] epoch={epoch + 1}/{self._epochs} "
                f"loss={training_loss:.6f} valAUC={auc:.6f} "
                f"seconds={seconds:.1f} stale={stale_epochs}/{self._patience}",
                flush=True,
            )
            if (
                self._max_projected_5fold_hours is not None
                and self._projected_5fold_training_seconds
                > self._max_projected_5fold_hours * 3600
            ):
                projected_hours = self._projected_5fold_training_seconds / 3600
                self._entry_abort_reason = (
                    f"Trompt 5-fold 학습 예상 {projected_hours:.2f}시간이 "
                    f"한도 {self._max_projected_5fold_hours:.2f}시간을 넘는다."
                )
                print(f"[trompt] entry-stop {self._entry_abort_reason}", flush=True)
                break
            if stale_epochs >= self._patience:
                break
        assert best_state is not None
        self._model.load_state_dict(best_state)
        return self._predict_encoded(validation)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_encoded(self._encoder.transform(X))

    def importance(self) -> pd.DataFrame:
        assert self._validation is not None
        X_va, y_va = self._validation
        if len(X_va) > self._perm_sample:
            keep = np.random.default_rng(self._seed).choice(
                len(X_va), size=self._perm_sample, replace=False
            )
            keep.sort()
            X_va = X_va.iloc[keep].copy()
            y_va = y_va[keep]
        base = roc_auc_score(y_va, self.predict(X_va))
        gains: list[float] = []
        for column_index, column in enumerate(self._encoder.columns):
            drops: list[float] = []
            for repeat in range(self._perm_repeats):
                rng = np.random.default_rng(
                    self._seed * 10007 + column_index * 101 + repeat
                )
                permuted = X_va.copy()
                order = rng.permutation(len(permuted))
                permuted[column] = X_va[column].iloc[order].set_axis(permuted.index)
                drops.append(base - roc_auc_score(y_va, self.predict(permuted)))
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._encoder.columns, "gain": gains})

    def entry_abort_reason(self) -> str | None:
        return self._entry_abort_reason

    def entry_diagnostics(self):
        from . import model

        return model.AdapterDiagnostics(
            assertions={"trompt_capacity_probe": bool(self._probe)},
            observations={
                "talent_revision": TALENT_REVISION,
                "requested_prompts": self._prompts,
                "effective_prompts": self._effective_prompts,
                "width": self._width,
                "cells": self._cells,
                "requested_batch_size": self._batch_size,
                "effective_batch_size": self._effective_batch_size,
                "requested_eval_batch_size": self._eval_batch_size,
                "effective_eval_batch_size": self._effective_eval_batch_size,
                "numeric_columns": len(self._encoder.numeric_columns),
                "categorical_columns": len(self._encoder.categorical_columns),
                "input_columns": len(self._encoder.columns),
                "capacity_attempts": self._capacity_attempts,
                "batch_probe": self._probe,
                "training_losses": self._training_losses,
                "epoch_validation_aucs": self._validation_aucs,
                "epoch_seconds": self._epoch_seconds,
                "best_epoch": self._best_epoch,
                "projected_5fold_training_seconds": self._projected_5fold_training_seconds,
                "max_projected_5fold_hours": self._max_projected_5fold_hours,
                "entry_abort_reason": self._entry_abort_reason,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            },
        )
