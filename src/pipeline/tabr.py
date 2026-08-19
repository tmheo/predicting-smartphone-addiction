"""전체판 TabR fold 학습기. (#199)

모델과 검색 수식은 Yandex Research 공식 MIT 구현의 ``bin/tabr.py``를
이 저장소의 ModelAdapter 계약에 맞게 옮겼다.
기준 판본은 yandex-research/tabular-dl-tabr 커밋
17baa9082506f8e7a0f8d11bb1e08212926a1507이다.

공식 기본 TabR(default-evaluation 설정)은 TabR-S(#142)와 같은 선형 encoder,
한 블록 predictor 구조를 쓰며, 차이는 학습 방식이다:
문맥을 고정하지 않고 매 배치 후보 키를 다시 계산하며,
검증 AUC 조기 종료(patience 16)까지 학습한다.
선택 항목으로 공식 튜닝판이 쓰는 PLR(lite) 수치 임베딩을 지원한다.

후보 키 계산은 공식 ``memory_efficient=True`` 경로를 따른다:
후보 전체 키는 기울기 없이 계산하고, 뽑힌 문맥 행만 기울기와 함께 다시 계산한다.

수치 열은 학습 fold에서 적합한 분위 정규화, 범주 열은 학습 fold에서 관측한 값의
one-hot 표현을 사용한다.
후보 저장소는 outer 학습 fold의 행과 라벨만 보유하고 학습 query는 자기 행을
문맥에서 제외한다.

CUDA 경로는 exact ``faiss.GpuIndexFlatL2``를 요구한다.
CPU 경로의 torch exact 검색은 소형 회귀 시험 전용이다.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor, nn

OFFICIAL_COMMIT = "17baa9082506f8e7a0f8d11bb1e08212926a1507"


class _SplitEncodedTable:
    """학습 fold 전용 수치 분위와 범주 어휘를 소유하고 수치·one-hot을 분리해 낸다."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self.columns: list[str] = []
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.medians: dict[str, float] = {}
        self.categories: dict[str, list[object]] = {}
        self.quantiles: QuantileTransformer | None = None
        self.numeric_dim = 0
        self.onehot_dim = 0

    def fit(self, X: pd.DataFrame) -> None:
        self.columns = list(X.columns)
        self.categorical = [
            column
            for column in self.columns
            if isinstance(X[column].dtype, pd.CategoricalDtype)
        ]
        self.numeric = [
            column for column in self.columns if column not in self.categorical
        ]
        numeric_values: list[np.ndarray] = []
        for column in self.numeric:
            values = pd.to_numeric(X[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            median = values.median()
            self.medians[column] = float(median) if pd.notna(median) else 0.0
            numeric_values.append(
                values.fillna(self.medians[column]).to_numpy(dtype="float64")
            )
        if numeric_values:
            matrix = np.column_stack(numeric_values)
            self.quantiles = QuantileTransformer(
                n_quantiles=min(1000, len(X)),
                output_distribution="normal",
                subsample=2_000_000_000,
                random_state=self._seed,
            )
            self.quantiles.fit(matrix)
        for column in self.categorical:
            values = X[column].astype(object)
            self.categories[column] = sorted(
                pd.unique(values.dropna()).tolist(), key=lambda value: str(value)
            )
        self.numeric_dim = len(self.numeric)
        self.onehot_dim = sum(len(values) + 1 for values in self.categories.values())
        if self.numeric_dim + self.onehot_dim == 0:
            raise ValueError("TabR 학습 행렬에 피처가 없다.")

    def transform(self, X: pd.DataFrame) -> tuple[Tensor, Tensor]:
        if list(X.columns) != self.columns:
            raise AssertionError("TabR 입력 컬럼이 학습 때와 다르다.")
        if self.numeric:
            matrix = np.column_stack(
                [
                    pd.to_numeric(X[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(self.medians[column])
                    .to_numpy(dtype="float64")
                    for column in self.numeric
                ]
            )
            assert self.quantiles is not None
            numeric = torch.from_numpy(
                self.quantiles.transform(matrix).astype("float32")
            )
        else:
            numeric = torch.zeros((len(X), 0), dtype=torch.float32)
        blocks: list[np.ndarray] = []
        for column in self.categorical:
            values = X[column].astype(object)
            categories = self.categories[column]
            mapping = {value: index for index, value in enumerate(categories)}
            codes = values.map(mapping).fillna(len(categories)).to_numpy(dtype="int64")
            block = np.zeros((len(X), len(categories) + 1), dtype="float32")
            block[np.arange(len(X)), codes] = 1.0
            blocks.append(block)
        onehot = (
            torch.from_numpy(np.concatenate(blocks, axis=1))
            if blocks
            else torch.zeros((len(X), 0), dtype=torch.float32)
        )
        return numeric, onehot


class _PeriodicEmbeddings(nn.Module):
    """공식 lib.deep.PeriodicEmbeddings."""

    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        return torch.cat([torch.cos(x), torch.sin(x)], -1)


class _PLREmbeddings(nn.Sequential):
    """공식 lib.deep.PLREmbeddings의 lite 판."""

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
    ) -> None:
        super().__init__(
            _PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            nn.Linear(2 * n_frequencies, d_embedding),
            nn.ReLU(),
        )


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        d_main: int,
        d_block: int,
        dropout0: float,
        dropout1: float,
        *,
        prenorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if prenorm:
            layers.append(nn.LayerNorm(d_main))
        layers.extend(
            [
                nn.Linear(d_main, d_block),
                nn.ReLU(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _TabR(nn.Module):
    """공식 TabR의 E, R, P 수식. 선택적 PLR(lite) 수치 임베딩을 포함한다."""

    def __init__(
        self,
        n_numeric: int,
        d_onehot: int,
        d_main: int,
        d_multiplier: float,
        context_dropout: float,
        dropout0: float,
        dropout1: float,
        num_embeddings: dict | None,
    ) -> None:
        super().__init__()
        d_block = int(d_main * d_multiplier)
        if num_embeddings is None:
            self.num_embeddings = None
            d_in = n_numeric + d_onehot
        else:
            if n_numeric == 0:
                raise ValueError("PLR 임베딩은 수치 열이 있어야 쓸 수 있다.")
            self.num_embeddings = _PLREmbeddings(
                n_numeric,
                int(num_embeddings["n_frequencies"]),
                float(num_embeddings["frequency_scale"]),
                int(num_embeddings["d_embedding"]),
            )
            d_in = n_numeric * int(num_embeddings["d_embedding"]) + d_onehot
        self.linear = nn.Linear(d_in, d_main)
        self.key = nn.Linear(d_main, d_main)
        self.label_encoder = nn.Embedding(2, d_main)
        self.value = nn.Sequential(
            nn.Linear(d_main, d_block),
            nn.ReLU(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.context_dropout = nn.Dropout(context_dropout)
        self.predictor = _ResidualBlock(
            d_main, d_block, dropout0, dropout1, prenorm=True
        )
        self.head = nn.Sequential(nn.LayerNorm(d_main), nn.ReLU(), nn.Linear(d_main, 1))
        nn.init.uniform_(self.label_encoder.weight, -1.0, 1.0)

    def encode(self, x_numeric: Tensor, x_onehot: Tensor) -> tuple[Tensor, Tensor]:
        parts: list[Tensor] = []
        if x_numeric.shape[1]:
            parts.append(
                x_numeric
                if self.num_embeddings is None
                else self.num_embeddings(x_numeric).flatten(1)
            )
        if x_onehot.shape[1]:
            parts.append(x_onehot)
        encoded = self.linear(torch.cat(parts, dim=1))
        return encoded, self.key(encoded)

    def forward_with_context(
        self,
        query_numeric: Tensor,
        query_onehot: Tensor,
        context_numeric: Tensor,
        context_onehot: Tensor,
        context_y: Tensor,
    ) -> Tensor:
        encoded, query_key = self.encode(query_numeric, query_onehot)
        batch_size, context_size = context_y.shape
        context_key = self.encode(
            context_numeric.reshape(batch_size * context_size, context_numeric.shape[2]),
            context_onehot.reshape(batch_size * context_size, context_onehot.shape[2]),
        )[1].reshape(batch_size, context_size, -1)
        similarities = -(query_key[:, None, :] - context_key).square().sum(dim=-1)
        probabilities = self.context_dropout(F.softmax(similarities, dim=-1))
        values = self.label_encoder(context_y) + self.value(
            query_key[:, None, :] - context_key
        )
        encoded = encoded + (probabilities[:, None, :] @ values).squeeze(1)
        return self.head(self.predictor(encoded)).squeeze(-1)


class _SearchResult(NamedTuple):
    local_ids: Tensor
    global_ids: np.ndarray


@dataclass
class _Environment:
    search_backend: str
    faiss_version: str | None
    torch_version: str
    torch_cuda_version: str | None


class TabRFold:
    """fold 하나의 전처리, 후보 저장소, 조기 종료 학습, 예측과 중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._context_size = int(params.pop("context_size", 96))
        self._patience = int(params.pop("patience", 16))
        self._max_epochs = int(params.pop("max_epochs", 128))
        self._batch_size = int(params.pop("batch_size", 1024))
        self._eval_batch_size = int(params.pop("eval_batch_size", 8192))
        self._candidate_encoding_batch_size = int(
            params.pop("candidate_encoding_batch_size", 65536)
        )
        self._d_main = int(params.pop("d_main", 265))
        self._d_multiplier = float(params.pop("d_multiplier", 2.0))
        self._context_dropout = float(
            params.pop("context_dropout", 0.38920071545944357)
        )
        self._dropout0 = float(params.pop("dropout0", 0.38852797479169876))
        self._dropout1 = float(params.pop("dropout1", 0.0))
        self._lr = float(params.pop("lr", 0.0003121273641315169))
        self._weight_decay = float(params.pop("weight_decay", 1.2260352006404615e-06))
        self._num_embeddings = params.pop("num_embeddings", None)
        self._perm_sample = int(params.pop("perm_sample", 8192))
        self._perm_repeats = int(params.pop("perm_repeats", 1))
        self._diagnostic_context_sample = int(
            params.pop("diagnostic_context_sample", 256)
        )
        self._device_name = params.pop("device", None)
        if params:
            raise ValueError(f"tabr가 모르는 params: {sorted(params)}")
        if self._num_embeddings is not None:
            embedding_keys = set(self._num_embeddings)
            expected = {"n_frequencies", "frequency_scale", "d_embedding", "lite"}
            if embedding_keys != expected:
                raise ValueError(
                    f"num_embeddings 키는 {sorted(expected)}여야 한다: "
                    f"{sorted(embedding_keys)}"
                )
            if self._num_embeddings["lite"] is not True:
                raise ValueError("num_embeddings는 공식 lite 판만 지원한다.")
        if self._patience < 0:
            raise ValueError("patience는 0 이상이어야 한다.")
        if self._max_epochs < 1:
            raise ValueError("max_epochs는 1 이상이어야 한다.")
        if min(self._batch_size, self._eval_batch_size, self._perm_sample) < 1:
            raise ValueError("batch와 permutation 표본 크기는 1 이상이어야 한다.")
        self._seed = seed
        self._device = torch.device(
            self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self._device.type == "cpu":
            torch.set_num_threads(1)
        self._encoder = _SplitEncodedTable(seed)
        self._model: _TabR | None = None
        self._train_numeric: Tensor | None = None
        self._train_onehot: Tensor | None = None
        self._train_y: Tensor | None = None
        self._train_global_ids = np.array([], dtype="int64")
        self._validation_global_ids = np.array([], dtype="int64")
        self._validation: tuple[pd.DataFrame, np.ndarray] | None = None
        self._epoch_seconds: list[float] = []
        self._epoch_aucs: list[float] = []
        self._best_epoch: int | None = None
        self._stop_reason: str | None = None
        self._self_samples: list[dict[str, object]] = []
        self._self_excluded = True
        self._environment: _Environment | None = None

    def _encode_candidate_keys(self) -> Tensor:
        assert (
            self._model is not None
            and self._train_numeric is not None
            and self._train_onehot is not None
        )
        self._model.eval()
        keys: list[Tensor] = []
        with torch.no_grad():
            for start in range(
                0, len(self._train_numeric), self._candidate_encoding_batch_size
            ):
                stop = start + self._candidate_encoding_batch_size
                keys.append(
                    self._model.encode(
                        self._train_numeric[start:stop].to(self._device),
                        self._train_onehot[start:stop].to(self._device),
                    )[1]
                )
        return torch.cat(keys).contiguous()

    def _search(self, query_keys: Tensor, candidate_keys: Tensor, k: int) -> Tensor:
        if self._device.type == "cuda":
            try:
                import faiss
                import faiss.contrib.torch_utils
            except ImportError as exc:
                raise RuntimeError(
                    "CUDA TabR는 고정된 faiss-gpu-cu12 환경이 필요하다."
                ) from exc
            if not hasattr(self, "_faiss_resources"):
                self._faiss_resources = faiss.StandardGpuResources()
                self._faiss_index = faiss.GpuIndexFlatL2(
                    self._faiss_resources, candidate_keys.shape[1]
                )
            self._faiss_index.reset()
            self._faiss_index.add(candidate_keys)
            _, ids = self._faiss_index.search(query_keys.contiguous(), k)
            self._environment = _Environment(
                search_backend="faiss.GpuIndexFlatL2",
                faiss_version=getattr(faiss, "__version__", None),
                torch_version=torch.__version__,
                torch_cuda_version=torch.version.cuda,
            )
            return ids.long()
        distances = torch.cdist(query_keys.float(), candidate_keys.float())
        self._environment = _Environment(
            search_backend="torch.cdist-test-only",
            faiss_version=None,
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
        )
        return distances.topk(k, largest=False).indices

    def _contexts(
        self,
        query_numeric: Tensor,
        query_onehot: Tensor,
        candidate_keys: Tensor,
        *,
        query_local_ids: Tensor | None,
    ) -> _SearchResult:
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            query_keys = self._model.encode(query_numeric, query_onehot)[1]
            extra = 1 if query_local_ids is not None else 0
            local_ids = self._search(
                query_keys, candidate_keys, self._context_size + extra
            )
            if query_local_ids is not None:
                is_self = local_ids == query_local_ids[:, None]
                positions = torch.arange(
                    local_ids.shape[1], device=self._device
                ).expand_as(local_ids)
                order = torch.where(
                    is_self,
                    torch.full_like(positions, local_ids.shape[1]),
                    positions,
                ).argsort(dim=1)
                local_ids = local_ids.gather(1, order)[:, : self._context_size]
                self._self_excluded = self._self_excluded and not bool(
                    (local_ids == query_local_ids[:, None]).any().item()
                )
        global_ids = self._train_global_ids[local_ids.detach().cpu().numpy()]
        return _SearchResult(local_ids=local_ids, global_ids=global_ids)

    def _context_rows(self, local_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        assert (
            self._train_numeric is not None
            and self._train_onehot is not None
            and self._train_y is not None
        )
        cpu_ids = local_ids.cpu()
        return (
            self._train_numeric[cpu_ids].to(self._device),
            self._train_onehot[cpu_ids].to(self._device),
            self._train_y[cpu_ids].to(self._device),
        )

    def _predict_tensor(
        self, query_numeric: Tensor, query_onehot: Tensor
    ) -> np.ndarray:
        assert self._model is not None
        candidate_keys = self._encode_candidate_keys()
        self._model.eval()
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(query_numeric), self._eval_batch_size):
                stop = start + self._eval_batch_size
                numeric = query_numeric[start:stop].to(self._device)
                onehot = query_onehot[start:stop].to(self._device)
                found = self._contexts(
                    numeric, onehot, candidate_keys, query_local_ids=None
                )
                context_numeric, context_onehot, context_y = self._context_rows(
                    found.local_ids
                )
                logits = self._model.forward_with_context(
                    numeric, onehot, context_numeric, context_onehot, context_y
                )
                predictions.append(
                    torch.sigmoid(logits).cpu().numpy().astype("float64")
                )
        return np.concatenate(predictions)

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        if not X_tr.index.is_unique or not X_va.index.is_unique:
            raise ValueError(
                "TabR는 행 경계 검증을 위해 고유한 DataFrame index가 필요하다."
            )
        overlap = X_tr.index.intersection(X_va.index)
        if len(overlap):
            raise ValueError("TabR 학습 fold와 검증 fold의 행 index가 겹친다.")
        if len(X_tr) <= self._context_size:
            raise ValueError("후보 저장소 행 수가 context_size보다 커야 한다.")

        torch.manual_seed(self._seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)
            torch.backends.cuda.matmul.allow_tf32 = True
        self._encoder.fit(X_tr)
        self._train_numeric, self._train_onehot = self._encoder.transform(X_tr)
        self._train_numeric = self._train_numeric.contiguous()
        self._train_onehot = self._train_onehot.contiguous()
        validation_numeric, validation_onehot = self._encoder.transform(X_va)
        self._train_y = torch.from_numpy(y_tr.to_numpy(dtype="int64", copy=True))
        self._train_global_ids = X_tr.index.to_numpy(copy=True)
        self._validation_global_ids = X_va.index.to_numpy(copy=True)
        self._validation = (X_va.copy(), y_va.to_numpy(dtype="float64"))

        self._model = _TabR(
            self._encoder.numeric_dim,
            self._encoder.onehot_dim,
            self._d_main,
            self._d_multiplier,
            self._context_dropout,
            self._dropout0,
            self._dropout1,
            self._num_embeddings,
        ).to(self._device)
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self._seed)
        best_auc = -math.inf
        best_state: dict[str, Tensor] | None = None
        bad_epochs = 0
        y_va_np = y_va.to_numpy(dtype="float64")

        for epoch in range(self._max_epochs):
            epoch_started = time.monotonic()
            permutation = torch.randperm(
                len(self._train_numeric), generator=generator
            )
            for start in range(0, len(permutation), self._batch_size):
                local_cpu = permutation[start : start + self._batch_size]
                local = local_cpu.to(self._device)
                query_numeric = self._train_numeric[local_cpu].to(self._device)
                query_onehot = self._train_onehot[local_cpu].to(self._device)
                query_y = self._train_y[local_cpu].to(self._device).float()
                candidate_keys = self._encode_candidate_keys()
                found = self._contexts(
                    query_numeric, query_onehot, candidate_keys, query_local_ids=local
                )
                if len(self._self_samples) < 8:
                    for row, contexts in zip(
                        local_cpu[: 8 - len(self._self_samples)].tolist(),
                        found.global_ids[: 8 - len(self._self_samples)],
                    ):
                        self._self_samples.append(
                            {
                                "query_row_id": self._json_scalar(
                                    self._train_global_ids[row]
                                ),
                                "context_row_ids": [
                                    self._json_scalar(value) for value in contexts[:8]
                                ],
                            }
                        )
                context_numeric, context_onehot, context_y = self._context_rows(
                    found.local_ids
                )
                self._model.train()
                logits = self._model.forward_with_context(
                    query_numeric, query_onehot, context_numeric, context_onehot,
                    context_y,
                )
                loss = loss_fn(logits, query_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            val_pred = self._predict_tensor(validation_numeric, validation_onehot)
            auc = float(roc_auc_score(y_va_np, val_pred))
            self._epoch_aucs.append(auc)
            self._epoch_seconds.append(float(time.monotonic() - epoch_started))
            if auc > best_auc:
                best_auc = auc
                best_state = copy.deepcopy(self._model.state_dict())
                self._best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(
                f"[tabr] epoch={epoch + 1}/{self._max_epochs} "
                f"valAUC={auc:.6f} best={best_auc:.6f} "
                f"bad={bad_epochs}/{self._patience} "
                f"seconds={self._epoch_seconds[-1]:.1f}",
                flush=True,
            )
            if not np.isfinite(auc):
                self._stop_reason = "non_finite_validation_auc"
                break
            if bad_epochs > self._patience:
                self._stop_reason = "early_stopping"
                break
        if self._stop_reason is None:
            self._stop_reason = "max_epochs"
        if best_state is None:
            raise ValueError("모든 epoch의 검증 AUC가 유한하지 않아 최적 상태가 없다.")
        self._model.load_state_dict(best_state)
        return self._predict_tensor(validation_numeric, validation_onehot)

    @staticmethod
    def _json_scalar(value: object) -> object:
        return value.item() if isinstance(value, np.generic) else value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        numeric, onehot = self._encoder.transform(X)
        return self._predict_tensor(numeric.contiguous(), onehot.contiguous())

    def importance(self) -> pd.DataFrame:
        assert self._validation is not None
        X_va, y_va = self._validation
        if len(X_va) > self._perm_sample:
            keep = np.random.default_rng(self._seed).choice(
                len(X_va), size=self._perm_sample, replace=False
            )
            keep.sort()
            X_va = X_va.iloc[keep].reset_index(drop=True)
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
                permuted[column] = (
                    X_va[column]
                    .take(rng.permutation(len(X_va)))
                    .set_axis(permuted.index)
                )
                drops.append(base - roc_auc_score(y_va, self.predict(permuted)))
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._encoder.columns, "gain": gains})

    def entry_diagnostics(self):
        from . import model

        assert self._train_y is not None
        environment = self._environment or _Environment(
            search_backend="not-observed",
            faiss_version=None,
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
        )
        train_only = not bool(
            np.intersect1d(self._train_global_ids, self._validation_global_ids).size
        )
        return model.AdapterDiagnostics(
            assertions={
                model.ASSERT_CANDIDATE_STORE_TRAIN_ONLY: train_only,
                model.ASSERT_VALIDATION_LABELS_EXCLUDED: len(self._train_y)
                == len(self._train_global_ids),
                model.ASSERT_SELF_ROWS_EXCLUDED: self._self_excluded,
            },
            observations={
                "official_commit": OFFICIAL_COMMIT,
                "candidate_rows": len(self._train_global_ids),
                "validation_rows": len(self._validation_global_ids),
                "context_size": self._context_size,
                "patience": self._patience,
                "max_epochs": self._max_epochs,
                "num_embeddings": self._num_embeddings,
                "epoch_seconds": self._epoch_seconds,
                "epoch_validation_aucs": self._epoch_aucs,
                "best_epoch": self._best_epoch,
                "stop_reason": self._stop_reason,
                "self_exclusion_samples": self._self_samples,
                "search_backend": environment.search_backend,
                "faiss_version": environment.faiss_version,
                "torch_version": environment.torch_version,
                "torch_cuda_version": environment.torch_cuda_version,
            },
        )
