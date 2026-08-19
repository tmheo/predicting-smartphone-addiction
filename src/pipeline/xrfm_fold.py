"""xRFM 재귀 특성 커널 머신의 fold 실행 경계. (#198)

공식 ``xrfm`` 패키지(MIT, https://github.com/dmbeaglehole/xRFM)의 xRFM을
저장소 fold 규율에 맞게 감싼다.

- 전처리(결측 중앙값 대치, 결측 지시열, 표준화, 학습 fold 어휘 one-hot)는
  outer 학습 fold에서만 맞춘다.
- xRFM이 요구하는 튜닝 검증 자료는 outer 학습 fold 안에서 seed 고정
  층화 분할로 떼어낸다. outer 검증 fold의 행과 라벨은 학습에 쓰지 않는다.
- 중요도는 검증 fold의 seed 고정 표본에 대한 permutation AUC 하락이다.
"""

from __future__ import annotations

import importlib.metadata
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from .model import (
    ASSERT_CANDIDATE_STORE_TRAIN_ONLY,
    ASSERT_VALIDATION_LABELS_EXCLUDED,
    AdapterDiagnostics,
)

PACKAGE_VERSION = "0.4.5"

_KERNELS = {
    "l2",
    "laplace",
    "l2_high_dim",
    "l2_light",
    "l1",
    "product_laplace",
    "l1_power",
    "sum_power_laplace",
}


class _FoldKernelEncoder:
    """학습 fold에서만 대치 통계, 표준화 통계와 범주 어휘를 맞춘다."""

    def __init__(self) -> None:
        self.columns: list[str] | None = None
        self._numeric: dict[str, tuple[float, float, float]] = {}
        self._vocabulary: dict[str, tuple[object, ...]] = {}
        self.encoded_names: list[str] = []
        self.fit_rows: int | None = None

    @staticmethod
    def _object_values(series: pd.Series) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            return series.astype(object)
        return series

    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series) and not isinstance(
            series.dtype, pd.CategoricalDtype
        )

    def fit(self, X: pd.DataFrame) -> None:
        if not len(X):
            raise ValueError("xRFM 전처리에는 학습 행이 필요하다.")
        self.columns = list(X.columns)
        if not self.columns:
            raise ValueError("xRFM 입력 열이 비어 있다.")
        self._numeric = {}
        self._vocabulary = {}
        self.encoded_names = []
        for name in self.columns:
            values = X[name]
            if self._is_numeric(values):
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(
                    dtype="float64"
                )
                finite = numeric[np.isfinite(numeric)]
                median = float(np.median(finite)) if len(finite) else 0.0
                imputed = np.where(np.isfinite(numeric), numeric, median)
                mean = float(imputed.mean())
                std = float(imputed.std())
                if not np.isfinite(std) or std <= 0.0:
                    std = 1.0
                self._numeric[name] = (median, mean, std)
                self.encoded_names.append(name)
                self.encoded_names.append(f"{name}__missing")
                continue
            objects = self._object_values(values)
            vocabulary = tuple(sorted(pd.unique(objects.dropna()).tolist(), key=repr))
            self._vocabulary[name] = vocabulary
            self.encoded_names.extend(f"{name}=={value!r}" for value in vocabulary)
            self.encoded_names.append(f"{name}__unknown")
            self.encoded_names.append(f"{name}__missing")
        self.fit_rows = len(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns is None:
            raise RuntimeError("xRFM 전처리를 먼저 학습해야 한다.")
        if list(X.columns) != self.columns:
            raise AssertionError("xRFM 입력 열이 학습 때와 다르다.")
        output = np.zeros((len(X), len(self.encoded_names)), dtype="float32")
        offset = 0
        for name in self.columns:
            if name in self._numeric:
                median, mean, std = self._numeric[name]
                numeric = pd.to_numeric(X[name], errors="coerce").to_numpy(
                    dtype="float64"
                )
                missing = ~np.isfinite(numeric)
                imputed = np.where(missing, median, numeric)
                output[:, offset] = ((imputed - mean) / std).astype("float32")
                output[:, offset + 1] = missing.astype("float32")
                offset += 2
                continue
            vocabulary = self._vocabulary[name]
            objects = self._object_values(X[name])
            mapping = {value: index for index, value in enumerate(vocabulary)}
            ids = objects.map(mapping)
            missing = objects.isna().to_numpy()
            unknown = ids.isna().to_numpy() & ~missing
            known = ids.fillna(0).to_numpy(dtype="int64")
            rows = np.arange(len(X))
            keep = ~missing & ~unknown
            output[rows[keep], offset + known[keep]] = 1.0
            output[rows[unknown], offset + len(vocabulary)] = 1.0
            output[rows[missing], offset + len(vocabulary) + 1] = 1.0
            offset += len(vocabulary) + 2
        if not np.isfinite(output).all():
            raise RuntimeError("xRFM 전처리 결과에 유한하지 않은 값이 있다.")
        return output


class XRFMFold:
    """fold 하나의 전처리, 내부 튜닝 분할, 학습과 중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._device = str(params.pop("device", "cuda"))
        self._max_leaf_size = int(params.pop("max_leaf_size", 60000))
        self._n_trees = int(params.pop("n_trees", 1))
        self._inner_val_frac = float(params.pop("inner_val_frac", 0.1))
        self._eval_batch_size = int(params.pop("eval_batch_size", 65536))
        self._kernel = str(params.pop("kernel", "l2_high_dim"))
        self._exponent = float(params.pop("exponent", 1.0))
        self._bandwidth = float(params.pop("bandwidth", 10.0))
        self._diag = bool(params.pop("diag", False))
        self._bandwidth_mode = str(params.pop("bandwidth_mode", "constant"))
        self._reg = float(params.pop("reg", 1e-3))
        self._iters = int(params.pop("iters", 5))
        self._perm_sample = int(params.pop("perm_sample", 1000))
        self._perm_repeats = int(params.pop("perm_repeats", 1))
        self._verbose = bool(params.pop("verbose", True))
        self._package_version = str(params.pop("package_version", PACKAGE_VERSION))
        if params:
            raise ValueError(f"xrfm이 모르는 params: {sorted(params)}")

        if self._device not in {"cpu", "cuda"}:
            raise ValueError("xrfm device는 'cpu' 또는 'cuda'여야 한다.")
        if self._max_leaf_size < 2:
            raise ValueError("xrfm max_leaf_size는 2 이상이어야 한다.")
        if self._n_trees < 1:
            raise ValueError("xrfm n_trees는 1 이상이어야 한다.")
        if not 0.0 < self._inner_val_frac < 0.5:
            raise ValueError("xrfm inner_val_frac은 (0, 0.5) 안이어야 한다.")
        if self._eval_batch_size < 1:
            raise ValueError("xrfm eval_batch_size는 1 이상이어야 한다.")
        if self._kernel not in _KERNELS:
            raise ValueError(f"xrfm kernel은 {sorted(_KERNELS)} 중 하나여야 한다.")
        if self._reg <= 0.0:
            raise ValueError("xrfm reg는 0보다 커야 한다.")
        if self._iters < 0:
            raise ValueError("xrfm iters는 0 이상이어야 한다.")
        if self._perm_sample < 1 or self._perm_repeats != 1:
            raise ValueError("xrfm importance는 표본 1 이상과 반복 1회만 지원한다.")
        if self._package_version != PACKAGE_VERSION:
            raise ValueError(f"xrfm 패키지는 {PACKAGE_VERSION}만 허용한다.")

        self._seed = seed
        self._encoder = _FoldKernelEncoder()
        self._model = None
        self._columns: list[str] | None = None
        self._X_va: pd.DataFrame | None = None
        self._y_va: np.ndarray | None = None
        self._va_pred: np.ndarray | None = None
        self._train_index: pd.Index | None = None
        self._validation_index: pd.Index | None = None
        self._inner_train_rows: int | None = None
        self._inner_val_rows: int | None = None
        self._effective_max_leaf_size: int | None = None
        self._leaf_count: int | None = None
        self._timings: dict[str, float] = {}

    def _verify_runtime(self) -> None:
        installed = importlib.metadata.version("xrfm")
        if installed != self._package_version:
            raise RuntimeError(
                f"xrfm 설치 판본이 다르다: expected={self._package_version} actual={installed}"
            )
        if self._device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("xrfm device=cuda인데 CUDA를 사용할 수 없다.")

    def _new_model(self):
        from xrfm import xRFM

        rfm_params = {
            "model": {
                "kernel": self._kernel,
                "exponent": self._exponent,
                "bandwidth": self._bandwidth,
                "diag": self._diag,
                "bandwidth_mode": self._bandwidth_mode,
            },
            "fit": {
                "get_agop_best_model": True,
                "return_best_params": True,
                "reg": self._reg,
                "iters": self._iters,
                "early_stop_rfm": False,
            },
        }
        return xRFM(
            rfm_params=rfm_params,
            max_leaf_size=self._max_leaf_size,
            device=self._device,
            n_trees=self._n_trees,
            tuning_metric="auc",
            random_state=self._seed,
            verbose=self._verbose,
        )

    def _predict_proba_batched(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        outputs: list[np.ndarray] = []
        for start in range(0, len(X), self._eval_batch_size):
            chunk = X.iloc[start : start + self._eval_batch_size]
            encoded = self._encoder.transform(chunk)
            proba = np.asarray(self._model.predict_proba(encoded))
            if proba.ndim != 2 or proba.shape != (len(chunk), 2):
                raise RuntimeError(f"xrfm predict_proba 형태가 이상하다: {proba.shape}")
            outputs.append(proba[:, 1].astype("float64"))
        prediction = np.concatenate(outputs) if outputs else np.empty(0, dtype="float64")
        if not np.isfinite(prediction).all():
            raise RuntimeError("xrfm 예측에 유한하지 않은 값이 있다.")
        return prediction

    def _count_leaves(self) -> int | None:
        assert self._model is not None
        trees = getattr(self._model, "trees", None)
        collect = getattr(self._model, "_collect_leaf_nodes", None)
        if not trees or collect is None:
            return None
        try:
            return int(sum(len(collect(tree)) for tree in trees))
        except Exception:
            return None

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        if not X_tr.index.intersection(X_va.index).empty:
            raise ValueError("xRFM outer 학습 fold와 검증 fold의 index가 겹친다.")
        self._columns = list(X_tr.columns)
        if list(X_va.columns) != self._columns:
            raise ValueError("xRFM 학습과 검증 컬럼이 다르다.")
        labels = np.unique(y_tr.to_numpy())
        if not np.array_equal(labels, np.array([0, 1])):
            raise ValueError("xRFM은 0/1 이진 라벨만 지원한다.")
        self._train_index = X_tr.index.copy()
        self._validation_index = X_va.index.copy()
        self._X_va = X_va.copy()
        self._y_va = y_va.to_numpy(dtype="int64")

        started = time.monotonic()
        self._verify_runtime()
        self._timings["runtime_verification"] = time.monotonic() - started

        started = time.monotonic()
        self._encoder.fit(X_tr)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self._inner_val_frac, random_state=self._seed
        )
        inner_train_pos, inner_val_pos = next(
            splitter.split(np.zeros(len(X_tr)), y_tr.to_numpy())
        )
        inner_train_pos.sort()
        inner_val_pos.sort()
        Z_tr = self._encoder.transform(X_tr.iloc[inner_train_pos])
        Z_iv = self._encoder.transform(X_tr.iloc[inner_val_pos])
        y_tr_values = y_tr.to_numpy(dtype="int64")
        self._inner_train_rows = len(inner_train_pos)
        self._inner_val_rows = len(inner_val_pos)
        self._timings["preprocess_and_inner_split"] = time.monotonic() - started

        started = time.monotonic()
        self._model = self._new_model()
        self._model.fit(
            Z_tr,
            y_tr_values[inner_train_pos],
            Z_iv,
            y_tr_values[inner_val_pos],
        )
        self._effective_max_leaf_size = int(getattr(self._model, "max_leaf_size", 0))
        self._leaf_count = self._count_leaves()
        self._timings["fit"] = time.monotonic() - started

        started = time.monotonic()
        self._va_pred = self._predict_proba_batched(self._X_va)
        self._timings["validation_predict"] = time.monotonic() - started
        if self._va_pred.shape != (len(self._X_va),):
            raise ValueError("xRFM 검증 예측 행 수가 입력과 다르다.")
        print(
            f"[xrfm] fold valAUC={roc_auc_score(self._y_va, self._va_pred):.5f} "
            f"kernel={self._kernel} leaves={self._leaf_count} "
            f"max_leaf_size={self._effective_max_leaf_size}",
            flush=True,
        )
        return self._va_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if list(X.columns) != self._columns:
            raise ValueError("xRFM 예측 컬럼이 학습 때와 다르다.")
        return self._predict_proba_batched(X)

    def importance(self) -> pd.DataFrame:
        """검증 fold의 seed 고정 표본 permutation importance."""
        assert (
            self._X_va is not None
            and self._y_va is not None
            and self._va_pred is not None
        )
        if len(self._X_va) > self._perm_sample:
            keep = np.random.default_rng(self._seed).choice(
                len(self._X_va), size=self._perm_sample, replace=False
            )
            keep.sort()
        else:
            keep = np.arange(len(self._X_va))
        X_va = self._X_va.iloc[keep].copy()
        y_va = self._y_va[keep]
        base_pred = self._predict_proba_batched(X_va)
        base = roc_auc_score(y_va, base_pred)

        gains = []
        for index, column in enumerate(self._columns):
            rng = np.random.default_rng(self._seed * 10007 + index * 101)
            X_perm = X_va.copy()
            X_perm[column] = X_va[column].take(rng.permutation(len(X_va))).to_numpy()
            pred = self._predict_proba_batched(X_perm)
            gains.append(float(base - roc_auc_score(y_va, pred)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})

    def entry_diagnostics(self) -> AdapterDiagnostics:
        assert (
            self._X_va is not None
            and self._y_va is not None
            and self._va_pred is not None
        )
        missing = self._X_va.isna().any(axis=1).to_numpy()

        def subgroup_auc(mask: np.ndarray) -> float | None:
            labels = self._y_va[mask]
            if len(labels) == 0 or np.unique(labels).size < 2:
                return None
            return float(roc_auc_score(labels, self._va_pred[mask]))

        return AdapterDiagnostics(
            assertions={
                ASSERT_CANDIDATE_STORE_TRAIN_ONLY: self._train_index.intersection(
                    self._validation_index
                ).empty,
                ASSERT_VALIDATION_LABELS_EXCLUDED: True,
            },
            observations={
                "training_rows": len(self._train_index),
                "validation_rows": len(self._validation_index),
                "inner_train_rows": self._inner_train_rows,
                "inner_val_rows": self._inner_val_rows,
                "encoded_feature_count": len(self._encoder.encoded_names),
                "missing_rows": int(missing.sum()),
                "complete_rows": int((~missing).sum()),
                "missing_auc": subgroup_auc(missing),
                "complete_auc": subgroup_auc(~missing),
                "kernel": self._kernel,
                "exponent": self._exponent,
                "bandwidth": self._bandwidth,
                "bandwidth_mode": self._bandwidth_mode,
                "diag": self._diag,
                "reg": self._reg,
                "iters": self._iters,
                "n_trees": self._n_trees,
                "requested_max_leaf_size": self._max_leaf_size,
                "effective_max_leaf_size": self._effective_max_leaf_size,
                "leaf_count": self._leaf_count,
                "inner_val_frac": self._inner_val_frac,
                "device": self._device,
                "package_version": self._package_version,
                "timing_seconds": self._timings,
            },
        )
