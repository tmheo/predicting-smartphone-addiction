"""TabPFN-3 fold 구현. (#102)

fold별 학습 문맥을 ``fit_with_cache``로 올리고, 검증·테스트는 작은 청크로
예측한다. 범주형 열은 TabPFN에 인덱스를 명시하고 결측을 보존한 정수 코드로
바꾼다. 스모크 게이트와 같은 ``tabpfn`` 8.3.0, ``ModelVersion.V3`` 경로다.

The TABPFN-3 Model is licensed by Prior Labs GmbH under the
TABPFN-3 Non-Commercial License.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _is_cuda_oom(exc: Exception) -> bool:
    """정상 경로에서 torch를 먼저 적재하지 않고 CUDA OOM만 식별한다."""
    import torch

    return isinstance(exc, torch.cuda.OutOfMemoryError)


def _clear_cuda_cache() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


class TabPFN3Fold:
    """fold 하나의 전처리·캐시·청크 예측·중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._n_estimators = int(params.pop("n_estimators", 8))
        self._device = str(params.pop("device", "auto"))
        self._fit_mode = str(params.pop("fit_mode", "fit_with_cache"))
        self._memory_saving_mode: bool | str = params.pop("memory_saving_mode", "auto")
        self._chunk_rows = int(params.pop("chunk_rows", 1000))
        self._perm_repeats = int(params.pop("perm_repeats", 1))
        self._perm_sample = int(params.pop("perm_sample", 1000))
        if params:
            raise ValueError(f"tabpfn3가 모르는 params: {sorted(params)}")
        if self._n_estimators < 1:
            raise ValueError(f"n_estimators는 1 이상이어야 한다: {self._n_estimators}")
        if self._fit_mode != "fit_with_cache":
            raise ValueError(
                "tabpfn3 정식 경로는 fit_mode='fit_with_cache'만 허용한다."
            )
        if self._memory_saving_mode not in ("auto", True, False):
            raise ValueError("memory_saving_mode는 'auto' 또는 bool이어야 한다.")
        if self._chunk_rows < 1:
            raise ValueError(f"chunk_rows는 1 이상이어야 한다: {self._chunk_rows}")
        if self._perm_repeats < 1:
            raise ValueError(f"perm_repeats는 1 이상이어야 한다: {self._perm_repeats}")
        if self._perm_sample < 1:
            raise ValueError(f"perm_sample은 1 이상이어야 한다: {self._perm_sample}")

        self._seed = seed
        self._columns: list[str] | None = None
        self._cat_indices: list[int] = []
        self._cat_levels: dict[str, list] = {}
        self._model = None
        self._X_tr: pd.DataFrame | None = None
        self._y_tr: pd.Series | None = None
        self._X_va: pd.DataFrame | None = None
        self._y_va: np.ndarray | None = None
        self._va_pred: np.ndarray | None = None

    def _prepare(self, X: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        if fit:
            self._columns = list(X.columns)
            self._cat_indices = [
                i
                for i, col in enumerate(self._columns)
                if isinstance(X[col].dtype, pd.CategoricalDtype)
                or X[col].dtype == object
            ]
            self._cat_levels = {}
            for i in self._cat_indices:
                col = self._columns[i]
                values = X[col]
                self._cat_levels[col] = (
                    list(values.cat.categories)
                    if isinstance(values.dtype, pd.CategoricalDtype)
                    else list(pd.unique(values.dropna()))
                )
        assert list(X.columns) == self._columns, "입력 컬럼이 학습 때와 다르다."

        out = X.copy()
        for i, col in enumerate(self._columns):
            if i in self._cat_indices:
                codes = pd.Categorical(out[col], categories=self._cat_levels[col]).codes
                out[col] = pd.Series(codes, index=out.index, dtype="float64").where(
                    codes >= 0, np.nan
                )
            else:
                out[col] = out[col].astype("float64")
        return out

    def _new_classifier(self):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3,
            n_estimators=self._n_estimators,
            device=self._device,
            random_state=self._seed,
            fit_mode=self._fit_mode,
            categorical_features_indices=self._cat_indices or None,
            memory_saving_mode=self._memory_saving_mode,
        )

    def _fit_model(self) -> None:
        self._model = self._new_classifier()
        try:
            self._model.fit(self._X_tr, self._y_tr)
        except Exception as exc:
            if not _is_cuda_oom(exc) or self._memory_saving_mode is True:
                raise
            self._memory_saving_mode = True
            self._model = None
            _clear_cuda_cache()
            self._model = self._new_classifier()
            self._model.fit(self._X_tr, self._y_tr)

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        self._X_tr = self._prepare(X_tr, fit=True)
        self._y_tr = y_tr.copy()
        self._X_va = self._prepare(X_va)
        self._y_va = y_va.to_numpy(dtype="float64")
        self._fit_model()
        self._va_pred = self._predict_transformed(self._X_va)
        print(
            f"[tabpfn3] fold valAUC={roc_auc_score(self._y_va, self._va_pred):.5f} "
            f"memory_saving_mode={self._memory_saving_mode}",
            flush=True,
        )
        return self._va_pred

    def fit_full(self, X: pd.DataFrame, y: pd.Series) -> None:
        """검증 자료 없이 전체 훈련 자료를 TabPFN 문맥으로 고정한다."""
        self._X_tr = self._prepare(X, fit=True)
        self._y_tr = y.copy()
        self._fit_model()

    def _predict_once(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.empty(len(X), dtype="float64")
        for start in range(0, len(X), self._chunk_rows):
            stop = min(start + self._chunk_rows, len(X))
            pred[start:stop] = self._model.predict_proba(X.iloc[start:stop])[:, 1]
        return pred

    def _predict_transformed(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self._predict_once(X)
        except Exception as exc:
            if not _is_cuda_oom(exc) or self._memory_saving_mode is True:
                raise
            self._memory_saving_mode = True
            self._model = None
            _clear_cuda_cache()
            self._fit_model()
            return self._predict_once(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_transformed(self._prepare(X))

    def importance(self) -> pd.DataFrame:
        """검증 fold의 결정적 부분표본 permutation importance를 돌려준다."""
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
        X_va = self._X_va.iloc[keep].reset_index(drop=True)
        y_va = self._y_va[keep]
        base = roc_auc_score(y_va, self._va_pred[keep])

        gains = []
        for j, col in enumerate(self._columns):
            drops = []
            for repeat in range(self._perm_repeats):
                rng = np.random.default_rng(self._seed * 10007 + j * 101 + repeat)
                X_perm = X_va.copy()
                X_perm[col] = X_va[col].take(rng.permutation(len(X_va))).to_numpy()
                pred = self._predict_transformed(X_perm)
                drops.append(base - roc_auc_score(y_va, pred))
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})
