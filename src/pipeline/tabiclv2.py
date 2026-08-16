"""TabICLv2 공식 추론기와 fold 실행 경계. (#143)

공식 ``tabicl`` 전처리와 추론기를 그대로 사용한다.
코드 판본, 분류 가중치 이름과 내용 해시를 실행 전에 검증하고, 검증 fold의
결정적 1,000행에서 permutation importance를 한 번 측정한다.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .model import (
    ASSERT_CANDIDATE_STORE_TRAIN_ONLY,
    ASSERT_VALIDATION_LABELS_EXCLUDED,
    AdapterDiagnostics,
)

PACKAGE_VERSION = "2.1.1"
SOURCE_REVISION = "59a957cd644be4e1f2e1582757203ecbd630afa2"
CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
CHECKPOINT_REPOSITORY = "jingang/TabICL"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_cuda_oom(exc: Exception) -> bool:
    """정상 경로에서 torch를 먼저 적재하지 않고 CUDA 메모리 부족만 식별한다."""
    import torch

    return isinstance(exc, torch.cuda.OutOfMemoryError)


def _clear_cuda_cache() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


class TabICLv2Fold:
    """fold 하나의 공식 전처리, 문맥 추론과 중요도 상태."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._n_estimators = int(params.pop("n_estimators", 8))
        self._batch_size = int(params.pop("batch_size", 8))
        self._kv_cache = params.pop("kv_cache", "repr")
        self._device = str(params.pop("device", "cuda"))
        self._offload_mode = str(params.pop("offload_mode", "auto"))
        self._disk_offload_dir = Path(params.pop("disk_offload_dir"))
        self._checkpoint_path = Path(params.pop("checkpoint_path"))
        self._checkpoint_version = str(
            params.pop("checkpoint_version", CHECKPOINT_VERSION)
        )
        self._checkpoint_sha256 = str(
            params.pop("checkpoint_sha256", CHECKPOINT_SHA256)
        ).lower()
        self._package_version = str(params.pop("package_version", PACKAGE_VERSION))
        self._source_revision = str(params.pop("source_revision", SOURCE_REVISION))
        self._perm_sample = int(params.pop("perm_sample", 1000))
        self._perm_repeats = int(params.pop("perm_repeats", 1))
        if params:
            raise ValueError(f"tabiclv2가 모르는 params: {sorted(params)}")

        if self._n_estimators not in {1, 8}:
            raise ValueError(
                "tabiclv2 n_estimators는 진입 확인용 1 또는 정식 기본값 8이어야 한다."
            )
        if self._batch_size != 8:
            raise ValueError("tabiclv2 batch_size는 공식 기본값 8이어야 한다.")
        if self._kv_cache != "repr":
            raise ValueError("tabiclv2 반복 예측 캐시는 메모리 절약형 'repr'로 고정한다.")
        if self._device not in {"cpu", "cuda"}:
            raise ValueError("tabiclv2 device는 'cpu' 또는 'cuda'여야 한다.")
        if self._offload_mode not in {"auto", "cpu", "disk"}:
            raise ValueError(
                "tabiclv2 offload_mode은 'auto', 'cpu', 'disk' 중 하나여야 한다."
            )
        if self._checkpoint_path.name != self._checkpoint_version:
            raise ValueError("checkpoint_path 파일명과 checkpoint_version이 다르다.")
        if self._checkpoint_version != CHECKPOINT_VERSION:
            raise ValueError(f"TabICLv2 분류 가중치는 {CHECKPOINT_VERSION}만 허용한다.")
        if not re.fullmatch(r"[0-9a-f]{64}", self._checkpoint_sha256):
            raise ValueError("checkpoint_sha256은 64자리 소문자 16진수여야 한다.")
        if self._checkpoint_sha256 != CHECKPOINT_SHA256:
            raise ValueError("TabICLv2 가중치 SHA-256이 고정 판본과 다르다.")
        if self._package_version != PACKAGE_VERSION:
            raise ValueError(f"tabicl 패키지는 {PACKAGE_VERSION}만 허용한다.")
        if self._source_revision != SOURCE_REVISION:
            raise ValueError("tabicl 소스 판본이 고정 태그 커밋과 다르다.")
        if self._perm_sample != 1000 or self._perm_repeats != 1:
            raise ValueError("tabiclv2 importance는 검증 1,000행과 반복 1회로 고정한다.")

        self._seed = seed
        self._active_offload_mode = self._offload_mode
        self._checkpoint_actual_sha256: str | None = None
        self._model = None
        self._columns: list[str] | None = None
        self._X_va: pd.DataFrame | None = None
        self._y_va: np.ndarray | None = None
        self._va_pred: np.ndarray | None = None
        self._train_index: pd.Index | None = None
        self._validation_index: pd.Index | None = None
        self._timings: dict[str, float] = {}
        self._disk_retry = False

    def _verify_runtime(self) -> None:
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                f"TabICLv2 가중치 파일이 없다: {self._checkpoint_path}"
            )
        actual = _sha256(self._checkpoint_path)
        if actual != self._checkpoint_sha256:
            raise ValueError(
                "TabICLv2 가중치 SHA-256 불일치: "
                f"expected={self._checkpoint_sha256} actual={actual}"
            )
        self._checkpoint_actual_sha256 = actual
        installed = importlib.metadata.version("tabicl")
        if installed != self._package_version:
            raise RuntimeError(
                f"tabicl 설치 판본이 다르다: expected={self._package_version} actual={installed}"
            )
        self._disk_offload_dir.mkdir(parents=True, exist_ok=True)

    def _new_classifier(self):
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            n_estimators=self._n_estimators,
            norm_methods=None,
            feat_shuffle_method="latin",
            class_shuffle_method="shift",
            outlier_threshold=4.0,
            softmax_temperature=0.9,
            average_logits=True,
            support_many_classes=True,
            batch_size=self._batch_size,
            kv_cache=self._kv_cache,
            model_path=self._checkpoint_path,
            allow_auto_download=False,
            checkpoint_version=self._checkpoint_version,
            device=self._device,
            use_amp="auto",
            use_fa3="auto",
            offload_mode=self._active_offload_mode,
            disk_offload_dir=str(self._disk_offload_dir),
            random_state=self._seed,
            n_jobs=None,
            verbose=False,
            inference_config=None,
        )

    def _fit_model(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> None:
        self._model = self._new_classifier()
        self._model.fit(X_tr, y_tr)

    def _switch_to_disk(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> None:
        if self._active_offload_mode == "disk":
            raise RuntimeError("TabICLv2 디스크 이동 상태에서도 CUDA 메모리가 부족하다.")
        self._active_offload_mode = "disk"
        self._disk_retry = True
        self._model = None
        _clear_cuda_cache()
        self._fit_model(X_tr, y_tr)

    def _predict_with_retry(
        self, X: pd.DataFrame, X_tr: pd.DataFrame, y_tr: pd.Series
    ) -> np.ndarray:
        try:
            return np.asarray(self._model.predict_proba(X)[:, 1], dtype="float64")
        except Exception as exc:
            if not _is_cuda_oom(exc) or self._active_offload_mode == "disk":
                raise
            self._switch_to_disk(X_tr, y_tr)
            return np.asarray(self._model.predict_proba(X)[:, 1], dtype="float64")

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        if not X_tr.index.intersection(X_va.index).empty:
            raise ValueError("TabICLv2 outer 학습 fold와 검증 fold의 index가 겹친다.")
        self._columns = list(X_tr.columns)
        if list(X_va.columns) != self._columns:
            raise ValueError("TabICLv2 학습과 검증 컬럼이 다르다.")
        self._train_index = X_tr.index.copy()
        self._validation_index = X_va.index.copy()
        self._X_tr = X_tr.copy()
        self._y_tr = y_tr.copy()
        self._X_va = X_va.copy()
        self._y_va = y_va.to_numpy(dtype="int64")

        started = time.monotonic()
        self._verify_runtime()
        self._timings["runtime_verification"] = time.monotonic() - started

        started = time.monotonic()
        try:
            self._fit_model(self._X_tr, self._y_tr)
        except Exception as exc:
            if not _is_cuda_oom(exc) or self._active_offload_mode == "disk":
                raise
            self._switch_to_disk(self._X_tr, self._y_tr)
        self._timings["fit"] = time.monotonic() - started

        started = time.monotonic()
        self._va_pred = self._predict_with_retry(
            self._X_va, self._X_tr, self._y_tr
        )
        self._timings["validation_predict"] = time.monotonic() - started
        if self._va_pred.shape != (len(self._X_va),):
            raise ValueError("TabICLv2 검증 예측 행 수가 입력과 다르다.")
        print(
            f"[tabiclv2] fold valAUC={roc_auc_score(self._y_va, self._va_pred):.5f} "
            f"n_estimators={self._n_estimators} offload_mode={self._active_offload_mode}",
            flush=True,
        )
        return self._va_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if list(X.columns) != self._columns:
            raise ValueError("TabICLv2 예측 컬럼이 학습 때와 다르다.")
        return self._predict_with_retry(X, self._X_tr, self._y_tr)

    def importance(self) -> pd.DataFrame:
        """검증 fold의 seed 고정 1,000행 permutation importance."""
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
        base_pred = self._predict_with_retry(X_va, self._X_tr, self._y_tr)
        base = roc_auc_score(y_va, base_pred)

        gains = []
        for index, column in enumerate(self._columns):
            rng = np.random.default_rng(self._seed * 10007 + index * 101)
            X_perm = X_va.copy()
            X_perm[column] = X_va[column].take(rng.permutation(len(X_va))).to_numpy()
            pred = self._predict_with_retry(X_perm, self._X_tr, self._y_tr)
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
                "missing_rows": int(missing.sum()),
                "complete_rows": int((~missing).sum()),
                "missing_auc": subgroup_auc(missing),
                "complete_auc": subgroup_auc(~missing),
                "n_estimators": self._n_estimators,
                "batch_size": self._batch_size,
                "kv_cache": self._kv_cache,
                "requested_offload_mode": self._offload_mode,
                "active_offload_mode": self._active_offload_mode,
                "disk_retry": self._disk_retry,
                "package_version": self._package_version,
                "source_revision": self._source_revision,
                "checkpoint_repository": CHECKPOINT_REPOSITORY,
                "checkpoint_version": self._checkpoint_version,
                "checkpoint_sha256": self._checkpoint_actual_sha256,
                "timing_seconds": self._timings,
            },
        )
