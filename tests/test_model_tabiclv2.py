"""TabICLv2 공식 추론기 adapter 경계 회귀 시험. (#143)"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline import tabiclv2 as tabiclv2_mod
from pipeline.config import ModelConfig


class FakeTabICL:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> FakeTabICL:
        self.train = X.copy()
        self.y = y.copy()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        signal = X["signal"].fillna(0).to_numpy(dtype="float64")
        probability = 1 / (1 + np.exp(-signal))
        return np.column_stack([1 - probability, probability])


def _data(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(14)
    signal = rng.normal(size=n)
    index = np.arange(2000, 2000 + n)
    X = pd.DataFrame(
        {
            "signal": pd.Series(signal, index=index).mask(np.arange(n) % 19 == 0),
            "noise": pd.Series(rng.normal(size=n), index=index),
            "category": pd.Categorical(
                pd.Series(np.where(signal > 0, "high", "low"), index=index).mask(
                    np.arange(n) % 23 == 0
                ),
                categories=["high", "low"],
            ),
        },
        index=index,
    )
    y = pd.Series(
        (signal + rng.normal(scale=0.25, size=n) > 0).astype(int), index=index
    )
    return X, y


def _adapter(monkeypatch, tmp_path, **overrides) -> model_mod.TabICLv2Adapter:
    checkpoint = tmp_path / tabiclv2_mod.CHECKPOINT_VERSION
    checkpoint.write_bytes(b"test-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(tabiclv2_mod, "CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(
        tabiclv2_mod.TabICLv2Fold,
        "_verify_runtime",
        lambda self: setattr(self, "_checkpoint_actual_sha256", digest),
    )
    monkeypatch.setattr(
        tabiclv2_mod.TabICLv2Fold, "_new_classifier", lambda self: FakeTabICL()
    )
    params = {
        "n_estimators": 8,
        "batch_size": 8,
        "kv_cache": "repr",
        "device": "cpu",
        "offload_mode": "auto",
        "disk_offload_dir": str(tmp_path / "offload"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "perm_sample": 1000,
        "perm_repeats": 1,
    }
    params.update(overrides)
    adapter = model_mod.create(
        ModelConfig(kind="tabiclv2", params=params, fit={}), seed=42
    )
    assert isinstance(adapter, model_mod.TabICLv2Adapter)
    return adapter


def test_tabiclv2_adapter_contract_official_input_and_importance(monkeypatch, tmp_path):
    X, y = _data()
    adapter = _adapter(monkeypatch, tmp_path)
    validation_pred = adapter.fit(
        X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:]
    )

    assert validation_pred.shape == (60,)
    assert roc_auc_score(y.iloc[180:], validation_pred) > 0.9
    prediction = adapter.predict(X.iloc[10:25].reset_index(drop=True))
    assert prediction.shape == (15,)

    importance = adapter.importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert importance["feature"].tolist() == list(X.columns)
    assert importance.loc[importance["feature"] == "signal", "gain"].item() > (
        importance.loc[importance["feature"] == "noise", "gain"].item()
    )

    diagnostics = adapter.entry_diagnostics()
    assert all(diagnostics.assertions.values())
    assert diagnostics.observations["n_estimators"] == 8
    assert diagnostics.observations["kv_cache"] == "repr"
    assert diagnostics.observations["missing_rows"] > 0
    assert diagnostics.observations["complete_rows"] > 0
    assert diagnostics.observations["missing_auc"] is not None
    assert diagnostics.observations["complete_auc"] is not None


def test_tabiclv2_rejects_overlapping_outer_rows(monkeypatch, tmp_path):
    X, y = _data()
    adapter = _adapter(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="index가 겹친다"):
        adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[170:], y.iloc[170:])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_estimators": 2}, "1 또는"),
        ({"batch_size": 4}, "공식 기본값 8"),
        ({"kv_cache": False}, "메모리 절약형"),
        ({"perm_sample": 999}, "1,000행"),
        ({"source_revision": "0" * 40}, "소스 판본"),
        ({"unknown": 1}, "모르는 params"),
    ],
)
def test_tabiclv2_rejects_non_baseline_or_unknown_params(
    monkeypatch, tmp_path, override, message
):
    X, y = _data()
    adapter = _adapter(monkeypatch, tmp_path, **override)
    with pytest.raises(ValueError, match=message):
        adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])
