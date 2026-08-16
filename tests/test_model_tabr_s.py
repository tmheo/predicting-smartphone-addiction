"""TabR-S adapter와 후보 저장소 경계 회귀 시험. (#142)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig


def _data(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(12)
    signal = rng.normal(size=n)
    row_index = np.arange(1000, 1000 + n)
    category = pd.Series(np.where(signal > 0, "high", "low"), index=row_index)
    category.iloc[::17] = None
    X = pd.DataFrame(
        {
            "signal": pd.Series(signal, index=row_index).where(np.arange(n) % 19 != 0),
            "noise": pd.Series(rng.normal(size=n), index=row_index),
            "category": pd.Categorical(category, categories=["high", "low"]),
        },
        index=row_index,
    )
    y = pd.Series(
        ((signal + rng.normal(scale=0.35, size=n)) > 0).astype(int), index=X.index
    )
    return X, y


def _adapter(**overrides) -> model_mod.TabRSAdapter:
    params = {
        "context_size": 96,
        "freeze_contexts_after_n_epochs": 1,
        "epochs": 2,
        "batch_size": 64,
        "eval_batch_size": 128,
        "candidate_encoding_batch_size": 128,
        "d_main": 16,
        "d_multiplier": 2.0,
        "context_dropout": 0.0,
        "dropout0": 0.0,
        "dropout1": 0.0,
        "lr": 0.01,
        "perm_sample": 24,
        "perm_repeats": 1,
        "diagnostic_context_sample": 16,
        "device": "cpu",
    }
    params.update(overrides)
    adapter = model_mod.create(
        ModelConfig(kind="tabr_s", params=params, fit={}), seed=7
    )
    assert isinstance(adapter, model_mod.TabRSAdapter)
    return adapter


def test_tabr_s_adapter_contract_and_candidate_boundaries():
    X, y = _data()
    adapter = _adapter()
    validation_pred = adapter.fit(
        X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:]
    )

    assert validation_pred.shape == (60,)
    assert np.isfinite(validation_pred).all()
    assert ((validation_pred >= 0) & (validation_pred <= 1)).all()
    assert roc_auc_score(y.iloc[180:], validation_pred) > 0.75

    test_pred = adapter.predict(X.iloc[180:190].reset_index(drop=True))
    assert test_pred.shape == (10,)
    assert np.isfinite(test_pred).all()

    importance = adapter.importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert importance["feature"].tolist() == list(X.columns)
    assert np.isfinite(importance["gain"]).all()

    diagnostics = adapter.entry_diagnostics()
    assert all(diagnostics.assertions.values())
    assert diagnostics.observations["candidate_rows"] == 180
    assert diagnostics.observations["validation_rows"] == 60
    assert len(diagnostics.observations["epoch_seconds"]) == 2
    assert len(diagnostics.observations["epoch_validation_aucs"]) == 2
    assert diagnostics.observations["context_id_change_rate_epoch_1_to_2"] is not None
    for sample in diagnostics.observations["self_exclusion_samples"]:
        assert sample["query_row_id"] not in sample["context_row_ids"]


def test_tabr_s_rejects_overlapping_outer_fold_rows():
    X, y = _data()
    adapter = _adapter()
    with pytest.raises(ValueError, match="index가 겹친다"):
        adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[170:], y.iloc[170:])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"context_size": 32}, "context_size는 96"),
        ({"freeze_contexts_after_n_epochs": 2}, "첫 epoch 뒤"),
        ({"unknown": 1}, "모르는 params"),
    ],
)
def test_tabr_s_rejects_non_baseline_or_unknown_params(override, message):
    X, y = _data()
    adapter = _adapter(**override)
    with pytest.raises(ValueError, match=message):
        adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])
