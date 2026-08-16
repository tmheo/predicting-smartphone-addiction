"""Trompt 구조, 전처리, 결정성과 공개 이진 자료 건전성 시험. (#145)"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig


def _params(**overrides) -> dict:
    params = {
        "prompts": 4,
        "width": 12,
        "cells": 2,
        "epochs": 12,
        "batch_size": 64,
        "eval_batch_size": 128,
        "lr": 0.01,
        "weight_decay": 1e-5,
        "patience": 4,
        "perm_sample": 64,
        "perm_repeats": 1,
        "device": "cpu",
    }
    params.update(overrides)
    return params


def _adapter(seed: int = 7, **overrides) -> model_mod.TromptAdapter:
    adapter = model_mod.create(
        ModelConfig(kind="trompt", params=_params(**overrides), fit={}), seed=seed
    )
    assert isinstance(adapter, model_mod.TromptAdapter)
    return adapter


def _data(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(12)
    signal = rng.normal(size=n)
    numeric = pd.Series(signal).where(np.arange(n) % 13 != 0)
    category = pd.Categorical(np.where(signal > 0, "high", "low"))
    X = pd.DataFrame(
        {
            "signal": numeric,
            "noise": rng.normal(size=n),
            "category": category,
            "signal_missing": numeric.isna().astype("float32"),
        }
    )
    y = pd.Series(((signal + rng.normal(scale=0.4, size=n)) > 0).astype("int64"))
    return X, y


def test_trompt_adapter_contract_and_determinism():
    X, y = _data()
    first = _adapter()
    first_prediction = first.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    second = _adapter()
    second_prediction = second.fit(
        X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:]
    )

    np.testing.assert_array_equal(first_prediction, second_prediction)
    assert roc_auc_score(y.iloc[240:], first_prediction) > 0.8
    assert np.isfinite(first.predict(X.iloc[240:250])).all()

    importance = first.importance()
    assert importance["feature"].tolist() == list(X.columns)
    assert np.isfinite(importance["gain"]).all()

    diagnostics = first.entry_diagnostics()
    assert all(diagnostics.assertions.values())
    assert diagnostics.observations["batch_probe"]["cell_output_shape"] == [64, 2, 2]
    assert diagnostics.observations["batch_probe"]["deterministic"] is True
    assert diagnostics.observations["input_columns"] == 4
    assert diagnostics.observations["requested_eval_batch_size"] == 128
    assert diagnostics.observations["effective_eval_batch_size"] == 64
    assert (
        diagnostics.observations["training_losses"][-1]
        < diagnostics.observations["training_losses"][0]
    )
    assert diagnostics.observations["cublas_workspace_config"] in {
        ":4096:8",
        ":16:8",
    }
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}


def test_trompt_learns_public_breast_cancer_binary_dataset():
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.astype("float64")
    X.iloc[::17, 0] = np.nan
    X["mean radius_missing"] = X["mean radius"].isna().astype("float32")
    y = bunch.target.astype("int64")
    adapter = _adapter(epochs=10, patience=3)
    prediction = adapter.fit(X.iloc[:450], y.iloc[:450], X.iloc[450:], y.iloc[450:])

    diagnostics = adapter.entry_diagnostics().observations
    assert diagnostics["training_losses"][-1] < diagnostics["training_losses"][0]
    assert roc_auc_score(y.iloc[450:], prediction) > 0.5


def test_trompt_stops_after_first_epoch_when_runtime_projection_exceeds_limit():
    X, y = _data()
    adapter = _adapter(epochs=3, max_projected_5fold_hours=1e-9)
    prediction = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    diagnostics = adapter.entry_diagnostics().observations
    assert prediction.shape == (80,)
    assert len(diagnostics["epoch_seconds"]) == 1
    assert diagnostics["projected_5fold_training_seconds"] > 0
    assert adapter.entry_abort_reason() is not None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prompts": 3}, "짝수"),
        ({"max_projected_5fold_hours": 0}, "양수"),
        ({"unknown": 1}, "모르는 params"),
    ],
)
def test_trompt_rejects_invalid_or_unknown_params(overrides, message):
    X, y = _data()
    adapter = _adapter(**overrides)
    with pytest.raises(ValueError, match=message):
        adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
