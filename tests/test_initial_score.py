from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.data import file_sha256
from pipeline.initial_score import (
    KnownOriginalRule,
    OriginalProxyLightGBM,
    probabilities_to_logits,
)


def test_probabilities_to_logits_clips_extremes_and_is_finite():
    logits = probabilities_to_logits([0.0, 0.5, 1.0], clip=1e-3)
    assert np.isfinite(logits).all()
    assert logits[0] == pytest.approx(-logits[2])
    assert logits[1] == pytest.approx(0.0)


def test_known_original_rule_semantics_and_target_boundary():
    X = pd.DataFrame(
        {
            "daily_screen_time_hours": [9.0, 5.0, 7.0, np.nan],
            "social_media_hours": [1.0, 3.0, 3.0, 5.0],
        }
    )
    provider = KnownOriginalRule(clip=0.01)
    scores = provider.compute(X, X.copy(), seed=42)
    assert scores.train.iloc[0] > 0
    assert scores.train.iloc[1] < 0
    assert scores.train.iloc[2] == pytest.approx(0.0)
    assert scores.train.iloc[3] > 0

    with pytest.raises(ValueError, match="합성 타깃"):
        provider.compute(X.assign(addicted_label=[0, 1, 0, 1]), X, seed=42)


def test_original_proxy_model_uses_only_proxy_labels(tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    proxy = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "group": np.where(np.arange(n) % 2, "a", "b"),
        }
    )
    proxy["addicted_label"] = (proxy["x"] + (proxy["group"] == "a") > 0).astype(int)
    path = tmp_path / "proxy.csv"
    proxy.to_csv(path, index=False)
    provider = OriginalProxyLightGBM(
        path=str(path),
        cols=["x", "group"],
        categorical=["group"],
        model_params={
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 30,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbosity": -1,
        },
        n_splits=3,
        early_stopping_rounds=5,
        sha256=file_sha256(path),
    )
    X = proxy[["x", "group"]].iloc[:20].copy()
    scores = provider.compute(X, X.copy(), seed=42)
    assert scores.train.index.equals(X.index)
    assert scores.test.index.equals(X.index)
    assert np.isfinite(scores.train).all()
    assert scores.train.to_numpy() == pytest.approx(scores.test.to_numpy())
