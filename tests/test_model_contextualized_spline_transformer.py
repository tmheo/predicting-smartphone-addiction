"""Contextualized deep univariate Transformer 모델 계열 회귀 시험. (#149)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig, load_config
from pipeline.plan import FeaturePlan

SEED = 7
SMALL_PARAMS = {
    "exact_cols": ["v", "c"],
    "numeric_mode": "spline",
    "token_dim": 16,
    "attention_dim": 16,
    "attention_heads": 4,
    "default_width": 16,
    "default_depth": 1,
    "context_hidden": 8,
    "gate_hidden": 8,
    "residual_hidden": 32,
    "epochs": 4,
    "patience": 2,
    "batch_size": 64,
    "perm_repeats": 1,
    "dropout": 0.0,
}


def _data(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    values = rng.choice([1.5, 2.5, 3.5, 4.5], size=n)
    category = pd.Series(rng.choice(["Low", "High"], size=n), dtype="object")
    category = category.where(rng.uniform(size=n) > 0.1)
    X = pd.DataFrame(
        {
            "v": values,
            "c": category,
            "z": rng.normal(size=n),
        }
    )
    y = pd.Series((values > 2.5).astype(int) ^ (rng.uniform(size=n) < 0.08))
    return X, y.astype(int)


def _adapter(mode: str = "spline") -> model_mod.ContextualizedSplineTransformerAdapter:
    params = dict(SMALL_PARAMS, numeric_mode=mode)
    cfg = ModelConfig(
        kind="contextualized_spline_transformer",
        params=params,
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.ContextualizedSplineTransformerAdapter)
    return adapter


def test_m0_and_a0_keep_exp067_feature_plan_and_33_columns():
    baseline = load_config("configs/exp067_lookup_xgb_impute_comps5.yaml", "screen")
    m0 = load_config("configs/exp085_contextual_spline_m0.yaml", "screen")
    a0 = load_config("configs/exp086_contextual_periodic_a0.yaml", "screen")
    assert m0.features == baseline.features
    assert a0.features == baseline.features

    raw = pd.DataFrame(
        {
            "id": [0],
            "age": [20.0],
            "gender": pd.Categorical(["Female"]),
            "daily_screen_time_hours": [5.0],
            "social_media_hours": [1.0],
            "gaming_hours": [1.0],
            "work_study_hours": [2.0],
            "sleep_hours": [8.0],
            "notifications_per_day": [10.0],
            "app_opens_per_day": [5.0],
            "weekend_screen_time": [6.0],
            "stress_level": pd.Categorical(["Low"]),
            "academic_work_impact": pd.Categorical(["No"]),
            "addicted_label": [0],
        }
    )
    plan = FeaturePlan.from_config(m0.features)
    plan.apply_dataset_wide(raw, raw.drop(columns="addicted_label"))
    assert len(plan.all_columns()) == 33


@pytest.mark.parametrize("mode", ["spline", "periodic"])
def test_adapter_contract_and_learning(mode: str):
    X, y = _data()
    adapter = _adapter(mode)
    validation = adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])

    assert validation.shape == (60,)
    assert np.isfinite(validation).all()
    assert ((validation >= 0) & (validation <= 1)).all()
    assert roc_auc_score(y.iloc[180:], validation) > 0.75

    test_prediction = adapter.predict(X.iloc[:12])
    assert test_prediction.shape == (12,)
    assert np.isfinite(test_prediction).all()

    importance = adapter.importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert list(importance["feature"]) == list(X.columns)
    assert np.isfinite(importance["gain"]).all()
    pd.testing.assert_frame_equal(importance, adapter.importance())

    diagnostics = adapter.entry_diagnostics()
    assert diagnostics.assertions == {
        "preprocessing_training_rows_only": True,
        "validation_labels_excluded_from_preprocessing": True,
        "missing_and_unknown_ids_distinct": True,
    }
    assert diagnostics.observations["numeric_mode"] == mode
    assert diagnostics.observations["trainable_parameters"] > 0


def test_training_only_vocab_distinguishes_unknown_and_missing():
    X, y = _data()
    adapter = _adapter()
    adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])

    probe = pd.DataFrame(
        {
            "v": [9.9, np.nan],
            "c": ["NeverSeen", None],
            "z": [0.0, np.nan],
        }
    )
    prediction = adapter.predict(probe)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()

    encoded = adapter._impl._encode(probe)
    exact_ids = encoded[1].cpu().numpy()
    assert (exact_ids[0] != exact_ids[1]).all()
    assert (exact_ids[1] == 0).all()


def test_periodic_ablation_keeps_parameter_scale_close_to_spline():
    X, y = _data(80)
    counts = {}
    for mode in ("spline", "periodic"):
        adapter = _adapter(mode)
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])
        counts[mode] = adapter.entry_diagnostics().observations["trainable_parameters"]

    relative_difference = abs(counts["spline"] - counts["periodic"]) / counts["spline"]
    assert relative_difference < 0.05


@pytest.mark.parametrize("mode", ["unknown", "pwl", "fourier"])
def test_rejects_unknown_numeric_mode(mode: str):
    params = dict(SMALL_PARAMS, numeric_mode=mode)
    cfg = ModelConfig(
        kind="contextualized_spline_transformer",
        params=params,
        fit={},
    )
    with pytest.raises(ValueError, match="numeric_mode"):
        model_mod.create(cfg, seed=SEED)


def test_rejects_unknown_params_and_fit_settings():
    X, y = _data(80)
    cfg = ModelConfig(
        kind="contextualized_spline_transformer",
        params=dict(SMALL_PARAMS, no_such_param=1),
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    with pytest.raises(ValueError, match="no_such_param"):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])

    cfg = ModelConfig(
        kind="contextualized_spline_transformer",
        params=dict(SMALL_PARAMS),
        fit={"no_such_fit_setting": 1},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    with pytest.raises(ValueError, match="fit 설정"):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])


def test_rejects_initial_score():
    X, y = _data(80)
    adapter = _adapter()
    with pytest.raises(ValueError, match="초기 점수"):
        adapter.fit(
            X.iloc[:60],
            y.iloc[:60],
            X.iloc[60:],
            y.iloc[60:],
            pd.Series(np.zeros(60)),
            pd.Series(np.zeros(20)),
        )
