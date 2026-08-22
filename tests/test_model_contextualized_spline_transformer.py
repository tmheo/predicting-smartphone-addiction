"""Contextualized deep univariate Transformer 모델 계열 회귀 시험. (#149)"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
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


def _adapter(
    mode: str = "spline", **overrides
) -> model_mod.ContextualizedSplineTransformerAdapter:
    params = dict(SMALL_PARAMS, numeric_mode=mode, **overrides)
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


def test_contextual_spline_muon_config_is_exp085_single_optimizer_delta():
    baseline = load_config("configs/exp085_contextual_spline_m0.yaml", "screen")
    challenger = load_config("configs/exp135_contextual_spline_muon.yaml", "screen")

    assert challenger.name == "exp135_contextual_spline_muon"
    assert challenger.data == baseline.data
    assert challenger.features == baseline.features
    assert challenger.model.kind == baseline.model.kind
    assert challenger.model.fit == baseline.model.fit
    assert challenger.model.params == {**baseline.model.params, "optimizer": "muon"}


def test_multilevel_config_is_exp085_single_output_path_delta():
    baseline = load_config("configs/exp085_contextual_spline_m0.yaml", "screen")
    challenger = load_config(
        "configs/exp136_contextual_spline_multilevel.yaml", "screen"
    )

    assert challenger.name == "exp136_contextual_spline_multilevel"
    assert challenger.data == baseline.data
    assert challenger.features == baseline.features
    assert challenger.model.kind == baseline.model.kind
    assert challenger.model.fit == baseline.model.fit
    assert challenger.model.params == {
        **baseline.model.params,
        "output_path": "multilevel_stage1",
    }


def test_recon_widths_config_is_exp085_three_feature_delta():
    from pipeline.features import ConstrainedImputeAux

    baseline = load_config("configs/exp085_contextual_spline_m0.yaml", "screen")
    challenger = load_config(
        "configs/exp137_contextual_spline_recon_widths.yaml", "screen"
    )

    assert challenger.name == "exp137_contextual_spline_recon_widths"
    assert challenger.data == baseline.data
    assert challenger.features.base == baseline.features.base
    assert challenger.features.categorical == baseline.features.categorical
    assert challenger.features.exclude == baseline.features.exclude
    assert challenger.model == baseline.model

    expected_providers = [dict(provider) for provider in baseline.features.providers]
    constrained = next(
        provider
        for provider in expected_providers
        if provider["kind"] == "constrained_impute_aux"
    )
    constrained["widths"] = True
    assert challenger.features.providers == expected_providers

    baseline_provider = next(
        provider
        for provider in FeaturePlan.from_config(baseline.features).fold_fit_transformers()
        if isinstance(provider, ConstrainedImputeAux)
    )
    challenger_provider = next(
        provider
        for provider in FeaturePlan.from_config(challenger.features).fold_fit_transformers()
        if isinstance(provider, ConstrainedImputeAux)
    )
    baseline_columns = baseline_provider.columns()
    challenger_columns = challenger_provider.columns()
    assert challenger_columns[: len(baseline_columns)] == baseline_columns
    assert set(challenger_columns) - set(baseline_columns) == {
        "gaming_hours_recon_width",
        "social_media_hours_recon_width",
        "work_study_hours_recon_width",
    }


def test_orig_cdf_diff_config_is_exp085_five_feature_delta():
    baseline = yaml.safe_load(
        Path("configs/exp085_contextual_spline_m0.yaml").read_text(encoding="utf-8")
    )
    challenger = yaml.safe_load(
        Path("configs/exp140_contextual_spline_orig_cdf_diff.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert challenger["name"] == "exp140_contextual_spline_orig_cdf_diff"
    assert challenger["data"] == baseline["data"]
    assert challenger["features"]["base"] == baseline["features"]["base"]
    assert (
        challenger["features"]["categorical"]
        == baseline["features"]["categorical"]
    )
    assert challenger["features"].get("exclude", []) == baseline["features"].get(
        "exclude", []
    )
    assert challenger["model"] == baseline["model"]

    cdf_provider = {
        "kind": "original_cdf_diff",
        "path": "data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv",
        "cols": [
            "daily_screen_time_hours",
            "weekend_screen_time",
            "social_media_hours",
            "notifications_per_day",
            "app_opens_per_day",
        ],
    }
    expected_providers = [
        dict(provider) for provider in baseline["features"]["providers"]
    ]
    expected_providers.insert(1, cdf_provider)
    assert challenger["features"]["providers"] == expected_providers

    added_columns = [f"{column}_orig_cdf_diff" for column in cdf_provider["cols"]]
    assert added_columns == [
        "daily_screen_time_hours_orig_cdf_diff",
        "weekend_screen_time_orig_cdf_diff",
        "social_media_hours_orig_cdf_diff",
        "notifications_per_day_orig_cdf_diff",
        "app_opens_per_day_orig_cdf_diff",
    ]
    assert all("kde" not in column for column in added_columns)


def test_multilevel_path_preserves_shared_initial_weights_and_favors_final_head():
    import torch

    from pipeline.contextualized_spline_transformer import (
        ContextualizedSplineTransformerFold,
    )

    X, _ = _data(80)

    def build(output_path: str):
        fold = ContextualizedSplineTransformerFold(
            dict(SMALL_PARAMS, output_path=output_path), seed=SEED
        )
        fold._seed_everything()
        fold._fit_preprocessing(X)
        numeric, exact = fold._encode(X)
        return fold._build_model(numeric.numpy()), numeric, exact

    baseline, _, _ = build("direct_final")
    challenger, numeric, exact = build("multilevel_stage1")
    baseline_state = baseline.state_dict()
    challenger_state = challenger.state_dict()

    assert set(baseline_state) < set(challenger_state)
    for name, value in baseline_state.items():
        torch.testing.assert_close(value, challenger_state[name])

    output = challenger(numeric[:12], exact[:12])
    assert output["mixer_weights"].shape == (12, 4)
    torch.testing.assert_close(
        output["mixer_weights"].sum(dim=1), torch.ones(12)
    )
    assert torch.all(output["mixer_weights"][:, 3] > 0.7)


def test_multilevel_path_contract_and_diagnostics():
    X, y = _data(120)
    adapter = _adapter(
        output_path="multilevel_stage1", epochs=2, patience=1
    )

    prediction = adapter.fit(X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:])

    assert prediction.shape == (30,)
    assert np.isfinite(prediction).all()
    diagnostics = adapter.training_diagnostics()
    assert diagnostics["output_path"] == "multilevel_stage1"
    assert diagnostics["best_base_final_auc"] >= 0
    assert diagnostics["best_univariate_auc"] >= 0
    assert diagnostics["best_attention_auc"] >= 0
    assert sum(diagnostics["best_mixer_mean_weights"]) == pytest.approx(1.0)


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
    assert diagnostics.observations["optimizer"] == "adamw"
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


def test_full_fit_uses_fixed_epoch_budget_and_predicts():
    X, y = _data(80)
    adapter = _adapter()

    model_mod.fit_full(adapter, X, y, training_budget=2)

    prediction = adapter.predict(X.iloc[:12])
    assert prediction.shape == (12,)
    assert np.isfinite(prediction).all()
    assert adapter.training_diagnostics() == {
        "initialization_seed": SEED,
        "numeric_mode": "spline",
        "optimizer": "adamw",
        "output_path": "direct_final",
        "configured_epochs": 4,
        "end_epoch": 2,
        "best_epoch": 2,
        "observed_best_epoch": None,
        "best_validation_auc": None,
        "full_fit": True,
    }


def test_full_fit_requires_fixed_epoch_budget():
    X, y = _data(80)
    adapter = _adapter()

    with pytest.raises(ValueError, match="고정 epoch 수"):
        model_mod.fit_full(adapter, X, y, training_budget=None)


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


def test_contextual_spline_rejects_unknown_output_path():
    X, y = _data(80)
    with pytest.raises(ValueError, match="output_path"):
        _adapter(output_path="unknown").fit(
            X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:]
        )


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


def test_contextual_spline_muon_selects_hidden_linear_matrices():
    from pipeline.contextualized_spline_transformer import (
        ContextualizedSplineTransformerFold,
        _muon_parameter_names,
    )

    X, y = _data(80)
    fold = ContextualizedSplineTransformerFold(
        dict(SMALL_PARAMS, optimizer="muon"), seed=SEED
    )
    fold._fit_preprocessing(X)
    numeric, _ = fold._encode(X)
    model = fold._build_model(numeric.numpy())
    names = _muon_parameter_names(model)
    named = dict(model.named_parameters())

    assert names
    assert all(named[name].ndim == 2 for name in names)
    assert "interaction.attention.in_proj_weight" in names
    assert "interaction.attention.out_proj.weight" in names
    assert "final_head.0.weight" in names
    assert "final_head.4.weight" in names
    assert "final_head.8.weight" not in names
    assert not any(name.startswith("additive_heads.") for name in names)
    assert not any(name.startswith("exact_embeddings.") for name in names)


def test_contextual_spline_muon_contract_and_learning():
    X, y = _data(120)
    adapter = _adapter(optimizer="muon", epochs=2, patience=1)

    prediction = adapter.fit(X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:])

    assert prediction.shape == (30,)
    assert np.isfinite(prediction).all()
    assert adapter.training_diagnostics()["optimizer"] == "muon"


def test_contextual_spline_rejects_unknown_optimizer():
    X, y = _data(80)
    with pytest.raises(ValueError, match="optimizer"):
        _adapter(optimizer="sgd").fit(
            X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:]
        )
