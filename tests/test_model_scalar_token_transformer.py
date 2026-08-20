"""스칼라 token Transformer 모델 계열 회귀 시험. (#178)"""

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
    "mixing": "attention",
    "relu_basis_dim": 4,
    "periodic_dim": 4,
    "model_dim": 16,
    "attention_heads": 4,
    "layers": 1,
    "feedforward_dim": 32,
    "backbone_dims": [32, 16, 8],
    "epochs": 8,
    "patience": 4,
    "batch_size": 32,
    "prediction_batch_size": 128,
    "perm_repeats": 1,
    "dropout": 0.0,
    "head_dropout": 0.0,
    "mixup_alpha": 0.0,
    "ema_decay": 0.5,
    "quantiles": 64,
}


def _data(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    value = rng.normal(size=n)
    category = pd.Series(rng.choice(["Low", "High"], size=n), dtype="object")
    category = category.where(rng.uniform(size=n) > 0.08)
    X = pd.DataFrame(
        {
            "value": value,
            "category": category,
            "noise": rng.normal(size=n),
        }
    )
    target = (value > 0.0).astype(int)
    target ^= rng.uniform(size=n) < 0.02
    return X, pd.Series(target, dtype="int64")


def _adapter(mixing: str = "attention") -> model_mod.ScalarTokenTransformerAdapter:
    params = dict(SMALL_PARAMS, mixing=mixing)
    cfg = ModelConfig(kind="scalar_token_transformer", params=params, fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.ScalarTokenTransformerAdapter)
    return adapter


def test_m0_and_a0_keep_champion_feature_plan_and_33_columns():
    champion = load_config("configs/exp081_lookup_fold_initialization_avg3.yaml", "screen")
    m0 = load_config("configs/exp115_scalar_token_transformer_m0.yaml", "screen")
    a0 = load_config("configs/exp116_scalar_token_mlp_a0.yaml", "screen")
    assert m0.features == champion.features
    assert a0.features == champion.features

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


def test_oof_target_mean_arm_adds_twelve_treatments_and_one_canary():
    baseline = load_config("configs/exp115_scalar_token_transformer_m0.yaml", "screen")
    cfg = load_config("configs/exp133_scalar_token_transformer_oof_te.yaml", "screen")
    assert cfg.model == baseline.model
    assert cfg.features.providers[:-1] == baseline.features.providers
    assert cfg.features.base == baseline.features.base
    assert cfg.features.categorical == baseline.features.categorical
    assert cfg.features.exclude == baseline.features.exclude

    plan = FeaturePlan.from_config(cfg.features)
    transformer = plan.fold_fit_transformers()[-1]

    assert transformer.inner_folds == 10
    assert transformer.smoothing == 10
    assert transformer.columns() == [
        "age_te",
        "daily_screen_time_hours_te",
        "social_media_hours_te",
        "gaming_hours_te",
        "work_study_hours_te",
        "sleep_hours_te",
        "notifications_per_day_te",
        "app_opens_per_day_te",
        "weekend_screen_time_te",
        "gender_te",
        "stress_level_te",
        "academic_work_impact_te",
        "placebo_noise_te",
    ]


@pytest.mark.parametrize("mixing", ["attention", "token_mlp"])
def test_adapter_contract_and_learning(mixing: str):
    X, y = _data()
    adapter = _adapter(mixing)
    validation = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    assert validation.shape == (80,)
    assert np.isfinite(validation).all()
    assert ((validation >= 0) & (validation <= 1)).all()
    assert roc_auc_score(y.iloc[240:], validation) > 0.8

    prediction = adapter.predict(X.iloc[:12])
    assert prediction.shape == (12,)
    assert np.isfinite(prediction).all()

    importance = adapter.importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert list(importance["feature"]) == list(X.columns)
    assert np.isfinite(importance["gain"]).all()
    pd.testing.assert_frame_equal(importance, adapter.importance())

    diagnostics = adapter.entry_diagnostics()
    assert diagnostics.assertions == {
        "preprocessing_training_rows_only": True,
        "validation_labels_excluded_from_preprocessing": True,
        "missing_and_unknown_categories_distinct": True,
        "attention_ablation_parameter_matched": True,
    }
    assert diagnostics.observations["mixing"] == mixing
    assert diagnostics.observations["feature_count"] == 3
    assert diagnostics.observations["target_encodings"] == 0
    assert diagnostics.observations["trainable_parameters"] > 0


def test_diagnostics_count_target_encoding_columns():
    X, y = _data()
    X = X.rename(columns={"value": "value_te"})
    adapter = _adapter()
    adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    assert adapter.entry_diagnostics().observations["target_encodings"] == 1


def test_training_only_category_mapping_distinguishes_unknown_and_missing():
    X, y = _data()
    adapter = _adapter()
    adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    probe = pd.DataFrame(
        {
            "value": [0.0, 0.0],
            "category": ["NeverSeen", None],
            "noise": [0.0, 0.0],
        }
    )
    encoded = adapter._impl._raw_scalars(probe)
    category_index = list(probe.columns).index("category")
    assert encoded[0, category_index] != encoded[1, category_index]
    assert encoded[0, category_index] == 1.0
    assert encoded[1, category_index] == 0.0
    assert np.isfinite(adapter.predict(probe)).all()


def test_attention_ablation_keeps_parameter_scale_close():
    from pipeline import scalar_token_transformer as scalar_mod

    attention = scalar_mod._AttentionBlock(64, 4, 256, 0.1)
    token_mlp = scalar_mod._TokenMLPBlock(64, 256, 0.1)
    attention_count = sum(parameter.numel() for parameter in attention.parameters())
    token_mlp_count = sum(parameter.numel() for parameter in token_mlp.parameters())

    assert attention_count == scalar_mod._attention_block_parameters(64, 256)
    assert token_mlp_count == scalar_mod._token_mlp_block_parameters(64, 256)
    assert abs(attention_count - token_mlp_count) / attention_count < 0.05


@pytest.mark.parametrize("mixing", ["unknown", "transformer", "mlp"])
def test_rejects_unknown_mixing(mixing: str):
    cfg = ModelConfig(
        kind="scalar_token_transformer",
        params=dict(SMALL_PARAMS, mixing=mixing),
        fit={},
    )
    with pytest.raises(ValueError, match="mixing"):
        model_mod.create(cfg, seed=SEED)


def test_rejects_unknown_params_and_fit_settings():
    X, y = _data(100)
    cfg = ModelConfig(
        kind="scalar_token_transformer",
        params=dict(SMALL_PARAMS, no_such_param=1),
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    with pytest.raises(ValueError, match="no_such_param"):
        adapter.fit(X.iloc[:80], y.iloc[:80], X.iloc[80:], y.iloc[80:])

    cfg = ModelConfig(
        kind="scalar_token_transformer",
        params=dict(SMALL_PARAMS),
        fit={"no_such_fit_setting": 1},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    with pytest.raises(ValueError, match="fit 설정"):
        adapter.fit(X.iloc[:80], y.iloc[:80], X.iloc[80:], y.iloc[80:])


def test_rejects_initial_score():
    X, y = _data(100)
    adapter = _adapter()
    with pytest.raises(ValueError, match="초기 점수"):
        adapter.fit(
            X.iloc[:80],
            y.iloc[:80],
            X.iloc[80:],
            y.iloc[80:],
            pd.Series(np.zeros(80)),
            pd.Series(np.zeros(20)),
        )
