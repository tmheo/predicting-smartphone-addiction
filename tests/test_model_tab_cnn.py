"""표 합성곱망 M0와 매개변수 규모를 맞춘 제거 대조 시험. (#177)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig, load_config
from pipeline.plan import FeaturePlan


def _params(**overrides) -> dict:
    params = {
        "interaction_mode": "convolution",
        "plr_bins": 4,
        "periodic_dim": 4,
        "periodic_sigma": 1.0,
        "dropout": 0.0,
        "head_samples": 2,
        "head_dropout": 0.0,
        "epochs": 10,
        "patience": 4,
        "batch_size": 32,
        "eval_batch_size": 128,
        "lr": 0.005,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "scheduler_t0": 5,
        "n_quantiles": 64,
        "perm_sample": 64,
        "perm_repeats": 1,
        "device": "cpu",
    }
    params.update(overrides)
    return params


def _adapter(**overrides) -> model_mod.TabCNNAdapter:
    adapter = model_mod.create(
        ModelConfig(kind="tab_cnn", params=_params(**overrides), fit={}), seed=7
    )
    assert isinstance(adapter, model_mod.TabCNNAdapter)
    return adapter


def _data(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(177)
    signal = rng.normal(size=n)
    category = pd.Series(np.where(signal > 0, "high", "low"), dtype="object")
    category.iloc[::23] = None
    numeric = pd.Series(signal).where(np.arange(n) % 19 != 0)
    X = pd.DataFrame(
        {
            "signal": numeric,
            "noise": rng.normal(size=n),
            "category": pd.Categorical(category, categories=["high", "low"]),
            "placebo_noise": rng.normal(size=n),
        },
        index=np.arange(5000, 5000 + n),
    )
    y = pd.Series(
        ((signal + rng.normal(scale=0.35, size=n)) > 0).astype("int64"),
        index=X.index,
    )
    return X, y


def test_m0_and_a0_keep_exp081_feature_plan_and_33_columns():
    baseline = load_config("configs/exp081_lookup_fold_initialization_avg3.yaml", "screen")
    m0 = load_config("configs/exp113_tab_cnn_m0.yaml", "screen")
    a0 = load_config("configs/exp114_tab_cnn_dense_a0.yaml", "screen")
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


def test_convolution_and_dense_ablation_have_matching_parameter_scale():
    from pipeline import tab_cnn

    convolution = tab_cnn._TabCNN(
        33,
        "convolution",
        plr_bins=12,
        periodic_dim=12,
        periodic_sigma=2.33,
        dropout=0.08,
        head_samples=5,
        head_dropout=0.2,
    )
    dense = tab_cnn._TabCNN(
        33,
        "dense",
        plr_bins=12,
        periodic_dim=12,
        periodic_sigma=2.33,
        dropout=0.08,
        head_samples=5,
        head_dropout=0.2,
    )
    assert any(isinstance(module, tab_cnn.nn.Conv1d) for module in convolution.modules())
    assert not any(isinstance(module, tab_cnn.nn.Conv1d) for module in dense.modules())
    relative_difference = abs(
        convolution.interaction_parameters - dense.interaction_parameters
    ) / convolution.interaction_parameters
    assert relative_difference < 0.05


def test_adapter_contract_learning_importance_and_determinism():
    X, y = _data()
    first = _adapter()
    first_prediction = first.fit(
        X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:]
    )
    second = _adapter()
    second_prediction = second.fit(
        X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:]
    )

    np.testing.assert_array_equal(first_prediction, second_prediction)
    assert first_prediction.shape == (80,)
    assert np.isfinite(first_prediction).all()
    assert roc_auc_score(y.iloc[240:], first_prediction) > 0.8
    assert np.isfinite(first.predict(X.iloc[240:250])).all()

    importance = first.importance()
    assert importance["feature"].tolist() == list(X.columns)
    assert np.isfinite(importance["gain"]).all()
    pd.testing.assert_frame_equal(importance, first.importance())

    diagnostics = first.entry_diagnostics()
    assert all(diagnostics.assertions.values())
    assert diagnostics.observations["interaction_mode"] == "convolution"
    assert diagnostics.observations["input_feature_count"] == 4
    assert diagnostics.observations["best_validation_auc"] > 0.8
    assert diagnostics.observations["source_script_version_id"] == 342747549

    training = first.training_diagnostics()
    assert all(training["integrity_assertions"].values())
    assert training["preprocessing_fit_rows"] == 240
    assert training["training_rows"] == 240
    assert training["validation_rows"] == 80
    assert training["prediction_calls"] > 0
    assert training["all_predictions_finite"] is True
    assert training["importance_values_finite"] is True
    assert np.isfinite(training["placebo_importance"])
    assert training["fit_seconds"] > 0
    assert training["importance_seconds"] > 0
    assert training["fold_adapter_seconds"] > training["fit_seconds"]
    assert training["cuda_max_allocated_bytes"] is None
    assert training["cuda_max_reserved_bytes"] is None
    assert training["cuda_device_total_bytes"] is None


def test_fold_encoder_distinguishes_unknown_and_missing_categories():
    from pipeline import tab_cnn

    X, _ = _data(80)
    encoder = tab_cnn._FoldQuantileEncoder(32, seed=7)
    encoder.fit(X.iloc[:60])
    probe = X.iloc[:2].copy()
    probe["category"] = pd.Categorical(
        ["never-seen", None], categories=["high", "low", "never-seen"]
    )
    raw = encoder.raw_transform(probe)
    category_index = list(X.columns).index("category")
    assert raw[0, category_index] != raw[1, category_index]
    assert raw[1, category_index] == 0
    assert np.isfinite(encoder.transform(probe).numpy()).all()


def test_dense_ablation_runs_finite_forward_and_backward():
    from pipeline import tab_cnn

    network = tab_cnn._TabCNN(
        4,
        "dense",
        plr_bins=4,
        periodic_dim=4,
        periodic_sigma=1.0,
        dropout=0.0,
        head_samples=2,
        head_dropout=0.0,
    )
    output = network(tab_cnn.torch.randn(32, 4))
    loss = output.square().mean()
    loss.backward()
    assert output.shape == (32,)
    assert tab_cnn.torch.isfinite(output).all()
    assert all(
        parameter.grad is None or tab_cnn.torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"interaction_mode": "attention"}, "interaction_mode"),
        ({"periodic_dim": 3}, "짝수"),
        ({"unknown": 1}, "모르는 params"),
    ],
)
def test_rejects_invalid_or_unknown_params(overrides, message):
    X, y = _data(80)
    adapter = _adapter(**overrides)
    with pytest.raises(ValueError, match=message):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])


def test_rejects_unknown_fit_settings_and_initial_score():
    X, y = _data(80)
    adapter = model_mod.create(
        ModelConfig(kind="tab_cnn", params=_params(), fit={"unknown": 1}), seed=7
    )
    with pytest.raises(ValueError, match="fit 설정"):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])

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
