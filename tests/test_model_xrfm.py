"""xRFM 재귀 특성 커널 머신 adapter 시험. (#198)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig, load_config


def _params(**overrides) -> dict:
    params = {
        "device": "cpu",
        "max_leaf_size": 2000,
        "inner_val_frac": 0.25,
        "eval_batch_size": 128,
        "perm_sample": 64,
        "perm_repeats": 1,
        "verbose": False,
    }
    params.update(overrides)
    return params


def _adapter(**overrides) -> model_mod.XRFMAdapter:
    adapter = model_mod.create(
        ModelConfig(kind="xrfm", params=_params(**overrides), fit={}), seed=7
    )
    assert isinstance(adapter, model_mod.XRFMAdapter)
    return adapter


def _data(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(198)
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


def test_exp123_keeps_exp081_feature_plan():
    baseline = load_config("configs/exp081_lookup_fold_initialization_avg3.yaml", "screen")
    challenger = load_config("configs/exp123_xrfm.yaml", "screen")
    assert challenger.features == baseline.features
    assert challenger.model.kind == "xrfm"
    assert challenger.model.fit == {}


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
    assert (first_prediction >= 0).all() and (first_prediction <= 1).all()
    assert roc_auc_score(y.iloc[240:], first_prediction) > 0.8
    assert np.isfinite(first.predict(X.iloc[240:250])).all()

    importance = first.importance()
    assert importance["feature"].tolist() == list(X.columns)
    assert np.isfinite(importance["gain"]).all()
    pd.testing.assert_frame_equal(importance, first.importance())

    diagnostics = first.entry_diagnostics()
    assert all(diagnostics.assertions.values())
    assert diagnostics.observations["training_rows"] == 240
    assert diagnostics.observations["validation_rows"] == 80
    assert diagnostics.observations["inner_train_rows"] == 180
    assert diagnostics.observations["inner_val_rows"] == 60
    assert diagnostics.observations["package_version"] == "0.4.5"
    assert diagnostics.observations["timing_seconds"]["fit"] > 0
    # 숫자 3열 x (값 + 결측 지시) + 범주 1열 (어휘 2 + unknown + missing)
    assert diagnostics.observations["encoded_feature_count"] == 10


def test_fold_encoder_distinguishes_unknown_and_missing_categories():
    from pipeline import xrfm_fold

    X, _ = _data(80)
    encoder = xrfm_fold._FoldKernelEncoder()
    encoder.fit(X.iloc[:60])
    probe = X.iloc[:2].copy()
    probe["category"] = pd.Categorical(
        ["never-seen", None], categories=["high", "low", "never-seen"]
    )
    encoded = encoder.transform(probe)
    unknown_index = encoder.encoded_names.index("category__unknown")
    missing_index = encoder.encoded_names.index("category__missing")
    assert encoded[0, unknown_index] == 1.0 and encoded[0, missing_index] == 0.0
    assert encoded[1, unknown_index] == 0.0 and encoded[1, missing_index] == 1.0
    assert np.isfinite(encoded).all()


def test_fold_encoder_imputes_and_flags_missing_numeric():
    from pipeline import xrfm_fold

    X, _ = _data(80)
    encoder = xrfm_fold._FoldKernelEncoder()
    encoder.fit(X.iloc[:60])
    encoded = encoder.transform(X.iloc[:60])
    signal_missing_index = encoder.encoded_names.index("signal__missing")
    expected_missing = X["signal"].iloc[:60].isna().to_numpy()
    np.testing.assert_array_equal(
        encoded[:, signal_missing_index] == 1.0, expected_missing
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device": "tpu"}, "device"),
        ({"kernel": "rbf"}, "kernel"),
        ({"tuning_metric": "auc"}, "tuning_metric"),
        ({"inner_val_frac": 0.9}, "inner_val_frac"),
        ({"unknown": 1}, "모르는 params"),
        ({"package_version": "0.0.1"}, "패키지"),
    ],
)
def test_rejects_invalid_or_unknown_params(overrides, message):
    from pipeline import xrfm_fold

    with pytest.raises(ValueError, match=message):
        xrfm_fold.XRFMFold(_params(**overrides), seed=7)


def test_rejects_non_binary_labels_and_overlapping_folds():
    X, y = _data(80)
    adapter = _adapter()
    with pytest.raises(ValueError, match="index가 겹친다"):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[50:], y.iloc[50:])
    with pytest.raises(ValueError, match="이진 라벨"):
        adapter.fit(X.iloc[:60], y.iloc[:60] + 1, X.iloc[60:], y.iloc[60:] + 1)


def test_rejects_unknown_fit_settings_and_initial_score():
    X, y = _data(80)
    adapter = model_mod.create(
        ModelConfig(kind="xrfm", params=_params(), fit={"unknown": 1}), seed=7
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
