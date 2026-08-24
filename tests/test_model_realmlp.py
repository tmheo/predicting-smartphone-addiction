"""고정 일정 RealMLP adapter 회귀 검사. (#180)

- 공개 계약의 핵심 설정과 원시 입력 경계를 고정한다.
- outer 학습 부분에서만 전처리와 내부 OOF 목표 인코딩을 맞춘다.
- fold 안 두 초기화 평균, 고정 epoch, 예측과 순열 중요도 계약을 확인한다.
- 검증 목표값은 학습이나 모형 선택에 영향을 주지 않아야 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import model as model_mod
from pipeline.config import ModelConfig, load_config
from pipeline.training_length import (
    FIXED_COUNT,
    observe_declaration,
)


REPO = Path(__file__).resolve().parents[1]
SEED = 42
RAW_DECIMAL_COLUMNS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "weekend_screen_time",
]
SMALL_PARAMS = {
    "n_ens": 1,
    "embed_dim": 2,
    "onehot_thresh": 4,
    "hidden_dims": [16],
    "dropout": 0.0,
    "pbld_hidden_dim": 4,
    "pbld_out_dim": 2,
    "fixed_epochs": 1,
    "schedule_epochs": 2,
    "batch_size": 64,
    "eval_batch_size": 128,
    "n_init_avg": 2,
    "inner_folds": 3,
    "perm_sample": 48,
    "perm_repeats": 1,
    "device": "cpu",
    "verbosity": 0,
}


def _data(n: int = 180) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(180)
    daily = rng.uniform(1.0, 12.0, n).round(1)
    social = rng.uniform(0.0, 7.0, n).round(1)
    frame = pd.DataFrame(
        {
            "age": rng.integers(13, 60, n).astype(float),
            "daily_screen_time_hours": daily,
            "social_media_hours": social,
            "gaming_hours": rng.uniform(0.0, 6.0, n).round(1),
            "work_study_hours": rng.uniform(0.0, 10.0, n).round(1),
            "sleep_hours": rng.uniform(4.0, 10.0, n).round(1),
            "notifications_per_day": rng.integers(0, 400, n).astype(float),
            "app_opens_per_day": rng.integers(0, 250, n).astype(float),
            "weekend_screen_time": rng.uniform(1.0, 16.0, n).round(1),
            "gender": pd.Categorical(np.resize(["Male", "Female", "Other"], n)),
            "stress_level": pd.Categorical(np.resize(["Low", "Medium", "High"], n)),
            "academic_work_impact": pd.Categorical(np.resize(["Low", "High"], n)),
            "placebo_noise": rng.normal(size=n),
        }
    )
    frame.loc[::13, "social_media_hours"] = np.nan
    frame.loc[::17, "sleep_hours"] = np.nan
    logit = daily + 0.8 * social + rng.normal(scale=1.0, size=n)
    target = pd.Series((logit >= np.median(logit)).astype("int64"))
    return frame, target


def _adapter(**overrides) -> model_mod.RealMLPAdapter:
    cfg = ModelConfig(
        kind="realmlp", params=dict(SMALL_PARAMS, **overrides), fit={}
    )
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.RealMLPAdapter)
    return adapter


def test_realmlp_public_contract_config():
    cfg = load_config(
        REPO / "configs" / "exp121_realmlp_fixed4_two_init.yaml", "screen"
    )
    params = cfg.model.params
    assert cfg.model.kind == "realmlp"
    assert cfg.features.providers == []
    assert params["hidden_dims"] == [512, 512, 512]
    assert params["pbld_hidden_dim"] == 20
    assert params["pbld_out_dim"] == 5
    assert params["n_ens"] == 8
    assert params["fixed_epochs"] == 4
    assert params["schedule_epochs"] == 8
    assert params["batch_size"] == 256
    assert params["inner_folds"] == 5
    assert params["n_init_avg"] == 2


def test_realmlp_muon_config_is_exp124_single_optimizer_delta():
    baseline = load_config(REPO / "configs" / "exp124_realmlp_dtype_fix.yaml", "screen")
    challenger = load_config(REPO / "configs" / "exp134_realmlp_muon.yaml", "screen")

    assert challenger.name == "exp134_realmlp_muon"
    assert challenger.data == baseline.data
    assert challenger.features == baseline.features
    assert challenger.model.kind == baseline.model.kind
    assert challenger.model.fit == baseline.model.fit
    assert challenger.model.params == {**baseline.model.params, "optimizer": "muon"}


def test_realmlp_orig_cdf_diff_config_is_exp124_feature_only_delta(monkeypatch):
    proxy_columns = [
        "daily_screen_time_hours",
        "weekend_screen_time",
        "social_media_hours",
        "notifications_per_day",
        "app_opens_per_day",
    ]
    proxy = pd.DataFrame(
        {
            **{column: [0.0, 1.0] for column in proxy_columns},
            "addicted_label": [0, 1],
        }
    )
    monkeypatch.setattr(
        "pipeline.features._load_locked_proxy",
        lambda path, sha256, cols: proxy,
    )

    baseline = load_config(REPO / "configs" / "exp124_realmlp_dtype_fix.yaml", "screen")
    challenger = load_config(
        REPO / "configs" / "exp140_realmlp_orig_cdf_diff.yaml", "screen"
    )

    assert challenger.name == "exp140_realmlp_orig_cdf_diff"
    assert challenger.data == baseline.data
    assert challenger.features.base == baseline.features.base
    assert challenger.features.categorical == baseline.features.categorical
    assert challenger.features.exclude == baseline.features.exclude
    assert challenger.features.providers == [
        {
            "kind": "original_cdf_diff",
            "path": "data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv",
            "cols": proxy_columns,
        }
    ]
    assert challenger.model == baseline.model


def test_realmlp_exact_frequency_config_is_exp124_feature_only_delta():
    from pipeline.features import FrequencyEncoder
    from pipeline.plan import FeaturePlan

    columns = [
        "age",
        "daily_screen_time_hours",
        "social_media_hours",
        "work_study_hours",
        "sleep_hours",
        "notifications_per_day",
        "app_opens_per_day",
        "weekend_screen_time",
    ]
    baseline = load_config(REPO / "configs" / "exp124_realmlp_dtype_fix.yaml", "screen")
    challenger = load_config(
        REPO / "configs" / "exp141_realmlp_exact_frequency.yaml", "screen"
    )

    assert challenger.name == "exp141_realmlp_exact_frequency"
    assert challenger.data == baseline.data
    assert challenger.features.base == baseline.features.base
    assert challenger.features.categorical == baseline.features.categorical
    assert challenger.features.exclude == baseline.features.exclude
    assert challenger.features.providers == [
        {"kind": "frequency_encoding", "cols": columns}
    ]
    assert challenger.model == baseline.model

    providers = FeaturePlan.from_config(challenger.features).fold_fit_transformers()
    assert len(providers) == 1
    assert isinstance(providers[0], FrequencyEncoder)
    assert providers[0].columns() == [f"{column}_ce" for column in columns]


def test_realmlp_muon_recon_widths_config_is_exp134_feature_only_delta():
    from pipeline.features import ConstrainedImputeAux
    from pipeline.plan import FeaturePlan

    baseline = load_config(REPO / "configs" / "exp134_realmlp_muon.yaml", "screen")
    challenger = load_config(
        REPO / "configs" / "exp136_realmlp_muon_recon_widths.yaml", "screen"
    )

    assert challenger.name == "exp136_realmlp_muon_recon_widths"
    assert challenger.data == baseline.data
    assert challenger.features.base == baseline.features.base
    assert challenger.features.categorical == baseline.features.categorical
    assert challenger.features.exclude == baseline.features.exclude
    assert challenger.model == baseline.model

    plan = FeaturePlan.from_config(challenger.features)
    providers = plan.fold_fit_transformers()
    assert len(providers) == 1
    assert isinstance(providers[0], ConstrainedImputeAux)
    assert providers[0].columns() == [
        "gaming_hours_recon_width",
        "social_media_hours_recon_width",
        "work_study_hours_recon_width",
    ]


def test_realmlp_reference_qnormal_configs_change_only_the_declared_scope():
    from pipeline.realmlp import RAW_NUMERICAL

    baseline = load_config(
        REPO / "configs" / "exp136_realmlp_muon_recon_widths.yaml", "screen"
    )
    control = load_config(
        REPO / "configs" / "exp138_realmlp_reference_qnormal_fold_train.yaml",
        "screen",
    )
    candidate = load_config(
        REPO / "configs" / "exp139_realmlp_reference_qnormal_train_test.yaml",
        "screen",
    )

    assert control.data == baseline.data == candidate.data
    assert control.features == baseline.features == candidate.features
    assert control.model.kind == baseline.model.kind == candidate.model.kind
    assert control.model.fit == baseline.model.fit == candidate.model.fit
    expected = {
        **baseline.model.params,
        "reference_qnormal_columns": RAW_NUMERICAL,
        "preprocessing_scope": "fold_train",
    }
    assert control.model.params == expected
    assert candidate.model.params == {**expected, "preprocessing_scope": "train_test"}


def test_reference_qnormal_uses_only_declared_reference_and_keeps_fold_state():
    from sklearn.preprocessing import QuantileTransformer

    from pipeline.realmlp import (
        RAW_NUMERICAL,
        REFERENCE_QNORMAL_SUFFIX,
        _FoldFeatureEngineer,
    )

    X, _ = _data(120)
    train = X.iloc[:90].copy()
    extra = X.iloc[90:].copy()
    extra["age"] = extra["age"] + 1000.0
    reference = pd.concat([X, extra], ignore_index=True)

    control = _FoldFeatureEngineer(RAW_NUMERICAL, reference_seed=SEED)
    candidate = _FoldFeatureEngineer(RAW_NUMERICAL, reference_seed=SEED)
    control_values = control.fit_transform(train, train[RAW_NUMERICAL])
    candidate_values = candidate.fit_transform(train, reference[RAW_NUMERICAL])

    assert control.fit_rows == candidate.fit_rows == 90
    assert control.medians == candidate.medians
    assert control.category_maps == candidate.category_maps
    assert control.reference_qnormal_reference_rows == 90
    assert candidate.reference_qnormal_reference_rows == len(reference)
    expected_fit_rows = {
        column: int(reference[column].notna().sum()) for column in RAW_NUMERICAL
    }
    assert candidate.reference_qnormal_fit_rows == expected_fit_rows
    assert not np.array_equal(
        control_values[f"age{REFERENCE_QNORMAL_SUFFIX}"].to_numpy(),
        candidate_values[f"age{REFERENCE_QNORMAL_SUFFIX}"].to_numpy(),
    )

    observed = reference["age"].dropna().to_numpy(dtype="float64")
    expected = QuantileTransformer(
        n_quantiles=min(1000, len(observed)),
        output_distribution="normal",
        subsample=2_000_000_000,
        random_state=SEED,
    ).fit(observed.reshape(-1, 1))
    expected_train = expected.transform(
        train["age"].to_numpy(dtype="float64").reshape(-1, 1)
    ).ravel()
    assert np.allclose(
        candidate_values[f"age{REFERENCE_QNORMAL_SUFFIX}"],
        expected_train.astype("float32"),
    )
    missing = train["social_media_hours"].isna()
    assert (
        candidate_values.loc[
            missing, f"social_media_hours{REFERENCE_QNORMAL_SUFFIX}"
        ]
        == 0.0
    ).all()
    assert not set(candidate.reference_qnormal_output_columns) & set(
        candidate.output_cat_cols
    )


def test_realmlp_train_test_reference_scope_is_target_free_and_diagnostic():
    from pipeline.realmlp import RAW_NUMERICAL, REFERENCE_QNORMAL_SUFFIX

    X, y = _data(120)
    adapter = _adapter(
        perm_sample=24,
        reference_qnormal_columns=RAW_NUMERICAL,
        preprocessing_scope="train_test",
    )
    reference_test = X.iloc[:18].copy()
    model_mod.set_dataset_reference(adapter, X, reference_test)
    prediction = adapter.fit(
        X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:]
    )
    assert np.isfinite(prediction).all()

    diagnostics = adapter.entry_diagnostics()
    assert all(diagnostics.assertions.values())
    observations = diagnostics.observations
    assert observations["preprocessing_fit_rows"] == 90
    assert observations["target_encoding_fit_rows"] == 90
    assert observations["reference_qnormal_reference_rows"] == 138
    assert observations["dataset_reference_train_rows"] == 120
    assert observations["dataset_reference_test_rows"] == 18
    assert observations["engineered_feature_count"] == 63
    assert observations["reference_qnormal_columns"] == [
        f"{column}{REFERENCE_QNORMAL_SUFFIX}" for column in RAW_NUMERICAL
    ]

    with_target = X.assign(addicted_label=y.to_numpy())
    rejected = _adapter(
        n_init_avg=1,
        reference_qnormal_columns=RAW_NUMERICAL,
        preprocessing_scope="train_test",
    )
    with pytest.raises(ValueError, match="목표값"):
        model_mod.set_dataset_reference(
            rejected, with_target, with_target.iloc[:18].copy()
        )


def test_realmlp_fold_preprocessing_is_train_only_and_has_54_features():
    from pipeline.realmlp import _FoldFeatureEngineer, _FoldTargetEncoder

    X, y = _data(120)
    train = X.iloc[:90].copy()
    validation = X.iloc[90:].copy()
    validation["gender"] = validation["gender"].astype("string")
    validation.loc[validation.index[0], "gender"] = "Unseen"
    validation["gender"] = pd.Categorical(validation["gender"])

    engineer = _FoldFeatureEngineer()
    transformed_train = engineer.fit_transform(train)
    transformed_validation = engineer.transform(validation)
    encoder = _FoldTargetEncoder(inner_folds=3, seed=SEED)
    encoded_train = encoder.fit_transform(
        transformed_train, y.iloc[:90], engineer.output_cat_cols
    )
    encoded_validation = encoder.transform(transformed_validation)

    assert engineer.fit_rows == 90
    assert encoder.fit_rows == 90
    assert len(engineer.output_cat_cols) == 23
    assert len(encoder.output_names) == 21
    assert encoded_train.shape == (90, 54)
    assert encoded_validation.shape == (30, 54)
    assert transformed_validation.loc[validation.index[0], "gender"] == 0
    assert np.isfinite(encoded_train.to_numpy(dtype="float64")).all()
    assert np.isfinite(encoded_validation.to_numpy(dtype="float64")).all()


def test_realmlp_numeric_category_mapping_keeps_fit_dtype():
    """소수 값 열의 정확값 범주가 형 변환으로 죽지 않아야 한다. (#243)"""
    from pipeline.realmlp import _FoldFeatureEngineer

    X, _ = _data(120)
    train = X.iloc[:90].copy()
    validation = X.iloc[90:].copy()

    engineer = _FoldFeatureEngineer()
    transformed_train = engineer.fit_transform(train)
    transformed_validation = engineer.transform(validation)

    # 학습 부분의 값은 전부 어휘에 있으므로 unknown이 하나도 없어야 한다.
    assert engineer.unknown_value_count(transformed_train) == 0
    # 검증 부분의 unknown은 어휘 밖 신규 값에서만 나와야 한다.
    for column in RAW_DECIMAL_COLUMNS:
        name = f"{column}_cat_"
        fitted_values = pd.to_numeric(train[column], errors="coerce").fillna(
            engineer.medians[column]
        )
        seen = validation[column].isin(set(fitted_values.tolist())).to_numpy()
        assert (transformed_validation[name].to_numpy()[seen] != 0).all()
    # 수치 열의 최종 출력 dtype 계약은 float32로 유지된다.
    assert transformed_train["daily_screen_time_hours"].dtype == np.float32


def test_realmlp_adapter_contract_and_diagnostics():
    import os

    X, y = _data()
    adapter = _adapter()
    validation_prediction = adapter.fit(
        X.iloc[:135], y.iloc[:135], X.iloc[135:], y.iloc[135:]
    )
    assert validation_prediction.shape == (45,)
    assert ((validation_prediction >= 0) & (validation_prediction <= 1)).all()
    assert np.isfinite(adapter.predict(X.iloc[:8])).all()

    diagnostics = adapter.entry_diagnostics()
    assert diagnostics.assertions
    assert all(diagnostics.assertions.values())
    observations = diagnostics.observations
    assert observations["engineered_feature_count"] == 54
    assert observations["fold_initialization_average_count"] == 2
    assert observations["validation_selection"] == "final_fixed_epoch"
    assert observations["pytabkit_estimator_used"] is False
    assert observations["optimizer"] == "adamw"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}
    assert observations["cublas_workspace_config"] == os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ]

    importance = adapter.importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert list(importance["feature"]) == list(X.columns)
    assert np.isfinite(importance["gain"]).all()
    training = adapter.training_diagnostics()
    assert training["all_predictions_finite"] is True
    assert np.isfinite(training["placebo_importance"])


def test_realmlp_validation_labels_do_not_change_predictions():
    X, y = _data(120)
    first = _adapter(n_init_avg=1, perm_sample=24)
    second = _adapter(n_init_avg=1, perm_sample=24)
    prediction = first.fit(X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:])
    flipped = second.fit(
        X.iloc[:90], y.iloc[:90], X.iloc[90:], 1 - y.iloc[90:]
    )
    assert np.array_equal(prediction, flipped)


def test_realmlp_full_fit_and_unknown_settings():
    X, y = _data(96)
    adapter = _adapter(n_init_avg=1)
    model_mod.fit_full(adapter, X, y, 1)
    assert np.isfinite(adapter.predict(X.iloc[:8])).all()
    assert adapter.training_diagnostics()["full_fit"] is True

    bad = _adapter(no_such_param=1)
    with pytest.raises(ValueError, match="no_such_param"):
        bad.fit(X.iloc[:72], y.iloc[:72], X.iloc[72:], y.iloc[72:])


def test_realmlp_muon_splits_internal_ensemble_hidden_matrices():
    from pipeline.realmlp import _DEFAULTS, _RealMLP, _muon_parameters

    config = {**_DEFAULTS, **SMALL_PARAMS, "n_ens": 3, "optimizer": "muon"}
    model = _RealMLP([3, 4], numerical_features=5, config=config)
    selected = _muon_parameters(model)

    assert len(selected) == 3 * len(config["hidden_dims"])
    assert all(parameter.ndim == 2 for parameter in selected)
    selected_ids = {id(parameter) for parameter in selected}
    assert all(
        id(parameter) not in selected_ids for parameter in model.output.parameters()
    )
    assert all(
        id(parameter) not in selected_ids
        for parameter in model.categorical.parameters()
    )
    assert all(
        id(parameter) not in selected_ids for parameter in model.numerical.parameters()
    )


def test_realmlp_split_hidden_matrix_keeps_initialization_and_forward_values():
    import torch

    from pipeline.realmlp import _NTPLinear

    torch.manual_seed(123)
    dense = _NTPLinear(3, 5, 4)
    torch.manual_seed(123)
    split = _NTPLinear(3, 5, 4, split_weight=True)
    values = torch.randn(7, 3, 5)

    assert torch.equal(dense.weight, torch.stack(list(split.weights)))
    assert torch.equal(dense.bias, split.bias)
    assert torch.equal(dense(values), split(values))


def test_realmlp_muon_contract_and_learning():
    X, y = _data(120)
    adapter = _adapter(optimizer="muon", n_init_avg=1, perm_sample=24)

    prediction = adapter.fit(X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:])

    assert prediction.shape == (30,)
    assert np.isfinite(prediction).all()
    assert adapter.training_diagnostics()["optimizer"] == "muon"


def test_realmlp_rejects_unknown_optimizer():
    X, y = _data(80)
    with pytest.raises(ValueError, match="optimizer"):
        _adapter(optimizer="sgd").fit(
            X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:]
        )


def test_realmlp_declares_fixed_count_evidence():
    """고정 일정도 숫자를 예산으로 바로 쓰지 않고 고정 횟수 근거로 남긴다. (#372)"""
    X, y = _data(96)
    adapter = _adapter(n_init_avg=1)
    adapter.fit(X.iloc[:72], y.iloc[:72], X.iloc[72:], y.iloc[72:])

    declaration = adapter.training_length_evidence()
    fixed_epochs = adapter.training_diagnostics()["fixed_epochs"]
    assert declaration.model_family == "realmlp"
    assert declaration.raw_field == "fixed_epochs"
    assert declaration.raw_meaning == FIXED_COUNT
    assert [item.raw_value for item in declaration.selections] == [fixed_epochs]
    assert [item.inner_member for item in declaration.selections] == [None]

    evidence = observe_declaration(declaration, seed=SEED, outer_fold=0)
    assert [item.value for item in evidence.observations] == [fixed_epochs]


def test_realmlp_full_fit_declares_no_training_length_evidence():
    X, y = _data(96)
    adapter = _adapter(n_init_avg=1)

    model_mod.fit_full(adapter, X, y, 1)

    with pytest.raises(RuntimeError, match="검증 분할"):
        adapter.training_length_evidence()
