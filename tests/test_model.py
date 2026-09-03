"""모델 계열 레지스트리와 cv 루프 테스트. (#72 완료 기준)

- 레지스트리 dispatch: kind -> adapter 해석, 미등록 kind는 명확한 오류.
- 결정적 가짜 adapter로 cv 루프 전체 검증: OOF 배치, test 예측 fold 평균,
  fold별 recorder 통지, importance 조립. 실제 LightGBM 학습 없이 돈다.
- lightgbm adapter 소형 데이터 스모크 테스트.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import cv
from pipeline import model as model_mod
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.features import PLACEBO
from pipeline.plan import FeaturePlan
from pipeline.training_length import (
    FIXED_COUNT,
    ONE_BASED_COUNT,
    ZERO_BASED_POSITION,
    RawTrainingLengthSelection,
    TrainingLengthContract,
    TrainingLengthError,
    observe_declaration,
)

SEED = 7
N_FOLDS = 3


class FakeAdapter:
    """고정 예측을 돌려주는 결정적 가짜 모델.

    fold마다 새로 만들어지는 계약을 이용해 팩토리가 fold 순서대로 서로 다른
    fold_value를 주입한다. 검증 예측은 fold_value, test 예측은 fold_value * 10.
    """

    def __init__(self, params: dict, fit: dict, seed: int, fold_value: float) -> None:
        self.params = params
        self.fit_args = fit
        self.seed = seed
        self.fold_value = fold_value
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        self._feature_names = list(X_tr.columns)
        return np.full(len(X_va), self.fold_value)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        return np.full(len(X), self.fold_value * 10)

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self._feature_names, "gain": np.ones(len(self._feature_names))}
        )


class FakeFullAdapter:
    def __init__(self) -> None:
        self.arguments = None

    def fit_full(self, X, y, training_budget, initial_score=None) -> None:
        self.arguments = (X, y, training_budget, initial_score)


class FakeDatasetReferenceAdapter:
    def __init__(self) -> None:
        self.reference = None

    def set_dataset_reference(self, X_train, X_test) -> None:
        self.reference = (X_train, X_test)


class SpyRecorder:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.folds: list[tuple[int, int, float]] = []

    def stage(self, name: str) -> None:
        self.stages.append(name)

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None:
        self.folds.append((seed_index, fold_index, auc))


def fake_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="fake_cv_test",
        data=DataConfig(
            train=Path("unused"),
            test=Path("unused"),
            sample_submission=Path("unused"),
            folds=Path("unused"),
        ),
        features=FeatureConfig(base="raw", categorical=[], providers=[]),
        model=ModelConfig(kind="fake", params={"p": 1}, fit={"f": 2}),
        initial_score=None,
        seeds=[SEED],
        stage="screen",
        source_path=Path("unused"),
    )


def toy_train_test(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": rng.uniform(1, 10, n).round(1),
            # placebo 마스크 원본 열. 결측 패턴 포함.
            "social_media_hours": rng.uniform(0, 5, n).round(1),
            "sleep_hours": rng.uniform(3, 9, n).round(1),
            "addicted_label": np.tile([0, 1], n // 2),
        }
    )
    test = train.drop(columns=["addicted_label"]).copy()
    test["id"] = test["id"] + n
    return train, test


def test_registry_dispatches_kind_to_adapter(monkeypatch):
    created = []

    def factory(params: dict, fit: dict, seed: int) -> FakeAdapter:
        adapter = FakeAdapter(params, fit, seed, fold_value=0.5)
        created.append(adapter)
        return adapter

    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "fake", factory)
    cfg = ModelConfig(kind="fake", params={"p": 1}, fit={"f": 2})
    adapter = model_mod.create(cfg, seed=SEED)
    assert created == [adapter]
    # params·fit·seed가 그대로 adapter에 전달된다. 해석은 adapter 소유다.
    assert (adapter.params, adapter.fit_args, adapter.seed) == (
        {"p": 1},
        {"f": 2},
        SEED,
    )


def test_lightgbm_kind_resolves_to_lightgbm_adapter():
    cfg = ModelConfig(kind="lightgbm", params={}, fit={"early_stopping_rounds": 5})
    # 인스턴스 생성은 lightgbm import 없이 된다(lazy import 유지).
    assert isinstance(model_mod.create(cfg, seed=SEED), model_mod.LightGBMAdapter)


def test_unregistered_kind_fails_with_clear_error():
    cfg = ModelConfig(kind="no_such_model", params={}, fit={})
    with pytest.raises(ValueError, match="알 수 없는 model.kind 'no_such_model'"):
        model_mod.create(cfg, seed=SEED)


def test_run_cv_keeps_public_signature():
    signature = inspect.signature(cv.run_cv)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "cfg",
        "plan",
        "train",
        "test",
        "seed",
        "recorder",
        "recovery",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    )
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in parameters[:5]
    )
    assert [parameter.default for parameter in parameters[5:]] == [None, None]
    assert signature.return_annotation == "CVResult"


def test_fit_full_validates_budget_and_dispatches_optional_contract():
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.Series([0, 1])
    score = pd.Series([-0.2, 0.2])
    adapter = FakeFullAdapter()

    model_mod.fit_full(adapter, X, y, 17, score)

    assert adapter.arguments[0] is X
    assert adapter.arguments[1] is y
    assert adapter.arguments[2] == 17
    assert adapter.arguments[3] is score
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="양의 정수"):
            model_mod.fit_full(adapter, X, y, invalid)
    with pytest.raises(ValueError, match="지원하지 않는다"):
        model_mod.fit_full(object(), X, y, None)


def test_set_dataset_reference_dispatches_optional_contract():
    X_train = pd.DataFrame({"x": [1.0, 2.0]})
    X_test = pd.DataFrame({"x": [3.0]})
    adapter = FakeDatasetReferenceAdapter()

    model_mod.set_dataset_reference(adapter, X_train, X_test)

    assert adapter.reference == (X_train, X_test)
    model_mod.set_dataset_reference(object(), X_train, X_test)
    with pytest.raises(ValueError, match="train/test 열"):
        model_mod.set_dataset_reference(adapter, X_train, pd.DataFrame({"y": [3.0]}))


def test_lightgbm_adapter_full_fit_uses_fixed_budget():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = pd.Series((X["a"] > 0).astype(int))
    adapter = model_mod.LightGBMAdapter(
        {
            "objective": "binary",
            "n_estimators": 999,
            "num_leaves": 7,
            "verbosity": -1,
        },
        {"early_stopping_rounds": 5},
        SEED,
    )

    model_mod.fit_full(adapter, X, y, 9)

    assert adapter._model.n_estimators_ == 9
    assert adapter.predict(X.iloc[:5]).shape == (5,)


def test_run_cv_with_fake_adapter_verifies_loop_wiring(monkeypatch):
    """cv 루프 전체를 실제 학습 없이 검증한다: OOF 배치, test fold 평균,
    recorder 통지, importance 조립."""
    made: list[FakeAdapter] = []

    def factory(params: dict, fit: dict, seed: int) -> FakeAdapter:
        # fold 순서대로 0.1, 0.2, 0.3을 돌려주는 결정적 가짜.
        adapter = FakeAdapter(params, fit, seed, fold_value=0.1 * (len(made) + 1))
        made.append(adapter)
        return adapter

    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "fake", factory)
    cfg = fake_experiment_config()
    plan = FeaturePlan.from_config(cfg.features)
    train, test = toy_train_test()
    train, test = plan.apply_dataset_wide(train, test)
    train["fold"] = np.arange(len(train)) % N_FOLDS
    recorder = SpyRecorder()

    result = cv.run_cv(cfg, plan, train, test, seed=SEED, recorder=recorder)

    # fold마다 adapter 인스턴스를 새로 만든다.
    assert len(made) == N_FOLDS
    assert all(a.seed == SEED for a in made)

    # OOF 배치: fold f의 검증 행 전부가 그 fold adapter의 예측값을 받는다.
    for fold, adapter in enumerate(made):
        fold_mask = result.oof["fold"] == fold
        assert (result.oof.loc[fold_mask, "pred"] == adapter.fold_value).all()

    # test 예측: fold별 predict의 평균.
    expected = np.mean([a.fold_value * 10 for a in made])
    assert result.test_pred["pred"].to_numpy() == pytest.approx(expected)
    assert list(result.oof.columns) == ["id", "fold", "pred"]
    assert list(result.test_pred.columns) == ["id", "pred"]

    # recorder 통지: 단계 전환과 fold 완료(상수 예측이라 AUC 0.5).
    assert recorder.stages == ["feature_build", "training"]
    assert recorder.folds == [(0, fold, pytest.approx(0.5)) for fold in range(N_FOLDS)]

    # importance 조립: fold별 프레임에 fold·seed가 붙어 하나로 합쳐진다.
    assert list(result.importance.columns) == ["feature", "gain", "fold", "seed"]
    assert sorted(result.importance["fold"].unique()) == list(range(N_FOLDS))
    assert set(result.importance["seed"]) == {SEED}
    features_per_fold = set(
        result.importance[result.importance["fold"] == 0]["feature"]
    )
    assert features_per_fold == set(result.feature_names)
    assert (
        PLACEBO in features_per_fold
    )  # placebo 자동 삽입까지 fake 학습 입력에 닿는다.


def test_run_cv_propagates_model_error_before_fold_completion(monkeypatch):
    class FailingAdapter(FakeAdapter):
        def fit(self, *args, **kwargs) -> np.ndarray:
            raise RuntimeError("model-fit-failed")

    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY,
        "fake",
        lambda params, fit, seed: FailingAdapter(params, fit, seed, fold_value=0.5),
    )
    cfg = fake_experiment_config()
    plan = FeaturePlan.from_config(cfg.features)
    train, test = toy_train_test()
    train, test = plan.apply_dataset_wide(train, test)
    train["fold"] = np.arange(len(train)) % N_FOLDS
    recorder = SpyRecorder()

    with pytest.raises(RuntimeError, match="model-fit-failed"):
        cv.run_cv(cfg, plan, train, test, seed=SEED, recorder=recorder)

    assert recorder.stages == ["feature_build", "training"]
    assert recorder.folds == []


def test_lightgbm_adapter_smoke():
    """소형 데이터로 lightgbm adapter의 fit/predict/importance 계약을 확인한다."""
    rng = np.random.default_rng(1)
    n = 240
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    cfg = ModelConfig(
        kind="lightgbm",
        params={
            "objective": "binary",
            "n_estimators": 30,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbosity": -1,
            "max_bin": 31,
            "max_bin_by_feature": {"a": 63},
        },
        fit={"early_stopping_rounds": 5},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    va_pred = adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])
    assert va_pred.shape == (60,)
    assert ((va_pred >= 0) & (va_pred <= 1)).all()

    test_pred = adapter.predict(X.iloc[:10])
    assert test_pred.shape == (10,)

    imp = adapter.importance()
    assert list(imp.columns) == ["feature", "gain"]
    assert list(imp["feature"]) == ["a", "b"]

    repeated = model_mod.create(cfg, seed=SEED)
    repeated_pred = repeated.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])
    np.testing.assert_array_equal(repeated_pred, va_pred)
    pd.testing.assert_frame_equal(repeated.importance(), imp, check_exact=True)


def test_lightgbm_max_bin_by_feature_resolves_names_in_matrix_order():
    params = {
        "max_bin": 1023,
        "max_bin_by_feature": {"screen": 1439, "weekend": 2047},
    }

    resolved = model_mod._resolve_lightgbm_params(
        params, ["age", "screen", "derived", "weekend"]
    )

    assert resolved["max_bin_by_feature"] == [1023, 1439, 1023, 2047]
    assert resolved["deterministic"] is True
    assert params["max_bin_by_feature"] == {"screen": 1439, "weekend": 2047}


def test_lightgbm_explicit_deterministic_setting_is_preserved():
    resolved = model_mod._resolve_lightgbm_params({"deterministic": False}, ["screen"])

    assert resolved["deterministic"] is False


def test_lightgbm_max_bin_by_feature_rejects_unknown_feature():
    with pytest.raises(ValueError, match="학습 행렬에 없는 열.*typo"):
        model_mod._resolve_lightgbm_params(
            {"max_bin": 1023, "max_bin_by_feature": {"typo": 1439}}, ["screen"]
        )


@pytest.mark.parametrize("value", [1, 1.5, True])
def test_lightgbm_max_bin_by_feature_rejects_invalid_bin_count(value):
    with pytest.raises(ValueError, match="2 이상의 정수"):
        model_mod._resolve_lightgbm_params(
            {"max_bin": 1023, "max_bin_by_feature": {"screen": value}}, ["screen"]
        )


def _smoke_data() -> tuple[pd.DataFrame, pd.Series]:
    """수치 2열 + 결측 있는 category 1열. 세 adapter의 native categorical 경로를 지난다."""
    rng = np.random.default_rng(1)
    n = 240
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    cat = pd.Series(rng.choice(["Low", "High"], size=n)).where(
        rng.uniform(size=n) > 0.1
    )
    X["c"] = pd.Categorical(cat, categories=["High", "Low"])
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return X, y


def _assert_adapter_contract(
    adapter: object,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    supports_initial_score: bool = False,
) -> None:
    """fit/predict/importance 계약: 확률 범위, 예측 shape, (feature, gain) 프레임."""
    va_pred = adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])
    assert va_pred.shape == (60,)
    assert ((va_pred >= 0) & (va_pred <= 1)).all()
    assert adapter.predict(X.iloc[:10]).shape == (10,)
    imp = adapter.importance()
    assert list(imp.columns) == ["feature", "gain"]
    assert list(imp["feature"]) == list(X.columns)
    if not supports_initial_score:
        with pytest.raises(ValueError, match="초기 점수"):
            adapter.fit(
                X.iloc[:180],
                y.iloc[:180],
                X.iloc[180:],
                y.iloc[180:],
                pd.Series(np.zeros(180)),
                pd.Series(np.zeros(60)),
            )


def test_xgboost_adapter_smoke():
    cfg = ModelConfig(
        kind="xgboost",
        params={
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "eval_metric": "auc",
        },
        fit={"early_stopping_rounds": 5},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.XGBoostAdapter)
    X, y = _smoke_data()
    _assert_adapter_contract(adapter, X, y, supports_initial_score=True)
    diagnostics = model_mod.collect_training_diagnostics(adapter)
    assert diagnostics is not None
    assert diagnostics["best_iteration"] >= 0
    assert 0.0 <= diagnostics["best_score"] <= 1.0


def test_xgboost_accepts_float_category_copies():
    """정확값 범주 복제 열(범주가 부동소수)을 native 범주로 학습하고 예측한다. (#622)"""
    cfg = ModelConfig(
        kind="xgboost",
        params={
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "eval_metric": "auc",
        },
        fit={"early_stopping_rounds": 5},
    )
    X, y = _smoke_data()
    rng = np.random.default_rng(2)
    values = pd.Series(rng.choice([0.5, 1.25, 7.5, np.nan], size=len(X)))
    X["v_cat"] = pd.Categorical(values, categories=[0.5, 1.25, 7.5])
    adapter = model_mod.create(cfg, seed=SEED)
    _assert_adapter_contract(adapter, X, y, supports_initial_score=True)
    # 범주 이름만 문자열이 되고 코드 배정과 피처 이름·순서는 그대로다.
    converted = model_mod._xgboost_categorical_frame(X)
    assert list(converted.columns) == list(X.columns)
    assert list(converted["v_cat"].cat.categories) == ["0.5", "1.25", "7.5"]
    assert np.array_equal(converted["v_cat"].cat.codes, X["v_cat"].cat.codes)
    assert converted["c"].dtype == X["c"].dtype
    assert list(adapter._model.feature_names_in_) == list(X.columns)


def test_xgboost_paired_fixed_fit_reports_fixed_training_schedule():
    cfg = ModelConfig(
        kind="xgboost",
        params={
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "eval_metric": "auc",
        },
        fit={"early_stopping_rounds": 5},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    X, y = _smoke_data()
    adapter.fit_full(X.iloc[:180], y.iloc[:180], training_budget=7)

    assert model_mod.collect_training_diagnostics(adapter) == {
        "training_schedule": FIXED_COUNT,
        "n_estimators": 7,
    }


def test_catboost_adapter_smoke():
    cfg = ModelConfig(
        kind="catboost",
        params={"iterations": 30, "depth": 3, "learning_rate": 0.1},
        fit={"early_stopping_rounds": 5},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.CatBoostAdapter)
    X, y = _smoke_data()
    _assert_adapter_contract(adapter, X, y, supports_initial_score=True)


def test_hist_gradient_boosting_adapter_smoke():
    cfg = ModelConfig(
        kind="hist_gradient_boosting",
        params={"max_iter": 30, "max_leaf_nodes": 7, "learning_rate": 0.1},
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.HistGradientBoostingAdapter)
    X, y = _smoke_data()
    _assert_adapter_contract(adapter, X, y)


def test_logistic_onehot_adapter_smoke():
    cfg = ModelConfig(
        kind="logistic_onehot",
        params={"C": 1.0, "max_iter": 200, "onehot_max_card": 10},
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.LogisticOnehotAdapter)
    # _smoke_data의 수치 2열은 카디널리티가 onehot_max_card를 넘어 표준화 통과,
    # 결측 있는 category 열은 결측 지시자를 포함한 one-hot이 된다.
    X, y = _smoke_data()
    _assert_adapter_contract(adapter, X, y)


def test_logistic_onehot_encodes_exact_values_without_leaking_unseen():
    """정확값 one-hot: 학습 fold 값 집합만 카테고리가 되고, 검증에만 있는 값과
    학습에 결측이 없던 컬럼의 검증 결측은 영벡터 블록으로 처리된다."""
    rng = np.random.default_rng(3)
    n = 300
    values = rng.choice([1.5, 2.5, 3.5, 4.5], size=n)
    X = pd.DataFrame({"v": values, "w": rng.choice([0.1, 0.2], size=n)})
    y = pd.Series((values > 2.5).astype(int) ^ (rng.uniform(size=n) < 0.1).astype(int))
    adapter = model_mod.create(
        ModelConfig(kind="logistic_onehot", params={"max_iter": 200}, fit={}), seed=SEED
    )
    va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    assert ((va_pred >= 0) & (va_pred <= 1)).all()
    # 값별 라벨 평균을 학습했으므로 값이 다르면 예측이 갈린다.
    assert roc_auc_score(y.iloc[240:], va_pred) > 0.8

    unseen = pd.DataFrame({"v": [9.9, np.nan], "w": [0.1, 0.2]})
    pred = adapter.predict(unseen)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()

    imp = adapter.importance()
    assert list(imp["feature"]) == ["v", "w"]
    assert (imp["gain"] >= 0).all()
    assert (
        imp.loc[imp["feature"] == "v", "gain"].item()
        > imp.loc[imp["feature"] == "w", "gain"].item()
    )


def test_logistic_onehot_penalty_variants():
    """#200: l1·elasticnet은 saga로 학습되고, l1_ratio는 elasticnet 전용이다."""
    rng = np.random.default_rng(5)
    n = 300
    values = rng.choice([1.5, 2.5, 3.5, 4.5], size=n)
    X = pd.DataFrame({"v": values, "w": rng.choice([0.1, 0.2], size=n)})
    y = pd.Series((values > 2.5).astype(int) ^ (rng.uniform(size=n) < 0.1).astype(int))
    for params in (
        {"penalty": "l1", "max_iter": 2000},
        {"penalty": "elasticnet", "l1_ratio": 0.5, "max_iter": 2000},
    ):
        adapter = model_mod.create(
            ModelConfig(kind="logistic_onehot", params=params, fit={}), seed=SEED
        )
        va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
        assert roc_auc_score(y.iloc[240:], va_pred) > 0.8

    # L1 좌표 하강 liblinear는 saga의 빠른 대안으로 명시 선택할 수 있다.
    adapter = model_mod.create(
        ModelConfig(
            kind="logistic_onehot",
            params={"penalty": "l1", "solver": "liblinear", "max_iter": 2000},
            fit={},
        ),
        seed=SEED,
    )
    va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    assert roc_auc_score(y.iloc[240:], va_pred) > 0.8

    with pytest.raises(ValueError, match="l1_ratio"):
        model_mod.create(
            ModelConfig(
                kind="logistic_onehot",
                params={"penalty": "l1", "l1_ratio": 0.5},
                fit={},
            ),
            seed=SEED,
        )
    with pytest.raises(ValueError, match="penalty"):
        model_mod.create(
            ModelConfig(kind="logistic_onehot", params={"penalty": "none"}, fit={}),
            seed=SEED,
        )
    with pytest.raises(ValueError, match="solver"):
        model_mod.create(
            ModelConfig(
                kind="logistic_onehot",
                params={
                    "penalty": "elasticnet",
                    "l1_ratio": 0.5,
                    "solver": "liblinear",
                },
                fit={},
            ),
            seed=SEED,
        )


def test_logistic_onehot_cross_pairs_encode_train_fold_pairs_only():
    """#200: 교차 블록은 학습 fold에서 cross_min_count 이상 관측된 쌍만 카테고리가
    되고, 미관측 쌍·결측 포함 행은 영벡터, importance에 교차 feature가 붙는다."""
    rng = np.random.default_rng(7)
    n = 400
    a = rng.choice([1.0, 2.0], size=n)
    b = rng.choice([10.0, 20.0], size=n)
    X = pd.DataFrame({"a": a, "b": b})
    # 라벨이 a·b의 XOR 조합에 달려 있어 단일 컬럼 one-hot으로는 못 맞춘다.
    y = pd.Series(
        ((a == 2.0) ^ (b == 20.0)).astype(int)
        ^ (rng.uniform(size=n) < 0.05).astype(int)
    )
    plain = model_mod.create(
        ModelConfig(kind="logistic_onehot", params={"max_iter": 300}, fit={}), seed=SEED
    )
    crossed = model_mod.create(
        ModelConfig(
            kind="logistic_onehot",
            params={"max_iter": 300, "cross_pairs": [["a", "b"]], "cross_min_count": 2},
            fit={},
        ),
        seed=SEED,
    )
    plain_auc = roc_auc_score(
        y.iloc[320:], plain.fit(X.iloc[:320], y.iloc[:320], X.iloc[320:], y.iloc[320:])
    )
    crossed_auc = roc_auc_score(
        y.iloc[320:],
        crossed.fit(X.iloc[:320], y.iloc[:320], X.iloc[320:], y.iloc[320:]),
    )
    assert plain_auc < 0.6
    assert crossed_auc > 0.9

    # 학습 fold에 없던 쌍과 결측 포함 행은 교차 블록이 영벡터라 예측이 유한하다.
    unseen = pd.DataFrame({"a": [9.0, np.nan], "b": [10.0, 20.0]})
    assert np.isfinite(crossed.predict(unseen)).all()

    imp = crossed.importance()
    assert list(imp["feature"]) == ["a", "b", "a*b"]
    assert imp.loc[imp["feature"] == "a*b", "gain"].item() > 0

    # 관측 쌍 수가 cross_max_card를 넘으면 설정 오류다.
    capped = model_mod.create(
        ModelConfig(
            kind="logistic_onehot",
            params={"max_iter": 300, "cross_pairs": [["a", "b"]], "cross_max_card": 2},
            fit={},
        ),
        seed=SEED,
    )
    with pytest.raises(ValueError, match="cross_max_card"):
        capped.fit(X.iloc[:320], y.iloc[:320], X.iloc[320:], y.iloc[320:])


def test_lightgbm_adapter_adds_initial_score_back_to_predictions():
    rng = np.random.default_rng(2)
    n = 240
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    margin = pd.Series(np.where(X["a"] > 0, 0.7, -0.7), index=X.index)
    cfg = ModelConfig(
        kind="lightgbm",
        params={
            "objective": "binary",
            "n_estimators": 30,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbosity": -1,
        },
        fit={"early_stopping_rounds": 5},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    va_pred = adapter.fit(
        X.iloc[:180],
        y.iloc[:180],
        X.iloc[180:],
        y.iloc[180:],
        margin.iloc[:180],
        margin.iloc[180:],
    )
    assert va_pred.shape == (60,)
    assert ((va_pred > 0) & (va_pred < 1)).all()
    with pytest.raises(ValueError, match="예측에도 같은 출처의 초기 점수"):
        adapter.predict(X.iloc[:10])
    assert adapter.predict(X.iloc[:10], margin.iloc[:10]).shape == (10,)


# ---- 반복형 계열이 관측 학습 길이 근거를 남기는 계약 (#372) ----


# 이슈 372가 확정한 아홉 반복형 계열의 표. 코드가 이 표에서 벗어나면 여기서 걸린다.
TRAINING_LENGTH_TABLE = {
    "lightgbm": ("best_iteration_", ONE_BASED_COUNT),
    "xgboost": ("best_iteration", ZERO_BASED_POSITION),
    "catboost": ("get_best_iteration()", ZERO_BASED_POSITION),
    "lookup_transformer": ("best_epoch", ZERO_BASED_POSITION),
    "contextualized_spline_transformer": ("best_epoch", ONE_BASED_COUNT),
    "scalar_token_transformer": ("best_epoch", ONE_BASED_COUNT),
    "tab_cnn": ("best_epoch", ONE_BASED_COUNT),
    "tabm": ("selected_epoch_count", ONE_BASED_COUNT),
    "realmlp": ("fixed_epochs", FIXED_COUNT),
}

FIXED_TREE_TRAINING_LENGTH_TABLE = {
    "lightgbm": ("n_estimators", FIXED_COUNT),
    "xgboost": ("n_estimators", FIXED_COUNT),
    "catboost": ("iterations", FIXED_COUNT),
}

# 반복 수가 없는 계열. 이 계열들은 관측을 지어내지 않는다.
NON_ITERATIVE_KINDS = ("logistic_onehot", "tabpfn3")


class StubTrainingLengthImpl:
    """fold 학습이 남긴 원시 선택값만 흉내 내는 구현. 무거운 학습을 하지 않는다."""

    def __init__(self, *selections: int) -> None:
        self._selections = tuple(selections)

    def raw_training_length_selections(self) -> tuple[int, ...]:
        return self._selections


def declaration_from_stub(adapter, *selections: int):
    """구현 자리에 stub을 끼워 연결부 선언만 꺼낸다."""
    adapter._impl = StubTrainingLengthImpl(*selections)
    return adapter.training_length_evidence()


def test_training_length_contracts_match_the_confirmed_table():
    assert set(model_mod.TRAINING_LENGTH_CONTRACTS) == set(TRAINING_LENGTH_TABLE)
    for kind, (raw_field, raw_meaning) in TRAINING_LENGTH_TABLE.items():
        contract = model_mod.TRAINING_LENGTH_CONTRACTS[kind]
        assert contract.model_family == kind
        assert contract.raw_field == raw_field
        assert contract.raw_meaning == raw_meaning
        # 변환기 식별자는 원시 의미와 같은 눈금이다. 장부도 이 눈금으로 대조한다.
        assert contract.converter == raw_meaning
    assert set(model_mod.FIXED_COUNT_TREE_CONTRACTS) == set(
        FIXED_TREE_TRAINING_LENGTH_TABLE
    )
    for kind, (raw_field, raw_meaning) in FIXED_TREE_TRAINING_LENGTH_TABLE.items():
        contract = model_mod.FIXED_COUNT_TREE_CONTRACTS[kind]
        assert contract.model_family == kind
        assert contract.raw_field == raw_field
        assert contract.raw_meaning == raw_meaning


def test_connector_declaration_agrees_with_the_refit_ledger_table():
    """연결부 선언과 장부 관문이 같은 계열 표를 두고 서로 대조할 수 있어야 한다.

    두 표는 일부러 따로 적는다. 장부는 실행 코드에 기대지 않고 근거를 다시 맞춰 봐야
    하기 때문이다. 대신 한쪽에만 계열이 늘거나 원시 의미가 갈리면 여기서 걸린다.
    """
    from pipeline.refit_plan import MODEL_FAMILY_CONVERTERS

    expected = {
        kind: tuple(
            contract.converter
            for contract in model_mod._training_length_contract_variants(kind)
        )
        for kind in model_mod.TRAINING_LENGTH_CONTRACTS
    }
    assert MODEL_FAMILY_CONVERTERS == expected


def test_every_contracted_kind_is_a_registered_model_that_declares_evidence():
    for kind in TRAINING_LENGTH_TABLE:
        adapter_factory = model_mod.MODEL_REGISTRY[kind]
        assert hasattr(adapter_factory, "training_length_evidence"), kind


@pytest.mark.parametrize("kind", NON_ITERATIVE_KINDS)
def test_families_without_iterations_declare_nothing(kind):
    """반복 수가 없는 계열은 계약을 구현하지 않고 근거도 만들지 않는다."""
    assert kind not in model_mod.TRAINING_LENGTH_CONTRACTS
    adapter = model_mod.MODEL_REGISTRY[kind]
    assert not hasattr(adapter, "training_length_evidence")
    assert model_mod.collect_training_length_declaration(object(), kind) is None


@pytest.mark.parametrize(
    ("kind", "selections", "expected"),
    [
        ("contextualized_spline_transformer", (14,), [(None, 14, 14)]),
        ("scalar_token_transformer", (14,), [(None, 14, 14)]),
        ("tab_cnn", (9,), [(None, 9, 9)]),
        ("realmlp", (4,), [(None, 4, 4)]),
        ("lookup_transformer", (12, 14), [(0, 12, 13), (1, 14, 15)]),
        ("tabm", (12, 14), [(0, 12, 12), (1, 14, 14)]),
    ],
)
def test_each_family_converts_its_own_raw_selection(kind, selections, expected):
    """계열마다 원시 값이 관측 학습 길이로 바뀌는 규칙을 무거운 학습 없이 고정한다."""
    adapter = model_mod.create(ModelConfig(kind=kind, params={}, fit={}), seed=SEED)
    declaration = declaration_from_stub(adapter, *selections)

    checked = model_mod.collect_training_length_declaration(adapter, kind)
    assert checked == declaration

    evidence = observe_declaration(declaration, seed=SEED, outer_fold=2)
    raw_field, raw_meaning = TRAINING_LENGTH_TABLE[kind]
    assert evidence.model_family == kind
    assert evidence.raw_field == raw_field
    assert evidence.raw_meaning == raw_meaning
    assert evidence.converter == raw_meaning
    assert [
        (item.inner_member, item.raw_value, item.value)
        for item in evidence.observations
    ] == expected
    assert {item.seed for item in evidence.observations} == {SEED}
    assert {item.outer_fold for item in evidence.observations} == {2}
    assert all(item.raw_path for item in evidence.observations)


def test_inner_member_families_name_the_member_in_the_raw_path():
    adapter = model_mod.create(
        ModelConfig(kind="lookup_transformer", params={}, fit={}), seed=SEED
    )
    declaration = declaration_from_stub(adapter, 12, 14)

    assert [selection.raw_path for selection in declaration.selections] == [
        "training_diagnostics.fold_initialization_members[0].best_epoch",
        "training_diagnostics.fold_initialization_members[1].best_epoch",
    ]


def test_single_member_family_rejects_more_than_one_raw_selection():
    adapter = model_mod.create(
        ModelConfig(kind="tab_cnn", params={}, fit={}), seed=SEED
    )
    with pytest.raises(ValueError):
        declaration_from_stub(adapter, 9, 10)


def test_declaring_evidence_before_fitting_is_refused():
    adapter = model_mod.create(ModelConfig(kind="tabm", params={}, fit={}), seed=SEED)
    with pytest.raises(TrainingLengthError, match="fold 학습을 마친 뒤에만"):
        adapter.training_length_evidence()


def test_collector_rejects_a_declaration_that_leaves_the_registered_contract():
    class DriftedAdapter:
        def training_length_evidence(self):
            return TrainingLengthContract(
                "lightgbm", "best_iteration_", ZERO_BASED_POSITION
            ).declare([RawTrainingLengthSelection(raw_path="x", raw_value=3)])

    with pytest.raises(TrainingLengthError, match="raw_meaning"):
        model_mod.collect_training_length_declaration(DriftedAdapter(), "lightgbm")


def test_collector_rejects_evidence_from_an_unregistered_family():
    class UnregisteredAdapter:
        def training_length_evidence(self):
            raise AssertionError("계약이 없는 계열은 선언을 읽기 전에 막아야 한다.")

    with pytest.raises(TrainingLengthError, match="등록되지 않은 계열"):
        model_mod.collect_training_length_declaration(UnregisteredAdapter(), "tabr")


def test_collector_rejects_a_non_declaration_return_value():
    class WrongTypeAdapter:
        def training_length_evidence(self):
            return {"model_family": "lightgbm"}

    with pytest.raises(TypeError, match="TrainingLengthDeclaration"):
        model_mod.collect_training_length_declaration(WrongTypeAdapter(), "lightgbm")


def test_run_cv_records_evidence_with_seed_and_outer_fold_coordinates(monkeypatch):
    """fold 실행부가 좌표를 채워 구조화 학습 진단에 같은 형식으로 남긴다."""

    class EvidenceAdapter(FakeAdapter):
        def training_length_evidence(self):
            return model_mod.TRAINING_LENGTH_CONTRACTS["fake_iterative"].declare(
                [
                    RawTrainingLengthSelection(
                        raw_path="fake.best_round",
                        raw_value=round(self.fold_value * 10),
                    )
                ]
            )

    made: list[EvidenceAdapter] = []

    def factory(params: dict, fit: dict, seed: int) -> EvidenceAdapter:
        adapter = EvidenceAdapter(params, fit, seed, fold_value=0.1 * (len(made) + 1))
        made.append(adapter)
        return adapter

    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "fake_iterative", factory)
    monkeypatch.setitem(
        model_mod.TRAINING_LENGTH_CONTRACTS,
        "fake_iterative",
        TrainingLengthContract("fake_iterative", "best_round", ZERO_BASED_POSITION),
    )
    cfg = fake_experiment_config()
    cfg = replace(cfg, model=ModelConfig(kind="fake_iterative", params={}, fit={}))
    plan = FeaturePlan.from_config(cfg.features)
    train, test = toy_train_test()
    train, test = plan.apply_dataset_wide(train, test)
    train["fold"] = np.arange(len(train)) % N_FOLDS

    result = cv.run_cv(cfg, plan, train, test, seed=SEED)

    assert [item["fold"] for item in result.model_training_diagnostics] == list(
        range(N_FOLDS)
    )
    for fold, item in enumerate(result.model_training_diagnostics):
        # 기존 모델별 진단은 그대로 두고 표준 근거를 나란히 남긴다.
        assert item["model_kind"] == "fake_iterative"
        assert item["details"] is None
        evidence = item["training_length_evidence"]
        assert evidence["model_family"] == "fake_iterative"
        assert evidence["converter"] == "zero_based_position"
        assert evidence["observations"] == [
            {
                "seed": SEED,
                "outer_fold": fold,
                "inner_member": None,
                "raw_field": "best_round",
                "raw_path": "fake.best_round",
                "raw_value": fold + 1,
                "raw_meaning": ZERO_BASED_POSITION,
                "observed_training_length": fold + 2,
            }
        ]
        # 기록 전에 JSON 직렬화 가능성이 이미 확인된 값이다.
        json.dumps(evidence, allow_nan=False)


TREE_SMOKE_CONFIGS = {
    "lightgbm": ModelConfig(
        kind="lightgbm",
        params={
            "objective": "binary",
            "n_estimators": 30,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbosity": -1,
        },
        fit={"early_stopping_rounds": 5},
    ),
    "xgboost": ModelConfig(
        kind="xgboost",
        params={
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "eval_metric": "auc",
        },
        fit={"early_stopping_rounds": 5},
    ),
    "catboost": ModelConfig(
        kind="catboost",
        params={"iterations": 30, "depth": 3, "learning_rate": 0.1},
        fit={"early_stopping_rounds": 5},
    ),
}

TREE_FIXED_SMOKE_CONFIGS = {
    "lightgbm": ModelConfig(
        kind="lightgbm",
        params={
            "objective": "binary",
            "n_estimators": 8,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbosity": -1,
        },
        fit={},
    ),
    "xgboost": ModelConfig(
        kind="xgboost",
        params={
            "n_estimators": 8,
            "max_depth": 3,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "eval_metric": "auc",
        },
        fit={},
    ),
    "catboost": ModelConfig(
        kind="catboost",
        params={"iterations": 8, "depth": 3, "learning_rate": 0.1},
        fit={},
    ),
}


@pytest.mark.parametrize("kind", sorted(TREE_SMOKE_CONFIGS))
def test_tree_families_declare_evidence_from_their_own_raw_field(kind):
    """트리 계열도 원시 선택값을 표준 근거로 남긴다. LightGBM만 `+1`하지 않는다."""
    X, y = _smoke_data()
    adapter = model_mod.create(TREE_SMOKE_CONFIGS[kind], seed=SEED)
    adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])

    declaration = model_mod.collect_training_length_declaration(adapter, kind)
    raw_field, raw_meaning = TRAINING_LENGTH_TABLE[kind]
    assert declaration.raw_field == raw_field
    assert declaration.raw_meaning == raw_meaning
    (selection,) = declaration.selections
    assert selection.inner_member is None

    raw_value = {
        "lightgbm": lambda model: model.best_iteration_,
        "xgboost": lambda model: model.best_iteration,
        "catboost": lambda model: model.get_best_iteration(),
    }[kind](adapter._model)
    assert selection.raw_value == raw_value

    (observation,) = observe_declaration(
        declaration, seed=SEED, outer_fold=0
    ).observations
    increment = 1 if raw_meaning == ZERO_BASED_POSITION else 0
    assert observation.value == raw_value + increment
    assert observation.value >= 1


@pytest.mark.parametrize("kind", sorted(TREE_FIXED_SMOKE_CONFIGS))
def test_tree_families_without_early_stopping_declare_the_configured_fixed_count(kind):
    """고정 일정 트리는 설정 반복 수를 그대로 근거로 남기고 검증 선택값을 만들지 않는다."""
    X, y = _smoke_data()
    cfg = TREE_FIXED_SMOKE_CONFIGS[kind]
    adapter = model_mod.create(cfg, seed=SEED)
    adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])

    declaration = model_mod.collect_training_length_declaration(adapter, kind)
    raw_field, raw_meaning = FIXED_TREE_TRAINING_LENGTH_TABLE[kind]
    assert declaration.raw_field == raw_field
    assert declaration.raw_meaning == raw_meaning == FIXED_COUNT
    (selection,) = declaration.selections
    assert selection.raw_path == f"model.params.{raw_field}"
    assert selection.raw_value == cfg.params[raw_field]

    (observation,) = observe_declaration(
        declaration, seed=SEED, outer_fold=0
    ).observations
    assert observation.raw_value == observation.value == cfg.params[raw_field]


@pytest.mark.parametrize("kind", sorted(TREE_FIXED_SMOKE_CONFIGS))
def test_fixed_count_tree_requires_an_explicit_positive_iteration_count(kind):
    cfg = TREE_FIXED_SMOKE_CONFIGS[kind]
    raw_field, _ = FIXED_TREE_TRAINING_LENGTH_TABLE[kind]
    params = dict(cfg.params)
    params.pop(raw_field)

    with pytest.raises(ValueError, match=f"model.params.{raw_field}"):
        model_mod.create(replace(cfg, params=params), seed=SEED)


@pytest.mark.parametrize("kind", sorted(TREE_SMOKE_CONFIGS))
def test_tree_families_refuse_evidence_after_a_full_data_refit(kind):
    """전체 자료 재학습에는 조기 종료가 없으므로 관측을 지어내지 않는다."""
    X, y = _smoke_data()
    adapter = model_mod.create(TREE_SMOKE_CONFIGS[kind], seed=SEED)
    model_mod.fit_full(adapter, X, y, training_budget=5)

    with pytest.raises(TrainingLengthError, match="검증 분할로 학습한 뒤에만"):
        adapter.training_length_evidence()
