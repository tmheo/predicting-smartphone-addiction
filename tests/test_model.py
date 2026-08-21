"""모델 계열 레지스트리와 cv 루프 테스트. (#72 완료 기준)

- 레지스트리 dispatch: kind -> adapter 해석, 미등록 kind는 명확한 오류.
- 결정적 가짜 adapter로 cv 루프 전체 검증: OOF 배치, test 예측 fold 평균,
  fold별 recorder 통지, importance 조립. 실제 LightGBM 학습 없이 돈다.
- lightgbm adapter 소형 데이터 스모크 테스트.
"""

from __future__ import annotations

import inspect
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
            train=Path("unused"), test=Path("unused"),
            sample_submission=Path("unused"), folds=Path("unused"),
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
    assert (adapter.params, adapter.fit_args, adapter.seed) == ({"p": 1}, {"f": 2}, SEED)


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
    features_per_fold = set(result.importance[result.importance["fold"] == 0]["feature"])
    assert features_per_fold == set(result.feature_names)
    assert PLACEBO in features_per_fold  # placebo 자동 삽입까지 fake 학습 입력에 닿는다.


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


def test_lightgbm_max_bin_by_feature_resolves_names_in_matrix_order():
    params = {
        "max_bin": 1023,
        "max_bin_by_feature": {"screen": 1439, "weekend": 2047},
    }

    resolved = model_mod._resolve_lightgbm_params(
        params, ["age", "screen", "derived", "weekend"]
    )

    assert resolved["max_bin_by_feature"] == [1023, 1439, 1023, 2047]
    assert params["max_bin_by_feature"] == {"screen": 1439, "weekend": 2047}


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
    cat = pd.Series(rng.choice(["Low", "High"], size=n)).where(rng.uniform(size=n) > 0.1)
    X["c"] = pd.Categorical(cat, categories=["High", "Low"])
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return X, y


def _assert_adapter_contract(adapter: object, X: pd.DataFrame, y: pd.Series) -> None:
    """fit/predict/importance 계약: 확률 범위, 예측 shape, (feature, gain) 프레임."""
    va_pred = adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])
    assert va_pred.shape == (60,)
    assert ((va_pred >= 0) & (va_pred <= 1)).all()
    assert adapter.predict(X.iloc[:10]).shape == (10,)
    imp = adapter.importance()
    assert list(imp.columns) == ["feature", "gain"]
    assert list(imp["feature"]) == list(X.columns)
    with pytest.raises(ValueError, match="초기 점수"):
        adapter.fit(
            X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:],
            pd.Series(np.zeros(180)), pd.Series(np.zeros(60)),
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
    _assert_adapter_contract(adapter, X, y)
    diagnostics = model_mod.collect_training_diagnostics(adapter)
    assert diagnostics is not None
    assert diagnostics["best_iteration"] >= 0
    assert 0.0 <= diagnostics["best_score"] <= 1.0


def test_catboost_adapter_smoke():
    cfg = ModelConfig(
        kind="catboost",
        params={"iterations": 30, "depth": 3, "learning_rate": 0.1},
        fit={"early_stopping_rounds": 5},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.CatBoostAdapter)
    X, y = _smoke_data()
    _assert_adapter_contract(adapter, X, y)


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
    assert imp.loc[imp["feature"] == "v", "gain"].item() > imp.loc[
        imp["feature"] == "w", "gain"
    ].item()


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
            ModelConfig(kind="logistic_onehot", params={"penalty": "l1", "l1_ratio": 0.5}, fit={}),
            seed=SEED,
        )
    with pytest.raises(ValueError, match="penalty"):
        model_mod.create(
            ModelConfig(kind="logistic_onehot", params={"penalty": "none"}, fit={}), seed=SEED
        )
    with pytest.raises(ValueError, match="solver"):
        model_mod.create(
            ModelConfig(
                kind="logistic_onehot",
                params={"penalty": "elasticnet", "l1_ratio": 0.5, "solver": "liblinear"},
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
        (((a == 2.0) ^ (b == 20.0)).astype(int) ^ (rng.uniform(size=n) < 0.05).astype(int))
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
        y.iloc[320:], crossed.fit(X.iloc[:320], y.iloc[:320], X.iloc[320:], y.iloc[320:])
    )
    assert plain_auc < 0.6 < 0.9 < crossed_auc

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
