"""모델 계열 레지스트리와 cv 루프 테스트. (#72 완료 기준)

- 레지스트리 dispatch: kind -> adapter 해석, 미등록 kind는 명확한 오류.
- 결정적 가짜 adapter로 cv 루프 전체 검증: OOF 배치, test 예측 fold 평균,
  fold별 recorder 통지, importance 조립. 실제 LightGBM 학습 없이 돈다.
- lightgbm adapter 소형 데이터 스모크 테스트.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import cv, model as model_mod
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
