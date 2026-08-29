"""바깥쪽 분할 계약 초기 점수(nested_logistic_onehot)의 누출 경계와 계보. (#505)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import initial_score
from pipeline.config import InitialScoreConfig
from pipeline.data import TARGET
from pipeline.initial_score import KnownOriginalRule, NestedLogisticOnehot


def _frame(n: int, seed: int, with_target: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": rng.integers(0, 12, size=n).astype("float64"),
            "social_media_hours": rng.integers(0, 6, size=n).astype("float64"),
            "gender": rng.choice(["Male", "Female"], size=n),
        }
    )
    df.loc[df.index[:3], "social_media_hours"] = np.nan
    if with_target:
        df[TARGET] = (
            (df["daily_screen_time_hours"] + df["social_media_hours"].fillna(0) + rng.normal(size=n))
            > 8
        ).astype(int)
    return df


def _provider() -> NestedLogisticOnehot:
    return NestedLogisticOnehot(
        cols=["daily_screen_time_hours", "social_media_hours", "gender"],
        categorical=["gender"],
        C=1.0,
        max_iter=200,
        onehot_max_card=100,
        inner_splits=3,
    )


def test_registry_creates_outer_fold_provider():
    provider = initial_score.create(
        InitialScoreConfig(
            kind="nested_logistic_onehot",
            params={
                "cols": ["daily_screen_time_hours", "gender"],
                "categorical": ["gender"],
                "inner_splits": 2,
            },
        )
    )
    assert initial_score.is_outer_fold_provider(provider)
    assert not initial_score.is_outer_fold_provider(KnownOriginalRule())
    assert provider.input_paths() == {}


def test_constructor_rejects_target_placebo_and_unknown_params():
    with pytest.raises(ValueError, match="설명변수로 쓸 수 없다"):
        NestedLogisticOnehot(cols=["x", TARGET], categorical=[])
    with pytest.raises(ValueError, match="설명변수로 쓸 수 없다"):
        NestedLogisticOnehot(cols=["x", "placebo_noise"], categorical=[])
    with pytest.raises(ValueError, match="부분집합"):
        NestedLogisticOnehot(cols=["x"], categorical=["y"])
    with pytest.raises(ValueError, match="inner_splits"):
        NestedLogisticOnehot(cols=["x"], categorical=[], inner_splits=1)
    with pytest.raises(ValueError, match="initial_score nested_logistic_onehot"):
        initial_score.create(
            InitialScoreConfig(
                kind="nested_logistic_onehot",
                params={"cols": ["x"], "categorical": [], "penalty": "l1"},
            )
        )


def test_outer_fold_scores_follow_indexes_and_record_lineage():
    train = _frame(240, seed=1, with_target=True)
    test = _frame(50, seed=2, with_target=False)
    training_index = train.index[train["id"] % 4 != 0]
    validation_index = train.index[train["id"] % 4 == 0]
    provider = _provider()

    scores = provider.compute_outer_fold(
        train.loc[training_index],
        train.loc[validation_index].drop(columns=[TARGET]),
        test,
        seed=42,
        outer_fold=0,
    )
    assert scores.training.index.equals(training_index)
    assert scores.validation.index.equals(validation_index)
    assert scores.test.index.equals(test.index)
    for series in (scores.training, scores.validation, scores.test):
        assert series.dtype == "float64"
        assert np.isfinite(series).all()
    evidence = scores.evidence
    assert evidence["kind"] == "nested_logistic_onehot"
    assert evidence["outer_fold"] == 0
    assert evidence["seed"] == 42
    assert evidence["inner_splits"] == 3
    assert len(evidence["inner_fit_iterations"]) == 3
    assert 0.0 <= evidence["inner_oof_auc"] <= 1.0
    assert set(evidence["sha256"]) == {"training", "validation", "test"}
    assert set(evidence["logit_range"]) == {"training", "validation", "test"}
    assert "validation_first_stage_auc" not in evidence  # 검증 목표값은 생성기가 보지 않는다.

    # 같은 입력과 시드는 같은 로짓을 만든다(계보 해시가 재현 근거가 된다).
    again = provider.compute_outer_fold(
        train.loc[training_index],
        train.loc[validation_index].drop(columns=[TARGET]),
        test,
        seed=42,
        outer_fold=0,
    )
    assert again.evidence["sha256"] == evidence["sha256"]


def test_training_scores_are_inner_oof_not_in_sample():
    """학습 부분 행의 로짓은 그 행을 뺀 내부 학습에서 나와야 한다."""
    train = _frame(240, seed=3, with_target=True)
    test = _frame(20, seed=4, with_target=False)
    training_index = train.index[:200]
    validation_index = train.index[200:]
    provider = _provider()
    validation = train.loc[validation_index].drop(columns=[TARGET])
    base = provider.compute_outer_fold(
        train.loc[training_index], validation, test, seed=7, outer_fold=1
    )
    # 학습 부분 전체로 맞춘 회귀의 학습 행 예측(표본 내)은 내부 OOF와 달라야 한다.
    adapter = provider._adapter(7)
    X = train.loc[training_index, provider.cols]
    adapter.fit_full(X, train.loc[training_index, TARGET], None)
    in_sample = initial_score.probabilities_to_logits(adapter.predict(X), provider.clip)
    assert not np.allclose(in_sample, base.training.to_numpy())


def test_validation_and_test_must_not_carry_target():
    train = _frame(120, seed=5, with_target=True)
    test = _frame(20, seed=6, with_target=False)
    training_index = train.index[:90]
    validation_index = train.index[90:]
    provider = _provider()
    with pytest.raises(ValueError, match="합성 타깃"):
        provider.compute_outer_fold(
            train.loc[training_index], train.loc[validation_index], test, seed=1, outer_fold=0
        )
    with pytest.raises(ValueError, match="합성 타깃"):
        provider.compute_outer_fold(
            train.loc[training_index],
            train.loc[validation_index].drop(columns=[TARGET]),
            test.assign(**{TARGET: 0}),
            seed=1,
            outer_fold=0,
        )
    with pytest.raises(ValueError, match="겹친다"):
        provider.compute_outer_fold(
            train.loc[training_index],
            train.loc[training_index[:10]].drop(columns=[TARGET]),
            test,
            seed=1,
            outer_fold=0,
        )
    with pytest.raises(ValueError, match="목표값"):
        provider.compute_outer_fold(
            train.loc[training_index].drop(columns=[TARGET]),
            train.loc[validation_index].drop(columns=[TARGET]),
            test,
            seed=1,
            outer_fold=0,
        )
    with pytest.raises(ValueError, match="합성 타깃"):
        provider.compute_full(train, test.assign(**{TARGET: 1}), seed=1)


def test_fold_scores_dispatch_for_both_contracts():
    train = _frame(120, seed=8, with_target=True)
    test = _frame(20, seed=9, with_target=False)
    train["fold"] = train["id"] % 2
    training_index = train.index[train["fold"] != 0]
    validation_index = train.index[train["fold"] == 0]

    legacy = KnownOriginalRule(clip=0.01)
    seed_scores = initial_score.seed_level_scores(legacy, train, test, seed=3)
    assert seed_scores is not None
    sliced = initial_score.fold_scores(
        legacy, seed_scores, train, test, 3, 0, training_index, validation_index
    )
    assert sliced.evidence is None
    assert sliced.training.index.equals(training_index)
    assert sliced.validation.index.equals(validation_index)
    assert sliced.test.equals(seed_scores.test)

    nested = _provider()
    assert initial_score.seed_level_scores(nested, train, test, seed=3) is None
    fresh = initial_score.fold_scores(
        nested, None, train, test, 3, 0, training_index, validation_index
    )
    assert fresh.evidence["outer_fold"] == 0
    assert fresh.training.index.equals(training_index)
    assert fresh.validation.index.equals(validation_index)
    assert fresh.test.index.equals(test.index)

    assert initial_score.fold_scores(
        None, None, train, test, 3, 0, training_index, validation_index
    ) is None


def test_full_data_scores_use_inner_oof_for_train_and_full_fit_for_test():
    train = _frame(150, seed=10, with_target=True)
    test = _frame(30, seed=11, with_target=False)
    nested = _provider()
    scores = initial_score.full_data_scores(nested, train, test, seed=5)
    assert scores.train.index.equals(train.index)
    assert scores.test.index.equals(test.index)
    assert np.isfinite(scores.train).all() and np.isfinite(scores.test).all()

    legacy = KnownOriginalRule(clip=0.01)
    legacy_scores = initial_score.full_data_scores(legacy, train, test, seed=5)
    assert legacy_scores.train.index.equals(train.index)
