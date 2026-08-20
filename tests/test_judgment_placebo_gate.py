"""플라시보 게이트 단위 테스트. (#94, 지도 #91)

- 기준값은 플라시보 원본의 평균 gain 하나이고, 미기록이면 게이트는 실패다.
- 카나리아는 플라시보와 0 중 큰 상한 이하가 통과이고, 미기록은 0.0으로 본다.
- 새 피처는 기준값 초과가 통과이고, gain 미기록은 실패다.
"""

from __future__ import annotations

import pandas as pd

from pipeline.features import PLACEBO
from pipeline.judgment import (
    check_canaries,
    check_new_features,
    mean_gain_of,
    placebo_gain_of,
)

CANARY = f"{PLACEBO}_te"


def test_mean_gain_averages_over_fold_and_seed():
    importance = pd.DataFrame(
        {
            "feature": ["age", "age", "age", PLACEBO, PLACEBO, PLACEBO],
            "fold": [0, 1, 0, 0, 1, 0],
            "seed": [42, 42, 43, 42, 42, 43],
            "gain": [300.0, 500.0, 400.0, 90.0, 110.0, 100.0],
        }
    )
    mean_gain = mean_gain_of(importance)
    assert mean_gain["age"] == 400.0
    assert placebo_gain_of(mean_gain) == 100.0


def test_placebo_gain_missing_is_none():
    assert placebo_gain_of(pd.Series({"age": 400.0})) is None


def test_canary_below_placebo_passes():
    report = check_canaries({CANARY, "age"}, pd.Series({PLACEBO: 100.0, CANARY: 10.0}))
    assert report.ok
    assert report.placebo_gain == 100.0
    assert [c.feature for c in report.checks] == [CANARY]


def test_canary_at_placebo_passes_and_above_placebo_fails():
    assert check_canaries({CANARY}, pd.Series({PLACEBO: 100.0, CANARY: 100.0})).ok
    assert not check_canaries({CANARY}, pd.Series({PLACEBO: 100.0, CANARY: 500.0})).ok


def test_zero_canary_passes_when_placebo_gain_is_negative():
    assert check_canaries({CANARY}, pd.Series({PLACEBO: -1.0, CANARY: 0.0})).ok
    assert not check_canaries({CANARY}, pd.Series({PLACEBO: -1.0, CANARY: 0.1})).ok


def test_canary_unrecorded_counts_as_zero():
    report = check_canaries({CANARY}, pd.Series({PLACEBO: 100.0}))
    assert report.ok
    assert report.checks[0].gain == 0.0


def test_canary_fails_without_placebo_baseline():
    report = check_canaries({CANARY}, pd.Series({CANARY: 0.0}))
    assert not report.ok
    assert report.placebo_gain is None


def test_new_feature_above_placebo_passes():
    report = check_new_features(
        {"age"}, {"age", "age_te"}, pd.Series({PLACEBO: 100.0, "age_te": 400.0})
    )
    assert report.ok
    assert report.new_features == ["age_te"]


def test_new_feature_at_or_below_placebo_fails():
    base, features = {"age"}, {"age", "age_te"}
    assert not check_new_features(
        base, features, pd.Series({PLACEBO: 100.0, "age_te": 100.0})
    ).ok
    assert not check_new_features(
        base, features, pd.Series({PLACEBO: 100.0, "age_te": 10.0})
    ).ok


def test_new_feature_unrecorded_gain_fails():
    report = check_new_features({"age"}, {"age", "age_te"}, pd.Series({PLACEBO: 100.0}))
    assert not report.ok
    assert report.checks[0].gain is None


def test_new_features_fail_without_placebo_baseline():
    report = check_new_features({"age"}, {"age", "age_te"}, pd.Series({"age_te": 400.0}))
    assert not report.ok
    assert report.placebo_gain is None


def test_no_new_features_passes_vacuously():
    # 플라시보와 카나리아는 새 피처가 아니므로 게이트 대상에서 빠진다.
    report = check_new_features(
        {"age"}, {"age", PLACEBO, CANARY}, pd.Series({"age": 400.0})
    )
    assert report.ok
    assert report.new_features == []
