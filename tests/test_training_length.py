"""관측 학습 길이 변환과 재학습 예산 산정의 공통 계약 테스트. (#371)"""

from __future__ import annotations

import pytest

from pipeline.training_length import (
    FIXED_COUNT,
    ONE_BASED_COUNT,
    ZERO_BASED_POSITION,
    ObservedTrainingLength,
    RefitBudgetPolicy,
    TrainingLengthError,
    derive_refit_budgets,
    observe_training_length,
    observed_length_from_raw,
    round_half_up,
)


def observations(raw_meaning: str, raw_field: str, per_seed: dict[int, list[list[int]]]):
    """시드 -> 바깥쪽 분할 -> 내부 구성원 원시 값 표를 관측 목록으로 편다.

    내부 구성원이 하나뿐인 계열도 같은 표로 적어 좌표 표기를 하나로 유지한다.
    """
    flattened = []
    for seed, folds in per_seed.items():
        for outer_fold, inner_values in enumerate(folds):
            for inner_member, raw_value in enumerate(inner_values):
                flattened.append(
                    observe_training_length(
                        seed=seed,
                        outer_fold=outer_fold,
                        raw_field=raw_field,
                        raw_value=raw_value,
                        raw_meaning=raw_meaning,
                        inner_member=None if len(inner_values) == 1 else inner_member,
                    )
                )
    return flattened


def single_member(per_seed: dict[int, list[int]]) -> dict[int, list[list[int]]]:
    """내부 구성원이 없는 계열의 분할별 값을 공통 표 모양으로 감싼다."""
    return {seed: [[value] for value in values] for seed, values in per_seed.items()}


# 원시 의미 변환


def test_zero_based_position_becomes_one_based_length():
    assert observed_length_from_raw(0, ZERO_BASED_POSITION) == 1
    assert observed_length_from_raw(7806, ZERO_BASED_POSITION) == 7807


@pytest.mark.parametrize("raw_meaning", [ONE_BASED_COUNT, FIXED_COUNT])
def test_counts_pass_through_unchanged(raw_meaning):
    assert observed_length_from_raw(1, raw_meaning) == 1
    assert observed_length_from_raw(24, raw_meaning) == 24


@pytest.mark.parametrize("raw_meaning", [ZERO_BASED_POSITION, ONE_BASED_COUNT, FIXED_COUNT])
@pytest.mark.parametrize("raw_value", [True, False])
def test_booleans_are_rejected(raw_meaning, raw_value):
    with pytest.raises(TrainingLengthError, match="불리언"):
        observed_length_from_raw(raw_value, raw_meaning)


@pytest.mark.parametrize("raw_value", [1.0, "1", None, 12.5])
def test_non_integers_are_rejected(raw_value):
    with pytest.raises(TrainingLengthError, match="정수"):
        observed_length_from_raw(raw_value, ONE_BASED_COUNT)


def test_negative_position_is_rejected():
    with pytest.raises(TrainingLengthError, match="음수"):
        observed_length_from_raw(-1, ZERO_BASED_POSITION)


@pytest.mark.parametrize("raw_meaning", [ONE_BASED_COUNT, FIXED_COUNT])
@pytest.mark.parametrize("raw_value", [0, -3])
def test_non_positive_counts_are_rejected(raw_meaning, raw_value):
    with pytest.raises(TrainingLengthError, match="1 이상"):
        observed_length_from_raw(raw_value, raw_meaning)


@pytest.mark.parametrize("raw_meaning", ["best_epoch", "zero_based", "", None])
def test_unknown_raw_meaning_is_rejected(raw_meaning):
    with pytest.raises(TrainingLengthError, match="알 수 없는 원시 의미"):
        observed_length_from_raw(3, raw_meaning)


def test_already_converted_length_cannot_be_converted_again():
    observed = observe_training_length(
        seed=42,
        outer_fold=0,
        raw_field="best_iteration",
        raw_value=7806,
        raw_meaning=ZERO_BASED_POSITION,
    )

    with pytest.raises(TrainingLengthError, match="다시 변환할 수 없다"):
        observed_length_from_raw(observed, ZERO_BASED_POSITION)


def test_observation_keeps_raw_evidence_beside_converted_length():
    observed = observe_training_length(
        seed=43,
        outer_fold=2,
        raw_field="best_epoch",
        raw_value=11,
        raw_meaning=ZERO_BASED_POSITION,
        inner_member=1,
    )

    assert observed.raw_value == 11
    assert observed.raw_meaning == ZERO_BASED_POSITION
    assert observed.value == 12
    assert (observed.seed, observed.outer_fold, observed.inner_member) == (43, 2, 1)


@pytest.mark.parametrize(
    "coordinate", [{"outer_fold": -1}, {"inner_member": -1}, {"raw_field": ""}]
)
def test_invalid_coordinates_are_rejected(coordinate):
    kwargs = {
        "seed": 42,
        "outer_fold": 0,
        "raw_field": "best_epoch",
        "raw_value": 5,
        "raw_meaning": ONE_BASED_COUNT,
    }
    kwargs.update(coordinate)

    with pytest.raises(TrainingLengthError):
        observe_training_length(**kwargs)


def test_directly_built_observation_still_enforces_positive_length():
    with pytest.raises(TrainingLengthError, match="양의 정수"):
        ObservedTrainingLength(
            seed=42,
            outer_fold=0,
            raw_field="best_epoch",
            raw_value=0,
            raw_meaning=ZERO_BASED_POSITION,
            value=0,
        )


# 사사오입


def test_round_half_up_does_not_round_half_to_even():
    assert round_half_up(12.5) == 13
    assert round_half_up(9758.75) == 9759
    assert round_half_up(15.0) == 15
    assert round_half_up(11.25) == 11


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf")])
def test_round_half_up_requires_a_positive_finite_value(value):
    with pytest.raises(TrainingLengthError, match="양의 유한값"):
        round_half_up(value)


# 재학습 예산 산정


def test_xgboost_confirmed_case_derives_9759_10394_10369():
    """exp135_xgb_hpo_trial30의 확정 원시 위치(#327)로 고정한다."""
    evidence = observations(
        ZERO_BASED_POSITION,
        "best_iteration",
        single_member(
            {
                42: [8491, 7806, 7382, 7488, 8320],
                43: [8314, 7608, 7032, 8450, 8404],
                44: [8057, 8484, 7157, 8294, 8442],
            }
        ),
    )

    derivation = derive_refit_budgets(evidence)

    assert derivation.budgets() == {42: 9759, 43: 10394, 44: 10369}


def test_lookup_transformer_confirmed_cases_derive_13_15_15_and_15_15_15():
    """exp127_lookup_muon과 exp131_lookup_bivariate_plr5의 확정 원시 위치(#327)로 고정한다."""
    muon = observations(
        ZERO_BASED_POSITION,
        "best_epoch",
        {
            42: [[11, 11, 9], [11, 9, 9], [11, 11, 9], [9, 9, 9], [9, 9, 11]],
            43: [[11, 9, 11], [11, 11, 11], [11, 11, 11], [9, 9, 9], [11, 9, 11]],
            44: [[11, 11, 11], [11, 9, 11], [11, 11, 11], [11, 9, 11], [11, 11, 9]],
        },
    )
    bivariate = observations(
        ZERO_BASED_POSITION,
        "best_epoch",
        {
            42: [[11, 11, 11], [11, 11, 11], [11, 11, 11], [11, 11, 9], [11, 11, 9]],
            43: [[9, 11, 11], [9, 11, 11], [9, 11, 11], [9, 9, 11], [9, 11, 11]],
            44: [[11, 11, 9], [11, 11, 9], [11, 11, 11], [11, 11, 9], [11, 11, 11]],
        },
    )

    assert derive_refit_budgets(muon).budgets() == {42: 13, 43: 15, 44: 15}
    assert derive_refit_budgets(bivariate).budgets() == {42: 15, 43: 15, 44: 15}


def test_derivation_exposes_every_intermediate_value():
    evidence = observations(
        ZERO_BASED_POSITION,
        "best_iteration",
        single_member({42: [8491, 7806, 7382, 7488, 8320]}),
    )

    (seed_derivation,) = derive_refit_budgets(evidence).seeds

    assert seed_derivation.seed == 42
    assert seed_derivation.observed_lengths == (8492, 7807, 7383, 7489, 8321)
    assert seed_derivation.median == 7807.0
    assert seed_derivation.scaled == 9758.75
    assert seed_derivation.budget == 9759


def test_fixed_schedule_family_goes_through_the_same_calculation():
    """RealMLP 고정 4회도 숫자를 그대로 예산으로 쓰지 않고 공통 계산을 거친다."""
    evidence = observations(
        FIXED_COUNT, "fixed_epochs", single_member({42: [4, 4, 4, 4, 4]})
    )

    assert derive_refit_budgets(evidence).budgets() == {42: 5}


def test_one_based_count_family_is_not_incremented():
    """exp133_scalar_token_transformer_oof_te 시드 42의 1부터 세는 횟수(#327)다."""
    evidence = observations(
        ONE_BASED_COUNT, "best_epoch", single_member({42: [16, 12, 12, 16, 16]})
    )

    (seed_derivation,) = derive_refit_budgets(evidence).seeds

    assert seed_derivation.observed_lengths == (16, 12, 12, 16, 16)
    assert seed_derivation.budget == 20


def test_seed_order_follows_first_appearance():
    evidence = observations(
        ONE_BASED_COUNT, "best_iteration_", single_member({44: [4], 42: [8], 43: [6]})
    )

    assert [seed.seed for seed in derive_refit_budgets(evidence).seeds] == [44, 42, 43]


def test_empty_evidence_is_rejected():
    with pytest.raises(TrainingLengthError, match="하나도 없다"):
        derive_refit_budgets([])


def test_raw_values_cannot_bypass_the_converter():
    with pytest.raises(TrainingLengthError, match="검증된 관측 학습 길이"):
        derive_refit_budgets([7806, 8314])


def test_default_policy_is_the_confirmed_protocol():
    policy = RefitBudgetPolicy()

    assert (policy.statistic, policy.multiplier, policy.rounding) == (
        "median",
        1.25,
        "half_up",
    )


@pytest.mark.parametrize(
    "override",
    [
        {"statistic": "mean"},
        {"statistic": "max"},
        {"rounding": "half_even"},
        {"rounding": "floor"},
        {"multiplier": 0},
        {"multiplier": -1.25},
        {"multiplier": "1.25"},
    ],
)
def test_other_protocols_are_rejected(override):
    with pytest.raises(TrainingLengthError):
        RefitBudgetPolicy(**override)
