"""관측 학습 길이 변환과 재학습 예산 산정의 공통 계약 테스트. (#371)"""

from __future__ import annotations

import pytest

from pipeline.training_length import (
    FIXED_COUNT,
    ONE_BASED_COUNT,
    ZERO_BASED_POSITION,
    ObservedTrainingLength,
    RawTrainingLengthSelection,
    RefitBudgetPolicy,
    TrainingLengthContract,
    TrainingLengthError,
    converter_identifier,
    derive_refit_budgets,
    observe_declaration,
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


# ---- 연결부가 fold 하나마다 선언하는 근거 (#372) ----


LIGHTGBM = TrainingLengthContract("lightgbm", "best_iteration_", ONE_BASED_COUNT)
LOOKUP = TrainingLengthContract("lookup_transformer", "best_epoch", ZERO_BASED_POSITION)


def test_converter_identifier_is_the_raw_meaning_itself():
    """원시 의미 하나에 변환기 하나라 식별자가 원시 의미와 같은 눈금이다."""
    for raw_meaning in (ZERO_BASED_POSITION, ONE_BASED_COUNT, FIXED_COUNT):
        assert converter_identifier(raw_meaning) == raw_meaning


def test_unknown_raw_meaning_has_no_converter():
    with pytest.raises(TrainingLengthError, match="알 수 없는 원시 의미"):
        converter_identifier("best_iteration")


def test_contract_declares_the_converter_its_raw_meaning_fixes():
    assert LIGHTGBM.converter == "one_based_count"
    assert LOOKUP.converter == "zero_based_position"


@pytest.mark.parametrize(
    ("model_family", "raw_field", "raw_meaning"),
    [
        ("", "best_epoch", ONE_BASED_COUNT),
        ("tabm", "", ONE_BASED_COUNT),
        ("tabm", "best_epoch", "one_based"),
    ],
)
def test_incomplete_contracts_are_rejected(model_family, raw_field, raw_meaning):
    with pytest.raises(TrainingLengthError):
        TrainingLengthContract(model_family, raw_field, raw_meaning)


def test_observed_evidence_carries_every_field_the_ledger_needs():
    declaration = LOOKUP.declare(
        [
            RawTrainingLengthSelection(
                raw_path="training_diagnostics.fold_initialization_members[0].best_epoch",
                raw_value=12,
                inner_member=0,
            ),
            RawTrainingLengthSelection(
                raw_path="training_diagnostics.fold_initialization_members[1].best_epoch",
                raw_value=14,
                inner_member=1,
            ),
        ]
    )
    evidence = observe_declaration(declaration, seed=42, outer_fold=3)

    assert evidence.to_json() == {
        "model_family": "lookup_transformer",
        "converter": "zero_based_position",
        "raw_field": "best_epoch",
        "raw_meaning": ZERO_BASED_POSITION,
        "observations": [
            {
                "seed": 42,
                "outer_fold": 3,
                "inner_member": 0,
                "raw_field": "best_epoch",
                "raw_path": (
                    "training_diagnostics.fold_initialization_members[0].best_epoch"
                ),
                "raw_value": 12,
                "raw_meaning": ZERO_BASED_POSITION,
                "observed_training_length": 13,
            },
            {
                "seed": 42,
                "outer_fold": 3,
                "inner_member": 1,
                "raw_field": "best_epoch",
                "raw_path": (
                    "training_diagnostics.fold_initialization_members[1].best_epoch"
                ),
                "raw_value": 14,
                "raw_meaning": ZERO_BASED_POSITION,
                "observed_training_length": 15,
            },
        ],
    }


def test_declared_evidence_feeds_the_common_budget_calculation():
    """계열 연결부의 선언과 공통 계산부가 실제로 이어지는지 본다."""
    evidence = [
        observation
        for seed, positions in {42: (12, 14, 14), 43: (11, 11, 13)}.items()
        for outer_fold, position in enumerate(positions)
        for observation in observe_declaration(
            LOOKUP.declare(
                [
                    RawTrainingLengthSelection(
                        raw_path="training_diagnostics.best_epoch", raw_value=position
                    )
                ]
            ),
            seed=seed,
            outer_fold=outer_fold,
        ).observations
    ]

    assert derive_refit_budgets(evidence).budgets() == {42: 19, 43: 15}


def test_declaration_without_inner_members_allows_only_one_selection():
    with pytest.raises(TrainingLengthError, match="내부 구성원 좌표 없이"):
        LIGHTGBM.declare(
            [
                RawTrainingLengthSelection(raw_path="a", raw_value=3),
                RawTrainingLengthSelection(raw_path="b", raw_value=4),
            ]
        )


@pytest.mark.parametrize(
    "inner_members",
    [
        (0, None),
        (0, 0),
        (1, 2),
        (0, 2),
        (1, 0),
    ],
)
def test_broken_inner_member_coordinates_are_rejected(inner_members):
    """빠지거나 중복되거나 순서가 어긋난 내부 구성원 좌표를 모두 막는다."""
    with pytest.raises(TrainingLengthError, match="내부 구성원"):
        LOOKUP.declare(
            [
                RawTrainingLengthSelection(
                    raw_path=f"m[{index}]", raw_value=5, inner_member=inner_member
                )
                for index, inner_member in enumerate(inner_members)
            ]
        )


def test_empty_declaration_is_rejected():
    with pytest.raises(TrainingLengthError, match="원시 선택값이 하나도 없다"):
        LIGHTGBM.declare([])


@pytest.mark.parametrize(
    ("raw_path", "raw_value", "inner_member"),
    [
        ("", 3, None),
        ("a", True, None),
        ("a", 3.0, None),
        ("a", "3", None),
        ("a", 3, -1),
        ("a", 3, True),
    ],
)
def test_invalid_raw_selections_are_rejected(raw_path, raw_value, inner_member):
    with pytest.raises(TrainingLengthError):
        RawTrainingLengthSelection(
            raw_path=raw_path, raw_value=raw_value, inner_member=inner_member
        )


def test_declaration_rejects_values_its_raw_meaning_forbids():
    """선언 시점에는 통과해도 좌표를 채울 때 원시 의미 검증을 다시 받는다."""
    declaration = LIGHTGBM.declare(
        [RawTrainingLengthSelection(raw_path="best_iteration_", raw_value=0)]
    )
    with pytest.raises(TrainingLengthError, match="횟수는 1 이상"):
        observe_declaration(declaration, seed=42, outer_fold=0)


def test_first_position_zero_becomes_length_one():
    declaration = LOOKUP.declare(
        [RawTrainingLengthSelection(raw_path="best_epoch", raw_value=0)]
    )
    evidence = observe_declaration(declaration, seed=42, outer_fold=0)
    assert [item.value for item in evidence.observations] == [1]


def test_only_a_declaration_can_be_given_coordinates():
    with pytest.raises(TrainingLengthError, match="근거 선언 형식이 아니다"):
        observe_declaration({"model_family": "lightgbm"}, seed=42, outer_fold=0)
