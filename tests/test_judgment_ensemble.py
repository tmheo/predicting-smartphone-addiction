"""계열 3 판정(judge_ensemble)의 규칙 테스트. (ADR 0001 계열 3, #104)

- 채택 가능: nested OOF AUC가 champion 대비 +0.00002 이상.
- 확정: 채택 가능한 전략 중 nested OOF AUC 최고 1개.
  채택 가능 전략이 없으면 채택 없음.
- 전략 간 fold별 승리 수는 보조 증거로 기록만 한다.
"""

from __future__ import annotations

import pytest

from pipeline.judgment import JudgmentError, StrategyOutcome, judge_ensemble

CHAMPION_AUC = 0.96900
FOLDS = [0, 1, 2, 3, 4]


def outcome(
    name: str,
    nested_auc: float,
    fold_aucs: dict[int, float] | None = None,
) -> StrategyOutcome:
    return StrategyOutcome(
        name=name,
        nested_auc=nested_auc,
        fold_aucs=fold_aucs or {f: nested_auc for f in FOLDS},
    )


def test_adoption_recommends_clear_winner():
    verdict = judge_ensemble(
        [outcome("rank_mean", 0.96901), outcome("ridge_logit", 0.96905)],
        champion_auc=CHAMPION_AUC,
    )
    assert [a.name for a in verdict.assessments] == ["ridge_logit", "rank_mean"]
    assert verdict.assessments[0].eligible
    assert not verdict.assessments[1].eligible  # +0.00001은 문턱 미만.
    assert verdict.recommended == "ridge_logit"


def test_no_eligible_strategy_keeps_solo_champion():
    verdict = judge_ensemble(
        [outcome("rank_mean", 0.96880), outcome("ridge_logit", 0.96901)],
        champion_auc=CHAMPION_AUC,
    )
    assert verdict.recommended is None


def test_close_scores_still_recommend_highest_nested_auc():
    # 차이가 0.00002보다 작아도 별도 동률 처리 없이 최고 nested OOF AUC를 선택한다.
    verdict = judge_ensemble(
        [
            outcome("a", 0.96906),
            outcome("b", 0.969045),
            outcome("c", 0.96903),
        ],
        champion_auc=CHAMPION_AUC,
    )
    assert verdict.recommended == "a"


def test_fold_wins_are_auxiliary_and_strict():
    fold_a = {0: 0.9, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.7}
    fold_b = {0: 0.8, 1: 0.8, 2: 0.9, 3: 0.7, 4: 0.6}  # fold 3은 동점이라 무승부.
    verdict = judge_ensemble(
        [outcome("a", 0.96700, fold_a), outcome("b", 0.96690, fold_b)],
        champion_auc=CHAMPION_AUC,
    )
    wins = {a.name: a.fold_wins for a in verdict.assessments}
    assert wins == {"a": 3, "b": 1}
    # 승리 수는 판정에 영향을 주지 않는다: 둘 다 미달이므로 채택 없음.
    assert verdict.recommended is None


def test_rejects_empty_or_duplicated_strategies():
    with pytest.raises(JudgmentError, match="없다"):
        judge_ensemble([], champion_auc=CHAMPION_AUC)
    with pytest.raises(JudgmentError, match="중복"):
        judge_ensemble(
            [outcome("a", 0.9), outcome("a", 0.9)], champion_auc=CHAMPION_AUC
        )


def test_rejects_mismatched_fold_sets():
    with pytest.raises(JudgmentError, match="fold 구성"):
        judge_ensemble(
            [
                outcome("a", 0.9),
                outcome("b", 0.9, {f: 0.9 for f in [0, 1, 2]}),
            ],
            champion_auc=CHAMPION_AUC,
        )
