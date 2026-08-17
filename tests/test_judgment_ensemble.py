"""계열 3 판정(judge_ensemble)의 규칙 테스트. (ADR 0001 계열 3, #104)

- 채택 가능: nested OOF AUC가 champion 대비 +0.00002 이상.
- 동률 그룹: 1위 고정 기준, 1위와의 차이가 0.00002 미만인 채택 가능 전략만
  (연쇄 확장 없음).
- 확정: 동률 그룹에서 복잡도 서열 최저 1개. 채택 가능 전략이 없으면 채택 없음.
- 전략 간 fold별 승리 수는 보조 증거로 기록만 한다.
"""

from __future__ import annotations

import pytest

from pipeline.judgment import JudgmentError, StrategyOutcome, judge_ensemble

CHAMPION_AUC = 0.96900
FOLDS = [0, 1, 2, 3, 4]


def outcome(
    name: str,
    complexity: int,
    nested_auc: float,
    fold_aucs: dict[int, float] | None = None,
) -> StrategyOutcome:
    return StrategyOutcome(
        name=name,
        complexity=complexity,
        nested_auc=nested_auc,
        fold_aucs=fold_aucs or {f: nested_auc for f in FOLDS},
    )


def test_adoption_recommends_clear_winner():
    verdict = judge_ensemble(
        [outcome("rank_mean", 1, 0.96901), outcome("ridge_logit", 4, 0.96905)],
        champion_auc=CHAMPION_AUC,
    )
    assert [a.name for a in verdict.assessments] == ["ridge_logit", "rank_mean"]
    assert verdict.assessments[0].eligible
    assert not verdict.assessments[1].eligible  # +0.00001은 문턱 미만.
    assert verdict.tie_group == ["ridge_logit"]
    assert verdict.recommended == "ridge_logit"


def test_no_eligible_strategy_keeps_solo_champion():
    verdict = judge_ensemble(
        [outcome("rank_mean", 1, 0.96880), outcome("ridge_logit", 4, 0.96901)],
        champion_auc=CHAMPION_AUC,
    )
    assert verdict.recommended is None
    assert verdict.tie_group == []


def test_tie_resolved_by_lowest_complexity():
    # 1위 ridge와 0.00001 차이(0.00002 미만)면 동률이고, 복잡도 서열이 낮은 쪽이 이긴다.
    verdict = judge_ensemble(
        [outcome("rank_mean", 1, 0.96904), outcome("ridge_logit", 4, 0.96905)],
        champion_auc=CHAMPION_AUC,
    )
    assert verdict.tie_group == ["ridge_logit", "rank_mean"]
    assert verdict.recommended == "rank_mean"


def test_tie_group_is_anchored_to_top_without_chaining():
    # b는 1위 a와 0.000015 차이로 동률이지만, c는 1위와 0.00003 차이라 b와 가까워도
    # 그룹에 들어가지 않는다(연쇄 확장 없음).
    verdict = judge_ensemble(
        [
            outcome("a", 3, 0.96906),
            outcome("b", 2, 0.969045),
            outcome("c", 1, 0.96903),
        ],
        champion_auc=CHAMPION_AUC,
    )
    assert verdict.tie_group == ["a", "b"]
    assert verdict.recommended == "b"


def test_tie_group_holds_only_eligible_strategies():
    # 1위와 0.00002 미만 차이라도 채택 가능 문턱에 못 미치면 동률 그룹에 못 든다.
    verdict = judge_ensemble(
        [outcome("rank_mean", 1, 0.969015), outcome("ridge_logit", 4, 0.96903)],
        champion_auc=CHAMPION_AUC,
    )
    assert verdict.tie_group == ["ridge_logit"]
    assert verdict.recommended == "ridge_logit"


def test_fold_wins_are_auxiliary_and_strict():
    fold_a = {0: 0.9, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.7}
    fold_b = {0: 0.8, 1: 0.8, 2: 0.9, 3: 0.7, 4: 0.6}  # fold 3은 동점이라 무승부.
    verdict = judge_ensemble(
        [outcome("a", 1, 0.96700, fold_a), outcome("b", 2, 0.96690, fold_b)],
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
            [outcome("a", 1, 0.9), outcome("a", 2, 0.9)], champion_auc=CHAMPION_AUC
        )


def test_rejects_mismatched_fold_sets():
    with pytest.raises(JudgmentError, match="fold 구성"):
        judge_ensemble(
            [
                outcome("a", 1, 0.9),
                outcome("b", 2, 0.9, {f: 0.9 for f in [0, 1, 2]}),
            ],
            champion_auc=CHAMPION_AUC,
        )
