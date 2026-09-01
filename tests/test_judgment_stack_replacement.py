"""스택 교체 판정 정본 테스트. (지도 #550의 #551)

계약 시험은 경계값(문턱 정확히 같음, 5/5·4/5, >= 표준화가 3/5 설정에서도 일관),
판정 불가 상태, to_record 키 안정성을 겨냥한다. golden 시험은 동결된 #513 판정
기록(docs/research/extended-stack-pool-reassembly/issue513/comparison.json)의
fold AUC를 입력으로 같은 판정 결과를 재현한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pipeline.judgment import (
    AUC_THRESHOLD,
    STACK_VERDICT_FAIL,
    STACK_VERDICT_PASS,
    STACK_VERDICT_UNDECIDABLE,
    FoldScores,
    JudgmentError,
    StackGate,
    judge_stack_replacement,
)

ISSUE513_COMPARISON = (
    Path(__file__).parents[1]
    / "docs/research/extended-stack-pool-reassembly/issue513/comparison.json"
)


def make_scores(nested_auc: float, fold_deltas: dict[int, float]) -> FoldScores:
    base = 0.97
    return FoldScores(
        nested_auc=nested_auc,
        fold_aucs={fold: base + delta for fold, delta in fold_deltas.items()},
    )


def make_reference(folds: int = 5) -> FoldScores:
    return make_scores(0.97, {fold: 0.0 for fold in range(folds)})


def test_default_gate_reuses_auc_threshold() -> None:
    assert StackGate().delta_required == AUC_THRESHOLD
    assert StackGate().folds_required_positive == 5


def test_delta_exactly_at_threshold_passes() -> None:
    candidate = make_scores(0.97 + AUC_THRESHOLD, {fold: 1e-6 for fold in range(5)})
    verdict = judge_stack_replacement(candidate, make_reference(), StackGate())
    assert verdict.status == STACK_VERDICT_PASS
    assert verdict.passes_gate
    assert verdict.gate_delta_passes and verdict.gate_folds_passes
    assert verdict.delta == pytest.approx(AUC_THRESHOLD)


def test_delta_below_threshold_fails() -> None:
    candidate = make_scores(
        0.97 + AUC_THRESHOLD - 1e-9, {fold: 1e-6 for fold in range(5)}
    )
    verdict = judge_stack_replacement(candidate, make_reference(), StackGate())
    assert verdict.status == STACK_VERDICT_FAIL
    assert not verdict.passes_gate
    assert not verdict.gate_delta_passes
    assert verdict.gate_folds_passes


def test_four_of_five_folds_fails_with_five_required() -> None:
    deltas = {0: 1e-6, 1: 1e-6, 2: 1e-6, 3: 1e-6, 4: -1e-6}
    candidate = make_scores(0.97 + AUC_THRESHOLD, deltas)
    verdict = judge_stack_replacement(candidate, make_reference(), StackGate())
    assert verdict.status == STACK_VERDICT_FAIL
    assert verdict.folds_positive == 4
    assert verdict.gate_delta_passes
    assert not verdict.gate_folds_passes


def test_zero_fold_delta_is_not_positive() -> None:
    deltas = {0: 1e-6, 1: 1e-6, 2: 1e-6, 3: 1e-6, 4: 0.0}
    candidate = make_scores(0.97 + AUC_THRESHOLD, deltas)
    verdict = judge_stack_replacement(candidate, make_reference(), StackGate())
    assert verdict.folds_positive == 4
    assert verdict.status == STACK_VERDICT_FAIL


def test_ge_standardization_passes_surplus_folds_under_three_required() -> None:
    # 요구 수를 3/5로 낮춰도 >= 비교라 4/5·5/5가 전부 통과한다(== 표준화 금지).
    gate = StackGate(folds_required_positive=3)
    for positive in (3, 4, 5):
        deltas = {
            fold: (1e-6 if fold < positive else -1e-6) for fold in range(5)
        }
        candidate = make_scores(0.97 + AUC_THRESHOLD, deltas)
        verdict = judge_stack_replacement(candidate, make_reference(), gate)
        assert verdict.status == STACK_VERDICT_PASS, positive
    deltas = {fold: (1e-6 if fold < 2 else -1e-6) for fold in range(5)}
    candidate = make_scores(0.97 + AUC_THRESHOLD, deltas)
    assert (
        judge_stack_replacement(candidate, make_reference(), gate).status
        == STACK_VERDICT_FAIL
    )


def test_empty_fold_aucs_is_undecidable() -> None:
    candidate = FoldScores(nested_auc=0.97, fold_aucs={})
    verdict = judge_stack_replacement(candidate, make_reference(), StackGate())
    assert verdict.status == STACK_VERDICT_UNDECIDABLE
    assert verdict.undecidable_reason is not None
    assert verdict.delta is None
    assert not verdict.passes_gate
    with pytest.raises(JudgmentError, match="판정 불가"):
        verdict.require_decidable()


def test_fold_set_mismatch_is_undecidable() -> None:
    candidate = make_scores(0.971, {fold: 1e-6 for fold in range(4)})
    verdict = judge_stack_replacement(candidate, make_reference(5), StackGate())
    assert verdict.status == STACK_VERDICT_UNDECIDABLE
    assert "분할 구성" in verdict.undecidable_reason


def test_nan_auc_is_undecidable() -> None:
    candidate = make_scores(math.nan, {fold: 1e-6 for fold in range(5)})
    verdict = judge_stack_replacement(candidate, make_reference(), StackGate())
    assert verdict.status == STACK_VERDICT_UNDECIDABLE
    assert "NaN" in verdict.undecidable_reason


def test_decidable_verdict_passes_require_decidable() -> None:
    candidate = make_scores(0.97 + AUC_THRESHOLD, {fold: 1e-6 for fold in range(5)})
    verdict = judge_stack_replacement(candidate, make_reference(), StackGate())
    verdict.require_decidable()  # 예외 없음.


def test_to_record_key_stability() -> None:
    candidate = make_scores(0.97 + AUC_THRESHOLD, {fold: 1e-6 for fold in range(5)})
    record = judge_stack_replacement(candidate, make_reference(), StackGate()).to_record()
    assert list(record) == [
        "status",
        "undecidable_reason",
        "delta",
        "delta_minus_gate",
        "fold_deltas",
        "folds_positive",
        "gate",
        "gate_delta_passes",
        "gate_folds_passes",
        "passes_gate",
    ]
    assert list(record["gate"]) == ["delta_required", "folds_required_positive"]
    assert list(record["fold_deltas"]) == ["0", "1", "2", "3", "4"]
    assert record["delta_minus_gate"] == pytest.approx(0.0)
    assert record["passes_gate"] is True


def test_to_record_undecidable_has_null_numbers() -> None:
    candidate = FoldScores(nested_auc=0.97, fold_aucs={})
    record = judge_stack_replacement(candidate, make_reference(), StackGate()).to_record()
    assert record["status"] == STACK_VERDICT_UNDECIDABLE
    assert record["delta"] is None
    assert record["delta_minus_gate"] is None
    assert record["fold_deltas"] == {}
    assert record["folds_positive"] is None
    assert record["passes_gate"] is False


def test_golden_issue513_comparison_reproduced() -> None:
    recorded = json.loads(ISSUE513_COMPARISON.read_text())
    reference = FoldScores(
        nested_auc=recorded["baseline"]["nested_auc"],
        fold_aucs={int(k): v for k, v in recorded["baseline"]["fold_aucs"].items()},
    )
    candidate = FoldScores(
        nested_auc=recorded["reassembled"]["nested_auc"],
        fold_aucs={int(k): v for k, v in recorded["reassembled"]["fold_aucs"].items()},
    )
    verdict = judge_stack_replacement(candidate, reference, StackGate())

    assert verdict.status == STACK_VERDICT_PASS
    assert verdict.delta == recorded["delta_vs_current_submission"]
    assert verdict.folds_positive == recorded["folds_positive"]
    assert verdict.gate_delta_passes == recorded["gate_delta_passes"]
    assert verdict.gate_folds_passes == recorded["gate_folds_passes"]
    assert verdict.passes_gate == recorded["passes_gate"]
    record = verdict.to_record()
    assert record["fold_deltas"] == recorded["fold_deltas"]
    assert record["delta_minus_gate"] == recorded["delta_minus_gate"]
