"""실행 로그에서 원시 학습 길이를 복원하는 읽기부의 계약 테스트. (#374)

복원 프로그램의 값어치는 "로그가 확정하지 못하는 좌표를 확정한 척하지 않는다"에 있다.
그래서 여기서는 계열별 읽기부가 좌표 수를 맞추는지, 표시 정밀도 안에서 동점인 좌표에
후보를 둘 다 남기는지, 확정 근거가 그 후보 안에 있을 때만 받아들이는지를 본다.

실행 저장소가 있어야 하는 부분은 보지 않는다. 그 관문은
`uv run --frozen python -m pipeline.refit_plan artifacts/full-refit-plan.yaml --validate-only`가
실제 저장소에서 본다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location(
        "recover_training_length_evidence",
        REPO / "scripts" / "recover_training_length_evidence.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    # dataclass의 미룬 주석을 풀려면 모듈이 sys.modules에 있어야 한다.
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_lightgbm_reads_both_stopping_messages(module):
    """조기 종료로 멈춘 fold와 최대 반복까지 간 fold는 서로 다른 문구를 쓴다."""
    lines = [
        "Early stopping, best iteration is:",
        "[312]\tvalid_0's auc: 0.965512",
        "Did not meet early stopping. Best iteration is:",
        "[19993]\tvalid_0's auc: 0.968121",
    ]

    assert module.LOG_READERS["lightgbm"](lines) == [[[312]], [[19993]]]


def test_tagged_readers_take_the_recorded_position(module):
    catboost = ["[catboost] early stopping: best_iteration=4213 best_score={'AUC': 0.96}"]
    xgboost = ["[xgboost] early stopping: best_iteration=1489 best_score=0.9675"]

    assert module.LOG_READERS["catboost"](catboost) == [[[4213]]]
    assert module.LOG_READERS["xgboost"](xgboost) == [[[1489]]]


def _lookup_fold(points: list[tuple[int, str]]) -> list[str]:
    return ["UserWarning: enable_nested_tensor is True"] + [
        f"[lookup_transformer] ep {epoch:2d} valAUC={score} best={score}"
        for epoch, score in points
    ]


def test_a_display_precision_tie_leaves_both_candidates(module):
    """로그가 소수점 다섯 자리까지만 남기면 두 epoch가 같은 값으로 찍힌다."""
    lines = _lookup_fold([(9, "0.96761"), (11, "0.96797"), (13, "0.96797")])

    assert module.LOG_READERS["lookup_single"](lines) == [[[11, 13]]]


def test_a_single_best_leaves_one_candidate(module):
    lines = _lookup_fold([(9, "0.96761"), (11, "0.96797"), (13, "0.96795")])

    assert module.LOG_READERS["lookup_single"](lines) == [[[11]]]


def test_fold_initialization_members_keep_their_own_curves(module):
    lines = [
        "[lookup_transformer] member 1/2 seed=42 device=cuda:0",
        "[lookup_transformer] member 2/2 seed=1042 device=cuda:1",
        "[lookup_transformer] seed=42 ep  9 valAUC=0.96788 best=0.96788",
        "[lookup_transformer] seed=1042 ep  9 valAUC=0.96788 best=0.96788",
        "[lookup_transformer] seed=42 ep 11 valAUC=0.96822 best=0.96822",
        "[lookup_transformer] seed=1042 ep 11 valAUC=0.96780 best=0.96788",
    ]

    assert module.LOG_READERS["lookup_members"](lines) == [[[11], [9]]]


def test_an_empty_curve_is_refused(module):
    with pytest.raises(module.RecoveryError):
        module.LOG_READERS["lookup_single"](["UserWarning: enable_nested_tensor is True"])


def test_the_confirmed_cells_of_issue_367_are_declared_for_the_two_ambiguous_members(
    module,
):
    """네 셀만 재실행으로 확정했다. 그 선언이 코드에서 사라지면 복원이 조용히 달라진다."""
    declared = {
        config: sorted((cell.seed, cell.outer_fold) for cell in recovery.confirmed)
        for config, recovery in module.RECOVERY.items()
        if recovery.confirmed
    }

    assert declared == {
        "exp059_lookup_transformer": [(42, 4), (43, 1), (44, 4)],
        "exp133_scalar_token_transformer_oof_te": [(44, 0)],
    }


def test_every_pool_member_has_a_recovery_declaration(module):
    from pipeline.ledger import Pool

    assert sorted(module.RECOVERY) == sorted(
        member.config for member in Pool.load().members
    )
