"""확정 재검증 판정 테스트. (ADR 0001, 지도 #91의 #93)

challenger는 InMemoryRunStore에 심고 load_run_facts로 읽어, metric 이름 규약
(auc_oof_seed_*, auc_fold_*)의 해석까지 함께 검증한다. 플라시보 게이트는 판정을
막지 않도록 새 피처·카나리아 없는 구성으로 고정하고, 시드 평균본 문턱·2/3 시드
게이트·경계 구간 fold 승리 게이트를 각각 겨냥한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.features import PLACEBO
from pipeline.judgment import JudgmentError, judge_confirmation, load_run_facts
from pipeline.ledger import Champion
from pipeline.runs import InMemoryRunStore

FEATURES = f"age,{PLACEBO}"


def make_champion() -> Champion:
    return Champion(
        run_id="champ",
        oof_auc=0.96700,
        seed_aucs={42: 0.96690, 43: 0.96700, 44: 0.96710},
        fold_aucs={f: 0.96700 for f in range(5)},
        config="exp_champ",
        features=set(FEATURES.split(",")),
        git_commit="cafebabe",
        adopted_at="2026-08-13",
        reason="테스트 champion",
    )


def make_challenger(
    *,
    auc_oof: float,
    seed_aucs: dict[int, float],
    fold_aucs: dict[int, float] | None = None,
):
    metrics = {"auc_oof": auc_oof}
    metrics.update({f"auc_oof_seed_{s}": v for s, v in seed_aucs.items()})
    metrics.update({f"auc_fold_{f}": v for f, v in (fold_aucs or {}).items()})
    store = InMemoryRunStore()
    store.add_run(
        "challenger",
        params={"experiment": "exp_test", "features": FEATURES, "seeds": "42,43,44"},
        metrics=metrics,
        tags={"git_commit": "deadbeef", "git_dirty": "False", "sha256.folds": "folds-sha"},
        importance=pd.DataFrame(
            {
                "feature": ["age", PLACEBO],
                "fold": [0, 0],
                "seed": [42, 42],
                "gain": [500.0, 100.0],
            }
        ),
        config={"model": {"kind": "test", "params": {}, "fit": {}}},
    )
    return load_run_facts("challenger", store)


def all_seeds_improved() -> dict[int, float]:
    return {42: 0.96790, 43: 0.96800, 44: 0.96810}


def folds(wins: int) -> dict[int, float]:
    """champion(전부 0.96700) 대비 앞에서부터 wins개 fold만 이긴 fold AUC."""
    return {f: 0.96800 if f < wins else 0.96600 for f in range(5)}


def test_confirmation_passes_outside_boundary_without_fold_gate():
    # delta +0.003은 경계 구간 밖이므로 fold 승리 2/5여도 보조 증거일 뿐 게이트가 아니다.
    challenger = make_challenger(
        auc_oof=0.97000, seed_aucs=all_seeds_improved(), fold_aucs=folds(wins=2)
    )
    verdict = judge_confirmation(make_champion(), challenger)
    assert verdict.passed
    assert not verdict.boundary
    assert verdict.fold_wins == 2 and verdict.fold_ok


def test_confirmation_requires_seed_mean_threshold():
    # 시드 평균본 delta +0.00005는 문턱 +0.0001 미만이라 CV 잡음으로 본다.
    challenger = make_challenger(
        auc_oof=0.96705, seed_aucs=all_seeds_improved(), fold_aucs=folds(wins=5)
    )
    verdict = judge_confirmation(make_champion(), challenger)
    assert not verdict.auc_ok
    assert not verdict.passed


def test_confirmation_requires_two_of_three_seed_wins():
    # 시드 42만 개선(동률은 개선이 아니다) → 1/3으로 시드 게이트 미달.
    challenger = make_challenger(
        auc_oof=0.97000,
        seed_aucs={42: 0.96695, 43: 0.96700, 44: 0.96705},
        fold_aucs=folds(wins=5),
    )
    verdict = judge_confirmation(make_champion(), challenger)
    assert verdict.seed_wins == 1
    assert not verdict.seed_ok
    assert not verdict.passed


@pytest.mark.parametrize(("fold_wins", "passed"), [(2, False), (3, True)])
def test_confirmation_boundary_adds_fold_win_gate(fold_wins, passed):
    # delta +0.00015는 경계 구간이라 fold 승리 3/5 이상을 추가로 요구한다.
    challenger = make_challenger(
        auc_oof=0.96715, seed_aucs=all_seeds_improved(), fold_aucs=folds(wins=fold_wins)
    )
    verdict = judge_confirmation(make_champion(), challenger)
    assert verdict.boundary
    assert verdict.fold_wins == fold_wins
    assert verdict.passed is passed


def test_confirmation_rejects_run_without_seed_metrics():
    challenger = make_challenger(
        auc_oof=0.97000, seed_aucs={42: 0.96790}, fold_aucs=folds(wins=5)
    )
    with pytest.raises(JudgmentError, match="시드별 OOF AUC 지표"):
        judge_confirmation(make_champion(), challenger)
