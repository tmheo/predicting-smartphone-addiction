from pathlib import Path

import pytest
import yaml

from pipeline.data import file_sha256
from pipeline.ledger import Pool
from pipeline.refit_plan import RefitPlan


RESULTS_PATH = Path("artifacts/issue388-bulk-selection-results.yaml")
ROUND_1_PATH = Path("artifacts/judgments/issue388-bulk-selection-2.yaml")
ROUND_2_PATH = Path("artifacts/judgments/issue388-bulk-selection-3.yaml")
EXP144_RUN_ID = "89e3913d74a1490792f19e283989116e"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_issue388_first_round_registers_one_boundary_contribution() -> None:
    record = _load(ROUND_1_PATH)

    assert record["contract_version"] == "candidate-pool-v1"
    assert record["selection"]["kind"] == "nested_selection"
    assert len(record["selection"]["candidates"]) == 7
    assert record["frozen_input"]["candidate_pool"]["member_count"] == 32
    assert record["change"]["candidate"] == {
        "run_id": EXP144_RUN_ID,
        "config": "exp144_issue387_xgb_trial6",
        "model_lineage_group": "issue387-bulk-tree-linear",
    }
    comparison = record["nested_oof_comparison"]
    assert comparison["delta"] == pytest.approx(0.000026956543791922805)
    assert comparison["outer_fold_wins"] == 5
    assert all(value > 0.0 for value in comparison["outer_fold_delta"].values())
    assert comparison["boundary_contribution"] is True
    assert record["result"]["state"] == "adopted"


def test_issue388_second_round_stops_serial_registration() -> None:
    record = _load(ROUND_2_PATH)

    assert record["selection"]["kind"] == "nested_selection"
    assert len(record["selection"]["candidates"]) == 6
    assert record["frozen_input"]["candidate_pool"]["member_count"] == 33
    assert record["change"]["candidate"]["config"] == "exp151_issue387_cat_trial3"
    comparison = record["nested_oof_comparison"]
    assert comparison["delta"] == pytest.approx(-0.00000512996689672196)
    assert comparison["outer_fold_wins"] == 1
    assert record["result"]["state"] == "rejected"
    assert record["result"]["decision"] == "do_not_admit"


def test_issue388_results_cover_every_handoff_candidate_and_record_hash() -> None:
    results = _load(RESULTS_PATH)

    assert results["issue"] == 388
    assert len(results["candidates"]) == 7
    assert results["final"]["registered_count"] == 1
    assert results["final"]["registered_candidates"] == [
        "exp144_issue387_xgb_trial6"
    ]
    statuses = {
        candidate["name"]: candidate["final_status"]
        for candidate in results["candidates"]
    }
    assert statuses["exp144_issue387_xgb_trial6"] == "registered"
    assert set(statuses.values()) == {
        "registered",
        "not_registered_after_procedure_rejection",
    }
    for round_result in results["rounds"]:
        assert file_sha256(Path(round_result["record"]["path"])) == round_result[
            "record"
        ]["sha256"]


def test_issue388_pool_and_refit_plan_are_aligned() -> None:
    results = _load(RESULTS_PATH)
    pool = Pool.load()
    plan = RefitPlan.load(Path("artifacts/full-refit-plan.yaml"))

    # #512가 결측 증강 교체 6개와 초기 점수 후보 1개를 반영해
    # 현재 풀은 36개, 재학습 106회다.
    assert len(pool.members) == 36
    assert results["final"]["final_pool"]["sha256"] == (
        "e430d32dc6d80b7feb34151ed93d431adc42dc9b61ca344368218768c30fc349"
    )
    assert "exp144_issue387_xgb_trial6" not in {
        member.config for member in pool.members
    }
    assert all(
        member.config not in {candidate["name"] for candidate in results["candidates"][1:]}
        for member in pool.members
    )
    assert plan.source_pool_sha256 == file_sha256(Path("artifacts/pool.yaml"))
    assert [(member.config, member.lineage.source_run_id) for member in plan.members] == [
        (member.config, member.run_id) for member in pool.members
    ]
    assert len(plan.members) == 36
    assert sum(len(member.budget_derivation.seeds) for member in plan.members) == 106
    correction = results["training_length_correction"]
    assert correction["pool_member"]["membership_changed"] is False
    assert correction["remeasurement"]["run_id"] == (
        "49e1433f3696419b9a5f4b7fbae7efc6"
    )
    assert correction["corrected_refit_budget_by_seed"] == {
        42: 65238,
        43: 63411,
        44: 63504,
    }
    assert correction[
        "current_full_refit_plan"
    ]["sha256"] == "f554327283e03d1be67fa954ebac556eef17722fd2a64ade1d86ab3abc1d1ffe"

    removal = results["pool_removal"]
    assert removal["member"] == {
        "config": "exp144_issue387_xgb_trial6",
        "run_id": EXP144_RUN_ID,
    }
    assert removal["nested_oof_impact"]["delta_after_minus_before"] == pytest.approx(
        -0.0000257162076638
    )
    # 제거 직후의 장부 해시는 결과 파일에 기록된 값이며, 이후 진입으로 현재 파일과는 다르다.
    assert removal["final_pool"]["member_count"] == 32
    assert removal["final_pool"]["sha256"] == (
        "c273ad60a4747740340cc9353312896c5cfb7f42b94f3cf9e2b51bd0b58ead8b"
    )
    assert removal["full_refit_plan"]["refit_count"] == 94
    assert removal["full_refit_plan"]["sha256"] == (
        "31982216b16460e4e5e77789b322b112afec349bf3a9c1f795f54ba56347dbac"
    )
    assert removal["judgment"]["sha256"] == file_sha256(
        Path(removal["judgment"]["path"])
    )
