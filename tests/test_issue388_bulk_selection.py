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

    assert len(pool.members) == 33
    assert file_sha256(Path("artifacts/pool.yaml")) == results["final"]["final_pool"][
        "sha256"
    ]
    assert [member.config for member in pool.members].count(
        "exp144_issue387_xgb_trial6"
    ) == 1
    assert all(
        member.config not in {candidate["name"] for candidate in results["candidates"][1:]}
        for member in pool.members
    )
    assert plan.source_pool_sha256 == results["final"]["final_pool"]["sha256"]
    assert [(member.config, member.lineage.source_run_id) for member in plan.members] == [
        (member.config, member.run_id) for member in pool.members
    ]
    (exp144,) = [
        member
        for member in plan.members
        if member.config == "exp144_issue387_xgb_trial6"
    ]
    assert {seed.seed: seed.budget for seed in exp144.budget_derivation.seeds} == {
        42: 12500,
        43: 12500,
        44: 12500,
    }
