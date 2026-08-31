from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path("scripts/judge_issue512_missingness_propagation_batch.py")
SPEC = importlib.util.spec_from_file_location("judge_issue512_missingness_batch", SCRIPT)
JUDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(JUDGE)


def test_refit_readiness_validates_the_plan_without_training(monkeypatch):
    member = SimpleNamespace(
        config="augmented",
        config_path=Path("configs/augmented.yaml"),
        budgets={42: 15},
        entry_sha256="member-sha",
    )
    executable = SimpleNamespace(members=(member,))
    validations = []

    class LoadedPlan:
        def validate_for_refit(self, **kwargs):
            validations.append(kwargs)
            return executable

    monkeypatch.setattr(JUDGE.RefitPlan, "load", lambda path: LoadedPlan())
    monkeypatch.setattr(JUDGE.Pool, "load", lambda path: object())
    monkeypatch.setattr(
        JUDGE,
        "file_sha256",
        lambda path: "pool-sha",
    )

    result = JUDGE._refit_readiness(
        precommit={
            "pairs": [
                {
                    "ordinal": 1,
                    "comparison_arms": [
                        {"arm": "missingness_augmented", "name": "augmented"}
                    ],
                }
            ]
        },
        search={"proposal": {"selected_ordinals": [1]}},
        selection={"selection_evidence_sha256": "selection-sha"},
        proposal_pool={"schema": "pool"},
        proposal_plan={"source_pool_sha256": "pool-sha"},
        store=object(),
        recorded_at_utc="2026-08-30T09:22:30+00:00",
    )

    assert len(validations) == 1
    assert result["schema"].endswith("/1")
    assert result["model_training_executed"] is False
    assert result["members"][0]["planned_budgets"] == {"42": 15}
