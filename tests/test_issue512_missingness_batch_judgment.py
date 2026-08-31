from __future__ import annotations

import importlib.util
import sys
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


def test_verify_only_does_not_revalidate_the_replaced_frozen_ledgers(
    monkeypatch, tmp_path: Path
):
    precommit = {"precommit_sha256": "precommit-sha"}
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--verify-only",
            "--output-root",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(JUDGE, "_load_json", lambda path: precommit)
    monkeypatch.setattr(
        JUDGE,
        "verify_self_hash",
        lambda payload, field: calls.append(("self_hash", payload, field)),
    )
    monkeypatch.setattr(
        JUDGE,
        "_verify",
        lambda output_root, payload: calls.append(("verify", output_root, payload)),
    )
    monkeypatch.setattr(
        JUDGE,
        "validate_precommit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("공식화 뒤 현재 장부를 동결 전 해시로 다시 검사했다.")
        ),
    )

    JUDGE.main()

    assert calls == [
        ("self_hash", precommit, "precommit_sha256"),
        ("verify", tmp_path.resolve(), precommit),
    ]
