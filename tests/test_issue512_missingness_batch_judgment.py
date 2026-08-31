from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/judge_issue512_missingness_propagation_batch.py")
SPEC = importlib.util.spec_from_file_location("judge_issue512_missingness_batch", SCRIPT)
JUDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(JUDGE)


def test_refit_rehearsal_uses_one_training_unit_instead_of_the_final_budget(
    monkeypatch, tmp_path: Path
):
    member = SimpleNamespace(
        config="augmented",
        budgets={42: 15},
        entry_sha256="member-sha",
    )
    executable = SimpleNamespace(members=(member,))
    calls = []

    class LoadedPlan:
        def validate_for_refit(self, **kwargs):
            return executable

    def rehearse_member(plan, selected, output, *, seed, data_root):
        calls.append((plan, selected, seed, data_root))
        member_dir = output / selected.config
        member_dir.mkdir(parents=True)
        prediction_path = member_dir / "test_pred_seed_42.parquet"
        prediction = np.array([0.25, 0.75], dtype=np.float64)
        pd.DataFrame({"id": [1, 2], "pred": prediction}).to_parquet(
            prediction_path,
            index=False,
        )
        prediction_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "execution_mode": "full_data_rehearsal",
                    "planned_training_budget": 15,
                    "training_budget": 1,
                    "rehearsal_budget_rule": "one_iteration_or_non_iterative",
                    "prediction_sha256": JUDGE.prediction_array_sha256(prediction),
                    "training_rows": {
                        "coordinate_scope": "full_data",
                        "training_row_count": 30,
                        "state_fit_row_count": 10,
                        "assertions": {"replicas_excluded_from_state_fit": True},
                    },
                }
            )
            + "\n"
        )
        return prediction_path

    monkeypatch.setattr(JUDGE.RefitPlan, "load", lambda path: LoadedPlan())
    monkeypatch.setattr(JUDGE.Pool, "load", lambda path: object())
    monkeypatch.setattr(
        JUDGE,
        "file_sha256",
        lambda path: "pool-sha" if Path(path).name == "pool.yaml" else "record-sha",
    )
    monkeypatch.setattr(JUDGE.refit_module, "rehearse_member", rehearse_member)
    monkeypatch.setattr(
        JUDGE.refit_module,
        "run_member",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("최종 재학습 실행기가 예행에서 호출됐다.")
        ),
    )

    result = JUDGE._refit_rehearsal(
        source_root=tmp_path,
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

    assert len(calls) == 1
    assert calls[0][2] == 42
    assert result["schema"].endswith("/2")
    assert result["results"][0]["planned_training_budget"] == 15
    assert result["results"][0]["rehearsal_training_budget"] == 1

