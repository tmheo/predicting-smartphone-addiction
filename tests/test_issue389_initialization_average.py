import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


OFFSETS_8 = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]


@pytest.mark.parametrize(
    ("baseline_path", "candidate_path", "candidate_name"),
    [
        (
            "configs/exp131_lookup_bivariate_plr5.yaml",
            "configs/exp156_lookup_bivariate_plr5_initavg8.yaml",
            "exp156_lookup_bivariate_plr5_initavg8",
        ),
        (
            "configs/exp127_lookup_muon.yaml",
            "configs/exp157_lookup_muon_initavg8.yaml",
            "exp157_lookup_muon_initavg8",
        ),
        (
            "configs/exp081_lookup_fold_initialization_avg3.yaml",
            "configs/exp158_lookup_fold_initialization_avg8.yaml",
            "exp158_lookup_fold_initialization_avg8",
        ),
    ],
)
def test_issue389_candidates_change_only_initialization_offsets(
    baseline_path: str, candidate_path: str, candidate_name: str
) -> None:
    baseline = yaml.safe_load(Path(baseline_path).read_text())
    candidate = yaml.safe_load(Path(candidate_path).read_text())

    assert candidate["name"] == candidate_name
    assert candidate["model"]["params"]["fold_seed_offsets"] == OFFSETS_8

    expected = deepcopy(baseline)
    expected["name"] = candidate_name
    expected["model"]["params"]["fold_seed_offsets"] = OFFSETS_8
    assert candidate == expected


def test_issue389_adoption_records_match_confirmed_results() -> None:
    champion = yaml.safe_load(Path("artifacts/champion.yaml").read_text())
    pool = yaml.safe_load(Path("artifacts/pool.yaml").read_text())
    first = yaml.safe_load(
        Path(
            "artifacts/judgments/issue389-exp157-lookup-muon-initavg8-replacement.yaml"
        ).read_text()
    )
    second = yaml.safe_load(
        Path(
            "artifacts/judgments/issue389-exp158-lookup-initavg8-replacement.yaml"
        ).read_text()
    )

    assert champion["run_id"] == "6911a461866b43dc9556553eba6783b7"
    assert champion["config"] == "exp156_lookup_bivariate_plr5_initavg8"
    assert champion["oof_auc"] == pytest.approx(0.9693676105620948)
    assert set(champion["seed_aucs"]) == {42, 43, 44}

    run_ids = {member["run_id"] for member in pool["members"]}
    assert "bb7be9baf1b64888818600d7e0b5927b" in run_ids
    assert "7124425b5b51421dbbeba597229554da" not in run_ids
    assert "d55d1cd49c194eb8bf7b5128e548df81" in run_ids
    assert "b8ea9ece449c486fb072b427b75e6003" not in run_ids

    assert first["result"]["state"] == "adopted"
    assert first["nested_oof_comparison"]["delta"] == pytest.approx(
        1.3623247564487073e-06
    )
    assert second["result"]["state"] == "rejected"
    assert second["nested_oof_comparison"]["delta"] == pytest.approx(
        -3.848426621821943e-07
    )


def test_issue389_refit_plan_tracks_the_adopted_pool_replacement() -> None:
    pool_path = Path("artifacts/pool.yaml")
    pool = yaml.safe_load(pool_path.read_text())
    plan = yaml.safe_load(Path("artifacts/full-refit-plan.yaml").read_text())

    assert plan["source_pool_sha256"] == hashlib.sha256(pool_path.read_bytes()).hexdigest()
    assert [member["config"] for member in plan["members"]] == [
        member["config"] for member in pool["members"]
    ]
    assert sum(
        len(member["refit_budget_derivation"]["seeds"])
        for member in plan["members"]
    ) == 94

    members = {member["config"]: member for member in plan["members"]}
    assert "exp127_lookup_muon" not in members
    replacement = members["exp157_lookup_muon_initavg8"]
    assert replacement["lineage"]["source_run_id"] == (
        "bb7be9baf1b64888818600d7e0b5927b"
    )
    observations = replacement["training_length_evidence"]["observations"]
    assert len(observations) == 120
    assert {item["inner_member"] for item in observations} == set(range(8))
    assert {
        seed["seed"]: seed["budget"]
        for seed in replacement["refit_budget_derivation"]["seeds"]
    } == {42: 14, 43: 15, 44: 15}
