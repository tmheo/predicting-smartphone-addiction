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
