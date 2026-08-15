"""Lookup-Transformer 조회 어휘 미등록값 진단의 핵심 계산 테스트. (#128)"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_lookup_unk.py"
SPEC = importlib.util.spec_from_file_location("diagnose_lookup_unk", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnosis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnosis
SPEC.loader.exec_module(diagnosis)

auc_error_contribution = diagnosis.auc_error_contribution
profile_unknowns = diagnosis.profile_unknowns
unknown_mask = diagnosis.unknown_mask


def test_unknown_mask_distinguishes_missing_from_unseen() -> None:
    reference = pd.DataFrame({"value": [1.0, 2.0, None], "category": ["a", "b", None]})
    evaluated = pd.DataFrame({"value": [2.0, 3.0, None], "category": ["b", "c", None]})

    actual = unknown_mask(reference, evaluated, ["value", "category"])

    assert actual.to_dict(orient="list") == {
        "value": [False, True, False],
        "category": [False, True, False],
    }


def test_profile_unknowns_uses_each_fold_training_vocabulary() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fold": [0, 0, 1, 1],
            "value": [1.0, 2.0, 1.0, 3.0],
        }
    )
    test = pd.DataFrame({"id": [5, 6], "value": [1.0, 4.0]})

    details, summaries, rows = profile_unknowns(train, test, ["value"])

    validation = details[details["split"] == "validation"].set_index("fold")
    assert validation["unknown"].to_dict() == {0: 1, 1: 1}
    assert summaries.set_index("fold")["test_any_unknown"].to_dict() == {0: 1, 1: 1}
    assert rows.set_index("id")["unknown_count"].to_dict() == {1: 0, 2: 1, 3: 0, 4: 1}


def test_auc_error_contribution_is_exact_pairwise_decomposition() -> None:
    labels = pd.Series([0, 0, 1, 1])
    predictions = pd.Series([0.1, 0.8, 0.7, 0.9])
    any_unknown = pd.Series([False, True, False, False])

    actual = auc_error_contribution(labels, predictions, any_unknown)

    assert actual.related_pair_share == pytest.approx(0.5)
    assert actual.related_auc_loss == pytest.approx(0.25)
    assert actual.related_error_share == pytest.approx(1.0)
    assert actual.related_pair_error_rate == pytest.approx(0.5)


def test_auc_error_contribution_is_zero_without_unknown_rows() -> None:
    actual = auc_error_contribution(
        pd.Series([0, 1]), pd.Series([0.1, 0.9]), pd.Series([False, False])
    )

    assert actual.related_pair_share == 0.0
    assert actual.related_auc_loss == 0.0
    assert actual.related_error_share == 0.0
    assert actual.related_pair_error_rate == 0.0
