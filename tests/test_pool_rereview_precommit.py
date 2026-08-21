"""실제 제거 결과를 보기 전에 #341 장부의 경계를 고정하는 특성화 시험."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


SCRIPT = Path("scripts/validate_pool_rereview_precommit.py")
SPEC = importlib.util.spec_from_file_location("validate_pool_rereview_precommit", SCRIPT)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_frozen_precommit_matches_baseline_and_strategy_registry():
    notes = VALIDATOR.validate()

    assert any("후보 35개" in note for note in notes)
    assert any("결합 전략 19개" in note for note in notes)
    assert any("영점 대조 210건" in note for note in notes)


def test_validator_rejects_candidate_order_change(tmp_path):
    payload = yaml.safe_load(VALIDATOR.DEFAULT_LEDGER.read_text(encoding="utf-8"))
    members = payload["candidate_pool"]["members"]
    members[0], members[1] = members[1], members[0]
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(VALIDATOR.PrecommitValidationError, match="후보와 순서"):
        VALIDATOR.validate(changed)


def test_validator_rejects_strategy_removal(tmp_path):
    payload = yaml.safe_load(VALIDATOR.DEFAULT_LEDGER.read_text(encoding="utf-8"))
    payload["strategies"]["included"].pop()
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(VALIDATOR.PrecommitValidationError, match="기본 결합 전략"):
        VALIDATOR.validate(changed)
