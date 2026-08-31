from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pipeline.data import file_sha256
from pipeline.runs import RunMeta


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "record_initial_score_pair.py"
SPEC = importlib.util.spec_from_file_location("record_initial_score_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _LegacyRunStore:
    def __init__(self, config: dict, config_sha256: str) -> None:
        self._config = config
        self._config_sha256 = config_sha256

    def facts_of(self, run_id: str) -> RunMeta:
        return RunMeta(
            run_id=run_id,
            run_name=self._config["name"],
            params={},
            metrics={},
            tags={
                "remote.provider": "kaggle",
                "sha256.train": "train",
                "sha256.test": "test",
                "sha256.folds": "folds",
            },
        )

    def config_of(self, run_id: str) -> dict:
        return self._config

    def artifact_sha256_of(self, run_id: str, name: str) -> str:
        return self._config_sha256


def test_legacy_reproduction_run_preserves_incomplete_runtime_identity(monkeypatch) -> None:
    precommit = json.loads(
        (REPO_ROOT / "artifacts/issue520-initial-score-extension-precommit.json").read_text()
    )
    frozen = next(pair for pair in precommit["pairs"] if pair["key"] == "lightgbm_fixed20")[
        "baseline"
    ]
    config_path = REPO_ROOT / frozen["path"]
    config = yaml.safe_load(config_path.read_text())
    store = _LegacyRunStore(config, file_sha256(config_path))
    monkeypatch.setattr(
        MODULE,
        "load_run_facts",
        lambda run_id, run_store: SimpleNamespace(
            experiment=config["name"],
            seeds=[42, 43, 44],
            auc_oof=0.0,
            seed_aucs={42: 0.0, 43: 0.0, 44: 0.0},
            fold_aucs={},
            git_commit="legacy",
            git_dirty=False,
        ),
    )

    _, identity = MODULE._run_identity(
        store,
        "legacy-run",
        frozen,
        initial_score_required=False,
        require_complete_runtime_identity=False,
    )

    assert identity["runtime_identity"] == {
        "provider": "kaggle",
        "runtime_class": "not_recorded",
        "record_complete": False,
    }

    with pytest.raises(ValueError, match="원격 공급자와 실행 환경 등급 기록이 비대칭"):
        MODULE._run_identity(
            store,
            "formal-pair-run",
            frozen,
            initial_score_required=False,
        )
