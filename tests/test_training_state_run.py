"""학습 상태 대조축의 부모 실행부터 세 실행 기록 묶음까지 잇는 통합 회귀 검사."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from test_model import toy_train_test
from test_training_state_cv import FakeTrainingStateAdapter

from pipeline import model as model_mod
from pipeline import tracking, training_state_run
from pipeline.bundle import export_bundle, import_bundle
from pipeline.runs import MlflowRunStore
from pipeline.training_state_manifest import (
    MANIFEST_NAME as TRAINING_STATE_MANIFEST_NAME,
    TRAINING_STATE_BUNDLE_SCHEMA_VERSION,
    validate_candidate_manifest,
    validate_candidate_parent_lineage,
)


REPO = Path(__file__).resolve().parents[1]
CANDIDATES = (6, 8, 12)
FOLD_TOTAL = 5


def _candidate_document(completed_epochs: int) -> dict[str, object]:
    name = f"candidate_epoch{completed_epochs}"
    return {
        "name": name,
        "data": {
            "train": "data/train.csv",
            "test": "data/test.csv",
            "sample_submission": "data/sample_submission.csv",
            "folds": "artifacts/folds.parquet",
        },
        "features": {
            "base": "raw",
            "categorical": [],
            "providers": [],
        },
        "model": {
            "kind": "lookup_transformer",
            "params": {
                "lookup_cols": [
                    "daily_screen_time_hours",
                    "social_media_hours",
                    "sleep_hours",
                ],
                "validation_selection": "final",
                "epochs": 24,
            },
            "fit": {},
        },
        "training_state": {
            "trajectory": "test-24epoch",
            "candidates": list(CANDIDATES),
            "selected": completed_epochs,
            "schedule_horizon_epochs": 24,
            "trajectory_end_epochs": 24,
            "state_kind": "ema",
            "selection_rule": "precommitted",
        },
    }


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _prepare_committed_workspace(root: Path) -> list[str]:
    for directory in ("configs", "data", "artifacts"):
        (root / directory).mkdir()
    for name in (".gitignore", "pyproject.toml", "uv.lock"):
        shutil.copy2(REPO / name, root / name)

    train, test = toy_train_test(20)
    train.to_csv(root / "data/train.csv", index=False)
    test.to_csv(root / "data/test.csv", index=False)
    pd.DataFrame(
        {"id": test["id"], "addicted_label": np.zeros(len(test), dtype="float64")}
    ).to_csv(root / "data/sample_submission.csv", index=False)
    pd.DataFrame(
        {
            "id": train["id"],
            "fold": np.repeat(np.arange(FOLD_TOTAL), len(train) // FOLD_TOTAL),
        }
    ).to_parquet(root / "artifacts/folds.parquet", index=False)

    config_paths = []
    for completed_epochs in CANDIDATES:
        document = _candidate_document(completed_epochs)
        path = root / "configs" / f"{document['name']}.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        config_paths.append(path.relative_to(root).as_posix())

    _git(root, "init", "-q")
    _git(
        root,
        "add",
        ".gitignore",
        "pyproject.toml",
        "uv.lock",
        "configs",
        "artifacts/folds.parquet",
    )
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    assert _git(root, "status", "--porcelain").stdout == ""
    return config_paths


def test_confirm_run_publishes_three_ready_children_and_exports_every_bundle(
    monkeypatch,
    tmp_path,
):
    config_paths = _prepare_committed_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls: list[int] = []
    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY,
        "lookup_transformer",
        lambda params, fit, seed: FakeTrainingStateAdapter(calls),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline.training_state_run",
            *config_paths,
            "--stage",
            "confirm",
            "--recovery-dir",
            "run-recovery/training-state-test",
        ],
    )

    training_state_run.main()

    client, experiment_id = tracking.mlflow_client()
    runs = client.search_runs([experiment_id])
    parents = [
        run
        for run in runs
        if run.data.tags.get("run.kind") == "training_state_trajectory"
    ]
    children = sorted(
        (
            run
            for run in runs
            if run.data.tags.get("run.kind") == "training_state_snapshot"
        ),
        key=lambda run: int(run.data.tags["training_state.completed_epochs"]),
    )
    assert calls == [4] * 15
    assert [(run.info.status, run.data.tags["training_state.ready"]) for run in children] == [
        ("FINISHED", "true"),
        ("FINISHED", "true"),
        ("FINISHED", "true"),
    ]
    assert len(parents) == 1
    assert parents[0].info.status == "FINISHED"

    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    bundle_paths = []
    bundle_manifests = []
    for child in children:
        path = export_bundle(
            child.info.run_id,
            bundle_dir / f"{child.data.params['experiment']}.bundle.zip",
            tracking_uri=tracking_uri,
        )
        bundle_paths.append(path)
        with zipfile.ZipFile(path) as archive:
            bundle_manifests.append(json.loads(archive.read("manifest.json")))

    assert [manifest["schema_version"] for manifest in bundle_manifests] == [
        TRAINING_STATE_BUNDLE_SCHEMA_VERSION,
    ] * 3
    assert {manifest["source_run_id"] for manifest in bundle_manifests} == {
        child.info.run_id for child in children
    }

    imported_tracking_uri = f"sqlite:///{tmp_path / 'imported-mlflow.db'}"
    imported_run_ids = [
        import_bundle(path, tracking_uri=imported_tracking_uri)
        for path in bundle_paths
    ]
    source_store = MlflowRunStore(tracking_uri=tracking_uri)
    imported_store = MlflowRunStore(tracking_uri=imported_tracking_uri)
    parent_run_id = parents[0].info.run_id
    for child, imported_run_id in zip(children, imported_run_ids, strict=True):
        imported = imported_store.facts_of(imported_run_id)
        assert imported.status == "FINISHED"
        assert imported.tags["source.kind"] == "bundle"
        assert imported.tags["source.run_id"] == child.info.run_id
        assert imported.tags["source.trajectory_run_id"] == parent_run_id

        source_manifest = source_store.artifact_bytes_of(
            child.info.run_id, TRAINING_STATE_MANIFEST_NAME
        )
        imported_manifest = imported_store.artifact_bytes_of(
            imported_run_id, TRAINING_STATE_MANIFEST_NAME
        )
        assert imported_manifest == source_manifest
        document = validate_candidate_manifest(
            manifest_bytes=imported_manifest,
            tags=imported.tags,
            params=imported.params,
            artifact_bytes_of=lambda name, run_id=imported_run_id: (
                imported_store.artifact_bytes_of(run_id, name)
            ),
        )
        validate_candidate_parent_lineage(
            child_run_id=imported_run_id,
            child_document=document,
            child_tags=imported.tags,
            facts_of=imported_store.facts_of,
            artifact_bytes_of=imported_store.artifact_bytes_of,
        )
