"""실행 저장소의 계약 테스트. (CONTEXT.md 용어: 실행 저장소)

두 adapter(MlflowRunStore, InMemoryRunStore)가 같은 interface 계약을 지키는지
동일한 테스트 한 벌로 검증한다. 계약:
- facts_of는 기록 원형(params/metrics/tags)을 그대로 돌려준다.
- oof_of는 id 인덱스의 예측 Series다.
- annotate는 뒤늦은 태그·지표를 남기고, facts_of가 이를 반영한다.
- artifact_bytes_of는 이름 있는 산출물을 원본 바이트로 읽고, artifact_sha256_of는
  그 내용의 SHA-256을 돌려준다.
- 없는 run은 RunNotFound, 있는 run의 없는 산출물은 ArtifactNotFound.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.runs import (
    ArtifactNotFound,
    InMemoryRunStore,
    MlflowRunStore,
    RunNotFound,
    RunStoreError,
    sha256_of,
)

RUN_NAME = "exp_test"
PARAMS = {"experiment": "exp_test", "seeds": "42"}
METRICS = {"auc_oof": 0.9}
TAGS = {"git_commit": "deadbeef", "git_dirty": "False", "sha256.folds": "abc123"}
CONFIG = {"name": "exp_test", "model": {"kind": "lightgbm", "params": {}, "fit": {}}}
# 형식을 모르는 산출물의 대표. 계보 검증은 이것을 바이트와 해시로만 다룬다.
DIAGNOSTICS_NAME = "model_training_diagnostics.json"
DIAGNOSTICS_BYTES = b'[{"seed": 42, "outer_fold": 0, "raw_value": 7805}]\n'



def make_oof() -> pd.DataFrame:
    return pd.DataFrame({"id": [3, 1, 2], "fold": [0, 1, 0], "pred": [0.9, 0.1, 0.5]})


def make_importance() -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": ["age"], "fold": [0], "seed": [42], "gain": [10.0]}
    )


def make_inmemory(tmp_path: Path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,addicted_label\n1,0.5\n")
    store = InMemoryRunStore()
    full = store.add_run(
        "run_full",
        run_name=RUN_NAME,
        params=PARAMS,
        metrics=METRICS,
        tags=TAGS,
        oof=make_oof(),
        importance=make_importance(),
        config=CONFIG,
        submission_path=submission,
        artifacts={DIAGNOSTICS_NAME: DIAGNOSTICS_BYTES},
    )
    empty = store.add_run("run_empty", run_name=RUN_NAME, params=PARAMS)
    return store, full, empty


def make_mlflow(tmp_path: Path):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = client.create_experiment(
        "contract-test", artifact_location=(tmp_path / "mlartifacts").as_uri()
    )

    def create_run(with_artifacts: bool) -> str:
        run = client.create_run(experiment_id, run_name=RUN_NAME)
        run_id = run.info.run_id
        for key, value in PARAMS.items():
            client.log_param(run_id, key, value)
        if with_artifacts:
            for key, value in METRICS.items():
                client.log_metric(run_id, key, value)
            for key, value in TAGS.items():
                client.set_tag(run_id, key, value)
            make_oof().to_parquet(tmp_path / "oof.parquet", index=False)
            make_importance().to_parquet(
                tmp_path / "feature_importance.parquet", index=False
            )
            (tmp_path / "exp_test.yaml").write_text(yaml.safe_dump(CONFIG))
            (tmp_path / "submission.csv").write_text("id,addicted_label\n1,0.5\n")
            (tmp_path / DIAGNOSTICS_NAME).write_bytes(DIAGNOSTICS_BYTES)
            for name in (
                "oof.parquet",
                "feature_importance.parquet",
                "exp_test.yaml",
                "submission.csv",
                DIAGNOSTICS_NAME,
            ):
                client.log_artifact(run_id, str(tmp_path / name))
        return run_id

    store = MlflowRunStore(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    return store, create_run(with_artifacts=True), create_run(with_artifacts=False)


@pytest.fixture(params=["inmemory", "mlflow"])
def stores(request, tmp_path):
    """(store, 산출물이 다 있는 run_id, 산출물이 없는 run_id)."""
    maker = make_inmemory if request.param == "inmemory" else make_mlflow
    return maker(tmp_path)


def test_facts_of_returns_recorded_form(stores):
    store, full, _ = stores
    meta = store.facts_of(full)
    assert meta.run_id == full
    assert meta.run_name == RUN_NAME
    assert meta.params == PARAMS
    assert meta.metrics == METRICS
    # MLflow는 mlflow.* 시스템 태그를 덧붙이므로 포함 관계로 검증한다.
    assert TAGS.items() <= meta.tags.items()


def test_oof_of_returns_id_indexed_series(stores):
    store, full, _ = stores
    oof = store.oof_of(full)
    expected = make_oof().set_index("id")["pred"]
    pd.testing.assert_series_equal(oof, expected)


def test_importance_of_returns_recorded_frame(stores):
    store, full, _ = stores
    pd.testing.assert_frame_equal(store.importance_of(full), make_importance())


def test_config_of_returns_parsed_yaml(stores):
    store, full, _ = stores
    assert store.config_of(full) == CONFIG


def test_submission_path_of_points_at_existing_file(stores):
    store, full, _ = stores
    path = store.submission_path_of(full)
    assert path.exists()
    assert path.read_text().startswith("id,addicted_label")


def test_annotate_is_visible_through_facts_of(stores):
    store, full, _ = stores
    store.annotate(full, tags={"submitted_at": "2026-08-13"}, metrics={"public_auc": 0.95})
    meta = store.facts_of(full)
    assert meta.tags["submitted_at"] == "2026-08-13"
    assert meta.metrics["public_auc"] == 0.95


def test_artifact_bytes_of_returns_stored_content(stores):
    store, full, _ = stores
    assert store.artifact_bytes_of(full, DIAGNOSTICS_NAME) == DIAGNOSTICS_BYTES


def test_artifact_sha256_of_hashes_stored_content(stores):
    store, full, _ = stores
    assert store.artifact_sha256_of(full, DIAGNOSTICS_NAME) == sha256_of(
        DIAGNOSTICS_BYTES
    )


def test_attach_artifact_adds_a_readable_artifact_to_a_finished_run(stores):
    store, full, _ = stores
    payload = b'{"observations": []}\n'

    digest = store.attach_artifact(full, "training_length_evidence.json", payload)

    assert digest == sha256_of(payload)
    assert store.artifact_bytes_of(full, "training_length_evidence.json") == payload
    assert store.artifact_sha256_of(full, "training_length_evidence.json") == digest


def test_attach_artifact_refuses_to_overwrite_a_recorded_artifact(stores):
    store, full, _ = stores

    with pytest.raises(RunStoreError):
        store.attach_artifact(full, DIAGNOSTICS_NAME, b"other")

    assert store.artifact_bytes_of(full, DIAGNOSTICS_NAME) == DIAGNOSTICS_BYTES


def test_attach_artifact_refuses_a_name_that_carries_a_path(stores):
    store, full, _ = stores

    with pytest.raises(RunStoreError):
        store.attach_artifact(full, "logs/run.log", b"x")


def test_unknown_artifact_name_raises_artifact_not_found(stores):
    store, full, _ = stores
    for op in (store.artifact_bytes_of, store.artifact_sha256_of):
        with pytest.raises(ArtifactNotFound):
            op(full, "no_such_artifact.json")


def test_unknown_run_raises_run_not_found(stores):
    store, _, _ = stores
    for op in (
        store.facts_of,
        store.oof_of,
        store.importance_of,
        store.config_of,
        store.submission_path_of,
    ):
        with pytest.raises(RunNotFound):
            op("no_such_run")
    for op in (store.artifact_bytes_of, store.artifact_sha256_of):
        with pytest.raises(RunNotFound):
            op("no_such_run", DIAGNOSTICS_NAME)
    with pytest.raises(RunNotFound):
        store.attach_artifact("no_such_run", "late.json", b"x")
    with pytest.raises(RunNotFound):
        store.annotate("no_such_run", tags={"x": "1"})


def test_missing_artifact_raises_artifact_not_found(stores):
    store, _, empty = stores
    assert store.facts_of(empty).params == PARAMS  # run 자체는 존재한다.
    for op in (
        store.oof_of,
        store.importance_of,
        store.config_of,
        store.submission_path_of,
    ):
        with pytest.raises(ArtifactNotFound):
            op(empty)
    for op in (store.artifact_bytes_of, store.artifact_sha256_of):
        with pytest.raises(ArtifactNotFound):
            op(empty, DIAGNOSTICS_NAME)
