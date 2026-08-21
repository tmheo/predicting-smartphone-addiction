"""정식 CV fold 복구 경계의 중단과 재개 회귀 시험. (#141)"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pipeline import model as model_mod
from pipeline import recovery as recovery_mod
from pipeline import tracking
from pipeline.bundle import export_bundle
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.cv import CVResult, run_cv
from pipeline.data import file_sha256
from pipeline.plan import FeaturePlan
from pipeline.recovery import FoldRecovery, RecoveryError

SEED = 7
N_FOLDS = 5


class RecoveryFakeAdapter:
    """fold 번호로 결정적인 예측을 만들고 지정 fold에서 한 번 중단하는 가짜 모델."""

    fail_fold: ClassVar[int | None] = None
    fitted_folds: ClassVar[list[int]] = []

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self.seed = seed
        self.fold: int | None = None
        self.features: list[str] = []

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        self.fold = int(X_va.index[0] % N_FOLDS)
        self.features = list(X_tr.columns)
        type(self).fitted_folds.append(self.fold)
        if self.fold == type(self).fail_fold:
            raise KeyboardInterrupt(f"fold {self.fold} 강제 중단")
        return (0.2 + 0.6 * y_va.to_numpy(dtype="float64") + self.fold * 1e-4).astype(
            "float64"
        )

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        return (
            0.3 + self.fold * 0.05 + np.arange(len(X), dtype="float64") * 1e-6
        )

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.features,
                "gain": np.arange(1, len(self.features) + 1, dtype="float64") + self.fold,
            }
        )

    def training_diagnostics(self) -> dict[str, object]:
        return {"fake_fold": self.fold, "losses": [0.5, 0.4]}


class ProgressRecorder:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.folds: list[tuple[int, int, float]] = []

    def stage(self, name: str) -> None:
        self.stages.append(name)

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None:
        self.folds.append((seed_index, fold_index, auc))


@pytest.fixture
def recovery_env(tmp_path, monkeypatch):
    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "recovery_fake", RecoveryFakeAdapter)
    n = 60
    train = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "daily_screen_time_hours": np.linspace(1, 10, n),
            "social_media_hours": np.linspace(0, 5, n),
            "addicted_label": np.tile([0, 1], n // 2),
        }
    )
    test = train.drop(columns=["addicted_label"]).iloc[:12].copy()
    test["id"] += n
    folds = pd.DataFrame({"id": train["id"], "fold": np.arange(n) % N_FOLDS})
    sample = pd.DataFrame({"id": test["id"], "addicted_label": 0.0})
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    artifacts_dir.mkdir()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    folds_path = artifacts_dir / "folds.parquet"
    config_path = tmp_path / "exp_recovery.yaml"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)
    folds.to_parquet(folds_path, index=False)
    config_path.write_text("name: recovery_test\nmodel:\n  kind: recovery_fake\n")
    cfg = ExperimentConfig(
        name="recovery_test",
        data=DataConfig(
            train=train_path,
            test=test_path,
            sample_submission=sample_path,
            folds=folds_path,
        ),
        features=FeatureConfig(base="raw", categorical=[], providers=[]),
        model=ModelConfig(kind="recovery_fake", params={}, fit={}),
        initial_score=None,
        seeds=[SEED],
        stage="screen",
        source_path=config_path,
    )
    prepared_train = train.merge(folds, on="id", validate="one_to_one")
    plan = FeaturePlan.from_config(cfg.features)
    prepared_train, prepared_test = plan.apply_dataset_wide(prepared_train, test)
    hashes = {
        "train": file_sha256(train_path),
        "test": file_sha256(test_path),
        "folds": file_sha256(folds_path),
    }

    def store(name: str) -> FoldRecovery:
        return FoldRecovery.for_run(
            tmp_path / name,
            cfg,
            hashes,
            git_commit="a" * 40,
            model_dependencies={"python": "test", "recovery_fake": "1"},
        )

    RecoveryFakeAdapter.fail_fold = None
    RecoveryFakeAdapter.fitted_folds = []
    return cfg, plan, prepared_train, prepared_test, store


def _with_seed_metric(result: CVResult) -> CVResult:
    result.fold_aucs[f"auc_oof_seed_{SEED}"] = result.fold_aucs["auc_oof"]
    return result


def _log_final(tmp_path: Path, name: str, cfg: ExperimentConfig, result: CVResult):
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / f'{name}.db'}"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        name, artifact_location=(tmp_path / f"{name}-artifacts").as_uri()
    )
    run = client.create_run(experiment_id, run_name=name)
    tracking.log_start_records(client, run.info.run_id, cfg)
    tracking.log_input_hashes(
        client,
        run.info.run_id,
        {
            "train": file_sha256(cfg.data.train),
            "test": file_sha256(cfg.data.test),
            "folds": file_sha256(cfg.data.folds),
        },
    )
    tracking.log_final_records(client, run.info.run_id, cfg, result, {SEED: result.oof})
    client.set_terminated(run.info.run_id, status="FINISHED")
    artifact_dir = Path(client.download_artifacts(run.info.run_id, "", tmp_path / f"{name}-out"))
    return client.get_run(run.info.run_id).data, artifact_dir, uri, run.info.run_id


def test_interrupted_run_reuses_completed_fold_and_matches_uninterrupted_mlflow(recovery_env, tmp_path):
    cfg, plan, train, test, store = recovery_env
    interrupted_store = store("interrupted")
    RecoveryFakeAdapter.fail_fold = 1

    with pytest.raises(KeyboardInterrupt, match="fold 1"):
        run_cv(cfg, plan, train, test, SEED, recovery=interrupted_store)
    assert RecoveryFakeAdapter.fitted_folds == [0, 1]
    assert (tmp_path / "interrupted" / f"seed_{SEED}" / "fold_0" / "manifest.json").is_file()
    assert not (tmp_path / "interrupted" / f"seed_{SEED}" / "fold_1").exists()

    RecoveryFakeAdapter.fail_fold = None
    RecoveryFakeAdapter.fitted_folds = []
    resumed = _with_seed_metric(
        run_cv(cfg, plan, train, test, SEED, recovery=interrupted_store)
    )
    assert RecoveryFakeAdapter.fitted_folds == [1, 2, 3, 4]
    assert [(item["fold"], item["reused"]) for item in resumed.recovery_evidence] == [
        (0, True),
        (1, False),
        (2, False),
        (3, False),
        (4, False),
    ]

    RecoveryFakeAdapter.fitted_folds = []
    uninterrupted = _with_seed_metric(run_cv(cfg, plan, train, test, SEED, recovery=store("full")))
    assert RecoveryFakeAdapter.fitted_folds == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(resumed.oof, uninterrupted.oof, check_exact=True)
    pd.testing.assert_frame_equal(resumed.test_pred, uninterrupted.test_pred, check_exact=True)
    pd.testing.assert_frame_equal(resumed.importance, uninterrupted.importance, check_exact=True)
    assert resumed.fold_aucs == uninterrupted.fold_aucs
    assert resumed.model_training_diagnostics == uninterrupted.model_training_diagnostics

    resumed_meta, resumed_artifacts, resumed_uri, resumed_run_id = _log_final(
        tmp_path, "resumed", cfg, resumed
    )
    full_meta, full_artifacts, _, _ = _log_final(tmp_path, "full", cfg, uninterrupted)
    assert resumed_meta.params == full_meta.params
    assert resumed_meta.metrics == full_meta.metrics
    for artifact in (
        "oof.parquet",
        f"oof_seed_{SEED}.parquet",
        "test_pred.parquet",
        "feature_importance.parquet",
        "submission.csv",
        "model_training_diagnostics.json",
    ):
        assert (resumed_artifacts / artifact).read_bytes() == (full_artifacts / artifact).read_bytes()
    evidence = json.loads((resumed_artifacts / recovery_mod.EVIDENCE_NAME).read_text())
    assert evidence["boundary"] == "completed_seed_fold"
    assert evidence["checkpoints"][0]["reused"] is True
    identity = evidence["checkpoints"][0]["execution_identity"]
    assert identity["git_commit"] == "a" * 40
    assert identity["config_sha256"] == file_sha256(cfg.source_path)
    assert identity["folds_sha256"] == file_sha256(cfg.data.folds)
    assert identity["model_dependencies"] == {"python": "test", "recovery_fake": "1"}
    assert not any(path.name.endswith(".tmp") for path in resumed_artifacts.rglob("*"))

    bundle_path = export_bundle(
        resumed_run_id, tmp_path / "resumed.bundle.zip", tracking_uri=resumed_uri
    )
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
    assert f"artifacts/{recovery_mod.EVIDENCE_NAME}" in names
    assert "artifacts/oof.parquet" in names
    assert "artifacts/test_pred.parquet" in names
    assert not any(name.endswith(".tmp") or "/.fold_" in name for name in names)


def test_four_of_five_and_full_recovery_keep_public_result_and_progress(
    recovery_env, tmp_path
):
    cfg, plan, train, test, store = recovery_env
    baseline_store = store("baseline")
    baseline_recorder = ProgressRecorder()
    baseline = run_cv(
        cfg,
        plan,
        train,
        test,
        SEED,
        recorder=baseline_recorder,
        recovery=baseline_store,
    )

    partial_store = store("four-of-five")
    partial_seed_dir = partial_store.root / f"seed_{SEED}"
    partial_seed_dir.mkdir(parents=True)
    for fold in range(N_FOLDS - 1):
        shutil.copytree(
            baseline_store.root / f"seed_{SEED}" / f"fold_{fold}",
            partial_seed_dir / f"fold_{fold}",
        )

    RecoveryFakeAdapter.fitted_folds = []
    partial_recorder = ProgressRecorder()
    partial = run_cv(
        cfg,
        plan,
        train,
        test,
        SEED,
        recorder=partial_recorder,
        recovery=partial_store,
    )
    assert RecoveryFakeAdapter.fitted_folds == [N_FOLDS - 1]
    assert [item["reused"] for item in partial.recovery_evidence] == [
        True,
        True,
        True,
        True,
        False,
    ]

    RecoveryFakeAdapter.fitted_folds = []
    full_recorder = ProgressRecorder()
    fully_recovered = run_cv(
        cfg,
        plan,
        train,
        test,
        SEED,
        recorder=full_recorder,
        recovery=baseline_store,
    )
    assert RecoveryFakeAdapter.fitted_folds == []
    assert [item["reused"] for item in fully_recovered.recovery_evidence] == [True] * N_FOLDS

    for recovered in (partial, fully_recovered):
        pd.testing.assert_frame_equal(recovered.oof, baseline.oof, check_exact=True)
        pd.testing.assert_frame_equal(recovered.test_pred, baseline.test_pred, check_exact=True)
        pd.testing.assert_frame_equal(recovered.importance, baseline.importance, check_exact=True)
        assert recovered.fold_aucs == baseline.fold_aucs
        assert recovered.feature_names == baseline.feature_names
        assert recovered.model_training_diagnostics == baseline.model_training_diagnostics

    expected_progress = [
        (0, fold, baseline.fold_aucs[f"auc_fold_{fold}"]) for fold in range(N_FOLDS)
    ]
    for recorder in (baseline_recorder, partial_recorder, full_recorder):
        assert recorder.stages == ["feature_build", "training"]
        assert recorder.folds == expected_progress


def _save_fold_zero(recovery_env) -> tuple[FoldRecovery, ExperimentConfig, FeaturePlan, pd.DataFrame, pd.DataFrame]:
    cfg, plan, train, test, store = recovery_env
    recovery = store("tamper")
    run_cv(cfg, plan, train, test, SEED, recovery=recovery)
    return recovery, cfg, plan, train, test


def test_tampered_manifest_is_rejected(recovery_env):
    recovery, cfg, plan, train, test = _save_fold_zero(recovery_env)
    manifest_path = recovery.root / f"seed_{SEED}" / "fold_0" / recovery_mod.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics"]["auc"] = 0.123
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RecoveryError, match="manifest 내용 해시"):
        run_cv(cfg, plan, train, test, SEED, recovery=recovery)


def test_changed_execution_identity_cannot_reuse_fold(recovery_env):
    recovery, cfg, plan, train, test = _save_fold_zero(recovery_env)
    changed_identity = {
        **recovery.execution_identity,
        "input_sha256": {
            **recovery.execution_identity["input_sha256"],
            "train": "0" * 64,
        },
    }

    with pytest.raises(RecoveryError, match="실행 정체성"):
        run_cv(
            cfg,
            plan,
            train,
            test,
            SEED,
            recovery=FoldRecovery(recovery.root, changed_identity),
        )


def test_validation_row_reordering_is_rejected_even_with_updated_hashes(recovery_env):
    recovery, cfg, plan, train, test = _save_fold_zero(recovery_env)
    fold_dir = recovery.root / f"seed_{SEED}" / "fold_0"
    prediction_path = fold_dir / recovery_mod.VALIDATION_NAME
    predictions = pd.read_parquet(prediction_path).iloc[::-1].reset_index(drop=True)
    predictions.to_parquet(prediction_path, index=False)
    manifest_path = fold_dir / recovery_mod.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][recovery_mod.VALIDATION_NAME]["sha256"] = file_sha256(prediction_path)
    manifest["manifest_content_sha256"] = recovery_mod._content_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RecoveryError, match="행 순서"):
        run_cv(cfg, plan, train, test, SEED, recovery=recovery)


def test_incomplete_temporary_state_and_duplicate_fold_are_rejected(recovery_env):
    recovery, cfg, plan, train, test = _save_fold_zero(recovery_env)
    seed_dir = recovery.root / f"seed_{SEED}"
    temporary = seed_dir / ".fold_1.interrupted.tmp"
    temporary.mkdir()
    with pytest.raises(RecoveryError, match="불완전한 fold 임시 상태"):
        run_cv(cfg, plan, train, test, SEED, recovery=recovery)

    temporary.rmdir()
    duplicate = seed_dir / "fold_0_copy"
    shutil.copytree(seed_dir / "fold_0", duplicate)
    with pytest.raises(RecoveryError, match="중복 fold"):
        run_cv(cfg, plan, train, test, SEED, recovery=recovery)


def test_dependency_snapshot_skips_requirements_for_other_environments(tmp_path):
    # faiss-gpu-cu12처럼 다른 환경 전용 표식이 붙은 의존성은 설치를 요구하지 않고,
    # 표식 없는 의존성은 설치돼 있어야 한다. (#258 회귀: macOS 로컬 실행 차단)
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    lock.write_text("")
    pyproject.write_text(
        "[project]\n"
        'name = "snapshot-test"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        "    \"pytest>=1\",\n"
        "    \"nonexistent-elsewhere-only==1.0; sys_platform == 'nonexistent-os'\",\n"
        "]\n"
    )
    snapshot = recovery_mod.model_dependency_snapshot(pyproject, lock)
    packages = snapshot["project_packages"]
    assert "pytest" in packages
    assert "nonexistent-elsewhere-only" not in packages


def test_dependency_snapshot_requires_applicable_requirements(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    lock.write_text("")
    pyproject.write_text(
        "[project]\n"
        'name = "snapshot-test"\n'
        'version = "0.0.0"\n'
        'dependencies = ["nonexistent-package-for-snapshot==1.0"]\n'
    )
    with pytest.raises(RecoveryError, match="설치되지 않았다"):
        recovery_mod.model_dependency_snapshot(pyproject, lock)
