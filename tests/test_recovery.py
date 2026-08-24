"""정식 CV fold 복구 경계의 중단과 재개 회귀 시험. (#141)"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pipeline import initial_score as initial_score_mod
from pipeline import model as model_mod
from pipeline import plan as plan_mod
from pipeline import recovery as recovery_mod
from pipeline import tracking
from pipeline.bundle import export_bundle
from pipeline.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    InitialScoreConfig,
    ModelConfig,
)
from pipeline.cv import CVResult, run_cv
from pipeline.data import file_sha256
from pipeline.initial_score import InitialScores
from pipeline.plan import FeaturePlan, ProviderKind
from pipeline.recovery import FoldRecovery, RecoveryError
from pipeline.training_length import (
    ONE_BASED_COUNT,
    RawTrainingLengthSelection,
    TrainingLengthContract,
)

SEED = 7
N_FOLDS = 5
# 가짜 계열도 실제 계열과 같은 계약으로만 근거를 낸다. (#372)
RECOVERY_FAKE_CONTRACT = TrainingLengthContract(
    "recovery_fake", "selected_rounds", ONE_BASED_COUNT
)


class RecoveryFakeAdapter:
    """fold 번호로 결정적인 예측을 만들고 지정 fold에서 한 번 중단하는 가짜 모델."""

    fail_fold: ClassVar[int | None] = None
    fitted_folds: ClassVar[list[int]] = []
    created_count: ClassVar[int] = 0

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        type(self).created_count += 1
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

    def training_length_evidence(self):
        """fold마다 다른 원시 횟수를 선언해 복구 지점이 근거를 보존하는지 본다."""
        return RECOVERY_FAKE_CONTRACT.declare(
            [
                RawTrainingLengthSelection(
                    raw_path="fake.selected_rounds",
                    raw_value=10 + self.fold,
                )
            ]
        )


class RecoveryRowWiseProvider:
    """공통 행렬 계산 호출을 기록하는 결정적 컬럼 제공자."""

    uses_target = False
    compute_calls: ClassVar[int] = 0

    def columns(self) -> list[str]:
        return ["recovery_row_probe"]

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        type(self).compute_calls += 1
        return pd.DataFrame(
            {"recovery_row_probe": frame["daily_screen_time_hours"] * 0.5},
            index=frame.index,
        )


class RecoveryFoldFitProvider:
    """폴드별 fit과 학습 및 시험 변환 호출을 기록하는 결정적 컬럼 제공자."""

    uses_target = False
    fit_calls: ClassVar[int] = 0
    transform_calls: ClassVar[int] = 0

    def columns(self) -> list[str]:
        return ["recovery_fold_probe"]

    def reuse_input_columns(self) -> list[str]:
        return ["daily_screen_time_hours"]

    def reuse_settings(self) -> dict[str, object]:
        return {}

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        if hasattr(self, "value"):
            raise AssertionError("같은 fold-fit 제공자 인스턴스를 두 폴드에서 재사용했다.")
        type(self).fit_calls += 1
        self.value = float(train_fold["daily_screen_time_hours"].mean())

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        type(self).transform_calls += 1
        return pd.DataFrame(
            {"recovery_fold_probe": self.value},
            index=frame.index,
        )


class RecoveryInitialScoreProvider:
    """완전 복구에서 초기 점수 생성도 생략되는지 기록한다."""

    created_count: ClassVar[int] = 0
    compute_calls: ClassVar[int] = 0

    def __init__(self) -> None:
        type(self).created_count += 1

    def compute(
        self, train: pd.DataFrame, test: pd.DataFrame, seed: int
    ) -> InitialScores:
        type(self).compute_calls += 1
        return InitialScores(
            train=pd.Series(0.0, index=train.index, dtype="float64"),
            test=pd.Series(0.0, index=test.index, dtype="float64"),
        )

    def input_paths(self) -> dict[str, Path]:
        return {}


class ProgressRecorder:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.folds: list[tuple[int, int, float]] = []
        self.timings: list[dict[str, object]] = []

    def stage(self, name: str) -> None:
        self.stages.append(name)

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None:
        self.folds.append((seed_index, fold_index, auc))

    def record_timing(self, event: dict[str, object]) -> None:
        self.timings.append(dict(event))


@pytest.fixture
def recovery_env(tmp_path, monkeypatch):
    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "recovery_fake", RecoveryFakeAdapter)
    monkeypatch.setitem(
        model_mod.TRAINING_LENGTH_CONTRACTS, "recovery_fake", RECOVERY_FAKE_CONTRACT
    )
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
    RecoveryFakeAdapter.created_count = 0
    return cfg, plan, prepared_train, prepared_test, store


def _with_seed_metric(result: CVResult) -> CVResult:
    result.fold_aucs[f"auc_oof_seed_{SEED}"] = result.fold_aucs["auc_oof"]
    return result


def _tracking_recovery_inputs(recovery_env, monkeypatch):
    cfg, _, train, test, store = recovery_env
    monkeypatch.setitem(
        plan_mod.REGISTRY,
        "recovery_row_probe",
        ProviderKind(plan_mod.ROW_WISE, RecoveryRowWiseProvider),
    )
    monkeypatch.setitem(
        plan_mod.REGISTRY,
        "recovery_fold_probe",
        ProviderKind(plan_mod.FOLD_FIT, RecoveryFoldFitProvider),
    )
    monkeypatch.setitem(
        initial_score_mod.REGISTRY,
        "recovery_initial_probe",
        RecoveryInitialScoreProvider,
    )
    cfg = replace(
        cfg,
        features=FeatureConfig(
            base="raw",
            categorical=[],
            providers=[
                {"kind": "recovery_row_probe"},
                {"kind": "recovery_fold_probe"},
            ],
        ),
        initial_score=InitialScoreConfig(kind="recovery_initial_probe", params={}),
    )
    plan = FeaturePlan.from_config(cfg.features)
    train, test = plan.apply_dataset_wide(train, test)
    return cfg, plan, train, test, store


def _reset_recovery_call_counts() -> None:
    RecoveryRowWiseProvider.compute_calls = 0
    RecoveryFoldFitProvider.fit_calls = 0
    RecoveryFoldFitProvider.transform_calls = 0
    RecoveryInitialScoreProvider.created_count = 0
    RecoveryInitialScoreProvider.compute_calls = 0
    RecoveryFakeAdapter.created_count = 0
    RecoveryFakeAdapter.fitted_folds = []


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
    interrupted_recorder = ProgressRecorder()

    with pytest.raises(KeyboardInterrupt, match="fold 1"):
        run_cv(
            cfg,
            plan,
            train,
            test,
            SEED,
            recorder=interrupted_recorder,
            recovery=interrupted_store,
        )
    assert RecoveryFakeAdapter.fitted_folds == [0, 1]
    assert (tmp_path / "interrupted" / f"seed_{SEED}" / "fold_0" / "manifest.json").is_file()
    assert not (tmp_path / "interrupted" / f"seed_{SEED}" / "fold_1").exists()
    failed = [
        event
        for event in interrupted_recorder.timings
        if event["fold"] == 1 and event["outcome"] == "failed"
    ]
    assert [event["operation"] for event in failed] == [
        "fold_finalize.model_fit",
        "fold_finalize",
    ]
    assert all(event["reason"] == "KeyboardInterrupt" for event in failed)

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
    # 재사용한 fold 0의 관측 학습 길이 근거가 새로 학습한 실행과 같아야 한다. (#372)
    reused_evidence = resumed.model_training_diagnostics[0]["training_length_evidence"]
    assert reused_evidence["model_family"] == "recovery_fake"
    assert reused_evidence["converter"] == "count_as_is"
    assert reused_evidence["observations"] == [
        {
            "seed": SEED,
            "outer_fold": 0,
            "inner_member": None,
            "raw_field": "selected_rounds",
            "raw_path": "fake.selected_rounds",
            "raw_value": 10,
            "raw_meaning": ONE_BASED_COUNT,
            "observed_training_length": 10,
        }
    ]
    assert [
        (
            item["training_length_evidence"]["observations"][0]["outer_fold"],
            item["training_length_evidence"]["observations"][0]["observed_training_length"],
        )
        for item in resumed.model_training_diagnostics
    ] == [(fold, 10 + fold) for fold in range(N_FOLDS)]

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

    partial_finalize = [
        event
        for event in partial_recorder.timings
        if event["operation"] == "fold_finalize"
    ]
    assert [event["outcome"] for event in partial_finalize] == [
        "reused",
        "reused",
        "reused",
        "reused",
        "success",
    ]
    partial_feature = [
        event
        for event in partial_recorder.timings
        if event["operation"] == "fold_feature"
    ]
    assert [event["outcome"] for event in partial_feature] == [
        "skipped",
        "skipped",
        "skipped",
        "skipped",
        "success",
    ]
    partial_operations = [event["operation"] for event in partial_recorder.timings]
    assert partial_operations[:N_FOLDS] == ["recovery.read_validate"] * N_FOLDS
    full_finalize = [
        event
        for event in full_recorder.timings
        if event["operation"] == "fold_finalize"
    ]
    assert [event["outcome"] for event in full_finalize] == ["reused"] * N_FOLDS
    full_feature = [
        event
        for event in full_recorder.timings
        if event["operation"] == "fold_feature"
    ]
    assert [event["outcome"] for event in full_feature] == ["skipped"] * N_FOLDS
    skipped_model_work = [
        event
        for event in full_recorder.timings
        if str(event["operation"]).startswith("fold_finalize.")
    ]
    assert skipped_model_work
    assert all(
        event["outcome"] == "skipped"
        and event["duration_ns"] is None
        and event["reason"] == "checkpoint_reused"
        for event in skipped_model_work
    )


def test_partial_and_full_recovery_only_compute_missing_fold_work(
    recovery_env, monkeypatch
):
    cfg, plan, train, test, store = _tracking_recovery_inputs(recovery_env, monkeypatch)
    recovery = store("full-feature-skip")
    baseline = run_cv(cfg, plan, train, test, SEED, recovery=recovery)

    partial_recovery = store("partial-feature-skip")
    partial_seed_dir = partial_recovery.root / f"seed_{SEED}"
    partial_seed_dir.mkdir(parents=True)
    for fold in range(N_FOLDS - 1):
        shutil.copytree(
            recovery.root / f"seed_{SEED}" / f"fold_{fold}",
            partial_seed_dir / f"fold_{fold}",
        )

    _reset_recovery_call_counts()
    partial = run_cv(cfg, plan, train, test, SEED, recovery=partial_recovery)

    assert RecoveryRowWiseProvider.compute_calls == 2
    assert RecoveryFoldFitProvider.fit_calls == 1
    assert RecoveryFoldFitProvider.transform_calls == 2
    assert RecoveryInitialScoreProvider.created_count == 1
    assert RecoveryInitialScoreProvider.compute_calls == 1
    assert RecoveryFakeAdapter.created_count == 1
    assert RecoveryFakeAdapter.fitted_folds == [N_FOLDS - 1]
    pd.testing.assert_frame_equal(partial.oof, baseline.oof, check_exact=True)
    pd.testing.assert_frame_equal(partial.test_pred, baseline.test_pred, check_exact=True)
    pd.testing.assert_frame_equal(partial.importance, baseline.importance, check_exact=True)
    assert partial.fold_aucs == baseline.fold_aucs
    assert partial.feature_names == baseline.feature_names

    _reset_recovery_call_counts()
    recovered = run_cv(cfg, plan, train, test, SEED, recovery=recovery)

    assert RecoveryRowWiseProvider.compute_calls == 0
    assert RecoveryFoldFitProvider.fit_calls == 0
    assert RecoveryFoldFitProvider.transform_calls == 0
    assert RecoveryInitialScoreProvider.created_count == 0
    assert RecoveryInitialScoreProvider.compute_calls == 0
    assert RecoveryFakeAdapter.created_count == 0
    assert RecoveryFakeAdapter.fitted_folds == []
    pd.testing.assert_frame_equal(recovered.oof, baseline.oof, check_exact=True)
    pd.testing.assert_frame_equal(recovered.test_pred, baseline.test_pred, check_exact=True)
    pd.testing.assert_frame_equal(recovered.importance, baseline.importance, check_exact=True)
    assert recovered.fold_aucs == baseline.fold_aucs
    assert recovered.feature_names == baseline.feature_names


def test_later_fold_feature_mismatch_fails_before_any_computation(
    recovery_env, monkeypatch
):
    cfg, plan, train, test, store = _tracking_recovery_inputs(recovery_env, monkeypatch)
    recovery = store("later-feature-mismatch")
    run_cv(cfg, plan, train, test, SEED, recovery=recovery)
    fold_dir = recovery.root / f"seed_{SEED}" / f"fold_{N_FOLDS - 1}"
    importance_path = fold_dir / recovery_mod.IMPORTANCE_NAME
    importance = pd.read_parquet(importance_path)
    importance.loc[0, "feature"] = "unexpected_recovery_feature"
    importance.to_parquet(importance_path, index=False)
    manifest_path = fold_dir / recovery_mod.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][recovery_mod.IMPORTANCE_NAME]["sha256"] = file_sha256(
        importance_path
    )
    manifest["manifest_content_sha256"] = recovery_mod._content_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest))
    _reset_recovery_call_counts()
    recorder = ProgressRecorder()

    with pytest.raises(RecoveryError, match="특성 행 순서"):
        run_cv(cfg, plan, train, test, SEED, recorder=recorder, recovery=recovery)

    assert RecoveryRowWiseProvider.compute_calls == 0
    assert RecoveryFoldFitProvider.fit_calls == 0
    assert RecoveryFoldFitProvider.transform_calls == 0
    assert RecoveryInitialScoreProvider.created_count == 0
    assert RecoveryInitialScoreProvider.compute_calls == 0
    assert RecoveryFakeAdapter.created_count == 0
    assert RecoveryFakeAdapter.fitted_folds == []
    assert recorder.stages == ["feature_build"]
    assert [event["operation"] for event in recorder.timings] == [
        "recovery.read_validate"
    ] * N_FOLDS
    assert [event["outcome"] for event in recorder.timings] == [
        "success",
        "success",
        "success",
        "success",
        "failed",
    ]


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
