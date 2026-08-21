"""fold 실행 시간 및 자원 관측 계약의 회귀 검사."""

from __future__ import annotations

import gzip
import json
import signal
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_model import FakeAdapter, fake_experiment_config, toy_train_test

from pipeline import cv
from pipeline import fold_observability as observability_mod
from pipeline import model as model_mod
from pipeline import plan as plan_mod
from pipeline.config import FeatureConfig, ModelConfig
from pipeline.fold_observability import (
    ARTIFACT_NAME,
    FoldExecutionRecorder,
    ObservabilityPersistenceError,
    ObservabilitySchemaError,
    read_fold_observability,
)
from pipeline.plan import FeaturePlan, ProviderKind
from pipeline.observe import RunObserver

SEED = 7
N_FOLDS = 3


class TimingRecorder:
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


class FakeProbe:
    metadata = {"cpu_scope": "test_scope", "gpu_expected": True}

    def sample(self, observed_ns: int) -> dict[str, object]:
        raise AssertionError("수동 표본 검사에서는 자동 측정을 호출하지 않는다.")


def _prepared_inputs(cfg):
    plan = FeaturePlan.from_config(cfg.features)
    train, test = toy_train_test()
    train, test = plan.apply_dataset_wide(train, test)
    train["fold"] = np.arange(len(train)) % N_FOLDS
    return plan, train, test


def _event(
    *,
    started_ns: int,
    duration_ns: int | None,
    operation: str,
    outcome: str = "success",
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "seed": SEED,
        "fold": 0,
        "operation": operation,
        "actor_kind": "test",
        "actor_name": "test_actor",
        "worker_id": "worker-1",
        "device_id": None,
        "dataset": None,
        "started_ns": started_ns,
        "duration_ns": duration_ns,
        "outcome": outcome,
        "reason": reason,
    }


def test_enabled_observability_keeps_cv_result_exactly_equal(monkeypatch):
    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY,
        "fake",
        lambda params, fit, seed: FakeAdapter(params, fit, seed, fold_value=0.5),
    )
    cfg = fake_experiment_config()
    plan, train, test = _prepared_inputs(cfg)

    without_observability = cv.run_cv(cfg, plan, train, test, seed=SEED)
    recorder = TimingRecorder()
    with_observability = cv.run_cv(
        cfg, plan, train, test, seed=SEED, recorder=recorder
    )

    pd.testing.assert_frame_equal(
        with_observability.oof, without_observability.oof, check_exact=True
    )
    pd.testing.assert_frame_equal(
        with_observability.test_pred,
        without_observability.test_pred,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        with_observability.importance,
        without_observability.importance,
        check_exact=True,
    )
    assert with_observability.fold_aucs == without_observability.fold_aucs
    assert with_observability.feature_names == without_observability.feature_names
    assert with_observability.recovery_evidence == without_observability.recovery_evidence
    assert (
        with_observability.model_training_diagnostics
        == without_observability.model_training_diagnostics
    )
    assert sum(event["operation"] == "fold_feature" for event in recorder.timings) == N_FOLDS
    assert sum(event["operation"] == "fold_finalize" for event in recorder.timings) == N_FOLDS
    disabled = [
        event
        for event in recorder.timings
        if str(event["operation"]).startswith("recovery.")
    ]
    assert disabled
    assert all(
        event["outcome"] == "skipped"
        and event["duration_ns"] is None
        and event["reason"] == "disabled"
        for event in disabled
    )


def test_enabled_observability_keeps_lightgbm_result_exactly_equal():
    cfg = replace(
        fake_experiment_config(),
        name="lightgbm_observability_equivalence",
        model=ModelConfig(
            kind="lightgbm",
            params={
                "objective": "binary",
                "n_estimators": 30,
                "num_leaves": 7,
                "learning_rate": 0.1,
                "verbosity": -1,
                "deterministic": True,
                "force_row_wise": True,
                "n_jobs": 1,
            },
            fit={"early_stopping_rounds": 5},
        ),
    )
    plan, train, test = _prepared_inputs(cfg)

    without_observability = cv.run_cv(cfg, plan, train, test, seed=SEED)
    recorder = TimingRecorder()
    with_observability = cv.run_cv(
        cfg,
        plan,
        train,
        test,
        seed=SEED,
        recorder=recorder,
    )

    pd.testing.assert_frame_equal(
        with_observability.oof,
        without_observability.oof,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        with_observability.test_pred,
        without_observability.test_pred,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        with_observability.importance,
        without_observability.importance,
        check_exact=True,
    )
    assert with_observability.fold_aucs == without_observability.fold_aucs
    assert with_observability.recovery_evidence == without_observability.recovery_evidence


def test_model_failure_records_failed_leaf_and_parent(monkeypatch):
    class FailingAdapter(FakeAdapter):
        def fit(self, *args, **kwargs) -> np.ndarray:
            raise RuntimeError("model-fit-failed")

    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY,
        "fake",
        lambda params, fit, seed: FailingAdapter(params, fit, seed, fold_value=0.5),
    )
    cfg = fake_experiment_config()
    plan, train, test = _prepared_inputs(cfg)
    recorder = TimingRecorder()

    with pytest.raises(RuntimeError, match="model-fit-failed"):
        cv.run_cv(cfg, plan, train, test, seed=SEED, recorder=recorder)

    failed = [event for event in recorder.timings if event["outcome"] == "failed"]
    assert [event["operation"] for event in failed] == [
        "fold_finalize.model_fit",
        "fold_finalize",
    ]
    assert all(event["reason"] == "RuntimeError" for event in failed)


def test_fold_fit_provider_emits_fit_and_train_test_transform_events(monkeypatch):
    class TimedProvider:
        uses_target = False

        def columns(self) -> list[str]:
            return ["timed_feature"]

        def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
            self.value = float(seed)

        def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"timed_feature": self.value}, index=frame.index)

    monkeypatch.setitem(
        plan_mod.REGISTRY,
        "timed_provider",
        ProviderKind(plan_mod.FOLD_FIT, TimedProvider),
    )
    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY,
        "fake",
        lambda params, fit, seed: FakeAdapter(params, fit, seed, fold_value=0.5),
    )
    cfg = fake_experiment_config()
    object.__setattr__(
        cfg,
        "features",
        FeatureConfig(
            base="raw",
            categorical=[],
            providers=[{"kind": "timed_provider"}],
        ),
    )
    plan, train, test = _prepared_inputs(cfg)
    recorder = TimingRecorder()

    cv.run_cv(cfg, plan, train, test, seed=SEED, recorder=recorder)

    fits = [
        event
        for event in recorder.timings
        if event["operation"] == "fold_feature.provider_fit"
    ]
    transforms = [
        event
        for event in recorder.timings
        if event["operation"] == "fold_feature.provider_transform"
    ]
    assert len(fits) == N_FOLDS
    assert len(transforms) == N_FOLDS * 2
    assert {event["dataset"] for event in transforms} == {"train", "test"}
    assert {event["actor_name"] for event in [*fits, *transforms]} == {
        "timed_provider"
    }


def test_versioned_artifact_aggregates_union_unclassified_and_resources(tmp_path):
    recorder = FoldExecutionRecorder(
        tmp_path / "observability",
        {"seeds": [SEED], "source": "remote_measured"},
        resource_probe=FakeProbe(),
        start_sampler=False,
    )
    recorder.start()
    recorder.configure_run_shape(seed_total=1, fold_total=1, provider_total=0)
    started_ns = time.monotonic_ns()
    recorder.record_timing(
        _event(
            started_ns=started_ns,
            duration_ns=100_000_000,
            operation="fold_finalize",
        )
    )
    recorder.record_timing(
        _event(
            started_ns=started_ns + 10_000_000,
            duration_ns=20_000_000,
            operation="fold_finalize.model_fit",
        )
    )
    recorder.record_resource(
        {
            "observed_ns": started_ns,
            "interval_ns": None,
            "cpu_scope": "test_scope",
            "cpu_cores": None,
            "memory_mib": 10.0,
            "cpu_supported": True,
            "cpu_reason": None,
            "gpu_expected": True,
            "gpu_supported": True,
            "gpu_reason": None,
            "gpus": [],
        }
    )
    recorder.record_resource(
        {
            "observed_ns": started_ns + 2_000_000_000,
            "interval_ns": 2_000_000_000,
            "cpu_scope": "test_scope",
            "cpu_cores": 1.5,
            "memory_mib": 12.0,
            "cpu_supported": True,
            "cpu_reason": None,
            "gpu_expected": True,
            "gpu_supported": True,
            "gpu_reason": None,
            "gpus": [
                {
                    "device_id": "0",
                    "name": "gpu-a",
                    "utilization_percent": 0.0,
                    "memory_used_mib": 100.0,
                    "power_watts": 20.0,
                },
                {
                    "device_id": "1",
                    "name": "gpu-b",
                    "utilization_percent": 50.0,
                    "memory_used_mib": 200.0,
                    "power_watts": 40.0,
                },
            ],
        }
    )

    finalized = recorder.finalize()
    records = read_fold_observability(finalized.path)

    assert finalized.path.name == ARTIFACT_NAME
    assert len(finalized.sha256) == 64
    assert records[0]["record_type"] == "metadata"
    assert records[-1]["record_type"] == "summary"
    summary = records[-1]
    assert summary["timing"]["operations"]["fold_finalize"][
        "union_seconds"
    ] == pytest.approx(0.1)
    assert summary["timing"]["unclassified"] == [
        {
            "fold": 0,
            "operation": "fold_finalize",
            "seed": SEED,
            "unclassified_seconds": pytest.approx(0.08),
        }
    ]
    assert summary["resource"]["cpu_cores_mean"] == pytest.approx(1.5)
    assert summary["resource"]["gpu_utilization_mean"] == pytest.approx(25.0)
    assert summary["resource"]["gpu_idle_fraction"] == pytest.approx(0.5)
    assert summary["resource"]["gpu_idle_device_seconds"] == pytest.approx(2.0)
    assert summary["resource"]["gpu_all_idle_wall_seconds"] == pytest.approx(0.0)
    assert summary["resource"]["gpu_memory_peak_mib"] == pytest.approx(200.0)
    assert summary["resource"]["gpu_power_mean_watts"] == pytest.approx(30.0)
    metric_names = {name for name, _, _ in finalized.metrics}
    assert "time.fold_finalize_seconds" in metric_names
    assert "time.model_fit_seconds" in metric_names
    assert "resource.gpu_idle_fraction" in metric_names


def test_resource_sampling_failure_is_nonfatal_and_counted_as_missing(tmp_path):
    recorder = FoldExecutionRecorder(
        tmp_path / "observability",
        {"seeds": [SEED], "source": "local_measured"},
        resource_probe=FakeProbe(),
        start_sampler=False,
    )
    recorder.start()
    observed_ns = time.monotonic_ns()
    recorder.record_resource(
        {
            "observed_ns": observed_ns,
            "interval_ns": 2_000_000_000,
            "cpu_scope": "test_scope",
            "cpu_cores": None,
            "memory_mib": None,
            "cpu_supported": False,
            "cpu_reason": "PermissionError",
            "gpu_expected": True,
            "gpu_supported": False,
            "gpu_reason": "nvidia_smi_failed",
            "gpus": [],
        }
    )

    finalized = recorder.finalize()

    assert finalized.summary["resource"]["observation_missing_fraction"] == 1.0


def test_process_tree_cpu_fallback_and_no_gpu_are_normal(monkeypatch, tmp_path):
    class ProcessTreeScope:
        name = "process_tree"

        def __init__(self) -> None:
            self.values = iter([(1.0, 10 * 1024 * 1024), (4.0, 12 * 1024 * 1024)])

        def read(self):
            return next(self.values)

    monkeypatch.setattr(observability_mod, "_resource_scope", ProcessTreeScope)
    monkeypatch.setattr(observability_mod.shutil, "which", lambda name: None)
    probe = observability_mod.SystemResourceProbe()
    recorder = FoldExecutionRecorder(
        tmp_path / "observability",
        {"seeds": [SEED], "source": "local_measured"},
        resource_probe=probe,
        start_sampler=False,
    )
    recorder.start()
    first_ns = time.monotonic_ns()
    first = probe.sample(first_ns)
    second = probe.sample(first_ns + 2_000_000_000)
    assert first["cpu_scope"] == "process_tree"
    assert second["cpu_cores"] == pytest.approx(1.5)
    assert second["gpu_expected"] is False
    assert second["gpu_supported"] is False
    assert second["gpu_reason"] == "gpu_not_allocated"

    recorder.record_resource(first)
    recorder.record_resource(second)
    finalized = recorder.finalize()
    metric_names = {name for name, _, _ in finalized.metrics}
    assert "resource.cpu_cores_mean" in metric_names
    assert "resource.observation_missing_fraction" in metric_names
    assert not any(name.startswith("resource.gpu_") for name in metric_names)
    assert finalized.summary["resource"]["gpu_expected"] is False
    assert finalized.summary["resource"]["gpu_supported"] is False


def test_timing_persistence_failure_is_fatal(tmp_path):
    recorder = FoldExecutionRecorder(
        tmp_path / "observability",
        {"seeds": [SEED], "source": "local_measured"},
        resource_probe=FakeProbe(),
        start_sampler=False,
    )
    recorder.start()

    class BrokenStream:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped

        def write(self, value: str) -> int:
            raise OSError("disk-full")

        def flush(self) -> None:
            self.wrapped.flush()

        def close(self) -> None:
            self.wrapped.close()

    recorder._stream = BrokenStream(recorder._stream)

    recorder.record_timing(
        _event(
            started_ns=time.monotonic_ns(),
            duration_ns=1,
            operation="fold_feature",
        )
    )
    with pytest.raises(ObservabilityPersistenceError, match="기록이 실패"):
        recorder.finalize()
    recorder.abandon()


def test_timing_recording_p99_and_memory_stay_within_contract(tmp_path):
    recorder = FoldExecutionRecorder(
        tmp_path / "observability",
        {"seeds": [SEED], "source": "local_measured"},
        resource_probe=FakeProbe(),
        start_sampler=False,
    )
    recorder.start()
    for _ in range(500):
        recorder.record_timing(
            _event(
                started_ns=time.monotonic_ns(),
                duration_ns=1,
                operation="fold_feature",
            )
        )

    finalized = recorder.finalize()
    overhead = finalized.summary["overhead"]
    assert overhead["timing_event_count"] == 500
    assert overhead["recording_duration_p99_ns"] <= 50_000
    assert overhead["recording_duration_p99_within_limit"] is True
    assert overhead["instrumentation_memory_upper_bound_bytes"] <= 32 * 1024 * 1024
    assert overhead["instrumentation_memory_within_limit"] is True


def test_reader_rejects_unknown_schema_and_operation(tmp_path):
    path = tmp_path / "bad.jsonl.gz"
    records = [
        {"schema_version": 1, "record_type": "metadata", "observed_offset_ns": 0},
        {
            "schema_version": 1,
            "record_type": "timing",
            "seed": SEED,
            "fold": 0,
            "operation": "unknown.operation",
            "actor_kind": "test",
            "actor_name": "test",
            "worker_id": "1",
            "device_id": None,
            "dataset": None,
            "started_offset_ns": 0,
            "duration_ns": 1,
            "outcome": "success",
            "reason": None,
            "source": "local_measured",
        },
        {
            "schema_version": 1,
            "record_type": "summary",
            "ended_offset_ns": 1,
            "timing": {},
            "resource": {},
            "overhead": {},
            "source": "local_measured",
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    with pytest.raises(ObservabilitySchemaError, match="알 수 없는.*작업"):
        read_fold_observability(path)

    records[1]["operation"] = "fold_feature"
    records[0]["schema_version"] = 999
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    with pytest.raises(ObservabilitySchemaError, match="판본"):
        read_fold_observability(path)


class ObserverClient:
    def __init__(self, *, fail_observability: bool = False) -> None:
        self.fail_observability = fail_observability
        self.calls: list[tuple] = []

    def log_metric(self, run_id, name, value, step=None) -> None:
        self.calls.append(("metric", name, value, step))

    def set_tag(self, run_id, name, value) -> None:
        self.calls.append(("tag", name, value))

    def log_artifact(self, run_id, path, artifact_path=None) -> None:
        self.calls.append(("artifact", artifact_path, Path(path).name))
        if self.fail_observability and artifact_path == "observability":
            raise OSError("artifact-store-down")

    def set_terminated(self, run_id, status) -> None:
        self.calls.append(("terminated", status))


def _observer_with_manual_artifacts(tmp_path, monkeypatch, *, fail_observability=False):
    from pipeline import observe as observe_mod

    monkeypatch.setattr(observe_mod, "RUN_LOGS_ROOT", tmp_path / "run-logs")
    cfg = fake_experiment_config()
    client = ObserverClient(fail_observability=fail_observability)
    observer = RunObserver(cfg, client, "run-1")
    observer._run_dir.mkdir(parents=True)
    observer._log_path.write_text("log\n")
    observer._fold_observability = FoldExecutionRecorder(
        observer._run_dir / "observability",
        {"seeds": [SEED], "source": "local_measured"},
        resource_probe=FakeProbe(),
        start_sampler=False,
    )
    observer._fold_observability.start()
    observer._fold_observability.configure_run_shape(
        seed_total=1, fold_total=1, provider_total=0
    )
    return observer, client


def test_run_observer_preserves_observability_before_run_log(tmp_path, monkeypatch):
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    try:
        observer, client = _observer_with_manual_artifacts(tmp_path, monkeypatch)

        observer.succeed()
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)

    artifacts = [call for call in client.calls if call[0] == "artifact"]
    assert artifacts == [
        ("artifact", "observability", ARTIFACT_NAME),
        ("artifact", "logs", "run.log"),
    ]
    assert ("terminated", "FINISHED") in client.calls


def test_run_observer_fails_successful_run_when_observability_cannot_be_preserved(
    tmp_path, monkeypatch
):
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    try:
        observer, client = _observer_with_manual_artifacts(
            tmp_path, monkeypatch, fail_observability=True
        )

        with pytest.raises(ObservabilityPersistenceError, match="보존"):
            observer.succeed()
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)

    assert ("terminated", "FAILED") in client.calls
    assert ("artifact", "logs", "run.log") in client.calls
    assert observer._fold_observability.finalize().path.exists()
