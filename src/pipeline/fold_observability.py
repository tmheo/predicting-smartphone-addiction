"""fold 실행의 판본화 시간 및 자원 관측 원본.

CV와 시드 워커는 저장 위치나 MLflow 이름을 알지 않고 시간 사건만 보낸다.
이 모듈이 고정 이름 검증, 줄 단위 JSON 기록, 자원 표본 수집, 병렬 구간 집계와
압축 산출물 생성을 소유한다.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

SCHEMA_VERSION = 1
ARTIFACT_NAME = "fold_execution.jsonl.gz"
ARTIFACT_PATH = f"observability/{ARTIFACT_NAME}"
SAMPLE_INTERVAL_SECONDS = 2.0
GPU_IDLE_THRESHOLD_PERCENT = 10.0
RECORDING_P99_LIMIT_NS = 50_000
INSTRUMENTATION_CPU_LIMIT_CORES = 0.05
INSTRUMENTATION_MEMORY_LIMIT_BYTES = 32 * 1024 * 1024
MLFLOW_METRICS = frozenset(
    {
        "time.fold_feature_seconds",
        "time.fold_finalize_seconds",
        "time.recovery_seconds",
        "time.model_fit_seconds",
        "time.test_prediction_seconds",
        "time.importance_seconds",
        "resource.cpu_cores_mean",
        "resource.gpu_utilization_mean",
        "resource.gpu_utilization_p95",
        "resource.gpu_idle_fraction",
        "resource.gpu_idle_device_seconds",
        "resource.gpu_all_idle_wall_seconds",
        "resource.gpu_memory_peak_mib",
        "resource.gpu_power_mean_watts",
        "resource.observation_missing_fraction",
    }
)

OPERATIONS = frozenset(
    {
        "fold_feature",
        "fold_feature.provider_fit",
        "fold_feature.provider_transform",
        "fold_finalize",
        "fold_finalize.model_fit",
        "fold_finalize.test_prediction",
        "fold_finalize.importance_prepare",
        "fold_finalize.importance_reinference",
        "fold_finalize.importance_score",
        "fold_finalize.training_diagnostics",
        "fold_finalize.fold_score",
        "training_state.trajectory_fit",
        "recovery.read_validate",
        "recovery.write_commit",
    }
)
OUTCOMES = frozenset({"success", "reused", "skipped", "failed"})


class ObservabilityPersistenceError(RuntimeError):
    """시간 사건 원본을 생성, 기록, 압축 또는 보존할 수 없다."""


class ObservabilitySchemaError(ValueError):
    """지원하지 않거나 불완전한 fold 관측 원본이다."""


class ResourceProbe(Protocol):
    """2초 자원 표본의 물리 측정 뒷단."""

    metadata: dict[str, object]

    def sample(self, observed_ns: int) -> dict[str, object]: ...


@dataclass(frozen=True)
class FinalizedObservability:
    path: Path
    sha256: str
    metrics: list[tuple[str, float, int | None]]
    summary: dict[str, object]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def read_fold_observability(path: Path) -> list[dict[str, object]]:
    """압축 관측 원본 전체를 판본과 고정 작업 이름까지 검증해 읽는다."""
    required = {
        "metadata": {"record_type", "schema_version", "observed_offset_ns"},
        "timing": {
            "record_type",
            "schema_version",
            "seed",
            "fold",
            "operation",
            "actor_kind",
            "actor_name",
            "worker_id",
            "device_id",
            "dataset",
            "started_offset_ns",
            "duration_ns",
            "outcome",
            "reason",
            "source",
        },
        "resource": {
            "record_type",
            "schema_version",
            "observed_offset_ns",
            "interval_ns",
            "cpu_scope",
            "cpu_cores",
            "memory_mib",
            "cpu_supported",
            "cpu_reason",
            "gpu_expected",
            "gpu_supported",
            "gpu_reason",
            "gpus",
            "source",
        },
        "summary": {
            "record_type",
            "schema_version",
            "ended_offset_ns",
            "timing",
            "resource",
            "overhead",
            "source",
        },
    }
    records: list[dict[str, object]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ObservabilitySchemaError(
                        f"fold 관측 {line_number}행이 JSON 객체가 아니다."
                    )
                if record.get("schema_version") != SCHEMA_VERSION:
                    raise ObservabilitySchemaError(
                        f"지원하지 않는 fold 관측 스키마 판본이다: "
                        f"{record.get('schema_version')}"
                    )
                record_type = record.get("record_type")
                if record_type not in required:
                    raise ObservabilitySchemaError(
                        f"알 수 없는 fold 관측 record_type이다: {record_type}"
                    )
                missing = required[record_type] - set(record)
                if missing:
                    raise ObservabilitySchemaError(
                        f"fold 관측 {record_type} 필수 필드가 없다: {sorted(missing)}"
                    )
                if record_type == "timing":
                    if record["operation"] not in OPERATIONS:
                        raise ObservabilitySchemaError(
                            f"알 수 없는 fold 관측 작업 이름이다: {record['operation']}"
                        )
                    if record["outcome"] not in OUTCOMES:
                        raise ObservabilitySchemaError(
                            f"알 수 없는 fold 관측 결과다: {record['outcome']}"
                        )
                records.append(record)
    except ObservabilitySchemaError:
        raise
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObservabilitySchemaError(f"fold 관측 원본을 읽을 수 없다: {path}") from exc
    if not records or records[0].get("record_type") != "metadata":
        raise ObservabilitySchemaError("fold 관측 원본의 첫 행은 metadata여야 한다.")
    if records[-1].get("record_type") != "summary":
        raise ObservabilitySchemaError("fold 관측 원본의 마지막 행은 summary여야 한다.")
    return records


def _measurement_source() -> str:
    remote_markers = (
        "KAGGLE_KERNEL_RUN_TYPE",
        "RUNPOD_POD_ID",
        "VAST_CONTAINERLABEL",
        "VAST_INSTANCE_ID",
        "REMOTE_RUN_JOB_ID",
    )
    return (
        "remote_measured"
        if any(os.environ.get(name) for name in remote_markers)
        else "local_measured"
    )


def _cpu_model() -> str | None:
    processor = platform.processor().strip()
    if processor:
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", maxsplit=1)[1].strip()
    return None


def runtime_metadata(cfg: Any, run_id: str) -> dict[str, object]:
    """실행 시작 시점에 비밀 없이 고정할 수 있는 장비와 계측기 정보."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    seed_gpus = os.environ.get("PIPELINE_SEED_GPUS")
    fold_gpus = os.environ.get("PIPELINE_FOLD_GPUS")
    return {
        "run_id": run_id,
        "experiment": cfg.name,
        "model_kind": cfg.model.kind,
        "seeds": list(cfg.seeds),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "cuda_visible_devices": visible,
        "seed_gpu_assignment": seed_gpus,
        "fold_gpu_assignment": fold_gpus,
        "shared_fold_seed_workers": os.environ.get(
            "PIPELINE_SHARED_FOLD_SEED_WORKERS"
        ),
        "remote_job_id": os.environ.get("REMOTE_RUN_JOB_ID"),
        "remote_provider": os.environ.get("REMOTE_RUN_PROVIDER"),
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "gpu_idle_threshold_percent": GPU_IDLE_THRESHOLD_PERCENT,
        "source": _measurement_source(),
    }


def _device_id() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or visible.strip() == "-1":
        return None
    return visible.strip()


def _deliver_timing(recorder: object | None, event: dict[str, object]) -> None:
    if recorder is None:
        return
    sink = getattr(recorder, "record_timing", None)
    if sink is not None:
        sink(event)


@contextmanager
def timed_operation(
    recorder: object | None,
    *,
    seed: int,
    fold: int,
    operation: str,
    actor_kind: str,
    actor_name: str,
    dataset: str | None = None,
    outcome: str = "success",
    reason: str | None = None,
) -> Iterator[None]:
    """도메인 호출 하나를 재지 않고도 끌 수 있는 좁은 통지 경계로 감싼다."""
    if recorder is None or getattr(recorder, "record_timing", None) is None:
        yield
        return
    started_ns = time.monotonic_ns()
    try:
        yield
    except BaseException as exc:
        _deliver_timing(
            recorder,
            {
                "seed": seed,
                "fold": fold,
                "operation": operation,
                "actor_kind": actor_kind,
                "actor_name": actor_name,
                "worker_id": str(os.getpid()),
                "device_id": _device_id(),
                "dataset": dataset,
                "started_ns": started_ns,
                "duration_ns": time.monotonic_ns() - started_ns,
                "outcome": "failed",
                "reason": type(exc).__name__,
            },
        )
        raise
    _deliver_timing(
        recorder,
        {
            "seed": seed,
            "fold": fold,
            "operation": operation,
            "actor_kind": actor_kind,
            "actor_name": actor_name,
            "worker_id": str(os.getpid()),
            "device_id": _device_id(),
            "dataset": dataset,
            "started_ns": started_ns,
            "duration_ns": time.monotonic_ns() - started_ns,
            "outcome": outcome,
            "reason": reason,
        },
    )


def skipped_operation(
    recorder: object | None,
    *,
    seed: int,
    fold: int,
    operation: str,
    actor_kind: str,
    actor_name: str,
    reason: str,
    dataset: str | None = None,
) -> None:
    """실제로 수행하지 않은 작업은 0초가 아니라 duration_ns=null로 남긴다."""
    _deliver_timing(
        recorder,
        {
            "seed": seed,
            "fold": fold,
            "operation": operation,
            "actor_kind": actor_kind,
            "actor_name": actor_name,
            "worker_id": str(os.getpid()),
            "device_id": _device_id(),
            "dataset": dataset,
            "started_ns": time.monotonic_ns(),
            "duration_ns": None,
            "outcome": "skipped",
            "reason": reason,
        },
    )


def recorded_operation(
    recorder: object | None,
    *,
    seed: int,
    fold: int,
    operation: str,
    actor_kind: str,
    actor_name: str,
    started_ns: int,
    duration_ns: int,
    outcome: str,
    reason: str | None = None,
    dataset: str | None = None,
) -> None:
    """다른 구성 요소가 잰 실제 구간을 고정 시간 사건으로 전달한다."""
    _deliver_timing(
        recorder,
        {
            "seed": seed,
            "fold": fold,
            "operation": operation,
            "actor_kind": actor_kind,
            "actor_name": actor_name,
            "worker_id": str(os.getpid()),
            "device_id": _device_id(),
            "dataset": dataset,
            "started_ns": started_ns,
            "duration_ns": duration_ns,
            "outcome": outcome,
            "reason": reason,
        },
    )


class _CgroupScope:
    name = "cgroup"

    def __init__(self, root: Path = Path("/sys/fs/cgroup")) -> None:
        self._cpu_stat = root / "cpu.stat"
        self._memory = root / "memory.current"
        if not self._cpu_stat.is_file() or not self._memory.is_file():
            raise OSError("cgroup v2 자원 파일이 없다.")
        self.read()

    def read(self) -> tuple[float, int]:
        fields = dict(
            line.split(maxsplit=1) for line in self._cpu_stat.read_text().splitlines()
        )
        return int(fields["usage_usec"]) / 1_000_000.0, int(self._memory.read_text())


class _ProcessTreeScope:
    name = "process_tree"

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._root = psutil.Process()

    def read(self) -> tuple[float, int]:
        cpu_seconds = 0.0
        memory_bytes = 0
        try:
            processes = [self._root, *self._root.children(recursive=True)]
        except self._psutil.Error:
            processes = [self._root]
        for process in processes:
            try:
                cpu = process.cpu_times()
                cpu_seconds += float(cpu.user + cpu.system)
                memory_bytes += int(process.memory_info().rss)
            except self._psutil.Error:
                continue
        return cpu_seconds, memory_bytes


def _resource_scope() -> _CgroupScope | _ProcessTreeScope:
    try:
        return _CgroupScope()
    except (OSError, KeyError, TypeError, ValueError):
        return _ProcessTreeScope()


def _float_or_none(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


class SystemResourceProbe:
    """cgroup 우선 CPU 측정과 nvidia-smi 장치 표본을 한 관측 구간으로 묶는다."""

    def __init__(self) -> None:
        self._scope = _resource_scope()
        self._previous_observed_ns: int | None = None
        self._previous_cpu_seconds: float | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        seed_gpus = os.environ.get("PIPELINE_SEED_GPUS")
        selected = visible if visible is not None else seed_gpus
        self._selected_devices = (
            [item.strip() for item in selected.split(",") if item.strip()]
            if selected and selected.strip() != "-1"
            else []
        )
        explicitly_disabled = visible is not None and (
            not visible.strip() or visible.strip() == "-1"
        )
        self._gpu_expected = not explicitly_disabled and (
            self._nvidia_smi is not None or bool(self._selected_devices)
        )
        self.metadata = {
            "cpu_scope": self._scope.name,
            "gpu_expected": self._gpu_expected,
            "nvidia_smi_available": self._nvidia_smi is not None,
            "selected_devices": self._selected_devices,
        }

    def _gpu_sample(self) -> tuple[list[dict[str, object]], str | None]:
        if self._nvidia_smi is None:
            return [], "nvidia_smi_unavailable" if self._gpu_expected else "gpu_not_allocated"
        command = [
            self._nvidia_smi,
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            reason = completed.stderr.strip().splitlines()
            return [], f"nvidia_smi_failed:{(reason or ['unknown'])[0][:200]}"
        devices: list[dict[str, object]] = []
        selected = set(self._selected_devices)
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", maxsplit=5)]
            if len(parts) != 6:
                continue
            index, uuid, name, utilization, memory, power = parts
            if selected and index not in selected and uuid not in selected:
                continue
            devices.append(
                {
                    "device_id": index,
                    "uuid": uuid,
                    "name": name,
                    "utilization_percent": _float_or_none(utilization),
                    "memory_used_mib": _float_or_none(memory),
                    "power_watts": _float_or_none(power),
                }
            )
        if not devices:
            return [], "selected_gpu_not_reported" if selected else "gpu_not_allocated"
        return devices, None

    def sample(self, observed_ns: int) -> dict[str, object]:
        cpu_error = None
        try:
            cpu_seconds, memory_bytes = self._scope.read()
        except Exception as exc:  # noqa: BLE001 - 자원 표본 실패는 실행 실패가 아니다.
            cpu_seconds, memory_bytes = None, None
            cpu_error = type(exc).__name__
        interval_ns = (
            None
            if self._previous_observed_ns is None
            else observed_ns - self._previous_observed_ns
        )
        cpu_cores = None
        if (
            interval_ns is not None
            and interval_ns > 0
            and cpu_seconds is not None
            and self._previous_cpu_seconds is not None
        ):
            cpu_cores = max(
                0.0,
                (cpu_seconds - self._previous_cpu_seconds) / (interval_ns / 1_000_000_000),
            )
        self._previous_observed_ns = observed_ns
        if cpu_seconds is not None:
            self._previous_cpu_seconds = cpu_seconds
        try:
            devices, gpu_reason = self._gpu_sample()
        except Exception as exc:  # noqa: BLE001 - 자원 표본 실패는 실행 실패가 아니다.
            devices, gpu_reason = [], type(exc).__name__
        return {
            "observed_ns": observed_ns,
            "interval_ns": interval_ns,
            "cpu_scope": self._scope.name,
            "cpu_cores": cpu_cores,
            "memory_mib": (
                None if memory_bytes is None else memory_bytes / (1024.0 * 1024.0)
            ),
            "cpu_supported": cpu_error is None,
            "cpu_reason": cpu_error,
            "gpu_expected": self._gpu_expected,
            "gpu_supported": bool(devices),
            "gpu_reason": gpu_reason,
            "gpus": devices,
        }


def _interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _weighted_percentile(
    values: list[tuple[float, int]], quantile: float
) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    total_weight = sum(weight for _, weight in ordered)
    threshold = total_weight * quantile
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


class FoldExecutionRecorder:
    """한 실행의 관측 원본과 집계를 소유하는 기록기."""

    def __init__(
        self,
        root: Path,
        metadata: dict[str, object],
        *,
        resource_probe: ResourceProbe | None = None,
        sample_interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
        start_sampler: bool = True,
    ) -> None:
        self.root = root
        self._metadata = dict(metadata)
        self._sample_interval_seconds = sample_interval_seconds
        self._probe = resource_probe or SystemResourceProbe()
        self._start_sampler = start_sampler
        self._t0_ns = time.monotonic_ns()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._timing_writer: threading.Thread | None = None
        self._timing_queue: queue.SimpleQueue[dict[str, object] | object] = (
            queue.SimpleQueue()
        )
        self._timing_sentinel = object()
        self._stream = None
        self._raw_path = root / ARTIFACT_NAME.removesuffix(".gz")
        self._final_path = root / ARTIFACT_NAME
        self._timings: list[dict[str, object]] = []
        self._resources: list[dict[str, object]] = []
        self._recording_cost_ns: list[int] = []
        self._instrumentation_cpu_ns = 0
        self._serialized_bytes = 0
        self._event_count_limit: int | None = None
        self._fold_total: int | None = None
        self._last_resource_observed_ns: int | None = None
        self._fatal_error: BaseException | None = None
        self._finalized: FinalizedObservability | None = None

    def start(self) -> None:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
            self._stream = self._raw_path.open("x", encoding="utf-8", buffering=64 * 1024)
            self.record_metadata({**self._metadata, "resource_probe": self._probe.metadata})
            self._timing_writer = threading.Thread(
                target=self._timing_writer_loop,
                name="fold-timing-writer",
                daemon=True,
            )
            self._timing_writer.start()
            if self._start_sampler:
                self._sampler = threading.Thread(
                    target=self._resource_loop,
                    name="fold-resource-sampler",
                    daemon=True,
                )
                self._sampler.start()
        except BaseException as exc:
            raise ObservabilityPersistenceError("fold 관측 원본을 시작할 수 없다.") from exc

    def _check_fatal(self) -> None:
        if self._fatal_error is not None:
            raise ObservabilityPersistenceError("fold 관측 원본 기록이 실패했다.") from self._fatal_error

    def _write(self, record: dict[str, object], *, measure_cost: bool = True) -> None:
        self._check_fatal()
        if self._stream is None:
            raise ObservabilityPersistenceError("fold 관측 원본이 열려 있지 않다.")
        full = {"schema_version": SCHEMA_VERSION, **record}
        started_cpu_ns = time.thread_time_ns()
        try:
            encoded = _canonical_json(full) + "\n"
            with self._lock:
                self._stream.write(encoded)
                self._serialized_bytes += len(encoded.encode())
        except BaseException as exc:
            self._fatal_error = exc
            raise ObservabilityPersistenceError("fold 관측 원본을 직렬화하거나 기록할 수 없다.") from exc
        cost_ns = time.thread_time_ns() - started_cpu_ns
        self._instrumentation_cpu_ns += cost_ns
        if measure_cost:
            self._recording_cost_ns.append(cost_ns)

    def _timing_writer_loop(self) -> None:
        while True:
            persisted = self._timing_queue.get()
            if persisted is self._timing_sentinel:
                return
            if self._fatal_error is not None:
                continue
            try:
                self._write(persisted, measure_cost=False)
            except BaseException as exc:  # noqa: BLE001 - 부모 호출이나 finalize가 치명 오류로 올린다.
                self._fatal_error = exc

    def _stop_timing_writer(self) -> None:
        if self._timing_writer is None:
            return
        self._timing_queue.put(self._timing_sentinel)
        self._timing_writer.join(timeout=10.0)
        if self._timing_writer.is_alive():
            self._fatal_error = TimeoutError("fold 시간 사건 기록 스레드가 끝나지 않았다.")
        self._timing_writer = None

    def record_metadata(self, values: dict[str, object]) -> None:
        self._write(
            {
                "record_type": "metadata",
                "observed_offset_ns": max(0, time.monotonic_ns() - self._t0_ns),
                **values,
            },
            measure_cost=False,
        )

    def record_execution_identity(self, identity: dict[str, object]) -> None:
        self.record_metadata(
            {
                "metadata_kind": "execution_identity",
                "execution_identity_sha256": _content_sha256(identity),
            }
        )

    def configure_run_shape(self, *, seed_total: int, fold_total: int, provider_total: int) -> None:
        self._fold_total = fold_total
        self._event_count_limit = seed_total * fold_total * (12 + 3 * provider_total)
        self.record_metadata(
            {
                "metadata_kind": "run_shape",
                "seed_total": seed_total,
                "fold_total": fold_total,
                "fold_fit_provider_total": provider_total,
                "timing_event_limit": self._event_count_limit,
            }
        )

    def record_timing(self, event: dict[str, object]) -> None:
        operation = event.get("operation")
        outcome = event.get("outcome")
        duration_ns = event.get("duration_ns")
        if operation not in OPERATIONS:
            raise ObservabilityPersistenceError(f"알 수 없는 fold 관측 작업 이름이다: {operation}")
        if outcome not in OUTCOMES:
            raise ObservabilityPersistenceError(f"알 수 없는 fold 관측 결과다: {outcome}")
        if outcome == "skipped" and duration_ns is not None:
            raise ObservabilityPersistenceError("생략된 fold 작업의 duration_ns는 null이어야 한다.")
        if outcome != "skipped" and (
            isinstance(duration_ns, bool) or not isinstance(duration_ns, int) or duration_ns < 0
        ):
            raise ObservabilityPersistenceError("실행한 fold 작업의 duration_ns가 잘못됐다.")
        started_ns = event.get("started_ns")
        if isinstance(started_ns, bool) or not isinstance(started_ns, int):
            raise ObservabilityPersistenceError("fold 시간 사건의 started_ns가 잘못됐다.")
        started_offset_ns = started_ns - self._t0_ns
        if started_offset_ns < 0:
            raise ObservabilityPersistenceError("fold 시간 사건이 실행 시작보다 앞선다.")
        reason = event.get("reason")
        if outcome in {"skipped", "failed"} and not reason:
            raise ObservabilityPersistenceError("생략 또는 실패한 fold 작업에는 reason이 필요하다.")
        persisted = {
            "record_type": "timing",
            "seed": int(event["seed"]),
            "fold": int(event["fold"]),
            "operation": operation,
            "actor_kind": str(event["actor_kind"]),
            "actor_name": str(event["actor_name"]),
            "worker_id": str(event["worker_id"]),
            "device_id": event.get("device_id"),
            "dataset": event.get("dataset"),
            "started_offset_ns": started_offset_ns,
            "duration_ns": duration_ns,
            "outcome": outcome,
            "reason": reason,
            "source": self._metadata.get("source", "local_measured"),
        }
        if self._event_count_limit is not None and len(self._timings) >= self._event_count_limit:
            raise ObservabilityPersistenceError(
                f"fold 시간 사건 수가 상한을 넘었다: "
                f"{len(self._timings) + 1} > {self._event_count_limit}"
            )
        self._check_fatal()
        started_cpu_ns = time.thread_time_ns()
        self._timing_queue.put(persisted)
        cost_ns = time.thread_time_ns() - started_cpu_ns
        self._instrumentation_cpu_ns += cost_ns
        self._recording_cost_ns.append(cost_ns)
        self._timings.append(persisted)

    def record_resource(self, sample: dict[str, object]) -> None:
        observed_ns = sample.get("observed_ns")
        if isinstance(observed_ns, bool) or not isinstance(observed_ns, int):
            raise ObservabilityPersistenceError("자원 표본 시각이 잘못됐다.")
        observed_offset_ns = observed_ns - self._t0_ns
        if observed_offset_ns < 0:
            raise ObservabilityPersistenceError("자원 표본 시각이 실행 시작보다 앞선다.")
        normalized = dict(sample)
        if normalized.get("interval_ns") is None and self._last_resource_observed_ns is not None:
            normalized["interval_ns"] = observed_ns - self._last_resource_observed_ns
        self._last_resource_observed_ns = observed_ns
        persisted = {
            "record_type": "resource",
            **normalized,
            "observed_offset_ns": observed_offset_ns,
            "source": self._metadata.get("source", "local_measured"),
        }
        persisted.pop("observed_ns", None)
        self._write(persisted, measure_cost=False)
        self._resources.append(persisted)

    def _sample_once(self) -> None:
        started_cpu_ns = time.thread_time_ns()
        observed_ns = time.monotonic_ns()
        try:
            sample = self._probe.sample(observed_ns)
        except BaseException as exc:  # noqa: BLE001 - 측정 실패는 결측 표본으로 보존한다.
            sample = {
                "observed_ns": observed_ns,
                "interval_ns": None,
                "cpu_scope": self._probe.metadata.get("cpu_scope"),
                "cpu_cores": None,
                "memory_mib": None,
                "cpu_supported": False,
                "cpu_reason": type(exc).__name__,
                "gpu_expected": bool(self._probe.metadata.get("gpu_expected")),
                "gpu_supported": False,
                "gpu_reason": type(exc).__name__,
                "gpus": [],
            }
        probe_cpu_ns = time.thread_time_ns() - started_cpu_ns
        try:
            self.record_resource(sample)
        except BaseException as exc:
            self._fatal_error = exc
        self._instrumentation_cpu_ns += probe_cpu_ns

    def _resource_loop(self) -> None:
        self._sample_once()
        while not self._stop.wait(self._sample_interval_seconds):
            self._sample_once()

    def _timing_summary(self) -> tuple[dict[str, object], list[tuple[str, float, int | None]]]:
        by_operation: dict[str, dict[str, object]] = {}
        for operation in sorted(OPERATIONS):
            rows = [
                row
                for row in self._timings
                if row["operation"] == operation and row["duration_ns"] is not None
            ]
            intervals = [
                (
                    int(row["started_offset_ns"]),
                    int(row["started_offset_ns"]) + int(row["duration_ns"]),
                )
                for row in rows
            ]
            by_operation[operation] = {
                "event_count": sum(row["operation"] == operation for row in self._timings),
                "completed_count": len(rows),
                "cumulative_seconds": sum(int(row["duration_ns"]) for row in rows)
                / 1_000_000_000,
                "union_seconds": _interval_union_ns(intervals) / 1_000_000_000,
            }

        unclassified: list[dict[str, object]] = []
        for parent in ("fold_feature", "fold_finalize"):
            for row in [item for item in self._timings if item["operation"] == parent]:
                if row["duration_ns"] is None:
                    continue
                parent_start = int(row["started_offset_ns"])
                parent_end = parent_start + int(row["duration_ns"])
                children = []
                for child in self._timings:
                    if (
                        str(child["operation"]).startswith(parent + ".")
                        and child["seed"] == row["seed"]
                        and child["fold"] == row["fold"]
                        and child["worker_id"] == row["worker_id"]
                        and child["duration_ns"] is not None
                    ):
                        start = int(child["started_offset_ns"])
                        end = start + int(child["duration_ns"])
                        if start < parent_start or end > parent_end:
                            raise ObservabilityPersistenceError(
                                f"{child['operation']} 구간이 {parent} 상위 구간 밖에 있다."
                            )
                        if end > start:
                            children.append((start, end))
                remainder_ns = int(row["duration_ns"]) - _interval_union_ns(children)
                if remainder_ns < 0:
                    raise ObservabilityPersistenceError(
                        f"{parent} 하위 구간 합집합이 상위 구간보다 길다."
                    )
                unclassified.append(
                    {
                        "seed": row["seed"],
                        "fold": row["fold"],
                        "operation": parent,
                        "unclassified_seconds": remainder_ns / 1_000_000_000,
                    }
                )

        metrics: list[tuple[str, float, int | None]] = []
        if self._fold_total is not None:
            for seed_index, seed in enumerate(self._metadata.get("seeds", [])):
                for fold in range(self._fold_total):
                    rows = [
                        row
                        for row in self._timings
                        if row["seed"] == seed
                        and row["fold"] == fold
                        and row["duration_ns"] is not None
                    ]
                    step = seed_index * self._fold_total + fold

                    def seconds_of(names: set[str]) -> float | None:
                        values = [int(row["duration_ns"]) for row in rows if row["operation"] in names]
                        return sum(values) / 1_000_000_000 if values else None

                    fixed = {
                        "time.fold_feature_seconds": seconds_of({"fold_feature"}),
                        "time.fold_finalize_seconds": seconds_of({"fold_finalize"}),
                        "time.recovery_seconds": seconds_of(
                            {"recovery.read_validate", "recovery.write_commit"}
                        ),
                        "time.model_fit_seconds": seconds_of({"fold_finalize.model_fit"}),
                        "time.test_prediction_seconds": seconds_of(
                            {"fold_finalize.test_prediction"}
                        ),
                        "time.importance_seconds": seconds_of(
                            {
                                "fold_finalize.importance_prepare",
                                "fold_finalize.importance_reinference",
                                "fold_finalize.importance_score",
                            }
                        ),
                    }
                    metrics.extend(
                        (name, value, step) for name, value in fixed.items() if value is not None
                    )
        worker_assignments = sorted(
            {
                (
                    str(row["worker_id"]),
                    None if row["device_id"] is None else str(row["device_id"]),
                    int(row["seed"]),
                )
                for row in self._timings
            },
            key=lambda item: (item[0], item[1] or "", item[2]),
        )
        return {
            "operations": by_operation,
            "unclassified": unclassified,
            "worker_assignments": [
                {"worker_id": worker, "device_id": device, "seed": seed}
                for worker, device, seed in worker_assignments
            ],
        }, metrics

    def _resource_summary(self) -> tuple[dict[str, object], list[tuple[str, float, int | None]]]:
        valid = [
            row
            for row in self._resources
            if isinstance(row.get("interval_ns"), int) and int(row["interval_ns"]) > 0
        ]
        total_interval_ns = sum(int(row["interval_ns"]) for row in valid)
        cpu_weighted = [
            (float(row["cpu_cores"]), int(row["interval_ns"]))
            for row in valid
            if row.get("cpu_cores") is not None
        ]
        cpu_observed_ns = sum(weight for _, weight in cpu_weighted)
        cpu_mean = (
            sum(value * weight for value, weight in cpu_weighted) / cpu_observed_ns
            if cpu_observed_ns
            else None
        )
        missing_ns = 0
        gpu_values: list[tuple[float, int]] = []
        gpu_weighted_sum = 0.0
        gpu_observed_device_ns = 0
        gpu_idle_device_ns = 0
        gpu_all_idle_wall_ns = 0
        gpu_power_weighted_sum = 0.0
        gpu_power_observed_ns = 0
        gpu_memory_values: list[float] = []
        gpu_expected = any(bool(row.get("gpu_expected")) for row in valid)
        for row in valid:
            interval_ns = int(row["interval_ns"])
            cpu_missing = not row.get("cpu_supported") or row.get("cpu_cores") is None
            devices = row.get("gpus") or []
            utilizations = []
            for device in devices:
                utilization = device.get("utilization_percent")
                if utilization is not None:
                    value = float(utilization)
                    utilizations.append(value)
                    gpu_values.append((value, interval_ns))
                    gpu_weighted_sum += value * interval_ns
                    gpu_observed_device_ns += interval_ns
                    if value <= GPU_IDLE_THRESHOLD_PERCENT:
                        gpu_idle_device_ns += interval_ns
                memory = device.get("memory_used_mib")
                if memory is not None:
                    gpu_memory_values.append(float(memory))
                power = device.get("power_watts")
                if power is not None:
                    gpu_power_weighted_sum += float(power) * interval_ns
                    gpu_power_observed_ns += interval_ns
            gpu_missing = bool(row.get("gpu_expected")) and (
                not row.get("gpu_supported")
                or not devices
                or len(utilizations) != len(devices)
            )
            if cpu_missing or gpu_missing:
                missing_ns += interval_ns
            if (
                utilizations
                and len(utilizations) == len(devices)
                and all(value <= GPU_IDLE_THRESHOLD_PERCENT for value in utilizations)
            ):
                gpu_all_idle_wall_ns += interval_ns
        missing_fraction = missing_ns / total_interval_ns if total_interval_ns else 0.0
        memory_values = [
            float(row["memory_mib"]) for row in valid if row.get("memory_mib") is not None
        ]
        resource = {
            "sample_count": len(self._resources),
            "weighted_interval_seconds": total_interval_ns / 1_000_000_000,
            "cpu_scope": self._probe.metadata.get("cpu_scope"),
            "cpu_cores_mean": cpu_mean,
            "memory_peak_mib": max(memory_values) if memory_values else None,
            "gpu_expected": gpu_expected,
            "gpu_supported": gpu_observed_device_ns > 0,
            "gpu_utilization_mean": (
                gpu_weighted_sum / gpu_observed_device_ns
                if gpu_observed_device_ns
                else None
            ),
            "gpu_utilization_p95": _weighted_percentile(gpu_values, 0.95),
            "gpu_idle_fraction": (
                gpu_idle_device_ns / gpu_observed_device_ns
                if gpu_observed_device_ns
                else None
            ),
            "gpu_idle_device_seconds": (
                gpu_idle_device_ns / 1_000_000_000
                if gpu_observed_device_ns
                else None
            ),
            "gpu_all_idle_wall_seconds": (
                gpu_all_idle_wall_ns / 1_000_000_000
                if gpu_observed_device_ns
                else None
            ),
            "gpu_memory_peak_mib": max(gpu_memory_values) if gpu_memory_values else None,
            "gpu_power_mean_watts": (
                gpu_power_weighted_sum / gpu_power_observed_ns
                if gpu_power_observed_ns
                else None
            ),
            "observation_missing_fraction": missing_fraction,
        }
        metric_names = {
            "resource.cpu_cores_mean": resource["cpu_cores_mean"],
            "resource.gpu_utilization_mean": resource["gpu_utilization_mean"],
            "resource.gpu_utilization_p95": resource["gpu_utilization_p95"],
            "resource.gpu_idle_fraction": resource["gpu_idle_fraction"],
            "resource.gpu_idle_device_seconds": resource["gpu_idle_device_seconds"],
            "resource.gpu_all_idle_wall_seconds": resource["gpu_all_idle_wall_seconds"],
            "resource.gpu_memory_peak_mib": resource["gpu_memory_peak_mib"],
            "resource.gpu_power_mean_watts": resource["gpu_power_mean_watts"],
            "resource.observation_missing_fraction": resource[
                "observation_missing_fraction"
            ],
        }
        metrics = [
            (name, float(value), None)
            for name, value in metric_names.items()
            if value is not None
        ]
        return resource, metrics

    def finalize(self) -> FinalizedObservability:
        if self._finalized is not None:
            return self._finalized
        try:
            self._stop.set()
            if self._sampler is not None:
                self._sampler.join(timeout=max(10.0, self._sample_interval_seconds * 2))
            self._stop_timing_writer()
            self._check_fatal()
            timing, timing_metrics = self._timing_summary()
            resource, resource_metrics = self._resource_summary()
            elapsed_ns = max(1, time.monotonic_ns() - self._t0_ns)
            p99_ns = _percentile([float(value) for value in self._recording_cost_ns], 0.99)
            instrumentation_cpu_cores = self._instrumentation_cpu_ns / elapsed_ns
            memory_upper_bound = (
                self._serialized_bytes * 4
                + len(self._timings) * 512
                + len(self._resources) * 1024
            )
            overhead = {
                "timing_event_count": len(self._timings),
                "timing_event_limit": self._event_count_limit,
                "recording_duration_p99_ns": p99_ns,
                "recording_duration_p99_within_limit": (
                    p99_ns is None or p99_ns <= RECORDING_P99_LIMIT_NS
                ),
                "instrumentation_cpu_cores_mean": instrumentation_cpu_cores,
                "instrumentation_cpu_within_limit": (
                    instrumentation_cpu_cores <= INSTRUMENTATION_CPU_LIMIT_CORES
                ),
                "instrumentation_memory_upper_bound_bytes": memory_upper_bound,
                "instrumentation_memory_within_limit": (
                    memory_upper_bound <= INSTRUMENTATION_MEMORY_LIMIT_BYTES
                ),
            }
            summary = {
                "record_type": "summary",
                "ended_offset_ns": elapsed_ns,
                "timing": timing,
                "resource": resource,
                "overhead": overhead,
                "source": self._metadata.get("source", "local_measured"),
            }
            self._write(summary, measure_cost=False)
            with self._lock:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()
                self._stream = None
            temporary = self._final_path.with_suffix(self._final_path.suffix + ".tmp")
            with self._raw_path.open("rb") as source, temporary.open("xb") as target:
                with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self._final_path)
            self._raw_path.unlink()
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._finalized = FinalizedObservability(
                path=self._final_path,
                sha256=_sha256(self._final_path),
                metrics=[*timing_metrics, *resource_metrics],
                summary=summary,
            )
            return self._finalized
        except ObservabilityPersistenceError:
            raise
        except BaseException as exc:
            self._fatal_error = exc
            raise ObservabilityPersistenceError("fold 관측 원본을 확정할 수 없다.") from exc

    def abandon(self) -> None:
        """시작 실패 때 표본 스레드와 파일 기술자만 닫고 로컬 원본은 남긴다."""
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=max(10.0, self._sample_interval_seconds * 2))
        self._stop_timing_writer()
        if self._stream is not None:
            try:
                self._stream.flush()
                self._stream.close()
            finally:
                self._stream = None
