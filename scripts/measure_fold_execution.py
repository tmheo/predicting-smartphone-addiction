#!/usr/bin/env python3
"""정식 실행의 계산을 바꾸지 않고 폴드 세부 시간을 임시 계측한다.

이 스크립트는 이슈 320의 근거 수집 전용이다.
제품 관측 계약을 정하지 않기 위해 기존 객체의 좁은 메서드만 실행 중에 감싼다.
시드 병렬 실행은 지원하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import MethodType
from typing import Any


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Recorder:
    def __init__(self, label: str) -> None:
        self.label = label
        self.started_at = _utc_now()
        self.started_ns = time.perf_counter_ns()
        self.process_started = time.process_time()
        self.events: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def record(
        self,
        category: str,
        operation: str,
        started_ns: int,
        **metadata: object,
    ) -> None:
        finished_ns = time.perf_counter_ns()
        event = {
            "sequence": 0,
            "category": category,
            "operation": operation,
            "started_offset_seconds": (started_ns - self.started_ns) / 1e9,
            "duration_seconds": (finished_ns - started_ns) / 1e9,
            **metadata,
        }
        with self._lock:
            event["sequence"] = len(self.events)
            self.events.append(event)


_STATE = threading.local()


def _set_state(**values: object) -> dict[str, object]:
    previous = {name: getattr(_STATE, name, None) for name in values}
    for name, value in values.items():
        setattr(_STATE, name, value)
    return previous


def _restore_state(previous: dict[str, object]) -> None:
    for name, value in previous.items():
        setattr(_STATE, name, value)


def _install_plan_timing(recorder: _Recorder) -> None:
    from pipeline.plan import FOLD_FIT, FeaturePlan

    original_from_config = FeaturePlan.from_config

    def measured_from_config(cls, cfg):
        plan = original_from_config(cfg)
        for provider_kind, provider in plan._stages[FOLD_FIT]:
            state = {"fold": -1, "transform_call": 0}
            original_fit = provider.fit
            original_transform = provider.transform

            def measured_fit(
                _self,
                train_fold,
                seed,
                *,
                _kind=provider_kind,
                _original=original_fit,
                _state=state,
            ):
                _state["fold"] += 1
                _state["transform_call"] = 0
                fold = int(_state["fold"])
                _STATE.fold = fold
                started_ns = time.perf_counter_ns()
                try:
                    return _original(train_fold, seed)
                finally:
                    recorder.record(
                        "fold_fit",
                        "fit",
                        started_ns,
                        provider=_kind,
                        provider_class=type(_self).__name__,
                        seed=int(seed),
                        fold=fold,
                        rows=len(train_fold),
                    )

            def measured_transform(
                _self,
                frame,
                *,
                _kind=provider_kind,
                _original=original_transform,
                _state=state,
            ):
                _state["transform_call"] += 1
                transform_call = int(_state["transform_call"])
                target = "train_all" if transform_call % 2 == 1 else "test"
                started_ns = time.perf_counter_ns()
                try:
                    return _original(frame)
                finally:
                    recorder.record(
                        "fold_fit",
                        "transform",
                        started_ns,
                        provider=_kind,
                        provider_class=type(_self).__name__,
                        fold=int(_state["fold"]),
                        target=target,
                        rows=len(frame),
                    )

            provider.fit = MethodType(measured_fit, provider)
            provider.transform = MethodType(measured_transform, provider)
        return plan

    FeaturePlan.from_config = classmethod(measured_from_config)


class _MeasuredAdapter:
    def __init__(self, delegate, recorder: _Recorder, model_kind: str, seed: int, fold: int):
        self._delegate = delegate
        self._recorder = recorder
        self._model_kind = model_kind
        self._seed = seed
        self._fold = fold

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def fit(self, *args, **kwargs):
        previous = _set_state(
            model_phase="model_fit", model_kind=self._model_kind, fold=self._fold
        )
        started_ns = time.perf_counter_ns()
        try:
            return self._delegate.fit(*args, **kwargs)
        finally:
            self._recorder.record(
                "model",
                "fit_with_validation_prediction",
                started_ns,
                model_kind=self._model_kind,
                seed=self._seed,
                fold=self._fold,
            )
            _restore_state(previous)

    def predict(self, *args, **kwargs):
        previous = _set_state(
            model_phase="test_prediction", model_kind=self._model_kind, fold=self._fold
        )
        started_ns = time.perf_counter_ns()
        try:
            return self._delegate.predict(*args, **kwargs)
        finally:
            self._recorder.record(
                "model",
                "test_prediction",
                started_ns,
                model_kind=self._model_kind,
                seed=self._seed,
                fold=self._fold,
                rows=len(args[0]) if args else None,
            )
            _restore_state(previous)

    def importance(self):
        previous = _set_state(
            model_phase="importance", model_kind=self._model_kind, fold=self._fold
        )
        started_ns = time.perf_counter_ns()
        try:
            return self._delegate.importance()
        finally:
            importance_kind = (
                "permutation" if self._model_kind == "tab_cnn" else "built_in"
            )
            self._recorder.record(
                "importance",
                "total",
                started_ns,
                model_kind=self._model_kind,
                importance_kind=importance_kind,
                seed=self._seed,
                fold=self._fold,
            )
            _restore_state(previous)


def _install_model_timing(recorder: _Recorder, measured_model_kind: str) -> None:
    from pipeline import model

    original_create = model.create

    def measured_create(cfg, seed):
        delegate = original_create(cfg, seed)
        fold = int(getattr(_STATE, "fold", -1))
        return _MeasuredAdapter(delegate, recorder, cfg.kind, int(seed), fold)

    model.create = measured_create

    if measured_model_kind != "tab_cnn":
        return

    from pipeline import tab_cnn

    original_predict = tab_cnn.TabCNNFold.predict

    def measured_predict(self, frame):
        phase = getattr(_STATE, "model_phase", None)
        fold = int(getattr(_STATE, "fold", -1))
        started_ns = time.perf_counter_ns()
        try:
            return original_predict(self, frame)
        finally:
            if phase is not None:
                operation = (
                    "preparation_reinference"
                    if phase == "model_fit"
                    else "reinference"
                )
                recorder.record(
                    "importance" if phase in {"model_fit", "importance"} else "model",
                    operation,
                    started_ns,
                    model_kind="tab_cnn",
                    phase=phase,
                    fold=fold,
                    rows=len(frame),
                )

    tab_cnn.TabCNNFold.predict = measured_predict

    original_score = tab_cnn.roc_auc_score

    def measured_score(y_true, y_score, *args, **kwargs):
        phase = getattr(_STATE, "model_phase", None)
        fold = int(getattr(_STATE, "fold", -1))
        started_ns = time.perf_counter_ns()
        try:
            return original_score(y_true, y_score, *args, **kwargs)
        finally:
            if phase is not None:
                recorder.record(
                    "importance" if phase == "importance" else "model",
                    "score_calculation",
                    started_ns,
                    model_kind="tab_cnn",
                    phase=phase,
                    fold=fold,
                    rows=len(y_true),
                )

    tab_cnn.roc_auc_score = measured_score


def _install_recovery_timing(recorder: _Recorder) -> None:
    from pipeline.recovery import FoldRecovery

    original_load = FoldRecovery.load
    original_save = FoldRecovery.save

    def measured_load(self, seed, fold, **kwargs):
        _STATE.fold = int(fold)
        started_ns = time.perf_counter_ns()
        checkpoint = None
        try:
            checkpoint = original_load(self, seed, fold, **kwargs)
            return checkpoint
        finally:
            recorder.record(
                "recovery",
                "read_and_validate",
                started_ns,
                seed=int(seed),
                fold=int(fold),
                reused=checkpoint is not None,
            )

    def measured_save(self, seed, fold, **kwargs):
        started_ns = time.perf_counter_ns()
        try:
            return original_save(self, seed, fold, **kwargs)
        finally:
            recorder.record(
                "recovery",
                "write_and_validate",
                started_ns,
                seed=int(seed),
                fold=int(fold),
            )

    FoldRecovery.load = measured_load
    FoldRecovery.save = measured_save


def _calibrate_recorder() -> dict[str, object]:
    samples: list[int] = []
    sink: list[dict[str, object]] = []
    iterations = 20_000
    for sequence in range(iterations):
        started_ns = time.perf_counter_ns()
        finished_ns = time.perf_counter_ns()
        sink.append(
            {
                "sequence": sequence,
                "duration_seconds": (finished_ns - started_ns) / 1e9,
                "category": "calibration",
                "operation": "record",
            }
        )
        samples.append(time.perf_counter_ns() - started_ns)
    ordered = sorted(samples)
    return {
        "iterations": iterations,
        "median_nanoseconds": int(statistics.median(ordered)),
        "p99_nanoseconds": int(ordered[int(len(ordered) * 0.99)]),
        "maximum_nanoseconds": int(max(ordered)),
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_evidence(
    output: Path,
    recorder: _Recorder,
    calibration: dict[str, object],
    pipeline_args: list[str],
    config_path: Path,
    status: str,
    exit_code: int,
) -> None:
    finished_ns = time.perf_counter_ns()
    p99_ns = int(calibration["p99_nanoseconds"])
    evidence = {
        "schema_version": 1,
        "purpose": "issue-320-fold-execution-timing",
        "label": recorder.label,
        "status": status,
        "exit_code": exit_code,
        "started_at": recorder.started_at,
        "finished_at": _utc_now(),
        "wall_seconds": (finished_ns - recorder.started_ns) / 1e9,
        "process_cpu_seconds": time.process_time() - recorder.process_started,
        "execution": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pipeline_arguments": pipeline_args,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        },
        "instrumentation": {
            "method": "runtime wrappers around selected provider, model, importance, and recovery methods",
            "clock": "time.perf_counter_ns",
            "calibration": calibration,
            "recorded_event_count": len(recorder.events),
            "estimated_p99_recording_overhead_seconds": (
                len(recorder.events) * p99_ns / 1e9
            ),
            "limitations": [
                "The estimate covers clock, dictionary, and list recording work, not method dispatch.",
                "GPU utilization is sampled by the external run controller.",
                "TabCNN permutation setup is derived as importance total minus reinference and scoring.",
            ],
        },
        "events": recorder.events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pipeline_args = list(args.pipeline_args)
    if pipeline_args and pipeline_args[0] == "--":
        pipeline_args.pop(0)
    if not pipeline_args:
        parser.error("-- 뒤에 pipeline.run 인수를 지정해야 한다.")
    if os.environ.get("PIPELINE_SEED_GPUS", "").strip():
        parser.error("세부 계측은 PIPELINE_SEED_GPUS 시드 병렬 실행을 지원하지 않는다.")

    config_path = Path(pipeline_args[0])
    if not config_path.is_file():
        parser.error(f"설정 파일이 없다: {config_path}")
    import yaml

    config_document = yaml.safe_load(config_path.read_text())
    measured_model_kind = str(config_document["model"]["kind"])

    recorder = _Recorder(args.label)
    calibration = _calibrate_recorder()
    _install_plan_timing(recorder)
    _install_model_timing(recorder, measured_model_kind)
    _install_recovery_timing(recorder)

    from pipeline import run

    original_argv = sys.argv
    sys.argv = ["pipeline.run", *pipeline_args]
    status = "success"
    exit_code = 0
    try:
        run.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
        status = "success" if exit_code == 0 else "failed"
        raise
    except BaseException:
        status = "failed"
        exit_code = 1
        raise
    finally:
        sys.argv = original_argv
        _write_evidence(
            args.evidence,
            recorder,
            calibration,
            pipeline_args,
            config_path,
            status,
            exit_code,
        )


if __name__ == "__main__":
    main()
