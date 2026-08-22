"""시드 반복 실행. 순차(기본)와 GPU별 프로세스 병렬을 같은 계약으로 제공한다. (#99)

Kaggle T4 x2처럼 GPU가 여럿일 때 PIPELINE_SEED_GPUS(예: "0,1")를 주면 시드
단위로 워커 프로세스를 띄워 GPU를 나눠 쓴다. 환경 변수가 없거나 시드가
하나면 기존 순차 경로 그대로다.

재현성: 각 adapter가 fold 학습 시작 때 자기 시드로 전역 RNG를 다시 심으므로
시드 간 실행 순서로 상태가 흐르지 않고, 같은 GPU 모델이면 병렬 결과가 순차
실행과 같다. 워커는 첫 CUDA 초기화 전에 CUDA_VISIBLE_DEVICES를 배정받아
자기 GPU 하나만 본다(adapter의 device="cuda"는 그 GPU로 해석된다).

진행 기록: 순차 경로는 시드별 feature_build/training 단계 전환과 fold 완료를
기존처럼 통지한다. 병렬 경로는 시드별 단계가 겹치므로 부모가 training 단계
하나로 묶고, 워커의 fold 완료 통지만 큐로 받아 부모 recorder에 전달한다.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pandas as pd

from . import cv
from .config import ExperimentConfig
from .plan import FeaturePlan
from .recovery import FoldRecovery

ENV_GPUS = "PIPELINE_SEED_GPUS"


def run_seeds(
    cfg: ExperimentConfig,
    plan: FeaturePlan,
    train: pd.DataFrame,
    test: pd.DataFrame,
    recorder: cv.RunRecorder | None = None,
    recovery: FoldRecovery | None = None,
) -> list[cv.CVResult]:
    """cfg.seeds 전체를 실행해 시드 순서대로 CVResult를 돌려준다."""
    gpus = [g.strip() for g in os.environ.get(ENV_GPUS, "").split(",") if g.strip()]
    if len(gpus) < 2 or len(cfg.seeds) < 2:
        return [
            cv.run_cv(cfg, plan, train, test, seed, recorder=recorder, recovery=recovery)
            for seed in cfg.seeds
        ]
    return _run_parallel(cfg, plan, train, test, recorder, recovery, gpus)


class _QueueRecorder:
    """워커의 진행 및 시간 사건을 부모 기록기 큐로 옮긴다."""

    def __init__(self, events) -> None:
        self._events = events

    def stage(self, name: str) -> None:
        pass  # 시드별 단계가 겹치므로 병렬 경로의 단계 기록은 부모가 소유한다.

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None:
        self._events.put(
            {
                "kind": "fold_completed",
                "seed_index": seed_index,
                "fold_index": fold_index,
                "auc": auc,
            }
        )

    def record_timing(self, event: dict[str, object]) -> None:
        self._events.put({"kind": "timing", "event": event})


def _pin_gpu(gpu_queue) -> None:
    """워커 시작 직후, torch의 CUDA 초기화 전에 GPU 하나를 배정받는다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_queue.get()


def _run_seed(cfg, plan, train, test, seed, events, recovery) -> cv.CVResult:
    return cv.run_cv(
        cfg,
        plan,
        train,
        test,
        seed,
        recorder=_QueueRecorder(events),
        recovery=recovery,
    )


def _forward_events(events, recorder: cv.RunRecorder, done: threading.Event) -> None:
    while not done.is_set() or not events.empty():
        try:
            item = events.get(timeout=0.2)
        except queue.Empty:
            continue
        if item["kind"] == "fold_completed":
            recorder.fold_completed(item["seed_index"], item["fold_index"], item["auc"])
        elif item["kind"] == "timing":
            sink = getattr(recorder, "record_timing", None)
            if sink is not None:
                sink(item["event"])
        else:
            raise ValueError(f"알 수 없는 시드 워커 사건이다: {item['kind']}")


def _run_parallel(
    cfg, plan, train, test, recorder, recovery: FoldRecovery | None, gpus: list[str]
) -> list[cv.CVResult]:
    if recorder is not None:
        recorder.stage("training")
    # fork는 부모의 CUDA·MLflow 상태를 물려받으므로 spawn으로 깨끗하게 시작한다.
    ctx = get_context("spawn")
    with ctx.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)
        events = manager.Queue()
        done = threading.Event()
        forwarder = None
        if recorder is not None:
            forwarder = threading.Thread(
                target=_forward_events, args=(events, recorder, done), daemon=True
            )
            forwarder.start()
        try:
            with ProcessPoolExecutor(
                max_workers=min(len(gpus), len(cfg.seeds)),
                mp_context=ctx,
                initializer=_pin_gpu,
                initargs=(gpu_queue,),
            ) as pool:
                futures = [
                    pool.submit(_run_seed, cfg, plan, train, test, seed, events, recovery)
                    for seed in cfg.seeds
                ]
                return [f.result() for f in futures]
        finally:
            done.set()
            if forwarder is not None:
                forwarder.join()
