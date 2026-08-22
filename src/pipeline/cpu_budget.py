"""한 머신의 시드 병렬 워커에 CPU 할당량을 나누는 작은 실행 규약."""

from __future__ import annotations

import math
import os
from pathlib import Path


XGB_N_JOBS_ENV = "PIPELINE_XGB_N_JOBS"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_cpu_quota() -> float | None:
    """cgroup v2 또는 v1이 정한 CPU 개수를 돌려준다. 제한이 없으면 None이다."""
    raw = _read_text(Path("/sys/fs/cgroup/cpu.max"))
    if raw:
        quota, period = raw.split()
        if quota != "max":
            return int(quota) / int(period)

    for root in (Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct")):
        quota = _read_text(root / "cpu.cfs_quota_us")
        period = _read_text(root / "cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            return int(quota) / int(period)
    return None


def _affinity_cpu_count() -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return len(os.sched_getaffinity(0))


def effective_cpu_count() -> int:
    """보이는 CPU, 실행 허용 집합과 cgroup 할당량 중 가장 작은 정수값이다."""
    candidates = [
        value
        for value in (os.cpu_count(), _affinity_cpu_count(), cgroup_cpu_quota())
        if value
    ]
    if not candidates:
        return 1
    return max(1, math.floor(min(candidates)))


def threads_per_worker(worker_count: int) -> int:
    """동시 워커가 CPU 할당량을 넘지 않도록 같은 몫으로 나눈다."""
    if worker_count <= 0:
        raise ValueError(f"worker_count는 양수여야 한다: {worker_count}")
    return max(1, effective_cpu_count() // worker_count)


def xgb_n_jobs_from_environment() -> int | None:
    """시드 병렬 초기화가 전달한 XGBoost 전용 작업 흐름 수를 읽는다."""
    raw = os.environ.get(XGB_N_JOBS_ENV)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{XGB_N_JOBS_ENV}는 양의 정수여야 한다: {raw!r}") from error
    if value <= 0:
        raise ValueError(f"{XGB_N_JOBS_ENV}는 양의 정수여야 한다: {raw!r}")
    return value
