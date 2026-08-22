"""시드 병렬 작업이 CPU 할당량을 안전하게 나누는 규칙 테스트."""

from __future__ import annotations

import pytest

from pipeline import cpu_budget


def test_threads_per_worker_uses_cgroup_quota_before_visible_cpu_count(monkeypatch):
    monkeypatch.setattr(cpu_budget.os, "cpu_count", lambda: 192)
    monkeypatch.setattr(cpu_budget, "_affinity_cpu_count", lambda: 192)
    monkeypatch.setattr(cpu_budget, "cgroup_cpu_quota", lambda: 92.16)

    assert cpu_budget.threads_per_worker(3) == 30


def test_threads_per_worker_falls_back_to_affinity(monkeypatch):
    monkeypatch.setattr(cpu_budget.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(cpu_budget, "_affinity_cpu_count", lambda: 48)
    monkeypatch.setattr(cpu_budget, "cgroup_cpu_quota", lambda: None)

    assert cpu_budget.threads_per_worker(3) == 16


@pytest.mark.parametrize("worker_count", [0, -1])
def test_threads_per_worker_rejects_non_positive_worker_count(worker_count):
    with pytest.raises(ValueError, match="양수"):
        cpu_budget.threads_per_worker(worker_count)


def test_xgb_n_jobs_environment_is_optional_and_validated(monkeypatch):
    monkeypatch.delenv(cpu_budget.XGB_N_JOBS_ENV, raising=False)
    assert cpu_budget.xgb_n_jobs_from_environment() is None

    monkeypatch.setenv(cpu_budget.XGB_N_JOBS_ENV, "30")
    assert cpu_budget.xgb_n_jobs_from_environment() == 30

    monkeypatch.setenv(cpu_budget.XGB_N_JOBS_ENV, "0")
    with pytest.raises(ValueError, match="양의 정수"):
        cpu_budget.xgb_n_jobs_from_environment()
