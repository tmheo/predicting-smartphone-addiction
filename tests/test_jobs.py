"""pipeline.jobs: 분할 작업 오케스트레이션의 계약 시험. (#541)

빠른 더미 명령(python -c)으로 완료 건너뜀, 실패 집계, workers 상한 준수,
lock 파일 재진입 거부, 로그 파일 생성·닫힘, 스레드 env 설정을 확인한다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pipeline.jobs import Job, JobsFailed, JobsRefused, run_jobs

POLL = 0.05


def touch_command(output: Path, *extra_lines: str) -> list[str]:
    code = "\n".join([*extra_lines, f"import pathlib; pathlib.Path({str(output)!r}).write_text('done')"])
    return [sys.executable, "-c", code]


def test_job_refuses_bad_tag_and_empty_command(tmp_path):
    with pytest.raises(JobsRefused):
        Job("a/b", [sys.executable], tmp_path / "out")
    with pytest.raises(JobsRefused):
        Job("", [sys.executable], tmp_path / "out")
    with pytest.raises(JobsRefused):
        Job("ok", [], tmp_path / "out")


def test_run_jobs_refuses_bad_arguments(tmp_path):
    job = Job("a", touch_command(tmp_path / "a.json"), tmp_path / "a.json")
    with pytest.raises(JobsRefused):
        run_jobs([job], workers=0, threads=1, log_dir=tmp_path / "logs")
    with pytest.raises(JobsRefused):
        run_jobs([job], workers=1, threads=0, log_dir=tmp_path / "logs")
    with pytest.raises(JobsRefused):
        run_jobs([job], workers=1, threads=1, log_dir=tmp_path / "logs", poll_seconds=0)
    duplicate_tag = Job("a", touch_command(tmp_path / "b.json"), tmp_path / "b.json")
    with pytest.raises(JobsRefused):
        run_jobs([job, duplicate_tag], workers=1, threads=1, log_dir=tmp_path / "logs")
    duplicate_output = Job("b", touch_command(tmp_path / "a.json"), tmp_path / "a.json")
    with pytest.raises(JobsRefused):
        run_jobs([job, duplicate_output], workers=1, threads=1, log_dir=tmp_path / "logs")


def test_done_jobs_are_skipped(tmp_path):
    output = tmp_path / "a.json"
    output.write_text("already")
    marker = tmp_path / "ran.marker"
    job = Job("a", touch_command(marker), output)
    run_jobs([job], workers=1, threads=1, log_dir=tmp_path / "logs", poll_seconds=POLL)
    assert not marker.exists(), "완료된 작업의 command가 실행됐다"
    assert output.read_text() == "already"
    assert not (tmp_path / "logs" / "a.log").exists()


def test_failures_are_aggregated_into_jobs_failed(tmp_path):
    ok = Job("ok", touch_command(tmp_path / "ok.json"), tmp_path / "ok.json")
    crash = Job("crash", [sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "crash.json")
    silent = Job("silent", [sys.executable, "-c", "pass"], tmp_path / "silent.json")
    unrunnable = Job("unrunnable", ["/nonexistent-binary-541"], tmp_path / "unrunnable.json")
    with pytest.raises(JobsFailed) as excinfo:
        run_jobs(
            [ok, crash, silent, unrunnable],
            workers=4,
            threads=1,
            log_dir=tmp_path / "logs",
            poll_seconds=POLL,
        )
    assert excinfo.value.tags == ("crash", "silent", "unrunnable")
    assert ok.output.exists(), "실패와 무관한 작업은 끝까지 돈다"


def test_workers_cap_is_respected(tmp_path):
    flags = tmp_path / "flags"
    flags.mkdir()
    jobs = []
    for index in range(5):
        output = tmp_path / f"out-{index}.json"
        code = (
            "import pathlib, time\n"
            f"flags = pathlib.Path({str(flags)!r})\n"
            f"flag = flags / 'flag-{index}'\n"
            "flag.write_text('')\n"
            "seen = len(list(flags.glob('flag-*')))\n"
            "time.sleep(0.3)\n"
            "flag.unlink()\n"
            f"pathlib.Path({str(output)!r}).write_text(str(seen))\n"
        )
        jobs.append(Job(f"job-{index}", [sys.executable, "-c", code], output))
    run_jobs(jobs, workers=2, threads=1, log_dir=tmp_path / "logs", poll_seconds=POLL)
    concurrency = [int(job.output.read_text()) for job in jobs]
    assert max(concurrency) <= 2, f"동시 상한 위반: {concurrency}"


def test_thread_env_is_set_for_workers(tmp_path):
    output = tmp_path / "env.json"
    code = (
        "import os, pathlib\n"
        "keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')\n"
        f"pathlib.Path({str(output)!r}).write_text(','.join(os.environ[k] for k in keys))\n"
    )
    job = Job("env", [sys.executable, "-c", code], output)
    run_jobs([job], workers=1, threads=7, log_dir=tmp_path / "logs", poll_seconds=POLL)
    assert output.read_text() == "7,7,7,7"


def test_live_lock_defers_to_foreign_worker(tmp_path):
    """살아 있는 lock이 잡은 작업은 띄우지 않고 output 출현을 기다린다."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output = tmp_path / "a.json"
    marker = tmp_path / "ran.marker"
    (log_dir / "a.lock").write_text(f"{os.getpid()}\n")

    def foreign_worker():
        time.sleep(0.3)
        output.write_text("foreign")

    thread = threading.Thread(target=foreign_worker)
    thread.start()
    job = Job("a", touch_command(marker), output)
    run_jobs([job], workers=1, threads=1, log_dir=log_dir, poll_seconds=POLL)
    thread.join()
    assert not marker.exists(), "외부 작업자가 잡은 작업을 겹쳐 띄웠다"
    assert output.read_text() == "foreign"


def test_stale_lock_is_reclaimed(tmp_path):
    """소유자가 죽은 lock은 잔재로 보고 직접 실행한다."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (log_dir / "a.lock").write_text(f"{dead.pid}\n")
    output = tmp_path / "a.json"
    job = Job("a", touch_command(output), output)
    run_jobs([job], workers=1, threads=1, log_dir=log_dir, poll_seconds=POLL)
    assert output.exists()
    assert not (log_dir / "a.lock").exists(), "완료한 작업의 lock이 남았다"


def test_log_files_are_written_and_closed(tmp_path):
    output = tmp_path / "a.json"
    code = (
        "import pathlib, sys\n"
        "print('to stdout')\n"
        "print('to stderr', file=sys.stderr)\n"
        f"pathlib.Path({str(output)!r}).write_text('done')\n"
    )
    job = Job("a", [sys.executable, "-c", code], output)
    run_jobs([job], workers=1, threads=1, log_dir=tmp_path / "logs", poll_seconds=POLL)
    log = (tmp_path / "logs" / "a.log").read_text()
    assert "to stdout" in log and "to stderr" in log
    assert not (tmp_path / "logs" / "a.lock").exists()
