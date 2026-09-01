"""분할 작업 subprocess 오케스트레이션의 정본 구현.

판정 스크립트들이 복사해 쓰던 run_jobs 루프(작업 실행, 동시 상한, 재진입 탐지,
로그 배치, 실패 집계)를 한 곳으로 모은다. 작업의 완료 여부는 output 파일의
존재로 판정하므로, 같은 작업 목록으로 다시 부르면 끝난 작업은 건너뛴다.

재진입 탐지는 log_dir의 lock 파일(작업자 pid + 생존 확인)로 한다.
살아 있는 소유자가 잡은 작업은 띄우지 않고 완료(output 출현)를 기다리며,
소유자가 죽은 채 output이 없으면 잔재 lock을 걷고 직접 실행한다.
lock은 실수로 겹쳐 뜬 드라이버를 막는 협조적 장치이지 적대적 잠금이 아니다.

동시 workers 상한은 같은 기계의 외부 작업자(살아 있는 lock)까지 세어 지킨다.
shrunk 계열은 동시 3개까지가 안전하고 5개에서 커널 패닉 전례가 있어,
상한을 지키는 책임이 이 module에 있다.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class JobsError(RuntimeError):
    """분할 작업 오케스트레이션 계약 위반."""


class JobsRefused(JobsError):
    """작업 목록이나 실행 인자가 계약을 위반해 실행을 거부했다."""


class JobsFailed(JobsError):
    """하나 이상의 작업이 실패했다(0이 아닌 종료 코드 또는 output 부재)."""

    def __init__(self, tags: Iterable[str]):
        self.tags = tuple(sorted(tags))
        super().__init__(f"실패한 작업: {', '.join(self.tags)}")


@dataclass(frozen=True)
class Job:
    """분할 작업 하나. 완료 여부는 output 파일의 존재로 판정한다."""

    tag: str
    command: tuple[str, ...]
    output: Path

    def __init__(self, tag: str, command: Iterable[str], output: Path | str) -> None:
        if not tag or "/" in tag or os.sep in tag or tag in (".", ".."):
            raise JobsRefused(f"tag는 파일 이름으로 쓸 수 있어야 한다: {tag!r}")
        command = tuple(command)
        if not command:
            raise JobsRefused(f"작업 {tag}의 command가 비어 있다.")
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "output", Path(output))

    @property
    def done(self) -> bool:
        return self.output.exists()


@dataclass
class _Running:
    job: Job
    process: subprocess.Popen
    handle: IO[str]
    lock: Path
    started: float = field(default_factory=time.monotonic)


def _write_lock(lock: Path, pid: int) -> None:
    """lock 내용을 원자적으로 교체한다(빈 파일을 읽는 창을 없앤다)."""
    staging = lock.with_suffix(".lock.staging")
    staging.write_text(f"{pid}\n", encoding="utf-8")
    os.replace(staging, lock)


def _lock_holder_alive(lock: Path) -> bool:
    try:
        pid = int(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _reap(active: dict[str, _Running], failed: list[str]) -> None:
    for tag, running in list(active.items()):
        code = running.process.poll()
        if code is None:
            continue
        running.handle.close()
        running.lock.unlink(missing_ok=True)
        del active[tag]
        elapsed = time.monotonic() - running.started
        if code != 0:
            failed.append(tag)
            print(f"[jobs] 실패({code}) {tag} {elapsed:.0f}s {_now()}", flush=True)
        elif not running.job.done:
            failed.append(tag)
            print(f"[jobs] 실패(output 없음) {tag} {elapsed:.0f}s {_now()}", flush=True)
        else:
            print(f"[jobs] 완료 {tag} {elapsed:.0f}s {_now()}", flush=True)


def _launch(
    pending: list[Job],
    active: dict[str, _Running],
    failed: list[str],
    *,
    workers: int,
    env: dict[str, str],
    log_dir: Path,
) -> list[Job]:
    """띄울 수 있는 만큼 띄우고 아직 대기해야 하는 작업을 돌려준다."""
    waiting: list[Job] = []
    launchable: list[Job] = []
    foreign = 0
    for job in pending:
        if job.done:
            print(f"[jobs] 완료(외부 실행) {job.tag} {_now()}", flush=True)
            continue
        lock = log_dir / f"{job.tag}.lock"
        if lock.exists() and _lock_holder_alive(lock):
            foreign += 1
            waiting.append(job)
        else:
            launchable.append(job)
    for job in launchable:
        if len(active) + foreign >= workers:
            waiting.append(job)
            continue
        lock = log_dir / f"{job.tag}.lock"
        lock.unlink(missing_ok=True)  # 죽은 소유자의 잔재
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:  # 방금 다른 드라이버가 잡았다
            foreign += 1
            waiting.append(job)
            continue
        os.write(descriptor, f"{os.getpid()}\n".encode("utf-8"))
        os.close(descriptor)
        handle = (log_dir / f"{job.tag}.log").open("w")
        try:
            process = subprocess.Popen(
                job.command, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            handle.close()
            lock.unlink(missing_ok=True)
            failed.append(job.tag)
            print(f"[jobs] 실패(실행 불가: {exc}) {job.tag} {_now()}", flush=True)
            continue
        _write_lock(lock, process.pid)
        active[job.tag] = _Running(job, process, handle, lock)
        print(f"[jobs] 시작 {job.tag} {_now()}", flush=True)
    return waiting


def run_jobs(
    jobs: Iterable[Job],
    *,
    workers: int,
    threads: int,
    log_dir: Path,
    poll_seconds: float = 10.0,
) -> None:
    """남은 작업을 동시 상한 안에서 병렬 실행한다.

    돌아오면 모든 작업의 output이 존재한다. 그렇게 만들 수 없으면(0이 아닌
    종료 코드, 종료했는데 output 부재, 실행 불가) 나머지 작업을 모두 끝까지
    돌린 뒤 실패 tag를 모아 JobsFailed로 던진다. CLI가 종료 메시지로 번역한다.

    workers는 이 기계에서 동시에 도는 작업 수의 상한이고, 다른 드라이버가
    남긴 살아 있는 작업자(lock 파일)도 세어 지킨다. threads는 작업별
    BLAS/OMP 스레드 env 4종에 설정한다.
    """
    jobs = list(jobs)
    if workers < 1:
        raise JobsRefused(f"workers는 1 이상이어야 한다: {workers}")
    if threads < 1:
        raise JobsRefused(f"threads는 1 이상이어야 한다: {threads}")
    if poll_seconds <= 0:
        raise JobsRefused(f"poll_seconds는 양수여야 한다: {poll_seconds}")
    tags = [job.tag for job in jobs]
    if len(set(tags)) != len(tags):
        duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
        raise JobsRefused(f"tag가 중복됐다: {', '.join(duplicates)}")
    outputs = [job.output.resolve() for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise JobsRefused("서로 다른 작업이 같은 output을 가리킨다.")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in _THREAD_ENV_KEYS:
        env[key] = str(threads)

    pending = [job for job in jobs if not job.done]
    active: dict[str, _Running] = {}
    failed: list[str] = []
    print(
        f"[jobs] 남은 작업 {len(pending)}/{len(jobs)}개, 동시 상한 {workers}, 스레드 {threads}",
        flush=True,
    )
    while pending or active:
        _reap(active, failed)
        pending = _launch(
            pending, active, failed, workers=workers, env=env, log_dir=log_dir
        )
        if pending or active:
            time.sleep(poll_seconds)
    _reap(active, failed)
    if failed:
        raise JobsFailed(failed)
    print("[jobs] all jobs finished", flush=True)
