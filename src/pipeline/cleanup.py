"""스테일 실행 판정과 정리. (#42)

사용법:
    uv run python -m pipeline.cleanup            # 판정 후 정리까지 수행
    uv run python -m pipeline.cleanup --dry-run  # 정리 없이 보고만

새 실험 실행이 시작될 때도 실행 생성 전에 같은 로직(cleanup_stale)이 자동 수행된다.

판정 규약 (#42):
- 대상은 이 프로젝트 실험의 RUNNING 실행.
- 마지막 활동이 10분을 초과했고, 같은 호스트에서 해당 PID의 프로세스가 없을 때만 스테일.
- 프로세스가 살아 있으면 시간이 얼마나 지났든 정리하지 않는다.
- 호스트가 다르거나 프로세스 태그가 없으면 자동 정리에서 제외하고 목록으로만 보고한다.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from . import tracking

STALE_AFTER_SECONDS = 600  # 마지막 활동 10분 초과. (#42)
RUN_LOGS_ROOT = Path("run-logs")


def _last_activity_ms(run) -> int:
    """마지막 활동 시각(밀리초). progress.last_activity_at 태그(#40)가 기준이고,
    첫 생존 신호 전에 죽은 실행은 시작 시각을 마지막 활동으로 본다."""
    tag = run.data.tags.get("progress.last_activity_at")
    if tag:
        return int(datetime.fromisoformat(tag).timestamp() * 1000)
    return run.info.start_time


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 신호를 못 보내도 프로세스는 존재한다.
    return True


def find_stale(client, experiment_id: str) -> tuple[list, list]:
    """(스테일 실행 목록, 수동 점검 대상 목록)을 돌려준다."""
    running = client.search_runs([experiment_id], filter_string="attributes.status = 'RUNNING'")
    hostname = socket.gethostname()
    now_ms = time.time() * 1000
    stale, manual = [], []
    for run in running:
        pid_tag = run.data.tags.get("process.pid")
        host_tag = run.data.tags.get("process.hostname")
        if not pid_tag or not host_tag or host_tag != hostname:
            manual.append(run)
            continue
        if _pid_alive(int(pid_tag)):
            continue
        if now_ms - _last_activity_ms(run) <= STALE_AFTER_SECONDS * 1000:
            continue  # 프로세스는 없지만 아직 10분이 지나지 않았다. 다음 기회에 정리한다.
        stale.append(run)
    return stale, manual


def cleanup_stale(client, experiment_id: str, dry_run: bool = False) -> None:
    """스테일 실행을 KILLED로 정리하고 남은 로컬 실행 로그를 보존한다. (#42)"""
    stale, manual = find_stale(client, experiment_id)
    for run in manual:
        print(
            f"수동 점검 대상: run_id={run.info.run_id} "
            f"host={run.data.tags.get('process.hostname', '(없음)')} "
            f"pid={run.data.tags.get('process.pid', '(없음)')} - 자동 정리에서 제외"
        )
    for run in stale:
        run_id = run.info.run_id
        if dry_run:
            print(f"스테일 실행(정리 예정): run_id={run_id}")
            continue
        # 종료 시각을 마지막 활동 시각으로 지정해 Duration이 실제 생존 시간과 가깝게 한다.
        # 태그가 어긋나도 종료 시각이 시작 시각보다 앞서지는 않게 한다.
        end_time = max(_last_activity_ms(run), run.info.start_time)
        client.set_terminated(run_id, status="KILLED", end_time=end_time)
        # 정리로 만들어진 KILLED와 실행 코드가 정상 처리한 KILLED를 구분한다.
        client.set_tag(run_id, "cleanup.reason", "stale_process_missing")
        client.set_tag(
            run_id,
            "cleanup.performed_at",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        leftover = RUN_LOGS_ROOT / run_id / "run.log"
        if leftover.exists():
            client.log_artifact(run_id, str(leftover), artifact_path="logs")
            # 업로드가 성공한 경우에만 로컬 파일과 빈 디렉터리를 삭제한다. (#39, #42)
            leftover.unlink()
            leftover.parent.rmdir()
        print(f"스테일 실행 정리 완료: run_id={run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="스테일 실행 점검과 정리")
    parser.add_argument("--dry-run", action="store_true", help="정리 없이 판정 결과만 출력")
    args = parser.parse_args()
    client, experiment_id = tracking.mlflow_client()
    cleanup_stale(client, experiment_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
