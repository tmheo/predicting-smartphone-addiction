"""실행 관찰 기록기. 실험 실행 하나의 MLflow 실행 수명주기를 소유한다. (#43)

선행 규약의 집행 지점:
- 생성 절차: 스테일 정리 -> MLflow 실행 생성 -> 실행 로그 캡처 -> 생존 신호 -> 시작 기록. (#39, #42)
- 진행 기록: progress.*/time.* 지표·태그의 이름과 step 규약은 전부 이 모듈 내부 소유다. (#40)
- 종료 처리: traceback -> 로그 닫기 -> 로그 보존 -> 오류 태그 -> 상태 종료의 5단계 순서. (#42)

파이프라인의 다른 모듈은 이 기록기에 일어난 일을 통지할 뿐 MLflow 지표 이름을 알지 못한다.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import cleanup, summary, tracking
from .config import ExperimentConfig
from .cv import CVResult

# 저장소 루트 기준. tracking URI와 같은 이유로 루트 실행이 전제다. (#39)
RUN_LOGS_ROOT = Path("run-logs")
HEARTBEAT_SECONDS = 60.0
# 단계 어휘는 여섯 개로 고정. (#40)
STAGES = ("setup", "data_load", "feature_build", "training", "evaluation", "artifacts")


class TerminationRequested(Exception):
    """SIGTERM을 예외로 변환해 일반 정리 경로로 보낸다. KILLED로 종료된다. (#42)"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _LogCapture:
    """표준 출력·오류를 OS 수준 파일 기술자 복제(dup2 + tee 파이프)로 가로챈다. (#39, #43)

    하위 프로세스와 네이티브 라이브러리(LightGBM 등)의 출력까지 수신된 순서대로
    한 로그 파일에 담고, 같은 내용을 원래 터미널에도 그대로 흘린다.
    로그 쓰기는 무버퍼(os.write)라서 실행 중에도 tail -f로 바로 읽힌다.
    """

    def __init__(self, log_path: Path) -> None:
        # O_EXCL: 예상과 달리 같은 경로가 이미 있으면 새 실행을 실패시킨다. (#39)
        self._log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._orig_stdout = os.dup(1)
        self._orig_stderr = os.dup(2)
        self._pump_thread: threading.Thread | None = None

    def start(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        # fd가 파이프로 바뀌면 print가 블록 버퍼링되므로 줄 단위 버퍼링으로 되돌린다.
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
        self._pump_thread = threading.Thread(
            target=self._pump, args=(read_fd,), name="run-log-pump", daemon=True
        )
        self._pump_thread.start()

    def _pump(self, read_fd: int) -> None:
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            try:
                os.write(self._orig_stdout, chunk)
            except OSError:
                pass  # 터미널이 닫혀도 로그 기록은 계속한다.
            os.write(self._log_fd, chunk)
        os.close(read_fd)

    def stop(self) -> None:
        """fd를 되돌리고 남은 출력을 전부 로그로 밀어 넣은 뒤 동기화하고 닫는다. (#39)"""
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._orig_stdout, 1)
        os.dup2(self._orig_stderr, 2)
        # 파이프 쓰기 끝의 마지막 사본이 fd 1·2에서 사라지면 펌프 스레드가 EOF로 끝난다.
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=10)
        os.fsync(self._log_fd)
        os.close(self._log_fd)
        os.close(self._orig_stdout)
        os.close(self._orig_stderr)


class RunObserver:
    """MLflow 실행 수명주기의 유일한 소유자. run.py가 만들고 cv.run_cv에 주입된다. (#43)"""

    def __init__(self, cfg: ExperimentConfig, client, run_id: str) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self._client = client
        self._t0 = time.monotonic()
        self._mlflow_lock = threading.Lock()  # 생존 신호 스레드와 메인 스레드의 기록 직렬화
        self._beat_step = 0
        self._beat_lock = threading.Lock()
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._capture: _LogCapture | None = None
        self._run_dir = RUN_LOGS_ROOT / run_id
        self._log_path = self._run_dir / "run.log"
        self._stage: str | None = None
        self._stage_started = 0.0
        self._stage_end_counts: dict[str, int] = {}
        self._durations: list[tuple[str, int, float]] = []  # (stage, step, seconds)
        self._completed_units = 0

    # ------------------------------------------------------------------ 생성

    @classmethod
    def begin(cls, cfg: ExperimentConfig) -> "RunObserver":
        client, experiment_id = tracking.mlflow_client()
        # 실행 생성 전에 스테일 실행을 자동 정리한다. (#42)
        cleanup.cleanup_stale(client, experiment_id)
        run = client.create_run(experiment_id, run_name=cfg.name)
        obs = cls(cfg, client, run.info.run_id)
        try:
            obs._open_log()
            obs._start_heartbeat()
            # 백그라운드 셸에서 시작되면 SIGINT가 무시 상태로 상속될 수 있으므로,
            # 사용자 중단(#42)이 항상 KeyboardInterrupt로 전달되게 명시적으로 설치한다.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, obs._on_sigterm)
            obs._log_start_records()
        except BaseException:
            # 준비 중 실패한 실행을 RUNNING으로 남기지 않는다.
            client.set_terminated(obs.run_id, status="FAILED")
            raise
        return obs

    def _open_log(self) -> None:
        # 실행 디렉터리와 로그 파일은 run_id를 받은 직후, 다른 실행 작업보다 먼저 만든다. (#39)
        RUN_LOGS_ROOT.mkdir(mode=0o700, exist_ok=True)
        self._run_dir.mkdir(mode=0o700)  # 이미 있으면 실패시킨다. 재사용 금지. (#39)
        self._capture = _LogCapture(self._log_path)
        self._capture.start()
        # 파일 기록 연결 직후의 첫 기록. 터미널에도 그대로 보인다. (#39)
        print(f"run_id={self.run_id}")
        print(f"run_log={self._log_path.resolve()}")

    def _start_heartbeat(self) -> None:
        self._beat()  # 시작 즉시 첫 신호를 남겨 마지막 활동 시각이 항상 존재하게 한다.
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="run-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _on_sigterm(self, signum, frame) -> None:
        raise TerminationRequested("SIGTERM")

    def _log_start_records(self) -> None:
        with self._mlflow_lock:
            self._client.set_tag(self.run_id, "process.pid", str(os.getpid()))
            self._client.set_tag(self.run_id, "process.hostname", socket.gethostname())
            tracking.log_start_records(self._client, self.run_id, self.cfg)

    # ------------------------------------------------------------- 생존 신호

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(HEARTBEAT_SECONDS):
            try:
                self._beat()
            except Exception as exc:  # 생존 신호 실패가 실험을 죽여서는 안 된다.
                print(f"경고: 생존 신호 기록 실패: {exc}", file=sys.stderr)

    def _beat(self) -> None:
        """60초 주기와 단계 경계에서 경과 시간과 마지막 활동 시각을 남긴다. (#40)"""
        with self._beat_lock:
            step = self._beat_step
            self._beat_step += 1
        elapsed = time.monotonic() - self._t0
        with self._mlflow_lock:
            self._client.log_metric(
                self.run_id, "progress.elapsed_seconds", elapsed, step=step
            )
            self._client.set_tag(self.run_id, "progress.last_activity_at", _utc_now_iso())

    # ------------------------------------------------------------- 진행 기록

    def stage(self, name: str) -> None:
        """단계 전환. 직전 단계의 소요 시간을 기록하고 현재 단계 태그를 갱신한다. (#40)"""
        assert name in STAGES, f"알 수 없는 단계: {name}"
        self._close_stage()
        self._stage = name
        self._stage_started = time.monotonic()
        with self._mlflow_lock:
            self._client.set_tag(self.run_id, "progress.stage", name)
        self._beat()  # 메인 스레드도 단계 경계에서 생존 신호를 한 번씩 남긴다. (#40)

    def _close_stage(self) -> None:
        if self._stage is None:
            return
        seconds = time.monotonic() - self._stage_started
        # 반복 단계의 step은 종료 횟수와 같다: 시드마다 feature_build/training이
        # 한 번씩 끝나므로 step이 곧 시드 순번이 된다. (#40)
        step = self._stage_end_counts.get(self._stage, 0)
        self._stage_end_counts[self._stage] = step + 1
        with self._mlflow_lock:
            self._client.log_metric(
                self.run_id, f"time.{self._stage}_seconds", seconds, step=step
            )
        self._durations.append((self._stage, step, seconds))
        self._stage = None

    def record_input_hashes(self, hashes: dict[str, str]) -> None:
        """setup 단계 안, 입력 해시 계산 완료 직후 호출된다. (#43 기록 시점 분배)"""
        with self._mlflow_lock:
            tracking.log_input_hashes(self._client, self.run_id, hashes)

    def data_loaded(self, seed_total: int, fold_total: int) -> None:
        """fold 수가 확정되는 데이터 적재 완료 직후 총량 지표를 한 번 기록한다. (#40)"""
        with self._mlflow_lock:
            self._client.log_metric(self.run_id, "progress.total_units", seed_total * fold_total)
            self._client.log_metric(self.run_id, "progress.seed_total", seed_total)
            self._client.log_metric(self.run_id, "progress.fold_total", fold_total)

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None:
        """fold 하나가 끝날 때마다 진행률과 실행 중 fold AUC를 쌓는다. (#40)"""
        step = self._completed_units
        self._completed_units += 1
        with self._mlflow_lock:
            self._client.log_metric(self.run_id, "progress.fold_auc", auc, step=step)
            self._client.log_metric(
                self.run_id, "progress.completed_units", self._completed_units, step=step
            )
            self._client.log_metric(self.run_id, "progress.seed_index", seed_index, step=step)
            self._client.log_metric(self.run_id, "progress.fold_index", fold_index, step=step)

    # ------------------------------------------------------------- 최종 기록

    def stage_durations(self) -> list[tuple[str, int, float]]:
        """지금까지 끝난 단계의 (stage, step, seconds). 요약 생성기의 입력이다. (#41, #43)"""
        return list(self._durations)

    def log_final(self, result: CVResult) -> None:
        """최종 지표·원본 산출물·결과 요약을 활성 실행 안에 기록한다. artifacts 단계 소관."""
        with self._mlflow_lock:
            tracking.log_final_records(self._client, self.run_id, self.cfg, result)
            summary.generate_and_log(
                self._client, self.run_id, self.cfg, result, self.stage_durations()
            )

    # ------------------------------------------------------------- 종료 처리

    def succeed(self) -> None:
        self._finalize("FINISHED", None)

    def fail(self, exc: BaseException) -> None:
        status = (
            "KILLED"
            if isinstance(exc, (KeyboardInterrupt, TerminationRequested))
            else "FAILED"
        )
        self._finalize(status, exc)

    def _finalize(self, status: str, exc: BaseException | None) -> None:
        """#42의 종료 처리 순서. 중간 단계가 실패해도 끝까지 시도한다."""
        # 정리 중 두 번째 신호는 막지 않는다: 기본 동작으로 되돌려 즉시 종료를 허용한다. (#42)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        error_stage = self._stage
        if exc is not None:
            # 1) 전체 traceback을 실행 로그에 기록한다(캡처 중이므로 로그와 터미널 모두).
            traceback.print_exception(exc, file=sys.stderr)
        else:
            self._close_stage()  # 성공 시 마지막 단계(artifacts)의 소요 시간을 기록한다.

        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=10)

        # 2) 실행 로그를 동기화하고 닫는다. 이후의 정리 출력은 기록 범위 밖이다. (#39)
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception as stop_exc:
                print(f"경고: 실행 로그 닫기 실패: {stop_exc}", file=sys.stderr)

        # 3) 닫힌 로그를 MLflow 산출물 logs/run.log로 보존한다. (#39)
        uploaded = False
        try:
            self._client.log_artifact(self.run_id, str(self._log_path), artifact_path="logs")
            uploaded = True
        except Exception as upload_exc:
            print(f"경고: 실행 로그 보존 실패, 로컬 파일을 남긴다: {upload_exc}", file=sys.stderr)

        # 4) 오류 태그. (#42)
        if exc is not None:
            try:
                message = (str(exc).splitlines() or [""])[0][:500]
                self._client.set_tag(self.run_id, "error.stage", error_stage or "")
                self._client.set_tag(self.run_id, "error.type", type(exc).__name__)
                self._client.set_tag(self.run_id, "error.message", message)
            except Exception as tag_exc:
                print(f"경고: 오류 태그 기록 실패: {tag_exc}", file=sys.stderr)

        # 5) 상태 종료. 종료 시각은 기본값인 현재 시각. (#42)
        try:
            self._client.set_terminated(self.run_id, status=status)
        except Exception as term_exc:
            print(f"경고: 실행 상태 종료 실패: {term_exc}", file=sys.stderr)

        # 산출물 보존이 성공한 뒤에만 로컬 로그와 빈 디렉터리를 정리한다. (#39)
        if uploaded:
            try:
                self._log_path.unlink()
                self._run_dir.rmdir()
            except OSError as rm_exc:
                print(f"경고: 로컬 실행 로그 정리 실패: {rm_exc}", file=sys.stderr)
