"""Kaggle 제출. MLflow run의 submission artifact를 제출하고 public 점수를 run에 기록한다. (#20, #21)

사용법:
    uv run python -m pipeline.submit <run_id>
    uv run python -m pipeline.submit <run_id> --force  # 이미 제출한 run의 재제출
    uv run python -m pipeline.submit --record-existing <submission_ref> \
      --submission <csv> --run-name <name> --source-run-id <run_id> \
      --git-commit <commit> [--artifact <path>] [--param key=value] [--tag key=value]

제출은 마일스톤 단위 건전성 점검 용도다. 판단 기준은 CV(OOF)이고 public 점수는
CV와 같은 방향인지 확인하는 데만 쓴다. 일반 제출에서는 run_id가 필수다. (#20)

실수 방지 장치 (#20):
- git_dirty=True로 기록된 run은 제출 거부. 우회 옵션 없음.
- 이미 제출된 run(태그 submitted_at 또는 metric public_auc 존재)은 재제출 거부.
  --force로만 우회한다(일일 10회 한도 낭비 방지).
- 제출 메시지는 run 이름·run_id 앞 8자리·커밋 해시·OOF AUC로 자동 생성한다.

제출 직후 submitted_at 태그를 남기고, 점수가 나오면 metric public_auc를 기록한다.
점수 회수가 시간 초과로 끊긴 경우 같은 명령을 다시 실행하면 제출 없이
점수 회수만 다시 시도한다(메시지가 run 내용에서 결정되므로 제출 이력과 대조 가능).
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID, TARGET, file_sha256
from .runs import TRACKING_URI, MlflowRunStore, RunStoreError
from .tracking import mlflow_client

COMPETITION = "playground-series-s6e8"
SCORE_POLL_SECONDS = 5
SCORE_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class ExistingSubmissionResult:
    run_id: str
    created: bool


def _submission_rows_and_sha256(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise ValueError(f"제출 파일이 없다: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != [ID, TARGET]:
        raise ValueError(f"제출 열이 {[ID, TARGET]}가 아니다: {list(frame.columns)}")
    if frame.empty:
        raise ValueError("제출 파일에 행이 없다.")
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("제출 ID에 결측값이나 중복이 있다.")
    prediction = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("제출 예측에 유한하지 않은 값이 있다.")
    return len(frame), file_sha256(path)


def _submitted_at(value: datetime.datetime) -> str:
    if not isinstance(value, datetime.datetime):
        raise ValueError("Kaggle 제출 시각이 없다.")
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC).isoformat(timespec="seconds")


def _find_submission(api, submission_ref: int):
    from kagglesdk.competitions.types.submission_status import SubmissionStatus

    submissions = api.competition_submissions(COMPETITION, page_size=200) or []
    matches = [item for item in submissions if item and item.ref == submission_ref]
    if len(matches) != 1:
        raise ValueError(
            f"Kaggle 제출 {submission_ref}을 최근 제출 목록에서 하나로 찾지 못했다."
        )
    submission = matches[0]
    if submission.status != SubmissionStatus.COMPLETE:
        raise ValueError(f"Kaggle 제출 {submission_ref} 상태가 완료가 아니다: {submission.status}")
    if not submission.public_score:
        raise ValueError(f"Kaggle 제출 {submission_ref}에 공개 점수가 없다.")
    return submission


def _single_existing_run(client, experiment_id: str, submission_ref: int):
    matches = [
        run
        for run in client.search_runs([experiment_id], max_results=10_000)
        if run.data.tags.get("kaggle.submission_ref") == str(submission_ref)
    ]
    if len(matches) > 1:
        raise ValueError(f"Kaggle 제출 {submission_ref}이 MLflow에 중복 기록돼 있다.")
    return matches[0] if matches else None


def record_existing_submission(
    *,
    api,
    submission_ref: int,
    submission_path: Path,
    run_name: str,
    source_run_id: str,
    git_commit: str,
    artifacts: tuple[Path, ...] = (),
    params: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str = TRACKING_URI,
) -> ExistingSubmissionResult:
    """이미 접수된 Kaggle 제출을 별도 파생 실행으로 검증해 기록한다."""
    rows, submission_sha256 = _submission_rows_and_sha256(submission_path)
    if len(git_commit) != 40 or any(ch not in "0123456789abcdef" for ch in git_commit.lower()):
        raise ValueError("제출 당시 git commit은 40자리 16진수여야 한다.")

    client, experiment_id = mlflow_client(tracking_uri)
    existing = _single_existing_run(client, experiment_id, submission_ref)
    if existing is not None:
        if existing.data.tags.get("sha256.submission") != submission_sha256:
            raise ValueError(
                f"Kaggle 제출 {submission_ref}의 기존 MLflow 기록과 CSV SHA-256이 다르다."
            )
        return ExistingSubmissionResult(run_id=existing.info.run_id, created=False)

    from mlflow.exceptions import MlflowException

    try:
        source_run = client.get_run(source_run_id)
    except MlflowException as exc:
        raise ValueError(f"원본 MLflow 실행을 찾지 못했다: {source_run_id}") from exc

    reserved_params = {
        "experiment": run_name,
        "stage": "final_submission",
        "model.kind": "derived_submission",
        "source.run_id": source_run_id,
        "submission.original_name": submission_path.name,
    }
    reserved_tags = {
        "git_commit": git_commit,
        "git_dirty": "False",
        "source.kind": "derived_submission",
        "source.run_id": source_run_id,
        "kaggle.competition": COMPETITION,
        "kaggle.submission_ref": str(submission_ref),
        "sha256.submission": submission_sha256,
    }
    for supplied, reserved, label in (
        (params or {}, reserved_params, "param"),
        (tags or {}, reserved_tags, "tag"),
    ):
        overlap = sorted(set(supplied) & set(reserved))
        if overlap:
            raise ValueError(f"예약된 {label}을 덮어쓸 수 없다: {overlap}")

    extra_artifacts = tuple(Path(path) for path in artifacts)
    reserved_names = {"submission.csv", "submission_record.json"}
    artifact_names = [path.name for path in extra_artifacts]
    if any(not path.is_file() for path in extra_artifacts):
        missing = [str(path) for path in extra_artifacts if not path.is_file()]
        raise ValueError(f"추가 산출물이 없다: {missing}")
    if len(set(artifact_names)) != len(artifact_names) or reserved_names & set(artifact_names):
        raise ValueError("추가 산출물 이름이 중복되거나 예약된 이름과 겹친다.")

    submission = _find_submission(api, submission_ref)
    submitted_at = _submitted_at(submission.date)
    reserved_tags.update(
        {
            "kaggle.submission_message": submission.description,
            "kaggle.submission_file": submission.file_name,
            "submitted_at": submitted_at,
            "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        }
    )
    artifact_hashes = {path.name: file_sha256(path) for path in extra_artifacts}
    record = {
        "schema_version": 1,
        "kaggle": {
            "competition": COMPETITION,
            "submission_ref": submission_ref,
            "submitted_at": submitted_at,
            "message": submission.description,
            "file_name": submission.file_name,
            "public_score": submission.public_score,
            "status": submission.status.name,
        },
        "submission": {
            "original_name": submission_path.name,
            "rows": rows,
            "sha256": submission_sha256,
        },
        "source_run_id": source_run_id,
        "git_commit": git_commit,
        "artifacts_sha256": artifact_hashes,
    }

    run = client.create_run(experiment_id, run_name=run_name)
    run_id = run.info.run_id
    try:
        for key, value in {**reserved_params, **(params or {})}.items():
            client.log_param(run_id, key, value)
        for key, value in {**reserved_tags, **(tags or {})}.items():
            client.set_tag(run_id, key, value)
        client.log_metric(run_id, "public_auc", float(submission.public_score))
        if "auc_oof" in source_run.data.metrics:
            client.log_metric(run_id, "source_auc_oof", source_run.data.metrics["auc_oof"])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            shutil.copyfile(submission_path, tmp_dir / "submission.csv")
            (tmp_dir / "submission_record.json").write_text(
                json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            client.log_artifact(run_id, str(tmp_dir / "submission.csv"))
            client.log_artifact(run_id, str(tmp_dir / "submission_record.json"))
        for path in extra_artifacts:
            client.log_artifact(run_id, str(path))
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    return ExistingSubmissionResult(run_id=run_id, created=True)


def submission_message(run_name: str, run_id: str, git_commit: str, auc_oof: float) -> str:
    return f"{run_name} run={run_id[:8]} commit={git_commit[:8]} oof_auc={auc_oof:.5f}"


def fetch_public_score(api, message: str) -> str:
    """제출 목록에서 message와 일치하는 최신 제출의 public 점수를 기다려 돌려준다."""
    from kagglesdk.competitions.types.submission_status import SubmissionStatus

    deadline = time.monotonic() + SCORE_TIMEOUT_SECONDS
    while True:
        submissions = api.competition_submissions(COMPETITION) or []
        match = next((s for s in submissions if s and s.description == message), None)
        if match is not None:
            if match.status == SubmissionStatus.ERROR:
                sys.exit(f"제출이 오류로 끝났다: {match.error_description or '사유 미상'}")
            if match.status == SubmissionStatus.COMPLETE and match.public_score:
                return match.public_score
        if time.monotonic() >= deadline:
            sys.exit(
                "점수 회수 시간 초과: 제출은 접수됐다. 같은 명령을 다시 실행하면 "
                "제출 없이 점수 회수만 다시 시도한다."
            )
        time.sleep(SCORE_POLL_SECONDS)


def _key_values(values: list[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"{label}은 key=value 형식이어야 한다: {item}")
        if key in parsed:
            raise ValueError(f"{label} 키가 중복됐다: {key}")
        parsed[key] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow run의 submission artifact를 Kaggle에 제출")
    parser.add_argument("run_id", nargs="?", help="제출할 MLflow run_id")
    parser.add_argument("--force", action="store_true", help="이미 제출된 run의 재제출을 허용")
    parser.add_argument(
        "--record-existing",
        type=int,
        metavar="SUBMISSION_REF",
        help="이미 접수된 Kaggle 제출을 새 파생 실행으로 기록",
    )
    parser.add_argument("--submission", type=Path, help="이미 제출한 CSV 경로")
    parser.add_argument("--run-name", help="사후 등록할 MLflow 실행 이름")
    parser.add_argument("--source-run-id", help="제출 예측의 원본 MLflow 실행 ID")
    parser.add_argument("--git-commit", help="제출 파일을 만든 40자리 git commit")
    parser.add_argument(
        "--artifact", type=Path, action="append", default=[], help="함께 기록할 산출물 경로"
    )
    parser.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE", help="추가 MLflow param"
    )
    parser.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE", help="추가 MLflow tag"
    )
    args = parser.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    if args.record_existing is not None:
        if args.run_id is not None or args.force:
            parser.error("--record-existing은 run_id 또는 --force와 함께 쓸 수 없다.")
        required = {
            "--submission": args.submission,
            "--run-name": args.run_name,
            "--source-run-id": args.source_run_id,
            "--git-commit": args.git_commit,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"--record-existing에 필요한 인자가 없다: {', '.join(missing)}")
        try:
            params = _key_values(args.param, "param")
            tags = _key_values(args.tag, "tag")
        except ValueError as exc:
            parser.error(str(exc))
        api = KaggleApi()
        api.authenticate()
        try:
            result = record_existing_submission(
                api=api,
                submission_ref=args.record_existing,
                submission_path=args.submission,
                run_name=args.run_name,
                source_run_id=args.source_run_id,
                git_commit=args.git_commit,
                artifacts=tuple(args.artifact),
                params=params,
                tags=tags,
            )
        except (RunStoreError, ValueError) as exc:
            sys.exit(str(exc))
        action = "기록 완료" if result.created else "이미 같은 내용으로 기록됨"
        print(f"{action}: run {result.run_id}")
        return

    if args.run_id is None:
        parser.error("제출할 run_id 또는 --record-existing이 필요하다.")
    if any(
        value
        for value in (
            args.submission,
            args.run_name,
            args.source_run_id,
            args.git_commit,
            args.artifact,
            args.param,
            args.tag,
        )
    ):
        parser.error("사후 등록 전용 인자는 일반 제출에 쓸 수 없다.")

    store = MlflowRunStore()
    try:
        meta = store.facts_of(args.run_id)
    except RunStoreError as exc:
        sys.exit(str(exc))

    if meta.tags.get("git_dirty") == "True":
        sys.exit("제출 거부: git_dirty=True로 기록된 run이다. 우회 옵션은 없다. 커밋 후 재실행할 것.")

    already_scored = "public_auc" in meta.metrics
    already_submitted = "submitted_at" in meta.tags
    if already_scored and not args.force:
        sys.exit(
            f"제출 거부: 이 run에는 이미 public_auc={meta.metrics['public_auc']}가 있다. "
            "재제출하려면 --force."
        )

    auc_oof = meta.metrics["auc_oof"]
    message = submission_message(
        meta.run_name, args.run_id, meta.tags["git_commit"], auc_oof
    )

    api = KaggleApi()
    api.authenticate()

    if already_submitted and not args.force:
        # 앞선 실행이 제출 후 점수 회수에서 끊긴 경우: 제출 없이 회수만 재시도.
        print(f"이미 제출된 run({meta.tags['submitted_at']}): 점수 회수만 다시 시도한다.")
    else:
        try:
            path = store.submission_path_of(args.run_id)
        except RunStoreError as exc:
            sys.exit(str(exc))
        print(f"제출: {COMPETITION} ← run {args.run_id[:8]}")
        print(f"메시지: {message}")
        result = api.competition_submit(str(path), message, COMPETITION, quiet=True)
        if result.message == api.COMPETITION_SUBMIT_UPLOAD_FAILED_MESSAGE:
            sys.exit(f"제출 실패: {result.message}")
        submitted_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        store.annotate(args.run_id, tags={"submitted_at": submitted_at})

    public_score = fetch_public_score(api, message)
    store.annotate(args.run_id, metrics={"public_auc": float(public_score)})
    print(f"public_auc={public_score} (oof_auc={auc_oof:.5f}) → run에 기록 완료.")


if __name__ == "__main__":
    main()
