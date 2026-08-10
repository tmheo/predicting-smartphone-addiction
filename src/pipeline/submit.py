"""Kaggle 제출. MLflow run의 submission artifact를 제출하고 public 점수를 run에 기록한다. (#20, #21)

사용법:
    uv run python -m pipeline.submit <run_id>
    uv run python -m pipeline.submit <run_id> --force  # 이미 제출한 run의 재제출

제출은 마일스톤 단위 건전성 점검 용도다. 판단 기준은 CV(OOF)이고 public 점수는
CV와 같은 방향인지 확인하는 데만 쓴다. 그래서 run_id는 필수이고 기본값이 없다. (#20)

실수 방지 장치 (#20):
- git_dirty=True로 기록된 run은 제출 거부. 우회 옵션 없음.
- 이미 제출된 run(태그 submitted_at 또는 metric public_auc 존재)은 재제출 거부.
  --force로만 우회한다(일일 5회 한도 낭비 방지).
- 제출 메시지는 run 이름·run_id 앞 8자리·커밋 해시·OOF AUC로 자동 생성한다.

제출 직후 submitted_at 태그를 남기고, 점수가 나오면 metric public_auc를 기록한다.
점수 회수가 시간 초과로 끊긴 경우 같은 명령을 다시 실행하면 제출 없이
점수 회수만 다시 시도한다(메시지가 run 내용에서 결정되므로 제출 이력과 대조 가능).
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time

COMPETITION = "playground-series-s6e8"
TRACKING_URI = "sqlite:///mlflow.db"
SCORE_POLL_SECONDS = 5
SCORE_TIMEOUT_SECONDS = 300


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


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow run의 submission artifact를 Kaggle에 제출")
    parser.add_argument("run_id", help="제출할 MLflow run_id (필수, 기본값 없음)")
    parser.add_argument("--force", action="store_true", help="이미 제출된 run의 재제출을 허용")
    args = parser.parse_args()

    import mlflow
    from kaggle.api.kaggle_api_extended import KaggleApi

    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(args.run_id)

    if run.data.tags.get("git_dirty") == "True":
        sys.exit("제출 거부: git_dirty=True로 기록된 run이다. 우회 옵션은 없다. 커밋 후 재실행할 것.")

    already_scored = "public_auc" in run.data.metrics
    already_submitted = "submitted_at" in run.data.tags
    if already_scored and not args.force:
        sys.exit(
            f"제출 거부: 이 run에는 이미 public_auc={run.data.metrics['public_auc']}가 있다. "
            "재제출하려면 --force."
        )

    auc_oof = run.data.metrics["auc_oof"]
    message = submission_message(
        run.info.run_name, args.run_id, run.data.tags["git_commit"], auc_oof
    )

    api = KaggleApi()
    api.authenticate()

    if already_submitted and not args.force:
        # 앞선 실행이 제출 후 점수 회수에서 끊긴 경우: 제출 없이 회수만 재시도.
        print(f"이미 제출된 run({run.data.tags['submitted_at']}): 점수 회수만 다시 시도한다.")
    else:
        path = client.download_artifacts(args.run_id, "submission.csv")
        print(f"제출: {COMPETITION} ← run {args.run_id[:8]}")
        print(f"메시지: {message}")
        result = api.competition_submit(path, message, COMPETITION, quiet=True)
        if result.message == api.COMPETITION_SUBMIT_UPLOAD_FAILED_MESSAGE:
            sys.exit(f"제출 실패: {result.message}")
        submitted_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        client.set_tag(args.run_id, "submitted_at", submitted_at)

    public_score = fetch_public_score(api, message)
    client.log_metric(args.run_id, "public_auc", float(public_score))
    print(f"public_auc={public_score} (oof_auc={auc_oof:.5f}) → run에 기록 완료.")


if __name__ == "__main__":
    main()
