"""fold 0 짝비교 선별 결과를 로컬 MLflow에 기록한다. (#385)

`scripts/screen_muon_lr.py`(#385)나 같은 구조의 선별 스크립트가 남긴 JSON Lines를
읽어 묶음 실행 하나와 후보별 자식 실행을 만든다.

## 이 실행은 판정 대상이 아니다

fold 하나와 시드 하나만 도는 약식 검증이므로 5-fold OOF가 없다.
그래서 판정 경로가 읽는 이름을 **일부러 쓰지 않는다.**

- `auc_oof`와 `auc_oof_seed_*` metric을 기록하지 않는다. fold 0 값은
  `auc_fold_0_seed_42`로만 남는다.
- `oof.parquet`, `test_pred.parquet` 산출물을 붙이지 않는다.
  풀 감사(`pipeline.pool_audit`)가 이 실행을 후보로 잘못 집으면 산출물 부재로 즉시 실패한다.
- `judgment.eligible=false`와 `screening.scope=fold0_only` 태그로 의도를 명시한다.

풀 구성원은 `artifacts/pool.yaml`의 명시적 run_id로만 정해지므로 이 실행이 저절로
후보가 되는 경로는 없다. 위 세 가지는 사람이 실수로 집었을 때의 방어선이다.

## 입력 계약

JSON Lines의 각 줄은 다음을 가진다.

- `candidate`: 후보 식별 문자열(자식 실행 이름의 꼬리).
- `auc`: fold 0 AUC.
- `seed`, `fold`, `fit_seconds`.
- 나머지 스칼라 키는 metric 또는 param으로 기록한다.
  수치는 metric, 그 밖은 param이다.
- `model_training_diagnostics`: 있으면 자식 실행의 산출물 JSON으로 붙인다.

사용법:
    uv run python scripts/record_fold0_screening.py \\
        --records run-logs/vast-issue385/results/results/muon-lr-fold0.jsonl \\
        --group-name screen-issue385-muon-lr \\
        --baseline-candidate 0.002:1 \\
        --note "이슈 385 1단계: 공유 최고 학습률" \\
        --attach docs/research/champion-muon-learning-rate-fold0-pairwise.md
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runs import TRACKING_URI  # noqa: E402
from pipeline.tracking import git_state, mlflow_client  # noqa: E402

# 판정 경로가 읽는 이름. 선별 실행에는 절대 기록하지 않는다.
FORBIDDEN_METRIC_PREFIXES = ("auc_oof",)
# 후보별 기록에서 metric·param으로 옮기지 않고 따로 다루는 키.
STRUCTURAL_KEYS = {"candidate", "auc", "model_training_diagnostics"}


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"기록이 비어 있다: {path}")
    seen = [record["candidate"] for record in records]
    if len(set(seen)) != len(seen):
        raise ValueError(f"후보 이름이 겹친다: {seen}")
    return records


def scalar_fields(record: dict) -> tuple[dict[str, float], dict[str, str]]:
    """후보 기록의 스칼라를 metric과 param으로 가른다."""
    metrics: dict[str, float] = {}
    params: dict[str, str] = {}
    for key, value in record.items():
        if key in STRUCTURAL_KEYS:
            continue
        if isinstance(value, bool) or value is None:
            params[key] = str(value)
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
        elif isinstance(value, str):
            params[key] = value
        else:
            params[key] = json.dumps(value, ensure_ascii=False)
    return metrics, params


def guard_metric_names(names) -> None:
    for name in names:
        if name.startswith(FORBIDDEN_METRIC_PREFIXES):
            raise ValueError(
                f"선별 실행은 판정 metric 이름을 쓸 수 없다: {name}. "
                "fold 하나짜리 값을 OOF로 읽히게 두면 안 된다."
            )


def summarize(records: list[dict], baseline: str | None) -> dict[str, float]:
    """묶음 실행에 남길 요약. 기준 후보가 있으면 짝차이의 범위도 남긴다."""
    aucs = [float(record["auc"]) for record in records]
    summary = {
        "candidate_count": float(len(records)),
        "auc_fold_0_best": max(aucs),
        "auc_fold_0_worst": min(aucs),
        "auc_fold_0_spread": max(aucs) - min(aucs),
        "fit_seconds_total": float(
            sum(float(record.get("fit_seconds", 0.0)) for record in records)
        ),
    }
    if baseline is None:
        return summary
    base = next((r for r in records if r["candidate"] == baseline), None)
    if base is None:
        raise ValueError(f"기준 후보를 찾지 못했다: {baseline}")
    diffs = [
        float(r["auc"]) - float(base["auc"])
        for r in records
        if r["candidate"] != baseline
    ]
    summary["auc_fold_0_baseline"] = float(base["auc"])
    if diffs:
        summary["diff_vs_baseline_best"] = max(diffs)
        summary["diff_vs_baseline_worst"] = min(diffs)
        summary["positive_candidate_count"] = float(sum(1 for d in diffs if d >= 0))
    return summary


def record(
    *,
    records: list[dict],
    group_name: str,
    note: str,
    baseline: str | None,
    execution_commit: str | None,
    attachments: list[Path],
    records_path: Path,
    tracking_uri: str = TRACKING_URI,
) -> str:
    client, experiment_id = mlflow_client(tracking_uri)
    state = git_state()

    common_tags = {
        "screening.scope": "fold0_only",
        "screening.group": group_name,
        "judgment.eligible": "false",
        "recorded.git_commit": state["git_commit"],
        "recorded.git_dirty": state["git_dirty"],
    }
    if execution_commit:
        common_tags["execution.git_commit"] = execution_commit

    parent = client.create_run(
        experiment_id,
        run_name=group_name,
        tags={**common_tags, "mlflow.note.content": note},
    )
    summary = summarize(records, baseline)
    guard_metric_names(summary)
    for key, value in summary.items():
        client.log_metric(parent.info.run_id, key, value)
    if baseline:
        client.log_param(parent.info.run_id, "baseline_candidate", baseline)
    client.log_param(parent.info.run_id, "records_source", records_path.name)
    for attachment in [records_path, *attachments]:
        client.log_artifact(parent.info.run_id, str(attachment))

    for item in records:
        metrics, params = scalar_fields(item)
        guard_metric_names(metrics)
        if baseline:
            base = next(r for r in records if r["candidate"] == baseline)
            metrics["diff_vs_baseline"] = float(item["auc"]) - float(base["auc"])
        metrics[f"auc_fold_{int(item.get('fold', 0))}_seed_{int(item.get('seed', 0))}"] = (
            float(item["auc"])
        )
        child = client.create_run(
            experiment_id,
            run_name=f"{group_name}/{item['candidate']}",
            tags={
                **common_tags,
                "mlflow.parentRunId": parent.info.run_id,
                "screening.candidate": item["candidate"],
                "screening.is_baseline": str(item["candidate"] == baseline),
            },
        )
        for key, value in metrics.items():
            client.log_metric(child.info.run_id, key, value)
        for key, value in params.items():
            client.log_param(child.info.run_id, key, value[:5000])
        diagnostics = item.get("model_training_diagnostics")
        if diagnostics:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model_training_diagnostics.json"
                path.write_text(
                    json.dumps(diagnostics, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                client.log_artifact(child.info.run_id, str(path))
        client.set_terminated(child.info.run_id, "FINISHED")

    client.set_terminated(parent.info.run_id, "FINISHED")
    return parent.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="fold 0 선별 결과를 MLflow에 기록한다")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--baseline-candidate",
        help="같은 기계 재현 기준 후보. 주면 짝차이를 함께 기록한다.",
    )
    parser.add_argument(
        "--execution-commit", help="원격 실행에 쓴 커밋. 기록 시점 커밋과 다를 수 있다."
    )
    parser.add_argument(
        "--attach",
        type=Path,
        nargs="*",
        default=[],
        help="묶음 실행에 함께 붙일 파일(연구 문서, 실행 장부 등).",
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    args = parser.parse_args()

    records = load_records(args.records)
    run_id = record(
        records=records,
        group_name=args.group_name,
        note=args.note,
        baseline=args.baseline_candidate,
        execution_commit=args.execution_commit,
        attachments=list(args.attach),
        records_path=args.records,
        tracking_uri=args.tracking_uri,
    )
    print(f"묶음 실행 {run_id}에 후보 {len(records)}개를 기록했다.")


if __name__ == "__main__":
    main()
