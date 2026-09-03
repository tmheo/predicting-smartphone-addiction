"""원격 실행 결과 4개를 검증하고 실행 기록 묶음으로 내보낸다. (이슈 #623, RealMLP 사다리)"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

from pipeline.bundle import export_bundle
from pipeline.data import file_sha256
from pipeline.tracking import mlflow_client


LINE = re.compile(
    r"^run_id=(?P<run_id>[0-9a-f]{32}) auc_oof=(?P<auc>[0-9.]+) auc_oof_weighted=(?P<weighted>[0-9.]+)$"
)
EXPECTED_INPUT_SHA = {
    "train": "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c",
    "test": "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e",
    "folds": "5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4",
}
REQUIRED_ARTIFACTS = {
    "oof.parquet",
    "oof_seed_42.parquet",
    "oof_seed_43.parquet",
    "oof_seed_44.parquet",
    "test_pred.parquet",
    "submission.csv",
    "feature_importance.parquet",
    "fold_feature_reuse.json",
    "model_training_diagnostics.json",
    "feature_importance_summary.csv",
    "stage_durations.csv",
    "summary.html",
}


def artifact_paths(client, run_id: str, root: str = "") -> set[str]:
    paths: set[str] = set()
    for item in client.list_artifacts(run_id, root):
        if item.is_dir:
            paths.update(artifact_paths(client, run_id, item.path))
        else:
            paths.add(item.path)
    return paths


def finalize_one(client, experiment: str, log: Path, args) -> dict:
    matches = [
        match.groupdict()
        for line in log.read_text(errors="replace").splitlines()
        if (match := LINE.fullmatch(line.strip()))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{experiment} 실행 완료 줄이 정확히 하나가 아니다: {matches}")
    run_id = matches[0]["run_id"]
    run = client.get_run(run_id)
    tags = dict(run.data.tags)
    params = dict(run.data.params)
    if run.info.status != "FINISHED":
        raise RuntimeError(f"{run_id} 상태가 FINISHED가 아니다: {run.info.status}")
    expected_tags = {
        "git_commit": args.expected_commit,
        "git_dirty": "False",
        "remote.provider": "vast",
        "remote.job_id": args.job_id,
    }
    for key, expected in expected_tags.items():
        if tags.get(key) != expected:
            raise RuntimeError(f"{run_id} 태그 {key} 불일치: {tags.get(key)!r} != {expected!r}")
    for name, digest in EXPECTED_INPUT_SHA.items():
        if tags.get(f"sha256.{name}") != digest:
            raise RuntimeError(f"{run_id} 입력 해시 {name}이 다르다.")
    if params.get("experiment") != experiment:
        raise RuntimeError(f"{run_id} 실험 이름이 다르다: {params.get('experiment')}")
    if params.get("stage") != "confirm" or params.get("seeds") != "42,43,44":
        raise RuntimeError(f"{run_id} 확정 단계 또는 난수 목록이 다르다: {params}")
    paths = artifact_paths(client, run_id)
    missing = sorted(REQUIRED_ARTIFACTS - paths)
    if missing:
        raise RuntimeError(f"{run_id} 필수 산출물 누락: {missing}")
    auc = float(run.data.metrics["auc_oof"])
    if not math.isfinite(auc):
        raise RuntimeError(f"{run_id} OOF AUC가 유한하지 않다: {auc}")
    seed_aucs = {seed: float(run.data.metrics[f"auc_oof_seed_{seed}"]) for seed in (42, 43, 44)}

    bundle_path = args.results / f"{experiment}.bundle.zip"
    export_bundle(run_id, bundle_path)
    run_log = Path("run-logs") / run_id / "run.log"
    if run_log.is_file():
        shutil.copy2(run_log, args.results / f"pipeline-run-log-{experiment}.txt")
    return {
        "experiment": experiment,
        "run_id": run_id,
        "status": run.info.status,
        "auc_oof": auc,
        "auc_oof_seed": seed_aucs,
        "auc_oof_weighted": float(run.data.metrics.get("auc_oof_weighted", float("nan"))),
        "fold_aucs": {
            key: float(value) for key, value in run.data.metrics.items() if key.startswith("auc_fold_")
        },
        "bundle": bundle_path.name,
        "bundle_size": bundle_path.stat().st_size,
        "bundle_sha256": file_sha256(bundle_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--experiment", action="append", required=True, metavar="NAME=LOG", help="실험 이름=실행 로그 경로"
    )
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    client, _ = mlflow_client()
    runs = []
    for item in args.experiment:
        experiment, _, log = item.partition("=")
        if not experiment or not log:
            raise RuntimeError(f"--experiment 형식이 NAME=LOG가 아니다: {item}")
        runs.append(finalize_one(client, experiment, Path(log), args))

    audit = {
        "schema_version": 1,
        "issue": 623,
        "job_id": args.job_id,
        "vast_instance_id": int(args.instance_id),
        "git_commit": args.expected_commit,
        "input_sha256": EXPECTED_INPUT_SHA,
        "runs": runs,
    }
    audit_path = args.results / "remote-run-audit.json"
    part_path = audit_path.with_suffix(".json.part")
    part_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    part_path.replace(audit_path)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
