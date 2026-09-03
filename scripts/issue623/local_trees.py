"""이슈 #623 로컬 CPU 실행: 나무 3계열의 기준 1개 + 사다리 3개를 3시드 확정 단계로 학습한다.

계열(lgb, xgb, cat)마다 차선(lane) 하나가 기준 -> raw4 -> cats_te -> ratio_round 순서로
`pipeline.run --stage confirm`을 돌린다. 세 차선은 `pipeline.jobs.run_jobs`로 동시에 뜨고
OpenMP 계열 스레드는 차선당 `--threads`로 나눈다(CatBoost는 자체 스레드 풀이라 이 값을 무시한다).

이어달리기: 설정마다 결과 JSON(`<out-root>/local/<experiment>.json`)이 있으면 건너뛰고,
없는 설정은 `pipeline.run`을 다시 부르는데 fold 복구(run-recovery)가 끝난 (seed, fold)를
재사용하므로 중단 지점부터 이어진다. 살아 있는 차선이 있으면 lock 때문에 다시 뜨지 않는다.

사용법(작업 폴더 = 실행 커밋의 깨끗한 체크아웃):
    .venv/bin/python scripts/issue623/local_trees.py drive --out-root <메인>/run-logs/issue623
    .venv/bin/python scripts/issue623/local_trees.py status --out-root <메인>/run-logs/issue623
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from pipeline.jobs import Job, run_jobs  # noqa: E402

STAGE = "confirm"
SEEDS = (42, 43, 44)
RUN_LINE = re.compile(
    r"^run_id=(?P<run_id>[0-9a-f]{32}) auc_oof=(?P<auc>[0-9.]+) auc_oof_weighted=(?P<weighted>[0-9.]+)$"
)

# 계열별 차선: (실험 이름, 설정 경로). 기준이 먼저다(짝비교 기준이자 fold-fit 재사용의 출처).
LANES: dict[str, list[tuple[str, str]]] = {
    "lgb": [
        ("exp117_ag25_gbm_r21", "configs/exp117_ag25_gbm_r21.yaml"),
        ("cdv2_lgb_raw4", "configs/constraint-derived/01_lgb_exp117_raw4.yaml"),
        ("cdv2_lgb_cats_te", "configs/constraint-derived/02_lgb_exp117_cats_te.yaml"),
        ("cdv2_lgb_ratio_round", "configs/constraint-derived/03_lgb_exp117_ratio_round.yaml"),
    ],
    "xgb": [
        ("exp135_xgb_hpo_trial30", "configs/exp135_xgb_hpo_trial30.yaml"),
        ("cdv2_xgb_raw4", "configs/constraint-derived/04_xgb_exp135_raw4.yaml"),
        ("cdv2_xgb_cats_te", "configs/constraint-derived/05_xgb_exp135_cats_te.yaml"),
        ("cdv2_xgb_ratio_round", "configs/constraint-derived/06_xgb_exp135_ratio_round.yaml"),
    ],
    "cat": [
        ("exp070_cat_exact_cats", "configs/exp070_cat_exact_cats.yaml"),
        ("cdv2_cat_raw4", "configs/constraint-derived/07_cat_exp070_raw4.yaml"),
        ("cdv2_cat_cats_te", "configs/constraint-derived/08_cat_exp070_cats_te.yaml"),
        ("cdv2_cat_ratio_round", "configs/constraint-derived/09_cat_exp070_ratio_round.yaml"),
    ],
}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return commit, dirty


def result_path(out_root: Path, experiment: str) -> Path:
    return out_root / "local" / f"{experiment}.json"


def verify_and_record(
    experiment: str, config: str, family: str, run_id: str, out_root: Path, started: str, seconds: float
) -> dict:
    """MLflow 실행을 되읽어 계약(FINISHED, 커밋, 깨끗함, 시드, 실험 이름)을 확인하고 묶음을 내보낸다."""
    import math

    from pipeline.bundle import export_bundle
    from pipeline.data import file_sha256
    from pipeline.tracking import mlflow_client

    client, _ = mlflow_client()
    run = client.get_run(run_id)
    tags, params, metrics = dict(run.data.tags), dict(run.data.params), dict(run.data.metrics)
    commit, _ = git_state()
    if run.info.status != "FINISHED":
        raise RuntimeError(f"{experiment} {run_id} 상태가 FINISHED가 아니다: {run.info.status}")
    if tags.get("git_commit") != commit or tags.get("git_dirty") != "False":
        raise RuntimeError(
            f"{experiment} {run_id} git 태그 불일치: {tags.get('git_commit')} dirty={tags.get('git_dirty')}"
        )
    if params.get("experiment") != experiment:
        raise RuntimeError(f"{experiment} {run_id} 실험 이름이 다르다: {params.get('experiment')}")
    if params.get("stage") != STAGE or params.get("seeds") != ",".join(map(str, SEEDS)):
        raise RuntimeError(f"{experiment} {run_id} 단계 또는 시드가 다르다: {params}")
    auc = float(metrics["auc_oof"])
    if not math.isfinite(auc):
        raise RuntimeError(f"{experiment} {run_id} OOF AUC가 유한하지 않다.")
    bundle_dir = out_root / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle_dir / f"{experiment}.bundle.zip"
    if bundle.exists():
        bundle.unlink()
    export_bundle(run_id, bundle)
    return {
        "schema_version": 1,
        "issue": 623,
        "provider": "local",
        "family": family,
        "experiment": experiment,
        "config": config,
        "run_id": run_id,
        "git_commit": commit,
        "stage": STAGE,
        "seeds": list(SEEDS),
        "auc_oof": auc,
        "auc_oof_seed": {seed: float(metrics[f"auc_oof_seed_{seed}"]) for seed in SEEDS},
        "fold_aucs": {k: float(v) for k, v in metrics.items() if k.startswith("auc_fold_")},
        "started_at": started,
        "finished_at": now(),
        "wall_seconds": round(seconds, 1),
        "bundle": str(bundle),
        "bundle_sha256": file_sha256(bundle),
    }


def run_one(experiment: str, config: str, family: str, out_root: Path) -> None:
    target = result_path(out_root, experiment)
    if target.exists():
        print(f"[lane {family}] 완료됨(건너뜀) {experiment} {now()}", flush=True)
        return
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError(f"작업 폴더가 깨끗하지 않아 {experiment}를 시작하지 않는다.")
    log_dir = out_root / "local"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment}.log"
    started = now()
    clock = time.monotonic()
    print(f"[lane {family}] 시작 {experiment} commit={commit[:8]} {started}", flush=True)
    with log_path.open("a") as handle:
        handle.write(f"=== start {started} commit={commit}\n")
        handle.flush()
        completed = subprocess.run(
            [sys.executable, "-m", "pipeline.run", config, "--stage", STAGE],
            cwd=PROJECT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    seconds = time.monotonic() - clock
    if completed.returncode != 0:
        raise RuntimeError(f"{experiment} pipeline.run 종료 코드 {completed.returncode} (로그 {log_path})")
    matches = [
        m.groupdict()
        for line in log_path.read_text(errors="replace").splitlines()
        if (m := RUN_LINE.fullmatch(line.strip()))
    ]
    if not matches:
        raise RuntimeError(f"{experiment} 로그에 run_id 줄이 없다: {log_path}")
    run_id = matches[-1]["run_id"]
    record = verify_and_record(experiment, config, family, run_id, out_root, started, seconds)
    part = target.with_suffix(".json.part")
    part.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    part.replace(target)
    print(f"[lane {family}] 완료 {experiment} run={run_id[:8]} auc={record['auc_oof']:.6f} {now()}", flush=True)


def run_lane(family: str, out_root: Path) -> None:
    for experiment, config in LANES[family]:
        run_one(experiment, config, family, out_root)
    marker = out_root / "local" / f"lane-{family}.done"
    marker.write_text(now() + "\n")


def drive(out_root: Path, workers: int, threads: int) -> None:
    commit, dirty = git_state()
    if dirty:
        raise SystemExit("작업 폴더가 깨끗하지 않다. 실행 기록의 git_dirty가 True가 되므로 시작하지 않는다.")
    print(f"[drive] commit={commit} lanes={list(LANES)} workers={workers} threads={threads}", flush=True)
    jobs = [
        Job(
            f"lane-{family}",
            [sys.executable, str(Path(__file__).resolve()), "run-lane", "--family", family, "--out-root", str(out_root)],
            out_root / "local" / f"lane-{family}.done",
        )
        for family in LANES
    ]
    run_jobs(jobs, workers=workers, threads=threads, log_dir=out_root / "local" / "lanes")


def status(out_root: Path) -> None:
    for family, items in LANES.items():
        parts = []
        for experiment, _ in items:
            target = result_path(out_root, experiment)
            if target.exists():
                record = json.loads(target.read_text())
                parts.append(f"{experiment}=done({record['auc_oof']:.6f},{record['wall_seconds'] / 60:.0f}m)")
            else:
                recovery = PROJECT / "run-recovery" / f"{experiment}-{STAGE}"
                folds = len(list(recovery.glob("seed_*/fold_*/manifest.json"))) if recovery.exists() else 0
                parts.append(f"{experiment}=pending(folds {folds}/15)")
        print(f"[{family}] " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="이슈 #623 로컬 나무 3계열 3시드 실행")
    sub = parser.add_subparsers(dest="command", required=True)
    p_drive = sub.add_parser("drive")
    p_drive.add_argument("--out-root", type=Path, required=True)
    p_drive.add_argument("--workers", type=int, default=3)
    p_drive.add_argument("--threads", type=int, default=4)
    p_lane = sub.add_parser("run-lane")
    p_lane.add_argument("--family", choices=sorted(LANES), required=True)
    p_lane.add_argument("--out-root", type=Path, required=True)
    p_status = sub.add_parser("status")
    p_status.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "drive":
        drive(args.out_root.resolve(), args.workers, args.threads)
    elif args.command == "run-lane":
        run_lane(args.family, args.out_root.resolve())
    else:
        status(args.out_root.resolve())


if __name__ == "__main__":
    main()
