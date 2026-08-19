"""fold 0 검증 예측과 풀 구성원 OOF fold 0 조각의 스피어만 상관을 잰다. (#199 게이트 3)

main 작업 폴더(mlflow.db가 있는 곳)에서 실행한다.

사용법:
    uv run python scripts/diagnose_tabr_pool_correlation.py \
        --predictions run-logs/.../validation_predictions.parquet \
        --out run-logs/.../pool_fold0_correlation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.judgment import spearman
from pipeline.ledger import Pool
from pipeline.runs import MlflowStore

GATE_THRESHOLD = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabR fold 0 풀 상관 진단 (#199)")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    fold_rows = predictions[predictions["fold"] == args.fold]
    if fold_rows.empty:
        raise ValueError(f"예측에 fold {args.fold} 행이 없다.")
    candidate = fold_rows.set_index("id")["pred"]
    if not candidate.index.is_unique:
        raise ValueError("검증 예측의 id가 고유하지 않다.")

    store = MlflowStore()
    pool = Pool.load()
    correlations: dict[str, dict[str, object]] = {}
    for member in pool.members:
        member_oof = store.oof_of(member.run_id).reindex(candidate.index)
        if member_oof.isna().any():
            raise ValueError(
                f"구성원 {member.run_id}의 OOF에 fold {args.fold} id가 빠져 있다."
            )
        correlations[member.run_id] = {
            "config": member.config,
            "spearman": spearman(candidate, member_oof),
        }

    nearest_id = max(correlations, key=lambda rid: correlations[rid]["spearman"])
    nearest = correlations[nearest_id]
    payload = {
        "diagnostic": "tabr_pool_fold0_correlation",
        "issue": 199,
        "fold": args.fold,
        "candidate_rows": int(len(candidate)),
        "pool_members": len(correlations),
        "correlations": correlations,
        "nearest": {
            "run_id": nearest_id,
            "config": nearest["config"],
            "spearman": nearest["spearman"],
        },
        "gate3": {
            "threshold": GATE_THRESHOLD,
            "passed": bool(nearest["spearman"] < GATE_THRESHOLD),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[pool-corr] nearest={nearest['config']} spearman={nearest['spearman']:.5f} "
        f"gate3={'pass' if payload['gate3']['passed'] else 'stop'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
