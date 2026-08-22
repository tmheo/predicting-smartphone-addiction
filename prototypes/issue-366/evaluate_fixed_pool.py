"""사전 지정한 제거 집합에 등록 결합 전략 19개를 다시 적용한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.pool_rereview import StrategyEvaluator, load_inputs

from fast_single_removal import _write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--anchor-decision", type=Path, required=True)
    parser.add_argument("--remove", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()

    anchor = json.loads(args.anchor_decision.read_text())
    context = load_inputs(args.predictions)
    if context.prediction_file_sha256 != anchor["input"]["prediction_sha256"]:
        raise ValueError("기준 판정과 현재 예측 파일의 해시가 다르다.")
    removed = tuple(args.remove)
    unknown = sorted(set(removed) - set(context.members))
    if unknown:
        raise ValueError(f"35개 기준 풀에 없는 제거 대상이다: {unknown}")
    selected = tuple(member for member in context.members if member not in removed)

    with StrategyEvaluator(context, jobs=args.jobs) as evaluator:
        result = evaluator.evaluate(selected, excluded_fold=None)

    anchor_result = anchor["splits"]["final"]["all_strategies"]
    payload = {
        "schema_version": 1,
        "issue": 346,
        "prediction_sha256": context.prediction_file_sha256,
        "anchor_member_count": len(context.members),
        "removed_members": list(removed),
        "selected_member_count": len(selected),
        "selected_members": list(selected),
        "registered_strategy_count": len(result.strategy_auc),
        "anchor_best_strategy": anchor_result["anchor_best_strategy"],
        "anchor_best_auc": anchor_result["anchor_best_auc"],
        "selected_best_strategy": result.best_strategy,
        "selected_best_auc": result.best_auc,
        "delta_vs_anchor": result.best_auc - anchor_result["anchor_best_auc"],
        "strategy_auc": result.strategy_auc,
        "strategy_fold_auc": result.strategy_fold_auc,
        "strategy_fits": evaluator.fits,
    }
    sha256 = _write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256,
        "selected_best_strategy": result.best_strategy,
        "selected_best_auc": result.best_auc,
        "delta_vs_anchor": payload["delta_vs_anchor"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
