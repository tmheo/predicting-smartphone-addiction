"""빠른 선별 점수가 양수인 단일 제거 후보를 정확 평가한다."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pipeline.pool_rereview import StrategyEvaluator, load_inputs

from fast_single_removal import FIXED_STRATEGY, _batch_single_removals, _write_json


def _candidate_order(decision: dict[str, Any]) -> list[dict[str, Any]]:
    screened = decision["splits"]["final"]["trajectory"][0]["screen_candidates"]
    return sorted(
        (
            {
                "removed": entry["removed"],
                "screen_delta": float(entry["delta_vs_working"]),
            }
            for entry in screened
            if float(entry["delta_vs_working"]) > 0.0
        ),
        key=lambda entry: (-entry["screen_delta"], entry["removed"]),
    )


def _checkpoint_payload(
    *,
    context: Any,
    anchor_auc: float,
    baseline_removed_members: tuple[str, ...],
    working_member_count: int,
    candidates: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "issue": 346,
        "status": "in_progress",
        "prediction_sha256": context.prediction_file_sha256,
        "strategy": FIXED_STRATEGY,
        "anchor_auc": anchor_auc,
        "baseline_removed_members": list(baseline_removed_members),
        "working_member_count": working_member_count,
        "candidates": candidates,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--screen-decision", type=Path, required=True)
    parser.add_argument("--seed-result", type=Path)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--baseline-remove", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--deadline-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("묶음 크기는 1 이상이어야 한다.")

    started = time.monotonic()
    context = load_inputs(args.predictions)
    decision = json.loads(args.screen_decision.read_text())
    if context.prediction_file_sha256 != decision["input"]["prediction_sha256"]:
        raise ValueError("선별 판정과 현재 예측 파일의 해시가 다르다.")
    baseline_removed = tuple(args.baseline_remove)
    unknown_baseline = sorted(set(baseline_removed) - set(context.members))
    if unknown_baseline:
        raise ValueError(f"35개 기준 풀에 없는 선행 제거 대상이다: {unknown_baseline}")
    working_members = tuple(
        member for member in context.members if member not in baseline_removed
    )
    if args.baseline_result is None:
        if baseline_removed:
            raise ValueError("선행 제거가 있으면 그 기준 결과가 필요하다.")
        anchor_auc = float(
            decision["splits"]["final"]["all_strategies"]["anchor_best_auc"]
        )
    else:
        baseline_result = json.loads(args.baseline_result.read_text())
        if baseline_result["prediction_sha256"] != context.prediction_file_sha256:
            raise ValueError("선행 제거 결과와 현재 예측 파일의 해시가 다르다.")
        if tuple(baseline_result["removed_members"]) != baseline_removed:
            raise ValueError("선행 제거 결과와 요청한 제거 목록이 다르다.")
        anchor_auc = float(baseline_result["selected_best_auc"])
    candidates = [
        entry
        for entry in _candidate_order(decision)
        if entry["removed"] not in baseline_removed
    ]
    candidate_names = [entry["removed"] for entry in candidates]
    results: dict[str, dict[str, Any]] = {}

    if args.checkpoint.exists():
        saved = json.loads(args.checkpoint.read_text())
        if saved["prediction_sha256"] != context.prediction_file_sha256:
            raise ValueError("중간 저장과 현재 예측 파일의 해시가 다르다.")
        if saved["strategy"] != FIXED_STRATEGY:
            raise ValueError("중간 저장과 현재 고정 전략이 다르다.")
        if tuple(saved["baseline_removed_members"]) != baseline_removed:
            raise ValueError("중간 저장과 현재 선행 제거 목록이 다르다.")
        if [entry["removed"] for entry in saved["candidates"]] != candidate_names:
            raise ValueError("중간 저장과 현재 후보 순서가 다르다.")
        results.update(saved["results"])

    if args.seed_result is not None:
        if baseline_removed:
            raise ValueError("선행 제거 뒤 평가는 35개 기준 결과를 재사용할 수 없다.")
        seed = json.loads(args.seed_result.read_text())
        if seed["prediction_sha256"] != context.prediction_file_sha256:
            raise ValueError("재사용 결과와 현재 예측 파일의 해시가 다르다.")
        if seed["strategy"] != FIXED_STRATEGY:
            raise ValueError("재사용 결과와 현재 고정 전략이 다르다.")
        for removed, result in seed["removals"].items():
            if removed in candidate_names:
                results.setdefault(removed, result)

    _write_json(
        args.checkpoint,
        _checkpoint_payload(
            context=context,
            anchor_auc=anchor_auc,
            baseline_removed_members=baseline_removed,
            working_member_count=len(working_members),
            candidates=candidates,
            results=results,
        ),
    )

    remaining = [name for name in candidate_names if name not in results]
    with StrategyEvaluator(context, jobs=args.jobs) as evaluator:
        for offset in range(0, len(remaining), args.chunk_size):
            if time.monotonic() - started >= args.deadline_seconds:
                raise TimeoutError("30분 상한에 도달해 새 묶음을 시작하지 않는다.")
            chunk = tuple(remaining[offset : offset + args.chunk_size])
            outcomes = _batch_single_removals(
                evaluator,
                working_members,
                excluded_fold=None,
                removal_candidates=chunk,
            )
            for removed, outcome in outcomes.items():
                results[removed] = {
                    "auc": float(outcome["auc"]),
                    "delta_vs_anchor": float(outcome["auc"] - anchor_auc),
                    "fold_auc": outcome["fold_auc"],
                }
            _write_json(
                args.checkpoint,
                _checkpoint_payload(
                    context=context,
                    anchor_auc=anchor_auc,
                    baseline_removed_members=baseline_removed,
                    working_member_count=len(working_members),
                    candidates=candidates,
                    results=results,
                ),
            )
            print(json.dumps({
                "completed": len(results),
                "total": len(candidate_names),
                "chunk": list(chunk),
            }, ensure_ascii=False, sort_keys=True), flush=True)

    ranked = sorted(
        (
            {
                "removed": name,
                "screen_delta": next(
                    entry["screen_delta"]
                    for entry in candidates
                    if entry["removed"] == name
                ),
                **results[name],
            }
            for name in candidate_names
        ),
        key=lambda entry: (-entry["delta_vs_anchor"], entry["removed"]),
    )
    payload = {
        "schema_version": 1,
        "issue": 346,
        "prediction_sha256": context.prediction_file_sha256,
        "strategy": FIXED_STRATEGY,
        "anchor_auc": anchor_auc,
        "baseline_removed_members": list(baseline_removed),
        "working_member_count": len(working_members),
        "candidate_count": len(candidate_names),
        "positive_exact_count": sum(
            entry["delta_vs_anchor"] > 0.0 for entry in ranked
        ),
        "ranked_results": ranked,
        "best": ranked[0],
        "elapsed_seconds": time.monotonic() - started,
    }
    sha256 = _write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256,
        "candidate_count": payload["candidate_count"],
        "positive_exact_count": payload["positive_exact_count"],
        "best": payload["best"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
