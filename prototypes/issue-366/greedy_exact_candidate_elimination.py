"""사전 선별 후보를 양의 개선이 사라질 때까지 조건부 정확 제거한다."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pipeline.pool_rereview import StrategyEvaluator, load_inputs

from fast_single_removal import FIXED_STRATEGY, _batch_single_removals, _write_json


def _screened_candidates(decision: dict[str, Any]) -> list[str]:
    screened = decision["splits"]["final"]["trajectory"][0]["screen_candidates"]
    return [
        entry["removed"]
        for entry in sorted(
            screened,
            key=lambda entry: (-float(entry["delta_vs_working"]), entry["removed"]),
        )
        if float(entry["delta_vs_working"]) > 0.0
    ]


def _save_checkpoint(
    path: Path,
    *,
    context: Any,
    initial_removed: tuple[str, ...],
    candidate_names: list[str],
    current_removed: list[str],
    current_auc: float,
    remaining: list[str],
    rounds: list[dict[str, Any]],
    active_round: dict[str, Any] | None,
    status: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "issue": 346,
            "status": status,
            "prediction_sha256": context.prediction_file_sha256,
            "fixed_strategy": FIXED_STRATEGY,
            "initial_removed_members": list(initial_removed),
            "candidate_names": candidate_names,
            "current_removed_members": current_removed,
            "current_fixed_strategy_auc": current_auc,
            "remaining_candidates": remaining,
            "rounds": rounds,
            "active_round": active_round,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--screen-decision", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--initial-remove", action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--deadline-seconds", type=float, default=10800.0)
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("묶음 크기는 1 이상이어야 한다.")

    started = time.monotonic()
    deadline = started + args.deadline_seconds
    context = load_inputs(args.predictions)
    decision = json.loads(args.screen_decision.read_text())
    baseline = json.loads(args.baseline_result.read_text())
    if context.prediction_file_sha256 != decision["input"]["prediction_sha256"]:
        raise ValueError("선별 판정과 현재 예측 파일의 해시가 다르다.")
    if context.prediction_file_sha256 != baseline["prediction_sha256"]:
        raise ValueError("시작 풀 결과와 현재 예측 파일의 해시가 다르다.")

    initial_removed = tuple(args.initial_remove)
    if tuple(baseline["removed_members"]) != initial_removed:
        raise ValueError("시작 풀 결과와 요청한 선행 제거 목록이 다르다.")
    candidate_names = [
        name for name in _screened_candidates(decision) if name not in initial_removed
    ]

    if args.checkpoint.exists():
        saved = json.loads(args.checkpoint.read_text())
        if saved["prediction_sha256"] != context.prediction_file_sha256:
            raise ValueError("중간 저장과 현재 예측 파일의 해시가 다르다.")
        if tuple(saved["initial_removed_members"]) != initial_removed:
            raise ValueError("중간 저장과 현재 선행 제거 목록이 다르다.")
        if saved["candidate_names"] != candidate_names:
            raise ValueError("중간 저장과 현재 후보 목록이 다르다.")
        current_removed = list(saved["current_removed_members"])
        current_auc = float(saved["current_fixed_strategy_auc"])
        remaining = list(saved["remaining_candidates"])
        rounds = list(saved["rounds"])
        active_round = saved["active_round"]
    else:
        current_removed = list(initial_removed)
        current_auc = float(baseline["selected_best_auc"])
        remaining = list(candidate_names)
        rounds: list[dict[str, Any]] = []
        active_round: dict[str, Any] | None = None
        _save_checkpoint(
            args.checkpoint,
            context=context,
            initial_removed=initial_removed,
            candidate_names=candidate_names,
            current_removed=current_removed,
            current_auc=current_auc,
            remaining=remaining,
            rounds=rounds,
            active_round=active_round,
            status="in_progress",
        )

    converged = False
    with StrategyEvaluator(context, jobs=args.jobs) as evaluator:
        while remaining:
            if active_round is None:
                active_round = {
                    "round": len(rounds) + 1,
                    "anchor_auc": current_auc,
                    "anchor_removed_members": list(current_removed),
                    "results": {},
                }
                _save_checkpoint(
                    args.checkpoint,
                    context=context,
                    initial_removed=initial_removed,
                    candidate_names=candidate_names,
                    current_removed=current_removed,
                    current_auc=current_auc,
                    remaining=remaining,
                    rounds=rounds,
                    active_round=active_round,
                    status="in_progress",
                )

            pending = [
                name for name in remaining if name not in active_round["results"]
            ]
            current_members = tuple(
                member for member in context.members if member not in current_removed
            )
            for offset in range(0, len(pending), args.chunk_size):
                if time.monotonic() >= deadline:
                    _save_checkpoint(
                        args.checkpoint,
                        context=context,
                        initial_removed=initial_removed,
                        candidate_names=candidate_names,
                        current_removed=current_removed,
                        current_auc=current_auc,
                        remaining=remaining,
                        rounds=rounds,
                        active_round=active_round,
                        status="checkpointed",
                    )
                    raise TimeoutError("실행 상한에 도달해 새 묶음을 시작하지 않는다.")
                chunk = tuple(pending[offset : offset + args.chunk_size])
                outcomes = _batch_single_removals(
                    evaluator,
                    current_members,
                    excluded_fold=None,
                    removal_candidates=chunk,
                )
                for removed, outcome in outcomes.items():
                    active_round["results"][removed] = {
                        "auc": float(outcome["auc"]),
                        "delta_vs_anchor": float(outcome["auc"] - current_auc),
                        "fold_auc": outcome["fold_auc"],
                    }
                _save_checkpoint(
                    args.checkpoint,
                    context=context,
                    initial_removed=initial_removed,
                    candidate_names=candidate_names,
                    current_removed=current_removed,
                    current_auc=current_auc,
                    remaining=remaining,
                    rounds=rounds,
                    active_round=active_round,
                    status="in_progress",
                )
                print(json.dumps({
                    "round": active_round["round"],
                    "completed": len(active_round["results"]),
                    "total": len(remaining),
                    "chunk": list(chunk),
                }, ensure_ascii=False, sort_keys=True), flush=True)

            ranked = sorted(
                (
                    {"removed": name, **result}
                    for name, result in active_round["results"].items()
                ),
                key=lambda entry: (-entry["delta_vs_anchor"], entry["removed"]),
            )
            best = ranked[0]
            accepted = float(best["delta_vs_anchor"]) > 0.0
            round_result = {
                "round": active_round["round"],
                "anchor_auc": current_auc,
                "anchor_removed_members": list(current_removed),
                "ranked_results": ranked,
                "selected_removal": best["removed"] if accepted else None,
                "selected_auc": best["auc"] if accepted else current_auc,
                "selected_delta": best["delta_vs_anchor"] if accepted else 0.0,
                "accepted": accepted,
            }
            rounds.append(round_result)
            active_round = None
            if not accepted:
                converged = True
                _save_checkpoint(
                    args.checkpoint,
                    context=context,
                    initial_removed=initial_removed,
                    candidate_names=candidate_names,
                    current_removed=current_removed,
                    current_auc=current_auc,
                    remaining=remaining,
                    rounds=rounds,
                    active_round=active_round,
                    status="converged",
                )
                print(json.dumps({
                    "round": round_result["round"],
                    "accepted": False,
                    "best_candidate": best,
                }, ensure_ascii=False, sort_keys=True), flush=True)
                break

            current_removed.append(best["removed"])
            current_auc = float(best["auc"])
            remaining.remove(best["removed"])
            _save_checkpoint(
                args.checkpoint,
                context=context,
                initial_removed=initial_removed,
                candidate_names=candidate_names,
                current_removed=current_removed,
                current_auc=current_auc,
                remaining=remaining,
                rounds=rounds,
                active_round=active_round,
                status="in_progress",
            )
            print(json.dumps({
                "round": round_result["round"],
                "accepted": True,
                "removed": best["removed"],
                "auc": best["auc"],
                "delta": best["delta_vs_anchor"],
                "remaining": len(remaining),
            }, ensure_ascii=False, sort_keys=True), flush=True)

        if not remaining:
            converged = True
        final_members = tuple(
            member for member in context.members if member not in current_removed
        )
        registered = evaluator.evaluate(final_members, excluded_fold=None)

    original_anchor_auc = float(
        decision["splits"]["final"]["all_strategies"]["anchor_best_auc"]
    )
    payload = {
        "schema_version": 1,
        "issue": 346,
        "prediction_sha256": context.prediction_file_sha256,
        "candidate_names": candidate_names,
        "initial_removed_members": list(initial_removed),
        "rounds": rounds,
        "converged": converged,
        "final_removed_members": current_removed,
        "final_members": list(final_members),
        "final_member_count": len(final_members),
        "fixed_strategy": FIXED_STRATEGY,
        "final_fixed_strategy_auc": current_auc,
        "registered_strategy_count": len(registered.strategy_auc),
        "final_best_strategy": registered.best_strategy,
        "final_best_auc": registered.best_auc,
        "delta_vs_original_35": registered.best_auc - original_anchor_auc,
        "strategy_auc": registered.strategy_auc,
        "strategy_fold_auc": registered.strategy_fold_auc,
        "elapsed_seconds": time.monotonic() - started,
    }
    sha256 = _write_json(args.output, payload)
    _save_checkpoint(
        args.checkpoint,
        context=context,
        initial_removed=initial_removed,
        candidate_names=candidate_names,
        current_removed=current_removed,
        current_auc=current_auc,
        remaining=remaining,
        rounds=rounds,
        active_round=None,
        status="completed",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256,
        "round_count": len(rounds),
        "final_removed_members": current_removed,
        "final_member_count": len(final_members),
        "final_best_strategy": registered.best_strategy,
        "final_best_auc": registered.best_auc,
        "delta_vs_original_35": payload["delta_vs_original_35"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
