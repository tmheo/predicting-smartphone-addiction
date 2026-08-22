"""이슈 366의 일회성 빠른 단일 제거 평가기.

제거 탐색에서는 35개 기준 풀의 최선 전략 하나만 고정한다.
각 바깥쪽 학습 부분과 전체 OOF에서 구성원 하나씩 뺀 후보를 최대 5회
탐욕적으로 비교하고, 전체 OOF 최종 후보에만 등록 전략 19개를 다시 적용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from pipeline.pool_rereview import (
    StrategyEvaluator,
    _fit_outer_prediction,
    _refit_counts,
    load_inputs,
)


FIXED_STRATEGY = "shrunk_rank_logit_logistic"


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _batch_single_removals(
    evaluator: StrategyEvaluator,
    members: tuple[str, ...],
    excluded_fold: int | None,
) -> dict[str, dict[str, Any]]:
    requests = [
        (
            FIXED_STRATEGY,
            tuple(candidate for candidate in members if candidate != removed),
            excluded_fold,
            None,
            False,
        )
        for removed in members
    ]
    outcomes = evaluator._map(requests)  # 일회성 원형은 기존 일괄 실행 통로를 재사용한다.
    failures = [
        (removed, outcome["failure"])
        for removed, outcome in zip(members, outcomes, strict=True)
        if outcome["failure"] is not None
    ]
    if failures:
        raise RuntimeError(f"단일 제거 평가 실패: {failures}")
    evaluator.fits += sum(int(outcome["fits"]) for outcome in outcomes)
    evaluator.arm_evaluations += len(outcomes)
    return dict(zip(members, outcomes, strict=True))


def _greedy_split(
    evaluator: StrategyEvaluator,
    members: tuple[str, ...],
    excluded_fold: int | None,
    max_steps: int,
    deadline: float,
) -> dict[str, Any]:
    anchor = evaluator.evaluate_one(
        FIXED_STRATEGY,
        members,
        excluded_fold=excluded_fold,
    )
    working_members = members
    working_auc = anchor.auc
    working_fold_auc = anchor.fold_auc
    trajectory: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("2시간 실행 상한에 도달했다.")
        outcomes = _batch_single_removals(
            evaluator,
            working_members,
            excluded_fold,
        )
        ordered = sorted(
            outcomes.items(),
            key=lambda item: (-float(item[1]["auc"]), item[0]),
        )
        removed, best = ordered[0]
        delta = float(best["auc"] - working_auc)
        candidate_table = [
            {
                "removed": candidate,
                "auc": float(outcome["auc"]),
                "delta_vs_working": float(outcome["auc"] - working_auc),
                "fold_delta": {
                    key: float(outcome["fold_auc"][key] - working_fold_auc[key])
                    for key in sorted(working_fold_auc, key=int)
                },
            }
            for candidate, outcome in sorted(outcomes.items())
        ]
        accepted = delta > 0.0
        trajectory.append(
            {
                "step": step,
                "working_member_count": len(working_members),
                "selected_removal": removed,
                "selected_auc": float(best["auc"]),
                "delta_vs_working": delta,
                "accepted": accepted,
                "candidates": candidate_table,
            }
        )
        if not accepted:
            break
        working_members = tuple(
            candidate for candidate in working_members if candidate != removed
        )
        working_auc = float(best["auc"])
        working_fold_auc = {
            key: float(value) for key, value in best["fold_auc"].items()
        }

    return {
        "excluded_outer_fold": excluded_fold,
        "anchor_strategy": FIXED_STRATEGY,
        "anchor_auc": anchor.auc,
        "anchor_fold_auc": anchor.fold_auc,
        "selected_members": list(working_members),
        "selected_member_count": len(working_members),
        "selected_fixed_strategy_auc": working_auc,
        "delta_vs_anchor": float(working_auc - anchor.auc),
        "trajectory": trajectory,
    }


def _member_metadata(context: Any, removed_members: list[str]) -> list[dict[str, Any]]:
    lineage_by_member = {
        member: group_name
        for group_name, entry in context.ledger["lineage_groups"].items()
        for member in entry["members"]
    }
    perspectives_by_member = {
        member: [
            name
            for name, entry in context.ledger["information_perspectives"].items()
            if member in entry["members"]
        ]
        for member in context.members
    }
    refit_counts = _refit_counts(context.ledger)
    return [
        {
            "config": member,
            "lineage_group": lineage_by_member[member],
            "information_perspectives": perspectives_by_member[member],
            "full_refit_reduction": refit_counts[member],
        }
        for member in removed_members
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    context = load_inputs(args.predictions)
    split_results: dict[str, dict[str, Any]] = {}

    with StrategyEvaluator(context, jobs=args.jobs) as evaluator:
        for fold in range(5):
            split = _greedy_split(
                evaluator,
                context.members,
                excluded_fold=fold,
                max_steps=args.max_steps,
                deadline=deadline,
            )
            selected_members = tuple(split["selected_members"])
            selected_prediction = _fit_outer_prediction(
                context,
                FIXED_STRATEGY,
                selected_members,
                held_out_fold=fold,
            )
            anchor_prediction = _fit_outer_prediction(
                context,
                FIXED_STRATEGY,
                context.members,
                held_out_fold=fold,
            )
            held_out = context.folds.to_numpy() == fold
            labels = context.labels.to_numpy()[held_out]
            selected_auc = float(roc_auc_score(labels, selected_prediction))
            anchor_auc = float(roc_auc_score(labels, anchor_prediction))
            split["held_out"] = {
                "selected_auc": selected_auc,
                "anchor_auc": anchor_auc,
                "delta": selected_auc - anchor_auc,
                "winner": (
                    "selected"
                    if selected_auc > anchor_auc
                    else "anchor"
                    if selected_auc < anchor_auc
                    else "tie"
                ),
            }
            split_results[f"outer-{fold}"] = split

        final = _greedy_split(
            evaluator,
            context.members,
            excluded_fold=None,
            max_steps=args.max_steps,
            deadline=deadline,
        )
        selected_members = tuple(final["selected_members"])
        anchor_all = evaluator.evaluate(context.members, excluded_fold=None)
        selected_all = evaluator.evaluate(selected_members, excluded_fold=None)
        final["all_strategies"] = {
            "anchor_best_strategy": anchor_all.best_strategy,
            "anchor_best_auc": anchor_all.best_auc,
            "anchor_strategy_auc": anchor_all.strategy_auc,
            "selected_best_strategy": selected_all.best_strategy,
            "selected_best_auc": selected_all.best_auc,
            "selected_strategy_auc": selected_all.strategy_auc,
            "delta_vs_anchor": selected_all.best_auc - anchor_all.best_auc,
        }
        split_results["final"] = final
        fits = evaluator.fits
        arm_evaluations = evaluator.arm_evaluations

    removed_members = [
        member for member in context.members if member not in set(final["selected_members"])
    ]
    held_out = [split_results[f"outer-{fold}"]["held_out"] for fold in range(5)]
    wins = sum(item["winner"] == "selected" for item in held_out)
    full_improved = final["all_strategies"]["delta_vs_anchor"] > 0.0
    payload = {
        "schema_version": 1,
        "issue": 366,
        "question": "고정 결합 전략의 최대 5회 단일 제거가 35개 기준보다 높은 풀을 찾는가?",
        "input": {
            "prediction_path": str(args.predictions.resolve()),
            "prediction_sha256": context.prediction_file_sha256,
            "rows": len(context.predictions),
            "members": len(context.members),
        },
        "execution": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "fixed_strategy": FIXED_STRATEGY,
            "max_steps": args.max_steps,
            "jobs": args.jobs,
            "deadline_seconds": args.deadline_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "strategy_fits": fits,
            "pool_arm_evaluations": arm_evaluations,
        },
        "splits": split_results,
        "final_removed_members": _member_metadata(context, removed_members),
        "validation": {
            "held_out_wins": wins,
            "held_out_anchor_wins": sum(item["winner"] == "anchor" for item in held_out),
            "held_out_ties": sum(item["winner"] == "tie" for item in held_out),
            "requires_wins": 3,
            "full_oof_improved": full_improved,
            "accepted": full_improved and wins >= 3,
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--deadline-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    payload = run(args)
    data = _json_bytes(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(json.dumps({
        "output": str(args.output),
        "sha256": hashlib.sha256(data).hexdigest(),
        "validation": payload["validation"],
        "execution": payload["execution"],
        "final": payload["splits"]["final"]["all_strategies"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
