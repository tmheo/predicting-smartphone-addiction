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
from collections.abc import Callable
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
SCREEN_STRATEGY = "rank_mean"


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> str:
    data = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_bytes(data)
    temporary_path.replace(path)
    return hashlib.sha256(data).hexdigest()


def _batch_single_removals(
    evaluator: StrategyEvaluator,
    members: tuple[str, ...],
    excluded_fold: int | None,
    *,
    strategy: str = FIXED_STRATEGY,
    removal_candidates: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    removed_members = removal_candidates or members
    requests = [
        (
            strategy,
            tuple(candidate for candidate in members if candidate != removed),
            excluded_fold,
            None,
            False,
        )
        for removed in removed_members
    ]
    outcomes = evaluator._map(requests)  # 일회성 원형은 기존 일괄 실행 통로를 재사용한다.
    failures = [
        (removed, outcome["failure"])
        for removed, outcome in zip(removed_members, outcomes, strict=True)
        if outcome["failure"] is not None
    ]
    if failures:
        raise RuntimeError(f"단일 제거 평가 실패: {failures}")
    evaluator.fits += sum(int(outcome["fits"]) for outcome in outcomes)
    evaluator.arm_evaluations += len(outcomes)
    return dict(zip(removed_members, outcomes, strict=True))


def _greedy_split(
    evaluator: StrategyEvaluator,
    members: tuple[str, ...],
    excluded_fold: int | None,
    max_steps: int,
    deadline: float,
    screen_limit: int,
    resume: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if resume is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("실행 상한에 도달해 새 분할을 시작하지 않는다.")
        anchor = evaluator.evaluate_one(
            FIXED_STRATEGY,
            members,
            excluded_fold=excluded_fold,
        )
        anchor_auc = anchor.auc
        anchor_fold_auc = anchor.fold_auc
        working_members = members
        working_auc = anchor.auc
        working_fold_auc = anchor.fold_auc
        trajectory: list[dict[str, Any]] = []
    else:
        anchor_auc = float(resume["anchor_auc"])
        anchor_fold_auc = {
            key: float(value) for key, value in resume["anchor_fold_auc"].items()
        }
        working_members = tuple(resume["selected_members"])
        working_auc = float(resume["selected_fixed_strategy_auc"])
        accepted_steps = [
            step for step in resume["trajectory"] if step["accepted"]
        ]
        if accepted_steps:
            working_fold_auc = {
                key: float(value)
                for key, value in accepted_steps[-1]["selected_fold_auc"].items()
            }
        else:
            working_fold_auc = anchor_fold_auc
        trajectory = list(resume["trajectory"])

    def snapshot() -> dict[str, Any]:
        return {
            "excluded_outer_fold": excluded_fold,
            "anchor_strategy": FIXED_STRATEGY,
            "anchor_auc": anchor_auc,
            "anchor_fold_auc": anchor_fold_auc,
            "selected_members": list(working_members),
            "selected_member_count": len(working_members),
            "selected_fixed_strategy_auc": working_auc,
            "delta_vs_anchor": float(working_auc - anchor_auc),
            "trajectory": trajectory,
        }

    if checkpoint is not None:
        checkpoint(snapshot())
    if trajectory and not trajectory[-1]["accepted"]:
        return snapshot()

    for step in range(len(trajectory) + 1, max_steps + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("실행 상한에 도달해 새 제거 단계를 시작하지 않는다.")
        screen_anchor = evaluator.evaluate_one(
            SCREEN_STRATEGY,
            working_members,
            excluded_fold=excluded_fold,
        )
        screen_outcomes = _batch_single_removals(
            evaluator,
            working_members,
            excluded_fold,
            strategy=SCREEN_STRATEGY,
        )
        screened_removals = tuple(
            removed
            for removed, _outcome in sorted(
                screen_outcomes.items(),
                key=lambda item: (-float(item[1]["auc"]), item[0]),
            )[:screen_limit]
        )
        outcomes = _batch_single_removals(
            evaluator,
            working_members,
            excluded_fold,
            removal_candidates=screened_removals,
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
        screen_table = [
            {
                "removed": candidate,
                "auc": float(outcome["auc"]),
                "delta_vs_working": float(outcome["auc"] - screen_anchor.auc),
                "selected_for_exact_evaluation": candidate in screened_removals,
            }
            for candidate, outcome in sorted(screen_outcomes.items())
        ]
        accepted = delta > 0.0
        trajectory.append(
            {
                "step": step,
                "working_member_count": len(working_members),
                "selected_removal": removed,
                "selected_auc": float(best["auc"]),
                "selected_fold_auc": {
                    key: float(value) for key, value in best["fold_auc"].items()
                },
                "delta_vs_working": delta,
                "accepted": accepted,
                "screen_strategy": SCREEN_STRATEGY,
                "screen_limit": screen_limit,
                "screen_candidates": screen_table,
                "candidates": candidate_table,
            }
        )
        if accepted:
            working_members = tuple(
                candidate for candidate in working_members if candidate != removed
            )
            working_auc = float(best["auc"])
            working_fold_auc = {
                key: float(value) for key, value in best["fold_auc"].items()
            }
        if checkpoint is not None:
            checkpoint(snapshot())
        if not accepted:
            break

    return snapshot()


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
    checkpoint_path = args.checkpoint or args.output.with_name(
        f"{args.output.name}.checkpoint"
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    split_results: dict[str, dict[str, Any]] = {}
    resumed_from_checkpoint = False
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text())
        saved_input = saved["input"]
        saved_execution = saved["execution"]
        if saved_input["prediction_sha256"] != context.prediction_file_sha256:
            raise ValueError("중간 저장 입력 해시가 현재 예측 파일과 다르다.")
        if saved_execution["fixed_strategy"] != FIXED_STRATEGY:
            raise ValueError("중간 저장 결합 전략이 현재 실행과 다르다.")
        if int(saved_execution["max_steps"]) != args.max_steps:
            raise ValueError("중간 저장 최대 제거 횟수가 현재 실행과 다르다.")
        if int(saved_execution["screen_limit"]) != args.screen_limit:
            raise ValueError("중간 저장 정밀 평가 후보 수가 현재 실행과 다르다.")
        split_results = saved["splits"]
        resumed_from_checkpoint = True

    with StrategyEvaluator(context, jobs=args.jobs) as evaluator:
        def save_checkpoint() -> None:
            _write_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "issue": 366,
                    "status": "in_progress",
                    "input": {
                        "prediction_path": str(args.predictions.resolve()),
                        "prediction_sha256": context.prediction_file_sha256,
                        "rows": len(context.predictions),
                        "members": len(context.members),
                    },
                    "execution": {
                        "git_commit": git_commit,
                        "fixed_strategy": FIXED_STRATEGY,
                        "max_steps": args.max_steps,
                        "screen_strategy": SCREEN_STRATEGY,
                        "screen_limit": args.screen_limit,
                        "jobs": args.jobs,
                        "deadline_seconds": args.deadline_seconds,
                        "elapsed_seconds": time.monotonic() - started,
                        "strategy_fits_this_process": evaluator.fits,
                        "pool_arm_evaluations_this_process": (
                            evaluator.arm_evaluations
                        ),
                    },
                    "splits": split_results,
                },
            )

        for fold in range(5):
            split_key = f"outer-{fold}"
            saved_split = split_results.get(split_key)
            if saved_split is not None and "held_out" in saved_split:
                continue

            def save_outer_partial(
                partial: dict[str, Any], *, key: str = split_key
            ) -> None:
                split_results[key] = partial
                save_checkpoint()

            split = _greedy_split(
                evaluator,
                context.members,
                excluded_fold=fold,
                max_steps=args.max_steps,
                deadline=deadline,
                screen_limit=args.screen_limit,
                resume=saved_split,
                checkpoint=save_outer_partial,
            )
            if time.monotonic() >= deadline:
                save_checkpoint()
                raise TimeoutError(
                    "실행 상한에 도달해 바깥쪽 채점 계산을 시작하지 않는다."
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
            split_results[split_key] = split
            save_checkpoint()

        saved_final = split_results.get("final")

        def save_final_partial(partial: dict[str, Any]) -> None:
            split_results["final"] = partial
            save_checkpoint()

        final_greedy_complete = bool(
            saved_final is not None
            and (
                len(saved_final["trajectory"]) >= args.max_steps
                or (
                    saved_final["trajectory"]
                    and not saved_final["trajectory"][-1]["accepted"]
                )
            )
        )
        if saved_final is not None and (
            "all_strategies" in saved_final or final_greedy_complete
        ):
            final = saved_final
        else:
            final = _greedy_split(
                evaluator,
                context.members,
                excluded_fold=None,
                max_steps=args.max_steps,
                deadline=deadline,
                screen_limit=args.screen_limit,
                resume=saved_final,
                checkpoint=save_final_partial,
            )
        selected_members = tuple(final["selected_members"])
        strategy_recheck = final.setdefault("strategy_recheck_checkpoint", {})
        if "anchor" not in strategy_recheck:
            if time.monotonic() >= deadline:
                save_final_partial(final)
                raise TimeoutError(
                    "실행 상한에 도달해 19개 전략 재평가를 시작하지 않는다."
                )
            anchor_all = evaluator.evaluate(context.members, excluded_fold=None)
            strategy_recheck["anchor"] = {
                "best_strategy": anchor_all.best_strategy,
                "best_auc": anchor_all.best_auc,
                "strategy_auc": anchor_all.strategy_auc,
            }
            save_final_partial(final)
        if "selected" not in strategy_recheck:
            if time.monotonic() >= deadline:
                save_final_partial(final)
                raise TimeoutError(
                    "실행 상한에 도달해 최종 후보 전략 재평가를 시작하지 않는다."
                )
            selected_all = evaluator.evaluate(selected_members, excluded_fold=None)
            strategy_recheck["selected"] = {
                "best_strategy": selected_all.best_strategy,
                "best_auc": selected_all.best_auc,
                "strategy_auc": selected_all.strategy_auc,
            }
            save_final_partial(final)
        anchor_result = strategy_recheck["anchor"]
        selected_result = strategy_recheck["selected"]
        final["all_strategies"] = {
            "anchor_best_strategy": anchor_result["best_strategy"],
            "anchor_best_auc": anchor_result["best_auc"],
            "anchor_strategy_auc": anchor_result["strategy_auc"],
            "selected_best_strategy": selected_result["best_strategy"],
            "selected_best_auc": selected_result["best_auc"],
            "selected_strategy_auc": selected_result["strategy_auc"],
            "delta_vs_anchor": (
                selected_result["best_auc"] - anchor_result["best_auc"]
            ),
        }
        del final["strategy_recheck_checkpoint"]
        split_results["final"] = final
        save_checkpoint()
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
            "git_commit": git_commit,
            "fixed_strategy": FIXED_STRATEGY,
            "max_steps": args.max_steps,
            "screen_strategy": SCREEN_STRATEGY,
            "screen_limit": args.screen_limit,
            "jobs": args.jobs,
            "deadline_seconds": args.deadline_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "strategy_fits": fits,
            "pool_arm_evaluations": arm_evaluations,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "checkpoint_path": str(checkpoint_path.resolve()),
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--screen-limit", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=10800.0)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or args.output.with_name(
        f"{args.output.name}.checkpoint"
    )
    try:
        payload = run(args)
    except TimeoutError as error:
        print(json.dumps({
            "status": "checkpointed",
            "checkpoint": str(checkpoint_path),
            "reason": str(error),
        }, ensure_ascii=False, sort_keys=True, indent=2))
        raise SystemExit(2) from error
    sha256 = _write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256,
        "validation": payload["validation"],
        "execution": payload["execution"],
        "final": payload["splits"]["final"]["all_strategies"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
