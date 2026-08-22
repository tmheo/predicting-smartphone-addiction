"""이슈 366의 분할 하나를 독립 실행하는 빠른 평가 작업자."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from pipeline.pool_rereview import StrategyEvaluator, _fit_outer_prediction, load_inputs

from fast_single_removal import (
    FIXED_STRATEGY,
    SCREEN_STRATEGY,
    _greedy_split,
    _write_json,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    context = load_inputs(args.predictions)
    split: dict[str, Any] | None = None
    if args.checkpoint.exists():
        saved = json.loads(args.checkpoint.read_text())
        if saved["input"]["prediction_sha256"] != context.prediction_file_sha256:
            raise ValueError("중간 저장 입력 해시가 현재 예측 파일과 다르다.")
        split = saved["split"]

    excluded_fold = None if args.fold == "final" else int(args.fold)
    with StrategyEvaluator(context, jobs=args.jobs) as evaluator:
        def checkpoint(partial: dict[str, Any]) -> None:
            nonlocal split
            split = partial
            _write_json(
                args.checkpoint,
                {
                    "schema_version": 1,
                    "issue": 366,
                    "status": "in_progress",
                    "input": {
                        "prediction_path": str(args.predictions.resolve()),
                        "prediction_sha256": context.prediction_file_sha256,
                    },
                    "execution": {
                        "git_commit": _git_commit(),
                        "fold": args.fold,
                        "fixed_strategy": FIXED_STRATEGY,
                        "screen_strategy": SCREEN_STRATEGY,
                        "screen_limit": args.screen_limit,
                        "max_steps": args.max_steps,
                        "deadline_seconds": args.deadline_seconds,
                        "elapsed_seconds": time.monotonic() - started,
                        "strategy_fits_this_process": evaluator.fits,
                    },
                    "split": split,
                },
            )

        greedy_complete = bool(
            split is not None
            and (
                len(split["trajectory"]) >= args.max_steps
                or (split["trajectory"] and not split["trajectory"][-1]["accepted"])
            )
        )
        if not greedy_complete:
            split = _greedy_split(
                evaluator,
                context.members,
                excluded_fold=excluded_fold,
                max_steps=args.max_steps,
                deadline=deadline,
                screen_limit=args.screen_limit,
                resume=split,
                checkpoint=checkpoint,
            )
        assert split is not None
        selected_members = tuple(split["selected_members"])

        if excluded_fold is not None and "held_out" not in split:
            if time.monotonic() >= deadline:
                checkpoint(split)
                raise TimeoutError("실행 상한에 도달해 바깥쪽 채점을 시작하지 않는다.")
            selected_prediction = _fit_outer_prediction(
                context,
                FIXED_STRATEGY,
                selected_members,
                held_out_fold=excluded_fold,
            )
            anchor_prediction = _fit_outer_prediction(
                context,
                FIXED_STRATEGY,
                context.members,
                held_out_fold=excluded_fold,
            )
            held_out = context.folds.to_numpy() == excluded_fold
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
            checkpoint(split)

        if excluded_fold is None and "all_strategies" not in split:
            recheck = split.setdefault("strategy_recheck_checkpoint", {})
            if "anchor" not in recheck:
                if time.monotonic() >= deadline:
                    checkpoint(split)
                    raise TimeoutError("실행 상한에 도달해 앵커 전략 재평가를 시작하지 않는다.")
                result = evaluator.evaluate(context.members, excluded_fold=None)
                recheck["anchor"] = {
                    "best_strategy": result.best_strategy,
                    "best_auc": result.best_auc,
                    "strategy_auc": result.strategy_auc,
                }
                checkpoint(split)
            if "selected" not in recheck:
                if time.monotonic() >= deadline:
                    checkpoint(split)
                    raise TimeoutError("실행 상한에 도달해 후보 전략 재평가를 시작하지 않는다.")
                result = evaluator.evaluate(selected_members, excluded_fold=None)
                recheck["selected"] = {
                    "best_strategy": result.best_strategy,
                    "best_auc": result.best_auc,
                    "strategy_auc": result.strategy_auc,
                }
                checkpoint(split)
            anchor = recheck["anchor"]
            selected = recheck["selected"]
            split["all_strategies"] = {
                "anchor_best_strategy": anchor["best_strategy"],
                "anchor_best_auc": anchor["best_auc"],
                "anchor_strategy_auc": anchor["strategy_auc"],
                "selected_best_strategy": selected["best_strategy"],
                "selected_best_auc": selected["best_auc"],
                "selected_strategy_auc": selected["strategy_auc"],
                "delta_vs_anchor": selected["best_auc"] - anchor["best_auc"],
            }
            del split["strategy_recheck_checkpoint"]
            checkpoint(split)

        return {
            "schema_version": 1,
            "issue": 366,
            "input": {
                "prediction_path": str(args.predictions.resolve()),
                "prediction_sha256": context.prediction_file_sha256,
            },
            "execution": {
                "git_commit": _git_commit(),
                "fold": args.fold,
                "fixed_strategy": FIXED_STRATEGY,
                "screen_strategy": SCREEN_STRATEGY,
                "screen_limit": args.screen_limit,
                "max_steps": args.max_steps,
                "jobs": args.jobs,
                "deadline_seconds": args.deadline_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "strategy_fits": evaluator.fits,
            },
            "split": split,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", choices=("0", "1", "2", "3", "4", "final"), required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--screen-limit", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=5400.0)
    args = parser.parse_args()
    try:
        payload = run(args)
    except TimeoutError as error:
        print(json.dumps({
            "status": "checkpointed",
            "checkpoint": str(args.checkpoint),
            "reason": str(error),
        }, ensure_ascii=False, sort_keys=True, indent=2))
        raise SystemExit(2) from error
    sha256 = _write_json(args.output, payload)
    print(json.dumps({
        "status": "completed",
        "fold": args.fold,
        "output": str(args.output),
        "sha256": sha256,
        "elapsed_seconds": payload["execution"]["elapsed_seconds"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
