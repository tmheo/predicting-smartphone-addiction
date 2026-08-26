"""이슈 #337 값 좌표 관점 조건부 진단.

최종 선택 결합 전략을 고정한 뒤, 동결 풀과 지정 구성원을 뺀 풀을 같은 nested 절차로
평가한다(`evaluate` 모드). `contrast` 모드는 전체 풀 결과와 제거 풀 결과를 짝지어
전체 OOF 차이, 가중 OOF 차이, 바깥쪽 검증 분할 승패, 남은 구성원의 선택 빈도와
결합 계수 변화를 JSON으로 남긴다.

구성원을 소급 제거하는 관문이 아니라 조건부 기여를 설명하는 보조 판정이므로
후보 풀과 champion 장부는 읽기만 한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.data import ID, TRAIN_PATH, file_sha256, labels
from pipeline.ensemble import (
    MISSINGNESS_TEST_PATH,
    combiner_for_context,
    evaluate_nested,
    member_matrix,
    member_stats,
    missingness_bands,
)
from pipeline.judgment import FOLDS_PATH, missingness_reweighting
from pipeline.ledger import POOL_PATH, Pool
from pipeline.runs import MlflowRunStore


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def evaluate(strategy: str, excluded: list[str], output: Path) -> None:
    pool = Pool.load()
    known = {member.config for member in pool.members}
    unknown = sorted(set(excluded) - known)
    if unknown:
        sys.exit(f"후보 풀에 없는 제외 구성원: {', '.join(unknown)}")
    members = [
        (member.config, member.run_id)
        for member in pool.members
        if member.config not in set(excluded)
    ]
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    store = MlflowRunStore()
    band_of = missingness_bands(TRAIN_PATH, MISSINGNESS_TEST_PATH)
    reweighting = missingness_reweighting(TRAIN_PATH, MISSINGNESS_TEST_PATH)
    matrix = member_matrix(members, store, fold_of.index)
    combiner = combiner_for_context(strategy, fold_of=fold_of, band_of=band_of)
    evaluation = evaluate_nested(combiner, matrix, fold_of, y, reweighting)
    weighted = evaluation.weighted
    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "strategy": strategy,
        "excluded": list(excluded),
        "member_count": len(members),
        "members": [{"config": config, "run_id": run_id} for config, run_id in members],
        "input_sha256": {
            "pool": file_sha256(POOL_PATH),
            "folds": file_sha256(FOLDS_PATH),
            "train": file_sha256(TRAIN_PATH),
            "test": file_sha256(MISSINGNESS_TEST_PATH),
        },
        "nested_oof_auc": evaluation.nested_auc,
        "weighted_oof_auc": None if weighted is None else weighted.auc,
        "weighted_oof_sample": None
        if weighted is None
        else {
            "effective_sample_size": weighted.effective_sample_size,
            "effective_sample_fraction": weighted.effective_sample_fraction,
            "zero_weight_rows": weighted.zero_weight_rows,
            "test_only_pattern_count": weighted.test_only_pattern_count,
        },
        "elapsed_seconds": evaluation.elapsed_seconds,
        "outer_fold_auc": {str(o.fold): o.auc for o in evaluation.folds},
        "member_stats": [
            {
                "member": stat.member,
                "selected": stat.selected,
                "fold_total": stat.fold_total,
                "mean_weight": stat.mean_weight,
            }
            for stat in member_stats(evaluation)
        ],
        "outer_fold_weights": {
            str(o.fold): {member: float(w) for member, w in o.summary.items()}
            for o in evaluation.folds
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{strategy} 제외 {excluded or '없음'}: 구성원 {len(members)}, "
        f"nested {evaluation.nested_auc:.10f}, "
        f"가중 {'-' if weighted is None else f'{weighted.auc:.10f}'}, "
        f"{evaluation.elapsed_seconds:.0f}초 → {output}"
    )


def contrast(full_path: Path, ablation_paths: list[Path], output: Path) -> None:
    full = json.loads(full_path.read_text())
    full_stats = {s["member"]: s for s in full["member_stats"]}
    contrasts = []
    for path in ablation_paths:
        ablated = json.loads(path.read_text())
        assert ablated["strategy"] == full["strategy"], "전략이 다르다."
        assert ablated["input_sha256"] == full["input_sha256"], "입력 해시가 다르다."
        fold_delta = {
            fold: ablated["outer_fold_auc"][fold] - full["outer_fold_auc"][fold]
            for fold in full["outer_fold_auc"]
        }
        member_changes = []
        for stat in ablated["member_stats"]:
            before = full_stats[stat["member"]]
            member_changes.append(
                {
                    "member": stat["member"],
                    "selected_before": before["selected"],
                    "selected_after": stat["selected"],
                    "mean_weight_before": before["mean_weight"],
                    "mean_weight_after": stat["mean_weight"],
                    "mean_weight_delta": stat["mean_weight"] - before["mean_weight"],
                }
            )
        member_changes.sort(key=lambda c: -abs(c["mean_weight_delta"]))
        removed = [
            {
                "member": name,
                "selected_in_full_pool": full_stats[name]["selected"],
                "mean_weight_in_full_pool": full_stats[name]["mean_weight"],
            }
            for name in ablated["excluded"]
        ]
        contrasts.append(
            {
                "excluded": ablated["excluded"],
                "member_count": ablated["member_count"],
                "nested_oof_auc": ablated["nested_oof_auc"],
                "nested_oof_delta_vs_full": ablated["nested_oof_auc"] - full["nested_oof_auc"],
                "weighted_oof_auc": ablated["weighted_oof_auc"],
                "weighted_oof_delta_vs_full": None
                if ablated["weighted_oof_auc"] is None
                else ablated["weighted_oof_auc"] - full["weighted_oof_auc"],
                "outer_fold_auc": ablated["outer_fold_auc"],
                "outer_fold_delta_vs_full": fold_delta,
                "outer_fold_full_wins": sum(delta < 0 for delta in fold_delta.values()),
                "outer_fold_full_losses": sum(delta > 0 for delta in fold_delta.values()),
                "removed_members_in_full_pool": removed,
                "remaining_member_changes": member_changes,
                "source": str(path),
            }
        )
    payload = {
        "strategy": full["strategy"],
        "git_commit": full["git_commit"],
        "input_sha256": full["input_sha256"],
        "full_pool": {
            "member_count": full["member_count"],
            "nested_oof_auc": full["nested_oof_auc"],
            "weighted_oof_auc": full["weighted_oof_auc"],
            "outer_fold_auc": full["outer_fold_auc"],
            "source": str(full_path),
        },
        "contrasts": contrasts,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"전체 풀 {full['member_count']}: nested {full['nested_oof_auc']:.10f}, 가중 {full['weighted_oof_auc']:.10f}")
    for c in contrasts:
        print(
            f"- 제외 {c['excluded']}: nested {c['nested_oof_delta_vs_full']:+.10f}, "
            f"가중 {c['weighted_oof_delta_vs_full']:+.10f}, "
            f"전체 풀 승 {c['outer_fold_full_wins']}/5"
        )
    print(f"대조 산출물: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--strategy", required=True)
    ev.add_argument("--exclude", action="append", default=[])
    ev.add_argument("--output", type=Path, required=True)
    ct = sub.add_parser("contrast")
    ct.add_argument("--full", type=Path, required=True)
    ct.add_argument("--ablation", type=Path, action="append", required=True)
    ct.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.strategy, args.exclude, args.output)
    else:
        contrast(args.full, args.ablation, args.output)


if __name__ == "__main__":
    main()
