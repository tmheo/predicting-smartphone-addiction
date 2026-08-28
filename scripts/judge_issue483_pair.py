"""이슈 #483 2단계 진입 진단과 3단계 확정 짝비교 기록.

미포함판(기준)과 포함판(후보)을 같은 분할·난수로 짝지어 다음을 기록한다.

- 2단계 경로 1: 난수 42 OOF AUC가 기준보다 높은지(분할별 승수는 보조 기록).
- 2단계 경로 2: 진입 하한(champion − 0.01), 후보 풀 최근접 구성원 스피어만 순위 상관 < 0.998.
  세 번째 조건인 자체 풀 예비 포함 nested 기여는 앞 두 조건을 통과했을 때만
  ``pipeline.pool_judgment``로 별도 계산한다.
- 3단계: 후보가 3난수 실행이고 기준의 시드별 지표가 있으면 ADR 0001 계열 1 규칙
  (``judge_confirmation``)을 기준 실행을 champion 자리에 두고 적용한다.

사용법:
    uv run python scripts/judge_issue483_pair.py \
        --baseline-run <run_id> --candidate-run <run_id> \
        --tracking-uri sqlite:////abs/mlflow.db --pool-tracking-uri sqlite:////abs/main/mlflow.db \
        --baseline-seed42-auc 0.9693147 --out run-issue483/judgments/lookup.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET, TRAIN_PATH, file_sha256
from pipeline.judgment import (
    AUC_THRESHOLD,
    DUPLICATE_SPEARMAN,
    ENTRY_FLOOR_MARGIN,
    judge_confirmation,
    load_run_facts,
    spearman,
)
from pipeline.ledger import Champion, Pool
from pipeline.runs import MlflowRunStore


def _seed_oof(store: MlflowRunStore, run_id: str, seed: int) -> pd.Series:
    path = store._artifact(run_id, f"oof_seed_{seed}.parquet")  # noqa: SLF001 - 판정 전용 읽기
    frame = pd.read_parquet(path)
    return frame.set_index(ID)[["fold", "pred"]]


def _fold_aucs(oof: pd.DataFrame, y: pd.Series) -> dict[int, float]:
    out = {}
    for fold, part in oof.groupby("fold"):
        out[int(fold)] = float(roc_auc_score(y.reindex(part.index), part["pred"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--tracking-uri", required=True, help="후보 실행이 있는 MLflow")
    parser.add_argument(
        "--baseline-tracking-uri", default=None, help="기준 실행이 다른 MLflow에 있으면 지정한다"
    )
    parser.add_argument("--pool-tracking-uri", required=True, help="후보 풀 구성원 OOF가 있는 main MLflow")
    parser.add_argument(
        "--baseline-seed42-auc",
        type=float,
        default=None,
        help="기준 실행에 auc_oof_seed_42가 없거나 기록값을 쓸 때의 난수 42 OOF AUC",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    store = MlflowRunStore(args.tracking_uri)
    baseline_store = MlflowRunStore(args.baseline_tracking_uri or args.tracking_uri)
    pool_store = MlflowRunStore(args.pool_tracking_uri)
    baseline = load_run_facts(args.baseline_run, baseline_store)
    candidate = load_run_facts(args.candidate_run, store)
    champion = Champion.load()
    pool = Pool.load()
    y = pd.read_csv(TRAIN_PATH, usecols=[ID, TARGET]).set_index(ID)[TARGET]

    # 2단계 경로 1: 난수 42 짝비교
    cand42 = _seed_oof(store, candidate.run_id, 42)
    cand42_auc = float(roc_auc_score(y.reindex(cand42.index), cand42["pred"]))
    cand42_folds = _fold_aucs(cand42, y)
    if 42 in baseline.seed_aucs:
        base42 = _seed_oof(baseline_store, baseline.run_id, 42)
        base42_auc = float(roc_auc_score(y.reindex(base42.index), base42["pred"]))
        base42_folds = _fold_aucs(base42, y)
        base42_source = f"run {baseline.run_id}"
    else:
        if args.baseline_seed42_auc is None:
            raise SystemExit("기준 실행에 난수 42 OOF가 없다: --baseline-seed42-auc가 필요하다.")
        base42_auc = float(args.baseline_seed42_auc)
        base42_folds = {}
        base42_source = "recorded"
    if args.baseline_seed42_auc is not None and 42 in baseline.seed_aucs:
        base42_source += f" (기록값 {args.baseline_seed42_auc:.7f}도 함께 기록)"
    fold_wins_42 = sum(
        cand42_folds[f] > base42_folds[f] for f in sorted(base42_folds)
    ) if base42_folds else None
    path1 = {
        "baseline_seed42_auc": base42_auc,
        "baseline_seed42_source": base42_source,
        "baseline_seed42_auc_recorded": args.baseline_seed42_auc,
        "candidate_seed42_auc": cand42_auc,
        "delta": cand42_auc - base42_auc,
        "candidate_seed42_fold_aucs": cand42_folds,
        "baseline_seed42_fold_aucs": base42_folds,
        "fold_wins": fold_wins_42,
        "passed": cand42_auc > base42_auc,
    }
    if args.baseline_seed42_auc is not None:
        path1["delta_vs_recorded"] = cand42_auc - float(args.baseline_seed42_auc)
        path1["passed_vs_recorded"] = cand42_auc > float(args.baseline_seed42_auc)

    # 2단계 경로 2: 진입 하한과 중복(난수 42 OOF와 시드 평균본 둘 다 기록)
    floor = champion.oof_auc - ENTRY_FLOOR_MARGIN
    cand_oof = store.oof_of(candidate.run_id)
    correlations = {}
    for member in pool.members:
        pred = pool_store.oof_of(member.run_id).reindex(cand_oof.index)
        assert pred.notna().all(), member.run_id
        correlations[member.config] = {
            "run_id": member.run_id,
            "spearman_seed_avg": float(spearman(cand_oof, pred)),
            "spearman_seed42": float(spearman(cand42["pred"], pred.reindex(cand42.index))),
        }
    nearest = max(correlations, key=lambda k: correlations[k]["spearman_seed_avg"])
    nearest42 = max(correlations, key=lambda k: correlations[k]["spearman_seed42"])
    path2 = {
        "champion_run_id": champion.run_id,
        "champion_auc": champion.oof_auc,
        "entry_floor": floor,
        "candidate_seed42_auc": cand42_auc,
        "candidate_seed_avg_auc": candidate.auc_oof,
        "floor_ok_seed42": cand42_auc >= floor,
        "floor_ok_seed_avg": candidate.auc_oof >= floor,
        "nearest_member_seed42": {"config": nearest42, **correlations[nearest42]},
        "nearest_member_seed_avg": {"config": nearest, **correlations[nearest]},
        "duplicate_threshold": DUPLICATE_SPEARMAN,
        "duplicate_seed42": correlations[nearest42]["spearman_seed42"] >= DUPLICATE_SPEARMAN,
        "duplicate_seed_avg": correlations[nearest]["spearman_seed_avg"] >= DUPLICATE_SPEARMAN,
        "nested_contribution": "앞 두 조건 통과 시 pipeline.pool_judgment로 별도 계산",
    }
    path2["prerequisites_passed_seed42"] = path2["floor_ok_seed42"] and not path2["duplicate_seed42"]

    # 3단계: 기준 실행을 champion 자리에 둔 확정 짝비교
    confirmation = None
    if candidate.seeds == [42, 43, 44] and baseline.seeds == [42, 43, 44]:
        pseudo = Champion(
            run_id=baseline.run_id,
            oof_auc=baseline.auc_oof,
            seed_aucs=baseline.seed_aucs,
            fold_aucs=baseline.fold_aucs,
            config=baseline.experiment,
            features=baseline.features,
            git_commit=baseline.git_commit,
            adopted_at="",
            reason="이슈 #483 짝비교용 기준(장부 champion 아님)",
        )
        verdict = judge_confirmation(pseudo, candidate)
        confirmation = {
            "baseline_auc": verdict.champion_auc,
            "candidate_auc": verdict.challenger_auc,
            "delta": verdict.delta,
            "threshold": AUC_THRESHOLD,
            "auc_ok": verdict.auc_ok,
            "seed_comparisons": [dataclasses.asdict(c) for c in verdict.seed_comparisons],
            "seed_wins": verdict.seed_wins,
            "seed_ok": verdict.seed_ok,
            "fold_wins": verdict.fold_wins,
            "fold_total": verdict.fold_total,
            "boundary": verdict.boundary,
            "fold_ok": verdict.fold_ok,
            "canary_ok": verdict.canary.ok,
            "new_features": [dataclasses.asdict(c) for c in verdict.new_features.checks],
            "new_features_ok": verdict.new_features.ok,
            "placebo_gain": verdict.new_features.placebo_gain,
            "passed": verdict.passed,
        }

    stage2_passed = bool(path1["passed"] or path2["prerequisites_passed_seed42"])
    result = {
        "schema_version": 1,
        "issue": 483,
        "baseline": {
            "run_id": baseline.run_id,
            "experiment": baseline.experiment,
            "seeds": baseline.seeds,
            "auc_oof": baseline.auc_oof,
            "seed_aucs": baseline.seed_aucs,
            "git_commit": baseline.git_commit,
        },
        "candidate": {
            "run_id": candidate.run_id,
            "experiment": candidate.experiment,
            "seeds": candidate.seeds,
            "auc_oof": candidate.auc_oof,
            "seed_aucs": candidate.seed_aucs,
            "fold_aucs": candidate.fold_aucs,
            "git_commit": candidate.git_commit,
            "git_dirty": candidate.git_dirty,
        },
        "folds_sha256": file_sha256(Path("artifacts/folds.parquet")),
        "stage2_path1_seed42_pairwise": path1,
        "stage2_path2_entry_and_duplicate": path2,
        "stage2_eligible_for_stage3": stage2_passed,
        "stage3_confirmation_vs_baseline": confirmation,
        "pool_size": len(pool.members),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {
                "path1_delta": path1["delta"],
                "path1_passed": path1["passed"],
                "floor_ok": path2["floor_ok_seed42"],
                "nearest_seed42": path2["nearest_member_seed42"]["config"],
                "spearman_seed42": path2["nearest_member_seed42"]["spearman_seed42"],
                "stage2_eligible": stage2_passed,
                "stage3_passed": None if confirmation is None else confirmation["passed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
