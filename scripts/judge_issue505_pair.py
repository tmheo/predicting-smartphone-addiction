"""이슈 #505 정확값 선형 OOF 로짓 초기 점수 후보의 짝비교 기록.

같은 커밋, 같은 분할, 같은 난수로 실행한 기준(exp110)과 후보(exp209)를 짝지어 다음을 남긴다.

- 난수 42 짝비교: 전체와 분할별 OOF AUC 차이, 분할 승수.
- 기록된 기준 실행과의 재현 대조: 로컬 재실행 기준이 기록값과 얼마나 다른지.
- 1단 계보: 후보 실행의 ``initial_score_evidence.json``에서 분할별 내부 OOF AUC,
  검증 1단 AUC, 초기 로짓 범위, 해시.
- 계열 2 진입 진단: 진입 하한(champion − 0.01)과 후보 풀 최근접 구성원 스피어만 순위 상관.
  세 번째 조건인 nested 기여는 앞 두 조건을 통과했을 때만 ``pipeline.pool_judgment``로
  별도 계산한다.
- 확정 대조: 후보가 3난수 실행이고 기준도 3난수면 ADR 0001 계열 1 규칙을 기준을
  champion 자리에 두고 적용한다.

사용법:
    uv run python scripts/judge_issue505_pair.py \
        --baseline-run <run_id> --candidate-run <run_id> \
        --recorded-baseline-run <run_id> \
        --out artifacts/issue505-lr-init-score-screen.json
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
from pipeline.runs import TRACKING_URI, MlflowRunStore


def _seed_oof(store: MlflowRunStore, run_id: str, seed: int) -> pd.DataFrame:
    path = store._artifact(run_id, f"oof_seed_{seed}.parquet")  # noqa: SLF001 - 판정 전용 읽기
    return pd.read_parquet(path).set_index(ID)[["fold", "pred"]]


def _fold_aucs(oof: pd.DataFrame, y: pd.Series) -> dict[int, float]:
    return {
        int(fold): float(roc_auc_score(y.reindex(part.index), part["pred"]))
        for fold, part in oof.groupby("fold")
    }


def _pairwise(
    baseline: pd.DataFrame, candidate: pd.DataFrame, y: pd.Series
) -> dict[str, object]:
    if not baseline.index.equals(candidate.index):
        candidate = candidate.reindex(baseline.index)
        assert candidate["pred"].notna().all(), "후보 OOF 행이 기준과 다르다."
    assert (baseline["fold"].to_numpy() == candidate["fold"].to_numpy()).all(), (
        "기준과 후보의 분할 배정이 다르다."
    )
    base_auc = float(roc_auc_score(y.reindex(baseline.index), baseline["pred"]))
    cand_auc = float(roc_auc_score(y.reindex(candidate.index), candidate["pred"]))
    base_folds = _fold_aucs(baseline, y)
    cand_folds = _fold_aucs(candidate, y)
    fold_deltas = {fold: cand_folds[fold] - base_folds[fold] for fold in sorted(base_folds)}
    return {
        "baseline_auc": base_auc,
        "candidate_auc": cand_auc,
        "delta": cand_auc - base_auc,
        "baseline_fold_aucs": base_folds,
        "candidate_fold_aucs": cand_folds,
        "fold_deltas": fold_deltas,
        "fold_wins": int(sum(delta > 0 for delta in fold_deltas.values())),
        "fold_total": len(fold_deltas),
        "spearman_baseline_candidate": float(spearman(baseline["pred"], candidate["pred"])),
        "passed": cand_auc > base_auc,
    }


def _first_stage_evidence(store: MlflowRunStore, run_id: str) -> dict[str, object]:
    path = store._artifact(run_id, "initial_score_evidence.json")  # noqa: SLF001 - 판정 전용 읽기
    payload = json.loads(Path(path).read_text())
    entries = payload["entries"]
    by_seed: dict[int, list[dict[str, object]]] = {}
    for entry in entries:
        by_seed.setdefault(int(entry["seed"]), []).append(entry)
    summary = {}
    for seed, items in sorted(by_seed.items()):
        items = sorted(items, key=lambda item: item["outer_fold"])
        summary[seed] = {
            "outer_folds": [item["outer_fold"] for item in items],
            "inner_oof_auc": [item["inner_oof_auc"] for item in items],
            "validation_first_stage_auc": [item["validation_first_stage_auc"] for item in items],
            "training_logit_range": [item["logit_range"]["training"] for item in items],
            "validation_logit_range": [item["logit_range"]["validation"] for item in items],
            "test_logit_range": [item["logit_range"]["test"] for item in items],
            "full_fit_iterations": [item["full_fit_iterations"] for item in items],
            "inner_fit_iterations_max": [max(item["inner_fit_iterations"]) for item in items],
            "n_features": [item["n_features"] for item in items],
            "seconds": [item["seconds"] for item in items],
            "sha256": [item["sha256"] for item in items],
        }
    return {
        "artifact_sha256": file_sha256(Path(path)),
        "schema_version": payload["schema_version"],
        "kind": payload["kind"],
        "entry_count": len(entries),
        "by_seed": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True, help="같은 커밋에서 재실행한 exp110")
    parser.add_argument("--candidate-run", required=True, help="exp209 실행")
    parser.add_argument(
        "--recorded-baseline-run",
        default=None,
        help="기록된 exp110 실행(재현 대조용). 시드 42 OOF가 있어야 한다.",
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    store = MlflowRunStore(args.tracking_uri)
    baseline = load_run_facts(args.baseline_run, store)
    candidate = load_run_facts(args.candidate_run, store)
    champion = Champion.load()
    pool = Pool.load()
    y = pd.read_csv(TRAIN_PATH, usecols=[ID, TARGET]).set_index(ID)[TARGET]

    if baseline.git_commit != candidate.git_commit:
        raise SystemExit(
            f"기준과 후보의 커밋이 다르다: {baseline.git_commit} != {candidate.git_commit}"
        )
    if 42 not in baseline.seed_aucs or 42 not in candidate.seed_aucs:
        raise SystemExit("기준과 후보 모두 난수 42 OOF가 있어야 한다.")

    base42 = _seed_oof(store, baseline.run_id, 42)
    cand42 = _seed_oof(store, candidate.run_id, 42)
    pairwise42 = _pairwise(base42, cand42, y)

    reproduction = None
    if args.recorded_baseline_run:
        recorded = load_run_facts(args.recorded_baseline_run, store)
        rec42 = _seed_oof(store, recorded.run_id, 42)
        reproduction = {
            "recorded_run_id": recorded.run_id,
            "recorded_git_commit": recorded.git_commit,
            "recorded_seed42_auc": float(roc_auc_score(y.reindex(rec42.index), rec42["pred"])),
            "local_seed42_auc": pairwise42["baseline_auc"],
            "local_minus_recorded": pairwise42["baseline_auc"]
            - float(roc_auc_score(y.reindex(rec42.index), rec42["pred"])),
            "spearman_local_recorded": float(
                spearman(base42["pred"], rec42["pred"].reindex(base42.index))
            ),
            "max_abs_pred_diff": float(
                np.abs(base42["pred"].to_numpy() - rec42["pred"].reindex(base42.index).to_numpy()).max()
            ),
        }

    first_stage = _first_stage_evidence(store, candidate.run_id)

    floor = champion.oof_auc - ENTRY_FLOOR_MARGIN
    cand_oof = store.oof_of(candidate.run_id)
    correlations = {}
    for member in pool.members:
        pred = store.oof_of(member.run_id).reindex(cand_oof.index)
        assert pred.notna().all(), member.run_id
        correlations[member.config] = {
            "run_id": member.run_id,
            "spearman_seed_avg": float(spearman(cand_oof, pred)),
            "spearman_seed42": float(spearman(cand42["pred"], pred.reindex(cand42.index))),
        }
    nearest = max(correlations, key=lambda k: correlations[k]["spearman_seed_avg"])
    nearest42 = max(correlations, key=lambda k: correlations[k]["spearman_seed42"])
    ranked42 = sorted(
        correlations.items(), key=lambda item: item[1]["spearman_seed42"], reverse=True
    )
    entry = {
        "champion_run_id": champion.run_id,
        "champion_auc": champion.oof_auc,
        "entry_floor": floor,
        "candidate_seed42_auc": pairwise42["candidate_auc"],
        "candidate_seed_avg_auc": candidate.auc_oof,
        "floor_ok_seed42": pairwise42["candidate_auc"] >= floor,
        "floor_ok_seed_avg": candidate.auc_oof >= floor,
        "nearest_member_seed42": {"config": nearest42, **correlations[nearest42]},
        "nearest_member_seed_avg": {"config": nearest, **correlations[nearest]},
        "top5_seed42": [
            {"config": config, **values} for config, values in ranked42[:5]
        ],
        "baseline_member": correlations.get(baseline.experiment),
        "duplicate_threshold": DUPLICATE_SPEARMAN,
        "duplicate_seed42": correlations[nearest42]["spearman_seed42"] >= DUPLICATE_SPEARMAN,
        "duplicate_seed_avg": correlations[nearest]["spearman_seed_avg"] >= DUPLICATE_SPEARMAN,
        "nested_contribution": "앞 두 조건 통과 시 pipeline.pool_judgment로 별도 계산",
    }
    entry["prerequisites_passed_seed42"] = entry["floor_ok_seed42"] and not entry["duplicate_seed42"]

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
            reason="이슈 #505 짝비교용 기준(장부 champion 아님)",
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
            "passed": verdict.passed,
        }

    stop = not pairwise42["passed"] and not entry["prerequisites_passed_seed42"]
    result = {
        "schema_version": 1,
        "issue": 505,
        "git_commit": candidate.git_commit,
        "folds_sha256": file_sha256(Path("artifacts/folds.parquet")),
        "train_sha256": file_sha256(Path(TRAIN_PATH)),
        "baseline": {
            "run_id": baseline.run_id,
            "experiment": baseline.experiment,
            "seeds": baseline.seeds,
            "auc_oof": baseline.auc_oof,
            "seed_aucs": baseline.seed_aucs,
            "git_commit": baseline.git_commit,
            "git_dirty": baseline.git_dirty,
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
        "seed42_pairwise": pairwise42,
        "baseline_reproduction": reproduction,
        "first_stage": first_stage,
        "series2_entry_and_duplicate": entry,
        "confirmation_vs_baseline": confirmation,
        "stop_without_three_seeds": stop,
        "pool_size": len(pool.members),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {
                "seed42_delta": pairwise42["delta"],
                "seed42_fold_wins": f"{pairwise42['fold_wins']}/{pairwise42['fold_total']}",
                "seed42_passed": pairwise42["passed"],
                "floor_ok": entry["floor_ok_seed42"],
                "nearest_seed42": entry["nearest_member_seed42"]["config"],
                "spearman_seed42": entry["nearest_member_seed42"]["spearman_seed42"],
                "prerequisites_passed": entry["prerequisites_passed_seed42"],
                "stop_without_three_seeds": stop,
                "confirmation_passed": None if confirmation is None else confirmation["passed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
