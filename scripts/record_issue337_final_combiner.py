"""이슈 #337 최종 결합 전략 판정 기록을 만든다.

입력: 동결 입력 해시(frozen-inputs.yaml), 22개 전략 비교 산출물(ensemble-evaluation.json),
비교 실행의 stdout(ensemble.log), 전략별 OOF 예측(strategy-oof.parquet),
값 좌표 관점 제거 대조(ablation-contrast.json).
출력: artifacts/judgments/issue337-final-combiner.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TRAIN_PATH, labels
from pipeline.ensemble import MISSINGNESS_TEST_PATH
from pipeline.judgment import FOLDS_PATH, missingness_reweighting, weighted_oof_auc

EQUIVALENCE_BAND_LOWER = -0.000027669802  # docs/research/pool-reduction-judgment-rule.md


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_log(log_path: Path) -> tuple[dict[str, dict], list[str]]:
    """stdout 리포트에서 전략별 outer fold AUC(5자리)와 구성원 선택 빈도·평균 가중치를 읽는다."""
    strategies: dict[str, dict] = {}
    derived_runs: list[str] = []
    current = None
    for line in log_path.read_text().splitlines():
        m = re.match(r"^전략 (\S+): nested OOF AUC ([0-9.]+)$", line)
        if m:
            current = m.group(1)
            strategies[current] = {"members": {}}
            continue
        m = re.match(r"^전략 (\S+): 점수 비교 제외 \((.*)\)$", line)
        if m:
            strategies[m.group(1)] = {"failed": m.group(2)}
            current = None
            continue
        m = re.match(r"^\s+outer fold AUC: (.*)$", line)
        if m and current:
            strategies[current]["outer_fold_auc_log"] = {
                k: float(v) for k, v in (t.split("=") for t in m.group(1).split())
            }
            continue
        m = re.match(r"^\s{4}(\S+): (\d+)/(\d+), ([+-][0-9.]+)$", line)
        if m and current:
            strategies[current]["members"][m.group(1)] = {
                "selected": int(m.group(2)),
                "fold_total": int(m.group(3)),
                "mean_weight": float(m.group(4)),
            }
            continue
        m = re.match(r"^파생 앙상블 실행 기록: (\w+) ", line)
        if m:
            derived_runs.append(m.group(1))
    return strategies, derived_runs


def rescore(parquet_path: Path) -> dict[str, dict]:
    """전략별 OOF 예측에서 전체·분할별·가중 AUC를 전체 정밀도로 다시 잰다."""
    frame = pd.read_parquet(parquet_path).set_index(ID)
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"].reindex(frame.index)
    assert fold_of.notna().all()
    y = labels(frame.index)
    reweighting = missingness_reweighting(TRAIN_PATH, MISSINGNESS_TEST_PATH)
    out = {}
    for name in frame.columns:
        pred = frame[name]
        folds = {
            str(int(f)): float(roc_auc_score(y[fold_of == f], pred[fold_of == f]))
            for f in sorted(fold_of.unique())
        }
        weighted = weighted_oof_auc(pred.rename("prediction"), y, reweighting)
        out[name] = {
            "nested_oof_auc_rescored": float(roc_auc_score(y, pred)),
            "weighted_oof_auc_rescored": weighted.auc,
            "outer_fold_auc": folds,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("run-logs/issue337"))
    parser.add_argument("--contrast", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-rank-rescore", type=Path, required=True)
    parser.add_argument("--selected-derived-run", required=True)
    parser.add_argument("--nested-best-derived-run", required=True)
    args = parser.parse_args()
    root = args.root

    frozen = yaml.safe_load((root / "frozen-inputs.yaml").read_text())
    recorded_at = frozen["recorded_at_utc"]
    if not isinstance(recorded_at, str):
        recorded_at = recorded_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    frozen["recorded_at_utc"] = recorded_at
    evaluation = json.loads((root / "ensemble-evaluation.json").read_text())
    log_stats, derived_runs = parse_log(root / "ensemble.log")
    rescored = rescore(root / "strategy-oof.parquet")
    contrast = json.loads(args.contrast.read_text())
    fold_rank = json.loads(args.fold_rank_rescore.read_text())

    completed = [s for s in evaluation["strategies"] if s["status"] == "completed"]
    failed = [s for s in evaluation["strategies"] if s["status"] != "completed"]
    assert len(evaluation["strategies"]) == 22, len(evaluation["strategies"])
    by_nested = sorted(completed, key=lambda s: -s["nested_oof_auc"])
    by_weighted = sorted(completed, key=lambda s: -s["weighted_oof_auc"])
    nested_rank = {s["name"]: i + 1 for i, s in enumerate(by_nested)}
    weighted_rank = {s["name"]: i + 1 for i, s in enumerate(by_weighted)}
    selected = by_weighted[0]
    nested_best = by_nested[0]

    strategies = []
    for s in evaluation["strategies"]:
        name = s["name"]
        entry = {
            "name": name,
            "status": s["status"],
            "elapsed_seconds": s["elapsed_seconds"],
        }
        if s["status"] != "completed":
            entry["reason"] = s.get("reason")
            strategies.append(entry)
            continue
        r = rescored[name]
        assert abs(r["nested_oof_auc_rescored"] - s["nested_oof_auc"]) < 1e-12
        assert abs(r["weighted_oof_auc_rescored"] - s["weighted_oof_auc"]) < 1e-12
        entry.update(
            {
                "nested_oof_auc": s["nested_oof_auc"],
                "weighted_oof_auc": s["weighted_oof_auc"],
                "nested_rank": nested_rank[name],
                "weighted_rank": weighted_rank[name],
                "nested_delta_vs_selected": s["nested_oof_auc"] - selected["nested_oof_auc"],
                "weighted_delta_vs_selected": s["weighted_oof_auc"] - selected["weighted_oof_auc"],
                "outer_fold_auc": r["outer_fold_auc"],
                "member_selection": log_stats[name]["members"],
            }
        )
        strategies.append(entry)

    payload = {
        "schema_version": 1,
        "judgment_id": "issue337-final-combiner",
        "contract_version": "candidate-pool-v1",
        "created_at": frozen["recorded_at_utc"][:10],
        "decision_issue": 337,
        "action": "final_combiner_selection",
        "status": "adopted",
        "question": (
            "동결 후보 풀에서 등록된 22개 결합 전략 중 어느 것이 최고 가중 OOF를 내며, "
            "nested OOF 눈금과 순위가 갈리는가? 값 좌표 관점 두 구현은 함께 있는 조건에서도 "
            "각각 또는 묶음으로 양의 기여를 유지하는가?"
        ),
        "execution_note": (
            "티켓의 달력 시작 조건(2026-08-28T00:00:00Z)보다 앞서 사용자 지시로 실행했다. "
            "실행 시점에 후보 풀을 바꿀 수 있는 열린 실험 티켓이 없어 풀을 동결 상태로 본다. "
            "2026-08-28 전에 후보 풀이나 등록 전략이 바뀌면 이 판정은 오래된 판정이 된다."
        ),
        "frozen_input": {
            "recorded_at_utc": frozen["recorded_at_utc"],
            "git_commit": frozen["git_commit"],
            "git_dirty": frozen["git_dirty"],
            "candidate_pool": {
                "path": "artifacts/pool.yaml",
                "sha256": frozen["sha256"]["artifacts/pool.yaml"],
                "member_count": evaluation["member_count"],
                "members": [m["config"] for m in evaluation["members"]],
                "member_run_ids": {m["config"]: m["run_id"] for m in evaluation["members"]},
            },
            "champion": {"path": "artifacts/champion.yaml", "sha256": frozen["sha256"]["artifacts/champion.yaml"]},
            "folds": {"path": "artifacts/folds.parquet", "sha256": frozen["sha256"]["artifacts/folds.parquet"]},
            "full_refit_plan_before": {"path": "artifacts/full-refit-plan.yaml", "sha256": frozen["sha256"]["artifacts/full-refit-plan.yaml"]},
            "data": {
                "train": frozen["sha256"]["data/train.csv"],
                "test": frozen["sha256"]["data/test.csv"],
                "sample_submission": frozen["sha256"]["data/sample_submission.csv"],
            },
            "registered_combiners": {
                "scope": "all_registered",
                "count": 22,
                "names": [s["name"] for s in evaluation["strategies"]],
                "names_sha256": hashlib.sha256(
                    json.dumps([s["name"] for s in evaluation["strategies"]], separators=(",", ":")).encode()
                ).hexdigest(),
            },
            "evaluation_artifacts": {
                "ensemble_evaluation_json_sha256": _sha256(root / "ensemble-evaluation.json"),
                "strategy_oof_parquet_sha256": _sha256(root / "strategy-oof.parquet"),
                "weighted_oof_sample": evaluation.get("weighted_oof_sample"),
            },
        },
        "procedure": {
            "kind": "compare_all_registered_combiners_once",
            "outer_folds": 5,
            "selection_scale": "weighted_oof_auc",
            "secondary_scale": "nested_oof_auc",
            "equivalence_band_lower": EQUIVALENCE_BAND_LOWER,
            "strategies_added_or_reconfigured_after_results": False,
            "strategies_excluded_after_results": False,
            "coefficient_precision_note": "member_selection의 mean_weight는 비교 실행 stdout의 소수 5자리 값이다. 선택 전략의 전체 정밀도 계수는 파생 앙상블 실행의 member_weights.csv에 있다.",
        },
        "result": {
            "selected_strategy": selected["name"],
            "selected_weighted_oof_auc": selected["weighted_oof_auc"],
            "selected_nested_oof_auc": selected["nested_oof_auc"],
            "selected_outer_fold_auc": rescored[selected["name"]]["outer_fold_auc"],
            "nested_best_strategy": nested_best["name"],
            "nested_best_nested_oof_auc": nested_best["nested_oof_auc"],
            "nested_best_weighted_oof_auc": nested_best["weighted_oof_auc"],
            "scales_agree": selected["name"] == nested_best["name"],
            "nested_gap_selected_minus_nested_best": selected["nested_oof_auc"] - nested_best["nested_oof_auc"],
            "weighted_gap_selected_minus_nested_best": selected["weighted_oof_auc"] - nested_best["weighted_oof_auc"],
            "top5_by_weighted": [s["name"] for s in by_weighted[:5]],
            "top5_by_nested": [s["name"] for s in by_nested[:5]],
            "top5_weighted_spread": by_weighted[0]["weighted_oof_auc"] - by_weighted[4]["weighted_oof_auc"],
            "top5_nested_spread": by_nested[0]["nested_oof_auc"] - by_nested[4]["nested_oof_auc"],
            "completed_strategy_count": len(completed),
            "failed_strategy_count": len(failed),
            "total_elapsed_seconds": sum(s["elapsed_seconds"] for s in evaluation["strategies"]),
            "derived_ensemble_runs": {
                "selected_strategy_run_id": args.selected_derived_run,
                "nested_best_run_id": args.nested_best_derived_run,
                "recorded_in_log": derived_runs,
                "baseline_run_id": frozen["baseline_derived_run"],
            },
        },
        "outer_fold_winners": {
            fold: max(completed, key=lambda s: rescored[s["name"]]["outer_fold_auc"][fold])["name"]
            for fold in rescored[selected["name"]]["outer_fold_auc"]
        },
        "selected_vs_top5_outer_fold_wins": {
            s["name"]: sum(
                rescored[selected["name"]]["outer_fold_auc"][f] > rescored[s["name"]]["outer_fold_auc"][f]
                for f in rescored[selected["name"]]["outer_fold_auc"]
            )
            for s in by_weighted[1:5]
        },
        "fold_rank_normalized_diagnostic": {
            "purpose": (
                "선택 전략은 바깥 분할 블록마다 백분위 순위를 내므로 nested 연결에서 분할 간 눈금 차이가 "
                "사라진다. 모든 전략의 OOF 예측을 같은 바깥 분할 백분위 순위로 바꿔 다시 채점해 같은 "
                "발판에서 비교한 보조 진단이며, 결과 확인 뒤 만든 눈금이라 선택 기준으로 쓰지 않는다."
            ),
            "selected_lambda_by_outer_fold": "5개 분할 모두 1.0 (분할 안 스피어만 1.0으로 rank_logit_logistic과 순위 동일)",
            "ranking": [
                {
                    "name": name,
                    "weighted_oof_auc_fold_rank": v["weighted_fold_rank"],
                    "nested_oof_auc_fold_rank": v["nested_fold_rank"],
                }
                for name, v in sorted(fold_rank.items(), key=lambda kv: -kv[1]["weighted_fold_rank"])
            ],
        },
        "strategies": strategies,
        "value_coordinate_ablation": {
            "purpose": "값 좌표 관점 두 구현(exp106 Lookup-Transformer, exp139 RealMLP)의 조건부 기여 설명. 소급 제거 관문이 아니다.",
            "fixed_strategy": contrast["strategy"],
            "full_pool": contrast["full_pool"],
            "contrasts": [
                {k: v for k, v in c.items() if k != "source"} for c in contrast["contrasts"]
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    yaml.SafeDumper.ignore_aliases = lambda *_: True  # 같은 값을 두 곳에 적어도 앵커를 만들지 않는다.
    args.output.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100))
    print(f"판정 기록 저장: {args.output}")
    print(f"선택 {selected['name']} 가중 {selected['weighted_oof_auc']:.10f} nested {selected['nested_oof_auc']:.10f}")
    print(f"nested 최고 {nested_best['name']} nested {nested_best['nested_oof_auc']:.10f} 가중 {nested_best['weighted_oof_auc']:.10f}")


if __name__ == "__main__":
    main()
