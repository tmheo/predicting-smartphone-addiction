"""규제 강도 선택 확장 스택의 전체 OOF 제안 조립. (#489, 사용자 결정에 따른 조립)

이슈 #489의 선택 절차 대조는 사전 고정 문턱(+0.00002, 분할 5/5)에 미달했다.
2026-08-28 사용자가 그 문턱을 사후에 접고 C 선택판을 두 번째 최종 제출로 바꾸기로 결정했다.
이 도구는 그 결정을 manifest에 명시하고, 판정과 같은 결합기(`CSelectedShrunkRankLogitCombiner`)를
전체 313 OOF에 한 번 맞춰 (C, λ)를 고른 뒤 현재 313 시험 예측에 적용해 제출 CSV를 만든다.

판정 도구 `judge_logistic_c_selection.py`의 `full`은 통과 판정에서만 열리므로 쓰지 않는다.
대신 precommit.json의 무결성, 입력 해시, 결합기 module 해시(판정 때와 같은 코드)를 다시
확인하고 comparison.json의 판정 결과를 그대로 manifest에 옮긴다.

사용법:
    uv run python scripts/assemble_c_selected_extended_stack.py --decision "<사용자 결정 요약>"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble_extended_stack as prior_assembly
import judge_extended_stack as ladder
import judge_logistic_c_selection as judge
import judge_strict_external_selection as strict

from pipeline import ensemble
from pipeline.data import ID, TARGET, file_sha256
from pipeline.pool_audit import prediction_array_sha256

RECORD_DIR = Path("docs/research/logistic-c-selection/issue489")
SUBMISSION_PATH = judge.SUBMISSION_DIR / "issue489-c-selected-extended-stack.csv"


def verified_precommit(run_dir: Path) -> dict:
    payload = judge.read_json(run_dir / "precommit.json")
    judge._require(judge.canonical_sha256({k: v for k, v in payload.items() if k != "precommit_sha256"}) == payload["precommit_sha256"], "precommit.json이 제자리에서 바뀌었다.")
    for key, entry in payload["inputs"].items():
        if isinstance(entry, dict) and "path" in entry:
            judge._require(file_sha256(Path(entry["path"])) == entry["sha256"], f"{key} 해시가 precommit과 다르다.")
    for name, digest in payload["cache"].items():
        judge._require(file_sha256(run_dir / "cache" / name) == digest, f"캐시 {name}이 precommit과 다르다.")
    judge._require(file_sha256(strict.ENSEMBLE_SOURCE) == payload["code_state"]["ensemble_module"]["sha256"], "결합기 module이 판정 때와 다르다. 같은 코드로만 조립한다.")
    judge._require(payload["candidate"]["c_grid"] == list(judge.C_GRID) and payload["candidate"]["lambda_grid"] == list(judge.LAMBDA_GRID), "후보값이 precommit과 다르다.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=judge.OUT_DIR)
    parser.add_argument("--decision", required=True, help="문턱을 접고 조립하기로 한 사용자 결정 요약(날짜 포함)")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    started = time.monotonic()
    payload = verified_precommit(run_dir)
    comparison = judge.read_json(run_dir / "comparison.json")
    judge._require(comparison["precommit_sha256"] == payload["precommit_sha256"], "comparison.json이 다른 precommit에서 나왔다.")
    judge._require(comparison["reproduction"]["passes"], "대조군 재현이 통과하지 않은 판정이다.")

    fold_of, y = strict.load_folds_and_labels()
    oof = judge.load_matrix(run_dir, payload, fold_of)
    test_ids = pd.read_csv(judge.TEST_PATH, usecols=[ID])[ID]
    judge._require(len(test_ids) == judge.N_TEST and not test_ids.duplicated().any(), "test.csv의 행 수나 id가 기대와 다르다.")
    own_test = pd.read_parquet(judge.OWN_TEST_PATH)
    judge._require(own_test[ID].to_numpy().tolist() == test_ids.to_numpy().tolist(), "5:1 혼합판 시험 예측의 id 순서가 test.csv와 다르다.")
    own_test = own_test.set_index(ID)
    columns: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    for entry in payload["members"]["rows"]:
        column = entry["column"]
        if entry["origin"] == "own":
            values = own_test[column].to_numpy(np.float64)
            sources[column] = {"kind": "own_cv5_full1_mix", "test_path": str(judge.OWN_TEST_PATH)}
        else:
            values = ladder.load_ledger_array(entry["test_path"])
            sources[column] = {"kind": "external_cv_fold_average", "test_path": entry["test_path"]}
        judge._require(values.shape == (judge.N_TEST,) and bool(np.isfinite(values).all()), f"{column}: 시험 배열 형태 {values.shape} 또는 비유한값")
        judge._require(prediction_array_sha256(values) == entry["test_prediction_sha256"], f"{column}: 시험 배열 해시가 #457 manifest와 다르다.")
        columns[column] = values
        sources[column]["prediction_sha256"] = entry["test_prediction_sha256"]
    test = pd.DataFrame(columns, index=test_ids.to_numpy()).astype(np.float64)
    judge._require(list(test.columns) == list(oof.columns), "시험 행렬의 열 순서가 OOF와 다르다.")

    combiner = ensemble.CSelectedShrunkRankLogitCombiner(fold_of=fold_of, c_grid=judge.C_GRID, lambda_grid=judge.LAMBDA_GRID, max_iter=judge.META_MAX_ITER)
    fitted = combiner.fit(oof, y)
    prediction = np.asarray(fitted.predict(test), dtype=np.float64)
    judge._require(prediction.shape == (judge.N_TEST,) and bool(np.isfinite(prediction).all()), "제안 시험 예측이 유한하지 않다.")
    in_sample = float(roc_auc_score(y.to_numpy(), np.asarray(fitted.predict(oof), dtype=np.float64)))
    current = pd.read_csv(judge.CURRENT_SUBMISSION_PATH)
    judge._require(current[ID].to_numpy().tolist() == test_ids.to_numpy().tolist(), "현재 제출의 id 순서가 test.csv와 다르다.")
    rho = float(spearmanr(prediction, current[TARGET].to_numpy(np.float64)).correlation)
    judge.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test_ids.to_numpy(), TARGET: prediction}).to_csv(SUBMISSION_PATH, index=False)

    manifest = {
        "schema_version": 1,
        "issue": judge.ISSUE,
        "precommit_sha256": payload["precommit_sha256"],
        "git": strict.git_state(),
        "strategy": judge.CANDIDATE_STRATEGY,
        "user_override": {
            "decision": args.decision,
            "judged_verdict": comparison["verdict"],
            "gate": payload["gate"],
            "note": "판정 도구의 문턱(+0.00002, 분할 5/5)에 미달한 후보를 사용자 결정으로 조립한다. 문턱과 판정 기록은 바꾸지 않았다.",
        },
        "judged": {
            "candidate_nested_auc": comparison["candidate"]["nested_auc"],
            "control_nested_auc": comparison["control"]["nested_auc"],
            "delta_vs_control": comparison["delta_vs_control_reference"],
            "folds_positive": comparison["folds_positive"],
            "fold_deltas": comparison["fold_deltas"],
            "candidate_weighted_oof_auc": comparison["candidate"]["weighted_oof_auc"],
            "candidate_fold_aucs": comparison["candidate"]["fold_aucs"],
            "fold_selected_c": comparison["candidate"]["fold_selected_c"],
            "fold_selected_lambda": comparison["candidate"]["fold_selected_lambda"],
            "reproduction_passes": comparison["reproduction"]["passes"],
        },
        "assembled": {"member_count": payload["members"]["count"], "own_member_count": payload["members"]["own_count"], "external_member_count": payload["members"]["external_count"], "config": payload["members"]["config"], "composition_sha256": payload["members"]["composition_sha256"], "test_composition_sha256": payload["members"]["test_composition_sha256"]},
        "combiner": {
            "selected_c": fitted.c,
            "selected_lambda": fitted.shrinkage_lambda,
            "c_grid": list(judge.C_GRID),
            "lambda_grid": list(judge.LAMBDA_GRID),
            "selection_aucs": [{"c": c, "lambda": lam, "auc": value} for (c, lam), value in fitted.selection_aucs.items()],
            "selection_note": "전체 OOF 5분할 leave-one-fold-out 내부 예측의 AUC. 통과 여부 판정에 쓰지 않는다.",
            "inner_fits": [fit.__dict__ for fit in fitted.inner_fits],
            "final_iterations": fitted.final_iterations,
            "final_coefficient_l2_norm": fitted.final_coefficient_l2_norm,
            "fit_protocol": "전체 OOF 1회 적합, (C, λ)는 5분할 leave-one-fold-out",
            "in_sample_oof_auc": in_sample,
        },
        "inputs": {**{k: v for k, v in payload["inputs"].items() if isinstance(v, dict)}, "ensemble_module_sha256": payload["code_state"]["ensemble_module"]["sha256"], "judge_script_sha256": payload["code_state"]["script"]["sha256"], "judge_git_commit": payload["code_state"]["git"]["commit"]},
        "members": [{"column": column, "weight": float(weight), "test": sources[column]} for column, weight in fitted.summary().items()],
        "submission": {"path": str(SUBMISSION_PATH), "file_sha256": file_sha256(SUBMISSION_PATH), "prediction_sha256": prediction_array_sha256(prediction), "checks": prior_assembly.rank_space_checks(prediction, test_ids), "spearman_vs_current_submission": {"path": str(judge.CURRENT_SUBMISSION_PATH), "sha256": file_sha256(judge.CURRENT_SUBMISSION_PATH), "run_id": judge.CURRENT_RUN_ID, "spearman": rho}},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": judge.peak_rss_bytes(),
        "finished_at": judge.now_iso(),
    }
    for target in (run_dir / "full" / "assembly-manifest.json", RECORD_DIR / "assembly-manifest.json"):
        judge.write_json(target, manifest)
    print(f"전체 OOF 제안 C={fitted.c} λ={fitted.shrinkage_lambda} (전체 OOF 적합 반복 {fitted.final_iterations}, 계수 L2 {fitted.final_coefficient_l2_norm:.4f}, in-sample {in_sample:.7f})")
    print(f"제출 {SUBMISSION_PATH} sha256 {manifest['submission']['file_sha256']}, 현재 제출과 스피어만 {rho:.6f}, manifest {RECORD_DIR / 'assembly-manifest.json'} ({manifest['elapsed_seconds']:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except judge.JudgmentError as exc:
        sys.exit(f"조립 불가: {exc}")
