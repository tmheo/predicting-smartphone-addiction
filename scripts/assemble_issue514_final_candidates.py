"""전체 데이터 재학습판으로 이슈 514의 두 제출 후보를 결정적으로 조립한다.

첫 후보는 공식 자체 풀 36개의 전체 데이터 재학습 시험 예측만 사용한다.
둘째 후보는 같은 자체 풀 36개와 동결된 외부 OOF 278개를 합친 314개 구성에
이슈 513에서 통과한 C 선택 결합 절차를 적용한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble_extended_stack as prior_assembly
import judge_extended_stack as ladder
import judge_issue513_extended_stack_reassembly as issue513
import judge_logistic_c_selection as logistic

from pipeline import ensemble, refit
from pipeline.data import ID, TARGET, file_sha256
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import MlflowRunStore


ISSUE = 514
SCHEMA = "final-full-refit-candidates/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = Path("artifacts/pool.yaml")
PLAN_PATH = Path("artifacts/full-refit-plan.yaml")
ISSUE513_PRECOMMIT = Path(
    "docs/research/extended-stack-pool-reassembly/issue513/precommit.json"
)
ISSUE513_COMPARISON = Path(
    "docs/research/extended-stack-pool-reassembly/issue513/comparison.json"
)
ISSUE512_DIRECT_GATE = Path(
    "docs/research/missingness-propagation-batch/issue512/direct-nested-gate.json"
)
ISSUE489_PRECOMMIT = Path(
    "docs/research/logistic-c-selection/issue489/precommit.json"
)
BASELINE_MANIFEST = Path("docs/research/extended-stack-submission-2-manifest.json")
OWN_CSV = "issue514-pool36-full.csv"
EXTENDED_CSV = "issue514-extended314-own-full.csv"
MANIFEST = "manifest.json"
CHECKSUMS = "manifest.sha256"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    )
    require(not dirty.stdout, "제출 조립은 깨끗한 git 작업 폴더에서만 실행한다.")
    return result.stdout.strip()


def source_spec(source_root: Path, specification: str) -> str:
    path, separator, selector = specification.partition("[")
    resolved = str(source_root / path)
    return resolved if not separator else f"{resolved}[{selector}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--full-refit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    require(Path.cwd().resolve() == REPO_ROOT, "저장소 루트에서 실행해야 한다.")
    source_root = args.source_root.expanduser().resolve()
    full_refit_dir = args.full_refit_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    require(not out_dir.exists(), f"새 출력 폴더가 아니다: {out_dir}")
    commit = git_commit()

    issue513_precommit = logistic.read_json(ISSUE513_PRECOMMIT)
    issue513_comparison = logistic.read_json(ISSUE513_COMPARISON)
    issue512_direct_gate = logistic.read_json(ISSUE512_DIRECT_GATE)
    require(issue512_direct_gate["passed"] is True, "이슈 512 자체 풀 관문이 통과 상태가 아니다.")
    require(
        issue512_direct_gate["proposal"]["best_strategy"]
        == "shrunk_rank_logit_logistic",
        "이슈 512 자체 풀의 최선 결합 방식이 등록 방식과 다르다.",
    )
    require(issue513_comparison["passes_gate"] is True, "이슈 513 교체 문턱이 통과 상태가 아니다.")
    require(issue513_comparison["folds_positive"] == 5, "이슈 513 분할 5/5 통과가 아니다.")
    require(
        issue513_comparison["precommit_sha256"]
        == issue513_precommit["precommit_sha256"],
        "이슈 513 판정과 사전 고정 기록이 다르다.",
    )

    source_train = source_root / "data/train.csv"
    source_test = source_root / "data/test.csv"
    source_mlflow = source_root / "mlflow.db"
    for path in (source_train, source_test, source_mlflow, full_refit_dir):
        require(path.exists(), f"입력을 찾지 못했다: {path}")

    train = pd.read_csv(source_train)
    test = pd.read_csv(source_test)
    train_index = pd.Index(train[ID], name=ID)
    test_index = pd.Index(test[ID], name=ID)
    fold_frame = pd.read_parquet("artifacts/folds.parquet")
    fold_of = fold_frame.set_index(ID)["fold"].reindex(train_index)
    require(fold_of.notna().all(), "고정 분할 id가 학습 자료와 맞지 않는다.")
    y = train.set_index(ID).loc[train_index, TARGET]

    store = MlflowRunStore(tracking_uri=f"sqlite:///{source_mlflow}")
    plan = refit.load_executable_plan(PLAN_PATH, store=store)
    members = [(member.config, member.run_id) for member in plan.members]
    require(len(members) == 36, "공식 자체 풀이 36개가 아니다.")
    own_oof = ensemble.member_matrix(members, store, train_index).astype(np.float64)

    full_columns: dict[str, pd.Series] = {}
    full_member_records = []
    for member in plan.members:
        values = refit._load_member_full_prediction(
            plan,
            member,
            full_refit_dir,
            test[ID],
        )
        full_columns[member.config] = pd.Series(values, index=test_index)
        member_dir = full_refit_dir / member.config
        full_member_records.append(
            {
                "column": member.config,
                "run_id": member.run_id,
                "member_entry_sha256": member.entry_sha256,
                "prediction_sha256": prediction_array_sha256(values),
                "prediction_file_sha256": file_sha256(member_dir / "test_pred_full.parquet"),
                "manifest_file_sha256": file_sha256(member_dir / "manifest.json"),
            }
        )
    own_full = pd.DataFrame(full_columns, index=test_index, dtype=np.float64)
    require(list(own_full.columns) == list(own_oof.columns), "자체 OOF와 전체 데이터 예측 열이 다르다.")

    registered = ensemble.COMBINER_REGISTRY[plan.protocol.combiner]
    fitted_own = registered.fit(own_oof, y)
    own_prediction = np.asarray(fitted_own.predict(own_full), dtype=np.float64)
    require(own_prediction.shape == (len(test),) and np.isfinite(own_prediction).all(), "자체 풀 제출 예측이 유효하지 않다.")

    reference_path = full_refit_dir / "submission_full.csv"
    reference = pd.read_csv(reference_path)
    require(reference[ID].to_numpy().tolist() == test[ID].to_numpy().tolist(), "재학습 기준 제출 id가 다르다.")
    require(
        np.array_equal(reference[TARGET].to_numpy(np.float64), own_prediction),
        "자체 풀 전체 데이터 제출이 표준 재학습 조립 결과와 다르다.",
    )

    baseline_manifest = logistic.read_json(BASELINE_MANIFEST)
    issue489_precommit = logistic.read_json(ISSUE489_PRECOMMIT)
    baseline_by_column = {
        row["column"]: row for row in issue489_precommit["members"]["rows"]
    }
    external_manifest = [
        row for row in baseline_manifest["members"] if row["origin"] == "external"
    ]
    require(len(external_manifest) == 278, "동결 외부 구성이 278개가 아니다.")

    external_oof: dict[str, np.ndarray] = {}
    external_test: dict[str, np.ndarray] = {}
    external_records = []
    for member in external_manifest:
        column = member["column"]
        frozen = baseline_by_column[column]
        oof_values = np.asarray(
            ladder.load_ledger_array(source_spec(source_root, member["oof_path"])),
            dtype=np.float64,
        )
        test_path = source_root / frozen["test_path"]
        test_values = np.asarray(ladder.load_ledger_array(str(test_path)), dtype=np.float64)
        require(oof_values.shape == (len(train),) and np.isfinite(oof_values).all(), f"{column}: 외부 OOF가 유효하지 않다.")
        require(test_values.shape == (len(test),) and np.isfinite(test_values).all(), f"{column}: 외부 시험 예측이 유효하지 않다.")
        require(prediction_array_sha256(oof_values) == frozen["oof_sha256"], f"{column}: 외부 OOF 해시가 다르다.")
        require(prediction_array_sha256(test_values) == frozen["test_prediction_sha256"], f"{column}: 외부 시험 예측 해시가 다르다.")
        require(file_sha256(test_path) == frozen["test_sha256"], f"{column}: 외부 시험 파일 해시가 다르다.")
        external_oof[column] = oof_values
        external_test[column] = test_values
        external_records.append(
            {
                "column": column,
                "oof_sha256": frozen["oof_sha256"],
                "test_prediction_sha256": frozen["test_prediction_sha256"],
                "test_file_sha256": frozen["test_sha256"],
                "test_path": frozen["test_path"],
            }
        )

    external_oof_frame = pd.DataFrame(external_oof, index=train_index, dtype=np.float64)
    external_test_frame = pd.DataFrame(external_test, index=test_index, dtype=np.float64)
    extended_oof = pd.concat([own_oof, external_oof_frame], axis=1)
    extended_test = pd.concat([own_full, external_test_frame], axis=1)
    expected_rows = issue513_precommit["reassembled"]["members"]
    require(list(extended_oof.columns) == [row["column"] for row in expected_rows], "314개 OOF 순서가 이슈 513과 다르다.")
    require(list(extended_test.columns) == list(extended_oof.columns), "314개 시험 예측 순서가 OOF와 다르다.")
    extended_oof_composition = logistic.canonical_sha256(
        [
            (column, prediction_array_sha256(extended_oof[column].to_numpy(np.float64)))
            for column in extended_oof.columns
        ]
    )
    require(
        extended_oof_composition
        == issue513_precommit["reassembled"]["composition_sha256"],
        "314개 OOF 구성 해시가 이슈 513과 다르다.",
    )
    extended_test_composition = logistic.canonical_sha256(
        [
            (column, prediction_array_sha256(extended_test[column].to_numpy(np.float64)))
            for column in extended_test.columns
        ]
    )

    combiner = ensemble.CSelectedShrunkRankLogitCombiner(
        fold_of=fold_of,
        c_grid=issue513.C_GRID,
        lambda_grid=issue513.LAMBDA_GRID,
        max_iter=issue513.META_MAX_ITER,
    )
    fitted_extended = combiner.fit(extended_oof, y)
    extended_prediction = np.asarray(
        fitted_extended.predict(extended_test), dtype=np.float64
    )
    require(
        extended_prediction.shape == (len(test),)
        and np.isfinite(extended_prediction).all(),
        "314개 확장 제출 예측이 유효하지 않다.",
    )

    out_dir.mkdir(parents=True)
    own_path = out_dir / OWN_CSV
    extended_path = out_dir / EXTENDED_CSV
    pd.DataFrame({ID: test[ID], TARGET: own_prediction}).to_csv(own_path, index=False)
    pd.DataFrame({ID: test[ID], TARGET: extended_prediction}).to_csv(
        extended_path, index=False
    )

    own_oof_composition = logistic.canonical_sha256(
        [
            (column, prediction_array_sha256(own_oof[column].to_numpy(np.float64)))
            for column in own_oof.columns
        ]
    )
    own_test_composition = logistic.canonical_sha256(
        [
            (column, prediction_array_sha256(own_full[column].to_numpy(np.float64)))
            for column in own_full.columns
        ]
    )
    manifest = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "git_commit": commit,
        "public_score_used": False,
        "uploaded": False,
        "inputs": {
            "pool_sha256": file_sha256(POOL_PATH),
            "plan_sha256": file_sha256(PLAN_PATH),
            "train_sha256": file_sha256(source_train),
            "test_sha256": file_sha256(source_test),
            "folds_sha256": file_sha256(Path("artifacts/folds.parquet")),
            "issue513_precommit_sha256": file_sha256(ISSUE513_PRECOMMIT),
            "issue513_comparison_sha256": file_sha256(ISSUE513_COMPARISON),
            "issue512_direct_gate_sha256": file_sha256(ISSUE512_DIRECT_GATE),
            "issue489_precommit_sha256": file_sha256(ISSUE489_PRECOMMIT),
            "baseline_manifest_sha256": file_sha256(BASELINE_MANIFEST),
            "full_refit_manifest_sha256": file_sha256(full_refit_dir / "manifest.json"),
            "full_refit_reference_submission_sha256": file_sha256(reference_path),
        },
        "issue513_gate": {
            "nested_auc": issue513_comparison["reassembled"]["nested_auc"],
            "delta_vs_current_submission": issue513_comparison[
                "delta_vs_current_submission"
            ],
            "folds_positive": issue513_comparison["folds_positive"],
            "prediction_sha256": issue513_comparison["reassembled"][
                "prediction_sha256"
            ],
            "verdict": issue513_comparison["verdict"],
        },
        "issue512_pool36_gate": {
            "nested_auc": issue512_direct_gate["proposal"]["best_auc"],
            "fold_aucs": issue512_direct_gate["proposal"]["best_fold_auc"],
            "strategy": issue512_direct_gate["proposal"]["best_strategy"],
            "delta_vs_previous_pool": issue512_direct_gate["best_strategy_delta"],
            "folds_positive": issue512_direct_gate["diagnostics"]["outer_fold_wins"],
        },
        "full_refit_members": full_member_records,
        "external_members": external_records,
        "candidates": {
            "pool36_full": {
                "member_count": 36,
                "strategy": plan.protocol.combiner,
                "nested_oof_auc": issue512_direct_gate["proposal"]["best_auc"],
                "nested_fold_aucs": issue512_direct_gate["proposal"]["best_fold_auc"],
                "selected_lambda": fitted_own.shrinkage_lambda,
                "weights": fitted_own.summary(),
                "oof_composition_sha256": own_oof_composition,
                "test_composition_sha256": own_test_composition,
                "prediction_sha256": prediction_array_sha256(own_prediction),
                "in_sample_oof_auc": float(
                    roc_auc_score(y.to_numpy(), fitted_own.predict(own_oof))
                ),
                "submission": {
                    "name": OWN_CSV,
                    "sha256": file_sha256(own_path),
                    "checks": prior_assembly.rank_space_checks(
                        own_prediction, test[ID]
                    ),
                    "matches_standard_refit_assembly": True,
                },
            },
            "extended314_own_full": {
                "member_count": 314,
                "own_member_count": 36,
                "external_member_count": 278,
                "strategy": ensemble.CSelectedShrunkRankLogitCombiner.name,
                "nested_oof_auc": issue513_comparison["reassembled"]["nested_auc"],
                "nested_fold_aucs": issue513_comparison["reassembled"]["fold_aucs"],
                "selected_c": fitted_extended.c,
                "selected_lambda": fitted_extended.shrinkage_lambda,
                "selection_aucs": [
                    {"c": c, "lambda": value, "auc": auc}
                    for (c, value), auc in fitted_extended.selection_aucs.items()
                ],
                "inner_fits": [fit.__dict__ for fit in fitted_extended.inner_fits],
                "final_iterations": fitted_extended.final_iterations,
                "final_coefficient_l2_norm": fitted_extended.final_coefficient_l2_norm,
                "weights": fitted_extended.summary(),
                "oof_composition_sha256": extended_oof_composition,
                "test_composition_sha256": extended_test_composition,
                "prediction_sha256": prediction_array_sha256(
                    extended_prediction
                ),
                "in_sample_oof_auc": float(
                    roc_auc_score(
                        y.to_numpy(), fitted_extended.predict(extended_oof)
                    )
                ),
                "submission": {
                    "name": EXTENDED_CSV,
                    "sha256": file_sha256(extended_path),
                    "checks": prior_assembly.rank_space_checks(
                        extended_prediction, test[ID]
                    ),
                },
            },
        },
    }
    write_json(out_dir / MANIFEST, manifest)
    checksum_lines = [
        f"{file_sha256(out_dir / name)}  {name}"
        for name in (OWN_CSV, EXTENDED_CSV, MANIFEST)
    ]
    (out_dir / CHECKSUMS).write_text("\n".join(checksum_lines) + "\n")
    print(
        json.dumps(
            {
                "pool36_full_sha256": file_sha256(own_path),
                "extended314_own_full_sha256": file_sha256(extended_path),
                "manifest_sha256": file_sha256(out_dir / MANIFEST),
                "selected_c": fitted_extended.c,
                "selected_lambda": fitted_extended.shrinkage_lambda,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, logistic.JudgmentError) as exc:
        sys.exit(f"조립 불가: {exc}")
