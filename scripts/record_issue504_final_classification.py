"""이슈 504의 독립 결측 증강 최종 근거와 분류를 생성하고 검증한다."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pipeline.data import file_sha256
from pipeline.judgment import POOL_EQUIVALENCE_BAND_UPPER


ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/504"
MAP_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/499"
REPOSITORY_URL = "https://github.com/tmheo/predicting-smartphone-addiction"

DIAGNOSTIC_PATH = Path("artifacts/issue500-training-row-diagnostic.json")
SCREEN_PATH = Path("artifacts/issue501-seed42-missingness-screen.json")
CONFIRMATION_PATH = Path(
    "artifacts/issue502-three-seed-missingness-confirmation.json"
)
POOL_FREEZE_PATH = Path(
    "artifacts/issue503-missingness-candidate-pool-freeze.json"
)
POOL_RESULT_PATH = Path(
    "artifacts/issue503-missingness-pool-contribution.json"
)
POOL_PATH = Path("artifacts/pool.yaml")
REFIT_PLAN_PATH = Path("artifacts/full-refit-plan.yaml")
REPLEAF_REPORT_PATH = Path(
    "docs/research/repleafgbm-entry-audit-2026-08-28.md"
)

CONFIG_PATHS = {
    "baseline": Path("configs/exp117_ag25_gbm_r21.yaml"),
    "original": Path("configs/exp206_issue500_ag25_original.yaml"),
    "tripled": Path("configs/exp207_issue500_ag25_tripled.yaml"),
    "missingness_augmented": Path(
        "configs/exp208_issue500_ag25_missingness_augmented.yaml"
    ),
}
CODE_PATHS = (
    Path("src/pipeline/config.py"),
    Path("src/pipeline/training_rows.py"),
    Path("src/pipeline/cv_seed_execution.py"),
    Path("src/pipeline/seed_reuse.py"),
    Path("scripts/diagnose_issue500_training_rows.py"),
    Path("scripts/record_issue502_confirmation.py"),
    Path("scripts/freeze_issue503_candidate_pool_input.py"),
    Path("scripts/record_issue503_pool_contribution.py"),
)

IMPLEMENTATION_COMMIT = "1aa1cafc4abab6dd61f89e983b7ed11c7248764e"
CONFIRMATION_EXECUTION_COMMIT = "7f714622653185b0eda0519716ce633f79edf1c3"
CONFIRMATION_RECORD_COMMIT = "e080bc32f54a5bb149115331c1e742fdff36c4f0"
POOL_FREEZE_COMMIT = "185d985cdaf680ecd5fe9d8939fec8e8d0902c84"
POOL_RESULT_COMMIT = "ce0c0f7c6631c12f11d0d61a82bb00fbafe779c0"
CANDIDATE_RUN_ID = "e46d1ca38e0746209e049970d3dd2ab6"
REPLACED_RUN_ID = "d107ea874ebe4dbe8094694141a162b6"

DEFAULT_JSON_OUTPUT = Path(
    "artifacts/issue504-missingness-augmentation-final-classification.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/research/missingness-augmentation-final-classification.md"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="기존 결과 파일을 현재 저장소 근거와 다시 대조한다.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError(f"JSON 최상위 값이 사전이 아니다: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError(f"YAML 최상위 값이 사전이 아니다: {path}")
    return value


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}이 다르다: {actual!r} != {expected!r}")


def _assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-15:
        raise AssertionError(f"{label}이 다르다: {actual} != {expected}")


def _git_blob(commit: str, path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _path_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AssertionError(f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def _commit_path_record(commit: str, path: Path) -> dict[str, str]:
    return {
        "commit": commit,
        "path": str(path),
        "git_blob_sha1": _git_blob(commit, path),
        "url": f"{REPOSITORY_URL}/blob/{commit}/{path}",
    }


def _validate_diagnostic(record: dict[str, Any]) -> None:
    _assert_equal(record["purpose"], "execution-boundary-diagnostic-only", "진단 목적")
    _assert_equal(record["improvement_judgment"], False, "진단 개선 판정 여부")
    _assert_equal(record["original_compatibility"]["passed"], True, "기존 경로 동등성")
    _assert_equal(record["pzero_equivalence"]["passed"], True, "p=0 동등성")
    _assert_close(
        record["pzero_equivalence"]["oof_max_abs_difference"],
        0.0,
        "p=0 OOF 최대 절대 차이",
    )
    _assert_close(
        record["pzero_equivalence"]["test_max_abs_difference"],
        0.0,
        "p=0 시험 예측 최대 절대 차이",
    )
    _assert_equal(record["mask_replay"]["passed"], True, "마스크 재현")
    _assert_equal(
        record["config_sha256"],
        file_sha256(CONFIG_PATHS["baseline"]),
        "기준 설정 해시",
    )
    if not all(record["assertions"].values()):
        raise AssertionError("학습 행 경계 진단 단언 중 실패가 있다.")


def _validate_screen(record: dict[str, Any]) -> None:
    _assert_equal(record["decision"]["status"], "pass", "시드 42 관문")
    if record["comparison"]["missingness_augmented_minus_tripled"] < 0:
        raise AssertionError("시드 42 직접 차이가 음수다.")
    _assert_equal(record["decision"]["all_folds_positive"], True, "시드 42 분할 부호")
    _assert_equal(
        record["training_row_integrity"]["all_assertions_passed"],
        True,
        "시드 42 학습 행 무결성",
    )
    _assert_equal(
        record["training_row_integrity"]["validation_and_test_not_augmented"],
        True,
        "시드 42 검증 및 시험 비증강",
    )
    for arm in ("original", "tripled", "missingness_augmented"):
        _assert_equal(
            record["freeze"]["config_sha256"][arm],
            file_sha256(CONFIG_PATHS[arm]),
            f"시드 42 {arm} 설정 해시",
        )
        _assert_equal(record["runs"][arm]["status"], "FINISHED", f"시드 42 {arm} 상태")
        _assert_equal(record["runs"][arm]["git_dirty"], False, f"시드 42 {arm} 작업 트리")


def _validate_confirmation(record: dict[str, Any]) -> None:
    decision = record["decision"]
    _assert_equal(decision["status"], "pass", "세 시드 관문")
    if not all(decision["gates"].values()):
        raise AssertionError("세 시드 확인 관문 중 실패가 있다.")
    _assert_equal(decision["observed"]["seed_wins"], 3, "시드 승수")
    _assert_equal(decision["observed"]["fold_wins"], 5, "분할 승수")
    _assert_equal(record["freeze"]["seeds"], [42, 43, 44], "확인 시드")
    _assert_equal(record["freeze"]["folds"], [0, 1, 2, 3, 4], "확인 분할")
    _assert_equal(
        record["freeze"]["execution_commit"],
        CONFIRMATION_EXECUTION_COMMIT,
        "확인 실행 커밋",
    )
    _assert_equal(record["freeze"]["execution_tree_clean"], True, "확인 작업 트리")
    _assert_equal(record["freeze"]["settings_changed_after_results"], False, "사후 설정 변경")
    _assert_equal(record["freeze"]["public_repleaf_used"], False, "공개 배열 사용")
    _assert_equal(record["freeze"]["kaggle_submission_uploaded"], False, "캐글 제출")
    integrity = record["training_row_integrity"]
    for key in (
        "all_assertions_passed",
        "all_mask_hashes_unique",
        "pair_identity_matches",
        "target_encoding_canary_valid",
        "validation_and_test_not_augmented",
    ):
        _assert_equal(integrity[key], True, f"세 시드 무결성 {key}")
    _assert_equal(integrity["mask_count"], 30, "세 시드 마스크 개수")
    failed = record["failed_attempt"]
    _assert_equal(failed["status"], "FAILED", "실패 실행 상태")
    _assert_equal(failed["results_used"], False, "실패 실행 사용 여부")
    input_records = []
    for arm in ("original", "tripled", "missingness_augmented"):
        run = record["runs"][arm]
        _assert_equal(run["status"], "FINISHED", f"세 시드 {arm} 상태")
        _assert_equal(run["git_dirty"], False, f"세 시드 {arm} 작업 트리")
        _assert_equal(run["git_commit"], CONFIRMATION_EXECUTION_COMMIT, f"세 시드 {arm} 커밋")
        _assert_equal(len(run["seed_auc"]), 3, f"세 시드 {arm} 시드 점수 개수")
        _assert_equal(len(run["fold_auc"]), 5, f"세 시드 {arm} 분할 점수 개수")
        config_name = f"{run['experiment']}.yaml"
        _assert_equal(
            run["artifact_sha256"][config_name],
            file_sha256(CONFIG_PATHS[arm]),
            f"세 시드 {arm} 실행 설정 해시",
        )
        for artifact in ("oof.parquet", "test_pred.parquet"):
            if len(run["artifact_sha256"][artifact]) != 64:
                raise AssertionError(f"세 시드 {arm} {artifact} 해시가 올바르지 않다.")
        _assert_equal(
            run["target_encoding_importance_canary"]["ok"],
            True,
            f"세 시드 {arm} 피처 중요도 카나리아",
        )
        input_records.append(run["input_sha256"])
    if any(item != input_records[0] for item in input_records[1:]):
        raise AssertionError("세 비교 팔의 입력 해시가 다르다.")
    _assert_equal(input_records[0], record["freeze"]["input_sha256"], "동결 입력 해시")


def _validate_pool(
    freeze: dict[str, Any], result: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _assert_equal(
        result["pre_result_freeze"]["sha256"],
        file_sha256(POOL_FREEZE_PATH),
        "후보 풀 사전 동결 해시",
    )
    _assert_equal(
        result["pre_result_freeze"]["commit"],
        POOL_FREEZE_COMMIT,
        "후보 풀 사전 동결 커밋",
    )
    _assert_equal(result["candidate"]["run_id"], CANDIDATE_RUN_ID, "후보 실행")
    _assert_equal(result["eligibility"]["entry_floor_passed"], True, "후보 진입 하한")
    if not all(
        result["eligibility"][key]
        for key in (
            "finished",
            "clean_execution_commit",
            "input_hashes_match",
            "training_row_integrity",
            "target_encoding_canary",
        )
    ):
        raise AssertionError("후보 자격 검사 중 실패가 있다.")
    for label in ("admission", "replacement"):
        judgment = result[label]
        if not 0 < judgment["delta"] <= POOL_EQUIVALENCE_BAND_UPPER:
            raise AssertionError(f"{label} 결과가 양의 경계 기여가 아니다.")
        _assert_equal(judgment["outer_fold_wins"], 5, f"{label} 바깥 분할 승수")
        _assert_equal(judgment["boundary_contribution"], True, f"{label} 경계 기여")
        _assert_equal(judgment["state"], "adopted", f"{label} 판정 상태")
    duplicate = result["duplicate_check"]
    _assert_equal(duplicate["replacement_target_run_id"], REPLACED_RUN_ID, "원자 교체 대상")
    _assert_equal(
        duplicate["candidate_vs_remaining_all_below_threshold"],
        True,
        "원자 교체 뒤 중복",
    )
    final = result["final"]
    _assert_equal(final["registered"], True, "최종 후보 풀 등록")
    _assert_equal(final["decision"], "register_by_atomic_replacement", "최종 등록 방식")
    _assert_equal(final["candidate_run_id"], CANDIDATE_RUN_ID, "최종 후보 실행")
    _assert_equal(final["replaced_run_id"], REPLACED_RUN_ID, "최종 교체 실행")
    _assert_equal(final["pool_sha256"], file_sha256(POOL_PATH), "현재 후보 풀 해시")
    _assert_equal(
        final["refit_plan_sha256"],
        file_sha256(REFIT_PLAN_PATH),
        "현재 전체 자료 재학습 계획 해시",
    )
    _assert_equal(final["current_submission_assembly_changed"], False, "현재 제출 결합 변경")
    _assert_equal(
        freeze["decision_contract"]["version"],
        "candidate-pool-v2",
        "후보 풀 계약 판본",
    )
    _assert_equal(
        freeze["decision_contract"]["submission_assembly"],
        False,
        "후보 풀 제출 결합 범위",
    )
    pool = _load_yaml(POOL_PATH)
    run_ids = [member["run_id"] for member in pool["members"]]
    _assert_equal(run_ids.count(CANDIDATE_RUN_ID), 1, "현재 후보 실행 개수")
    _assert_equal(REPLACED_RUN_ID in run_ids, False, "교체된 실행 잔존 여부")
    return duplicate, final


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    config_name = f"{run['experiment']}.yaml"
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "experiment": run["experiment"],
        "git_commit": run["git_commit"],
        "git_dirty": run["git_dirty"],
        "auc_oof": run["auc_oof"],
        "auc_oof_weighted": run["auc_oof_weighted"],
        "seed_auc": run["seed_auc"],
        "fold_auc": run["fold_auc"],
        "input_sha256": run["input_sha256"],
        "config_sha256": run["artifact_sha256"][config_name],
        "oof_sha256": run["artifact_sha256"]["oof.parquet"],
        "seed_oof_sha256": run["seed_oof_sha256"],
        "test_prediction_sha256": run["artifact_sha256"]["test_pred.parquet"],
        "submission_artifact_sha256": run["artifact_sha256"]["submission.csv"],
        "training_row_evidence_sha256": run["artifact_sha256"][
            "training_row_evidence.json"
        ],
        "target_encoding_importance_canary": run[
            "target_encoding_importance_canary"
        ],
    }


def _build_record(recorded_at_utc: str) -> dict[str, Any]:
    diagnostic = _load_json(DIAGNOSTIC_PATH)
    screen = _load_json(SCREEN_PATH)
    confirmation = _load_json(CONFIRMATION_PATH)
    pool_freeze = _load_json(POOL_FREEZE_PATH)
    pool_result = _load_json(POOL_RESULT_PATH)

    _validate_diagnostic(diagnostic)
    _validate_screen(screen)
    _validate_confirmation(confirmation)
    duplicate, final_pool = _validate_pool(pool_freeze, pool_result)

    upstream = {
        "issue500_boundary_diagnostic": _path_record(DIAGNOSTIC_PATH),
        "issue501_seed42_screen": _path_record(SCREEN_PATH),
        "issue502_three_seed_confirmation": _path_record(CONFIRMATION_PATH),
        "issue503_pre_result_freeze": _path_record(POOL_FREEZE_PATH),
        "issue503_pool_contribution": _path_record(POOL_RESULT_PATH),
        "public_repleaf_audit": _path_record(REPLEAF_REPORT_PATH),
    }
    configs = {name: _path_record(path) for name, path in CONFIG_PATHS.items()}
    code = [_path_record(path) for path in CODE_PATHS]

    seed42_direct_delta = screen["comparison"][
        "missingness_augmented_minus_tripled"
    ]
    confirmation_direct_delta = confirmation["comparison"][
        "missingness_augmented_minus_tripled"
    ]
    admission = pool_result["admission"]
    replacement = pool_result["replacement"]
    _assert_close(
        admission["delta"],
        0.00002178780846240347,
        "일반 추가 최종 차이",
    )
    _assert_close(
        replacement["delta"],
        0.000019363665046956413,
        "원자 교체 최종 차이",
    )

    return {
        "schema_version": 1,
        "issue": {"number": 504, "url": ISSUE_URL},
        "map": {"number": 499, "url": MAP_URL},
        "recorded_at_utc": recorded_at_utc,
        "final_classification": {
            "code": "boundary_contribution_member",
            "label_ko": "경계 기여 구성원",
            "candidate_pool_member": True,
            "registration": "atomic_replacement",
            "candidate_run_id": CANDIDATE_RUN_ID,
            "replaced_run_id": REPLACED_RUN_ID,
            "reason": (
                "세 시드 확인을 통과했고 일반 추가와 원자 교체가 모두 양수이면서 "
                "성능 동등 대역 안의 5/5 분할 기여를 냈다."
            ),
        },
        "traceability": {
            "upstream_records": upstream,
            "configurations": configs,
            "current_code": code,
            "history": {
                "implementation": {
                    "commit": IMPLEMENTATION_COMMIT,
                    "paths": [
                        _commit_path_record(
                            IMPLEMENTATION_COMMIT,
                            Path("src/pipeline/training_rows.py"),
                        ),
                        _commit_path_record(
                            IMPLEMENTATION_COMMIT,
                            Path("src/pipeline/cv_seed_execution.py"),
                        ),
                    ],
                },
                "confirmation_execution": {
                    "commit": CONFIRMATION_EXECUTION_COMMIT,
                    "paths": [
                        _commit_path_record(
                            CONFIRMATION_EXECUTION_COMMIT,
                            CONFIG_PATHS["missingness_augmented"],
                        ),
                        _commit_path_record(
                            CONFIRMATION_EXECUTION_COMMIT,
                            Path("src/pipeline/seed_reuse.py"),
                        ),
                    ],
                },
                "confirmation_record": {"commit": CONFIRMATION_RECORD_COMMIT},
                "pool_pre_result_freeze": {"commit": POOL_FREEZE_COMMIT},
                "pool_result": {"commit": POOL_RESULT_COMMIT},
            },
        },
        "fixed_contract": {
            "baseline": {
                "name": "exp117_ag25_gbm_r21",
                **configs["baseline"],
            },
            "arms": {
                arm: {
                    "configuration": configs[arm],
                    "training_rows": screen["freeze"]["arms"][arm][
                        "training_rows"
                    ],
                }
                for arm in ("original", "tripled", "missingness_augmented")
            },
            "common_model": screen["freeze"]["arms"]["original"]["model"],
            "common_features": screen["freeze"]["arms"]["original"]["features"],
            "common_config_sha256_excluding_name_and_training_rows": screen[
                "freeze"
            ]["common_config_sha256_excluding_name_and_training_rows"],
            "training_rows_contract_sha256": screen["freeze"][
                "training_rows_contract_sha256"
            ],
            "raw_columns": screen["freeze"]["raw_columns"],
            "seeds": confirmation["freeze"]["seeds"],
            "folds": confirmation["freeze"]["folds"],
            "input_sha256": confirmation["freeze"]["input_sha256"],
            "model_dependencies": confirmation["freeze"]["model_dependencies"],
            "execution_environment": confirmation["freeze"][
                "execution_environment"
            ],
            "mask_contract": {
                "scope": "원자료 12열의 기존 관측 셀",
                "probability": 0.25,
                "replica_count": 2,
                "determinants": ["model_seed", "outer_fold", "replica_number"],
                "validation_augmented": False,
                "test_augmented": False,
            },
            "post_result_changes": {
                "replica_count": False,
                "missingness_probability": False,
                "baseline_model": False,
                "feature_plan": False,
                "training_length": False,
                "decision_thresholds": False,
            },
        },
        "execution_boundary_diagnostic": {
            "purpose": diagnostic["purpose"],
            "original_compatibility": diagnostic["original_compatibility"],
            "pzero_equivalence": diagnostic["pzero_equivalence"],
            "mask_replay": diagnostic["mask_replay"],
            "assertions": diagnostic["assertions"],
            "improvement_judgment": False,
        },
        "seed42_screen": {
            "execution_commit": screen["freeze"]["execution_commit"],
            "seed": screen["freeze"]["seed"],
            "runs": {
                arm: {
                    "run_id": screen["runs"][arm]["run_id"],
                    "auc_oof": screen["runs"][arm]["auc_oof"],
                    "auc_oof_weighted": screen["runs"][arm]["auc_oof_weighted"],
                    "fold_auc": screen["runs"][arm]["fold_auc"],
                    "oof_sha256": screen["runs"][arm]["artifact_sha256"][
                        "oof.parquet"
                    ],
                    "test_prediction_sha256": screen["runs"][arm][
                        "artifact_sha256"
                    ]["test_pred.parquet"],
                }
                for arm in ("original", "tripled", "missingness_augmented")
            },
            "comparison": screen["comparison"],
            "decision": screen["decision"],
            "training_row_integrity": {
                key: value
                for key, value in screen["training_row_integrity"].items()
                if key != "mask_records"
            },
            "mask_sha256": [
                item["mask_sha256"]
                for item in screen["training_row_integrity"]["mask_records"]
            ],
            "training_limit_observation": screen["training_limit_observation"],
        },
        "three_seed_confirmation": {
            "execution_commit": confirmation["freeze"]["execution_commit"],
            "runs": {
                arm: _run_summary(confirmation["runs"][arm])
                for arm in ("original", "tripled", "missingness_augmented")
            },
            "comparison": confirmation["comparison"],
            "decision": confirmation["decision"],
            "training_row_integrity": confirmation["training_row_integrity"],
            "failed_attempt": confirmation["failed_attempt"],
        },
        "candidate_pool_contribution": {
            "pre_result_freeze": pool_result["pre_result_freeze"],
            "candidate": {
                "run_id": pool_result["candidate"]["run_id"],
                "execution": pool_result["candidate"]["execution"],
                "configuration": pool_result["candidate"]["configuration"],
                "input_sha256": pool_result["candidate"]["input_sha256"],
                "prediction_artifacts": pool_result["candidate"][
                    "prediction_artifacts"
                ],
                "artifact_manifest_sha256": pool_result["candidate"][
                    "artifact_manifest_sha256"
                ],
                "artifact_count": pool_result["candidate"]["artifact_count"],
            },
            "decision_contract": pool_freeze["decision_contract"],
            "eligibility": pool_result["eligibility"],
            "admission": admission,
            "duplicate_check": {
                key: value
                for key, value in duplicate.items()
                if key != "all_correlations"
            },
            "replacement": replacement,
            "final": final_pool,
        },
        "decision_sequence": [
            {
                "order": 1,
                "stage": "execution_boundary_diagnostic",
                "executed": True,
                "result": "pass_without_improvement_judgment",
            },
            {
                "order": 2,
                "stage": "seed42_stop_gate",
                "executed": True,
                "threshold": 0.0,
                "observed": seed42_direct_delta,
                "result": "pass",
            },
            {
                "order": 3,
                "stage": "three_seed_confirmation",
                "executed": True,
                "threshold": confirmation["decision"]["rule"],
                "observed": confirmation["decision"]["observed"],
                "result": "pass",
            },
            {
                "order": 4,
                "stage": "candidate_pool_admission",
                "executed": True,
                "threshold": "delta > 0",
                "observed": {
                    "delta": admission["delta"],
                    "outer_fold_wins": admission["outer_fold_wins"],
                },
                "result": "boundary_contribution",
            },
            {
                "order": 5,
                "stage": "atomic_replacement",
                "executed": True,
                "threshold": "delta > 0 and remaining correlations < 0.998",
                "observed": {
                    "delta": replacement["delta"],
                    "outer_fold_wins": replacement["outer_fold_wins"],
                    "nearest_remaining_spearman": duplicate[
                        "nearest_after_replacement"
                    ]["spearman"],
                },
                "result": "register_boundary_contribution_member",
            },
        ],
        "conditional_stages": {
            "skipped": [],
            "reason": "각 선행 관문이 통과되어 계획된 조건부 판정 단계를 모두 실행했다.",
        },
        "scope_and_provenance": {
            "public_repleaf": {
                "role": "evidence_comparison_only",
                "audit": upstream["public_repleaf_audit"],
                "used_for_training": False,
                "used_for_combination": False,
                "registered_as_candidate": False,
                "used_for_submission": False,
            },
            "current_313_submission_assembly_changed": False,
            "kaggle_upload_performed": False,
            "out_of_scope_training_length_50000_executed": False,
            "failed_attempt_results_used": False,
        },
        "map_destination": {
            "reproducible_record_complete": True,
            "final_classification_complete": True,
            "candidate_pool_registration_complete": True,
            "submission_scope_preserved": True,
            "destination_reached": True,
        },
    }


def _run_table(lines: list[str], record: dict[str, Any]) -> None:
    runs = record["three_seed_confirmation"]["runs"]
    comparison = record["three_seed_confirmation"]["comparison"]
    lines.extend(
        [
            "| 학습군 | 세 시드 OOF AUC | 일반 기준군 대비 | 가중 OOF AUC |",
            "| --- | ---: | ---: | ---: |",
            f"| 일반 기준군 | `{runs['original']['auc_oof']:.10f}` | 기준 | `{runs['original']['auc_oof_weighted']:.10f}` |",
            f"| 3배 행 대조군 | `{runs['tripled']['auc_oof']:.10f}` | `{comparison['delta_vs_original']['tripled']:+.10f}` | `{runs['tripled']['auc_oof_weighted']:.10f}` |",
            f"| 결측 증강군 | `{runs['missingness_augmented']['auc_oof']:.10f}` | `{comparison['delta_vs_original']['missingness_augmented']:+.10f}` | `{runs['missingness_augmented']['auc_oof_weighted']:.10f}` |",
        ]
    )


def _report(record: dict[str, Any]) -> str:
    contract = record["fixed_contract"]
    diagnostic = record["execution_boundary_diagnostic"]
    screen = record["seed42_screen"]
    confirmation = record["three_seed_confirmation"]
    runs = confirmation["runs"]
    comparison = confirmation["comparison"]
    integrity = confirmation["training_row_integrity"]
    pool = record["candidate_pool_contribution"]
    admission = pool["admission"]
    replacement = pool["replacement"]
    duplicate = pool["duplicate_check"]
    final = pool["final"]
    environment = contract["execution_environment"]
    dependencies = contract["model_dependencies"]
    paths = record["traceability"]["upstream_records"]

    lines = [
        "# 독립 결측 증강 최종 근거와 분류",
        "",
        f"이 문서는 [독립 결측 증강 실험의 근거와 최종 판정을 확정한다]({ISSUE_URL})의 통합 연구 기록이다.",
        f"판정 지도는 [독립 결측 증강을 강한 나무 모형에 적용해 채택 여부를 판정한다]({MAP_URL})이다.",
        "",
        "## 최종 분류",
        "",
        "독립 결측 증강 후보의 최종 분류는 **경계 기여 구성원**이다.",
        f"세 시드 확인에서 3배 행 대조군보다 `{comparison['missingness_augmented_minus_tripled']:+.12f}` 높았고 시드 `3/3`, 평균 분할 `5/5`가 양수였다.",
        f"후보 풀 일반 추가는 `{admission['delta']:+.12f}`, 기존 `exp117_ag25_gbm_r21`과의 원자 교체는 `{replacement['delta']:+.12f}`였으며 두 비교 모두 바깥 분할 `5/5`가 양수였다.",
        f"두 후보 풀 차이는 성능 동등 대역 상한 `+{POOL_EQUIVALENCE_BAND_UPPER:.12f}` 이내이므로 경계 기여로 분류한다.",
        f"기존 실행 `{final['replaced_run_id']}`을 결측 증강 평균본 `{final['candidate_run_id']}`으로 원자 교체해 후보 풀에 등록했다.",
        "현재 313개 제출 결합과 제출물은 변경하지 않았다.",
        "",
        "## 고정 비교 계약",
        "",
        f"기준 모형은 `exp117_ag25_gbm_r21`이며 설정 파일 SHA-256은 `{contract['baseline']['sha256']}`이다.",
        "세 비교 팔은 모형 설정, 피처 계획, 입력 자료, 고정 분할과 조기 종료 조건을 공유하고 학습 행 구성만 다르게 고정했다.",
        "",
        "| 학습군 | 복제본 수 | 관측 셀 추가 결측 확률 | 설정 SHA-256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for arm, label in (
        ("original", "일반 기준군"),
        ("tripled", "3배 행 대조군"),
        ("missingness_augmented", "결측 증강군"),
    ):
        arm_record = contract["arms"][arm]
        rows = arm_record["training_rows"]
        lines.append(
            f"| {label} | `{rows['replica_count']}` | `{rows['observed_cell_mask_probability']}` | `{arm_record['configuration']['sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"공통 설정 해시는 `{contract['common_config_sha256_excluding_name_and_training_rows']}`이고 학습 행 계약 해시는 `{contract['training_rows_contract_sha256']}`이다.",
            "결측 처리는 식별자와 목표값을 제외한 원자료 12열의 기존 관측 셀에만 적용했다.",
            "복제본 두 개의 마스크는 모형 시드, 바깥쪽 분할 번호와 복제본 번호로 결정했다.",
            "검증 및 시험 자료에는 증강을 적용하지 않았다.",
            f"학습, 시험, 고정 분할 SHA-256은 각각 `{contract['input_sha256']['train']}`, `{contract['input_sha256']['test']}`, `{contract['input_sha256']['folds']}`이다.",
            f"확인 시드는 `{contract['seeds']}`, 고정 분할은 `{contract['folds']}`다.",
            "",
            "## 실행 경계 진단",
            "",
            "축소 한 분할 진단은 개선 판정에 쓰지 않고 실행 경계와 결정성만 확인했다.",
            f"기존 기본 경로와 명시적 일반 기준군의 OOF 및 시험 예측 최대 절대 차이는 각각 `{diagnostic['original_compatibility']['oof_max_abs_difference']}`, `{diagnostic['original_compatibility']['test_max_abs_difference']}`였다.",
            f"결측 확률 0인 증강군과 3배 행 대조군의 OOF 및 시험 예측 최대 절대 차이는 각각 `{diagnostic['pzero_equivalence']['oof_max_abs_difference']}`, `{diagnostic['pzero_equivalence']['test_max_abs_difference']}`였다.",
            f"마스크 재생 진단의 실측 추가 결측률은 `{diagnostic['mask_replay']['actual_added_missing_rate']:.10f}`였고 같은 좌표의 마스크 해시가 다시 일치했다.",
            "부모 행 순서, 목표값, 바깥쪽 분할 상속, 상태 적합 제외, 기존 결측 보존과 복제본 신원 비노출 단언을 모두 통과했다.",
            "",
            "## 판정 순서와 관문",
            "",
            "| 순서 | 단계 | 사전 고정 관문 | 관측값 | 판정 |",
            "| ---: | --- | --- | --- | --- |",
            "| 1 | 실행 경계 진단 | 동등성 및 결정성 | 모든 단언 통과 | 개선 판정 없이 통과 |",
            f"| 2 | 시드 42 중단 관문 | 직접 차이 `>= 0` | `{screen['comparison']['missingness_augmented_minus_tripled']:+.10f}`, 분할 `5/5` 양수 | 통과 |",
            f"| 3 | 세 시드 확인 | 평균 `>= +0.00002`, 시드 `>= 2/3`, 적용 시 분할 `>= 3/5` | `{comparison['missingness_augmented_minus_tripled']:+.10f}`, 시드 `3/3`, 분할 `5/5` | 통과 |",
            f"| 4 | 후보 풀 일반 추가 | 중첩 OOF 차이 `> 0` | `{admission['delta']:+.12f}`, 분할 `5/5` | 경계 기여 |",
            f"| 5 | 중복 처리와 원자 교체 | 차이 `> 0`, 남은 상관 `< 0.998` | `{replacement['delta']:+.12f}`, 분할 `5/5`, 최대 상관 `{duplicate['nearest_after_replacement']['spearman']:.12f}` | 경계 기여 구성원 등록 |",
            "",
            "모든 선행 관문을 통과했으므로 계획된 조건부 단계 중 실행하지 않은 단계는 없다.",
            "50,000회 학습 길이 비교, 현재 313개 제출 결합 변경과 Kaggle 업로드는 조건부 판정 단계가 아니라 이 지도의 명시적 범위 밖이다.",
            "",
            "## 세 시드 전체 결과",
            "",
        ]
    )
    _run_table(lines, record)
    lines.extend(
        [
            "",
            f"결측 증강군과 3배 행 대조군의 직접 차이는 `{comparison['missingness_augmented_minus_tripled']:+.12f}`다.",
            "",
            "### 시드별 점수",
            "",
            "| 시드 | 일반 기준군 | 3배 행 대조군 | 결측 증강군 | 직접 차이 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in contract["seeds"]:
        key = str(seed)
        lines.append(
            f"| {seed} | `{runs['original']['seed_auc'][key]:.10f}` | `{runs['tripled']['seed_auc'][key]:.10f}` | `{runs['missingness_augmented']['seed_auc'][key]:.10f}` | `{comparison['seed_missingness_augmented_minus_tripled'][key]:+.10f}` |"
        )
    lines.extend(
        [
            "",
            "### 분할별 점수",
            "",
            "| 분할 | 일반 기준군 | 3배 행 대조군 | 결측 증강군 | 직접 차이 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in contract["folds"]:
        key = str(fold)
        lines.append(
            f"| {fold} | `{runs['original']['fold_auc'][key]:.10f}` | `{runs['tripled']['fold_auc'][key]:.10f}` | `{runs['missingness_augmented']['fold_auc'][key]:.10f}` | `{comparison['fold_missingness_augmented_minus_tripled'][key]:+.10f}` |"
        )
    lines.extend(
        [
            "",
            "### 검증 행의 기존 결측 수별 점수",
            "",
            "| 기존 결측 수 | 행 수 | 일반 기준군 | 3배 행 대조군 | 결측 증강군 | 직접 차이 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket in comparison["missing_count_buckets"]:
        auc = bucket["auc"]
        lines.append(
            f"| {bucket['bucket']} | {bucket['row_count']:,} | `{auc['original']:.10f}` | `{auc['tripled']:.10f}` | `{auc['missingness_augmented']:.10f}` | `{bucket['missingness_augmented_minus_tripled']:+.10f}` |"
        )
    lines.extend(
        [
            "",
            "## 실행 무결성과 환경",
            "",
            f"결측 증강군은 원래 관측된 학습 셀 {integrity['missingness_augmented_eligible_observed_cells']:,}개 중 {integrity['missingness_augmented_added_missing_cells']:,}개를 추가로 비워 실측 비율 `{integrity['missingness_augmented_added_missing_rate']:.8%}`를 만들었다.",
            f"시드, 분할과 복제본별 마스크 `{integrity['mask_count']}`개는 모두 서로 다른 결정적 해시를 가졌다.",
            "3배 행 대조군과 결측 증강군의 부모 행 순서, 원본 행 번호 순서, 목표값 순서와 전체 행 수가 일치했다.",
            "목표값 통계 부호화는 복제본을 통계 산출에서 제외하고 부모 분할, 목표값과 위약 잡음을 물려받으며 증강 원자료의 결측 마스크만 다시 계산했다.",
            "세 학습군의 피처 중요도 위약 카나리아도 모두 통과했다.",
            f"정식 확인은 `{environment['hostname']}`의 `{environment['platform']}`에서 CPU `{environment['cpu_count']}`개와 Python `{environment['python']}`으로 실행했다.",
            f"LightGBM 판본은 `{dependencies['project_packages']['lightgbm']}`, scikit-learn 판본은 `{dependencies['project_packages']['scikit-learn']}`, 의존성 잠금 파일 SHA-256은 `{dependencies['uv_lock_sha256']}`이다.",
            "원격 계산 자원과 GPU는 사용하지 않았다.",
            "",
            "| 학습군 | 중앙 MLflow 실행 ID | OOF SHA-256 | 시험 예측 SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for arm, label in (
        ("original", "일반 기준군"),
        ("tripled", "3배 행 대조군"),
        ("missingness_augmented", "결측 증강군"),
    ):
        run = runs[arm]
        lines.append(
            f"| {label} | `{run['run_id']}` | `{run['oof_sha256']}` | `{run['test_prediction_sha256']}` |"
        )
    failed = confirmation["failed_attempt"]
    lines.extend(
        [
            "",
            f"첫 원본 시도 `{failed['run_id']}`은 재사용 피처 선언 순서 오류로 최종 평가에서 중단됐다.",
            "오류를 실제 실행으로 재현한 뒤 현재 피처 계획의 선언 순서를 보존하도록 고쳤고 실패 실행 결과는 판정에 사용하지 않았다.",
            "이 수정은 성공한 정식 결과를 확인하기 전에 이루어졌으며 복제 수, 결측 확률, 기준 모형, 피처 계획, 학습 길이와 판정 문턱을 바꾸지 않았다.",
            "",
            "## 후보 풀 기여와 등록",
            "",
            f"결과 확인 전 동결 기록은 커밋 `{pool['pre_result_freeze']['commit']}`에 있으며 SHA-256은 `{pool['pre_result_freeze']['sha256']}`이다.",
            f"후보 실행 산출물 `{pool['candidate']['artifact_count']}`개의 전체 목록 해시는 `{pool['candidate']['artifact_manifest_sha256']}`다.",
            f"일반 추가 중첩 OOF는 `{admission['before']['best_auc']:.12f}`에서 `{admission['after']['best_auc']:.12f}`로 올랐고 차이는 `{admission['delta']:+.12f}`다.",
            f"후보는 기존 `exp117_ag25_gbm_r21`과만 스피어만 순위 상관 `{duplicate['duplicates'][0]['spearman']:.12f}`로 중복됐다.",
            f"원자 교체 중첩 OOF는 `{replacement['before']['best_auc']:.12f}`에서 `{replacement['after']['best_auc']:.12f}`로 올랐고 차이는 `{replacement['delta']:+.12f}`다.",
            f"원자 교체 뒤 남은 구성원과의 최대 상관은 `{duplicate['nearest_after_replacement']['spearman']:.12f}`로 문턱 아래다.",
            f"최종 후보 풀은 `{final['pool_member_count']}`개이며 SHA-256은 `{final['pool_sha256']}`이다.",
            f"전체 자료 재학습 계획은 같은 `{final['refit_plan_member_count']}`개 구성원과 총 `{final['refit_plan_total_runs']}`회 실행으로 검증됐고 SHA-256은 `{final['refit_plan_sha256']}`이다.",
            "",
            "## 공개 RepLeaf와 범위 보존",
            "",
            f"공개 RepLeaf 배열의 감사 기록은 `{REPLEAF_REPORT_PATH}`에 있으며 SHA-256은 `{paths['public_repleaf_audit']['sha256']}`이다.",
            "공개 배열은 선행 근거 대조에만 사용했고 이 지도의 학습, 결합, 후보 등록과 제출에는 사용하지 않았다.",
            "정식 판정은 저장소의 자체 학습 실행만 사용했다.",
            "현재 313개 제출 결합 변경과 Kaggle 업로드는 수행하지 않았다.",
            "결과 확인 뒤 복제 수, 결측 확률, 기준 모형, 피처 계획, 학습 길이와 판정 문턱을 바꾸지 않았다.",
            "",
            "## 재현 경로",
            "",
            f"실행 경계 진단은 `{paths['issue500_boundary_diagnostic']['path']}`이며 SHA-256은 `{paths['issue500_boundary_diagnostic']['sha256']}`이다.",
            f"시드 42 선별 기록은 `{paths['issue501_seed42_screen']['path']}`이며 SHA-256은 `{paths['issue501_seed42_screen']['sha256']}`이다.",
            f"세 시드 확인 기록은 `{paths['issue502_three_seed_confirmation']['path']}`이며 SHA-256은 `{paths['issue502_three_seed_confirmation']['sha256']}`이다.",
            f"후보 풀 사전 동결은 `{paths['issue503_pre_result_freeze']['path']}`이며 SHA-256은 `{paths['issue503_pre_result_freeze']['sha256']}`이다.",
            f"후보 풀 판정은 `{paths['issue503_pool_contribution']['path']}`이며 SHA-256은 `{paths['issue503_pool_contribution']['sha256']}`이다.",
            f"세 팔 실행 코드는 커밋 `{IMPLEMENTATION_COMMIT}`, 세 시드 정식 실행 코드는 커밋 `{CONFIRMATION_EXECUTION_COMMIT}`, 후보 풀 결과는 커밋 `{POOL_RESULT_COMMIT}`에서 추적할 수 있다.",
            f"이 통합 기록의 기계 판독 원본은 `{DEFAULT_JSON_OUTPUT}`이다.",
            "",
            "## 지도 목적지",
            "",
            "고정 비교 계약, 실행 경계, OOF와 시험 예측, 분할 및 시드 점수, 결측 수 구간, 카나리아, 환경, 후보 풀 기여와 최종 등록을 같은 저장소 이력에서 추적할 수 있다.",
            "따라서 독립 결측 증강을 강한 나무 모형에 적용해 채택 여부를 판정한다는 지도 목적지에 도달했다.",
            "최종 분류는 후보 풀에 원자 교체로 등록된 경계 기여 구성원이다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if args.verify:
        if not args.json_output.is_file() or not args.report_output.is_file():
            raise SystemExit("검증할 JSON 또는 Markdown 결과 파일이 없다.")
        expected = _load_json(args.json_output)
        actual = _build_record(expected["recorded_at_utc"])
        if actual != expected:
            raise SystemExit("기계 판독 결과와 현재 저장소 근거가 다르다.")
        if args.report_output.read_text() != _report(actual):
            raise SystemExit("사람이 읽는 결과와 현재 저장소 근거가 다르다.")
        print(
            f"결과 검증 통과: {args.json_output} ({file_sha256(args.json_output)}), "
            f"{args.report_output} ({file_sha256(args.report_output)})"
        )
        return

    for path in (args.json_output, args.report_output):
        if path.exists():
            raise SystemExit(f"변경 불가 결과 파일이 이미 있다: {path}")
    record = _build_record(datetime.now(UTC).isoformat())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    args.report_output.write_text(_report(record))
    print(
        f"기계 판독 결과 생성: {args.json_output} "
        f"({file_sha256(args.json_output)})"
    )
    print(
        f"연구 기록 생성: {args.report_output} "
        f"({file_sha256(args.report_output)})"
    )


if __name__ == "__main__":
    main()
