"""이슈 503 후보 풀 기여 판정을 재검증해 JSON과 Markdown으로 기록한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.data import file_sha256
from pipeline.judgment import DUPLICATE_SPEARMAN, POOL_EQUIVALENCE_BAND_UPPER
from pipeline.refit_plan import RefitPlan


ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/503"
PREDECESSOR_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/502"
NEXT_ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/504"
CANDIDATE_RUN_ID = "e46d1ca38e0746209e049970d3dd2ab6"
REPLACED_RUN_ID = "d107ea874ebe4dbe8094694141a162b6"
PRE_RESULT_COMMIT = "185d985cdaf680ecd5fe9d8939fec8e8d0902c84"
FREEZE_PATH = Path("artifacts/issue503-missingness-candidate-pool-freeze.json")
PREDECESSOR_PATH = Path("artifacts/issue502-three-seed-missingness-confirmation.json")
ADMISSION_PATH = Path("artifacts/judgments/issue503-exp208-admission.yaml")
REPLACEMENT_PATH = Path(
    "artifacts/judgments/issue503-exp208-replacement-exp117.yaml"
)
POOL_PATH = Path("artifacts/pool.yaml")
CHAMPION_PATH = Path("artifacts/champion.yaml")
REFIT_PLAN_PATH = Path("artifacts/full-refit-plan.yaml")
DEFAULT_JSON_OUTPUT = Path(
    "artifacts/issue503-missingness-pool-contribution.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/research/missingness-augmentation-pool-contribution.md"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="기존 결과 파일을 현재 근거와 다시 대조한다.",
    )
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError(f"YAML 최상위 값이 사전이 아니다: {path}")
    return value


def _assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-15:
        raise AssertionError(f"{label}이 다르다: {actual} != {expected}")


def _judgment(path: Path) -> dict[str, Any]:
    record = _load_yaml(path)
    evidence_path = Path(record["evidence"]["path"])
    if file_sha256(evidence_path) != record["evidence"]["sha256"]:
        raise AssertionError(f"판정 근거 해시가 다르다: {evidence_path}")
    evidence = json.loads(evidence_path.read_text())
    snapshot = evidence["input_artifacts"]["evaluation_snapshot"]
    snapshot_path = Path(snapshot["path"])
    if file_sha256(snapshot_path) != snapshot["sha256"]:
        raise AssertionError(f"평가 입력 사본 해시가 다르다: {snapshot_path}")
    if record["contract_version"] != "candidate-pool-v2":
        raise AssertionError("후보 풀 판정 계약 판본이 candidate-pool-v2가 아니다.")
    if record["result"]["state"] != "adopted":
        raise AssertionError(f"판정이 채택 상태가 아니다: {path}")
    if record["change"]["candidate"]["run_id"] != CANDIDATE_RUN_ID:
        raise AssertionError(f"판정 후보 실행이 다르다: {path}")

    evaluation = evidence["evaluation"]
    comparison = evaluation["nested_oof_comparison"]
    recorded = record["nested_oof_comparison"]
    _assert_close(comparison["before_auc"], recorded["before"]["auc"], "제외 AUC")
    _assert_close(comparison["after_auc"], recorded["after"]["auc"], "포함 AUC")
    _assert_close(comparison["delta"], recorded["delta"], "전체 차이")
    if comparison["boundary_contribution"] != recorded["boundary_contribution"]:
        raise AssertionError("경계 기여 표시가 판정 기록과 근거에서 다르다.")

    strategies = []
    for name in record["frozen_input"]["registered_combiners"]["names"]:
        before_auc = evaluation["before"]["strategy_auc"][name]
        after_auc = evaluation["after"]["strategy_auc"][name]
        strategies.append(
            {
                "name": name,
                "before_auc": before_auc,
                "after_auc": after_auc,
                "delta": after_auc - before_auc,
            }
        )
    return {
        "judgment_id": record["judgment_id"],
        "action": record["change"]["action"],
        "record_path": str(path),
        "record_sha256": file_sha256(path),
        "evidence_path": str(evidence_path),
        "evidence_sha256": record["evidence"]["sha256"],
        "input_snapshot_path": str(snapshot_path),
        "input_snapshot_sha256": snapshot["sha256"],
        "pool_sha256": record["frozen_input"]["candidate_pool"]["sha256"],
        "pool_member_count": record["frozen_input"]["candidate_pool"]["member_count"],
        "combiner_scope": record["selection"]["combiner_scope"],
        "combiner_names": record["frozen_input"]["registered_combiners"]["names"],
        "combiner_names_sha256": record["frozen_input"]["registered_combiners"][
            "names_sha256"
        ],
        "before": {
            "member_count": evaluation["before"]["member_count"],
            "best_strategy": evaluation["before"]["best_strategy"],
            "best_auc": evaluation["before"]["best_auc"],
        },
        "after": {
            "member_count": evaluation["after"]["member_count"],
            "best_strategy": evaluation["after"]["best_strategy"],
            "best_auc": evaluation["after"]["best_auc"],
        },
        "strategies": strategies,
        "delta": comparison["delta"],
        "outer_fold_delta": comparison["outer_fold_delta"],
        "outer_fold_wins": comparison["outer_fold_wins"],
        "boundary_contribution": comparison["boundary_contribution"],
        "nested_prediction_sha256": evaluation["nested_prediction_sha256"],
        "state": record["result"]["state"],
        "decision": record["result"]["decision"],
    }


def _correlation_record(admission: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    snapshot = pd.read_parquet(admission["input_snapshot_path"])
    candidate_column = f"candidate:{CANDIDATE_RUN_ID}"
    if candidate_column not in snapshot:
        raise AssertionError("일반 추가 평가 입력에 후보 열이 없다.")
    member_by_run = {
        member["run_id"]: member
        for member in freeze["frozen_input"]["candidate_pool"]["members"]
    }
    correlations = []
    for run_id, member in member_by_run.items():
        column = f"pool:{run_id}"
        if column not in snapshot:
            raise AssertionError(f"일반 추가 평가 입력에 풀 구성원 열이 없다: {run_id}")
        correlations.append(
            {
                "run_id": run_id,
                "config": member["config"],
                "spearman": float(
                    snapshot[candidate_column].corr(snapshot[column], method="spearman")
                ),
            }
        )
    correlations.sort(key=lambda item: item["spearman"], reverse=True)
    duplicates = [
        item for item in correlations if item["spearman"] >= DUPLICATE_SPEARMAN
    ]
    if [item["run_id"] for item in duplicates] != [REPLACED_RUN_ID]:
        raise AssertionError(f"원자 교체 대상이 하나로 고정되지 않는다: {duplicates}")
    remaining = [item for item in correlations if item["run_id"] != REPLACED_RUN_ID]
    return {
        "threshold": DUPLICATE_SPEARMAN,
        "all_correlations": correlations,
        "duplicates": duplicates,
        "replacement_target_run_id": REPLACED_RUN_ID,
        "nearest_after_replacement": remaining[0],
        "candidate_vs_remaining_all_below_threshold": all(
            item["spearman"] < DUPLICATE_SPEARMAN for item in remaining
        ),
    }


def _final_state(replacement: dict[str, Any]) -> dict[str, Any]:
    pool = _load_yaml(POOL_PATH)
    members = pool["members"]
    candidate = [member for member in members if member["run_id"] == CANDIDATE_RUN_ID]
    replaced = [member for member in members if member["run_id"] == REPLACED_RUN_ID]
    if len(candidate) != 1 or replaced:
        raise AssertionError("후보 풀의 원자 교체 결과가 기대와 다르다.")
    pointer = candidate[0].get("judgment")
    if pointer is None or pointer["sha256"] != replacement["record_sha256"]:
        raise AssertionError("후보 풀 장부가 원자 교체 판정 기록을 가리키지 않는다.")

    plan_document = _load_yaml(REFIT_PLAN_PATH)
    pool_sha256 = file_sha256(POOL_PATH)
    if plan_document["source_pool_sha256"] != pool_sha256:
        raise AssertionError("재학습 계획의 후보 풀 해시가 현재 장부와 다르다.")
    plan_member_runs = [member["lineage"]["source_run_id"] for member in plan_document["members"]]
    pool_member_runs = [member["run_id"] for member in members]
    if plan_member_runs != pool_member_runs:
        raise AssertionError("재학습 계획 구성원 순서가 후보 풀과 다르다.")
    executable = RefitPlan.load(REFIT_PLAN_PATH).validate_for_refit()
    refit_candidate = executable.member(candidate[0]["config"])
    return {
        "decision": "register_by_atomic_replacement",
        "registered": True,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "replaced_run_id": REPLACED_RUN_ID,
        "pool_path": str(POOL_PATH),
        "pool_sha256": pool_sha256,
        "pool_member_count": len(members),
        "pool_entry": candidate[0],
        "refit_plan_path": str(REFIT_PLAN_PATH),
        "refit_plan_sha256": file_sha256(REFIT_PLAN_PATH),
        "refit_plan_schema_version": plan_document["schema_version"],
        "refit_plan_member_count": len(plan_document["members"]),
        "refit_plan_total_runs": sum(len(member.budgets) for member in executable.members),
        "candidate_refit_budgets": {
            str(seed): budget for seed, budget in refit_candidate.budgets.items()
        },
        "current_submission_assembly_changed": False,
    }


def _build_record(recorded_at_utc: str) -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text())
    predecessor = json.loads(PREDECESSOR_PATH.read_text())
    admission = _judgment(ADMISSION_PATH)
    replacement = _judgment(REPLACEMENT_PATH)
    if admission["action"] != "admission" or admission["decision"] != "admit":
        raise AssertionError("일반 추가 판정이 채택 기록이 아니다.")
    if replacement["action"] != "replacement" or replacement["decision"] != "replace":
        raise AssertionError("원자 교체 판정이 채택 기록이 아니다.")
    if replacement["pool_sha256"] != admission["pool_sha256"]:
        raise AssertionError("일반 추가와 원자 교체의 동결 후보 풀이 다르다.")
    if admission["pool_sha256"] != freeze["frozen_input"]["candidate_pool"]["sha256"]:
        raise AssertionError("판정 기록과 사전 동결의 후보 풀 해시가 다르다.")
    if admission["combiner_names"] != freeze["frozen_input"]["registered_combiners"]["names"]:
        raise AssertionError("판정 기록과 사전 동결의 결합 방식이 다르다.")
    if not (0 < admission["delta"] <= POOL_EQUIVALENCE_BAND_UPPER):
        raise AssertionError("일반 추가 판정이 양수 경계 기여가 아니다.")
    if not (0 < replacement["delta"] <= POOL_EQUIVALENCE_BAND_UPPER):
        raise AssertionError("원자 교체 판정이 양수 경계 기여가 아니다.")

    champion = _load_yaml(CHAMPION_PATH)
    candidate_auc = freeze["candidate"]["metrics"]["auc_oof"]
    entry_floor = champion["oof_auc"] - 0.01
    candidate_integrity = predecessor["training_row_integrity"]
    canary = predecessor["runs"]["missingness_augmented"][
        "target_encoding_importance_canary"
    ]
    eligibility = {
        "finished": freeze["candidate"]["status"] == "FINISHED",
        "seeds": [42, 43, 44],
        "clean_execution_commit": not freeze["candidate"]["execution"]["git_dirty"],
        "input_hashes_match": freeze["candidate"]["input_sha256"]
        == {
            "train": freeze["frozen_input"]["datasets_and_folds"]["train"]["sha256"],
            "test": freeze["frozen_input"]["datasets_and_folds"]["test"]["sha256"],
            "folds": freeze["frozen_input"]["datasets_and_folds"]["folds"]["sha256"],
        },
        "training_row_integrity": candidate_integrity["all_assertions_passed"],
        "target_encoding_canary": canary["ok"],
        "candidate_oof_auc": candidate_auc,
        "champion_oof_auc": champion["oof_auc"],
        "entry_floor": entry_floor,
        "entry_floor_passed": candidate_auc >= entry_floor,
    }
    if not all(
        value
        for key, value in eligibility.items()
        if key
        in {
            "finished",
            "clean_execution_commit",
            "input_hashes_match",
            "training_row_integrity",
            "target_encoding_canary",
            "entry_floor_passed",
        }
    ):
        raise AssertionError(f"후보 자격 검사에 실패했다: {eligibility}")

    correlation = _correlation_record(admission, freeze)
    if not correlation["candidate_vs_remaining_all_below_threshold"]:
        raise AssertionError("원자 교체 뒤 후보가 남은 구성원과 중복된다.")
    return {
        "schema_version": 1,
        "issue": ISSUE_URL,
        "recorded_at_utc": recorded_at_utc,
        "pre_result_freeze": {
            "path": str(FREEZE_PATH),
            "sha256": file_sha256(FREEZE_PATH),
            "commit": PRE_RESULT_COMMIT,
        },
        "predecessor": {
            "issue": PREDECESSOR_URL,
            "path": str(PREDECESSOR_PATH),
            "sha256": file_sha256(PREDECESSOR_PATH),
            "decision": predecessor["decision"],
        },
        "candidate": freeze["candidate"],
        "eligibility": eligibility,
        "admission": admission,
        "duplicate_check": correlation,
        "replacement": replacement,
        "final": _final_state(replacement),
        "next_issue": NEXT_ISSUE_URL,
    }


def _strategy_table(lines: list[str], judgment: dict[str, Any]) -> None:
    lines.extend(
        [
            "| 결합 방식 | 제외 AUC | 포함 또는 교체 AUC | 차이 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in judgment["strategies"]:
        lines.append(
            f"| `{row['name']}` | `{row['before_auc']:.12f}` | `{row['after_auc']:.12f}` | `{row['delta']:+.12f}` |"
        )


def _fold_table(lines: list[str], judgment: dict[str, Any]) -> None:
    lines.extend(["| 바깥 분할 | AUC 차이 |", "| ---: | ---: |"])
    for fold, delta in judgment["outer_fold_delta"].items():
        lines.append(f"| {fold} | `{delta:+.12f}` |")


def _report(record: dict[str, Any]) -> str:
    freeze = record["pre_result_freeze"]
    candidate = record["candidate"]
    frozen_input = json.loads(FREEZE_PATH.read_text())["frozen_input"]
    admission = record["admission"]
    replacement = record["replacement"]
    duplicate = record["duplicate_check"]
    final = record["final"]
    lines = [
        "# 결측 증강 후보의 후보 풀 기여 판정",
        "",
        f"이 문서는 [통과한 결측 증강 후보의 후보 풀 기여를 판정한다]({ISSUE_URL})의 동결 입력, 중첩 OOF 비교, 중복 처리와 등록 결과를 기록한다.",
        "",
        "## 결론",
        "",
        f"결측 증강 평균본은 일반 추가에서 중첩 OOF AUC `{admission['delta']:+.12f}`, 기존 `exp117_ag25_gbm_r21`과의 원자 교체에서 `{replacement['delta']:+.12f}`의 양의 기여를 냈다.",
        f"두 비교 모두 바깥 분할 `{admission['outer_fold_wins']}/5`, `{replacement['outer_fold_wins']}/5`가 양수였다.",
        f"두 차이는 성능 동등 대역 상한 `+{POOL_EQUIVALENCE_BAND_UPPER:.12f}` 이내이므로 경계 기여다.",
        "후보는 기존 구성원 한 건과만 중복됐고 원자 교체 뒤 남은 구성원과의 최대 상관이 문턱 아래이므로 후보 풀에 지속 등록했다.",
        "현재 313개 제출 결합과 제출물은 변경하지 않았다.",
        "",
        "## 결과 확인 전 동결",
        "",
        f"동결 기록은 커밋 `{freeze['commit']}`에서 먼저 확정했고 파일 SHA-256은 `{freeze['sha256']}`다.",
        f"후보 실행은 `{candidate['run_id']}`이며 실행 커밋은 `{candidate['execution']['git_commit']}`다.",
        f"학습은 `{candidate['execution']['start_time_utc']}`부터 `{candidate['execution']['end_time_utc']}`까지 수행됐다.",
        f"실행 산출물 `{candidate['artifact_count']}`개의 전체 목록 해시는 `{candidate['artifact_manifest_sha256']}`다.",
        f"OOF 파일 SHA-256은 `{candidate['prediction_artifacts']['oof']['file_sha256']}`, 시험 예측 파일 SHA-256은 `{candidate['prediction_artifacts']['test']['file_sha256']}`다.",
        f"학습, 시험, 고정 분할 SHA-256은 각각 `{candidate['input_sha256']['train']}`, `{candidate['input_sha256']['test']}`, `{candidate['input_sha256']['folds']}`다.",
        f"판정 전 후보 풀은 `{frozen_input['candidate_pool']['member_count']}`개이고 SHA-256은 `{frozen_input['candidate_pool']['sha256']}`다.",
        f"핵심 결합 방식 목록 해시는 `{frozen_input['registered_combiners']['names_sha256']}`다.",
        "",
        "## 자격 검사",
        "",
        f"후보는 완료 상태, 난수 42·43·44 평균본, 깨끗한 실행 커밋, 입력 및 분할 해시 일치, 학습 행 무결성과 목표값 부호화 카나리아 검사를 모두 통과했다.",
        f"OOF AUC는 `{record['eligibility']['candidate_oof_auc']:.12f}`로 진입 하한 `{record['eligibility']['entry_floor']:.12f}`보다 높다.",
        "",
        "## 일반 추가 대조",
        "",
        f"현재 35개 풀과 후보를 더한 36개 풀을 같은 핵심 결합 방식 세 개로 비교했다.",
        "",
    ]
    _strategy_table(lines, admission)
    lines.extend(
        [
            "",
            f"양쪽 최선 방식은 모두 `{admission['before']['best_strategy']}`였고 최선끼리의 차이는 `{admission['delta']:+.12f}`다.",
            "",
        ]
    )
    _fold_table(lines, admission)
    lines.extend(
        [
            "",
            "## 중복 검사와 원자 교체",
            "",
            f"후보와 기존 풀 전체의 스피어만 순위 상관을 계산한 결과 문턱 `{duplicate['threshold']}` 이상은 `exp117_ag25_gbm_r21` 한 건뿐이었다.",
            f"해당 상관은 `{duplicate['duplicates'][0]['spearman']:.12f}`다.",
            f"이를 제거한 뒤 후보와 가장 가까운 구성원은 `{duplicate['nearest_after_replacement']['config']}`이고 상관은 `{duplicate['nearest_after_replacement']['spearman']:.12f}`로 문턱 아래다.",
            "",
        ]
    )
    _strategy_table(lines, replacement)
    lines.extend(
        [
            "",
            f"원자 교체에서도 양쪽 최선 방식은 `{replacement['before']['best_strategy']}`였고 최선끼리의 차이는 `{replacement['delta']:+.12f}`다.",
            "",
        ]
    )
    _fold_table(lines, replacement)
    lines.extend(
        [
            "",
            "## 등록과 재학습 계획",
            "",
            f"후보 풀은 35개를 유지하면서 기존 실행 `{final['replaced_run_id']}`을 후보 실행 `{final['candidate_run_id']}`로 교체했다.",
            f"새 후보 풀 SHA-256은 `{final['pool_sha256']}`다.",
            f"전체 자료 재학습 계획은 후보 풀과 같은 35개 구성원, 총 `{final['refit_plan_total_runs']}`회 실행으로 검증됐다.",
            f"결측 증강 후보의 시드별 학습 예산은 42가 `{final['candidate_refit_budgets']['42']}`, 43이 `{final['candidate_refit_budgets']['43']}`, 44가 `{final['candidate_refit_budgets']['44']}`회다.",
            "",
            "## 근거",
            "",
            f"사전 동결은 `{freeze['path']}`에 있다.",
            f"일반 추가 판정은 `{admission['record_path']}`이며 SHA-256은 `{admission['record_sha256']}`다.",
            f"원자 교체 판정은 `{replacement['record_path']}`이며 SHA-256은 `{replacement['record_sha256']}`다.",
            f"기계 판독 최종 결과는 `{DEFAULT_JSON_OUTPUT}`에 있다.",
            f"다음 단계는 [독립 결측 증강 실험의 근거와 최종 판정을 확정한다]({NEXT_ISSUE_URL})다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if args.verify:
        if not args.json_output.is_file() or not args.report_output.is_file():
            raise SystemExit("검증할 JSON 또는 Markdown 결과 파일이 없다.")
        expected = json.loads(args.json_output.read_text())
        actual = _build_record(expected["recorded_at_utc"])
        if actual != expected:
            raise SystemExit("기계 판독 결과와 현재 근거가 다르다.")
        report = _report(actual)
        if args.report_output.read_text() != report:
            raise SystemExit("사람이 읽는 결과와 현재 근거가 다르다.")
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
    args.json_output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    args.report_output.write_text(_report(record))
    print(f"기계 판독 결과 생성: {args.json_output} ({file_sha256(args.json_output)})")
    print(f"연구 기록 생성: {args.report_output} ({file_sha256(args.report_output)})")


if __name__ == "__main__":
    main()
