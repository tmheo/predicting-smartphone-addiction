"""이슈 511의 완결 짝 실행을 중앙 MLflow에서 다시 검증해 최종 기록을 만든다."""

from __future__ import annotations

import argparse
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET
from pipeline.runs import MlflowRunStore


ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/511"
PRECOMMIT_PATH = Path("artifacts/issue510-missingness-propagation-precommit.json")
SOURCE_DEFAULT = "c831c40f80e4115856364586643b6312a7a1bca5"
SOURCE_XGB_DIAGNOSTIC_FIX = "7000df00329b3e0d71fc0dcf7bbd1be293144b69"
SOURCE_NEURAL_CORRECTION = "b9be8aaba1a5c307084fab9c678312bdab536959"
NEURAL_CONTRACT = "paired-parent-balanced-exposure-v2"
INPUT_SHA256 = {
    "train": "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c",
    "test": "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e",
    "folds": "5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4",
}
RESOURCE_CHECKED_AT_UTC = "2026-08-30T07:40:13Z"
RESOURCE_END_CREDIT_USD = 20.743656999368994


PAIR_RUN_IDS = {
    1: ("11d48af0dd964803aafd45eabd1d1968", "590e5975d00641119c79ebc8970c737f"),
    2: ("dc8aaaa66d3d4e44a50ab570c33c91c3", "b0832fb2477446768da52a4d95f99825"),
    3: ("d1427312166340899ce943ebf22c1428", "6e770b47cc524454986ef00d8110cf26"),
    4: ("82f5bbf431c54590b16b2bead6ca12e6", "636908c5d68d4259a9fb86553961bfb2"),
    5: ("84da7b7079ac4c57acc4b9b5458b5d1d", "c3bd36067dbe4c9fb68935da48bb4f98"),
    6: ("1369fdd2f80f4cc4897b7f1905544100", "8ad47dda4f75432e946330c23fccf317"),
    7: ("042527e9bd3d419ab3ccea50b904808d", "f525ee26f7e54d4b96f515c858e39e99"),
    8: ("6b75c87cf644438288c0ae00175248e2", "90aed4413e4b48e7a56f92fa9ecd5285"),
    10: ("ef0c6024d86d48ec8a75930cdaf17c2d", "4fd10d9fc1324583a023552acc2cf77f"),
    12: ("470b1202750647a483b6006cef8b29e5", "9c51438d9842439881ed151ba66b0c86"),
    13: ("ccab5cf432bb4da9b841251dfb93c013", "53b1699cffdd48fa8b5268899bede34c"),
    14: ("b196a34ef666481e8995b02048edd96a", "ac74499f041e4ef1a5c7f39d9f17e47c"),
    15: ("a1e26156491d4a239d460c4423996bdb", "23b474bbad6f45e481e201b22fa602f2"),
    17: ("17001b45e8a44c35aa72c9eab0a08293", "e18005f1342d442a9f0aa432d8ed0280"),
    18: ("7309d90b64614affb6e041064b12a690", "721a05f89e5f443bbd89c903d878d4f2"),
    19: ("05d2663a98ed451aaf85c4521f6c5d98", "51eedd29ad7a448cb9225aa2c8cf4236"),
    21: ("25bfea63e32941219a4eb850fa49be69", "b731e5f297db44a093a42709f9280605"),
    22: ("856923ca55a24706a8d3430e660f75d1", "9b847a787b25429f872f6eadb7497659"),
    24: ("a41e5a86c4fe45b49b1e206643cc1e7d", "367f2c4297c04debb3c1ae9e4606c90a"),
    28: ("44c0553c555e4e9b806fa8428922dcf8", "3e740473d43b4532bec9a0a6aaf927db"),
    31: ("2d32528f4ac445d89811a938d64b871e", "c37cf00ffee4445db8dcf97ba5e375c7"),
    32: ("d0aa583196474fef870e9ced20d7c750", "f364a98fcbf94320b549c1fea55a0518"),
    33: ("1ea5aeed6228443d89414345e0f2192f", "47f83128009f42afb7e2df3e8a1cb5fb"),
    34: ("62f691639416428aa26a6181442e0c3c", "8281439d83304f1b8b983cf20ba21e6f"),
}
NEURAL_ORDINALS = {15, 17, 22, 24, 28}
XGB_EXCEPTION_ORDINALS = {13, 21}
TAB_CNN_EXCLUDED = {16, 26, 27}
USER_SELECTED_OUT = {9, 11, 20, 23, 25, 29, 30}
INVALID_NEURAL_RUNS = {
    "15-exp106": {
        "first_round": ["dec47ce45a624bfab6db8927db2573d5", "6434a350bbf94a2d88fc20b0b19f0119"],
        "second_round": ["57df44db51a5457f88780f59290fc8fb", "ffe23b8acd7243e998ffec8c13c5798e"],
    },
    "17-exp085": {
        "first_round": ["2b9b9fd2d8c747ae857fa8dede4b96e8", "e63904815bff4a74922a70d8987e1b58"],
        "second_round": [],
    },
    "22-exp131": {
        "first_round": ["296c93858cbc45e5a96d426f9e48ffb8", "7ea59e3f835d489187d3c5083bfb2bd4"],
        "second_round": [],
    },
    "24-exp137": {"first_round": [], "second_round": []},
    "28-exp139": {
        "first_round": ["2d207e326b664305b094dead15211721", "0bb24f819b724ff59cf73ffcd36be006"],
        "second_round": [],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/issue511-missingness-propagation-confirmation.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("docs/research/issue511-missingness-propagation-confirmation.md"),
    )
    return parser.parse_args()


def _expected_source(ordinal: int) -> str:
    if ordinal in NEURAL_ORDINALS:
        return SOURCE_NEURAL_CORRECTION
    if ordinal in XGB_EXCEPTION_ORDINALS:
        return SOURCE_XGB_DIAGNOSTIC_FIX
    return SOURCE_DEFAULT


def _run_record(
    store: MlflowRunStore,
    run_id: str,
    expected_name: str,
    expected_source: str,
    targets: pd.DataFrame,
    neural: bool,
) -> dict[str, Any]:
    facts = store.facts_of(run_id)
    if facts.status != "FINISHED":
        raise AssertionError(f"실행 {run_id} 상태가 FINISHED가 아니다: {facts.status}")
    if facts.params["experiment"] != expected_name:
        raise AssertionError(f"실행 {run_id} 이름이 고정값과 다르다.")
    if facts.params["seeds"] != "42,43,44":
        raise AssertionError(f"실행 {run_id} 시드가 42,43,44가 아니다.")
    if facts.tags["git_commit"] != expected_source or facts.tags["git_dirty"] != "False":
        raise AssertionError(f"실행 {run_id} 출처 또는 깨끗한 상태가 고정값과 다르다.")
    for name, expected in INPUT_SHA256.items():
        if facts.tags[f"sha256.{name}"] != expected:
            raise AssertionError(f"실행 {run_id}의 {name} 입력 해시가 다르다.")
    if neural:
        if facts.tags.get("issue511.judgment_status") != "valid":
            raise AssertionError(f"신경망 실행 {run_id}에 유효 판정 꼬리표가 없다.")
        if facts.tags.get("issue511.validation_contract") != NEURAL_CONTRACT:
            raise AssertionError(f"신경망 실행 {run_id}의 교정 계약이 다르다.")

    oof = pd.read_parquet(io.BytesIO(store.artifact_bytes_of(run_id, "oof.parquet")))
    scored = oof.merge(targets, on=ID, validate="one_to_one")
    rescored_auc = float(roc_auc_score(scored[TARGET], scored["pred"]))
    if abs(rescored_auc - facts.metrics["auc_oof"]) > 1e-12:
        raise AssertionError(f"실행 {run_id}의 OOF 재채점값이 기록값과 다르다.")

    evidence = json.loads(store.artifact_bytes_of(run_id, "training_row_evidence.json"))["entries"]
    if len(evidence) != 15:
        raise AssertionError(f"실행 {run_id}의 학습 행 근거가 15개가 아니다.")
    if not all(all(entry["assertions"].values()) for entry in evidence):
        raise AssertionError(f"실행 {run_id}의 학습 행 단언 가운데 실패가 있다.")
    if any(entry["validation_augmented"] or entry["test_augmented"] for entry in evidence):
        raise AssertionError(f"실행 {run_id}가 검증 또는 시험 자료를 증강했다.")

    return {
        "run_id": run_id,
        "experiment": facts.params["experiment"],
        "auc_oof": rescored_auc,
        "git_commit": facts.tags["git_commit"],
        "git_dirty": False,
        "provider": facts.tags["remote.provider"],
        "runtime_class": facts.tags["remote.runtime_class"],
        "input_sha256": INPUT_SHA256,
        "bundle_manifest_sha256": store.artifact_sha256_of(run_id, "bundle/manifest.json"),
        "oof_sha256": store.artifact_sha256_of(run_id, "oof.parquet"),
        "training_row_evidence_sha256": store.artifact_sha256_of(
            run_id, "training_row_evidence.json"
        ),
        "coordinates_verified": 15,
        "neural_validation_contract": NEURAL_CONTRACT if neural else None,
    }


def _render_report(record: dict[str, Any]) -> str:
    lines = [
        "# 결측 증강 전파 선별 짝비교 최종 실행 기록",
        "",
        f"이 문서는 GitHub 이슈 [#511]({ISSUE_URL})의 완결된 3시드 짝비교와 중앙 반입 결과를 기록한다.",
        "결과 확인 뒤 사용자가 GPU 후보를 선별했으므로 사전 동결 34짝 전체가 아니라 최종 선택된 24짝만 완결했다.",
        "TabCNN 계열 3짝은 실행 범위에서 제외했고 GPU 후보 7짝은 비용 검토 뒤 사용자가 선택하지 않았다.",
        "",
        "## 결론",
        "",
        f"- 완결 짝은 {record['summary']['completed_pair_count']}개이며 결측 증강군이 3배 대조군보다 높은 짝은 {record['summary']['positive_delta_count']}개, 낮은 짝은 {record['summary']['negative_delta_count']}개다.",
        f"- 가장 높은 결측 증강 OOF AUC는 `{record['summary']['best_augmented']['member']}`의 `{record['summary']['best_augmented']['auc_oof']:.10f}`다.",
        f"- 가장 큰 직접 개선은 `{record['summary']['largest_gain']['member']}`의 `{record['summary']['largest_gain']['delta']:+.10f}`다.",
        "- 신경망 5짝은 첫 실행과 두 번째 교정을 무효화한 뒤 부모별 학습 경로를 보존하는 세 번째 교정으로 다시 실행했다.",
        "- Vast.ai 계산 자원과 별도 저장 공간 목록은 모두 비어 있으며 추가 과금 자원은 남아 있지 않다.",
        "- 이 기록은 직접 짝비교 실행과 중앙 반입의 완료 근거이며 후보 풀 변경이나 중첩 선별 채택을 수행하지 않는다.",
        "",
        "## 완결 결과",
        "",
        "| 짝 | 후보 | 공급자 | 3배 대조군 | 결측 증강군 | 차이 |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for pair in record["completed_pairs"]:
        lines.append(
            f"| {pair['ordinal']:02d} | `{pair['member']}` | {pair['provider']} | "
            f"{pair['tripled']['auc_oof']:.10f} | {pair['missingness_augmented']['auc_oof']:.10f} | "
            f"{pair['missingness_augmented_minus_tripled']:+.10f} |"
        )
    lines.extend(
        [
            "",
            "## 무결성과 교정",
            "",
            "모든 완결 짝은 같은 공급자와 실행 환경 등급에서 두 팔을 함께 끝냈다.",
            "각 중앙 실행은 고정된 세 시드, 입력 해시, 깨끗한 출처, 15개 학습 좌표, OOF 재채점과 중앙 묶음 산출물 해시를 통과했다.",
            "`13-exp111`과 `21-exp135`만 XGBoost 고정 학습 길이 진단 수정 출처를 사용했고 설정 해시는 원래 고정 출처와 동일하다.",
            "신경망 5짝은 원본 물리 배치 크기, 부모 행 노출 순서, 최적화 갱신 수, 학습률 일정 위치와 원본 전처리 범위를 보존했다.",
            "교정 3배 대조군이 각 역사적 원본 성능을 재현했으므로 세 번째 신경망 짝비교만 유효하다.",
            "",
            "## 실행하지 않은 짝",
            "",
        ]
    )
    for item in record["excluded_pairs"]:
        lines.append(f"- `{item['ordinal']:02d}-{item['member']}`: {item['reason']}.")
    lines.extend(
        [
            "",
            "## 자원 정리",
            "",
            f"Vast.ai 계정은 {record['resource_cleanup']['checked_at_utc']}에 다시 조회했다.",
            f"활성 인스턴스 {record['resource_cleanup']['active_instance_count']}개, 별도 저장 공간 {record['resource_cleanup']['volume_count']}개이며 잔액은 `${record['resource_cleanup']['ending_credit_usd']:.6f}`다.",
            "실행별 비용과 실패·재시도 자원 정산은 이슈 댓글과 로컬 `run-logs/issue511` 장부에 보존한다.",
            "",
            "## 근거",
            "",
            "- 최종 기계 판독 기록: `artifacts/issue511-missingness-propagation-confirmation.json`",
            "- 실행 전 고정 기록: `artifacts/issue510-missingness-propagation-precommit.json`",
            "- 학습 길이 고정 기록: `artifacts/issue510-paired-training-lengths.json`",
            "- 신경망 교정 계약: `docs/adr/0007-preserve-neural-optimizer-steps-in-replicated-row-comparisons.md`",
            "- 중앙 실행 식별자, OOF 해시와 묶음 manifest 해시는 기계 판독 기록의 각 짝 항목에 있다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    precommit = json.loads(PRECOMMIT_PATH.read_text())
    pair_by_ordinal = {int(item["ordinal"]): item for item in precommit["pairs"]}
    if set(PAIR_RUN_IDS) | TAB_CNN_EXCLUDED | USER_SELECTED_OUT != set(pair_by_ordinal):
        raise AssertionError("완결·제외 짝 집합이 실행 전 고정 34짝과 정확히 일치하지 않는다.")

    targets = pd.read_csv("data/train.csv", usecols=[ID, TARGET])
    store = MlflowRunStore()
    completed = []
    for ordinal, (tripled_id, augmented_id) in sorted(PAIR_RUN_IDS.items()):
        frozen = pair_by_ordinal[ordinal]
        arm_by_name = {arm["arm"]: arm for arm in frozen["arms"]}
        source = _expected_source(ordinal)
        tripled = _run_record(
            store,
            tripled_id,
            arm_by_name["tripled"]["name"],
            source,
            targets,
            ordinal in NEURAL_ORDINALS,
        )
        augmented = _run_record(
            store,
            augmented_id,
            arm_by_name["missingness_augmented"]["name"],
            source,
            targets,
            ordinal in NEURAL_ORDINALS,
        )
        if (tripled["provider"], tripled["runtime_class"]) != (
            augmented["provider"],
            augmented["runtime_class"],
        ):
            raise AssertionError(f"{ordinal:02d} 짝의 공급자 또는 실행 환경 등급이 다르다.")
        completed.append(
            {
                "ordinal": ordinal,
                "member": frozen["member"],
                "provider": tripled["provider"],
                "runtime_class": tripled["runtime_class"],
                "source_commit": source,
                "common_config_semantic_sha256": frozen["common_config_semantic_sha256"],
                "tripled": tripled,
                "missingness_augmented": augmented,
                "missingness_augmented_minus_tripled": augmented["auc_oof"] - tripled["auc_oof"],
            }
        )

    excluded = []
    for ordinal in sorted(TAB_CNN_EXCLUDED | USER_SELECTED_OUT):
        frozen = pair_by_ordinal[ordinal]
        excluded.append(
            {
                "ordinal": ordinal,
                "member": frozen["member"],
                "reason": (
                    "TabCNN 계열 사전 제외"
                    if ordinal in TAB_CNN_EXCLUDED
                    else "비용 검토 뒤 GPU 선별 대상에서 제외"
                ),
            }
        )

    positive = [pair for pair in completed if pair["missingness_augmented_minus_tripled"] > 0]
    negative = [pair for pair in completed if pair["missingness_augmented_minus_tripled"] < 0]
    best_augmented = max(completed, key=lambda item: item["missingness_augmented"]["auc_oof"])
    largest_gain = max(completed, key=lambda item: item["missingness_augmented_minus_tripled"])
    record = {
        "schema": "issue511-missingness-propagation-confirmation/1",
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "issue": {"number": 511, "url": ISSUE_URL},
        "scope": {
            "frozen_pair_count": len(pair_by_ordinal),
            "completed_pair_count": len(completed),
            "excluded_pair_count": len(excluded),
            "seeds": [42, 43, 44],
            "folds": [0, 1, 2, 3, 4],
            "decision_scope": "paired_execution_and_central_import_only",
        },
        "source_commits": {
            "default": SOURCE_DEFAULT,
            "xgb_diagnostic_fix_exception": SOURCE_XGB_DIAGNOSTIC_FIX,
            "neural_parent_balanced_correction": SOURCE_NEURAL_CORRECTION,
        },
        "input_sha256": INPUT_SHA256,
        "completed_pairs": completed,
        "excluded_pairs": excluded,
        "invalid_neural_runs": INVALID_NEURAL_RUNS,
        "summary": {
            "completed_pair_count": len(completed),
            "positive_delta_count": len(positive),
            "negative_delta_count": len(negative),
            "zero_delta_count": len(completed) - len(positive) - len(negative),
            "best_augmented": {
                "ordinal": best_augmented["ordinal"],
                "member": best_augmented["member"],
                "auc_oof": best_augmented["missingness_augmented"]["auc_oof"],
            },
            "largest_gain": {
                "ordinal": largest_gain["ordinal"],
                "member": largest_gain["member"],
                "delta": largest_gain["missingness_augmented_minus_tripled"],
            },
        },
        "resource_cleanup": {
            "provider": "vast",
            "checked_at_utc": RESOURCE_CHECKED_AT_UTC,
            "active_instance_count": 0,
            "volume_count": 0,
            "ending_credit_usd": RESOURCE_END_CREDIT_USD,
            "billing_stopped": True,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    args.report_output.write_text(_render_report(record))


if __name__ == "__main__":
    main()
