"""이슈 515의 결측 증강 전파 일괄 판정 계약을 결과 확인 전에 고정한다.

사용법:
    uv run python scripts/freeze_issue515_missingness_propagation_batch.py
    uv run python scripts/freeze_issue515_missingness_propagation_batch.py --verify-only

이 도구는 실험 실행 저장소와 예측값을 읽지 않는다.
현재 장부와 이슈 510의 실행 명세만으로 36개 풀, 고정 구성원 2개,
34개 원본과 결측 증강판의 상호 배타 대응, 검색과 채택 규칙을 봉인한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.data import file_sha256
from pipeline.missingness_propagation_batch import (
    ADOPTION_RULES,
    ADOPTION_STRATEGIES,
    CHAMPION_FLOOR_MARGIN,
    CONDITIONAL_RULES,
    CONTRACT_VERSION,
    DUPLICATE_RULES,
    FAILURE_RULES,
    FIXED_MEMBERS,
    FORMALIZATION_RULES,
    OUTER_FOLDS,
    PAIR_COUNT,
    POOL_MEMBER_COUNT,
    PRECOMMIT_SCHEMA,
    SEARCH_RULES,
    SEARCH_STRATEGY,
    canonical_sha256,
    identity_sha256,
    self_hashed_payload,
    validate_precommit,
    verify_search_mechanics,
)

OUTPUT_PATH = (
    REPO_ROOT / "artifacts/issue515-missingness-propagation-batch-precommit.json"
)
POOL_PATH = REPO_ROOT / "artifacts/pool.yaml"
CHAMPION_PATH = REPO_ROOT / "artifacts/champion.yaml"
FOLDS_PATH = REPO_ROOT / "artifacts/folds.parquet"
REFIT_PLAN_PATH = REPO_ROOT / "artifacts/full-refit-plan.yaml"
CAPACITY_PATH = REPO_ROOT / "artifacts/issue509-parallel-capacity-freeze.json"
PAIR_FREEZE_PATH = (
    REPO_ROOT / "artifacts/issue510-missingness-propagation-precommit.json"
)
LENGTH_EVIDENCE_PATH = REPO_ROOT / "artifacts/issue510-paired-training-lengths.json"
POOL_BASELINE_PATH = REPO_ROOT / "artifacts/pool-baseline-2026-08-21.yaml"
EXP208_FREEZE_PATH = (
    REPO_ROOT / "artifacts/issue503-missingness-candidate-pool-freeze.json"
)
CONTRACT_MODULE_PATH = REPO_ROOT / "src/pipeline/missingness_propagation_batch.py"
EXPECTED_POOL_SHA256 = (
    "c513443b6d1cc8af348dc06f8c547ed2728a659261cf7d78dc4e17a27ca668d9"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="기존 사전 기록과 구현 및 입력의 내용 해시만 검증한다.",
    )
    return parser.parse_args()


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 루트가 객체가 아니다: {_relative(path)}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"YAML 루트가 객체가 아니다: {_relative(path)}")
    return value


def _input(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": file_sha256(path)}


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixed_identity(
    member: str,
    *,
    position: int,
    pool_member: dict[str, Any],
    refit_member: dict[str, Any],
    pool_baseline_by_name: dict[str, dict[str, Any]],
    exp208_freeze: dict[str, Any],
) -> dict[str, Any]:
    lineage = refit_member["lineage"]
    identity: dict[str, Any] = {
        "role": "fixed_current_pool_member",
        "member": member,
        "pool_position": position,
        "run_id": pool_member["run_id"],
        "pool_entry_sha256": canonical_sha256(pool_member),
        "source_git_commit": lineage["source_git_commit"],
        "source_config_path": lineage["source_config_path"],
        "source_config_sha256": lineage["source_config_sha256"],
        "oof_auc": float(pool_member["oof_auc"]),
    }
    if member == "exp067_tabpfn3":
        baseline = pool_baseline_by_name[member]
        identity["oof_identity"] = {
            "kind": "prediction_array_sha256",
            "sha256": baseline["integrity"]["oof_sha256"],
        }
    elif member == "exp208_issue500_ag25_missingness_augmented":
        candidate = exp208_freeze["candidate"]
        oof = candidate["prediction_artifacts"]["oof"]
        identity["oof_identity"] = {
            "kind": "prediction_array_sha256",
            "sha256": oof["prediction_array_sha256"],
            "artifact_file_sha256": oof["file_sha256"],
        }
    else:
        raise ValueError(f"알 수 없는 고정 구성원: {member}")
    return {
        "member": member,
        "pool_position": position,
        "identity": identity,
        "identity_sha256": identity_sha256(identity),
    }


def build_precommit() -> dict[str, Any]:
    if file_sha256(POOL_PATH) != EXPECTED_POOL_SHA256:
        raise ValueError("후보 풀 내용 해시가 이슈 507에서 고정한 36개 풀과 다르다.")

    pool = _load_yaml(POOL_PATH)
    champion = _load_yaml(CHAMPION_PATH)
    refit_plan = _load_yaml(REFIT_PLAN_PATH)
    capacity = _load_json(CAPACITY_PATH)
    pair_freeze = _load_json(PAIR_FREEZE_PATH)
    length_evidence = _load_json(LENGTH_EVIDENCE_PATH)
    pool_baseline = _load_yaml(POOL_BASELINE_PATH)
    exp208_freeze = _load_json(EXP208_FREEZE_PATH)

    members = pool["members"]
    if len(members) != POOL_MEMBER_COUNT:
        raise ValueError("현재 후보 풀이 36개가 아니다.")
    pool_by_name = {member["config"]: member for member in members}
    if len(pool_by_name) != POOL_MEMBER_COUNT:
        raise ValueError("현재 후보 풀 이름이 중복된다.")
    pool_position = {
        member["config"]: position for position, member in enumerate(members, start=1)
    }
    refit_by_name = {member["config"]: member for member in refit_plan["members"]}
    pool_baseline_by_name = {
        member["config"]: member for member in pool_baseline["members"]
    }

    fixed = [
        _fixed_identity(
            member,
            position=pool_position[member],
            pool_member=pool_by_name[member],
            refit_member=refit_by_name[member],
            pool_baseline_by_name=pool_baseline_by_name,
            exp208_freeze=exp208_freeze,
        )
        for member in FIXED_MEMBERS
    ]

    evidence_by_name = {
        member["member"]: member for member in length_evidence["members"]
    }
    pair_freeze_by_name = {pair["member"]: pair for pair in pair_freeze["pairs"]}
    expected_pair_names = [
        member["config"] for member in members if member["config"] not in FIXED_MEMBERS
    ]
    if pair_freeze["scope"]["member_order"] != expected_pair_names:
        raise ValueError("이슈 510의 34개 짝 순서가 현재 후보 풀과 다르다.")
    if set(evidence_by_name) != set(expected_pair_names):
        raise ValueError("짝비교 학습 길이 근거가 현재 34개 원본을 정확히 덮지 않는다.")

    execution_contract_sha256 = canonical_sha256(pair_freeze["pair_contract"])
    pairs: list[dict[str, Any]] = []
    for ordinal, member in enumerate(expected_pair_names, start=1):
        evidence = evidence_by_name[member]
        source_identity = evidence["source_identity"]
        generated = pair_freeze_by_name[member]
        augmented_arm = next(
            arm for arm in generated["arms"] if arm["arm"] == "missingness_augmented"
        )
        original_identity = {
            "role": "current_pool_original",
            "member": member,
            "pool_position": pool_position[member],
            "run_id": pool_by_name[member]["run_id"],
            "pool_entry_sha256": canonical_sha256(pool_by_name[member]),
            "source_identity_sha256": canonical_sha256(source_identity),
            "source_git_commit": source_identity["git_commit"],
            "source_config_artifact_sha256": source_identity["config_artifact_sha256"],
            "normalized_config_sha256": source_identity["normalized_config_sha256"],
            "oof_artifact_sha256": source_identity["oof_artifact_sha256"],
            "input_sha256": source_identity["input_sha256"],
        }
        original_hash = identity_sha256(original_identity)
        augmented_identity = {
            "role": "missingness_augmented_replacement",
            "member": augmented_arm["name"],
            "slot_original_member": member,
            "ordinal": ordinal,
            "config_path": augmented_arm["path"],
            "config_sha256": augmented_arm["sha256"],
            "common_config_semantic_sha256": generated["common_config_semantic_sha256"],
            "source_original_identity_sha256": original_hash,
            "pair_execution_contract_sha256": execution_contract_sha256,
            "required_seeds": list(pair_freeze["pair_contract"]["seeds"]),
            "required_outer_folds": list(pair_freeze["pair_contract"]["outer_folds"]),
            "runtime_class": generated["runtime_class"],
        }
        pairs.append(
            {
                "ordinal": ordinal,
                "slot": member,
                "comparison_arms": generated["arms"],
                "original": {
                    "member": member,
                    "identity": original_identity,
                    "identity_sha256": original_hash,
                },
                "missingness_augmented": {
                    "member": augmented_arm["name"],
                    "identity": augmented_identity,
                    "identity_sha256": identity_sha256(augmented_identity),
                },
            }
        )

    inputs = {
        "candidate_pool": _input(POOL_PATH),
        "champion": _input(CHAMPION_PATH),
        "folds": _input(FOLDS_PATH),
        "full_refit_plan": _input(REFIT_PLAN_PATH),
        "parallel_capacity_freeze": _input(CAPACITY_PATH),
        "pair_execution_freeze": _input(PAIR_FREEZE_PATH),
        "paired_training_length_evidence": _input(LENGTH_EVIDENCE_PATH),
        "fixed_tabpfn_identity_evidence": _input(POOL_BASELINE_PATH),
        "fixed_exp208_identity_evidence": _input(EXP208_FREEZE_PATH),
    }
    inputs["candidate_pool"]["member_count"] = POOL_MEMBER_COUNT
    inputs["folds"]["role"] = "고정 5분할과 행 순서"

    cutoff = capacity["deadlines_utc"]["central_import"]
    threshold = float(champion["oof_auc"]) - CHAMPION_FLOOR_MARGIN
    payload: dict[str, Any] = {
        "schema": PRECOMMIT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "frozen_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "issue": {
            "number": 515,
            "title": "결측 증강 전파 일괄 원자 교체 판정 계약을 구현하고 사전 고정한다",
            "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/515",
        },
        "map": {
            "number": 506,
            "title": "지도: 결측 증강을 후보 풀 전 계열에 전파해 마지막 제출 개선을 판정한다",
            "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/506",
        },
        "source_commit_before_contract": _git_head(),
        "inputs": inputs,
        "scope": {
            "pool_member_count": POOL_MEMBER_COUNT,
            "pool_member_order": [member["config"] for member in members],
            "fixed_members": list(FIXED_MEMBERS),
            "pair_count": PAIR_COUNT,
            "pair_member_order": expected_pair_names,
            "proposal_member_count": POOL_MEMBER_COUNT,
            "prediction_values_read_while_freezing": False,
        },
        "fixed_members": fixed,
        "pair_execution_contract": {
            "sha256": execution_contract_sha256,
            "value": pair_freeze["pair_contract"],
        },
        "pairs": pairs,
        "identity_manifest_sha256": canonical_sha256(
            [
                *(record["identity_sha256"] for record in fixed),
                *(
                    identity_hash
                    for pair in pairs
                    for identity_hash in (
                        pair["original"]["identity_sha256"],
                        pair["missingness_augmented"]["identity_sha256"],
                    )
                ),
            ]
        ),
        "collection_contract": {
            "central_import_cutoff_utc": cutoff,
            "execution_source_commit": _git_head(),
            "state_rows": PAIR_COUNT,
            "complete_pair_requires_both_arms": True,
            "complete_pair_included_regardless_of_direct_delta": True,
            "same_provider": True,
            "same_runtime_class": True,
            "same_container_image_digest": True,
            "same_dependency_lock_sha256": True,
            "cross_provider_partial_merge": False,
            "incomplete_pair_records_reason_only": True,
            "required_arm_integrity": [
                "input hashes",
                "source commit",
                "clean state",
                "three-seed OOF rescore",
                "required diagnostics",
                "provider tag",
                "runtime class tag",
                "execution-record bundle hash",
            ],
        },
        "search_parameters": {
            "strategy": SEARCH_STRATEGY,
            "champion": {
                "config": champion["config"],
                "run_id": champion["run_id"],
                "oof_auc": float(champion["oof_auc"]),
            },
            "champion_floor_margin": CHAMPION_FLOOR_MARGIN,
            "missingness_augmented_entry_threshold": threshold,
            "entry_floor_applies_only_to_forward_search_moves": True,
            "duplicate_spearman": DUPLICATE_RULES["threshold"],
            "strict_positive_move": True,
            "floating_point_tolerance": None,
            "outer_folds": list(OUTER_FOLDS),
            "adoption_strategies": list(ADOPTION_STRATEGIES),
        },
        "search": SEARCH_RULES,
        "duplicate_invariant": DUPLICATE_RULES,
        "conditional_procedure": CONDITIONAL_RULES,
        "adoption_gates": ADOPTION_RULES,
        "failure_handling": FAILURE_RULES,
        "formalization": FORMALIZATION_RULES,
        "required_judgment_outputs": [
            "immutable input bundle with all 34 collection states",
            "full OOF search trajectory and proposal pool",
            "five outer conditional search trajectories and sealed predictions",
            "selection stability diagnostics",
            "three-strategy direct nested comparison",
            "duplicate diagnostics for every accepted state",
            "selected replacement full-refit rehearsal manifest",
            "final verdict and file SHA-256 manifest",
        ],
        "code": {
            "contract_module": _input(CONTRACT_MODULE_PATH),
            "freeze_script": _input(Path(__file__).resolve()),
        },
        "search_mechanics_verification": verify_search_mechanics(),
    }
    return self_hashed_payload(payload, "precommit_sha256")


def main() -> None:
    args = _args()
    if args.verify_only:
        payload = _load_json(OUTPUT_PATH)
        validate_precommit(payload, REPO_ROOT)
        mechanics = verify_search_mechanics()
        print(
            "검증 완료: "
            f"pool={payload['scope']['pool_member_count']} "
            f"pairs={payload['scope']['pair_count']} "
            f"precommit={payload['precommit_sha256']} "
            f"search={mechanics}"
        )
        return

    if OUTPUT_PATH.exists():
        raise ValueError(
            f"변경 불가 사전 기록이 이미 있다: {_relative(OUTPUT_PATH)}. "
            "검증은 --verify-only를 사용한다."
        )
    payload = build_precommit()
    validate_precommit(payload, REPO_ROOT)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "동결 완료: "
        f"pool={payload['scope']['pool_member_count']} "
        f"pairs={payload['scope']['pair_count']} "
        f"cutoff={payload['collection_contract']['central_import_cutoff_utc']} "
        f"precommit={payload['precommit_sha256']}"
    )


if __name__ == "__main__":
    main()
