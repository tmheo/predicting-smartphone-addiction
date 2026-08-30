"""이슈 512의 결측 증강 전파 일괄 판정 입력과 종결 판정을 기록한다.

사전 기록의 출처 커밋과 중앙 반입 결과를 다시 맞춰 34개 짝의 완결 상태를
변경 불가 입력 묶음으로 만든다.
현재 공식 풀의 중복 위반을 허용된 원자 교체로 해소할 수 없는 경우에는 검색과
후속 관문을 시작하지 않고 두 공식 장부 유지 판정을 기록한다.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET, file_sha256
from pipeline.missingness_propagation_batch import (
    INPUT_BUNDLE_SCHEMA,
    MissingnessPropagationBatchError,
    self_hashed_payload,
    spearman_violations,
    validate_input_bundle,
    validate_precommit,
    verify_self_hash,
)
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import ArtifactNotFound, MlflowRunStore

REPO_ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT_PATH = (
    REPO_ROOT / "artifacts/issue515-missingness-propagation-batch-precommit.json"
)
CONFIRMATION_PATH = (
    REPO_ROOT / "artifacts/issue511-missingness-propagation-confirmation.json"
)
POOL_PATH = REPO_ROOT / "artifacts/pool.yaml"
REFIT_PLAN_PATH = REPO_ROOT / "artifacts/full-refit-plan.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs/research/missingness-propagation-batch/issue512"
ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/512"
MAP_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/506"
INPUT_NAME = "input-bundle.json"
PREFLIGHT_NAME = "preflight.json"
JUDGMENT_NAME = "judgment.json"
REPORT_NAME = "report.md"
MANIFEST_NAME = "manifest.sha256"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
        help="비공개 자료와 중앙 MLflow가 있는 저장소 경로",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="생략하면 source-root의 mlflow.db를 사용한다.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MissingnessPropagationBatchError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON 루트가 객체가 아니다: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"YAML 루트가 객체가 아니다: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime_identity() -> dict[str, Any]:
    status = _git(
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    _require(not status, "판정 실행 전에 작업 폴더가 깨끗해야 한다.")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": True,
        "script": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }


def _tracking_uri(source_root: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    database = (source_root / "mlflow.db").resolve()
    _require(database.is_file(), f"중앙 MLflow 데이터베이스가 없다: {database}")
    return f"sqlite:///{database.as_posix()}"


def _targets(source_root: Path) -> pd.DataFrame:
    train_path = source_root / "data/train.csv"
    _require(train_path.is_file(), f"학습 자료가 없다: {train_path}")
    frame = pd.read_csv(train_path, usecols=[ID, TARGET])
    _require(not frame[ID].duplicated().any(), "학습 자료 id가 중복된다.")
    _require(set(frame[TARGET].unique()) == {0, 1}, "목표값이 이진 0, 1이 아니다.")
    return frame.set_index(ID)


def _oof_series(
    store: MlflowRunStore,
    run_id: str,
    targets: pd.DataFrame,
) -> tuple[pd.Series, str, str, float]:
    payload = store.artifact_bytes_of(run_id, "oof.parquet")
    frame = pd.read_parquet(io.BytesIO(payload))
    _require(
        list(frame.columns) in ([ID, "pred"], [ID, "fold", "pred"]),
        f"{run_id}: OOF 열이 id, 선택적 fold, pred와 다르다.",
    )
    _require(not frame[ID].duplicated().any(), f"{run_id}: OOF id가 중복된다.")
    prediction = frame.set_index(ID)["pred"].reindex(targets.index)
    _require(
        not prediction.isna().any(), f"{run_id}: OOF가 학습 id 전부를 덮지 않는다."
    )
    values = prediction.to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), f"{run_id}: OOF에 유한하지 않은 값이 있다.")
    return (
        prediction,
        store.artifact_sha256_of(run_id, "oof.parquet"),
        prediction_array_sha256(values),
        float(roc_auc_score(targets[TARGET].to_numpy(), values)),
    )


def _training_coordinates(
    store: MlflowRunStore, run_id: str
) -> tuple[list[int], list[int], str]:
    payload = store.artifact_bytes_of(run_id, "training_row_evidence.json")
    evidence = json.loads(payload)
    entries = evidence.get("entries")
    _require(
        isinstance(entries, list) and len(entries) == 15,
        f"{run_id}: 학습 좌표가 15개가 아니다.",
    )
    _require(
        all(
            all(bool(value) for value in entry["assertions"].values())
            for entry in entries
        ),
        f"{run_id}: 학습 행 단언 가운데 실패가 있다.",
    )
    _require(
        not any(
            entry["validation_augmented"] or entry["test_augmented"]
            for entry in entries
        ),
        f"{run_id}: 검증 또는 시험 자료가 증강됐다.",
    )
    seeds = sorted({int(entry["seed"]) for entry in entries})
    folds = sorted({int(entry["outer_fold"]) for entry in entries})
    return seeds, folds, store.artifact_sha256_of(run_id, "training_row_evidence.json")


def _runtime_identity_fields(
    store: MlflowRunStore,
    run_id: str,
    arm_name: str,
    facts: Any,
) -> tuple[str, str, str | None]:
    container_identity = facts.tags.get("remote.container_image_identity_sha256")
    dependency_lock = facts.tags.get("remote.dependency_lock_sha256")
    if container_identity:
        _require(bool(dependency_lock), f"{run_id}: 의존성 잠금 신원이 없다.")
        return str(container_identity), str(dependency_lock), None

    artifact_name = f"runtime-environment-{arm_name.replace('_', '-')}.json"
    try:
        payload = store.artifact_bytes_of(run_id, artifact_name)
    except ArtifactNotFound as error:
        raise MissingnessPropagationBatchError(
            f"{run_id}: 컨테이너 신원 꼬리표와 실행 환경 근거가 모두 없다."
        ) from error
    runtime = json.loads(payload)
    _require(
        runtime.get("provider") == facts.tags.get("remote.provider"),
        f"{run_id}: 실행 환경 근거의 공급자가 꼬리표와 다르다.",
    )
    _require(
        runtime.get("runtime_class") == facts.tags.get("remote.runtime_class"),
        f"{run_id}: 실행 환경 근거의 등급이 꼬리표와 다르다.",
    )
    _require(
        runtime.get("dependency_lock_sha256") == dependency_lock,
        f"{run_id}: 실행 환경 근거의 의존성 잠금이 꼬리표와 다르다.",
    )
    _require(
        bool(runtime.get("container_image_digest")),
        f"{run_id}: 실행 환경 근거에 컨테이너 신원이 없다.",
    )
    return (
        str(runtime["container_image_digest"]),
        str(dependency_lock),
        store.artifact_sha256_of(run_id, artifact_name),
    )


def _arm_record(
    *,
    store: MlflowRunStore,
    run_id: str,
    arm_name: str,
    expected: dict[str, Any],
    pair: dict[str, Any],
    precommit: dict[str, Any],
    targets: pd.DataFrame,
) -> tuple[dict[str, Any], pd.Series]:
    facts = store.facts_of(run_id)
    expected_seeds = list(precommit["pair_execution_contract"]["value"]["seeds"])
    expected_folds = list(precommit["search_parameters"]["outer_folds"])
    _require(facts.status == "FINISHED", f"{run_id}: 완료 실행이 아니다.")
    _require(
        facts.params.get("experiment") == expected["name"],
        f"{run_id}: 실행 이름이 다르다.",
    )
    config_name = Path(expected["path"]).name
    config_sha256 = store.artifact_sha256_of(run_id, config_name)
    _require(config_sha256 == expected["sha256"], f"{run_id}: 실행 설정 해시가 다르다.")
    git_commit = facts.tags.get("git_commit")
    _require(
        git_commit == precommit["collection_contract"]["execution_source_commit"],
        f"{run_id}: 출처 커밋 {git_commit}이 사전 기록과 다르다.",
    )
    _require(
        facts.tags.get("git_dirty") == "False", f"{run_id}: 깨끗하지 않은 코드 상태다."
    )
    provider = facts.tags.get("remote.provider")
    runtime_class = facts.tags.get("remote.runtime_class")
    _require(bool(provider), f"{run_id}: 공급자 꼬리표가 없다.")
    _require(bool(runtime_class), f"{run_id}: 실행 환경 등급 꼬리표가 없다.")
    container_identity, dependency_lock, runtime_environment_sha256 = (
        _runtime_identity_fields(store, run_id, arm_name, facts)
    )
    input_sha256 = {
        name: facts.tags.get(f"sha256.{name}") for name in ("folds", "test", "train")
    }
    _require(
        input_sha256 == pair["original"]["identity"]["input_sha256"],
        f"{run_id}: 입력 자료 해시가 원본 신원과 다르다.",
    )
    seeds, folds, diagnostics_sha256 = _training_coordinates(store, run_id)
    _require(seeds == expected_seeds, f"{run_id}: 세 시드가 사전 기록과 다르다.")
    _require(folds == expected_folds, f"{run_id}: 바깥 분할이 사전 기록과 다르다.")
    prediction, oof_sha256, oof_prediction_sha256, oof_auc = _oof_series(
        store, run_id, targets
    )
    _require(
        abs(oof_auc - float(facts.metrics["auc_oof"])) <= 1e-12,
        f"{run_id}: OOF 재채점값이 중앙 기록과 다르다.",
    )
    imported_at = facts.tags.get("import.imported_at")
    _require(bool(imported_at), f"{run_id}: 중앙 반입 시각이 없다.")
    imported = datetime.fromisoformat(str(imported_at))
    cutoff = datetime.fromisoformat(
        precommit["collection_contract"]["central_import_cutoff_utc"]
    )
    _require(imported <= cutoff, f"{run_id}: 중앙 반입 마감 뒤에 들어왔다.")
    bundle_sha256 = store.artifact_sha256_of(run_id, "bundle/manifest.json")
    return (
        {
            "arm": arm_name,
            "run_id": run_id,
            "status": facts.status,
            "experiment": expected["name"],
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "git_dirty": False,
            "provider": provider,
            "runtime_class": runtime_class,
            "container_image_digest": container_identity,
            "dependency_lock_sha256": dependency_lock,
            "runtime_environment_sha256": runtime_environment_sha256,
            "seeds": seeds,
            "outer_folds": folds,
            "input_sha256": input_sha256,
            "central_imported_at_utc": str(imported_at),
            "oof_sha256": oof_sha256,
            "oof_prediction_sha256": oof_prediction_sha256,
            "oof_auc": oof_auc,
            "required_diagnostics_sha256": diagnostics_sha256,
            "execution_record_bundle_sha256": bundle_sha256,
            "integrity_verdict": "pass",
        },
        prediction,
    )


def _input_bundle(
    *,
    precommit: dict[str, Any],
    confirmation: dict[str, Any],
    store: MlflowRunStore,
    targets: pd.DataFrame,
    recorded_at_utc: str,
) -> tuple[dict[str, Any], dict[str, pd.Series], list[dict[str, Any]]]:
    completed_by_member = {
        record["member"]: record for record in confirmation["completed_pairs"]
    }
    excluded_by_member = {
        record["member"]: record for record in confirmation["excluded_pairs"]
    }
    collection: list[dict[str, Any]] = []
    augmented_predictions: dict[str, pd.Series] = {}
    classification: list[dict[str, Any]] = []
    for pair in precommit["pairs"]:
        member = pair["slot"]
        completed = completed_by_member.get(member)
        if completed is None:
            excluded = excluded_by_member.get(member)
            _require(excluded is not None, f"{member}: 중앙 반입 기록에 상태가 없다.")
            reason = excluded["reason"]
            collection.append(
                {"member": member, "status": "incomplete", "reason": reason}
            )
            classification.append(
                {
                    "member": member,
                    "source_status": "not_completed",
                    "batch_status": "incomplete",
                    "reason": reason,
                }
            )
            continue
        expected_source = precommit["collection_contract"]["execution_source_commit"]
        if completed["source_commit"] != expected_source:
            reason = (
                f"중앙 반입 출처 커밋 {completed['source_commit']}이 "
                f"사전 기록 출처 {expected_source}와 다르다"
            )
            collection.append(
                {"member": member, "status": "incomplete", "reason": reason}
            )
            classification.append(
                {
                    "member": member,
                    "source_status": "completed_after_execution_correction",
                    "batch_status": "incomplete",
                    "reason": reason,
                }
            )
            continue
        expected_arms = {arm["arm"]: arm for arm in pair["comparison_arms"]}
        arms: list[dict[str, Any]] = []
        predictions: dict[str, pd.Series] = {}
        try:
            for arm_name in ("tripled", "missingness_augmented"):
                arm, prediction = _arm_record(
                    store=store,
                    run_id=completed[arm_name]["run_id"],
                    arm_name=arm_name,
                    expected=expected_arms[arm_name],
                    pair=pair,
                    precommit=precommit,
                    targets=targets,
                )
                arms.append(arm)
                predictions[arm_name] = prediction
        except MissingnessPropagationBatchError as error:
            reason = f"중앙 반입 무결성 관문 실패: {error}"
            collection.append(
                {"member": member, "status": "incomplete", "reason": reason}
            )
            classification.append(
                {
                    "member": member,
                    "source_status": "central_integrity_failed",
                    "batch_status": "incomplete",
                    "reason": reason,
                }
            )
            continue
        _require(
            len({arm["provider"] for arm in arms}) == 1,
            f"{member}: 두 팔의 공급자가 다르다.",
        )
        _require(
            len({arm["runtime_class"] for arm in arms}) == 1,
            f"{member}: 두 팔의 실행 환경 등급이 다르다.",
        )
        _require(
            len({arm["container_image_digest"] for arm in arms}) == 1,
            f"{member}: 두 팔의 컨테이너 신원이 다르다.",
        )
        _require(
            len({arm["dependency_lock_sha256"] for arm in arms}) == 1,
            f"{member}: 두 팔의 의존성 잠금 신원이 다르다.",
        )
        by_name = {arm["arm"]: arm for arm in arms}
        direct_delta = float(by_name["missingness_augmented"]["oof_auc"]) - float(
            by_name["tripled"]["oof_auc"]
        )
        collection.append(
            {
                "member": member,
                "status": "complete",
                "arms": arms,
                "direct_oof_delta": direct_delta,
            }
        )
        augmented_predictions[member] = predictions["missingness_augmented"]
        classification.append(
            {
                "member": member,
                "source_status": "completed",
                "batch_status": "complete",
                "reason": None,
            }
        )
    complete_members = [
        record["member"] for record in collection if record["status"] == "complete"
    ]
    payload = self_hashed_payload(
        {
            "schema": INPUT_BUNDLE_SCHEMA,
            "recorded_at_utc": recorded_at_utc,
            "issue": {"number": 512, "url": ISSUE_URL},
            "map": {"number": 506, "url": MAP_URL},
            "precommit_sha256": precommit["precommit_sha256"],
            "source_confirmation": {
                "path": str(CONFIRMATION_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(CONFIRMATION_PATH),
            },
            "collection": collection,
            "complete_pair_members": complete_members,
            "classification": classification,
        },
        "input_bundle_sha256",
    )
    validate_input_bundle(payload, precommit=precommit)
    return payload, augmented_predictions, classification


def _current_pool_predictions(
    *,
    precommit: dict[str, Any],
    pool: dict[str, Any],
    store: MlflowRunStore,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    members = pool["members"]
    _require(
        [member["config"] for member in members]
        == precommit["scope"]["pool_member_order"],
        "현재 후보 풀 순서가 사전 기록과 다르다.",
    )
    pair_by_member = {pair["slot"]: pair for pair in precommit["pairs"]}
    fixed_by_member = {
        record["member"]: record for record in precommit["fixed_members"]
    }
    columns: dict[str, pd.Series] = {}
    for member in members:
        name = member["config"]
        prediction, artifact_sha256, array_sha256, _auc = _oof_series(
            store, member["run_id"], targets
        )
        if name in pair_by_member:
            expected_artifact = pair_by_member[name]["original"]["identity"][
                "oof_artifact_sha256"
            ]
            _require(
                artifact_sha256 == expected_artifact,
                f"{name}: 현재 원본 OOF 해시가 다르다.",
            )
        else:
            fixed = fixed_by_member[name]
            expected = fixed["identity"]["oof_identity"]
            if "artifact_file_sha256" in expected:
                _require(
                    artifact_sha256 == expected["artifact_file_sha256"],
                    f"{name}: 고정 구성원 OOF 파일 해시가 다르다.",
                )
            _require(
                array_sha256 == expected["sha256"],
                f"{name}: 고정 구성원 OOF 배열 해시가 다르다.",
            )
        columns[name] = prediction
    return pd.DataFrame(columns, index=targets.index)


def _preflight(
    *,
    precommit: dict[str, Any],
    bundle: dict[str, Any],
    current_predictions: pd.DataFrame,
    recorded_at_utc: str,
) -> dict[str, Any]:
    pool_order = list(precommit["scope"]["pool_member_order"])
    violations = sorted(
        spearman_violations(current_predictions, pool_order),
        key=lambda pair: (pool_order.index(pair[0]), pool_order.index(pair[1])),
    )
    correlations = current_predictions.corr(method="spearman")
    complete_by_member = {
        record["member"]: record
        for record in bundle["collection"]
        if record["status"] == "complete"
    }
    pair_by_member = {pair["slot"]: pair for pair in precommit["pairs"]}
    threshold = float(
        precommit["search_parameters"]["missingness_augmented_entry_threshold"]
    )
    eligibility: dict[str, dict[str, Any]] = {}
    for member in precommit["scope"]["pair_member_order"]:
        record = complete_by_member.get(member)
        if record is None:
            collection = next(
                item for item in bundle["collection"] if item["member"] == member
            )
            eligibility[member] = {
                "complete": False,
                "forward_eligible": False,
                "reason": collection["reason"],
            }
            continue
        augmented = next(
            arm for arm in record["arms"] if arm["arm"] == "missingness_augmented"
        )
        auc = float(augmented["oof_auc"])
        eligibility[member] = {
            "complete": True,
            "forward_eligible": auc >= threshold,
            "augmented_oof_auc": auc,
            "threshold": threshold,
            "reason": None
            if auc >= threshold
            else "결측 증강판 OOF가 검색 진입 하한보다 낮다",
        }
    violation_records = []
    unresolved = []
    for left, right in violations:
        resolver_slots = [name for name in (left, right) if name in pair_by_member]
        eligible_resolvers = [
            name for name in resolver_slots if eligibility[name]["forward_eligible"]
        ]
        item = {
            "left": left,
            "right": right,
            "spearman": float(correlations.loc[left, right]),
            "threshold": float(precommit["search_parameters"]["duplicate_spearman"]),
            "resolver_slots": resolver_slots,
            "eligible_resolvers": eligible_resolvers,
            "resolver_status": {name: eligibility[name] for name in resolver_slots},
            "reachable_resolution": bool(eligible_resolvers),
        }
        violation_records.append(item)
        if not eligible_resolvers:
            unresolved.append(item)
    payload = self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/preflight/1",
            "recorded_at_utc": recorded_at_utc,
            "precommit_sha256": precommit["precommit_sha256"],
            "input_bundle_sha256": bundle["input_bundle_sha256"],
            "current_pool": {
                "member_count": len(pool_order),
                "members": pool_order,
                "sha256": file_sha256(POOL_PATH),
            },
            "forward_eligibility": eligibility,
            "current_duplicate_violations": violation_records,
            "unresolved_duplicate_violations": unresolved,
            "reachable_valid_proposal": not unresolved,
            "proof": (
                "원자 교체는 자기 자리의 예측만 바꾼다. 현재 중복 위반의 두 자리 가운데 "
                "검색 이동 자격이 있는 완결 결측 증강판이 하나도 없으므로 모든 도달 가능한 "
                "상태가 같은 중복 위반을 보존한다. 따라서 전체 OOF 제안 풀 중복 불변식을 "
                "만족하는 도달 가능한 상태가 없다."
            ),
        },
        "preflight_sha256",
    )
    return payload


def _judgment(
    *,
    precommit: dict[str, Any],
    bundle: dict[str, Any],
    preflight: dict[str, Any],
    runtime: dict[str, Any],
    recorded_at_utc: str,
) -> dict[str, Any]:
    _require(
        not preflight["reachable_valid_proposal"],
        "유효 제안 풀이 도달 가능하므로 종결 판정 경로를 사용할 수 없다.",
    )
    pool_sha256 = file_sha256(POOL_PATH)
    refit_sha256 = file_sha256(REFIT_PLAN_PATH)
    complete_count = len(bundle["complete_pair_members"])
    source_mismatch_count = sum(
        item["source_status"] == "completed_after_execution_correction"
        for item in bundle["classification"]
    )
    payload = self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/judgment/1",
            "recorded_at_utc": recorded_at_utc,
            "issue": {"number": 512, "url": ISSUE_URL},
            "map": {"number": 506, "url": MAP_URL},
            "runtime": runtime,
            "precommit_sha256": precommit["precommit_sha256"],
            "input_bundle_sha256": bundle["input_bundle_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "collection": {
                "state_count": len(bundle["collection"]),
                "complete_pair_count": complete_count,
                "incomplete_pair_count": len(bundle["collection"]) - complete_count,
                "source_commit_mismatch_count": source_mismatch_count,
            },
            "search": {
                "status": "not_started",
                "evaluated_state_count": 0,
                "reason_code": "proposal_duplicate_invariant_unreachable",
                "reason": preflight["proof"],
                "partial_result_adopted": False,
            },
            "conditional_gate": {"status": "not_run", "passed": False},
            "direct_nested_gate": {"status": "not_run", "passed": False},
            "full_refit_rehearsal": {"status": "not_run", "passed": False},
            "verdict": {
                "status": "keep_current_ledgers",
                "proposal_pool": None,
                "selected_replacements": [],
                "public_score_used": False,
                "formalized": False,
            },
            "official_ledgers": {
                "candidate_pool": {
                    "path": str(POOL_PATH.relative_to(REPO_ROOT)),
                    "before_sha256": pool_sha256,
                    "after_sha256": pool_sha256,
                    "changed": False,
                },
                "full_refit_plan": {
                    "path": str(REFIT_PLAN_PATH.relative_to(REPO_ROOT)),
                    "before_sha256": refit_sha256,
                    "after_sha256": refit_sha256,
                    "changed": False,
                },
            },
        },
        "judgment_sha256",
    )
    return payload


def _report(
    bundle: dict[str, Any],
    preflight: dict[str, Any],
    judgment: dict[str, Any],
) -> str:
    mismatches = [
        item["member"]
        for item in bundle["classification"]
        if item["source_status"] == "completed_after_execution_correction"
    ]
    violations = preflight["unresolved_duplicate_violations"]
    lines = [
        "# 결측 증강 전파 일괄 판정",
        "",
        f"이 문서는 GitHub 이슈 [결측 증강 전파 후보를 동결 OOF 조건부로 일괄 판정해 공식 풀을 확정한다]({ISSUE_URL})의 변경 불가 종결 기록이다.",
        "",
        "## 결론",
        "",
        "현재 후보 풀과 전체 자료 재학습 계획을 그대로 유지한다.",
        "허용된 원자 교체로 현재 풀의 중복 위반을 해소할 수 없어 전체 OOF 제안 풀 자체가 존재하지 않는다.",
        "따라서 검색 점수, 동결 OOF 조건부 절차 관문, 핵심 결합 방식 세 가지의 직접 중첩 관문과 전체 자료 재학습 스모크 예행은 시작하지 않았다.",
        "부분 결과와 Public 점수는 판정에 사용하지 않았다.",
        "",
        "## 판정 입력",
        "",
        f"사전 고정한 34개 짝 가운데 일괄 판정 입력 묶음의 완결 짝은 {judgment['collection']['complete_pair_count']}개이고 미완결 짝은 {judgment['collection']['incomplete_pair_count']}개다.",
        f"중앙 반입에서 완결로 기록됐지만 사전 기록 출처 커밋과 달라 미완결로 분류한 짝은 {judgment['collection']['source_commit_mismatch_count']}개다.",
        "해당 짝은 다음과 같다.",
        "",
        *[f"- `{member}`" for member in mismatches],
        "",
        "그 밖의 미완결 짝은 이슈 511에서 TabCNN 계열 제외 또는 비용 검토 뒤 미실행으로 확정한 짝이다.",
        "완결 짝의 직접 OOF 차이 부호는 입력 포함 여부에 사용하지 않았다.",
        "",
        "## 도달 가능성 판정",
        "",
    ]
    for violation in violations:
        lines.extend(
            [
                f"- `{violation['left']}`와 `{violation['right']}`의 스피어만 순위 상관은 `{violation['spearman']:.10f}`로 문턱 `{violation['threshold']}` 이상이다.",
                f"- 바꿀 수 있는 자리는 {', '.join(f'`{name}`' for name in violation['resolver_slots'])}이지만 판정 입력을 통과해 검색 이동 자격을 얻은 결측 증강판은 없다.",
            ]
        )
    lines.extend(
        [
            "",
            "원자 교체는 해당 자리의 예측만 바꾸므로 다른 자리를 바꾸어 이 중복 관계를 없앨 수 없다.",
            "모든 도달 가능한 상태가 같은 위반을 보존하므로 전체 OOF 제안 풀의 모든 구성원 쌍이 `0.998` 미만이어야 한다는 조건을 만족할 수 없다.",
            "이 판정은 검색 결과를 본 뒤 문턱을 바꾼 것이 아니라 사전 기록의 출처와 중복 조건을 그대로 적용한 결과다.",
            "",
            "## 공식 장부",
            "",
            f"- `artifacts/pool.yaml`: `{judgment['official_ledgers']['candidate_pool']['after_sha256']}`",
            f"- `artifacts/full-refit-plan.yaml`: `{judgment['official_ledgers']['full_refit_plan']['after_sha256']}`",
            "",
            "두 파일은 판정 전후에 바이트 단위로 같고 이번 이슈에서 수정하지 않았다.",
            "",
            "## 근거 파일",
            "",
            f"- 입력 묶음: `{INPUT_NAME}` (`{bundle['input_bundle_sha256']}`)",
            f"- 도달 가능성 기록: `{PREFLIGHT_NAME}` (`{preflight['preflight_sha256']}`)",
            f"- 최종 판정: `{JUDGMENT_NAME}` (`{judgment['judgment_sha256']}`)",
            f"- 파일 목록: `{MANIFEST_NAME}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(output_root: Path) -> None:
    files = sorted(
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name != MANIFEST_NAME
    )
    lines = [f"{file_sha256(path)}  {path.name}" for path in files]
    (output_root / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify(output_root: Path, precommit: dict[str, Any]) -> None:
    bundle = _load_json(output_root / INPUT_NAME)
    preflight = _load_json(output_root / PREFLIGHT_NAME)
    judgment = _load_json(output_root / JUDGMENT_NAME)
    validate_input_bundle(bundle, precommit=precommit)
    verify_self_hash(preflight, "preflight_sha256")
    verify_self_hash(judgment, "judgment_sha256")
    _require(
        preflight["input_bundle_sha256"] == bundle["input_bundle_sha256"],
        "도달 가능성 기록이 다른 입력 묶음을 가리킨다.",
    )
    _require(
        judgment["preflight_sha256"] == preflight["preflight_sha256"],
        "최종 판정이 다른 도달 가능성 기록을 가리킨다.",
    )
    for ledger in judgment["official_ledgers"].values():
        path = REPO_ROOT / ledger["path"]
        actual = file_sha256(path)
        _require(
            actual == ledger["before_sha256"] == ledger["after_sha256"],
            f"공식 장부가 판정 뒤 바뀌었다: {path}",
        )
        _require(
            ledger["changed"] is False, f"공식 장부 변경 꼬리표가 잘못됐다: {path}"
        )
    manifest = output_root / MANIFEST_NAME
    _require(manifest.is_file(), "파일 SHA-256 목록이 없다.")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = output_root / name
        _require(path.is_file(), f"파일 SHA-256 목록의 파일이 없다: {name}")
        _require(file_sha256(path) == digest, f"파일 SHA-256이 다르다: {name}")


def main() -> None:
    args = _args()
    output_root = args.output_root.resolve()
    precommit = _load_json(PRECOMMIT_PATH)
    validate_precommit(precommit, REPO_ROOT)
    if args.verify_only:
        _verify(output_root, precommit)
        print(f"verified {output_root}")
        return

    runtime = _runtime_identity()
    recorded_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
    source_root = args.source_root.resolve()
    confirmation = _load_json(CONFIRMATION_PATH)
    targets = _targets(source_root)
    store = MlflowRunStore(_tracking_uri(source_root, args.tracking_uri))
    bundle, _augmented_predictions, _classification = _input_bundle(
        precommit=precommit,
        confirmation=confirmation,
        store=store,
        targets=targets,
        recorded_at_utc=recorded_at_utc,
    )
    pool = _load_yaml(POOL_PATH)
    current_predictions = _current_pool_predictions(
        precommit=precommit,
        pool=pool,
        store=store,
        targets=targets,
    )
    preflight = _preflight(
        precommit=precommit,
        bundle=bundle,
        current_predictions=current_predictions,
        recorded_at_utc=recorded_at_utc,
    )
    judgment = _judgment(
        precommit=precommit,
        bundle=bundle,
        preflight=preflight,
        runtime=runtime,
        recorded_at_utc=recorded_at_utc,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / INPUT_NAME, bundle)
    _write_json(output_root / PREFLIGHT_NAME, preflight)
    _write_json(output_root / JUDGMENT_NAME, judgment)
    (output_root / REPORT_NAME).write_text(
        _report(bundle, preflight, judgment), encoding="utf-8"
    )
    _write_manifest(output_root)
    _verify(output_root, precommit)
    print(json.dumps(judgment["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
