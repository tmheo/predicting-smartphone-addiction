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

from pipeline import ensemble as ensemble_module
from pipeline.data import ID, TARGET, file_sha256
from pipeline.missingness_propagation_batch import (
    ADOPTION_STRATEGIES,
    INPUT_BUNDLE_SCHEMA,
    MaximumGainSearch,
    MissingnessPropagationBatchError,
    self_hashed_payload,
    spearman_violations,
    validate_input_bundle,
    validate_precommit,
    verify_self_hash,
)
from pipeline.pool_audit import prediction_array_sha256
from pipeline.pool_rereview import InputContext, PoolScore, StrategyEvaluator
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
SEARCH_NAME = "search.json"
SCORE_CACHE_NAME = "search-score-cache.json"
CONDITIONAL_NAME = "conditional-gate.json"
DIRECT_NAME = "direct-nested-gate.json"
REFIT_REHEARSAL_NAME = "full-refit-rehearsal.json"
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
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--score-cache",
        type=Path,
        default=None,
        help="중단 뒤 같은 입력의 정확 검색 점수를 이어 쓸 파일",
    )
    parser.add_argument(
        "--gates-only",
        action="store_true",
        help="정확 검색과 두 OOF 관문까지만 진단하고 공식 장부는 쓰지 않는다.",
    )
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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


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


def _source_contract(
    *,
    member: str,
    completed: dict[str, Any],
    confirmation: dict[str, Any],
    precommit: dict[str, Any],
) -> dict[str, Any]:
    source_commit = str(completed["source_commit"])
    sources = confirmation["source_commits"]
    default_source = precommit["collection_contract"]["execution_source_commit"]
    if source_commit == default_source == sources["default"]:
        return {
            "kind": "precommitted_execution_source",
            "source_commit": source_commit,
            "validation_contract": None,
        }
    if source_commit == sources["xgb_diagnostic_fix_exception"]:
        _require(
            member in {"exp111_xgb_depth8_no_te", "exp135_xgb_hpo_trial30"},
            f"{member}: XGBoost 교정 출처를 허용한 대상이 아니다.",
        )
        return {
            "kind": "issue511_xgb_diagnostic_fix",
            "source_commit": source_commit,
            "validation_contract": "issue511-confirmed-xgb-diagnostic-fix",
        }
    if source_commit == sources["neural_parent_balanced_correction"]:
        contracts = {
            completed["tripled"].get("neural_validation_contract"),
            completed["missingness_augmented"].get("neural_validation_contract"),
        }
        _require(
            contracts == {"paired-parent-balanced-exposure-v2"},
            f"{member}: 신경망 교정 검증 계약이 다르다.",
        )
        return {
            "kind": "issue511_neural_parent_balanced_correction",
            "source_commit": source_commit,
            "validation_contract": "paired-parent-balanced-exposure-v2",
        }
    raise MissingnessPropagationBatchError(
        f"{member}: 이슈 511 최종 기록에 허용되지 않은 실행 출처다: {source_commit}"
    )


def _arm_record(
    *,
    store: MlflowRunStore,
    run_id: str,
    arm_name: str,
    expected: dict[str, Any],
    pair: dict[str, Any],
    precommit: dict[str, Any],
    source_contract: dict[str, Any],
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
        git_commit == source_contract["source_commit"],
        f"{run_id}: 출처 커밋 {git_commit}이 허용된 교정 계보와 다르다.",
    )
    _require(
        facts.tags.get("git_dirty") == "False", f"{run_id}: 깨끗하지 않은 코드 상태다."
    )
    if source_contract["kind"] == "issue511_neural_parent_balanced_correction":
        _require(
            facts.tags.get("issue511.valid") == "true"
            and facts.tags.get("issue511.judgment_status") == "valid"
            and facts.tags.get("issue511.validation_contract")
            == source_contract["validation_contract"],
            f"{run_id}: 이슈 511 신경망 교정 유효성 꼬리표가 다르다.",
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
            "source_contract": source_contract,
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
        source_contract = _source_contract(
            member=member,
            completed=completed,
            confirmation=confirmation,
            precommit=precommit,
        )
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
                    source_contract=source_contract,
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
                "source_status": source_contract["kind"],
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
    allowed_source_commits = {
        record["member"]: next(
            arm for arm in record["arms"] if arm["arm"] == "tripled"
        )["git_commit"]
        for record in collection
        if record["status"] == "complete"
    }
    validate_input_bundle(
        payload,
        precommit=precommit,
        allowed_source_commits=allowed_source_commits,
    )
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
    augmented_predictions: dict[str, pd.Series],
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
        resolving_moves = []
        for name in eligible_resolvers:
            candidate = current_predictions.copy()
            candidate[name] = augmented_predictions[name]
            after = sorted(spearman_violations(candidate, pool_order))
            resolving_moves.append(
                {
                    "member": name,
                    "violations_after": [list(pair) for pair in after],
                    "reaches_valid_full_oof_pool": not after,
                }
            )
        reachable = any(
            move["reaches_valid_full_oof_pool"] for move in resolving_moves
        )
        item = {
            "left": left,
            "right": right,
            "spearman": float(correlations.loc[left, right]),
            "threshold": float(precommit["search_parameters"]["duplicate_spearman"]),
            "resolver_slots": resolver_slots,
            "eligible_resolvers": eligible_resolvers,
            "resolving_moves": resolving_moves,
            "resolver_status": {name: eligibility[name] for name in resolver_slots},
            "reachable_resolution": reachable,
        }
        violation_records.append(item)
        if not reachable:
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
                "이슈 511에서 유효 판정된 exp131 교정 결측 증강판은 현재 exp131과 exp157의 "
                "기존 중복 위반을 해소하면서 새 위반을 만들지 않는다. 따라서 현재 풀을 "
                "점수 기준점으로 두고 첫 채택 이동부터 전체 OOF 중복 불변식을 만족하는 "
                "정확 검색을 시작할 수 있다."
            ),
        },
        "preflight_sha256",
    )
    return payload


def _allowed_source_commits(bundle: dict[str, Any]) -> dict[str, str]:
    return {
        record["member"]: next(
            arm for arm in record["arms"] if arm["arm"] == "tripled"
        )["git_commit"]
        for record in bundle["collection"]
        if record["status"] == "complete"
    }


def _evaluation_context(
    *,
    source_root: Path,
    precommit: dict[str, Any],
    current_predictions: pd.DataFrame,
    augmented_predictions: dict[str, pd.Series],
    targets: pd.DataFrame,
    strategies: list[str],
) -> tuple[InputContext, dict[int, str], dict[str, str]]:
    folds_frame = pd.read_parquet(REPO_ROOT / precommit["inputs"]["folds"]["path"])
    ids = current_predictions.index
    _require(folds_frame[ID].tolist() == ids.tolist(), "현재 OOF와 분할 행 순서가 다르다.")
    folds = pd.Series(folds_frame["fold"].to_numpy(), index=ids, name="fold")
    labels = targets[TARGET].reindex(ids)
    _require(not labels.isna().any(), "현재 OOF와 목표값 행이 다르다.")
    bands = ensemble_module.missingness_bands(
        source_root / "data/train.csv", source_root / "data/test.csv"
    ).reindex(ids)
    _require(not bands.isna().any(), "결측 개수 구간이 현재 OOF 행을 덮지 않는다.")

    predictions = current_predictions.copy()
    augmented_columns: dict[str, str] = {}
    for member, prediction in augmented_predictions.items():
        column = f"missingness_augmented:{member}"
        predictions[column] = prediction.reindex(ids)
        augmented_columns[member] = column
    _require(
        np.isfinite(predictions.to_numpy(dtype=np.float64)).all(),
        "검색 예측 행렬에 유한하지 않은 값이 있다.",
    )
    ordinal_members = {
        int(pair["ordinal"]): str(pair["slot"]) for pair in precommit["pairs"]
    }
    context = InputContext(
        predictions=predictions,
        labels=labels,
        folds=folds,
        missingness_bands=bands.astype(np.int8),
        ledger={
            "strategies": {"included": strategies},
            "candidate_pool": {
                "members": list(precommit["scope"]["pool_member_order"])
            },
        },
        baseline={},
        source_hashes={},
        prediction_file_sha256="",
        member_prediction_sha256={},
    )
    return context, ordinal_members, augmented_columns


def _members_for_state(
    state: tuple[int, ...],
    *,
    pool_order: list[str],
    ordinal_members: dict[int, str],
    augmented_columns: dict[str, str],
) -> tuple[str, ...]:
    selected = {ordinal_members[ordinal] for ordinal in state}
    return tuple(
        augmented_columns[member] if member in selected else member
        for member in pool_order
    )


def _correlation_violations(
    correlations: pd.DataFrame,
    members: tuple[str, ...],
    threshold: float,
) -> tuple[tuple[str, str], ...]:
    violations = []
    for position, left in enumerate(members):
        for right in members[position + 1 :]:
            if float(correlations.loc[left, right]) >= threshold:
                violations.append((left, right))
    return tuple(violations)


def _pool_score_payload(score: PoolScore) -> dict[str, Any]:
    return {
        "members": list(score.members),
        "best_strategy": score.best_strategy,
        "best_auc": score.best_auc,
        "best_fold_auc": score.best_fold_auc,
        "strategy_auc": score.strategy_auc,
        "strategy_fold_auc": score.strategy_fold_auc,
    }


def _search_and_gates(
    *,
    source_root: Path,
    precommit: dict[str, Any],
    bundle: dict[str, Any],
    preflight: dict[str, Any],
    current_predictions: pd.DataFrame,
    augmented_predictions: dict[str, pd.Series],
    targets: pd.DataFrame,
    jobs: int,
    score_cache_path: Path,
    recorded_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(preflight["reachable_valid_proposal"], "유효 제안 풀에 도달할 수 없다.")
    pool_order = list(precommit["scope"]["pool_member_order"])
    search_strategy = str(precommit["search_parameters"]["strategy"])
    search_context, ordinal_members, augmented_columns = _evaluation_context(
        source_root=source_root,
        precommit=precommit,
        current_predictions=current_predictions,
        augmented_predictions=augmented_predictions,
        targets=targets,
        strategies=[search_strategy],
    )
    pair_order = tuple(int(pair["ordinal"]) for pair in precommit["pairs"])
    eligible = tuple(
        int(pair["ordinal"])
        for pair in precommit["pairs"]
        if preflight["forward_eligibility"][pair["slot"]]["forward_eligible"]
    )
    threshold = float(precommit["search_parameters"]["duplicate_spearman"])
    all_correlations = search_context.predictions.corr(method="spearman")
    outer_correlations = {
        fold: search_context.predictions.loc[
            search_context.folds.to_numpy() != fold
        ].corr(method="spearman")
        for fold in precommit["search_parameters"]["outer_folds"]
    }
    cache_identity = {
        "schema": "missingness-propagation-batch-v1/search-score-cache/1",
        "precommit_sha256": precommit["precommit_sha256"],
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "strategy": search_strategy,
    }
    if score_cache_path.is_file():
        score_cache = _load_json(score_cache_path)
        verify_self_hash(score_cache, "score_cache_sha256")
        _require(
            all(score_cache.get(key) == value for key, value in cache_identity.items()),
            "정확 검색 점수 이어쓰기 파일의 입력 신원이 다르다.",
        )
    else:
        score_cache = {**cache_identity, "scores": {}}
    score_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    cache_misses = 0

    def flush_score_cache() -> None:
        payload = self_hashed_payload(
            {
                key: value
                for key, value in score_cache.items()
                if key != "score_cache_sha256"
            },
            "score_cache_sha256",
        )
        score_cache.clear()
        score_cache.update(payload)
        _write_json_atomic(score_cache_path, score_cache)

    def members(state: tuple[int, ...]) -> tuple[str, ...]:
        return _members_for_state(
            state,
            pool_order=pool_order,
            ordinal_members=ordinal_members,
            augmented_columns=augmented_columns,
        )

    def run_search(
        evaluator: StrategyEvaluator, excluded_fold: int | None
    ) -> tuple[Any, tuple[tuple[str, str], ...]]:
        correlations = (
            all_correlations
            if excluded_fold is None
            else outer_correlations[excluded_fold]
        )
        baseline_members = members(())
        baseline_violations = _correlation_violations(
            correlations, baseline_members, threshold
        )

        def allowed(state: tuple[int, ...]) -> bool:
            violations = _correlation_violations(correlations, members(state), threshold)
            if excluded_fold is None:
                return not violations
            return set(violations) <= set(baseline_violations)

        def score_many(states: tuple[tuple[int, ...], ...]) -> list[float]:
            nonlocal cache_hits, cache_misses
            scope = "full_oof" if excluded_fold is None else f"outer_{excluded_fold}"
            cached = score_cache["scores"].setdefault(scope, {})
            keys = [",".join(str(value) for value in state) for state in states]
            missing_positions = [
                position for position, key in enumerate(keys) if key not in cached
            ]
            cache_hits += len(states) - len(missing_positions)
            cache_misses += len(missing_positions)
            if missing_positions:
                missing_states = [states[position] for position in missing_positions]
                evaluated = evaluator.evaluate_many(
                    [(members(state), None) for state in missing_states],
                    excluded_fold=excluded_fold,
                )
                for position, score in zip(
                    missing_positions, evaluated, strict=True
                ):
                    cached[keys[position]] = score.best_auc
                flush_score_cache()
            return [float(cached[key]) for key in keys]

        search = MaximumGainSearch(
            pair_order=pair_order,
            eligible=eligible,
            score=lambda state: score_many((state,))[0],
            score_many=score_many,
            allowed=allowed,
            allow_invalid_start=excluded_fold is None and bool(baseline_violations),
        ).run()
        return search, baseline_violations

    with StrategyEvaluator(search_context, jobs) as evaluator:
        full_search, full_baseline_violations = run_search(evaluator, None)
        outer_searches = {}
        outer_baseline_violations = {}
        for fold in precommit["search_parameters"]["outer_folds"]:
            result, violations = run_search(evaluator, int(fold))
            outer_searches[str(fold)] = result
            outer_baseline_violations[str(fold)] = violations
        search_fits = evaluator.fits
        search_arm_evaluations = evaluator.arm_evaluations

    proposal_members = members(full_search.selected)
    proposal_violations = _correlation_violations(
        all_correlations, proposal_members, threshold
    )
    _require(not proposal_violations, "전체 OOF 검색 제안 풀이 중복 불변식을 어겼다.")
    search_payload = self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/search/2",
            "recorded_at_utc": recorded_at_utc,
            "precommit_sha256": precommit["precommit_sha256"],
            "input_bundle_sha256": bundle["input_bundle_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "strategy": search_strategy,
            "eligible_ordinals": list(eligible),
            "eligible_members": [ordinal_members[value] for value in eligible],
            "full_oof": full_search.as_payload(),
            "outer": {
                fold: result.as_payload() for fold, result in outer_searches.items()
            },
            "baseline_duplicate_violations": {
                "full_oof": [list(pair) for pair in full_baseline_violations],
                "outer": {
                    fold: [list(pair) for pair in violations]
                    for fold, violations in outer_baseline_violations.items()
                },
            },
            "proposal": {
                "selected_ordinals": list(full_search.selected),
                "selected_replacements": [
                    ordinal_members[value] for value in full_search.selected
                ],
                "members": list(proposal_members),
                "duplicate_violations": [list(pair) for pair in proposal_violations],
            },
            "execution": {
                "jobs": jobs,
                "fits": search_fits,
                "arm_evaluations": search_arm_evaluations,
                "score_cache": {
                    "path": str(score_cache_path),
                    "sha256": score_cache["score_cache_sha256"],
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "state_count": sum(
                        len(values) for values in score_cache["scores"].values()
                    ),
                },
            },
        },
        "search_sha256",
    )

    labels = search_context.labels.to_numpy(dtype=np.int8)
    folds = search_context.folds.to_numpy(dtype=np.int8)
    baseline_sealed = np.full(len(labels), np.nan, dtype=np.float64)
    proposal_sealed = np.full(len(labels), np.nan, dtype=np.float64)
    conditional_folds = {}
    for fold in precommit["search_parameters"]["outer_folds"]:
        fold = int(fold)
        train_mask = folds != fold
        test_mask = folds == fold
        fold_proposal = members(outer_searches[str(fold)].selected)
        fold_record = {}
        for label, selected_members, target in (
            ("current", members(()), baseline_sealed),
            ("proposal", fold_proposal, proposal_sealed),
        ):
            combiner = ensemble_module.combiner_for_context(
                search_strategy,
                fold_of=search_context.folds,
                band_of=search_context.missingness_bands,
            )
            fitted = combiner.fit(
                search_context.predictions.loc[train_mask, list(selected_members)],
                search_context.labels.loc[train_mask],
            )
            prediction = np.asarray(
                fitted.predict(
                    search_context.predictions.loc[test_mask, list(selected_members)]
                ),
                dtype=np.float64,
            )
            target[test_mask] = prediction
            fold_record[label] = {
                "members": list(selected_members),
                "auc": float(roc_auc_score(labels[test_mask], prediction)),
                "prediction_sha256": prediction_array_sha256(prediction),
            }
        conditional_folds[str(fold)] = fold_record
    _require(
        np.isfinite(baseline_sealed).all() and np.isfinite(proposal_sealed).all(),
        "조건부 절차 예측이 전체 행을 덮지 않는다.",
    )
    baseline_conditional_auc = float(roc_auc_score(labels, baseline_sealed))
    proposal_conditional_auc = float(roc_auc_score(labels, proposal_sealed))
    conditional_delta = proposal_conditional_auc - baseline_conditional_auc
    selected_sets = [set(result.selected) for result in outer_searches.values()]
    conditional_payload = self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/conditional-gate/1",
            "recorded_at_utc": recorded_at_utc,
            "search_sha256": search_payload["search_sha256"],
            "strategy": search_strategy,
            "folds": conditional_folds,
            "current": {
                "auc": baseline_conditional_auc,
                "prediction_sha256": prediction_array_sha256(baseline_sealed),
            },
            "proposal": {
                "auc": proposal_conditional_auc,
                "prediction_sha256": prediction_array_sha256(proposal_sealed),
            },
            "delta": conditional_delta,
            "passed": conditional_delta > 0.0,
            "selection_stability": [
                {
                    "ordinal": ordinal,
                    "member": ordinal_members[ordinal],
                    "selected_outer_fold_count": sum(
                        ordinal in selected for selected in selected_sets
                    ),
                    "selected_in_full_oof_proposal": ordinal
                    in set(full_search.selected),
                }
                for ordinal in eligible
            ],
        },
        "conditional_gate_sha256",
    )

    direct_context, _, _ = _evaluation_context(
        source_root=source_root,
        precommit=precommit,
        current_predictions=current_predictions,
        augmented_predictions=augmented_predictions,
        targets=targets,
        strategies=list(ADOPTION_STRATEGIES),
    )
    with StrategyEvaluator(direct_context, jobs) as evaluator:
        current_score, proposal_score = evaluator.evaluate_many(
            [(members(()), None), (proposal_members, None)], excluded_fold=None
        )
        direct_fits = evaluator.fits
    direct_delta = proposal_score.best_auc - current_score.best_auc
    direct_payload = self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/direct-nested-gate/1",
            "recorded_at_utc": recorded_at_utc,
            "search_sha256": search_payload["search_sha256"],
            "strategies": list(ADOPTION_STRATEGIES),
            "current": _pool_score_payload(current_score),
            "proposal": _pool_score_payload(proposal_score),
            "best_strategy_delta": direct_delta,
            "passed": direct_delta > 0.0,
            "diagnostics": {
                "fold_delta": {
                    fold: proposal_score.best_fold_auc[fold]
                    - current_score.best_fold_auc[fold]
                    for fold in current_score.best_fold_auc
                },
                "outer_fold_wins": sum(
                    proposal_score.best_fold_auc[fold]
                    > current_score.best_fold_auc[fold]
                    for fold in current_score.best_fold_auc
                ),
                "fits": direct_fits,
            },
        },
        "direct_nested_gate_sha256",
    )
    return search_payload, conditional_payload, direct_payload


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
    validate_precommit(
        precommit,
        REPO_ROOT,
        allow_contract_module_correction=True,
    )
    if args.verify_only:
        _verify(output_root, precommit)
        print(f"verified {output_root}")
        return

    runtime = _runtime_identity()
    recorded_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
    source_root = args.source_root.resolve()
    score_cache_path = (
        args.score_cache.resolve()
        if args.score_cache is not None
        else output_root / SCORE_CACHE_NAME
    )
    confirmation = _load_json(CONFIRMATION_PATH)
    targets = _targets(source_root)
    store = MlflowRunStore(_tracking_uri(source_root, args.tracking_uri))
    bundle, augmented_predictions, _classification = _input_bundle(
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
        augmented_predictions=augmented_predictions,
        recorded_at_utc=recorded_at_utc,
    )
    if args.gates_only:
        search, conditional, direct = _search_and_gates(
            source_root=source_root,
            precommit=precommit,
            bundle=bundle,
            preflight=preflight,
            current_predictions=current_predictions,
            augmented_predictions=augmented_predictions,
            targets=targets,
            jobs=args.jobs,
            score_cache_path=score_cache_path,
            recorded_at_utc=recorded_at_utc,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / INPUT_NAME, bundle)
        _write_json(output_root / PREFLIGHT_NAME, preflight)
        _write_json(output_root / SEARCH_NAME, search)
        _write_json(output_root / CONDITIONAL_NAME, conditional)
        _write_json(output_root / DIRECT_NAME, direct)
        _write_manifest(output_root)
        print(
            json.dumps(
                {
                    "selected_replacements": search["proposal"][
                        "selected_replacements"
                    ],
                    "full_oof_search_auc": search["full_oof"]["score"],
                    "conditional_delta": conditional["delta"],
                    "conditional_passed": conditional["passed"],
                    "direct_delta": direct["best_strategy_delta"],
                    "direct_passed": direct["passed"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
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
