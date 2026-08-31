"""이슈 512의 결측 증강 전파 일괄 판정 입력과 종결 판정을 기록한다.

사전 기록의 출처 커밋과 중앙 반입 결과를 다시 맞춰 34개 짝의 완결 상태를
변경 불가 입력 묶음으로 만든다.
현재 공식 풀의 중복 위반을 허용된 원자 교체로 해소할 수 없는 경우에는 검색과
후속 관문을 시작하지 않고 두 공식 장부 유지 판정을 기록한다.
제안이 두 OOF 관문을 통과하면 제안 풀과 재학습 계획의 정적 준비 상태만 검증하며,
모델 학습과 시험 예측 생성은 후속 생산 단계로 넘긴다.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from pipeline import ensemble as ensemble_module
from pipeline.data import ID, TARGET, file_sha256
from pipeline.ledger import Pool
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
from pipeline.refit_plan import RefitPlan
from pipeline.runs import ArtifactNotFound, MlflowRunStore
from pipeline.training_length import derive_refit_budgets, observe_training_length

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
SELECTION_NAME = "selection-evidence.json"
REFIT_READINESS_NAME = "full-refit-readiness.json"
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
        "--score-batch-size",
        type=int,
        default=16,
        help="정확 검색 점수를 이 개수씩 평가하고 이어쓰기 파일에 반영한다.",
    )
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
    parser.add_argument(
        "--recorded-at-utc",
        default=None,
        help=argparse.SUPPRESS,
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


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_yaml(temporary, payload)
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
    score_batch_size: int,
    score_cache_path: Path,
    recorded_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(preflight["reachable_valid_proposal"], "유효 제안 풀에 도달할 수 없다.")
    _require(score_batch_size >= 1, "검색 점수 묶음 크기는 1 이상이어야 한다.")
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
                for start in range(0, len(missing_states), score_batch_size):
                    state_chunk = missing_states[start : start + score_batch_size]
                    position_chunk = missing_positions[
                        start : start + score_batch_size
                    ]
                    evaluated = evaluator.evaluate_many(
                        [(members(state), None) for state in state_chunk],
                        excluded_fold=excluded_fold,
                    )
                    for position, score in zip(
                        position_chunk, evaluated, strict=True
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
        if excluded_fold is None and search.selected:
            # 공식 장부는 선택된 각 교체의 단일 제거 점수를 요구한다. 중복 해소를
            # 강제하는 교체를 빼면 허용되지 않는 상태가 되므로 검색 이동에서는 그
            # 상태를 평가하지 않는다. 선택에는 쓰지 않고 기여 진단에만 쓸 단일 제거
            # 상태를 여기서 명시적으로 점수화한다.
            score_many(
                tuple(
                    tuple(
                        value
                        for value in search.selected
                        if value != removed
                    )
                    for removed in search.selected
                )
            )
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
    selected_slots = {ordinal_members[value] for value in full_search.selected}
    formal_member_by_slot = {
        record["member"]: next(
            arm["experiment"]
            for arm in record["arms"]
            if arm["arm"] == "missingness_augmented"
        )
        for record in bundle["collection"]
        if record["status"] == "complete"
    }
    formal_proposal_members = [
        formal_member_by_slot[member] if member in selected_slots else member
        for member in pool_order
    ]
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
                "formal_members": formal_proposal_members,
                "duplicate_violations": [list(pair) for pair in proposal_violations],
            },
            "execution": {
                "jobs": jobs,
                "score_batch_size": score_batch_size,
                "fits": search_fits,
                "arm_evaluations": search_arm_evaluations,
                "score_cache": {
                    "path": SCORE_CACHE_NAME,
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


def _selection_evidence(
    *,
    precommit: dict[str, Any],
    bundle: dict[str, Any],
    preflight: dict[str, Any],
    search: dict[str, Any],
    conditional: dict[str, Any],
    direct: dict[str, Any],
    recorded_at_utc: str,
) -> dict[str, Any]:
    _require(conditional["passed"], "조건부 절차 관문을 통과하지 못했다.")
    _require(direct["passed"], "직접 중첩 관문을 통과하지 못했다.")
    return self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/selection-evidence/1",
            "recorded_at_utc": recorded_at_utc,
            "issue": {"number": 512, "url": ISSUE_URL},
            "precommit_sha256": precommit["precommit_sha256"],
            "input_bundle_sha256": bundle["input_bundle_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "search_sha256": search["search_sha256"],
            "conditional_gate_sha256": conditional["conditional_gate_sha256"],
            "direct_nested_gate_sha256": direct["direct_nested_gate_sha256"],
            "selected_ordinals": search["proposal"]["selected_ordinals"],
            "selected_replacements": search["proposal"]["selected_replacements"],
            "proposal_members": search["proposal"]["formal_members"],
            "public_score_used": False,
        },
        "selection_evidence_sha256",
    )


def _proposal_pool_document(
    *,
    precommit: dict[str, Any],
    bundle: dict[str, Any],
    pool: dict[str, Any],
    search: dict[str, Any],
    conditional: dict[str, Any],
    direct: dict[str, Any],
    selection: dict[str, Any],
    current_predictions: pd.DataFrame,
    augmented_predictions: dict[str, pd.Series],
    score_cache: dict[str, Any],
) -> dict[str, Any]:
    proposal = copy.deepcopy(pool)
    pool_order = list(precommit["scope"]["pool_member_order"])
    selected = set(search["proposal"]["selected_ordinals"])
    pair_by_ordinal = {int(pair["ordinal"]): pair for pair in precommit["pairs"]}
    complete_by_member = {
        record["member"]: record
        for record in bundle["collection"]
        if record["status"] == "complete"
    }
    state = tuple(sorted(selected))
    state_key = ",".join(str(value) for value in state)
    full_scores = score_cache["scores"]["full_oof"]
    _require(
        abs(float(full_scores[state_key]) - float(search["full_oof"]["score"]))
        <= 1e-15,
        "제안 풀 점수가 정확 검색 이어쓰기 기록과 다르다.",
    )

    selected_slots: dict[str, dict[str, Any]] = {}
    for ordinal in sorted(selected):
        pair = pair_by_ordinal[ordinal]
        slot = pair["slot"]
        collection = complete_by_member[slot]
        augmented = next(
            arm for arm in collection["arms"] if arm["arm"] == "missingness_augmented"
        )
        expected = next(
            arm
            for arm in pair["comparison_arms"]
            if arm["arm"] == "missingness_augmented"
        )
        _require(
            augmented["experiment"] == expected["name"],
            f"{slot}: 결측 증강 실행 이름이 사전 기록과 다르다.",
        )
        selected_slots[slot] = {
            "ordinal": ordinal,
            "augmented": augmented,
            "expected": expected,
        }

    proposal_series: dict[str, pd.Series] = {}
    proposal_run_ids: dict[str, str] = {}
    for position, entry in enumerate(proposal["members"]):
        slot = pool_order[position]
        selected_record = selected_slots.get(slot)
        if selected_record is None:
            proposal_series[entry["config"]] = current_predictions[slot]
            proposal_run_ids[entry["config"]] = entry["run_id"]
            continue
        augmented = selected_record["augmented"]
        new_name = augmented["experiment"]
        entry["run_id"] = augmented["run_id"]
        entry["config"] = new_name
        entry["oof_auc"] = augmented["oof_auc"]
        entry["seeds"] = ",".join(str(seed) for seed in augmented["seeds"])
        entry["entered_at"] = "2026-08-30"
        without = tuple(value for value in state if value != selected_record["ordinal"])
        without_key = ",".join(str(value) for value in without)
        _require(
            without_key in full_scores,
            f"{slot}: 제안 풀 기여를 계산할 역방향 검색 점수가 없다.",
        )
        entry["reason"] = (
            "이슈 512: 교정된 24짝의 결정적 최대 상승 검색과 동결 OOF 조건부 절차 및 "
            "직접 중첩 관문을 통과한 원자 교체"
        )
        entry["evidence"] = {
            "champion_run_id": precommit["search_parameters"]["champion"]["run_id"],
            "champion_oof_auc": precommit["search_parameters"]["champion"]["oof_auc"],
            "floor_margin": float(augmented["oof_auc"])
            - float(
                precommit["search_parameters"][
                    "missingness_augmented_entry_threshold"
                ]
            ),
            "nearest_run_id": None,
            "nearest_spearman": None,
            "ensemble_auc_with": float(search["full_oof"]["score"]),
            "ensemble_auc_without": float(full_scores[without_key]),
            "contribution": float(search["full_oof"]["score"])
            - float(full_scores[without_key]),
        }
        entry["judgment"] = {
            "judgment_id": "issue512-missingness-propagation-batch-selection",
            "contract_version": "missingness-propagation-batch-v1",
            "path": str((DEFAULT_OUTPUT_ROOT / SELECTION_NAME).relative_to(REPO_ROOT)),
            "sha256": selection["selection_evidence_sha256"],
        }
        proposal_series[new_name] = augmented_predictions[slot]
        proposal_run_ids[new_name] = augmented["run_id"]

    proposal_matrix = pd.DataFrame(proposal_series, index=current_predictions.index)
    correlations = proposal_matrix.corr(method="spearman")
    for entry in proposal["members"]:
        if "judgment" not in entry or entry["judgment"].get("judgment_id") != (
            "issue512-missingness-propagation-batch-selection"
        ):
            continue
        name = entry["config"]
        neighbors = correlations[name].drop(index=name)
        nearest_name = str(neighbors.idxmax())
        entry["evidence"]["nearest_run_id"] = proposal_run_ids[nearest_name]
        entry["evidence"]["nearest_spearman"] = float(neighbors.loc[nearest_name])

    _require(
        [entry["config"] for entry in proposal["members"]]
        == search["proposal"]["formal_members"],
        "공식화 후보 풀 순서가 검색 제안과 다르다.",
    )
    _require(
        conditional["passed"] and direct["passed"],
        "채택 관문을 모두 통과하지 않은 제안을 후보 풀로 만들 수 없다.",
    )
    return proposal


def _paired_lengths_from_diagnostics(
    store: MlflowRunStore,
    run_id: str,
    *,
    expected_member: str,
    expected_model_kind: str,
) -> dict[tuple[int, int, int | None], int]:
    entries = json.loads(store.artifact_bytes_of(run_id, "model_training_diagnostics.json"))
    _require(isinstance(entries, list) and len(entries) == 15, f"{run_id}: 학습 길이 진단이 15개가 아니다.")
    coordinates: dict[tuple[int, int, int | None], int] = {}
    for entry in entries:
        evidence = entry.get("training_length_evidence")
        _require(isinstance(evidence, dict), f"{run_id}: 짝비교 학습 길이 진단이 없다.")
        _require(
            evidence.get("contract") == "paired-training-length-v1"
            and evidence.get("member") == expected_member
            and evidence.get("model_kind") == expected_model_kind,
            f"{run_id}: 짝비교 학습 길이 계보가 다르다.",
        )
        values = evidence.get("observed_training_lengths")
        if values is None:
            continue
        _require(isinstance(values, list) and values, f"{run_id}: 관측 학습 길이가 비었다.")
        inner_coordinates = [None] if len(values) == 1 else list(range(len(values)))
        for inner_member, value in zip(inner_coordinates, values, strict=True):
            coordinate = (
                int(evidence["seed"]),
                int(evidence["outer_fold"]),
                inner_member,
            )
            _require(coordinate not in coordinates, f"{run_id}: 학습 길이 좌표가 중복된다.")
            coordinates[coordinate] = int(value)
    return coordinates


def _proposal_refit_plan_document(
    *,
    precommit: dict[str, Any],
    bundle: dict[str, Any],
    search: dict[str, Any],
    store: MlflowRunStore,
    proposal_pool_sha256: str,
) -> dict[str, Any]:
    proposal = copy.deepcopy(_load_yaml(REFIT_PLAN_PATH))
    proposal["source_pool_sha256"] = proposal_pool_sha256
    pair_by_ordinal = {int(pair["ordinal"]): pair for pair in precommit["pairs"]}
    complete_by_member = {
        record["member"]: record
        for record in bundle["collection"]
        if record["status"] == "complete"
    }
    current_pool = _load_yaml(POOL_PATH)
    plan_by_name = {entry["config"]: entry for entry in proposal["members"]}
    for position, pool_member in enumerate(current_pool["members"]):
        if pool_member["config"] in plan_by_name:
            continue
        entry = _structured_refit_plan_member(store, pool_member)
        proposal["members"].insert(position, entry)
        plan_by_name[entry["config"]] = entry
    for ordinal in search["proposal"]["selected_ordinals"]:
        pair = pair_by_ordinal[int(ordinal)]
        slot = pair["slot"]
        current = plan_by_name[slot]
        collection = complete_by_member[slot]
        augmented = next(
            arm for arm in collection["arms"] if arm["arm"] == "missingness_augmented"
        )
        expected = next(
            arm
            for arm in pair["comparison_arms"]
            if arm["arm"] == "missingness_augmented"
        )
        model_kind = str(current["training_length_evidence"]["model_family"])
        paired = _paired_lengths_from_diagnostics(
            store,
            augmented["run_id"],
            expected_member=slot,
            expected_model_kind=model_kind,
        )
        expected_lengths = {
            (
                int(observation["seed"]),
                int(observation["outer_fold"]),
                observation["inner_member"],
            ): int(observation["observed_training_length"])
            for observation in current["training_length_evidence"]["observations"]
        }
        _require(
            paired == expected_lengths,
            f"{slot}: 결측 증강 실행이 물려받은 학습 길이가 현재 재학습 근거와 다르다.",
        )
        current["config"] = augmented["experiment"]
        current["config_path"] = expected["path"]
        current["lineage"] = {
            "source_run_id": augmented["run_id"],
            "source_git_commit": augmented["git_commit"],
            "source_config_path": expected["path"],
            "source_config_sha256": augmented["config_sha256"],
            "evidence_artifact_path": "model_training_diagnostics.json",
            "evidence_artifact_sha256": store.artifact_sha256_of(
                augmented["run_id"], "model_training_diagnostics.json"
            ),
        }
    _require(
        len(proposal["members"]) == len(current_pool["members"]),
        "제안 전체 자료 재학습 계획의 구성원 수가 후보 풀과 다르다.",
    )
    return proposal


def _structured_refit_plan_member(
    store: MlflowRunStore,
    pool_member: dict[str, Any],
) -> dict[str, Any]:
    name = str(pool_member["config"])
    run_id = str(pool_member["run_id"])
    config_path = Path("configs") / f"{name}.yaml"
    _require(config_path.is_file(), f"{name}: 전체 자료 재학습 설정 파일이 없다.")
    artifact_name = "model_training_diagnostics.json"
    entries = json.loads(store.artifact_bytes_of(run_id, artifact_name))
    _require(isinstance(entries, list) and entries, f"{name}: 학습 길이 진단이 없다.")
    observations: list[dict[str, Any]] = []
    observed = []
    families: set[str] = set()
    converters: set[str] = set()
    for entry in entries:
        evidence = entry.get("training_length_evidence")
        _require(isinstance(evidence, dict), f"{name}: 구조화 학습 길이 근거가 없다.")
        families.add(str(evidence["model_family"]))
        converters.add(str(evidence["converter"]))
        for item in evidence["observations"]:
            record = {
                key: item[key]
                for key in (
                    "seed",
                    "outer_fold",
                    "inner_member",
                    "raw_field",
                    "raw_value",
                    "raw_meaning",
                    "observed_training_length",
                )
            }
            observations.append(record)
            observed.append(
                observe_training_length(
                    seed=int(record["seed"]),
                    outer_fold=int(record["outer_fold"]),
                    inner_member=record["inner_member"],
                    raw_field=str(record["raw_field"]),
                    raw_value=int(record["raw_value"]),
                    raw_meaning=str(record["raw_meaning"]),
                )
            )
    _require(len(families) == 1 and len(converters) == 1, f"{name}: 학습 길이 규약이 하나가 아니다.")
    derivation = derive_refit_budgets(observed)
    facts = store.facts_of(run_id)
    return {
        "config": name,
        "config_path": str(config_path),
        "lineage": {
            "source_run_id": run_id,
            "source_git_commit": facts.tags["git_commit"],
            "source_config_path": str(config_path),
            "source_config_sha256": store.artifact_sha256_of(run_id, config_path.name),
            "evidence_artifact_path": artifact_name,
            "evidence_artifact_sha256": store.artifact_sha256_of(run_id, artifact_name),
        },
        "training_length_evidence": {
            "status": "confirmed",
            "model_family": next(iter(families)),
            "converter": next(iter(converters)),
            "observations": observations,
        },
        "refit_budget_derivation": {
            "statistic": derivation.policy.statistic,
            "multiplier": derivation.policy.multiplier,
            "rounding": derivation.policy.rounding,
            "seeds": [
                {
                    "seed": seed.seed,
                    "observed_lengths": list(seed.observed_lengths),
                    "median": seed.median,
                    "scaled": seed.scaled,
                    "budget": seed.budget,
                }
                for seed in derivation.seeds
            ],
        },
    }


def _refit_readiness(
    *,
    precommit: dict[str, Any],
    search: dict[str, Any],
    selection: dict[str, Any],
    proposal_pool: dict[str, Any],
    proposal_plan: dict[str, Any],
    store: MlflowRunStore,
    recorded_at_utc: str,
) -> dict[str, Any]:
    selected_ordinals = set(search["proposal"]["selected_ordinals"])
    selected_names = {
        next(
            arm["name"]
            for arm in pair["comparison_arms"]
            if arm["arm"] == "missingness_augmented"
        )
        for pair in precommit["pairs"]
        if int(pair["ordinal"]) in selected_ordinals
    }
    members: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="issue512-refit-readiness-") as directory:
        root = Path(directory)
        pool_path = root / "pool.yaml"
        plan_path = root / "full-refit-plan.yaml"
        _write_yaml(pool_path, proposal_pool)
        _write_yaml(plan_path, proposal_plan)
        proposal_pool_sha256 = file_sha256(pool_path)
        _require(
            proposal_plan["source_pool_sha256"] == proposal_pool_sha256,
            "재학습 준비 계획이 다른 후보 풀 해시를 가리킨다.",
        )
        executable = RefitPlan.load(plan_path).validate_for_refit(
            store=store,
            pool=Pool.load(pool_path),
            pool_sha256=proposal_pool_sha256,
        )
        for member in executable.members:
            if member.config not in selected_names:
                continue
            members.append(
                {
                    "config": member.config,
                    "config_path": str(member.config_path),
                    "planned_budgets": {
                        str(seed): budget for seed, budget in member.budgets.items()
                    },
                    "member_entry_sha256": member.entry_sha256,
                }
            )
    _require(
        {member["config"] for member in members} == selected_names,
        "새 결측 증강 구성원 전체가 검증된 재학습 계획에 들어 있지 않다.",
    )
    return self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/full-refit-readiness/1",
            "recorded_at_utc": recorded_at_utc,
            "selection_evidence_sha256": selection["selection_evidence_sha256"],
            "proposal_pool_sha256": proposal_plan["source_pool_sha256"],
            "selected_members": sorted(selected_names),
            "members": members,
            "validation_scope": [
                "proposal_pool_and_refit_plan_hash_match",
                "full_refit_plan_schema_and_lineage",
                "selected_replacements_present_in_refit_plan",
            ],
            "model_training_executed": False,
            "deferred_to_follow_up": "production full-data refit and prediction generation",
            "scope_correction": {
                "decision": "replace_issue512_model_training_with_static_readiness_validation",
                "reason": "issue 512 selects and formalizes the OOF proposal pool; model training belongs to the follow-up production path",
            },
            "passed": True,
        },
        "full_refit_readiness_sha256",
    )


def _formalize_ledgers(
    *,
    proposal_pool: dict[str, Any],
    proposal_plan: dict[str, Any],
) -> None:
    pool_before = POOL_PATH.read_bytes()
    plan_before = REFIT_PLAN_PATH.read_bytes()
    pool_temporary = POOL_PATH.with_suffix(".yaml.issue512.tmp")
    plan_temporary = REFIT_PLAN_PATH.with_suffix(".yaml.issue512.tmp")
    _write_yaml(pool_temporary, proposal_pool)
    _write_yaml(plan_temporary, proposal_plan)
    _require(
        file_sha256(pool_temporary) == proposal_plan["source_pool_sha256"],
        "공식화할 재학습 계획이 다른 후보 풀을 가리킨다.",
    )
    try:
        pool_temporary.replace(POOL_PATH)
        plan_temporary.replace(REFIT_PLAN_PATH)
    except Exception:
        POOL_PATH.write_bytes(pool_before)
        REFIT_PLAN_PATH.write_bytes(plan_before)
        pool_temporary.unlink(missing_ok=True)
        plan_temporary.unlink(missing_ok=True)
        raise


def _judgment(
    *,
    precommit: dict[str, Any],
    bundle: dict[str, Any],
    preflight: dict[str, Any],
    search: dict[str, Any],
    conditional: dict[str, Any],
    direct: dict[str, Any],
    selection: dict[str, Any] | None,
    refit_readiness: dict[str, Any] | None,
    runtime: dict[str, Any],
    recorded_at_utc: str,
    pool_before_sha256: str,
    refit_before_sha256: str,
) -> dict[str, Any]:
    _require(preflight["reachable_valid_proposal"], "유효 제안 풀에 도달할 수 없다.")
    gates_passed = bool(conditional["passed"] and direct["passed"])
    readiness_passed = bool(refit_readiness and refit_readiness["passed"])
    formalized = gates_passed and readiness_passed
    _require(
        (selection is not None) == gates_passed,
        "채택 관문 결과와 선택 근거 존재 여부가 다르다.",
    )
    _require(
        (refit_readiness is not None) == gates_passed,
        "채택 관문 결과와 재학습 준비 상태 존재 여부가 다르다.",
    )
    pool_after_sha256 = file_sha256(POOL_PATH)
    refit_after_sha256 = file_sha256(REFIT_PLAN_PATH)
    complete_count = len(bundle["complete_pair_members"])
    source_correction_count = sum(
        item["source_status"]
        in {
            "issue511_xgb_diagnostic_fix",
            "issue511_neural_parent_balanced_correction",
        }
        for item in bundle["classification"]
    )
    evaluated_state_count = int(search["full_oof"]["evaluated_state_count"]) + sum(
        int(result["evaluated_state_count"]) for result in search["outer"].values()
    )
    failure_reasons = []
    if not conditional["passed"]:
        failure_reasons.append("conditional_gate_not_strictly_positive")
    if not direct["passed"]:
        failure_reasons.append("direct_nested_gate_not_strictly_positive")
    if gates_passed and not readiness_passed:
        failure_reasons.append("full_refit_readiness_not_passed")
    payload = self_hashed_payload(
        {
            "schema": "missingness-propagation-batch-v1/judgment/3",
            "recorded_at_utc": recorded_at_utc,
            "issue": {"number": 512, "url": ISSUE_URL},
            "map": {"number": 506, "url": MAP_URL},
            "runtime": runtime,
            "scope_correction": {
                "decision": "issue512_performs_static_refit_readiness_validation_only",
                "model_training_executed": False,
                "deferred_work": "production full-data refit and prediction generation",
            },
            "precommit_sha256": precommit["precommit_sha256"],
            "input_bundle_sha256": bundle["input_bundle_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "collection": {
                "state_count": len(bundle["collection"]),
                "complete_pair_count": complete_count,
                "incomplete_pair_count": len(bundle["collection"]) - complete_count,
                "accepted_correction_count": source_correction_count,
            },
            "search": {
                "status": "completed",
                "sha256": search["search_sha256"],
                "evaluated_state_count": evaluated_state_count,
                "selected_ordinals": search["proposal"]["selected_ordinals"],
                "selected_replacements": search["proposal"]["selected_replacements"],
                "full_oof_auc": search["full_oof"]["score"],
                "partial_result_adopted": False,
            },
            "conditional_gate": {
                "status": "completed",
                "sha256": conditional["conditional_gate_sha256"],
                "delta": conditional["delta"],
                "passed": conditional["passed"],
            },
            "direct_nested_gate": {
                "status": "completed",
                "sha256": direct["direct_nested_gate_sha256"],
                "delta": direct["best_strategy_delta"],
                "passed": direct["passed"],
            },
            "full_refit_readiness": (
                {
                    "status": "completed",
                    "sha256": refit_readiness["full_refit_readiness_sha256"],
                    "passed": refit_readiness["passed"],
                    "model_training_executed": False,
                }
                if refit_readiness is not None
                else {"status": "not_run", "passed": False}
            ),
            "verdict": {
                "status": (
                    "formalize_proposal_ledgers"
                    if formalized
                    else "keep_current_ledgers"
                ),
                "proposal_pool": search["proposal"]["formal_members"],
                "selected_replacements": search["proposal"]["selected_replacements"],
                "failure_reason_codes": failure_reasons,
                "public_score_used": False,
                "formalized": formalized,
            },
            "official_ledgers": {
                "candidate_pool": {
                    "path": str(POOL_PATH.relative_to(REPO_ROOT)),
                    "before_sha256": pool_before_sha256,
                    "after_sha256": pool_after_sha256,
                    "changed": pool_before_sha256 != pool_after_sha256,
                },
                "full_refit_plan": {
                    "path": str(REFIT_PLAN_PATH.relative_to(REPO_ROOT)),
                    "before_sha256": refit_before_sha256,
                    "after_sha256": refit_after_sha256,
                    "changed": refit_before_sha256 != refit_after_sha256,
                },
            },
        },
        "judgment_sha256",
    )
    return payload


def _report(
    bundle: dict[str, Any],
    preflight: dict[str, Any],
    search: dict[str, Any],
    conditional: dict[str, Any],
    direct: dict[str, Any],
    selection: dict[str, Any] | None,
    refit_readiness: dict[str, Any] | None,
    judgment: dict[str, Any],
) -> str:
    corrected = [
        item["member"]
        for item in bundle["classification"]
        if item["source_status"]
        in {
            "issue511_xgb_diagnostic_fix",
            "issue511_neural_parent_balanced_correction",
        }
    ]
    completed = [
        record for record in bundle["collection"] if record["status"] == "complete"
    ]
    direct_positive = sum(float(record["direct_oof_delta"]) > 0.0 for record in completed)
    baseline_search_auc = float(direct["current"]["strategy_auc"][search["strategy"]])
    full_search_auc = float(search["full_oof"]["score"])
    full_search_delta = full_search_auc - baseline_search_auc
    formalized = bool(judgment["verdict"]["formalized"])
    lines = [
        "# 결측 증강 전파 일괄 판정",
        "",
        f"이 문서는 GitHub 이슈 [결측 증강 전파 후보를 동결 OOF 조건부로 일괄 판정해 공식 풀을 확정한다]({ISSUE_URL})의 변경 불가 종결 기록이다.",
        "",
        "## 결론",
        "",
        (
            "교정 실행을 포함한 정확 검색의 제안이 두 OOF 관문과 재학습 계획의 정적 준비 상태 검증을 통과해 후보 풀과 전체 자료 재학습 계획을 함께 바꿨다."
            if formalized
            else "교정 실행을 포함한 정확 검색은 유효 제안 풀을 만들었지만 필수 후속 관문 가운데 하나 이상을 통과하지 못해 후보 풀과 전체 자료 재학습 계획을 유지했다."
        ),
        f"전체 OOF 검색은 현재 풀 AUC `{baseline_search_auc:.12f}`에서 `{full_search_auc:.12f}`로 `{full_search_delta:+.12f}` 개선되는 {len(search['proposal']['selected_replacements'])}개 원자 교체를 선택했다.",
        "선택된 원본 자리는 다음과 같다.",
        "",
        *[f"- `{member}`" for member in search["proposal"]["selected_replacements"]],
        "",
        "부분 결과와 Public 점수는 판정에 사용하지 않았다.",
        "",
        "## 기존 판정 정정",
        "",
        "앞선 종결 기록은 이슈 511에서 유효성이 확인된 교정 실행 7개를 사전 기록의 오래된 출처 커밋과 다르다는 이유로 제외했다.",
        "그 결과 실제 완결 짝 24개를 17개로 줄여 읽었고, 현재 풀의 기존 중복 위반을 해소하는 `exp131_lookup_bivariate_plr5` 교정판도 제외했다.",
        "검색 상태를 한 건도 평가하지 않은 채 제안 풀에 도달할 수 없다고 결론 내린 것은 잘못이었다.",
        f"교정 계약을 적용한 이번 입력은 완결 {judgment['collection']['complete_pair_count']}짝과 미완결 {judgment['collection']['incomplete_pair_count']}짝이며, 완결 짝 중 직접 OOF 차이가 양수인 짝은 {direct_positive}개다.",
        "직접 짝비교 차이의 부호는 검색 입력 포함이나 최종 교체를 단독으로 결정하지 않았다.",
        "허용한 교정 실행은 다음과 같다.",
        "",
        *[f"- `{member}`" for member in corrected],
        "",
        "## 도달 가능성과 정확 검색",
        "",
        preflight["proof"],
        f"전체 OOF와 바깥 분할 검색에서 중복 불변식을 지킨 채 총 {judgment['search']['evaluated_state_count']}개 고유 상태를 정확 채점했다.",
        f"최종 전체 OOF 제안의 선택 번호는 `{search['proposal']['selected_ordinals']}`이고 중복 위반은 `{search['proposal']['duplicate_violations']}`다.",
        "",
        "## 채택 관문",
        "",
        f"동결 OOF 조건부 절차 점수 차이는 `{conditional['delta']:+.12f}`이며 관문 통과 여부는 `{str(conditional['passed']).lower()}`다.",
        f"핵심 결합 방식 세 가지에서 각 풀의 최선 방식끼리 비교한 직접 중첩 OOF 차이는 `{direct['best_strategy_delta']:+.12f}`이며 관문 통과 여부는 `{str(direct['passed']).lower()}`다.",
        f"현재 풀의 최선 방식은 `{direct['current']['best_strategy']}`, 제안 풀의 최선 방식은 `{direct['proposal']['best_strategy']}`다.",
        f"직접 중첩 비교의 바깥 분할 승수는 `{direct['diagnostics']['outer_fold_wins']}/5`다.",
        "",
        "## 재학습 준비 상태",
        "",
    ]
    if refit_readiness is None:
        lines.append("두 OOF 관문을 모두 통과하지 못해 재학습 계획의 준비 상태를 만들지 않았다.")
    else:
        lines.append(
            f"새로 선택된 결측 증강판 {len(refit_readiness['members'])}개가 제안 풀과 같은 해시의 검증된 재학습 계획에 포함되는지 정적으로 확인했다."
        )
        lines.append(
            "이슈 512에서는 모델 학습과 시험 예측 생성을 실행하지 않았으며 실제 전체 자료 재학습은 후속 생산 단계로 넘겼다."
        )
        lines.extend(
            [
                f"- `{member['config']}`: 계획 예산 `{member['planned_budgets']}`, 항목 해시 `{member['member_entry_sha256']}`"
                for member in refit_readiness["members"]
            ]
        )
    lines.extend(
        [
            "",
            "## 공식 장부",
            "",
            f"- `artifacts/pool.yaml`: `{judgment['official_ledgers']['candidate_pool']['after_sha256']}`",
            f"- `artifacts/full-refit-plan.yaml`: `{judgment['official_ledgers']['full_refit_plan']['after_sha256']}`",
            "",
            (
                "두 파일은 같은 공식화 경로에서 함께 바뀌었다."
                if formalized
                else "두 파일은 판정 전후에 바이트 단위로 같고 이번 이슈에서 수정하지 않았다."
            ),
            "",
            "## 근거 파일",
            "",
            f"- 입력 묶음: `{INPUT_NAME}` (`{bundle['input_bundle_sha256']}`)",
            f"- 도달 가능성 기록: `{PREFLIGHT_NAME}` (`{preflight['preflight_sha256']}`)",
            f"- 정확 검색: `{SEARCH_NAME}` (`{search['search_sha256']}`)",
            f"- 조건부 절차 관문: `{CONDITIONAL_NAME}` (`{conditional['conditional_gate_sha256']}`)",
            f"- 직접 중첩 관문: `{DIRECT_NAME}` (`{direct['direct_nested_gate_sha256']}`)",
            *(
                [f"- 선택 근거: `{SELECTION_NAME}` (`{selection['selection_evidence_sha256']}`)"]
                if selection is not None
                else []
            ),
            *(
                [f"- 재학습 준비 상태: `{REFIT_READINESS_NAME}` (`{refit_readiness['full_refit_readiness_sha256']}`)"]
                if refit_readiness is not None
                else []
            ),
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
    search = _load_json(output_root / SEARCH_NAME)
    score_cache = _load_json(output_root / SCORE_CACHE_NAME)
    conditional = _load_json(output_root / CONDITIONAL_NAME)
    direct = _load_json(output_root / DIRECT_NAME)
    judgment = _load_json(output_root / JUDGMENT_NAME)
    selection = (
        _load_json(output_root / SELECTION_NAME)
        if (output_root / SELECTION_NAME).is_file()
        else None
    )
    refit_readiness = (
        _load_json(output_root / REFIT_READINESS_NAME)
        if (output_root / REFIT_READINESS_NAME).is_file()
        else None
    )
    validate_input_bundle(
        bundle,
        precommit=precommit,
        allowed_source_commits=_allowed_source_commits(bundle),
    )
    verify_self_hash(preflight, "preflight_sha256")
    verify_self_hash(search, "search_sha256")
    verify_self_hash(score_cache, "score_cache_sha256")
    verify_self_hash(conditional, "conditional_gate_sha256")
    verify_self_hash(direct, "direct_nested_gate_sha256")
    if selection is not None:
        verify_self_hash(selection, "selection_evidence_sha256")
    if refit_readiness is not None:
        verify_self_hash(refit_readiness, "full_refit_readiness_sha256")
    verify_self_hash(judgment, "judgment_sha256")
    _require(
        preflight["input_bundle_sha256"] == bundle["input_bundle_sha256"],
        "도달 가능성 기록이 다른 입력 묶음을 가리킨다.",
    )
    _require(
        search["preflight_sha256"] == preflight["preflight_sha256"],
        "정확 검색이 다른 도달 가능성 기록을 가리킨다.",
    )
    _require(
        search["execution"]["score_cache"]["sha256"]
        == score_cache["score_cache_sha256"],
        "정확 검색이 다른 점수 이어쓰기 기록을 가리킨다.",
    )
    _require(
        conditional["search_sha256"] == search["search_sha256"]
        and direct["search_sha256"] == search["search_sha256"],
        "채택 관문이 다른 정확 검색을 가리킨다.",
    )
    gates_passed = bool(conditional["passed"] and direct["passed"])
    _require(
        (selection is not None) == gates_passed,
        "채택 관문과 선택 근거 존재 여부가 다르다.",
    )
    _require(
        (refit_readiness is not None) == gates_passed,
        "채택 관문과 재학습 준비 상태 존재 여부가 다르다.",
    )
    if selection is not None:
        _require(
            selection["search_sha256"] == search["search_sha256"]
            and selection["conditional_gate_sha256"]
            == conditional["conditional_gate_sha256"]
            and selection["direct_nested_gate_sha256"]
            == direct["direct_nested_gate_sha256"],
            "선택 근거가 다른 검색 또는 관문을 가리킨다.",
        )
    if refit_readiness is not None:
        _require(
            selection is not None
            and refit_readiness["selection_evidence_sha256"]
            == selection["selection_evidence_sha256"],
            "재학습 준비 상태가 다른 선택 근거를 가리킨다.",
        )
        _require(
            not refit_readiness["model_training_executed"],
            "이슈 512 재학습 준비 상태에서 모델 학습을 실행했다.",
        )
    _require(
        judgment["preflight_sha256"] == preflight["preflight_sha256"]
        and judgment["search"]["sha256"] == search["search_sha256"]
        and judgment["conditional_gate"]["sha256"]
        == conditional["conditional_gate_sha256"]
        and judgment["direct_nested_gate"]["sha256"]
        == direct["direct_nested_gate_sha256"]
        and (
            refit_readiness is None
            or judgment["full_refit_readiness"]["sha256"]
            == refit_readiness["full_refit_readiness_sha256"]
        ),
        "최종 판정이 다른 입력, 검색 또는 관문을 가리킨다.",
    )
    for ledger in judgment["official_ledgers"].values():
        path = REPO_ROOT / ledger["path"]
        actual = file_sha256(path)
        _require(actual == ledger["after_sha256"], f"공식 장부 해시가 다르다: {path}")
        _require(
            bool(ledger["changed"])
            == (ledger["before_sha256"] != ledger["after_sha256"]),
            f"공식 장부 변경 꼬리표가 잘못됐다: {path}",
        )
    formalized = bool(judgment["verdict"]["formalized"])
    ledger_changes = [
        bool(ledger["changed"]) for ledger in judgment["official_ledgers"].values()
    ]
    _require(
        ledger_changes == [formalized, formalized],
        "후보 풀과 전체 자료 재학습 계획의 원자 변경 상태가 다르다.",
    )
    Pool.load(POOL_PATH)
    RefitPlan.load(REFIT_PLAN_PATH)
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
    recorded_at_utc = (
        datetime.fromisoformat(args.recorded_at_utc).isoformat(timespec="seconds")
        if args.recorded_at_utc is not None
        else datetime.now(UTC).isoformat(timespec="seconds")
    )
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
    search, conditional, direct = _search_and_gates(
        source_root=source_root,
        precommit=precommit,
        bundle=bundle,
        preflight=preflight,
        current_predictions=current_predictions,
        augmented_predictions=augmented_predictions,
        targets=targets,
        jobs=args.jobs,
        score_batch_size=args.score_batch_size,
        score_cache_path=score_cache_path,
        recorded_at_utc=recorded_at_utc,
    )
    if args.gates_only:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / INPUT_NAME, bundle)
        _write_json(output_root / PREFLIGHT_NAME, preflight)
        _write_json(output_root / SEARCH_NAME, search)
        (output_root / SCORE_CACHE_NAME).write_bytes(score_cache_path.read_bytes())
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
    pool_before_sha256 = file_sha256(POOL_PATH)
    refit_before_sha256 = file_sha256(REFIT_PLAN_PATH)
    selection = None
    refit_readiness = None
    if conditional["passed"] and direct["passed"]:
        selection = _selection_evidence(
            precommit=precommit,
            bundle=bundle,
            preflight=preflight,
            search=search,
            conditional=conditional,
            direct=direct,
            recorded_at_utc=recorded_at_utc,
        )
        score_cache = _load_json(score_cache_path)
        verify_self_hash(score_cache, "score_cache_sha256")
        proposal_pool = _proposal_pool_document(
            precommit=precommit,
            bundle=bundle,
            pool=pool,
            search=search,
            conditional=conditional,
            direct=direct,
            selection=selection,
            current_predictions=current_predictions,
            augmented_predictions=augmented_predictions,
            score_cache=score_cache,
        )
        with tempfile.TemporaryDirectory(prefix="issue512-ledger-proposal-") as directory:
            proposal_pool_path = Path(directory) / "pool.yaml"
            _write_yaml(proposal_pool_path, proposal_pool)
            proposal_pool_sha256 = file_sha256(proposal_pool_path)
        proposal_plan = _proposal_refit_plan_document(
            precommit=precommit,
            bundle=bundle,
            search=search,
            store=store,
            proposal_pool_sha256=proposal_pool_sha256,
        )
        refit_readiness = _refit_readiness(
            precommit=precommit,
            search=search,
            selection=selection,
            proposal_pool=proposal_pool,
            proposal_plan=proposal_plan,
            store=store,
            recorded_at_utc=recorded_at_utc,
        )
        if refit_readiness["passed"]:
            _formalize_ledgers(
                proposal_pool=proposal_pool,
                proposal_plan=proposal_plan,
            )
    judgment = _judgment(
        precommit=precommit,
        bundle=bundle,
        preflight=preflight,
        search=search,
        conditional=conditional,
        direct=direct,
        selection=selection,
        refit_readiness=refit_readiness,
        runtime=runtime,
        recorded_at_utc=recorded_at_utc,
        pool_before_sha256=pool_before_sha256,
        refit_before_sha256=refit_before_sha256,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / INPUT_NAME, bundle)
    _write_json(output_root / PREFLIGHT_NAME, preflight)
    _write_json(output_root / SEARCH_NAME, search)
    (output_root / SCORE_CACHE_NAME).write_bytes(score_cache_path.read_bytes())
    _write_json(output_root / CONDITIONAL_NAME, conditional)
    _write_json(output_root / DIRECT_NAME, direct)
    for optional_name in (
        SELECTION_NAME,
        REFIT_READINESS_NAME,
        "full-refit-rehearsal.json",
    ):
        (output_root / optional_name).unlink(missing_ok=True)
    if selection is not None:
        _write_json(output_root / SELECTION_NAME, selection)
    if refit_readiness is not None:
        _write_json(output_root / REFIT_READINESS_NAME, refit_readiness)
    _write_json(output_root / JUDGMENT_NAME, judgment)
    (output_root / REPORT_NAME).write_text(
        _report(
            bundle,
            preflight,
            search,
            conditional,
            direct,
            selection,
            refit_readiness,
            judgment,
        ),
        encoding="utf-8",
    )
    _write_manifest(output_root)
    _verify(output_root, precommit)
    print(json.dumps(judgment["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
