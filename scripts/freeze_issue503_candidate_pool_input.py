"""후보 풀 기여를 계산하기 전에 이슈 503의 입력과 판정 규칙을 동결한다."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from pipeline.data import ID, file_sha256
from pipeline.ensemble import CANDIDATE_POOL_CORE_COMBINER_NAMES
from pipeline.judgment import (
    DUPLICATE_SPEARMAN,
    ENTRY_FLOOR_MARGIN,
    POOL_EQUIVALENCE_BAND_UPPER,
    POOL_JUDGMENT_CONTRACT_VERSION,
    canonical_name_list_sha256,
)
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import MlflowRunStore


CANDIDATE_RUN_ID = "e46d1ca38e0746209e049970d3dd2ab6"
CANDIDATE_CONFIG = "exp208_issue500_ag25_missingness_augmented"
MODEL_LINEAGE_GROUP = "exp117_ag25_gbm_r21"
ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/503"
PREDECESSOR_ISSUE_URL = (
    "https://github.com/tmheo/predicting-smartphone-addiction/issues/502"
)
PREDECESSOR_RECORD = Path(
    "artifacts/issue502-three-seed-missingness-confirmation.json"
)
POOL_PATH = Path("artifacts/pool.yaml")
FOLDS_PATH = Path("artifacts/folds.parquet")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
DEFAULT_OUTPUT = Path("artifacts/issue503-missingness-candidate-pool-freeze.json")
TRACKING_URI = "sqlite:///mlflow.db"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="기존 동결 파일을 현재 입력과 다시 대조한다.",
    )
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256(payload)


def _iso_utc(milliseconds: int | None) -> str | None:
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()


def _git_file_at_commit(commit: str, path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _artifact_paths(client: MlflowClient, run_id: str, root: str = "") -> list[str]:
    paths: list[str] = []
    for item in client.list_artifacts(run_id, root):
        if item.is_dir:
            paths.extend(_artifact_paths(client, run_id, item.path))
        else:
            paths.append(item.path)
    return sorted(paths)


def _prediction_record(payload: bytes, expected_ids: pd.Series) -> dict[str, Any]:
    frame = pd.read_parquet(io.BytesIO(payload))
    if ID not in frame or "pred" not in frame:
        raise AssertionError("예측 산출물에 id 또는 pred 열이 없다.")
    if frame[ID].duplicated().any():
        raise AssertionError("예측 산출물의 id가 중복됐다.")
    if not frame[ID].equals(expected_ids):
        raise AssertionError("예측 산출물의 id와 입력 자료의 id 순서가 다르다.")
    values = frame["pred"].to_numpy(dtype="float64")
    if not pd.Series(values).map(lambda value: bool(float("-inf") < value < float("inf"))).all():
        raise AssertionError("예측 산출물에 유한하지 않은 값이 있다.")
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "prediction_array_sha256": prediction_array_sha256(values),
    }


def _candidate_record(
    client: MlflowClient,
    store: MlflowRunStore,
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> dict[str, Any]:
    run = client.get_run(CANDIDATE_RUN_ID)
    if run.info.status != "FINISHED":
        raise AssertionError(f"후보 실행이 완료 상태가 아니다: {run.info.status}")
    if run.data.params.get("experiment") != CANDIDATE_CONFIG:
        raise AssertionError("후보 실행의 설정 이름이 사전 지정값과 다르다.")
    if run.data.params.get("seeds") != "42,43,44":
        raise AssertionError("후보 실행이 난수 42, 43, 44 평균본이 아니다.")
    if run.data.tags.get("git_dirty") != "False":
        raise AssertionError("후보 실행의 작업 트리가 깨끗하지 않다.")

    artifact_paths = _artifact_paths(client, CANDIDATE_RUN_ID)
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_payloads: dict[str, bytes] = {}
    for name in artifact_paths:
        payload = store.artifact_bytes_of(CANDIDATE_RUN_ID, name)
        artifact_payloads[name] = payload
        artifacts[name] = {"bytes": len(payload), "sha256": _sha256(payload)}

    config_path = Path("configs") / f"{CANDIDATE_CONFIG}.yaml"
    config_artifact_name = f"{CANDIDATE_CONFIG}.yaml"
    config_payload = config_path.read_bytes()
    run_commit = run.data.tags["git_commit"]
    commit_config_payload = _git_file_at_commit(run_commit, config_path)
    if config_payload != artifact_payloads[config_artifact_name]:
        raise AssertionError("현재 저장소 설정과 실행에 붙은 설정이 다르다.")
    if commit_config_payload != artifact_payloads[config_artifact_name]:
        raise AssertionError("실행 커밋의 설정과 실행에 붙은 설정이 다르다.")

    oof_name = "oof.parquet"
    test_prediction_name = "test_pred.parquet"
    prediction_records = {
        "oof": {
            "artifact": oof_name,
            "file_sha256": artifacts[oof_name]["sha256"],
            **_prediction_record(artifact_payloads[oof_name], train_ids),
        },
        "test": {
            "artifact": test_prediction_name,
            "file_sha256": artifacts[test_prediction_name]["sha256"],
            **_prediction_record(artifact_payloads[test_prediction_name], test_ids),
        },
    }

    artifact_manifest = [
        {"path": path, **artifacts[path]} for path in sorted(artifacts)
    ]
    return {
        "run_id": CANDIDATE_RUN_ID,
        "status": run.info.status,
        "experiment": CANDIDATE_CONFIG,
        "model_lineage_group": MODEL_LINEAGE_GROUP,
        "execution": {
            "git_commit": run_commit,
            "git_dirty": False,
            "start_time_utc": _iso_utc(run.info.start_time),
            "end_time_utc": _iso_utc(run.info.end_time),
            "artifact_uri": run.info.artifact_uri,
        },
        "configuration": {
            "repository_path": str(config_path),
            "artifact_path": config_artifact_name,
            "sha256": _sha256(config_payload),
            "repository_matches_run_artifact": True,
            "execution_commit_matches_run_artifact": True,
            "params": dict(sorted(run.data.params.items())),
        },
        "input_sha256": {
            "train": run.data.tags["sha256.train"],
            "test": run.data.tags["sha256.test"],
            "folds": run.data.tags["sha256.folds"],
        },
        "prediction_artifacts": prediction_records,
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_sha256": _canonical_sha256(artifact_manifest),
        "artifact_count": len(artifact_manifest),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest),
        "metrics": dict(sorted(run.data.metrics.items())),
        "tags": dict(sorted(run.data.tags.items())),
    }


def _pool_record(store: MlflowRunStore) -> dict[str, Any]:
    raw = yaml.safe_load(POOL_PATH.read_text())
    members = raw["members"]
    frozen_members = []
    for member in members:
        oof_payload = store.artifact_bytes_of(member["run_id"], "oof.parquet")
        frozen_members.append(
            {
                "run_id": member["run_id"],
                "config": member["config"],
                "oof_auc": member["oof_auc"],
                "oof_artifact_sha256": _sha256(oof_payload),
            }
        )
    return {
        "path": str(POOL_PATH),
        "sha256": file_sha256(POOL_PATH),
        "member_count": len(frozen_members),
        "members": frozen_members,
        "member_oof_manifest_sha256": _canonical_sha256(frozen_members),
    }


def _input_record() -> tuple[dict[str, Any], pd.Series, pd.Series]:
    train_ids = pd.read_csv(TRAIN_PATH, usecols=[ID])[ID]
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    folds = pd.read_parquet(FOLDS_PATH)
    if list(folds.columns) != [ID, "fold"]:
        raise AssertionError("고정 분할 파일의 열이 id, fold가 아니다.")
    if not folds[ID].equals(train_ids):
        raise AssertionError("학습 자료와 고정 분할의 id 순서가 다르다.")
    return (
        {
            "train": {
                "path": str(TRAIN_PATH),
                "sha256": file_sha256(TRAIN_PATH),
                "rows": len(train_ids),
            },
            "test": {
                "path": str(TEST_PATH),
                "sha256": file_sha256(TEST_PATH),
                "rows": len(test_ids),
            },
            "folds": {
                "path": str(FOLDS_PATH),
                "sha256": file_sha256(FOLDS_PATH),
                "rows": len(folds),
                "fold_counts": {
                    str(int(fold)): int(count)
                    for fold, count in folds["fold"].value_counts().sort_index().items()
                },
            },
        },
        train_ids,
        test_ids,
    )


def _predecessor_record() -> dict[str, Any]:
    raw = json.loads(PREDECESSOR_RECORD.read_text())
    decision = raw["decision"]
    candidate = raw["runs"]["missingness_augmented"]
    if decision["status"] != "pass":
        raise AssertionError("선행 세 시드 확인이 통과 상태가 아니다.")
    if decision["next_step"] != "build_pre_registered_candidate_pool_in_issue_503":
        raise AssertionError("선행 판정의 다음 단계가 이슈 503이 아니다.")
    if candidate["run_id"] != CANDIDATE_RUN_ID:
        raise AssertionError("선행 판정의 후보 실행이 사전 지정값과 다르다.")
    return {
        "issue": PREDECESSOR_ISSUE_URL,
        "path": str(PREDECESSOR_RECORD),
        "sha256": file_sha256(PREDECESSOR_RECORD),
        "decision_status": decision["status"],
        "next_step": decision["next_step"],
        "candidate_run_id": candidate["run_id"],
        "mean_delta": decision["observed"]["mean_delta"],
        "seed_wins": decision["observed"]["seed_wins"],
        "fold_wins": decision["observed"]["fold_wins"],
    }


def _build_record(frozen_at_utc: str) -> dict[str, Any]:
    inputs, train_ids, test_ids = _input_record()
    store = MlflowRunStore(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    candidate = _candidate_record(client, store, train_ids, test_ids)
    if candidate["input_sha256"]["train"] != inputs["train"]["sha256"]:
        raise AssertionError("후보 실행의 학습 자료 해시와 현재 입력이 다르다.")
    if candidate["input_sha256"]["test"] != inputs["test"]["sha256"]:
        raise AssertionError("후보 실행의 시험 자료 해시와 현재 입력이 다르다.")
    if candidate["input_sha256"]["folds"] != inputs["folds"]["sha256"]:
        raise AssertionError("후보 실행의 고정 분할 해시와 현재 입력이 다르다.")

    combiner_names = tuple(CANDIDATE_POOL_CORE_COMBINER_NAMES)
    return {
        "schema_version": 1,
        "issue": ISSUE_URL,
        "frozen_at_utc": frozen_at_utc,
        "purpose": "후보 풀 포함 및 제외 결과를 계산하기 전에 평가 입력과 판정 규칙을 고정한다.",
        "predecessor": _predecessor_record(),
        "candidate": candidate,
        "frozen_input": {
            "datasets_and_folds": inputs,
            "candidate_pool": _pool_record(store),
            "registered_combiners": {
                "scope": "core",
                "names": list(combiner_names),
                "names_sha256": canonical_name_list_sha256(combiner_names),
            },
        },
        "decision_contract": {
            "version": POOL_JUDGMENT_CONTRACT_VERSION,
            "action_first": "admission",
            "selection_kind": "precommitted_single",
            "candidate_selection": "선행 세 시드 확인을 통과한 결측 증강 평균본 한 건만 평가한다.",
            "comparison": "포함 풀과 제외 풀은 각각 핵심 결합 방식 세 개 가운데 최선의 nested OOF AUC를 고른다.",
            "admit_when": "included_best_auc_minus_excluded_best_auc > 0",
            "entry_floor_margin": ENTRY_FLOOR_MARGIN,
            "boundary_contribution_upper": POOL_EQUIVALENCE_BAND_UPPER,
            "duplicate_spearman_threshold": DUPLICATE_SPEARMAN,
            "duplicate_route": "문턱 이상 기존 구성원이 있으면 일반 추가와 함께 원자 교체 대조를 수행하고 결과 풀의 모든 쌍이 문턱 미만일 때만 등록한다.",
            "submission_assembly": False,
        },
    }


def main() -> None:
    args = _parse_args()
    if args.verify:
        if not args.output.is_file():
            raise SystemExit(f"동결 파일이 없다: {args.output}")
        expected = json.loads(args.output.read_text())
        actual = _build_record(expected["frozen_at_utc"])
        if actual != expected:
            raise SystemExit("동결 파일과 현재 입력이 다르다.")
        print(f"동결 검증 통과: {args.output} ({file_sha256(args.output)})")
        return

    if args.output.exists():
        raise SystemExit(f"변경 불가 동결 파일이 이미 있다: {args.output}")
    frozen_at_utc = datetime.now(UTC).isoformat()
    record = _build_record(frozen_at_utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"동결 기록 생성: {args.output} ({file_sha256(args.output)})")


if __name__ == "__main__":
    main()
