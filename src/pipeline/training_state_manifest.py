"""학습 시점 후보 child의 게시 manifest를 한 규칙으로 검증한다.

게시자, 실행 기록 묶음 반입기, 후보 풀 판정기가 이 모듈의 같은 검증기를
사용한다. 파일 SHA-256뿐 아니라 예측·중요도 내용, 복구 경계, 학습 길이
관측, 실행 정체성을 서로 결속한다.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath

import pandas as pd
import yaml

from .training_length import TrainingLengthError, observed_length_from_raw
from .training_state_contract import (
    CANDIDATE_RUNS_NAME,
    PARENT_MANIFEST_NAME,
    content_sha256,
    frame_content_sha256,
)
from .training_state_recovery import (
    BOUNDARY as RECOVERY_BOUNDARY,
    EVIDENCE_SCHEMA_VERSION as RECOVERY_EVIDENCE_SCHEMA_VERSION,
    SCHEMA_NAMESPACE as RECOVERY_SCHEMA_NAMESPACE,
)

CANDIDATE_MANIFEST_SCHEMA_VERSION = 2
TRAINING_STATE_BUNDLE_SCHEMA_VERSION = 2
RUN_KIND = "training_state_snapshot"
MANIFEST_NAME = "training_state_manifest.json"
RECOVERY_NAME = "training_state_recovery.json"
DIAGNOSTICS_NAME = "model_training_diagnostics.json"
FOLD_FEATURE_REUSE_NAME = "fold_feature_reuse.json"
OUTER_FOLDS = tuple(range(5))

_SHA256_LENGTH = 64
_SUMMARY_ARTIFACTS = {
    "feature_importance_summary.csv",
    "stage_durations.csv",
    "summary.html",
    "top30_gain.png",
}
_CORE_ARTIFACTS = {
    "oof.parquet",
    "test_pred.parquet",
    "feature_importance.parquet",
    "submission.csv",
    RECOVERY_NAME,
    DIAGNOSTICS_NAME,
    FOLD_FEATURE_REUSE_NAME,
    *_SUMMARY_ARTIFACTS,
}


class TrainingStateManifestError(ValueError):
    """학습 시점 후보 child의 게시 기록이 완전한 실행을 증명하지 못한다."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingStateManifestError(message)


def _positive_int(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1,
        f"{label}은 1 이상의 정수여야 한다.",
    )
    return value


def _sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        f"{label}은 소문자 SHA-256이어야 한다.",
    )
    return value


def _json_document(payload: bytes, label: str) -> object:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingStateManifestError(f"{label}을 JSON으로 읽을 수 없다.") from exc


def _artifact_payload(
    artifact_bytes_of: Callable[[str], bytes], name: str
) -> bytes:
    try:
        payload = artifact_bytes_of(name)
    except Exception as exc:
        raise TrainingStateManifestError(f"게시 산출물 {name}을 읽을 수 없다.") from exc
    _require(isinstance(payload, bytes), f"게시 산출물 {name} 원본이 bytes가 아니다.")
    return payload


def _frame(payload: bytes, name: str) -> pd.DataFrame:
    try:
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(payload))
        return pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise TrainingStateManifestError(f"게시 산출물 {name}을 자료틀로 읽을 수 없다.") from exc


def _canonical_config_path(value: object, config_name: str) -> str:
    _require(isinstance(value, str) and bool(value), "후보 config 경로가 비어 있다.")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.parent == PurePosixPath("configs")
        and path.suffix in {".yaml", ".yml"}
        and path.stem == config_name,
        "후보 config 경로는 configs/<config_name>.yaml 형식이어야 한다.",
    )
    return path.as_posix()


def _manifest_fields(document: dict[str, object]) -> None:
    expected = {
        "schema_version",
        "run_kind",
        "trajectory_run_id",
        "trajectory",
        "trajectory_identity_sha256",
        "candidate_set_sha256",
        "precommitted_candidates",
        "selection_rule",
        "validation_target_used_for_selection",
        "state_kind",
        "completed_epochs",
        "schedule_horizon_epochs",
        "trajectory_end_epochs",
        "candidate",
        "git_commit",
        "input_sha256",
        "stage",
        "seeds",
        "model_kind",
        "prediction_content_sha256",
        "importance_content_sha256",
        "artifact_file_sha256",
        "manifest_content_sha256",
    }
    _require(
        set(document) == expected,
        "학습 시점 후보 manifest 필드 집합이 현재 게시 계약과 다르다.",
    )


def _validate_config(
    payload: bytes,
    *,
    candidate: dict[str, object],
    document: dict[str, object],
) -> dict[str, object]:
    try:
        raw = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise TrainingStateManifestError("후보 config YAML을 읽을 수 없다.") from exc
    _require(isinstance(raw, dict), "후보 config YAML의 루트가 객체가 아니다.")
    _require(raw.get("name") == candidate["config_name"], "후보 config 이름이 계보와 다르다.")
    state = raw.get("training_state")
    _require(isinstance(state, dict), "후보 config에 training_state 선언이 없다.")
    expected_state = {
        "trajectory": document["trajectory"],
        "candidates": document["precommitted_candidates"],
        "selected": document["completed_epochs"],
        "schedule_horizon_epochs": document["schedule_horizon_epochs"],
        "trajectory_end_epochs": document["trajectory_end_epochs"],
        "state_kind": document["state_kind"],
        "selection_rule": document["selection_rule"],
    }
    _require(state == expected_state, "후보 config의 training_state 선언이 계보와 다르다.")
    model = raw.get("model")
    _require(
        isinstance(model, dict) and model.get("kind") == document["model_kind"],
        "후보 config의 모델 종류가 계보와 다르다.",
    )
    return raw


def _validate_diagnostics(
    payload: bytes,
    *,
    expected_coordinates: set[tuple[int, int]],
    document: dict[str, object],
) -> None:
    diagnostics = _json_document(payload, DIAGNOSTICS_NAME)
    _require(isinstance(diagnostics, list), "학습 관측 산출물이 목록이 아니다.")
    coordinates: set[tuple[int, int]] = set()
    for item in diagnostics:
        _require(isinstance(item, dict), "학습 관측 항목이 객체가 아니다.")
        seed = item.get("seed")
        fold = item.get("fold")
        _require(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(fold, int)
            and not isinstance(fold, bool)
            and fold >= 0,
            "학습 관측의 seed 또는 fold가 올바른 정수가 아니다.",
        )
        coordinate = (seed, fold)
        _require(coordinate not in coordinates, "학습 관측에 중복 seed/fold가 있다.")
        coordinates.add(coordinate)
        _require(item.get("model_kind") == document["model_kind"], "학습 관측 모델이 다르다.")
        state = item.get("training_state")
        _require(isinstance(state, dict), "학습 관측에 training_state가 없다.")
        expected_state = {
            "completed_epochs": document["completed_epochs"],
            "schedule_horizon_epochs": document["schedule_horizon_epochs"],
            "trajectory_end_epochs": document["trajectory_end_epochs"],
            "selection_rule": document["selection_rule"],
            "state_kind": document["state_kind"],
        }
        _require(
            all(state.get(key) == value for key, value in expected_state.items()),
            "학습 관측의 시점 계약이 child 계보와 다르다.",
        )
        evidence = item.get("training_length_evidence")
        _require(isinstance(evidence, dict), "학습 관측에 실제 학습 길이 근거가 없다.")
        _require(
            evidence.get("model_family") == document["model_kind"],
            "학습 길이 근거의 모델 종류가 child 계보와 다르다.",
        )
        raw_field = evidence.get("raw_field")
        raw_meaning = evidence.get("raw_meaning")
        _require(
            isinstance(raw_field, str)
            and bool(raw_field)
            and isinstance(raw_meaning, str)
            and evidence.get("converter") == raw_meaning,
            "학습 길이 근거의 원시 필드 또는 변환기 선언이 불완전하다.",
        )
        observations = evidence.get("observations")
        _require(isinstance(observations, list) and bool(observations), "학습 길이 관측값이 비어 있다.")
        inner_members: set[int | None] = set()
        for observation in observations:
            _require(isinstance(observation, dict), "학습 길이 관측값이 객체가 아니다.")
            _require(
                observation.get("seed") == seed
                and observation.get("outer_fold") == fold
                and observation.get("raw_field") == raw_field
                and observation.get("raw_meaning") == raw_meaning,
                "학습 길이 관측값의 좌표 또는 원시 의미가 다르다.",
            )
            try:
                converted = observed_length_from_raw(
                    observation.get("raw_value"), raw_meaning
                )
            except TrainingLengthError as exc:
                raise TrainingStateManifestError(
                    "학습 길이 관측값의 원시 값 변환이 올바르지 않다."
                ) from exc
            _require(
                converted
                == observation.get("observed_training_length")
                == document["completed_epochs"],
                "학습 길이 관측값이 후보 완료 시점과 다르다.",
            )
            inner_member = observation.get("inner_member")
            _require(
                inner_member is None
                or (isinstance(inner_member, int) and not isinstance(inner_member, bool) and inner_member >= 0),
                "학습 길이 관측값의 내부 구성원 좌표가 올바르지 않다.",
            )
            _require(inner_member not in inner_members, "학습 길이 관측값의 내부 구성원 좌표가 중복됐다.")
            inner_members.add(inner_member)
    _require(coordinates == expected_coordinates, "학습 관측의 seed/fold 집합이 불완전하다.")


def _validate_recovery(
    payload: bytes,
    *,
    expected_coordinates: set[tuple[int, int]],
    candidate: dict[str, object],
    document: dict[str, object],
) -> list[dict[str, object]]:
    evidence = _json_document(payload, RECOVERY_NAME)
    _require(isinstance(evidence, dict), "복구 근거 산출물이 객체가 아니다.")
    _require(
        evidence.get("schema_namespace") == RECOVERY_SCHEMA_NAMESPACE
        and evidence.get("schema_version") == RECOVERY_EVIDENCE_SCHEMA_VERSION
        and evidence.get("boundary") == RECOVERY_BOUNDARY,
        "복구 근거의 namespace, schema 또는 원자적 경계가 다르다.",
    )
    checkpoints = evidence.get("checkpoints")
    _require(isinstance(checkpoints, list), "복구 근거 checkpoints가 목록이 아니다.")
    coordinates: set[tuple[int, int]] = set()
    expected_candidate_set: list[dict[str, object]] | None = None
    for checkpoint in checkpoints:
        _require(isinstance(checkpoint, dict), "복구 checkpoint가 객체가 아니다.")
        _require(
            checkpoint.get("schema_namespace") == RECOVERY_SCHEMA_NAMESPACE
            and checkpoint.get("schema_version") == RECOVERY_EVIDENCE_SCHEMA_VERSION
            and checkpoint.get("boundary") == RECOVERY_BOUNDARY,
            "복구 checkpoint의 namespace, schema 또는 원자적 경계가 다르다.",
        )
        seed = checkpoint.get("seed")
        fold = checkpoint.get("fold")
        _require(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(fold, int)
            and not isinstance(fold, bool)
            and fold >= 0,
            "복구 checkpoint의 seed 또는 fold가 올바른 정수가 아니다.",
        )
        coordinate = (seed, fold)
        _require(coordinate not in coordinates, "복구 근거에 중복 seed/fold가 있다.")
        coordinates.add(coordinate)
        candidate_set = checkpoint.get("candidate_set")
        _require(isinstance(candidate_set, list) and candidate_set, "복구 후보 집합이 비어 있다.")
        _require(
            content_sha256(candidate_set) == document["candidate_set_sha256"]
            == checkpoint.get("candidate_set_sha256"),
            "복구 후보 집합의 내용 해시가 child 계보와 다르다.",
        )
        if expected_candidate_set is None:
            expected_candidate_set = candidate_set
        _require(candidate_set == expected_candidate_set, "fold마다 복구 후보 집합이 다르다.")
        identity = checkpoint.get("execution_identity")
        _require(isinstance(identity, dict), "복구 실행 정체성이 객체가 아니다.")
        _require(
            content_sha256(identity) == checkpoint.get("execution_identity_sha256"),
            "복구 실행 정체성 내용 해시가 다르다.",
        )
        run_identity = identity.get("run")
        _require(isinstance(run_identity, dict), "복구 실행 정체성에 run 문서가 없다.")
        expected_run_identity = {
            "git_commit": document["git_commit"],
            "trajectory_identity_sha256": document["trajectory_identity_sha256"],
            "input_sha256": document["input_sha256"],
            "folds_sha256": document["input_sha256"]["folds"],
            "stage": document["stage"],
            "seeds": document["seeds"],
            "model_kind": document["model_kind"],
            "trajectory_end_epochs": document["trajectory_end_epochs"],
        }
        _require(
            all(run_identity.get(key) == value for key, value in expected_run_identity.items()),
            "복구 실행 정체성이 child 계보와 다르다.",
        )
        _require(identity.get("seed") == seed and identity.get("fold") == fold, "복구 좌표가 다르다.")
        _require(
            identity.get("candidate_set") == candidate_set
            and identity.get("candidate_set_sha256") == document["candidate_set_sha256"],
            "복구 실행 정체성의 후보 집합이 다르다.",
        )
        snapshots = checkpoint.get("snapshots")
        _require(isinstance(snapshots, dict), "복구 checkpoint snapshots가 객체가 아니다.")
        snapshot_names = {
            record.get("config_name")
            for record in candidate_set
            if isinstance(record, dict)
        }
        _require(
            set(snapshots) == snapshot_names,
            "복구 checkpoint snapshot 후보 집합이 불완전하다.",
        )
        unhashed = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"reused", "manifest_content_sha256"}
        }
        _require(
            content_sha256(unhashed) == checkpoint.get("manifest_content_sha256"),
            "복구 checkpoint manifest 내용 해시가 다르다.",
        )
    _require(coordinates == expected_coordinates, "복구 근거의 seed/fold 집합이 불완전하다.")
    assert expected_candidate_set is not None
    candidate_names: set[str] = set()
    completed: list[int] = []
    child_record = {
        "config_name": candidate["config_name"],
        "config_path": candidate["config_path"],
        "config_sha256": candidate["config_sha256"],
        "completed_epochs": candidate["completed_epochs"],
        "schedule_horizon_epochs": document["schedule_horizon_epochs"],
    }
    for record in expected_candidate_set:
        _require(isinstance(record, dict), "복구 후보 항목이 객체가 아니다.")
        _require(
            set(record)
            == {
                "config_name",
                "config_path",
                "config_sha256",
                "completed_epochs",
                "schedule_horizon_epochs",
            },
            "복구 후보 항목 필드가 실행 계약과 다르다.",
        )
        _sha256(record.get("config_sha256"), "복구 후보 config")
        _require(record.get("schedule_horizon_epochs") == document["schedule_horizon_epochs"], "복구 후보 일정 지평이 다르다.")
        name = record.get("config_name")
        _require(isinstance(name, str) and name not in candidate_names, "복구 후보 이름이 비었거나 중복됐다.")
        candidate_names.add(name)
        completed.append(_positive_int(record.get("completed_epochs"), "복구 후보 완료 시점"))
    _require(sorted(completed) == document["precommitted_candidates"], "복구 후보 시점 집합이 사전 선언과 다르다.")
    _require(child_record in expected_candidate_set, "현재 child 후보가 복구 후보 집합에 없다.")
    return expected_candidate_set


def validate_candidate_manifest(
    *,
    manifest_bytes: bytes,
    tags: Mapping[str, str],
    params: Mapping[str, str],
    artifact_bytes_of: Callable[[str], bytes],
) -> dict[str, object]:
    """child manifest와 모든 최초 게시 산출물을 검증해 정규 문서를 돌려준다."""
    claimed_file_sha = tags.get("sha256.training_state_manifest")
    _sha256(claimed_file_sha, "학습 시점 후보 manifest 파일")
    _require(
        hashlib.sha256(manifest_bytes).hexdigest() == claimed_file_sha,
        "학습 시점 후보 manifest 파일 SHA-256이 태그와 다르다.",
    )
    parsed = _json_document(manifest_bytes, MANIFEST_NAME)
    _require(isinstance(parsed, dict), "학습 시점 후보 manifest가 객체가 아니다.")
    document: dict[str, object] = parsed
    _manifest_fields(document)
    _require(
        document["schema_version"] == CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "지원하지 않는 학습 시점 후보 manifest schema다.",
    )
    _require(document["run_kind"] == RUN_KIND, "학습 시점 후보 run kind가 다르다.")
    unhashed = {
        key: value for key, value in document.items() if key != "manifest_content_sha256"
    }
    _require(
        content_sha256(unhashed) == document["manifest_content_sha256"],
        "학습 시점 후보 manifest 내부 내용 해시가 다르다.",
    )

    candidate = document["candidate"]
    _require(isinstance(candidate, dict), "학습 시점 후보 정체성이 객체가 아니다.")
    _require(
        set(candidate)
        == {
            "config_name",
            "config_path",
            "config_sha256",
            "completed_epochs",
            "snapshot_identity_sha256",
        },
        "학습 시점 후보 정체성 필드가 다르다.",
    )
    config_name = candidate.get("config_name")
    _require(isinstance(config_name, str) and bool(config_name), "학습 시점 후보 이름이 비어 있다.")
    candidate["config_path"] = _canonical_config_path(candidate.get("config_path"), config_name)
    _sha256(candidate.get("config_sha256"), "후보 config")
    _sha256(candidate.get("snapshot_identity_sha256"), "후보 snapshot 정체성")
    _sha256(document["trajectory_identity_sha256"], "학습 궤적 정체성")
    _sha256(document["candidate_set_sha256"], "후보 집합")
    _sha256(document["importance_content_sha256"], "importance 내용")
    completed_epochs = _positive_int(document["completed_epochs"], "후보 완료 시점")
    _require(candidate.get("completed_epochs") == completed_epochs, "후보 완료 시점이 계보 안에서 다르다.")
    horizon = _positive_int(document["schedule_horizon_epochs"], "일정 지평")
    trajectory_end = _positive_int(document["trajectory_end_epochs"], "궤적 종료 시점")
    candidates_raw = document["precommitted_candidates"]
    _require(isinstance(candidates_raw, list) and bool(candidates_raw), "사전 고정 후보 집합이 비어 있다.")
    candidates = [_positive_int(value, "사전 고정 후보 시점") for value in candidates_raw]
    _require(candidates == sorted(set(candidates)), "사전 고정 후보 시점은 중복 없는 오름차순이어야 한다.")
    _require(completed_epochs in candidates, "현재 child 완료 시점이 사전 고정 후보에 없다.")
    _require(max(candidates) <= trajectory_end <= horizon, "후보 시점, 궤적 종료, 일정 지평 순서가 다르다.")
    _require(document["state_kind"] == "ema", "학습 시점 후보는 EMA 상태여야 한다.")
    _require(document["selection_rule"] == "precommitted", "학습 시점 후보 선택은 사전 고정이어야 한다.")
    _require(document["validation_target_used_for_selection"] is False, "검증 목표값으로 학습 시점을 선택한 후보는 허용하지 않는다.")
    seeds = document["seeds"]
    _require(isinstance(seeds, list) and bool(seeds), "학습 시점 후보 시드가 비어 있다.")
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "학습 시점 후보 시드가 정수 고유 목록이 아니다.",
    )
    input_sha256 = document["input_sha256"]
    _require(isinstance(input_sha256, dict), "학습 시점 입력 해시가 객체가 아니다.")
    _require({"train", "test", "folds"} <= set(input_sha256), "학습 시점 핵심 입력 해시가 없다.")
    for name, digest in input_sha256.items():
        _require(isinstance(name, str), "학습 시점 입력 이름이 문자열이 아니다.")
        _sha256(digest, f"학습 시점 입력 {name}")

    expected_tags = {
        "run.kind": RUN_KIND,
        "training_state.ready": "true",
        "training_state.trajectory": document["trajectory"],
        "training_state.trajectory_run_id": document["trajectory_run_id"],
        "training_state.trajectory_identity_sha256": document["trajectory_identity_sha256"],
        "training_state.candidate_set_sha256": document["candidate_set_sha256"],
        "training_state.snapshot_identity_sha256": candidate["snapshot_identity_sha256"],
        "training_state.completed_epochs": str(completed_epochs),
        "training_state.schedule_horizon_epochs": str(horizon),
        "training_state.state_kind": document["state_kind"],
        "training_state.selection_rule": document["selection_rule"],
        "git_commit": document["git_commit"],
    }
    _require(
        all(tags.get(key) == value for key, value in expected_tags.items()),
        "학습 시점 후보 manifest와 실행 태그가 다르다.",
    )
    for name, digest in input_sha256.items():
        _require(tags.get(f"sha256.{name}") == digest, f"입력 {name} 해시 태그가 계보와 다르다.")
    expected_params = {
        "experiment": config_name,
        "seeds": ",".join(str(seed) for seed in seeds),
        "stage": document["stage"],
        "model.kind": document["model_kind"],
    }
    _require(
        all(params.get(key) == value for key, value in expected_params.items()),
        "학습 시점 후보 manifest와 실행 매개변수가 다르다.",
    )

    config_artifact = PurePosixPath(candidate["config_path"]).name
    expected_artifacts = {
        *_CORE_ARTIFACTS,
        config_artifact,
        *(f"oof_seed_{seed}.parquet" for seed in seeds),
    }
    artifact_hashes = document["artifact_file_sha256"]
    _require(isinstance(artifact_hashes, dict), "게시 산출물 해시 목록이 객체가 아니다.")
    _require(set(artifact_hashes) == expected_artifacts, "최초 게시 산출물 해시 집합이 불완전하다.")
    payloads: dict[str, bytes] = {}
    for name, digest in artifact_hashes.items():
        _require(
            isinstance(name, str) and PurePosixPath(name).name == name,
            "게시 산출물 이름은 루트의 안전한 파일 이름이어야 한다.",
        )
        claimed = _sha256(digest, f"게시 산출물 {name}")
        payload = _artifact_payload(artifact_bytes_of, name)
        _require(hashlib.sha256(payload).hexdigest() == claimed, f"게시 산출물 {name}의 파일 해시가 다르다.")
        payloads[name] = payload
    _require(artifact_hashes[config_artifact] == candidate["config_sha256"], "후보 config 파일 해시가 정체성과 다르다.")
    _require(tags.get("sha256.training_state_recovery") == artifact_hashes[RECOVERY_NAME], "복구 근거 해시 태그가 게시 manifest와 다르다.")
    _require(tags.get("sha256.model_training_diagnostics") == artifact_hashes[DIAGNOSTICS_NAME], "학습 관측 해시 태그가 게시 manifest와 다르다.")
    config_document = _validate_config(
        payloads[config_artifact], candidate=candidate, document=document
    )

    prediction_hashes = document["prediction_content_sha256"]
    expected_prediction_names = {"oof", "test", *(f"oof_seed_{seed}" for seed in seeds)}
    _require(isinstance(prediction_hashes, dict) and set(prediction_hashes) == expected_prediction_names, "예측 내용 해시 집합이 불완전하다.")
    prediction_artifacts = {
        "oof": "oof.parquet",
        "test": "test_pred.parquet",
        **{f"oof_seed_{seed}": f"oof_seed_{seed}.parquet" for seed in seeds},
    }
    frames = {
        name: _frame(payloads[artifact], artifact)
        for name, artifact in prediction_artifacts.items()
    }
    for name, frame in frames.items():
        _sha256(prediction_hashes[name], f"예측 {name} 내용")
        _require(frame_content_sha256(frame) == prediction_hashes[name], f"예측 {name} 내용 해시가 다르다.")
    importance = _frame(payloads["feature_importance.parquet"], "feature_importance.parquet")
    _require(frame_content_sha256(importance) == document["importance_content_sha256"], "importance 내용 해시가 다르다.")
    oof = frames["oof"]
    _require({"id", "fold", "pred"} <= set(oof.columns), "OOF 열이 불완전하다.")
    try:
        folds = sorted(int(value) for value in oof["fold"].unique())
    except (TypeError, ValueError) as exc:
        raise TrainingStateManifestError("OOF fold 값을 정수로 읽을 수 없다.") from exc
    _require(folds == list(OUTER_FOLDS), "OOF fold 집합은 정확히 0..4여야 한다.")
    expected_coordinates = {(seed, fold) for seed in seeds for fold in folds}
    _validate_diagnostics(
        payloads[DIAGNOSTICS_NAME],
        expected_coordinates=expected_coordinates,
        document=document,
    )
    candidate_set = _validate_recovery(
        payloads[RECOVERY_NAME],
        expected_coordinates=expected_coordinates,
        candidate=candidate,
        document=document,
    )
    normalized_config = json.loads(json.dumps(config_document))
    normalized_config.pop("name", None)
    normalized_state = normalized_config.get("training_state")
    _require(isinstance(normalized_state, dict), "정규화할 training_state config가 없다.")
    normalized_state.pop("selected", None)
    trajectory_document = {
        "schema_version": 1,
        "trajectory": document["trajectory"],
        "git_commit": document["git_commit"],
        "input_sha256": document["input_sha256"],
        "stage": document["stage"],
        "seeds": document["seeds"],
        "model_kind": document["model_kind"],
        "shared_config_sha256": content_sha256(normalized_config),
        "candidate_set_sha256": content_sha256(candidate_set),
        "precommitted_candidates": document["precommitted_candidates"],
        "schedule_horizon_epochs": document["schedule_horizon_epochs"],
        "trajectory_end_epochs": document["trajectory_end_epochs"],
        "state_kind": document["state_kind"],
        "selection_rule": document["selection_rule"],
    }
    expected_trajectory_sha256 = content_sha256(trajectory_document)
    _require(
        expected_trajectory_sha256 == document["trajectory_identity_sha256"],
        "학습 궤적 정체성 해시가 실제 config, 입력, 후보 집합에서 재계산한 값과 다르다.",
    )
    expected_snapshot_sha256 = content_sha256(
        {
            "trajectory_identity_sha256": expected_trajectory_sha256,
            "config_sha256": candidate["config_sha256"],
            "completed_epochs": document["completed_epochs"],
            "schedule_horizon_epochs": document["schedule_horizon_epochs"],
        }
    )
    _require(
        expected_snapshot_sha256 == candidate["snapshot_identity_sha256"],
        "snapshot 정체성 해시가 실제 궤적과 config에서 재계산한 값과 다르다.",
    )
    return document


def validate_candidate_parent_lineage(
    *,
    child_run_id: str,
    child_document: Mapping[str, object],
    child_tags: Mapping[str, str],
    facts_of: Callable[[str], object],
    artifact_bytes_of: Callable[[str, str], bytes],
) -> None:
    """직접 실행 child를 FINISHED 부모와 사전 후보 전체에 결속한다.

    묶음 반입 child는 self-contained manifest가 이동성 경계다. 원격 부모와 형제가
    로컬에 없을 수 있으므로 ``source.kind=bundle``이면 부모 조회를 요구하지 않는다.
    """
    if child_tags.get("source.kind") == "bundle":
        _sha256(child_tags.get("import.bundle_sha256"), "반입 묶음")
        source_run_id = child_tags.get("source.run_id")
        _require(isinstance(source_run_id, str) and bool(source_run_id), "반입 출처 child run id가 없다.")
        _require(
            child_tags.get("source.trajectory_run_id")
            == child_document.get("trajectory_run_id"),
            "반입 출처 부모 run id가 child 계보와 다르다.",
        )
        try:
            bundle_manifest = _json_document(
                artifact_bytes_of(child_run_id, "bundle/manifest.json"),
                "bundle/manifest.json",
            )
        except Exception as exc:
            raise TrainingStateManifestError("반입 출처 묶음 manifest를 읽을 수 없다.") from exc
        _require(isinstance(bundle_manifest, dict), "반입 출처 묶음 manifest가 객체가 아니다.")
        bundle_tags = bundle_manifest.get("tags")
        _require(
            bundle_manifest.get("schema_version") == TRAINING_STATE_BUNDLE_SCHEMA_VERSION
            and bundle_manifest.get("source_run_id") == source_run_id
            and isinstance(bundle_tags, dict)
            and bundle_tags.get("run.kind") == RUN_KIND
            and bundle_tags.get("training_state.trajectory_identity_sha256")
            == child_document["trajectory_identity_sha256"]
            and bundle_tags.get("training_state.candidate_set_sha256")
            == child_document["candidate_set_sha256"]
            and bundle_tags.get("training_state.snapshot_identity_sha256")
            == child_document["candidate"]["snapshot_identity_sha256"],
            "반입 출처 묶음 manifest가 현재 child 정체성과 다르다.",
        )
        return
    parent_run_id = child_document.get("trajectory_run_id")
    _require(
        isinstance(parent_run_id, str) and bool(parent_run_id),
        "학습 시점 child의 부모 run id가 비어 있다.",
    )
    try:
        parent = facts_of(parent_run_id)
    except Exception as exc:
        raise TrainingStateManifestError(
            f"학습 시점 child의 부모 run {parent_run_id}을 읽을 수 없다."
        ) from exc
    parent_tags = getattr(parent, "tags", None)
    _require(getattr(parent, "status", None) == "FINISHED", "학습 궤적 부모 run이 FINISHED가 아니다.")
    _require(isinstance(parent_tags, dict), "학습 궤적 부모 run 태그가 객체가 아니다.")
    expected_parent_tags = {
        "run.kind": "training_state_trajectory",
        "judgment.eligible": "false",
        "training_state.trajectory": child_document["trajectory"],
        "training_state.trajectory_identity_sha256": child_document[
            "trajectory_identity_sha256"
        ],
        "training_state.candidate_set_sha256": child_document[
            "candidate_set_sha256"
        ],
        "git_commit": child_document["git_commit"],
        "git_dirty": "False",
        **{
            f"sha256.{name}": digest
            for name, digest in child_document["input_sha256"].items()
        },
    }
    _require(
        all(parent_tags.get(key) == value for key, value in expected_parent_tags.items()),
        "학습 궤적 부모 run 태그가 child 계보와 다르다.",
    )
    try:
        parent_manifest_bytes = artifact_bytes_of(parent_run_id, PARENT_MANIFEST_NAME)
        candidate_runs_bytes = artifact_bytes_of(parent_run_id, CANDIDATE_RUNS_NAME)
        parent_manifest = _json_document(parent_manifest_bytes, PARENT_MANIFEST_NAME)
        candidate_runs = _json_document(candidate_runs_bytes, CANDIDATE_RUNS_NAME)
    except TrainingStateManifestError:
        raise
    except Exception as exc:
        raise TrainingStateManifestError("학습 궤적 부모 계보 산출물을 읽을 수 없다.") from exc
    _require(isinstance(parent_manifest, dict), "학습 궤적 부모 manifest가 객체가 아니다.")
    _require(
        hashlib.sha256(parent_manifest_bytes).hexdigest()
        == parent_tags.get("sha256.training_state_trajectory")
        and hashlib.sha256(candidate_runs_bytes).hexdigest()
        == parent_tags.get("sha256.training_state_candidate_runs"),
        "학습 궤적 부모 계보 산출물 파일 해시가 태그와 다르다.",
    )
    expected_parent = {
        "schema_version": 1,
        "run_kind": "training_state_trajectory",
        "trajectory": child_document["trajectory"],
        "trajectory_identity_sha256": child_document["trajectory_identity_sha256"],
        "candidate_set_sha256": child_document["candidate_set_sha256"],
        "git_commit": child_document["git_commit"],
        "input_sha256": child_document["input_sha256"],
        "stage": child_document["stage"],
        "seeds": child_document["seeds"],
        "model_kind": child_document["model_kind"],
        "state_kind": child_document["state_kind"],
        "selection_rule": child_document["selection_rule"],
        "precommitted_candidates": child_document["precommitted_candidates"],
        "schedule_horizon_epochs": child_document["schedule_horizon_epochs"],
        "trajectory_end_epochs": child_document["trajectory_end_epochs"],
    }
    _require(
        all(parent_manifest.get(key) == value for key, value in expected_parent.items()),
        "학습 궤적 부모 manifest가 child 계보와 다르다.",
    )
    parent_candidates = parent_manifest.get("candidates")
    _require(isinstance(parent_candidates, list), "학습 궤적 부모 후보 목록이 없다.")
    _require(
        [item.get("completed_epochs") for item in parent_candidates if isinstance(item, dict)]
        == child_document["precommitted_candidates"],
        "학습 궤적 부모 후보 목록이 사전 고정 집합과 다르다.",
    )
    _require(isinstance(candidate_runs, dict), "부모 candidate-runs 산출물이 객체가 아니다.")
    _require(
        candidate_runs.get("schema_version") == 1
        and candidate_runs.get("trajectory_identity_sha256")
        == child_document["trajectory_identity_sha256"],
        "부모 candidate-runs 산출물의 schema 또는 궤적 정체성이 다르다.",
    )
    runs = candidate_runs.get("runs")
    _require(isinstance(runs, list), "부모 candidate-runs 목록이 없다.")
    _require(len(runs) == len(parent_candidates), "부모 candidate-runs 목록이 불완전하다.")
    current_candidate = child_document["candidate"]
    seen_run_ids: set[str] = set()
    current_found = False
    for identity, record in zip(parent_candidates, runs, strict=True):
        _require(isinstance(identity, dict) and isinstance(record, dict), "부모 후보 계보 항목이 객체가 아니다.")
        run_id = record.get("run_id")
        _require(isinstance(run_id, str) and run_id not in seen_run_ids, "부모 후보 run id가 비었거나 중복됐다.")
        seen_run_ids.add(run_id)
        expected_record = {**identity, "run_id": run_id, "status": "FINISHED"}
        _require(record == expected_record, "부모 candidate-runs 항목이 후보 정체성과 다르다.")
        try:
            sibling = facts_of(run_id)
        except Exception as exc:
            raise TrainingStateManifestError(f"형제 후보 run {run_id}을 읽을 수 없다.") from exc
        sibling_tags = getattr(sibling, "tags", None)
        sibling_params = getattr(sibling, "params", None)
        _require(getattr(sibling, "status", None) == "FINISHED", f"형제 후보 run {run_id}이 FINISHED가 아니다.")
        _require(isinstance(sibling_tags, dict), f"형제 후보 run {run_id} 태그가 객체가 아니다.")
        _require(isinstance(sibling_params, dict), f"형제 후보 run {run_id} 매개변수가 객체가 아니다.")
        _require(
            sibling_tags.get("run.kind") == RUN_KIND
            and sibling_tags.get("training_state.ready") == "true"
            and sibling_tags.get("mlflow.parentRunId") == parent_run_id
            and sibling_tags.get("training_state.trajectory_run_id") == parent_run_id
            and sibling_tags.get("training_state.trajectory_identity_sha256")
            == child_document["trajectory_identity_sha256"]
            and sibling_tags.get("training_state.candidate_set_sha256")
            == child_document["candidate_set_sha256"],
            f"형제 후보 run {run_id}의 게시 태그가 후보 집합과 다르다.",
        )
        try:
            sibling_document = validate_candidate_manifest(
                manifest_bytes=artifact_bytes_of(run_id, MANIFEST_NAME),
                tags=sibling_tags,
                params=sibling_params,
                artifact_bytes_of=lambda name, sibling_run_id=run_id: artifact_bytes_of(
                    sibling_run_id, name
                ),
            )
        except TrainingStateManifestError:
            raise
        except Exception as exc:
            raise TrainingStateManifestError(
                f"형제 후보 run {run_id}의 게시 산출물을 읽을 수 없다."
            ) from exc
        _require(
            sibling_document["candidate"] == identity,
            f"형제 후보 run {run_id}의 self-contained 정체성이 부모 목록과 다르다.",
        )
        if run_id == child_run_id:
            _require(identity == current_candidate, "현재 child 정체성이 부모 후보 목록과 다르다.")
            current_found = True
    _require(current_found, "현재 child run이 부모 candidate-runs 목록에 없다.")
