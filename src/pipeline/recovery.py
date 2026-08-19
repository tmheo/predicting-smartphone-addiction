"""정식 CV 실행의 fold 복구 경계.

모델 내부 상태는 저장하지 않는다.
완료된 ``(seed, fold)``의 검증 예측, 테스트 예측, 중요도와 AUC만 원자적으로
확정하고, 같은 실행 정체성을 다시 제시한 실행에서만 읽는다.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from packaging.requirements import InvalidRequirement, Requirement
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig
from .data import ID, file_sha256

SCHEMA_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_NAME = "fold_recovery.json"
MANIFEST_NAME = "manifest.json"
VALIDATION_NAME = "validation_predictions.parquet"
TEST_NAME = "test_predictions.parquet"
IMPORTANCE_NAME = "feature_importance.parquet"
TEMP_SUFFIX = ".tmp"


class RecoveryError(Exception):
    """복구 산출물을 안전하게 저장하거나 재사용할 수 없다."""


@dataclass(frozen=True)
class FoldCheckpoint:
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    importance: pd.DataFrame
    auc: float
    model_training_diagnostics: dict[str, object] | None
    manifest: dict[str, object]

    def evidence(self, reused: bool) -> dict[str, object]:
        return {
            "seed": self.manifest["seed"],
            "fold": self.manifest["fold"],
            "reused": reused,
            "execution_identity": self.manifest["execution_identity"],
            "execution_identity_sha256": self.manifest["execution_identity_sha256"],
            "manifest_content_sha256": self.manifest["manifest_content_sha256"],
            "artifacts": self.manifest["artifacts"],
            "metrics": self.manifest["metrics"],
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _requirement_of(requirement: str) -> Requirement:
    try:
        return Requirement(requirement)
    except InvalidRequirement as exc:
        raise RecoveryError(f"pyproject 의존성을 해석할 수 없다: {requirement!r}") from exc


def model_dependency_snapshot(
    pyproject_path: Path = Path("pyproject.toml"), lock_path: Path = Path("uv.lock")
) -> dict[str, object]:
    """현재 프로젝트가 선언한 모델 실행 의존성의 실제 판본과 잠금 해시를 고정한다.

    환경 표식이 붙은 의존성(faiss-gpu-cu12의 sys_platform == 'linux' 등)은 표식이
    현재 환경에 맞을 때만 설치를 요구한다. 스냅샷은 실행 환경의 정체성이므로,
    이 환경에 설치될 수 없는 의존성은 판본 목록에 넣지 않는다.
    """
    if not pyproject_path.is_file() or not lock_path.is_file():
        raise RecoveryError("pyproject.toml과 uv.lock이 있어야 복구 실행 의존성을 고정할 수 있다.")
    project = tomllib.loads(pyproject_path.read_text())
    requirements = project.get("project", {}).get("dependencies", [])
    versions: dict[str, str] = {}
    for requirement in requirements:
        req = _requirement_of(requirement)
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            versions[req.name] = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RecoveryError(f"선언된 실행 의존성이 설치되지 않았다: {req.name}") from exc
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "uv_lock_sha256": file_sha256(lock_path),
        "project_packages": dict(sorted(versions.items())),
    }


class FoldRecovery:
    """내용으로 고정한 한 정식 실행의 fold 완료 저장소."""

    def __init__(self, root: Path, execution_identity: dict[str, object]) -> None:
        self.root = root
        self.execution_identity = execution_identity

    @classmethod
    def for_run(
        cls,
        root: Path,
        cfg: ExperimentConfig,
        input_sha256: dict[str, str],
        *,
        git_commit: str,
        model_dependencies: dict[str, object] | None = None,
    ) -> FoldRecovery:
        if "folds" not in input_sha256:
            raise RecoveryError("복구 실행 정체성에 folds 입력 해시가 없다.")
        identity = {
            "git_commit": git_commit,
            "config_sha256": file_sha256(cfg.source_path),
            "input_sha256": dict(sorted(input_sha256.items())),
            "folds_sha256": input_sha256["folds"],
            "stage": cfg.stage,
            "seeds": list(cfg.seeds),
            "model_kind": cfg.model.kind,
            "model_dependencies": (
                model_dependencies
                if model_dependencies is not None
                else model_dependency_snapshot()
            ),
        }
        return cls(root, identity)

    def _seed_dir(self, seed: int) -> Path:
        return self.root / f"seed_{seed}"

    def _fold_dir(self, seed: int, fold: int) -> Path:
        return self._seed_dir(seed) / f"fold_{fold}"

    def _identity_for(self, seed: int, fold: int) -> dict[str, object]:
        return {**self.execution_identity, "seed": seed, "fold": fold}

    def _validate_seed_layout(self, seed: int) -> None:
        seed_dir = self._seed_dir(seed)
        if not seed_dir.exists():
            return
        temporary = sorted(p.name for p in seed_dir.iterdir() if p.name.endswith(TEMP_SUFFIX))
        if temporary:
            raise RecoveryError(
                f"불완전한 fold 임시 상태를 재사용할 수 없다: {', '.join(temporary)}"
            )
        seen: set[tuple[int, int]] = set()
        for manifest_path in sorted(seed_dir.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
                key = (int(manifest["seed"]), int(manifest["fold"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RecoveryError(f"복구 manifest를 읽을 수 없다: {manifest_path}") from exc
            if key in seen:
                raise RecoveryError(f"중복 fold 복구 산출물이 있다: seed={key[0]} fold={key[1]}")
            seen.add(key)
            expected = self._fold_dir(*key) / MANIFEST_NAME
            if manifest_path != expected:
                raise RecoveryError(
                    f"fold manifest 경로와 seed/fold가 다르다: {manifest_path} != {expected}"
                )
        incomplete = sorted(
            str(p) for p in seed_dir.glob("fold_*") if p.is_dir() and not (p / MANIFEST_NAME).is_file()
        )
        if incomplete:
            raise RecoveryError(f"manifest가 없는 불완전한 fold 상태가 있다: {', '.join(incomplete)}")

    def load(
        self,
        seed: int,
        fold: int,
        *,
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_ids: pd.Series,
        feature_names: list[str],
    ) -> FoldCheckpoint | None:
        self._validate_seed_layout(seed)
        fold_dir = self._fold_dir(seed, fold)
        if not fold_dir.exists():
            return None
        checkpoint = self._read_checkpoint(fold_dir)
        self._validate_checkpoint(
            checkpoint,
            seed,
            fold,
            validation_ids,
            validation_labels,
            test_ids,
            feature_names,
        )
        return checkpoint

    def save(
        self,
        seed: int,
        fold: int,
        *,
        validation_predictions: pd.DataFrame,
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_predictions: pd.DataFrame,
        test_ids: pd.Series,
        importance: pd.DataFrame,
        feature_names: list[str],
        model_training_diagnostics: dict[str, object] | None = None,
    ) -> FoldCheckpoint:
        self._validate_seed_layout(seed)
        final_dir = self._fold_dir(seed, fold)
        if final_dir.exists():
            raise RecoveryError(f"중복 fold 완료 저장을 거부한다: seed={seed} fold={fold}")
        seed_dir = self._seed_dir(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        staging = seed_dir / f".fold_{fold}.{uuid.uuid4().hex}{TEMP_SUFFIX}"
        staging.mkdir()
        try:
            auc = float(roc_auc_score(validation_labels, validation_predictions["pred"]))
            self._validate_payload(
                validation_predictions,
                test_predictions,
                importance,
                auc,
                seed,
                fold,
                validation_ids,
                validation_labels,
                test_ids,
                feature_names,
            )
            self._validate_model_training_diagnostics(model_training_diagnostics)
            validation_path = staging / VALIDATION_NAME
            test_path = staging / TEST_NAME
            importance_path = staging / IMPORTANCE_NAME
            validation_predictions.to_parquet(validation_path, index=False)
            test_predictions.to_parquet(test_path, index=False)
            importance.to_parquet(importance_path, index=False)
            manifest: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "fold": fold,
                "execution_identity": self._identity_for(seed, fold),
                "execution_identity_sha256": _content_sha256(self._identity_for(seed, fold)),
                "artifacts": {
                    VALIDATION_NAME: self._artifact_record(validation_path, validation_predictions),
                    TEST_NAME: self._artifact_record(test_path, test_predictions),
                    IMPORTANCE_NAME: self._artifact_record(importance_path, importance),
                },
                "metrics": {"auc": auc},
                "model_training_diagnostics": model_training_diagnostics,
            }
            manifest["manifest_content_sha256"] = _content_sha256(manifest)
            manifest_path = staging / MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
                + "\n"
            )
            for path in (validation_path, test_path, importance_path, manifest_path):
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            os.replace(staging, final_dir)
            directory_fd = os.open(seed_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        checkpoint = self._read_checkpoint(final_dir)
        self._validate_checkpoint(
            checkpoint,
            seed,
            fold,
            validation_ids,
            validation_labels,
            test_ids,
            feature_names,
        )
        return checkpoint

    @staticmethod
    def _artifact_record(path: Path, frame: pd.DataFrame) -> dict[str, object]:
        return {
            "sha256": file_sha256(path),
            "rows": len(frame),
            "columns": list(frame.columns),
        }

    @staticmethod
    def _read_checkpoint(fold_dir: Path) -> FoldCheckpoint:
        manifest_path = fold_dir / MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text())
            metrics = manifest.get("metrics", {})
            if not isinstance(metrics, dict):
                raise TypeError("metrics가 객체가 아니다.")
            return FoldCheckpoint(
                validation_predictions=pd.read_parquet(fold_dir / VALIDATION_NAME),
                test_predictions=pd.read_parquet(fold_dir / TEST_NAME),
                importance=pd.read_parquet(fold_dir / IMPORTANCE_NAME),
                auc=float(metrics.get("auc", float("nan"))),
                model_training_diagnostics=manifest.get("model_training_diagnostics"),
                manifest=manifest,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RecoveryError(f"완료된 fold 복구 상태를 읽을 수 없다: {fold_dir}") from exc

    def _validate_checkpoint(
        self,
        checkpoint: FoldCheckpoint,
        seed: int,
        fold: int,
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_ids: pd.Series,
        feature_names: list[str],
    ) -> None:
        manifest = checkpoint.manifest
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RecoveryError(
                f"지원하지 않는 fold 복구 스키마다: {manifest.get('schema_version')}"
            )
        claimed_manifest_hash = manifest.get("manifest_content_sha256")
        unhashed = {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
        if claimed_manifest_hash != _content_sha256(unhashed):
            raise RecoveryError("fold 복구 manifest 내용 해시가 일치하지 않는다.")
        expected_identity = self._identity_for(seed, fold)
        if manifest.get("execution_identity") != expected_identity:
            raise RecoveryError("fold 복구 실행 정체성이 현재 원격 실행 명세와 다르다.")
        expected_identity_hash = _content_sha256(expected_identity)
        if manifest.get("execution_identity_sha256") != expected_identity_hash:
            raise RecoveryError("fold 복구 실행 정체성 내용 해시가 일치하지 않는다.")
        if manifest.get("seed") != seed or manifest.get("fold") != fold:
            raise RecoveryError("fold 복구 manifest의 seed 또는 fold 번호가 요청과 다르다.")
        self._validate_model_training_diagnostics(checkpoint.model_training_diagnostics)

        frames = {
            VALIDATION_NAME: checkpoint.validation_predictions,
            TEST_NAME: checkpoint.test_predictions,
            IMPORTANCE_NAME: checkpoint.importance,
        }
        artifact_records = manifest.get("artifacts")
        if not isinstance(artifact_records, dict) or set(artifact_records) != set(frames):
            raise RecoveryError("fold 복구 artifact 스키마가 완전하지 않다.")
        fold_dir = self._fold_dir(seed, fold)
        for name, frame in frames.items():
            record = artifact_records[name]
            if not isinstance(record, dict):
                raise RecoveryError(f"fold 복구 artifact 기록 형식이 잘못됐다: {name}")
            if record.get("sha256") != file_sha256(fold_dir / name):
                raise RecoveryError(f"fold 복구 artifact 내용 해시가 일치하지 않는다: {name}")
            if record.get("rows") != len(frame) or record.get("columns") != list(frame.columns):
                raise RecoveryError(f"fold 복구 artifact 행 또는 열 스키마가 다르다: {name}")

        self._validate_payload(
            checkpoint.validation_predictions,
            checkpoint.test_predictions,
            checkpoint.importance,
            checkpoint.auc,
            seed,
            fold,
            validation_ids,
            validation_labels,
            test_ids,
            feature_names,
        )

    def _validate_payload(
        self,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        importance: pd.DataFrame,
        auc: float,
        seed: int,
        fold: int,
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_ids: pd.Series,
        feature_names: list[str],
    ) -> None:
        self._require_columns(validation, [ID, "fold", "pred"], VALIDATION_NAME)
        self._require_columns(test, [ID, "pred"], TEST_NAME)
        self._require_columns(importance, ["feature", "gain", "fold", "seed"], IMPORTANCE_NAME)
        self._require_order(validation[ID], validation_ids, f"{VALIDATION_NAME} id")
        self._require_order(test[ID], test_ids, f"{TEST_NAME} id")
        if validation[ID].duplicated().any() or test[ID].duplicated().any():
            raise RecoveryError("fold 복구 예측에 중복 id가 있다.")
        if not (validation["fold"] == fold).all():
            raise RecoveryError("검증 예측의 fold 값이 manifest와 다르다.")
        if not (importance["fold"] == fold).all() or not (importance["seed"] == seed).all():
            raise RecoveryError("importance의 seed 또는 fold 값이 manifest와 다르다.")
        if importance["feature"].tolist() != feature_names:
            raise RecoveryError("importance의 특성 행 순서가 현재 학습 열 순서와 다르다.")
        if importance["feature"].duplicated().any():
            raise RecoveryError("importance에 중복 특성 행이 있다.")
        for name, values in {
            "validation pred": validation["pred"],
            "test pred": test["pred"],
            "importance gain": importance["gain"],
        }.items():
            array = values.to_numpy()
            if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise RecoveryError(f"fold 복구 {name} 값은 유한한 부동소수점이어야 한다.")
        actual_auc = float(roc_auc_score(validation_labels, validation["pred"]))
        if not np.isfinite(auc) or auc != actual_auc:
            raise RecoveryError("fold 복구 AUC가 검증 예측 재채점과 다르다.")

    @staticmethod
    def _require_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
        if list(frame.columns) != expected:
            raise RecoveryError(f"{name} 열 순서가 다르다: {list(frame.columns)} != {expected}")

    @staticmethod
    def _validate_model_training_diagnostics(value: dict[str, object] | None) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise RecoveryError("fold 학습 관측은 객체여야 한다.")
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RecoveryError("fold 학습 관측은 유한한 JSON 값이어야 한다.") from exc

    @staticmethod
    def _require_order(actual: pd.Series, expected: pd.Series, name: str) -> None:
        if len(actual) != len(expected) or actual.tolist() != expected.tolist():
            raise RecoveryError(f"{name} 행 순서가 현재 입력과 다르다.")


def recovery_evidence(checkpoints: list[dict[str, object]]) -> dict[str, object]:
    """MLflow와 실행 기록 묶음에 남길 비밀 없는 최종 복구 근거를 만든다."""
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "boundary": "completed_seed_fold",
        "checkpoints": sorted(checkpoints, key=lambda item: (item["seed"], item["fold"])),
    }
