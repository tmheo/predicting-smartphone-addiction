"""한 학습 궤적의 사전 고정 시점 집합을 위한 fold 복구 경계.

기존 :mod:`pipeline.recovery`의 단일 시점 계약과 저장 공간은 사용하지 않는다.
모델 내부 상태도 저장하지 않는다.
완료된 ``(seed, fold)``에서 선언된 모든 학습 시점의 검증 예측, 테스트 예측,
중요도와 학습 관측을 한 디렉터리로 원자적으로 확정하고, 같은 실행 정체성과
같은 후보 집합을 다시 제시한 실행에서만 읽는다.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import ID, file_sha256
from .recovery import (
    RecoveryError as FoldRecoveryError,
    _content_sha256,
    model_dependency_snapshot,
)

SCHEMA_NAMESPACE = "training_state_recovery"
SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
BOUNDARY = "completed_seed_fold_snapshot_set"
EVIDENCE_NAME = "training_state_recovery.json"
MANIFEST_NAME = "manifest.json"
SNAPSHOTS_DIRECTORY = "snapshots"
VALIDATION_NAME = "validation_predictions.parquet"
TEST_NAME = "test_predictions.parquet"
IMPORTANCE_NAME = "feature_importance.parquet"
DIAGNOSTICS_NAME = "model_training_diagnostics.json"
TEMP_SUFFIX = ".tmp"


class TrainingStateRecoveryError(Exception):
    """학습 시점 집합을 안전하게 저장하거나 재사용할 수 없다."""


@dataclass(frozen=True)
class TrainingStateCandidate:
    """결과 확인 전에 고정한 후보 학습 시점 하나.

    후보 집합 문서는 :mod:`pipeline.training_state_contract`와 같은 필드로
    정규화한다.
    ``config_name``은 실행과 반입을 거쳐도 유지되는 후보 식별자다.
    ``completed_epochs``는 1부터 세는 종료 시점이며 모든 후보는 같은
    ``schedule_horizon_epochs``를 선언해야 한다.
    """

    config_name: str
    config_path: str
    config_sha256: str
    completed_epochs: int
    schedule_horizon_epochs: int

    @property
    def candidate_id(self) -> str:
        return self.config_name

    def identity_document(self) -> dict[str, object]:
        return {
            "config_name": self.config_name,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "completed_epochs": self.completed_epochs,
            "schedule_horizon_epochs": self.schedule_horizon_epochs,
        }


@dataclass(frozen=True)
class TrainingStateSnapshot:
    """후보 시점 하나에서 얻은 fold 출력."""

    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    importance: pd.DataFrame
    model_training_diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class TrainingStateCheckpoint:
    """완전히 확정된 한 ``(seed, fold)``의 후보 시점 집합."""

    snapshots: dict[str, TrainingStateSnapshot]
    aucs: dict[str, float]
    manifest: dict[str, object]

    def evidence(self, reused: bool) -> dict[str, object]:
        """MLflow에 남길 후보 집합 단위 복구 근거."""
        return {
            "schema_namespace": SCHEMA_NAMESPACE,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "boundary": BOUNDARY,
            "seed": self.manifest["seed"],
            "fold": self.manifest["fold"],
            "reused": reused,
            "execution_identity": self.manifest["execution_identity"],
            "execution_identity_sha256": self.manifest[
                "execution_identity_sha256"
            ],
            "candidate_set": self.manifest["candidate_set"],
            "candidate_set_sha256": self.manifest["candidate_set_sha256"],
            "manifest_content_sha256": self.manifest["manifest_content_sha256"],
            "snapshots": self.manifest["snapshots"],
        }


class TrainingStateRecovery:
    """내용으로 고정한 한 다중 시점 실행의 fold 완료 저장소."""

    def __init__(
        self,
        root: Path,
        execution_identity: dict[str, object],
        candidates: Sequence[TrainingStateCandidate],
    ) -> None:
        self.root = Path(root)
        self.run_identity = self._normalized_json_object(
            execution_identity, "학습 시점 복구 실행 정체성"
        )
        self.candidates = self._validate_candidates(candidates)
        model_kind = self.run_identity.get("model_kind")
        if not isinstance(model_kind, str) or not model_kind:
            raise TrainingStateRecoveryError(
                "학습 시점 복구 실행 정체성에 모델 종류가 없다."
            )
        trajectory_end_epochs = self.run_identity.get("trajectory_end_epochs")
        if (
            isinstance(trajectory_end_epochs, bool)
            or not isinstance(trajectory_end_epochs, int)
            or trajectory_end_epochs <= 0
        ):
            raise TrainingStateRecoveryError(
                "학습 시점 복구 실행 정체성의 궤적 종료 epoch가 잘못됐다."
            )
        schedule_horizon_epochs = self.candidates[0].schedule_horizon_epochs
        if (
            self.candidates[-1].completed_epochs > trajectory_end_epochs
            or trajectory_end_epochs > schedule_horizon_epochs
        ):
            raise TrainingStateRecoveryError(
                "학습 시점 복구 궤적 종료 epoch는 마지막 후보 이상이고 "
                "학습률 일정 지평 이하여야 한다."
            )
        self.model_kind = model_kind
        self.trajectory_end_epochs = trajectory_end_epochs
        self._candidate_set = [candidate.identity_document() for candidate in self.candidates]
        self.candidate_set_sha256 = _content_sha256(self._candidate_set)
        self.execution_identity = {
            "run": self.run_identity,
            "candidate_set": self._candidate_set,
            "candidate_set_sha256": self.candidate_set_sha256,
        }

    @classmethod
    def for_run(
        cls,
        root: Path,
        candidates: Sequence[TrainingStateCandidate],
        input_sha256: dict[str, str],
        *,
        git_commit: str,
        trajectory_identity_sha256: str,
        stage: str,
        seeds: Sequence[int],
        model_kind: str,
        trajectory_end_epochs: int,
        model_dependencies: dict[str, object] | None = None,
        declared_candidate_set_sha256: str | None = None,
    ) -> TrainingStateRecovery:
        """정식 다중 시점 CV가 쓰는 실행 정체성을 만들고 저장소를 연다."""
        if "folds" not in input_sha256:
            raise TrainingStateRecoveryError(
                "학습 시점 복구 실행 정체성에 folds 입력 해시가 없다."
            )
        for name, digest in input_sha256.items():
            cls._require_sha256(digest, f"입력 파일 {name}")
        cls._require_sha256(trajectory_identity_sha256, "학습 궤적 정체성")
        normalized_seeds = cls._validate_seeds(seeds)
        if not isinstance(git_commit, str) or not git_commit:
            raise TrainingStateRecoveryError("학습 시점 복구 Git 커밋이 비어 있다.")
        if not isinstance(stage, str) or not stage:
            raise TrainingStateRecoveryError("학습 시점 복구 실행 단계가 비어 있다.")
        if not isinstance(model_kind, str) or not model_kind:
            raise TrainingStateRecoveryError("학습 시점 복구 모델 종류가 비어 있다.")
        if (
            isinstance(trajectory_end_epochs, bool)
            or not isinstance(trajectory_end_epochs, int)
            or trajectory_end_epochs <= 0
        ):
            raise TrainingStateRecoveryError(
                "학습 시점 복구 궤적 종료 epoch는 양의 정수여야 한다."
            )
        try:
            dependencies = (
                model_dependencies
                if model_dependencies is not None
                else model_dependency_snapshot()
            )
        except FoldRecoveryError as exc:
            raise TrainingStateRecoveryError(
                "학습 시점 복구 모델 의존성을 고정할 수 없다."
            ) from exc
        identity = {
            "git_commit": git_commit,
            "trajectory_identity_sha256": trajectory_identity_sha256,
            "input_sha256": dict(sorted(input_sha256.items())),
            "folds_sha256": input_sha256["folds"],
            "stage": stage,
            "seeds": normalized_seeds,
            "model_kind": model_kind,
            "trajectory_end_epochs": trajectory_end_epochs,
            "model_dependencies": dependencies,
        }
        recovery = cls(root, identity, candidates)
        if declared_candidate_set_sha256 is not None:
            cls._require_sha256(declared_candidate_set_sha256, "학습 시점 후보 집합")
            if recovery.candidate_set_sha256 != declared_candidate_set_sha256:
                raise TrainingStateRecoveryError(
                    "학습 시점 복구 후보 집합 해시가 실행 계약과 다르다."
                )
        return recovery

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    def load(
        self,
        seed: int,
        fold: int,
        *,
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_ids: pd.Series,
        feature_names: list[str],
    ) -> TrainingStateCheckpoint | None:
        """선언 후보가 전부 완전한 fold만 읽고, 없는 fold는 ``None``을 돌려준다."""
        self._validate_coordinate(seed, fold)
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
        snapshots: Mapping[str, TrainingStateSnapshot],
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_ids: pd.Series,
        feature_names: list[str],
    ) -> TrainingStateCheckpoint:
        """선언 후보 전체를 검증하고 하나의 원자적 fold 완료로 확정한다."""
        self._validate_coordinate(seed, fold)
        ordered_snapshots = self._ordered_snapshots(snapshots)
        aucs: dict[str, float] = {}
        for candidate, snapshot in zip(
            self.candidates, ordered_snapshots.values(), strict=True
        ):
            auc = float(
                roc_auc_score(
                    validation_labels,
                    snapshot.validation_predictions["pred"],
                )
            )
            self._validate_payload(
                candidate,
                snapshot,
                auc,
                seed,
                fold,
                validation_ids,
                validation_labels,
                test_ids,
                feature_names,
            )
            aucs[candidate.candidate_id] = auc

        self._validate_seed_layout(seed)
        final_dir = self._fold_dir(seed, fold)
        if final_dir.exists():
            raise TrainingStateRecoveryError(
                f"중복 학습 시점 fold 완료 저장을 거부한다: seed={seed} fold={fold}"
            )
        seed_dir = self._seed_dir(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        staging = seed_dir / f".fold_{fold}.{uuid.uuid4().hex}{TEMP_SUFFIX}"
        staging.mkdir()
        try:
            snapshot_records: dict[str, object] = {}
            snapshots_dir = staging / SNAPSHOTS_DIRECTORY
            snapshots_dir.mkdir()
            for index, (candidate, snapshot) in enumerate(
                zip(self.candidates, ordered_snapshots.values(), strict=True)
            ):
                relative_dir = self._snapshot_relative_dir(index)
                snapshot_dir = staging / relative_dir
                snapshot_dir.mkdir()
                validation_path = snapshot_dir / VALIDATION_NAME
                test_path = snapshot_dir / TEST_NAME
                importance_path = snapshot_dir / IMPORTANCE_NAME
                diagnostics_path = snapshot_dir / DIAGNOSTICS_NAME
                snapshot.validation_predictions.to_parquet(validation_path, index=False)
                snapshot.test_predictions.to_parquet(test_path, index=False)
                snapshot.importance.to_parquet(importance_path, index=False)
                diagnostics_path.write_text(
                    json.dumps(
                        snapshot.model_training_diagnostics,
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                snapshot_records[candidate.candidate_id] = {
                    "directory": relative_dir.as_posix(),
                    "artifacts": {
                        VALIDATION_NAME: self._frame_record(
                            validation_path, snapshot.validation_predictions
                        ),
                        TEST_NAME: self._frame_record(
                            test_path, snapshot.test_predictions
                        ),
                        IMPORTANCE_NAME: self._frame_record(
                            importance_path, snapshot.importance
                        ),
                        DIAGNOSTICS_NAME: {"sha256": file_sha256(diagnostics_path)},
                    },
                    "metrics": {"auc": aucs[candidate.candidate_id]},
                }
                for path in (
                    validation_path,
                    test_path,
                    importance_path,
                    diagnostics_path,
                ):
                    self._fsync_file(path)
                self._fsync_directory(snapshot_dir)
            self._fsync_directory(snapshots_dir)

            identity = self._identity_for(seed, fold)
            manifest: dict[str, object] = {
                "schema_namespace": SCHEMA_NAMESPACE,
                "schema_version": SCHEMA_VERSION,
                "boundary": BOUNDARY,
                "seed": seed,
                "fold": fold,
                "execution_identity": identity,
                "execution_identity_sha256": _content_sha256(identity),
                "candidate_set": self._candidate_set,
                "candidate_set_sha256": self.candidate_set_sha256,
                "snapshots": snapshot_records,
            }
            manifest["manifest_content_sha256"] = _content_sha256(manifest)
            manifest_path = staging / MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            self._fsync_file(manifest_path)
            self._fsync_directory(staging)
            os.replace(staging, final_dir)
            self._fsync_directory(seed_dir)
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

    def _seed_dir(self, seed: int) -> Path:
        return self.root / f"seed_{seed}"

    def _fold_dir(self, seed: int, fold: int) -> Path:
        return self._seed_dir(seed) / f"fold_{fold}"

    @staticmethod
    def _snapshot_relative_dir(index: int) -> Path:
        return Path(SNAPSHOTS_DIRECTORY) / f"snapshot_{index:04d}"

    def _identity_for(self, seed: int, fold: int) -> dict[str, object]:
        return {
            **self.execution_identity,
            "seed": seed,
            "fold": fold,
        }

    def _validate_seed_layout(self, seed: int) -> None:
        seed_dir = self._seed_dir(seed)
        if not seed_dir.exists():
            return
        if not seed_dir.is_dir():
            raise TrainingStateRecoveryError(
                f"학습 시점 복구 seed 경로가 디렉터리가 아니다: {seed_dir}"
            )
        temporary = sorted(
            path.name for path in seed_dir.iterdir() if path.name.endswith(TEMP_SUFFIX)
        )
        if temporary:
            raise TrainingStateRecoveryError(
                "불완전한 학습 시점 fold 임시 상태를 재사용할 수 없다: "
                + ", ".join(temporary)
            )
        seen: set[tuple[int, int]] = set()
        fold_directories = sorted(seed_dir.glob("fold_*"))
        unexpected = sorted(
            path.name
            for path in seed_dir.iterdir()
            if path not in fold_directories and not path.name.endswith(TEMP_SUFFIX)
        )
        if unexpected:
            raise TrainingStateRecoveryError(
                "학습 시점 복구 seed 경로에 알 수 없는 항목이 있다: "
                + ", ".join(unexpected)
            )
        for fold_dir in fold_directories:
            manifest_path = fold_dir / MANIFEST_NAME
            if not fold_dir.is_dir() or not manifest_path.is_file():
                raise TrainingStateRecoveryError(
                    f"manifest가 없는 불완전한 학습 시점 fold 상태가 있다: {fold_dir}"
                )
            try:
                manifest = json.loads(manifest_path.read_text())
                key = (int(manifest["seed"]), int(manifest["fold"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TrainingStateRecoveryError(
                    f"학습 시점 복구 manifest를 읽을 수 없다: {manifest_path}"
                ) from exc
            if key in seen:
                raise TrainingStateRecoveryError(
                    f"중복 학습 시점 fold 복구 산출물이 있다: "
                    f"seed={key[0]} fold={key[1]}"
                )
            seen.add(key)
            expected = self._fold_dir(*key) / MANIFEST_NAME
            if manifest_path != expected:
                raise TrainingStateRecoveryError(
                    "학습 시점 fold manifest 경로와 seed/fold가 다르다: "
                    f"{manifest_path} != {expected}"
                )

    def _ordered_snapshots(
        self, snapshots: Mapping[str, TrainingStateSnapshot]
    ) -> dict[str, TrainingStateSnapshot]:
        if not isinstance(snapshots, Mapping):
            raise TrainingStateRecoveryError("학습 시점 출력 집합은 식별자별 매핑이어야 한다.")
        expected = set(self.candidate_ids)
        actual = set(snapshots)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise TrainingStateRecoveryError(
                "선언된 학습 시점 출력 집합이 완전하지 않다: "
                f"missing={missing}, extra={extra}"
            )
        ordered: dict[str, TrainingStateSnapshot] = {}
        for candidate_id in self.candidate_ids:
            snapshot = snapshots[candidate_id]
            if not isinstance(snapshot, TrainingStateSnapshot):
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} 출력 형식이 TrainingStateSnapshot이 아니다."
                )
            ordered[candidate_id] = snapshot
        return ordered

    def _read_checkpoint(self, fold_dir: Path) -> TrainingStateCheckpoint:
        manifest_path = fold_dir / MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text())
            snapshot_records = manifest.get("snapshots")
            if not isinstance(snapshot_records, dict):
                raise TypeError("snapshots가 객체가 아니다.")
            snapshots: dict[str, TrainingStateSnapshot] = {}
            aucs: dict[str, float] = {}
            for index, candidate in enumerate(self.candidates):
                record = snapshot_records.get(candidate.candidate_id)
                if not isinstance(record, dict):
                    raise TypeError(
                        f"후보 {candidate.candidate_id} snapshot 기록이 객체가 아니다."
                    )
                directory = fold_dir / self._snapshot_relative_dir(index)
                diagnostics = json.loads((directory / DIAGNOSTICS_NAME).read_text())
                snapshots[candidate.candidate_id] = TrainingStateSnapshot(
                    validation_predictions=pd.read_parquet(directory / VALIDATION_NAME),
                    test_predictions=pd.read_parquet(directory / TEST_NAME),
                    importance=pd.read_parquet(directory / IMPORTANCE_NAME),
                    model_training_diagnostics=diagnostics,
                )
                metrics = record.get("metrics")
                if not isinstance(metrics, dict):
                    raise TypeError(
                        f"후보 {candidate.candidate_id} metrics가 객체가 아니다."
                    )
                aucs[candidate.candidate_id] = float(metrics.get("auc", float("nan")))
            return TrainingStateCheckpoint(
                snapshots=snapshots,
                aucs=aucs,
                manifest=manifest,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainingStateRecoveryError(
                f"완료된 학습 시점 fold 복구 상태를 읽을 수 없다: {fold_dir}"
            ) from exc

    def _validate_checkpoint(
        self,
        checkpoint: TrainingStateCheckpoint,
        seed: int,
        fold: int,
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_ids: pd.Series,
        feature_names: list[str],
    ) -> None:
        manifest = checkpoint.manifest
        if manifest.get("schema_namespace") != SCHEMA_NAMESPACE:
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 namespace가 현재 계약과 다르다."
            )
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise TrainingStateRecoveryError(
                "지원하지 않는 학습 시점 fold 복구 스키마다: "
                f"{manifest.get('schema_version')}"
            )
        if manifest.get("boundary") != BOUNDARY:
            raise TrainingStateRecoveryError(
                f"학습 시점 fold 복구 경계가 {BOUNDARY}가 아니다."
            )
        claimed_manifest_hash = manifest.get("manifest_content_sha256")
        unhashed = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_content_sha256"
        }
        if claimed_manifest_hash != _content_sha256(unhashed):
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 manifest 내용 해시가 일치하지 않는다."
            )
        expected_identity = self._identity_for(seed, fold)
        if manifest.get("execution_identity") != expected_identity:
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 실행 정체성이 현재 실행 명세와 다르다."
            )
        expected_identity_hash = _content_sha256(expected_identity)
        if manifest.get("execution_identity_sha256") != expected_identity_hash:
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 실행 정체성 내용 해시가 일치하지 않는다."
            )
        if manifest.get("seed") != seed or manifest.get("fold") != fold:
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 manifest의 seed 또는 fold 번호가 요청과 다르다."
            )
        if manifest.get("candidate_set") != self._candidate_set:
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 후보 집합이 현재 사전 선언과 다르다."
            )
        if manifest.get("candidate_set_sha256") != self.candidate_set_sha256:
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 후보 집합 내용 해시가 일치하지 않는다."
            )
        if set(checkpoint.snapshots) != set(self.candidate_ids) or set(
            checkpoint.aucs
        ) != set(self.candidate_ids):
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 후보 출력 집합이 완전하지 않다."
            )

        fold_dir = self._fold_dir(seed, fold)
        self._validate_fold_layout(fold_dir)
        snapshot_records = manifest.get("snapshots")
        if not isinstance(snapshot_records, dict) or set(snapshot_records) != set(
            self.candidate_ids
        ):
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 snapshot manifest가 완전하지 않다."
            )
        for index, candidate in enumerate(self.candidates):
            candidate_id = candidate.candidate_id
            record = snapshot_records[candidate_id]
            if not isinstance(record, dict) or set(record) != {
                "directory",
                "artifacts",
                "metrics",
            }:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} snapshot 기록 형식이 잘못됐다."
                )
            relative_dir = self._snapshot_relative_dir(index)
            if record.get("directory") != relative_dir.as_posix():
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} snapshot 경로가 선언과 다르다."
                )
            snapshot_dir = fold_dir / relative_dir
            snapshot = checkpoint.snapshots[candidate_id]
            artifacts = record.get("artifacts")
            frames = {
                VALIDATION_NAME: snapshot.validation_predictions,
                TEST_NAME: snapshot.test_predictions,
                IMPORTANCE_NAME: snapshot.importance,
            }
            expected_artifacts = {*frames, DIAGNOSTICS_NAME}
            if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} artifact 스키마가 완전하지 않다."
                )
            for name, frame in frames.items():
                artifact = artifacts[name]
                if not isinstance(artifact, dict):
                    raise TrainingStateRecoveryError(
                        f"학습 시점 {candidate_id} artifact 기록 형식이 잘못됐다: {name}"
                    )
                path = snapshot_dir / name
                if artifact.get("sha256") != file_sha256(path):
                    raise TrainingStateRecoveryError(
                        f"학습 시점 {candidate_id} artifact 내용 해시가 일치하지 않는다: "
                        f"{name}"
                    )
                if artifact.get("rows") != len(frame) or artifact.get(
                    "columns"
                ) != list(frame.columns):
                    raise TrainingStateRecoveryError(
                        f"학습 시점 {candidate_id} artifact 행 또는 열 스키마가 다르다: "
                        f"{name}"
                    )
            diagnostics_record = artifacts[DIAGNOSTICS_NAME]
            if not isinstance(diagnostics_record, dict) or set(diagnostics_record) != {
                "sha256"
            }:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} 학습 관측 기록 형식이 잘못됐다."
                )
            if diagnostics_record["sha256"] != file_sha256(
                snapshot_dir / DIAGNOSTICS_NAME
            ):
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} 학습 관측 내용 해시가 일치하지 않는다."
                )
            metrics = record.get("metrics")
            if not isinstance(metrics, dict) or set(metrics) != {"auc"}:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} metric 스키마가 완전하지 않다."
                )
            try:
                claimed_auc = float(metrics["auc"])
            except (TypeError, ValueError) as exc:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} AUC 기록 형식이 잘못됐다."
                ) from exc
            if checkpoint.aucs[candidate_id] != claimed_auc:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} AUC 판독값이 manifest와 다르다."
                )
            self._validate_payload(
                candidate,
                snapshot,
                claimed_auc,
                seed,
                fold,
                validation_ids,
                validation_labels,
                test_ids,
                feature_names,
            )

    def _validate_fold_layout(self, fold_dir: Path) -> None:
        expected_root = {MANIFEST_NAME, SNAPSHOTS_DIRECTORY}
        actual_root = {path.name for path in fold_dir.iterdir()}
        if actual_root != expected_root:
            raise TrainingStateRecoveryError(
                "학습 시점 fold 복구 루트 항목이 완전하지 않다: "
                f"{sorted(actual_root)} != {sorted(expected_root)}"
            )
        snapshots_dir = fold_dir / SNAPSHOTS_DIRECTORY
        if not snapshots_dir.is_dir():
            raise TrainingStateRecoveryError("학습 시점 snapshot 디렉터리가 없다.")
        expected_directories = {
            self._snapshot_relative_dir(index).name
            for index, _candidate in enumerate(self.candidates)
        }
        actual_directories = {path.name for path in snapshots_dir.iterdir()}
        if actual_directories != expected_directories or any(
            not path.is_dir() for path in snapshots_dir.iterdir()
        ):
            raise TrainingStateRecoveryError(
                "학습 시점 snapshot 디렉터리 집합이 사전 선언과 다르다."
            )
        expected_files = {
            VALIDATION_NAME,
            TEST_NAME,
            IMPORTANCE_NAME,
            DIAGNOSTICS_NAME,
        }
        for directory in snapshots_dir.iterdir():
            actual_files = {path.name for path in directory.iterdir()}
            if actual_files != expected_files or any(
                not path.is_file() for path in directory.iterdir()
            ):
                raise TrainingStateRecoveryError(
                    f"학습 시점 snapshot 파일 집합이 완전하지 않다: {directory}"
                )

    def _validate_payload(
        self,
        candidate: TrainingStateCandidate,
        snapshot: TrainingStateSnapshot,
        auc: float,
        seed: int,
        fold: int,
        validation_ids: pd.Series,
        validation_labels: pd.Series,
        test_ids: pd.Series,
        feature_names: list[str],
    ) -> None:
        candidate_id = candidate.candidate_id
        validation = snapshot.validation_predictions
        test = snapshot.test_predictions
        importance = snapshot.importance
        self._require_columns(
            validation,
            [ID, "fold", "pred"],
            f"{candidate_id}/{VALIDATION_NAME}",
        )
        self._require_columns(test, [ID, "pred"], f"{candidate_id}/{TEST_NAME}")
        self._require_columns(
            importance,
            ["feature", "gain", "fold", "seed"],
            f"{candidate_id}/{IMPORTANCE_NAME}",
        )
        self._require_order(
            validation[ID], validation_ids, f"{candidate_id}/{VALIDATION_NAME} id"
        )
        self._require_order(test[ID], test_ids, f"{candidate_id}/{TEST_NAME} id")
        if len(validation_labels) != len(validation):
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} 검증 예측과 타깃 길이가 다르다."
            )
        if validation[ID].duplicated().any() or test[ID].duplicated().any():
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 복구 예측에 중복 id가 있다."
            )
        if not (validation["fold"] == fold).all():
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} 검증 예측의 fold 값이 manifest와 다르다."
            )
        if not (importance["fold"] == fold).all() or not (
            importance["seed"] == seed
        ).all():
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} importance의 seed 또는 fold 값이 다르다."
            )
        if importance["feature"].tolist() != feature_names:
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} importance의 특성 행 순서가 현재 학습 열 순서와 다르다."
            )
        if importance["feature"].duplicated().any():
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} importance에 중복 특성 행이 있다."
            )
        for name, values in {
            "validation pred": validation["pred"],
            "test pred": test["pred"],
            "importance gain": importance["gain"],
        }.items():
            array = values.to_numpy()
            if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate_id} fold 복구 {name} 값은 "
                    "유한한 부동소수점이어야 한다."
                )
        self._validate_diagnostics(snapshot.model_training_diagnostics, candidate_id)
        diagnostics = snapshot.model_training_diagnostics
        expected_diagnostics = {
            "seed": seed,
            "fold": fold,
            "model_kind": self.model_kind,
        }
        if not isinstance(diagnostics, dict) or any(
            diagnostics.get(key) != value
            for key, value in expected_diagnostics.items()
        ):
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 학습 관측의 "
                "seed, fold 또는 모델 종류가 실행 좌표와 다르다."
            )
        state_diagnostics = (
            diagnostics.get("training_state")
            if isinstance(diagnostics, dict)
            else None
        )
        expected_state_diagnostics = {
            "completed_epochs": candidate.completed_epochs,
            "schedule_horizon_epochs": candidate.schedule_horizon_epochs,
            "trajectory_end_epochs": self.trajectory_end_epochs,
            "selection_rule": "precommitted",
            "state_kind": "ema",
        }
        if not isinstance(state_diagnostics, dict) or any(
            state_diagnostics.get(key) != value
            for key, value in expected_state_diagnostics.items()
        ):
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 학습 관측의 시점 계약이 후보 선언과 다르다."
            )
        evidence = diagnostics.get("training_length_evidence")
        observations = evidence.get("observations") if isinstance(evidence, dict) else None
        if not isinstance(observations, list) or not observations:
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 학습 길이 근거가 없다."
            )
        if evidence.get("model_family") != self.model_kind or any(
            not isinstance(observation, dict)
            or observation.get("seed") != seed
            or observation.get("outer_fold") != fold
            for observation in observations
        ):
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 학습 길이 근거의 "
                "seed, fold 또는 모델 종류가 실행 좌표와 다르다."
            )
        observed_lengths = {
            observation.get("observed_training_length")
            for observation in observations
            if isinstance(observation, dict)
        }
        if observed_lengths != {candidate.completed_epochs}:
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 학습 길이 근거가 완료 시점과 다르다."
            )
        actual_auc = float(roc_auc_score(validation_labels, validation["pred"]))
        if not np.isfinite(auc) or auc != actual_auc:
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 복구 AUC가 검증 예측 재채점과 다르다."
            )

    @staticmethod
    def _frame_record(path: Path, frame: pd.DataFrame) -> dict[str, object]:
        return {
            "sha256": file_sha256(path),
            "rows": len(frame),
            "columns": list(frame.columns),
        }

    @staticmethod
    def _require_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
        if list(frame.columns) != expected:
            raise TrainingStateRecoveryError(
                f"{name} 열 순서가 다르다: {list(frame.columns)} != {expected}"
            )

    @staticmethod
    def _require_order(actual: pd.Series, expected: pd.Series, name: str) -> None:
        if len(actual) != len(expected) or actual.tolist() != expected.tolist():
            raise TrainingStateRecoveryError(f"{name} 행 순서가 현재 입력과 다르다.")

    @staticmethod
    def _validate_diagnostics(value: dict[str, object] | None, candidate_id: str) -> None:
        if value is not None and not isinstance(value, dict):
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 학습 관측은 객체 또는 null이어야 한다."
            )
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TrainingStateRecoveryError(
                f"학습 시점 {candidate_id} fold 학습 관측은 유한한 JSON 값이어야 한다."
            ) from exc

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[TrainingStateCandidate],
    ) -> tuple[TrainingStateCandidate, ...]:
        values = tuple(candidates)
        if not values:
            raise TrainingStateRecoveryError("사전 고정한 학습 시점 후보가 없다.")
        for candidate in values:
            if not isinstance(candidate, TrainingStateCandidate):
                raise TrainingStateRecoveryError(
                    "학습 시점 후보 형식이 TrainingStateCandidate가 아니다."
                )
            if not isinstance(candidate.config_name, str) or not candidate.config_name:
                raise TrainingStateRecoveryError("학습 시점 후보 식별자가 비어 있다.")
            if not isinstance(candidate.config_path, str) or not candidate.config_path:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate.config_name} 설정 경로가 비어 있다."
                )
            TrainingStateRecovery._require_sha256(
                candidate.config_sha256,
                f"학습 시점 {candidate.config_name} 설정",
            )
            for name, value in (
                ("완료 epoch", candidate.completed_epochs),
                ("학습률 일정 지평", candidate.schedule_horizon_epochs),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise TrainingStateRecoveryError(
                        f"학습 시점 {candidate.config_name}의 {name}은 양의 정수여야 한다."
                    )
            if candidate.completed_epochs > candidate.schedule_horizon_epochs:
                raise TrainingStateRecoveryError(
                    f"학습 시점 {candidate.config_name}의 완료 epoch가 "
                    "학습률 일정 지평을 넘는다."
                )
        ids = [candidate.candidate_id for candidate in values]
        if len(ids) != len(set(ids)):
            raise TrainingStateRecoveryError("학습 시점 후보 식별자가 중복됐다.")
        config_paths = [candidate.config_path for candidate in values]
        if len(config_paths) != len(set(config_paths)):
            raise TrainingStateRecoveryError("학습 시점 후보 설정 경로가 중복됐다.")
        config_hashes = [candidate.config_sha256 for candidate in values]
        if len(config_hashes) != len(set(config_hashes)):
            raise TrainingStateRecoveryError("학습 시점 후보 설정 내용 해시가 중복됐다.")
        horizons = {candidate.schedule_horizon_epochs for candidate in values}
        if len(horizons) != 1:
            raise TrainingStateRecoveryError(
                "학습 시점 후보들이 같은 학습률 일정 지평을 공유하지 않는다."
            )
        completed_epochs = [candidate.completed_epochs for candidate in values]
        if completed_epochs != sorted(set(completed_epochs)):
            raise TrainingStateRecoveryError(
                "학습 시점 후보는 중복 없는 완료 epoch 오름차순으로 선언해야 한다."
            )
        return values

    @staticmethod
    def _validate_seeds(seeds: Sequence[int]) -> list[int]:
        values = list(seeds)
        if not values or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in values
        ):
            raise TrainingStateRecoveryError(
                "학습 시점 복구 시드는 비어 있지 않은 음이 아닌 정수 목록이어야 한다."
            )
        if len(values) != len(set(values)):
            raise TrainingStateRecoveryError("학습 시점 복구 시드가 중복됐다.")
        return values

    @staticmethod
    def _validate_coordinate(seed: int, fold: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise TrainingStateRecoveryError("학습 시점 복구 seed는 음이 아닌 정수여야 한다.")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise TrainingStateRecoveryError("학습 시점 복구 fold는 음이 아닌 정수여야 한다.")

    @staticmethod
    def _require_sha256(value: object, label: str) -> None:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise TrainingStateRecoveryError(f"{label} SHA-256 형식이 잘못됐다.")

    @staticmethod
    def _normalized_json_object(
        value: dict[str, object], label: str
    ) -> dict[str, object]:
        if not isinstance(value, dict) or not value:
            raise TrainingStateRecoveryError(f"{label}이 비어 있거나 객체가 아니다.")
        try:
            normalized = json.loads(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise TrainingStateRecoveryError(f"{label}은 유한한 JSON 객체여야 한다.") from exc
        return normalized

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def training_state_recovery_evidence(
    checkpoints: list[dict[str, object]],
) -> dict[str, object]:
    """MLflow와 실행 기록 묶음에 남길 후보 집합 단위 최종 복구 근거."""
    return {
        "schema_namespace": SCHEMA_NAMESPACE,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "boundary": BOUNDARY,
        "checkpoints": sorted(
            checkpoints,
            key=lambda item: (item["seed"], item["fold"]),
        ),
    }
