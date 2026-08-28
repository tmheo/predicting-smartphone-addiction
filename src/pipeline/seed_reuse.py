"""검증된 단일 시드 실행을 다중 시드 확정 실행에서 재사용한다.

재사용은 계산 생략일 뿐 판정 증거 생략이 아니다.
소스 실행의 설정, 입력, 의존성, 예측, 점수와 시드별 산출물을 현재 실행 조건과
다시 대조한 뒤 ``CVResult``로 복원한다.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pandas as pd
import yaml

from . import cv, data
from .config import ExperimentConfig
from .recovery import model_dependency_snapshot
from .runs import MlflowRunStore, RunStore


class SeedReuseError(ValueError):
    """단일 시드 실행이 현재 확정 실행과 같은 조건이라는 증거가 부족하다."""


@dataclass(frozen=True)
class ReusedSeed:
    seed: int
    source_run_id: str
    result: cv.CVResult
    provenance: dict[str, object]


def load_reused_seed(
    cfg: ExperimentConfig,
    source_run_id: str,
    input_sha256: dict[str, str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    store: RunStore | None = None,
) -> ReusedSeed:
    """완료된 단일 시드 실행을 검증하고 현재 실행의 결과 하나로 복원한다."""

    store = store or MlflowRunStore()
    meta = store.facts_of(source_run_id)
    _require(meta.status == "FINISHED", f"소스 실행이 완료 상태가 아니다: {meta.status}")
    source_seeds = _parse_seeds(meta.params.get("seeds", ""))
    _require(len(source_seeds) == 1, f"소스 실행이 단일 시드가 아니다: {source_seeds}")
    seed = source_seeds[0]
    _require(seed in cfg.seeds, f"소스 시드 {seed}가 현재 확정 시드 {cfg.seeds}에 없다.")
    _require(meta.tags.get("git_dirty") == "False", "소스 실행의 코드 상태가 깨끗하지 않다.")
    _require(
        meta.params.get("experiment") == cfg.name,
        "소스 실행과 현재 설정의 실험 이름이 다르다.",
    )
    _require(
        store.config_of(source_run_id) == yaml.safe_load(cfg.source_path.read_text()),
        "소스 실행과 현재 실행의 설정 내용이 다르다.",
    )
    for name, digest in sorted(input_sha256.items()):
        _require(
            meta.tags.get(f"sha256.{name}") == digest,
            f"소스 실행의 {name} 입력 해시가 현재 실행과 다르다.",
        )

    recovery = _json_artifact(store, source_run_id, "fold_recovery.json")
    checkpoints = recovery.get("checkpoints")
    _require(
        recovery.get("boundary") == "completed_seed_fold" and isinstance(checkpoints, list),
        "소스 fold 복구 근거 형식이 올바르지 않다.",
    )
    folds = sorted(int(value) for value in train["fold"].unique())
    _require(
        sorted((int(item["seed"]), int(item["fold"])) for item in checkpoints)
        == [(seed, fold) for fold in folds],
        "소스 fold 복구 근거가 단일 시드의 모든 분할을 정확히 포함하지 않는다.",
    )
    dependencies = model_dependency_snapshot()
    config_sha256 = data.file_sha256(cfg.source_path)
    for item in checkpoints:
        identity = item.get("execution_identity", {})
        _require(
            identity.get("config_sha256") == config_sha256,
            "소스 설정 해시가 다르다.",
        )
        _require(
            identity.get("input_sha256") == dict(sorted(input_sha256.items())),
            "소스 입력 계보가 다르다.",
        )
        _require(
            identity.get("model_dependencies") == dependencies,
            "소스 실행 의존성이 다르다.",
        )
        _require(identity.get("model_kind") == cfg.model.kind, "소스 모델 계열이 다르다.")

    oof = _parquet_artifact(store, source_run_id, f"oof_seed_{seed}.parquet")
    recorded_oof = _parquet_artifact(store, source_run_id, "oof.parquet")
    _require(oof.equals(recorded_oof), "단일 시드 OOF와 대표 OOF의 저장 내용이 다르다.")
    expected_oof = train[[data.ID, "fold"]].reset_index(drop=True)
    _require(
        oof[[data.ID, "fold"]].reset_index(drop=True).equals(expected_oof),
        "소스 OOF의 행이나 분할 순서가 현재 입력과 다르다.",
    )
    fold_aucs = cv.score_predictions(
        train[data.TARGET], train["fold"], oof["pred"].to_numpy()
    )
    _require_close(
        fold_aucs["auc_oof"],
        meta.metrics.get(f"auc_oof_seed_{seed}"),
        "소스 시드 OOF 재채점값이 기록 지표와 다르다.",
    )
    for fold in folds:
        _require_close(
            fold_aucs[f"auc_fold_{fold}"],
            meta.metrics.get(f"auc_fold_{fold}"),
            f"소스 분할 {fold} OOF 재채점값이 기록 지표와 다르다.",
        )

    test_pred = _parquet_artifact(store, source_run_id, "test_pred.parquet")
    _require(
        test_pred[data.ID]
        .reset_index(drop=True)
        .equals(test[data.ID].reset_index(drop=True)),
        "소스 시험 예측의 행 순서가 현재 입력과 다르다.",
    )
    importance = _parquet_artifact(store, source_run_id, "feature_importance.parquet")
    _require(
        set(importance["seed"].astype(int)) == {seed},
        "소스 중요도에 다른 시드가 섞였다.",
    )
    feature_names = sorted(filter(None, meta.params.get("features", "").split(",")))
    _require(feature_names, "소스 실행에 확정 피처 목록이 없다.")
    _require(set(importance["feature"]) == set(feature_names), "소스 중요도와 피처 목록이 다르다.")

    fold_feature_reuse = _entries_artifact(
        store, source_run_id, "fold_feature_reuse.json"
    )
    model_diagnostics = _json_artifact(
        store, source_run_id, "model_training_diagnostics.json"
    )
    _require(isinstance(model_diagnostics, list), "소스 학습 진단이 목록이 아니다.")
    training_rows = _entries_artifact(store, source_run_id, "training_row_evidence.json")
    for name, entries in (
        ("fold-fit 재사용 근거", fold_feature_reuse),
        ("학습 진단", model_diagnostics),
        ("학습 행 구성 근거", training_rows),
    ):
        _require(
            entries and {int(item["seed"]) for item in entries} == {seed},
            f"소스 {name}의 시드가 다르다.",
        )

    artifact_names = [
        f"oof_seed_{seed}.parquet",
        "oof.parquet",
        "test_pred.parquet",
        "feature_importance.parquet",
        "fold_recovery.json",
        "fold_feature_reuse.json",
        "model_training_diagnostics.json",
        "training_row_evidence.json",
    ]
    provenance = {
        "schema_version": 1,
        "seed": seed,
        "source_run_id": source_run_id,
        "source_stage": meta.params.get("stage"),
        "source_git_commit": meta.tags.get("git_commit"),
        "config_sha256": config_sha256,
        "input_sha256": dict(sorted(input_sha256.items())),
        "model_dependencies": dependencies,
        "artifact_sha256": {
            name: store.artifact_sha256_of(source_run_id, name) for name in artifact_names
        },
        "auc_oof": fold_aucs["auc_oof"],
        "fold_auc": [fold_aucs[f"auc_fold_{fold}"] for fold in folds],
        "validation": {
            "finished_single_seed": True,
            "clean_source_tree": True,
            "config_equal": True,
            "inputs_equal": True,
            "model_dependencies_equal": True,
            "oof_rows_equal": True,
            "test_rows_equal": True,
            "stored_metrics_recomputed": True,
            "per_seed_evidence_complete": True,
        },
    }
    return ReusedSeed(
        seed=seed,
        source_run_id=source_run_id,
        result=cv.CVResult(
            oof=oof,
            test_pred=test_pred,
            fold_aucs=fold_aucs,
            feature_names=feature_names,
            importance=importance,
            recovery_evidence=checkpoints,
            fold_feature_reuse_evidence=fold_feature_reuse,
            model_training_diagnostics=model_diagnostics,
            training_row_evidence=training_rows,
        ),
        provenance=provenance,
    )


def _parse_seeds(raw: str) -> list[int]:
    try:
        return [int(value) for value in raw.split(",") if value]
    except ValueError as exc:
        raise SeedReuseError(f"소스 시드 목록을 읽을 수 없다: {raw!r}") from exc


def _json_artifact(store: RunStore, run_id: str, name: str):
    try:
        return json.loads(store.artifact_bytes_of(run_id, name))
    except json.JSONDecodeError as exc:
        raise SeedReuseError(f"소스 산출물 {name}을 JSON으로 읽을 수 없다.") from exc


def _entries_artifact(store: RunStore, run_id: str, name: str) -> list[dict[str, object]]:
    payload = _json_artifact(store, run_id, name)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    _require(isinstance(entries, list), f"소스 산출물 {name}에 entries 목록이 없다.")
    return entries


def _parquet_artifact(store: RunStore, run_id: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.artifact_bytes_of(run_id, name)))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SeedReuseError(message)


def _require_close(actual: float, expected: float | None, message: str) -> None:
    _require(expected is not None and abs(actual - expected) <= 1e-12, message)
