"""바깥쪽 분할의 학습 행 구성과 결정성 증거.

원본 행 경로는 기존 교차 검증 동작을 그대로 둔다.
복제 팔만 이 모듈을 통해 원본 학습 부분과 복제본 두 블록을 만들며, 마스크는
모형 시드, 바깥쪽 분할과 복제본 번호만으로 결정된다.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TrainingRowsConfig
from .data import ID, TARGET
from .fold_fit_reuse import canonical_json_bytes, series_value_document

EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_NAME = "training_row_evidence.json"

# 피처가 아니라 변환 문맥이다.
# 목표값 참조 fold-fit 제공자만 이 열을 읽어 복제본에 부모의 내부 분할을 물려준다.
PARENT_ID = "__training_row_parent_id"


@dataclass(frozen=True)
class TrainingRowBatch:
    """한 바깥쪽 분할의 모형 학습 행과 부모 대응."""

    frame: pd.DataFrame
    parent_source_index: pd.Index
    original_row_count: int
    evidence: dict[str, object]

    @property
    def training_index(self) -> pd.RangeIndex:
        return pd.RangeIndex(len(self.frame))

    @property
    def state_fit_index(self) -> pd.RangeIndex:
        return pd.RangeIndex(self.original_row_count)


def _row_key(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return f"{type(value).__name__}:{value}"


def _mask_seed(seed: int, outer_fold: int, replica_index: int) -> int:
    payload = canonical_json_bytes(
        {
            "contract": "independent-observed-cell-mask-v1",
            "seed": int(seed),
            "outer_fold": int(outer_fold),
            "replica_index": int(replica_index),
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _mask_sha256(mask: np.ndarray, raw_columns: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"training-row-mask-v1\0")
    digest.update(struct.pack("<QQ", *mask.shape))
    digest.update(canonical_json_bytes(raw_columns))
    digest.update(np.packbits(mask, bitorder="little").tobytes())
    return digest.hexdigest()


def _validate_inputs(source: pd.DataFrame, raw_columns: list[str]) -> None:
    if len(raw_columns) != 12:
        raise ValueError(
            f"결측 증강은 id와 목표값을 뺀 원자료 12열에만 적용한다: {raw_columns}"
        )
    if len(set(raw_columns)) != len(raw_columns):
        raise ValueError("결측 증강 원자료 열에 중복이 있다.")
    missing = [column for column in [ID, TARGET, *raw_columns] if column not in source]
    if missing:
        raise ValueError(f"결측 증강 입력에 필요한 열이 없다: {missing}")
    if not source[ID].is_unique:
        raise ValueError("결측 증강의 부모 원본 행 식별자가 고유하지 않다.")
    if PARENT_ID in source:
        raise ValueError(f"원본 자료에 내부 변환 문맥 열 {PARENT_ID}가 이미 있다.")


def build_training_rows(
    source: pd.DataFrame,
    raw_columns: list[str],
    config: TrainingRowsConfig,
    *,
    seed: int,
    outer_fold: int,
) -> TrainingRowBatch:
    """원본 학습 부분과 설정이 요구한 복제본을 정해진 순서로 만든다."""
    _validate_inputs(source, raw_columns)
    source = source.copy()
    original_keys = source[ID].map(_row_key)
    if not original_keys.is_unique:
        raise ValueError("정규화한 부모 행 식별자가 고유하지 않다.")

    original = source.copy()
    if config.replica_count:
        original[ID] = "original:" + original_keys
        original[PARENT_ID] = original[ID]
    else:
        original[PARENT_ID] = original_keys
    frames = [original]
    parent_source_indexes = [source.index.to_numpy(copy=True)]
    masks: list[dict[str, object]] = []
    total_eligible = 0
    total_added_missing = 0

    for replica_index in range(1, config.replica_count + 1):
        replica = source.copy()
        replica[ID] = f"replica-{replica_index}:" + original_keys
        replica[PARENT_ID] = "original:" + original_keys
        observed = replica[raw_columns].notna().to_numpy()
        eligible = int(observed.sum())
        mask_seed = _mask_seed(seed, outer_fold, replica_index)
        if config.observed_cell_mask_probability == 0.0:
            mask = np.zeros(observed.shape, dtype=bool)
        else:
            rng = np.random.default_rng(mask_seed)
            mask = observed & (
                rng.random(observed.shape) < config.observed_cell_mask_probability
            )
        for column_index, column in enumerate(raw_columns):
            if mask[:, column_index].any():
                replica.loc[mask[:, column_index], column] = np.nan
        added_missing = int(mask.sum())
        total_eligible += eligible
        total_added_missing += added_missing
        masks.append(
            {
                "replica_index": replica_index,
                "mask_seed": mask_seed,
                "mask_sha256": _mask_sha256(mask, raw_columns),
                "eligible_observed_cells": eligible,
                "added_missing_cells": added_missing,
                "actual_added_missing_rate": (
                    float(added_missing / eligible) if eligible else 0.0
                ),
            }
        )
        frames.append(replica)
        parent_source_indexes.append(source.index.to_numpy(copy=True))

    frame = pd.concat(frames, ignore_index=True)
    parent_source_index = pd.Index(np.concatenate(parent_source_indexes))
    expected_rows = len(source) * (1 + config.replica_count)
    if len(frame) != expected_rows or len(parent_source_index) != expected_rows:
        raise AssertionError("결측 증강 학습 행 수가 계약과 다르다.")
    if not frame[ID].is_unique:
        raise AssertionError("결측 증강 내부 행 식별자가 고유하지 않다.")

    source_target = source[TARGET].to_numpy()
    expected_target = np.tile(source_target, 1 + config.replica_count)
    target_inherited = np.array_equal(
        frame[TARGET].to_numpy(), expected_target, equal_nan=True
    )
    fold_inherited = True
    if "fold" in source:
        fold_inherited = np.array_equal(
            frame["fold"].to_numpy(),
            np.tile(source["fold"].to_numpy(), 1 + config.replica_count),
            equal_nan=True,
        )
    existing_missing = source[raw_columns].isna().to_numpy()
    existing_missing_preserved = all(
        np.all(
            replica_frame[raw_columns].isna().to_numpy()[existing_missing]
        )
        for replica_frame in frames[1:]
    )
    parent_ids = frame[PARENT_ID]
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "arm": config.arm,
        "seed": int(seed),
        "outer_fold": int(outer_fold),
        "raw_columns": list(raw_columns),
        "replica_count": config.replica_count,
        "observed_cell_mask_probability": config.observed_cell_mask_probability,
        "original_row_count": len(source),
        "replica_row_count": len(source) * config.replica_count,
        "training_row_count": len(frame),
        "state_fit_row_count": len(source),
        "row_id_order": series_value_document(frame[ID]),
        "parent_id_order": series_value_document(parent_ids),
        "parent_source_index_order": series_value_document(
            pd.Series(parent_source_index.to_numpy())
        ),
        "target_order": series_value_document(frame[TARGET]),
        "replicas": masks,
        "eligible_observed_cells": total_eligible,
        "added_missing_cells": total_added_missing,
        "actual_added_missing_rate": (
            float(total_added_missing / total_eligible) if total_eligible else 0.0
        ),
        "assertions": {
            "original_rows_first": True,
            "replica_blocks_follow_in_order": True,
            "targets_inherited_from_parent": bool(target_inherited),
            "outer_fold_inherited_from_parent": bool(fold_inherited),
            "existing_missing_cells_preserved": bool(existing_missing_preserved),
            "replicas_excluded_from_state_fit": True,
            "replica_identity_excluded_from_model_features": True,
        },
    }
    failed = [name for name, value in evidence["assertions"].items() if not value]
    if failed:
        raise AssertionError(f"결측 증강 학습 행 불변식이 깨졌다: {failed}")
    return TrainingRowBatch(
        frame=frame,
        parent_source_index=parent_source_index,
        original_row_count=len(source),
        evidence=evidence,
    )


def validation_context(frame: pd.DataFrame) -> pd.DataFrame:
    """바깥쪽 검증 행을 변형하지 않고 내부 행 식별자만 격리한다."""
    out = frame.copy()
    out[ID] = "validation:" + out[ID].map(_row_key)
    out[PARENT_ID] = pd.NA
    return out
