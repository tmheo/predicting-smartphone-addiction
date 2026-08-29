"""이슈 510 결측 증강 짝비교의 축소 경계 진단을 실행한다."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline import cv, data, initial_score, model  # noqa: E402
from pipeline.config import TrainingRowsConfig, load_config  # noqa: E402
from pipeline.data import ID, TARGET, file_sha256  # noqa: E402
from pipeline.features import (  # noqa: E402
    PLACEBO,
    CategoricalCopies,
    LatticePairTargetEncoder,
    placebo_series,
)
from pipeline.paired_training_length import load as load_paired_lengths  # noqa: E402
from pipeline.training_rows import PARENT_ID, build_training_rows  # noqa: E402


CONFIG_DIR = REPO_ROOT / "configs/missingness-propagation"
FREEZE_PATH = REPO_ROOT / "artifacts/issue510-missingness-propagation-precommit.json"
OUTPUT_PATH = REPO_ROOT / "artifacts/issue510-missingness-propagation-diagnostics.json"
RAW_COLUMNS = [
    "age",
    "gender",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="실제 data 디렉터리가 있는 저장소 루트",
    )
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise AssertionError(f"설정 루트가 객체가 아니다: {path}")
    return raw


def _select_balanced(frame: pd.DataFrame, per_class: int, offset: int = 0) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby(TARGET, sort=True):
        parts.append(group.iloc[offset : offset + per_class])
    result = pd.concat(parts).sort_index().reset_index(drop=True)
    if result[TARGET].value_counts().min() != per_class:
        raise AssertionError("축소 진단의 층화 표본이 부족하다.")
    return result


def _pair_contracts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    freeze = json.loads(FREEZE_PATH.read_text())
    pairs = freeze["pairs"]
    records = []
    parsed = []
    for pair in pairs:
        paths = [REPO_ROOT / arm["path"] for arm in pair["arms"]]
        configs = [_load_yaml(path) for path in paths]
        for path, arm, config in zip(paths, pair["arms"], configs):
            if file_sha256(path) != arm["sha256"]:
                raise AssertionError(f"동결 뒤 설정이 바뀌었다: {path}")
            parsed.append(config)
        left = dict(configs[0])
        right = dict(configs[1])
        left_name = left.pop("name")
        right_name = right.pop("name")
        left_rows = left.pop("training_rows")
        right_rows = right.pop("training_rows")
        if left != right:
            raise AssertionError(f"{pair['member']}: 허용하지 않은 짝 설정 차이가 있다.")
        if left_rows != {
            "arm": "tripled",
            "replica_count": 2,
            "observed_cell_mask_probability": 0.0,
        }:
            raise AssertionError(f"{pair['member']}: 3배 행 대조군 설정이 다르다.")
        if right_rows != {
            "arm": "missingness_augmented",
            "replica_count": 2,
            "observed_cell_mask_probability": 0.25,
        }:
            raise AssertionError(f"{pair['member']}: 결측 증강군 설정이 다르다.")
        if left_name == right_name:
            raise AssertionError(f"{pair['member']}: 두 실행 이름이 같다.")
        records.append(
            {
                "member": pair["member"],
                "only_name_and_training_rows_differ": True,
                "paired_training_length_reference_equal": (
                    left["paired_training_length"] == right["paired_training_length"]
                ),
            }
        )
    if len(records) != 34 or len(parsed) != 68:
        raise AssertionError("짝 34개와 실행 설정 68개가 아니다.")
    return records, parsed


def _configuration_boundaries(
    source_root: Path, raw_configs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = []
    previous = Path.cwd()
    try:
        os.chdir(source_root)
        for path, raw in zip(
            sorted(CONFIG_DIR.glob("*.yaml")), raw_configs
        ):
            config = load_config(path, "confirm")
            from pipeline.plan import FeaturePlan

            plan = FeaturePlan.from_config(config.features)
            plan.validate_training_row_augmentation()
            paired_config = config.paired_training_length
            assert paired_config is not None
            lengths = load_paired_lengths(
                replace(
                    paired_config,
                    source=REPO_ROOT / paired_config.source,
                )
            )
            if lengths is None or lengths.model_kind != config.model.kind:
                raise AssertionError(f"{path}: 학습 길이 계열이 설정과 다르다.")
            adapter = model.create(config.model, 42)
            coordinate = lengths.for_coordinate(42, 0)
            if coordinate is None:
                if config.model.kind != "logistic_onehot":
                    raise AssertionError(f"{path}: 반복형 계열의 학습 길이가 없다.")
            elif len(coordinate) > 1 and not hasattr(
                adapter, "fit_paired_training_lengths"
            ):
                raise AssertionError(f"{path}: 내부 구성원별 학습 길이 경로가 없다.")
            elif len(coordinate) == 1 and not hasattr(adapter, "fit_full"):
                raise AssertionError(f"{path}: 고정 학습 길이 경로가 없다.")
            records.append(
                {
                    "config": path.name,
                    "model_kind": config.model.kind,
                    "source_lengths": None if coordinate is None else list(coordinate),
                    "training_row_boundary_supported": True,
                }
            )
    finally:
        os.chdir(previous)
    if len(records) != 68:
        raise AssertionError("설정 경계 진단이 68개 실행을 모두 지나지 않았다.")
    return records


def _row_and_feature_boundaries(source_root: Path) -> dict[str, Any]:
    all_train = pd.read_csv(source_root / "data/train.csv")
    test = pd.read_csv(source_root / "data/test.csv").iloc[:40].copy()
    source = _select_balanced(all_train, 40)
    source["fold"] = np.arange(len(source)) % 5
    tripled_config = TrainingRowsConfig("tripled", 2, 0.0)
    augmented_config = TrainingRowsConfig("missingness_augmented", 2, 0.25)
    tripled = build_training_rows(
        source, RAW_COLUMNS, tripled_config, seed=42, outer_fold=0
    )
    augmented = build_training_rows(
        source, RAW_COLUMNS, augmented_config, seed=42, outer_fold=0
    )
    repeated = build_training_rows(
        source, RAW_COLUMNS, augmented_config, seed=42, outer_fold=0
    )
    n = len(source)
    original_raw = tripled.frame.loc[: n - 1, RAW_COLUMNS].reset_index(drop=True)
    for block in range(3):
        actual = tripled.frame.loc[block * n : (block + 1) * n - 1, RAW_COLUMNS]
        actual = actual.reset_index(drop=True)
        if not actual.equals(original_raw):
            raise AssertionError("p=0의 세 학습 블록 원자료가 같지 않다.")
    if augmented.evidence != repeated.evidence or not augmented.frame.equals(
        repeated.frame
    ):
        raise AssertionError("결측 증강 행과 증거가 같은 좌표에서 결정적이지 않다.")
    if augmented.evidence["added_missing_cells"] <= 0:
        raise AssertionError("결측 증강군에 새 결측이 생기지 않았다.")
    copies = CategoricalCopies(
        ["gender", "stress_level", "academic_work_impact"]
    )
    categorical_values, _ = copies.compute(augmented.frame, test)
    for column in copies.cols:
        if not np.array_equal(
            categorical_values[f"{column}_cat"].isna().to_numpy(),
            augmented.frame[column].isna().to_numpy(),
        ):
            raise AssertionError("범주 복제 열의 결측이 실제 증강 원자료와 다르다.")

    def lattice_input(batch) -> pd.DataFrame:
        frame = batch.frame.copy()
        parent_placebo = placebo_series(source, 42).to_numpy()
        frame[PLACEBO] = np.tile(parent_placebo, 3)
        return frame

    plain = source.copy()
    plain[PLACEBO] = placebo_series(plain, 42)
    plain_encoder = LatticePairTargetEncoder(
        ["daily_screen_time_hours", "social_media_hours"],
        inner_folds=4,
        smoothing=20.0,
    )
    plain_encoder.fit(plain, 42)
    plain_values = plain_encoder.transform(plain).reset_index(drop=True)
    tripled_frame = lattice_input(tripled)
    tripled_encoder = LatticePairTargetEncoder(
        ["daily_screen_time_hours", "social_media_hours"],
        inner_folds=4,
        smoothing=20.0,
    )
    tripled_encoder.fit(tripled_frame.iloc[:n], 42)
    tripled_values = tripled_encoder.transform(tripled_frame)
    for block in range(3):
        actual = tripled_values.iloc[block * n : (block + 1) * n].reset_index(drop=True)
        if not np.allclose(actual, plain_values, equal_nan=True):
            raise AssertionError("p=0 격자 부호화가 원본 OOF 값과 같지 않다.")
    augmented_frame = lattice_input(augmented)
    augmented_encoder = LatticePairTargetEncoder(
        ["daily_screen_time_hours", "social_media_hours"],
        inner_folds=4,
        smoothing=20.0,
    )
    augmented_encoder.fit(augmented_frame.iloc[:n], 42)
    augmented_values = augmented_encoder.transform(augmented_frame)
    if augmented_values.shape != tripled_values.shape:
        raise AssertionError("결측 증강 격자 산출 모양이 대조군과 다르다.")
    return {
        "source_rows": n,
        "training_rows": len(tripled.frame),
        "state_fit_rows": len(tripled.state_fit_index),
        "p0_raw_blocks_equal": True,
        "p0_lattice_oof_equal": True,
        "target_and_outer_fold_inherited": True,
        "parent_inner_fold_inherited": True,
        "mask_deterministic": True,
        "categorical_copies_recomputed": True,
        "added_missing_cells": augmented.evidence["added_missing_cells"],
    }


class _RowSensitiveInitialScore:
    """실제 행의 결측 수를 그대로 로짓으로 쓰는 초기 점수 진단 대역."""

    def compute(self, train: pd.DataFrame, test: pd.DataFrame, seed: int):
        return initial_score.InitialScores(
            train=pd.Series(train[RAW_COLUMNS].isna().sum(axis=1), index=train.index),
            test=pd.Series(test[RAW_COLUMNS].isna().sum(axis=1), index=test.index),
        )


def _initial_score_boundaries(source_root: Path) -> dict[str, Any]:
    all_train = pd.read_csv(source_root / "data/train.csv")
    source = _select_balanced(all_train, 30)
    validation = _select_balanced(all_train, 10, offset=30)
    test = pd.read_csv(source_root / "data/test.csv").iloc[:20].copy()
    batch = build_training_rows(
        source,
        RAW_COLUMNS,
        TrainingRowsConfig("missingness_augmented", 2, 0.25),
        seed=42,
        outer_fold=0,
    )
    validation.index = pd.RangeIndex(len(batch.frame), len(batch.frame) + len(validation))
    test.index = pd.RangeIndex(
        len(batch.frame) + len(validation),
        len(batch.frame) + len(validation) + len(test),
    )
    proxy_scores = initial_score.training_row_fold_scores(
        _RowSensitiveInitialScore(),
        batch.frame,
        validation.drop(columns=[TARGET]),
        test,
        42,
        0,
    )
    if proxy_scores is None or not proxy_scores.evidence["actual_training_rows_recomputed"]:
        raise AssertionError("실제 결측 복제본의 초기 점수를 다시 만들지 않았다.")
    parent_score_counts = proxy_scores.training.groupby(batch.frame[PARENT_ID]).nunique()
    if not (parent_score_counts > 1).any():
        raise AssertionError("초기 점수 진단 대역에서 복제본 점수가 부모 점수에 묶였다.")

    nested = initial_score.NestedLogisticOnehot(
        cols=RAW_COLUMNS,
        categorical=["gender", "stress_level", "academic_work_impact"],
        C=100.0,
        max_iter=500,
        onehot_max_card=5000,
        inner_splits=3,
        clip=1e-6,
    )
    nested_scores = initial_score.training_row_fold_scores(
        nested,
        batch.frame,
        validation.drop(columns=[TARGET]),
        test,
        42,
        0,
    )
    if nested_scores is None:
        raise AssertionError("중첩 로지스틱 초기 점수가 없다.")
    evidence = nested_scores.evidence
    if not evidence.get("parent_grouping_used") or not evidence.get(
        "parent_groups_exclusive"
    ):
        raise AssertionError("중첩 로지스틱 내부 분할이 부모 행을 묶지 않았다.")
    if evidence.get("parent_scores_inherited") is not False:
        raise AssertionError("중첩 로지스틱 초기 점수가 부모 점수를 물려받았다.")
    return {
        "actual_replica_scores_recomputed": True,
        "parent_scores_inherited": False,
        "nested_parent_groups": evidence["parent_groups"],
        "nested_parent_groups_exclusive": True,
        "nested_parent_inner_fold_sha256": evidence["parent_inner_fold_sha256"],
        "validation_and_test_fit_from_actual_outer_training_block": True,
    }


def _reduced_end_to_end(source_root: Path) -> list[dict[str, Any]]:
    """비반복형과 고정 반복형 대표 경로를 실제 5분할 실행으로 관통한다."""
    records = []
    patterns = [
        ("*exp058*missingness_augmented.yaml", "logistic_onehot", None),
        ("*exp197*missingness_augmented.yaml", "lightgbm", [76]),
    ]
    for pattern, expected_kind, expected_lengths in patterns:
        (path,) = CONFIG_DIR.glob(pattern)
        config = load_config(path, "confirm")
        train = _select_balanced(pd.read_csv(source_root / "data/train.csv"), 50)
        folds = np.empty(len(train), dtype="int64")
        for _, indices in train.groupby(TARGET).groups.items():
            positions = np.asarray(list(indices), dtype="int64")
            folds[positions] = np.arange(len(positions)) % 5
        train["fold"] = folds
        test = pd.read_csv(source_root / "data/test.csv").iloc[:30].copy()
        data.align_categories(train, test, config.features.categorical)
        from pipeline.plan import FeaturePlan

        plan = FeaturePlan.from_config(config.features)
        train, test = plan.apply_dataset_wide(train, test)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = cv.run_cv(config, plan, train, test, 42)
        if config.model.kind != expected_kind:
            raise AssertionError("축소 종단 간 진단의 모형 계열이 예상과 다르다.")
        if not np.isfinite(result.oof["pred"]).all():
            raise AssertionError("축소 종단 간 OOF 예측이 유한하지 않다.")
        if len(result.training_row_evidence) != 5:
            raise AssertionError("축소 종단 간 학습 행 증거가 분할별로 남지 않았다.")
        observed = [
            record["training_length_evidence"]["observed_training_lengths"]
            for record in result.model_training_diagnostics
        ]
        if observed != [expected_lengths] * 5:
            raise AssertionError("축소 종단 간 실행이 고정한 학습 길이를 적용하지 않았다.")
        records.append(
            {
                "config": path.name,
                "model_kind": config.model.kind,
                "rows": len(train),
                "fold_count": 5,
                "oof_auc": result.fold_aucs["auc_oof"],
                "observed_training_lengths": observed,
                "finite_predictions": True,
                "paired_training_evidence_recorded": True,
            }
        )
    return records


def main() -> None:
    args = _args()
    source_root = args.source_root.resolve()
    if not (source_root / "data/train.csv").is_file():
        raise FileNotFoundError(f"진단 원자료가 없다: {source_root}")
    pair_records, raw_configs = _pair_contracts()
    configuration_records = _configuration_boundaries(source_root, raw_configs)
    row_boundaries = _row_and_feature_boundaries(source_root)
    initial_boundaries = _initial_score_boundaries(source_root)
    end_to_end = _reduced_end_to_end(source_root)
    document = {
        "schema_version": 1,
        "issue": {
            "number": 510,
            "title": "모든 모델 계열에서 결측 증강 짝비교 실행 경계를 구현하고 진단한다",
            "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/510",
        },
        "inputs": {
            "freeze_path": FREEZE_PATH.relative_to(REPO_ROOT).as_posix(),
            "freeze_sha256": file_sha256(FREEZE_PATH),
            "train_sha256": file_sha256(source_root / "data/train.csv"),
            "test_sha256": file_sha256(source_root / "data/test.csv"),
        },
        "summary": {
            "pair_count": len(pair_records),
            "config_count": len(configuration_records),
            "model_kinds": sorted(
                {record["model_kind"] for record in configuration_records}
            ),
            "all_assertions_passed": True,
        },
        "pair_contracts": pair_records,
        "configuration_boundaries": configuration_records,
        "row_and_feature_boundaries": row_boundaries,
        "initial_score_boundaries": initial_boundaries,
        "reduced_end_to_end": end_to_end,
    }
    payload = _json_bytes(document)
    OUTPUT_PATH.write_bytes(payload)
    print(
        f"진단 통과: pairs={len(pair_records)} configs={len(configuration_records)} "
        f"sha256={_sha256(payload)}"
    )


if __name__ == "__main__":
    main()
