"""이슈 #500 학습 행 구성의 값싼 한 분할 진단.

성능을 고르지 않는다.
3배 행 대조군과 결측 확률 0인 증강군의 예측 동등성, 결측 확률 0.25 마스크의
재실행 결정성과 실행 증거 불변식만 확인한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline import cv, data
from pipeline.config import TrainingRowsConfig, load_config
from pipeline.fold_fit_reuse import canonical_json_bytes
from pipeline.plan import FeaturePlan
from pipeline.training_rows import build_training_rows


def _sha256_array(values: np.ndarray) -> str:
    values = np.asarray(values, dtype="<f8")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _stratified_prefix(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.sort_index()
    labels = sorted(frame[data.TARGET].unique())
    base, remainder = divmod(limit, len(labels))
    parts = [
        frame[frame[data.TARGET] == label].head(base + (index < remainder))
        for index, label in enumerate(labels)
    ]
    out = pd.concat(parts).sort_index()
    if len(out) != limit:
        raise ValueError(f"층화 진단 표본을 {limit}행으로 만들지 못했다: {len(out)}")
    return out


def _diagnostic_data(cfg, fold: int, row_limit: int | None, test_limit: int | None):
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)
    train = data.attach_folds(train, cfg.data.folds)
    validation = _stratified_prefix(train[train["fold"] == fold], row_limit)
    fitting = _stratified_prefix(train[train["fold"] != fold], row_limit)
    diagnostic = pd.concat([fitting, validation]).sort_index().reset_index(drop=True)
    diagnostic["fold"] = np.where(
        diagnostic[ID].isin(validation[ID]), 0, -1
    )
    if test_limit is not None:
        test = test.head(test_limit).copy()
    return diagnostic, test


ID = data.ID


def _run_arm(
    cfg, train, test, rows: TrainingRowsConfig | None, seed: int
):
    local_train = train.copy()
    local_test = test.copy()
    plan = FeaturePlan.from_config(cfg.features)
    local_train, local_test = plan.apply_dataset_wide(local_train, local_test)
    arm_cfg = replace(cfg, seeds=[seed], training_rows=rows)
    return cv.run_cv(arm_cfg, plan, local_train, local_test, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="이슈 #500 한 분할 학습 행 진단")
    parser.add_argument(
        "--config", default="configs/exp117_ag25_gbm_r21.yaml"
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--row-limit",
        type=int,
        help="진단 전용으로 학습 부분과 검증 부분에서 각각 취할 최대 행 수",
    )
    parser.add_argument("--test-limit", type=int)
    parser.add_argument(
        "--n-estimators",
        type=int,
        help="진단 전용 LightGBM 반복 상한. 생략하면 고정 설정을 그대로 쓴다.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/issue500-training-row-diagnostic.json")
    )
    args = parser.parse_args()
    if args.row_limit is not None and args.row_limit < 20:
        raise ValueError("안쪽 10분할 진단에는 학습 및 검증 표본이 각각 20행 이상 필요하다.")
    if args.n_estimators is not None and args.n_estimators < 1:
        raise ValueError("--n-estimators는 1 이상이어야 한다.")

    cfg = load_config(args.config, "screen")
    if cfg.model.kind != "lightgbm":
        raise ValueError("이 진단은 고정 LightGBM 기준 설정만 지원한다.")
    if args.n_estimators is not None:
        params = {**cfg.model.params, "n_estimators": args.n_estimators}
        cfg = replace(cfg, model=replace(cfg.model, params=params))
    train, test = _diagnostic_data(cfg, args.fold, args.row_limit, args.test_limit)

    legacy = _run_arm(cfg, train, test, None, args.seed)
    explicit_original = _run_arm(
        cfg, train, test, TrainingRowsConfig("original", 0, 0.0), args.seed
    )
    tripled_rows = TrainingRowsConfig("tripled", 2, 0.0)
    pzero_rows = TrainingRowsConfig("missingness_augmented", 2, 0.0)
    tripled = _run_arm(cfg, train, test, tripled_rows, args.seed)
    pzero = _run_arm(cfg, train, test, pzero_rows, args.seed)
    validation = train["fold"] == 0
    legacy_oof = legacy.oof.loc[validation, "pred"].to_numpy()
    explicit_original_oof = explicit_original.oof.loc[validation, "pred"].to_numpy()
    legacy_test = legacy.test_pred["pred"].to_numpy()
    explicit_original_test = explicit_original.test_pred["pred"].to_numpy()
    original_oof_max_abs = float(
        np.max(np.abs(legacy_oof - explicit_original_oof))
    )
    original_test_max_abs = float(
        np.max(np.abs(legacy_test - explicit_original_test))
    )
    tripled_oof = tripled.oof.loc[validation, "pred"].to_numpy()
    pzero_oof = pzero.oof.loc[validation, "pred"].to_numpy()
    tripled_test = tripled.test_pred["pred"].to_numpy()
    pzero_test = pzero.test_pred["pred"].to_numpy()
    oof_max_abs = float(np.max(np.abs(tripled_oof - pzero_oof)))
    test_max_abs = float(np.max(np.abs(tripled_test - pzero_test)))
    tripled_auc = float(roc_auc_score(train.loc[validation, data.TARGET], tripled_oof))
    pzero_auc = float(roc_auc_score(train.loc[validation, data.TARGET], pzero_oof))
    tolerance = 1e-12
    if original_oof_max_abs > tolerance or original_test_max_abs > tolerance:
        raise AssertionError(
            "기존 기본 경로와 명시적 원본 행 팔이 다르다: "
            f"oof={original_oof_max_abs}, test={original_test_max_abs}, tol={tolerance}"
        )
    if oof_max_abs > tolerance or test_max_abs > tolerance:
        raise AssertionError(
            f"p=0 동등성 실패: oof={oof_max_abs}, test={test_max_abs}, tol={tolerance}"
        )
    if abs(tripled_auc - pzero_auc) > tolerance:
        raise AssertionError("p=0 OOF AUC가 3배 행 대조군과 다르다.")

    raw_plan = FeaturePlan.from_config(cfg.features)
    raw_train = train.copy()
    raw_test = test.copy()
    raw_train, raw_test = raw_plan.apply_dataset_wide(raw_train, raw_test)
    source = raw_train[raw_train["fold"] != 0]
    mask_rows = TrainingRowsConfig("missingness_augmented", 2, 0.25)
    first = build_training_rows(
        source, raw_plan.raw_columns(), mask_rows, seed=args.seed, outer_fold=args.fold
    )
    second = build_training_rows(
        source, raw_plan.raw_columns(), mask_rows, seed=args.seed, outer_fold=args.fold
    )
    first_masks = [item["mask_sha256"] for item in first.evidence["replicas"]]
    second_masks = [item["mask_sha256"] for item in second.evidence["replicas"]]
    if first_masks != second_masks or first.evidence != second.evidence:
        raise AssertionError("같은 좌표의 결측 증강 마스크나 실행 증거가 재실행에서 달라졌다.")

    payload = {
        "schema_version": 1,
        "purpose": "execution-boundary-diagnostic-only",
        "improvement_judgment": False,
        "config": str(Path(args.config)),
        "config_sha256": data.file_sha256(Path(args.config)),
        "fold": args.fold,
        "seed": args.seed,
        "diagnostic_overrides": {
            "row_limit_per_role": args.row_limit,
            "test_limit": args.test_limit,
            "n_estimators": args.n_estimators,
        },
        "rows": {
            "fit_original": int((train["fold"] != 0).sum()),
            "validation": int(validation.sum()),
            "test": len(test),
        },
        "original_compatibility": {
            "tolerance": tolerance,
            "oof_max_abs_difference": original_oof_max_abs,
            "test_max_abs_difference": original_test_max_abs,
            "legacy_oof_sha256": _sha256_array(legacy_oof),
            "explicit_original_oof_sha256": _sha256_array(
                explicit_original_oof
            ),
            "legacy_test_sha256": _sha256_array(legacy_test),
            "explicit_original_test_sha256": _sha256_array(
                explicit_original_test
            ),
            "passed": True,
        },
        "pzero_equivalence": {
            "tolerance": tolerance,
            "oof_max_abs_difference": oof_max_abs,
            "test_max_abs_difference": test_max_abs,
            "tripled_oof_auc": tripled_auc,
            "pzero_oof_auc": pzero_auc,
            "tripled_oof_sha256": _sha256_array(tripled_oof),
            "pzero_oof_sha256": _sha256_array(pzero_oof),
            "tripled_test_sha256": _sha256_array(tripled_test),
            "pzero_test_sha256": _sha256_array(pzero_test),
            "passed": True,
        },
        "mask_replay": {
            "probability": 0.25,
            "first_mask_sha256": first_masks,
            "second_mask_sha256": second_masks,
            "evidence_sha256": hashlib.sha256(
                canonical_json_bytes(first.evidence)
            ).hexdigest(),
            "actual_added_missing_rate": first.evidence[
                "actual_added_missing_rate"
            ],
            "passed": True,
        },
        "assertions": first.evidence["assertions"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    print(args.out)
    print(json.dumps(payload["original_compatibility"], sort_keys=True))
    print(json.dumps(payload["pzero_equivalence"], sort_keys=True))
    print(json.dumps(payload["mask_replay"], sort_keys=True))


if __name__ == "__main__":
    main()
