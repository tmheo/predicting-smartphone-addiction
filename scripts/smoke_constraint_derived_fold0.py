"""제약 파생 4열 승격 사다리 설정 12개의 fold 0 스모크. (#622)

설정을 정식 경로(`load_config`, `FeaturePlan`, `model.create`)로 읽어 fold 0, seed 42의
행렬을 만들고, 사다리 단계가 선언한 새 열이 행렬에 실제로 있는지와 4열의 결측 규약·격자
규약을 확인한 뒤 adapter를 한 번 학습해 fold 0 AUC를 남긴다.

판정 대상이 아니다. MLflow 실행을 만들지 않고 결과는 `--out` JSON Lines에만 쌓는다.
이미 기록된 실험 이름은 건너뛰므로 중단한 실행을 그대로 다시 시작하면 남은 설정부터 이어 달린다.

`--rows`와 `--device`는 연결 점검 전용이다. GPU가 없는 로컬에서 RealMLP 설정을 돌릴 때
행 표본을 줄이고 device를 cpu로 바꾼다. 이 두 옵션을 켠 기록은 fold 0 AUC를 읽지 않는다.

사용법:
    uv run python scripts/smoke_constraint_derived_fold0.py \\
        --out run-logs/issue-622/smoke-lgb.jsonl configs/constraint-derived/0[1-3]_*.yaml
    uv run python scripts/smoke_constraint_derived_fold0.py \\
        --out run-logs/issue-622/smoke-realmlp-cpu.jsonl --rows 20000 --device cpu \\
        configs/constraint-derived/1[0-2]_*.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import data  # noqa: E402
from pipeline import model as model_mod  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.features import (  # noqa: E402
    CONSTRAINT_DERIVED_COLS,
    CONSTRAINT_DERIVED_DECIMALS,
    CONSTRAINT_DERIVED_NAMES,
)
from pipeline.plan import FeaturePlan, prepare_fold_fit_input  # noqa: E402

SEED = 42
VALID_FOLD = 0
STAGE = "screen"
EXPERIMENT_PREFIX = "cdv2_"
LADDER_SUFFIXES = ("raw4", "cats_te", "ratio_round")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="제약 파생 4열 사다리 fold 0 스모크 (#622)")
    parser.add_argument("configs", nargs="+", help="configs/constraint-derived/*.yaml")
    parser.add_argument("--out", type=Path, required=True, help="JSON Lines 결과 경로")
    parser.add_argument(
        "--rows", type=int, default=None, help="연결 점검 전용 행 표본 수. 판정용 실행에서는 켜지 않는다."
    )
    parser.add_argument(
        "--device", default=None, help="연결 점검 전용 device 재정의(예: cpu). 모형 params에 device가 있을 때만 적용한다."
    )
    return parser.parse_args()


def ladder_step(experiment: str) -> str:
    if not experiment.startswith(EXPERIMENT_PREFIX):
        raise ValueError(f"실험 이름이 {EXPERIMENT_PREFIX!r}로 시작하지 않는다: {experiment}")
    for suffix in LADDER_SUFFIXES:
        if experiment.endswith(f"_{suffix}"):
            return suffix
    raise ValueError(f"실험 이름이 사다리 단계 접미 {LADDER_SUFFIXES}로 끝나지 않는다: {experiment}")


def expected_new_columns(step: str, model_kind: str) -> list[str]:
    """단계와 계열이 행렬에 반드시 가져야 하는 새 열. RealMLP의 범주 복제는 adapter 안 처리다."""
    columns = list(CONSTRAINT_DERIVED_COLS)
    if step in ("cats_te", "ratio_round"):
        if model_kind != "realmlp":
            columns += [f"{col}_cat" for col in CONSTRAINT_DERIVED_COLS]
        columns += [f"{col}_te" for col in CONSTRAINT_DERIVED_COLS] + ["placebo_noise_te"]
    if step == "ratio_round":
        columns += [name for name in CONSTRAINT_DERIVED_NAMES if name not in CONSTRAINT_DERIVED_COLS]
        columns += [f"{col}_te_r1" for col in CONSTRAINT_DERIVED_COLS] + ["placebo_noise_te_r1"]
    return columns


def check_four_columns(X: pd.DataFrame, train: pd.DataFrame) -> dict[str, object]:
    """4열의 결측 규약(성분 하나라도 결측이면 결측)과 0.01 격자 규약을 행렬에서 확인한다."""
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    daily = "daily_screen_time_hours"
    expected_defined = {
        "fake_daily": train[parts].notna().all(axis=1),
        "fake_social": train[[daily, "gaming_hours", "work_study_hours"]].notna().all(axis=1),
        "fake_work": train[[daily, "social_media_hours", "gaming_hours"]].notna().all(axis=1),
        "fake_game": train[[daily, "social_media_hours", "work_study_hours"]].notna().all(axis=1),
    }
    summary: dict[str, object] = {}
    for col in CONSTRAINT_DERIVED_COLS:
        values = X[col]
        defined = values.notna()
        if not defined.to_numpy().tolist() == expected_defined[col].to_numpy().tolist():
            raise AssertionError(f"{col}의 정의 행이 성분 관측 규약과 다르다.")
        observed = values[defined].to_numpy(dtype="float64")
        if not np.array_equal(observed, np.round(observed, CONSTRAINT_DERIVED_DECIMALS)):
            raise AssertionError(f"{col}에 0.01 격자를 벗어난 값이 있다.")
        if (observed < 0).any():
            raise AssertionError(f"{col}에 음수가 있다.")
        summary[f"{col}_defined_rate"] = round(float(defined.mean()), 4)
        summary[f"{col}_unique_values"] = int(len(np.unique(observed)))
    return summary


def build_fold0_matrices(cfg, plan: FeaturePlan, rows: int | None):
    """fold 0의 학습·검증·test 행렬을 정식 경로와 같은 순서로 만든다."""
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    if rows:
        train = train.head(rows).copy()
        test = test.head(rows).copy()
    data.align_categories(train, test, cfg.features.categorical)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)

    y = train[data.TARGET]
    va_idx = train.index[train["fold"] == VALID_FOLD]
    tr_idx = train.index[train["fold"] != VALID_FOLD]

    X = plan.build_matrix(train, SEED)
    X_test = plan.build_matrix(test, SEED)
    four_column_summary = check_four_columns(X, train)

    providers = plan.new_fold_fit_providers()
    X_fold, X_test_fold = X, X_test
    fold_fit_seconds: dict[str, float] = {}
    if providers:
        train_ff = prepare_fold_fit_input(train, X)
        test_ff = prepare_fold_fit_input(test, X_test)
        for kind, transformer in providers:
            started = time.time()
            train_values, test_values, _ = plan.materialize_fold_fit_provider(
                kind=kind,
                transformer=transformer,
                train_input=train_ff,
                test_input=test_ff,
                training_index=tr_idx,
                validation_index=va_idx,
                seed=SEED,
                fold=VALID_FOLD,
                recorder=None,
            )
            collision = set(train_values.columns) & set(X_fold.columns)
            if collision:
                raise AssertionError(f"fold-fit 컬럼 이름 충돌: {sorted(collision)}")
            X_fold = pd.concat([X_fold, train_values], axis=1)
            X_test_fold = pd.concat([X_test_fold, test_values], axis=1)
            label = f"{kind}:{','.join(transformer.columns()[:2])}"
            fold_fit_seconds[label] = round(time.time() - started, 1)
            print(f"[fold-fit] {label} {fold_fit_seconds[label]}s 컬럼 {len(train_values.columns)}개", flush=True)

    feature_names = plan.all_columns()
    assert list(X_fold.columns) == feature_names, "fold 0 컬럼 집합이 피처 계획과 다르다."
    assert list(X_test_fold.columns) == feature_names, "test 컬럼 집합이 피처 계획과 다르다."
    return X_fold, X_test_fold, y, tr_idx, va_idx, four_column_summary, fold_fit_seconds


def run_config(path: str, rows: int | None, device: str | None) -> dict[str, object]:
    cfg = load_config(path, STAGE)
    step = ladder_step(cfg.name)
    plan = FeaturePlan.from_config(cfg.features)
    expected = expected_new_columns(step, cfg.model.kind)

    started = time.time()
    X_fold, X_test_fold, y, tr_idx, va_idx, four_column_summary, fold_fit_seconds = (
        build_fold0_matrices(cfg, plan, rows)
    )
    feature_seconds = time.time() - started
    missing = [column for column in expected if column not in X_fold.columns]
    if missing:
        raise AssertionError(f"{cfg.name}: 단계 {step}가 선언해야 할 열이 행렬에 없다: {missing}")

    params = dict(cfg.model.params)
    if device is not None and "device" in params:
        params["device"] = device
    model_cfg = dataclasses.replace(cfg.model, params=params)
    adapter = model_mod.create(model_cfg, SEED)
    model_mod.set_dataset_reference(adapter, X_fold, X_test_fold)
    started = time.time()
    va_pred = adapter.fit(
        X_fold.loc[tr_idx], y.loc[tr_idx], X_fold.loc[va_idx], y.loc[va_idx], None, None
    )
    fit_seconds = time.time() - started
    va_pred = np.asarray(va_pred, dtype="float64")
    if va_pred.shape != (len(va_idx),) or not np.isfinite(va_pred).all():
        raise AssertionError(f"{cfg.name}: fold 0 예측 형태나 값이 잘못됐다.")
    auc = float(roc_auc_score(y.loc[va_idx], va_pred))
    diagnostics = model_mod.collect_training_diagnostics(adapter)

    record: dict[str, object] = {
        "experiment": cfg.name,
        "config": path,
        "model_kind": cfg.model.kind,
        "ladder_step": step,
        "seed": SEED,
        "fold": VALID_FOLD,
        "rows_override": rows,
        "device_override": device if "device" in params else None,
        "connectivity_only": bool(rows or device),
        "auc_fold_0": auc,
        "feature_count": len(plan.all_columns()),
        "expected_new_columns": len(expected),
        "training_rows": int(len(tr_idx)),
        "validation_rows": int(len(va_idx)),
        "feature_seconds": round(feature_seconds, 1),
        "fold_fit_seconds": fold_fit_seconds,
        "fit_seconds": round(fit_seconds, 1),
        **four_column_summary,
    }
    if cfg.model.kind == "realmlp" and diagnostics is not None:
        record["extra_raw_numeric_columns"] = diagnostics.get("extra_raw_numeric_columns")
        engineer = adapter._impl.engineer  # noqa: SLF001 스모크 전용 내부 확인
        record["realmlp_output_cat_cols"] = len(engineer.output_cat_cols)
        record["realmlp_extra_cat_embeddings"] = [
            name for name in engineer.output_cat_cols
            if name.endswith("_cat_") and name[: -len("_cat_")] in CONSTRAINT_DERIVED_COLS
        ]
        declared = list(params.get("extra_raw_numeric_columns", []))
        if sorted(record["realmlp_extra_cat_embeddings"]) != sorted(f"{c}_cat_" for c in declared):
            raise AssertionError(f"{cfg.name}: RealMLP 추가 열의 정확값 임베딩이 선언과 다르다.")
    return record


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["experiment"])
    for path in args.configs:
        cfg = load_config(path, STAGE)
        if cfg.name in done:
            print(f"[skip] {cfg.name} 이미 기록됨", flush=True)
            continue
        print(f"[start] {cfg.name} ({path})", flush=True)
        record = run_config(path, args.rows, args.device)
        with args.out.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[done] {record['experiment']} auc_fold_0={record['auc_fold_0']:.7f} "
            f"features={record['feature_count']} fit={record['fit_seconds']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
