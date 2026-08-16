"""정식 실행과 같은 피처 계획·모델 adapter를 쓰는 fold 진입 진단. (#140)

사용법:
    uv run python -m pipeline.entry_diagnostic configs/expNNN.yaml \
        --out-dir artifacts/entry-expNNN

기본값은 fold 0과 seed 42다.
이 명령은 MLflow 실행을 만들지 않고 champion/pool 장부를 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from . import data, initial_score, model, tracking
from .config import ExperimentConfig, load_config
from .data import ID, TARGET
from .ledger import CHAMPION_PATH
from .plan import FeaturePlan

SCHEMA_VERSION = 1
DEFAULT_FOLD = 0
DEFAULT_SEED = 42
DEFAULT_LIMIT_HOURS = 24.0
MODEL_LIMIT_HOURS = {"tabr_s": 20.0}
MODEL_FOLD_LIMIT_HOURS = {"tabr_s": 4.0}
MODEL_CUDA_MEMORY_FRACTION = {"tabr_s": 0.90}
AUC_FLOOR_MARGIN = 0.01
RESULT_NAME = "entry_diagnostic.json"
PREDICTIONS_NAME = "validation_predictions.parquet"
IMPORTANCE_NAME = "feature_importance.parquet"


@dataclass
class DiagnosticRun:
    """메모리 안의 진입 진단 결과와 저장할 표 형식 산출물."""

    result: dict[str, object]
    predictions: pd.DataFrame
    importance: pd.DataFrame


def _seconds(clock: Callable[[], float], started: float) -> float:
    return float(clock() - started)


def _reset_cuda_peak() -> bool:
    """PyTorch CUDA를 쓰는 adapter의 fold 최고 메모리 측정을 시작한다."""
    # CUDA가 없는 호스트에서 torch를 먼저 불러오면 이후 XGBoost/LightGBM의 OpenMP와
    # 충돌할 수 있다. NVIDIA 장치가 보이는 실행 환경에서만 torch를 불러온다.
    if not Path("/dev/nvidiactl").exists() and shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return True


def _cuda_peak(enabled: bool) -> dict[str, object]:
    if not enabled:
        return {
            "available": False,
            "source": "torch.cuda",
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
            "device_total_bytes": None,
        }
    import torch

    return {
        "available": True,
        "source": "torch.cuda",
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def _with_built_columns(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    extra = [column for column in X.columns if column not in df.columns]
    return pd.concat([df, X[extra]], axis=1) if extra else df


def _validate_importance(importance: pd.DataFrame, features: list[str]) -> None:
    if list(importance.columns) != ["feature", "gain"]:
        raise AssertionError("adapter importance는 feature, gain 두 컬럼이어야 한다.")
    if importance["feature"].tolist() != features:
        raise AssertionError("adapter importance의 피처 순서가 실제 학습 행렬과 다르다.")
    if not np.isfinite(importance["gain"].to_numpy(dtype="float64")).all():
        raise AssertionError("adapter importance에 유한하지 않은 값이 있다.")


def run_fold_diagnostic(
    cfg: ExperimentConfig,
    plan: FeaturePlan,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    fold: int = DEFAULT_FOLD,
    seed: int = DEFAULT_SEED,
    champion_fold_auc: float,
    limit_hours: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    prior_timings: dict[str, float] | None = None,
) -> DiagnosticRun:
    """준비된 자료에서 fold 하나만 실행하고 승격 진단 결과를 만든다.

    ``train``에는 커밋된 fold 배정이 이미 붙어 있어야 한다.
    ``plan.apply_dataset_wide``도 호출자가 정식 run.py와 같은 순서로 적용한다.
    """
    if fold not in set(train["fold"].astype(int)):
        raise ValueError(f"자료에 fold {fold}가 없다.")
    if seed < 0:
        raise ValueError("seed는 0 이상이어야 한다.")
    if limit_hours is None:
        limit_hours = MODEL_LIMIT_HOURS.get(cfg.model.kind, DEFAULT_LIMIT_HOURS)
        limit_source = "model" if cfg.model.kind in MODEL_LIMIT_HOURS else "common"
    else:
        limit_source = "command"
    if not np.isfinite(limit_hours) or limit_hours <= 0:
        raise ValueError("5-fold 시간 한도는 양수여야 한다.")

    timings = dict(prior_timings or {})
    total_started = clock()

    started = clock()
    initial_provider = initial_score.create(cfg.initial_score)
    initial_scores = (
        initial_provider.compute(train.drop(columns=[TARGET]), test, seed)
        if initial_provider is not None
        else None
    )
    if initial_scores is not None:
        assert initial_scores.train.index.equals(train.index), "train 초기 점수 인덱스가 다르다."
        assert initial_scores.test.index.equals(test.index), "test 초기 점수 인덱스가 다르다."
    X = plan.build_matrix(train, seed)
    X_test = plan.build_matrix(test, seed)
    timings["feature_build"] = _seconds(clock, started)

    va_idx = train.index[train["fold"] == fold]
    tr_idx = train.index[train["fold"] != fold]
    expected_ids = train.loc[va_idx, ID].reset_index(drop=True)
    y = train[TARGET]

    started = clock()
    X_fold, X_test_fold = X, X_test
    transformers = plan.fold_fit_transformers()
    if transformers:
        train_ff = _with_built_columns(train, X)
        test_ff = _with_built_columns(test, X_test)
        for transformer in transformers:
            transformer.fit(train_ff.loc[tr_idx], seed)
        X_fold = plan.add_fold_fit_columns(X, train_ff)
        X_test_fold = plan.add_fold_fit_columns(X_test, test_ff)
    assert list(X_fold.columns) == list(X_test_fold.columns), (
        "train/test의 fold-fit 컬럼 집합이 다르다."
    )
    assert list(X_fold.columns) == plan.all_columns(), (
        f"학습 컬럼이 피처 계획의 선언과 다르다: {list(X_fold.columns)} != {plan.all_columns()}"
    )
    timings["fold_prepare"] = _seconds(clock, started)

    adapter = model.create(cfg.model, seed)
    cuda_enabled = _reset_cuda_peak()

    started = clock()
    raw_validation_pred = adapter.fit(
        X_fold.loc[tr_idx],
        y.loc[tr_idx],
        X_fold.loc[va_idx],
        y.loc[va_idx],
        initial_scores.train.loc[tr_idx] if initial_scores is not None else None,
        initial_scores.train.loc[va_idx] if initial_scores is not None else None,
    )
    timings["fit_and_validation_predict"] = _seconds(clock, started)
    started = clock()
    adapter_diagnostics = model.collect_entry_diagnostics(adapter)
    adapter_abort_reason = model.collect_entry_abort_reason(adapter)
    timings["adapter_diagnostics"] = _seconds(clock, started)

    validation_pred = np.asarray(raw_validation_pred, dtype="float64")
    row_count_ok = validation_pred.shape == (len(va_idx),)
    flat_validation_pred = validation_pred.reshape(-1)
    source_order_ok = not isinstance(raw_validation_pred, pd.Series) or (
        raw_validation_pred.index.equals(va_idx)
    )
    artifact_ids = expected_ids.reindex(range(len(flat_validation_pred)))
    predictions = pd.DataFrame(
        {
            ID: artifact_ids.to_numpy(),
            "fold": np.full(len(flat_validation_pred), fold, dtype="int64"),
            "pred": flat_validation_pred,
        }
    )
    row_order_ok = row_count_ok and source_order_ok and predictions[ID].equals(expected_ids)
    finite_ok = bool(np.isfinite(flat_validation_pred).all())
    auc = (
        float(roc_auc_score(y.loc[va_idx], flat_validation_pred))
        if row_count_ok and row_order_ok and finite_ok
        else None
    )

    if adapter_abort_reason is None:
        started = clock()
        test_pred = np.asarray(
            adapter.predict(
                X_test_fold,
                initial_scores.test if initial_scores is not None else None,
            ),
            dtype="float64",
        )
        timings["test_predict"] = _seconds(clock, started)
        test_prediction_ok = test_pred.shape == (len(test),) and bool(
            np.isfinite(test_pred).all()
        )

        started = clock()
        importance = adapter.importance().copy()
        _validate_importance(importance, list(X_fold.columns))
        timings["importance"] = _seconds(clock, started)
    else:
        test_prediction_ok = False
        importance = pd.DataFrame(
            {
                "feature": pd.Series(dtype="object"),
                "gain": pd.Series(dtype="float64"),
            }
        )
        timings["test_predict"] = 0.0
        timings["importance"] = 0.0
    cuda = _cuda_peak(cuda_enabled)

    fold_seconds = sum(
        timings[name]
        for name in (
            "fold_prepare",
            "fit_and_validation_predict",
            "test_predict",
            "importance",
            "adapter_diagnostics",
        )
    )
    one_time_seconds = sum(
        value
        for name, value in timings.items()
        if name not in {
            "fold_prepare",
            "fit_and_validation_predict",
            "test_predict",
            "importance",
            "adapter_diagnostics",
            "total",
        }
    )
    projected_seconds = one_time_seconds + 5 * fold_seconds
    adapter_projected_seconds = adapter_diagnostics.observations.get(
        "projected_5fold_training_seconds"
    )
    if isinstance(adapter_projected_seconds, (int, float)) and np.isfinite(
        adapter_projected_seconds
    ):
        projected_seconds = max(projected_seconds, float(adapter_projected_seconds))
    timings["total"] = (
        sum(prior_timings.values()) + _seconds(clock, total_started)
        if prior_timings
        else _seconds(clock, total_started)
    )

    auc_floor = float(champion_fold_auc - AUC_FLOOR_MARGIN)
    fold_limit_hours = MODEL_FOLD_LIMIT_HOURS.get(cfg.model.kind)
    memory_fraction_limit = MODEL_CUDA_MEMORY_FRACTION.get(cfg.model.kind)
    fold_time_ok = fold_limit_hours is None or fold_seconds <= fold_limit_hours * 3600
    cuda_memory_fraction = (
        None
        if not cuda["available"]
        else float(int(cuda["max_reserved_bytes"]) / int(cuda["device_total_bytes"]))
    )
    cuda_memory_ok = (
        memory_fraction_limit is None
        or cuda_memory_fraction is None
        or cuda_memory_fraction <= memory_fraction_limit
    )
    cuda_memory_limit_bytes = None
    if cfg.model.kind == "trompt" and cuda["available"]:
        device_total = int(cuda["device_total_bytes"])
        cuda_memory_limit_bytes = (14 if device_total <= 20 * 1024**3 else 20) * 1024**3
        cuda_memory_ok = (
            cuda_memory_ok
            and int(cuda["max_reserved_bytes"]) <= cuda_memory_limit_bytes
        )
    checks = {
        "validation_row_count": row_count_ok,
        "validation_row_order": row_order_ok,
        "validation_predictions_finite": finite_ok,
        "test_prediction_shape_and_finiteness": test_prediction_ok,
        "adapter_assertions": all(adapter_diagnostics.assertions.values()),
        "adapter_entry_abort": adapter_abort_reason is None,
        "fold_auc_floor": auc is not None and auc >= auc_floor,
        "fold_time_limit": fold_time_ok,
        "cuda_memory_limit": cuda_memory_ok,
        "projected_time_limit": projected_seconds <= limit_hours * 3600,
    }
    reasons: list[str] = []
    if not row_count_ok:
        reasons.append("검증 예측 행 수가 fold 배정과 다르다.")
    if not row_order_ok:
        reasons.append("검증 예측 id 순서가 커밋된 fold 행 순서와 다르다.")
    if not finite_ok:
        reasons.append("검증 예측에 유한하지 않은 값이 있다.")
    if not test_prediction_ok:
        reasons.append("테스트 예측의 행 수가 다르거나 유한하지 않은 값이 있다.")
    for name, passed in adapter_diagnostics.assertions.items():
        if not passed:
            reasons.append(f"모델 assertion 실패: {name}")
    if adapter_abort_reason is not None:
        reasons.append(f"모델 진입 중단: {adapter_abort_reason}")
    if not checks["fold_auc_floor"]:
        reasons.append(f"fold {fold} AUC가 승격 하한 {auc_floor:.6f}보다 낮다.")
    if not checks["fold_time_limit"]:
        reasons.append(
            f"fold {fold} 시간이 모델 한도 {fold_limit_hours:.1f}시간을 넘는다."
        )
    if not checks["cuda_memory_limit"]:
        if cuda_memory_limit_bytes is not None:
            reasons.append(
                f"최고 CUDA 예약 메모리 {int(cuda['max_reserved_bytes'])}바이트가 "
                f"Trompt 한도 {cuda_memory_limit_bytes}바이트를 넘는다."
            )
        else:
            reasons.append(
                f"최고 CUDA 예약 메모리 비율 {cuda_memory_fraction:.3f}이 "
                f"모델 한도 {memory_fraction_limit:.2f}를 넘는다."
            )
    if not checks["projected_time_limit"]:
        reasons.append(
            f"seed 42 5-fold 예상 시간이 모델 한도 {limit_hours:.1f}시간을 넘는다."
        )
    passed = all(checks.values())
    if passed:
        reasons.append("공통 무결성 검사와 승격 문턱을 모두 통과했다.")

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "model_entry",
        "experiment": cfg.name,
        "config": str(cfg.source_path),
        "model_kind": cfg.model.kind,
        "fold": fold,
        "seed": seed,
        "features": list(X_fold.columns),
        "rows": {
            "training": len(tr_idx),
            "validation_expected": len(va_idx),
            "validation_predictions": len(predictions),
            "test": len(test),
        },
        "validation": {
            "auc": auc,
            "champion_fold_auc": float(champion_fold_auc),
            "auc_floor": auc_floor,
            "row_count_ok": row_count_ok,
            "row_order_ok": row_order_ok,
            "finite": finite_ok,
        },
        "timing_seconds": timings,
        "cuda": cuda,
        "adapter": {
            "assertions": adapter_diagnostics.assertions,
            "observations": adapter_diagnostics.observations,
            "abort_reason": adapter_abort_reason,
        },
        "projection": {
            "seed": seed,
            "seed_5fold_seconds": projected_seconds,
            "limit_hours": limit_hours,
            "limit_source": limit_source,
            "fold_limit_hours": fold_limit_hours,
            "cuda_memory_fraction": cuda_memory_fraction,
            "cuda_memory_fraction_limit": memory_fraction_limit,
            "cuda_memory_limit_bytes": cuda_memory_limit_bytes,
            "formula": "one_time_stages + 5 * fold_stages",
        },
        "decision": {
            "passed": passed,
            "status": "pass" if passed else "stop",
            "checks": checks,
            "reasons": reasons,
        },
        "artifacts": {
            "validation_predictions": PREDICTIONS_NAME,
            "feature_importance": IMPORTANCE_NAME,
        },
    }
    return DiagnosticRun(result=result, predictions=predictions, importance=importance)


def write_diagnostic(run: DiagnosticRun, out_dir: Path) -> None:
    """공통 JSON과 표 산출물을 새 디렉터리에 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=False)
    run.predictions.to_parquet(out_dir / PREDICTIONS_NAME, index=False)
    run.importance.to_parquet(out_dir / IMPORTANCE_NAME, index=False)
    (out_dir / RESULT_NAME).write_text(
        json.dumps(run.result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _champion_fold_auc(fold: int) -> float:
    raw = yaml.safe_load(CHAMPION_PATH.read_text())
    fold_aucs = raw["fold_aucs"]
    value = fold_aucs.get(fold, fold_aucs.get(str(fold)))
    if value is None:
        raise ValueError(f"{CHAMPION_PATH}에 fold {fold} AUC가 없다.")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="공통 fold 진입 진단 (#140)")
    parser.add_argument("config", help="정식 실행과 공유하는 실험 설정 YAML")
    parser.add_argument("--out-dir", required=True, help="새 진단 산출물 디렉터리")
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-5fold-hours",
        type=float,
        default=None,
        help="생략하면 공통 24시간, TabR-S 20시간 한도를 적용한다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_started = time.monotonic()
    cfg = load_config(args.config, "screen")
    plan = FeaturePlan.from_config(cfg.features)
    input_hashes = {
        "train": data.file_sha256(cfg.data.train),
        "test": data.file_sha256(cfg.data.test),
        "folds": data.file_sha256(cfg.data.folds),
    }
    input_hashes.update(
        {
            name: data.file_sha256(path)
            for name, path in initial_score.input_paths(cfg.initial_score).items()
        }
    )
    setup_seconds = time.monotonic() - setup_started

    data_started = time.monotonic()
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)
    data_load_seconds = time.monotonic() - data_started

    run = run_fold_diagnostic(
        cfg,
        plan,
        train,
        test,
        fold=args.fold,
        seed=args.seed,
        champion_fold_auc=_champion_fold_auc(args.fold),
        limit_hours=args.max_5fold_hours,
        prior_timings={"setup": setup_seconds, "data_load": data_load_seconds},
    )
    run.result["input_sha256"] = input_hashes
    run.result["git"] = tracking.git_state()
    write_diagnostic(run, Path(args.out_dir))
    decision = run.result["decision"]
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"result={Path(args.out_dir) / RESULT_NAME}")


if __name__ == "__main__":
    main()
