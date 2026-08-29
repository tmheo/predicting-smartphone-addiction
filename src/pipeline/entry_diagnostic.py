"""정식 실행과 같은 피처 계획·모델 adapter를 쓰는 fold 진입 진단. (#140)

사용법:
    uv run python -m pipeline.entry_diagnostic configs/champion.yaml \
        --out-dir artifacts/entry-baseline --reference \
        --expected-baseline-auc 0.968294911389327
    uv run python -m pipeline.entry_diagnostic configs/challenger.yaml \
        --out-dir artifacts/entry-challenger \
        --baseline-diagnostic artifacts/entry-baseline/entry_diagnostic.json \
        --baseline-predictions artifacts/entry-baseline/validation_predictions.parquet

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
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data, initial_score, model, tracking
from .config import ExperimentConfig, load_config
from .data import ID, TARGET
from .ledger import Champion
from .plan import FeaturePlan
from .recovery import model_dependency_snapshot

SCHEMA_VERSION = 2
DEFAULT_FOLD = 0
DEFAULT_SEED = 42
DEFAULT_LIMIT_HOURS = 24.0
MODEL_LIMIT_HOURS = {"tabr_s": 20.0, "tabr": 24.0}
MODEL_FOLD_LIMIT_HOURS = {"tabr_s": 4.0, "tabr": 4.5}
MODEL_CUDA_MEMORY_FRACTION = {"tabr_s": 0.90, "tabr": 0.90}
AUC_FLOOR_MARGIN = 0.01
METRIC_TOLERANCE = 1e-9
MAX_EXPLICIT_REFERENCE_TOLERANCE = 2e-4
REFERENCE_MODE = "reference"
CHAMPION_IMPROVEMENT_MODE = "champion-improvement"
NEW_MODEL_FAMILY_MODE = "new-model-family"
COMPARISON_MODES = (CHAMPION_IMPROVEMENT_MODE, NEW_MODEL_FAMILY_MODE)
RESULT_NAME = "entry_diagnostic.json"
PREDICTIONS_NAME = "validation_predictions.parquet"
IMPORTANCE_NAME = "feature_importance.parquet"


@dataclass
class DiagnosticRun:
    """메모리 안의 진입 진단 결과와 저장할 표 형식 산출물."""

    result: dict[str, object]
    predictions: pd.DataFrame
    importance: pd.DataFrame


@dataclass(frozen=True)
class BaselineEvidence:
    """저장된 같은 단계 champion 진입 진단과 검증 예측."""

    result: dict[str, Any]
    predictions: pd.DataFrame
    auc: float


class DiagnosticEvidenceError(ValueError):
    """짝비교 증거가 불완전하거나 실행 정체성이 다르다."""


def apply_reference_reproduction_check(
    run: DiagnosticRun,
    expected_auc: float,
    *,
    tolerance: float = METRIC_TOLERANCE,
) -> None:
    """기준 재실행이 저장된 같은 단계 champion AUC를 재현하는지 확인한다."""
    if not np.isfinite(expected_auc):
        raise ValueError("저장된 같은 단계 champion AUC는 유한해야 한다.")
    if not np.isfinite(tolerance) or not 0 < tolerance <= MAX_EXPLICIT_REFERENCE_TOLERANCE:
        raise ValueError(
            "기준 AUC 허용 범위는 0보다 크고 "
            f"{MAX_EXPLICIT_REFERENCE_TOLERANCE} 이하여야 한다."
        )
    reproduced_auc = run.result["validation"]["auc"]
    matches = reproduced_auc is not None and abs(
        float(reproduced_auc) - expected_auc
    ) <= tolerance
    run.result["reference_reproduction"] = {
        "expected_auc": float(expected_auc),
        "reproduced_auc": reproduced_auc,
        "difference": (
            None if reproduced_auc is None else float(reproduced_auc) - expected_auc
        ),
        "tolerance": float(tolerance),
        "tolerance_source": (
            "strict" if tolerance == METRIC_TOLERANCE else "explicit_hardware_environment"
        ),
        "matches": matches,
    }
    checks = run.result["decision"]["checks"]
    checks["same_stage_champion_auc_reproduced"] = matches
    if not matches:
        run.result["decision"]["reasons"].append(
            "기준 재현 AUC가 저장된 같은 단계 champion 결과와 허용 범위 밖으로 다르다."
        )
    elif tolerance > METRIC_TOLERANCE:
        run.result["decision"]["reasons"].append(
            "기준 재현 AUC가 명시한 하드웨어 환경 허용 범위 안에 있다."
        )
    passed = all(checks.values())
    run.result["decision"]["passed"] = passed
    run.result["decision"]["status"] = "pass" if passed else "stop"


def build_execution_identity(
    cfg: ExperimentConfig,
    plan: FeaturePlan,
    input_sha256: dict[str, str],
    *,
    fold: int,
    seed: int,
    model_dependencies: dict[str, object],
) -> dict[str, object]:
    """진입 진단 둘을 짝지을 때 같아야 할 실행 정체성을 만든다."""
    if "folds" not in input_sha256:
        raise ValueError("실행 정체성에 folds 입력 해시가 없다.")
    initial = None
    if cfg.initial_score is not None:
        initial = {"kind": cfg.initial_score.kind, "params": cfg.initial_score.params}
    return {
        "stage": cfg.stage,
        "fold": fold,
        "seed": seed,
        "input_sha256": dict(sorted(input_sha256.items())),
        "folds_sha256": input_sha256["folds"],
        "feature_plan": {
            "base": cfg.features.base,
            "categorical": list(cfg.features.categorical),
            "providers": cfg.features.providers,
            "exclude": list(cfg.features.exclude),
            "columns": plan.all_columns(),
        },
        "initial_score": initial,
        "model": {
            "kind": cfg.model.kind,
            "params": cfg.model.params,
            "fit": cfg.model.fit,
        },
        "model_dependencies": model_dependencies,
    }


def _prediction_auc(predictions: pd.DataFrame, *, source: str) -> float:
    expected = [ID, "fold", TARGET, "pred"]
    if list(predictions.columns) != expected:
        raise DiagnosticEvidenceError(
            f"{source} 검증 예측 열 순서가 다르다: {list(predictions.columns)} != {expected}"
        )
    if predictions.empty:
        raise DiagnosticEvidenceError(f"{source} 검증 예측이 비어 있다.")
    if predictions[ID].duplicated().any():
        raise DiagnosticEvidenceError(f"{source} 검증 예측에 중복 id가 있다.")
    target = predictions[TARGET].to_numpy()
    pred = predictions["pred"].to_numpy()
    if not np.isfinite(target).all() or not np.isfinite(pred).all():
        raise DiagnosticEvidenceError(f"{source} 검증 목표값 또는 예측에 유한하지 않은 값이 있다.")
    return float(roc_auc_score(target, pred))


def load_baseline_evidence(
    diagnostic_path: Path, predictions_path: Path
) -> BaselineEvidence:
    """기준 JSON과 예측을 함께 읽고 저장 AUC를 재현한다."""
    try:
        result = json.loads(diagnostic_path.read_text())
        predictions = pd.read_parquet(predictions_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DiagnosticEvidenceError(f"기준 진입 진단 산출물을 읽을 수 없다: {exc}") from exc
    if not isinstance(result, dict):
        raise DiagnosticEvidenceError("기준 진입 진단 JSON 최상위 값이 객체가 아니다.")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise DiagnosticEvidenceError(
            f"기준 진입 진단 스키마가 다르다: {result.get('schema_version')} != {SCHEMA_VERSION}"
        )
    if result.get("mode") != REFERENCE_MODE:
        raise DiagnosticEvidenceError("기준 진입 진단 JSON은 --reference로 만든 결과여야 한다.")
    reproduction = result.get("reference_reproduction")
    if not isinstance(reproduction, dict) or reproduction.get("matches") is not True:
        raise DiagnosticEvidenceError(
            "저장된 같은 단계 champion AUC를 재현하지 못한 기준 진입 진단은 사용할 수 없다."
        )
    decision = result.get("decision")
    if not isinstance(decision, dict):
        raise DiagnosticEvidenceError("기준 진입 진단 JSON에 구조화된 결정이 없다.")
    if decision.get("status") == "stop":
        raise DiagnosticEvidenceError("중단된 기준 진입 진단은 짝비교에 사용할 수 없다.")
    try:
        stored_auc = float(result["validation"]["auc"])
        identity = result["execution_identity"]
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticEvidenceError("기준 진입 진단 JSON에 AUC 또는 실행 정체성이 없다.") from exc
    if not isinstance(identity, dict):
        raise DiagnosticEvidenceError("기준 실행 정체성이 객체가 아니다.")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DiagnosticEvidenceError("기준 진입 진단 JSON에 산출물 기록이 없다.")
    expected_prediction_sha = artifacts.get("validation_predictions_sha256")
    if expected_prediction_sha != data.file_sha256(predictions_path):
        raise DiagnosticEvidenceError(
            "기준 진입 진단 JSON과 검증 예측의 내용 해시가 일치하지 않는다."
        )
    reproduced_auc = _prediction_auc(predictions, source="기준")
    if abs(stored_auc - reproduced_auc) > METRIC_TOLERANCE:
        raise DiagnosticEvidenceError(
            "기준 재현 AUC가 저장된 같은 단계 champion 결과와 다르다: "
            f"stored={stored_auc:.12f} reproduced={reproduced_auc:.12f} "
            f"tolerance={METRIC_TOLERANCE}"
        )
    return BaselineEvidence(result=result, predictions=predictions, auc=reproduced_auc)


_MISSING = object()


def _model_differences(
    baseline: object, challenger: object, path: str = ""
) -> list[dict[str, object]]:
    if isinstance(baseline, dict) and isinstance(challenger, dict):
        differences: list[dict[str, object]] = []
        for key in sorted(set(baseline) | set(challenger)):
            child = f"{path}.{key}" if path else str(key)
            differences.extend(
                _model_differences(
                    baseline.get(key, _MISSING), challenger.get(key, _MISSING), child
                )
            )
        return differences
    if baseline == challenger:
        return []
    return [
        {
            "path": path,
            "baseline": None if baseline is _MISSING else baseline,
            "challenger": None if challenger is _MISSING else challenger,
            "baseline_present": baseline is not _MISSING,
            "challenger_present": challenger is not _MISSING,
        }
    ]


def _normalise_allowed_model_axes(axes: list[str]) -> list[str]:
    normalised: list[str] = []
    for raw in axes:
        axis = raw.removeprefix("model.").strip(".")
        if axis not in {"kind", "params", "fit"} and not axis.startswith(
            ("params.", "fit.")
        ):
            raise DiagnosticEvidenceError(
                f"허용 모델 차이 축이 잘못됐다: {raw!r}. kind, params, fit 또는 하위 경로를 쓸 것."
            )
        if axis not in normalised:
            normalised.append(axis)
    return normalised


def validate_pairing_identity(
    baseline: BaselineEvidence,
    challenger_identity: dict[str, object],
    allowed_model_axes: list[str],
) -> dict[str, object]:
    """학습 전에 짝비교의 정적 정체성과 허용 모델 차이를 검사한다."""
    baseline_identity = baseline.result["execution_identity"]
    checks: dict[str, bool] = {}
    labels = {
        "stage": "실행 단계",
        "input_sha256": "입력 내용 해시",
        "folds_sha256": "fold 파일",
        "fold": "fold 번호",
        "seed": "시드",
        "feature_plan": "피처 계획",
        "initial_score": "초기 점수 계획",
        "model_dependencies": "실행 의존성 판본",
    }
    for key, label in labels.items():
        checks[key] = baseline_identity.get(key) == challenger_identity.get(key)
        if not checks[key]:
            raise DiagnosticEvidenceError(f"짝비교 불가: 기준과 challenger의 {label}이 다르다.")

    axes = _normalise_allowed_model_axes(allowed_model_axes)
    differences = _model_differences(
        baseline_identity.get("model", {}), challenger_identity.get("model", {})
    )
    unexpected = [
        difference
        for difference in differences
        if not any(
            difference["path"] == axis
            or str(difference["path"]).startswith(f"{axis}.")
            for axis in axes
        )
    ]
    checks["model_difference_scope"] = not unexpected
    if unexpected:
        paths = ", ".join(str(item["path"]) for item in unexpected)
        raise DiagnosticEvidenceError(f"짝비교 불가: 허용하지 않은 모델 설정 차이가 있다: {paths}")
    return {
        "checks": checks,
        "allowed_model_axes": axes,
        "model_differences": differences,
    }


def validate_baseline_rows(
    baseline: BaselineEvidence, train: pd.DataFrame, *, fold: int
) -> None:
    """저장 기준 예측이 현재 입력의 같은 검증 행과 목표값을 가리키는지 검사한다."""
    expected = train.loc[train["fold"] == fold, [ID, "fold", TARGET]].reset_index(drop=True)
    actual = baseline.predictions[[ID, "fold", TARGET]].reset_index(drop=True)
    if not actual.equals(expected):
        raise DiagnosticEvidenceError(
            "짝비교 불가: 기준 검증 예측의 id 순서, fold 또는 목표값이 현재 입력과 다르다."
        )


def apply_paired_comparison(
    run: DiagnosticRun,
    baseline: BaselineEvidence,
    *,
    mode: str,
    pairing: dict[str, object],
    baseline_diagnostic_path: Path,
    baseline_predictions_path: Path,
) -> None:
    """같은 저장 예측을 재채점해 짝비교 문턱과 최종 결정을 적용한다."""
    if mode not in COMPARISON_MODES:
        raise ValueError(f"알 수 없는 짝비교 모드다: {mode}")
    baseline_rows = baseline.predictions[[ID, "fold", TARGET]].reset_index(drop=True)
    challenger_rows = run.predictions[[ID, "fold", TARGET]].reset_index(drop=True)
    row_pair_ok = baseline_rows.equals(challenger_rows)
    if not row_pair_ok:
        raise DiagnosticEvidenceError(
            "짝비교 불가: 기준과 challenger의 검증 id 순서, fold 또는 목표값이 다르다."
        )
    challenger_auc = _prediction_auc(run.predictions, source="challenger")
    stored_challenger_auc = run.result["validation"]["auc"]
    if stored_challenger_auc is None or abs(
        float(stored_challenger_auc) - challenger_auc
    ) > METRIC_TOLERANCE:
        raise DiagnosticEvidenceError("challenger 저장 AUC가 검증 예측 재채점과 다르다.")
    delta = challenger_auc - baseline.auc
    threshold = baseline.auc if mode == CHAMPION_IMPROVEMENT_MODE else baseline.auc - AUC_FLOOR_MARGIN
    performance_ok = challenger_auc >= threshold
    checks = run.result["decision"]["checks"]
    checks["paired_validation_rows"] = True
    checks["paired_auc_threshold"] = performance_ok
    reasons = run.result["decision"]["reasons"]
    if not performance_ok:
        if mode == CHAMPION_IMPROVEMENT_MODE:
            reasons.append(f"짝지은 fold AUC 차이 {delta:+.12f}가 개선 승격 문턱 0보다 낮다.")
        else:
            reasons.append(
                f"fold AUC가 새 모델 계열 진입 하한 {threshold:.6f}보다 낮다."
            )
    passed = all(checks.values())
    run.result["decision"]["passed"] = passed
    run.result["decision"]["status"] = "pass" if passed else "stop"
    if passed:
        reasons.append("짝비교 무결성 검사와 승격 문턱을 모두 통과했다.")
    run.result["mode"] = mode
    run.result["comparison"] = {
        **pairing,
        "baseline_diagnostic": str(baseline_diagnostic_path),
        "baseline_predictions": str(baseline_predictions_path),
        "baseline_experiment": baseline.result["experiment"],
        "baseline_auc": baseline.auc,
        "challenger_auc": challenger_auc,
        "auc_delta": delta,
        "auc_threshold": threshold,
        "metric_tolerance": METRIC_TOLERANCE,
    }


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
    execution_identity: dict[str, object],
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
    initial_scores = initial_score.seed_level_scores(initial_provider, train, test, seed)
    X = plan.build_matrix(train, seed)
    X_test = plan.build_matrix(test, seed)
    timings["feature_build"] = _seconds(clock, started)

    va_idx = train.index[train["fold"] == fold]
    tr_idx = train.index[train["fold"] != fold]
    expected_ids = train.loc[va_idx, ID].reset_index(drop=True)
    y = train[TARGET]

    started = clock()
    # 바깥쪽 분할 계약 초기 점수는 fold 준비 단계에서 학습 부분 목표값만으로 만든다. (#505)
    fold_initial = initial_score.fold_scores(
        initial_provider, initial_scores, train, test, seed, fold, tr_idx, va_idx
    )
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
    model.set_dataset_reference(adapter, X_fold, X_test_fold)
    cuda_enabled = _reset_cuda_peak()

    started = clock()
    raw_validation_pred = adapter.fit(
        X_fold.loc[tr_idx],
        y.loc[tr_idx],
        X_fold.loc[va_idx],
        y.loc[va_idx],
        fold_initial.training if fold_initial is not None else None,
        fold_initial.validation if fold_initial is not None else None,
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
            TARGET: y.loc[va_idx].reset_index(drop=True).reindex(
                range(len(flat_validation_pred))
            ),
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
                fold_initial.test if fold_initial is not None else None,
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
        reasons.append("공통 진입 진단 무결성 검사를 통과했다.")

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "model_entry",
        "mode": REFERENCE_MODE,
        "experiment": cfg.name,
        "config": str(cfg.source_path),
        "model_kind": cfg.model.kind,
        "fold": fold,
        "seed": seed,
        "execution_identity": execution_identity,
        "features": list(X_fold.columns),
        "rows": {
            "training": len(tr_idx),
            "validation_expected": len(va_idx),
            "validation_predictions": len(predictions),
            "test": len(test),
        },
        "validation": {
            "auc": auc,
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
    predictions_path = out_dir / PREDICTIONS_NAME
    importance_path = out_dir / IMPORTANCE_NAME
    run.predictions.to_parquet(predictions_path, index=False)
    run.importance.to_parquet(importance_path, index=False)
    run.result["artifacts"]["validation_predictions_sha256"] = data.file_sha256(
        predictions_path
    )
    run.result["artifacts"]["feature_importance_sha256"] = data.file_sha256(
        importance_path
    )
    (out_dir / RESULT_NAME).write_text(
        json.dumps(run.result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="짝지은 공통 fold 진입 진단 (#159)")
    parser.add_argument("config", help="정식 실행과 공유하는 실험 설정 YAML")
    parser.add_argument("--out-dir", required=True, help="새 진단 산출물 디렉터리")
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="현재 champion의 같은 fold·시드 기준 진단 산출물을 만든다.",
    )
    parser.add_argument(
        "--expected-baseline-auc",
        type=float,
        help="저장된 같은 단계 champion fold AUC. --reference 재현 검사에 필수다.",
    )
    parser.add_argument(
        "--reference-auc-tolerance",
        type=float,
        default=None,
        help=(
            "다른 GPU 환경의 기준 재현에만 명시하는 AUC 허용 범위. "
            f"기본값은 {METRIC_TOLERANCE}, 상한은 "
            f"{MAX_EXPLICIT_REFERENCE_TOLERANCE}다."
        ),
    )
    parser.add_argument(
        "--baseline-diagnostic",
        type=Path,
        help="--reference 기준 실행이 저장한 entry_diagnostic.json",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        help="--reference 기준 실행이 저장한 validation_predictions.parquet",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=COMPARISON_MODES,
        default=NEW_MODEL_FAMILY_MODE,
        help=(
            "champion-improvement는 짝지은 AUC 차이 0 이상, new-model-family는 "
            "champion - 0.01을 승격 문턱으로 쓴다."
        ),
    )
    parser.add_argument(
        "--allow-model-diff",
        action="append",
        default=[],
        metavar="PATH",
        help="비교 대상으로 허용할 모델 설정 축(kind, params, fit 또는 하위 경로). 반복 가능.",
    )
    parser.add_argument(
        "--max-5fold-hours",
        type=float,
        default=None,
        help="생략하면 공통 24시간, TabR-S 20시간 한도를 적용한다.",
    )
    args = parser.parse_args()
    has_baseline = args.baseline_diagnostic is not None or args.baseline_predictions is not None
    if args.reference:
        if has_baseline:
            parser.error("--reference는 기준 진단·예측 입력과 함께 쓸 수 없다.")
        if args.allow_model_diff:
            parser.error("--reference에는 --allow-model-diff를 쓸 수 없다.")
        if args.expected_baseline_auc is None:
            parser.error("--reference에는 --expected-baseline-auc가 필요하다.")
        if args.reference_auc_tolerance is not None and not (
            0 < args.reference_auc_tolerance <= MAX_EXPLICIT_REFERENCE_TOLERANCE
        ):
            parser.error(
                "--reference-auc-tolerance은 0보다 크고 "
                f"{MAX_EXPLICIT_REFERENCE_TOLERANCE} 이하여야 한다."
            )
    elif args.baseline_diagnostic is None or args.baseline_predictions is None:
        parser.error(
            "challenger 진단에는 --baseline-diagnostic과 --baseline-predictions가 모두 필요하다."
        )
    elif args.expected_baseline_auc is not None:
        parser.error("challenger 진단에는 --expected-baseline-auc를 쓸 수 없다.")
    elif args.reference_auc_tolerance is not None:
        parser.error("challenger 진단에는 --reference-auc-tolerance을 쓸 수 없다.")
    return args


def main() -> None:
    args = parse_args()
    setup_started = time.monotonic()
    cfg = load_config(args.config, "screen")
    plan = FeaturePlan.from_config(cfg.features)
    input_hashes: dict[str, str] = {
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

    identity = build_execution_identity(
        cfg,
        plan,
        input_hashes,
        fold=args.fold,
        seed=args.seed,
        model_dependencies=model_dependency_snapshot(),
    )
    baseline: BaselineEvidence | None = None
    pairing: dict[str, object] | None = None
    if args.reference:
        champion = Champion.load()
        if cfg.name != champion.config:
            raise DiagnosticEvidenceError(
                f"기준 진입 진단은 현재 champion 설정 {champion.config!r}만 허용한다: {cfg.name!r}"
            )
    else:
        baseline = load_baseline_evidence(
            args.baseline_diagnostic, args.baseline_predictions
        )
        champion = Champion.load()
        if baseline.result.get("experiment") != champion.config:
            raise DiagnosticEvidenceError(
                "기준 진입 진단이 현재 champion 설정과 다르다: "
                f"{baseline.result.get('experiment')!r} != {champion.config!r}"
            )
        pairing = validate_pairing_identity(
            baseline, identity, args.allow_model_diff
        )
        validate_baseline_rows(baseline, train, fold=args.fold)

    run = run_fold_diagnostic(
        cfg,
        plan,
        train,
        test,
        fold=args.fold,
        seed=args.seed,
        execution_identity=identity,
        limit_hours=args.max_5fold_hours,
        prior_timings={"setup": setup_seconds, "data_load": data_load_seconds},
    )
    if args.reference:
        apply_reference_reproduction_check(
            run,
            args.expected_baseline_auc,
            tolerance=args.reference_auc_tolerance or METRIC_TOLERANCE,
        )
    elif baseline is not None:
        assert pairing is not None
        apply_paired_comparison(
            run,
            baseline,
            mode=args.comparison_mode,
            pairing=pairing,
            baseline_diagnostic_path=args.baseline_diagnostic,
            baseline_predictions_path=args.baseline_predictions,
        )
    run.result["input_sha256"] = input_hashes
    run.result["git"] = tracking.git_state()
    write_diagnostic(run, Path(args.out_dir))
    decision = run.result["decision"]
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"result={Path(args.out_dir) / RESULT_NAME}")


if __name__ == "__main__":
    main()
