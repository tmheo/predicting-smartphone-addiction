"""교사-학생 순위 잔차 보정의 누출 없는 nested OOF 평가기.

사용법:
    uv run python -m pipeline.teacher_student_residual \
        configs/exp137_teacher_student_rank_residual.yaml \
        --baseline-oof /absolute/path/to/oof.parquet \
        --out-dir /absolute/path/to/results

각 바깥 분할에서 나머지 네 분할만 사용한다.
교사 OOF 목표와 학생 OOF 재구성은 그 네 분할 안의 내부 분할 네 개로 만든다.
바깥 검증 목표값은 최종 AUC 채점에만 사용한다.

완료된 평가는 MLflow 실행으로 기록한다.
일반 실행 기록 묶음 경로를 그대로 쓰기 위해 보정 OOF를 단일 고정 시드 OOF로 기록한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig, load_config
from .data import ID, TARGET, align_categories, attach_folds, file_sha256, load_csv
from .plan import FeaturePlan, prepare_fold_fit_input
from .runs import TRACKING_URI


Role = Literal["teacher", "student"]
SCHEMA_VERSION = 1
BASELINE_AUC_TOLERANCE = 1e-12
CHECKPOINT_PREFIX = "outer-fold-"


class ResidualEvaluationError(Exception):
    """실행 계약이나 입력 무결성이 어긋난 경우."""


@dataclass(frozen=True)
class BaselineSpec:
    run_id: str
    strategy: str
    pool_members: int
    auc_oof: float
    oof_sha256: str


@dataclass(frozen=True)
class ModelSpec:
    params: dict[str, Any]
    early_stopping_rounds: int


@dataclass(frozen=True)
class ResidualExperimentConfig:
    experiment: ExperimentConfig
    baseline: BaselineSpec
    seed: int
    refit_iteration_multiplier: float
    teacher: ModelSpec
    student: ModelSpec
    correction_weight: float
    clip_to_reference_range: bool
    minimum_auc_gain: float
    minimum_fold_wins: int


@dataclass(frozen=True)
class FitResult:
    prediction: np.ndarray
    best_iteration: int


class ResidualTrainer(Protocol):
    def fit_with_validation(
        self,
        role: Role,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        spec: ModelSpec,
        seed: int,
    ) -> FitResult: ...

    def fit_full(
        self,
        role: Role,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_predict: pd.DataFrame,
        spec: ModelSpec,
        seed: int,
        iterations: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ContrastCalibration:
    reference_std: float
    candidate_std: float
    scale: float
    clip_lower: float
    clip_upper: float


@dataclass(frozen=True)
class FoldOutcome:
    fold: int
    baseline_auc: float
    corrected_auc: float
    auc_gain: float
    teacher_refit_iterations: int
    student_refit_iterations: int
    calibration: ContrastCalibration


@dataclass(frozen=True)
class NestedResidualEvaluation:
    source_baseline_auc: float
    rank_control_auc: float
    corrected_auc: float
    source_auc_gain: float
    residual_auc_gain: float
    fold_wins: int
    passed: bool
    folds: list[FoldOutcome]
    oof: pd.DataFrame


def _model_spec(raw: dict[str, Any], label: str) -> ModelSpec:
    params = raw.get("params")
    fit = raw.get("fit")
    if not isinstance(params, dict) or not isinstance(fit, dict):
        raise ResidualEvaluationError(f"{label} 설정에는 params와 fit이 필요하다.")
    rounds = fit.get("early_stopping_rounds")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ResidualEvaluationError(
            f"{label}.fit.early_stopping_rounds는 양의 정수여야 한다."
        )
    estimators = params.get("n_estimators")
    if isinstance(estimators, bool) or not isinstance(estimators, int) or estimators < 1:
        raise ResidualEvaluationError(f"{label}.params.n_estimators는 양의 정수여야 한다.")
    return ModelSpec(params=dict(params), early_stopping_rounds=rounds)


def load_residual_config(path: str | Path) -> ResidualExperimentConfig:
    """표준 피처 계획과 전용 잔차 보정 블록을 함께 검증한다."""
    source = Path(path)
    experiment = load_config(source, "screen")
    raw = yaml.safe_load(source.read_text())
    residual = raw.get("teacher_student_residual")
    if not isinstance(residual, dict):
        raise ResidualEvaluationError("teacher_student_residual 설정 블록이 없다.")

    baseline_raw = residual.get("baseline")
    if not isinstance(baseline_raw, dict):
        raise ResidualEvaluationError("teacher_student_residual.baseline 설정이 없다.")
    baseline = BaselineSpec(
        run_id=str(baseline_raw["run_id"]),
        strategy=str(baseline_raw["strategy"]),
        pool_members=int(baseline_raw["pool_members"]),
        auc_oof=float(baseline_raw["auc_oof"]),
        oof_sha256=str(baseline_raw["oof_sha256"]),
    )
    if len(baseline.oof_sha256) != 64:
        raise ResidualEvaluationError("baseline.oof_sha256은 SHA-256 형식이어야 한다.")

    seed = residual.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ResidualEvaluationError("teacher_student_residual.seed는 정수여야 한다.")
    multiplier = float(residual.get("refit_iteration_multiplier", 0.0))
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ResidualEvaluationError("refit_iteration_multiplier는 양수여야 한다.")

    teacher = _model_spec(
        {"params": experiment.model.params, "fit": experiment.model.fit}, "teacher"
    )
    student = _model_spec(residual.get("student", {}), "student")

    correction = residual.get("correction")
    judgment = residual.get("judgment")
    if not isinstance(correction, dict) or not isinstance(judgment, dict):
        raise ResidualEvaluationError("correction과 judgment 설정이 필요하다.")
    weight = float(correction.get("weight", 0.0))
    if not np.isfinite(weight) or weight <= 0:
        raise ResidualEvaluationError("correction.weight는 양수여야 한다.")
    minimum_gain = float(judgment.get("minimum_auc_gain", -1.0))
    minimum_wins = judgment.get("minimum_fold_wins")
    if minimum_gain < 0 or not np.isfinite(minimum_gain):
        raise ResidualEvaluationError("minimum_auc_gain은 0 이상의 유한값이어야 한다.")
    if isinstance(minimum_wins, bool) or not isinstance(minimum_wins, int):
        raise ResidualEvaluationError("minimum_fold_wins는 정수여야 한다.")
    if minimum_wins < 0 or minimum_wins > 5:
        raise ResidualEvaluationError("minimum_fold_wins는 0 이상 5 이하여야 한다.")

    return ResidualExperimentConfig(
        experiment=experiment,
        baseline=baseline,
        seed=seed,
        refit_iteration_multiplier=multiplier,
        teacher=teacher,
        student=student,
        correction_weight=weight,
        clip_to_reference_range=bool(
            correction.get("clip_to_reference_range", False)
        ),
        minimum_auc_gain=minimum_gain,
        minimum_fold_wins=minimum_wins,
    )


def percentile_rank(values: pd.Series | np.ndarray) -> np.ndarray:
    """현재 평가 블록 안에서 평균 동률 백분위 순위를 계산한다."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ResidualEvaluationError("백분위 순위 입력은 비어 있지 않은 유한 1차원 값이어야 한다.")
    return pd.Series(array).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def signed_square_contrast(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    clip_to_reference_range: bool,
) -> tuple[np.ndarray, ContrastCalibration]:
    """후보 잔차 분산을 학습 부분 기준에 맞춘 뒤 부호 보존 제곱을 만든다."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 1 or cand.ndim != 1 or len(ref) == 0 or len(cand) == 0:
        raise ResidualEvaluationError("잔차 기준과 후보는 비어 있지 않은 1차원 값이어야 한다.")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ResidualEvaluationError("잔차 기준과 후보에 유한하지 않은 값이 있다.")
    reference_std = float(np.std(ref))
    candidate_std = float(np.std(cand))
    if reference_std <= 0 or candidate_std <= 0:
        raise ResidualEvaluationError("잔차 표준편차가 0이라 보정 크기를 맞출 수 없다.")
    scale = reference_std / candidate_std
    scaled = cand * scale
    lower = float(np.min(ref))
    upper = float(np.max(ref))
    if clip_to_reference_range:
        scaled = np.clip(scaled, lower, upper)
    signal = np.sign(scaled) * np.square(np.abs(scaled))
    return signal, ContrastCalibration(
        reference_std=reference_std,
        candidate_std=candidate_std,
        scale=scale,
        clip_lower=lower,
        clip_upper=upper,
    )


def _refit_iterations(best_iterations: list[int], multiplier: float) -> int:
    if not best_iterations or any(value < 1 for value in best_iterations):
        raise ResidualEvaluationError("내부 학습 반복 수가 비어 있거나 양수가 아니다.")
    scaled = float(np.median(best_iterations)) * multiplier
    return max(1, int(math.floor(scaled + 0.5)))


class LightGBMResidualTrainer:
    """분류 교사와 회귀 학생을 같은 고정 LightGBM 실행 경계로 감싼다."""

    @staticmethod
    def _model(role: Role, params: dict[str, Any], seed: int):
        import lightgbm as lgb

        resolved = dict(params)
        resolved["random_state"] = seed
        if role == "teacher":
            return lgb.LGBMClassifier(**resolved)
        return lgb.LGBMRegressor(**resolved)

    @staticmethod
    def _prediction(role: Role, model: Any, X: pd.DataFrame) -> np.ndarray:
        if role == "teacher":
            return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
        return np.asarray(model.predict(X), dtype=np.float64)

    def fit_with_validation(
        self,
        role: Role,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        spec: ModelSpec,
        seed: int,
    ) -> FitResult:
        import lightgbm as lgb

        model = self._model(role, spec.params, seed)
        model.fit(
            X_train,
            y_train,
            eval_X=X_valid,
            eval_y=y_valid,
            callbacks=[
                lgb.early_stopping(spec.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(model.best_iteration_ or spec.params["n_estimators"])
        prediction = self._prediction(role, model, X_valid)
        if prediction.shape != (len(X_valid),) or not np.isfinite(prediction).all():
            raise ResidualEvaluationError(f"{role} 검증 예측이 유한한 1차원 값이 아니다.")
        return FitResult(prediction=prediction, best_iteration=best_iteration)

    def fit_full(
        self,
        role: Role,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_predict: pd.DataFrame,
        spec: ModelSpec,
        seed: int,
        iterations: int,
    ) -> np.ndarray:
        params = dict(spec.params)
        params["n_estimators"] = iterations
        model = self._model(role, params, seed)
        model.fit(X_train, y_train)
        prediction = self._prediction(role, model, X_predict)
        if prediction.shape != (len(X_predict),) or not np.isfinite(prediction).all():
            raise ResidualEvaluationError(f"{role} 전체 맞춤 예측이 유한한 1차원 값이 아니다.")
        return prediction


class FeatureMatrixFactory:
    """표준 피처 계획을 쓰되 각 내부 학습 부분에서 fold-fit 상태를 다시 맞춘다."""

    def __init__(
        self,
        cfg: ResidualExperimentConfig,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        self.seed = cfg.seed
        self.plan = FeaturePlan.from_config(cfg.experiment.features)
        prepared_train, prepared_test = self.plan.apply_dataset_wide(train, test)
        self.train = prepared_train
        self.static = self.plan.build_matrix(prepared_train, cfg.seed)
        self.fold_input = prepare_fold_fit_input(prepared_train, self.static)
        self.transformers = self.plan.fold_fit_transformers()
        if any(transformer.uses_target for transformer in self.transformers):
            raise ResidualEvaluationError(
                "교사-학생 피처 계획은 목표값을 쓰는 fold-fit 제공자를 허용하지 않는다."
            )
        expected = self.plan.all_columns()
        if len(expected) != len(set(expected)):
            raise ResidualEvaluationError("피처 계획에 중복 열이 있다.")

    @property
    def feature_names(self) -> list[str]:
        return self.plan.all_columns()

    def pair(
        self, fit_index: pd.Index, predict_index: pd.Index
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(fit_index) == 0 or len(predict_index) == 0:
            raise ResidualEvaluationError("피처 행렬의 학습 부분과 예측 부분은 비어 있을 수 없다.")
        if set(fit_index) & set(predict_index):
            raise ResidualEvaluationError("피처 행렬의 학습 행과 예측 행이 겹친다.")
        X_fit = self.static.loc[fit_index].copy()
        X_predict = self.static.loc[predict_index].copy()
        for transformer in self.transformers:
            transformer.fit(self.fold_input.loc[fit_index], self.seed)
            fit_new = transformer.transform(self.fold_input.loc[fit_index])
            predict_new = transformer.transform(self.fold_input.loc[predict_index])
            X_fit = pd.concat([X_fit, fit_new], axis=1)
            X_predict = pd.concat([X_predict, predict_new], axis=1)
        expected = self.feature_names
        if list(X_fit.columns) != expected or list(X_predict.columns) != expected:
            raise ResidualEvaluationError("실제 교사-학생 피처 열이 선언과 다르다.")
        return X_fit, X_predict


def _fold_indexes(train: pd.DataFrame, outer_fold: int) -> tuple[pd.Index, pd.Index, list[int]]:
    outer_index = train.index[train["fold"] == outer_fold]
    inner_index = train.index[train["fold"] != outer_fold]
    inner_folds = sorted(int(value) for value in train.loc[inner_index, "fold"].unique())
    if len(inner_folds) != 4:
        raise ResidualEvaluationError(
            f"바깥 분할 {outer_fold}의 내부 분할이 4개가 아니다: {inner_folds}"
        )
    return inner_index, outer_index, inner_folds


def evaluate_outer_fold(
    cfg: ResidualExperimentConfig,
    train: pd.DataFrame,
    baseline: pd.Series,
    outer_fold: int,
    features: FeatureMatrixFactory,
    trainer: ResidualTrainer,
) -> tuple[FoldOutcome, pd.DataFrame]:
    """바깥 분할 하나를 완전히 닫힌 내부 OOF 절차로 평가한다."""
    inner_index, outer_index, inner_folds = _fold_indexes(train, outer_fold)
    y = train[TARGET].astype(np.float64)

    teacher_oof = pd.Series(np.nan, index=inner_index, dtype=np.float64)
    teacher_best: list[int] = []
    for inner_fold in inner_folds:
        valid_index = inner_index[train.loc[inner_index, "fold"] == inner_fold]
        fit_index = inner_index.difference(valid_index, sort=False)
        X_fit, X_valid = features.pair(fit_index, valid_index)
        result = trainer.fit_with_validation(
            "teacher",
            X_fit,
            y.loc[fit_index],
            X_valid,
            y.loc[valid_index],
            cfg.teacher,
            cfg.seed,
        )
        teacher_oof.loc[valid_index] = result.prediction
        teacher_best.append(result.best_iteration)
    if teacher_oof.isna().any():
        raise ResidualEvaluationError(f"바깥 분할 {outer_fold}의 교사 OOF가 완성되지 않았다.")
    teacher_target = pd.Series(
        percentile_rank(teacher_oof), index=inner_index, dtype=np.float64
    )

    student_oof = pd.Series(np.nan, index=inner_index, dtype=np.float64)
    student_best: list[int] = []
    for inner_fold in inner_folds:
        valid_index = inner_index[train.loc[inner_index, "fold"] == inner_fold]
        fit_index = inner_index.difference(valid_index, sort=False)
        X_fit, X_valid = features.pair(fit_index, valid_index)
        result = trainer.fit_with_validation(
            "student",
            X_fit,
            teacher_target.loc[fit_index],
            X_valid,
            teacher_target.loc[valid_index],
            cfg.student,
            cfg.seed,
        )
        student_oof.loc[valid_index] = result.prediction
        student_best.append(result.best_iteration)
    if student_oof.isna().any():
        raise ResidualEvaluationError(f"바깥 분할 {outer_fold}의 학생 OOF가 완성되지 않았다.")

    X_inner, X_outer = features.pair(inner_index, outer_index)
    teacher_iterations = _refit_iterations(
        teacher_best, cfg.refit_iteration_multiplier
    )
    student_iterations = _refit_iterations(
        student_best, cfg.refit_iteration_multiplier
    )
    teacher_outer = trainer.fit_full(
        "teacher",
        X_inner,
        y.loc[inner_index],
        X_outer,
        cfg.teacher,
        cfg.seed,
        teacher_iterations,
    )
    student_outer = trainer.fit_full(
        "student",
        X_inner,
        teacher_target,
        X_outer,
        cfg.student,
        cfg.seed,
        student_iterations,
    )

    reference_contrast = teacher_target.to_numpy() - percentile_rank(student_oof)
    outer_contrast = percentile_rank(teacher_outer) - percentile_rank(student_outer)
    signal, calibration = signed_square_contrast(
        reference_contrast,
        outer_contrast,
        clip_to_reference_range=cfg.clip_to_reference_range,
    )
    baseline_outer = baseline.loc[outer_index]
    rank_control = percentile_rank(baseline_outer)
    corrected = rank_control + cfg.correction_weight * signal
    baseline_auc = float(roc_auc_score(y.loc[outer_index], baseline_outer))
    corrected_auc = float(roc_auc_score(y.loc[outer_index], corrected))
    outcome = FoldOutcome(
        fold=outer_fold,
        baseline_auc=baseline_auc,
        corrected_auc=corrected_auc,
        auc_gain=corrected_auc - baseline_auc,
        teacher_refit_iterations=teacher_iterations,
        student_refit_iterations=student_iterations,
        calibration=calibration,
    )
    frame = pd.DataFrame(
        {
            ID: train.loc[outer_index, ID].to_numpy(),
            "fold": outer_fold,
            "baseline_pred": baseline_outer.to_numpy(dtype=np.float64),
            "rank_control_pred": rank_control,
            "teacher_pred": teacher_outer,
            "student_pred": student_outer,
            "contrast": outer_contrast,
            "correction_signal": signal,
            "pred": corrected,
        },
        index=outer_index,
    )
    return outcome, frame


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _checkpoint_paths(root: Path, fold: int) -> tuple[Path, Path]:
    return (
        root / f"{CHECKPOINT_PREFIX}{fold}.parquet",
        root / f"{CHECKPOINT_PREFIX}{fold}.json",
    )


def evaluate_nested_residual(
    cfg: ResidualExperimentConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline: pd.Series,
    *,
    trainer: ResidualTrainer | None = None,
    checkpoint_dir: Path | None = None,
    execution_identity: dict[str, str] | None = None,
) -> NestedResidualEvaluation:
    """다섯 바깥 분할을 평가하고 중단된 분할 단위 결과를 재사용한다."""
    active_trainer = trainer or LightGBMResidualTrainer()
    features = FeatureMatrixFactory(cfg, train, test)
    folds = sorted(int(value) for value in train["fold"].unique())
    if folds != [0, 1, 2, 3, 4]:
        raise ResidualEvaluationError(f"고정 바깥 분할은 0부터 4여야 한다: {folds}")

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        identity_path = checkpoint_dir / "execution-identity.json"
        identity = execution_identity or {}
        if identity_path.exists():
            existing = json.loads(identity_path.read_text())
            if existing != identity:
                raise ResidualEvaluationError("기존 체크포인트의 실행 식별자가 현재 입력과 다르다.")
        else:
            _write_json_atomic(identity_path, identity)

    outcomes: list[FoldOutcome] = []
    frames: list[pd.DataFrame] = []
    for fold in folds:
        checkpoint = metadata = None
        if checkpoint_dir is not None:
            checkpoint, metadata = _checkpoint_paths(checkpoint_dir, fold)
        if checkpoint is not None and metadata is not None and checkpoint.exists() and metadata.exists():
            frame = pd.read_parquet(checkpoint)
            stored = json.loads(metadata.read_text())
            calibration = ContrastCalibration(**stored.pop("calibration"))
            outcome = FoldOutcome(calibration=calibration, **stored)
        elif checkpoint is not None and metadata is not None and (
            checkpoint.exists() or metadata.exists()
        ):
            raise ResidualEvaluationError(
                f"바깥 분할 {fold} 체크포인트가 일부만 있어 재사용할 수 없다."
            )
        else:
            outcome, frame = evaluate_outer_fold(
                cfg, train, baseline, fold, features, active_trainer
            )
            if checkpoint is not None and metadata is not None:
                _write_parquet_atomic(checkpoint, frame.reset_index(drop=True))
                _write_json_atomic(metadata, asdict(outcome))
        outcomes.append(outcome)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if combined[ID].duplicated().any() or len(combined) != len(train):
        raise ResidualEvaluationError("바깥 분할 결과의 id가 전체 훈련 자료를 한 번씩 덮지 않는다.")
    order = pd.Index(train[ID], name=ID)
    combined = combined.set_index(ID).reindex(order).reset_index()
    if combined.isna().any().any():
        raise ResidualEvaluationError("결합한 보정 OOF에 결측값이 있다.")
    y = train.set_index(ID)[TARGET].reindex(order).to_numpy(dtype=np.float64)
    source_baseline_auc = float(roc_auc_score(y, combined["baseline_pred"]))
    rank_control_auc = float(roc_auc_score(y, combined["rank_control_pred"]))
    corrected_auc = float(roc_auc_score(y, combined["pred"]))
    source_gain = corrected_auc - source_baseline_auc
    residual_gain = corrected_auc - rank_control_auc
    wins = sum(outcome.auc_gain > 0 for outcome in outcomes)
    passed = (
        source_gain >= cfg.minimum_auc_gain
        and residual_gain >= cfg.minimum_auc_gain
        and wins >= cfg.minimum_fold_wins
    )
    return NestedResidualEvaluation(
        source_baseline_auc=source_baseline_auc,
        rank_control_auc=rank_control_auc,
        corrected_auc=corrected_auc,
        source_auc_gain=source_gain,
        residual_auc_gain=residual_gain,
        fold_wins=wins,
        passed=passed,
        folds=outcomes,
        oof=combined,
    )


def load_inputs(
    cfg: ResidualExperimentConfig, baseline_oof_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, str]]:
    """자료, 고정 분할과 기준 OOF의 id, 해시와 점수를 함께 검증한다."""
    experiment = cfg.experiment
    train = attach_folds(load_csv(experiment.data.train), experiment.data.folds)
    test = load_csv(experiment.data.test)
    align_categories(train, test, experiment.features.categorical)

    hashes = {
        "train": file_sha256(experiment.data.train),
        "test": file_sha256(experiment.data.test),
        "folds": file_sha256(experiment.data.folds),
        "baseline_oof": file_sha256(baseline_oof_path),
        "config": file_sha256(experiment.source_path),
    }
    if hashes["baseline_oof"] != cfg.baseline.oof_sha256:
        raise ResidualEvaluationError(
            "기준 OOF SHA-256이 설정과 다르다: "
            f"{hashes['baseline_oof']} != {cfg.baseline.oof_sha256}"
        )
    frame = pd.read_parquet(baseline_oof_path)
    prediction_column = "prediction" if "prediction" in frame.columns else "pred"
    if ID not in frame.columns or prediction_column not in frame.columns:
        raise ResidualEvaluationError("기준 OOF에는 id와 prediction 또는 pred 열이 필요하다.")
    if frame[ID].duplicated().any():
        raise ResidualEvaluationError("기준 OOF id가 중복된다.")
    expected_ids = pd.Index(train[ID], name=ID)
    baseline = frame.set_index(ID)[prediction_column].reindex(expected_ids)
    if baseline.isna().any() or not np.isfinite(baseline.to_numpy(dtype=np.float64)).all():
        raise ResidualEvaluationError("기준 OOF id가 훈련 자료와 다르거나 예측이 유한하지 않다.")
    if len(frame) != len(train):
        raise ResidualEvaluationError("기준 OOF 행 수가 훈련 자료와 다르다.")
    baseline.index = train.index
    actual_auc = float(roc_auc_score(train[TARGET], baseline))
    if abs(actual_auc - cfg.baseline.auc_oof) > BASELINE_AUC_TOLERANCE:
        raise ResidualEvaluationError(
            f"기준 OOF AUC {actual_auc:.16f}가 설정값 {cfg.baseline.auc_oof:.16f}와 다르다."
        )
    return train, test, baseline.astype(np.float64), hashes


def _result_payload(
    cfg: ResidualExperimentConfig,
    evaluation: NestedResidualEvaluation,
    input_hashes: dict[str, str],
    git_state: dict[str, str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": cfg.experiment.name,
        "baseline": asdict(cfg.baseline),
        "correction": {
            "weight": cfg.correction_weight,
            "clip_to_reference_range": cfg.clip_to_reference_range,
        },
        "judgment": {
            "minimum_auc_gain": cfg.minimum_auc_gain,
            "minimum_fold_wins": cfg.minimum_fold_wins,
            "passed": evaluation.passed,
        },
        "metrics": {
            "source_baseline_auc": evaluation.source_baseline_auc,
            "rank_control_auc": evaluation.rank_control_auc,
            "corrected_auc": evaluation.corrected_auc,
            "source_auc_gain": evaluation.source_auc_gain,
            "residual_auc_gain": evaluation.residual_auc_gain,
            "fold_wins": evaluation.fold_wins,
        },
        "folds": [asdict(outcome) for outcome in evaluation.folds],
        "input_sha256": input_hashes,
        "git": git_state,
        "elapsed_seconds": elapsed_seconds,
    }


def record_evaluation(
    cfg: ResidualExperimentConfig,
    evaluation: NestedResidualEvaluation,
    input_hashes: dict[str, str],
    result_payload: dict[str, Any],
    out_dir: Path,
    *,
    tracking_uri: str = TRACKING_URI,
) -> str:
    """묶음 반입이 재채점할 수 있는 완료 실행으로 평가를 기록한다."""
    from .tracking import git_state, mlflow_client

    state = git_state()
    if state["git_dirty"] != "False":
        raise ResidualEvaluationError("dirty 작업 폴더의 결과는 실행 저장소에 기록하지 않는다.")

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    oof = evaluation.oof.copy()
    oof_path = final_dir / "oof.parquet"
    seed_oof_path = final_dir / f"oof_seed_{cfg.seed}.parquet"
    result_path = final_dir / "result.json"
    _write_parquet_atomic(oof_path, oof)
    _write_parquet_atomic(seed_oof_path, oof[[ID, "fold", "pred"]])
    _write_json_atomic(result_path, result_payload)

    client, experiment_id = mlflow_client(tracking_uri)
    run_name = cfg.experiment.name
    run_id = client.create_run(experiment_id, run_name=run_name).info.run_id
    try:
        raw_columns = list(pd.read_csv(cfg.experiment.data.train, nrows=0).columns)
        feature_names = [
            column
            for _, _, columns, _ in FeaturePlan.from_config(
                cfg.experiment.features
            ).describe(raw_columns)
            for column in columns
        ]
        params = {
            "experiment": run_name,
            "seeds": str(cfg.seed),
            "stage": "confirm",
            "model.kind": "teacher_student_rank_residual",
            "features": ",".join(feature_names),
            "baseline.run_id": cfg.baseline.run_id,
            "baseline.strategy": cfg.baseline.strategy,
            "baseline.pool_members": str(cfg.baseline.pool_members),
            "baseline.auc_oof": repr(cfg.baseline.auc_oof),
            "correction.weight": repr(cfg.correction_weight),
            "refit_iteration_multiplier": repr(cfg.refit_iteration_multiplier),
        }
        for key, value in params.items():
            client.log_param(run_id, key, value)
        client.log_metric(run_id, "auc_oof", evaluation.corrected_auc)
        client.log_metric(run_id, f"auc_oof_seed_{cfg.seed}", evaluation.corrected_auc)
        client.log_metric(
            run_id, "auc_source_baseline", evaluation.source_baseline_auc
        )
        client.log_metric(run_id, "auc_rank_control", evaluation.rank_control_auc)
        client.log_metric(
            run_id, "delta_vs_source_baseline", evaluation.source_auc_gain
        )
        client.log_metric(
            run_id, "delta_vs_rank_control", evaluation.residual_auc_gain
        )
        client.log_metric(run_id, "fold_wins", evaluation.fold_wins)
        for outcome in evaluation.folds:
            client.log_metric(run_id, f"auc_fold_{outcome.fold}", outcome.corrected_auc)
            client.log_metric(
                run_id, f"delta_fold_{outcome.fold}", outcome.auc_gain
            )
        for key, value in state.items():
            client.set_tag(run_id, key, value)
        for name, digest in input_hashes.items():
            client.set_tag(run_id, f"sha256.{name}", digest)
        client.set_tag(run_id, "sha256.oof_prediction", file_sha256(oof_path))
        client.set_tag(run_id, "source.kind", "derived_ensemble_postprocess")
        client.set_tag(run_id, "source.issue", "186")
        client.log_artifact(run_id, str(cfg.experiment.source_path))
        client.log_artifact(run_id, str(oof_path))
        client.log_artifact(run_id, str(seed_oof_path))
        client.log_artifact(run_id, str(result_path))
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    (out_dir / "run_id.txt").write_text(run_id + "\n")
    return run_id


def _execution_identity(
    cfg: ResidualExperimentConfig,
    input_hashes: dict[str, str],
    git_state: dict[str, str],
) -> dict[str, str]:
    return {
        "experiment": cfg.experiment.name,
        "baseline_run_id": cfg.baseline.run_id,
        "git_commit": git_state["git_commit"],
        **{f"sha256.{name}": digest for name, digest in input_hashes.items()},
    }


def run_command(
    config_path: Path,
    baseline_oof_path: Path,
    out_dir: Path,
    tracking_uri: str,
) -> str:
    from .tracking import git_state

    state = git_state()
    if state["git_dirty"] != "False":
        raise ResidualEvaluationError("실행은 커밋된 clean 작업 폴더에서만 허용한다.")
    cfg = load_residual_config(config_path)
    train, test, baseline, hashes = load_inputs(cfg, baseline_oof_path)
    identity = _execution_identity(cfg, hashes, state)
    started = time.monotonic()
    evaluation = evaluate_nested_residual(
        cfg,
        train,
        test,
        baseline,
        checkpoint_dir=out_dir / "checkpoints",
        execution_identity=identity,
    )
    elapsed = time.monotonic() - started
    payload = _result_payload(cfg, evaluation, hashes, state, elapsed)
    run_id = record_evaluation(
        cfg,
        evaluation,
        hashes,
        payload,
        out_dir,
        tracking_uri=tracking_uri,
    )
    print(f"현재 풀 OOF AUC: {evaluation.source_baseline_auc:.16f}")
    print(f"분할별 순위 대조 AUC: {evaluation.rank_control_auc:.16f}")
    print(f"보정 OOF AUC: {evaluation.corrected_auc:.16f}")
    print(f"현재 풀 대비: {evaluation.source_auc_gain:+.16f}")
    print(f"순위 대조 대비: {evaluation.residual_auc_gain:+.16f}")
    print(f"분할 승리: {evaluation.fold_wins}/5")
    print(f"판정: {'통과' if evaluation.passed else '기각'}")
    print(f"run_id: {run_id}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="교사-학생 순위 잔차 보정 nested OOF 평가")
    parser.add_argument("config", type=Path)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    args = parser.parse_args()
    try:
        run_command(args.config, args.baseline_oof, args.out_dir, args.tracking_uri)
    except ResidualEvaluationError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
