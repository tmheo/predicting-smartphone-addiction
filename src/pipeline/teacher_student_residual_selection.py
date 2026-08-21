"""교사-학생 잔차 보정의 의사 목표 영향력과 보정 가중치를 중첩 OOF로 선택한다.

이 모듈은 이슈 186의 고정 `rho=0`, `beta=0.10` 경로와 분리된다.
각 바깥 분할의 목표값을 잠근 뒤 바깥 학습 부분의 OOF로만 `rho`와 `beta`를
선택한다.
선택에 쓰는 기준 풀 예측도 현재 바깥 분할을 제외하고 다시 맞춘
이중 중첩 산출물만 받아 기준 풀의 목표값 경계를 유지한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .data import ID, TARGET, file_sha256, labels
from .ensemble import (
    COMBINER_REGISTRY,
    Combiner,
    MlflowRunStore,
    evaluate_nested,
    member_matrix,
)
from .judgment import FOLDS_PATH
from .ledger import Pool
from .plan import FeaturePlan
from .runs import TRACKING_URI
from .teacher_student_residual import (
    ContrastCalibration,
    FeatureMatrixFactory,
    LightGBMResidualTrainer,
    ModelSpec,
    ResidualEvaluationError,
    ResidualExperimentConfig,
    ResidualTrainer,
    _execution_identity,
    _refit_iterations,
    _write_json_atomic,
    _write_parquet_atomic,
    load_residual_config,
    percentile_rank,
    signed_square_contrast,
)


SCHEMA_VERSION = 2
SELECTION_BASELINE_SCHEMA_VERSION = 1
SELECTION_COLUMN_PREFIX = "selection_pred_excluding_outer_"


@dataclass(frozen=True)
class SearchSpace:
    rhos: tuple[float, ...]
    betas: tuple[float, ...]


@dataclass(frozen=True)
class SelectionInputSpec:
    pool_sha256: str
    selection_baseline_sha256: str
    baseline_test_sha256: str


@dataclass(frozen=True)
class ResidualSelectionConfig:
    control: ResidualExperimentConfig
    inputs: SelectionInputSpec
    search: SearchSpace


@dataclass(frozen=True)
class CandidateScore:
    rho: float
    beta: float
    auc: float
    auc_gain: float
    fold_wins: int


@dataclass(frozen=True)
class SelectionDecision:
    rho: float
    beta: float
    auc: float
    auc_gain: float
    fold_wins: int


@dataclass(frozen=True)
class SelectionFoldOutcome:
    fold: int
    selected: SelectionDecision
    baseline_auc: float
    corrected_auc: float
    auc_gain: float
    teacher_refit_iterations: int
    student_refit_iterations: int
    pseudo_row_weight: float
    calibration: ContrastCalibration


@dataclass(frozen=True)
class FullSelectionOutcome:
    selected: SelectionDecision
    teacher_refit_iterations: int
    student_refit_iterations: int
    pseudo_row_weight: float
    calibration: ContrastCalibration


@dataclass(frozen=True)
class NestedSelectionEvaluation:
    source_baseline_auc: float
    corrected_auc: float
    auc_gain: float
    fold_rank_control_auc: float
    fold_rank_corrected_auc: float
    fold_rank_gain: float
    fold_wins: int
    passed_official: bool
    passed_final_correction: bool
    folds: list[SelectionFoldOutcome]
    candidate_tables: dict[int, list[CandidateScore]]
    full_candidates: list[CandidateScore]
    oof: pd.DataFrame
    full: FullSelectionOutcome
    test: pd.DataFrame


class WeightedResidualTrainer(ResidualTrainer, Protocol):
    def fit_weighted(
        self,
        role: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: pd.Series,
        X_predict: pd.DataFrame,
        spec: ModelSpec,
        seed: int,
        iterations: int,
    ) -> np.ndarray: ...


class LightGBMWeightedResidualTrainer(LightGBMResidualTrainer):
    """의사 목표 행의 총 가중치를 명시적으로 받는 학생 학습기."""

    def fit_weighted(
        self,
        role: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: pd.Series,
        X_predict: pd.DataFrame,
        spec: ModelSpec,
        seed: int,
        iterations: int,
    ) -> np.ndarray:
        if role != "student":
            raise ResidualEvaluationError("가중 학습은 학생 모형에만 허용한다.")
        if not X_train.index.equals(y_train.index) or not X_train.index.equals(
            sample_weight.index
        ):
            raise ResidualEvaluationError("가중 학습의 행 인덱스가 서로 다르다.")
        weights = sample_weight.to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ResidualEvaluationError("학습 가중치는 유한한 0 이상이고 합이 양수여야 한다.")
        params = dict(spec.params)
        params["n_estimators"] = iterations
        model = self._model(role, params, seed)
        model.fit(X_train, y_train, sample_weight=weights)
        prediction = self._prediction(role, model, X_predict)
        if prediction.shape != (len(X_predict),) or not np.isfinite(prediction).all():
            raise ResidualEvaluationError("가중 학생 예측이 유한한 1차원 값이 아니다.")
        return prediction


def _finite_grid(raw: Any, label: str, *, allow_zero: bool) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ResidualEvaluationError(f"{label}은 비어 있지 않은 목록이어야 한다.")
    values = tuple(float(value) for value in raw)
    lower_ok = all(value >= 0 if allow_zero else value > 0 for value in values)
    if not lower_ok or not all(np.isfinite(value) for value in values):
        boundary = "0 이상" if allow_zero else "0 초과"
        raise ResidualEvaluationError(f"{label}의 값은 {boundary}의 유한값이어야 한다.")
    if tuple(sorted(set(values))) != values:
        raise ResidualEvaluationError(f"{label}은 중복 없는 오름차순이어야 한다.")
    return values


def load_selection_config(path: str | Path) -> ResidualSelectionConfig:
    control = load_residual_config(path)
    source = Path(path)
    raw = yaml.safe_load(source.read_text())
    block = raw.get("nested_selection")
    if not isinstance(block, dict):
        raise ResidualEvaluationError("nested_selection 설정 블록이 없다.")
    inputs = block.get("inputs")
    search = block.get("search")
    if not isinstance(inputs, dict) or not isinstance(search, dict):
        raise ResidualEvaluationError("nested_selection.inputs와 search가 필요하다.")
    input_spec = SelectionInputSpec(
        pool_sha256=str(inputs["pool_sha256"]),
        selection_baseline_sha256=str(inputs["selection_baseline_sha256"]),
        baseline_test_sha256=str(inputs["baseline_test_sha256"]),
    )
    for label, digest in asdict(input_spec).items():
        if len(digest) != 64:
            raise ResidualEvaluationError(f"{label}은 SHA-256 형식이어야 한다.")
    rhos = _finite_grid(search.get("rhos"), "search.rhos", allow_zero=True)
    betas = _finite_grid(search.get("betas"), "search.betas", allow_zero=True)
    if not rhos or rhos[0] != 0.0 or not betas or betas[0] != 0.0:
        raise ResidualEvaluationError("rho와 beta 탐색 공간은 0을 첫 값으로 포함해야 한다.")
    return ResidualSelectionConfig(
        control=control,
        inputs=input_spec,
        search=SearchSpace(rhos=rhos, betas=betas),
    )


def pseudo_row_weight(rho: float, labeled_count: int, pseudo_count: int) -> float:
    if not np.isfinite(rho) or rho < 0:
        raise ResidualEvaluationError("rho는 0 이상의 유한값이어야 한다.")
    if labeled_count < 1 or pseudo_count < 1:
        raise ResidualEvaluationError("학습 행과 의사 목표 행 수는 양수여야 한다.")
    return float(rho * labeled_count / pseudo_count)


def candidate_pairs(search: SearchSpace) -> list[tuple[float, float]]:
    pairs = [(0.0, 0.0)]
    pairs.extend(
        (rho, beta)
        for rho in search.rhos
        for beta in search.betas
        if beta > 0.0
    )
    return pairs


def select_candidate(scores: list[CandidateScore]) -> SelectionDecision:
    if not scores:
        raise ResidualEvaluationError("선택할 안쪽 후보 점수가 없다.")
    best = max(scores, key=lambda score: (score.auc, score.fold_wins, -score.beta, -score.rho))
    return SelectionDecision(**asdict(best))


def prepare_selection_baseline(
    combiner: Combiner,
    member_predictions: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    outer_prediction: pd.Series,
) -> pd.DataFrame:
    """각 바깥 분할을 제외한 이중 중첩 기준 OOF를 만든다."""
    index = member_predictions.index
    if not index.equals(fold_of.index) or not index.equals(y.index):
        raise ResidualEvaluationError("기준 풀 행렬, 분할과 목표값 인덱스가 다르다.")
    outer = outer_prediction.reindex(index)
    if outer.isna().any() or not np.isfinite(outer.to_numpy(dtype=np.float64)).all():
        raise ResidualEvaluationError("기준 풀 바깥 OOF가 전체 id를 덮지 않는다.")
    folds = sorted(int(value) for value in fold_of.unique())
    if folds != [0, 1, 2, 3, 4]:
        raise ResidualEvaluationError(f"고정 분할은 0부터 4여야 한다: {folds}")
    result = pd.DataFrame(
        {ID: index.to_numpy(), "fold": fold_of.to_numpy(), "baseline_pred": outer.to_numpy()},
        index=index,
    )
    for excluded_outer in folds:
        column = np.full(len(index), np.nan, dtype=np.float64)
        for predicted_fold in folds:
            if predicted_fold == excluded_outer:
                continue
            fit = ~fold_of.isin((excluded_outer, predicted_fold))
            predict = fold_of == predicted_fold
            fitted = combiner.fit(member_predictions.loc[fit], y.loc[fit])
            column[predict.to_numpy()] = np.asarray(
                fitted.predict(member_predictions.loc[predict]), dtype=np.float64
            )
        expected_missing = (fold_of == excluded_outer).to_numpy()
        if not np.array_equal(np.isnan(column), expected_missing):
            raise ResidualEvaluationError("이중 중첩 기준 OOF의 행 경계가 다르다.")
        result[f"{SELECTION_COLUMN_PREFIX}{excluded_outer}"] = column
    return result.reset_index(drop=True)


def validate_selection_baseline(
    frame: pd.DataFrame, train: pd.DataFrame, baseline: pd.Series
) -> dict[int, pd.Series]:
    expected_columns = [ID, "fold", "baseline_pred"] + [
        f"{SELECTION_COLUMN_PREFIX}{fold}" for fold in range(5)
    ]
    if list(frame.columns) != expected_columns:
        raise ResidualEvaluationError("이중 중첩 기준 OOF 열이 설정 계약과 다르다.")
    if frame[ID].duplicated().any() or len(frame) != len(train):
        raise ResidualEvaluationError("이중 중첩 기준 OOF id가 전체 훈련 자료와 다르다.")
    aligned = frame.set_index(ID).reindex(pd.Index(train[ID], name=ID))
    if aligned["fold"].isna().any() or not np.array_equal(
        aligned["fold"].to_numpy(dtype=np.int64), train["fold"].to_numpy(dtype=np.int64)
    ):
        raise ResidualEvaluationError("이중 중첩 기준 OOF 분할이 고정 분할과 다르다.")
    if not np.array_equal(
        aligned["baseline_pred"].to_numpy(dtype=np.float64),
        baseline.to_numpy(dtype=np.float64),
    ):
        raise ResidualEvaluationError("이중 중첩 산출물의 바깥 예측이 기준 OOF와 다르다.")
    selections: dict[int, pd.Series] = {}
    fold_values = train["fold"].to_numpy(dtype=np.int64)
    for excluded_outer in range(5):
        values = aligned[f"{SELECTION_COLUMN_PREFIX}{excluded_outer}"]
        missing = values.isna().to_numpy()
        if not np.array_equal(missing, fold_values == excluded_outer):
            raise ResidualEvaluationError("이중 중첩 산출물의 제외 분할 경계가 다르다.")
        selections[excluded_outer] = pd.Series(
            values.to_numpy(dtype=np.float64), index=train.index, dtype=np.float64
        )
    return selections


@dataclass(frozen=True)
class _SelectionContext:
    decision: SelectionDecision
    candidates: list[CandidateScore]
    teacher_target: pd.Series
    reference_contrast: np.ndarray
    teacher_best_iterations: list[int]
    student_best_iterations: list[int]


@dataclass(frozen=True)
class _OuterComputation:
    fold: int
    selected: SelectionDecision
    teacher_refit_iterations: int
    student_refit_iterations: int
    pseudo_weight: float
    calibration: ContrastCalibration
    candidates: list[CandidateScore]
    frame: pd.DataFrame


def _weighted_student_prediction(
    cfg: ResidualSelectionConfig,
    trainer: WeightedResidualTrainer,
    X_labeled: pd.DataFrame,
    labeled_target: pd.Series,
    X_pseudo: pd.DataFrame,
    pseudo_target: pd.Series,
    *,
    rho: float,
    iterations: int,
) -> tuple[np.ndarray, float]:
    weight = pseudo_row_weight(rho, len(X_labeled), len(X_pseudo))
    if rho == 0.0:
        prediction = trainer.fit_full(
            "student",
            X_labeled,
            labeled_target,
            X_pseudo,
            cfg.control.student,
            cfg.control.seed,
            iterations,
        )
        return prediction, weight
    combined_X = pd.concat([X_labeled, X_pseudo], axis=0)
    combined_target = pd.concat([labeled_target, pseudo_target], axis=0).astype(
        np.float64
    )
    weights = pd.Series(
        np.concatenate(
            (
                np.ones(len(X_labeled), dtype=np.float64),
                np.full(len(X_pseudo), weight, dtype=np.float64),
            )
        ),
        index=combined_X.index,
        dtype=np.float64,
    )
    prediction = trainer.fit_weighted(
        "student",
        combined_X,
        combined_target,
        weights,
        X_pseudo,
        cfg.control.student,
        cfg.control.seed,
        iterations,
    )
    return prediction, weight


def _selection_context(
    cfg: ResidualSelectionConfig,
    train: pd.DataFrame,
    indexes: pd.Index,
    baseline: pd.Series,
    features: FeatureMatrixFactory,
    trainer: WeightedResidualTrainer,
) -> _SelectionContext:
    y = train[TARGET].astype(np.float64)
    folds = sorted(int(value) for value in train.loc[indexes, "fold"].unique())
    if len(folds) < 2:
        raise ResidualEvaluationError("안쪽 OOF 선택에는 분할이 2개 이상 필요하다.")

    teacher_oof = pd.Series(np.nan, index=indexes, dtype=np.float64)
    teacher_best: list[int] = []
    for fold in folds:
        valid_index = indexes[train.loc[indexes, "fold"] == fold]
        fit_index = indexes.difference(valid_index, sort=False)
        X_fit, X_valid = features.pair(fit_index, valid_index)
        result = trainer.fit_with_validation(
            "teacher",
            X_fit,
            y.loc[fit_index],
            X_valid,
            y.loc[valid_index],
            cfg.control.teacher,
            cfg.control.seed,
        )
        teacher_oof.loc[valid_index] = result.prediction
        teacher_best.append(result.best_iteration)
    if teacher_oof.isna().any():
        raise ResidualEvaluationError("안쪽 교사 OOF가 완성되지 않았다.")
    teacher_target = pd.Series(
        percentile_rank(teacher_oof), index=indexes, dtype=np.float64
    )

    student_oof = {
        rho: pd.Series(np.nan, index=indexes, dtype=np.float64)
        for rho in cfg.search.rhos
    }
    student_best: list[int] = []
    for fold in folds:
        valid_index = indexes[train.loc[indexes, "fold"] == fold]
        fit_index = indexes.difference(valid_index, sort=False)
        X_fit, X_valid = features.pair(fit_index, valid_index)
        control = trainer.fit_with_validation(
            "student",
            X_fit,
            teacher_target.loc[fit_index],
            X_valid,
            teacher_target.loc[valid_index],
            cfg.control.student,
            cfg.control.seed,
        )
        student_oof[0.0].loc[valid_index] = control.prediction
        student_best.append(control.best_iteration)
        pseudo_target = pd.Series(
            percentile_rank(teacher_oof.loc[valid_index]),
            index=valid_index,
            dtype=np.float64,
        )
        for rho in cfg.search.rhos:
            if rho == 0.0:
                continue
            prediction, _ = _weighted_student_prediction(
                cfg,
                trainer,
                X_fit,
                teacher_target.loc[fit_index],
                X_valid,
                pseudo_target,
                rho=rho,
                iterations=control.best_iteration,
            )
            student_oof[rho].loc[valid_index] = prediction

    rank_control = percentile_rank(baseline.loc[indexes])
    baseline_auc = float(roc_auc_score(y.loc[indexes], rank_control))
    baseline_fold_aucs = {
        fold: float(
            roc_auc_score(
                y.loc[indexes[train.loc[indexes, "fold"] == fold]],
                rank_control[train.loc[indexes, "fold"].to_numpy() == fold],
            )
        )
        for fold in folds
    }
    contrasts: dict[float, np.ndarray] = {}
    candidates: list[CandidateScore] = []
    for rho, values in student_oof.items():
        if values.isna().any():
            raise ResidualEvaluationError(f"rho={rho}의 안쪽 학생 OOF가 완성되지 않았다.")
        contrast = teacher_target.to_numpy() - percentile_rank(values)
        contrasts[rho] = contrast
    for rho, beta in candidate_pairs(cfg.search):
        signal = np.sign(contrasts[rho]) * np.square(np.abs(contrasts[rho]))
        prediction = rank_control + beta * signal
        auc = float(roc_auc_score(y.loc[indexes], prediction))
        wins = 0
        for fold in folds:
            mask = train.loc[indexes, "fold"].to_numpy() == fold
            candidate_auc = float(roc_auc_score(y.loc[indexes[mask]], prediction[mask]))
            wins += candidate_auc > baseline_fold_aucs[fold]
        candidates.append(
            CandidateScore(
                rho=rho,
                beta=beta,
                auc=auc,
                auc_gain=auc - baseline_auc,
                fold_wins=wins,
            )
        )
    decision = select_candidate(candidates)
    return _SelectionContext(
        decision=decision,
        candidates=candidates,
        teacher_target=teacher_target,
        reference_contrast=contrasts[decision.rho],
        teacher_best_iterations=teacher_best,
        student_best_iterations=student_best,
    )


def _evaluate_outer(
    cfg: ResidualSelectionConfig,
    train: pd.DataFrame,
    baseline: pd.Series,
    selection_baseline: pd.Series,
    outer_fold: int,
    features: FeatureMatrixFactory,
    trainer: WeightedResidualTrainer,
) -> _OuterComputation:
    outer_index = train.index[train["fold"] == outer_fold]
    inner_index = train.index[train["fold"] != outer_fold]
    if selection_baseline.loc[inner_index].isna().any():
        raise ResidualEvaluationError(f"바깥 분할 {outer_fold}의 선택 기준 OOF가 비어 있다.")
    context = _selection_context(
        cfg,
        train,
        inner_index,
        selection_baseline,
        features,
        trainer,
    )
    y = train[TARGET].astype(np.float64)
    X_inner, X_outer = features.pair(inner_index, outer_index)
    teacher_iterations = _refit_iterations(
        context.teacher_best_iterations, cfg.control.refit_iteration_multiplier
    )
    student_iterations = _refit_iterations(
        context.student_best_iterations, cfg.control.refit_iteration_multiplier
    )
    teacher_outer = trainer.fit_full(
        "teacher",
        X_inner,
        y.loc[inner_index],
        X_outer,
        cfg.control.teacher,
        cfg.control.seed,
        teacher_iterations,
    )
    pseudo_target = pd.Series(
        percentile_rank(teacher_outer), index=X_outer.index, dtype=np.float64
    )
    student_outer, weight = _weighted_student_prediction(
        cfg,
        trainer,
        X_inner,
        context.teacher_target,
        X_outer,
        pseudo_target,
        rho=context.decision.rho,
        iterations=student_iterations,
    )
    outer_contrast = percentile_rank(teacher_outer) - percentile_rank(student_outer)
    signal, calibration = signed_square_contrast(
        context.reference_contrast,
        outer_contrast,
        clip_to_reference_range=cfg.control.clip_to_reference_range,
    )
    frame = pd.DataFrame(
        {
            ID: train.loc[outer_index, ID].to_numpy(),
            "fold": outer_fold,
            "baseline_pred": baseline.loc[outer_index].to_numpy(dtype=np.float64),
            "teacher_pred": teacher_outer,
            "student_pred": student_outer,
            "contrast": outer_contrast,
            "correction_signal": signal,
            "selected_rho": context.decision.rho,
            "selected_beta": context.decision.beta,
            "pseudo_row_weight": weight,
        },
        index=outer_index,
    )
    return _OuterComputation(
        fold=outer_fold,
        selected=context.decision,
        teacher_refit_iterations=teacher_iterations,
        student_refit_iterations=student_iterations,
        pseudo_weight=weight,
        calibration=calibration,
        candidates=context.candidates,
        frame=frame,
    )


def _fit_full(
    cfg: ResidualSelectionConfig,
    train: pd.DataFrame,
    baseline: pd.Series,
    baseline_test: pd.Series,
    features: FeatureMatrixFactory,
    trainer: WeightedResidualTrainer,
) -> tuple[FullSelectionOutcome, list[CandidateScore], pd.DataFrame]:
    indexes = train.index
    context = _selection_context(
        cfg, train, indexes, baseline, features, trainer
    )
    y = train[TARGET].astype(np.float64)
    X_train, X_test = features.external_pair(indexes)
    teacher_iterations = _refit_iterations(
        context.teacher_best_iterations, cfg.control.refit_iteration_multiplier
    )
    student_iterations = _refit_iterations(
        context.student_best_iterations, cfg.control.refit_iteration_multiplier
    )
    teacher_test = trainer.fit_full(
        "teacher",
        X_train,
        y,
        X_test,
        cfg.control.teacher,
        cfg.control.seed,
        teacher_iterations,
    )
    pseudo_target = pd.Series(
        percentile_rank(teacher_test), index=X_test.index, dtype=np.float64
    )
    student_test, weight = _weighted_student_prediction(
        cfg,
        trainer,
        X_train,
        context.teacher_target,
        X_test,
        pseudo_target,
        rho=context.decision.rho,
        iterations=student_iterations,
    )
    contrast = percentile_rank(teacher_test) - percentile_rank(student_test)
    signal, calibration = signed_square_contrast(
        context.reference_contrast,
        contrast,
        clip_to_reference_range=cfg.control.clip_to_reference_range,
    )
    rank_control = percentile_rank(baseline_test)
    prediction = rank_control + context.decision.beta * signal
    frame = pd.DataFrame(
        {
            ID: baseline_test.index.to_numpy(),
            "baseline_pred": baseline_test.to_numpy(dtype=np.float64),
            "rank_control_pred": rank_control,
            "teacher_pred": teacher_test,
            "student_pred": student_test,
            "contrast": contrast,
            "correction_signal": signal,
            "selected_rho": context.decision.rho,
            "selected_beta": context.decision.beta,
            "pseudo_row_weight": weight,
            "pred": prediction,
        }
    )
    outcome = FullSelectionOutcome(
        selected=context.decision,
        teacher_refit_iterations=teacher_iterations,
        student_refit_iterations=student_iterations,
        pseudo_row_weight=weight,
        calibration=calibration,
    )
    return outcome, context.candidates, frame


def _outer_checkpoint_paths(root: Path, fold: int) -> tuple[Path, Path]:
    return root / f"outer-fold-{fold}.parquet", root / f"outer-fold-{fold}.json"


def _save_outer_checkpoint(root: Path, value: _OuterComputation) -> None:
    frame_path, metadata_path = _outer_checkpoint_paths(root, value.fold)
    _write_parquet_atomic(frame_path, value.frame.reset_index(drop=True))
    _write_json_atomic(
        metadata_path,
        {
            "fold": value.fold,
            "selected": asdict(value.selected),
            "teacher_refit_iterations": value.teacher_refit_iterations,
            "student_refit_iterations": value.student_refit_iterations,
            "pseudo_weight": value.pseudo_weight,
            "calibration": asdict(value.calibration),
            "candidates": [asdict(candidate) for candidate in value.candidates],
        },
    )


def _load_outer_checkpoint(root: Path, fold: int) -> _OuterComputation | None:
    frame_path, metadata_path = _outer_checkpoint_paths(root, fold)
    if not frame_path.exists() and not metadata_path.exists():
        return None
    if not frame_path.exists() or not metadata_path.exists():
        raise ResidualEvaluationError(
            f"바깥 분할 {fold} 체크포인트가 일부만 있다."
        )
    raw = json.loads(metadata_path.read_text())
    if int(raw["fold"]) != fold:
        raise ResidualEvaluationError("바깥 분할 체크포인트 번호가 다르다.")
    return _OuterComputation(
        fold=fold,
        selected=SelectionDecision(**raw["selected"]),
        teacher_refit_iterations=int(raw["teacher_refit_iterations"]),
        student_refit_iterations=int(raw["student_refit_iterations"]),
        pseudo_weight=float(raw["pseudo_weight"]),
        calibration=ContrastCalibration(**raw["calibration"]),
        candidates=[CandidateScore(**candidate) for candidate in raw["candidates"]],
        frame=pd.read_parquet(frame_path),
    )


def _save_full_checkpoint(
    root: Path,
    outcome: FullSelectionOutcome,
    candidates: list[CandidateScore],
    frame: pd.DataFrame,
) -> None:
    _write_parquet_atomic(root / "full-test.parquet", frame)
    _write_json_atomic(
        root / "full-test.json",
        {
            "outcome": asdict(outcome),
            "candidates": [asdict(candidate) for candidate in candidates],
        },
    )


def _load_full_checkpoint(
    root: Path,
) -> tuple[FullSelectionOutcome, list[CandidateScore], pd.DataFrame] | None:
    frame_path = root / "full-test.parquet"
    metadata_path = root / "full-test.json"
    if not frame_path.exists() and not metadata_path.exists():
        return None
    if not frame_path.exists() or not metadata_path.exists():
        raise ResidualEvaluationError("전체 자료 체크포인트가 일부만 있다.")
    raw = json.loads(metadata_path.read_text())
    outcome_raw = raw["outcome"]
    outcome_raw["selected"] = SelectionDecision(**outcome_raw["selected"])
    outcome_raw["calibration"] = ContrastCalibration(**outcome_raw["calibration"])
    return (
        FullSelectionOutcome(**outcome_raw),
        [CandidateScore(**candidate) for candidate in raw["candidates"]],
        pd.read_parquet(frame_path),
    )


def evaluate_nested_selection(
    cfg: ResidualSelectionConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline: pd.Series,
    selection_baselines: dict[int, pd.Series],
    baseline_test: pd.Series,
    *,
    trainer: WeightedResidualTrainer | None = None,
    checkpoint_dir: Path | None = None,
    execution_identity: dict[str, str] | None = None,
) -> NestedSelectionEvaluation:
    active_trainer = trainer or LightGBMWeightedResidualTrainer()
    features = FeatureMatrixFactory(cfg.control, train, test)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        identity_path = checkpoint_dir / "execution-identity.json"
        identity = execution_identity or {}
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ResidualEvaluationError("기존 체크포인트의 실행 식별자가 현재 입력과 다르다.")
        else:
            _write_json_atomic(identity_path, identity)
    computations = []
    for fold in range(5):
        value = (
            _load_outer_checkpoint(checkpoint_dir, fold)
            if checkpoint_dir is not None
            else None
        )
        if value is None:
            value = _evaluate_outer(
                cfg,
                train,
                baseline,
                selection_baselines[fold],
                fold,
                features,
                active_trainer,
            )
            if checkpoint_dir is not None:
                _save_outer_checkpoint(checkpoint_dir, value)
        computations.append(value)
    combined = pd.concat([value.frame for value in computations], ignore_index=True)
    if combined[ID].duplicated().any() or len(combined) != len(train):
        raise ResidualEvaluationError("선택한 바깥 OOF가 훈련 id를 한 번씩 덮지 않는다.")
    order = pd.Index(train[ID], name=ID)
    combined = combined.set_index(ID).reindex(order).reset_index()
    if combined.isna().any().any():
        raise ResidualEvaluationError("선택한 바깥 OOF에 결측값이 있다.")
    combined["rank_control_pred"] = percentile_rank(combined["baseline_pred"])
    combined["pred"] = (
        combined["rank_control_pred"].to_numpy(dtype=np.float64)
        + combined["selected_beta"].to_numpy(dtype=np.float64)
        * combined["correction_signal"].to_numpy(dtype=np.float64)
    )
    combined["fold_rank_control_pred"] = np.nan
    combined["fold_rank_pred"] = np.nan
    for fold in range(5):
        mask = combined["fold"].to_numpy(dtype=np.int64) == fold
        fold_control = percentile_rank(combined.loc[mask, "baseline_pred"])
        combined.loc[mask, "fold_rank_control_pred"] = fold_control
        combined.loc[mask, "fold_rank_pred"] = (
            fold_control
            + combined.loc[mask, "selected_beta"].to_numpy(dtype=np.float64)
            * combined.loc[mask, "correction_signal"].to_numpy(dtype=np.float64)
        )
    y = train.set_index(ID)[TARGET].reindex(order).to_numpy(dtype=np.float64)
    source_auc = float(roc_auc_score(y, combined["baseline_pred"]))
    corrected_auc = float(roc_auc_score(y, combined["pred"]))
    gain = corrected_auc - source_auc
    fold_rank_control_auc = float(
        roc_auc_score(y, combined["fold_rank_control_pred"])
    )
    fold_rank_corrected_auc = float(roc_auc_score(y, combined["fold_rank_pred"]))
    outcomes: list[SelectionFoldOutcome] = []
    for value in computations:
        mask = combined["fold"].to_numpy(dtype=np.int64) == value.fold
        baseline_auc = float(roc_auc_score(y[mask], combined.loc[mask, "baseline_pred"]))
        corrected_fold_auc = float(roc_auc_score(y[mask], combined.loc[mask, "pred"]))
        outcomes.append(
            SelectionFoldOutcome(
                fold=value.fold,
                selected=value.selected,
                baseline_auc=baseline_auc,
                corrected_auc=corrected_fold_auc,
                auc_gain=corrected_fold_auc - baseline_auc,
                teacher_refit_iterations=value.teacher_refit_iterations,
                student_refit_iterations=value.student_refit_iterations,
                pseudo_row_weight=value.pseudo_weight,
                calibration=value.calibration,
            )
        )
    wins = sum(outcome.auc_gain > 0 for outcome in outcomes)
    full_checkpoint = (
        _load_full_checkpoint(checkpoint_dir) if checkpoint_dir is not None else None
    )
    if full_checkpoint is None:
        full, full_candidates, test_frame = _fit_full(
            cfg,
            train,
            baseline,
            baseline_test,
            features,
            active_trainer,
        )
        if checkpoint_dir is not None:
            _save_full_checkpoint(
                checkpoint_dir, full, full_candidates, test_frame
            )
    else:
        full, full_candidates, test_frame = full_checkpoint
    threshold = cfg.control.minimum_auc_gain
    return NestedSelectionEvaluation(
        source_baseline_auc=source_auc,
        corrected_auc=corrected_auc,
        auc_gain=gain,
        fold_rank_control_auc=fold_rank_control_auc,
        fold_rank_corrected_auc=fold_rank_corrected_auc,
        fold_rank_gain=fold_rank_corrected_auc - fold_rank_control_auc,
        fold_wins=wins,
        passed_official=gain >= threshold and wins >= cfg.control.minimum_fold_wins,
        passed_final_correction=0.0 < gain < threshold and wins == 5,
        folds=outcomes,
        candidate_tables={value.fold: value.candidates for value in computations},
        full_candidates=full_candidates,
        oof=combined,
        full=full,
        test=test_frame,
    )


def load_selection_inputs(
    cfg: ResidualSelectionConfig,
    baseline_oof_path: Path,
    selection_baseline_path: Path,
    baseline_test_path: Path,
    pool_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    dict[int, pd.Series],
    pd.Series,
    dict[str, str],
]:
    from .teacher_student_residual import load_inputs

    train, test, baseline, hashes = load_inputs(cfg.control, baseline_oof_path)
    extra_hashes = {
        "selection_baseline": file_sha256(selection_baseline_path),
        "baseline_test": file_sha256(baseline_test_path),
        "pool": file_sha256(pool_path),
    }
    expected = {
        "selection_baseline": cfg.inputs.selection_baseline_sha256,
        "baseline_test": cfg.inputs.baseline_test_sha256,
        "pool": cfg.inputs.pool_sha256,
    }
    for name, actual in extra_hashes.items():
        if actual != expected[name]:
            raise ResidualEvaluationError(
                f"{name} SHA-256이 설정과 다르다: {actual} != {expected[name]}"
            )
    selection_frame = pd.read_parquet(selection_baseline_path)
    selection_baselines = validate_selection_baseline(
        selection_frame, train, baseline
    )
    test_frame = pd.read_csv(baseline_test_path)
    if list(test_frame.columns) != [ID, TARGET]:
        raise ResidualEvaluationError("기준 시험 예측은 id와 target 열만 포함해야 한다.")
    if test_frame[ID].duplicated().any() or len(test_frame) != len(test):
        raise ResidualEvaluationError("기준 시험 예측 id가 시험 자료와 다르다.")
    test_index = pd.Index(test[ID], name=ID)
    baseline_test = test_frame.set_index(ID)[TARGET].reindex(test_index)
    if baseline_test.isna().any() or not np.isfinite(
        baseline_test.to_numpy(dtype=np.float64)
    ).all():
        raise ResidualEvaluationError("기준 시험 예측이 id와 맞지 않거나 유한하지 않다.")
    return (
        train,
        test,
        baseline,
        selection_baselines,
        baseline_test.astype(np.float64),
        {**hashes, **extra_hashes},
    )


def _candidate_frame(evaluation: NestedSelectionEvaluation) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, candidates in evaluation.candidate_tables.items():
        rows.extend(
            {"scope": "outer", "outer_fold": fold, **asdict(candidate)}
            for candidate in candidates
        )
    rows.extend(
        {"scope": "full", "outer_fold": -1, **asdict(candidate)}
        for candidate in evaluation.full_candidates
    )
    return pd.DataFrame(rows)


def _result_payload(
    cfg: ResidualSelectionConfig,
    evaluation: NestedSelectionEvaluation,
    input_hashes: dict[str, str],
    git_state: dict[str, str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": cfg.control.experiment.name,
        "baseline": asdict(cfg.control.baseline),
        "inputs": asdict(cfg.inputs),
        "search": asdict(cfg.search),
        "judgment": {
            "minimum_auc_gain": cfg.control.minimum_auc_gain,
            "minimum_fold_wins": cfg.control.minimum_fold_wins,
            "passed_official": evaluation.passed_official,
            "passed_final_correction": evaluation.passed_final_correction,
        },
        "metrics": {
            "source_baseline_auc": evaluation.source_baseline_auc,
            "corrected_auc": evaluation.corrected_auc,
            "auc_gain": evaluation.auc_gain,
            "fold_rank_control_auc": evaluation.fold_rank_control_auc,
            "fold_rank_corrected_auc": evaluation.fold_rank_corrected_auc,
            "fold_rank_gain": evaluation.fold_rank_gain,
            "fold_wins": evaluation.fold_wins,
        },
        "folds": [asdict(outcome) for outcome in evaluation.folds],
        "full": asdict(evaluation.full),
        "input_sha256": input_hashes,
        "git": git_state,
        "elapsed_seconds": elapsed_seconds,
    }


def record_selection_evaluation(
    cfg: ResidualSelectionConfig,
    evaluation: NestedSelectionEvaluation,
    input_hashes: dict[str, str],
    payload: dict[str, Any],
    out_dir: Path,
    *,
    tracking_uri: str = TRACKING_URI,
) -> str:
    from .tracking import git_state, mlflow_client

    state = git_state()
    if state["git_dirty"] != "False":
        raise ResidualEvaluationError("dirty 작업 폴더의 결과는 실행 저장소에 기록하지 않는다.")
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    oof_path = final_dir / "oof.parquet"
    seed_oof_path = final_dir / f"oof_seed_{cfg.control.seed}.parquet"
    test_path = final_dir / "test_prediction.parquet"
    submission_path = final_dir / "submission.csv"
    candidates_path = final_dir / "candidate_tables.csv"
    result_path = final_dir / "result.json"
    _write_parquet_atomic(oof_path, evaluation.oof)
    _write_parquet_atomic(
        seed_oof_path, evaluation.oof[[ID, "fold", "pred"]]
    )
    _write_parquet_atomic(test_path, evaluation.test)
    evaluation.test[[ID, "pred"]].rename(columns={"pred": TARGET}).to_csv(
        submission_path, index=False
    )
    _candidate_frame(evaluation).to_csv(candidates_path, index=False)
    _write_json_atomic(result_path, payload)

    client, experiment_id = mlflow_client(tracking_uri)
    run_name = cfg.control.experiment.name
    run_id = client.create_run(experiment_id, run_name=run_name).info.run_id
    try:
        raw_columns = list(
            pd.read_csv(cfg.control.experiment.data.train, nrows=0).columns
        )
        feature_names = [
            column
            for _, _, columns, _ in FeaturePlan.from_config(
                cfg.control.experiment.features
            ).describe(raw_columns)
            for column in columns
        ]
        params = {
            "experiment": run_name,
            "seeds": str(cfg.control.seed),
            "stage": "confirm",
            "model.kind": "teacher_student_nested_selection",
            "features": ",".join(feature_names),
            "baseline.run_id": cfg.control.baseline.run_id,
            "baseline.strategy": cfg.control.baseline.strategy,
            "baseline.pool_members": str(cfg.control.baseline.pool_members),
            "baseline.auc_oof": repr(cfg.control.baseline.auc_oof),
            "search.rhos": ",".join(map(str, cfg.search.rhos)),
            "search.betas": ",".join(map(str, cfg.search.betas)),
            "full.selected_rho": repr(evaluation.full.selected.rho),
            "full.selected_beta": repr(evaluation.full.selected.beta),
            "refit_iteration_multiplier": repr(
                cfg.control.refit_iteration_multiplier
            ),
        }
        for key, value in params.items():
            client.log_param(run_id, key, value)
        client.log_metric(run_id, "auc_oof", evaluation.corrected_auc)
        client.log_metric(
            run_id, f"auc_oof_seed_{cfg.control.seed}", evaluation.corrected_auc
        )
        client.log_metric(run_id, "auc_source_baseline", evaluation.source_baseline_auc)
        client.log_metric(run_id, "delta_vs_source_baseline", evaluation.auc_gain)
        client.log_metric(run_id, "auc_fold_rank_control", evaluation.fold_rank_control_auc)
        client.log_metric(
            run_id, "auc_fold_rank_corrected", evaluation.fold_rank_corrected_auc
        )
        client.log_metric(run_id, "delta_fold_rank", evaluation.fold_rank_gain)
        client.log_metric(run_id, "fold_wins", evaluation.fold_wins)
        for outcome in evaluation.folds:
            client.log_metric(run_id, f"auc_fold_{outcome.fold}", outcome.corrected_auc)
            client.log_metric(run_id, f"delta_fold_{outcome.fold}", outcome.auc_gain)
            client.log_metric(run_id, f"rho_fold_{outcome.fold}", outcome.selected.rho)
            client.log_metric(run_id, f"beta_fold_{outcome.fold}", outcome.selected.beta)
        for key, value in state.items():
            client.set_tag(run_id, key, value)
        for name, digest in input_hashes.items():
            client.set_tag(run_id, f"sha256.{name}", digest)
        client.set_tag(run_id, "sha256.oof_prediction", file_sha256(oof_path))
        client.set_tag(run_id, "sha256.submission", file_sha256(submission_path))
        client.set_tag(run_id, "source.kind", "derived_ensemble_postprocess")
        client.set_tag(run_id, "source.issue", "316")
        for artifact in (
            cfg.control.experiment.source_path,
            oof_path,
            seed_oof_path,
            test_path,
            submission_path,
            candidates_path,
            result_path,
        ):
            client.log_artifact(run_id, str(artifact))
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    (out_dir / "run_id.txt").write_text(run_id + "\n")
    return run_id


def prepare_baseline_command(
    baseline_oof_path: Path,
    strategy: str,
    output_path: Path,
    tracking_uri: str,
) -> None:
    if strategy not in COMBINER_REGISTRY:
        raise ResidualEvaluationError(f"알 수 없는 기준 풀 결합 전략: {strategy}")
    pool = Pool.load()
    members = [(member.config, member.run_id) for member in pool.members]
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    store = MlflowRunStore(tracking_uri=tracking_uri)
    matrix = member_matrix(members, store, fold_of.index)
    frame = pd.read_parquet(baseline_oof_path)
    prediction_column = "prediction" if "prediction" in frame.columns else "pred"
    if ID not in frame or prediction_column not in frame:
        raise ResidualEvaluationError("기준 OOF에 id와 예측 열이 없다.")
    outer = frame.set_index(ID)[prediction_column].reindex(fold_of.index)
    recomputed = evaluate_nested(
        COMBINER_REGISTRY[strategy], matrix, fold_of, y
    ).prediction
    if not np.array_equal(
        outer.to_numpy(dtype=np.float64),
        recomputed.to_numpy(dtype=np.float64),
    ):
        raise ResidualEvaluationError(
            "기준 OOF가 현재 풀과 지정 결합 전략의 재계산값과 다르다."
        )
    prepared = prepare_selection_baseline(
        COMBINER_REGISTRY[strategy], matrix, fold_of, y, outer
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(output_path, prepared)
    metadata = {
        "schema_version": SELECTION_BASELINE_SCHEMA_VERSION,
        "strategy": strategy,
        "pool_members": len(members),
        "sha256_pool": file_sha256(Path("artifacts/pool.yaml")),
        "sha256_baseline_oof": file_sha256(baseline_oof_path),
        "sha256_output": file_sha256(output_path),
    }
    _write_json_atomic(output_path.with_suffix(".json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def run_command(
    config_path: Path,
    baseline_oof_path: Path,
    selection_baseline_path: Path,
    baseline_test_path: Path,
    pool_path: Path,
    out_dir: Path,
    tracking_uri: str,
) -> str:
    from .tracking import git_state

    state = git_state()
    if state["git_dirty"] != "False":
        raise ResidualEvaluationError("실행은 커밋된 clean 작업 폴더에서만 허용한다.")
    cfg = load_selection_config(config_path)
    train, test, baseline, selection_baselines, baseline_test, hashes = (
        load_selection_inputs(
            cfg,
            baseline_oof_path,
            selection_baseline_path,
            baseline_test_path,
            pool_path,
        )
    )
    identity = _execution_identity(cfg.control, hashes, state)
    started = time.monotonic()
    evaluation = evaluate_nested_selection(
        cfg,
        train,
        test,
        baseline,
        selection_baselines,
        baseline_test,
        checkpoint_dir=out_dir / "checkpoints",
        execution_identity=identity,
    )
    elapsed = time.monotonic() - started
    payload = _result_payload(cfg, evaluation, hashes, state, elapsed)
    run_id = record_selection_evaluation(
        cfg,
        evaluation,
        hashes,
        payload,
        out_dir,
        tracking_uri=tracking_uri,
    )
    print(f"현재 풀 OOF AUC: {evaluation.source_baseline_auc:.16f}")
    print(f"선택 보정 OOF AUC: {evaluation.corrected_auc:.16f}")
    print(f"현재 풀 대비: {evaluation.auc_gain:+.16f}")
    print(
        f"분할별 순위 진단 차이: {evaluation.fold_rank_gain:+.16f} "
        "(채택 근거에서 제외)"
    )
    print(f"바깥 분할 승리: {evaluation.fold_wins}/5")
    print(
        "판정: "
        + (
            "공식 풀 채택"
            if evaluation.passed_official
            else "최종 보정 채택"
            if evaluation.passed_final_correction
            else "기각"
        )
    )
    print(
        f"전체 자료 선택: rho={evaluation.full.selected.rho}, "
        f"beta={evaluation.full.selected.beta}"
    )
    print(f"run_id: {run_id}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-baseline")
    prepare.add_argument("--baseline-oof", type=Path, required=True)
    prepare.add_argument("--strategy", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--tracking-uri", default=TRACKING_URI)
    run = commands.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--baseline-oof", type=Path, required=True)
    run.add_argument("--selection-baseline", type=Path, required=True)
    run.add_argument("--baseline-test", type=Path, required=True)
    run.add_argument("--pool", type=Path, default=Path("artifacts/pool.yaml"))
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--tracking-uri", default=TRACKING_URI)
    args = parser.parse_args()
    try:
        if args.command == "prepare-baseline":
            prepare_baseline_command(
                args.baseline_oof,
                args.strategy,
                args.out,
                args.tracking_uri,
            )
        else:
            run_command(
                args.config,
                args.baseline_oof,
                args.selection_baseline,
                args.baseline_test,
                args.pool,
                args.out_dir,
                args.tracking_uri,
            )
    except ResidualEvaluationError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
