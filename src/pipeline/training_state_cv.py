"""한 물리 학습 궤적에서 여러 고정 시점 후보를 만드는 전용 CV 경로."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import model as model_mod
from . import training_length as training_length_mod
from .config import ExperimentConfig
from .cv import CVResult, RunRecorder, score_predictions
from .data import ID, TARGET
from .fold_observability import timed_operation
from .plan import FeaturePlan

if TYPE_CHECKING:
    from .training_state_recovery import TrainingStateRecovery


@dataclass
class _CandidateAccumulator:
    oof_prediction: np.ndarray
    test_prediction: np.ndarray
    importances: list[pd.DataFrame] = field(default_factory=list)
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    recovery_evidence: list[dict[str, object]] = field(default_factory=list)


def run_training_state_seed(
    cfg: ExperimentConfig,
    plan: FeaturePlan,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    *,
    recorder: RunRecorder | None = None,
    recovery: TrainingStateRecovery | None = None,
) -> dict[int, CVResult]:
    """시드 하나를 한 번 학습하고 후보 시점별 평범한 CVResult를 돌려준다."""
    state = cfg.training_state
    if state is None:
        raise ValueError("여러 학습 시점 CV에는 training_state 설정이 필요하다.")
    if cfg.initial_score is not None:
        raise ValueError("여러 학습 시점 CV는 initial_score를 지원하지 않는다.")
    if plan.fold_fit_providers():
        raise ValueError("여러 학습 시점 CV는 fold-fit 특성 제공자를 아직 지원하지 않는다.")
    if recorder is not None:
        recorder.stage("feature_build")
    y = train[TARGET]
    n_folds = int(train["fold"].max()) + 1
    feature_names = plan.all_columns()
    X = plan.build_matrix(train, seed)
    X_test = plan.build_matrix(test, seed)
    if list(X.columns) != feature_names or list(X_test.columns) != feature_names:
        raise AssertionError("여러 시점 CV 학습 열이 피처 계획 선언과 다르다.")

    accumulators = {
        completed_epochs: _CandidateAccumulator(
            oof_prediction=np.zeros(len(train), dtype="float64"),
            test_prediction=np.zeros(len(test), dtype="float64"),
        )
        for completed_epochs in state.candidates
    }
    recovery_candidate_ids = (
        dict(zip(state.candidates, recovery.candidate_ids))
        if recovery is not None
        else {}
    )
    if recorder is not None:
        recorder.stage("training")

    for fold in range(n_folds):
        va_idx = train.index[train["fold"] == fold]
        tr_idx = train.index[train["fold"] != fold]
        checkpoint = (
            recovery.load(
                seed,
                fold,
                validation_ids=train.loc[va_idx, ID],
                validation_labels=y.loc[va_idx],
                test_ids=test[ID],
                feature_names=feature_names,
            )
            if recovery is not None
            else None
        )
        if checkpoint is None:
            adapter = model_mod.create(cfg.model, seed)
            model_mod.set_dataset_reference(adapter, X, X_test)
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="training_state.trajectory_fit",
                actor_kind="model",
                actor_name=cfg.model.kind,
            ):
                trajectory = model_mod.fit_predict_training_states(
                    adapter,
                    cfg.model.kind,
                    X.loc[tr_idx],
                    y.loc[tr_idx],
                    X.loc[va_idx],
                    y.loc[va_idx],
                    X_test,
                    state,
                )
            fold_points = {
                point.completed_epochs: _materialize_fold_point(
                    point,
                    seed=seed,
                    fold=fold,
                    validation_ids=train.loc[va_idx, ID],
                    validation_labels=y.loc[va_idx],
                    test_ids=test[ID],
                    schedule_horizon_epochs=state.schedule_horizon_epochs,
                    trajectory_end_epochs=state.trajectory_end_epochs,
                )
                for point in trajectory.points
            }
            if recovery is not None:
                checkpoint = recovery.save(
                    seed,
                    fold,
                    snapshots={
                        recovery_candidate_ids[completed_epochs]: point
                        for completed_epochs, point in fold_points.items()
                    },
                    validation_ids=train.loc[va_idx, ID],
                    validation_labels=y.loc[va_idx],
                    test_ids=test[ID],
                    feature_names=feature_names,
                )
                fold_points = {
                    completed_epochs: checkpoint.snapshots[candidate_id]
                    for completed_epochs, candidate_id in recovery_candidate_ids.items()
                }
                reused = False
            else:
                reused = False
        else:
            fold_points = {
                completed_epochs: checkpoint.snapshots[candidate_id]
                for completed_epochs, candidate_id in recovery_candidate_ids.items()
            }
            reused = True

        if tuple(sorted(fold_points)) != state.candidates:
            raise ValueError(
                f"seed={seed} fold={fold} 후보 시점 집합이 설정과 다르다: "
                f"{tuple(sorted(fold_points))} != {state.candidates}"
            )
        for completed_epochs in state.candidates:
            point = fold_points[completed_epochs]
            accumulator = accumulators[completed_epochs]
            accumulator.oof_prediction[va_idx] = point.validation_predictions[
                "pred"
            ].to_numpy()
            accumulator.test_prediction += (
                point.test_predictions["pred"].to_numpy() / n_folds
            )
            accumulator.importances.append(point.importance)
            if point.model_training_diagnostics is not None:
                accumulator.diagnostics.append(point.model_training_diagnostics)
            if checkpoint is not None:
                accumulator.recovery_evidence.append(
                    checkpoint.evidence(reused=reused)
                )
        if recorder is not None:
            last = fold_points[state.candidates[-1]]
            recorder.fold_completed(
                cfg.seeds.index(seed),
                fold,
                float(
                    roc_auc_score(
                        y.loc[va_idx], last.validation_predictions["pred"]
                    )
                ),
            )

    results: dict[int, CVResult] = {}
    for completed_epochs, accumulator in accumulators.items():
        results[completed_epochs] = CVResult(
            oof=pd.DataFrame(
                {
                    ID: train[ID],
                    "fold": train["fold"],
                    "pred": accumulator.oof_prediction,
                }
            ),
            test_pred=pd.DataFrame(
                {ID: test[ID], "pred": accumulator.test_prediction}
            ),
            fold_aucs=score_predictions(
                y, train["fold"], accumulator.oof_prediction
            ),
            feature_names=feature_names,
            importance=pd.concat(accumulator.importances, ignore_index=True),
            recovery_evidence=accumulator.recovery_evidence,
            model_training_diagnostics=accumulator.diagnostics,
            recovery_evidence_name="training_state_recovery.json",
        )
    return results


def _materialize_fold_point(
    point: model_mod.FoldTrainingStatePoint,
    *,
    seed: int,
    fold: int,
    validation_ids: pd.Series,
    validation_labels: pd.Series,
    test_ids: pd.Series,
    schedule_horizon_epochs: int,
    trajectory_end_epochs: int,
):
    """모델 시점 결과에 fold 좌표와 학습 길이 근거를 붙인다."""
    from .training_state_recovery import TrainingStateSnapshot

    evidence = training_length_mod.observe_declaration(
        point.training_length_declaration,
        seed=seed,
        outer_fold=fold,
    ).to_json()
    observed = {
        item["observed_training_length"] for item in evidence["observations"]
    }
    if observed != {point.completed_epochs}:
        raise ValueError(
            f"seed={seed} fold={fold} 학습 길이 근거가 후보 시점과 다르다: {observed}"
        )
    diagnostics = {
        "model_kind": point.training_length_declaration.model_family,
        "seed": seed,
        "fold": fold,
        "details": point.training_diagnostics,
        "training_state": {
            "completed_epochs": point.completed_epochs,
            "schedule_horizon_epochs": schedule_horizon_epochs,
            "trajectory_end_epochs": trajectory_end_epochs,
            "selection_rule": "precommitted",
            "state_kind": "ema",
        },
        "training_length_evidence": evidence,
    }
    return TrainingStateSnapshot(
        validation_predictions=pd.DataFrame(
            {
                ID: validation_ids,
                "fold": fold,
                "pred": point.validation_prediction,
            }
        ),
        test_predictions=pd.DataFrame(
            {ID: test_ids, "pred": point.test_prediction}
        ),
        importance=point.importance.assign(fold=fold, seed=seed),
        model_training_diagnostics=diagnostics,
    )
