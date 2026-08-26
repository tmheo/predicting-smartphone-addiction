"""여러 학습 시점 CV의 관측 및 복구 경계를 짧게 완주하는 회귀 검사."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from test_model import SEED, fake_experiment_config, toy_train_test

from pipeline import model as model_mod
from pipeline import training_state_cv
from pipeline.config import ModelConfig, TrainingStateConfig
from pipeline.fold_observability import FoldExecutionRecorder
from pipeline.plan import FeaturePlan
from pipeline.training_length import (
    RawTrainingLengthSelection,
    ZERO_BASED_POSITION,
    TrainingLengthContract,
)
from pipeline.training_state_recovery import (
    TrainingStateCandidate,
    TrainingStateRecovery,
)


N_FOLDS = 3
STATE = TrainingStateConfig(
    trajectory="test-24epoch",
    candidates=(6, 8, 12),
    selected=6,
    schedule_horizon_epochs=24,
    trajectory_end_epochs=24,
    state_kind="ema",
    selection_rule="precommitted",
)


class NoSampleProbe:
    metadata = {"cpu_scope": "test_scope", "gpu_expected": False}

    def sample(self, observed_ns: int) -> dict[str, object]:
        raise AssertionError("자동 자원 표본을 시작하지 않는다.")


class PersistingRecorder:
    def __init__(self, root) -> None:
        self.fold_recorder = FoldExecutionRecorder(
            root,
            {"seeds": [SEED], "source": "local_measured"},
            resource_probe=NoSampleProbe(),
            start_sampler=False,
        )
        self.fold_recorder.start()
        self.fold_recorder.configure_run_shape(
            seed_total=1,
            fold_total=N_FOLDS,
            provider_total=0,
        )
        self.completed: list[tuple[int, int, float]] = []

    def stage(self, name: str) -> None:
        pass

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None:
        self.completed.append((seed_index, fold_index, auc))

    def record_timing(self, event: dict[str, object]) -> None:
        self.fold_recorder.record_timing(event)


class FakeTrainingStateAdapter:
    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    def fit_predict_training_states(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        X_test: pd.DataFrame,
        state: TrainingStateConfig,
    ) -> model_mod.FoldTrainingStateTrajectory:
        self.calls.append(len(X_va))
        points = []
        for completed_epochs in state.candidates:
            declaration = TrainingLengthContract(
                "lookup_transformer",
                "best_epoch",
                ZERO_BASED_POSITION,
            ).declare(
                [
                    RawTrainingLengthSelection(
                        raw_path="test.best_epoch",
                        raw_value=completed_epochs - 1,
                    )
                ]
            )
            points.append(
                model_mod.FoldTrainingStatePoint(
                    completed_epochs=completed_epochs,
                    validation_prediction=np.linspace(
                        0.1 + completed_epochs * 1e-4,
                        0.9 + completed_epochs * 1e-4,
                        len(X_va),
                        dtype="float64",
                    ),
                    test_prediction=np.full(
                        len(X_test), completed_epochs / 100, dtype="float64"
                    ),
                    importance=pd.DataFrame(
                        {
                            "feature": list(X_tr.columns),
                            "gain": np.full(
                                len(X_tr.columns), completed_epochs, dtype="float64"
                            ),
                        }
                    ),
                    training_diagnostics={"completed_epochs": completed_epochs},
                    training_length_declaration=declaration,
                )
            )
        return model_mod.FoldTrainingStateTrajectory(
            schedule_horizon_epochs=state.schedule_horizon_epochs,
            trajectory_end_epochs=state.trajectory_end_epochs,
            points=tuple(points),
        )


def test_training_state_cv_persists_observability_and_reuses_complete_folds(
    monkeypatch,
    tmp_path,
):
    cfg = replace(
        fake_experiment_config(),
        model=ModelConfig(kind="lookup_transformer", params={}, fit={}),
        training_state=STATE,
    )
    plan = FeaturePlan.from_config(cfg.features)
    train, test = toy_train_test()
    train, test = plan.apply_dataset_wide(train, test)
    train["fold"] = np.arange(len(train)) % N_FOLDS
    candidates = [
        TrainingStateCandidate(
            config_name=f"candidate-{completed_epochs}",
            config_path=f"configs/candidate-{completed_epochs}.yaml",
            config_sha256=character * 64,
            completed_epochs=completed_epochs,
            schedule_horizon_epochs=24,
        )
        for completed_epochs, character in zip(STATE.candidates, "bcd", strict=True)
    ]
    recovery = TrainingStateRecovery.for_run(
        tmp_path / "recovery",
        candidates,
        {"train": "1" * 64, "test": "2" * 64, "folds": "3" * 64},
        git_commit="test-commit",
        trajectory_identity_sha256="a" * 64,
        stage="confirm",
        seeds=[SEED],
        model_kind="lookup_transformer",
        trajectory_end_epochs=24,
        model_dependencies={"python": "test"},
    )
    calls: list[int] = []
    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY,
        "lookup_transformer",
        lambda params, fit, seed: FakeTrainingStateAdapter(calls),
    )
    first_recorder = PersistingRecorder(tmp_path / "observability-first")

    first = training_state_cv.run_training_state_seed(
        cfg,
        plan,
        train,
        test,
        SEED,
        recorder=first_recorder,
        recovery=recovery,
    )
    first_observability = first_recorder.fold_recorder.finalize()

    assert set(first) == set(STATE.candidates)
    assert calls == [20, 20, 20]
    assert len(first_recorder.completed) == N_FOLDS
    assert first_observability.summary["timing"]["operations"][
        "training_state.trajectory_fit"
    ]["completed_count"] == N_FOLDS
    assert all(len(result.recovery_evidence) == N_FOLDS for result in first.values())

    second_recorder = PersistingRecorder(tmp_path / "observability-second")
    second = training_state_cv.run_training_state_seed(
        cfg,
        plan,
        train,
        test,
        SEED,
        recorder=second_recorder,
        recovery=recovery,
    )
    second_observability = second_recorder.fold_recorder.finalize()

    assert calls == [20, 20, 20]
    assert second_observability.summary["timing"]["operations"][
        "training_state.trajectory_fit"
    ]["event_count"] == 0
    for completed_epochs in STATE.candidates:
        pd.testing.assert_frame_equal(
            first[completed_epochs].oof,
            second[completed_epochs].oof,
            check_exact=True,
        )
        assert {
            entry["reused"] for entry in second[completed_epochs].recovery_evidence
        } == {True}
