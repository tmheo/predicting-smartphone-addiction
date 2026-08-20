"""교사-학생 순위 잔차 보정의 분할 경계, 변환과 기록 계약."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.bundle import export_bundle
from pipeline.data import ID, TARGET
from pipeline.teacher_student_residual import (
    FitResult,
    NestedResidualEvaluation,
    evaluate_nested_residual,
    load_residual_config,
    percentile_rank,
    record_evaluation,
    signed_square_contrast,
)


CONFIG_PATH = Path("configs/exp137_teacher_student_rank_residual.yaml")


def make_frames(rows: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(19)
    age = np.linspace(18, 65, rows)
    daily = rng.uniform(1.0, 12.0, rows)
    social = rng.uniform(0.0, 5.0, rows)
    gaming = rng.uniform(0.0, 4.0, rows)
    work = rng.uniform(0.0, 6.0, rows)
    sleep = rng.uniform(4.0, 10.0, rows)
    notifications = rng.integers(0, 150, rows).astype(float)
    app_opens = rng.integers(1, 80, rows).astype(float)
    weekend = rng.uniform(1.0, 14.0, rows)
    gender = np.resize(np.array(["Female", "Male"]), rows)
    stress = np.resize(np.array(["Low", "Medium", "High"]), rows)
    academic = np.resize(np.array(["No", "Yes"]), rows)
    target = (age + 0.08 * notifications + 0.7 * gaming > 45).astype(int)
    train = pd.DataFrame(
        {
            ID: np.arange(1, rows + 1),
            "age": age,
            "daily_screen_time_hours": daily,
            "social_media_hours": social,
            "gaming_hours": gaming,
            "work_study_hours": work,
            "sleep_hours": sleep,
            "notifications_per_day": notifications,
            "app_opens_per_day": app_opens,
            "weekend_screen_time": weekend,
            "gender": gender,
            "stress_level": stress,
            "academic_work_impact": academic,
            TARGET: target,
            "fold": np.arange(rows) % 5,
        }
    )
    train.loc[::11, "social_media_hours"] = np.nan
    train.loc[::13, "sleep_hours"] = np.nan
    test = train.drop(columns=[TARGET, "fold"]).iloc[:20].copy()
    baseline = pd.Series(
        age + 0.08 * notifications + rng.normal(0.0, 2.0, rows),
        index=train.index,
        dtype=np.float64,
    )
    return train, test, baseline


class RecordingTrainer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, set[int], set[int]]] = []

    @staticmethod
    def _predict(role: str, X: pd.DataFrame) -> np.ndarray:
        age = X["age"].to_numpy(dtype=np.float64)
        notifications = X["notifications_per_day"].to_numpy(dtype=np.float64)
        gaming = X["gaming_hours"].to_numpy(dtype=np.float64)
        if role == "teacher":
            return age + 0.08 * notifications + 0.5 * np.square(gaming)
        return age + 0.02 * notifications + 0.1 * gaming

    def fit_with_validation(
        self, role, X_train, y_train, X_valid, y_valid, spec, seed
    ) -> FitResult:
        assert set(X_train.index).isdisjoint(X_valid.index)
        assert X_train.index.equals(y_train.index)
        assert X_valid.index.equals(y_valid.index)
        self.calls.append((role, "validation", set(X_train.index), set(X_valid.index)))
        return FitResult(self._predict(role, X_valid), best_iteration=20)

    def fit_full(
        self, role, X_train, y_train, X_predict, spec, seed, iterations
    ) -> np.ndarray:
        assert set(X_train.index).isdisjoint(X_predict.index)
        assert X_train.index.equals(y_train.index)
        assert iterations == 25
        self.calls.append((role, "full", set(X_train.index), set(X_predict.index)))
        return self._predict(role, X_predict)


def test_config_fixes_current_pool29_baseline_and_recipe():
    cfg = load_residual_config(CONFIG_PATH)
    assert cfg.baseline.run_id == "33bde7e429d8407cb8b46b4737450265"
    assert cfg.baseline.auc_oof == pytest.approx(0.9696999775221133)
    assert cfg.baseline.pool_members == 29
    assert cfg.correction_weight == 0.10
    assert cfg.minimum_auc_gain == pytest.approx(0.00002)
    assert cfg.minimum_fold_wins == 3


def test_percentile_rank_uses_average_ties():
    np.testing.assert_allclose(percentile_rank([1.0, 2.0, 2.0, 4.0]), [0.25, 0.625, 0.625, 1.0])


def test_signed_square_contrast_matches_reference_scale_before_squaring():
    signal, calibration = signed_square_contrast(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([-1.0, -0.5, 0.5, 1.0]),
        clip_to_reference_range=True,
    )
    np.testing.assert_allclose(signal, [-4.0, -1.0, 1.0, 4.0])
    assert calibration.scale == pytest.approx(2.0)
    assert calibration.clip_lower == -2.0
    assert calibration.clip_upper == 2.0


def test_nested_evaluation_keeps_every_outer_row_out_of_all_fits():
    cfg = load_residual_config(CONFIG_PATH)
    train, test, baseline = make_frames()
    trainer = RecordingTrainer()

    evaluation = evaluate_nested_residual(
        cfg, train, test, baseline, trainer=trainer
    )

    assert len(trainer.calls) == 50
    assert len(evaluation.oof) == len(train)
    assert evaluation.oof[ID].is_unique
    assert np.isfinite(evaluation.oof["pred"]).all()
    assert evaluation.fold_wins == sum(fold.auc_gain > 0 for fold in evaluation.folds)
    for role in ("teacher", "student"):
        full_calls = [call for call in trainer.calls if call[0] == role and call[1] == "full"]
        assert len(full_calls) == 5
        assert sorted(len(predict_ids) for _, _, _, predict_ids in full_calls) == [20] * 5


def test_completed_outer_fold_checkpoints_are_reused(tmp_path):
    cfg = load_residual_config(CONFIG_PATH)
    train, test, baseline = make_frames()
    first_trainer = RecordingTrainer()
    identity = {"git_commit": "a" * 40, "sha256.config": "b" * 64}
    first = evaluate_nested_residual(
        cfg,
        train,
        test,
        baseline,
        trainer=first_trainer,
        checkpoint_dir=tmp_path,
        execution_identity=identity,
    )

    class FailingTrainer(RecordingTrainer):
        def fit_with_validation(self, *args, **kwargs):
            raise AssertionError("완료된 분할을 다시 학습했다.")

        def fit_full(self, *args, **kwargs):
            raise AssertionError("완료된 분할을 다시 학습했다.")

    second = evaluate_nested_residual(
        cfg,
        train,
        test,
        baseline,
        trainer=FailingTrainer(),
        checkpoint_dir=tmp_path,
        execution_identity=identity,
    )

    pd.testing.assert_frame_equal(first.oof, second.oof)
    assert first.corrected_auc == second.corrected_auc


def test_recorded_result_can_be_exported_as_verified_run_bundle(tmp_path, monkeypatch):
    cfg = load_residual_config(CONFIG_PATH)
    train, _, baseline = make_frames()
    oof = pd.DataFrame(
        {
            ID: train[ID],
            "fold": train["fold"],
            "baseline_pred": baseline,
            "rank_control_pred": percentile_rank(baseline),
            "teacher_pred": baseline + 0.1,
            "student_pred": baseline,
            "contrast": np.linspace(-0.1, 0.1, len(train)),
            "correction_signal": np.linspace(-0.01, 0.01, len(train)),
            "pred": baseline + np.linspace(-0.01, 0.01, len(train)),
        }
    )
    corrected_auc = 0.8
    evaluation = NestedResidualEvaluation(
        source_baseline_auc=0.79,
        rank_control_auc=0.795,
        corrected_auc=corrected_auc,
        source_auc_gain=0.01,
        residual_auc_gain=0.005,
        fold_wins=5,
        passed=True,
        folds=[],
        oof=oof,
    )
    state = {"git_commit": "a" * 40, "git_dirty": "False"}
    monkeypatch.setattr("pipeline.tracking.git_state", lambda: state)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    payload = {
        "schema_version": 1,
        "metrics": {"corrected_auc": corrected_auc},
    }
    input_hashes = {
        "train": "1" * 64,
        "test": "2" * 64,
        "folds": "3" * 64,
        "baseline_oof": "4" * 64,
        "config": "5" * 64,
    }

    run_id = record_evaluation(
        cfg,
        evaluation,
        input_hashes,
        payload,
        tmp_path / "result",
        tracking_uri=tracking_uri,
    )
    out = export_bundle(
        run_id, tmp_path / "result.bundle.zip", tracking_uri=tracking_uri
    )

    assert out.exists()
    from mlflow.tracking import MlflowClient

    run = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["baseline.run_id"] == cfg.baseline.run_id
    assert run.data.metrics["auc_oof"] == corrected_auc
