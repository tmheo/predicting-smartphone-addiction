from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import pytest

from pipeline.bundle import export_bundle
from pipeline.data import ID, TARGET
from pipeline.teacher_student_residual import FitResult, load_residual_config
from pipeline.teacher_student_residual_selection import (
    CandidateScore,
    ResidualSelectionConfig,
    SearchSpace,
    SelectionInputSpec,
    _evaluate_outer,
    candidate_pairs,
    evaluate_nested_selection,
    load_selection_config,
    prepare_selection_baseline,
    pseudo_row_weight,
    record_selection_evaluation,
    select_candidate,
)


CONFIG_PATH = "configs/exp137_teacher_student_rank_residual.yaml"
SELECTION_CONFIG_PATH = "configs/exp138_teacher_student_nested_selection.yaml"


def selection_config() -> ResidualSelectionConfig:
    return ResidualSelectionConfig(
        control=load_residual_config(CONFIG_PATH),
        inputs=SelectionInputSpec(
            pool_sha256="1" * 64,
            selection_baseline_sha256="2" * 64,
            baseline_test_sha256="3" * 64,
        ),
        search=SearchSpace(rhos=(0.0, 0.125), betas=(0.0, 0.1)),
    )


def make_frames(rows: int = 100):
    rng = np.random.default_rng(316)
    age = np.linspace(13, 29, rows)
    daily = rng.uniform(2.0, 12.0, rows)
    social = rng.uniform(0.2, 5.0, rows)
    gaming = rng.uniform(0.0, 4.0, rows)
    work = rng.uniform(0.5, 6.0, rows)
    sleep = rng.uniform(4.0, 10.0, rows)
    notifications = rng.integers(10, 240, rows).astype(float)
    app_opens = rng.integers(4, 110, rows).astype(float)
    weekend = daily + rng.normal(1.0, 0.5, rows)
    gender = pd.Categorical(np.where(np.arange(rows) % 2, "Male", "Female"))
    stress = pd.Categorical(np.where(np.arange(rows) % 3, "Medium", "High"))
    academic = pd.Categorical(np.where(np.arange(rows) % 4, "Yes", "No"))
    target = (age + 0.05 * notifications + 0.3 * gaming > 27).astype(int)
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
    test = train.drop(columns=[TARGET, "fold"]).iloc[:20].copy()
    baseline = pd.Series(
        age + 0.05 * notifications + rng.normal(0.0, 1.0, rows),
        index=train.index,
        dtype=np.float64,
    )
    return train, test, baseline


def test_pseudo_weight_preserves_total_influence_and_grid_normalizes_beta_zero():
    assert pseudo_row_weight(0.125, labeled_count=80, pseudo_count=20) == 0.5
    assert pseudo_row_weight(0.125, labeled_count=300, pseudo_count=100) == 0.375
    assert candidate_pairs(SearchSpace((0.0, 0.1), (0.0, 0.2))) == [
        (0.0, 0.0),
        (0.0, 0.2),
        (0.1, 0.2),
    ]


def test_selection_config_fixes_current_pool_and_complete_search_space():
    cfg = load_selection_config(SELECTION_CONFIG_PATH)
    assert cfg.control.baseline.run_id == "e5efdf08435e40368add5f0dcb1771b1"
    assert cfg.control.baseline.auc_oof == pytest.approx(0.9697435407980921)
    assert cfg.control.baseline.pool_members == 32
    assert cfg.inputs.pool_sha256 == (
        "943cbfd760cfe81036ec8551d9368b640f3873c1e6d8b16a2f487e1095db4f1d"
    )
    assert cfg.search.rhos == (0.0, 0.025, 0.05, 0.10, 0.125, 0.20, 0.40, 0.80)
    assert cfg.search.betas == (
        0.0,
        0.0125,
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
    )
    assert len(candidate_pairs(cfg.search)) == 65


def test_candidate_ties_prefer_wins_then_smaller_beta_then_smaller_rho():
    scores = [
        CandidateScore(0.0, 0.1, 0.8, 0.01, 2),
        CandidateScore(0.2, 0.05, 0.8, 0.01, 3),
        CandidateScore(0.1, 0.05, 0.8, 0.01, 3),
        CandidateScore(0.0, 0.1, 0.79, 0.0, 4),
    ]
    selected = select_candidate(scores)
    assert (selected.rho, selected.beta) == (0.1, 0.05)


class RecordingCombiner:
    name = "recording"

    def __init__(self):
        self.calls: list[tuple[set[int], set[int]]] = []

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series):
        owner = self
        fit_ids = set(inner_preds.index)

        @dataclass
        class Fitted:
            def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
                owner.calls.append((fit_ids, set(outer_preds.index)))
                return outer_preds.mean(axis=1).to_numpy(dtype=np.float64)

            def summary(self):
                return {column: 0.5 for column in inner_preds.columns}

        return Fitted()


def test_selection_baseline_excludes_outer_and_current_prediction_fold():
    index = pd.Index(np.arange(25), name=ID)
    fold_of = pd.Series(np.arange(25) % 5, index=index)
    y = pd.Series(np.arange(25) % 2, index=index)
    members = pd.DataFrame(
        {"a": np.linspace(0, 1, 25), "b": np.linspace(1, 0, 25)},
        index=index,
    )
    outer = pd.Series(np.linspace(0.1, 0.9, 25), index=index)
    combiner = RecordingCombiner()

    prepared = prepare_selection_baseline(
        combiner, members, fold_of, y, outer
    )

    assert len(combiner.calls) == 20
    for fit_ids, predict_ids in combiner.calls:
        fit_folds = set(fold_of.loc[list(fit_ids)])
        predict_folds = set(fold_of.loc[list(predict_ids)])
        assert len(predict_folds) == 1
        assert fit_ids.isdisjoint(predict_ids)
        assert fit_folds.isdisjoint(predict_folds)
        assert len(fit_folds) == 3
    for excluded in range(5):
        values = prepared[f"selection_pred_excluding_outer_{excluded}"]
        assert values[prepared["fold"] == excluded].isna().all()
        assert values[prepared["fold"] != excluded].notna().all()


class RecordingWeightedTrainer:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "kind": "validation",
                "role": role,
                "train": set(X_train.index),
                "predict": set(X_valid.index),
            }
        )
        return FitResult(self._predict(role, X_valid), best_iteration=20)

    def fit_full(
        self, role, X_train, y_train, X_predict, spec, seed, iterations
    ) -> np.ndarray:
        self.calls.append(
            {
                "kind": "full",
                "role": role,
                "train": set(X_train.index),
                "predict": set(X_predict.index),
            }
        )
        return self._predict(role, X_predict)

    def fit_weighted(
        self,
        role,
        X_train,
        y_train,
        sample_weight,
        X_predict,
        spec,
        seed,
        iterations,
    ) -> np.ndarray:
        assert X_train.index.equals(y_train.index)
        assert X_train.index.equals(sample_weight.index)
        self.calls.append(
            {
                "kind": "weighted",
                "role": role,
                "train": set(X_train.index),
                "predict": set(X_predict.index),
                "weights": sample_weight.copy(),
                "target": y_train.copy(),
            }
        )
        return self._predict(role, X_predict)


def test_outer_selection_never_gives_outer_targets_to_teacher_or_control_student():
    cfg = selection_config()
    train, test, baseline = make_frames()
    outer_fold = 0
    outer_ids = set(train.index[train["fold"] == outer_fold])
    selection_baseline = baseline.copy()
    selection_baseline.loc[train["fold"] == outer_fold] = np.nan
    from pipeline.teacher_student_residual import FeatureMatrixFactory

    trainer = RecordingWeightedTrainer()
    outcome = _evaluate_outer(
        cfg,
        train,
        baseline,
        selection_baseline,
        outer_fold,
        FeatureMatrixFactory(cfg.control, train, test),
        trainer,
    )

    assert len(outcome.frame) == len(outer_ids)
    for call in trainer.calls:
        if call["role"] == "teacher" or call["kind"] == "validation":
            assert set(call["train"]).isdisjoint(outer_ids)
        if call["kind"] == "weighted" and set(call["train"]) & outer_ids:
            weights = call["weights"]
            target = call["target"]
            assert isinstance(weights, pd.Series)
            assert isinstance(target, pd.Series)
            assert (weights.loc[list(outer_ids)] > 0).all()
            assert not np.array_equal(
                target.loc[list(outer_ids)].to_numpy(),
                train.loc[list(outer_ids), TARGET].to_numpy(),
            )


def test_nested_selection_records_required_oof_fields_and_reuses_checkpoints(tmp_path):
    cfg = selection_config()
    train, test, baseline = make_frames()
    selection_baselines = {}
    for fold in range(5):
        values = baseline + fold * 0.001
        values = values.copy()
        values.loc[train["fold"] == fold] = np.nan
        selection_baselines[fold] = values
    baseline_test = pd.Series(
        baseline.iloc[: len(test)].to_numpy(),
        index=pd.Index(test[ID], name=ID),
        dtype=np.float64,
    )
    identity = {"git_commit": "a" * 40, "sha256.config": "b" * 64}
    first = evaluate_nested_selection(
        cfg,
        train,
        test,
        baseline,
        selection_baselines,
        baseline_test,
        trainer=RecordingWeightedTrainer(),
        checkpoint_dir=tmp_path,
        execution_identity=identity,
    )

    class FailingTrainer(RecordingWeightedTrainer):
        def fit_with_validation(self, *args, **kwargs):
            raise AssertionError("완료된 선택을 다시 학습했다.")

        def fit_full(self, *args, **kwargs):
            raise AssertionError("완료된 선택을 다시 학습했다.")

        def fit_weighted(self, *args, **kwargs):
            raise AssertionError("완료된 선택을 다시 학습했다.")

    second = evaluate_nested_selection(
        cfg,
        train,
        test,
        baseline,
        selection_baselines,
        baseline_test,
        trainer=FailingTrainer(),
        checkpoint_dir=tmp_path,
        execution_identity=identity,
    )

    required = {
        ID,
        "fold",
        "baseline_pred",
        "teacher_pred",
        "student_pred",
        "contrast",
        "correction_signal",
        "selected_rho",
        "selected_beta",
        "pseudo_row_weight",
        "rank_control_pred",
        "pred",
    }
    assert required <= set(first.oof.columns)
    assert len(first.oof) == len(train)
    assert all(len(table) == 3 for table in first.candidate_tables.values())
    pd.testing.assert_frame_equal(first.oof, second.oof)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_recorded_selection_can_be_exported_as_verified_run_bundle(
    tmp_path, monkeypatch
):
    cfg = selection_config()
    train, test, baseline = make_frames()
    selection_baselines = {}
    for fold in range(5):
        values = baseline.copy()
        values.loc[train["fold"] == fold] = np.nan
        selection_baselines[fold] = values
    baseline_test = pd.Series(
        baseline.iloc[: len(test)].to_numpy(),
        index=pd.Index(test[ID], name=ID),
        dtype=np.float64,
    )
    evaluation = evaluate_nested_selection(
        cfg,
        train,
        test,
        baseline,
        selection_baselines,
        baseline_test,
        trainer=RecordingWeightedTrainer(),
    )
    monkeypatch.setattr(
        "pipeline.tracking.git_state",
        lambda: {"git_commit": "a" * 40, "git_dirty": "False"},
    )
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    input_hashes = {
        "train": "1" * 64,
        "test": "2" * 64,
        "folds": "3" * 64,
        "baseline_oof": "4" * 64,
        "config": "5" * 64,
    }
    run_id = record_selection_evaluation(
        cfg,
        evaluation,
        input_hashes,
        {"schema_version": 2},
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
    assert run.data.params["features"]
    assert run.data.params["baseline.run_id"] == cfg.control.baseline.run_id
    assert run.data.metrics["auc_oof"] == evaluation.corrected_auc
