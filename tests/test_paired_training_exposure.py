"""결측 증강 짝의 신경망 학습 갱신량 보존 회귀 검사."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pipeline import model as model_mod
from pipeline import paired_training_length
from pipeline.training_length import observe_declaration


@pytest.mark.parametrize(
    ("model_kind", "params", "source_batch_size"),
    [
        ("lookup_transformer", {}, 2048),
        ("contextualized_spline_transformer", {}, 4096),
        ("tabm", {"batch_size": 128}, 128),
        ("realmlp", {"batch_size": 256}, 256),
    ],
)
def test_neural_replica_rows_preserve_physical_batch_and_optimizer_steps(
    model_kind, params, source_batch_size
):
    original_rows = 553_095
    training_rows = original_rows * 3

    plan = paired_training_length.build_optimizer_step_plan(
        model_kind,
        params,
        original_row_count=original_rows,
        training_row_count=training_rows,
    )

    assert plan is not None
    assert plan.row_multiplier == 3
    assert plan.source_batch_size == source_batch_size
    assert plan.paired_batch_size == source_batch_size
    assert plan.source_steps_per_epoch == math.ceil(original_rows / source_batch_size)
    assert plan.paired_steps_per_epoch == plan.source_steps_per_epoch
    adjusted = plan.apply(params)
    assert adjusted.get("batch_size", source_batch_size) == source_batch_size
    assert adjusted["paired_original_row_count"] == original_rows
    assert adjusted["paired_row_multiplier"] == 3


def test_tree_replica_rows_keep_existing_iteration_contract():
    assert (
        paired_training_length.build_optimizer_step_plan(
            "lightgbm",
            {"n_estimators": 20_000},
            original_row_count=80,
            training_row_count=240,
        )
        is None
    )


def test_neural_step_plan_rejects_non_integral_replica_rows():
    with pytest.raises(ValueError, match="정수배"):
        paired_training_length.build_optimizer_step_plan(
            "lookup_transformer",
            {},
            original_row_count=80,
            training_row_count=239,
        )


def test_lookup_paired_fit_preserves_source_schedule_horizon(monkeypatch):
    calls = []

    class FakeLookupFold:
        def fit_full_member_training_points(
            self, X, y, member_epochs, schedule_horizon_epochs
        ):
            calls.append(
                {
                    "rows": len(X),
                    "member_epochs": member_epochs,
                    "schedule_horizon_epochs": schedule_horizon_epochs,
                }
            )

    adapter = model_mod.LookupTransformerAdapter(
        {
            "lookup_cols": ["x"],
            "epochs": 32,
            "fold_seed_offsets": [0, 1000, 2000],
        },
        {},
        42,
    )
    monkeypatch.setattr(adapter, "_new_impl", lambda: FakeLookupFold())
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0])

    adapter.fit_paired_training_lengths(X, y, (12, 12, 12))

    assert calls == [
        {
            "rows": 3,
            "member_epochs": (12, 12, 12),
            "schedule_horizon_epochs": 32,
        }
    ]


def test_lookup_paired_fit_keeps_source_update_count_and_schedule_in_real_path():
    original_rows = 96
    row_multiplier = 3
    source_batch_size = 32
    plan = paired_training_length.build_optimizer_step_plan(
        "lookup_transformer",
        {"batch_size": source_batch_size},
        original_row_count=original_rows,
        training_row_count=original_rows * row_multiplier,
    )
    assert plan is not None
    params = plan.apply(
        {
            "lookup_cols": ["x"],
            "d_model": 8,
            "plr_k": 2,
            "layers": 1,
            "heads": 2,
            "epochs": 4,
            "batch_size": source_batch_size,
            "value_dropout": 0.0,
            "perm_repeats": 1,
        }
    )
    base_x = pd.Series(range(original_rows), dtype="float64") % 7
    base_y = (base_x > 3).astype("int64")
    X = pd.DataFrame({"x": pd.concat([base_x] * row_multiplier, ignore_index=True)})
    y = pd.concat([base_y] * row_multiplier, ignore_index=True)
    adapter = model_mod.LookupTransformerAdapter(params, {}, 42)

    adapter.fit_paired_training_lengths(X, y, (2,))

    member = adapter.training_diagnostics()["fold_initialization_members"][0]
    source_steps_per_epoch = math.ceil(original_rows / source_batch_size)
    assert member["completed_steps"] == source_steps_per_epoch * 2
    assert member["planned_total_steps"] == source_steps_per_epoch * 4 + 10
    assert member["trajectory_end_epochs"] == 2
    assert member["schedule_horizon_epochs"] == 4
    assert member["parent_balanced_exposure"] == {
        "original_row_count": original_rows,
        "row_multiplier": row_multiplier,
        "preprocessing_fit_rows": original_rows,
        "replica_selection": "deterministic_parent_rotation",
    }


def test_lookup_duplicate_control_reproduces_source_training_path():
    all_rows = 48
    original_rows = 36
    row_multiplier = 3
    params = {
        "lookup_cols": ["x"],
        "preprocessing_scope": "train_test",
        "d_model": 8,
        "plr_k": 2,
        "layers": 1,
        "heads": 2,
        "epochs": 3,
        "batch_size": 16,
        "value_dropout": 0.1,
        "perm_repeats": 1,
    }
    base_x = pd.Series(range(all_rows), dtype="float64") % 7
    base_y = (base_x > 3).astype("int64")
    source_X = pd.DataFrame({"x": base_x})
    source_y = base_y
    reference_test = pd.DataFrame({"x": [1.25, 5.75]})

    source = model_mod.LookupTransformerAdapter(params, {}, 42)
    model_mod.set_dataset_reference(source, source_X, reference_test)
    expected = source.fit(
        source_X.iloc[:original_rows],
        source_y.iloc[:original_rows],
        source_X.iloc[original_rows:],
        source_y.iloc[original_rows:],
    )
    evidence = observe_declaration(
        source.training_length_evidence(), seed=42, outer_fold=0
    )
    training_lengths = tuple(item.value for item in evidence.observations)

    plan = paired_training_length.build_optimizer_step_plan(
        "lookup_transformer",
        params,
        original_row_count=original_rows,
        training_row_count=original_rows * row_multiplier,
    )
    assert plan is not None
    paired_X = pd.concat(
        [source_X.iloc[:original_rows]] * row_multiplier, ignore_index=True
    )
    paired_y = pd.concat(
        [source_y.iloc[:original_rows]] * row_multiplier, ignore_index=True
    )
    paired = model_mod.LookupTransformerAdapter(plan.apply(params), {}, 42)
    model_mod.set_dataset_reference(paired, source_X, reference_test)
    paired.fit_paired_training_lengths(paired_X, paired_y, training_lengths)

    actual = paired.predict(source_X.iloc[original_rows:])
    assert actual == pytest.approx(expected, abs=1e-8)


def test_contextual_duplicate_control_reproduces_source_training_path():
    all_rows = 48
    original_rows = 36
    params = {
        "exact_cols": ["v", "c"],
        "numeric_mode": "spline",
        "token_dim": 8,
        "attention_dim": 8,
        "attention_heads": 2,
        "default_width": 8,
        "default_depth": 1,
        "context_hidden": 4,
        "gate_hidden": 4,
        "residual_hidden": 16,
        "epochs": 2,
        "patience": 2,
        "batch_size": 16,
        "perm_repeats": 1,
        "dropout": 0.0,
    }
    source_X = pd.DataFrame(
        {
            "v": np.resize([1.5, 2.5, 3.5, 4.5], all_rows),
            "c": np.resize(["Low", "High"], all_rows),
            "z": np.linspace(-1.0, 1.0, all_rows),
        }
    )
    source_y = pd.Series((source_X["v"] > 2.5).astype("int64"))
    source = model_mod.ContextualizedSplineTransformerAdapter(params, {}, 7)
    expected = source.fit(
        source_X.iloc[:original_rows],
        source_y.iloc[:original_rows],
        source_X.iloc[original_rows:],
        source_y.iloc[original_rows:],
    )
    evidence = observe_declaration(
        source.training_length_evidence(), seed=7, outer_fold=0
    )
    training_length = evidence.observations[0].value

    plan = paired_training_length.build_optimizer_step_plan(
        "contextualized_spline_transformer",
        params,
        original_row_count=original_rows,
        training_row_count=original_rows * 3,
    )
    assert plan is not None
    paired_X = pd.concat(
        [source_X.iloc[:original_rows]] * 3, ignore_index=True
    )
    paired_y = pd.concat(
        [source_y.iloc[:original_rows]] * 3, ignore_index=True
    )
    paired = model_mod.ContextualizedSplineTransformerAdapter(
        plan.apply(params), {}, 7
    )
    model_mod.fit_full(paired, paired_X, paired_y, training_budget=training_length)

    assert paired.predict(source_X.iloc[original_rows:]) == pytest.approx(
        expected, abs=1e-8
    )


def test_tabm_duplicate_control_reproduces_source_training_path():
    all_rows = 96
    original_rows = 72
    rng = np.random.default_rng(19)
    source_X = pd.DataFrame(
        {
            "v": rng.normal(size=all_rows),
            "c": pd.Categorical(np.resize(["Low", "High"], all_rows)),
            "z": rng.normal(size=all_rows),
        }
    )
    source_y = pd.Series((source_X["v"] > 0).astype("int64"))
    params = {
        "tabm_k": 2,
        "d_embedding": 4,
        "batch_size": 16,
        "lr": 5e-3,
        "n_epochs": 3,
        "d_block": 16,
        "n_blocks": 1,
        "patience": 3,
        "n_seed_avg": 1,
        "perm_repeats": 1,
        "perm_sample": 24,
    }
    source = model_mod.TabMAdapter(params, {}, 7)
    expected = source.fit(
        source_X.iloc[:original_rows],
        source_y.iloc[:original_rows],
        source_X.iloc[original_rows:],
        source_y.iloc[original_rows:],
    )
    selected_epoch = source.training_diagnostics()["members"][0][
        "selected_epoch_count"
    ]

    plan = paired_training_length.build_optimizer_step_plan(
        "tabm",
        params,
        original_row_count=original_rows,
        training_row_count=original_rows * 3,
    )
    assert plan is not None
    paired_X = pd.concat(
        [source_X.iloc[:original_rows]] * 3, ignore_index=True
    )
    paired_y = pd.concat(
        [source_y.iloc[:original_rows]] * 3, ignore_index=True
    )
    paired = model_mod.TabMAdapter(plan.apply(params), {}, 7)
    paired.fit_paired_training_lengths(
        paired_X, paired_y, (selected_epoch,)
    )

    assert paired.predict(source_X.iloc[original_rows:]) == pytest.approx(
        expected, abs=1e-7
    )


def test_realmlp_duplicate_control_reproduces_source_training_path():
    from pipeline.realmlp import RAW_NUMERICAL

    all_rows = 60
    original_rows = 45
    rng = np.random.default_rng(11)
    source_X = pd.DataFrame(
        {
            "age": rng.integers(13, 60, all_rows).astype(float),
            "daily_screen_time_hours": rng.uniform(1, 12, all_rows),
            "social_media_hours": rng.uniform(0, 7, all_rows),
            "gaming_hours": rng.uniform(0, 6, all_rows),
            "work_study_hours": rng.uniform(0, 10, all_rows),
            "sleep_hours": rng.uniform(4, 10, all_rows),
            "notifications_per_day": rng.integers(0, 400, all_rows).astype(float),
            "app_opens_per_day": rng.integers(0, 250, all_rows).astype(float),
            "weekend_screen_time": rng.uniform(1, 16, all_rows),
            "gender": pd.Categorical(np.resize(["Male", "Female"], all_rows)),
            "stress_level": pd.Categorical(
                np.resize(["Low", "Medium", "High"], all_rows)
            ),
            "academic_work_impact": pd.Categorical(
                np.resize(["Low", "High"], all_rows)
            ),
            "placebo_noise": rng.normal(size=all_rows),
        },
        index=pd.Index(np.arange(all_rows) * 2 + 1),
    )
    source_y = pd.Series(
        (source_X["daily_screen_time_hours"].to_numpy() > 6).astype("int64"),
        index=source_X.index,
    )
    params = {
        "n_ens": 1,
        "embed_dim": 2,
        "onehot_thresh": 4,
        "hidden_dims": [8],
        "dropout": 0.0,
        "pbld_hidden_dim": 4,
        "pbld_out_dim": 2,
        "fixed_epochs": 1,
        "schedule_epochs": 2,
        "batch_size": 16,
        "eval_batch_size": 64,
        "n_init_avg": 1,
        "inner_folds": 3,
        "reference_qnormal_columns": RAW_NUMERICAL,
        "preprocessing_scope": "train_test",
        "perm_sample": 24,
        "perm_repeats": 1,
        "device": "cpu",
        "verbosity": 0,
    }
    reference_test = source_X.iloc[:10].copy()
    source = model_mod.RealMLPAdapter(params, {}, 42)
    model_mod.set_dataset_reference(source, source_X, reference_test)
    expected = source.fit(
        source_X.iloc[:original_rows],
        source_y.iloc[:original_rows],
        source_X.iloc[original_rows:],
        source_y.iloc[original_rows:],
    )
    evidence = observe_declaration(
        source.training_length_evidence(), seed=42, outer_fold=0
    )
    training_length = evidence.observations[0].value

    plan = paired_training_length.build_optimizer_step_plan(
        "realmlp",
        params,
        original_row_count=original_rows,
        training_row_count=original_rows * 3,
        original_index=source_X.index[:original_rows],
    )
    assert plan is not None
    paired_X = pd.concat(
        [source_X.iloc[:original_rows]] * 3, ignore_index=True
    )
    paired_y = pd.concat(
        [source_y.iloc[:original_rows]] * 3, ignore_index=True
    )
    paired = model_mod.RealMLPAdapter(plan.apply(params), {}, 42)
    model_mod.set_dataset_reference(paired, source_X, reference_test)
    model_mod.fit_full(
        paired, paired_X, paired_y, training_budget=training_length
    )

    assert paired.predict(source_X.iloc[original_rows:]) == pytest.approx(
        expected, abs=1e-7
    )
