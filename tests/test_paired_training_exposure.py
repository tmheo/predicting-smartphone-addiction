"""결측 증강 짝의 신경망 학습 갱신량 보존 회귀 검사."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pipeline import model as model_mod
from pipeline import paired_training_length


@pytest.mark.parametrize(
    ("model_kind", "params", "source_batch_size"),
    [
        ("lookup_transformer", {}, 2048),
        ("contextualized_spline_transformer", {}, 4096),
        ("tabm", {"batch_size": 128}, 128),
        ("realmlp", {"batch_size": 256}, 256),
    ],
)
def test_neural_replica_rows_preserve_optimizer_steps_per_epoch(
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
    assert plan.paired_batch_size == source_batch_size * 3
    assert plan.source_steps_per_epoch == math.ceil(original_rows / source_batch_size)
    assert plan.paired_steps_per_epoch == plan.source_steps_per_epoch
    assert plan.apply(params)["batch_size"] == source_batch_size * 3


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
