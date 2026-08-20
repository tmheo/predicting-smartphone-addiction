import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

import pipeline.catboost_hpo as catboost_hpo
from pipeline.catboost_hpo import (
    BASELINE_FOLD0_AUC,
    GpuMemorySampler,
    PoolRanks,
    checkpoint_continues,
    export_pool_fold_predictions,
    parse_nvidia_smi_memory,
    promoted_trial_numbers,
    sample_model_params,
)
from pipeline.ledger import EntryEvidence, Pool, PoolMember
from pipeline.runs import InMemoryRunStore


def test_sample_model_params_matches_approved_gpu_space() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.01,
            "subsample": 0.8,
            "grow_policy": "Depthwise",
            "depth": 7,
            "l2_leaf_reg": 0.25,
            "leaf_estimation_iterations": 5,
            "one_hot_max_size": 32,
            "model_size_reg": 0.4,
        }
    )

    params = sample_model_params(trial, "0")

    assert params == {
        "iterations": 10000,
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": "0",
        "max_ctr_complexity": 1,
        "boosting_type": "Plain",
        "max_bin": 254,
        "bootstrap_type": "Bernoulli",
        "learning_rate": 0.01,
        "subsample": 0.8,
        "grow_policy": "Depthwise",
        "depth": 7,
        "l2_leaf_reg": 0.25,
        "leaf_estimation_iterations": 5,
        "one_hot_max_size": 32,
        "model_size_reg": 0.4,
    }
    assert "colsample_bylevel" not in params


def test_checkpoint_requires_full_point_zero_zero_zero_two_gain() -> None:
    assert not checkpoint_continues(BASELINE_FOLD0_AUC + 0.000199999999)
    assert checkpoint_continues(BASELINE_FOLD0_AUC + 0.0002)


def _completed_trial(study: optuna.Study, value: float) -> None:
    trial = study.ask()
    study.tell(trial, value)


def test_promotion_uses_one_winner_after_stop_and_two_after_continuation() -> None:
    study = optuna.create_study(direction="maximize")
    _completed_trial(study, 0.7)
    _completed_trial(study, 0.9)
    _completed_trial(study, 0.8)

    assert promoted_trial_numbers(study.trials, False) == [1]
    assert promoted_trial_numbers(study.trials, True) == [1, 2]


def test_nearest_pool_member_uses_average_tie_spearman() -> None:
    ranks = PoolRanks.from_frame(
        pd.DataFrame(
            {
                "reverse": [4.0, 3.0, 2.0, 1.0],
                "tied": [1.0, 1.0, 3.0, 4.0],
            }
        )
    )

    member, correlation = ranks.nearest(np.asarray([2.0, 2.0, 3.0, 4.0]))

    assert member == "tied"
    assert correlation == pytest.approx(1.0)


def _member(run_id: str, config: str) -> PoolMember:
    return PoolMember(
        run_id=run_id,
        config=config,
        oof_auc=0.8,
        seeds=[42, 43, 44],
        entered_at="2026-08-20",
        reason="test",
        evidence=EntryEvidence(
            champion_run_id="champion",
            champion_oof_auc=0.9,
            floor_margin=0.0,
            nearest_run_id=None,
            nearest_spearman=None,
            ensemble_auc_with=None,
            ensemble_auc_without=None,
            contribution=None,
        ),
    )


def test_export_pool_fold_predictions_aligns_every_member_by_id(tmp_path: Path) -> None:
    store = InMemoryRunStore()
    store.add_run(
        "run-a",
        oof=pd.DataFrame(
            {"id": [30, 10, 20], "fold": [0, 0, 1], "pred": [0.3, 0.1, 0.2]}
        ),
    )
    store.add_run(
        "run-b",
        oof=pd.DataFrame(
            {"id": [20, 30, 10], "fold": [1, 0, 0], "pred": [0.6, 0.7, 0.8]}
        ),
    )
    pool = Pool(members=[_member("run-a", "exp_a"), _member("run-b", "exp_b")])
    folds = pd.DataFrame({"id": [10, 20, 30], "fold": [0, 1, 0]})
    output = tmp_path / "pool.parquet"

    exported = export_pool_fold_predictions(pool, store, folds, output)

    assert list(exported["id"]) == [10, 30]
    assert list(exported["exp_a::run-a"]) == [0.1, 0.3]
    assert list(exported["exp_b::run-b"]) == [0.8, 0.7]
    pd.testing.assert_frame_equal(pd.read_parquet(output), exported)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("123, 24576\n", (123, 24576)),
        (" 0, 49140 ", (0, 49140)),
    ],
)
def test_parse_nvidia_smi_memory(text: str, expected: tuple[int, int]) -> None:
    assert parse_nvidia_smi_memory(text) == expected


@pytest.mark.parametrize("text", ["", "1", "1,2\n3,4", "-1,24576", "9,8"])
def test_parse_nvidia_smi_memory_rejects_invalid_output(text: str) -> None:
    with pytest.raises(ValueError):
        parse_nvidia_smi_memory(text)


def test_gpu_memory_sampler_keeps_peak_and_total() -> None:
    readings = [(100, 24576), (300, 24576), (200, 24576)]

    def query() -> tuple[int, int]:
        return readings.pop(0) if len(readings) > 1 else readings[0]

    sampler = GpuMemorySampler(query, interval_seconds=0.001)

    with sampler:
        import time

        time.sleep(0.004)

    assert sampler.start_mib == 100
    assert sampler.peak_mib == 300
    assert sampler.total_mib == 24576


def test_run_uses_config_repository_root_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = Path(catboost_hpo.__file__).resolve().parents[2]
    config_path = repository_root / "configs/exp070_cat_exact_cats.yaml"
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    captured: dict[str, object] = {}

    def fake_run_search(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        captured["working_directory"] = Path.cwd()
        return {"status": "completed"}

    monkeypatch.setattr(catboost_hpo, "run_search", fake_run_search)

    catboost_hpo._run(
        argparse.Namespace(
            config=config_path,
            pool_oof=Path("pool-fold0.parquet"),
            output_dir=Path("results"),
            device="0",
            memory_interval_seconds=0.5,
        )
    )

    assert captured["config_path"] == config_path
    assert captured["pool_oof"] == caller / "pool-fold0.parquet"
    assert captured["output_dir"] == caller / "results"
    assert captured["working_directory"] == repository_root
    assert Path.cwd() == caller
