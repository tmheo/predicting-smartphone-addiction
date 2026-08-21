"""#342 성능 동등 대역 측정 도구의 순수 계산 계약."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

SCRIPT = Path(__file__).parents[1] / "scripts" / "measure_pool_equivalence_band.py"
SPEC = importlib.util.spec_from_file_location("measure_pool_equivalence_band", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
band = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = band
SPEC.loader.exec_module(band)


CONFIGS = [f"exp{index:03d}" for index in range(35)]


def test_plan_covers_every_member_once_at_size_one() -> None:
    plan = band.build_contrast_plan(CONFIGS)

    clones = [c for c in plan if c.kind == band.CLONE and c.size == 1]

    assert [c.sources[0] for c in clones] == CONFIGS


def test_plan_repeat_counts_match_the_frozen_design() -> None:
    plan = band.build_contrast_plan(CONFIGS)

    noise_size_one = [c for c in plan if c.kind == band.NOISE and c.size == 1]
    assert len(noise_size_one) == band.SIZE1_NOISE_REPEATS
    for size in band.GROUP_SIZES:
        if size == 1:
            continue
        for kind in (band.CLONE, band.NOISE):
            matching = [c for c in plan if c.kind == kind and c.size == size]
            assert len(matching) == band.MULTI_REPEATS
    assert [c.index for c in plan] == list(range(len(plan)))


def test_plan_is_reproducible_and_samples_without_replacement() -> None:
    first = band.build_contrast_plan(CONFIGS)
    second = band.build_contrast_plan(CONFIGS)

    assert first == second
    for contrast in first:
        if contrast.kind != band.CLONE:
            assert contrast.sources == ()
            continue
        assert len(set(contrast.sources)) == contrast.size
        assert set(contrast.sources) <= set(CONFIGS)


def test_plan_rejects_a_duplicated_member_name() -> None:
    with pytest.raises(band.MeasurementError):
        band.build_contrast_plan(["a", "a"])


def test_added_columns_never_collide_with_pool_members() -> None:
    plan = band.build_contrast_plan(CONFIGS)

    for contrast in plan:
        names = contrast.added_columns()
        assert len(set(names)) == contrast.size
        assert not set(names) & set(CONFIGS)


def test_clone_arm_repeats_the_source_prediction_exactly() -> None:
    base = pd.DataFrame(
        {name: np.linspace(0.1, 0.9, 6) + index for index, name in enumerate(["a", "b"])}
    )
    contrast = band.NullContrast(index=0, kind=band.CLONE, size=1, sources=("b",))

    augmented = band.augmented_matrix(base, contrast)

    assert list(augmented.columns) == ["a", "b", "b__복제0"]
    np.testing.assert_array_equal(augmented["b__복제0"].to_numpy(), base["b"].to_numpy())


def test_noise_arm_is_a_permutation_rank_column_fixed_by_the_contrast_index() -> None:
    base = pd.DataFrame({"a": np.linspace(0.0, 1.0, 8)})
    contrast = band.NullContrast(index=3, kind=band.NOISE, size=2, sources=())

    first = band.augmented_matrix(base, contrast)
    second = band.augmented_matrix(base, contrast)
    other = band.augmented_matrix(
        base, band.NullContrast(index=4, kind=band.NOISE, size=2, sources=())
    )

    added = first.iloc[:, 1:].to_numpy()
    np.testing.assert_array_equal(added, second.iloc[:, 1:].to_numpy())
    assert not np.array_equal(added, other.iloc[:, 1:].to_numpy())
    for column in added.T:
        np.testing.assert_allclose(np.sort(column), (np.arange(8) + 1) / 8)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_weighted_auc_matches_roc_auc_score_on_the_expanded_resample(seed: int) -> None:
    rng = np.random.default_rng(seed)
    prediction = rng.normal(size=60)
    y = (rng.random(60) < 0.4).astype(np.float64)
    weights = rng.integers(0, 4, size=60).astype(np.float64)
    repeated = np.repeat(np.arange(60), weights.astype(int))
    if len(np.unique(y[repeated])) < 2:
        pytest.skip("재표본에 한쪽 라벨만 남았다.")

    measured = band.WeightedAucSorter(prediction, y).auc(weights)

    expected = roc_auc_score(y[repeated], prediction[repeated])
    assert measured == pytest.approx(expected)


def test_weighted_auc_handles_tied_predictions_as_midranks() -> None:
    prediction = np.asarray([0.2, 0.2, 0.2, 0.9])
    y = np.asarray([0.0, 1.0, 0.0, 1.0])
    weights = np.asarray([1.0, 2.0, 1.0, 1.0])
    repeated = np.repeat(np.arange(4), weights.astype(int))

    measured = band.WeightedAucSorter(prediction, y).auc(weights)

    assert measured == pytest.approx(roc_auc_score(y[repeated], prediction[repeated]))


def test_weighted_auc_refuses_a_single_class_resample() -> None:
    prediction = np.asarray([0.1, 0.9])
    y = np.asarray([0.0, 1.0])

    with pytest.raises(band.MeasurementError):
        band.WeightedAucSorter(prediction, y).auc(np.asarray([2.0, 0.0]))


def test_bootstrap_weights_preserve_each_outer_fold_row_count() -> None:
    positions = [np.asarray([0, 1, 2]), np.asarray([3, 4, 5, 6])]

    weights = band.stratified_bootstrap_weights(
        np.random.default_rng(band.BOOTSTRAP_SEED), positions, 7
    )

    assert weights[:3].sum() == 3
    assert weights[3:].sum() == 4


def test_paired_bootstrap_is_reproducible_and_brackets_its_median() -> None:
    rng = np.random.default_rng(11)
    y = np.repeat([0.0, 1.0], 60)
    small = rng.normal(size=120) + y * 0.8
    large = small + rng.normal(scale=0.01, size=120)
    positions = [np.arange(0, 60), np.arange(60, 120)]

    first = band.paired_row_bootstrap(small, large, y, positions, replicates=64)
    second = band.paired_row_bootstrap(small, large, y, positions, replicates=64)

    assert first == second
    assert first["minimum"] <= first["percentile_2p5"] <= first["median"]
    assert first["median"] <= first["percentile_97p5"] <= first["maximum"]


def test_band_takes_the_union_of_observed_and_bootstrap_envelopes() -> None:
    records = [
        {
            "delta_best": -1e-5,
            "negative_folds": 3,
            "fold_total": 5,
            "best_strategy_changed": True,
            "bootstrap": {"percentile_2p5": -4e-5, "percentile_97p5": 2e-5},
        },
        {
            "delta_best": 3e-5,
            "negative_folds": 1,
            "fold_total": 5,
            "best_strategy_changed": False,
            "bootstrap": {"percentile_2p5": -2e-5, "percentile_97p5": 1e-5},
        },
    ]

    measured = band.band_for_records(records)

    assert measured["lower"] == pytest.approx(-4e-5)
    assert measured["upper"] == pytest.approx(3e-5)
    assert measured["observed_minimum"] == pytest.approx(-1e-5)
    assert measured["observed_maximum"] == pytest.approx(3e-5)
    assert measured["max_negative_folds"] == 3
    assert measured["best_strategy_changes"] == 1


def _arm(index: int, size: int, kind: str, auc: float) -> dict:
    return {
        "index": index,
        "size": size,
        "kind": kind,
        "label": f"{kind} g={size} #{index}",
        "large_best_auc": auc,
        "delta_best": 0.0,
        "negative_folds": 0,
        "fold_total": 5,
        "best_strategy_changed": False,
        "bootstrap": {"percentile_2p5": 0.0, "percentile_97p5": 0.0},
    }


def test_same_size_pairs_only_join_arms_of_the_same_removal_size() -> None:
    records = [
        _arm(0, 1, band.CLONE, 0.96),
        _arm(1, 1, band.NOISE, 0.97),
        _arm(2, 2, band.NOISE, 0.95),
    ]

    pairs = band.same_size_pairs(records)

    assert {pair["size"] for pair in pairs} == {1}
    assert len(pairs) == 2


def test_same_size_distribution_is_symmetric_about_zero() -> None:
    records = [
        _arm(0, 1, band.NOISE, 0.960),
        _arm(1, 1, band.NOISE, 0.961),
        _arm(2, 1, band.NOISE, 0.964),
    ]

    summary = band.same_size_summary(records)["by_size"]["1"]["all"]

    assert summary["pairs"] == 6
    assert summary["observed_minimum"] == pytest.approx(-summary["observed_maximum"])
    assert summary["symmetric_envelope"] == pytest.approx(0.004)


def test_same_size_summary_splits_clone_and_noise_pairings() -> None:
    records = [
        _arm(0, 1, band.CLONE, 0.960),
        _arm(1, 1, band.CLONE, 0.962),
        _arm(2, 1, band.NOISE, 0.965),
    ]

    by_size = band.same_size_summary(records)["by_size"]["1"]

    mixed = " vs ".join(sorted((band.CLONE, band.NOISE)))
    assert by_size[f"{band.CLONE} vs {band.CLONE}"]["pairs"] == 2
    assert by_size[mixed]["pairs"] == 4
    assert f"{band.NOISE} vs {band.NOISE}" not in by_size


def test_summary_union_covers_both_bands() -> None:
    records = [
        {**_arm(0, 1, band.CLONE, 0.960), "delta_best": 2e-5},
        {**_arm(1, 1, band.NOISE, 0.966), "delta_best": 3e-5},
    ]

    summary = band.summarize(records)

    assert summary["union"]["lower"] <= summary["overall"]["lower"]
    assert summary["union"]["upper"] >= summary["overall"]["upper"]
    assert summary["union"]["lower"] == pytest.approx(
        summary["same_size"]["overall"]["observed_minimum"]
    )


def test_checkpoint_round_trips_and_drops_a_truncated_last_line(tmp_path) -> None:
    path = tmp_path / "contrasts.jsonl"
    band.append_checkpoint(path, {"index": 0, "delta_best": 1.0})
    band.append_checkpoint(path, {"index": 2, "delta_best": 2.0})
    with path.open("a") as handle:
        handle.write('{"index": 3, "delta_be')

    done = band.load_checkpoint(path)

    assert sorted(done) == [0, 2]
    assert done[2]["delta_best"] == 2.0


def test_checkpoint_of_a_missing_file_is_empty(tmp_path) -> None:
    assert band.load_checkpoint(tmp_path / "absent.jsonl") == {}


def test_baseline_cache_round_trips_the_arm_and_its_prediction(tmp_path) -> None:
    path = tmp_path / "baseline-arm.json"
    arm = band.ArmResult(
        label="기준 풀",
        members=35,
        strategy_auc={"rank_mean": 0.9},
        strategy_fold_auc={},
        failures={"greedy_rank_mean": "미수렴"},
        best_strategy="rank_mean",
        best_auc=0.9,
        best_fold_auc={0: 0.91, 1: 0.89},
        elapsed_seconds=671.0,
    )
    prediction = np.asarray([0.1, 0.2, 0.3])

    band.save_baseline_cache(path, arm, prediction)
    restored, restored_prediction = band.load_baseline_cache(path)

    assert restored.best_strategy == "rank_mean"
    assert restored.best_auc == 0.9
    assert restored.best_fold_auc == {0: 0.91, 1: 0.89}
    assert sorted(restored.failures) == ["greedy_rank_mean"]
    np.testing.assert_array_equal(restored_prediction, prediction)


def test_baseline_cache_is_absent_without_its_prediction_file(tmp_path) -> None:
    path = tmp_path / "baseline-arm.json"
    path.write_text("{}")

    assert band.load_baseline_cache(path) is None
