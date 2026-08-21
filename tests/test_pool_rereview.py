"""이슈 339 실제 재심사 실행기의 고정 판정 계약."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from pipeline.pool_rereview import (
    CandidateState,
    InputContext,
    PoolScore,
    SplitResult,
    StrategyEvaluator,
    WeightedAucSorter,
    choose_final_candidate,
    paired_bootstrap,
    run_split,
)


def _score(members: tuple[str, ...], auc: float, strategy: str = "rank_mean") -> PoolScore:
    return PoolScore(
        members=members,
        strategy_auc={strategy: auc},
        strategy_fold_auc={strategy: {"0": auc, "1": auc}},
        best_strategy=strategy,
        best_auc=auc,
        best_fold_auc={"0": auc, "1": auc},
    )


def test_weighted_auc_sorter_matches_sklearn_with_ties():
    prediction = np.array([0.1, 0.2, 0.2, 0.8, 0.9, 0.9], dtype=np.float64)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    weights = np.array([2, 1, 3, 2, 1, 4], dtype=np.float64)

    actual = WeightedAucSorter(prediction, labels).auc(weights)
    expected = roc_auc_score(labels, prediction, sample_weight=weights)

    assert actual == expected


def test_paired_bootstrap_is_zero_for_identical_predictions():
    prediction = np.linspace(0.01, 0.99, 40)
    labels = np.tile([0, 1], 20)
    folds = np.repeat(np.arange(5), 8)

    result = paired_bootstrap(
        prediction,
        prediction.copy(),
        labels,
        folds,
        np.random.default_rng(342342),
        25,
    )

    assert result["minimum"] == 0.0
    assert result["percentile_2p5"] == 0.0
    assert result["maximum"] == 0.0


def test_exclusion_contribution_uses_after_minus_before_direction():
    evaluator = object.__new__(StrategyEvaluator)
    evaluator.fits = 0
    evaluator._map = lambda payloads: [
        {"name": "rank_mean", "auc": 0.81, "fold_auc": {}, "fits": 2, "failure": None}
    ]
    anchor = _score(("a", "b"), 0.80)

    result = evaluator.contributions(
        "rank_mean", anchor, [("a",)], excluded_fold=None
    )

    assert result[("a",)] == pytest.approx(0.01)
    assert evaluator.fits == 2


def test_final_candidate_scans_all_accepted_states_not_only_terminal():
    anchor = _score(tuple("abcdef"), 0.90)
    middle = _score(tuple("abcd"), 0.90 - 0.00002)
    terminal = _score(tuple("ab"), 0.90 - 0.00004)
    result = SplitResult(
        label="final",
        excluded_fold=None,
        anchor=anchor,
        terminal=terminal,
        trajectory=[],
        accepted_states=[
            CandidateState(0, anchor.members, anchor),
            CandidateState(4, middle.members, middle),
            CandidateState(8, terminal.members, terminal),
        ],
        order={},
    )
    ledger = {
        "candidate_pool": {
            "members": list(anchor.members),
            "full_refit_count": {"default": 1, "overrides": {}},
        }
    }

    selected, previous = choose_final_candidate(result, -0.000027669802, ledger)

    assert selected.pool == middle.members
    assert previous is not None
    assert previous.pool == anchor.members


def test_final_candidate_same_size_prefers_higher_anchor_delta():
    anchor = _score(("a", "b", "c", "d"), 0.90)
    first = _score(("a", "b"), 0.90001)
    second = _score(("c", "d"), 0.90002)
    result = SplitResult(
        label="final",
        excluded_fold=None,
        anchor=anchor,
        terminal=second,
        trajectory=[],
        accepted_states=[
            CandidateState(0, anchor.members, anchor),
            CandidateState(1, first.members, first),
            CandidateState(2, second.members, second),
        ],
        order={},
    )
    ledger = {
        "candidate_pool": {
            "members": list(anchor.members),
            "full_refit_count": {"default": 1, "overrides": {}},
        }
    }

    selected, _ = choose_final_candidate(result, -0.000027669802, ledger)

    assert selected.pool == second.members


class _DeterministicEvaluator:
    def __init__(self, strategy_names: tuple[str, ...]) -> None:
        self.strategy_names = strategy_names

    def evaluate(self, members, *, excluded_fold, **_):
        pool = tuple(members)
        auc = 0.9 - (35 - len(pool)) * 0.000001
        fold_keys = [str(fold) for fold in range(5) if fold != excluded_fold]
        strategy_auc = {
            name: auc - index * 0.000001
            for index, name in enumerate(self.strategy_names)
        }
        return PoolScore(
            members=pool,
            strategy_auc=strategy_auc,
            strategy_fold_auc={
                name: {fold: value for fold in fold_keys}
                for name, value in strategy_auc.items()
            },
            best_strategy=self.strategy_names[0],
            best_auc=auc,
            best_fold_auc={fold: auc for fold in fold_keys},
        )

    def contributions(self, strategy, anchor, targets, *, excluded_fold):
        assert strategy == self.strategy_names[0]
        return {target: -len(target) * 0.000001 for target in targets}


def test_split_control_flow_covers_fixed_four_stages_and_required_fields():
    ledger = yaml.safe_load(
        Path("artifacts/pool-rereview-precommit-2026-08-22.yaml").read_text(
            encoding="utf-8"
        )
    )
    members = ledger["candidate_pool"]["members"]
    rows = 200
    index = pd.Index(np.arange(rows), name="id")
    rng = np.random.default_rng(339)
    labels = pd.Series(np.tile([0, 1], rows // 2), index=index)
    predictions = pd.DataFrame(
        {member: rng.random(rows) for member in members}, index=index
    )
    context = InputContext(
        predictions=predictions,
        labels=labels,
        folds=pd.Series(np.arange(rows) % 5, index=index),
        missingness_bands=pd.Series(np.arange(rows) % 3, index=index),
        ledger=ledger,
        baseline={},
        source_hashes={},
        prediction_file_sha256="0" * 64,
        member_prediction_sha256={member: "0" * 64 for member in members},
    )
    evaluator = _DeterministicEvaluator(tuple(ledger["strategies"]["included"]))

    result = run_split("outer-0", 0, context, evaluator, -0.000027669802)

    assert result.order["stage_1"][0] == sorted(
        ledger["lineage_groups"]["exp025_constrained_impute"]["members"]
    )
    assert {record["stage"] for record in result.trajectory} == {1, 3, 4}
    required = set(ledger["outputs"]["decision"]["contrast_required"])
    assert all(required <= set(record) for record in result.trajectory)
