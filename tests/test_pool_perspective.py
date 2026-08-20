"""후보 풀 정보 관점 고정과 제거 대조 계약. (#310)"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.pool_perspective import (
    DEFAULT_MAP_PATH,
    EvaluationSummary,
    FoldSummary,
    FrozenPerspectiveMap,
    MemberPerspective,
    PerspectiveComparison,
    PerspectiveDefinition,
    PerspectiveDiagnosticError,
    compare_perspective,
    diagnostic_combiners,
    load_frozen_map,
    member_selection,
)
from pipeline.ensemble import BaggedGreedyRankMeanCombiner


POOL_SHA256 = "290d96ac719b737cdfbfd8d1e6ee19ce621b472410c528a43bb4515d2eb3ec38"


def make_evaluation(
    name: str,
    nested_auc: float,
    fold_aucs: list[float],
    weights: list[dict[str, float]],
) -> EvaluationSummary:
    return EvaluationSummary(
        name=name,
        nested_auc=nested_auc,
        folds=tuple(
            FoldSummary(fold, auc, summary)
            for fold, (auc, summary) in enumerate(zip(fold_aucs, weights, strict=True))
        ),
    )


def test_repository_map_freezes_current_pool_and_model_families():
    frozen = load_frozen_map(DEFAULT_MAP_PATH)

    assert frozen.member_count == 29
    assert frozen.pool_sha256 == POOL_SHA256
    assert len(frozen.perspectives) == 14
    assert frozen.members[0].config == "exp006_te_drop_gaming"
    assert frozen.members[-1].config == "exp135_xgb_hpo_trial30"
    assert {member.primary for member in frozen.members} == set(frozen.perspectives)


def test_repository_map_rejects_changed_pool_before_reading_results(tmp_path: Path):
    changed_pool = tmp_path / "pool.yaml"
    changed_pool.write_bytes(Path("artifacts/pool.yaml").read_bytes() + b"\n")
    record = DEFAULT_MAP_PATH.read_text().replace(
        "path: artifacts/pool.yaml", f"path: {changed_pool}"
    )
    map_path = tmp_path / "map.yaml"
    map_path.write_text(record)

    with pytest.raises(PerspectiveDiagnosticError, match="후보 풀 내용 해시 변경"):
        load_frozen_map(map_path, pool_path=changed_pool)


def test_diagnostic_worker_override_preserves_registered_strategy_names():
    combiners = diagnostic_combiners(7)

    bagged = [
        combiner
        for combiner in combiners
        if isinstance(combiner, BaggedGreedyRankMeanCombiner)
    ]
    assert len(bagged) == 1
    assert bagged[0].workers == 7
    assert bagged[0].bags == 50
    assert bagged[0].name == "bagged_greedy_rank_mean"


def test_member_selection_aggregates_outer_fold_weights():
    evaluation = make_evaluation(
        "baseline",
        0.9,
        [0.9] * 5,
        [
            {"a": 0.5, "b": 0.0},
            {"a": 0.4, "b": 0.1},
            {"a": 0.0, "b": 0.2},
            {"a": 0.3, "b": 0.0},
            {"a": 0.2, "b": 0.3},
        ],
    )

    selection = member_selection(evaluation)

    assert selection["a"].selected == 4
    assert selection["a"].mean_weight == pytest.approx(0.28)
    assert selection["b"].selected == 3
    assert selection["b"].mean_weight == pytest.approx(0.12)


def test_perspective_comparison_applies_all_three_opening_conditions():
    definition = PerspectiveDefinition("view_a", "관점 A", "설명")
    frozen = FrozenPerspectiveMap(
        pool_path=Path("pool.yaml"),
        pool_sha256="pool",
        member_count=3,
        perspectives={"view_a": definition},
        members=(
            MemberPerspective("a", "run-a", "family-a", "view_a", ()),
            MemberPerspective("b", "run-b", "family-b", "other", ()),
            MemberPerspective("c", "run-c", "family-c", "other", ()),
        ),
        sha256="map",
    )
    baseline = make_evaluation(
        "full_best",
        0.90005,
        [0.91, 0.90, 0.89, 0.92, 0.88],
        [{"a": 0.2, "b": 0.3, "c": 0.5}] * 5,
    )
    removed = make_evaluation(
        "removed_best",
        0.90002,
        [0.90, 0.89, 0.88, 0.93, 0.89],
        [{"b": 0.4, "c": 0.6}] * 5,
    )

    comparison = compare_perspective(frozen, "view_a", baseline, removed)

    assert isinstance(comparison, PerspectiveComparison)
    assert comparison.loss == pytest.approx(0.00003)
    assert comparison.outer_worse == 3
    assert comparison.outer_better == 2
    assert comparison.opens_experiment
    assert comparison.excluded_selection[0][0] == "a"
    assert {shift.config for shift in comparison.shifts} == {"b", "c"}


@pytest.mark.parametrize(
    ("removed_auc", "folds", "families"),
    [
        (0.90004, [0.90, 0.89, 0.88, 0.93, 0.89], ("family-a",)),
        (0.90002, [0.90, 0.89, 0.90, 0.93, 0.89], ("family-a",)),
        (0.90002, [0.90, 0.89, 0.88, 0.93, 0.89], ("family-a", "family-b")),
    ],
)
def test_perspective_comparison_requires_delta_folds_and_single_family(
    removed_auc: float, folds: list[float], families: tuple[str, ...]
):
    members = tuple(
        MemberPerspective(
            chr(ord("a") + index),
            f"run-{index}",
            family,
            "view_a" if index < len(families) else "other",
            (),
        )
        for index, family in enumerate((*families, "remaining"))
    )
    frozen = FrozenPerspectiveMap(
        pool_path=Path("pool.yaml"),
        pool_sha256="pool",
        member_count=len(members),
        perspectives={"view_a": PerspectiveDefinition("view_a", "관점 A", "설명")},
        members=members,
        sha256="map",
    )
    configs = [member.config for member in members]
    baseline = make_evaluation(
        "full",
        0.90005,
        [0.91, 0.90, 0.89, 0.92, 0.88],
        [{config: 1.0 / len(configs) for config in configs}] * 5,
    )
    remaining = [member.config for member in members if member.primary != "view_a"]
    removed = make_evaluation(
        "removed",
        removed_auc,
        folds,
        [{config: 1.0 / len(remaining) for config in remaining}] * 5,
    )

    comparison = compare_perspective(frozen, "view_a", baseline, removed)

    assert not comparison.opens_experiment
