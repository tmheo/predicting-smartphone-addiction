import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce_s6e8_097110.py"
SPEC = importlib.util.spec_from_file_location("reproduce_s6e8_097110", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reproduction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reproduction
SPEC.loader.exec_module(reproduction)

candidate_predictions = reproduction.candidate_predictions
deterministic_rank = reproduction.deterministic_rank
leaderboard_history = reproduction.leaderboard_history
percentile_rank = reproduction.percentile_rank


def test_percentile_rank_uses_average_ties_and_half_step_scaling() -> None:
    ranked = percentile_rank(np.asarray([1.0, 1.0, 3.0]))

    np.testing.assert_allclose(ranked, [1 / 3, 1 / 3, 5 / 6])


def test_deterministic_rank_uses_secondary_values_for_ties() -> None:
    ranked = deterministic_rank(
        np.asarray([1.0, 1.0, 0.0]),
        np.asarray([2.0, 1.0, 0.0]),
    )

    np.testing.assert_allclose(ranked, [5 / 6, 1 / 2, 1 / 6])


def test_candidate_predictions_keep_the_documented_evaluation_order() -> None:
    teacher = np.asarray([0.1, 0.4])
    student = np.asarray([0.9, 0.2])
    weights = np.asarray([-0.08, 0.12])

    predictions = candidate_predictions(teacher, student, weights)

    expected = np.asarray(
        [
            1.08 * teacher - 0.08 * student,
            0.88 * teacher + 0.12 * student,
        ]
    )
    np.testing.assert_array_equal(predictions, expected)


def test_leaderboard_history_reports_team_level_limit_without_submission_ids() -> None:
    board = pd.DataFrame(
        {
            "episode": ["S6E2"] * 12,
            "is_host_baseline": [False] * 12,
            "public_rank": np.arange(1, 13),
            "private_rank": np.arange(12, 0, -1),
        }
    )

    history = leaderboard_history(board)

    assert history["all_episode_survivors"] == 8
    assert history["auc_episode_survivors"] == 8
    assert history["same_submission_control_available"] is False
