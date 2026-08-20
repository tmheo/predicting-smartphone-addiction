"""정확값 목표 인코딩의 내부 OOF와 평활 회귀 시험."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import ExactValueTargetEncoder


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(8),
            "value": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "addicted_label": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )


def test_smoothed_table_uses_fit_global_mean_and_unknown_fallback():
    encoder = ExactValueTargetEncoder(cols=["value"], inner_folds=2, smoothing=2)
    encoder.fit(_frame(), seed=7)

    probe = pd.DataFrame({"id": [100, 101, 102], "value": ["a", "b", "unknown"]})
    encoded = encoder.transform(probe)["value_te"]

    assert encoded.tolist() == pytest.approx([5 / 6, 1 / 6, 1 / 2])


def test_fit_rows_receive_smoothed_inner_oof_values():
    frame = _frame()
    encoder = ExactValueTargetEncoder(cols=["value"], inner_folds=2, smoothing=2)
    encoder.fit(frame, seed=7)

    encoded = encoder.transform(frame)["value_te"]

    assert encoded[frame["value"] == "a"].tolist() == pytest.approx([3 / 4] * 4)
    assert encoded[frame["value"] == "b"].tolist() == pytest.approx([1 / 4] * 4)
    assert not np.allclose(encoded.to_numpy(), [5 / 6] * 4 + [1 / 6] * 4)


@pytest.mark.parametrize("smoothing", [-1, np.inf, np.nan])
def test_rejects_invalid_smoothing(smoothing: float):
    with pytest.raises(ValueError, match="smoothing"):
        ExactValueTargetEncoder(cols=["value"], smoothing=smoothing)
