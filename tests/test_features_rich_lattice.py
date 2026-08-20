"""지정 격자 키 OOF TE와 셀 개수 제공자 테스트. (#266)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import PLACEBO, RichLatticeEncoder

RAW = ["daily", "social", "stress"]
NUMERIC = ["daily", "social"]
PAIRS = [["daily", "social"], ["stress", "daily"]]


def make_train(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "daily": np.tile([2.14, 2.18, 3.91], n // 3),
            "social": np.tile([0.21, 0.29, 1.11], n // 3),
            "stress": np.tile(["Low", "Low", "High"], n // 3),
            PLACEBO: np.linspace(-1.5, 1.5, n),
            "addicted_label": np.tile([0, 1], n // 2),
        }
    )


def make_encoder(**overrides: object) -> RichLatticeEncoder:
    params: dict[str, object] = {
        "raw_cols": RAW,
        "numeric_cols": NUMERIC,
        "pairs": PAIRS,
        "inner_folds": 2,
        "smoothing": 10.0,
    }
    params.update(overrides)
    return RichLatticeEncoder(**params)


def test_declares_source_key_block_and_placebo_canary() -> None:
    enc = make_encoder()
    assert enc.columns() == [
        "daily_rich_freq",
        "social_rich_freq",
        "stress_rich_freq",
        "daily_rich_r1_te",
        "daily_rich_r1_freq",
        "daily_rich_fl_te",
        "daily_rich_fl_freq",
        "social_rich_r1_te",
        "social_rich_r1_freq",
        "social_rich_fl_te",
        "social_rich_fl_freq",
        "daily__social_rich_pair_te",
        "daily__social_rich_pair_freq",
        "stress__daily_rich_pair_te",
        "stress__daily_rich_pair_freq",
        f"{PLACEBO}__daily_rich_pair_te",
        f"{PLACEBO}__daily_rich_pair_freq",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"raw_cols": []},
        {"raw_cols": ["daily", "daily"]},
        {"raw_cols": [PLACEBO, "daily"]},
        {"numeric_cols": []},
        {"numeric_cols": ["daily", "daily"]},
        {"numeric_cols": ["unknown"]},
        {"pairs": [["daily"]]},
        {"pairs": [["daily", "daily"]]},
        {"pairs": [["daily", "unknown"]]},
        {"pairs": [["daily", "social"], ["social", "daily"]]},
        {"inner_folds": 1},
        {"smoothing": 0.0},
    ],
)
def test_rejects_invalid_construction(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_encoder(**overrides)


def test_validation_rows_get_smoothed_te_and_full_fit_counts() -> None:
    train = make_train()
    enc = make_encoder()
    enc.fit(train, seed=0)
    row = pd.DataFrame(
        {"id": [999], "daily": [2.19], "social": [0.28], "stress": ["Low"], PLACEBO: [0.0]}
    )
    out = enc.transform(row)

    cell = np.floor(train["daily"]) == 2.0
    count = float(cell.sum())
    positive = float(train.loc[cell, "addicted_label"].sum())
    global_mean = float(train["addicted_label"].mean())
    expected = (positive + 10.0 * global_mean) / (count + 10.0)
    assert out["daily_rich_fl_te"].iloc[0] == pytest.approx(expected)
    assert out["daily_rich_fl_freq"].iloc[0] == count
    assert out["stress_rich_freq"].iloc[0] == float((train["stress"] == "Low").sum())


def test_fit_rows_get_inner_oof_te_and_counts() -> None:
    train = pd.DataFrame(
        {
            "id": np.arange(8),
            "daily": [2.1] * 8,
            "social": [0.5] * 8,
            "stress": ["Low"] * 8,
            PLACEBO: [0.1] * 8,
            "addicted_label": [0, 1] * 4,
        }
    )
    enc = make_encoder()
    enc.fit(train, seed=0)
    out = enc.transform(train)
    assert (out["daily_rich_freq"] == 4.0).all()
    assert (out["daily_rich_fl_freq"] == 4.0).all()
    assert (out["daily__social_rich_pair_freq"] == 4.0).all()
    assert np.isfinite(out.to_numpy()).all()


def test_r1_floor_and_mixed_pair_have_distinct_key_semantics() -> None:
    train = make_train()
    enc = make_encoder()
    enc.fit(train, seed=0)
    assert len(enc.freq_tables_["daily_rich_r1"]) == 3
    assert len(enc.freq_tables_["daily_rich_fl"]) == 2
    assert len(enc.freq_tables_["stress__daily_rich_pair"]) == 2


def test_unknown_and_missing_keys_use_safe_fallbacks() -> None:
    train = make_train()
    train.loc[0, "social"] = np.nan
    enc = make_encoder()
    enc.fit(train, seed=0)
    rows = pd.DataFrame(
        {
            "id": [998, 999],
            "daily": [99.0, 2.1],
            "social": [99.0, np.nan],
            "stress": ["Unknown", "Low"],
            PLACEBO: [99.0, 0.0],
        }
    )
    out = enc.transform(rows)
    assert out["daily_rich_fl_te"].iloc[0] == pytest.approx(enc.global_mean_)
    assert out["daily_rich_fl_freq"].iloc[0] == 0.0
    assert out["social_rich_freq"].iloc[1] == 1.0
    assert list(out.columns) == enc.columns()
    assert (out.dtypes == "float64").all()
