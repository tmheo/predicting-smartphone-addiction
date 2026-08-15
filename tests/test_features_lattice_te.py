"""격자 쌍 TE 블록 제공자 테스트. (#75)

- 선언: 전 쌍 전개 순서, 해상도별 접미어, placebo 쌍 카나리아 자동 포함.
- 적재 거부: 중복·부족 컬럼, 미지 해상도, placebo 수동 포함, 잘못된 평활.
- 골든: 검증/test 행의 평활 TE와 개수 열 값, 미지 셀의 전체 평균·0 대입.
- 누수 방어: 학습 fold 행은 자기 행이 빠진 내부 OOF 값을 받는다.
- 해상도 의미: floor는 정수 내림 셀, r1은 소수 첫째 자리 반올림 셀.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import PLACEBO, LatticePairTargetEncoder

COLS = ["daily_screen_time_hours", "social_media_hours", "gaming_hours"]


def make_train(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": rng.uniform(1, 4, n).round(1),
            "social_media_hours": rng.uniform(0, 3, n).round(1),
            "gaming_hours": rng.uniform(0, 3, n).round(1),
            PLACEBO: rng.normal(size=n),
            "addicted_label": np.tile([0, 1], n // 2),
        }
    )
    df.loc[::7, "social_media_hours"] = np.nan
    return df


def test_declares_all_pairs_with_canary_and_resolution_suffixes() -> None:
    enc = LatticePairTargetEncoder(cols=COLS, resolutions=["floor", "r1"])
    stems = [
        "daily_screen_time_hours__social_media_hours",
        "daily_screen_time_hours__gaming_hours",
        "social_media_hours__gaming_hours",
        f"{PLACEBO}__daily_screen_time_hours",
    ]
    expected = [
        f"{stem}_{sfx}_{kind}"
        for sfx in ("latf", "latr1")
        for stem in stems
        for kind in ("te", "ct")
    ]
    assert enc.columns() == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cols": ["a"]},
        {"cols": ["a", "a"]},
        {"cols": ["a", PLACEBO]},
        {"cols": COLS, "resolutions": ["exact"]},
        {"cols": COLS, "resolutions": []},
        {"cols": COLS, "resolutions": ["floor", "floor"]},
        {"cols": COLS, "inner_folds": 1},
        {"cols": COLS, "smoothing": 0.0},
    ],
)
def test_rejects_invalid_construction(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LatticePairTargetEncoder(**kwargs)


def fit_small(
    resolutions: list[str] | None = None,
) -> tuple[LatticePairTargetEncoder, pd.DataFrame]:
    train = make_train()
    resolutions = ["floor"] if resolutions is None else resolutions
    enc = LatticePairTargetEncoder(cols=COLS, resolutions=resolutions, inner_folds=2)
    enc.fit(train, seed=0)
    return enc, train


def test_transform_declared_columns_match_actual() -> None:
    enc, train = fit_small(resolutions=["floor", "r1"])
    out = enc.transform(train)
    assert list(out.columns) == enc.columns()
    assert (out.dtypes == "float64").all()


def test_valid_rows_get_smoothed_table_values_and_counts() -> None:
    enc, train = fit_small()
    # fit 행이 아닌 새 행: floor 셀 (2, 1)에 대한 전체 표 값을 받아야 한다.
    row = pd.DataFrame(
        {
            "id": [999],
            "daily_screen_time_hours": [2.3],
            "social_media_hours": [1.9],
            "gaming_hours": [0.5],
            PLACEBO: [0.0],
        }
    )
    y = train["addicted_label"]
    cell = (np.floor(train["daily_screen_time_hours"]) == 2.0) & (
        np.floor(train["social_media_hours"]) == 1.0
    )
    n, s = int(cell.sum()), float(y[cell].sum())
    assert n > 0, "골든 셀이 비어 있으면 테스트 데이터가 잘못된 것"
    g = float(y.mean())
    expected_te = (s + 20.0 * g) / (n + 20.0)
    out = enc.transform(row)
    stem = "daily_screen_time_hours__social_media_hours_latf"
    assert out[f"{stem}_te"].iloc[0] == pytest.approx(expected_te)
    assert out[f"{stem}_ct"].iloc[0] == float(n)


def test_unknown_cell_falls_back_to_global_mean_and_zero_count() -> None:
    enc, train = fit_small()
    row = pd.DataFrame(
        {
            "id": [999],
            "daily_screen_time_hours": [99.0],
            "social_media_hours": [99.0],
            "gaming_hours": [99.0],
            PLACEBO: [0.0],
        }
    )
    out = enc.transform(row)
    stem = "daily_screen_time_hours__social_media_hours_latf"
    assert out[f"{stem}_te"].iloc[0] == pytest.approx(float(train["addicted_label"].mean()))
    assert out[f"{stem}_ct"].iloc[0] == 0.0


def test_fit_rows_get_inner_oof_not_table_values() -> None:
    # 모든 행이 같은 셀에 있고 라벨이 반반이면, 내부 OOF 값은 자기 행을 뺀 나머지로만
    # 만들어져 전체 표 값과 달라야 한다(자기 행 포함이면 모든 행이 같은 표 값을 받는다).
    n = 8
    train = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": [2.1] * n,
            "social_media_hours": [1.5] * n,
            "gaming_hours": [0.5] * n,
            PLACEBO: [0.1] * n,
            "addicted_label": [0, 1] * (n // 2),
        }
    )
    enc = LatticePairTargetEncoder(cols=COLS, resolutions=["floor"], inner_folds=2)
    enc.fit(train, seed=0)
    out = enc.transform(train)
    stem = "daily_screen_time_hours__social_media_hours_latf"
    # 전체 표의 셀 개수는 8인데 학습 fold 행이 받는 개수는 내부 fold(절반) 기준 4다.
    # 자기 행 포함(표 값 직접 매핑)이었다면 8이 나온다. 개수가 내부 OOF 경로의 증거다.
    assert float(enc.ct_tables_[stem].iloc[0]) == 8.0
    assert (out[f"{stem}_ct"] == 4.0).all(), "학습 fold 행의 개수는 내부 fold 기준이어야 한다"


def test_r1_resolution_separates_cells_that_floor_merges() -> None:
    # floor로는 같은 셀(2.x), r1로는 다른 셀(2.1 vs 2.4)이 되는 두 그룹.
    n = 8
    train = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": [2.1, 2.4] * (n // 2),
            "social_media_hours": [1.0] * n,
            "gaming_hours": [0.5] * n,
            PLACEBO: [0.1] * n,
            "addicted_label": [0, 1] * (n // 2),
        }
    )
    enc = LatticePairTargetEncoder(cols=COLS, resolutions=["floor", "r1"], inner_folds=2)
    enc.fit(train, seed=0)
    stem_f = "daily_screen_time_hours__social_media_hours_latf"
    stem_r = "daily_screen_time_hours__social_media_hours_latr1"
    assert len(enc.ct_tables_[stem_f]) == 1, "floor 셀은 하나로 합쳐진다"
    assert len(enc.ct_tables_[stem_r]) == 2, "r1 셀은 소수 첫째 자리로 갈라진다"


def test_nan_forms_single_missing_bucket() -> None:
    enc, train = fit_small()
    stem = "daily_screen_time_hours__social_media_hours_latf"
    nan_keys = [k for k in enc.te_tables_[stem].index if "__nan__" in k]
    n_nan_cells = train["social_media_hours"].isna().sum()
    assert n_nan_cells > 0
    total = sum(enc.ct_tables_[stem].loc[k] for k in nan_keys)
    assert total == float(n_nan_cells)
