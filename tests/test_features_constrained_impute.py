"""제약 결측 재구성 제공자 테스트. (#74)

- 화면 블록 열이 imputer 입력에 없으면 적재 거부.
- 관측 셀은 원시 값 그대로(원시 열을 덮지 않는 병행 열).
- 클리핑 골든: daily 결측은 관측 성분 합 하한, 성분 결측은 [0, slack] 양측 경계.
- 폭 열: 결측이었던 성분 셀에만 slack, daily 결측 행은 NaN(상한 없음).
- 실데이터 성질: fit-transform 산출이 항상 실현 가능 구간 안에 있다.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from pipeline.features import SCREEN_PARTS, SCREEN_TOTAL, ConstrainedImputeAux

COLS = [SCREEN_TOTAL, *SCREEN_PARTS, "sleep_hours"]


def make_df() -> pd.DataFrame:
    # 행 0: 전관측. 행 1: daily 결측. 행 2: social 결측(daily 관측).
    # 행 3: social·gaming 결측(daily 관측, slack 공유). 행 4: daily·social 결측(상한 없음).
    return pd.DataFrame(
        {
            SCREEN_TOTAL: [8.0, np.nan, 6.0, 6.0, np.nan],
            "social_media_hours": [2.0, 3.0, np.nan, np.nan, np.nan],
            "gaming_hours": [1.0, 2.0, 1.0, np.nan, 1.0],
            "work_study_hours": [3.0, 1.5, 4.5, 2.0, 2.0],
            "sleep_hours": [7.0, 8.0, 6.5, 7.5, 7.0],
        }
    )


class FakeImputer:
    """클리핑 경로를 결정적으로 검증하기 위한 고정 추정치 대역."""

    def __init__(self, est: pd.DataFrame) -> None:
        self.est = est

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.est[list(X.columns)].to_numpy()


def make_provider(est: pd.DataFrame) -> ConstrainedImputeAux:
    provider = ConstrainedImputeAux(cols=COLS)
    provider.imputer_ = FakeImputer(est)
    return provider


def test_screen_block_must_be_in_cols():
    with pytest.raises(ValueError, match="화면 블록"):
        ConstrainedImputeAux(cols=["sleep_hours", SCREEN_TOTAL])


def test_declared_columns_match_output():
    df = make_df()
    provider = make_provider(df.ffill().bfill())
    out = provider.transform(df)
    assert list(out.columns) == provider.columns()


def test_observed_cells_keep_raw_values():
    df = make_df()
    provider = make_provider(df.ffill().bfill())
    out = provider.transform(df)
    for c in [SCREEN_TOTAL, *SCREEN_PARTS]:
        observed = df[c].notna()
        assert (out.loc[observed, f"{c}_recon"] == df.loc[observed, c]).all()


def test_daily_estimate_is_clipped_to_observed_parts_sum():
    df = make_df()
    est = df.copy()
    # 행 1의 관측 성분 합은 3+2+1.5=6.5인데 추정치 5.0은 규칙 위반 → 6.5로 클리핑.
    est.loc[1, SCREEN_TOTAL] = 5.0
    out = make_provider(est.ffill().bfill()).transform(df)
    assert out.loc[1, f"{SCREEN_TOTAL}_recon"] == 6.5


def test_part_estimate_is_clipped_into_zero_slack_interval():
    df = make_df()
    est = df.ffill().bfill()
    # 행 2의 slack은 6-(1+4.5)=0.5. 추정치 3.0은 상한 위반 → 0.5로 클리핑.
    est.loc[2, "social_media_hours"] = 3.0
    # 행 4는 daily 결측이라 상한이 없다. 음수 추정치는 하한 0으로만 클리핑.
    est.loc[4, "social_media_hours"] = -1.0
    out = make_provider(est).transform(df)
    assert out.loc[2, "social_media_hours_recon"] == 0.5
    assert out.loc[4, "social_media_hours_recon"] == 0.0


def test_shared_slack_bounds_each_missing_part():
    df = make_df()
    est = df.ffill().bfill()
    # 행 3의 slack은 6-2=4. 두 결측 성분 각각 4를 상한으로 잘린다.
    est.loc[3, "social_media_hours"] = 9.0
    est.loc[3, "gaming_hours"] = 5.0
    out = make_provider(est).transform(df)
    assert out.loc[3, "social_media_hours_recon"] == 4.0
    assert out.loc[3, "gaming_hours_recon"] == 4.0


def test_width_is_slack_only_on_reconstructed_cells():
    df = make_df()
    out = make_provider(df.ffill().bfill()).transform(df)
    width = out["social_media_hours_recon_width"]
    assert np.isnan(width[0])  # 관측 셀: 재구성 없음
    assert width[2] == 0.5  # 결측 + daily 관측: slack
    assert width[3] == 4.0
    assert np.isnan(width[4])  # 결측 + daily 결측: 상한 없음
    assert out["work_study_hours_recon_width"].isna().all()  # 전행 관측
    assert f"{SCREEN_TOTAL}_recon_width" not in out.columns  # daily 폭 열은 없다


def test_widths_false_emits_recon_columns_only():
    df = make_df()
    provider = ConstrainedImputeAux(cols=COLS, widths=False)
    provider.imputer_ = FakeImputer(df.ffill().bfill())
    out = provider.transform(df)
    assert list(out.columns) == [f"{c}_recon" for c in [SCREEN_TOTAL, *SCREEN_PARTS]]
    assert list(out.columns) == provider.columns()


def test_emit_width_subset_keeps_same_state_and_drops_recon_columns():
    df = make_df()
    estimate = df.ffill().bfill()
    full = ConstrainedImputeAux(cols=COLS)
    full.imputer_ = FakeImputer(estimate)
    widths = [f"{c}_recon_width" for c in SCREEN_PARTS]
    reduced = ConstrainedImputeAux(cols=COLS, emit=widths)
    reduced.imputer_ = FakeImputer(estimate)

    assert reduced.columns() == widths
    pd.testing.assert_frame_equal(full.transform(df)[widths], reduced.transform(df))
    assert not any(c.endswith("_recon") for c in reduced.transform(df).columns)


@pytest.mark.parametrize(
    "emit, match",
    [
        ([], "비어 있지 않은"),
        (
            ["gaming_hours_recon_width", "gaming_hours_recon_width"],
            "중복 없는",
        ),
        (["daily_screen_time_hours_recon_width"], "알 수 없는 열"),
        (["gaming_hours_recon_width"], "알 수 없는 열"),
    ],
)
def test_emit_rejects_invalid_output_subset(emit, match):
    widths = emit != ["gaming_hours_recon_width"]
    with pytest.raises(ValueError, match=match):
        ConstrainedImputeAux(cols=COLS, widths=widths, emit=emit)


def test_fit_transform_respects_feasible_intervals():
    rng = np.random.default_rng(0)
    n = 400
    parts = rng.uniform(0, 3, size=(n, 3))
    daily = parts.sum(axis=1) + rng.uniform(0, 2, size=n)
    df = pd.DataFrame(
        {
            SCREEN_TOTAL: daily,
            "social_media_hours": parts[:, 0],
            "gaming_hours": parts[:, 1],
            "work_study_hours": parts[:, 2],
            "sleep_hours": rng.uniform(5, 9, size=n),
        }
    )
    mask = rng.uniform(size=df.shape) < 0.2
    df = df.mask(pd.DataFrame(mask, columns=df.columns))

    provider = ConstrainedImputeAux(cols=COLS)
    # 무작위 소형 표본은 제한 반복 안에 수렴하지 않을 수 있다.
    # 이 테스트는 대체값의 정확도가 아니라 산술 경계 클리핑을 검증한다.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        provider.fit(df, seed=42)
    out = provider.transform(df)

    obs_sum = df[SCREEN_PARTS].sum(axis=1)
    slack = (df[SCREEN_TOTAL] - obs_sum).clip(lower=0.0)
    daily_missing = df[SCREEN_TOTAL].isna()
    assert (
        out.loc[daily_missing, f"{SCREEN_TOTAL}_recon"] >= obs_sum[daily_missing] - 1e-9
    ).all()
    for c in SCREEN_PARTS:
        m = df[c].isna()
        assert (out.loc[m, f"{c}_recon"] >= -1e-9).all()
        bounded = m & df[SCREEN_TOTAL].notna()
        assert (out.loc[bounded, f"{c}_recon"] <= slack[bounded] + 1e-9).all()
    for c in [SCREEN_TOTAL, *SCREEN_PARTS]:
        assert out[f"{c}_recon"].notna().all()
