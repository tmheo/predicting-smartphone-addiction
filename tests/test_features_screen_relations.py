"""화면 관계 7특성 블록(#181)의 값과 결측 정책 검증.

계약(public-stack-provenance 3순위, beicicc fixed4000 README):
- 차이 3개는 NaN을 자연 전파한다.
- 비율 4개는 분자·분모가 유한하고 분모가 양수일 때만 정의하며,
  epsilon·대체·클리핑·정의 여부 플래그를 쓰지 않는다.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.features import DERIVED_REGISTRY, DerivedColumns

SCREEN_RELATION_NAMES = [
    "gaming_minus_work",
    "screen_minus_work",
    "weekend_minus_daily",
    "social_share_screen",
    "gaming_share_screen",
    "work_share_screen",
    "screen_to_sleep",
]


def make_df() -> pd.DataFrame:
    # 행: 0 전관측, 1 분모 0, 2 분모 음수, 3 분모 결측, 4 분자 결측, 5 sleep 0
    return pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0, 0.0, -1.0, np.nan, 8.0, 4.0],
            "social_media_hours": [2.0, 1.0, 1.0, 1.0, np.nan, 1.0],
            "gaming_hours": [1.5, 1.0, 1.0, 1.0, 1.0, 1.0],
            "work_study_hours": [4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "sleep_hours": [6.0, 6.0, 6.0, 6.0, 6.0, 0.0],
            "weekend_screen_time": [9.0, 1.0, 1.0, 1.0, np.nan, 1.0],
        }
    )


def test_registry_has_screen_relation_block() -> None:
    assert set(SCREEN_RELATION_NAMES) <= set(DERIVED_REGISTRY)


def test_differences_propagate_nan_naturally() -> None:
    out = DerivedColumns(SCREEN_RELATION_NAMES).compute(make_df())
    assert out.loc[0, "gaming_minus_work"] == pytest.approx(1.5 - 4.0)
    assert out.loc[0, "screen_minus_work"] == pytest.approx(8.0 - 4.0)
    assert out.loc[0, "weekend_minus_daily"] == pytest.approx(9.0 - 8.0)
    # 항 하나가 결측이면 결과도 결측이다.
    assert np.isnan(out.loc[3, "screen_minus_work"])
    assert np.isnan(out.loc[3, "weekend_minus_daily"])
    assert np.isnan(out.loc[4, "weekend_minus_daily"])
    # 차이는 분모 조건과 무관하게 관측 행에서 정의된다.
    assert out.loc[1, "gaming_minus_work"] == pytest.approx(0.0)


def test_ratios_defined_only_for_finite_positive_denominator() -> None:
    out = DerivedColumns(SCREEN_RELATION_NAMES).compute(make_df())
    assert out.loc[0, "social_share_screen"] == pytest.approx(2.0 / 8.0)
    assert out.loc[0, "gaming_share_screen"] == pytest.approx(1.5 / 8.0)
    assert out.loc[0, "work_share_screen"] == pytest.approx(4.0 / 8.0)
    assert out.loc[0, "screen_to_sleep"] == pytest.approx(8.0 / 6.0)
    # 분모 0, 음수, 결측이면 미정의(NaN)이고 무한대나 대체값을 만들지 않는다.
    for row in (1, 2, 3):
        assert np.isnan(out.loc[row, "social_share_screen"])
    assert np.isnan(out.loc[5, "screen_to_sleep"])
    # 분자 결측이면 미정의다.
    assert np.isnan(out.loc[4, "social_share_screen"])
    # 어떤 행에서도 무한대가 나오지 않는다(클리핑·epsilon 없이 마스크만 쓴다).
    assert np.isfinite(out.to_numpy()[~np.isnan(out.to_numpy())]).all()
