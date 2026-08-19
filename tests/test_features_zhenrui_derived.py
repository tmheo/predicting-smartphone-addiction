"""zhenrui 파생 수치 16열 블록(#258)의 값과 결측 정책 검증.

계약(kernel 129907249, SHA-256 ef16bb88…b116)과 이식 규율:
- 합·차·로그·스트레스 상호작용은 NaN을 자연 전파한다(원본의 중앙값 대체를 쓰지 않는다).
- 비율은 분자·분모가 유한하고 분모가 양수일 때만 정의한다(원본의 +1e-6 대신 마스크).
- sleep_deficit은 9시간 기준 부족분을 0 하한으로 자르고 결측은 유지한다.
- 스트레스는 Low=0, Medium=1, High=2로 서수화하고 결측·미지 값은 NaN이다.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.features import DERIVED_REGISTRY, ZHENRUI_DERIVED_NAMES, DerivedColumns


def make_df() -> pd.DataFrame:
    # 행: 0 전관측, 1 분모 0, 2 수치 결측, 3 스트레스 결측, 4 수면 과다, 5 High 스트레스
    return pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0, 6.0, np.nan, 4.0, 2.0, 5.0],
            "social_media_hours": [2.0, 1.0, 1.0, 1.0, 1.0, 2.5],
            "gaming_hours": [1.5, 1.0, np.nan, 1.0, 0.5, 1.0],
            "work_study_hours": [4.0, 1.0, 1.0, 1.0, 6.0, 1.0],
            "sleep_hours": [6.0, 7.0, 8.0, np.nan, 10.0, 5.5],
            "notifications_per_day": [120.0, 80.0, np.nan, 50.0, 0.0, 90.0],
            "app_opens_per_day": [40.0, 0.0, 30.0, 20.0, 10.0, 60.0],
            "weekend_screen_time": [9.0, 5.0, 3.0, np.nan, 2.0, 6.0],
            "stress_level": ["Medium", "Low", "Low", np.nan, "Low", "High"],
        }
    )


def test_registry_has_zhenrui_block() -> None:
    assert len(ZHENRUI_DERIVED_NAMES) == 16
    assert set(ZHENRUI_DERIVED_NAMES) <= set(DERIVED_REGISTRY)


def test_sums_and_diffs_propagate_nan_naturally() -> None:
    out = DerivedColumns(ZHENRUI_DERIVED_NAMES).compute(make_df())
    assert out.loc[0, "total_screen"] == pytest.approx(8.0 + 9.0)
    assert out.loc[0, "activity_total"] == pytest.approx(2.0 + 1.5 + 4.0)
    assert out.loc[0, "engagement_total"] == pytest.approx(120.0 + 40.0)
    # 항 하나가 결측이면 결과도 결측이다(관측 성분만 합치지 않는다).
    assert np.isnan(out.loc[2, "total_screen"])
    assert np.isnan(out.loc[3, "total_screen"])
    assert np.isnan(out.loc[2, "activity_total"])
    assert np.isnan(out.loc[2, "engagement_total"])


def test_ratios_defined_only_for_finite_positive_denominator() -> None:
    out = DerivedColumns(ZHENRUI_DERIVED_NAMES).compute(make_df())
    assert out.loc[0, "activity_share_screen"] == pytest.approx(7.5 / 8.0)
    assert out.loc[0, "notif_per_app"] == pytest.approx(120.0 / 40.0)
    assert out.loc[0, "screen_per_app"] == pytest.approx(8.0 * 60.0 / 40.0)
    # 분모 0이면 미정의(NaN)이고 무한대나 대체값을 만들지 않는다.
    assert np.isnan(out.loc[1, "notif_per_app"])
    assert np.isnan(out.loc[1, "screen_per_app"])
    # 분자 성분이 결측이면 미정의다.
    assert np.isnan(out.loc[2, "activity_share_screen"])
    assert np.isnan(out.loc[2, "notif_per_app"])
    # 어떤 행에서도 무한대가 나오지 않는다.
    values = out.to_numpy(dtype=float)
    assert np.isfinite(values[~np.isnan(values)]).all()


def test_log_transforms() -> None:
    out = DerivedColumns(ZHENRUI_DERIVED_NAMES).compute(make_df())
    assert out.loc[0, "log_notifications"] == pytest.approx(np.log1p(120.0))
    assert out.loc[4, "log_notifications"] == pytest.approx(0.0)
    assert out.loc[0, "log_app_opens"] == pytest.approx(np.log1p(40.0))
    assert np.isnan(out.loc[2, "log_notifications"])


def test_sleep_deficit_clips_at_zero_and_keeps_nan() -> None:
    out = DerivedColumns(ZHENRUI_DERIVED_NAMES).compute(make_df())
    assert out.loc[0, "sleep_deficit"] == pytest.approx(3.0)
    # 9시간 초과 수면은 부족분 0이다(음수를 만들지 않는다).
    assert out.loc[4, "sleep_deficit"] == pytest.approx(0.0)
    assert np.isnan(out.loc[3, "sleep_deficit"])


def test_stress_interactions_are_ordinal_and_propagate_nan() -> None:
    out = DerivedColumns(ZHENRUI_DERIVED_NAMES).compute(make_df())
    # Medium=1, Low=0, High=2.
    assert out.loc[0, "screen_x_stress"] == pytest.approx(8.0 * 1.0)
    assert out.loc[1, "screen_x_stress"] == pytest.approx(0.0)
    assert out.loc[5, "screen_x_stress"] == pytest.approx(5.0 * 2.0)
    assert out.loc[5, "social_x_stress"] == pytest.approx(2.5 * 2.0)
    assert out.loc[5, "sleep_x_stress"] == pytest.approx(5.5 * 2.0)
    # 스트레스 결측이면 원본의 missing→1 대체 대신 NaN을 전파한다.
    assert np.isnan(out.loc[3, "screen_x_stress"])
    assert np.isnan(out.loc[3, "social_x_stress"])
    # 수치 쪽 결측도 전파된다.
    assert np.isnan(out.loc[2, "screen_x_stress"])
