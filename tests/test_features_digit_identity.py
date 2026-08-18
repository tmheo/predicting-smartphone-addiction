"""정체성·자리수 특성 블록(#184)의 값과 결측 정책 검증.

계약(public-stack-provenance 5순위, beicicc identity_digit_contract):
- 수치 9열별로 round0/1/2, absdiff_round0/1/2, is_round0, is_round1, tenths, hundredths.
- 타깃을 쓰지 않고 결측은 자연 전파한다(지시자·자리 값도 결측 행은 NaN).
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    DERIVED_REGISTRY,
    DIGIT_IDENTITY_COLS,
    DIGIT_IDENTITY_NAMES,
    DerivedColumns,
)


def make_df() -> pd.DataFrame:
    # 행: 0 정수, 1 소수 1자리, 2 소수 2자리, 3 결측, 4 부동소수 위험값(2.9), 5 반자리(0.25)
    values = [3.0, 2.5, 1.37, np.nan, 2.9, 0.25]
    return pd.DataFrame({col: values for col in DIGIT_IDENTITY_COLS})


def test_registry_has_all_90_columns() -> None:
    assert len(DIGIT_IDENTITY_NAMES) == 90
    assert set(DIGIT_IDENTITY_NAMES) <= set(DERIVED_REGISTRY)


def test_rounding_resolutions_and_absdiff() -> None:
    out = DerivedColumns(DIGIT_IDENTITY_NAMES).compute(make_df())
    col = DIGIT_IDENTITY_COLS[0]
    assert out.loc[2, f"{col}_round0"] == pytest.approx(1.0)
    assert out.loc[2, f"{col}_round1"] == pytest.approx(1.4)
    assert out.loc[2, f"{col}_round2"] == pytest.approx(1.37)
    assert out.loc[2, f"{col}_absdiff_round0"] == pytest.approx(0.37)
    assert out.loc[2, f"{col}_absdiff_round1"] == pytest.approx(0.03)
    assert out.loc[2, f"{col}_absdiff_round2"] == pytest.approx(0.0)


def test_identity_indicators() -> None:
    out = DerivedColumns(DIGIT_IDENTITY_NAMES).compute(make_df())
    col = DIGIT_IDENTITY_COLS[0]
    # 3.0은 정수이자 소수 1자리, 2.5는 소수 1자리만, 1.37은 둘 다 아니다.
    assert out.loc[0, f"{col}_is_round0"] == 1.0
    assert out.loc[0, f"{col}_is_round1"] == 1.0
    assert out.loc[1, f"{col}_is_round0"] == 0.0
    assert out.loc[1, f"{col}_is_round1"] == 1.0
    assert out.loc[2, f"{col}_is_round0"] == 0.0
    assert out.loc[2, f"{col}_is_round1"] == 0.0
    # 2.9는 이진 표현 오차에도 소수 1자리로 판정되어야 한다.
    assert out.loc[4, f"{col}_is_round0"] == 0.0
    assert out.loc[4, f"{col}_is_round1"] == 1.0
    # 0.25는 소수 2자리라 is_round1이 0이다.
    assert out.loc[5, f"{col}_is_round1"] == 0.0


def test_decimal_digit_values() -> None:
    out = DerivedColumns(DIGIT_IDENTITY_NAMES).compute(make_df())
    col = DIGIT_IDENTITY_COLS[0]
    assert out.loc[0, f"{col}_tenths"] == pytest.approx(0.0)
    assert out.loc[0, f"{col}_hundredths"] == pytest.approx(0.0)
    assert out.loc[1, f"{col}_tenths"] == pytest.approx(5.0)
    assert out.loc[1, f"{col}_hundredths"] == pytest.approx(0.0)
    assert out.loc[2, f"{col}_tenths"] == pytest.approx(3.0)
    assert out.loc[2, f"{col}_hundredths"] == pytest.approx(7.0)
    assert out.loc[4, f"{col}_tenths"] == pytest.approx(9.0)
    assert out.loc[5, f"{col}_hundredths"] == pytest.approx(5.0)


def test_missing_rows_propagate_nan_everywhere() -> None:
    out = DerivedColumns(DIGIT_IDENTITY_NAMES).compute(make_df())
    col = DIGIT_IDENTITY_COLS[0]
    for name in [n for n in DIGIT_IDENTITY_NAMES if n.startswith(f"{col}_")]:
        assert np.isnan(out.loc[3, name]), name


def test_block_is_target_free_and_deterministic() -> None:
    df = make_df()
    provider = DerivedColumns(DIGIT_IDENTITY_NAMES)
    assert provider.uses_target is False
    pd.testing.assert_frame_equal(provider.compute(df), provider.compute(df))
