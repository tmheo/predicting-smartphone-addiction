"""원본 프록시 prior 제공자 테스트. (#53)

- 해시 고정: 기대 해시와 다른 파일은 거부.
- 프록시 전용 열 사용 거부(#47 경계).
- 통계 정의(평활 평균, WoE, 엔트로피, 빈도)의 수치 골든.
- 미지 키 처리: nan 대 global 대입.
- 자료형 정렬: 프록시 int 열과 대회 float 열이 같은 키로 매핑.
- 합성 train 라벨 미사용 경계: 타깃을 바꿔도 산출이 불변.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.features import OriginalPriorColumns

LABEL = "addicted_label"


def write_proxy(tmp_path: Path, df: pd.DataFrame) -> tuple[str, str]:
    path = tmp_path / "proxy.csv"
    df.to_csv(path, index=False)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def small_proxy(tmp_path: Path) -> tuple[str, str]:
    # age 20이 4행(라벨 평균 0.75), 30이 2행(평균 0.0), 전체 평균 g=0.5.
    df = pd.DataFrame(
        {
            "age": [20, 20, 20, 20, 30, 30],
            "gender": ["F", "F", "M", "M", "F", "M"],
            LABEL: [1, 1, 1, 0, 0, 0],
        }
    )
    return write_proxy(tmp_path, df)


def test_wrong_hash_is_rejected(small_proxy):
    path, _ = small_proxy
    with pytest.raises(ValueError, match="해시 불일치"):
        OriginalPriorColumns(path=path, cols=["age"], sha256="0" * 64)


def test_proxy_only_columns_are_rejected(small_proxy):
    path, sha = small_proxy
    with pytest.raises(ValueError, match="프록시 전용"):
        OriginalPriorColumns(path=path, cols=["addiction_level"], sha256=sha)


def test_unknown_stat_and_zero_smoothing_woe_are_rejected(small_proxy):
    path, sha = small_proxy
    with pytest.raises(ValueError, match="알 수 없는 stat"):
        OriginalPriorColumns(path=path, cols=["age"], stats=["median"], sha256=sha)
    with pytest.raises(ValueError, match="smoothing > 0"):
        OriginalPriorColumns(path=path, cols=["age"], stats=["woe"], smoothing=0, sha256=sha)


def test_statistics_golden_values(small_proxy):
    path, sha = small_proxy
    provider = OriginalPriorColumns(
        path=path,
        cols=["age"],
        stats=["mean", "woe", "entropy", "count"],
        smoothing=2.0,
        sha256=sha,
    )
    df = pd.DataFrame({"age": [20.0, 30.0]})
    out = provider.compute(df)
    assert list(out.columns) == provider.columns()
    # m=2, g=0.5: p(20) = (4*0.75 + 2*0.5)/6 = 2/3, p(30) = (2*0.0 + 2*0.5)/4 = 0.25
    np.testing.assert_allclose(out["age_orig_mean"], [2 / 3, 0.25])
    # woe = logit(p) - logit(0.5) = logit(p)
    np.testing.assert_allclose(out["age_orig_woe"], [np.log(2), np.log(0.25 / 0.75)])
    ent = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))
    np.testing.assert_allclose(out["age_orig_entropy"], [ent(2 / 3), ent(0.25)])
    np.testing.assert_allclose(out["age_orig_count"], [np.log1p(4), np.log1p(2)])


def test_unknown_keys_nan_vs_global(small_proxy):
    path, sha = small_proxy
    df = pd.DataFrame({"age": [25.0, np.nan]})  # 25는 프록시에 없는 값, NaN은 결측 키
    kw = {
        "path": path,
        "cols": ["age"],
        "stats": ["mean", "count"],
        "smoothing": 2.0,
        "sha256": sha,
    }
    out_nan = OriginalPriorColumns(unknown="nan", **kw).compute(df)
    assert out_nan.isna().all().all()
    out_global = OriginalPriorColumns(unknown="global", **kw).compute(df)
    np.testing.assert_allclose(out_global["age_orig_mean"], [0.5, 0.5])  # g
    np.testing.assert_allclose(out_global["age_orig_count"], [0.0, 0.0])  # log1p(0)


def test_int_proxy_column_maps_float_competition_column(small_proxy):
    """프록시 age는 int64, 대회 train age는 float64로 읽힌다. 키가 정렬되어야 한다."""
    path, sha = small_proxy
    provider = OriginalPriorColumns(
        path=path, cols=["age"], stats=["mean"], smoothing=0.0, sha256=sha
    )
    out = provider.compute(pd.DataFrame({"age": pd.Series([20.0, 30.0], dtype="float64")}))
    np.testing.assert_allclose(out["age_orig_mean"], [0.75, 0.0])


def test_pair_spec_uses_joint_key(small_proxy):
    path, sha = small_proxy
    provider = OriginalPriorColumns(
        path=path, cols=[["age", "gender"]], stats=["mean"], smoothing=0.0, sha256=sha
    )
    df = pd.DataFrame({"age": [20.0, 20.0], "gender": ["F", "M"]})
    out = provider.compute(df)
    # (20,F): 라벨 [1,1] 평균 1.0 / (20,M): 라벨 [1,0] 평균 0.5
    np.testing.assert_allclose(out["age__gender_orig_mean"], [1.0, 0.5])


def test_output_is_invariant_to_competition_target(small_proxy):
    """합성 train 라벨 미사용 경계: 타깃 열을 뒤집거나 없애도 산출이 같아야 한다. (#53)"""
    path, sha = small_proxy
    provider = OriginalPriorColumns(path=path, cols=["age"], sha256=sha)
    assert provider.uses_target is False
    df = pd.DataFrame({"age": [20.0, 30.0, 25.0], LABEL: [1, 1, 1]})
    flipped = df.assign(**{LABEL: [0, 0, 0]})
    dropped = df.drop(columns=[LABEL])
    base = provider.compute(df)
    pd.testing.assert_frame_equal(base, provider.compute(flipped))
    pd.testing.assert_frame_equal(base, provider.compute(dropped))
