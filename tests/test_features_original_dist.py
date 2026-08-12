"""원본 클래스별 CDF 차·KDE 로그밀도비 제공자 테스트. (#84가 연 #87)

- 해시 고정: 기대 해시와 다른 파일은 거부.
- 프록시 전용 열·수치 아닌 열 거부(#47 경계).
- CDF 차의 수치 골든(우측 포함 CDF, F0 - F1)과 입력 결측 NaN 유지.
- 클래스에 관측값이 없으면 생성 시점 거부(레시피의 오류 조건).
- KDE 로그밀도비의 수치 골든(표준화·Silverman 대역폭·[-20, 20] 절단).
- 합성 train 라벨 미사용 경계: 타깃을 바꿔도 산출이 불변.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    OriginalClassCdfDiff,
    OriginalKdeLogRatio,
    _silverman_bandwidth,
)

LABEL = "addicted_label"


def write_proxy(tmp_path: Path, df: pd.DataFrame) -> tuple[str, str]:
    path = tmp_path / "proxy.csv"
    df.to_csv(path, index=False)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def small_proxy(tmp_path: Path) -> tuple[str, str]:
    # 클래스 0: x = [1, 2, 3], 클래스 1: x = [2, 4]. gender는 수치 아닌 열 거부 검사용.
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 2.0, 4.0],
            "gender": ["F", "M", "F", "M", "F"],
            LABEL: [0, 0, 0, 1, 1],
        }
    )
    return write_proxy(tmp_path, df)


def test_wrong_hash_is_rejected(small_proxy):
    path, _ = small_proxy
    with pytest.raises(ValueError, match="해시 불일치"):
        OriginalClassCdfDiff(path=path, cols=["x"], sha256="0" * 64)
    with pytest.raises(ValueError, match="해시 불일치"):
        OriginalKdeLogRatio(path=path, cols=["x"], sha256="0" * 64)


def test_forbidden_and_non_numeric_columns_are_rejected(small_proxy):
    path, sha = small_proxy
    with pytest.raises(ValueError, match="프록시 전용"):
        OriginalClassCdfDiff(path=path, cols=["addiction_level"], sha256=sha)
    with pytest.raises(ValueError, match="수치가 아닌"):
        OriginalClassCdfDiff(path=path, cols=["gender"], sha256=sha)
    with pytest.raises(ValueError, match="프록시 전용"):
        OriginalKdeLogRatio(path=path, cols=["addicted_label"], sha256=sha)
    with pytest.raises(ValueError, match="수치가 아닌"):
        OriginalKdeLogRatio(path=path, cols=["gender"], sha256=sha)


def test_empty_class_is_rejected(tmp_path):
    path, sha = write_proxy(
        tmp_path, pd.DataFrame({"x": [1.0, 2.0], LABEL: [0, 0]})
    )
    with pytest.raises(ValueError, match="클래스 1에 관측값이 없어"):
        OriginalClassCdfDiff(path=path, cols=["x"], sha256=sha)
    with pytest.raises(ValueError, match="클래스 1에 관측값이 없어"):
        OriginalKdeLogRatio(path=path, cols=["x"], sha256=sha)


def test_cdf_diff_golden_values(small_proxy):
    path, sha = small_proxy
    provider = OriginalClassCdfDiff(path=path, cols=["x"], sha256=sha)
    assert provider.columns() == ["x_orig_cdf_diff"]
    df = pd.DataFrame({"x": [0.5, 2.0, 3.0, 5.0, np.nan]})
    out = provider.compute(df)
    assert list(out.columns) == provider.columns()
    # F0: [1,2,3] 우측 포함, F1: [2,4] 우측 포함.
    # x=0.5: 0 - 0 = 0 / x=2: 2/3 - 1/2 / x=3: 1 - 1/2 / x=5: 1 - 1 = 0
    np.testing.assert_allclose(
        out["x_orig_cdf_diff"].to_numpy()[:4], [0.0, 2 / 3 - 0.5, 0.5, 0.0]
    )
    assert np.isnan(out["x_orig_cdf_diff"].iloc[4])


def test_cdf_is_right_inclusive(small_proxy):
    path, sha = small_proxy
    provider = OriginalClassCdfDiff(path=path, cols=["x"], sha256=sha)
    out = provider.compute(pd.DataFrame({"x": [1.0, 1.0 - 1e-9]}))
    # x=1은 클래스 0의 첫 관측값을 포함(1/3), 그보다 작은 값은 0.
    np.testing.assert_allclose(out["x_orig_cdf_diff"], [1 / 3, 0.0])


def test_silverman_bandwidth_rules():
    # 관측값이 둘보다 적으면 0.30.
    assert _silverman_bandwidth(np.array([1.0])) == 0.30
    # 값이 전부 같으면(scale 0) 0.30.
    assert _silverman_bandwidth(np.array([2.0, 2.0, 2.0])) == 0.30
    # 정상 경로: 0.9 * min(std, IQR/1.34) * n^(-1/5)를 [0.10, 1.00]으로 절단.
    vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    std = vals.std(ddof=1)
    iqr = np.subtract(*np.percentile(vals, [75, 25]))
    expected = np.clip(0.9 * min(std, iqr / 1.34) * len(vals) ** (-1 / 5), 0.10, 1.00)
    np.testing.assert_allclose(_silverman_bandwidth(vals), expected)


def test_kde_log_ratio_golden_values(small_proxy):
    path, sha = small_proxy
    provider = OriginalKdeLogRatio(path=path, cols=["x"], sha256=sha)
    assert provider.columns() == ["x_orig_kde_lr"]
    df = pd.DataFrame({"x": [2.0, np.nan]})
    out = provider.compute(df)

    # 손 계산 재현: 전체 [1,2,3,2,4]의 평균과 모집단 표준편차로 표준화한 뒤
    # 클래스별 Silverman 대역폭의 가우시안 KDE 로그밀도 차.
    full = np.array([1.0, 2.0, 3.0, 2.0, 4.0])
    mu, sd = full.mean(), full.std(ddof=0)
    z0 = (np.array([1.0, 2.0, 3.0]) - mu) / sd
    z1 = (np.array([2.0, 4.0]) - mu) / sd
    zq = (2.0 - mu) / sd

    def log_kde(z_ref: np.ndarray, zq: float, bw: float) -> float:
        dens = np.exp(-0.5 * ((zq - z_ref) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
        return float(np.log(dens.mean()))

    expected = log_kde(z1, zq, _silverman_bandwidth(z1)) - log_kde(
        z0, zq, _silverman_bandwidth(z0)
    )
    np.testing.assert_allclose(out["x_orig_kde_lr"].iloc[0], np.clip(expected, -20, 20))
    assert np.isnan(out["x_orig_kde_lr"].iloc[1])


def test_kde_log_ratio_is_clipped(tmp_path):
    # 클래스 분포를 멀리 떨어뜨려 로그밀도비가 절단 범위를 넘게 만든다.
    df = pd.DataFrame({"x": [0.0, 0.001, 1000.0, 1000.001], LABEL: [0, 0, 1, 1]})
    path, sha = write_proxy(tmp_path, df)
    provider = OriginalKdeLogRatio(path=path, cols=["x"], sha256=sha)
    out = provider.compute(pd.DataFrame({"x": [0.0, 1000.0]}))
    np.testing.assert_allclose(out["x_orig_kde_lr"], [-20.0, 20.0])


def test_output_is_invariant_to_competition_target(small_proxy):
    """합성 train 라벨 미사용 경계: 타깃 열을 뒤집거나 없애도 산출이 같아야 한다."""
    path, sha = small_proxy
    for provider in (
        OriginalClassCdfDiff(path=path, cols=["x"], sha256=sha),
        OriginalKdeLogRatio(path=path, cols=["x"], sha256=sha),
    ):
        assert provider.uses_target is False
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan], LABEL: [1, 1, 1]})
        flipped = df.assign(**{LABEL: [0, 0, 0]})
        dropped = df.drop(columns=[LABEL])
        base = provider.compute(df)
        pd.testing.assert_frame_equal(base, provider.compute(flipped))
        pd.testing.assert_frame_equal(base, provider.compute(dropped))
