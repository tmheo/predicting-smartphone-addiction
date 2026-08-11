"""원본 프록시 최근접 라벨 제공자 테스트. (#54)

- 해시 고정: 기대 해시와 다른 파일은 거부.
- 프록시 전용 열·비수치 열 사용 거부(#47 경계, 거리 정의).
- ks 검증: 1 미만, 중복, 프록시 행 수 초과 거부.
- 최근접 정의의 수치 골든: k=1 라벨과 k>1 접두 평균.
- NaN 인지 거리: 결측 컬럼은 거리에서 제외.
- 수치 전결측 행은 NaN.
- 합성 train 라벨 미사용 경계: 타깃을 바꿔도 산출이 불변.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.features import OriginalKnnColumns

LABEL = "addicted_label"


def write_proxy(tmp_path: Path, df: pd.DataFrame) -> tuple[str, str]:
    path = tmp_path / "proxy.csv"
    df.to_csv(path, index=False)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def line_proxy(tmp_path: Path) -> tuple[str, str]:
    # age 한 축 위의 점 4개. 표준화는 모든 점에 같은 배율이라 순위를 바꾸지 않는다.
    df = pd.DataFrame({"age": [0.0, 10.0, 20.0, 30.0], LABEL: [1, 1, 0, 0]})
    return write_proxy(tmp_path, df)


@pytest.fixture
def plane_proxy(tmp_path: Path) -> tuple[str, str]:
    # 두 축(age, sleep). age를 가리면 sleep만으로 최근접이 정해진다.
    df = pd.DataFrame(
        {
            "age": [0.0, 100.0, 200.0],
            "sleep_hours": [8.0, 4.0, 6.0],
            LABEL: [0, 1, 0],
        }
    )
    return write_proxy(tmp_path, df)


def test_wrong_hash_is_rejected(line_proxy):
    path, _ = line_proxy
    with pytest.raises(ValueError, match="해시 불일치"):
        OriginalKnnColumns(path=path, cols=["age"], sha256="0" * 64)


def test_proxy_only_and_non_numeric_columns_are_rejected(tmp_path):
    df = pd.DataFrame(
        {"age": [1.0, 2.0], "gender": ["F", "M"], "addiction_level": [1, 2], LABEL: [0, 1]}
    )
    path, sha = write_proxy(tmp_path, df)
    with pytest.raises(ValueError, match="프록시 전용"):
        OriginalKnnColumns(path=path, cols=["addiction_level"], sha256=sha)
    with pytest.raises(ValueError, match="수치가 아닌"):
        OriginalKnnColumns(path=path, cols=["gender"], sha256=sha)


def test_ks_validation(line_proxy):
    path, sha = line_proxy
    with pytest.raises(ValueError, match="1 이상의 정수"):
        OriginalKnnColumns(path=path, cols=["age"], ks=[0], sha256=sha)
    with pytest.raises(ValueError, match="중복"):
        OriginalKnnColumns(path=path, cols=["age"], ks=[1, 1], sha256=sha)
    with pytest.raises(ValueError, match="프록시 행 수"):
        OriginalKnnColumns(path=path, cols=["age"], ks=[5], sha256=sha)


def test_nearest_label_and_prefix_means(line_proxy):
    path, sha = line_proxy
    provider = OriginalKnnColumns(path=path, cols=["age"], ks=[1, 3], sha256=sha)
    assert provider.columns() == ["orig_nn1_mean", "orig_nn3_mean"]
    df = pd.DataFrame({"age": [1.0, 29.0]})
    out = provider.compute(df)
    # age 1: 최근접 0(라벨 1), 3-이웃 {0,10,20} 라벨 [1,1,0] 평균 2/3
    # age 29: 최근접 30(라벨 0), 3-이웃 {30,20,10} 라벨 [0,0,1] 평균 1/3
    np.testing.assert_allclose(out["orig_nn1_mean"], [1.0, 0.0])
    np.testing.assert_allclose(out["orig_nn3_mean"], [2 / 3, 1 / 3])


def test_missing_column_is_excluded_from_distance(plane_proxy):
    path, sha = plane_proxy
    provider = OriginalKnnColumns(path=path, cols=["age", "sleep_hours"], ks=[1], sha256=sha)
    # 표준화 배율: age std 100, sleep std 2. 표준화 좌표는 age [0,1,2], sleep [4,2,3].
    # age가 결측이면 sleep만 비교: 4.2(표준화 2.1)는 sleep 4.0 행(라벨 1)이 최근접.
    # age 200이 관측되면 (2, 2.1)이 되어 age 200 행(표준화 (2,3), 라벨 0)이 최근접.
    df = pd.DataFrame({"age": [np.nan, 200.0], "sleep_hours": [4.2, 4.2]})
    out = provider.compute(df)
    np.testing.assert_allclose(out["orig_nn1_mean"], [1.0, 0.0])


def test_all_numeric_missing_row_is_nan(plane_proxy):
    path, sha = plane_proxy
    provider = OriginalKnnColumns(path=path, cols=["age", "sleep_hours"], ks=[1, 2], sha256=sha)
    df = pd.DataFrame({"age": [np.nan, 100.0], "sleep_hours": [np.nan, 4.0]})
    out = provider.compute(df)
    assert out.iloc[0].isna().all()
    assert not out.iloc[1].isna().any()


def test_output_is_invariant_to_competition_target(line_proxy):
    """합성 train 라벨 미사용 경계: 타깃 열을 뒤집거나 없애도 산출이 같아야 한다."""
    path, sha = line_proxy
    provider = OriginalKnnColumns(path=path, cols=["age"], ks=[1, 3], sha256=sha)
    assert provider.uses_target is False
    df = pd.DataFrame({"age": [1.0, 15.0, 29.0], LABEL: [1, 1, 1]})
    flipped = df.assign(**{LABEL: [0, 0, 0]})
    dropped = df.drop(columns=[LABEL])
    base = provider.compute(df)
    pd.testing.assert_frame_equal(base, provider.compute(flipped))
    pd.testing.assert_frame_equal(base, provider.compute(dropped))
