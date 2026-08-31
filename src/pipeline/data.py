"""데이터 로딩과 fold 부여.

파이프라인은 fold를 계산하지 않는다.
scripts/make_folds.py가 한 번 만들어 커밋한 artifacts/folds.parquet을 읽기만 한다. (#15)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .identity import file_identity

TARGET = "addicted_label"
ID = "id"
TRAIN_PATH = Path("data/train.csv")


def labels(index: pd.Index, train_path: Path = TRAIN_PATH) -> pd.Series:
    """train 라벨을 주어진 id 순서로 정렬해 돌려준다. id가 어긋나면 즉시 실패한다."""
    aligned = pd.read_csv(train_path, usecols=[ID, TARGET]).set_index(ID)[TARGET].reindex(index)
    assert aligned.notna().all(), "요청한 id가 train과 일치하지 않는다."
    return aligned


def file_sha256(path: Path) -> str:
    """입력 파일 계보 기록용 해시. 실행마다 태그로 남긴다."""
    return file_identity(path)


def load_csv(path: Path) -> pd.DataFrame:
    # NaN 유지, 대치 없음. (#16)
    return pd.read_csv(path)


def align_categories(train: pd.DataFrame, test: pd.DataFrame, categorical: list[str]) -> None:
    """범주형 컬럼을 train/test 공통 카테고리 체계의 category dtype으로 맞춘다.

    각각 astype("category")를 하면 한쪽에만 있는 값이 코드 배정을 어긋나게 만들 수 있어,
    두 데이터의 값 합집합으로 카테고리를 고정한다. LightGBM 내장 범주형 처리는 이 코드를 쓴다. (#16)
    """
    for col in categorical:
        train[col], test[col] = union_categorical(train[col], test[col])


def union_categorical(a: pd.Series, b: pd.Series) -> tuple[pd.Categorical, pd.Categorical]:
    cats = sorted(set(a.dropna()) | set(b.dropna()))
    return pd.Categorical(a, categories=cats), pd.Categorical(b, categories=cats)


def attach_folds(train: pd.DataFrame, folds_path: Path) -> pd.DataFrame:
    """커밋된 fold 파일(id, fold)을 id로 병합한다.

    전 행이 정확히 한 번씩 fold를 받는지 검증한다. 어긋나면 데이터나 fold 파일이 바뀐 것이므로 즉시 실패.
    """
    folds = pd.read_parquet(folds_path)
    merged = train.merge(folds, on=ID, how="left", validate="one_to_one")
    assert merged["fold"].notna().all(), "fold가 없는 행이 있다. folds.parquet을 다시 확인할 것."
    return merged
