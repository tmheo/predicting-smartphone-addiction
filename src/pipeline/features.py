"""피처 구성.

baseline은 원시 피처 그대로. (#16)
타깃 인코딩, other_screen 잔차 같은 후속 피처는 이 모듈에 함수를 추가하고
설정 파일의 [features] 섹션에서 켜는 식으로 확장한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import ID, TARGET

PLACEBO = "placebo_noise"
# 플라시보에 입힐 결측 패턴의 원본 열. 결측 없는 잡음은 후보 피처(원본 열의 결측을
# 물려받음)와 다른 종류의 자가 되므로, 실제 열의 NaN 마스크를 복사한다. (#19 정정)
PLACEBO_MASK_SOURCE = "social_media_hours"


def build_features(df: pd.DataFrame, cfg: FeatureConfig, seed: int) -> pd.DataFrame:
    """모델 입력 행렬을 만든다. 반환 컬럼 목록이 곧 params로 기록되는 feature 목록."""
    if cfg.include == "raw":
        cols = [c for c in df.columns if c not in (ID, TARGET, "fold")]
    else:
        cols = list(cfg.include)
    X = df[cols].copy()
    if cfg.placebo:
        # 개선 판정 기준선: 이 피처보다 importance가 낮으면 노이즈로 본다. (#15)
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=len(X))
        noise[df[PLACEBO_MASK_SOURCE].isna().to_numpy()] = np.nan
        X[PLACEBO] = noise
    return X


class FoldFitTransformer(Protocol):
    """fold 루프 안에서 학습하는 fold-fit 피처의 단위. (#32, #35)

    fit은 타깃이 포함된 학습 fold의 DataFrame을 받아 상태를 새로 계산하고
    (fold마다 다시 불리므로 이전 fold의 상태를 남기면 안 된다),
    transform은 DataFrame을 받아 같은 인덱스의 새 컬럼 DataFrame을 돌려준다.
    두 입력 모두 원본 컬럼에 build_features 산출 컬럼(placebo 등)이 더해진 형태다. (#33 파급)

    transform은 fold마다 train 전체(학습 fold + 검증 fold 행)와 test로 두 번 불린다.
    학습 fold 행에 OOF 값을 줘야 하는 트랜스포머(타깃 인코딩의 내부 K-fold)는
    fit 때 학습 행의 id 집합을 저장해 행별로 OOF 값과 평균표 값을 구분해 돌려준다.
    위치 인덱스는 train과 test가 겹치므로 구분 기준은 id여야 한다.
    컬럼 이름은 트랜스포머 책임이며, 기존 컬럼과 겹치면 파이프라인이 즉시 실패한다.
    """

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None: ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


# kind -> 트랜스포머 팩토리. 새 fold-fit 피처는 여기 등록만 하면 설정에서 켤 수 있다.
FOLD_FIT_REGISTRY: dict[str, Callable[..., FoldFitTransformer]] = {}


def make_fold_fit(cfg: FeatureConfig) -> list[FoldFitTransformer]:
    """설정의 fold_fit 목록을 트랜스포머 인스턴스로 만든다. kind 외 키는 생성자 인자."""
    transformers: list[FoldFitTransformer] = []
    for spec in cfg.fold_fit:
        params = dict(spec)
        kind = params.pop("kind")
        transformers.append(FOLD_FIT_REGISTRY[kind](**params))
    return transformers


def add_fold_fit_columns(
    transformers: list[FoldFitTransformer], X: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:
    """fit된 트랜스포머들의 새 컬럼을 X에 붙인 행렬을 돌려준다. 추가 전용."""
    out = X
    for t in transformers:
        new = t.transform(df)
        assert new.index.equals(df.index), f"{type(t).__name__}의 transform 인덱스가 원본과 다르다."
        collision = set(new.columns) & set(out.columns)
        assert not collision, f"fold-fit 컬럼 이름 충돌: {sorted(collision)}"
        out = pd.concat([out, new], axis=1)
    return out
