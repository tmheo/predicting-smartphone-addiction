# PROTOTYPE (issue #17): 구조 확인용 뼈대.
"""피처 구성.

baseline은 원시 피처 그대로. (#16)
타깃 인코딩, other_screen 잔차 같은 후속 피처는 이 모듈에 함수를 추가하고
설정 파일의 [features] 섹션에서 켜는 식으로 확장한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import ID, TARGET

PLACEBO = "placebo_noise"


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
        X[PLACEBO] = rng.normal(size=len(X))
    return X
