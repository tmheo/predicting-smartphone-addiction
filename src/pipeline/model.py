"""모델 계열. kind -> adapter 팩토리 레지스트리로 학습기 구현을 dispatch한다. (#72)

cv 루프는 어떤 모델인지 모른다: cfg.model.kind를 여기 레지스트리에서 해석해
adapter 인스턴스를 만들고 fit/predict/importance만 부른다.
params 해석, 시드 적용(random_state 등 모델별 이름), fit 인자(early_stopping_rounds)
해석은 adapter가 소유하고, 학습된 모델 상태는 인스턴스 안에 갇힌다.
plan.REGISTRY와 같은 패턴: 새 모델 계열은 adapter를 구현하고 MODEL_REGISTRY에 등록한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from .config import ModelConfig


class ModelAdapter(Protocol):
    """fold 하나의 학습기 계약. 인스턴스는 fold마다 새로 만든다."""

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        """한 fold를 학습하고 검증 fold 예측을 돌려준다."""
        ...

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray: ...

    def importance(self) -> pd.DataFrame:
        """학습된 모델의 importance를 (feature, gain) 프레임으로 돌려준다. (#19)"""
        ...


class LightGBMAdapter:
    """LightGBM 이진 분류 adapter."""

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._model = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        # 무거운 의존성은 실제 학습 시점에만 import한다. --plan 실행은 이 모듈이 필요 없다.
        import lightgbm as lgb

        self._model = lgb.LGBMClassifier(**self._params, random_state=self._seed)
        if (initial_score_tr is None) != (initial_score_va is None):
            raise ValueError("학습과 검증 초기 점수는 함께 주거나 함께 생략해야 한다.")
        self._uses_initial_score = initial_score_tr is not None
        kwargs = {}
        if self._uses_initial_score:
            kwargs = {
                "init_score": initial_score_tr,
                "eval_init_score": [initial_score_va],
            }
        self._model.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            callbacks=[lgb.early_stopping(self._fit["early_stopping_rounds"])],
            **kwargs,
        )
        return self._predict(X_va, initial_score_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        return self._predict(X, initial_score)

    def _predict(self, X: pd.DataFrame, initial_score: pd.Series | None) -> np.ndarray:
        if not self._uses_initial_score:
            if initial_score is not None:
                raise ValueError("초기 점수 없이 학습한 모델에 예측 초기 점수가 전달됐다.")
            return self._model.predict_proba(X)[:, 1]
        if initial_score is None:
            raise ValueError("초기 점수로 학습한 모델은 예측에도 같은 출처의 초기 점수가 필요하다.")
        residual = np.asarray(self._model.predict(X, raw_score=True), dtype="float64")
        margin = np.asarray(initial_score, dtype="float64")
        if residual.shape != margin.shape:
            raise ValueError(f"예측과 초기 점수 길이가 다르다: {residual.shape} != {margin.shape}")
        total = residual + margin
        # 큰 음수에서도 overflow 경고 없이 안정적으로 sigmoid를 계산한다.
        out = np.empty_like(total)
        positive = total >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-total[positive]))
        exp_total = np.exp(total[~positive])
        out[~positive] = exp_total / (1.0 + exp_total)
        return out

    def importance(self) -> pd.DataFrame:
        booster = self._model.booster_
        return pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance(importance_type="gain"),
            }
        )


# kind -> adapter 팩토리. 인스턴스는 (params, fit, seed)로 만든다.
MODEL_REGISTRY: dict[str, Callable[[dict, dict, int], ModelAdapter]] = {
    "lightgbm": LightGBMAdapter,
}


def create(cfg: ModelConfig, seed: int) -> ModelAdapter:
    """cfg.kind를 레지스트리에서 해석해 fold 하나의 adapter를 만든다."""
    if cfg.kind not in MODEL_REGISTRY:
        raise ValueError(
            f"알 수 없는 model.kind {cfg.kind!r}. 등록된 kind: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    return MODEL_REGISTRY[cfg.kind](cfg.params, cfg.fit, seed)
