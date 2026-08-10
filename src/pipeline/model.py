"""모델 학습. 지금은 LightGBM 하나. 모델 종류가 늘면 kind별 함수를 추가한다."""

from __future__ import annotations

import pandas as pd

from .config import ModelConfig


def train_one_fold(
    cfg: ModelConfig,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    seed: int,
):
    """한 fold를 학습해 (모델, 검증 예측)을 돌려준다."""
    # 무거운 의존성은 실제 학습 시점에만 import한다. --plan 실행은 이 모듈이 필요 없다.
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**cfg.params, random_state=seed)
    model.fit(
        X_tr,
        y_tr,
        eval_X=X_va,
        eval_y=y_va,
        callbacks=[lgb.early_stopping(cfg.fit["early_stopping_rounds"])],
    )
    return model, model.predict_proba(X_va)[:, 1]


def predict_test(model, X_test: pd.DataFrame):
    return model.predict_proba(X_test)[:, 1]


def gain_importance(model) -> pd.DataFrame:
    """학습된 모델의 gain importance를 (feature, gain) 프레임으로 돌려준다. (#19)"""
    booster = model.booster_
    return pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
        }
    )
