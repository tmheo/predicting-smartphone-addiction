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


class XGBoostAdapter:
    """XGBoost 이진 분류 adapter. (#59)

    범주형은 category dtype 그대로 native 학습한다(enable_categorical).
    importance는 LightGBM gain과 같은 축척인 total_gain을 쓴다.
    """

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
        import xgboost as xgb

        _reject_initial_score("xgboost", initial_score_tr, initial_score_va)
        self._model = xgb.XGBClassifier(
            **self._params,
            random_state=self._seed,
            enable_categorical=True,
            early_stopping_rounds=self._fit["early_stopping_rounds"],
        )
        # LightGBM처럼 학습 과정이 실행 로그에 남게 200라운드마다 검증 지표를 찍고,
        # fold의 종착점(best iteration)을 한 줄로 요약한다.
        self._model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
        print(
            f"[xgboost] early stopping: best_iteration={self._model.best_iteration} "
            f"best_score={self._model.best_score:.6f}"
        )
        return self._model.predict_proba(X_va)[:, 1]

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("xgboost", initial_score, None)
        return self._model.predict_proba(X)[:, 1]

    def importance(self) -> pd.DataFrame:
        booster = self._model.get_booster()
        gain = booster.get_score(importance_type="total_gain")
        # get_score는 분기에 안 쓰인 피처를 생략하므로 0으로 채워 전 피처를 돌려준다.
        names = list(self._model.feature_names_in_)
        return pd.DataFrame(
            {"feature": names, "gain": [gain.get(n, 0.0) for n in names]}
        )


class CatBoostAdapter:
    """CatBoost 이진 분류 adapter. (#59)

    CatBoost는 cat 피처의 NaN을 거부하므로 category dtype 컬럼을 결측 sentinel이
    포함된 문자열로 바꿔 native categorical로 학습한다(행 단위 결정적 변환).
    importance는 gain이 없어 PredictionValuesChange를 gain 컬럼으로 돌려준다.
    """

    _MISSING = "__missing__"

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._model = None

    @classmethod
    def _prepare(cls, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out = X.copy()
        cat_cols = [
            c for c in out.columns if isinstance(out[c].dtype, pd.CategoricalDtype)
        ]
        for c in cat_cols:
            out[c] = (
                out[c].cat.add_categories([cls._MISSING]).fillna(cls._MISSING).astype(str)
            )
        return out, cat_cols

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from catboost import CatBoostClassifier

        _reject_initial_score("catboost", initial_score_tr, initial_score_va)
        X_tr, cat_cols = self._prepare(X_tr)
        X_va, _ = self._prepare(X_va)
        self._model = CatBoostClassifier(
            **self._params,
            random_seed=self._seed,
            cat_features=cat_cols,
            allow_writing_files=False,
        )
        # LightGBM처럼 학습 과정이 실행 로그에 남게 200라운드마다 검증 지표를 찍고,
        # fold의 종착점(best iteration)을 한 줄로 요약한다.
        self._model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=self._fit["early_stopping_rounds"],
            use_best_model=True,
            verbose=200,
        )
        best_score = self._model.get_best_score().get("validation", {})
        print(
            f"[catboost] early stopping: best_iteration={self._model.get_best_iteration()} "
            f"best_score={best_score}"
        )
        return self._model.predict_proba(X_va)[:, 1]

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("catboost", initial_score, None)
        X, _ = self._prepare(X)
        return self._model.predict_proba(X)[:, 1]

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self._model.feature_names_,
                "gain": self._model.get_feature_importance(type="PredictionValuesChange"),
            }
        )


class HistGradientBoostingAdapter:
    """scikit-learn HistGradientBoosting 이진 분류 adapter. (#59)

    외부 eval set 조기 종료가 없어 early stopping 설정은 params로 받는다
    (학습 fold 내부 분할만 쓰므로 검증 fold 누출은 없다).
    gain importance가 없어 검증 fold permutation importance(AUC 하락 폭)를
    gain 컬럼으로 돌려준다(#59 코멘트의 계열 무관 중요도 규약).
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._model = None
        self._X_va: pd.DataFrame | None = None
        self._y_va: pd.Series | None = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from sklearn.ensemble import HistGradientBoostingClassifier

        _reject_initial_score("hist_gradient_boosting", initial_score_tr, initial_score_va)
        self._model = HistGradientBoostingClassifier(
            **self._params,
            random_state=self._seed,
            categorical_features="from_dtype",
        )
        self._model.fit(X_tr, y_tr)
        # 내부 분할 조기 종료의 종착점을 실행 로그에 남긴다(반복별 로그는 없음).
        print(f"[hist_gradient_boosting] n_iter={self._model.n_iter_}")
        self._X_va, self._y_va = X_va, y_va
        return self._model.predict_proba(X_va)[:, 1]

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("hist_gradient_boosting", initial_score, None)
        return self._model.predict_proba(X)[:, 1]

    def importance(self) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            self._model,
            self._X_va,
            self._y_va,
            scoring="roc_auc",
            n_repeats=3,
            random_state=self._seed,
        )
        return pd.DataFrame(
            {"feature": list(self._X_va.columns), "gain": result.importances_mean}
        )


def _reject_initial_score(
    kind: str, initial_score_tr: pd.Series | None, initial_score_va: pd.Series | None
) -> None:
    if initial_score_tr is not None or initial_score_va is not None:
        raise ValueError(f"{kind} adapter는 초기 점수를 지원하지 않는다.")


# kind -> adapter 팩토리. 인스턴스는 (params, fit, seed)로 만든다.
MODEL_REGISTRY: dict[str, Callable[[dict, dict, int], ModelAdapter]] = {
    "lightgbm": LightGBMAdapter,
    "xgboost": XGBoostAdapter,
    "catboost": CatBoostAdapter,
    "hist_gradient_boosting": HistGradientBoostingAdapter,
}


def create(cfg: ModelConfig, seed: int) -> ModelAdapter:
    """cfg.kind를 레지스트리에서 해석해 fold 하나의 adapter를 만든다."""
    if cfg.kind not in MODEL_REGISTRY:
        raise ValueError(
            f"알 수 없는 model.kind {cfg.kind!r}. 등록된 kind: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    return MODEL_REGISTRY[cfg.kind](cfg.params, cfg.fit, seed)
