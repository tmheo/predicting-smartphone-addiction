"""모델 계열. kind -> adapter 팩토리 레지스트리로 학습기 구현을 dispatch한다. (#72)

cv 루프는 어떤 모델인지 모른다: cfg.model.kind를 여기 레지스트리에서 해석해
adapter 인스턴스를 만들고 fit/predict/importance만 부른다.
params 해석, 시드 적용(random_state 등 모델별 이름), fit 인자(early_stopping_rounds)
해석은 adapter가 소유하고, 학습된 모델 상태는 인스턴스 안에 갇힌다.
plan.REGISTRY와 같은 패턴: 새 모델 계열은 adapter를 구현하고 MODEL_REGISTRY에 등록한다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class AdapterDiagnostics:
    """진입 진단에 추가할 모델별 assertion과 관측값.

    검색·문맥형 모델은 아래 ``ASSERT_*`` 이름을 써서 후보 저장소가 학습 행으로만
    구성됐는지, 검증 라벨을 문맥에 쓰지 않았는지, 조회 결과에서 자기 행을 뺐는지
    보고한다. 다른 모델은 같은 스키마의 observations에 필요한 측정값만 추가한다.
    """

    assertions: dict[str, bool] = field(default_factory=dict)
    observations: dict[str, object] = field(default_factory=dict)


ASSERT_CANDIDATE_STORE_TRAIN_ONLY = "candidate_store_training_only"
ASSERT_VALIDATION_LABELS_EXCLUDED = "validation_labels_excluded_from_context"
ASSERT_SELF_ROWS_EXCLUDED = "self_rows_excluded_from_candidates"


class EntryDiagnosticAdapter(Protocol):
    """모델별 진입 진단을 제공하는 선택 계약."""

    def entry_diagnostics(self) -> AdapterDiagnostics: ...


def collect_entry_diagnostics(adapter: ModelAdapter) -> AdapterDiagnostics:
    """선택 계약을 구현한 adapter의 진단을 공통 스키마로 검증해 돌려준다."""
    provider = getattr(adapter, "entry_diagnostics", None)
    if provider is None:
        return AdapterDiagnostics()
    diagnostics = provider()
    if not isinstance(diagnostics, AdapterDiagnostics):
        raise TypeError("entry_diagnostics()는 AdapterDiagnostics를 돌려줘야 한다.")
    invalid = {
        name: value
        for name, value in diagnostics.assertions.items()
        if not isinstance(value, bool)
    }
    if invalid:
        raise TypeError(f"adapter 진단 assertion 값은 bool이어야 한다: {invalid}")
    try:
        json.dumps(diagnostics.observations, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("adapter 진단 observations는 유한한 JSON 값이어야 한다.") from exc
    return diagnostics


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

        params = _resolve_lightgbm_params(self._params, list(X_tr.columns))
        self._model = lgb.LGBMClassifier(**params, random_state=self._seed)
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


def _resolve_lightgbm_params(params: dict, feature_names: list[str]) -> dict:
    """열 이름별 max_bin 재정의를 LightGBM의 위치 목록으로 바꾼다.

    LightGBM 자체의 ``max_bin_by_feature``는 최종 행렬 순서와 길이가 같은 정수
    목록만 받는다. 설정에서는 ``{열 이름: max_bin}`` 매핑도 허용해 피처 계획의
    열 순서가 바뀌어도 다른 열에 조용히 적용되지 않게 한다. 목록 입력은 LightGBM
    원형을 써야 하는 경우를 위해 그대로 통과시킨다.
    """
    resolved = dict(params)
    by_feature = resolved.get("max_bin_by_feature")
    if not isinstance(by_feature, dict):
        return resolved
    if "max_bin" not in resolved:
        raise ValueError("열 이름별 max_bin_by_feature에는 기본값 max_bin이 필요하다.")

    unknown = sorted(set(by_feature) - set(feature_names))
    if unknown:
        raise ValueError(
            "max_bin_by_feature에 학습 행렬에 없는 열이 있다: " + ", ".join(unknown)
        )
    invalid = {
        name: value
        for name, value in by_feature.items()
        if isinstance(value, bool) or not isinstance(value, int) or value < 2
    }
    if invalid:
        raise ValueError(f"max_bin_by_feature 값은 2 이상의 정수여야 한다: {invalid}")

    default = resolved["max_bin"]
    resolved["max_bin_by_feature"] = [
        by_feature.get(name, default) for name in feature_names
    ]
    return resolved


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


class LogisticOnehotAdapter:
    """정확값 one-hot 로지스틱 회귀 adapter. (#56)

    각 컬럼의 학습 fold 정확값을 그대로 카테고리로 보고 one-hot한 희소 행렬에
    L2 로지스틱 회귀를 학습한다. 인코딩은 학습 fold 값 집합만 쓰므로 누출이 없고,
    검증·테스트에만 있는 값은 해당 컬럼 블록이 영벡터가 된다.
    결측은 학습 fold에 결측이 있는 컬럼에 한해 별도 지시자 카테고리가 된다.
    카디널리티가 onehot_max_card를 넘는 컬럼(placebo 등 연속 잡음)은 학습 fold
    평균·표준편차로 표준화한 수치 열로 통과시키고 결측은 0(평균)으로 둔다.
    importance는 컬럼 블록별 선형 기여(학습 fold 점수 조각)의 표준편차를 gain으로
    돌려준다. 표준화 수치 열에서는 |coef|와 같아 블록 간 축척이 맞는다.
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        params = dict(params)
        self._C = float(params.pop("C", 1.0))
        self._max_iter = int(params.pop("max_iter", 2000))
        self._max_card = int(params.pop("onehot_max_card", 10000))
        if params:
            raise ValueError(f"logistic_onehot이 모르는 params: {sorted(params)}")
        self._fit = fit
        self._seed = seed
        self._model = None
        self._columns: list[str] | None = None
        # 컬럼별 인코딩 스펙: one-hot이면 ("onehot", 카테고리 목록, 결측 지시자 여부),
        # 통과 수치면 ("numeric", 평균, 표준편차). 블록 오프셋은 인코딩 때 계산한다.
        self._specs: dict[str, tuple] = {}
        self._train_matrix = None  # importance용 학습 fold 인코딩 행렬.

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression

        _reject_initial_score("logistic_onehot", initial_score_tr, initial_score_va)
        self._columns = list(X_tr.columns)
        self._specs = {}
        for col in self._columns:
            values = X_tr[col]
            distinct = pd.unique(values.dropna())
            if len(distinct) <= self._max_card:
                self._specs[col] = ("onehot", list(distinct), bool(values.isna().any()))
            else:
                mean = float(values.mean())
                std = float(values.std())
                self._specs[col] = ("numeric", mean, std if std > 0 else 1.0)
        self._train_matrix = self._encode(X_tr)
        self._model = LogisticRegression(
            C=self._C, max_iter=self._max_iter, solver="lbfgs", random_state=self._seed
        )
        self._model.fit(self._train_matrix, y_tr)
        print(
            f"[logistic_onehot] n_features={self._train_matrix.shape[1]} "
            f"n_iter={int(self._model.n_iter_[0])}"
        )
        return self._model.predict_proba(self._encode(X_va))[:, 1]

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("logistic_onehot", initial_score, None)
        return self._model.predict_proba(self._encode(X))[:, 1]

    def _block_widths(self) -> list[int]:
        widths = []
        for col in self._columns:
            spec = self._specs[col]
            widths.append(1 if spec[0] == "numeric" else len(spec[1]) + int(spec[2]))
        return widths

    def _encode(self, X: pd.DataFrame):
        from scipy import sparse

        assert list(X.columns) == self._columns, "인코딩 입력 컬럼이 학습 때와 다르다."
        blocks = []
        for col in self._columns:
            spec = self._specs[col]
            if spec[0] == "numeric":
                _, mean, std = spec
                dense = ((X[col] - mean) / std).fillna(0.0).to_numpy(dtype="float64")
                blocks.append(sparse.csr_matrix(dense.reshape(-1, 1)))
                continue
            _, categories, has_nan = spec
            width = len(categories) + int(has_nan)
            values = X[col]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype(object)
            # Index.get_indexer는 학습 fold에 없던 값과 결측을 -1로 돌려준다.
            # pandas 4에서 금지될 미지 값을 가진 Categorical 생성을 피한다.
            codes = pd.Index(categories).get_indexer(values).astype("int64")
            if has_nan:
                codes = np.where(values.isna().to_numpy(), len(categories), codes)
            seen = codes >= 0  # 미관측 값(그리고 지시자 없는 결측)은 영벡터 블록.
            rows = np.flatnonzero(seen)
            block = sparse.csr_matrix(
                (np.ones(len(rows)), (rows, codes[seen])), shape=(len(X), width)
            )
            blocks.append(block)
        return sparse.hstack(blocks, format="csr")

    def importance(self) -> pd.DataFrame:
        coef = self._model.coef_[0]
        gains = []
        offset = 0
        for width in self._block_widths():
            contribution = self._train_matrix[:, offset : offset + width] @ coef[
                offset : offset + width
            ]
            gains.append(float(np.std(contribution)))
            offset += width
        return pd.DataFrame({"feature": self._columns, "gain": gains})


class LookupTransformerAdapter:
    """정확값 lookup embedding Transformer adapter. (#58)

    구현은 torch가 필요한 lookup_transformer 모듈에 있고 여기서 lazy import한다.
    어휘·rank-gauss 분위는 학습 fold에서만 fit하고(outer fold 규율), gain
    importance가 없어 검증 fold permutation importance(AUC 하락 폭)를 gain
    컬럼으로 돌려준다(ADR 0001 #97의 계열 무관 중요도). 환산은 시드로 결정적이다.
    fold_seed_offsets가 여러 개면 파이프라인 시드에 각 offset을 더한 초기화로
    같은 fold를 학습하고 확률 예측을 평균한다(#127).
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._impl = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from . import lookup_transformer

        _reject_initial_score("lookup_transformer", initial_score_tr, initial_score_va)
        self._impl = lookup_transformer.LookupTransformerFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("lookup_transformer", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()


class TabMAdapter:
    """TabM(pytabkit) adapter. (#61)

    구현은 pytabkit이 필요한 tabm 모듈에 있고 여기서 lazy import한다.
    수치 열의 남은 NaN 중앙값 대체는 학습 fold 통계만 쓰고(outer fold 규율),
    fold 안 시드 평균(원문 N_SEEDS=3)으로 예측 하나를 만든다. gain importance가
    없어 검증 fold permutation importance(AUC 하락 폭)를 gain 컬럼으로 돌려준다
    (ADR 0001 #97의 계열 무관 중요도). 환산은 시드로 결정적이다.
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._impl = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from . import tabm

        _reject_initial_score("tabm", initial_score_tr, initial_score_va)
        self._impl = tabm.TabMFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tabm", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()


class TabPFN3Adapter:
    """TabPFN-3 adapter. (#102)

    구현은 GPU와 gated 가중치가 필요한 tabpfn3 모듈에 있고 여기서 lazy import한다.
    fold 학습 문맥은 캐시하고 검증·테스트는 작은 청크로 예측한다. gain importance가
    없어 검증 fold 부분표본 permutation importance를 gain 컬럼으로 돌려준다.
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._impl = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from . import tabpfn3

        _reject_initial_score("tabpfn3", initial_score_tr, initial_score_va)
        self._impl = tabpfn3.TabPFN3Fold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tabpfn3", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()


class TabRSAdapter:
    """TabR-S와 첫 epoch 뒤 문맥 고정을 제공하는 검색형 모델 adapter. (#142)"""

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._impl = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from . import tabr_s

        _reject_initial_score("tabr_s", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"tabr_s가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = tabr_s.TabRSFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tabr_s", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()


class TabICLv2Adapter:
    """공식 TabICLv2 문맥 추론기 adapter. (#143)"""

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._impl = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from . import tabiclv2

        _reject_initial_score("tabiclv2", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"tabiclv2가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = tabiclv2.TabICLv2Fold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tabiclv2", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()


class TromptAdapter:
    """TALENT 기준 Trompt의 행별 열 중요도와 다중 Cell 학습 adapter. (#145)"""

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._impl = None

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        from . import trompt

        _reject_initial_score("trompt", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"trompt가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = trompt.TromptFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("trompt", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()


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
    "logistic_onehot": LogisticOnehotAdapter,
    "lookup_transformer": LookupTransformerAdapter,
    "tabm": TabMAdapter,
    "tabpfn3": TabPFN3Adapter,
    "tabr_s": TabRSAdapter,
    "tabiclv2": TabICLv2Adapter,
    "trompt": TromptAdapter,
}


def create(cfg: ModelConfig, seed: int) -> ModelAdapter:
    """cfg.kind를 레지스트리에서 해석해 fold 하나의 adapter를 만든다."""
    if cfg.kind not in MODEL_REGISTRY:
        raise ValueError(
            f"알 수 없는 model.kind {cfg.kind!r}. 등록된 kind: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    return MODEL_REGISTRY[cfg.kind](cfg.params, cfg.fit, seed)
