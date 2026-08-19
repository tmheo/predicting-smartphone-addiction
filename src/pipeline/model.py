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


class FullFitModelAdapter(Protocol):
    """검증 fold 없이 전체 자료를 고정 학습 길이로 맞추는 선택 계약."""

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None: ...


class DatasetReferenceAdapter(Protocol):
    """목표값 비참조 전처리 기준 집합을 받는 선택 계약."""

    def set_dataset_reference(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None: ...


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


class EntryDiagnosticAbortAdapter(Protocol):
    """진입 진단의 비싼 후속 단계를 생략해야 하는 모델 계약."""

    def entry_abort_reason(self) -> str | None: ...


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


def collect_entry_abort_reason(adapter: ModelAdapter) -> str | None:
    """모델이 측정 중 확정한 진입 중단 사유를 검증해 돌려준다."""
    provider = getattr(adapter, "entry_abort_reason", None)
    if provider is None:
        return None
    reason = provider()
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise TypeError("entry_abort_reason()은 비어 있지 않은 문자열 또는 None이어야 한다.")
    return reason


def collect_training_diagnostics(adapter: ModelAdapter) -> dict[str, object] | None:
    """선택 계약의 fold 학습 관측을 유한한 JSON 객체로 검증한다."""
    provider = getattr(adapter, "training_diagnostics", None)
    if provider is None:
        return None
    diagnostics = provider()
    if not isinstance(diagnostics, dict):
        raise TypeError("training_diagnostics()는 dict를 돌려줘야 한다.")
    try:
        json.dumps(diagnostics, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("학습 관측은 유한한 JSON 값이어야 한다.") from exc
    return diagnostics


def fit_full(
    adapter: ModelAdapter,
    X: pd.DataFrame,
    y: pd.Series,
    training_budget: int | None,
    initial_score: pd.Series | None = None,
) -> None:
    """adapter의 전체 자료 재학습 계약을 호출한다.

    전체 자료 재학습은 검증 자료가 없으므로 조기 종료를 허용하지 않는다.
    반복 학습기는 CV에서 미리 확정한 양의 정수 학습 길이를 받고, 반복 수 개념이
    없는 계열은 ``None``을 받는다. 모델별 해석은 adapter가 소유한다.
    """
    provider = getattr(adapter, "fit_full", None)
    if provider is None:
        raise ValueError("이 model adapter는 전체 자료 재학습을 지원하지 않는다.")
    if training_budget is not None and (
        isinstance(training_budget, bool)
        or not isinstance(training_budget, int)
        or training_budget < 1
    ):
        raise ValueError("전체 자료 재학습 길이는 양의 정수 또는 None이어야 한다.")
    provider(X, y, training_budget, initial_score)


def set_dataset_reference(
    adapter: ModelAdapter,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """지원하는 adapter에 train+test 설명변수 전처리 기준 집합을 건넨다."""
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("전처리 기준 집합의 train/test 열이 다르다.")
    provider = getattr(adapter, "set_dataset_reference", None)
    if provider is not None:
        provider(X_train, X_test)


class LightGBMAdapter:
    """LightGBM 이진 분류 adapter."""

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._model = None
        self._uses_initial_score = False

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

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        import lightgbm as lgb

        if training_budget is None:
            raise ValueError("lightgbm 전체 자료 재학습에는 고정 반복 수가 필요하다.")
        params = _resolve_lightgbm_params(self._params, list(X.columns))
        params["n_estimators"] = training_budget
        self._model = lgb.LGBMClassifier(**params, random_state=self._seed)
        self._uses_initial_score = initial_score is not None
        kwargs = {"init_score": initial_score} if self._uses_initial_score else {}
        self._model.fit(X, y, **kwargs)

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

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        import xgboost as xgb

        _reject_initial_score("xgboost", initial_score, None)
        if training_budget is None:
            raise ValueError("xgboost 전체 자료 재학습에는 고정 반복 수가 필요하다.")
        params = dict(self._params)
        params["n_estimators"] = training_budget
        self._model = xgb.XGBClassifier(
            **params,
            random_state=self._seed,
            enable_categorical=True,
        )
        self._model.fit(X, y, verbose=200)

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

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from catboost import CatBoostClassifier

        _reject_initial_score("catboost", initial_score, None)
        if training_budget is None:
            raise ValueError("catboost 전체 자료 재학습에는 고정 반복 수가 필요하다.")
        X, cat_cols = self._prepare(X)
        params = dict(self._params)
        params["iterations"] = training_budget
        self._model = CatBoostClassifier(
            **params,
            random_seed=self._seed,
            cat_features=cat_cols,
            allow_writing_files=False,
        )
        self._model.fit(X, y, verbose=200)

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

    #200 변형 갈래:
    - penalty: l2(기본, lbfgs) 외에 l1·elasticnet(둘 다 saga)을 지원한다.
      l1_ratio는 elasticnet에서만, 그리고 반드시 지정한다.
    - cross_pairs: 두 컬럼의 정확값 쌍을 학습 fold에서 관측된 조합 기준으로
      one-hot하는 명시적 교차 블록. 학습 fold에서 cross_min_count 미만으로
      나타난 쌍과 어느 한쪽이 결측인 행, 검증·테스트에만 있는 쌍은 영벡터
      블록이 된다(단일 컬럼의 미관측 값 처리와 같은 규약). 관측 쌍 수가
      cross_max_card를 넘으면 설정 오류로 거부한다.
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        params = dict(params)
        self._C = float(params.pop("C", 1.0))
        self._max_iter = int(params.pop("max_iter", 2000))
        self._max_card = int(params.pop("onehot_max_card", 10000))
        self._penalty = str(params.pop("penalty", "l2"))
        self._l1_ratio = params.pop("l1_ratio", None)
        solver = params.pop("solver", None)
        self._solver = None if solver is None else str(solver)
        cross_pairs = params.pop("cross_pairs", [])
        self._cross_min_count = int(params.pop("cross_min_count", 1))
        self._cross_max_card = int(params.pop("cross_max_card", 50000))
        if params:
            raise ValueError(f"logistic_onehot이 모르는 params: {sorted(params)}")
        if self._penalty not in {"l2", "l1", "elasticnet"}:
            raise ValueError(f"penalty는 l2·l1·elasticnet 중 하나다(받은 값: {self._penalty!r})")
        if (self._penalty == "elasticnet") != (self._l1_ratio is not None):
            raise ValueError("l1_ratio는 elasticnet에서만, 그리고 반드시 지정한다.")
        if self._l1_ratio is not None:
            self._l1_ratio = float(self._l1_ratio)
        allowed_solvers = {
            "l2": {"lbfgs"},  # exp058 재현성. 다른 L2 solver는 필요할 때 연다.
            "l1": {"saga", "liblinear"},  # liblinear 좌표 하강이 saga보다 훨씬 빠르다(#200).
            "elasticnet": {"saga"},  # sklearn에서 elasticnet은 saga 전용.
        }[self._penalty]
        default_solver = "lbfgs" if self._penalty == "l2" else "saga"
        if self._solver is None:
            self._solver = default_solver
        if self._solver not in allowed_solvers:
            raise ValueError(
                f"penalty {self._penalty!r}에 쓸 수 있는 solver는 {sorted(allowed_solvers)}다"
                f"(받은 값: {self._solver!r})"
            )
        self._cross_pairs: list[tuple[str, str]] = []
        for pair in cross_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(f"cross_pairs 항목은 서로 다른 두 컬럼이어야 한다: {pair}")
            self._cross_pairs.append((str(pair[0]), str(pair[1])))
        if self._cross_min_count < 1:
            raise ValueError(f"cross_min_count는 1 이상이어야 한다(받은 값: {self._cross_min_count})")
        self._fit = fit
        self._seed = seed
        self._model = None
        self._columns: list[str] | None = None
        # 컬럼별 인코딩 스펙: one-hot이면 ("onehot", 카테고리 목록, 결측 지시자 여부),
        # 통과 수치면 ("numeric", 평균, 표준편차). 블록 오프셋은 인코딩 때 계산한다.
        self._specs: dict[str, tuple] = {}
        # 교차 쌍별 학습 fold 관측 조합(pandas MultiIndex). 순서가 곧 블록 내 위치다.
        self._cross_specs: dict[tuple[str, str], pd.MultiIndex] = {}
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
        _reject_initial_score("logistic_onehot", initial_score_tr, initial_score_va)
        self._fit_model(X_tr, y_tr)
        return self._model.predict_proba(self._encode(X_va))[:, 1]

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        _reject_initial_score("logistic_onehot", initial_score, None)
        if training_budget is not None:
            raise ValueError("logistic_onehot은 고정 반복 수 대신 수렴 조건으로 학습한다.")
        self._fit_model(X, y)

    def _fit_model(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> None:
        from sklearn.linear_model import LogisticRegression

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
        self._cross_specs = {}
        for a, b in self._cross_pairs:
            missing = [c for c in (a, b) if c not in self._columns]
            if missing:
                raise ValueError(f"cross_pairs의 컬럼이 입력에 없다: {missing}")
            both = X_tr[[a, b]].dropna()
            counts = both.groupby([a, b], sort=True).size()
            kept = counts[counts >= self._cross_min_count]
            if len(kept) > self._cross_max_card:
                raise ValueError(
                    f"교차 {a}*{b}의 관측 쌍 {len(kept)}개가 cross_max_card="
                    f"{self._cross_max_card}를 넘는다."
                )
            self._cross_specs[(a, b)] = kept.index
        self._train_matrix = self._encode(X_tr)
        # sklearn 1.8부터 penalty 인자 대신 l1_ratio가 페널티 선언이다(0=L2, 1=L1,
        # 사이 값=elasticnet). solver는 __init__에서 페널티와의 조합을 검증했다.
        l1_ratio = {"l2": 0.0, "l1": 1.0}.get(self._penalty, self._l1_ratio)
        self._model = LogisticRegression(
            C=self._C,
            max_iter=self._max_iter,
            solver=self._solver,
            random_state=self._seed,
            l1_ratio=l1_ratio,
        )
        self._model.fit(self._train_matrix, y_tr)
        print(
            f"[logistic_onehot] n_features={self._train_matrix.shape[1]} "
            f"n_iter={int(self._model.n_iter_[0])}"
        )

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
        for pair in self._cross_pairs:
            widths.append(len(self._cross_specs[pair]))
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
        for pair in self._cross_pairs:
            categories = self._cross_specs[pair]
            keys = pd.MultiIndex.from_arrays([X[pair[0]], X[pair[1]]])
            # 미관측 쌍과 어느 한쪽 결측(NaN은 자기 자신과도 다르다)은 -1 → 영벡터.
            codes = categories.get_indexer(keys)
            rows = np.flatnonzero(codes >= 0)
            block = sparse.csr_matrix(
                (np.ones(len(rows)), (rows, codes[rows])),
                shape=(len(X), len(categories)),
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
        features = self._columns + [f"{a}*{b}" for a, b in self._cross_pairs]
        return pd.DataFrame({"feature": features, "gain": gains})


class LookupTransformerAdapter:
    """정확값 lookup embedding Transformer adapter. (#58)

    구현은 torch가 필요한 lookup_transformer 모듈에 있고 여기서 lazy import한다.
    기본 어휘·rank-gauss 분위는 학습 fold에서만 맞춘다.
    명시적인 실험은 목표값 비참조 train+test 전처리 기준 집합을 사용할 수 있다.
    gain importance가 없어 검증 fold permutation importance(AUC 하락 폭)를 gain
    컬럼으로 돌려준다(ADR 0001 #97의 계열 무관 중요도). 환산은 시드로 결정적이다.
    fold_seed_offsets가 여러 개면 파이프라인 시드에 각 offset을 더한 초기화로
    같은 fold를 학습하고 확률 예측을 평균한다(#127).
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._impl = None
        self._dataset_reference: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def set_dataset_reference(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        """정확값 어휘와 분위 변환에 쓸 목표값 비참조 기준 집합을 보관한다."""
        if self._impl is not None:
            raise RuntimeError("전처리 기준 집합은 Lookup-Transformer 학습 전에 정해야 한다.")
        self._dataset_reference = (X_train, X_test)

    def _new_impl(self):
        from . import lookup_transformer

        impl = lookup_transformer.LookupTransformerFold(self._params, self._seed)
        if self._dataset_reference is not None:
            impl.set_dataset_reference(*self._dataset_reference)
        return impl

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        _reject_initial_score("lookup_transformer", initial_score_tr, initial_score_va)
        self._impl = self._new_impl()
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        _reject_initial_score("lookup_transformer", initial_score, None)
        if training_budget is None:
            raise ValueError("lookup_transformer 전체 자료 재학습에는 고정 epoch 수가 필요하다.")
        self._impl = self._new_impl()
        self._impl.fit_full(X, y, training_budget)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("lookup_transformer", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def training_diagnostics(self) -> dict[str, object]:
        return self._impl.training_diagnostics()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return AdapterDiagnostics(observations=self.training_diagnostics())


class ContextualizedSplineTransformerAdapter:
    """단변량 선행 학습과 얕은 상호작용을 결합한 모델 adapter. (#149)

    구현은 torch가 필요한 contextualized_spline_transformer 모듈에 있고 여기서
    지연 import한다. 수치 표준화, knot와 정확값 어휘는 outer 학습 부분에서만
    맞추며, ``numeric_mode``로 M0 조각선형 경로와 A0 주기 제거 대조를 고른다.
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        mode = params.get("numeric_mode", "spline")
        if mode not in {"spline", "periodic"}:
            raise ValueError(
                f"numeric_mode는 ['periodic', 'spline'] 중 하나여야 한다: {mode!r}"
            )
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
        from . import contextualized_spline_transformer

        _reject_initial_score(
            "contextualized_spline_transformer", initial_score_tr, initial_score_va
        )
        if self._fit:
            raise ValueError(
                "contextualized_spline_transformer가 모르는 fit 설정: "
                f"{sorted(self._fit)}"
            )
        self._impl = (
            contextualized_spline_transformer.ContextualizedSplineTransformerFold(
                self._params, self._seed
            )
        )
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from . import contextualized_spline_transformer

        _reject_initial_score(
            "contextualized_spline_transformer", initial_score, None
        )
        if self._fit:
            raise ValueError(
                "contextualized_spline_transformer가 모르는 fit 설정: "
                f"{sorted(self._fit)}"
            )
        if training_budget is None:
            raise ValueError(
                "contextualized_spline_transformer 전체 자료 재학습에는 "
                "고정 epoch 수가 필요하다."
            )
        self._impl = (
            contextualized_spline_transformer.ContextualizedSplineTransformerFold(
                self._params, self._seed
            )
        )
        self._impl.fit_full(X, y, training_budget)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("contextualized_spline_transformer", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def training_diagnostics(self) -> dict[str, object]:
        return self._impl.training_diagnostics()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()


class ScalarTokenTransformerAdapter:
    """ReLU·주기 스칼라 token과 결합 블록을 쓰는 모델 adapter. (#178)

    구현은 torch가 필요한 ``scalar_token_transformer`` 모듈에 있고 여기서 지연
    import한다. 범주값 스칼라화와 분위 변환은 outer 학습 부분에서만 맞춘다.
    ``mixing``으로 M0 attention과 매개변수 규모를 맞춘 A0 열별 MLP를 고른다.
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        mixing = params.get("mixing", "attention")
        if mixing not in {"attention", "token_mlp"}:
            raise ValueError(
                "mixing은 ['attention', 'token_mlp'] 중 하나여야 한다: "
                f"{mixing!r}"
            )
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
        from . import scalar_token_transformer

        _reject_initial_score(
            "scalar_token_transformer", initial_score_tr, initial_score_va
        )
        if self._fit:
            raise ValueError(
                "scalar_token_transformer가 모르는 fit 설정: "
                f"{sorted(self._fit)}"
            )
        self._impl = scalar_token_transformer.ScalarTokenTransformerFold(
            self._params, self._seed
        )
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("scalar_token_transformer", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()


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

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from . import tabm

        _reject_initial_score("tabm", initial_score, None)
        if training_budget is None:
            raise ValueError("tabm 전체 자료 재학습에는 고정 epoch 수가 필요하다.")
        self._impl = tabm.TabMFold(self._params, self._seed)
        self._impl.fit_full(X, y, training_budget)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tabm", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def training_diagnostics(self) -> dict[str, object]:
        return self._impl.training_diagnostics()


class RealMLPAdapter:
    """고정 일정 RealMLP adapter. (#180)

    공개 노트북의 fold별 전처리와 내부 OOF 목표 인코딩, 병렬 모형,
    고정 epoch 학습은 realmlp 모듈에서 구현한다.
    같은 fold 안에서 초기화가 다른 두 예측을 평균하며, run.py가 담당하는
    파이프라인 시드 평균과는 분리한다.
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
        from . import realmlp

        _reject_initial_score("realmlp", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"realmlp가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = realmlp.RealMLPFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from . import realmlp

        _reject_initial_score("realmlp", initial_score, None)
        if self._fit:
            raise ValueError(f"realmlp가 모르는 fit 설정: {sorted(self._fit)}")
        if training_budget is None:
            raise ValueError("realmlp 전체 자료 재학습에는 고정 epoch 수가 필요하다.")
        self._impl = realmlp.RealMLPFold(self._params, self._seed)
        self._impl.fit_full(X, y, training_budget)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("realmlp", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()

    def training_diagnostics(self) -> dict[str, object]:
        return self._impl.training_diagnostics()


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

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from . import tabpfn3

        _reject_initial_score("tabpfn3", initial_score, None)
        if training_budget is not None:
            raise ValueError("tabpfn3는 CV에서 고를 반복 학습 길이가 없다.")
        self._impl = tabpfn3.TabPFN3Fold(self._params, self._seed)
        self._impl.fit_full(X, y)

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


class TabRAdapter:
    """공식 기본 TabR(조기 종료, 매 배치 문맥 재계산)을 제공하는 검색형 모델 adapter. (#199)"""

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
        from . import tabr

        _reject_initial_score("tabr", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"tabr가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = tabr.TabRFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tabr", initial_score, None)
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

    def entry_abort_reason(self) -> str | None:
        return self._impl.entry_abort_reason()


class AMFormerAdapter:
    """논문 수식으로 독립 구현한 AMFormer adapter. (#144)"""

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
        from . import amformer

        _reject_initial_score("amformer", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"amformer가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = amformer.AMFormerFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("amformer", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()


class TabCNNAdapter:
    """공개 구조를 누출 없이 고친 표 합성곱망과 제거 대조 adapter. (#177)"""

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
        from . import tab_cnn

        _reject_initial_score("tab_cnn", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"tab_cnn이 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = tab_cnn.TabCNNFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from . import tab_cnn

        _reject_initial_score("tab_cnn", initial_score, None)
        if self._fit:
            raise ValueError(f"tab_cnn이 모르는 fit 설정: {sorted(self._fit)}")
        if training_budget is None:
            raise ValueError("tab_cnn 전체 자료 재학습에는 고정 epoch 수가 필요하다.")
        self._impl = tab_cnn.TabCNNFold(self._params, self._seed)
        self._impl.fit_full(X, y, training_budget)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tab_cnn", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()

    def training_diagnostics(self) -> dict[str, object]:
        return self._impl.training_diagnostics()


class XRFMAdapter:
    """공식 xRFM 재귀 특성 커널 머신 adapter. (#198)"""

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
        from . import xrfm_fold

        _reject_initial_score("xrfm", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"xrfm이 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = xrfm_fold.XRFMFold(self._params, self._seed)
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("xrfm", initial_score, None)
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
    "contextualized_spline_transformer": ContextualizedSplineTransformerAdapter,
    "scalar_token_transformer": ScalarTokenTransformerAdapter,
    "tabm": TabMAdapter,
    "realmlp": RealMLPAdapter,
    "tabpfn3": TabPFN3Adapter,
    "tabr": TabRAdapter,
    "tabr_s": TabRSAdapter,
    "tabiclv2": TabICLv2Adapter,
    "trompt": TromptAdapter,
    "amformer": AMFormerAdapter,
    "tab_cnn": TabCNNAdapter,
    "xrfm": XRFMAdapter,
}


def create(cfg: ModelConfig, seed: int) -> ModelAdapter:
    """cfg.kind를 레지스트리에서 해석해 fold 하나의 adapter를 만든다."""
    if cfg.kind not in MODEL_REGISTRY:
        raise ValueError(
            f"알 수 없는 model.kind {cfg.kind!r}. 등록된 kind: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    return MODEL_REGISTRY[cfg.kind](cfg.params, cfg.fit, seed)
