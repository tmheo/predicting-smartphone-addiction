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
from sklearn.metrics import roc_auc_score

from .config import ModelConfig, TrainingStateConfig
from .data import TARGET
from .training_length import (
    FIXED_COUNT,
    ONE_BASED_COUNT,
    ZERO_BASED_POSITION,
    RawTrainingLengthSelection,
    TrainingLengthContract,
    TrainingLengthDeclaration,
    TrainingLengthError,
)


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


@dataclass(frozen=True)
class FoldTrainingStatePoint:
    """한 물리 학습 궤적에서 포착한 후보 시점 하나의 fold 결과."""

    completed_epochs: int
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    importance: pd.DataFrame
    training_diagnostics: dict[str, object] | None
    training_length_declaration: TrainingLengthDeclaration


@dataclass(frozen=True)
class FoldTrainingStateTrajectory:
    """같은 입력, 난수, 순서와 일정 지평을 공유하는 후보 시점 묶음."""

    schedule_horizon_epochs: int
    trajectory_end_epochs: int
    points: tuple[FoldTrainingStatePoint, ...]


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
        raise TypeError(
            "adapter 진단 observations는 유한한 JSON 값이어야 한다."
        ) from exc
    return diagnostics


def collect_entry_abort_reason(adapter: ModelAdapter) -> str | None:
    """모델이 측정 중 확정한 진입 중단 사유를 검증해 돌려준다."""
    provider = getattr(adapter, "entry_abort_reason", None)
    if provider is None:
        return None
    reason = provider()
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise TypeError(
            "entry_abort_reason()은 비어 있지 않은 문자열 또는 None이어야 한다."
        )
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


def fit_paired_training_lengths(
    adapter: ModelAdapter,
    kind: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    training_lengths: tuple[int, ...] | None,
    initial_score_tr: pd.Series | None = None,
    initial_score_va: pd.Series | None = None,
) -> np.ndarray:
    """출처 실행에서 고정한 노출량으로 학습하고 검증 예측을 만든다.

    반복 수 개념이 없는 로지스틱 회귀만 기존 적합 경로를 쓴다.
    반복형 계열은 검증 목표값을 학습 종료 결정에 쓰지 않고, 출처에서 관측한
    내부 구성원별 길이만큼 정확히 학습한다.
    """
    if training_lengths is None:
        if kind != "logistic_onehot":
            raise ValueError(f"{kind}: 반복형 모형의 짝비교 학습 길이가 없다.")
        return np.asarray(
            adapter.fit(
                X_tr,
                y_tr,
                X_va,
                y_va,
                initial_score_tr,
                initial_score_va,
            ),
            dtype="float64",
        )
    if not training_lengths or any(
        isinstance(length, bool) or not isinstance(length, int) or length < 1
        for length in training_lengths
    ):
        raise ValueError(f"{kind}: 짝비교 학습 길이는 양의 정수 묶음이어야 한다.")
    provider = getattr(adapter, "fit_paired_training_lengths", None)
    if provider is not None:
        provider(X_tr, y_tr, training_lengths, initial_score_tr)
    else:
        if len(training_lengths) != 1:
            raise ValueError(
                f"{kind}: 내부 구성원 {len(training_lengths)}개의 서로 다른 학습 길이를 "
                "지원하지 않는다."
            )
        fit_full(adapter, X_tr, y_tr, training_lengths[0], initial_score_tr)
    prediction = adapter.predict(X_va, initial_score_va)
    prediction = np.asarray(prediction, dtype="float64")
    if prediction.shape != (len(X_va),) or not np.isfinite(prediction).all():
        raise ValueError(f"{kind}: 고정 노출량 검증 예측이 유한한 1차원 배열이 아니다.")
    return prediction


def paired_permutation_importance(
    adapter: ModelAdapter,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    seed: int,
    initial_score_va: pd.Series | None = None,
) -> pd.DataFrame:
    """고정 노출량 경로에서 검증 목표값을 오직 순열 중요도 평가에만 쓴다."""
    if len(X_va) > 50_000:
        keep = np.random.default_rng(seed).choice(len(X_va), size=50_000, replace=False)
        keep.sort()
        X_va = X_va.iloc[keep].copy()
        y_va = y_va.iloc[keep].copy()
        if initial_score_va is not None:
            initial_score_va = initial_score_va.iloc[keep].copy()
    base_prediction = np.asarray(
        adapter.predict(X_va, initial_score_va), dtype="float64"
    )
    base_auc = roc_auc_score(y_va, base_prediction)
    gains: list[float] = []
    for column_index, column in enumerate(X_va.columns):
        drops: list[float] = []
        for repeat in range(3):
            generator = np.random.default_rng(
                seed * 10007 + column_index * 101 + repeat
            )
            permuted = X_va.copy()
            order = generator.permutation(len(permuted))
            permuted[column] = X_va[column].iloc[order].set_axis(permuted.index)
            score = roc_auc_score(
                y_va,
                np.asarray(
                    adapter.predict(permuted, initial_score_va), dtype="float64"
                ),
            )
            drops.append(float(base_auc - score))
        gains.append(float(np.mean(drops)))
    return pd.DataFrame({"feature": list(X_va.columns), "gain": gains})


def fit_predict_training_states(
    adapter: ModelAdapter,
    kind: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_test: pd.DataFrame,
    state: TrainingStateConfig,
) -> FoldTrainingStateTrajectory:
    """여러 고정 시점 선택 계약을 호출하고 완전한 fold 묶음을 검증한다."""
    provider = getattr(adapter, "fit_predict_training_states", None)
    if provider is None:
        raise ValueError(
            f"{kind} model adapter는 여러 학습 시점 실행을 지원하지 않는다."
        )
    trajectory = provider(X_tr, y_tr, X_va, y_va, X_test, state)
    if not isinstance(trajectory, FoldTrainingStateTrajectory):
        raise TypeError(
            "fit_predict_training_states()는 FoldTrainingStateTrajectory를 돌려줘야 한다."
        )
    if trajectory.schedule_horizon_epochs != state.schedule_horizon_epochs:
        raise ValueError("모델이 보고한 학습률 일정 지평이 설정 계약과 다르다.")
    if trajectory.trajectory_end_epochs != state.trajectory_end_epochs:
        raise ValueError("모델이 보고한 궤적 종료 시점이 설정 계약과 다르다.")
    completed = tuple(point.completed_epochs for point in trajectory.points)
    if completed != state.candidates:
        raise ValueError(
            f"모델이 포착한 학습 시점이 설정 계약과 다르다: {completed} != {state.candidates}"
        )
    for point in trajectory.points:
        _validate_training_state_point(point, kind, len(X_va), len(X_test))
    return trajectory


def _validate_training_state_point(
    point: FoldTrainingStatePoint,
    kind: str,
    validation_rows: int,
    test_rows: int,
) -> None:
    if not isinstance(point, FoldTrainingStatePoint):
        raise TypeError("학습 시점 결과 형식이 FoldTrainingStatePoint가 아니다.")
    for label, values, expected_rows in (
        ("검증", point.validation_prediction, validation_rows),
        ("시험", point.test_prediction, test_rows),
    ):
        if not isinstance(values, np.ndarray) or values.shape != (expected_rows,):
            raise ValueError(
                f"학습 시점 {point.completed_epochs}의 {label} 예측 모양이 다르다: "
                f"{getattr(values, 'shape', None)} != {(expected_rows,)}"
            )
        if (
            not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"학습 시점 {point.completed_epochs}의 {label} 예측은 유한한 부동소수점이어야 한다."
            )
    if list(point.importance.columns) != ["feature", "gain"]:
        raise ValueError(
            f"학습 시점 {point.completed_epochs}의 중요도 열이 다르다: "
            f"{list(point.importance.columns)}"
        )
    if point.importance["feature"].duplicated().any():
        raise ValueError(
            f"학습 시점 {point.completed_epochs}의 중요도 특성이 중복됐다."
        )
    gain = point.importance["gain"].to_numpy()
    if not np.issubdtype(gain.dtype, np.floating) or not np.isfinite(gain).all():
        raise ValueError(
            f"학습 시점 {point.completed_epochs}의 중요도는 유한해야 한다."
        )
    if point.training_diagnostics is not None:
        try:
            json.dumps(point.training_diagnostics, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("학습 시점 관측은 유한한 JSON 객체여야 한다.") from exc
    _validate_training_length_declaration(point.training_length_declaration, kind)
    observed = {
        (
            selection.raw_value + 1
            if point.training_length_declaration.raw_meaning == ZERO_BASED_POSITION
            else selection.raw_value
        )
        for selection in point.training_length_declaration.selections
    }
    if observed != {point.completed_epochs}:
        raise TrainingLengthError(
            f"학습 시점 근거가 실제 종료 {point.completed_epochs}와 다르다: {sorted(observed)}"
        )


def fit_full_training_state(
    adapter: ModelAdapter,
    X: pd.DataFrame,
    y: pd.Series,
    state: TrainingStateConfig,
    initial_score: pd.Series | None = None,
) -> dict[str, object]:
    """선택된 시점에서 끝내되 원래 일정 지평을 보존해 전체 자료를 학습한다."""
    provider = getattr(adapter, "fit_full_training_state", None)
    if provider is None:
        raise ValueError(
            "이 model adapter는 학습 시점 후보 전체 자료 재학습을 지원하지 않는다."
        )
    diagnostics = provider(X, y, state, initial_score)
    if not isinstance(diagnostics, dict):
        raise TypeError(
            "fit_full_training_state()는 실제 학습 계약 진단을 돌려줘야 한다."
        )
    expected = {
        "completed_epochs": state.selected,
        "schedule_horizon_epochs": state.schedule_horizon_epochs,
        "state_kind": state.state_kind,
    }
    actual = {key: diagnostics.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"전체 자료 학습의 실제 시점 계약이 요청과 다르다: {actual} != {expected}"
        )
    try:
        json.dumps(diagnostics, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "전체 자료 학습 시점 진단은 유한한 JSON 객체여야 한다."
        ) from exc
    return diagnostics


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


class TrainingLengthAdapter(Protocol):
    """관측 학습 길이 근거를 선언하는 선택 계약. (#372)

    반복형 계열만 구현한다. 반복 수가 없는 계열은 이 계약을 구현하지 않으며,
    그래서 없는 관측을 지어내지 않는다. 좌표 가운데 시드와 바깥쪽 분할은 fold
    실행부가 채우므로 연결부는 원시 값과 내부 구성원 좌표까지만 안다.
    """

    def training_length_evidence(self) -> TrainingLengthDeclaration: ...


# 계열 -> 조기 종료 또는 기존 원시 선택값 계약.
# 아래 고정 반복 트리 대체 계약과 함께 어떤 실행이 `+1`을 받는지 정하는 유일한 자리다.
# 원시 의미는 #326과 #413이 계열별 일정에 따라 확정했고, 연결부는 등록 계약으로만 선언한다.
TRAINING_LENGTH_CONTRACTS: dict[str, TrainingLengthContract] = {
    "lightgbm": TrainingLengthContract("lightgbm", "best_iteration_", ONE_BASED_COUNT),
    "xgboost": TrainingLengthContract("xgboost", "best_iteration", ZERO_BASED_POSITION),
    "catboost": TrainingLengthContract(
        "catboost", "get_best_iteration()", ZERO_BASED_POSITION
    ),
    "lookup_transformer": TrainingLengthContract(
        "lookup_transformer", "best_epoch", ZERO_BASED_POSITION
    ),
    "contextualized_spline_transformer": TrainingLengthContract(
        "contextualized_spline_transformer", "best_epoch", ONE_BASED_COUNT
    ),
    "scalar_token_transformer": TrainingLengthContract(
        "scalar_token_transformer", "best_epoch", ONE_BASED_COUNT
    ),
    "tab_cnn": TrainingLengthContract("tab_cnn", "best_epoch", ONE_BASED_COUNT),
    "tabm": TrainingLengthContract("tabm", "selected_epoch_count", ONE_BASED_COUNT),
    "realmlp": TrainingLengthContract("realmlp", "fixed_epochs", FIXED_COUNT),
}

# 조기 종료를 쓰지 않는 트리 변형은 검증이 고른 위치가 아니라 설정이 고정한
# 실제 부스팅 횟수를 근거로 낸다. 기존 계열 계약은 그대로 두고 이 대체 계약만
# 조건부로 허용해, 조기 종료 설정의 원시 의미가 조용히 바뀌지 않게 한다. (#413)
FIXED_COUNT_TREE_CONTRACTS: dict[str, TrainingLengthContract] = {
    "lightgbm": TrainingLengthContract("lightgbm", "n_estimators", FIXED_COUNT),
    "xgboost": TrainingLengthContract("xgboost", "n_estimators", FIXED_COUNT),
    "catboost": TrainingLengthContract("catboost", "iterations", FIXED_COUNT),
}


def _training_length_contract_variants(kind: str) -> tuple[TrainingLengthContract, ...]:
    primary = TRAINING_LENGTH_CONTRACTS.get(kind)
    if primary is None:
        return ()
    fixed = FIXED_COUNT_TREE_CONTRACTS.get(kind)
    return (primary,) if fixed is None else (primary, fixed)


def _registered_training_length_contracts(
    kind: str,
) -> tuple[TrainingLengthContract, ...]:
    """등록된 계열의 학습 길이 계약을 돌려주고, 미등록 계열은 거부한다."""
    contracts = _training_length_contract_variants(kind)
    if not contracts:
        raise TrainingLengthError(
            f"{kind}는 관측 학습 길이 계약이 등록되지 않은 계열인데 근거를 선언했다."
        )
    return contracts


def collect_training_length_declaration(
    adapter: ModelAdapter, kind: str
) -> TrainingLengthDeclaration | None:
    """선택 계약을 구현한 adapter의 원시 근거 선언을 계열 계약과 대조해 돌려준다.

    계약을 구현하지 않은 계열은 `None`을 돌려준다. 구현한 계열은 자기 kind로
    등록된 계약과 모델 계열, 원시 필드, 원시 의미가 모두 같아야 한다. 이 대조가
    표와 코드가 갈라지는 것을 실행 시점에 막는다.
    """
    provider = getattr(adapter, "training_length_evidence", None)
    if provider is None:
        return None
    _registered_training_length_contracts(kind)
    declaration = provider()
    if not isinstance(declaration, TrainingLengthDeclaration):
        raise TypeError(
            "training_length_evidence()는 TrainingLengthDeclaration을 돌려줘야 한다."
        )
    _validate_training_length_declaration(declaration, kind)
    return declaration


def _validate_training_length_declaration(
    declaration: TrainingLengthDeclaration, kind: str
) -> None:
    """한 선언이 등록된 모델 계열의 원시 학습 길이 계약과 같은지 확인한다."""
    if not isinstance(declaration, TrainingLengthDeclaration):
        raise TypeError("학습 길이 선언은 TrainingLengthDeclaration이어야 한다.")
    contracts = _registered_training_length_contracts(kind)
    declared_contract = (
        declaration.model_family,
        declaration.raw_field,
        declaration.raw_meaning,
    )
    expected_contracts = {
        (contract.model_family, contract.raw_field, contract.raw_meaning)
        for contract in contracts
    }
    if declared_contract not in expected_contracts:
        raise TrainingLengthError(
            f"{kind} 연결부 선언의 model_family/raw_field/raw_meaning이 등록된 계약과 "
            f"다르다: {declared_contract!r} (허용: {sorted(expected_contracts)!r})"
        )


def _early_stopping_rounds(fit: dict, kind: str) -> int | None:
    """조기 종료 설정을 검증한다. 키가 없으면 고정 반복 일정이다."""
    if "early_stopping_rounds" not in fit:
        return None
    value = fit["early_stopping_rounds"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{kind} fit.early_stopping_rounds는 1 이상의 정수여야 한다: {value!r}"
        )
    return value


def _fixed_iteration_count(params: dict, raw_field: str, kind: str) -> int:
    """고정 일정이 근거로 남길 설정 반복 수를 검증한다."""
    value = params.get(raw_field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"조기 종료를 쓰지 않는 {kind}는 model.params.{raw_field}에 "
            f"1 이상의 정수를 명시해야 한다: {value!r}"
        )
    return value


def _declare_training_length(
    impl: object,
    kind: str,
    raw_path: str,
    *,
    inner_members: bool,
) -> TrainingLengthDeclaration:
    """구현이 남긴 원시 선택값을 그 계열의 계약으로 묶는다. (#372)

    내부 구성원이 있는 계열은 구성원 순서를 그대로 좌표로 쓰고 하나도 빠뜨리지 않는다.
    없는 계열은 선택값이 정확히 하나여야 하며, 그렇지 않으면 여기서 터진다.
    """
    if impl is None:
        raise TrainingLengthError(
            f"{kind} 관측 학습 길이는 fold 학습을 마친 뒤에만 읽을 수 있다."
        )
    contract = TRAINING_LENGTH_CONTRACTS[kind]
    selections = impl.raw_training_length_selections()
    if inner_members:
        return contract.declare(
            RawTrainingLengthSelection(
                raw_path=raw_path.format(index=index),
                raw_value=raw_value,
                inner_member=index,
            )
            for index, raw_value in enumerate(selections)
        )
    (raw_value,) = selections
    return contract.declare(
        [RawTrainingLengthSelection(raw_path=raw_path, raw_value=raw_value)]
    )


def _validate_initial_score(
    initial_score: pd.Series | np.ndarray,
    expected_rows: int,
    label: str,
) -> np.ndarray:
    """초기 로짓을 유한한 float64 1차원 배열로 고정한다."""
    values = np.asarray(initial_score)
    if values.shape != (expected_rows,):
        raise ValueError(
            f"{label} 초기 점수 길이가 다르다: {values.shape} != {(expected_rows,)}"
        )
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.bool_
    ):
        raise ValueError(f"{label} 초기 점수는 숫자형이어야 한다: {values.dtype}")
    values = values.astype("float64", copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} 초기 점수는 모두 유한해야 한다.")
    return values


def _validate_initial_score_pair(
    initial_score_tr: pd.Series | None,
    initial_score_va: pd.Series | None,
    training_rows: int,
    validation_rows: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """학습과 검증 초기 로짓의 대칭 계약을 검증한다."""
    if (initial_score_tr is None) != (initial_score_va is None):
        raise ValueError("학습과 검증 초기 점수는 함께 주거나 함께 생략해야 한다.")
    if initial_score_tr is None:
        return None, None
    assert initial_score_va is not None
    return (
        _validate_initial_score(initial_score_tr, training_rows, "학습"),
        _validate_initial_score(initial_score_va, validation_rows, "검증"),
    )


def _validate_residual_margin(
    residual: np.ndarray,
    expected_rows: int,
    kind: str,
) -> np.ndarray:
    """잔차 부스팅 원시 출력을 유한한 부동소수점 벡터로 검증한다."""
    values = np.asarray(residual)
    if values.shape != (expected_rows,):
        raise ValueError(
            f"{kind} 잔차 원시 출력 길이가 다르다: {values.shape} != {(expected_rows,)}"
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(
            f"{kind} 잔차 원시 출력은 부동소수점이어야 한다: {values.dtype}"
        )
    values = values.astype("float64", copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{kind} 잔차 원시 출력은 모두 유한해야 한다.")
    return values


def _stable_sigmoid(margin: np.ndarray) -> np.ndarray:
    """큰 절댓값에서도 overflow 없이 로짓을 확률로 바꾼다."""
    out = np.empty_like(margin, dtype="float64")
    positive = margin >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-margin[positive]))
    exp_margin = np.exp(margin[~positive])
    out[~positive] = exp_margin / (1.0 + exp_margin)
    return out


class LightGBMAdapter:
    """LightGBM 이진 분류 adapter."""

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._early_stopping_rounds = _early_stopping_rounds(fit, "lightgbm")
        self._fixed_iteration_count = (
            None
            if self._early_stopping_rounds is not None
            else _fixed_iteration_count(params, "n_estimators", "lightgbm")
        )
        self._model = None
        self._uses_initial_score = False
        self._validated_fit = False

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
        if self._early_stopping_rounds is None:
            self._model.fit(X_tr, y_tr, **kwargs)
        else:
            self._model.fit(
                X_tr,
                y_tr,
                eval_X=X_va,
                eval_y=y_va,
                callbacks=[lgb.early_stopping(self._early_stopping_rounds)],
                **kwargs,
            )
        self._validated_fit = True
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
        # 전체 자료 재학습은 조기 종료가 없다. 없는 관측을 지어내지 않는다. (#372)
        self._validated_fit = False

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        return self._predict(X, initial_score)

    def _predict(self, X: pd.DataFrame, initial_score: pd.Series | None) -> np.ndarray:
        if not self._uses_initial_score:
            if initial_score is not None:
                raise ValueError(
                    "초기 점수 없이 학습한 모델에 예측 초기 점수가 전달됐다."
                )
            return self._model.predict_proba(X)[:, 1]
        if initial_score is None:
            raise ValueError(
                "초기 점수로 학습한 모델은 예측에도 같은 출처의 초기 점수가 필요하다."
            )
        residual = np.asarray(self._model.predict(X, raw_score=True), dtype="float64")
        margin = np.asarray(initial_score, dtype="float64")
        if residual.shape != margin.shape:
            raise ValueError(
                f"예측과 초기 점수 길이가 다르다: {residual.shape} != {margin.shape}"
            )
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

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """조기 종료 선택값 또는 설정의 고정 부스팅 횟수를 선언한다. (#372, #413)

        LightGBM의 `best_iteration_`은 학습 엔진이 이미 센 실제 횟수라 그대로 쓴다.
        """
        if self._model is None or not self._validated_fit:
            raise TrainingLengthError(
                "lightgbm 관측 학습 길이는 검증 분할로 학습한 뒤에만 읽을 수 있다."
            )
        if self._early_stopping_rounds is None:
            return FIXED_COUNT_TREE_CONTRACTS["lightgbm"].declare(
                [
                    RawTrainingLengthSelection(
                        raw_path="model.params.n_estimators",
                        raw_value=self._fixed_iteration_count,
                    )
                ]
            )
        return TRAINING_LENGTH_CONTRACTS["lightgbm"].declare(
            [
                RawTrainingLengthSelection(
                    raw_path="LGBMClassifier.best_iteration_",
                    raw_value=int(self._model.best_iteration_),
                )
            ]
        )


def _resolve_lightgbm_params(params: dict, feature_names: list[str]) -> dict:
    """열 이름별 max_bin 재정의를 LightGBM의 위치 목록으로 바꾼다.

    LightGBM 자체의 ``max_bin_by_feature``는 최종 행렬 순서와 길이가 같은 정수
    목록만 받는다. 설정에서는 ``{열 이름: max_bin}`` 매핑도 허용해 피처 계획의
    열 순서가 바뀌어도 다른 열에 조용히 적용되지 않게 한다. 목록 입력은 LightGBM
    원형을 써야 하는 경우를 위해 그대로 통과시킨다.
    """
    resolved = dict(params)
    # 같은 피처 값으로 재학습한 실행의 예측과 중요도가 정확히 같아야 한다.
    # LightGBM의 다중 스레드 누적 순서는 deterministic을 켜야 고정된다.
    # 명시적 설정은 보존하되 파이프라인 기본값은 결정적으로 둔다.
    resolved.setdefault("deterministic", True)
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


def _xgboost_categorical_frame(X: pd.DataFrame) -> pd.DataFrame:
    """XGBoost native 범주 학습이 받는 형태로 범주 열을 맞춘다.

    XGBoost는 범주 색인이 문자열이나 정수여야 한다고 요구한다. 정확값 범주 복제 열
    (<col>_cat)은 범주가 부동소수라 그대로 넘기면 거부되므로, 코드 배정은 그대로 둔 채
    범주 이름만 문자열로 바꾼다. 범주 집합은 train/test 합집합으로 고정돼 있어 학습과
    예측이 같은 이름을 받는다. (#622)
    """
    converted: dict[str, pd.Series] = {}
    for column in X.columns:
        dtype = X[column].dtype
        if isinstance(dtype, pd.CategoricalDtype) and pd.api.types.is_float_dtype(
            dtype.categories
        ):
            converted[column] = X[column].cat.rename_categories(
                [repr(float(value)) for value in dtype.categories]
            )
    if not converted:
        return X
    return X.assign(**converted)


class XGBoostAdapter:
    """XGBoost 이진 분류 adapter. (#59)

    범주형은 category dtype 그대로 native 학습한다(enable_categorical).
    범주가 부동소수인 열은 범주 이름만 문자열로 바꿔 넘긴다(_xgboost_categorical_frame).
    importance는 LightGBM gain과 같은 축척인 total_gain을 쓴다.
    """

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self._params = params
        self._fit = fit
        self._seed = seed
        self._early_stopping_rounds = _early_stopping_rounds(fit, "xgboost")
        self._fixed_iteration_count = (
            None
            if self._early_stopping_rounds is not None
            else _fixed_iteration_count(params, "n_estimators", "xgboost")
        )
        self._model = None
        self._uses_initial_score = False
        self._validated_fit = False

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

        score_tr, score_va = _validate_initial_score_pair(
            initial_score_tr, initial_score_va, len(X_tr), len(X_va)
        )
        X_tr = _xgboost_categorical_frame(X_tr)
        X_va = _xgboost_categorical_frame(X_va)
        self._uses_initial_score = score_tr is not None
        early_stopping = (
            {}
            if self._early_stopping_rounds is None
            else {"early_stopping_rounds": self._early_stopping_rounds}
        )
        self._model = xgb.XGBClassifier(
            **self._params,
            random_state=self._seed,
            enable_categorical=True,
            **early_stopping,
        )
        fit_kwargs = {"base_margin": score_tr} if self._uses_initial_score else {}
        # 조기 종료 설정만 검증 지표를 학습기에 넘겨 선택하게 한다. 고정 일정은
        # 바깥쪽 검증 자료를 선택에 쓰지 않고 설정 횟수 전부를 돈다.
        if self._early_stopping_rounds is None:
            self._model.fit(X_tr, y_tr, verbose=200, **fit_kwargs)
        else:
            if self._uses_initial_score:
                fit_kwargs["base_margin_eval_set"] = [score_va]
            self._model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=200,
                **fit_kwargs,
            )
        self._validated_fit = True
        if self._early_stopping_rounds is None:
            print(f"[xgboost] fixed count: n_estimators={self._fixed_iteration_count}")
        else:
            print(
                f"[xgboost] early stopping: best_iteration={self._model.best_iteration} "
                f"best_score={self._model.best_score:.6f}"
            )
        return self._predict(X_va, initial_score_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        import xgboost as xgb

        if training_budget is None:
            raise ValueError("xgboost 전체 자료 재학습에는 고정 반복 수가 필요하다.")
        score = (
            None
            if initial_score is None
            else _validate_initial_score(initial_score, len(X), "학습")
        )
        X = _xgboost_categorical_frame(X)
        self._uses_initial_score = score is not None
        params = dict(self._params)
        params["n_estimators"] = training_budget
        self._model = xgb.XGBClassifier(
            **params,
            random_state=self._seed,
            enable_categorical=True,
        )
        fit_kwargs = {"base_margin": score} if self._uses_initial_score else {}
        self._model.fit(X, y, verbose=200, **fit_kwargs)
        # 전체 자료 재학습은 조기 종료가 없다. 없는 관측을 지어내지 않는다. (#372)
        self._fixed_iteration_count = training_budget
        self._validated_fit = False

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        return self._predict(X, initial_score)

    def _predict(self, X: pd.DataFrame, initial_score: pd.Series | None) -> np.ndarray:
        X = _xgboost_categorical_frame(X)
        if not self._uses_initial_score:
            if initial_score is not None:
                raise ValueError(
                    "초기 점수 없이 학습한 모델에 예측 초기 점수가 전달됐다."
                )
            return self._model.predict_proba(X)[:, 1]
        if initial_score is None:
            raise ValueError(
                "초기 점수로 학습한 모델은 예측에도 같은 출처의 초기 점수가 필요하다."
            )
        margin = _validate_initial_score(initial_score, len(X), "예측")
        residual = _validate_residual_margin(
            self._model.predict(
                X,
                output_margin=True,
                base_margin=np.zeros(len(X), dtype="float64"),
            ),
            len(X),
            "xgboost",
        )
        return _stable_sigmoid(margin + residual)

    def importance(self) -> pd.DataFrame:
        booster = self._model.get_booster()
        gain = booster.get_score(importance_type="total_gain")
        # get_score는 분기에 안 쓰인 피처를 생략하므로 0으로 채워 전 피처를 돌려준다.
        names = list(self._model.feature_names_in_)
        return pd.DataFrame(
            {"feature": names, "gain": [gain.get(n, 0.0) for n in names]}
        )

    def training_diagnostics(self) -> dict[str, object]:
        """조기 종료 선택값 또는 설정이 고정한 반복 수를 기록한다."""
        if self._model is None:
            raise ValueError("xgboost 학습 관측은 fit 뒤에만 읽을 수 있다.")
        if not self._validated_fit or self._early_stopping_rounds is None:
            return {
                "training_schedule": FIXED_COUNT,
                "n_estimators": self._fixed_iteration_count,
            }
        return {
            "best_iteration": int(self._model.best_iteration),
            "best_score": float(self._model.best_score),
        }

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """조기 종료 위치 또는 설정의 고정 부스팅 횟수를 선언한다. (#372, #413)

        XGBoost의 `best_iteration`은 0부터 세는 위치라 실제 횟수보다 하나 작다.
        """
        if self._model is None or not self._validated_fit:
            raise TrainingLengthError(
                "xgboost 관측 학습 길이는 검증 분할로 학습한 뒤에만 읽을 수 있다."
            )
        if self._early_stopping_rounds is None:
            return FIXED_COUNT_TREE_CONTRACTS["xgboost"].declare(
                [
                    RawTrainingLengthSelection(
                        raw_path="model.params.n_estimators",
                        raw_value=self._fixed_iteration_count,
                    )
                ]
            )
        return TRAINING_LENGTH_CONTRACTS["xgboost"].declare(
            [
                RawTrainingLengthSelection(
                    raw_path="XGBClassifier.best_iteration",
                    raw_value=int(self._model.best_iteration),
                )
            ]
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
        self._early_stopping_rounds = _early_stopping_rounds(fit, "catboost")
        self._fixed_iteration_count = (
            None
            if self._early_stopping_rounds is not None
            else _fixed_iteration_count(params, "iterations", "catboost")
        )
        self._model = None
        self._uses_initial_score = False
        self._validated_fit = False

    @classmethod
    def _prepare(cls, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out = X.copy()
        cat_cols = [
            c for c in out.columns if isinstance(out[c].dtype, pd.CategoricalDtype)
        ]
        for c in cat_cols:
            out[c] = (
                out[c]
                .cat.add_categories([cls._MISSING])
                .fillna(cls._MISSING)
                .astype(str)
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
        from catboost import CatBoostClassifier, Pool

        score_tr, score_va = _validate_initial_score_pair(
            initial_score_tr, initial_score_va, len(X_tr), len(X_va)
        )
        self._uses_initial_score = score_tr is not None
        X_tr, cat_cols = self._prepare(X_tr)
        X_va, _ = self._prepare(X_va)
        self._model = CatBoostClassifier(
            **self._params,
            random_seed=self._seed,
            cat_features=cat_cols,
            allow_writing_files=False,
        )
        training_data = (
            Pool(X_tr, y_tr, cat_features=cat_cols, baseline=score_tr)
            if self._uses_initial_score
            else X_tr
        )
        if self._early_stopping_rounds is None:
            if self._uses_initial_score:
                self._model.fit(training_data, verbose=200)
            else:
                self._model.fit(training_data, y_tr, verbose=200)
        else:
            validation_data = (
                Pool(X_va, y_va, cat_features=cat_cols, baseline=score_va)
                if self._uses_initial_score
                else (X_va, y_va)
            )
            fit_kwargs = {
                "eval_set": validation_data,
                "early_stopping_rounds": self._early_stopping_rounds,
                "use_best_model": True,
                "verbose": 200,
            }
            if self._uses_initial_score:
                self._model.fit(training_data, **fit_kwargs)
            else:
                self._model.fit(training_data, y_tr, **fit_kwargs)
        self._validated_fit = True
        if self._early_stopping_rounds is None:
            print(f"[catboost] fixed count: iterations={self._fixed_iteration_count}")
        else:
            best_score = self._model.get_best_score().get("validation", {})
            print(
                f"[catboost] early stopping: best_iteration={self._model.get_best_iteration()} "
                f"best_score={best_score}"
            )
        return self._predict(X_va, initial_score_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from catboost import CatBoostClassifier, Pool

        if training_budget is None:
            raise ValueError("catboost 전체 자료 재학습에는 고정 반복 수가 필요하다.")
        score = (
            None
            if initial_score is None
            else _validate_initial_score(initial_score, len(X), "학습")
        )
        self._uses_initial_score = score is not None
        X, cat_cols = self._prepare(X)
        params = dict(self._params)
        params["iterations"] = training_budget
        self._model = CatBoostClassifier(
            **params,
            random_seed=self._seed,
            cat_features=cat_cols,
            allow_writing_files=False,
        )
        if self._uses_initial_score:
            self._model.fit(
                Pool(X, y, cat_features=cat_cols, baseline=score), verbose=200
            )
        else:
            self._model.fit(X, y, verbose=200)
        # 전체 자료 재학습은 조기 종료가 없다. 없는 관측을 지어내지 않는다. (#372)
        self._validated_fit = False

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        return self._predict(X, initial_score)

    def _predict(self, X: pd.DataFrame, initial_score: pd.Series | None) -> np.ndarray:
        if not self._uses_initial_score:
            if initial_score is not None:
                raise ValueError(
                    "초기 점수 없이 학습한 모델에 예측 초기 점수가 전달됐다."
                )
            X, _ = self._prepare(X)
            return self._model.predict_proba(X)[:, 1]
        if initial_score is None:
            raise ValueError(
                "초기 점수로 학습한 모델은 예측에도 같은 출처의 초기 점수가 필요하다."
            )
        margin = _validate_initial_score(initial_score, len(X), "예측")
        X, _ = self._prepare(X)
        residual = _validate_residual_margin(
            self._model.predict(X, prediction_type="RawFormulaVal"),
            len(X),
            "catboost",
        )
        return _stable_sigmoid(margin + residual)

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self._model.feature_names_,
                "gain": self._model.get_feature_importance(
                    type="PredictionValuesChange"
                ),
            }
        )

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """조기 종료 위치 또는 설정의 고정 부스팅 횟수를 선언한다. (#372, #413)

        CatBoost의 `get_best_iteration()`은 0부터 세는 위치라 실제 횟수보다 하나 작다.
        """
        if self._model is None or not self._validated_fit:
            raise TrainingLengthError(
                "catboost 관측 학습 길이는 검증 분할로 학습한 뒤에만 읽을 수 있다."
            )
        if self._early_stopping_rounds is None:
            return FIXED_COUNT_TREE_CONTRACTS["catboost"].declare(
                [
                    RawTrainingLengthSelection(
                        raw_path="model.params.iterations",
                        raw_value=self._fixed_iteration_count,
                    )
                ]
            )
        return TRAINING_LENGTH_CONTRACTS["catboost"].declare(
            [
                RawTrainingLengthSelection(
                    raw_path="CatBoostClassifier.get_best_iteration()",
                    raw_value=int(self._model.get_best_iteration()),
                )
            ]
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

        _reject_initial_score(
            "hist_gradient_boosting", initial_score_tr, initial_score_va
        )
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
            raise ValueError(
                f"penalty는 l2·l1·elasticnet 중 하나다(받은 값: {self._penalty!r})"
            )
        if (self._penalty == "elasticnet") != (self._l1_ratio is not None):
            raise ValueError("l1_ratio는 elasticnet에서만, 그리고 반드시 지정한다.")
        if self._l1_ratio is not None:
            self._l1_ratio = float(self._l1_ratio)
        allowed_solvers = {
            "l2": {"lbfgs"},  # exp058 재현성. 다른 L2 solver는 필요할 때 연다.
            "l1": {
                "saga",
                "liblinear",
            },  # liblinear 좌표 하강이 saga보다 훨씬 빠르다(#200).
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
                raise ValueError(
                    f"cross_pairs 항목은 서로 다른 두 컬럼이어야 한다: {pair}"
                )
            self._cross_pairs.append((str(pair[0]), str(pair[1])))
        if self._cross_min_count < 1:
            raise ValueError(
                f"cross_min_count는 1 이상이어야 한다(받은 값: {self._cross_min_count})"
            )
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
            raise ValueError(
                "logistic_onehot은 고정 반복 수 대신 수렴 조건으로 학습한다."
            )
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

    def training_iterations(self) -> int:
        """마지막 적합이 쓴 solver 반복 수. 초기 점수 계보 기록이 읽는다. (#505)"""
        if self._model is None:
            raise ValueError("적합 전에는 반복 수가 없다.")
        return int(self._model.n_iter_[0])

    def feature_count(self) -> int:
        """마지막 적합의 인코딩 열 수."""
        if self._train_matrix is None:
            raise ValueError("적합 전에는 인코딩 열 수가 없다.")
        return int(self._train_matrix.shape[1])

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
            contribution = (
                self._train_matrix[:, offset : offset + width]
                @ coef[offset : offset + width]
            )
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
            raise RuntimeError(
                "전처리 기준 집합은 Lookup-Transformer 학습 전에 정해야 한다."
            )
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
            raise ValueError(
                "lookup_transformer 전체 자료 재학습에는 고정 epoch 수가 필요하다."
            )
        self._impl = self._new_impl()
        self._impl.fit_full(X, y, training_budget)

    def fit_paired_training_lengths(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_lengths: tuple[int, ...],
        initial_score: pd.Series | None = None,
    ) -> None:
        _reject_initial_score("lookup_transformer", initial_score, None)
        self._impl = self._new_impl()
        self._impl.fit_full_member_training_points(
            X,
            y,
            training_lengths,
            int(self._params.get("epochs", 32)),
        )

    def fit_predict_training_states(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        X_test: pd.DataFrame,
        state: TrainingStateConfig,
    ) -> FoldTrainingStateTrajectory:
        """한 Lookup 궤적에서 미리 고정한 EMA 시점들을 물질화한다."""
        if self._fit:
            raise ValueError(
                f"lookup_transformer가 모르는 fit 설정: {sorted(self._fit)}"
            )
        self._impl = self._new_impl()
        self._impl.fit_training_trajectory(
            X_tr,
            y_tr,
            X_va,
            y_va,
            completed_epochs=state.candidates,
            trajectory_end_epochs=state.trajectory_end_epochs,
            schedule_horizon_epochs=state.schedule_horizon_epochs,
        )
        points = []
        for completed_epochs in state.candidates:
            validation_prediction = np.asarray(
                self._impl.select_training_point(completed_epochs), dtype="float64"
            )
            test_prediction = np.asarray(self._impl.predict(X_test), dtype="float64")
            importance = self._impl.importance().copy()
            diagnostics = json.loads(
                json.dumps(self._impl.training_diagnostics(), allow_nan=False)
            )
            declaration = self.training_length_evidence()
            points.append(
                FoldTrainingStatePoint(
                    completed_epochs=completed_epochs,
                    validation_prediction=validation_prediction,
                    test_prediction=test_prediction,
                    importance=importance,
                    training_diagnostics=diagnostics,
                    training_length_declaration=declaration,
                )
            )
        return FoldTrainingStateTrajectory(
            schedule_horizon_epochs=state.schedule_horizon_epochs,
            trajectory_end_epochs=state.trajectory_end_epochs,
            points=tuple(points),
        )

    def fit_full_training_state(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        state: TrainingStateConfig,
        initial_score: pd.Series | None = None,
    ) -> dict[str, object]:
        _reject_initial_score("lookup_transformer", initial_score, None)
        self._impl = self._new_impl()
        self._impl.fit_full_training_point(
            X,
            y,
            trajectory_end_epochs=state.selected,
            schedule_horizon_epochs=state.schedule_horizon_epochs,
        )
        details = self._impl.training_diagnostics()
        members = details.get("fold_initialization_members")
        if not isinstance(members, list) or not members:
            raise ValueError(
                "Lookup-Transformer 전체 자료 학습 진단에 초기화 구성원이 없다."
            )
        if any(
            not isinstance(member, dict)
            or member.get("end_epoch") != state.selected - 1
            for member in members
        ):
            raise ValueError(
                "Lookup-Transformer 전체 자료 학습이 선택 시점에서 끝나지 않았다."
            )
        if any(
            member.get("schedule_horizon_epochs") != state.schedule_horizon_epochs
            for member in members
        ):
            raise ValueError(
                "Lookup-Transformer 전체 자료 학습률 일정 지평이 요청과 다르다."
            )
        return {
            "completed_epochs": state.selected,
            "schedule_horizon_epochs": state.schedule_horizon_epochs,
            "state_kind": state.state_kind,
            "model_diagnostics": details,
        }

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

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """초기화 구성원마다 검증이 고른 0부터 세는 epoch 위치를 선언한다. (#372)"""
        return _declare_training_length(
            self._impl,
            "lookup_transformer",
            "training_diagnostics.fold_initialization_members[{index}].best_epoch",
            inner_members=True,
        )


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

        _reject_initial_score("contextualized_spline_transformer", initial_score, None)
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

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """검증이 고른 1부터 세는 epoch 횟수를 선언한다. (#372)"""
        return _declare_training_length(
            self._impl,
            "contextualized_spline_transformer",
            "training_diagnostics.best_epoch",
            inner_members=False,
        )


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
                f"mixing은 ['attention', 'token_mlp'] 중 하나여야 한다: {mixing!r}"
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
                f"scalar_token_transformer가 모르는 fit 설정: {sorted(self._fit)}"
            )
        self._impl = scalar_token_transformer.ScalarTokenTransformerFold(
            self._params, self._seed
        )
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        from . import scalar_token_transformer

        _reject_initial_score("scalar_token_transformer", initial_score, None)
        if self._fit:
            raise ValueError(
                f"scalar_token_transformer가 모르는 fit 설정: {sorted(self._fit)}"
            )
        if training_budget is None:
            raise ValueError(
                "scalar_token_transformer 전체 자료 재학습에는 "
                "고정 epoch 수가 필요하다."
            )
        self._impl = scalar_token_transformer.ScalarTokenTransformerFold(
            self._params, self._seed
        )
        self._impl.fit_full(X, y, training_budget)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("scalar_token_transformer", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def entry_diagnostics(self) -> AdapterDiagnostics:
        return self._impl.entry_diagnostics()

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """검증이 고른 1부터 세는 epoch 횟수를 선언한다. (#372)"""
        return _declare_training_length(
            self._impl,
            "scalar_token_transformer",
            "entry_diagnostics.best_epoch",
            inner_members=False,
        )


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

    def fit_paired_training_lengths(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_lengths: tuple[int, ...],
        initial_score: pd.Series | None = None,
    ) -> None:
        from . import tabm

        _reject_initial_score("tabm", initial_score, None)
        self._impl = tabm.TabMFold(self._params, self._seed)
        self._impl.fit_full_member_epochs(X, y, training_lengths)

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        _reject_initial_score("tabm", initial_score, None)
        return self._impl.predict(X)

    def importance(self) -> pd.DataFrame:
        return self._impl.importance()

    def training_diagnostics(self) -> dict[str, object]:
        return self._impl.training_diagnostics()

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """내부 구성원마다 저장소가 이미 센 선택 epoch 횟수를 선언한다. (#372)"""
        return _declare_training_length(
            self._impl,
            "tabm",
            "training_diagnostics.members[{index}].selected_epoch_count",
            inner_members=True,
        )


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
        self._dataset_reference: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def set_dataset_reference(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        """분위-정규 좌표에 쓸 목표값 비참조 기준 집합을 보관한다."""
        if self._impl is not None:
            raise RuntimeError("전처리 기준 집합은 RealMLP 학습 전에 정해야 한다.")
        if TARGET in X_train.columns or TARGET in X_test.columns:
            raise ValueError("RealMLP 전처리 기준 집합은 목표값을 포함할 수 없다.")
        self._dataset_reference = (X_train, X_test)

    def _new_impl(self):
        from . import realmlp

        impl = realmlp.RealMLPFold(self._params, self._seed)
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
        _reject_initial_score("realmlp", initial_score_tr, initial_score_va)
        if self._fit:
            raise ValueError(f"realmlp가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = self._new_impl()
        return self._impl.fit(X_tr, y_tr, X_va, y_va)

    def fit_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training_budget: int | None,
        initial_score: pd.Series | None = None,
    ) -> None:
        _reject_initial_score("realmlp", initial_score, None)
        if self._fit:
            raise ValueError(f"realmlp가 모르는 fit 설정: {sorted(self._fit)}")
        if training_budget is None:
            raise ValueError("realmlp 전체 자료 재학습에는 고정 epoch 수가 필요하다.")
        self._impl = self._new_impl()
        self._impl.fit_full(X, y, training_budget)

    def fit_predict_training_states(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        X_test: pd.DataFrame,
        state: TrainingStateConfig,
    ) -> FoldTrainingStateTrajectory:
        """한 RealMLP 궤적에서 미리 고정한 EMA 시점들을 물질화한다."""
        if self._fit:
            raise ValueError(f"realmlp가 모르는 fit 설정: {sorted(self._fit)}")
        self._impl = self._new_impl()
        self._impl.fit_training_trajectory(
            X_tr,
            y_tr,
            X_va,
            y_va,
            completed_epochs=state.candidates,
        )
        points = []
        for completed_epochs in state.candidates:
            validation_prediction = np.asarray(
                self._impl.select_training_point(completed_epochs), dtype="float64"
            )
            test_prediction = np.asarray(self._impl.predict(X_test), dtype="float64")
            importance = self._impl.importance().copy()
            diagnostics = json.loads(
                json.dumps(self._impl.training_diagnostics(), allow_nan=False)
            )
            declaration = self.training_length_evidence()
            points.append(
                FoldTrainingStatePoint(
                    completed_epochs=completed_epochs,
                    validation_prediction=validation_prediction,
                    test_prediction=test_prediction,
                    importance=importance,
                    training_diagnostics=diagnostics,
                    training_length_declaration=declaration,
                )
            )
        return FoldTrainingStateTrajectory(
            schedule_horizon_epochs=state.schedule_horizon_epochs,
            trajectory_end_epochs=state.trajectory_end_epochs,
            points=tuple(points),
        )

    def fit_full_training_state(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        state: TrainingStateConfig,
        initial_score: pd.Series | None = None,
    ) -> dict[str, object]:
        _reject_initial_score("realmlp", initial_score, None)
        self._impl = self._new_impl()
        self._impl.fit_full(X, y, state.selected)
        details = self._impl.training_diagnostics()
        if details.get("full_training_budget") != state.selected:
            raise ValueError("RealMLP 전체 자료 학습이 선택 시점에서 끝나지 않았다.")
        if details.get("schedule_horizon_epochs") != state.schedule_horizon_epochs:
            raise ValueError("RealMLP 전체 자료 학습률 일정 지평이 요청과 다르다.")
        return {
            "completed_epochs": state.selected,
            "schedule_horizon_epochs": state.schedule_horizon_epochs,
            "state_kind": state.state_kind,
            "model_diagnostics": details,
        }

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

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """설정이 고정한 실제 epoch 횟수를 선언한다. 검증이 고른 위치가 아니다. (#372)"""
        return _declare_training_length(
            self._impl,
            "realmlp",
            "training_diagnostics.fixed_epochs",
            inner_members=False,
        )


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

    def training_length_evidence(self) -> TrainingLengthDeclaration:
        """검증이 고른 1부터 세는 epoch 횟수를 선언한다. (#372)"""
        return _declare_training_length(
            self._impl,
            "tab_cnn",
            "training_diagnostics.best_epoch",
            inner_members=False,
        )


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
