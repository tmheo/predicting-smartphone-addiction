"""nested OOF 평가기와 결합 전략 계약. (ADR 0001 계열 3, #104)

사용법:
    uv run python -m pipeline.ensemble                    # 기본 전략 평가·비교·판정
    uv run python -m pipeline.ensemble --only rank_mean   # 개발·디버깅용 부분 실행
    uv run python -m pipeline.ensemble --only rank_mean --only ridge_logit
    uv run python -m pipeline.ensemble --only rank_logit_logistic --submission <path>
    uv run python -m pipeline.ensemble --record-issue 202 --baseline-run <run_id>

이 module은 측정이다: outer fold 루프, float64 강제 캐스팅, 결합 전략 Protocol과
adapter, COMBINER_REGISTRY, 선택 빈도 집계, CLI. 계열 3 판정(judge_ensemble)은
judgment module 소관이고, 순환을 막기 위해 최상단 import는 judgment → ensemble
단방향만 둔다. 이 module은 리포트 함수 안에서만 judgment를 지역 import한다
(run.py가 observe를 main 안에서 import하는 전례).

nested OOF 정의(ADR 0001 계열 3): outer fold k마다 나머지 4개 fold의 OOF만으로
구성원 선택과 가중치를 학습하고, 그 결과로 fold k 예측을 만들어 5개 fold를 합쳐
채점한다. 앙상블 구성원은 후보 풀(pool.yaml)의 시드 평균본 전체다.

결합 전략 계약:
- 순위 변환·logit 변환 같은 전처리는 각 전략이 소유한다. 평가기는 float64 원시
  예측만 건넨다.
- 결측 개수처럼 목표값을 쓰지 않는 행 맥락도 필요한 전략이 소유한다. 평가기는
  행 맥락을 해석하거나 구간을 고르지 않는다.
- Fitted.predict는 outer fold 행만 받는다. 순위 변환의 모집단은 채점 블록 자신이다.
  제출 시점의 결합이 test 예측만으로 순위를 매기므로, 평가도 같은 조건이어야
  nested 점수가 제출 동작을 대변한다.
- Fitted.summary()(구성원 이름 → 가중치/선택 여부)를 outer fold 5개에서 모아
  구성원별 선택 빈도·평균 가중치 표를 만든다. #62의 "선택 빈도와 fold별 승리 기록"
  요구를 adapter 추가만으로 감당하기 위한 규약이다.

구성원 예측 행렬의 컬럼 키는 config 이름이다(풀 장부가 유일성 보장). run_id 대응은
리포트 머리에 한 번 출력한다.

평가 자체는 MLflow run을 만들지 않고 stdout으로 남긴다(실험 하나 = run 하나 규약).
각 실행은 전략별 nested OOF AUC와 경과 시간, 기본 평가에서 제외한 정밀 결합 전략을
run-logs/ensemble-evaluation.json에 함께 남긴다.
CLI는 전략마다 가중 OOF AUC(test 결측 패턴 구성비 재채점, judgment 소관)를 함께 재서
같은 JSON에 남기고, 전략별 nested OOF 예측을 run-logs/strategy-oof.parquet에
저장한다. 점수 하나만 남기면 사후에 다른 눈금으로 다시 비교할 수 없기 때문이다(#383).
판정은 nested OOF만 쓴다.
--record-issue가 주어지면 최선 전략의 nested 결과 하나를 파생 앙상블 실행
(source.kind=derived_ensemble)으로 MLflow에 기록한다(#179·#183 선례의 상설화, #202).
채택 전략은 전체 OOF로 다시 학습해 구성원 시험 예측에 적용한 제출 파일을 명시적
경로에 만들 수 있다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import ndtri
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import ID, TARGET, TRAIN_PATH, file_sha256, labels
from .ledger import CHAMPION_PATH, Champion, Pool
from .runs import TRACKING_URI, MlflowRunStore, RunStore, RunStoreError

if TYPE_CHECKING:  # 최상단 import는 judgment → ensemble 단방향만 둔다(module docstring).
    from .judgment import MissingnessReweighting, WeightedOof


def rank_mean(preds: pd.DataFrame) -> np.ndarray:
    """균등 순위 평균의 수식 단일 소스: 블록 내 백분위 순위의 구성원 평균.

    순위의 모집단은 전달된 블록 자신이다. judgment 계열 2 기여 참고값의
    rank_ensemble_auc는 "블록=전체 OOF"인 특수 사례로 같은 수식을 쓴다.
    """
    return preds.rank(pct=True).to_numpy().mean(axis=1)


class FittedCombiner(Protocol):
    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        """outer fold 블록의 구성원 예측 행렬을 결합 예측 하나로 만든다."""
        ...

    def summary(self) -> dict[str, float]:
        """구성원 이름 → 가중치(무학습 전략은 선택 여부에 해당하는 값)."""
        ...


class Combiner(Protocol):
    name: str

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedCombiner: ...


@dataclass(frozen=True)
class FittedRankMean:
    members: list[str]

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        return rank_mean(outer_preds[self.members])

    def summary(self) -> dict[str, float]:
        return {member: 1.0 / len(self.members) for member in self.members}


class RankMeanCombiner:
    """균등 순위 평균: 무학습 전략의 대표. fit은 항등, summary는 전 구성원 1/N."""

    name = "rank_mean"

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedRankMean:
        return FittedRankMean(list(inner_preds.columns))


@dataclass(frozen=True)
class FittedPerformanceWeightedRankMean:
    weights: pd.Series

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        ranks = (
            outer_preds[self.weights.index].rank(pct=True).to_numpy(dtype=np.float64)
        )
        return ranks @ self.weights.to_numpy(dtype=np.float64)

    def summary(self) -> dict[str, float]:
        return {member: float(weight) for member, weight in self.weights.items()}


class PerformanceWeightedRankMeanCombiner:
    """학습 부분의 단독 OOF AUC만 쓰는 제한된 가중 순위 평균.

    무작위 예측보다 나은 AUC 폭만 비음수 가중치로 쓰고 합을 1로 정규화한다.
    별도 최적화나 구성원 선택이 없어 outer 평가 결과에 가중치를 맞출 자유도가 없다.
    모든 구성원이 0.5 이하이면 균등 가중치로 되돌린다.
    """

    name = "performance_weighted_rank_mean"

    def fit(
        self, inner_preds: pd.DataFrame, y: pd.Series
    ) -> FittedPerformanceWeightedRankMean:
        advantages = pd.Series(
            {
                member: max(
                    float(roc_auc_score(y.to_numpy(), inner_preds[member].to_numpy()))
                    - 0.5,
                    0.0,
                )
                for member in inner_preds.columns
            },
            dtype=np.float64,
        )
        if float(advantages.sum()) == 0.0:
            weights = pd.Series(
                1.0 / len(advantages), index=advantages.index, dtype=np.float64
            )
        else:
            weights = advantages / float(advantages.sum())
        return FittedPerformanceWeightedRankMean(weights)


def _greedy_rank_weights(
    preds: pd.DataFrame,
    y: pd.Series,
    *,
    max_members: int,
    min_improvement: float,
) -> pd.Series:
    """학습 블록에서 AUC가 오르는 동안 구성원을 중복 없이 하나씩 추가한다."""
    ranks = preds.rank(pct=True).to_numpy(dtype=np.float64)
    labels = y.to_numpy()
    selected: list[int] = []
    remaining = list(range(ranks.shape[1]))
    running_sum = np.zeros(len(preds), dtype=np.float64)
    best_auc = -np.inf

    while remaining and len(selected) < max_members:
        scores = [
            float(
                roc_auc_score(
                    labels,
                    (running_sum + ranks[:, candidate]) / (len(selected) + 1),
                )
            )
            for candidate in remaining
        ]
        winner_position = int(np.argmax(scores))
        winner = remaining[winner_position]
        winner_auc = scores[winner_position]
        if selected and winner_auc <= best_auc + min_improvement:
            break
        selected.append(winner)
        remaining.remove(winner)
        running_sum += ranks[:, winner]
        best_auc = winner_auc

    weights = pd.Series(0.0, index=preds.columns, dtype=np.float64)
    weights.iloc[selected] = 1.0 / len(selected)
    return weights


class GreedyRankMeanCombiner:
    """inner OOF의 균등 순위 평균 AUC를 탐욕적으로 높이는 부분집합 선택."""

    name = "greedy_rank_mean"

    def __init__(
        self,
        *,
        max_members: int | None = None,
        min_improvement: float = 1e-12,
        name: str | None = None,
    ) -> None:
        self.max_members = max_members
        self.min_improvement = min_improvement
        self.name = name or type(self).name

    def fit(
        self, inner_preds: pd.DataFrame, y: pd.Series
    ) -> FittedPerformanceWeightedRankMean:
        weights = _greedy_rank_weights(
            inner_preds,
            y,
            max_members=self.max_members or len(inner_preds.columns),
            min_improvement=self.min_improvement,
        )
        return FittedPerformanceWeightedRankMean(weights)


class BaggedGreedyRankMeanCombiner:
    """층화 부분 표본의 탐욕 선택 빈도를 평균한 순위 결합."""

    name = "bagged_greedy_rank_mean"

    def __init__(
        self,
        *,
        bags: int = 50,
        sample_fraction: float = 0.5,
        seed: int = 42,
        workers: int = 4,
        max_members: int | None = None,
        min_improvement: float = 1e-12,
        name: str | None = None,
    ) -> None:
        if bags < 1:
            raise ValueError("bags는 1 이상이어야 한다.")
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction은 0보다 크고 1 이하여야 한다.")
        if workers < 1:
            raise ValueError("workers는 1 이상이어야 한다.")
        self.bags = bags
        self.sample_fraction = sample_fraction
        self.seed = seed
        self.workers = workers
        self.max_members = max_members
        self.min_improvement = min_improvement
        self.name = name or type(self).name

    def fit(
        self, inner_preds: pd.DataFrame, y: pd.Series
    ) -> FittedPerformanceWeightedRankMean:
        rng = np.random.default_rng(self.seed)
        labels = y.to_numpy()
        class_positions = [
            np.flatnonzero(labels == value) for value in np.unique(labels)
        ]
        samples = [
            np.sort(
                np.concatenate(
                    [
                        rng.choice(
                            positions,
                            size=max(1, round(len(positions) * self.sample_fraction)),
                            replace=False,
                        )
                        for positions in class_positions
                    ]
                )
            )
            for _ in range(self.bags)
        ]

        def select(sampled: np.ndarray) -> pd.Series:
            return _greedy_rank_weights(
                inner_preds.iloc[sampled],
                y.iloc[sampled],
                max_members=self.max_members or len(inner_preds.columns),
                min_improvement=self.min_improvement,
            )

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            bag_weights = list(executor.map(select, samples))
        aggregate = sum(
            bag_weights,
            start=pd.Series(0.0, index=inner_preds.columns, dtype=np.float64),
        )
        weights = aggregate / float(aggregate.sum())
        return FittedPerformanceWeightedRankMean(weights)


def _optuna_rank_weights(
    preds: pd.DataFrame,
    y: pd.Series,
    *,
    trials: int,
    seed: int,
) -> pd.Series:
    """Optuna TPE로 고른 이진 부분집합을 균등 가중치로 반환한다."""
    import optuna

    if trials < 2:
        raise ValueError("trials는 2 이상이어야 한다.")
    ranks = preds.rank(pct=True).to_numpy(dtype=np.float64)
    labels = y.to_numpy()
    members = list(preds.columns)

    def objective(trial: optuna.Trial) -> float:
        selected = np.array(
            [
                trial.suggest_categorical(f"member_{i}", [False, True])
                for i in range(len(members))
            ]
        )
        if not selected.any():
            return 0.5
        return float(roc_auc_score(labels, ranks[:, selected].mean(axis=1)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(32, max(1, trials // 4)),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    greedy = _greedy_rank_weights(
        preds,
        y,
        max_members=len(members),
        min_improvement=1e-12,
    )
    study.enqueue_trial(
        {f"member_{i}": bool(greedy.iloc[i] > 0.0) for i in range(len(members))}
    )
    study.enqueue_trial({f"member_{i}": True for i in range(len(members))})
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    selected = np.array(
        [bool(study.best_params[f"member_{i}"]) for i in range(len(members))]
    )
    weights = pd.Series(0.0, index=members, dtype=np.float64)
    weights.iloc[np.flatnonzero(selected)] = 1.0 / int(selected.sum())
    return weights


class OptunaSubsetRankMeanCombiner:
    """inner OOF에서 Optuna가 고른 부분집합의 균등 순위 평균."""

    name = "optuna_subset_rank_mean"

    def __init__(
        self, *, trials: int = 128, seed: int = 42, name: str | None = None
    ) -> None:
        self.trials = trials
        self.seed = seed
        self.name = name or type(self).name

    def fit(
        self, inner_preds: pd.DataFrame, y: pd.Series
    ) -> FittedPerformanceWeightedRankMean:
        return FittedPerformanceWeightedRankMean(
            _optuna_rank_weights(inner_preds, y, trials=self.trials, seed=self.seed)
        )


@dataclass(frozen=True)
class FittedSubsetRidgeLogit:
    fitted: FittedRidgeLogit
    all_members: list[str]

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        return self.fitted.predict(outer_preds)

    def summary(self) -> dict[str, float]:
        selected = self.fitted.summary()
        return {member: selected.get(member, 0.0) for member in self.all_members}


class OptunaSubsetRidgeLogitCombiner:
    """Optuna 부분집합을 고정한 뒤 inner OOF에서 Ridge 가중치를 학습한다."""

    name = "optuna_subset_ridge_logit"

    def __init__(
        self,
        *,
        trials: int = 128,
        seed: int = 42,
        alpha: float = 100.0,
        name: str | None = None,
    ) -> None:
        self.trials = trials
        self.seed = seed
        self.alpha = alpha
        self.name = name or type(self).name

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedSubsetRidgeLogit:
        weights = _optuna_rank_weights(
            inner_preds, y, trials=self.trials, seed=self.seed
        )
        selected = list(weights[weights > 0.0].index)
        fitted = RidgeLogitCombiner(alpha=self.alpha).fit(inner_preds[selected], y)
        return FittedSubsetRidgeLogit(fitted, list(inner_preds.columns))


def _logit(preds: pd.DataFrame, eps: float) -> np.ndarray:
    clipped = preds.to_numpy().clip(eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class FittedRidgeLogit:
    model: Ridge
    members: list[str]
    eps: float

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_logit(outer_preds[self.members], self.eps))

    def summary(self) -> dict[str, float]:
        return {
            member: float(coef)
            for member, coef in zip(self.members, self.model.coef_, strict=True)
        }


class RidgeLogitCombiner:
    """Ridge 결합: 학습 상태(계수)를 가진 전략의 대표. (#64 정의)

    구성원 예측을 logit으로 변환해 학습·적용한다. alpha 값 탐색은 #64 소관이라
    여기서는 생성자 인자에 기본값 1.0으로 고정한다.
    """

    name = "ridge_logit"
    LOGIT_EPS = 1e-6  # 0/1 포화 예측의 logit 발산을 막는 클리핑. adapter 소유 상수.

    def __init__(self, alpha: float = 1.0, *, name: str | None = None) -> None:
        self.alpha = alpha
        self.name = name or type(self).name

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedRidgeLogit:
        model = Ridge(alpha=self.alpha)
        model.fit(_logit(inner_preds, self.LOGIT_EPS), y.to_numpy())
        return FittedRidgeLogit(model, list(inner_preds.columns), self.LOGIT_EPS)


class CombinerConvergenceError(RuntimeError):
    """선형 결합기가 반복 한도 안에 수렴하지 못해 비교에서 제외되어야 한다."""


Representation = Literal["rank", "logit", "rank_logit", "rank_gauss"]


@dataclass
class EmpiricalCDFTransformer:
    """outer 학습 행 전체의 열별 경험적 누적분포를 고정한다."""

    output_distribution: Literal["uniform", "normal"]
    sorted_columns: tuple[np.ndarray, ...] | None = None

    def fit(self, values: np.ndarray) -> EmpiricalCDFTransformer:
        self.sorted_columns = tuple(
            np.sort(values[:, column]) for column in range(values.shape[1])
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.sorted_columns is None:
            raise RuntimeError("경험적 누적분포 변환기를 먼저 학습해야 한다.")
        transformed = np.empty(values.shape, dtype=np.float64)
        for column, reference in enumerate(self.sorted_columns):
            left = np.searchsorted(reference, values[:, column], side="left")
            right = np.searchsorted(reference, values[:, column], side="right")
            count = len(reference)
            # 동률은 평균 순위로 두고, 학습 범위 밖 값도 유한한 정규 분위수가 되게
            # 양끝을 반 관측치만큼 안쪽으로 자른다.
            midpoint = (left + right) / (2.0 * count)
            transformed[:, column] = np.clip(midpoint, 0.5 / count, 1.0 - 0.5 / count)
        if self.output_distribution == "normal":
            return ndtri(transformed)
        return transformed

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


def _linear_features(
    preds: pd.DataFrame,
    representation: Representation,
    *,
    quantiles: EmpiricalCDFTransformer | None,
    fit: bool,
    eps: float,
) -> tuple[np.ndarray, EmpiricalCDFTransformer | None]:
    values = preds.to_numpy(dtype=np.float64)
    if representation == "logit":
        return _logit(preds, eps), None

    if quantiles is None:
        distribution: Literal["uniform", "normal"] = (
            "normal" if representation == "rank_gauss" else "uniform"
        )
        quantiles = EmpiricalCDFTransformer(distribution)
    ranked = quantiles.fit_transform(values) if fit else quantiles.transform(values)
    ranked = np.asarray(ranked, dtype=np.float64)
    if representation == "rank_logit":
        return np.column_stack((ranked, _logit(preds, eps))), quantiles
    return ranked, quantiles


@dataclass(frozen=True)
class FittedLogisticLinear:
    model: LogisticRegression
    scaler: StandardScaler
    members: list[str]
    representation: Representation
    quantiles: EmpiricalCDFTransformer | None
    eps: float

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        features, _ = _linear_features(
            outer_preds[self.members],
            self.representation,
            quantiles=self.quantiles,
            fit=False,
            eps=self.eps,
        )
        scaled = self.scaler.transform(features)
        return self.model.predict_proba(scaled)[:, 1].astype(np.float64, copy=False)

    def summary(self) -> dict[str, float]:
        coefficients = self.model.coef_[0]
        if self.representation == "rank_logit":
            member_count = len(self.members)
            coefficients = coefficients[:member_count] + coefficients[member_count:]
        return {
            member: float(coefficient)
            for member, coefficient in zip(self.members, coefficients, strict=True)
        }


class LogisticLinearCombiner:
    """outer 학습 부분에서만 표현 변환, 표준화와 로지스틱 계수를 학습한다."""

    LOGIT_EPS = 1e-6

    def __init__(
        self,
        name: str,
        representation: Representation,
        *,
        c: float = 1.0,
        max_iter: int = 1_000,
    ) -> None:
        self.name = name
        self.representation = representation
        self.c = c
        self.max_iter = max_iter

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedLogisticLinear:
        features, quantiles = _linear_features(
            inner_preds,
            self.representation,
            quantiles=None,
            fit=True,
            eps=self.LOGIT_EPS,
        )
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features).astype(np.float64, copy=False)
        model = LogisticRegression(
            C=self.c,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled, y.to_numpy())
        iterations = int(np.max(model.n_iter_))
        if iterations >= self.max_iter:
            raise CombinerConvergenceError(
                f"max(n_iter_)={iterations}, max_iter={self.max_iter}"
            )
        return FittedLogisticLinear(
            model=model,
            scaler=scaler,
            members=list(inner_preds.columns),
            representation=self.representation,
            quantiles=quantiles,
            eps=self.LOGIT_EPS,
        )


@dataclass(frozen=True)
class FittedNNLSLinear:
    weights: pd.Series
    representation: Representation
    quantiles: EmpiricalCDFTransformer | None
    eps: float

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        features, _ = _linear_features(
            outer_preds[list(self.weights.index)],
            self.representation,
            quantiles=self.quantiles,
            fit=False,
            eps=self.eps,
        )
        return features @ self.weights.to_numpy(dtype=np.float64)

    def summary(self) -> dict[str, float]:
        return {member: float(weight) for member, weight in self.weights.items()}


class NNLSCombiner:
    """비음수 최소제곱 결합: 가중치 ≥ 0 제약이 음수 가중치의 분산 증폭을 막는다. (#223)

    학습 후 가중치를 합 1로 정규화한다(AUC는 순서 평가라 스케일 무관, summary 가독성
    목적). 모든 가중치가 0이면 균등 가중치로 되돌린다.
    """

    LOGIT_EPS = 1e-6

    def __init__(self, name: str, representation: Representation) -> None:
        if representation not in ("logit", "rank"):
            raise ValueError(
                f"NNLS는 구성원당 특성 1개인 표현만 받는다: {representation}"
            )
        self.name = name
        self.representation = representation

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedNNLSLinear:
        features, quantiles = _linear_features(
            inner_preds,
            self.representation,
            quantiles=None,
            fit=True,
            eps=self.LOGIT_EPS,
        )
        raw, _ = nnls(features, y.to_numpy(dtype=np.float64))
        total = float(raw.sum())
        if total == 0.0:
            raw = np.full(len(raw), 1.0)
            total = float(raw.sum())
        weights = pd.Series(raw / total, index=inner_preds.columns, dtype=np.float64)
        return FittedNNLSLinear(
            weights, self.representation, quantiles, self.LOGIT_EPS
        )


SHRINKAGE_LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0)


@lru_cache(maxsize=1)
def outer_fold_assignment() -> pd.Series:
    """artifacts/folds.parquet의 outer fold 배정. 수축 결합의 λ 선택이 재사용한다."""
    from .judgment import FOLDS_PATH  # 순환을 막는 지역 import(module docstring).

    return pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]


def _aligned_folds(fold_of: pd.Series, index: pd.Index) -> pd.Series:
    aligned = fold_of.reindex(index)
    if aligned.isna().any():
        raise ValueError("fold 배정에 요청한 id가 없다.")
    return aligned.astype(np.int64)


def _shrunk_prediction(
    meta: FittedLogisticLinear, block: pd.DataFrame, shrinkage_lambda: float
) -> np.ndarray:
    """순위 공간의 볼록 결합: λ·학습 메타 예측 + (1-λ)·rank_mean 예측."""
    meta_ranks = (
        pd.Series(meta.predict(block), index=block.index)
        .rank(pct=True)
        .to_numpy(dtype=np.float64)
    )
    return shrinkage_lambda * meta_ranks + (1.0 - shrinkage_lambda) * rank_mean(block)


@dataclass(frozen=True)
class FittedShrunkRankLogit:
    meta: FittedLogisticLinear
    members: list[str]
    shrinkage_lambda: float

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        return _shrunk_prediction(
            self.meta, outer_preds[self.members], self.shrinkage_lambda
        )

    def summary(self) -> dict[str, float]:
        meta_summary = self.meta.summary()
        uniform = 1.0 / len(self.members)
        return {
            member: float(
                self.shrinkage_lambda * meta_summary[member]
                + (1.0 - self.shrinkage_lambda) * uniform
            )
            for member in self.members
        }


class ShrunkRankLogitCombiner:
    """학습 메타를 균등 순위 평균으로 수축하는 볼록 결합. (#223)

    학습 메타는 현행 기본 표현인 rank_logit 이중 표현 로지스틱이다. λ는 outer 학습
    fold 안의 leave-one-fold-out으로 λ별 AUC를 재어 고르므로 outer fold 밖 정보를
    쓰지 않고, nested 계약이 유지된다. 격자는 오름차순이라 동률이면 수축이 큰
    (λ가 작은) 쪽을 채택한다.
    """

    name = "shrunk_rank_logit_logistic"

    def __init__(
        self,
        *,
        fold_of: pd.Series | None = None,
        lambda_grid: tuple[float, ...] = SHRINKAGE_LAMBDA_GRID,
        name: str | None = None,
    ) -> None:
        if not lambda_grid or not all(0.0 <= value <= 1.0 for value in lambda_grid):
            raise ValueError("λ 격자는 [0, 1] 안의 값 1개 이상이어야 한다.")
        self.fold_of = fold_of
        self.lambda_grid = tuple(sorted(lambda_grid))
        self.name = name or type(self).name

    @staticmethod
    def _meta() -> LogisticLinearCombiner:
        return LogisticLinearCombiner("shrinkage_meta", "rank_logit")

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedShrunkRankLogit:
        fold_of = self.fold_of if self.fold_of is not None else outer_fold_assignment()
        folds = _aligned_folds(fold_of, inner_preds.index)
        unique_folds = sorted(folds.unique())
        if len(unique_folds) < 2:
            raise ValueError("λ 선택의 leave-one-fold-out에는 fold 2개 이상이 필요하다.")

        combined = {
            shrinkage_lambda: np.full(len(inner_preds), np.nan, dtype=np.float64)
            for shrinkage_lambda in self.lambda_grid
        }
        for fold in unique_folds:
            train = (folds != fold).to_numpy()
            validate = (folds == fold).to_numpy()
            meta = self._meta().fit(inner_preds[train], y[train])
            block = inner_preds[validate]
            for shrinkage_lambda in self.lambda_grid:
                combined[shrinkage_lambda][validate] = _shrunk_prediction(
                    meta, block, shrinkage_lambda
                )
        label_values = y.to_numpy()
        aucs = [
            float(roc_auc_score(label_values, combined[shrinkage_lambda]))
            for shrinkage_lambda in self.lambda_grid
        ]
        best_lambda = self.lambda_grid[int(np.argmax(aucs))]
        return FittedShrunkRankLogit(
            meta=self._meta().fit(inner_preds, y),
            members=list(inner_preds.columns),
            shrinkage_lambda=float(best_lambda),
        )


MISSINGNESS_TEST_PATH = Path("data/test.csv")
MISSINGNESS_BAND_LABELS = (0, 1, 2)


@lru_cache(maxsize=4)
def missingness_bands(
    train_path: Path = TRAIN_PATH,
    test_path: Path = MISSINGNESS_TEST_PATH,
) -> pd.Series:
    """원시 특성의 결측 개수를 0-1, 2-3, 4개 이상 구간으로 고정한다.

    목표값과 id는 결측 개수에서 제외한다.
    train과 test를 함께 읽는 이유는 전체 OOF 학습 뒤 같은 결합 전략으로 시험 예측을
    만들기 위해서다.
    구간은 자료에서 학습하거나 목표값으로 고르지 않는다.
    """
    frames = []
    feature_columns: list[str] | None = None
    for path in (train_path, test_path):
        frame = pd.read_csv(path)
        current = [column for column in frame.columns if column not in {ID, TARGET}]
        if feature_columns is None:
            feature_columns = current
        elif current != feature_columns:
            raise ValueError("train과 test의 결측 개수 대상 특성 열이 다르다.")
        counts = frame[current].isna().sum(axis=1)
        bands = pd.Series(
            np.select([counts <= 1, counts <= 3], [0, 1], default=2),
            index=pd.Index(frame[ID], name=ID),
            dtype=np.int8,
        )
        frames.append(bands)
    combined = pd.concat(frames)
    if combined.index.has_duplicates:
        raise ValueError("train과 test의 id가 중복되어 결측 구간을 구분할 수 없다.")
    return combined


def _aligned_bands(band_of: pd.Series, index: pd.Index) -> pd.Series:
    aligned = band_of.reindex(index)
    if aligned.isna().any():
        raise ValueError("결측 개수 구간에 요청한 id가 없다.")
    unexpected = set(aligned.astype(int).unique()) - set(MISSINGNESS_BAND_LABELS)
    if unexpected:
        raise ValueError(f"알 수 없는 결측 개수 구간: {sorted(unexpected)}")
    return aligned.astype(np.int8)


@dataclass(frozen=True)
class FittedMissingnessSegmentedLogistic:
    models: dict[int, FittedLogisticLinear]
    global_model: FittedLogisticLinear | None
    band_of: pd.Series
    summary_weights: dict[str, float]

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        bands = _aligned_bands(self.band_of, outer_preds.index)
        prediction = np.full(len(outer_preds), np.nan, dtype=np.float64)
        for band in MISSINGNESS_BAND_LABELS:
            mask = (bands == band).to_numpy()
            if not mask.any():
                continue
            fitted = self.models.get(band, self.global_model)
            if fitted is None:
                raise ValueError(f"결측 개수 구간 {band}의 학습 모델이 없다.")
            prediction[mask] = fitted.predict(outer_preds[mask])
        if np.isnan(prediction).any():
            raise ValueError("결측 개수 구간 결합 예측이 전 행을 덮지 않는다.")
        return prediction

    def summary(self) -> dict[str, float]:
        return self.summary_weights


class MissingnessSegmentedLogisticCombiner:
    """결측 개수 구간별로 순위와 logit 이중 표현 선형 결합을 다시 맞춘다.

    specialized_bands가 None이면 세 구간 모두 독립 모델을 쓴다.
    값이 주어지면 나머지 구간은 전역 모델을 유지하고 지정 구간만 전용 모델로
    교체한다.
    """

    name = "missing_segmented_rank_logit"

    def __init__(
        self,
        *,
        band_of: pd.Series | None = None,
        specialized_bands: tuple[int, ...] | None = None,
        name: str | None = None,
    ) -> None:
        if specialized_bands is not None and not set(specialized_bands) <= set(
            MISSINGNESS_BAND_LABELS
        ):
            raise ValueError("전용 모델의 결측 개수 구간이 0, 1, 2 밖에 있다.")
        self.band_of = band_of
        self.specialized_bands = specialized_bands
        self.name = name or type(self).name

    def fit(
        self, inner_preds: pd.DataFrame, y: pd.Series
    ) -> FittedMissingnessSegmentedLogistic:
        band_of = self.band_of if self.band_of is not None else missingness_bands()
        bands = _aligned_bands(band_of, inner_preds.index)
        specialized = (
            MISSINGNESS_BAND_LABELS
            if self.specialized_bands is None
            else self.specialized_bands
        )
        global_model = None
        if self.specialized_bands is not None:
            global_model = LogisticLinearCombiner("global", "rank_logit").fit(
                inner_preds, y
            )
        models = {}
        for band in specialized:
            mask = (bands == band).to_numpy()
            if int(mask.sum()) < 2 or y[mask].nunique() < 2:
                raise ValueError(f"결측 개수 구간 {band}에 두 목표값의 학습 행이 부족하다.")
            models[band] = LogisticLinearCombiner(
                f"missing_band_{band}", "rank_logit"
            ).fit(inner_preds[mask], y[mask])

        counts = bands.value_counts(normalize=True)
        summaries = {}
        for band in MISSINGNESS_BAND_LABELS:
            fitted = models.get(band, global_model)
            assert fitted is not None
            summaries[band] = fitted.summary()
        summary_weights = {
            member: float(
                sum(
                    float(counts.get(band, 0.0)) * summaries[band][member]
                    for band in MISSINGNESS_BAND_LABELS
                )
            )
            for member in inner_preds.columns
        }
        return FittedMissingnessSegmentedLogistic(
            models=models,
            global_model=global_model,
            band_of=band_of,
            summary_weights=summary_weights,
        )


def _missingness_interaction_features(
    preds: pd.DataFrame,
    bands: pd.Series,
    *,
    quantiles: EmpiricalCDFTransformer | None,
    fit: bool,
) -> tuple[np.ndarray, EmpiricalCDFTransformer]:
    base, fitted_quantiles = _linear_features(
        preds,
        "rank_logit",
        quantiles=quantiles,
        fit=fit,
        eps=LogisticLinearCombiner.LOGIT_EPS,
    )
    assert fitted_quantiles is not None
    indicators = np.column_stack(
        [(bands.to_numpy() == band).astype(np.float64) for band in (1, 2)]
    )
    return (
        np.column_stack(
            (base, base * indicators[:, [0]], base * indicators[:, [1]], indicators)
        ),
        fitted_quantiles,
    )


@dataclass(frozen=True)
class FittedMissingnessInteractionLogistic:
    model: LogisticRegression
    scaler: StandardScaler
    quantiles: EmpiricalCDFTransformer
    members: list[str]
    band_of: pd.Series
    summary_weights: dict[str, float]

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        bands = _aligned_bands(self.band_of, outer_preds.index)
        features, _ = _missingness_interaction_features(
            outer_preds[self.members], bands, quantiles=self.quantiles, fit=False
        )
        return self.model.predict_proba(self.scaler.transform(features))[:, 1].astype(
            np.float64, copy=False
        )

    def summary(self) -> dict[str, float]:
        return self.summary_weights


class MissingnessInteractionLogisticCombiner:
    """전역 선형 결합에 결측 개수 구간별 계수 차이를 허용한다."""

    name = "missing_interaction_rank_logit"

    def __init__(
        self, *, band_of: pd.Series | None = None, max_iter: int = 1_000
    ) -> None:
        self.band_of = band_of
        self.max_iter = max_iter

    def fit(
        self, inner_preds: pd.DataFrame, y: pd.Series
    ) -> FittedMissingnessInteractionLogistic:
        band_of = self.band_of if self.band_of is not None else missingness_bands()
        bands = _aligned_bands(band_of, inner_preds.index)
        features, quantiles = _missingness_interaction_features(
            inner_preds, bands, quantiles=None, fit=True
        )
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features).astype(np.float64, copy=False)
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled, y.to_numpy())
        iterations = int(np.max(model.n_iter_))
        if iterations >= self.max_iter:
            raise CombinerConvergenceError(
                f"max(n_iter_)={iterations}, max_iter={self.max_iter}"
            )

        member_count = len(inner_preds.columns)
        coefficients = model.coef_[0]
        base = coefficients[: 2 * member_count]
        band_1 = coefficients[2 * member_count : 4 * member_count]
        band_2 = coefficients[4 * member_count : 6 * member_count]
        frequencies = bands.value_counts(normalize=True)
        effective = (
            base
            + float(frequencies.get(1, 0.0)) * band_1
            + float(frequencies.get(2, 0.0)) * band_2
        )
        effective = effective[:member_count] + effective[member_count:]
        summary_weights = {
            member: float(coefficient)
            for member, coefficient in zip(
                inner_preds.columns, effective, strict=True
            )
        }
        return FittedMissingnessInteractionLogistic(
            model=model,
            scaler=scaler,
            quantiles=quantiles,
            members=list(inner_preds.columns),
            band_of=band_of,
            summary_weights=summary_weights,
        )


@dataclass(frozen=True)
class FittedXGBoostRankLogit:
    model: Any
    quantiles: EmpiricalCDFTransformer
    members: list[str]

    def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
        features, _ = _linear_features(
            outer_preds[self.members],
            "rank_logit",
            quantiles=self.quantiles,
            fit=False,
            eps=LogisticLinearCombiner.LOGIT_EPS,
        )
        return np.asarray(self.model.predict_proba(features)[:, 1], dtype=np.float64)

    def summary(self) -> dict[str, float]:
        importance = np.asarray(self.model.feature_importances_, dtype=np.float64)
        member_count = len(self.members)
        combined = importance[:member_count] + importance[member_count:]
        return {
            member: float(value)
            for member, value in zip(self.members, combined, strict=True)
        }


class XGBoostRankLogitCombiner:
    """결과에 맞춘 탐색 없이 고정한 얕은 XGBoost 2단 결합."""

    name = "xgb_rank_logit"

    def __init__(self, *, n_estimators: int = 200, n_jobs: int = 4) -> None:
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedXGBoostRankLogit:
        from xgboost import XGBClassifier

        features, quantiles = _linear_features(
            inner_preds,
            "rank_logit",
            quantiles=None,
            fit=True,
            eps=LogisticLinearCombiner.LOGIT_EPS,
        )
        assert quantiles is not None
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=200,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=10.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            max_bin=128,
            n_jobs=self.n_jobs,
            random_state=42,
            verbosity=0,
        )
        model.fit(features, y.to_numpy())
        return FittedXGBoostRankLogit(model, quantiles, list(inner_preds.columns))


COMBINER_REGISTRY: dict[str, Combiner] = {
    combiner.name: combiner
    for combiner in (
        RankMeanCombiner(),
        PerformanceWeightedRankMeanCombiner(),
        LogisticLinearCombiner("logit_logistic", "logit"),
        LogisticLinearCombiner("rank_logistic", "rank"),
        RidgeLogitCombiner(alpha=0.01, name="ridge_logit_alpha_0p01"),
        RidgeLogitCombiner(alpha=0.1, name="ridge_logit_alpha_0p1"),
        RidgeLogitCombiner(),
        RidgeLogitCombiner(alpha=10.0, name="ridge_logit_alpha_10"),
        RidgeLogitCombiner(alpha=100.0, name="ridge_logit_alpha_100"),
        LogisticLinearCombiner("rank_gauss_logistic", "rank_gauss"),
        LogisticLinearCombiner("rank_logit_logistic", "rank_logit"),
        GreedyRankMeanCombiner(),
        BaggedGreedyRankMeanCombiner(),
        OptunaSubsetRankMeanCombiner(),
        OptunaSubsetRidgeLogitCombiner(),
        XGBoostRankLogitCombiner(),
        MissingnessSegmentedLogisticCombiner(),
        MissingnessInteractionLogisticCombiner(),
        MissingnessSegmentedLogisticCombiner(
            specialized_bands=(2,), name="missing_4plus_rank_logit"
        ),
        NNLSCombiner("nnls_logit", "logit"),
        NNLSCombiner("nnls_rank", "rank"),
        ShrunkRankLogitCombiner(),
    )
}

PRECISION_COMBINER_NAMES = (
    "bagged_greedy_rank_mean",
    "optuna_subset_rank_mean",
    "optuna_subset_ridge_logit",
)
DEFAULT_COMBINER_NAMES = tuple(
    name for name in COMBINER_REGISTRY if name not in PRECISION_COMBINER_NAMES
)
CANDIDATE_POOL_CORE_COMBINER_NAMES = tuple(
    name
    for name in DEFAULT_COMBINER_NAMES
    if name
    in {
        "missing_segmented_rank_logit",
        "missing_interaction_rank_logit",
        "shrunk_rank_logit_logistic",
    }
)
CANDIDATE_POOL_OPTIONAL_COMBINER_NAMES = tuple(
    name
    for name in DEFAULT_COMBINER_NAMES
    if name not in CANDIDATE_POOL_CORE_COMBINER_NAMES
)
DEFAULT_EVALUATION_OUTPUT = Path("run-logs/ensemble-evaluation.json")
DEFAULT_STRATEGY_OOF_OUTPUT = Path("run-logs/strategy-oof.parquet")
EVALUATION_ARTIFACT_NAME = "ensemble_evaluation.json"
STRATEGY_OOF_ARTIFACT_NAME = "strategy_oof.parquet"
MISSINGNESS_WEIGHTS_ARTIFACT_NAME = "missingness_weights.csv"
# 2: 전략별 weighted_oof_auc와 best_weighted_oof_auc·weighted_oof_sample 추가. (#383)
EVALUATION_ARTIFACT_SCHEMA_VERSION = 2


def combiner_for_context(
    name: str,
    *,
    fold_of: pd.Series,
    band_of: pd.Series,
) -> Combiner:
    """행 문맥이 필요한 등록 결합 전략을 현재 평가 입력에 묶어 돌려준다."""
    if name not in COMBINER_REGISTRY:
        raise ValueError(f"결합 전략 없음: {name}")
    if name == "shrunk_rank_logit_logistic":
        return ShrunkRankLogitCombiner(fold_of=fold_of)
    if name == "missing_segmented_rank_logit":
        return MissingnessSegmentedLogisticCombiner(band_of=band_of)
    if name == "missing_interaction_rank_logit":
        return MissingnessInteractionLogisticCombiner(band_of=band_of)
    if name == "missing_4plus_rank_logit":
        return MissingnessSegmentedLogisticCombiner(
            band_of=band_of,
            specialized_bands=(2,),
            name="missing_4plus_rank_logit",
        )
    return COMBINER_REGISTRY[name]


def select_combiners(
    only: list[str] | None,
) -> tuple[list[Combiner], tuple[str, ...]]:
    """기본 평가 범위 또는 사용자가 명시한 등록 전략을 돌려준다."""
    if only is None:
        return (
            [COMBINER_REGISTRY[name] for name in DEFAULT_COMBINER_NAMES],
            PRECISION_COMBINER_NAMES,
        )
    missing = [name for name in only if name not in COMBINER_REGISTRY]
    if missing:
        raise ValueError(
            f"결합 전략 없음: {', '.join(missing)} "
            f"(등록: {', '.join(COMBINER_REGISTRY)})"
        )
    return [COMBINER_REGISTRY[name] for name in only], ()


@dataclass(frozen=True)
class FoldOutcome:
    """outer fold 하나의 채점 결과와 그 fold에서 학습된 전략의 summary."""

    fold: int
    auc: float
    summary: dict[str, float]


@dataclass(frozen=True)
class NestedEvaluation:
    """전략 하나의 nested OOF 평가 결과."""

    name: str
    nested_auc: float
    elapsed_seconds: float
    folds: list[FoldOutcome]
    prediction: pd.Series  # id 인덱스의 nested OOF 예측. 파생 앙상블 실행 기록의 원본.
    weighted: WeightedOof | None = None  # 결측 패턴 재가중을 건넸을 때만 잰다. (#383)


@dataclass(frozen=True)
class FailedStrategyEvaluation:
    """미수렴으로 점수 비교에서 빠진 전략의 비용과 이유."""

    name: str
    elapsed_seconds: float
    reason: str


@dataclass(frozen=True)
class NestedEvaluationReport:
    """한 번의 결합 전략 비교에서 나온 성공, 실패와 기본 제외 범위."""

    evaluations: list[NestedEvaluation]
    failures: list[FailedStrategyEvaluation]
    default_excluded: tuple[str, ...]


def evaluate_nested(
    combiner: Combiner,
    preds: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    reweighting: MissingnessReweighting | None = None,
) -> NestedEvaluation:
    """outer fold 루프: 나머지 fold의 OOF로 fit하고 fold k만 predict해 합쳐 채점한다.

    reweighting을 건네면 같은 nested 예측을 test 결측 패턴 구성비로 한 번 더 채점해
    가중 OOF를 함께 남긴다. nested_auc는 그대로다(#383, 추가 눈금).
    """
    started = time.monotonic()
    preds = preds.astype(np.float64)  # 결합 전략에는 float64 원시 예측만 건넨다.
    nested = np.full(len(preds), np.nan)
    outcomes = []
    for fold in sorted(fold_of.unique()):
        inner = (fold_of != fold).to_numpy()
        outer = (fold_of == fold).to_numpy()
        try:
            fitted = combiner.fit(preds[inner], y[inner])
        except CombinerConvergenceError as exc:
            raise CombinerConvergenceError(
                f"outer fold {int(fold)}에서 미수렴: {exc}"
            ) from exc
        prediction = np.asarray(fitted.predict(preds[outer]), dtype=np.float64)
        nested[outer] = prediction
        outcomes.append(
            FoldOutcome(
                fold=int(fold),
                auc=float(roc_auc_score(y[outer].to_numpy(), prediction)),
                summary=fitted.summary(),
            )
        )
    assert not np.isnan(nested).any(), "fold 배정이 전 행을 덮지 않는다."
    prediction = pd.Series(nested, index=preds.index, name="prediction")
    weighted = None
    if reweighting is not None:
        from .judgment import weighted_oof_auc  # 지역 import(module docstring).

        weighted = weighted_oof_auc(prediction, y, reweighting)
    return NestedEvaluation(
        name=combiner.name,
        nested_auc=float(roc_auc_score(y.to_numpy(), nested)),
        elapsed_seconds=time.monotonic() - started,
        folds=outcomes,
        prediction=prediction,
        weighted=weighted,
    )


@dataclass(frozen=True)
class MemberStat:
    """구성원 하나의 outer fold 5개 집계: 선택 빈도와 평균 가중치."""

    member: str
    selected: int  # 가중치가 0이 아닌 outer fold 수.
    fold_total: int
    mean_weight: float


def member_stats(evaluation: NestedEvaluation) -> list[MemberStat]:
    """Fitted.summary()를 outer fold 전체에서 모아 구성원별 선택 빈도·평균 가중치를 만든다."""
    members = list(evaluation.folds[0].summary)
    stats = []
    for member in members:
        weights = [outcome.summary[member] for outcome in evaluation.folds]
        stats.append(
            MemberStat(
                member=member,
                selected=sum(weight != 0.0 for weight in weights),
                fold_total=len(weights),
                mean_weight=float(np.mean(weights)),
            )
        )
    return stats


def member_matrix(
    members: list[tuple[str, str]], store: RunStore, index: pd.Index
) -> pd.DataFrame:
    """구성원 OOF 예측 행렬. 컬럼 키는 config 이름(풀 장부가 유일성 보장), float64."""
    configs = [config for config, _ in members]
    assert len(set(configs)) == len(configs), "풀 장부의 config 이름이 중복된다."
    columns = {}
    for config, run_id in members:
        pred = store.oof_of(run_id).reindex(index)
        assert pred.notna().all(), (
            f"구성원 {config}(run {run_id})의 OOF id가 fold 배정과 일치하지 않는다."
        )
        columns[config] = pred
    return pd.DataFrame(columns).astype(np.float64)


def member_test_matrix(
    members: list[tuple[str, str]], store: RunStore, index: pd.Index
) -> pd.DataFrame:
    """구성원 제출 산출물에서 시험 예측 행렬을 읽고 기준 id 순서로 맞춘다."""
    columns = {}
    for config, run_id in members:
        submission = pd.read_csv(store.submission_path_of(run_id))
        if list(submission.columns) != [ID, TARGET]:
            raise ValueError(
                f"구성원 {config}(run {run_id})의 제출 열이 다르다: {list(submission.columns)}"
            )
        if submission[ID].duplicated().any():
            raise ValueError(f"구성원 {config}(run {run_id})의 제출 id가 중복된다.")
        pred = submission.set_index(ID)[TARGET].reindex(index)
        if pred.isna().any() or not np.isfinite(pred.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"구성원 {config}(run {run_id})의 시험 예측이 기준 id와 맞지 않거나 유한하지 않다."
            )
        columns[config] = pred
    return pd.DataFrame(columns, index=index).astype(np.float64)


def full_fit_predictions(
    combiner: Combiner,
    oof: pd.DataFrame,
    y: pd.Series,
    test_preds: pd.DataFrame,
) -> np.ndarray:
    """전체 OOF로 결합 전략을 학습하고 시험 예측을 만든다."""
    if list(oof.columns) != list(test_preds.columns):
        raise ValueError("OOF와 시험 예측의 구성원 순서가 다르다.")
    fitted = combiner.fit(oof.astype(np.float64), y)
    prediction = np.asarray(
        fitted.predict(test_preds.astype(np.float64)), dtype=np.float64
    )
    if prediction.shape != (len(test_preds),) or not np.isfinite(prediction).all():
        raise ValueError("결합 시험 예측의 길이가 다르거나 유한하지 않다.")
    return prediction


@dataclass(frozen=True)
class NestedBaseline:
    """직전 파생 앙상블 실행과의 비교 재료. main이 계산해 기록 함수에 건넨다."""

    run_id: str
    pool_size: int
    previous_best_auc: float
    same_strategy_auc: float | None  # 최선 전략을 직전 구성원 부분집합에 재평가한 값.
    new_member_configs: list[str]


def evaluation_artifact_payload(
    report: NestedEvaluationReport,
    members: list[tuple[str, str]],
) -> dict[str, Any]:
    """결합 전략 비교 결과를 장기 비교 가능한 JSON 값으로 만든다."""
    best = (
        max(report.evaluations, key=lambda evaluation: evaluation.nested_auc)
        if report.evaluations
        else None
    )
    strategies: list[dict[str, Any]] = []
    for evaluation in report.evaluations:
        entry: dict[str, Any] = {
            "name": evaluation.name,
            "status": "completed",
            "elapsed_seconds": evaluation.elapsed_seconds,
            "nested_oof_auc": evaluation.nested_auc,
        }
        if evaluation.weighted is not None:
            entry["weighted_oof_auc"] = evaluation.weighted.auc
        strategies.append(entry)
    strategies.extend(
        {
            "name": failure.name,
            "status": "failed",
            "elapsed_seconds": failure.elapsed_seconds,
            "reason": failure.reason,
        }
        for failure in report.failures
    )
    payload: dict[str, Any] = {
        "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "member_count": len(members),
        "members": [
            {"config": config, "run_id": run_id} for config, run_id in members
        ],
        "default_excluded_strategies": list(report.default_excluded),
        "best_strategy": best.name if best is not None else None,
        "best_nested_oof_auc": best.nested_auc if best is not None else None,
        "strategies": strategies,
    }
    if best is not None and best.weighted is not None:
        payload["best_weighted_oof_auc"] = best.weighted.auc
        payload["weighted_oof_sample"] = {
            "effective_sample_size": best.weighted.effective_sample_size,
            "effective_sample_fraction": best.weighted.effective_sample_fraction,
            "zero_weight_rows": best.weighted.zero_weight_rows,
            "test_only_pattern_count": best.weighted.test_only_pattern_count,
        }
    return payload


def strategy_oof_frame(report: NestedEvaluationReport) -> pd.DataFrame:
    """완료된 전략별 nested OOF 예측을 id 열 + 전략 열 하나씩으로 모은다. (#383)

    두 눈금 비교와 사후 재채점의 원본이다. 미수렴으로 빠진 전략은 예측이 없어 빠진다.
    """
    if not report.evaluations:
        raise ValueError("전략별 OOF 예측을 남길 완료 전략이 없다.")
    index = report.evaluations[0].prediction.index
    columns = {}
    for evaluation in report.evaluations:
        if not evaluation.prediction.index.equals(index):
            raise ValueError(f"전략 {evaluation.name}의 OOF id 순서가 다르다.")
        columns[evaluation.name] = evaluation.prediction.to_numpy(dtype=np.float64)
    return pd.DataFrame(columns, index=index).rename_axis(ID).reset_index()


def write_strategy_oof(report: NestedEvaluationReport, output_path: Path) -> None:
    """전략별 nested OOF 예측을 parquet으로 저장한다. (#383)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_oof_frame(report).to_parquet(output_path, index=False)


def write_evaluation_artifact(
    report: NestedEvaluationReport,
    members: list[tuple[str, str]],
    output_path: Path,
) -> None:
    """전략별 점수와 경과 시간을 기계 판독 JSON으로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            evaluation_artifact_payload(report, members),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )


def record_nested_evaluation(
    evaluation: NestedEvaluation,
    members: list[tuple[str, str]],
    *,
    issue: int,
    baseline: NestedBaseline | None,
    evaluation_report: NestedEvaluationReport | None = None,
    reweighting: MissingnessReweighting | None = None,
    input_hashes: dict[str, str],
    tracking_uri: str = TRACKING_URI,
) -> str:
    """최선 전략의 nested 결과를 파생 앙상블 실행으로 MLflow에 기록한다. (#202)

    #179(run 7fbe590b)·#183(run d845b5d1) 사후 등록의 스키마를 그대로 따른다:
    params에 구성원 계보와 git 상태, metrics에 auc_oof·fold별 AUC·직전 기록 대비 증분,
    artifacts에 oof.parquet(id, prediction), member_weights.csv와 전략별 비교 JSON,
    tags에 source.kind=derived_ensemble와 입력 sha256·산출물 sha256을 남긴다.

    git 상태는 params뿐 아니라 학습 실행과 같은 이름의 tags(git_commit·git_dirty)로도
    남긴다. pipeline.submit·bundle 등 실행 출처를 읽는 도구는 태그 규약만 보므로,
    태그가 없으면 파생 앙상블 실행을 제출할 수 없다. (#416)

    가중 OOF를 잰 실행은 전략별 nested OOF 예측(strategy_oof.parquet)과 결측 패턴별
    가중치 원본(missingness_weights.csv)을 함께 남긴다. (#383)
    """
    import hashlib
    import tempfile
    from datetime import datetime, timezone

    # 최상단 import는 judgment → ensemble 단방향만 둔다(module docstring). tracking은
    # judgment를 최상단에서 당기므로 지역 import로 순환을 피한다.
    from .tracking import git_state, mlflow_client

    run_name = f"ensemble_{evaluation.name}_issue{issue}_pool{len(members)}"
    configs = ",".join(config for config, _ in members)
    run_ids = ",".join(run_id for _, run_id in members)

    client, experiment_id = mlflow_client(tracking_uri)
    run_id = client.create_run(experiment_id, run_name=run_name).info.run_id

    git = git_state()
    params: dict[str, str] = {
        "experiment": run_name,
        "stage": "confirm",
        "model.kind": f"ensemble_{evaluation.name}",
        "ensemble.strategy": evaluation.name,
        "ensemble.member_count": str(len(members)),
        "ensemble.member_configs": configs,
        "ensemble.member_run_ids": run_ids,
        **git,
    }
    if baseline is not None:
        params["ensemble.baseline_run_id"] = baseline.run_id
        params["ensemble.new_member_configs"] = ",".join(baseline.new_member_configs)
    for key, value in params.items():
        client.log_param(run_id, key, value)

    client.log_metric(run_id, "auc_oof", evaluation.nested_auc)
    for outcome in evaluation.folds:
        client.log_metric(run_id, f"auc_fold_{outcome.fold}", outcome.auc)
    if evaluation.weighted is not None:
        for name, value in evaluation.weighted.metrics().items():
            client.log_metric(run_id, name, value)
    if baseline is not None:
        size = baseline.pool_size
        client.log_metric(
            run_id, f"auc_pool{size}_previous_best", baseline.previous_best_auc
        )
        client.log_metric(
            run_id,
            f"delta_vs_pool{size}_previous_best",
            evaluation.nested_auc - baseline.previous_best_auc,
        )
        if baseline.same_strategy_auc is not None:
            client.log_metric(
                run_id, f"auc_pool{size}_same_strategy", baseline.same_strategy_auc
            )
            client.log_metric(
                run_id,
                "delta_same_strategy",
                evaluation.nested_auc - baseline.same_strategy_auc,
            )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        oof_path = tmp_dir / "oof.parquet"
        evaluation.prediction.rename_axis(ID).reset_index().to_parquet(
            oof_path, index=False
        )
        weights_path = tmp_dir / "member_weights.csv"
        pd.DataFrame(
            [
                {
                    "member": stat.member,
                    "selected": stat.selected,
                    "fold_total": stat.fold_total,
                    "mean_weight": stat.mean_weight,
                }
                for stat in member_stats(evaluation)
            ]
        ).to_csv(weights_path, index=False)
        client.log_artifact(run_id, str(oof_path))
        client.log_artifact(run_id, str(weights_path))
        strategy_oof_sha256 = None
        if evaluation_report is not None:
            evaluation_path = tmp_dir / EVALUATION_ARTIFACT_NAME
            write_evaluation_artifact(evaluation_report, members, evaluation_path)
            client.log_artifact(run_id, str(evaluation_path))
            # 전략별 OOF 예측을 남겨야 사후에 두 눈금으로 다시 비교할 수 있다. (#383)
            strategy_oof_path = tmp_dir / STRATEGY_OOF_ARTIFACT_NAME
            write_strategy_oof(evaluation_report, strategy_oof_path)
            client.log_artifact(run_id, str(strategy_oof_path))
            strategy_oof_sha256 = hashlib.sha256(
                strategy_oof_path.read_bytes()
            ).hexdigest()
        if reweighting is not None:
            # 어떤 가중치로 잰 값인지가 없으면 가중 OOF는 감사할 수 없는 수치다. (#383)
            weights_table_path = tmp_dir / MISSINGNESS_WEIGHTS_ARTIFACT_NAME
            reweighting.patterns.to_csv(weights_table_path, index=False)
            client.log_artifact(run_id, str(weights_table_path))
        oof_sha256 = hashlib.sha256(oof_path.read_bytes()).hexdigest()

    tags = {
        **git,
        "source.issue": str(issue),
        "source.kind": "derived_ensemble",
        "ensemble.strategy": evaluation.name,
        "ensemble.member_run_ids": run_ids,
        "sha256.oof_prediction": oof_sha256,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{f"sha256.{name}": digest for name, digest in input_hashes.items()},
    }
    if strategy_oof_sha256 is not None:
        tags["sha256.strategy_oof_prediction"] = strategy_oof_sha256
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    client.set_terminated(run_id, "FINISHED")
    return run_id


def nested_baseline(
    baseline_run_id: str,
    best: NestedEvaluation,
    members: list[tuple[str, str]],
    store: RunStore,
    fold_of: pd.Series,
    y: pd.Series,
) -> NestedBaseline:
    """직전 파생 앙상블 실행에서 비교 재료를 만든다.

    신규 구성원은 직전 실행의 ensemble.member_configs와의 차집합이다. 직전 구성원이
    현재 풀의 부분집합일 때만 최선 전략을 그 부분집합에 재평가해 같은 전략 증분을
    잰다(구성원 이탈이 있으면 짝비교가 성립하지 않으므로 건너뛴다).
    """
    meta = store.facts_of(baseline_run_id)
    if "auc_oof" not in meta.metrics:
        raise RunStoreError(f"기준 실행 {baseline_run_id}에 auc_oof metric이 없다.")
    baseline_configs = [
        config
        for config in meta.params.get("ensemble.member_configs", "").split(",")
        if config
    ]
    if not baseline_configs:
        raise RunStoreError(
            f"기준 실행 {baseline_run_id}에 ensemble.member_configs param이 없다."
        )
    current_configs = [config for config, _ in members]
    same_strategy_auc = None
    if set(baseline_configs) <= set(current_configs):
        matrix = member_matrix(members, store, fold_of.index)
        same_strategy_auc = evaluate_nested(
            COMBINER_REGISTRY[best.name], matrix[baseline_configs], fold_of, y
        ).nested_auc
    return NestedBaseline(
        run_id=baseline_run_id,
        pool_size=len(baseline_configs),
        previous_best_auc=meta.metrics["auc_oof"],
        same_strategy_auc=same_strategy_auc,
        new_member_configs=[
            config for config in current_configs if config not in baseline_configs
        ],
    )


def run_report(
    combiners: list[Combiner],
    members: list[tuple[str, str]],
    store: RunStore,
    fold_of: pd.Series,
    y: pd.Series,
    champion_auc: float,
    *,
    default_excluded: tuple[str, ...] = (),
    reweighting: MissingnessReweighting | None = None,
) -> NestedEvaluationReport:
    """nested 평가부터 계열 3 판정까지의 stdout 리포트. CLI와 golden 테스트가 공유한다.

    reweighting을 건네면 전략마다 가중 OOF를 함께 재서 리포트와 산출물에 남긴다.
    판정은 nested OOF만 쓰므로 판정 줄과 결론은 달라지지 않는다. (#383)
    """
    # 최상단 import는 judgment → ensemble 단방향만 둔다(module docstring). 지역 import.
    from .judgment import AUC_THRESHOLD, StrategyOutcome, judge_ensemble

    matrix = member_matrix(members, store, fold_of.index)

    print(f"구성원 {len(members)}명 (config → run_id):")
    for config, run_id in members:
        print(f"  {config} → {run_id}")
    if default_excluded:
        print(
            "기본 평가에서 제외한 정밀 결합 전략 "
            f"(--only로 명시 선택 가능): {', '.join(default_excluded)}"
        )

    evaluations = []
    failures = []
    for combiner in combiners:
        started = time.monotonic()
        try:
            evaluations.append(
                evaluate_nested(combiner, matrix, fold_of, y, reweighting)
            )
        except CombinerConvergenceError as exc:
            failures.append(
                FailedStrategyEvaluation(
                    name=combiner.name,
                    elapsed_seconds=time.monotonic() - started,
                    reason=str(exc),
                )
            )
    for evaluation in evaluations:
        print(f"전략 {evaluation.name}: nested OOF AUC {evaluation.nested_auc:.5f}")
        if evaluation.weighted is not None:
            weighted = evaluation.weighted
            print(
                f"  가중 OOF AUC {weighted.auc:.5f} "
                f"(nested 대비 {weighted.auc - evaluation.nested_auc:+.5f}, "
                f"유효 표본 {weighted.effective_sample_size:,.0f}"
                f"/{weighted.effective_sample_fraction:.1%}, "
                f"0가중 행 {weighted.zero_weight_rows}, "
                f"test 전용 패턴 {weighted.test_only_pattern_count})"
            )
        folds = " ".join(f"{o.fold}={o.auc:.5f}" for o in evaluation.folds)
        print(f"  outer fold AUC: {folds}")
        print("  구성원별 선택 빈도·평균 가중치:")
        for stat in member_stats(evaluation):
            print(
                f"    {stat.member}: {stat.selected}/{stat.fold_total}, "
                f"{stat.mean_weight:+.5f}"
            )

    for failure in failures:
        print(f"전략 {failure.name}: 점수 비교 제외 ({failure.reason})")

    report = NestedEvaluationReport(
        evaluations=evaluations,
        failures=failures,
        default_excluded=default_excluded,
    )

    verdict = judge_ensemble(
        [
            StrategyOutcome(
                name=e.name,
                nested_auc=e.nested_auc,
                fold_aucs={o.fold: o.auc for o in e.folds},
            )
            for e in evaluations
        ],
        champion_auc=champion_auc,
    )

    wins = ", ".join(
        f"{a.name} {a.fold_wins}/{len(fold_of.unique())}" for a in verdict.assessments
    )
    print(f"fold 승리 (보조 증거): {wins}")
    print(
        f"계열 3 판정: champion OOF AUC {verdict.champion_auc:.5f} 대비 "
        f"문턱 +{AUC_THRESHOLD:.5f}"
    )
    for assessment in verdict.assessments:
        print(
            f"  {assessment.name}: nested {assessment.nested_auc:.5f} "
            f"(delta {assessment.delta:+.5f}) → {'채택 가능' if assessment.eligible else '미달'}"
        )
    if verdict.recommended is None:
        print("판정: 채택 없음, 단독 champion 유지")
        return report
    print(f"판정: {verdict.recommended} 채택 추천 (nested OOF AUC 최고)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="nested OOF 평가와 계열 3 판정 (ADR 0001 계열 3)"
    )
    parser.add_argument(
        "--only",
        action="append",
        help=(
            "이 이름의 등록 결합 전략만 평가. 여러 번 지정할 수 있으며, "
            "정밀 결합 전략도 이 인자로 실행한다."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT,
        help="전략별 nested OOF AUC·가중 OOF AUC와 경과 시간을 저장할 JSON 경로",
    )
    parser.add_argument(
        "--strategy-oof-output",
        type=Path,
        default=DEFAULT_STRATEGY_OOF_OUTPUT,
        help="전략별 nested OOF 예측을 저장할 parquet 경로 (#383)",
    )
    parser.add_argument(
        "--submission", type=Path, help="--only 전략의 전체 OOF 학습 제출 파일 경로"
    )
    parser.add_argument(
        "--record-issue",
        type=int,
        help="최선 전략의 nested 결과를 이 이슈 번호의 파생 앙상블 실행으로 MLflow에 기록",
    )
    parser.add_argument(
        "--baseline-run",
        help="직전 파생 앙상블 실행의 run_id. 증분 지표와 신규 구성원 계보를 함께 기록",
    )
    args = parser.parse_args()

    if args.submission is not None and args.only is None:
        sys.exit("--submission에는 제출 결합 전략을 고르는 --only가 필요하다.")
    if args.submission is not None and len(args.only) != 1:
        sys.exit("--submission에는 --only 전략을 정확히 하나 지정해야 한다.")
    if args.baseline_run is not None and args.record_issue is None:
        sys.exit("--baseline-run에는 기록을 여는 --record-issue가 필요하다.")

    try:
        combiners, default_excluded = select_combiners(args.only)
    except ValueError as exc:
        sys.exit(str(exc))

    if not CHAMPION_PATH.exists():
        sys.exit(f"{CHAMPION_PATH} 없음: 계열 3 판정의 기준 champion이 필요하다.")
    champion = Champion.load()
    pool = Pool.load()
    if not pool.members:
        sys.exit("후보 풀이 비어 있다: nested OOF 평가는 풀 구성원이 필요하다.")

    # run_report와 같은 이유의 지역 import.
    from .judgment import FOLDS_PATH, missingness_reweighting

    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    store = MlflowRunStore()
    reweighting = missingness_reweighting(TRAIN_PATH, MISSINGNESS_TEST_PATH)
    try:
        members = [(member.config, member.run_id) for member in pool.members]
        report = run_report(
            combiners,
            members,
            store,
            fold_of,
            y,
            champion.oof_auc,
            default_excluded=default_excluded,
            reweighting=reweighting,
        )
        write_evaluation_artifact(report, members, args.output)
        print(f"평가 산출물 저장: {args.output}")
        if report.evaluations:
            write_strategy_oof(report, args.strategy_oof_output)
            print(
                f"전략별 OOF 예측 저장: {args.strategy_oof_output} "
                f"({len(report.evaluations)}개 전략)"
            )
        evaluations = report.evaluations
        if args.record_issue is not None:
            from .tracking import git_state

            if git_state()["git_dirty"] == "True":
                sys.exit("기록 거부: git_dirty 상태다. 우회 옵션은 없다. 커밋 후 재실행할 것.")
            if not evaluations:
                sys.exit("기록할 평가가 없다: 모든 전략이 비교에서 제외됐다.")
            best = max(evaluations, key=lambda evaluation: evaluation.nested_auc)
            baseline = (
                nested_baseline(args.baseline_run, best, members, store, fold_of, y)
                if args.baseline_run is not None
                else None
            )
            recorded = record_nested_evaluation(
                best,
                members,
                issue=args.record_issue,
                baseline=baseline,
                evaluation_report=report,
                reweighting=reweighting,
                input_hashes={
                    "train": file_sha256(TRAIN_PATH),
                    "test": file_sha256(MISSINGNESS_TEST_PATH),
                    "folds": file_sha256(FOLDS_PATH),
                },
            )
            print(
                f"파생 앙상블 실행 기록: {recorded} "
                f"(전략 {best.name}, 구성원 {len(members)}명)"
            )
        if args.submission is not None:
            oof = member_matrix(members, store, fold_of.index)
            template = pd.read_csv("data/sample_submission.csv", usecols=[ID, TARGET])
            test_index = pd.Index(template[ID], name=ID)
            test_preds = member_test_matrix(members, store, test_index)
            prediction = full_fit_predictions(combiners[0], oof, y, test_preds)
            args.submission.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: test_index, TARGET: prediction}).to_csv(
                args.submission, index=False
            )
            print(f"제출 파일 저장: {args.submission} ({len(prediction)}행, float64)")
    except RunStoreError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
