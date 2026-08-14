"""nested OOF 평가기와 결합 전략 계약. (ADR 0001 계열 3, #104)

사용법:
    uv run python -m pipeline.ensemble                    # registry 전 전략 평가·비교·판정
    uv run python -m pipeline.ensemble --only rank_mean   # 개발·디버깅용 부분 실행

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
- Fitted.predict는 outer fold 행만 받는다. 순위 변환의 모집단은 채점 블록 자신이다.
  제출 시점의 결합이 test 예측만으로 순위를 매기므로, 평가도 같은 조건이어야
  nested 점수가 제출 동작을 대변한다.
- 각 adapter는 복잡도 서열(선택 자유도 순위)을 선언한다. 판정의 동률 해소가 이
  서열로 기계 판정된다.
- Fitted.summary()(구성원 이름 → 가중치/선택 여부)를 outer fold 5개에서 모아
  구성원별 선택 빈도·평균 가중치 표를 만든다. #62의 "선택 빈도와 fold별 승리 기록"
  요구를 adapter 추가만으로 감당하기 위한 규약이다.

구성원 예측 행렬의 컬럼 키는 config 이름이다(풀 장부가 유일성 보장). run_id 대응은
리포트 머리에 한 번 출력한다.

MLflow run을 만들지 않는다. stdout 리포트만 낸다(실험 하나 = run 하나 규약 유지).
결과는 티켓 코멘트로 남기는 기존 관행 그대로. 제출 결합(전체 OOF로 fit한 전략을
구성원 test 예측에 적용)은 이번 범위가 아니며, fit/predict 분리로 interface는 이미
감당 가능하므로 #64에서 전략이 채택된 뒤에 짓는다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from .data import ID, labels
from .ledger import CHAMPION_PATH, Champion, Pool
from .runs import MlflowRunStore, RunStore, RunStoreError


def rank_mean(preds: pd.DataFrame) -> np.ndarray:
    """균등 순위 평균의 수식 단일 소스: 블록 내 백분위 순위의 구성원 평균.

    순위의 모집단은 전달된 블록 자신이다. judgment 계열 2 기여 판정의
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
    complexity: int  # 복잡도 서열(선택 자유도 순위). 낮을수록 단순하다.

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
    complexity = 1  # 무학습이라 선택 자유도가 가장 낮다.

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedRankMean:
        return FittedRankMean(list(inner_preds.columns))


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
    # #64의 두 전략(제한 가중 순위 평균, logit 로지스틱 회귀)이 2·3으로 사이에 선언된다.
    complexity = 4
    LOGIT_EPS = 1e-6  # 0/1 포화 예측의 logit 발산을 막는 클리핑. adapter 소유 상수.

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series) -> FittedRidgeLogit:
        model = Ridge(alpha=self.alpha)
        model.fit(_logit(inner_preds, self.LOGIT_EPS), y.to_numpy())
        return FittedRidgeLogit(model, list(inner_preds.columns), self.LOGIT_EPS)


COMBINER_REGISTRY: dict[str, Combiner] = {
    combiner.name: combiner for combiner in (RankMeanCombiner(), RidgeLogitCombiner())
}


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
    complexity: int
    nested_auc: float
    folds: list[FoldOutcome]


def evaluate_nested(
    combiner: Combiner, preds: pd.DataFrame, fold_of: pd.Series, y: pd.Series
) -> NestedEvaluation:
    """outer fold 루프: 나머지 fold의 OOF로 fit하고 fold k만 predict해 합쳐 채점한다."""
    preds = preds.astype(np.float64)  # 결합 전략에는 float64 원시 예측만 건넨다.
    nested = np.full(len(preds), np.nan)
    outcomes = []
    for fold in sorted(fold_of.unique()):
        inner = (fold_of != fold).to_numpy()
        outer = (fold_of == fold).to_numpy()
        fitted = combiner.fit(preds[inner], y[inner])
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
    return NestedEvaluation(
        name=combiner.name,
        complexity=combiner.complexity,
        nested_auc=float(roc_auc_score(y.to_numpy(), nested)),
        folds=outcomes,
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


def run_report(
    combiners: list[Combiner],
    members: list[tuple[str, str]],
    store: RunStore,
    fold_of: pd.Series,
    y: pd.Series,
    champion_auc: float,
) -> None:
    """nested 평가부터 계열 3 판정까지의 stdout 리포트. CLI와 golden 테스트가 공유한다."""
    # 최상단 import는 judgment → ensemble 단방향만 둔다(module docstring). 지역 import.
    from .judgment import AUC_THRESHOLD, StrategyOutcome, judge_ensemble

    matrix = member_matrix(members, store, fold_of.index)

    print(f"구성원 {len(members)}명 (config → run_id):")
    for config, run_id in members:
        print(f"  {config} → {run_id}")

    evaluations = [evaluate_nested(combiner, matrix, fold_of, y) for combiner in combiners]
    for evaluation in evaluations:
        print(
            f"전략 {evaluation.name} (복잡도 서열 {evaluation.complexity}): "
            f"nested OOF AUC {evaluation.nested_auc:.5f}"
        )
        folds = " ".join(f"{o.fold}={o.auc:.5f}" for o in evaluation.folds)
        print(f"  outer fold AUC: {folds}")
        print("  구성원별 선택 빈도·평균 가중치:")
        for stat in member_stats(evaluation):
            print(
                f"    {stat.member}: {stat.selected}/{stat.fold_total}, "
                f"{stat.mean_weight:+.5f}"
            )

    verdict = judge_ensemble(
        [
            StrategyOutcome(
                name=e.name,
                complexity=e.complexity,
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
    print(f"계열 3 판정: champion OOF AUC {verdict.champion_auc:.5f} 대비 문턱 +{AUC_THRESHOLD}")
    for assessment in verdict.assessments:
        print(
            f"  {assessment.name}: nested {assessment.nested_auc:.5f} "
            f"(delta {assessment.delta:+.5f}) → {'채택 가능' if assessment.eligible else '미달'}"
        )
    if verdict.recommended is None:
        print("판정: 채택 없음, 단독 champion 유지")
        return
    print(f"동률 그룹 (1위와 차이 {AUC_THRESHOLD} 미만): {', '.join(verdict.tie_group)}")
    print(f"판정: {verdict.recommended} 채택 추천 (동률 그룹에서 복잡도 서열 최저)")


def main() -> None:
    parser = argparse.ArgumentParser(description="nested OOF 평가와 계열 3 판정 (ADR 0001 계열 3)")
    parser.add_argument("--only", help="이 이름의 결합 전략만 평가 (개발·디버깅용)")
    args = parser.parse_args()

    combiners = list(COMBINER_REGISTRY.values())
    if args.only is not None:
        if args.only not in COMBINER_REGISTRY:
            sys.exit(f"결합 전략 없음: {args.only} (등록: {', '.join(COMBINER_REGISTRY)})")
        combiners = [COMBINER_REGISTRY[args.only]]

    if not CHAMPION_PATH.exists():
        sys.exit(f"{CHAMPION_PATH} 없음: 계열 3 판정의 기준 champion이 필요하다.")
    champion = Champion.load()
    pool = Pool.load()
    if not pool.members:
        sys.exit("후보 풀이 비어 있다: nested OOF 평가는 풀 구성원이 필요하다.")

    from .judgment import FOLDS_PATH  # run_report와 같은 이유의 지역 import.

    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    store = MlflowRunStore()
    try:
        run_report(
            combiners,
            [(member.config, member.run_id) for member in pool.members],
            store,
            fold_of,
            y,
            champion.oof_auc,
        )
    except RunStoreError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
