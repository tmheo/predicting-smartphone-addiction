"""CV 루프. 실행당 산출물 규약의 원천.

반환 규약 (#14의 스키마 결정):
- oof: id, fold, pred 세 컬럼. 앙상블이 파일 하나로 fold 정렬 검증과 병합을 같이 한다.
- test_pred: id, pred 두 컬럼.
- fold_aucs: auc_fold_0 .. auc_fold_4, 그리고 전체 auc_oof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig
from .plan import FeaturePlan
from .recovery import FoldRecovery


class RunRecorder(Protocol):
    """CV가 진행 상황을 통지하는 좁은 계약. 구현은 observe.RunObserver. (#43)

    CV는 무슨 일이 있었는지만 알리고, MLflow 지표 이름과 step 규약은 기록기가 소유한다.
    """

    def stage(self, name: str) -> None: ...

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None: ...

    def record_timing(self, event: dict[str, object]) -> None: ...


@dataclass
class CVResult:
    oof: pd.DataFrame  # columns: id, fold, pred
    test_pred: pd.DataFrame  # columns: id, pred
    fold_aucs: dict[str, float]  # auc_fold_0..N, auc_oof
    feature_names: list[str]
    importance: pd.DataFrame  # columns: feature, fold, seed, gain (#19)
    recovery_evidence: list[dict[str, object]] = field(default_factory=list)
    fold_feature_reuse_evidence: list[dict[str, object]] = field(default_factory=list)
    model_training_diagnostics: list[dict[str, object]] = field(default_factory=list)


def score_predictions(y: pd.Series, folds: pd.Series, pred: np.ndarray) -> dict[str, float]:
    """fold별 AUC와 전체 OOF AUC를 계산한다. 시드 평균 예측의 재채점에도 쓴다. (#15)"""
    fold_aucs: dict[str, float] = {}
    for fold in sorted(folds.unique()):
        mask = folds == fold
        fold_aucs[f"auc_fold_{int(fold)}"] = roc_auc_score(y[mask], pred[mask])
    fold_aucs["auc_oof"] = roc_auc_score(y, pred)
    return fold_aucs


def run_cv(
    cfg: ExperimentConfig,
    plan: FeaturePlan,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    recorder: RunRecorder | None = None,
    recovery: FoldRecovery | None = None,
) -> CVResult:
    """커밋된 fold 배정대로 학습하고 OOF와 테스트 예측을 만든다.

    시드 반복(cfg.seeds가 여럿)일 때는 run.py가 이 함수를 시드별로 부르고 예측을 평균한다.
    피처 구성은 주입받은 피처 계획만 안다: 행렬은 plan.build_matrix가 만들고,
    fold-fit 컬럼은 plan.fold_fit_transformers를 fold 안에서 fit해 더한다. (#71)
    recorder가 없으면(단독 실행, 노트북) 아무것도 기록하지 않는다. (#43)
    """
    from .cv_seed_execution import execute_seed

    return execute_seed(
        cfg,
        plan,
        train,
        test,
        seed,
        recorder=recorder,
        recovery=recovery,
    )
