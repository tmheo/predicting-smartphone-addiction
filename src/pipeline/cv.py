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

from . import initial_score as initial_score_mod
from . import model as model_mod
from .config import ExperimentConfig
from .data import ID, TARGET
from .plan import FeaturePlan, prepare_fold_fit_input
from .recovery import FoldRecovery


class RunRecorder(Protocol):
    """CV가 진행 상황을 통지하는 좁은 계약. 구현은 observe.RunObserver. (#43)

    CV는 무슨 일이 있었는지만 알리고, MLflow 지표 이름과 step 규약은 기록기가 소유한다.
    """

    def stage(self, name: str) -> None: ...

    def fold_completed(self, seed_index: int, fold_index: int, auc: float) -> None: ...


@dataclass
class CVResult:
    oof: pd.DataFrame  # columns: id, fold, pred
    test_pred: pd.DataFrame  # columns: id, pred
    fold_aucs: dict[str, float]  # auc_fold_0..N, auc_oof
    feature_names: list[str]
    importance: pd.DataFrame  # columns: feature, fold, seed, gain (#19)
    recovery_evidence: list[dict[str, object]] = field(default_factory=list)
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
    if recorder is not None:
        recorder.stage("feature_build")
    initial_provider = initial_score_mod.create(cfg.initial_score)
    initial_scores = (
        initial_provider.compute(train.drop(columns=[TARGET]), test, seed)
        if initial_provider is not None
        else None
    )
    if initial_scores is not None:
        assert initial_scores.train.index.equals(train.index), "train 초기 점수 인덱스가 다르다."
        assert initial_scores.test.index.equals(test.index), "test 초기 점수 인덱스가 다르다."
    X = plan.build_matrix(train, seed)
    X_test = plan.build_matrix(test, seed)
    y = train[TARGET]
    transformers = plan.fold_fit_transformers()
    if transformers:
        train_ff = prepare_fold_fit_input(train, X)
        test_ff = prepare_fold_fit_input(test, X_test)

    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    n_folds = int(train["fold"].max()) + 1
    importances: list[pd.DataFrame] = []
    recovery_records: list[dict[str, object]] = []
    model_training_diagnostics: list[dict[str, object]] = []
    feature_names = list(X.columns)

    # fold 안의 fold-fit 변환 fit도 training 단계에 포함한다. (#40)
    if recorder is not None:
        recorder.stage("training")
    for fold in range(n_folds):
        va_idx = train.index[train["fold"] == fold]
        tr_idx = train.index[train["fold"] != fold]
        X_fold, X_test_fold = X, X_test
        if transformers:
            # fold-fit 단계: 학습 fold로만 fit하고, 같은 상태를 검증 fold와 test에 적용한다.
            # 전체 train으로 fit하는 별도 경로는 없다. (#32 결정 4)
            # transform은 학습 fold 행과 검증 fold 행이 섞인 train 전체를 받는다.
            # 학습 행에 OOF 값을 줘야 하는 트랜스포머는 fit 때 저장한 행 집합(id 기준,
            # test와 위치 인덱스가 겹치므로)으로 두 경우를 구분해 돌려준다. (#33 파급)
            for t in transformers:
                t.fit(train_ff.loc[tr_idx], seed)
            X_fold = plan.add_fold_fit_columns(X, train_ff)
            X_test_fold = plan.add_fold_fit_columns(X_test, test_ff)
            assert list(X_fold.columns) == list(X_test_fold.columns), (
                "train/test의 fold-fit 컬럼 집합이 다르다."
            )
            if fold == 0:
                feature_names = list(X_fold.columns)
            else:
                assert list(X_fold.columns) == feature_names, (
                    f"fold {fold}의 컬럼 집합이 fold 0과 다르다."
                )
        checkpoint = None
        if recovery is not None:
            checkpoint = recovery.load(
                seed,
                fold,
                validation_ids=train.loc[va_idx, ID],
                validation_labels=y.loc[va_idx],
                test_ids=test[ID],
                feature_names=feature_names,
            )
        if checkpoint is not None:
            oof_pred[va_idx] = checkpoint.validation_predictions["pred"].to_numpy()
            test_pred += checkpoint.test_predictions["pred"].to_numpy() / n_folds
            importances.append(checkpoint.importance)
            recovery_records.append(checkpoint.evidence(reused=True))
            if checkpoint.model_training_diagnostics is not None:
                model_training_diagnostics.append(checkpoint.model_training_diagnostics)
            if recorder is not None:
                recorder.fold_completed(cfg.seeds.index(seed), fold, checkpoint.auc)
            continue

        adapter = model_mod.create(cfg.model, seed)
        va_pred = np.asarray(
            adapter.fit(
                X_fold.loc[tr_idx],
                y.loc[tr_idx],
                X_fold.loc[va_idx],
                y.loc[va_idx],
                initial_scores.train.loc[tr_idx] if initial_scores is not None else None,
                initial_scores.train.loc[va_idx] if initial_scores is not None else None,
            ),
            dtype="float64",
        )
        oof_pred[va_idx] = va_pred
        fold_test_pred = np.asarray(
            adapter.predict(
                X_test_fold,
                initial_scores.test if initial_scores is not None else None,
            ),
            dtype="float64",
        )
        test_pred += fold_test_pred / n_folds
        fold_importance = adapter.importance().assign(fold=fold, seed=seed)
        fold_importance["gain"] = fold_importance["gain"].astype("float64")
        importances.append(fold_importance)
        adapter_training_diagnostics = model_mod.collect_training_diagnostics(adapter)
        fold_training_diagnostics = (
            {
                "model_kind": cfg.model.kind,
                "seed": seed,
                "fold": fold,
                "details": adapter_training_diagnostics,
            }
            if adapter_training_diagnostics is not None
            else None
        )
        if fold_training_diagnostics is not None:
            model_training_diagnostics.append(fold_training_diagnostics)
        fold_auc = roc_auc_score(y.loc[va_idx], va_pred)
        if recovery is not None:
            checkpoint = recovery.save(
                seed,
                fold,
                validation_predictions=pd.DataFrame(
                    {ID: train.loc[va_idx, ID], "fold": fold, "pred": va_pred}
                ),
                validation_ids=train.loc[va_idx, ID],
                validation_labels=y.loc[va_idx],
                test_predictions=pd.DataFrame({ID: test[ID], "pred": fold_test_pred}),
                test_ids=test[ID],
                importance=fold_importance,
                feature_names=feature_names,
                model_training_diagnostics=fold_training_diagnostics,
            )
            recovery_records.append(checkpoint.evidence(reused=False))
        if recorder is not None:
            # 실행 중 fold AUC는 해당 시드의 fold 예측 기준. 최종 auc_fold_*는
            # 시드 평균 예측으로 평가 단계에서 다시 채점한다. (#40)
            recorder.fold_completed(
                cfg.seeds.index(seed), fold, fold_auc
            )

    return CVResult(
        oof=pd.DataFrame({"id": train[ID], "fold": train["fold"], "pred": oof_pred}),
        test_pred=pd.DataFrame({"id": test[ID], "pred": test_pred}),
        fold_aucs=score_predictions(y, train["fold"], oof_pred),
        feature_names=feature_names,
        importance=pd.concat(importances, ignore_index=True),
        recovery_evidence=recovery_records,
        model_training_diagnostics=model_training_diagnostics,
    )
