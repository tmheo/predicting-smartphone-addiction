"""CV 루프. 실행당 산출물 규약의 원천.

반환 규약 (#14의 스키마 결정):
- oof: id, fold, pred 세 컬럼. 앙상블이 파일 하나로 fold 정렬 검증과 병합을 같이 한다.
- test_pred: id, pred 두 컬럼.
- fold_aucs: auc_fold_0 .. auc_fold_4, 그리고 전체 auc_oof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import model as model_mod
from .config import ExperimentConfig
from .data import ID, TARGET
from .features import add_fold_fit_columns, build_features, make_fold_fit


@dataclass
class CVResult:
    oof: pd.DataFrame  # columns: id, fold, pred
    test_pred: pd.DataFrame  # columns: id, pred
    fold_aucs: dict[str, float]  # auc_fold_0..N, auc_oof
    feature_names: list[str]
    importance: pd.DataFrame  # columns: feature, fold, seed, gain (#19)


def score_predictions(y: pd.Series, folds: pd.Series, pred: np.ndarray) -> dict[str, float]:
    """fold별 AUC와 전체 OOF AUC를 계산한다. 시드 평균 예측의 재채점에도 쓴다. (#15)"""
    fold_aucs: dict[str, float] = {}
    for fold in sorted(folds.unique()):
        mask = folds == fold
        fold_aucs[f"auc_fold_{int(fold)}"] = roc_auc_score(y[mask], pred[mask])
    fold_aucs["auc_oof"] = roc_auc_score(y, pred)
    return fold_aucs


def _with_built_columns(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """원본 df에 build_features가 만든 컬럼(placebo 등)을 더한 fold-fit 입력을 만든다.

    placebo 카나리아를 타깃 인코딩하려면 fold-fit 입력에 placebo 컬럼이 있어야 한다. (#33 파급)
    """
    extra = [c for c in X.columns if c not in df.columns]
    return pd.concat([df, X[extra]], axis=1) if extra else df


def run_cv(cfg: ExperimentConfig, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> CVResult:
    """커밋된 fold 배정대로 학습하고 OOF와 테스트 예측을 만든다.

    시드 반복(cfg.seeds가 여럿)일 때는 run.py가 이 함수를 시드별로 부르고 예측을 평균한다.
    """
    X = build_features(train, cfg.features, seed)
    X_test = build_features(test, cfg.features, seed)
    y = train[TARGET]
    transformers = make_fold_fit(cfg.features)
    if transformers:
        train_ff = _with_built_columns(train, X)
        test_ff = _with_built_columns(test, X_test)

    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    n_folds = int(train["fold"].max()) + 1
    importances: list[pd.DataFrame] = []
    feature_names = list(X.columns)

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
            X_fold = add_fold_fit_columns(transformers, X, train_ff)
            X_test_fold = add_fold_fit_columns(transformers, X_test, test_ff)
            assert list(X_fold.columns) == list(X_test_fold.columns), (
                "train/test의 fold-fit 컬럼 집합이 다르다."
            )
            if fold == 0:
                feature_names = list(X_fold.columns)
            else:
                assert list(X_fold.columns) == feature_names, (
                    f"fold {fold}의 컬럼 집합이 fold 0과 다르다."
                )
        model, va_pred = model_mod.train_one_fold(
            cfg.model, X_fold.loc[tr_idx], y.loc[tr_idx], X_fold.loc[va_idx], y.loc[va_idx], seed
        )
        oof_pred[va_idx] = va_pred
        test_pred += model_mod.predict_test(model, X_test_fold) / n_folds
        importances.append(model_mod.gain_importance(model).assign(fold=fold, seed=seed))

    return CVResult(
        oof=pd.DataFrame({"id": train[ID], "fold": train["fold"], "pred": oof_pred}),
        test_pred=pd.DataFrame({"id": test[ID], "pred": test_pred}),
        fold_aucs=score_predictions(y, train["fold"], oof_pred),
        feature_names=feature_names,
        importance=pd.concat(importances, ignore_index=True),
    )
