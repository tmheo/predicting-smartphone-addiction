"""한 시드의 교차 검증 실행을 소유하는 내부 구성 요소.

호출자는 ``cv.run_cv``만 사용한다.
이 파일의 단일 진입점인 ``execute_seed``는 피처 준비, 모델 호출,
폴드 결과 확정과 최종 결과 조립을 내부에 숨긴다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import initial_score as initial_score_mod
from . import model as model_mod
from .config import ExperimentConfig
from .data import ID, TARGET
from .plan import FeaturePlan, prepare_fold_fit_input
from .recovery import FoldRecovery

if TYPE_CHECKING:
    from .cv import CVResult, RunRecorder


def execute_seed(
    cfg: ExperimentConfig,
    plan: FeaturePlan,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    recorder: RunRecorder | None = None,
    recovery: FoldRecovery | None = None,
) -> CVResult:
    """한 시드의 현행 교차 검증 실행 전체를 수행한다."""
    from .cv import CVResult, score_predictions

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
            for transformer in transformers:
                transformer.fit(train_ff.loc[tr_idx], seed)
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
        model_mod.set_dataset_reference(adapter, X_fold, X_test_fold)
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
            recorder.fold_completed(cfg.seeds.index(seed), fold, fold_auc)

    return CVResult(
        oof=pd.DataFrame({"id": train[ID], "fold": train["fold"], "pred": oof_pred}),
        test_pred=pd.DataFrame({"id": test[ID], "pred": test_pred}),
        fold_aucs=score_predictions(y, train["fold"], oof_pred),
        feature_names=feature_names,
        importance=pd.concat(importances, ignore_index=True),
        recovery_evidence=recovery_records,
        model_training_diagnostics=model_training_diagnostics,
    )
