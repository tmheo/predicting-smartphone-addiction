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
from .fold_observability import skipped_operation, timed_operation
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
    providers = plan.fold_fit_providers()
    if providers:
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
        with timed_operation(
            recorder,
            seed=seed,
            fold=fold,
            operation="fold_feature",
            actor_kind="pipeline",
            actor_name="feature_plan",
        ):
            if providers:
                # fold-fit 단계: 학습 fold로만 fit하고, 같은 상태를 검증 fold와 test에 적용한다.
                # 전체 train으로 fit하는 별도 경로는 없다. (#32 결정 4)
                # transform은 학습 fold 행과 검증 fold 행이 섞인 train 전체를 받는다.
                for kind, transformer in providers:
                    with timed_operation(
                        recorder,
                        seed=seed,
                        fold=fold,
                        operation="fold_feature.provider_fit",
                        actor_kind="column_provider",
                        actor_name=kind,
                    ):
                        transformer.fit(train_ff.loc[tr_idx], seed)
                X_fold = X
                X_test_fold = X_test
                for kind, transformer in providers:
                    with timed_operation(
                        recorder,
                        seed=seed,
                        fold=fold,
                        operation="fold_feature.provider_transform",
                        actor_kind="column_provider",
                        actor_name=kind,
                        dataset="train",
                    ):
                        X_fold = plan.add_fold_fit_provider_columns(
                            X_fold, train_ff, kind, transformer
                        )
                    with timed_operation(
                        recorder,
                        seed=seed,
                        fold=fold,
                        operation="fold_feature.provider_transform",
                        actor_kind="column_provider",
                        actor_name=kind,
                        dataset="test",
                    ):
                        X_test_fold = plan.add_fold_fit_provider_columns(
                            X_test_fold, test_ff, kind, transformer
                        )
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
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="recovery.read_validate",
                actor_kind="recovery",
                actor_name="fold_recovery",
            ):
                checkpoint = recovery.load(
                    seed,
                    fold,
                    validation_ids=train.loc[va_idx, ID],
                    validation_labels=y.loc[va_idx],
                    test_ids=test[ID],
                    feature_names=feature_names,
                )
        else:
            skipped_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="recovery.read_validate",
                actor_kind="recovery",
                actor_name="fold_recovery",
                reason="disabled",
            )
        if checkpoint is not None:
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_finalize",
                actor_kind="pipeline",
                actor_name="fold_result",
                outcome="reused",
                reason="checkpoint_reused",
            ):
                for operation in (
                    "fold_finalize.model_fit",
                    "fold_finalize.test_prediction",
                    "fold_finalize.importance_prepare",
                    "fold_finalize.importance_reinference",
                    "fold_finalize.importance_score",
                    "fold_finalize.training_diagnostics",
                    "fold_finalize.fold_score",
                ):
                    skipped_operation(
                        recorder,
                        seed=seed,
                        fold=fold,
                        operation=operation,
                        actor_kind="model",
                        actor_name=cfg.model.kind,
                        reason="checkpoint_reused",
                    )
                oof_pred[va_idx] = checkpoint.validation_predictions["pred"].to_numpy()
                test_pred += checkpoint.test_predictions["pred"].to_numpy() / n_folds
                importances.append(checkpoint.importance)
                recovery_records.append(checkpoint.evidence(reused=True))
                if checkpoint.model_training_diagnostics is not None:
                    model_training_diagnostics.append(checkpoint.model_training_diagnostics)
            skipped_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="recovery.write_commit",
                actor_kind="recovery",
                actor_name="fold_recovery",
                reason="checkpoint_reused",
            )
            if recorder is not None:
                recorder.fold_completed(cfg.seeds.index(seed), fold, checkpoint.auc)
            continue

        with timed_operation(
            recorder,
            seed=seed,
            fold=fold,
            operation="fold_finalize",
            actor_kind="pipeline",
            actor_name="fold_result",
        ):
            adapter = model_mod.create(cfg.model, seed)
            model_mod.set_dataset_reference(adapter, X_fold, X_test_fold)
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_finalize.model_fit",
                actor_kind="model",
                actor_name=cfg.model.kind,
            ):
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
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_finalize.test_prediction",
                actor_kind="model",
                actor_name=cfg.model.kind,
                dataset="test",
            ):
                fold_test_pred = np.asarray(
                    adapter.predict(
                        X_test_fold,
                        initial_scores.test if initial_scores is not None else None,
                    ),
                    dtype="float64",
                )
            test_pred += fold_test_pred / n_folds
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_finalize.importance_prepare",
                actor_kind="model",
                actor_name=cfg.model.kind,
            ):
                raw_importance = adapter.importance()
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_finalize.importance_score",
                actor_kind="model",
                actor_name=cfg.model.kind,
            ):
                fold_importance = raw_importance.assign(fold=fold, seed=seed)
                fold_importance["gain"] = fold_importance["gain"].astype("float64")
            importances.append(fold_importance)
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_finalize.training_diagnostics",
                actor_kind="model",
                actor_name=cfg.model.kind,
            ):
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
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_finalize.fold_score",
                actor_kind="model",
                actor_name=cfg.model.kind,
            ):
                fold_auc = roc_auc_score(y.loc[va_idx], va_pred)
        if recovery is not None:
            with timed_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="recovery.write_commit",
                actor_kind="recovery",
                actor_name="fold_recovery",
            ):
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
        else:
            skipped_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="recovery.write_commit",
                actor_kind="recovery",
                actor_name="fold_recovery",
                reason="disabled",
            )
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
