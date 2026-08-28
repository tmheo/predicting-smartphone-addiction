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
from . import training_length as training_length_mod
from .config import ExperimentConfig
from .data import ID, TARGET
from .features import PLACEBO, PLACEBO_MASK_SOURCE
from .fold_fit_reuse import dataframe_value_sha256
from .fold_observability import skipped_operation, timed_operation
from .plan import FeaturePlan, prepare_fold_fit_input
from .recovery import FoldRecovery
from .training_rows import PARENT_ID, build_training_rows, validation_context

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
    y = train[TARGET]
    n_folds = int(train["fold"].max()) + 1
    feature_names = plan.all_columns()
    uses_replicas = bool(
        cfg.training_rows is not None and cfg.training_rows.replica_count
    )
    if uses_replicas:
        plan.validate_training_row_augmentation()
        if PLACEBO not in feature_names or PLACEBO_MASK_SOURCE not in plan.raw_columns():
            raise ValueError("복제 학습 행의 플라시보 카나리아 계약을 확인할 수 없다.")
        if PARENT_ID in feature_names:
            raise AssertionError(f"내부 부모 행 문맥이 모형 피처에 노출됐다: {PARENT_ID}")
    checkpoints = []
    for fold in range(n_folds):
        va_idx = train.index[train["fold"] == fold]
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
            checkpoint = None
            skipped_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="recovery.read_validate",
                actor_kind="recovery",
                actor_name="fold_recovery",
                reason="disabled",
            )
        checkpoints.append(checkpoint)

    provider_declarations = plan.fold_fit_providers()
    needs_computation = any(checkpoint is None for checkpoint in checkpoints)
    if needs_computation:
        initial_provider = initial_score_mod.create(cfg.initial_score)
        initial_scores = (
            initial_provider.compute(train.drop(columns=[TARGET]), test, seed)
            if initial_provider is not None
            else None
        )
        if initial_scores is not None:
            assert initial_scores.train.index.equals(train.index), (
                "train 초기 점수 인덱스가 다르다."
            )
            assert initial_scores.test.index.equals(test.index), (
                "test 초기 점수 인덱스가 다르다."
            )
        X = plan.build_matrix(train, seed)
        X_test = plan.build_matrix(test, seed)
        if provider_declarations:
            train_ff = None if uses_replicas else prepare_fold_fit_input(train, X)
            test_ff = prepare_fold_fit_input(test, X_test)
        else:
            train_ff = None
            test_ff = None
    else:
        initial_scores = None
        X = None
        X_test = None
        train_ff = None
        test_ff = None

    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    importances: list[pd.DataFrame] = []
    recovery_records: list[dict[str, object]] = []
    fold_feature_reuse_records: list[dict[str, object]] = []
    model_training_diagnostics: list[dict[str, object]] = []
    training_row_evidence: list[dict[str, object]] = []
    test_raw_sha256 = (
        dataframe_value_sha256(test[plan.raw_columns()])
        if cfg.training_rows is not None
        else None
    )

    # fold 안의 fold-fit 변환 fit도 training 단계에 포함한다. (#40)
    if recorder is not None:
        recorder.stage("training")
    for fold, checkpoint in enumerate(checkpoints):
        va_idx = train.index[train["fold"] == fold]
        tr_idx = train.index[train["fold"] != fold]
        row_batch = None
        row_evidence = None
        if cfg.training_rows is not None:
            row_batch = build_training_rows(
                train.loc[tr_idx],
                plan.raw_columns(),
                cfg.training_rows,
                seed=seed,
                outer_fold=fold,
            )
            row_evidence = dict(row_batch.evidence)
            row_evidence.update(
                {
                    "validation_augmented": False,
                    "test_augmented": False,
                    "validation_raw_sha256": dataframe_value_sha256(
                        train.loc[va_idx, plan.raw_columns()]
                    ),
                    "test_raw_sha256": test_raw_sha256,
                    "placebo_canary": {
                        "feature": PLACEBO,
                        "mask_source": PLACEBO_MASK_SOURCE,
                        "parent_noise_is_shared_by_replica": True,
                        "missing_mask_rebuilt_from_augmented_raw_source": True,
                    },
                }
            )
            training_row_evidence.append(row_evidence)
        if checkpoint is not None:
            fold_feature_reuse_records.extend(
                plan.unused_fold_fit_reuse_evidence(
                    seed=seed, fold=fold, reason="checkpoint_reused"
                )
            )
            skipped_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_feature",
                actor_kind="pipeline",
                actor_name="feature_plan",
                reason="checkpoint_reused",
            )
            for kind, _ in provider_declarations:
                skipped_operation(
                    recorder,
                    seed=seed,
                    fold=fold,
                    operation="fold_feature.provider_fit",
                    actor_kind="column_provider",
                    actor_name=kind,
                    reason="checkpoint_reused",
                )
                for dataset in ("train", "test"):
                    skipped_operation(
                        recorder,
                        seed=seed,
                        fold=fold,
                        operation="fold_feature.provider_transform",
                        actor_kind="column_provider",
                        actor_name=kind,
                        dataset=dataset,
                        reason="checkpoint_reused",
                    )
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

        assert X is not None and X_test is not None
        providers = plan.new_fold_fit_providers()
        if uses_replicas:
            assert row_batch is not None and row_evidence is not None
            X_training = plan.build_matrix(row_batch.frame, seed)
            parent_placebo = X.loc[row_batch.parent_source_index, PLACEBO].to_numpy(
                copy=True
            )
            augmented_source_missing = row_batch.frame[
                PLACEBO_MASK_SOURCE
            ].isna().to_numpy()
            parent_placebo[augmented_source_missing] = np.nan
            X_training[PLACEBO] = parent_placebo
            placebo_mask_matches = np.array_equal(
                X_training[PLACEBO].isna().to_numpy(), augmented_source_missing
            )
            if not placebo_mask_matches:
                raise AssertionError("플라시보 결측 마스크가 증강 원자료와 다르다.")
            row_evidence["placebo_canary"][
                "actual_missing_mask_matches_source"
            ] = True
            validation_frame = validation_context(train.loc[va_idx])
            fold_frame = pd.concat(
                [row_batch.frame, validation_frame], ignore_index=True
            )
            X_fold = pd.concat(
                [X_training, X.loc[va_idx].reset_index(drop=True)],
                ignore_index=True,
            )
            fold_train_ff = (
                prepare_fold_fit_input(fold_frame, X_fold) if providers else None
            )
            model_training_index = row_batch.training_index
            state_fit_index = row_batch.state_fit_index
            model_validation_index = pd.RangeIndex(len(row_batch.frame), len(fold_frame))
            model_training_target = row_batch.frame[TARGET]
            model_validation_target = y.loc[va_idx].reset_index(drop=True)
            model_validation_target.index = model_validation_index
        else:
            X_fold = X
            fold_train_ff = train_ff
            model_training_index = tr_idx
            state_fit_index = tr_idx
            model_validation_index = va_idx
            model_training_target = y.loc[tr_idx]
            model_validation_target = y.loc[va_idx]
        X_test_fold = X_test
        with timed_operation(
            recorder,
            seed=seed,
            fold=fold,
            operation="fold_feature",
            actor_kind="pipeline",
            actor_name="feature_plan",
        ):
            if providers:
                assert fold_train_ff is not None and test_ff is not None
                # fold-fit 단계: 학습 fold로만 fit하고, 같은 상태를 검증 fold와 test에 적용한다.
                # 전체 train으로 fit하는 별도 경로는 없다. (#32 결정 4)
                # transform은 학습 fold 행과 검증 fold 행이 섞인 train 전체를 받는다.
                for kind, transformer in providers:
                    train_values, test_values, reuse_evidence = (
                        plan.materialize_fold_fit_provider(
                            kind=kind,
                            transformer=transformer,
                            train_input=fold_train_ff,
                            test_input=test_ff,
                            training_index=state_fit_index,
                            validation_index=model_validation_index,
                            seed=seed,
                            fold=fold,
                            recorder=recorder,
                        )
                    )
                    fold_feature_reuse_records.append(reuse_evidence)
                    collision = set(train_values.columns) & set(X_fold.columns)
                    if collision:
                        raise AssertionError(
                            f"fold-fit 컬럼 이름 충돌: {sorted(collision)}"
                        )
                    X_fold = pd.concat([X_fold, train_values], axis=1)
                    X_test_fold = pd.concat([X_test_fold, test_values], axis=1)
                assert list(X_fold.columns) == list(X_test_fold.columns), (
                    "train/test의 fold-fit 컬럼 집합이 다르다."
                )
            assert list(X_fold.columns) == feature_names, (
                f"fold {fold}의 컬럼 집합이 피처 계획 선언과 다르다."
            )
            assert list(X_test_fold.columns) == feature_names, (
                f"fold {fold}의 test 컬럼 집합이 피처 계획 선언과 다르다."
            )

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
                        X_fold.loc[model_training_index],
                        model_training_target,
                        X_fold.loc[model_validation_index],
                        model_validation_target,
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
                # 반복형 계열은 원시 선택값과 관측 학습 길이를 계열과 무관한 같은
                # 형식으로 남긴다. 시드와 바깥쪽 분할 좌표는 여기서만 안다. (#372)
                declaration = model_mod.collect_training_length_declaration(
                    adapter, cfg.model.kind
                )
                training_length_evidence = (
                    None
                    if declaration is None
                    else training_length_mod.observe_declaration(
                        declaration, seed=seed, outer_fold=fold
                    ).to_json()
                )
                fold_training_diagnostics = (
                    {
                        "model_kind": cfg.model.kind,
                        "seed": seed,
                        "fold": fold,
                        "details": adapter_training_diagnostics,
                        "training_length_evidence": training_length_evidence,
                    }
                    if adapter_training_diagnostics is not None
                    or training_length_evidence is not None
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
        fold_feature_reuse_evidence=fold_feature_reuse_records,
        model_training_diagnostics=model_training_diagnostics,
        training_row_evidence=training_row_evidence,
    )
