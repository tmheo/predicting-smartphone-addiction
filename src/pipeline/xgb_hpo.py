"""XGBoost 모델 설정값 탐색 진입점. (이슈 288)"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data, model as model_mod, tracking
from .config import ExperimentConfig, ModelConfig, load_config
from .ledger import Pool
from .plan import FeaturePlan, prepare_fold_fit_input
from .runs import MlflowRunStore, RunStore

BASE_CONFIG = Path("configs/exp045_xgb_depth8.yaml")
OUTPUT_PATH = Path("artifacts/hpo/issue-288-xgb-search.json")
POOL_PATH = Path("artifacts/pool.yaml")
TRIALS = 50
SEED = 42
VALID_FOLD = 0


@dataclass(frozen=True)
class PoolPrediction:
    run_id: str
    config: str
    values: pd.Series


@dataclass(frozen=True)
class NearestPoolMember:
    run_id: str
    config: str
    spearman: float


@dataclass(frozen=True)
class TrialEvaluation:
    fold0_auc: float
    nearest: NearestPoolMember
    training_seconds: float
    best_iteration: int


@dataclass
class PreparedSearch:
    """한 번 만든 fold 0 피처 행렬과 후보 풀 예측을 모든 시행이 공유한다."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    validation_ids: pd.Index
    pool_predictions: list[PoolPrediction]
    adapter_factory: Callable[[ModelConfig, int], object]

    def evaluate(self, config: ModelConfig) -> TrialEvaluation:
        adapter = self.adapter_factory(config, SEED)
        started = time.monotonic()
        predictions = np.asarray(
            adapter.fit(self.X_train, self.y_train, self.X_valid, self.y_valid),
            dtype="float64",
        )
        training_seconds = time.monotonic() - started
        if predictions.shape != (len(self.validation_ids),):
            raise ValueError(
                f"fold 0 예측 크기가 다르다: {predictions.shape} != "
                f"({len(self.validation_ids)},)"
            )
        candidate = pd.Series(predictions, index=self.validation_ids, name="pred")
        diagnostics = model_mod.collect_training_diagnostics(adapter)
        if diagnostics is None or "best_iteration" not in diagnostics:
            raise ValueError("XGBoost 최적 반복 수를 기록하지 못했다.")
        return TrialEvaluation(
            fold0_auc=float(roc_auc_score(self.y_valid, predictions)),
            nearest=nearest_pool_member(candidate, self.pool_predictions),
            training_seconds=training_seconds,
            best_iteration=int(diagnostics["best_iteration"]),
        )


def nearest_pool_member(
    candidate: pd.Series, pool: list[PoolPrediction]
) -> NearestPoolMember:
    """검증 예측과 순위가 가장 비슷한 후보 풀 구성원을 찾는다."""
    if not pool:
        raise ValueError("후보 풀이 비어 있다.")
    nearest: NearestPoolMember | None = None
    for member in pool:
        if not member.values.index.equals(candidate.index):
            raise ValueError(f"후보 풀 구성원 {member.run_id}의 id 순서가 다르다.")
        spearman = float(candidate.corr(member.values, method="spearman"))
        if pd.isna(spearman):
            raise ValueError(f"후보 풀 구성원 {member.run_id}과의 상관을 계산할 수 없다.")
        if nearest is None or spearman > nearest.spearman:
            nearest = NearestPoolMember(member.run_id, member.config, spearman)
    assert nearest is not None
    return nearest


def load_pool_predictions(
    pool: Pool, store: RunStore, validation_ids: pd.Index
) -> list[PoolPrediction]:
    """후보 풀의 시드 평균 OOF를 검증 fold id 순서로 읽는다."""
    predictions: list[PoolPrediction] = []
    for member in pool.members:
        values = store.oof_of(member.run_id).reindex(validation_ids)
        if values.isna().any():
            raise ValueError(
                f"후보 풀 구성원 {member.run_id}의 OOF에 fold 0 id가 빠져 있다."
            )
        predictions.append(PoolPrediction(member.run_id, member.config, values))
    if not predictions:
        raise ValueError("후보 풀이 비어 있다.")
    return predictions


def prepare_search_data(
    config: ExperimentConfig, pool: Pool, store: RunStore
) -> PreparedSearch:
    """기준 피처 계획을 fold 0 분할에 한 번 맞춰 모든 시행이 공유하게 한다."""
    train = data.load_csv(config.data.train)
    test = data.load_csv(config.data.test)
    data.align_categories(train, test, config.features.categorical)
    plan = FeaturePlan.from_config(config.features)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, config.data.folds)

    X = plan.build_matrix(train, SEED)
    validation_mask = train["fold"] == VALID_FOLD
    va_idx = train.index[validation_mask]
    tr_idx = train.index[~validation_mask]
    if len(va_idx) == 0 or len(tr_idx) == 0:
        raise ValueError("fold 0 학습·검증 분할을 만들 수 없다.")
    transformers = plan.fold_fit_transformers()
    if transformers:
        train_ff = prepare_fold_fit_input(train, X)
        for transformer in transformers:
            transformer.fit(train_ff.loc[tr_idx], SEED)
        X = plan.add_fold_fit_columns(X, train_ff)
    if list(X.columns) != plan.all_columns():
        raise ValueError("탐색 피처 행렬이 기준 피처 계획과 다르다.")

    validation_ids = pd.Index(train.loc[va_idx, data.ID], name=data.ID)
    if validation_ids.has_duplicates:
        raise ValueError("fold 0 검증 id에 중복이 있다.")
    return PreparedSearch(
        X_train=X.loc[tr_idx],
        y_train=train.loc[tr_idx, data.TARGET],
        X_valid=X.loc[va_idx],
        y_valid=train.loc[va_idx, data.TARGET],
        validation_ids=validation_ids,
        pool_predictions=load_pool_predictions(pool, store, validation_ids),
        adapter_factory=model_mod.create,
    )


def model_config_for_trial(trial: optuna.trial.BaseTrial) -> ModelConfig:
    """이슈 288의 고정값과 탐색 공간에서 XGBoost 설정 하나를 만든다."""
    return ModelConfig(
        kind="xgboost",
        params={
            "tree_method": "hist",
            "eval_metric": "auc",
            "n_estimators": 10000,
            "learning_rate": trial.suggest_float(
                "learning_rate", 5e-3, 5e-2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 4, 10, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 5.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.6, 1.0
            ),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 5.0, log=True
            ),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "max_cat_to_onehot": trial.suggest_int(
                "max_cat_to_onehot", 8, 100, log=True
            ),
            "max_leaves": trial.suggest_int("max_leaves", 8, 1024, log=True),
        },
        fit={"early_stopping_rounds": 200},
    )


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, object]:
    record: dict[str, object] = {
        "number": trial.number,
        "state": trial.state.name,
        "model_params": trial.user_attrs.get("model_params"),
    }
    if trial.state == optuna.trial.TrialState.COMPLETE:
        record.update(
            {
                "fold0_auc": float(trial.value),
                "nearest_pool_member": trial.user_attrs["nearest_pool_member"],
                "training_seconds": trial.user_attrs["training_seconds"],
                "best_iteration": trial.user_attrs["best_iteration"],
            }
        )
    return record


def _write_artifact(
    study: optuna.Study,
    output_path: Path,
    n_trials: int,
    context: dict[str, object] | None,
) -> None:
    records = [_trial_record(trial) for trial in study.trials]
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    top_two = sorted(complete, key=lambda trial: (-float(trial.value), trial.number))[:2]
    artifact = {
        "schema_version": 1,
        "issue": 288,
        "objective": "fold_0_auc",
        "sampler": {"name": "TPESampler", "multivariate": True, "seed": 42},
        "trials_requested": n_trials,
        "trials": records,
        "top_two_trial_numbers": [trial.number for trial in top_two],
    }
    if context is not None:
        artifact["context"] = context
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )


def run_search(
    evaluate: Callable[[ModelConfig], TrialEvaluation],
    *,
    n_trials: int,
    output_path: Path,
    context: dict[str, object] | None = None,
) -> optuna.Study:
    """고정 TPE 탐색을 실행하고 각 시행 뒤 기계 판독 결과를 갱신한다."""
    if n_trials < 1:
        raise ValueError("탐색 시행 수는 1 이상이어야 한다.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        config = model_config_for_trial(trial)
        trial.set_user_attr("model_params", config.params)
        result = evaluate(config)
        trial.set_user_attr("nearest_pool_member", asdict(result.nearest))
        trial.set_user_attr("training_seconds", result.training_seconds)
        trial.set_user_attr("best_iteration", result.best_iteration)
        return result.fold0_auc

    def persist(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        _write_artifact(study, output_path, n_trials, context)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(
                f"trial {trial.number + 1}/{n_trials}: "
                f"fold0_auc={float(trial.value):.10f} "
                f"nearest={trial.user_attrs['nearest_pool_member']['config']} "
                f"rho={trial.user_attrs['nearest_pool_member']['spearman']:.10f} "
                f"best_iteration={trial.user_attrs['best_iteration']} "
                f"training={trial.user_attrs['training_seconds']:.1f}s"
            )

    study.optimize(objective, n_trials=n_trials, callbacks=[persist])
    return study


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="이슈 288 XGBoost 설정값 탐색")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--pool-ledger", type=Path, default=POOL_PATH)
    parser.add_argument("--pool-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--plan", action="store_true", help="계산 없이 고정 계약을 출력")
    args = parser.parse_args(argv)

    if args.plan:
        print(f"base config : {args.base_config}")
        print(f"validation  : fold {VALID_FOLD}")
        print(f"trials      : {args.trials}")
        print("sampler     : TPESampler(multivariate=True, seed=42)")
        print("tree_method : hist (local CPU)")
        print(f"output      : {args.output}")
        return

    config = load_config(args.base_config, "screen")
    canonical = load_config(BASE_CONFIG, "screen")
    if config.features != canonical.features:
        raise SystemExit(
            f"기준 피처 계획이 {BASE_CONFIG}와 다르다. 이슈 288 탐색을 중단한다."
        )
    if config.initial_score is not None:
        raise SystemExit("이슈 288 기준 설정은 초기 점수를 쓰지 않는다.")
    pool = Pool.load(args.pool_ledger)
    store = MlflowRunStore(args.pool_tracking_uri)
    print("공유 fold 0 피처 행렬과 후보 풀 예측을 준비한다.")
    prepared = prepare_search_data(config, pool, store)
    context = {
        "base_config": str(args.base_config),
        "sha256_base_config": data.file_sha256(args.base_config),
        "sha256_train": data.file_sha256(config.data.train),
        "sha256_test": data.file_sha256(config.data.test),
        "sha256_folds": data.file_sha256(config.data.folds),
        "sha256_pool_ledger": data.file_sha256(args.pool_ledger),
        "pool_run_ids": [member.run_id for member in pool.members],
        "validation_fold": VALID_FOLD,
        "seed": SEED,
        "feature_names": list(prepared.X_train.columns),
        "git": tracking.git_state(),
    }
    print(
        f"피처 행렬 준비 완료: train={prepared.X_train.shape} "
        f"valid={prepared.X_valid.shape}, pool={len(prepared.pool_predictions)}"
    )
    study = run_search(
        prepared.evaluate,
        n_trials=args.trials,
        output_path=args.output,
        context=context,
    )
    print(
        f"탐색 완료: best_trial={study.best_trial.number} "
        f"fold0_auc={study.best_value:.10f} output={args.output}"
    )


if __name__ == "__main__":
    main()
