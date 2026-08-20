"""CatBoost GPU fold 0 model-setting search for issue 289.

The search keeps the feature plan from ``exp070_cat_exact_cats`` fixed, prepares
its fold 0 matrix once, and reuses that matrix for every Optuna trial.  It also
exports the current candidate-pool predictions from the local MLflow ledger so
the remote job can record the nearest pool member for every trial without
needing access to ``mlflow.db``.

Usage from the main worktree, where the complete MLflow ledger lives::

    uv run python -m pipeline.catboost_hpo export-pool \
        --tracking-uri sqlite:////absolute/path/to/mlflow.db \
        --output run-logs/issue-289/pool-fold0.parquet

Usage on a CUDA worker::

    uv run python -m pipeline.catboost_hpo run \
        configs/exp070_cat_exact_cats.yaml \
        --pool-oof run-logs/issue-289/pool-fold0.parquet \
        --output-dir /workspace/issue-289/results/hpo

The search-space exception is intentional and recorded in issue 289.
CatBoost calls ``colsample_bylevel`` ``rsm`` and supports it on GPU only for
pairwise ranking.  This binary-classification search therefore keeps it at the
GPU default 1.0 instead of passing an unsupported parameter.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data
from . import model as model_mod
from . import tracking
from .config import ModelConfig, load_config
from .data import ID, TARGET
from .ledger import POOL_PATH, Pool
from .plan import FeaturePlan, prepare_fold_fit_input
from .runs import MlflowRunStore, RunStore

ISSUE = 289
SEED = 42
VALID_FOLD = 0
MAX_TRIALS = 50
CHECKPOINT_TRIALS = 25
MIN_CHECKPOINT_IMPROVEMENT = 0.0002
BASELINE_FOLD0_AUC = 0.967796533926754
MIN_GPU_MEMORY_MIB = 24 * 1024
STUDY_NAME = "issue-289-catboost-gpu"
SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sample_model_params(trial: optuna.trial.BaseTrial, device: str) -> dict[str, Any]:
    """Return the exact issue-289 GPU model settings for one trial."""
    sampled = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 5e-2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise"]
        ),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 5.0, log=True),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 20, log=True
        ),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 8, 100, log=True),
        "model_size_reg": trial.suggest_float("model_size_reg", 0.1, 1.5, log=True),
    }
    return {
        "iterations": 10000,
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": device,
        "max_ctr_complexity": 1,
        "boosting_type": "Plain",
        "max_bin": 254,
        "bootstrap_type": "Bernoulli",
        # colsample_bylevel is deliberately omitted.  The GPU default is 1.0.
        **sampled,
    }


def checkpoint_continues(
    best_auc: float,
    baseline_auc: float = BASELINE_FOLD0_AUC,
    minimum_improvement: float = MIN_CHECKPOINT_IMPROVEMENT,
) -> bool:
    """Return whether trials 26 through 50 are authorized."""
    return best_auc >= baseline_auc + minimum_improvement


def promoted_trial_numbers(
    trials: list[optuna.trial.FrozenTrial], continued_after_checkpoint: bool
) -> list[int]:
    """Choose the top two trials after 50, or the winner after an early stop."""
    completed = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        raise ValueError("완료된 탐색 시행이 없다.")
    completed.sort(key=lambda trial: (-float(trial.value), trial.number))
    count = 2 if continued_after_checkpoint else 1
    return [trial.number for trial in completed[:count]]


@dataclass(frozen=True)
class PoolRanks:
    members: tuple[str, ...]
    ranks: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> PoolRanks:
        if frame.empty or not len(frame.columns):
            raise ValueError("후보 풀 예측이 비어 있다.")
        if frame.isna().any().any():
            raise ValueError("후보 풀 예측에 결측값이 있다.")
        ranks = frame.rank(method="average").to_numpy(dtype="float64")
        return cls(members=tuple(map(str, frame.columns)), ranks=ranks)

    def nearest(self, prediction: np.ndarray) -> tuple[str, float]:
        candidate = pd.Series(np.asarray(prediction, dtype="float64")).rank(
            method="average"
        )
        if len(candidate) != len(self.ranks):
            raise ValueError("후보 예측과 후보 풀 예측의 행 수가 다르다.")
        correlations = np.asarray(
            [
                np.corrcoef(candidate.to_numpy(), self.ranks[:, index])[0, 1]
                for index in range(self.ranks.shape[1])
            ],
            dtype="float64",
        )
        if not np.isfinite(correlations).all():
            raise ValueError("후보 풀 스피어만 상관이 유한하지 않다.")
        best = int(correlations.argmax())
        return self.members[best], float(correlations[best])


def load_pool_ranks(path: Path, validation_ids: pd.Series) -> PoolRanks:
    frame = pd.read_parquet(path)
    if ID not in frame.columns:
        raise ValueError(f"후보 풀 예측에 {ID} 열이 없다.")
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError("후보 풀 예측 id가 비어 있거나 중복됐다.")
    expected = pd.Index(validation_ids.to_numpy())
    actual = pd.Index(frame[ID].to_numpy())
    if set(actual) != set(expected):
        raise ValueError("후보 풀 예측 id 집합이 fold 0 검증 id와 다르다.")
    aligned = frame.set_index(ID).reindex(expected)
    return PoolRanks.from_frame(aligned)


def export_pool_fold_predictions(
    pool: Pool,
    store: RunStore,
    folds: pd.DataFrame,
    output: Path,
    fold: int = VALID_FOLD,
) -> pd.DataFrame:
    """Export one fold of every committed pool member as a wide Parquet file."""
    required = {ID, "fold"}
    if not required.issubset(folds.columns):
        raise ValueError(f"fold 파일에 필요한 열이 없다: {sorted(required - set(folds))}")
    validation_ids = folds.loc[folds["fold"] == fold, ID]
    if validation_ids.empty or validation_ids.isna().any() or not validation_ids.is_unique:
        raise ValueError(f"fold {fold} id가 비어 있거나 중복됐다.")
    exported = pd.DataFrame({ID: validation_ids.to_numpy()})
    for member in pool.members:
        prediction = store.oof_of(member.run_id).reindex(validation_ids.to_numpy())
        if prediction.isna().any():
            raise ValueError(f"후보 풀 구성원 {member.run_id}의 fold {fold} 예측이 불완전하다.")
        label = f"{member.config}::{member.run_id}"
        if label in exported:
            raise ValueError(f"후보 풀 예측 열 이름이 중복됐다: {label}")
        exported[label] = prediction.to_numpy(dtype="float64")
    if len(exported.columns) == 1:
        raise ValueError("후보 풀 구성원이 없다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.to_parquet(output, index=False)
    return exported


def parse_nvidia_smi_memory(output: str) -> tuple[int, int]:
    """Parse ``memory.used,memory.total`` output in MiB."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"nvidia-smi 메모리 응답이 한 줄이 아니다: {lines}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 2:
        raise ValueError(f"nvidia-smi 메모리 응답 열이 두 개가 아니다: {fields}")
    used, total = (int(field) for field in fields)
    if used < 0 or total <= 0 or used > total:
        raise ValueError(f"nvidia-smi 메모리 값이 잘못됐다: used={used}, total={total}")
    return used, total


class MemoryQuery(Protocol):
    def __call__(self) -> tuple[int, int]: ...


def nvidia_smi_query(device: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            device,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return parse_nvidia_smi_memory(result.stdout)


class GpuMemorySampler:
    """Poll device memory while a single trial trains."""

    def __init__(self, query: MemoryQuery, interval_seconds: float = 0.5) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU 메모리 확인 간격은 양수여야 한다.")
        self._query = query
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.start_mib = 0
        self.peak_mib = 0
        self.total_mib = 0

    def __enter__(self) -> GpuMemorySampler:
        self.start_mib, self.total_mib = self._query()
        self.peak_mib = self.start_mib
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                used, total = self._query()
                if total != self.total_mib:
                    raise RuntimeError(
                        f"GPU 전체 메모리가 시행 중 바뀌었다: {total} != {self.total_mib}"
                    )
                self.peak_mib = max(self.peak_mib, used)
            except BaseException as exc:  # noqa: BLE001 - 전달할 표본 수집 실패를 보존한다.
                self._error = exc
                self._stop.set()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._interval * 4))
        if exc is None and self._error is not None:
            raise RuntimeError("GPU 메모리 표본 수집에 실패했다.") from self._error
        return False


@dataclass(frozen=True)
class SearchData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    validation_ids: pd.Series
    feature_names: tuple[str, ...]
    preparation_seconds: float


def prepare_search_data(config_path: Path) -> tuple[Any, SearchData]:
    started = time.monotonic()
    cfg = load_config(config_path, "screen")
    if cfg.name != "exp070_cat_exact_cats" or cfg.model.kind != "catboost":
        raise ValueError("기준 설정은 exp070_cat_exact_cats CatBoost여야 한다.")
    plan = FeaturePlan.from_config(cfg.features)
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)
    train, _ = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)
    X = plan.build_matrix(train, SEED)
    y = train[TARGET]
    fold_input = prepare_fold_fit_input(train, X)
    validation_index = train.index[train["fold"] == VALID_FOLD]
    training_index = train.index[train["fold"] != VALID_FOLD]
    for transformer in plan.fold_fit_transformers():
        transformer.fit(fold_input.loc[training_index], SEED)
    X_fold = plan.add_fold_fit_columns(X, fold_input)
    if list(X_fold.columns) != plan.all_columns():
        raise ValueError("탐색 피처 행렬이 exp070 선언과 다르다.")
    prepared = SearchData(
        X_train=X_fold.loc[training_index],
        y_train=y.loc[training_index],
        X_valid=X_fold.loc[validation_index],
        y_valid=y.loc[validation_index],
        validation_ids=train.loc[validation_index, ID],
        feature_names=tuple(X_fold.columns),
        preparation_seconds=time.monotonic() - started,
    )
    return cfg, prepared


def _best_iteration(adapter: Any) -> int:
    model = getattr(adapter, "_model", None)
    getter = getattr(model, "get_best_iteration", None)
    if not callable(getter):
        raise RuntimeError("CatBoost 최적 반복 수를 읽을 수 없다.")
    value = int(getter())
    if value < 0:
        raise RuntimeError(f"CatBoost 최적 반복 수가 잘못됐다: {value}")
    return value


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "fold0_auc": float(trial.value) if trial.value is not None else None,
        "sampled_params": dict(trial.params),
        **dict(trial.user_attrs),
    }


def write_trials_snapshot(study: optuna.Study, output_dir: Path) -> None:
    _atomic_json(
        output_dir / "trials.json",
        {
            "schema_version": SCHEMA_VERSION,
            "issue": ISSUE,
            "study_name": study.study_name,
            "direction": study.direction.name,
            "trials": [_trial_record(trial) for trial in study.trials],
        },
    )


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]


def _write_identity(path: Path, identity: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise ValueError("기존 탐색 디렉터리의 실행 정체성이 현재 입력과 다르다.")
        return
    _atomic_json(path, identity)


def _search_identity(config_path: Path, pool_oof: Path, cfg: Any) -> dict[str, Any]:
    git = tracking.git_state()
    if git["git_dirty"] != "False":
        raise ValueError("CatBoost 탐색은 깨끗한 커밋에서만 실행한다.")
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "git_commit": git["git_commit"],
        "base_config": str(config_path),
        "base_config_sha256": sha256(config_path),
        "train_sha256": data.file_sha256(cfg.data.train),
        "test_sha256": data.file_sha256(cfg.data.test),
        "folds_sha256": data.file_sha256(cfg.data.folds),
        "pool_oof_sha256": sha256(pool_oof),
        "seed": SEED,
        "valid_fold": VALID_FOLD,
        "baseline_fold0_auc": BASELINE_FOLD0_AUC,
        "checkpoint_trials": CHECKPOINT_TRIALS,
        "max_trials": MAX_TRIALS,
        "minimum_checkpoint_improvement": MIN_CHECKPOINT_IMPROVEMENT,
        "sampler": {"kind": "TPESampler", "multivariate": True, "seed": SEED},
        "colsample_bylevel": {
            "value": 1.0,
            "reason": "CatBoost GPU binary classification does not support rsm",
        },
    }


def run_search(
    config_path: Path,
    pool_oof: Path,
    output_dir: Path,
    device: str,
    memory_interval_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, search_data = prepare_search_data(config_path)
    identity = _search_identity(config_path, pool_oof, cfg)
    _write_identity(output_dir / "run-identity.json", identity)

    pool_ranks = load_pool_ranks(pool_oof, search_data.validation_ids)
    used_mib, total_mib = nvidia_smi_query(device)
    if total_mib < MIN_GPU_MEMORY_MIB:
        raise RuntimeError(
            f"GPU 메모리가 24GiB보다 작다: {total_mib}MiB < {MIN_GPU_MEMORY_MIB}MiB"
        )
    print(
        f"[gpu] device={device} memory_used={used_mib}MiB "
        f"memory_total={total_mib}MiB"
    )

    storage_path = (output_dir / "study.db").resolve()
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    predictions_dir = output_dir / "trial-predictions"
    predictions_dir.mkdir(exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = sample_model_params(trial, device)
        model_cfg = ModelConfig(
            kind="catboost",
            params=params,
            fit={"early_stopping_rounds": 200},
        )
        adapter = model_mod.create(model_cfg, SEED)
        query = lambda: nvidia_smi_query(device)
        started = time.monotonic()
        with GpuMemorySampler(query, memory_interval_seconds) as memory:
            prediction = np.asarray(
                adapter.fit(
                    search_data.X_train,
                    search_data.y_train,
                    search_data.X_valid,
                    search_data.y_valid,
                ),
                dtype="float64",
            )
        training_seconds = time.monotonic() - started
        auc = float(roc_auc_score(search_data.y_valid, prediction))
        nearest_member, nearest_spearman = pool_ranks.nearest(prediction)
        prediction_path = predictions_dir / f"trial-{trial.number:04d}.parquet"
        pd.DataFrame(
            {
                ID: search_data.validation_ids.to_numpy(),
                TARGET: search_data.y_valid.to_numpy(),
                "pred": prediction,
            }
        ).to_parquet(prediction_path, index=False)
        trial.set_user_attr("model_params", params)
        trial.set_user_attr("training_seconds", training_seconds)
        trial.set_user_attr("best_iteration", _best_iteration(adapter))
        trial.set_user_attr("nearest_pool_member", nearest_member)
        trial.set_user_attr("nearest_pool_spearman", nearest_spearman)
        trial.set_user_attr("gpu_memory_start_mib", memory.start_mib)
        trial.set_user_attr("gpu_peak_memory_mib", memory.peak_mib)
        trial.set_user_attr("gpu_total_memory_mib", memory.total_mib)
        trial.set_user_attr("prediction_file", str(prediction_path.relative_to(output_dir)))
        trial.set_user_attr("prediction_sha256", sha256(prediction_path))
        print(
            f"[trial {trial.number}] auc={auc:.10f} "
            f"nearest={nearest_member} rho={nearest_spearman:.8f} "
            f"best_iter={_best_iteration(adapter)} time={training_seconds:.1f}s "
            f"gpu_peak={memory.peak_mib}MiB"
        )
        return auc

    def snapshot_callback(
        completed_study: optuna.Study, _: optuna.trial.FrozenTrial
    ) -> None:
        write_trials_snapshot(completed_study, output_dir)

    while len(_completed_trials(study)) < CHECKPOINT_TRIALS:
        study.optimize(objective, n_trials=1, callbacks=[snapshot_callback])

    completed = _completed_trials(study)
    checkpoint_best = max(float(trial.value) for trial in completed[:CHECKPOINT_TRIALS])
    continued = checkpoint_continues(checkpoint_best)
    if continued:
        while len(_completed_trials(study)) < MAX_TRIALS:
            study.optimize(objective, n_trials=1, callbacks=[snapshot_callback])

    write_trials_snapshot(study, output_dir)
    promoted = promoted_trial_numbers(study.trials, continued)
    by_number = {trial.number: trial for trial in study.trials}
    promoted_records = [_trial_record(by_number[number]) for number in promoted]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "status": "completed" if continued else "stopped_after_checkpoint",
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "identity": identity,
        "feature_preparation_seconds": search_data.preparation_seconds,
        "feature_count": len(search_data.feature_names),
        "feature_names_sha256": hashlib.sha256(
            "\n".join(search_data.feature_names).encode()
        ).hexdigest(),
        "completed_trials": len(_completed_trials(study)),
        "checkpoint_best_auc": checkpoint_best,
        "checkpoint_delta_vs_baseline": checkpoint_best - BASELINE_FOLD0_AUC,
        "continued_after_checkpoint": continued,
        "promoted_trials": promoted_records,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _export_pool(args: argparse.Namespace) -> None:
    pool = Pool.load(args.pool)
    store = MlflowRunStore(tracking_uri=args.tracking_uri)
    folds = pd.read_parquet(args.folds)
    exported = export_pool_fold_predictions(pool, store, folds, args.output)
    print(
        f"[pool-export] rows={len(exported)} members={len(exported.columns) - 1} "
        f"sha256={sha256(args.output)} output={args.output}"
    )


def _repository_root(config_path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(config_path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def _run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    pool_oof = args.pool_oof.resolve()
    output_dir = args.output_dir.resolve()
    original_directory = Path.cwd()
    try:
        os.chdir(_repository_root(config_path))
        summary = run_search(
            config_path=config_path,
            pool_oof=pool_oof,
            output_dir=output_dir,
            device=args.device,
            memory_interval_seconds=args.memory_interval_seconds,
        )
    finally:
        os.chdir(original_directory)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="이슈 289 CatBoost GPU 설정값 탐색")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export-pool", help="현재 후보 풀의 fold 0 OOF를 원격 입력으로 내보낸다."
    )
    export.add_argument("--pool", type=Path, default=POOL_PATH)
    export.add_argument("--folds", type=Path, default=Path("artifacts/folds.parquet"))
    export.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=_export_pool)

    run = subparsers.add_parser("run", help="고정 fold 0에서 GPU 탐색을 실행한다.")
    run.add_argument("config", type=Path)
    run.add_argument("--pool-oof", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="0")
    run.add_argument("--memory-interval-seconds", type=float, default=0.5)
    run.set_defaults(func=_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
