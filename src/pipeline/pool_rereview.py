"""사전 고정한 35개 후보 풀 재심사 실행기. (이슈 339)

이 모듈은 ``artifacts/pool-rereview-precommit-2026-08-22.yaml``의 계약을
기계적으로 확인한 뒤 다음 순서만 실행한다.

1. 두 척도의 정확 복제 영점 대조 210건을 끝내고 하한을 한 번 확정한다.
2. 바깥쪽 검증 분할 다섯 개와 전체 OOF 최종 실행을 차례로 돈다.
3. 결정 산출물, 실행 계측, 중간 저장과 파일별 SHA-256 목록을 분리해 남긴다.

후보, 전략, 난수, 문턱과 순서는 명령줄 인자로 바꿀 수 없다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from . import ensemble as ensemble_module
from .data import ID, TARGET
from .pool_audit import prediction_array_sha256
from .runs import MlflowRunStore


REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMMIT_PATH = REPO_ROOT / "artifacts/pool-rereview-precommit-2026-08-22.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "run-logs/pool-rereview"
DEFAULT_STAGED_PREDICTIONS = REPO_ROOT / "run-logs/pool-rereview-input/predictions.parquet"
EXECUTION_SCHEMA_VERSION = 1
APPROXIMATION_NOTE = (
    "성능 동등 하한은 35개 근방 풀에서 측정했으며, 많이 축소된 작은 작업 풀에 "
    "같은 하한을 적용하는 것은 사전 고정한 근사다."
)


class PoolRereviewError(RuntimeError):
    """재심사를 시작하거나 계속할 수 없는 계약 위반."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoolRereviewError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_write(path, _json_bytes(payload))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"키-값 구조가 아닌 YAML이다: {path}")
    return payload


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _runtime_identity() -> dict[str, Any]:
    status = _git("--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=normal")
    _require(not status, "실행 코드가 커밋되지 않았거나 작업 폴더가 깨끗하지 않다.")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": True,
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "parallelism_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def _source_hashes(ledger: dict[str, Any], *, require_private: bool) -> dict[str, str]:
    actual: dict[str, str] = {}
    private = {"train", "test", "external_original"}
    for name, source in ledger["sources"].items():
        source_path = REPO_ROOT / source["path"]
        if not source_path.is_file():
            if name in private and not require_private:
                continue
            raise PoolRereviewError(f"동결 출처 파일이 없다: {source['path']}")
        digest = _sha256(source_path)
        _require(
            digest == source["sha256"],
            f"동결 출처 내용 해시 불일치: {source['path']} ({digest})",
        )
        actual[name] = digest
    return actual


@dataclass(frozen=True)
class InputContext:
    predictions: pd.DataFrame
    labels: pd.Series
    folds: pd.Series
    missingness_bands: pd.Series
    ledger: dict[str, Any]
    baseline: dict[str, Any]
    source_hashes: dict[str, str]
    prediction_file_sha256: str
    member_prediction_sha256: dict[str, str]

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(self.ledger["candidate_pool"]["members"])

    def identity(self, execution_code_sha256: str, runtime: dict[str, Any]) -> dict[str, Any]:
        return {
            "precommit_path": str(PRECOMMIT_PATH.relative_to(REPO_ROOT)),
            "precommit_sha256": _sha256(PRECOMMIT_PATH),
            "sources": self.source_hashes,
            "prediction_file_sha256": self.prediction_file_sha256,
            "member_prediction_sha256": self.member_prediction_sha256,
            "execution_code_sha256": execution_code_sha256,
            "git_commit": runtime["git_commit"],
        }


def stage_predictions(tracking_uri: str, output_path: Path) -> dict[str, Any]:
    """기준 장부의 35개 OOF를 내용 검증 뒤 한 전송 파일로 고정한다."""
    ledger = _load_yaml(PRECOMMIT_PATH)
    baseline_path = REPO_ROOT / ledger["sources"]["baseline_ledger"]["path"]
    baseline = _load_yaml(baseline_path)
    _source_hashes(ledger, require_private=True)

    folds_frame = pd.read_parquet(REPO_ROOT / ledger["sources"]["folds"]["path"])
    _require(list(folds_frame.columns) == [ID, "fold"], "fold 파일 열이 id, fold와 다르다.")
    ids = pd.Index(folds_frame[ID], name=ID)
    store = MlflowRunStore(tracking_uri=tracking_uri)
    frame = pd.DataFrame({ID: ids.to_numpy()})
    member_hashes: dict[str, str] = {}
    baseline_by_config = {entry["config"]: entry for entry in baseline["members"]}
    for config in ledger["candidate_pool"]["members"]:
        entry = baseline_by_config[config]
        prediction = store.oof_of(entry["run_id"]).reindex(ids)
        _require(not prediction.isna().any(), f"{config} OOF에 기준 id가 빠졌다.")
        values = prediction.to_numpy(dtype=np.float64)
        digest = prediction_array_sha256(values)
        expected = entry["integrity"]["oof_sha256"]
        _require(digest == expected, f"{config} OOF 내용 해시 불일치: {digest}")
        member_hashes[config] = digest
        frame[config] = values

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    payload = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "precommit_sha256": _sha256(PRECOMMIT_PATH),
        "tracking_uri": tracking_uri,
        "rows": len(frame),
        "members": len(member_hashes),
        "prediction_file": str(output_path),
        "prediction_file_sha256": _sha256(output_path),
        "member_prediction_sha256": member_hashes,
    }
    _atomic_json(output_path.with_suffix(".json"), payload)
    return payload


def load_inputs(prediction_path: Path) -> InputContext:
    ledger = _load_yaml(PRECOMMIT_PATH)
    baseline_path = REPO_ROOT / ledger["sources"]["baseline_ledger"]["path"]
    baseline = _load_yaml(baseline_path)
    source_hashes = _source_hashes(ledger, require_private=True)

    members = list(ledger["candidate_pool"]["members"])
    _require(len(members) == 35 and len(set(members)) == 35, "후보 35개 계약이 깨졌다.")
    _require(
        tuple(ledger["strategies"]["included"]) == ensemble_module.DEFAULT_COMBINER_NAMES,
        "등록 전략 이름이나 순서가 현재 구현과 다르다.",
    )
    _require(
        tuple(ledger["strategies"]["excluded_precision"])
        == ensemble_module.PRECISION_COMBINER_NAMES,
        "제외한 정밀 전략 목록이 현재 구현과 다르다.",
    )

    prediction_path = prediction_path.resolve()
    _require(prediction_path.is_file(), f"고정 OOF 전송 파일이 없다: {prediction_path}")
    staged = pd.read_parquet(prediction_path)
    _require(list(staged.columns) == [ID, *members], "고정 OOF 열과 순서가 장부와 다르다.")
    _require(not staged[ID].duplicated().any(), "고정 OOF id가 중복됐다.")

    folds_frame = pd.read_parquet(REPO_ROOT / ledger["sources"]["folds"]["path"])
    train = pd.read_csv(REPO_ROOT / ledger["sources"]["train"]["path"], usecols=[ID, TARGET])
    _require(staged[ID].equals(folds_frame[ID]), "고정 OOF id와 fold id 순서가 다르다.")
    _require(staged[ID].equals(train[ID]), "고정 OOF id와 학습 자료 id 순서가 다르다.")
    folds = pd.Series(folds_frame["fold"].to_numpy(), index=staged[ID], name="fold")
    labels = pd.Series(train[TARGET].to_numpy(), index=staged[ID], name=TARGET)
    _require(sorted(folds.unique()) == [0, 1, 2, 3, 4], "fold가 0부터 4까지를 덮지 않는다.")
    _require(set(labels.unique()) == {0, 1}, "목표값이 이진 0, 1이 아니다.")

    predictions = staged.set_index(ID).astype(np.float64)
    values = predictions.to_numpy(dtype=np.float64, copy=False)
    _require(np.isfinite(values).all(), "후보 OOF에 결측값이나 유한하지 않은 값이 있다.")
    baseline_by_config = {entry["config"]: entry for entry in baseline["members"]}
    member_hashes: dict[str, str] = {}
    for config in members:
        digest = prediction_array_sha256(predictions[config].to_numpy())
        expected = baseline_by_config[config]["integrity"]["oof_sha256"]
        _require(digest == expected, f"{config} OOF 내용 해시 불일치: {digest}")
        member_hashes[config] = digest

    band_of = ensemble_module.missingness_bands(
        REPO_ROOT / ledger["sources"]["train"]["path"],
        REPO_ROOT / ledger["sources"]["test"]["path"],
    ).reindex(predictions.index)
    _require(not band_of.isna().any(), "결측 개수 구간에 학습 id가 빠졌다.")

    return InputContext(
        predictions=predictions,
        labels=labels,
        folds=folds,
        missingness_bands=band_of.astype(np.int8),
        ledger=ledger,
        baseline=baseline,
        source_hashes=source_hashes,
        prediction_file_sha256=_sha256(prediction_path),
        member_prediction_sha256=member_hashes,
    )


@dataclass(frozen=True)
class StrategyScore:
    name: str
    auc: float
    fold_auc: dict[str, float]
    fits: int
    prediction: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PoolScore:
    members: tuple[str, ...]
    strategy_auc: dict[str, float]
    strategy_fold_auc: dict[str, dict[str, float]]
    best_strategy: str
    best_auc: float
    best_fold_auc: dict[str, float]
    prediction: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _WorkerContext:
    predictions: pd.DataFrame
    labels: pd.Series
    folds: pd.Series
    bands: pd.Series


_WORKER: _WorkerContext | None = None


def _init_worker(
    predictions: pd.DataFrame,
    labels: pd.Series,
    folds: pd.Series,
    bands: pd.Series,
) -> None:
    global _WORKER
    _WORKER = _WorkerContext(predictions, labels, folds, bands)


def _registry_combiner(name: str, context: _WorkerContext) -> ensemble_module.Combiner:
    if name == "shrunk_rank_logit_logistic":
        return ensemble_module.ShrunkRankLogitCombiner(fold_of=context.folds)
    if name == "missing_segmented_rank_logit":
        return ensemble_module.MissingnessSegmentedLogisticCombiner(band_of=context.bands)
    if name == "missing_interaction_rank_logit":
        return ensemble_module.MissingnessInteractionLogisticCombiner(band_of=context.bands)
    if name == "missing_4plus_rank_logit":
        return ensemble_module.MissingnessSegmentedLogisticCombiner(
            band_of=context.bands,
            specialized_bands=(2,),
            name="missing_4plus_rank_logit",
        )
    return ensemble_module.COMBINER_REGISTRY[name]


def _scope_mask(folds: pd.Series, excluded_fold: int | None) -> np.ndarray:
    if excluded_fold is None:
        return np.ones(len(folds), dtype=bool)
    return (folds.to_numpy() != excluded_fold)


def _pool_matrix(
    context: _WorkerContext,
    members: tuple[str, ...],
    excluded_fold: int | None,
    clone_source: str | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mask = _scope_mask(context.folds, excluded_fold)
    matrix = context.predictions.loc[mask, list(members)].copy()
    if clone_source is not None:
        _require(clone_source in members, f"복제 원본이 풀에 없다: {clone_source}")
        matrix[f"{clone_source}__exact_clone"] = matrix[clone_source].to_numpy()
    return matrix, context.folds.loc[mask], context.labels.loc[mask]


def _lofo_prediction(
    combiner: ensemble_module.Combiner,
    matrix: pd.DataFrame,
    fold_of: pd.Series,
    labels: pd.Series,
) -> tuple[np.ndarray, int]:
    unique = sorted(int(value) for value in fold_of.unique())
    _require(len(unique) >= 2, "학습 분할 안 채점에는 fold 두 개 이상이 필요하다.")
    prediction = np.full(len(matrix), np.nan, dtype=np.float64)
    for fold in unique:
        validate = (fold_of.to_numpy() == fold)
        train = ~validate
        fitted = combiner.fit(matrix.loc[train], labels.loc[train])
        prediction[validate] = np.asarray(fitted.predict(matrix.loc[validate]), dtype=np.float64)
    _require(np.isfinite(prediction).all(), "결합 전략 예측이 유한하지 않다.")
    return prediction, len(unique)


def _strategy_task(
    payload: tuple[str, tuple[str, ...], int | None, str | None, bool]
) -> dict[str, Any]:
    name, members, excluded_fold, clone_source, capture_prediction = payload
    if _WORKER is None:
        raise RuntimeError("결합 전략 작업자 입력이 준비되지 않았다.")
    try:
        matrix, fold_of, labels = _pool_matrix(
            _WORKER, members, excluded_fold, clone_source
        )
        prediction, fits = _lofo_prediction(
            _registry_combiner(name, _WORKER), matrix, fold_of, labels
        )
        fold_auc = {
            str(fold): float(
                roc_auc_score(
                    labels.loc[fold_of.to_numpy() == fold].to_numpy(),
                    prediction[fold_of.to_numpy() == fold],
                )
            )
            for fold in sorted(int(value) for value in fold_of.unique())
        }
        return {
            "name": name,
            "auc": float(roc_auc_score(labels.to_numpy(), prediction)),
            "fold_auc": fold_auc,
            "fits": fits,
            "prediction": prediction if capture_prediction else None,
            "failure": None,
        }
    except Exception as exc:  # 중단 계약을 부모 프로세스에 구조화해 전달한다.
        return {
            "name": name,
            "failure": f"{type(exc).__name__}: {exc}",
            "fits": 0,
            "prediction": None,
        }


class StrategyEvaluator:
    """등록 전략을 고정 순서로 평가하고 작업자 사이에서 입력을 재사용한다."""

    def __init__(self, context: InputContext, jobs: int) -> None:
        _require(jobs >= 1, "작업자 수는 1 이상이어야 한다.")
        self.context = context
        self.jobs = jobs
        self.names = tuple(context.ledger["strategies"]["included"])
        self.fits = 0
        self.arm_evaluations = 0
        self._executor: ProcessPoolExecutor | None = None
        _init_worker(
            context.predictions,
            context.labels,
            context.folds,
            context.missingness_bands,
        )
        if jobs > 1:
            self._executor = ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init_worker,
                initargs=(
                    context.predictions,
                    context.labels,
                    context.folds,
                    context.missingness_bands,
                ),
            )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> StrategyEvaluator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _map(self, payloads: list[tuple[str, tuple[str, ...], int | None, str | None, bool]]) -> list[dict[str, Any]]:
        if self._executor is None:
            return [_strategy_task(payload) for payload in payloads]
        return list(self._executor.map(_strategy_task, payloads))

    def evaluate(
        self,
        members: Sequence[str],
        *,
        excluded_fold: int | None,
        clone_source: str | None = None,
        capture_prediction: bool = False,
    ) -> PoolScore:
        pool = tuple(members)
        payloads = [
            (name, pool, excluded_fold, clone_source, False) for name in self.names
        ]
        outcomes = self._map(payloads)
        failures = {
            outcome["name"]: outcome["failure"]
            for outcome in outcomes
            if outcome["failure"] is not None
        }
        _require(
            not failures,
            "필수 등록 전략이 실패했다: "
            + "; ".join(f"{name}={reason}" for name, reason in failures.items()),
        )
        _require(len(outcomes) == 19, "필수 등록 전략 19개가 모두 실행되지 않았다.")
        self.fits += sum(int(outcome["fits"]) for outcome in outcomes)
        self.arm_evaluations += 1
        best = max(outcomes, key=lambda item: (item["auc"], -self.names.index(item["name"])))
        prediction = None
        if capture_prediction:
            replay = self._map(
                [(best["name"], pool, excluded_fold, clone_source, True)]
            )[0]
            _require(replay["failure"] is None, f"최선 전략 예측 재현 실패: {replay['failure']}")
            _require(replay["auc"] == best["auc"], "최선 전략 재실행 AUC가 달라졌다.")
            self.fits += int(replay["fits"])
            prediction = np.asarray(replay["prediction"], dtype=np.float64)
        return PoolScore(
            members=pool,
            strategy_auc={outcome["name"]: float(outcome["auc"]) for outcome in outcomes},
            strategy_fold_auc={outcome["name"]: outcome["fold_auc"] for outcome in outcomes},
            best_strategy=best["name"],
            best_auc=float(best["auc"]),
            best_fold_auc=best["fold_auc"],
            prediction=prediction,
        )

    def evaluate_one(
        self,
        strategy: str,
        members: Sequence[str],
        *,
        excluded_fold: int | None,
        capture_prediction: bool = False,
    ) -> StrategyScore:
        pool = tuple(members)
        outcome = self._map(
            [(strategy, pool, excluded_fold, None, capture_prediction)]
        )[0]
        _require(outcome["failure"] is None, f"{strategy} 평가 실패: {outcome['failure']}")
        self.fits += int(outcome["fits"])
        return StrategyScore(
            name=strategy,
            auc=float(outcome["auc"]),
            fold_auc=outcome["fold_auc"],
            fits=int(outcome["fits"]),
            prediction=(
                np.asarray(outcome["prediction"], dtype=np.float64)
                if capture_prediction
                else None
            ),
        )

    def contributions(
        self,
        strategy: str,
        anchor: PoolScore,
        targets: Sequence[tuple[str, ...]],
        *,
        excluded_fold: int | None,
    ) -> dict[tuple[str, ...], float]:
        payloads: list[tuple[str, tuple[str, ...], int | None, str | None, bool]] = []
        for target in targets:
            target_set = set(target)
            reduced = tuple(member for member in anchor.members if member not in target_set)
            _require(reduced, f"조사 순서 계산이 풀 전체를 제거한다: {target}")
            payloads.append((strategy, reduced, excluded_fold, None, False))
        outcomes = self._map(payloads)
        failures = [outcome for outcome in outcomes if outcome["failure"] is not None]
        _require(
            not failures,
            "조사 순서 제외 기여 계산 실패: "
            + "; ".join(f"{item['name']}={item['failure']}" for item in failures),
        )
        self.fits += sum(int(outcome["fits"]) for outcome in outcomes)
        return {
            target: float(outcome["auc"] - anchor.best_auc)
            for target, outcome in zip(targets, outcomes, strict=True)
        }


class WeightedAucSorter:
    """예측 정렬을 한 번만 만들어 가중 AUC를 선형 시간에 다시 계산한다."""

    def __init__(self, prediction: np.ndarray, labels: np.ndarray) -> None:
        order = np.argsort(prediction, kind="stable")
        sorted_prediction = prediction[order]
        boundary = np.empty(len(order), dtype=bool)
        boundary[0] = True
        np.not_equal(sorted_prediction[1:], sorted_prediction[:-1], out=boundary[1:])
        self.order = order
        self.group = np.cumsum(boundary) - 1
        self.groups = int(self.group[-1]) + 1
        self.positive = labels[order].astype(np.float64)

    def auc(self, weights: np.ndarray) -> float:
        ordered = weights[self.order].astype(np.float64, copy=False)
        total = np.bincount(self.group, weights=ordered, minlength=self.groups)
        positive = np.bincount(
            self.group, weights=ordered * self.positive, minlength=self.groups
        )
        below = np.concatenate(([0.0], np.cumsum(total)[:-1]))
        rank_sum = float(np.sum(positive * (below + (total + 1.0) / 2.0)))
        positives = float(positive.sum())
        negatives = float(total.sum()) - positives
        _require(positives > 0.0 and negatives > 0.0, "재표본에 목표값 한쪽만 남았다.")
        return (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def _strata(folds: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    return [
        np.flatnonzero((folds == fold) & (labels == label))
        for fold in sorted(np.unique(folds))
        for label in sorted(np.unique(labels))
    ]


def paired_bootstrap(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, float]:
    """fold와 목표값 층을 보존한 짝지은 행 부트스트랩."""
    before_sorter = WeightedAucSorter(before, labels)
    after_sorter = WeightedAucSorter(after, labels)
    groups = _strata(folds, labels)
    differences = np.empty(replicates, dtype=np.float64)
    probabilities = {
        len(group): np.full(len(group), 1.0 / len(group), dtype=np.float64)
        for group in groups
    }
    for index in range(replicates):
        weights = np.zeros(len(labels), dtype=np.float64)
        for group in groups:
            weights[group] = rng.multinomial(len(group), probabilities[len(group)])
        differences[index] = after_sorter.auc(weights) - before_sorter.auc(weights)
    quantiles = np.quantile(differences, [0.025, 0.5, 0.975])
    return {
        "replicates": replicates,
        "minimum": float(differences.min()),
        "percentile_2p5": float(quantiles[0]),
        "median": float(quantiles[1]),
        "percentile_97p5": float(quantiles[2]),
        "maximum": float(differences.max()),
    }


def _scope_arrays(context: InputContext, excluded_fold: int | None) -> tuple[np.ndarray, np.ndarray]:
    mask = _scope_mask(context.folds, excluded_fold)
    return (
        context.labels.to_numpy()[mask],
        context.folds.to_numpy()[mask],
    )


def _fold_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: float(after[key] - before[key]) for key in sorted(before, key=int)}


def _pool_score_payload(score: PoolScore) -> dict[str, Any]:
    return {
        "members": list(score.members),
        "member_count": len(score.members),
        "best_strategy": score.best_strategy,
        "best_auc": score.best_auc,
        "best_fold_auc": score.best_fold_auc,
        "strategy_auc": score.strategy_auc,
        "strategy_fold_auc": score.strategy_fold_auc,
    }


def _identity_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def run_null_band(
    context: InputContext,
    evaluator: StrategyEvaluator,
    output_root: Path,
    input_identity: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    ledger = context.ledger
    checkpoint_root = output_root / "null-checkpoints"
    block_specs = [("full-oof", None)] + [(f"outer-{fold}", fold) for fold in range(5)]
    resumed = 0
    blocks: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(ledger["randomness"]["bootstrap_seed"]))
    replicates = int(ledger["randomness"]["bootstrap_replicates"])
    block_identity = _identity_sha256({"input": input_identity, "phase": "null-band"})

    for block_name, excluded_fold in block_specs:
        checkpoint = checkpoint_root / f"{block_name}.json"
        if checkpoint.is_file():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            _require(payload["identity_sha256"] == block_identity, f"{block_name} 영점 중간 저장 입력이 다르다.")
            block = payload["block"]
            rng.bit_generator.state = payload["rng_state_after"]
            blocks.append(block)
            resumed += 1
            continue

        baseline = evaluator.evaluate(
            context.members,
            excluded_fold=excluded_fold,
            capture_prediction=True,
        )
        _require(baseline.prediction is not None, "영점 기준 팔 예측이 없다.")
        labels, folds = _scope_arrays(context, excluded_fold)
        contrasts: list[dict[str, Any]] = []
        for source in context.members:
            large = evaluator.evaluate(
                context.members,
                excluded_fold=excluded_fold,
                clone_source=source,
                capture_prediction=True,
            )
            _require(large.prediction is not None, f"{source} 복제 팔 예측이 없다.")
            fold_delta = _fold_delta(large.best_fold_auc, baseline.best_fold_auc)
            bootstrap = paired_bootstrap(
                large.prediction,
                baseline.prediction,
                labels,
                folds,
                rng,
                replicates,
            )
            contrasts.append(
                {
                    "source": source,
                    "small_members": 35,
                    "large_members": 36,
                    "small_best_strategy": baseline.best_strategy,
                    "large_best_strategy": large.best_strategy,
                    "best_strategy_changed": baseline.best_strategy != large.best_strategy,
                    "small_best_auc": baseline.best_auc,
                    "large_best_auc": large.best_auc,
                    "delta": baseline.best_auc - large.best_auc,
                    "fold_delta": fold_delta,
                    "negative_folds": sum(value < 0.0 for value in fold_delta.values()),
                    "fold_total": len(fold_delta),
                    "strategy_auc_small": baseline.strategy_auc,
                    "strategy_auc_large": large.strategy_auc,
                    "bootstrap": bootstrap,
                }
            )
        same_size = [
            {
                "left": left["source"],
                "right": right["source"],
                "delta": left["large_best_auc"] - right["large_best_auc"],
            }
            for left in contrasts
            for right in contrasts
            if left["source"] != right["source"]
        ]
        observed_lower = min(record["delta"] for record in contrasts)
        bootstrap_lower = min(record["bootstrap"]["percentile_2p5"] for record in contrasts)
        same_size_lower = min(record["delta"] for record in same_size)
        block = {
            "name": block_name,
            "excluded_outer_fold": excluded_fold,
            "rows": len(labels),
            "folds": sorted(int(value) for value in np.unique(folds)),
            "baseline": _pool_score_payload(baseline),
            "contrasts": contrasts,
            "same_size": {
                "directed_pairs": len(same_size),
                "minimum": same_size_lower,
                "maximum": max(record["delta"] for record in same_size),
                "pairs": same_size,
            },
            "lower_components": {
                "observed_minimum": observed_lower,
                "bootstrap_2p5_minimum": bootstrap_lower,
                "same_size_minimum": same_size_lower,
            },
            "lower": min(observed_lower, bootstrap_lower, same_size_lower),
        }
        _atomic_json(
            checkpoint,
            {
                "schema_version": EXECUTION_SCHEMA_VERSION,
                "identity_sha256": block_identity,
                "rng_state_after": rng.bit_generator.state,
                "block": block,
            },
        )
        blocks.append(block)

    full_lower = blocks[0]["lower"]
    inner_lower = min(block["lower"] for block in blocks[1:])
    existing = float(ledger["null_band"]["existing_lower"])
    adopted = min(existing, full_lower, inner_lower)
    sign_histogram: dict[str, dict[str, int]] = {}
    for scale_name, scale_blocks in (
        ("full-oof", blocks[:1]),
        ("inner-training", blocks[1:]),
    ):
        histogram: dict[str, int] = {}
        for block in scale_blocks:
            for contrast in block["contrasts"]:
                key = str(contrast["negative_folds"])
                histogram[key] = histogram.get(key, 0) + 1
        sign_histogram[scale_name] = dict(sorted(histogram.items(), key=lambda item: int(item[0])))
    payload = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "ticket_issue": 339,
        "map_issue": 338,
        "input_identity": input_identity,
        "bootstrap_seed": int(ledger["randomness"]["bootstrap_seed"]),
        "bootstrap_replicates": replicates,
        "blocks": blocks,
        "scale_lower": {"full-oof": full_lower, "inner-training": inner_lower},
        "existing_lower": existing,
        "adopted_lower": adopted,
        "no_narrowing_applied": adopted <= existing,
        "fold_sign_frequency": sign_histogram,
    }
    null_path = output_root / "null-band.json"
    _atomic_json(null_path, payload)
    return payload, resumed


def _target_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(values))


def _refit_counts(ledger: dict[str, Any]) -> dict[str, int]:
    default = int(ledger["candidate_pool"]["full_refit_count"]["default"])
    overrides = ledger["candidate_pool"]["full_refit_count"]["overrides"]
    return {
        member: int(overrides.get(member, default))
        for member in ledger["candidate_pool"]["members"]
    }


def _common_sort_key(
    target: tuple[str, ...],
    ledger: dict[str, Any],
    standalone_auc: dict[str, float],
) -> tuple[Any, ...]:
    target_set = set(target)
    perspectives = ledger["information_perspectives"]
    disappearing = sum(
        set(entry["members"]) <= target_set for entry in perspectives.values()
    )
    depths = [
        int(entry["predecessor_depth"][member])
        for entry in ledger["lineage_groups"].values()
        for member in target
        if member in entry["predecessor_depth"]
    ]
    predecessor_depth = min(depths) if depths else math.inf
    single_auc = min(standalone_auc[member] for member in target)
    refits = sum(_refit_counts(ledger)[member] for member in target)
    return (disappearing, predecessor_depth, single_auc, -refits, target)


def _sort_keys_payload(
    target: tuple[str, ...],
    contribution: float,
    ledger: dict[str, Any],
    standalone_auc: dict[str, float],
) -> dict[str, Any]:
    common = _common_sort_key(target, ledger, standalone_auc)
    return {
        "exclusion_contribution": contribution,
        "disappearing_information_perspectives": common[0],
        "predecessor_depth": None if math.isinf(common[1]) else common[1],
        "standalone_auc": common[2],
        "full_refit_reduction": -common[3],
        "config_tuple": list(target),
    }


@dataclass(frozen=True)
class CandidateState:
    step: int
    pool: tuple[str, ...]
    score: PoolScore


@dataclass
class SplitResult:
    label: str
    excluded_fold: int | None
    anchor: PoolScore
    terminal: PoolScore
    trajectory: list[dict[str, Any]]
    accepted_states: list[CandidateState]
    order: dict[str, Any]


def run_split(
    label: str,
    excluded_fold: int | None,
    context: InputContext,
    evaluator: StrategyEvaluator,
    adopted_lower: float,
) -> SplitResult:
    ledger = context.ledger
    anchor = evaluator.evaluate(context.members, excluded_fold=excluded_fold)
    mask = _scope_mask(context.folds, excluded_fold)
    y = context.labels.to_numpy()[mask]
    standalone_auc = {
        member: float(roc_auc_score(y, context.predictions[member].to_numpy()[mask]))
        for member in context.members
    }
    lineage_targets = [
        _target_tuple(entry["members"]) for entry in ledger["lineage_groups"].values()
    ]
    perspective_targets = [
        _target_tuple(entry["members"]) for entry in ledger["information_perspectives"].values()
    ]
    individual_targets = [(member,) for member in context.members]
    all_targets = list(dict.fromkeys([*lineage_targets, *perspective_targets, *individual_targets]))
    contributions = evaluator.contributions(
        anchor.best_strategy,
        anchor,
        all_targets,
        excluded_fold=excluded_fold,
    )

    common = lambda target: _common_sort_key(target, ledger, standalone_auc)
    lineages = sorted(
        lineage_targets,
        key=lambda target: (-len(target), contributions[target], *common(target)),
    )
    perspectives = sorted(
        perspective_targets,
        key=lambda target: (len(target), contributions[target], *common(target)),
    )
    individuals = sorted(
        individual_targets,
        key=lambda target: (contributions[target], *common(target)),
    )
    order = {
        "anchor_strategy": anchor.best_strategy,
        "standalone_auc": standalone_auc,
        "targets": {
            "+".join(target): _sort_keys_payload(
                target, contributions[target], ledger, standalone_auc
            )
            for target in all_targets
        },
        "stage_1": [list(target) for target in lineages],
        "stage_3": [list(target) for target in perspectives],
        "stage_4": [list(target) for target in individuals],
    }

    working = anchor
    trajectory: list[dict[str, Any]] = []
    accepted = [CandidateState(0, anchor.members, anchor)]
    kept_lineages: list[tuple[str, ...]] = []
    step = 0

    def attempt(stage: int, kind: str, frozen_target: tuple[str, ...]) -> str:
        nonlocal working, step
        step += 1
        removal = tuple(member for member in frozen_target if member in working.members)
        sort_keys = _sort_keys_payload(
            frozen_target, contributions[frozen_target], ledger, standalone_auc
        )
        if not removal or len(removal) == len(working.members):
            trajectory.append(
                {
                    "step": step,
                    "stage": stage,
                    "target_kind": kind,
                    "frozen_target": list(frozen_target),
                    "removed": list(removal),
                    "working_members_before": len(working.members),
                    "working_members_after": len(working.members),
                    "delta_vs_working": 0.0,
                    "verdict": "생략",
                    "delta_vs_anchor": working.best_auc - anchor.best_auc,
                    "before_strategy": working.best_strategy,
                    "after_strategy": working.best_strategy,
                    "strategy_changed": False,
                    "fold_delta": {key: 0.0 for key in working.best_fold_auc},
                    "negative_folds": 0,
                    "fold_total": len(working.best_fold_auc),
                    "sort_keys": sort_keys,
                    "strategy_auc_before": working.strategy_auc,
                    "strategy_auc_after": working.strategy_auc,
                    "skip_reason": "교집합이 비었음" if not removal else "작업 풀 전체가 사라짐",
                }
            )
            return "생략"
        candidate_members = tuple(member for member in working.members if member not in set(removal))
        candidate = evaluator.evaluate(candidate_members, excluded_fold=excluded_fold)
        delta = candidate.best_auc - working.best_auc
        verdict = "유지" if delta < adopted_lower else "제거"
        fold_delta = _fold_delta(working.best_fold_auc, candidate.best_fold_auc)
        trajectory.append(
            {
                "step": step,
                "stage": stage,
                "target_kind": kind,
                "frozen_target": list(frozen_target),
                "removed": list(removal),
                "working_members_before": len(working.members),
                "working_members_after": len(candidate.members),
                "delta_vs_working": delta,
                "verdict": verdict,
                "delta_vs_anchor": candidate.best_auc - anchor.best_auc,
                "before_strategy": working.best_strategy,
                "after_strategy": candidate.best_strategy,
                "strategy_changed": working.best_strategy != candidate.best_strategy,
                "fold_delta": fold_delta,
                "negative_folds": sum(value < 0.0 for value in fold_delta.values()),
                "fold_total": len(fold_delta),
                "sort_keys": sort_keys,
                "strategy_auc_before": working.strategy_auc,
                "strategy_auc_after": candidate.strategy_auc,
            }
        )
        if verdict == "제거":
            working = candidate
            accepted.append(CandidateState(step, working.members, working))
        return verdict

    for target in lineages:
        if attempt(1, "모델 계보 묶음", target) == "유지":
            kept_lineages.append(target)
    depths_by_member = {
        member: int(entry["predecessor_depth"][member])
        for entry in ledger["lineage_groups"].values()
        for member in entry["predecessor_depth"]
    }
    for group in kept_lineages:
        ordered = sorted(
            ((member,) for member in group),
            key=lambda target: (depths_by_member[target[0]], *common(target)),
        )
        for target in ordered:
            attempt(2, "유지된 모델 계보 묶음의 개별 구성원", target)
    for target in perspectives:
        attempt(3, "정보 관점 묶음", target)
    for target in individuals:
        attempt(4, "개별 구성원", target)

    return SplitResult(
        label=label,
        excluded_fold=excluded_fold,
        anchor=anchor,
        terminal=working,
        trajectory=trajectory,
        accepted_states=accepted,
        order=order,
    )


def _split_payload(result: SplitResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "excluded_outer_fold": result.excluded_fold,
        "anchor": _pool_score_payload(result.anchor),
        "terminal": _pool_score_payload(result.terminal),
        "order": result.order,
        "trajectory": result.trajectory,
        "accepted_trajectory": [
            {
                "step": state.step,
                "pool": list(state.pool),
                "member_count": len(state.pool),
                "best_strategy": state.score.best_strategy,
                "best_auc": state.score.best_auc,
                "delta_vs_anchor": state.score.best_auc - result.anchor.best_auc,
            }
            for state in result.accepted_states
        ],
    }


def _pool_score_from_payload(payload: dict[str, Any]) -> PoolScore:
    return PoolScore(
        members=tuple(payload["members"]),
        strategy_auc={key: float(value) for key, value in payload["strategy_auc"].items()},
        strategy_fold_auc={
            name: {key: float(value) for key, value in fold.items()}
            for name, fold in payload["strategy_fold_auc"].items()
        },
        best_strategy=payload["best_strategy"],
        best_auc=float(payload["best_auc"]),
        best_fold_auc={key: float(value) for key, value in payload["best_fold_auc"].items()},
    )


def _split_from_payload(payload: dict[str, Any]) -> SplitResult:
    anchor = _pool_score_from_payload(payload["anchor"])
    terminal = _pool_score_from_payload(payload["terminal"])
    score_by_pool = {anchor.members: anchor, terminal.members: terminal}
    states = []
    for entry in payload["accepted_trajectory"]:
        pool = tuple(entry["pool"])
        score = score_by_pool.get(pool)
        if score is None:
            score = PoolScore(
                members=pool,
                strategy_auc={},
                strategy_fold_auc={},
                best_strategy=entry["best_strategy"],
                best_auc=float(entry["best_auc"]),
                best_fold_auc={},
            )
        states.append(CandidateState(int(entry["step"]), pool, score))
    return SplitResult(
        label=payload["label"],
        excluded_fold=payload["excluded_outer_fold"],
        anchor=anchor,
        terminal=terminal,
        trajectory=payload["trajectory"],
        accepted_states=states,
        order=payload["order"],
    )


def _fit_outer_prediction(
    context: InputContext,
    strategy: str,
    members: tuple[str, ...],
    held_out_fold: int,
) -> np.ndarray:
    worker = _WorkerContext(
        context.predictions, context.labels, context.folds, context.missingness_bands
    )
    train = context.folds.to_numpy() != held_out_fold
    score = ~train
    combiner = _registry_combiner(strategy, worker)
    fitted = combiner.fit(
        context.predictions.loc[train, list(members)], context.labels.loc[train]
    )
    prediction = np.asarray(
        fitted.predict(context.predictions.loc[score, list(members)]), dtype=np.float64
    )
    _require(np.isfinite(prediction).all(), "바깥쪽 채점 예측이 유한하지 않다.")
    return prediction


def _refit_total(pool: tuple[str, ...], ledger: dict[str, Any]) -> int:
    counts = _refit_counts(ledger)
    return sum(counts[member] for member in pool)


def choose_final_candidate(
    result: SplitResult, adopted_lower: float, ledger: dict[str, Any]
) -> tuple[CandidateState, CandidateState | None]:
    eligible = [
        state
        for state in result.accepted_states
        if state.score.best_auc - result.anchor.best_auc >= adopted_lower
    ]
    _require(eligible, "앵커 자신이 최종 후보 목록에서 사라졌다.")
    selected = min(
        eligible,
        key=lambda state: (
            len(state.pool),
            -(state.score.best_auc - result.anchor.best_auc),
            _refit_total(state.pool, ledger),
            state.pool,
        ),
    )
    position = result.accepted_states.index(selected)
    previous = result.accepted_states[position - 1] if position > 0 else None
    return selected, previous


def _checkpoint_identity(input_identity: dict[str, Any], null_sha256: str) -> str:
    return _identity_sha256({"input": input_identity, "null_band_sha256": null_sha256})


def run_procedure(
    context: InputContext,
    evaluator: StrategyEvaluator,
    output_root: Path,
    input_identity: dict[str, Any],
    null_band: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    checkpoints = output_root / "checkpoints"
    predictions_root = output_root / "predictions"
    null_path = output_root / "null-band.json"
    null_sha = _sha256(null_path)
    identity = _checkpoint_identity(input_identity, null_sha)
    adopted_lower = float(null_band["adopted_lower"])
    splits: dict[str, SplitResult] = {}
    resumed = 0
    outer_fit_count = 0

    for fold in range(5):
        label = f"outer-{fold}"
        checkpoint = checkpoints / f"{label}.json"
        prediction_path = predictions_root / f"{label}.parquet"
        if checkpoint.is_file():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            _require(payload["identity_sha256"] == identity, f"{label} 중간 저장 입력이 다르다.")
            _require(prediction_path.is_file(), f"{label} 채점 예측 파일이 없다.")
            _require(_sha256(prediction_path) == payload["prediction_file_sha256"], f"{label} 채점 예측 파일 해시가 다르다.")
            split = _split_from_payload(payload["split"])
            resumed += 1
        else:
            split = run_split(label, fold, context, evaluator, adopted_lower)
            terminal_prediction = _fit_outer_prediction(
                context, split.terminal.best_strategy, split.terminal.members, fold
            )
            anchor_prediction = _fit_outer_prediction(
                context, split.anchor.best_strategy, split.anchor.members, fold
            )
            outer_fit_count += 2
            mask = context.folds.to_numpy() == fold
            prediction_frame = pd.DataFrame(
                {
                    ID: context.predictions.index.to_numpy()[mask],
                    "fold": fold,
                    "prediction": terminal_prediction,
                    "anchor_prediction": anchor_prediction,
                }
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction_path.with_name(f".{prediction_path.name}.tmp")
            prediction_frame.to_parquet(temporary, index=False)
            os.replace(temporary, prediction_path)
            payload = {
                "schema_version": EXECUTION_SCHEMA_VERSION,
                "identity_sha256": identity,
                "adopted_lower": adopted_lower,
                "split": _split_payload(split),
                "prediction_file": str(prediction_path.relative_to(output_root)),
                "prediction_file_sha256": _sha256(prediction_path),
                "prediction_content_sha256": prediction_array_sha256(terminal_prediction),
                "anchor_prediction_content_sha256": prediction_array_sha256(anchor_prediction),
            }
            _atomic_json(checkpoint, payload)
        splits[label] = split

    final_checkpoint = checkpoints / "final.json"
    if final_checkpoint.is_file():
        payload = json.loads(final_checkpoint.read_text(encoding="utf-8"))
        _require(payload["identity_sha256"] == identity, "final 중간 저장 입력이 다르다.")
        final = _split_from_payload(payload["split"])
        selected_payload = payload["selected_candidate"]
        previous_payload = payload["previous_candidate"]
        boundary = payload["boundary"]
        resumed += 1
    else:
        final = run_split("final", None, context, evaluator, adopted_lower)
        selected, previous = choose_final_candidate(final, adopted_lower, context.ledger)
        anchor_prediction = evaluator.evaluate_one(
            final.anchor.best_strategy,
            final.anchor.members,
            excluded_fold=None,
            capture_prediction=True,
        ).prediction
        selected_prediction = evaluator.evaluate_one(
            selected.score.best_strategy,
            selected.pool,
            excluded_fold=None,
            capture_prediction=True,
        ).prediction
        _require(anchor_prediction is not None and selected_prediction is not None, "경계 후보 예측이 없다.")
        labels, folds = _scope_arrays(context, None)
        boundary_bootstrap = paired_bootstrap(
            anchor_prediction,
            selected_prediction,
            labels,
            folds,
            np.random.default_rng(int(context.ledger["randomness"]["bootstrap_seed"])),
            int(context.ledger["randomness"]["bootstrap_replicates"]),
        )
        boundary = {
            "bootstrap": boundary_bootstrap,
            "is_boundary": boundary_bootstrap["percentile_2p5"] < adopted_lower,
            "condition": "paired bootstrap 2.5 percentile < adopted_lower",
        }
        selected_payload = {
            "step": selected.step,
            "pool": list(selected.pool),
            "member_count": len(selected.pool),
            "best_strategy": selected.score.best_strategy,
            "best_auc": selected.score.best_auc,
            "delta_vs_anchor": selected.score.best_auc - final.anchor.best_auc,
            "full_refit_count": _refit_total(selected.pool, context.ledger),
        }
        previous_payload = (
            {
                "step": previous.step,
                "pool": list(previous.pool),
                "member_count": len(previous.pool),
                "best_strategy": previous.score.best_strategy,
                "best_auc": previous.score.best_auc,
                "delta_vs_anchor": previous.score.best_auc - final.anchor.best_auc,
            }
            if previous is not None
            else None
        )
        payload = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "identity_sha256": identity,
            "adopted_lower": adopted_lower,
            "split": _split_payload(final),
            "selected_candidate": selected_payload,
            "previous_candidate": previous_payload,
            "boundary": boundary,
        }
        _atomic_json(final_checkpoint, payload)
    splits["final"] = final

    nested_parts = []
    anchor_parts = []
    validation_folds: dict[str, dict[str, Any]] = {}
    for fold in range(5):
        frame = pd.read_parquet(predictions_root / f"outer-{fold}.parquet")
        nested_parts.append(frame[[ID, "prediction"]])
        anchor_parts.append(frame[[ID, "anchor_prediction"]])
        fold_labels = context.labels.reindex(frame[ID]).to_numpy()
        selected_auc = float(roc_auc_score(fold_labels, frame["prediction"]))
        baseline_auc = float(roc_auc_score(fold_labels, frame["anchor_prediction"]))
        delta = selected_auc - baseline_auc
        validation_folds[str(fold)] = {
            "selected_auc": selected_auc,
            "anchor_auc": baseline_auc,
            "delta": delta,
            "winner": "selected" if delta > 0.0 else "anchor" if delta < 0.0 else "tie",
        }
    nested = pd.concat(nested_parts).set_index(ID)["prediction"].reindex(context.predictions.index)
    anchor_nested = pd.concat(anchor_parts).set_index(ID)["anchor_prediction"].reindex(context.predictions.index)
    _require(not nested.isna().any() and not anchor_nested.isna().any(), "바깥쪽 채점 예측이 전 행을 덮지 않는다.")
    nested_auc = float(roc_auc_score(context.labels.to_numpy(), nested.to_numpy()))
    anchor_nested_auc = float(
        roc_auc_score(context.labels.to_numpy(), anchor_nested.to_numpy())
    )
    stability = {
        member: sum(member in splits[f"outer-{fold}"].terminal.members for fold in range(5))
        for member in context.members
    }
    decision = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "ticket_issue": 339,
        "map_issue": 338,
        "input_identity": input_identity,
        "null_band_sha256": null_sha,
        "adopted_lower": adopted_lower,
        "adopted_lower_sources": {
            "existing": null_band["existing_lower"],
            **null_band["scale_lower"],
        },
        "splits": {label: _split_payload(splits[label]) for label in [f"outer-{fold}" for fold in range(5)] + ["final"]},
        "procedure_nested_oof_auc": nested_auc,
        "anchor_nested_oof_auc": anchor_nested_auc,
        "procedure_delta_vs_anchor": nested_auc - anchor_nested_auc,
        "validation_fold_results": validation_folds,
        "validation_fold_wins": {
            "selected": sum(item["winner"] == "selected" for item in validation_folds.values()),
            "anchor": sum(item["winner"] == "anchor" for item in validation_folds.values()),
            "tie": sum(item["winner"] == "tie" for item in validation_folds.values()),
        },
        "selection_stability": stability,
        "selected_candidate": selected_payload,
        "previous_candidate": previous_payload,
        "boundary": boundary,
        "null_fold_sign_frequency": null_band["fold_sign_frequency"],
        "small_pool_band_approximation": APPROXIMATION_NOTE,
    }
    return decision, resumed, outer_fit_count


def _manifest(output_root: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    return {str(path.relative_to(output_root)): _sha256(path) for path in files}


def _write_manifest(output_root: Path) -> None:
    manifest = _manifest(output_root)
    lines = [f"{digest}  {relative}" for relative, digest in manifest.items()]
    _atomic_write(output_root / "manifest.sha256", ("\n".join(lines) + "\n").encode("utf-8"))


def execute(prediction_path: Path, output_root: Path, jobs: int) -> dict[str, Any]:
    started = time.monotonic()
    runtime = _runtime_identity()
    code_path = Path(__file__).resolve()
    code_sha = _sha256(code_path)
    context = load_inputs(prediction_path)
    input_identity = context.identity(code_sha, runtime)
    output_root.mkdir(parents=True, exist_ok=True)

    with StrategyEvaluator(context, jobs) as evaluator:
        null_band, resumed_null = run_null_band(
            context, evaluator, output_root, input_identity
        )
        decision, resumed_splits, outer_fits = run_procedure(
            context, evaluator, output_root, input_identity, null_band
        )
        decision_bytes = _json_bytes(decision)
        existing_decision = output_root / "decision.json"
        if existing_decision.is_file():
            _require(
                existing_decision.read_bytes() == decision_bytes,
                "같은 입력의 재개 실행 decision.json이 글자 단위로 다르다.",
            )
        _atomic_write(existing_decision, decision_bytes)

        # 완료된 중간 저장만 다시 읽어 계산 없이 같은 결정을 만드는 재개 검사를 수행한다.
        replay, replay_resumed, replay_outer_fits = run_procedure(
            context, evaluator, output_root, input_identity, null_band
        )
        replay_bytes = _json_bytes(replay)
        _require(replay_bytes == decision_bytes, "완료 중간 저장 재개의 결정 산출물이 다르다.")
        _require(replay_resumed == 6 and replay_outer_fits == 0, "재개 검사가 완료 분할을 다시 계산했다.")

        telemetry = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "runtime_identity": {
                **runtime,
                "pyproject_sha256": _sha256(REPO_ROOT / "pyproject.toml"),
                "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
            },
            "workers": jobs,
            "elapsed_seconds": time.monotonic() - started,
            "top_level_strategy_fits": evaluator.fits + outer_fits,
            "pool_arm_evaluations": evaluator.arm_evaluations,
            "resumed_null_blocks": resumed_null,
            "resumed_splits": resumed_splits,
            "resume_verification": {
                "completed_splits_loaded": replay_resumed,
                "recomputed_outer_fits": replay_outer_fits,
                "decision_byte_identical": True,
                "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
            },
            "failures_and_retries": [],
        }
    _atomic_json(output_root / "telemetry.json", telemetry)
    _write_manifest(output_root)
    return {
        "decision": decision,
        "telemetry": telemetry,
        "manifest_sha256": _sha256(output_root / "manifest.sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="MLflow의 35개 OOF를 전송 파일로 고정한다.")
    stage.add_argument("--tracking-uri", required=True)
    stage.add_argument("--output", type=Path, default=DEFAULT_STAGED_PREDICTIONS)

    run = subparsers.add_parser("run", help="영점 대조와 여섯 실행 블록을 수행한다.")
    run.add_argument("--predictions", type=Path, default=DEFAULT_STAGED_PREDICTIONS)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))

    args = parser.parse_args()
    if args.command == "stage":
        result = stage_predictions(args.tracking_uri, args.output)
    else:
        result = execute(args.predictions.resolve(), args.output_root.resolve(), args.jobs)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
