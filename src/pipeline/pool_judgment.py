"""후보 평가 사본에서 candidate-pool-v1 판정 기록을 만드는 실행 경로.

사용법:
    uv run python -m pipeline.pool_judgment \
        --judgment-id issueNNN-expNNN \
        --candidate-run <run_id> \
        --model-lineage-group <name> \
        --selection-description "결과를 보기 전에 고정한 후보"

여러 ``--candidate-run``을 주면 바깥쪽 검증 분할마다 학습 부분 안에서 후보와
결합 전략을 다시 고르는 선택 절차 대조를 수행한다.
``--action replacement --replaces-run-id <run_id>``는 현재 풀과 신규 후보로 기존
구성원 하나를 바꾼 풀을 같은 조건에서 비교한다.

이 명령은 후보 풀을 변경하지 않는다.
완료된 판정 기록만 ``pipeline.pool --admit --judgment``가 별도 변환 없이 소비한다.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from . import ensemble as ensemble_module
from .data import ID, TARGET, file_sha256
from .judgment import (
    POOL_EQUIVALENCE_BAND_UPPER,
    POOL_JUDGMENT_CONTRACT_VERSION,
    canonical_name_list_sha256,
    check_adoption_eligibility,
    check_canaries,
    mean_gain_of,
)
from .ledger import POOL_PATH, Pool
from .pool_audit import prediction_array_sha256
from .pool_rereview import (
    InputContext,
    PoolScore,
    StrategyEvaluator,
)
from .runs import MlflowRunStore, RunStore, TRACKING_URI


REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDS_PATH = Path("artifacts/folds.parquet")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PoolJudgmentError(RuntimeError):
    """판정 기록을 시작하거나 안전하게 끝낼 수 없는 계약 위반."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoolJudgmentError(message)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_pair(repo_root: Path, path: Path, label: str) -> tuple[Path, Path]:
    _require(not path.is_absolute(), f"{label}은 저장소 상대 경로여야 한다.")
    _require(".." not in path.parts, f"{label}에 상위 경로 이동을 쓸 수 없다.")
    root = repo_root.resolve()
    absolute = (root / path).resolve()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise PoolJudgmentError(f"{label}이 저장소 밖을 가리킨다: {path}") from exc
    return relative, absolute


def _artifact_record(repo_root: Path, path: Path) -> dict[str, Any]:
    relative, absolute = _path_pair(repo_root, path, "입력 산출물 경로")
    _require(absolute.is_file(), f"입력 산출물이 없다: {relative}")
    return {"path": str(relative), "sha256": file_sha256(absolute)}


def _publish_new(path: Path, payload: bytes) -> None:
    """완성 바이트를 기존 파일을 덮지 않고 원자적으로 공개한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PoolJudgmentError(f"변경 불가 출력이 이미 있다: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_existing_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise PoolJudgmentError(f"변경 불가 출력이 이미 있다: {destination}") from exc


@dataclass(frozen=True)
class GenerationRequest:
    judgment_id: str
    action: str
    candidate_run_ids: tuple[str, ...]
    model_lineage_group: str
    selection_description: str
    output_path: Path
    evidence_path: Path
    replaces_run_id: str | None = None
    restores_judgment_path: Path | None = None
    jobs: int = max(1, min(8, os.cpu_count() or 1))


@dataclass(frozen=True)
class GenerationResult:
    state: str
    decision: str
    selected_candidate_run_id: str | None
    nested_oof_delta: float | None
    record_path: Path
    evidence_path: Path
    input_snapshot_path: Path | None


@dataclass(frozen=True)
class _FrozenInput:
    pool_relative: Path
    pool_absolute: Path
    pool_sha256: str
    folds_relative: Path
    folds_absolute: Path
    folds_sha256: str
    combiner_names: tuple[str, ...]
    combiner_names_sha256: str

    def payload(self, pool: Pool) -> dict[str, Any]:
        return {
            "candidate_pool": {
                "path": str(self.pool_relative),
                "sha256": self.pool_sha256,
                "member_count": len(pool.members),
                "members": [
                    {"config": member.config, "run_id": member.run_id}
                    for member in pool.members
                ],
            },
            "folds": {
                "path": str(self.folds_relative),
                "sha256": self.folds_sha256,
            },
            "registered_combiners": {
                "names": list(self.combiner_names),
                "names_sha256": self.combiner_names_sha256,
            },
        }


@dataclass(frozen=True)
class _CandidateRun:
    run_id: str
    config: str
    key: str
    oof_sha256: str
    auc_oof: float
    seeds: tuple[int, ...]
    git_commit: str | None


@dataclass(frozen=True)
class _PreparedInput:
    context: InputContext
    pool: Pool
    candidates: tuple[_CandidateRun, ...]
    before_members: tuple[str, ...]
    candidate_arms: tuple[tuple[str, ...], ...]
    display_name: dict[str, str]
    snapshot_frame: pd.DataFrame
    input_artifacts: dict[str, Any]


@dataclass(frozen=True)
class _Comparison:
    selected_candidate: _CandidateRun
    before_auc: float
    after_auc: float
    delta: float
    before_strategy: str
    after_strategy: str
    outer_fold_delta: dict[str, float]
    outer_fold_wins: int
    evidence: dict[str, Any]


class _Evaluator(Protocol):
    def __enter__(self) -> _Evaluator: ...

    def __exit__(self, exc_type, exc, traceback) -> None: ...

    def evaluate_many(
        self,
        arms: Sequence[tuple[Sequence[str], str | None]],
        *,
        excluded_fold: int | None,
        capture_prediction: bool = False,
    ) -> list[PoolScore]: ...

    def predict_outer(
        self, strategy: str, members: Sequence[str], held_out_fold: int
    ) -> np.ndarray: ...


class _DefaultEvaluator:
    """재심사와 같은 19개 결합 전략 평가기를 후보 판정에 맞춘 adapter."""

    def __init__(self, context: InputContext, jobs: int) -> None:
        self.context = context
        self._evaluator = StrategyEvaluator(context, jobs)

    def __enter__(self) -> _DefaultEvaluator:
        self._evaluator.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._evaluator.__exit__(exc_type, exc, traceback)

    def evaluate_many(self, arms, *, excluded_fold, capture_prediction=False):
        return self._evaluator.evaluate_many(
            arms,
            excluded_fold=excluded_fold,
            capture_prediction=capture_prediction,
        )

    def predict_outer(
        self, strategy: str, members: Sequence[str], held_out_fold: int
    ) -> np.ndarray:
        train = self.context.folds.to_numpy() != held_out_fold
        score = ~train
        combiner = ensemble_module.combiner_for_context(
            strategy,
            fold_of=self.context.folds,
            band_of=self.context.missingness_bands,
        )
        fitted = combiner.fit(
            self.context.predictions.loc[train, list(members)],
            self.context.labels.loc[train],
        )
        prediction = np.asarray(
            fitted.predict(self.context.predictions.loc[score, list(members)]),
            dtype=np.float64,
        )
        _require(
            prediction.shape == (int(score.sum()),)
            and np.isfinite(prediction).all(),
            f"{strategy} 바깥쪽 검증 예측이 올바르지 않다.",
        )
        return prediction


def _validate_request(request: GenerationRequest) -> None:
    _require(
        bool(_SAFE_ID.fullmatch(request.judgment_id)),
        "judgment_id는 영문자나 숫자로 시작하고 영문자, 숫자, 점, 밑줄, 붙임표만 써야 한다.",
    )
    _require(
        request.action in {"admission", "replacement", "restoration"},
        "action은 admission, replacement, restoration 중 하나여야 한다.",
    )
    _require(bool(request.candidate_run_ids), "후보 run_id가 하나 이상 필요하다.")
    _require(
        len(set(request.candidate_run_ids)) == len(request.candidate_run_ids),
        "후보 run_id가 중복됐다.",
    )
    _require(bool(request.model_lineage_group.strip()), "모델 계보 묶음이 필요하다.")
    _require(bool(request.selection_description.strip()), "후보 선택 경위가 필요하다.")
    _require(request.jobs >= 1, "작업자 수는 1 이상이어야 한다.")
    if request.action == "replacement":
        _require(
            bool(request.replaces_run_id),
            "replacement에는 replaces_run_id가 필요하다.",
        )
    else:
        _require(
            request.replaces_run_id is None,
            f"{request.action}에는 replaces_run_id를 쓸 수 없다.",
        )
    if request.action == "restoration":
        _require(
            request.restores_judgment_path is not None,
            "restoration에는 원래 제거 기록 경로가 필요하다.",
        )
        _require(
            len(request.candidate_run_ids) == 1,
            "restoration은 제거 기록의 후보 하나씩만 판정한다.",
        )
    else:
        _require(
            request.restores_judgment_path is None,
            f"{request.action}에는 원래 제거 기록을 쓸 수 없다.",
        )


def _freeze(
    repo_root: Path,
    pool_path: Path,
    folds_path: Path,
    registered_combiner_names_provider: Callable[[], tuple[str, ...]],
) -> tuple[_FrozenInput, Pool]:
    pool_relative, pool_absolute = _path_pair(repo_root, pool_path, "후보 풀 경로")
    folds_relative, folds_absolute = _path_pair(repo_root, folds_path, "fold 경로")
    _require(pool_absolute.is_file(), f"현재 후보 풀 파일이 없다: {pool_relative}")
    _require(folds_absolute.is_file(), f"현재 fold 파일이 없다: {folds_relative}")
    names = tuple(registered_combiner_names_provider())
    _require(len(names) == 19, f"기본 결합 전략은 19개여야 한다: {len(names)}개")
    _require(len(set(names)) == len(names), "기본 결합 전략 이름이 중복됐다.")
    pool = Pool.load(pool_absolute)
    _require(bool(pool.members), "후보 풀이 비어 있다.")
    _require(
        len({member.run_id for member in pool.members}) == len(pool.members),
        "후보 풀 run_id가 중복됐다.",
    )
    _require(
        len({member.config for member in pool.members}) == len(pool.members),
        "후보 풀 config가 중복됐다.",
    )
    return (
        _FrozenInput(
            pool_relative=pool_relative,
            pool_absolute=pool_absolute,
            pool_sha256=file_sha256(pool_absolute),
            folds_relative=folds_relative,
            folds_absolute=folds_absolute,
            folds_sha256=file_sha256(folds_absolute),
            combiner_names=names,
            combiner_names_sha256=canonical_name_list_sha256(names),
        ),
        pool,
    )


def _assert_frozen_unchanged(
    frozen: _FrozenInput,
    registered_combiner_names_provider: Callable[[], tuple[str, ...]],
) -> None:
    current_names = tuple(registered_combiner_names_provider())
    unchanged = (
        frozen.pool_absolute.is_file()
        and file_sha256(frozen.pool_absolute) == frozen.pool_sha256
        and frozen.folds_absolute.is_file()
        and file_sha256(frozen.folds_absolute) == frozen.folds_sha256
        and current_names == frozen.combiner_names
        and canonical_name_list_sha256(current_names)
        == frozen.combiner_names_sha256
    )
    _require(unchanged, "동결 입력이 평가 도중 바뀌어 판정 기록 쓰기를 거부한다.")


def _load_fold_and_data(
    repo_root: Path,
    frozen: _FrozenInput,
    train_path: Path,
    test_path: Path,
) -> tuple[
    pd.Index,
    pd.Series,
    pd.Series,
    pd.Series,
    dict[str, Any],
]:
    train_relative, train_absolute = _path_pair(repo_root, train_path, "학습 자료 경로")
    test_relative, test_absolute = _path_pair(repo_root, test_path, "시험 자료 경로")
    _require(train_absolute.is_file(), f"학습 자료가 없다: {train_relative}")
    _require(test_absolute.is_file(), f"시험 자료가 없다: {test_relative}")

    folds_frame = pd.read_parquet(frozen.folds_absolute)
    _require(list(folds_frame.columns) == [ID, "fold"], "fold 파일 열은 id, fold여야 한다.")
    _require(not folds_frame[ID].duplicated().any(), "fold 파일 id가 중복됐다.")
    _require(
        sorted(int(value) for value in folds_frame["fold"].unique()) == list(range(5)),
        "fold 파일은 0부터 4까지를 모두 포함해야 한다.",
    )
    ids = pd.Index(folds_frame[ID], name=ID)
    folds = pd.Series(
        folds_frame["fold"].to_numpy(dtype=np.int8), index=ids, name="fold"
    )

    train = pd.read_csv(train_absolute)
    test = pd.read_csv(test_absolute)
    _require(ID in train and TARGET in train, "학습 자료에 id 또는 목표값이 없다.")
    _require(ID in test and TARGET not in test, "시험 자료의 id 또는 목표값 열이 올바르지 않다.")
    _require(not train[ID].duplicated().any(), "학습 자료 id가 중복됐다.")
    _require(not test[ID].duplicated().any(), "시험 자료 id가 중복됐다.")
    _require(train[ID].equals(folds_frame[ID]), "학습 자료와 fold id 순서가 다르다.")
    labels = pd.Series(train[TARGET].to_numpy(), index=ids, name=TARGET)
    _require(set(labels.unique()) == {0, 1}, "목표값은 이진 0, 1이어야 한다.")

    train_features = [column for column in train if column not in {ID, TARGET}]
    test_features = [column for column in test if column != ID]
    _require(train_features == test_features, "학습 자료와 시험 자료의 특성 열이 다르다.")
    missing_count = train[train_features].isna().sum(axis=1)
    bands = pd.Series(
        np.select([missing_count <= 1, missing_count <= 3], [0, 1], default=2),
        index=ids,
        dtype=np.int8,
        name="missingness_band",
    )
    artifacts = {
        "candidate_pool": {
            "path": str(frozen.pool_relative),
            "sha256": frozen.pool_sha256,
        },
        "folds": {
            "path": str(frozen.folds_relative),
            "sha256": frozen.folds_sha256,
        },
        "train": {"path": str(train_relative), "sha256": file_sha256(train_absolute)},
        "test": {"path": str(test_relative), "sha256": file_sha256(test_absolute)},
    }
    return ids, folds, labels, bands, artifacts


def _validated_oof(
    run_id: str,
    store: RunStore,
    ids: pd.Index,
    *,
    label: str,
) -> tuple[pd.Series, str]:
    prediction = store.oof_of(run_id)
    _require(isinstance(prediction, pd.Series), f"{label} OOF가 Series가 아니다.")
    _require(not prediction.index.has_duplicates, f"{label} OOF id가 중복됐다.")
    _require(prediction.dtype == np.dtype("float64"), f"{label} OOF가 float64가 아니다.")
    aligned = prediction.reindex(ids)
    _require(not aligned.isna().any(), f"{label} OOF에 fold id가 빠졌다.")
    _require(
        len(prediction) == len(ids) and set(prediction.index) == set(ids),
        f"{label} OOF 행 집합이 fold와 다르다.",
    )
    values = aligned.to_numpy(dtype=np.float64, copy=False)
    _require(np.isfinite(values).all(), f"{label} OOF에 유한하지 않은 값이 있다.")
    return aligned.astype(np.float64), prediction_array_sha256(values)


def _candidate_run(
    run_id: str,
    store: RunStore,
    ids: pd.Index,
    labels: pd.Series,
    frozen: _FrozenInput,
) -> tuple[_CandidateRun, pd.Series]:
    meta = store.facts_of(run_id)
    try:
        config = meta.params["experiment"]
        seeds = tuple(int(value) for value in meta.params["seeds"].split(",") if value)
        git_dirty = meta.tags["git_dirty"] == "True"
        folds_sha256 = meta.tags["sha256.folds"]
        claimed_auc = float(meta.metrics["auc_oof"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PoolJudgmentError(f"후보 run {run_id}의 판정 메타데이터가 불완전하다.") from exc
    _require(bool(config), f"후보 run {run_id}의 config가 없다.")
    prediction, digest = _validated_oof(
        run_id, store, ids, label=f"후보 {config} run {run_id}"
    )
    actual_auc = float(roc_auc_score(labels.to_numpy(), prediction.to_numpy()))
    _require(
        math.isclose(actual_auc, claimed_auc, rel_tol=0.0, abs_tol=1e-9),
        f"후보 {config}의 OOF 재채점과 실행 지표가 다르다.",
    )
    eligibility = check_adoption_eligibility(
        seeds=list(seeds),
        git_dirty=git_dirty,
        folds_sha256=folds_sha256,
        committed_folds_sha256=frozen.folds_sha256,
    )
    _require(eligibility.seeds_ok, f"후보 {config}는 3시드 평균본이 아니다.")
    _require(not eligibility.git_dirty, f"후보 {config}는 git_dirty 실행이다.")
    _require(eligibility.folds_ok, f"후보 {config}의 fold 내용 해시가 다르다.")

    features = {value for value in meta.params.get("features", "").split(",") if value}
    importance = store.importance_of(run_id)
    _require(
        {"feature", "gain"} <= set(importance.columns),
        f"후보 {config}의 계열 무관 중요도 열이 불완전하다.",
    )
    canary = check_canaries(features, mean_gain_of(importance))
    _require(canary.ok, f"후보 {config}의 카나리아 유효성 판정이 실패했다.")
    return (
        _CandidateRun(
            run_id=run_id,
            config=config,
            key=f"candidate:{run_id}",
            oof_sha256=digest,
            auc_oof=actual_auc,
            seeds=seeds,
            git_commit=meta.tags.get("git_commit"),
        ),
        prediction,
    )


def _validate_restoration(
    request: GenerationRequest,
    repo_root: Path,
    frozen: _FrozenInput,
    candidate: _CandidateRun,
) -> None:
    if request.action != "restoration":
        return
    assert request.restores_judgment_path is not None
    relative, absolute = _path_pair(
        repo_root, request.restores_judgment_path, "원래 제거 기록 경로"
    )
    _require(absolute.is_file(), f"원래 제거 기록이 없다: {relative}")
    try:
        original = yaml.safe_load(absolute.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PoolJudgmentError(f"원래 제거 기록을 읽을 수 없다: {relative}") from exc
    _require(isinstance(original, dict), "원래 제거 기록은 키-값 구조여야 한다.")
    _require(
        original.get("contract_version") == POOL_JUDGMENT_CONTRACT_VERSION,
        "원래 제거 기록의 후보 풀 판정 계약 판본이 다르다.",
    )
    original_result = original.get("result")
    _require(
        original.get("status") == "adopted"
        or (
            isinstance(original_result, dict)
            and original_result.get("state") == "adopted"
        ),
        "채택된 원래 제거 기록만 복구 판정에 쓸 수 있다.",
    )
    removed_run_ids = {
        removal.get("member", {}).get("run_id")
        for removal in original.get("removals", [])
        if isinstance(removal, dict)
    }
    change = original.get("change", {})
    if isinstance(change, dict) and change.get("action") == "removal":
        change_candidate = change.get("candidate")
        if isinstance(change_candidate, dict):
            removed_run_ids.add(change_candidate.get("run_id"))
    _require(
        candidate.run_id in removed_run_ids,
        "복구 후보가 원래 제거 기록에 없다.",
    )
    original_frozen = original.get("frozen_input", {})
    original_pool_sha = original_frozen.get("candidate_pool", {}).get("sha256")
    original_names_sha = original_frozen.get("registered_combiners", {}).get(
        "names_sha256"
    )
    _require(
        original_pool_sha != frozen.pool_sha256
        or original_names_sha != frozen.combiner_names_sha256,
        "후보 풀이나 기본 결합 전략 집합이 바뀌지 않아 복구를 다시 판정할 수 없다.",
    )


def _prepare(
    request: GenerationRequest,
    *,
    store: RunStore,
    repo_root: Path,
    frozen: _FrozenInput,
    pool: Pool,
    train_path: Path,
    test_path: Path,
) -> _PreparedInput:
    ids, folds, labels, bands, artifacts = _load_fold_and_data(
        repo_root, frozen, train_path, test_path
    )
    predictions: dict[str, pd.Series] = {}
    display_name: dict[str, str] = {}
    pool_keys: list[str] = []
    pool_artifacts: list[dict[str, Any]] = []
    for member in pool.members:
        key = f"pool:{member.run_id}"
        prediction, digest = _validated_oof(
            member.run_id,
            store,
            ids,
            label=f"현재 풀 {member.config} run {member.run_id}",
        )
        actual_auc = float(roc_auc_score(labels.to_numpy(), prediction.to_numpy()))
        _require(
            math.isclose(actual_auc, member.oof_auc, rel_tol=0.0, abs_tol=1e-9),
            f"현재 풀 {member.config}의 OOF 재채점과 장부 지표가 다르다.",
        )
        predictions[key] = prediction
        display_name[key] = member.config
        pool_keys.append(key)
        pool_artifacts.append(
            {
                "config": member.config,
                "run_id": member.run_id,
                "snapshot_column": key,
                "oof_sha256": digest,
            }
        )

    candidates: list[_CandidateRun] = []
    candidate_artifacts: list[dict[str, Any]] = []
    for run_id in request.candidate_run_ids:
        candidate, prediction = _candidate_run(run_id, store, ids, labels, frozen)
        candidates.append(candidate)
        predictions[candidate.key] = prediction
        display_name[candidate.key] = candidate.config
        candidate_artifacts.append(
            {
                "config": candidate.config,
                "run_id": candidate.run_id,
                "snapshot_column": candidate.key,
                "oof_sha256": candidate.oof_sha256,
                "auc_oof": candidate.auc_oof,
                "seeds": list(candidate.seeds),
                "git_commit": candidate.git_commit,
            }
        )
    _require(
        len({candidate.config for candidate in candidates}) == len(candidates),
        "여러 후보 변형의 config가 중복됐다.",
    )
    if request.action == "restoration":
        _validate_restoration(request, repo_root, frozen, candidates[0])
    pool_run_ids = {member.run_id for member in pool.members}
    _require(
        not any(candidate.run_id in pool_run_ids for candidate in candidates),
        "후보 run이 이미 현재 풀에 있다.",
    )

    before_members = tuple(pool_keys)
    replaced_key = None
    if request.action == "replacement":
        replaced_key = next(
            (
                key
                for key, member in zip(pool_keys, pool.members, strict=True)
                if member.run_id == request.replaces_run_id
            ),
            None,
        )
        _require(replaced_key is not None, "교체 대상 run이 현재 후보 풀에 없다.")
    base_after = tuple(key for key in before_members if key != replaced_key)
    candidate_arms = tuple(base_after + (candidate.key,) for candidate in candidates)
    for candidate, arm in zip(candidates, candidate_arms, strict=True):
        configs = [display_name[key] for key in arm]
        _require(
            len(configs) == len(set(configs)),
            f"후보 {candidate.config}을 넣은 평가 사본의 config가 중복됐다.",
        )

    matrix = pd.DataFrame(predictions, index=ids).astype(np.float64)
    context = InputContext(
        predictions=matrix,
        labels=labels,
        folds=folds,
        missingness_bands=bands,
        ledger={"strategies": {"included": list(frozen.combiner_names)}},
        baseline={},
        source_hashes={},
        prediction_file_sha256="",
        member_prediction_sha256={
            record["snapshot_column"]: record["oof_sha256"]
            for record in [*pool_artifacts, *candidate_artifacts]
        },
    )
    snapshot = pd.DataFrame(
        {
            ID: ids.to_numpy(),
            "fold": folds.to_numpy(),
            TARGET: labels.to_numpy(),
            "missingness_band": bands.to_numpy(),
            **{key: matrix[key].to_numpy() for key in matrix},
        }
    )
    artifacts["pool_member_oof"] = pool_artifacts
    artifacts["candidate_oof"] = candidate_artifacts
    if request.restores_judgment_path is not None:
        artifacts["original_removal_judgment"] = _artifact_record(
            repo_root, request.restores_judgment_path
        )
    return _PreparedInput(
        context=context,
        pool=pool,
        candidates=tuple(candidates),
        before_members=before_members,
        candidate_arms=candidate_arms,
        display_name=display_name,
        snapshot_frame=snapshot,
        input_artifacts=artifacts,
    )


def _validate_score(score: PoolScore, names: tuple[str, ...]) -> None:
    _require(
        tuple(score.strategy_auc) == names,
        "평가 팔에서 기본 결합 전략 19개의 이름이나 순서가 다르다.",
    )
    _require(
        tuple(score.strategy_fold_auc) == names,
        "평가 팔에서 분할별 기본 결합 전략 결과가 불완전하다.",
    )
    _require(score.best_strategy in names, "평가 팔의 최선 결합 전략이 등록부에 없다.")
    _require(math.isfinite(score.best_auc), "평가 팔의 최선 AUC가 유한하지 않다.")
    _require(
        all(math.isfinite(float(value)) for value in score.strategy_auc.values()),
        "평가 팔의 결합 전략 AUC에 유한하지 않은 값이 있다.",
    )
    _require(
        math.isclose(
            score.best_auc,
            float(score.strategy_auc[score.best_strategy]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "평가 팔의 최선 전략과 최선 AUC가 맞지 않는다.",
    )
    expected_folds = set(score.best_fold_auc)
    _require(bool(expected_folds), "평가 팔의 분할별 AUC가 없다.")
    _require(
        all(
            set(fold_auc) == expected_folds
            and all(math.isfinite(float(value)) for value in fold_auc.values())
            for fold_auc in score.strategy_fold_auc.values()
        ),
        "평가 팔의 결합 전략별 분할 AUC가 불완전하다.",
    )


def _score_payload(score: PoolScore, display_name: dict[str, str]) -> dict[str, Any]:
    return {
        "members": [display_name[key] for key in score.members],
        "member_count": len(score.members),
        "best_strategy": score.best_strategy,
        "best_auc": score.best_auc,
        "best_fold_auc": score.best_fold_auc,
        "strategy_auc": score.strategy_auc,
        "strategy_fold_auc": score.strategy_fold_auc,
    }


def _best_candidate_index(scores: Sequence[PoolScore]) -> int:
    _require(bool(scores), "후보 평가 팔이 없다.")
    return max(range(len(scores)), key=lambda index: (scores[index].best_auc, -index))


def _fold_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    _require(set(before) == set(after) == {str(fold) for fold in range(5)}, "분할 AUC가 0부터 4까지 완전하지 않다.")
    return {key: float(after[key] - before[key]) for key in sorted(before, key=int)}


def _evaluate_single(prepared: _PreparedInput, evaluator: _Evaluator) -> _Comparison:
    scores = evaluator.evaluate_many(
        [
            (prepared.before_members, None),
            (prepared.candidate_arms[0], None),
        ],
        excluded_fold=None,
        capture_prediction=True,
    )
    _require(len(scores) == 2, "단일 후보 포함 및 제외 평가 결과 수가 다르다.")
    for score in scores:
        _validate_score(score, tuple(prepared.context.ledger["strategies"]["included"]))
    before, after = scores
    _require(
        before.prediction is not None and after.prediction is not None,
        "단일 후보 대조의 최선 전략 예측이 없다.",
    )
    labels = prepared.context.labels.to_numpy()
    before_auc = float(roc_auc_score(labels, before.prediction))
    after_auc = float(roc_auc_score(labels, after.prediction))
    _require(
        math.isclose(before_auc, before.best_auc, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(after_auc, after.best_auc, rel_tol=0.0, abs_tol=1e-15),
        "단일 후보 대조의 저장 예측 재채점과 최선 AUC가 다르다.",
    )
    fold_delta = _fold_deltas(before.best_fold_auc, after.best_fold_auc)
    delta = float(after_auc - before_auc)
    return _Comparison(
        selected_candidate=prepared.candidates[0],
        before_auc=before_auc,
        after_auc=after_auc,
        delta=delta,
        before_strategy=before.best_strategy,
        after_strategy=after.best_strategy,
        outer_fold_delta=fold_delta,
        outer_fold_wins=sum(value > 0.0 for value in fold_delta.values()),
        evidence={
            "mode": "direct_inclusion_or_replacement",
            "before": _score_payload(before, prepared.display_name),
            "after": _score_payload(after, prepared.display_name),
            "outer_fold_choices": {},
            "nested_prediction_sha256": {
                "before": prediction_array_sha256(before.prediction),
                "after": prediction_array_sha256(after.prediction),
            },
        },
    )


def _evaluate_selection(prepared: _PreparedInput, evaluator: _Evaluator) -> _Comparison:
    names = tuple(prepared.context.ledger["strategies"]["included"])
    full_scores = evaluator.evaluate_many(
        [(prepared.before_members, None)]
        + [(arm, None) for arm in prepared.candidate_arms],
        excluded_fold=None,
    )
    _require(
        len(full_scores) == 1 + len(prepared.candidates),
        "전체 OOF 후보 선택 평가 결과 수가 다르다.",
    )
    for score in full_scores:
        _validate_score(score, names)
    selected_index = _best_candidate_index(full_scores[1:])
    selected_candidate = prepared.candidates[selected_index]
    full_before = full_scores[0]
    full_after = full_scores[1 + selected_index]

    nested_before = np.full(len(prepared.context.labels), np.nan, dtype=np.float64)
    nested_after = np.full(len(prepared.context.labels), np.nan, dtype=np.float64)
    fold_delta: dict[str, float] = {}
    outer_choices: dict[str, Any] = {}
    for fold in range(5):
        inner_scores = evaluator.evaluate_many(
            [(prepared.before_members, None)]
            + [(arm, None) for arm in prepared.candidate_arms],
            excluded_fold=fold,
        )
        _require(
            len(inner_scores) == 1 + len(prepared.candidates),
            f"바깥쪽 검증 분할 {fold}의 후보 선택 결과 수가 다르다.",
        )
        for score in inner_scores:
            _validate_score(score, names)
        inner_selected = _best_candidate_index(inner_scores[1:])
        before_score = inner_scores[0]
        after_score = inner_scores[1 + inner_selected]
        candidate = prepared.candidates[inner_selected]
        before_prediction = evaluator.predict_outer(
            before_score.best_strategy, prepared.before_members, fold
        )
        after_prediction = evaluator.predict_outer(
            after_score.best_strategy, prepared.candidate_arms[inner_selected], fold
        )
        mask = prepared.context.folds.to_numpy() == fold
        _require(
            before_prediction.shape == after_prediction.shape == (int(mask.sum()),),
            f"바깥쪽 검증 분할 {fold}의 예측 행 수가 다르다.",
        )
        _require(
            np.isfinite(before_prediction).all()
            and np.isfinite(after_prediction).all(),
            f"바깥쪽 검증 분할 {fold}의 예측이 유한하지 않다.",
        )
        nested_before[mask] = before_prediction
        nested_after[mask] = after_prediction
        fold_labels = prepared.context.labels.to_numpy()[mask]
        before_auc = float(roc_auc_score(fold_labels, before_prediction))
        after_auc = float(roc_auc_score(fold_labels, after_prediction))
        fold_delta[str(fold)] = after_auc - before_auc
        outer_choices[str(fold)] = {
            "candidate_run_id": candidate.run_id,
            "candidate_config": candidate.config,
            "before_strategy": before_score.best_strategy,
            "after_strategy": after_score.best_strategy,
            "inner_before_auc": before_score.best_auc,
            "inner_after_auc": after_score.best_auc,
            "strategy_count": len(names),
        }
    _require(
        np.isfinite(nested_before).all() and np.isfinite(nested_after).all(),
        "선택 절차 대조 예측이 전 행을 덮지 않는다.",
    )
    labels = prepared.context.labels.to_numpy()
    before_auc = float(roc_auc_score(labels, nested_before))
    after_auc = float(roc_auc_score(labels, nested_after))
    return _Comparison(
        selected_candidate=selected_candidate,
        before_auc=before_auc,
        after_auc=after_auc,
        delta=after_auc - before_auc,
        before_strategy=full_before.best_strategy,
        after_strategy=full_after.best_strategy,
        outer_fold_delta=fold_delta,
        outer_fold_wins=sum(value > 0.0 for value in fold_delta.values()),
        evidence={
            "mode": "nested_selection",
            "before": _score_payload(full_before, prepared.display_name),
            "after": _score_payload(full_after, prepared.display_name),
            "full_candidate_selection": [
                {
                    "candidate_run_id": candidate.run_id,
                    "candidate_config": candidate.config,
                    "score": _score_payload(score, prepared.display_name),
                    "selected": index == selected_index,
                }
                for index, (candidate, score) in enumerate(
                    zip(prepared.candidates, full_scores[1:], strict=True)
                )
            ],
            "outer_fold_choices": outer_choices,
            "nested_prediction_sha256": {
                "before": prediction_array_sha256(nested_before),
                "after": prediction_array_sha256(nested_after),
            },
        },
    )


def _selection_payload(
    request: GenerationRequest,
    candidates: Sequence[_CandidateRun] | None,
) -> dict[str, Any]:
    nested = len(request.candidate_run_ids) > 1
    return {
        "kind": "nested_selection" if nested else "precommitted_single",
        "description": request.selection_description,
        "rule": (
            "각 평가 범위에서 후보 포함 풀의 기본 결합 전략 19개 중 최고 nested OOF AUC가 가장 높은 후보를 고른다."
            if nested
            else "사전 고정한 후보의 포함 전후를 직접 비교한다."
        ),
        "tie_break": (
            "후보 run_id 입력 순서가 먼저이고 결합 전략 등록 순서가 그다음이다."
            if nested
            else "결합 전략 AUC가 같으면 등록 순서가 앞선 전략을 고른다."
        ),
        "candidates": [
            {
                "run_id": candidate.run_id,
                "config": candidate.config,
                "oof_sha256": candidate.oof_sha256,
            }
            for candidate in (candidates or ())
        ],
    }


def _change_payload(
    request: GenerationRequest,
    candidate: _CandidateRun | None,
    repo_root: Path,
    *,
    allow_missing_restoration: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": request.action,
        "candidate": {
            "run_id": candidate.run_id if candidate is not None else request.candidate_run_ids[0],
            "config": candidate.config if candidate is not None else "unknown",
            "model_lineage_group": request.model_lineage_group,
        },
        "replaces_run_id": request.replaces_run_id,
    }
    if request.restores_judgment_path is not None:
        if allow_missing_restoration:
            try:
                relative, absolute = _path_pair(
                    repo_root,
                    request.restores_judgment_path,
                    "원래 제거 기록 경로",
                )
                payload["restores_judgment"] = {
                    "path": str(relative),
                    "sha256": file_sha256(absolute) if absolute.is_file() else None,
                }
            except PoolJudgmentError:
                payload["restores_judgment"] = {
                    "path": str(request.restores_judgment_path),
                    "sha256": None,
                }
        else:
            payload["restores_judgment"] = _artifact_record(
                repo_root, request.restores_judgment_path
            )
    return payload


def _result_for(request: GenerationRequest, delta: float) -> tuple[str, str, str]:
    if delta > 0.0:
        return (
            "adopted",
            "replace" if request.action == "replacement" else "admit",
            "전체 nested OOF AUC 차이가 엄격히 양수다.",
        )
    return (
        "rejected",
        "do_not_replace" if request.action == "replacement" else "do_not_admit",
        "전체 nested OOF AUC 차이가 0 이하라 양의 기여를 확인하지 못했다.",
    )


def _record_payload(
    request: GenerationRequest,
    frozen: _FrozenInput,
    pool: Pool,
    comparison: _Comparison,
    candidates: Sequence[_CandidateRun],
    evidence_relative: Path,
    evidence_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    state, decision, reason = _result_for(request, comparison.delta)
    boundary = 0.0 < comparison.delta <= POOL_EQUIVALENCE_BAND_UPPER
    nested_selection = len(request.candidate_run_ids) > 1
    before: dict[str, Any] = {
        "strategy": comparison.before_strategy,
        "auc": comparison.before_auc,
        "strategy_role": (
            "full_oof_final_selection_reference"
            if nested_selection
            else "direct_nested_oof"
        ),
    }
    after: dict[str, Any] = {
        "strategy": comparison.after_strategy,
        "auc": comparison.after_auc,
        "strategy_role": (
            "full_oof_final_selection_reference"
            if nested_selection
            else "direct_nested_oof"
        ),
    }
    if nested_selection:
        choices = comparison.evidence["outer_fold_choices"]
        before["outer_fold_strategy"] = {
            fold: choice["before_strategy"] for fold, choice in choices.items()
        }
        after["outer_fold_strategy"] = {
            fold: choice["after_strategy"] for fold, choice in choices.items()
        }
    return {
        "schema_version": 1,
        "judgment_id": request.judgment_id,
        "contract_version": POOL_JUDGMENT_CONTRACT_VERSION,
        "created_at": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "change": _change_payload(request, comparison.selected_candidate, repo_root),
        "selection": _selection_payload(request, candidates),
        "frozen_input": frozen.payload(pool),
        "nested_oof_comparison": {
            "before": before,
            "after": after,
            "delta": comparison.delta,
            "outer_fold_delta": comparison.outer_fold_delta,
            "outer_fold_wins": comparison.outer_fold_wins,
            "boundary_contribution": boundary,
        },
        "evidence": {
            "path": str(evidence_relative),
            "sha256": evidence_sha256,
        },
        "result": {"state": state, "decision": decision, "reason": reason},
    }


def _indeterminate_payloads(
    request: GenerationRequest,
    frozen: _FrozenInput,
    pool: Pool,
    reason: str,
    evidence_relative: Path,
    repo_root: Path,
) -> tuple[bytes, dict[str, Any]]:
    evidence = {
        "schema_version": 1,
        "judgment_id": request.judgment_id,
        "contract_version": POOL_JUDGMENT_CONTRACT_VERSION,
        "input_artifacts": {
            "candidate_pool": {
                "path": str(frozen.pool_relative),
                "sha256": frozen.pool_sha256,
            },
            "folds": {
                "path": str(frozen.folds_relative),
                "sha256": frozen.folds_sha256,
            },
        },
        "evaluation": {"state": "indeterminate", "reason": reason},
    }
    evidence_bytes = _json_bytes(evidence)
    record = {
        "schema_version": 1,
        "judgment_id": request.judgment_id,
        "contract_version": POOL_JUDGMENT_CONTRACT_VERSION,
        "created_at": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "change": _change_payload(
            request, None, repo_root, allow_missing_restoration=True
        ),
        "selection": _selection_payload(request, None),
        "frozen_input": frozen.payload(pool),
        "evidence": {
            "path": str(evidence_relative),
            "sha256": _sha256_bytes(evidence_bytes),
        },
        "result": {
            "state": "indeterminate",
            "decision": "no_change",
            "reason": reason,
        },
    }
    return evidence_bytes, record


def _cleanup_published(paths: Sequence[Path]) -> None:
    for path in reversed(paths):
        path.unlink(missing_ok=True)
    parents = sorted(
        {path.parent for path in paths}, key=lambda path: len(path.parts), reverse=True
    )
    for parent in parents:
        try:
            parent.rmdir()
        except OSError:
            pass


def generate_pool_judgment(
    request: GenerationRequest,
    *,
    store: RunStore | None = None,
    repo_root: Path = REPO_ROOT,
    pool_path: Path = POOL_PATH,
    folds_path: Path = FOLDS_PATH,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    evaluator_factory: Callable[[InputContext, int], _Evaluator] = _DefaultEvaluator,
    registered_combiner_names_provider: Callable[[], tuple[str, ...]] = (
        lambda: ensemble_module.DEFAULT_COMBINER_NAMES
    ),
) -> GenerationResult:
    """한 후보 판정을 실행하고 변경 불가 evidence와 YAML 기록을 함께 만든다."""
    _validate_request(request)
    repo_root = repo_root.resolve()
    output_relative, output_absolute = _path_pair(
        repo_root, request.output_path, "판정 기록 경로"
    )
    evidence_relative, evidence_absolute = _path_pair(
        repo_root, request.evidence_path, "근거 산출물 경로"
    )
    input_relative = evidence_relative.parent / "inputs.parquet"
    input_absolute = repo_root / input_relative
    for path in (output_absolute, evidence_absolute, input_absolute):
        _require(not path.exists(), f"변경 불가 출력이 이미 있다: {path}")

    frozen, pool = _freeze(
        repo_root, pool_path, folds_path, registered_combiner_names_provider
    )
    store = store or MlflowRunStore()
    with tempfile.TemporaryDirectory(prefix=".pool-judgment-", dir=repo_root) as temporary:
        temporary_root = Path(temporary)
        temporary_snapshot = temporary_root / "inputs.parquet"
        try:
            prepared = _prepare(
                request,
                store=store,
                repo_root=repo_root,
                frozen=frozen,
                pool=pool,
                train_path=train_path,
                test_path=test_path,
            )
            prepared.snapshot_frame.to_parquet(temporary_snapshot, index=False)
            snapshot_sha256 = file_sha256(temporary_snapshot)
            prepared.input_artifacts["evaluation_snapshot"] = {
                "path": str(input_relative),
                "sha256": snapshot_sha256,
                "rows": len(prepared.snapshot_frame),
                "columns": list(prepared.snapshot_frame.columns),
            }
            with evaluator_factory(prepared.context, request.jobs) as evaluator:
                comparison = (
                    _evaluate_single(prepared, evaluator)
                    if len(prepared.candidates) == 1
                    else _evaluate_selection(prepared, evaluator)
                )
            state, decision, _reason = _result_for(request, comparison.delta)
            evidence = {
                "schema_version": 1,
                "judgment_id": request.judgment_id,
                "contract_version": POOL_JUDGMENT_CONTRACT_VERSION,
                "input_artifacts": prepared.input_artifacts,
                "selection": _selection_payload(request, prepared.candidates),
                "evaluation": comparison.evidence
                | {
                    "nested_oof_comparison": {
                        "before_auc": comparison.before_auc,
                        "after_auc": comparison.after_auc,
                        "delta": comparison.delta,
                        "outer_fold_delta": comparison.outer_fold_delta,
                        "outer_fold_wins": comparison.outer_fold_wins,
                        "boundary_contribution": (
                            0.0
                            < comparison.delta
                            <= POOL_EQUIVALENCE_BAND_UPPER
                        ),
                    }
                },
            }
            evidence_bytes = _json_bytes(evidence)
            record = _record_payload(
                request,
                frozen,
                pool,
                comparison,
                prepared.candidates,
                evidence_relative,
                _sha256_bytes(evidence_bytes),
                repo_root,
            )
            input_snapshot_path: Path | None = input_absolute
        # 계산기와 실행 저장소의 오류 종류가 달라도 장부 변경은 같은 판정 불가로 닫는다.
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            evidence_bytes, record = _indeterminate_payloads(
                request,
                frozen,
                pool,
                reason,
                evidence_relative,
                repo_root,
            )
            state = "indeterminate"
            decision = "no_change"
            comparison = None
            input_snapshot_path = None

        _assert_frozen_unchanged(frozen, registered_combiner_names_provider)
        published: list[Path] = []
        try:
            if input_snapshot_path is not None:
                _publish_existing_new(temporary_snapshot, input_absolute)
                published.append(input_absolute)
            _publish_new(evidence_absolute, evidence_bytes)
            published.append(evidence_absolute)
            _publish_new(output_absolute, _yaml_bytes(record))
            published.append(output_absolute)
        except Exception:
            _cleanup_published(published)
            raise

    return GenerationResult(
        state=state,
        decision=decision,
        selected_candidate_run_id=(
            comparison.selected_candidate.run_id if comparison is not None else None
        ),
        nested_oof_delta=comparison.delta if comparison is not None else None,
        record_path=output_absolute,
        evidence_path=evidence_absolute,
        input_snapshot_path=input_snapshot_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgment-id", required=True)
    parser.add_argument(
        "--candidate-run",
        action="append",
        required=True,
        help="평가할 후보 run_id. 여러 변형이면 여러 번 지정한다.",
    )
    parser.add_argument("--model-lineage-group", required=True)
    parser.add_argument("--selection-description", required=True)
    parser.add_argument(
        "--action",
        choices=("admission", "replacement", "restoration"),
        default="admission",
    )
    parser.add_argument("--replaces-run-id")
    parser.add_argument("--restores-judgment", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument(
        "--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    args = parser.parse_args()
    output = args.output or Path("artifacts/judgments") / f"{args.judgment_id}.yaml"
    evidence = args.evidence or Path("run-logs/pool-judgments") / args.judgment_id / "evidence.json"
    request = GenerationRequest(
        judgment_id=args.judgment_id,
        action=args.action,
        candidate_run_ids=tuple(args.candidate_run),
        model_lineage_group=args.model_lineage_group,
        selection_description=args.selection_description,
        output_path=output,
        evidence_path=evidence,
        replaces_run_id=args.replaces_run_id,
        restores_judgment_path=args.restores_judgment,
        jobs=args.jobs,
    )
    try:
        result = generate_pool_judgment(
            request, store=MlflowRunStore(tracking_uri=args.tracking_uri)
        )
    except PoolJudgmentError as exc:
        sys.exit(str(exc))
    print(f"판정 상태: {result.state}")
    print(f"판정 기록: {result.record_path.relative_to(REPO_ROOT)}")
    print(f"근거 산출물: {result.evidence_path.relative_to(REPO_ROOT)}")
    if result.input_snapshot_path is not None:
        print(f"평가 입력 사본: {result.input_snapshot_path.relative_to(REPO_ROOT)}")
    if result.nested_oof_delta is not None:
        print(f"전체 nested OOF 차이: {result.nested_oof_delta:+.12f}")
    if result.state == "indeterminate":
        sys.exit("판정 불가: 기록의 result.reason을 확인할 것.")


if __name__ == "__main__":
    main()
