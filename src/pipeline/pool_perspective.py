"""후보 풀의 학습기 계열과 정보 관점 제거 nested OOF 진단. (#310)

분류 장부는 제거 결과를 보기 전에 후보 풀 내용 해시와 구성원 순서를 고정한다.
진단은 등록된 결합 전략 전부를 전체 풀과 대표 정보 관점별 제외 풀에 똑같이
적용하고, 각 경우의 최고 nested OOF를 선택한다.

사용법:
    uv run python -m pipeline.pool_perspective \
        --tracking-uri sqlite:////absolute/path/to/mlflow.db \
        --reference-run 33bde7e429d8407cb8b46b4737450265
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .data import ID, TRAIN_PATH, file_sha256, labels
from .ensemble import (
    BaggedGreedyRankMeanCombiner,
    COMBINER_REGISTRY,
    Combiner,
    CombinerConvergenceError,
    NestedEvaluation,
    evaluate_nested,
    member_matrix,
)
from .judgment import FOLDS_PATH
from .ledger import POOL_PATH, Pool
from .runs import TRACKING_URI, MlflowRunStore, RunStore

DEFAULT_MAP_PATH = Path("artifacts/pool-perspectives.yaml")
DEFAULT_REPORT_PATH = Path("docs/research/pool-perspective-diagnostic.md")
DEFAULT_JSON_PATH = Path("docs/research/pool-perspective-diagnostic.json")
CONFIG_DIR = Path("configs")
OPENING_DELTA = 0.00002
OPENING_FOLD_WINS = 3


class PerspectiveDiagnosticError(Exception):
    """고정 입력 또는 진단 결과가 계약을 만족하지 못한 상태."""


@dataclass(frozen=True)
class PerspectiveDefinition:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class MemberPerspective:
    config: str
    run_id: str
    model_family: str
    primary: str
    secondary: tuple[str, ...]


@dataclass(frozen=True)
class FrozenPerspectiveMap:
    pool_path: Path
    pool_sha256: str
    member_count: int
    perspectives: dict[str, PerspectiveDefinition]
    members: tuple[MemberPerspective, ...]
    sha256: str


@dataclass(frozen=True)
class FoldSummary:
    fold: int
    auc: float
    weights: dict[str, float]


@dataclass(frozen=True)
class EvaluationSummary:
    name: str
    nested_auc: float
    folds: tuple[FoldSummary, ...]


@dataclass(frozen=True)
class MemberQuality:
    config: str
    auc: float
    nearest: str | None
    nearest_spearman: float | None


@dataclass(frozen=True)
class MemberSelection:
    selected: int
    fold_total: int
    mean_weight: float


@dataclass(frozen=True)
class SelectionShift:
    config: str
    selected_before: int
    selected_after: int
    selected_delta: int
    weight_before: float
    weight_after: float
    weight_delta: float


@dataclass(frozen=True)
class PerspectiveComparison:
    perspective: str
    excluded: tuple[str, ...]
    model_families: tuple[str, ...]
    best: EvaluationSummary
    loss: float
    outer_worse: int
    outer_better: int
    outer_tied: int
    excluded_selection: tuple[tuple[str, MemberSelection], ...]
    shifts: tuple[SelectionShift, ...]
    opens_experiment: bool


@dataclass(frozen=True)
class ReferenceRun:
    run_id: str
    strategy: str | None
    member_count: int | None
    member_configs: tuple[str, ...]
    auc: float | None
    fold_aucs: dict[int, float]
    matches_pool: bool


@dataclass(frozen=True)
class DiagnosticResult:
    frozen_map: FrozenPerspectiveMap
    member_quality: tuple[MemberQuality, ...]
    registered_strategy_names: tuple[str, ...]
    bagged_workers: int
    baseline_evaluations: tuple[EvaluationSummary, ...]
    group_evaluations: dict[str, tuple[EvaluationSummary, ...]]
    baseline_best: EvaluationSummary
    comparisons: tuple[PerspectiveComparison, ...]
    reference: ReferenceRun | None
    elapsed_seconds: float
    peak_rss_bytes: int
    train_sha256: str
    folds_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PerspectiveDiagnosticError(f"{field}는 mapping이어야 한다.")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PerspectiveDiagnosticError(f"{field}는 list여야 한다.")
    return value


def load_frozen_map(
    path: Path = DEFAULT_MAP_PATH,
    *,
    pool_path: Path = POOL_PATH,
    config_dir: Path = CONFIG_DIR,
) -> FrozenPerspectiveMap:
    """분류 장부와 현재 풀, 실행 설정의 정합성을 확인해 불변 입력으로 만든다."""
    try:
        record = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise PerspectiveDiagnosticError(f"분류 장부를 읽지 못했다: {path}: {exc}") from exc
    record = _require_dict(record, "분류 장부")
    pool_record = _require_dict(record.get("pool"), "pool")
    declared_pool_path = Path(str(pool_record.get("path", "")))
    if declared_pool_path != pool_path:
        raise PerspectiveDiagnosticError(
            f"분류 장부 pool.path {declared_pool_path} != 실행 경로 {pool_path}"
        )
    actual_pool_sha256 = _sha256(pool_path)
    declared_pool_sha256 = str(pool_record.get("sha256", ""))
    if declared_pool_sha256 != actual_pool_sha256:
        raise PerspectiveDiagnosticError(
            f"후보 풀 내용 해시 변경: {actual_pool_sha256} != {declared_pool_sha256}"
        )
    try:
        declared_count = int(pool_record["member_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PerspectiveDiagnosticError("pool.member_count가 올바른 정수가 아니다.") from exc

    perspective_records = _require_dict(record.get("perspectives"), "perspectives")
    perspectives: dict[str, PerspectiveDefinition] = {}
    for key, raw in perspective_records.items():
        raw = _require_dict(raw, f"perspectives.{key}")
        label = str(raw.get("label", "")).strip()
        description = str(raw.get("description", "")).strip()
        if not label or not description:
            raise PerspectiveDiagnosticError(f"관점 {key}의 label과 description이 필요하다.")
        perspectives[str(key)] = PerspectiveDefinition(str(key), label, description)
    if not perspectives:
        raise PerspectiveDiagnosticError("정보 관점이 비어 있다.")

    member_records = _require_list(record.get("members"), "members")
    members: list[MemberPerspective] = []
    for index, raw in enumerate(member_records):
        raw = _require_dict(raw, f"members[{index}]")
        config = str(raw.get("config", ""))
        run_id = str(raw.get("run_id", ""))
        model_family = str(raw.get("model_family", ""))
        primary = str(raw.get("primary", ""))
        secondary_raw = _require_list(raw.get("secondary"), f"members[{index}].secondary")
        secondary = tuple(str(item) for item in secondary_raw)
        if not config or not run_id or not model_family:
            raise PerspectiveDiagnosticError(f"members[{index}]의 식별 필드가 비어 있다.")
        if primary not in perspectives:
            raise PerspectiveDiagnosticError(f"{config}의 대표 관점 {primary}가 정의되지 않았다.")
        unknown = [key for key in secondary if key not in perspectives]
        if unknown:
            raise PerspectiveDiagnosticError(f"{config}의 알 수 없는 보조 관점: {unknown}")
        if primary in secondary or len(set(secondary)) != len(secondary):
            raise PerspectiveDiagnosticError(f"{config}의 대표·보조 관점이 중복된다.")
        config_path = config_dir / f"{config}.yaml"
        try:
            config_record = yaml.safe_load(config_path.read_text())
            actual_family = config_record["model"]["kind"]
        except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
            raise PerspectiveDiagnosticError(f"설정을 읽지 못했다: {config_path}: {exc}") from exc
        if actual_family != model_family:
            raise PerspectiveDiagnosticError(
                f"{config} 모델 계열 {model_family} != 설정 model.kind {actual_family}"
            )
        members.append(
            MemberPerspective(config, run_id, model_family, primary, secondary)
        )

    pool = Pool.load(pool_path)
    actual_members = [(member.config, member.run_id) for member in pool.members]
    declared_members = [(member.config, member.run_id) for member in members]
    if declared_count != len(actual_members) or len(members) != len(actual_members):
        raise PerspectiveDiagnosticError(
            f"후보 풀 구성원 수 변경: 실제 {len(actual_members)}, "
            f"분류 장부 {declared_count}/{len(members)}"
        )
    if declared_members != actual_members:
        raise PerspectiveDiagnosticError("분류 장부의 구성원 순서 또는 run_id가 후보 풀과 다르다.")
    if len(set(declared_members)) != len(declared_members):
        raise PerspectiveDiagnosticError("분류 장부의 구성원이 중복된다.")
    used_primary = {member.primary for member in members}
    unused = set(perspectives) - used_primary
    if unused:
        raise PerspectiveDiagnosticError(f"대표 관점으로 쓰이지 않은 정의가 있다: {sorted(unused)}")

    return FrozenPerspectiveMap(
        pool_path=pool_path,
        pool_sha256=actual_pool_sha256,
        member_count=declared_count,
        perspectives=perspectives,
        members=tuple(members),
        sha256=_sha256(path),
    )


def _summarize(evaluation: NestedEvaluation) -> EvaluationSummary:
    return EvaluationSummary(
        name=evaluation.name,
        nested_auc=evaluation.nested_auc,
        folds=tuple(
            FoldSummary(outcome.fold, outcome.auc, dict(outcome.summary))
            for outcome in evaluation.folds
        ),
    )


def evaluate_all(
    combiners: Iterable[Combiner],
    matrix: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    *,
    existing: tuple[EvaluationSummary, ...] = (),
    processed: tuple[str, ...] = (),
    checkpoint: Callable[[tuple[EvaluationSummary, ...], tuple[str, ...]], None]
    | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[EvaluationSummary, ...]:
    """등록 순서를 유지해 모든 결합 전략을 평가하고 미수렴만 제외한다."""
    summaries = list(existing)
    processed_names = list(processed)
    processed_set = set(processed_names)
    for combiner in combiners:
        if combiner.name in processed_set:
            continue
        if progress is not None:
            progress(f"전략 시작: {combiner.name}")
        try:
            summary = _summarize(evaluate_nested(combiner, matrix, fold_of, y))
        except CombinerConvergenceError as exc:
            if progress is not None:
                progress(f"전략 제외: {combiner.name}: {exc}")
        else:
            summaries.append(summary)
            if progress is not None:
                progress(f"전략 완료: {combiner.name} {summary.nested_auc:.12f}")
        processed_names.append(combiner.name)
        processed_set.add(combiner.name)
        if checkpoint is not None:
            checkpoint(tuple(summaries), tuple(processed_names))
    if not summaries:
        raise PerspectiveDiagnosticError("점수를 낸 결합 전략이 없다.")
    return tuple(summaries)


def diagnostic_combiners(bagged_workers: int) -> tuple[Combiner, ...]:
    """전략 의미는 유지하고 배깅 작업 스레드 수만 실행 환경에 맞춘다."""
    if bagged_workers < 1:
        raise PerspectiveDiagnosticError("배깅 작업 스레드 수는 1 이상이어야 한다.")
    combiners: list[Combiner] = []
    for combiner in COMBINER_REGISTRY.values():
        if isinstance(combiner, BaggedGreedyRankMeanCombiner):
            combiners.append(
                BaggedGreedyRankMeanCombiner(
                    bags=combiner.bags,
                    sample_fraction=combiner.sample_fraction,
                    seed=combiner.seed,
                    workers=bagged_workers,
                    max_members=combiner.max_members,
                    min_improvement=combiner.min_improvement,
                    name=combiner.name,
                )
            )
        else:
            combiners.append(combiner)
    return tuple(combiners)


def best_evaluation(evaluations: Iterable[EvaluationSummary]) -> EvaluationSummary:
    """기존 결합 CLI처럼 등록 순서를 동률 결정 순서로 쓰는 최고점 선택."""
    return max(evaluations, key=lambda evaluation: evaluation.nested_auc)


def measure_member_quality(matrix: pd.DataFrame, y: pd.Series) -> tuple[MemberQuality, ...]:
    """단독 OOF AUC와 풀 안의 가장 가까운 스피어만 순위 상관을 계산한다."""
    ranked = matrix.rank(method="average")
    correlation = ranked.corr(method="pearson")
    quality: list[MemberQuality] = []
    for config in matrix:
        others = correlation.loc[config].drop(config)
        nearest = str(others.idxmax()) if len(others) else None
        nearest_spearman = float(others.loc[nearest]) if nearest is not None else None
        quality.append(
            MemberQuality(
                config=config,
                auc=float(roc_auc_score(y.to_numpy(), matrix[config].to_numpy())),
                nearest=nearest,
                nearest_spearman=nearest_spearman,
            )
        )
    return tuple(quality)


def member_selection(evaluation: EvaluationSummary) -> dict[str, MemberSelection]:
    """outer fold별 선택 여부와 계수를 구성원 단위로 집계한다."""
    members = list(evaluation.folds[0].weights)
    result = {}
    for member in members:
        weights = [fold.weights[member] for fold in evaluation.folds]
        result[member] = MemberSelection(
            selected=sum(weight != 0.0 for weight in weights),
            fold_total=len(weights),
            mean_weight=float(np.mean(weights)),
        )
    return result


def compare_perspective(
    frozen_map: FrozenPerspectiveMap,
    perspective: str,
    baseline: EvaluationSummary,
    best_removed: EvaluationSummary,
) -> PerspectiveComparison:
    excluded_members = tuple(
        member for member in frozen_map.members if member.primary == perspective
    )
    excluded = tuple(member.config for member in excluded_members)
    families = tuple(sorted({member.model_family for member in excluded_members}))
    baseline_folds = {fold.fold: fold.auc for fold in baseline.folds}
    removed_folds = {fold.fold: fold.auc for fold in best_removed.folds}
    if baseline_folds.keys() != removed_folds.keys():
        raise PerspectiveDiagnosticError(f"{perspective} 제거 대조의 outer fold가 다르다.")
    worse = sum(removed_folds[fold] < baseline_folds[fold] for fold in baseline_folds)
    better = sum(removed_folds[fold] > baseline_folds[fold] for fold in baseline_folds)
    tied = len(baseline_folds) - worse - better
    loss = baseline.nested_auc - best_removed.nested_auc

    baseline_selection = member_selection(baseline)
    removed_selection = member_selection(best_removed)
    excluded_selection = tuple((name, baseline_selection[name]) for name in excluded)
    shifts = []
    for config in removed_selection:
        before = baseline_selection[config]
        after = removed_selection[config]
        shifts.append(
            SelectionShift(
                config=config,
                selected_before=before.selected,
                selected_after=after.selected,
                selected_delta=after.selected - before.selected,
                weight_before=before.mean_weight,
                weight_after=after.mean_weight,
                weight_delta=after.mean_weight - before.mean_weight,
            )
        )
    return PerspectiveComparison(
        perspective=perspective,
        excluded=excluded,
        model_families=families,
        best=best_removed,
        loss=loss,
        outer_worse=worse,
        outer_better=better,
        outer_tied=tied,
        excluded_selection=excluded_selection,
        shifts=tuple(shifts),
        opens_experiment=(
            loss >= OPENING_DELTA
            and worse >= OPENING_FOLD_WINS
            and len(families) == 1
        ),
    )


def _fold_aucs(evaluation: EvaluationSummary) -> dict[int, float]:
    return {fold.fold: fold.auc for fold in evaluation.folds}


def load_reference_run(
    run_id: str, store: RunStore, frozen_map: FrozenPerspectiveMap
) -> ReferenceRun:
    meta = store.facts_of(run_id)
    member_configs = tuple(
        config
        for config in meta.params.get("ensemble.member_configs", "").split(",")
        if config
    )
    expected = tuple(member.config for member in frozen_map.members)
    member_count_raw = meta.params.get("ensemble.member_count")
    member_count = int(member_count_raw) if member_count_raw is not None else None
    return ReferenceRun(
        run_id=run_id,
        strategy=meta.params.get("ensemble.strategy"),
        member_count=member_count,
        member_configs=member_configs,
        auc=meta.metrics.get("auc_oof"),
        fold_aucs={
            fold: meta.metrics[f"auc_fold_{fold}"]
            for fold in range(5)
            if f"auc_fold_{fold}" in meta.metrics
        },
        matches_pool=(member_configs == expected and member_count == len(expected)),
    )


def _evaluation_to_json(evaluation: EvaluationSummary) -> dict[str, Any]:
    return {
        "name": evaluation.name,
        "nested_auc": evaluation.nested_auc,
        "folds": [asdict(fold) for fold in evaluation.folds],
    }


def _evaluation_from_json(record: dict[str, Any]) -> EvaluationSummary:
    return EvaluationSummary(
        name=str(record["name"]),
        nested_auc=float(record["nested_auc"]),
        folds=tuple(
            FoldSummary(
                fold=int(fold["fold"]),
                auc=float(fold["auc"]),
                weights={str(key): float(value) for key, value in fold["weights"].items()},
            )
            for fold in record["folds"]
        ),
    )


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    frozen_map: FrozenPerspectiveMap,
    strategy_names: tuple[str, ...],
) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "complete": False,
            "pool_sha256": frozen_map.pool_sha256,
            "perspective_map_sha256": frozen_map.sha256,
            "strategy_names": list(strategy_names),
            "baseline_evaluations": [],
            "baseline_processed_names": [],
            "group_evaluations": {},
            "group_processed_names": {},
        }
    try:
        checkpoint = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PerspectiveDiagnosticError(f"중간 결과를 읽지 못했다: {path}: {exc}") from exc
    expected = (
        checkpoint.get("schema_version") == 1
        and checkpoint.get("pool_sha256") == frozen_map.pool_sha256
        and checkpoint.get("perspective_map_sha256") == frozen_map.sha256
        and checkpoint.get("strategy_names") == list(strategy_names)
    )
    if not expected:
        raise PerspectiveDiagnosticError(
            f"중간 결과의 풀, 분류 또는 전략 목록이 현재 입력과 다르다: {path}"
        )
    return checkpoint


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def run_diagnostic(
    frozen_map: FrozenPerspectiveMap,
    store: RunStore,
    fold_of: pd.Series,
    y: pd.Series,
    combiners: tuple[Combiner, ...],
    *,
    checkpoint_path: Path,
    reference_run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> DiagnosticResult:
    started = time.perf_counter()
    members = [(member.config, member.run_id) for member in frozen_map.members]
    matrix = member_matrix(members, store, fold_of.index)
    quality = measure_member_quality(matrix, y)
    pool = Pool.load(frozen_map.pool_path)
    recorded_auc = {member.config: member.oof_auc for member in pool.members}
    for item in quality:
        if abs(item.auc - recorded_auc[item.config]) > 1e-9:
            raise PerspectiveDiagnosticError(
                f"{item.config} 단독 OOF 재채점 {item.auc:.12f}과 풀 장부가 다르다."
            )

    strategy_names = tuple(combiner.name for combiner in combiners)
    bagged_workers = next(
        combiner.workers
        for combiner in combiners
        if isinstance(combiner, BaggedGreedyRankMeanCombiner)
    )
    checkpoint = _load_checkpoint(
        checkpoint_path, frozen_map=frozen_map, strategy_names=strategy_names
    )
    baseline_evaluations = tuple(
        _evaluation_from_json(record)
        for record in checkpoint["baseline_evaluations"]
    )
    baseline_processed = tuple(checkpoint.get("baseline_processed_names", []))
    if set(baseline_processed) == set(strategy_names):
        if progress is not None:
            progress("전체 풀 결합 전략 결과를 중간 결과에서 복구했다.")
    else:
        if progress is not None:
            if baseline_processed:
                progress(
                    f"전체 풀 결합 전략 {len(baseline_processed)}개 뒤부터 재개"
                )
            else:
                progress("전체 풀 결합 전략 재평가 시작")

        def save_baseline(
            evaluations: tuple[EvaluationSummary, ...], processed: tuple[str, ...]
        ) -> None:
            checkpoint["baseline_evaluations"] = [
                _evaluation_to_json(evaluation) for evaluation in evaluations
            ]
            checkpoint["baseline_processed_names"] = list(processed)
            _write_json_atomic(checkpoint_path, checkpoint)

        baseline_evaluations = evaluate_all(
            combiners,
            matrix,
            fold_of,
            y,
            existing=baseline_evaluations,
            processed=baseline_processed,
            checkpoint=save_baseline,
            progress=progress,
        )
    baseline_best = best_evaluation(baseline_evaluations)

    group_records = checkpoint["group_evaluations"]
    group_processed_records = checkpoint.setdefault("group_processed_names", {})
    comparisons = []
    all_group_evaluations: dict[str, tuple[EvaluationSummary, ...]] = {}
    for perspective in frozen_map.perspectives:
        excluded = [
            member.config
            for member in frozen_map.members
            if member.primary == perspective
        ]
        group_evaluations = tuple(
            _evaluation_from_json(record)
            for record in group_records.get(perspective, [])
        )
        group_processed = tuple(group_processed_records.get(perspective, []))
        if set(group_processed) == set(strategy_names):
            if progress is not None:
                progress(f"관점 {perspective} 결과를 중간 결과에서 복구했다.")
        else:
            if progress is not None:
                progress(
                    f"관점 제거 시작: {perspective} ({len(excluded)}개 구성원 제외)"
                )
            subset = matrix.drop(columns=excluded)

            def save_group(
                evaluations: tuple[EvaluationSummary, ...], processed: tuple[str, ...]
            ) -> None:
                group_records[perspective] = [
                    _evaluation_to_json(evaluation) for evaluation in evaluations
                ]
                group_processed_records[perspective] = list(processed)
                _write_json_atomic(checkpoint_path, checkpoint)

            group_evaluations = evaluate_all(
                combiners,
                subset,
                fold_of,
                y,
                existing=group_evaluations,
                processed=group_processed,
                checkpoint=save_group,
                progress=progress,
            )
        all_group_evaluations[perspective] = group_evaluations
        comparisons.append(
            compare_perspective(
                frozen_map,
                perspective,
                baseline_best,
                best_evaluation(group_evaluations),
            )
        )

    reference = (
        load_reference_run(reference_run_id, store, frozen_map)
        if reference_run_id is not None
        else None
    )
    elapsed = time.perf_counter() - started
    return DiagnosticResult(
        frozen_map=frozen_map,
        member_quality=quality,
        registered_strategy_names=strategy_names,
        bagged_workers=bagged_workers,
        baseline_evaluations=baseline_evaluations,
        group_evaluations=all_group_evaluations,
        baseline_best=baseline_best,
        comparisons=tuple(comparisons),
        reference=reference,
        elapsed_seconds=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        train_sha256=file_sha256(TRAIN_PATH),
        folds_sha256=file_sha256(FOLDS_PATH),
    )


def _fmt(value: float) -> str:
    return f"{value:.12f}"


def _perspective_label(result: DiagnosticResult, key: str) -> str:
    return result.frozen_map.perspectives[key].label


def render_report(result: DiagnosticResult) -> str:
    """사람이 결정 근거를 검토할 수 있는 Markdown 보고서를 만든다."""
    lines = [
        "# 후보 풀 학습기 계열과 정보 관점 제거 진단",
        "",
        "## 고정 입력",
        "",
        f"후보 풀은 {result.frozen_map.member_count}개 구성원이며 SHA-256은 "
        f"`{result.frozen_map.pool_sha256}`다.",
        f"제거 결과보다 먼저 고정한 정보 관점 장부 SHA-256은 "
        f"`{result.frozen_map.sha256}`다.",
        f"훈련 자료 SHA-256은 `{result.train_sha256}`이고 fold 파일 SHA-256은 "
        f"`{result.folds_sha256}`다.",
        f"등록된 {len(result.registered_strategy_names)}개 결합 전략을 전체 풀과 모든 대표 "
        "정보 관점 제외 풀에 같은 순서로 적용했다.",
        f"배깅 탐욕 전략은 선택 규칙을 바꾸지 않고 {result.bagged_workers}개 작업 스레드로 실행했다.",
        "",
        "## 전체 풀 결합 전략 재평가",
        "",
        "| 결합 전략 | nested OOF AUC | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for evaluation in result.baseline_evaluations:
        folds = _fold_aucs(evaluation)
        lines.append(
            f"| `{evaluation.name}` | {_fmt(evaluation.nested_auc)} | "
            + " | ".join(_fmt(folds[fold]) for fold in sorted(folds))
            + " |"
        )
    lines.extend(
        [
            "",
            f"최신 동등 단계 기준 실행은 `{result.baseline_best.name}`이며 nested OOF "
            f"AUC는 `{_fmt(result.baseline_best.nested_auc)}`다.",
        ]
    )
    if result.reference is not None:
        reference = result.reference
        delta = (
            result.baseline_best.nested_auc - reference.auc
            if reference.auc is not None
            else None
        )
        lines.extend(
            [
                "",
                "### 기존 파생 앙상블 교차 확인",
                "",
                f"참고 run `{reference.run_id}`은 후보 풀 계보가 "
                f"{'일치한다' if reference.matches_pool else '일치하지 않는다'}.",
                f"기록 전략은 `{reference.strategy}`이고 OOF AUC는 "
                f"`{_fmt(reference.auc) if reference.auc is not None else '없음'}`다.",
                (
                    f"현재 재평가 최고점과의 차이는 `{delta:+.12f}`다."
                    if delta is not None
                    else "현재 재평가와 비교할 참고 OOF AUC가 없다."
                ),
            ]
        )

    baseline_selection = member_selection(result.baseline_best)
    quality = {item.config: item for item in result.member_quality}
    lines.extend(
        [
            "",
            "## 구성원 두 축 지도",
            "",
            "| 구성원 | 학습기 계열 | 대표 정보 관점 | 보조 정보 관점 | 단독 OOF AUC | 최근접 구성원 | 스피어만 | 선택 fold | 평균 계수 |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for member in result.frozen_map.members:
        item = quality[member.config]
        selection = baseline_selection[member.config]
        secondary = ", ".join(
            _perspective_label(result, key) for key in member.secondary
        ) or "없음"
        lines.append(
            f"| `{member.config}` | `{member.model_family}` | "
            f"{_perspective_label(result, member.primary)} | {secondary} | "
            f"{_fmt(item.auc)} | `{item.nearest}` | "
            f"{item.nearest_spearman:.9f} | {selection.selected}/{selection.fold_total} | "
            f"{selection.mean_weight:+.9f} |"
        )

    best_weights = {
        fold.fold: fold.weights for fold in result.baseline_best.folds
    }
    lines.extend(
        [
            "",
            "## 전체 풀 최고 전략의 outer fold별 구성원 계수",
            "",
            "계수가 0이면 해당 outer fold에서 선택되지 않은 구성원이다.",
            "",
            "| 구성원 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for member in result.frozen_map.members:
        lines.append(
            f"| `{member.config}` | "
            + " | ".join(
                f"{best_weights[fold][member.config]:+.9f}"
                for fold in sorted(best_weights)
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 대표 정보 관점 묶음 제거 대조",
            "",
            "손실은 전체 풀 최고 nested OOF AUC에서 해당 관점 제거 뒤 최고점을 뺀 값이다.",
            "outer 악화는 제거 뒤 fold AUC가 전체 풀보다 낮아진 fold 수다.",
            "선택 변화 합과 계수 L1 변화는 남은 구성원이 선택 절차 변화로 받은 총변화를 나타낸다.",
            "",
            "| 대표 정보 관점 | 제외 구성원 | 학습기 계열 | 제거 뒤 최고 전략 | 제거 뒤 AUC | 손실 | outer 악화/개선/동률 | 제외 전 선택 | 남은 선택 변화 합 | 남은 계수 L1 변화 | 최대 계수 변화 | 개방 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for comparison in result.comparisons:
        excluded_selection = ", ".join(
            f"`{config}` {selection.selected}/{selection.fold_total} "
            f"({selection.mean_weight:+.6f})"
            for config, selection in comparison.excluded_selection
        )
        selected_change = sum(abs(shift.selected_delta) for shift in comparison.shifts)
        weight_change = sum(abs(shift.weight_delta) for shift in comparison.shifts)
        max_shift = max(
            comparison.shifts,
            key=lambda shift: (abs(shift.weight_delta), shift.config),
        )
        lines.append(
            f"| {_perspective_label(result, comparison.perspective)} | "
            f"{', '.join(f'`{name}`' for name in comparison.excluded)} | "
            f"{', '.join(f'`{name}`' for name in comparison.model_families)} | "
            f"`{comparison.best.name}` | {_fmt(comparison.best.nested_auc)} | "
            f"{comparison.loss:+.12f} | "
            f"{comparison.outer_worse}/{comparison.outer_better}/{comparison.outer_tied} | "
            f"{excluded_selection} | {selected_change} | {weight_change:.9f} | "
            f"`{max_shift.config}` {max_shift.weight_delta:+.9f} | "
            f"{'충족' if comparison.opens_experiment else '미충족'} |"
        )

    lines.extend(["", "## 관점별 남은 구성원 선택과 계수 변화", ""])
    for comparison in result.comparisons:
        lines.extend(
            [
                f"### {_perspective_label(result, comparison.perspective)}",
                "",
                "| 구성원 | 선택 fold 전 | 선택 fold 후 | 변화 | 평균 계수 전 | 평균 계수 후 | 변화 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for shift in comparison.shifts:
            lines.append(
                f"| `{shift.config}` | {shift.selected_before} | {shift.selected_after} | "
                f"{shift.selected_delta:+d} | {shift.weight_before:+.9f} | "
                f"{shift.weight_after:+.9f} | {shift.weight_delta:+.9f} |"
            )
        lines.append("")

    opened = [comparison for comparison in result.comparisons if comparison.opens_experiment]
    lines.extend(["## 새 학습 실험 개방 판정", ""])
    if opened:
        labels = ", ".join(
            _perspective_label(result, comparison.perspective) for comparison in opened
        )
        lines.append(
            f"현재 풀 기여 경로의 세 조건을 모두 충족한 대표 정보 관점은 {labels}다."
        )
        lines.append("각 관점은 별도 후속 결정 티켓에서 다른 학습기 계열 이식 범위를 정해야 한다.")
    else:
        lines.append("현재 풀 기여 경로의 세 조건을 모두 충족한 대표 정보 관점은 없다.")
    lines.extend(
        [
            "최근 12개 Playground 상위권 해법 조사에서 현재 풀에 없는 관점 가운데 독립 제거 대조 1건 또는 독립 성공 사례 2건을 충족한 관점도 없다.",
            "Chris Deotte식 3단계 GBDT·신경망 중간 출력은 독립 제거 대조 0건과 독립 성공 사례 1건이므로 근거 부족을 유지한다.",
            "[조건부: 교사-학생 anti-residual 보정](https://github.com/tmheo/predicting-smartphone-addiction/issues/186)은 이 진단과 독립적으로 병행한다.",
            "",
            "## 자원 사용과 해석 경계",
            "",
            f"로컬 CPU 진단 경과 시간은 {result.elapsed_seconds:.1f}초이고 프로세스 최고 RSS는 "
            f"{result.peak_rss_bytes / 1024 / 1024:.1f} MiB다.",
            "이 진단은 새 기본 모델을 학습하지 않았고 champion, 후보 풀과 결합 전략 장부를 변경하지 않았다.",
            "대표 관점 제거 결과가 음수여도 다른 후보 구성이나 다른 결합 절차에서 기여할 수 없다는 뜻은 아니다.",
            "분류를 결과 뒤에 바꾸면 선택 편향이 생기므로 이 결과는 고정한 정보 관점 장부에만 유효하다.",
            "",
        ]
    )
    return "\n".join(lines)


def result_json(result: DiagnosticResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "complete": True,
        "pool_sha256": result.frozen_map.pool_sha256,
        "perspective_map_sha256": result.frozen_map.sha256,
        "strategy_names": list(result.registered_strategy_names),
        "bagged_workers": result.bagged_workers,
        "baseline_processed_names": list(result.registered_strategy_names),
        "baseline_evaluations": [
            _evaluation_to_json(evaluation) for evaluation in result.baseline_evaluations
        ],
        "group_evaluations": {
            perspective: [
                _evaluation_to_json(evaluation) for evaluation in evaluations
            ]
            for perspective, evaluations in result.group_evaluations.items()
        },
        "group_processed_names": {
            perspective: list(result.registered_strategy_names)
            for perspective in result.group_evaluations
        },
        "baseline_best": _evaluation_to_json(result.baseline_best),
        "member_quality": [asdict(item) for item in result.member_quality],
        "comparisons": [asdict(item) for item in result.comparisons],
        "reference": asdict(result.reference) if result.reference is not None else None,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_rss_bytes": result.peak_rss_bytes,
        "train_sha256": result.train_sha256,
        "folds_sha256": result.folds_sha256,
    }


def _write_report_atomic(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--pool", type=Path, default=POOL_PATH)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--reference-run")
    parser.add_argument(
        "--bagged-workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="배깅 탐욕 전략의 작업 스레드 수. 선택 결과에는 영향을 주지 않는다.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        frozen_map = load_frozen_map(args.map, pool_path=args.pool)
        fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
        y = labels(fold_of.index)
        result = run_diagnostic(
            frozen_map,
            MlflowRunStore(args.tracking_uri),
            fold_of,
            y,
            diagnostic_combiners(args.bagged_workers),
            checkpoint_path=args.json_output,
            reference_run_id=args.reference_run,
            progress=progress,
        )
        _write_report_atomic(args.report, render_report(result))
        _write_json_atomic(args.json_output, result_json(result))
    except PerspectiveDiagnosticError as exc:
        sys.exit(str(exc))
    print(
        f"진단 완료: {args.report} / {args.json_output} "
        f"(최고 {result.baseline_best.name} {result.baseline_best.nested_auc:.12f})"
    )


if __name__ == "__main__":
    main()
