"""후보 풀 축소의 성능 동등 대역을 측정한다. (#342)

사용법:
    uv run python scripts/measure_pool_equivalence_band.py plan
    uv run python scripts/measure_pool_equivalence_band.py measure --jobs 5
    uv run python scripts/measure_pool_equivalence_band.py report

설계는 `docs/research/pool-reduction-equivalence-band-design.md`가 결과를 보기 전에
고정했다. 이 스크립트는 그 설계를 그대로 실행하며, 실제 구성원을 기준 풀에서 빼는
대조는 하나도 만들지 않는다. 그 판정은 #339 소관이다.

영점 짝은 증강 제거형이다. 동결한 35개 기준 풀에 무정보 구성원 g개를 더한 큰 팔과
기준 풀 그대로인 작은 팔을 짝지어, 정보량이 바뀌지 않은 제거의 nested OOF AUC 차이를
모은다. 무정보 구성원은 고정 난수의 독립 순위 열 또는 기존 구성원의 정확 복제다.

모형은 다시 학습하지 않는다. 후보 풀 구성원의 기록된 OOF 예측만 읽는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline import ensemble as ensemble_module  # noqa: E402
from pipeline.data import ID, labels  # noqa: E402
from pipeline.runs import MlflowRunStore  # noqa: E402

BASELINE_PATH = Path("artifacts/pool-baseline-2026-08-21.yaml")
DEFAULT_EVIDENCE = Path("docs/research/pool-reduction-equivalence-band-evidence.json")
DEFAULT_REPORT = Path("docs/research/pool-reduction-equivalence-band.md")
TICKET_ISSUE = 342
MAP_ISSUE = 338

# 설계에서 고정한 난수. 실행 전에 정했고 결과를 보고 바꾸지 않는다.
NULL_SEED = 630063  # pipeline.pool_audit.NULL_SEED와 같은 계보를 유지한다.
BOOTSTRAP_SEED = 342342
BOOTSTRAP_REPLICATES = 2000

# #341이 사전 고정할 대조 목록의 실제 제거 집합 크기를 덮는 격자.
GROUP_SIZES = (1, 2, 3, 4, 5, 6, 13, 17, 22, 27)
SIZE1_NOISE_REPEATS = 10
MULTI_REPEATS = 3

CLONE = "복제"
NOISE = "난수"


class MeasurementError(Exception):
    """측정을 시작하기 전에 멈춰야 하는 계약 위반."""


@dataclass(frozen=True)
class NullContrast:
    """무정보 구성원 g개를 더한 큰 팔 하나. 작은 팔은 언제나 기준 풀이다."""

    index: int
    kind: str  # CLONE 또는 NOISE.
    size: int
    sources: tuple[str, ...]  # 복제 대상 구성원 이름. 난수 대조에서는 빈 튜플.

    @property
    def label(self) -> str:
        if self.kind == CLONE:
            return f"{CLONE} g={self.size} #{self.index}"
        return f"{NOISE} g={self.size} #{self.index}"

    def added_columns(self) -> tuple[str, ...]:
        if self.kind == CLONE:
            return tuple(f"{name}__복제{i}" for i, name in enumerate(self.sources))
        return tuple(f"__난수{self.index}_{i}" for i in range(self.size))


def build_contrast_plan(configs: list[str]) -> list[NullContrast]:
    """설계가 고정한 대조 목록을 결과와 무관한 난수로만 만든다.

    크기 1의 복제 대조는 35개 구성원을 전수로 덮는다. 개별 순차 제거가 재심사의 주된
    단위이기 때문이다. 크기 2 이상의 복제 대상은 NULL_SEED에서 나온 난수로 비복원
    표집하며, 표집이 결과와 무관하므로 선택 편향이 생기지 않는다.
    """
    if len(set(configs)) != len(configs):
        raise MeasurementError("기준 풀의 구성원 이름이 중복된다.")
    plan: list[NullContrast] = []
    index = 0
    for name in configs:
        plan.append(NullContrast(index=index, kind=CLONE, size=1, sources=(name,)))
        index += 1
    for _ in range(SIZE1_NOISE_REPEATS):
        plan.append(NullContrast(index=index, kind=NOISE, size=1, sources=()))
        index += 1
    rng = np.random.default_rng(NULL_SEED)
    for size in GROUP_SIZES:
        if size == 1:
            continue
        if size > len(configs):
            raise MeasurementError(f"복제 대상 크기 {size}가 풀 크기를 넘는다.")
        for _ in range(MULTI_REPEATS):
            chosen = rng.choice(len(configs), size=size, replace=False)
            sources = tuple(configs[position] for position in sorted(chosen))
            plan.append(NullContrast(index=index, kind=CLONE, size=size, sources=sources))
            index += 1
        for _ in range(MULTI_REPEATS):
            plan.append(NullContrast(index=index, kind=NOISE, size=size, sources=()))
            index += 1
    return plan


def noise_columns(contrast: NullContrast, rows: int) -> np.ndarray:
    """대조 하나가 쓰는 독립 순위 열. 대조 색인으로 갈라 실행 순서와 무관하게 만든다."""
    rng = np.random.default_rng([NULL_SEED, contrast.index])
    return np.column_stack(
        [(rng.permutation(rows) + 1).astype(np.float64) / rows for _ in range(contrast.size)]
    )


def augmented_matrix(base: pd.DataFrame, contrast: NullContrast) -> pd.DataFrame:
    """큰 팔의 구성원 예측 행렬. 기준 풀 열 뒤에 무정보 열을 붙인다."""
    names = contrast.added_columns()
    if contrast.kind == CLONE:
        added = base[list(contrast.sources)].to_numpy(dtype=np.float64)
    else:
        added = noise_columns(contrast, len(base))
    extra = pd.DataFrame(added, index=base.index, columns=list(names))
    return pd.concat([base, extra], axis=1)


@dataclass
class ArmResult:
    """풀 하나에 기존 등록 결합 절차를 적용한 결과."""

    label: str
    members: int
    strategy_auc: dict[str, float]
    strategy_fold_auc: dict[str, dict[int, float]]
    failures: dict[str, str]
    best_strategy: str
    best_auc: float
    best_fold_auc: dict[int, float]
    elapsed_seconds: float
    best_prediction: pd.Series | None = field(default=None, repr=False)


def evaluate_arm(
    label: str, matrix: pd.DataFrame, fold_of: pd.Series, y: pd.Series
) -> ArmResult:
    """등록된 기본 평가 전략 전부를 돌리고 전략별·최선 판독을 함께 남긴다."""
    started = time.monotonic()
    strategy_auc: dict[str, float] = {}
    strategy_fold_auc: dict[str, dict[int, float]] = {}
    failures: dict[str, str] = {}
    best: ensemble_module.NestedEvaluation | None = None
    for name in ensemble_module.DEFAULT_COMBINER_NAMES:
        combiner = ensemble_module.COMBINER_REGISTRY[name]
        try:
            evaluation = ensemble_module.evaluate_nested(combiner, matrix, fold_of, y)
        except ensemble_module.CombinerConvergenceError as exc:
            failures[name] = str(exc)
            continue
        strategy_auc[name] = evaluation.nested_auc
        strategy_fold_auc[name] = {o.fold: o.auc for o in evaluation.folds}
        if best is None or evaluation.nested_auc > best.nested_auc:
            best = evaluation
    if best is None:
        raise MeasurementError(f"{label}: 수렴한 등록 전략이 하나도 없다.")
    return ArmResult(
        label=label,
        members=matrix.shape[1],
        strategy_auc=strategy_auc,
        strategy_fold_auc=strategy_fold_auc,
        failures=failures,
        best_strategy=best.name,
        best_auc=best.nested_auc,
        best_fold_auc={o.fold: o.auc for o in best.folds},
        elapsed_seconds=time.monotonic() - started,
        best_prediction=best.prediction,
    )


class WeightedAucSorter:
    """예측 하나의 정렬 구조를 한 번 만들어 두고 가중 AUC를 O(n)으로 다시 계산한다.

    짝지은 행 부트스트랩은 재표본마다 AUC를 두 번씩 2000회 계산한다. 매번 정렬하면
    측정 비용이 결합 절차 실행에 맞먹으므로, 정렬은 한 번만 하고 재표본의 행 중복
    횟수를 가중치로 넘긴다. 동점은 중간 순위로 처리해 roc_auc_score와 같은 값을 준다.
    """

    def __init__(self, prediction: np.ndarray, y: np.ndarray) -> None:
        order = np.argsort(prediction, kind="stable")
        sorted_prediction = prediction[order]
        boundary = np.empty(len(order), dtype=bool)
        boundary[0] = True
        np.not_equal(sorted_prediction[1:], sorted_prediction[:-1], out=boundary[1:])
        self._order = order
        self._group = np.cumsum(boundary) - 1
        self._groups = int(self._group[-1]) + 1
        self._positive = y[order].astype(np.float64)

    def auc(self, weights: np.ndarray) -> float:
        ordered = weights[self._order].astype(np.float64)
        total = np.bincount(self._group, weights=ordered, minlength=self._groups)
        positive = np.bincount(
            self._group, weights=ordered * self._positive, minlength=self._groups
        )
        below = np.concatenate(([0.0], np.cumsum(total)[:-1]))
        rank_sum = float(np.sum(positive * (below + (total + 1.0) / 2.0)))
        positives = float(positive.sum())
        negatives = float(total.sum()) - positives
        if positives <= 0.0 or negatives <= 0.0:
            raise MeasurementError("재표본에 한쪽 라벨만 남았다.")
        return (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def stratified_bootstrap_weights(
    rng: np.random.Generator, fold_positions: list[np.ndarray], rows: int
) -> np.ndarray:
    """outer fold별 행 수를 보존한 복원추출 중복 횟수."""
    weights = np.zeros(rows, dtype=np.float64)
    for positions in fold_positions:
        size = len(positions)
        weights[positions] = rng.multinomial(size, np.full(size, 1.0 / size))
    return weights


def paired_row_bootstrap(
    small: np.ndarray,
    large: np.ndarray,
    y: np.ndarray,
    fold_positions: list[np.ndarray],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """같은 재표본 행에서 두 팔의 AUC 차이를 다시 계산한다. 재적합은 하지 않는다."""
    small_sorter = WeightedAucSorter(small, y)
    large_sorter = WeightedAucSorter(large, y)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        weights = stratified_bootstrap_weights(rng, fold_positions, len(y))
        differences[replicate] = small_sorter.auc(weights) - large_sorter.auc(weights)
    quantiles = np.quantile(differences, [0.025, 0.5, 0.975])
    return {
        "replicates": replicates,
        "minimum": float(differences.min()),
        "percentile_2p5": float(quantiles[0]),
        "median": float(quantiles[1]),
        "percentile_97p5": float(quantiles[2]),
        "maximum": float(differences.max()),
    }


def contrast_record(
    contrast: NullContrast,
    baseline: ArmResult,
    arm: ArmResult,
    bootstrap: dict[str, float],
) -> dict:
    """대조 하나의 기록. 전체 차이와 outer fold별 기록을 함께 남긴다."""
    delta_best = baseline.best_auc - arm.best_auc
    fold_delta = {
        fold: baseline.best_fold_auc[fold] - arm.best_fold_auc[fold]
        for fold in sorted(baseline.best_fold_auc)
    }
    strategy_delta = {
        name: baseline.strategy_auc[name] - arm.strategy_auc[name]
        for name in baseline.strategy_auc
        if name in arm.strategy_auc
    }
    return {
        "index": contrast.index,
        "kind": contrast.kind,
        "size": contrast.size,
        "label": contrast.label,
        "sources": list(contrast.sources),
        "added_columns": list(contrast.added_columns()),
        "large_members": arm.members,
        "small_members": baseline.members,
        "small_best_strategy": baseline.best_strategy,
        "large_best_strategy": arm.best_strategy,
        "best_strategy_changed": baseline.best_strategy != arm.best_strategy,
        "small_best_auc": baseline.best_auc,
        "large_best_auc": arm.best_auc,
        "delta_best": delta_best,
        "delta_by_strategy": strategy_delta,
        "fold_delta": {str(fold): value for fold, value in fold_delta.items()},
        "negative_folds": sum(1 for value in fold_delta.values() if value < 0.0),
        "fold_total": len(fold_delta),
        "large_failures": sorted(arm.failures),
        "bootstrap": bootstrap,
        "elapsed_seconds": arm.elapsed_seconds,
    }


def band_for_records(records: list[dict]) -> dict:
    """설계가 고정한 산식으로 크기별 대역을 낸다.

    대역은 관측 영점 Δ와 짝지은 행 부트스트랩 봉투의 합집합이다. 각 계열이 다른
    잡음원을 덮으므로 한쪽만 쓰면 대역이 좁아지고, 좁은 대역은 잡음을 유지 증거로
    오인해 풀을 필요 이상으로 크게 남긴다.
    """
    observed = [record["delta_best"] for record in records]
    lower_tail = [record["bootstrap"]["percentile_2p5"] for record in records]
    upper_tail = [record["bootstrap"]["percentile_97p5"] for record in records]
    return {
        "contrasts": len(records),
        "observed_minimum": min(observed),
        "observed_maximum": max(observed),
        "bootstrap_lower_minimum": min(lower_tail),
        "bootstrap_upper_maximum": max(upper_tail),
        "lower": min(min(observed), min(lower_tail)),
        "upper": max(max(observed), max(upper_tail)),
        "max_negative_folds": max(record["negative_folds"] for record in records),
        "fold_total": records[0]["fold_total"],
        "best_strategy_changes": sum(
            1 for record in records if record["best_strategy_changed"]
        ),
    }


def same_size_pairs(records: list[dict]) -> list[dict]:
    """크기가 같은 무정보 팔끼리의 차이. 추가 실행 없이 기록된 팔만 짝짓는다.

    증강 제거형은 두 팔의 풀 크기가 달라 쏠림과 잡음이 한 숫자에 섞인다. 여기서는 두
    팔이 같은 크기이고 둘 다 기준 풀에 정보를 더하지 않았으므로, 남는 것은 "어느
    무정보 열이 하필 들어왔는가"뿐이다. 양쪽 순서를 모두 넣어 분포를 0에 대칭으로 둔다.
    """
    pairs: list[dict] = []
    for size in sorted({record["size"] for record in records}):
        subset = [record for record in records if record["size"] == size]
        for left in subset:
            for right in subset:
                if left["index"] == right["index"]:
                    continue
                kinds = tuple(sorted((left["kind"], right["kind"])))
                pairs.append(
                    {
                        "size": size,
                        "kind_pair": " vs ".join(kinds),
                        "left": left["label"],
                        "right": right["label"],
                        "delta": left["large_best_auc"] - right["large_best_auc"],
                    }
                )
    return pairs


def _same_size_band(pairs: list[dict]) -> dict | None:
    if not pairs:
        return None
    deltas = [pair["delta"] for pair in pairs]
    return {
        "pairs": len(pairs),
        "observed_minimum": min(deltas),
        "observed_maximum": max(deltas),
        "symmetric_envelope": max(abs(value) for value in deltas),
    }


def same_size_summary(records: list[dict]) -> dict:
    """동일 크기 영점 대조의 전체·크기별·짝 종류별 대역."""
    pairs = same_size_pairs(records)
    by_size = {}
    for size in sorted({pair["size"] for pair in pairs}):
        subset = [pair for pair in pairs if pair["size"] == size]
        by_size[str(size)] = {
            "all": _same_size_band(subset),
            **{
                kind_pair: _same_size_band(
                    [pair for pair in subset if pair["kind_pair"] == kind_pair]
                )
                for kind_pair in sorted({pair["kind_pair"] for pair in subset})
            },
        }
    return {"overall": _same_size_band(pairs), "by_size": by_size}


def _band_or_none(records: list[dict]) -> dict | None:
    """대조가 하나도 없는 칸은 대역을 만들지 않는다(점검 실행에서만 생긴다)."""
    return band_for_records(records) if records else None


def summarize(records: list[dict]) -> dict:
    """전체·크기별·종류별 대역."""
    by_size = {}
    for size in sorted({record["size"] for record in records}):
        subset = [record for record in records if record["size"] == size]
        by_size[str(size)] = {
            "all": band_for_records(subset),
            CLONE: _band_or_none([r for r in subset if r["kind"] == CLONE]),
            NOISE: _band_or_none([r for r in subset if r["kind"] == NOISE]),
        }
    augmented = {
        "overall": band_for_records(records),
        "by_kind": {
            CLONE: _band_or_none([r for r in records if r["kind"] == CLONE]),
            NOISE: _band_or_none([r for r in records if r["kind"] == NOISE]),
        },
        "by_size": by_size,
    }
    same_size = same_size_summary(records)
    union = None
    if same_size["overall"] is not None:
        union = {
            "lower": min(
                augmented["overall"]["lower"], same_size["overall"]["observed_minimum"]
            ),
            "upper": max(
                augmented["overall"]["upper"], same_size["overall"]["observed_maximum"]
            ),
        }
    return {
        **augmented,
        "same_size": same_size,
        "union": union,
    }


SEED_READOUT_EVIDENCE = Path(
    "docs/research/pool-reduction-equivalence-band-seed-readout.json"
)
CONFIRM_SEEDS = (42, 43, 44)


def load_seed_matrices(
    index: pd.Index,
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """시드별 OOF 산출물이 있는 구성원만 모아 시드 조각 행렬을 만든다.

    설계가 약속한 보조 판독의 입력이다. 이 행렬은 대역을 정하는 데 쓰지 않는다.
    풀 장부가 시드 평균본만 구성원으로 받으므로 시드 조각 풀은 재심사의 판정 대상이
    아니고, 35개 중 8개가 #98 이전 실행이라 같은 대조를 낼 수 없기 때문이다.
    """
    from mlflow.tracking import MlflowClient

    from pipeline.runs import TRACKING_URI

    baseline = yaml.safe_load(BASELINE_PATH.read_text())
    client = MlflowClient(tracking_uri=TRACKING_URI)
    columns: dict[int, dict[str, pd.Series]] = {seed: {} for seed in CONFIRM_SEEDS}
    mean_columns: dict[str, pd.Series] = {}
    covered: list[str] = []
    missing: list[str] = []
    for member in baseline["members"]:
        run_id = member["run_id"]
        names = {artifact.path for artifact in client.list_artifacts(run_id)}
        wanted = {seed: f"oof_seed_{seed}.parquet" for seed in CONFIRM_SEEDS}
        if not set(wanted.values()) <= names:
            missing.append(member["config"])
            continue
        per_seed = []
        for seed, name in wanted.items():
            frame = pd.read_parquet(client.download_artifacts(run_id, name))
            series = frame.set_index(ID)["pred"].reindex(index).astype(np.float64)
            if series.isna().any():
                raise MeasurementError(
                    f"{member['config']} 시드 {seed} OOF가 fold 배정과 맞지 않는다."
                )
            columns[seed][member["config"]] = series
            per_seed.append(series)
        mean_columns[member["config"]] = sum(per_seed) / len(per_seed)
        covered.append(member["config"])
    matrices = {f"시드 {seed}": pd.DataFrame(columns[seed]) for seed in CONFIRM_SEEDS}
    matrices["시드 평균"] = pd.DataFrame(mean_columns)
    return matrices, covered, missing


def seed_readout(evidence_path: Path) -> dict:
    """시드 잡음이 행 재표집 잡음보다 큰지 작은지만 알려주는 보조 판독.

    시드 조각 풀마다 같은 복제 영점 짝을 만들어 Δ를 재고, 그 Δ가 시드에 따라 흔들리는
    폭을 시드 평균 짝의 짝지은 행 부트스트랩 폭과 나란히 놓는다. 대역은 바꾸지 않는다.
    """
    base, fold_of, y = load_inputs()
    matrices, covered, missing = load_seed_matrices(fold_of.index)
    if not covered:
        raise MeasurementError("시드별 OOF 산출물이 있는 구성원이 없다.")
    target = covered[0]
    positions = fold_positions_of(fold_of)
    print(
        f"보조 판독 구성원 {len(covered)}개, 제외 {len(missing)}개, 복제 대상 {target}",
        flush=True,
    )

    results = []
    for label in ["시드 평균", *[f"시드 {seed}" for seed in CONFIRM_SEEDS]]:
        matrix = matrices[label]
        small = evaluate_arm(f"{label} 작은 팔", matrix, fold_of, y)
        clone = matrix[[target]].to_numpy(dtype=np.float64)
        large_matrix = pd.concat(
            [
                matrix,
                pd.DataFrame(clone, index=matrix.index, columns=[f"{target}__복제0"]),
            ],
            axis=1,
        )
        large = evaluate_arm(f"{label} 큰 팔", large_matrix, fold_of, y)
        bootstrap = paired_row_bootstrap(
            small.best_prediction.to_numpy(dtype=np.float64),
            large.best_prediction.to_numpy(dtype=np.float64),
            y.to_numpy(),
            positions,
        )
        delta = small.best_auc - large.best_auc
        results.append(
            {
                "pool": label,
                "members": small.members,
                "small_best_strategy": small.best_strategy,
                "large_best_strategy": large.best_strategy,
                "small_best_auc": small.best_auc,
                "large_best_auc": large.best_auc,
                "delta_best": delta,
                "bootstrap": bootstrap,
            }
        )
        print(f"{label}: Δ={delta:+.12f}", flush=True)

    slices = [r for r in results if r["pool"] != "시드 평균"]
    seed_spread = max(r["delta_best"] for r in slices) - min(
        r["delta_best"] for r in slices
    )
    mean_result = next(r for r in results if r["pool"] == "시드 평균")
    bootstrap_width = (
        mean_result["bootstrap"]["percentile_97p5"]
        - mean_result["bootstrap"]["percentile_2p5"]
    )
    evidence = {
        "ticket_issue": TICKET_ISSUE,
        "purpose": "보조 판독. 성능 동등 대역을 정하는 데 쓰지 않는다.",
        "clone_target": target,
        "covered_members": covered,
        "excluded_members": missing,
        "confirm_seeds": list(CONFIRM_SEEDS),
        "results": results,
        "seed_delta_spread": seed_spread,
        "seed_mean_bootstrap_width": bootstrap_width,
        "seed_noise_exceeds_row_noise": seed_spread > bootstrap_width,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(
        f"시드 Δ 폭 {seed_spread:+.12f}, 행 재표집 Δ 폭 {bootstrap_width:+.12f}",
        flush=True,
    )
    return evidence


_WORKER_STATE: dict = {}


def _init_worker() -> None:
    base, fold_of, y = load_inputs()
    _WORKER_STATE["base"] = base
    _WORKER_STATE["fold_of"] = fold_of
    _WORKER_STATE["y"] = y


def _run_contrast(payload: tuple[NullContrast, np.ndarray, list[list[int]]]) -> dict:
    contrast, baseline_prediction, fold_positions = payload
    base = _WORKER_STATE["base"]
    fold_of = _WORKER_STATE["fold_of"]
    y = _WORKER_STATE["y"]
    matrix = augmented_matrix(base, contrast)
    arm = evaluate_arm(contrast.label, matrix, fold_of, y)
    bootstrap = paired_row_bootstrap(
        baseline_prediction,
        arm.best_prediction.to_numpy(dtype=np.float64),
        y.to_numpy(),
        [np.asarray(positions) for positions in fold_positions],
    )
    arm.best_prediction = None
    return {"arm": arm, "bootstrap": bootstrap, "contrast": contrast}


def load_inputs() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """동결한 기준 풀의 구성원 OOF 행렬, outer fold 배정, 목표값."""
    baseline = yaml.safe_load(BASELINE_PATH.read_text())
    members = [(m["config"], m["run_id"]) for m in baseline["members"]]
    store = MlflowRunStore()
    fold_of = ensemble_module.outer_fold_assignment()
    y = labels(fold_of.index)
    base = ensemble_module.member_matrix(members, store, fold_of.index)
    return base, fold_of, y


def fold_positions_of(fold_of: pd.Series) -> list[np.ndarray]:
    values = fold_of.to_numpy()
    return [np.flatnonzero(values == fold) for fold in sorted(fold_of.unique())]


def measure(jobs: int, limit: int | None, evidence_path: Path) -> dict:
    base, fold_of, y = load_inputs()
    configs = list(base.columns)
    plan = build_contrast_plan(configs)
    if limit is not None:
        plan = plan[:limit]
    print(f"기준 풀 {len(configs)}개, 행 {len(base)}, 영점 대조 {len(plan)}개", flush=True)

    baseline_arm = evaluate_arm("기준 풀", base, fold_of, y)
    baseline_prediction = baseline_arm.best_prediction.to_numpy(dtype=np.float64)
    print(
        f"기준 풀 최선 전략 {baseline_arm.best_strategy} "
        f"nested OOF AUC {baseline_arm.best_auc:.12f} "
        f"({baseline_arm.elapsed_seconds:.0f}초)",
        flush=True,
    )

    positions = [list(map(int, group)) for group in fold_positions_of(fold_of)]
    payloads = [(contrast, baseline_prediction, positions) for contrast in plan]
    records: list[dict] = []
    if jobs <= 1:
        _init_worker()
        outcomes = (_run_contrast(payload) for payload in payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker)
        outcomes = executor.map(_run_contrast, payloads)
    for outcome in outcomes:
        record = contrast_record(
            outcome["contrast"], baseline_arm, outcome["arm"], outcome["bootstrap"]
        )
        records.append(record)
        print(
            f"[{len(records)}/{len(plan)}] {record['label']} "
            f"Δ={record['delta_best']:+.12f} "
            f"음수 fold {record['negative_folds']}/{record['fold_total']} "
            f"전략 {record['small_best_strategy']}→{record['large_best_strategy']}",
            flush=True,
        )
    if jobs > 1:
        executor.shutdown()

    records.sort(key=lambda record: record["index"])
    evidence = {
        "ticket_issue": TICKET_ISSUE,
        "map_issue": MAP_ISSUE,
        "design": "docs/research/pool-reduction-equivalence-band-design.md",
        "baseline_pool": str(BASELINE_PATH),
        "null_seed": NULL_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "group_sizes": list(GROUP_SIZES),
        "size1_noise_repeats": SIZE1_NOISE_REPEATS,
        "multi_repeats": MULTI_REPEATS,
        "registered_strategies": list(ensemble_module.DEFAULT_COMBINER_NAMES),
        "excluded_precision_strategies": list(ensemble_module.PRECISION_COMBINER_NAMES),
        "baseline_arm": {
            "members": baseline_arm.members,
            "best_strategy": baseline_arm.best_strategy,
            "best_auc": baseline_arm.best_auc,
            "strategy_auc": baseline_arm.strategy_auc,
            "fold_auc": {str(k): v for k, v in baseline_arm.best_fold_auc.items()},
            "failures": sorted(baseline_arm.failures),
        },
        "contrasts": records,
        "summary": summarize(records),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(f"기록: {evidence_path}", flush=True)
    return evidence


def _fmt(value: float, digits: int = 12) -> str:
    return f"{value:+.{digits}f}"


def render_markdown(evidence: dict) -> str:
    summary = evidence["summary"]
    overall = summary["overall"]
    baseline = evidence["baseline_arm"]
    lines = [
        "# 후보 풀 축소 성능 동등 대역 측정 결과",
        "",
        f"이슈 [#{evidence['ticket_issue']}](https://github.com/tmheo/predicting-smartphone-addiction/issues/{evidence['ticket_issue']})의 결과 문서다.",
        f"설계는 [{Path(evidence['design']).name}]({Path(evidence['design']).name})가 결과를 보기 전에 고정했고, 기계 판독 자료는 `{DEFAULT_EVIDENCE.name}`다.",
        "",
        "## 결론",
        "",
        f"동결한 35개 기준 풀의 최선 전략은 `{baseline['best_strategy']}`이고 nested OOF AUC는 `{baseline['best_auc']:.12f}`다.",
        f"영점 대조 {overall['contrasts']}개 전체에서 성능 동등 대역은 `{_fmt(overall['lower'])}`에서 `{_fmt(overall['upper'])}`다.",
        f"제거 집합 크기 1만 보면 대역은 `{_fmt(summary['by_size']['1']['all']['lower'])}`에서 `{_fmt(summary['by_size']['1']['all']['upper'])}`다.",
        "",
        "크기 1의 대역을 큰 제거 집합에 그대로 쓰면 안 된다.",
        "아래 크기별 표가 대역이 제거 집합 크기와 함께 어떻게 움직이는지 보여준다.",
        "",
        "## 크기별 대역",
        "",
        "| 제거 집합 크기 | 대조 수 | 관측 최솟값 | 관측 최댓값 | 부트스트랩 2.5백분위 최솟값 | 부트스트랩 97.5백분위 최댓값 | 대역 하한 | 대역 상한 | 음수 fold 최댓값 | 최선 전략 교체 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for size in sorted(summary["by_size"], key=int):
        band = summary["by_size"][size]["all"]
        lines.append(
            f"| {size} | {band['contrasts']} | {_fmt(band['observed_minimum'])} | "
            f"{_fmt(band['observed_maximum'])} | {_fmt(band['bootstrap_lower_minimum'])} | "
            f"{_fmt(band['bootstrap_upper_maximum'])} | {_fmt(band['lower'])} | "
            f"{_fmt(band['upper'])} | {band['max_negative_folds']}/{band['fold_total']} | "
            f"{band['best_strategy_changes']}/{band['contrasts']} |"
        )
    lines += [
        "",
        "## 대조 종류별 대역",
        "",
        "| 종류 | 대조 수 | 대역 하한 | 대역 상한 | 음수 fold 최댓값 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind in (CLONE, NOISE):
        band = summary["by_kind"][kind]
        if band is None:
            continue
        lines.append(
            f"| {kind} 구성원 제거 | {band['contrasts']} | {_fmt(band['lower'])} | "
            f"{_fmt(band['upper'])} | {band['max_negative_folds']}/{band['fold_total']} |"
        )
    same_size = summary.get("same_size")
    if same_size and same_size["overall"]:
        lines += [
            "",
            "## 동일 크기 영점 대역",
            "",
            "크기가 같은 무정보 팔끼리의 차이다.",
            "두 팔이 같은 크기이고 둘 다 기준 풀에 정보를 더하지 않았으므로 분포는 0에 대칭이며, 풀 크기 변화가 만드는 쏠림이 빠진다.",
            "추가 실행 없이 위 표의 큰 팔들을 서로 짝지어 얻는다.",
            "",
            "| 제거 집합 크기 | 짝 수 | 최솟값 | 최댓값 | 대칭 봉투 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for size in sorted(same_size["by_size"], key=int):
            entry = same_size["by_size"][size]["all"]
            lines.append(
                f"| {size} | {entry['pairs']} | {_fmt(entry['observed_minimum'])} | "
                f"{_fmt(entry['observed_maximum'])} | {_fmt(entry['symmetric_envelope'])} |"
            )
        if summary.get("union"):
            lines += [
                "",
                f"증강 제거형 대역과 동일 크기 대역의 합집합은 `{_fmt(summary['union']['lower'])}`에서 `{_fmt(summary['union']['upper'])}`다.",
            ]
    lines += [
        "",
        "## 크기 1 복제 대조 전수",
        "",
        "35개 구성원을 각각 한 번씩 복제하고 그 복제본만 뺀 짝이다.",
        "정보량이 바뀌지 않으므로 여기 나타나는 차이는 전부 측정 잡음이다.",
        "",
        "| 복제 대상 | Δ | 음수 fold | 부트스트랩 2.5백분위 | 부트스트랩 97.5백분위 | 최선 전략 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in evidence["contrasts"]:
        if record["kind"] != CLONE or record["size"] != 1:
            continue
        strategy = record["large_best_strategy"]
        if record["best_strategy_changed"]:
            strategy = f"{record['small_best_strategy']} → {strategy}"
        lines.append(
            f"| `{record['sources'][0]}` | {_fmt(record['delta_best'])} | "
            f"{record['negative_folds']}/{record['fold_total']} | "
            f"{_fmt(record['bootstrap']['percentile_2p5'])} | "
            f"{_fmt(record['bootstrap']['percentile_97p5'])} | {strategy} |"
        )
    lines += [
        "",
        "## 크기 1 난수 대조",
        "",
        "| 대조 | Δ | 음수 fold | 부트스트랩 2.5백분위 | 부트스트랩 97.5백분위 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in evidence["contrasts"]:
        if record["kind"] != NOISE or record["size"] != 1:
            continue
        lines.append(
            f"| {record['label']} | {_fmt(record['delta_best'])} | "
            f"{record['negative_folds']}/{record['fold_total']} | "
            f"{_fmt(record['bootstrap']['percentile_2p5'])} | "
            f"{_fmt(record['bootstrap']['percentile_97p5'])} |"
        )
    lines += [
        "",
        "## 해석 제약",
        "",
        "대조 반복 횟수는 경험적 P값이나 오류율 추정으로 해석하지 않는다.",
        "크기 1 복제 대조 35회만 관측 분포로 읽고, 나머지 크기는 관측 최솟값과 최댓값으로만 읽는다.",
        "",
        "영점 짝은 증강 제거형이라 절대 풀 크기가 실제 대조보다 제거 집합 크기만큼 위에 있다.",
        "크기 1에서는 무시할 수 있고, 큰 크기에서는 근사값이다.",
        "",
        "이 문서는 통과와 탈락을 가르는 숫자 문턱을 정하지 않는다.",
        "문턱과 적용 규칙은 #347 소관이다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="후보 풀 축소의 성능 동등 대역 측정 (#342)")
    parser.add_argument(
        "command", choices=["plan", "measure", "seed-readout", "report"]
    )
    parser.add_argument("--jobs", type=int, default=1, help="동시에 돌릴 대조 수")
    parser.add_argument("--limit", type=int, help="앞에서부터 이만큼의 대조만 실행(점검용)")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed-evidence", type=Path, default=SEED_READOUT_EVIDENCE)
    args = parser.parse_args()

    if args.command == "plan":
        baseline = yaml.safe_load(BASELINE_PATH.read_text())
        configs = [m["config"] for m in baseline["members"]]
        plan = build_contrast_plan(configs)
        for contrast in plan:
            print(f"{contrast.index:3d} {contrast.kind} g={contrast.size:2d} "
                  f"{', '.join(contrast.sources) if contrast.sources else '독립 순위 열'}")
        print(f"대조 {len(plan)}개, 결합 절차 실행 {len(plan) + 1}회")
        return

    if args.command == "measure":
        measure(args.jobs, args.limit, args.evidence)
        return

    if args.command == "seed-readout":
        seed_readout(args.seed_evidence)
        return

    evidence = json.loads(args.evidence.read_text())
    evidence["summary"] = summarize(evidence["contrasts"])
    args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    args.report.write_text(render_markdown(evidence))
    print(f"보고서: {args.report}")


if __name__ == "__main__":
    main()
