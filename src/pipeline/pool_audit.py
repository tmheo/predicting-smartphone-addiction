"""후보 풀의 예측 무결성, 중복, 다양성과 기여 참고값 감사. (#63)

사용법:
    uv run python -m pipeline.pool_audit
    uv run python -m pipeline.pool_audit --output docs/research/oof-pool-audit.md

감사는 모델을 다시 학습하지 않고 현재 풀 장부가 가리키는 실행 산출물만 읽는다.
검사 순서는 다음과 같다.

1. 실행 출처, 입력 해시, OOF·시험 예측의 ID·fold·float64 정밀도를 검증한다.
2. 시드별 OOF가 있으면 평균본을 다시 계산하고, #98 이전 실행은 부분 확인으로 표시한다.
3. OOF와 시험 예측 배열의 SHA-256을 함께 비교해 정확 중복을 제거한다.
4. 단독 OOF, 최근접 순위 상관, 잔차 상관, 결측 개수 층별 AUC를 측정한다.
5. 고정 난수 순위와 각 구성원 복제로 표준 평가 앙상블 기여의 영점 대역을 잰다.

균등 순위 평균 기여와 영점 대역은 결합 방식 하나의 진단값일 뿐이다.
무결성과 중복 검사를 통과한 후보를 제거하는 게이트로 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .data import ID, TARGET, file_sha256
from .identity import array_identity
from .judgment import CONFIRM_SEEDS, DUPLICATE_SPEARMAN
from .ledger import Pool, PoolMember
from .runs import TRACKING_URI

TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
FOLDS_PATH = Path("artifacts/folds.parquet")
DEFAULT_OUTPUT = Path("docs/research/oof-pool-audit.md")
NULL_SEED = 630063
NULL_RANDOM_COUNT = 64
PRED_TOLERANCE = 1e-12
METRIC_TOLERANCE = 1e-9


class PoolAuditError(Exception):
    """감사를 시작하거나 완결할 수 없는 전역 전제 위반."""


@dataclass(frozen=True)
class CandidateArtifacts:
    """실행 저장소에서 읽은 후보 한 명의 기록 원형과 예측 산출물."""

    member: PoolMember
    status: str
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]
    config_bytes: bytes
    committed_config_bytes: bytes | None
    oof: pd.DataFrame
    test_pred: pd.DataFrame
    seed_oofs: dict[int, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditContext:
    train: pd.DataFrame
    test: pd.DataFrame
    folds: pd.DataFrame
    input_hashes: dict[str, str]


@dataclass(frozen=True)
class CandidateCheck:
    artifacts: CandidateArtifacts
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    seed_mean_status: str
    auc: float | None
    oof_missing: int
    test_missing: int
    oof_hash: str | None
    test_hash: str | None

    @property
    def config(self) -> str:
        return self.artifacts.member.config

    @property
    def run_id(self) -> str:
        return self.artifacts.member.run_id

    @property
    def valid(self) -> bool:
        return not self.failures

    @property
    def oof_series(self) -> pd.Series:
        return self.artifacts.oof.set_index(ID)["pred"]


@dataclass(frozen=True)
class DuplicateDecision:
    dropped: str
    kept: str
    reason: str
    spearman: float


@dataclass(frozen=True)
class CandidateQuality:
    config: str
    auc: float
    nearest: str | None
    nearest_spearman: float | None
    residual_correlation: float | None
    contribution: float | None
    segment_aucs: dict[str, float | None]
    action: str = ""


@dataclass(frozen=True)
class NullControls:
    seed: int
    random_count: int
    random_changes: tuple[float, ...]
    clone_changes: dict[str, float]
    best_member: str
    lower: float
    upper: float


@dataclass(frozen=True)
class PoolAudit:
    checks: tuple[CandidateCheck, ...]
    duplicate_decisions: tuple[DuplicateDecision, ...]
    retained_configs: tuple[str, ...]
    initial_ensemble_auc: float
    full_ensemble_auc: float
    quality: tuple[CandidateQuality, ...]
    controls: NullControls


def prediction_array_sha256(pred: pd.Series | np.ndarray) -> str:
    """정렬 검증 뒤의 float64 예측 배열 자체를 SHA-256으로 식별한다."""
    return array_identity(pred)


def _same_values(actual: pd.Series, expected: pd.Series) -> bool:
    return np.array_equal(actual.to_numpy(), expected.to_numpy())


def _prediction_frame_failures(
    frame: pd.DataFrame,
    *,
    expected_ids: pd.Series,
    expected_folds: pd.Series | None,
    name: str,
) -> list[str]:
    failures: list[str] = []
    required = [ID, "pred"] + (["fold"] if expected_folds is not None else [])
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        return [f"{name} 필수 열 없음: {','.join(missing_columns)}"]
    if len(frame) != len(expected_ids):
        failures.append(f"{name} 행 수 {len(frame)} != 기대 {len(expected_ids)}")
        return failures
    if frame[ID].duplicated().any():
        failures.append(f"{name} id 중복")
    if not _same_values(frame[ID], expected_ids):
        failures.append(f"{name} id 순서 불일치")
    if expected_folds is not None and not _same_values(frame["fold"], expected_folds):
        failures.append(f"{name} fold 배정 불일치")
    if frame["pred"].dtype != np.dtype("float64"):
        failures.append(f"{name} pred 정밀도 {frame['pred'].dtype} != float64")
    if not np.isfinite(frame["pred"].to_numpy(dtype=float)).all():
        failures.append(f"{name} pred에 NaN 또는 무한값 존재")
    return failures


def verify_candidate(candidate: CandidateArtifacts, context: AuditContext) -> CandidateCheck:
    """후보 한 명의 계보, 정렬, fold, 정밀도와 시드 평균을 검증한다."""
    failures: list[str] = []
    warnings: list[str] = []
    member = candidate.member
    train_ids = context.train[ID]
    test_ids = context.test[ID]
    fold_by_id = context.folds.set_index(ID)["fold"]
    expected_folds = train_ids.map(fold_by_id)

    if candidate.status != "FINISHED":
        failures.append(f"실행 상태 {candidate.status} != FINISHED")
    if candidate.params.get("experiment") != member.config:
        failures.append("장부 config와 실행 experiment 불일치")
    try:
        config = yaml.safe_load(candidate.config_bytes)
    except yaml.YAMLError as exc:
        failures.append(f"설정 YAML 해석 실패: {exc}")
        config = {}
    if config.get("name") != member.config:
        failures.append("장부 config와 설정 name 불일치")
    seeds = [int(seed) for seed in candidate.params.get("seeds", "").split(",") if seed]
    if seeds != CONFIRM_SEEDS or member.seeds != CONFIRM_SEEDS:
        failures.append(f"확정 시드 불일치: 실행 {seeds}, 장부 {member.seeds}")
    if candidate.tags.get("git_dirty") != "False":
        failures.append("git_dirty 실행")
    if candidate.committed_config_bytes is None:
        failures.append("실행 커밋에서 설정 파일을 찾지 못함")
    elif candidate.committed_config_bytes != candidate.config_bytes:
        failures.append("실행 커밋의 설정과 산출물 설정 불일치")
    for name, expected_hash in context.input_hashes.items():
        if candidate.tags.get(f"sha256.{name}") != expected_hash:
            failures.append(f"{name} 입력 해시 불일치")

    failures.extend(
        _prediction_frame_failures(
            candidate.oof,
            expected_ids=train_ids,
            expected_folds=expected_folds,
            name="OOF",
        )
    )
    failures.extend(
        _prediction_frame_failures(
            candidate.test_pred,
            expected_ids=test_ids,
            expected_folds=None,
            name="시험 예측",
        )
    )

    oof_missing = int(candidate.oof["pred"].isna().sum()) if "pred" in candidate.oof else -1
    test_missing = (
        int(candidate.test_pred["pred"].isna().sum()) if "pred" in candidate.test_pred else -1
    )
    auc: float | None = None
    oof_hash: str | None = None
    test_hash: str | None = None
    base_shape_ok = len(candidate.oof) == len(context.train) and "pred" in candidate.oof
    if base_shape_ok and np.isfinite(candidate.oof["pred"].to_numpy(dtype=float)).all():
        auc = float(roc_auc_score(context.train[TARGET], candidate.oof["pred"]))
        claimed = candidate.metrics.get("auc_oof")
        if claimed is None or abs(auc - claimed) > METRIC_TOLERANCE:
            failures.append(f"OOF 재채점 {auc:.12f}과 실행 지표 {claimed} 불일치")
        if abs(auc - member.oof_auc) > METRIC_TOLERANCE:
            failures.append(f"OOF 재채점 {auc:.12f}과 장부 지표 {member.oof_auc:.12f} 불일치")
        if candidate.oof["pred"].dtype == np.dtype("float64"):
            oof_hash = prediction_array_sha256(candidate.oof["pred"])
    if (
        len(candidate.test_pred) == len(context.test)
        and "pred" in candidate.test_pred
        and np.isfinite(candidate.test_pred["pred"].to_numpy(dtype=float)).all()
        and candidate.test_pred["pred"].dtype == np.dtype("float64")
    ):
        test_hash = prediction_array_sha256(candidate.test_pred["pred"])

    seed_mean_status = "완전 확인"
    if not candidate.seed_oofs:
        seed_mean_status = "기존 기록 부분 확인"
        warnings.append("#98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음")
    elif set(candidate.seed_oofs) != set(CONFIRM_SEEDS):
        seed_mean_status = "실패"
        failures.append(f"시드별 OOF 일부 누락: {sorted(candidate.seed_oofs)}")
    else:
        seed_predictions = []
        for seed in CONFIRM_SEEDS:
            frame = candidate.seed_oofs[seed]
            failures.extend(
                _prediction_frame_failures(
                    frame,
                    expected_ids=train_ids,
                    expected_folds=expected_folds,
                    name=f"seed {seed} OOF",
                )
            )
            if "pred" in frame and len(frame) == len(context.train):
                seed_predictions.append(frame["pred"].to_numpy(dtype=np.float64))
        if len(seed_predictions) == len(CONFIRM_SEEDS) and "pred" in candidate.oof:
            mean_prediction = np.mean(seed_predictions, axis=0)
            max_difference = float(
                np.max(np.abs(mean_prediction - candidate.oof["pred"].to_numpy(dtype=float)))
            )
            if max_difference > PRED_TOLERANCE:
                seed_mean_status = "실패"
                failures.append(f"시드 평균과 OOF 불일치: 최대 절대차 {max_difference:.3g}")

    return CandidateCheck(
        artifacts=candidate,
        failures=tuple(failures),
        warnings=tuple(warnings),
        seed_mean_status=seed_mean_status,
        auc=auc,
        oof_missing=oof_missing,
        test_missing=test_missing,
        oof_hash=oof_hash,
        test_hash=test_hash,
    )


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a.rank().to_numpy(), b.rank().to_numpy())[0, 1])


def deduplicate(checks: list[CandidateCheck]) -> tuple[list[CandidateCheck], list[DuplicateDecision]]:
    """정확 중복을 먼저, 그다음 ADR 0001 순위 중복을 성능 내림차순으로 제거한다."""
    valid = [check for check in checks if check.valid]
    order = {check.config: index for index, check in enumerate(valid)}
    strongest = sorted(valid, key=lambda check: (-float(check.auc), order[check.config]))
    retained: list[CandidateCheck] = []
    decisions: list[DuplicateDecision] = []
    for candidate in strongest:
        duplicate: DuplicateDecision | None = None
        for kept in retained:
            exact = (
                candidate.oof_hash is not None
                and candidate.test_hash is not None
                and candidate.oof_hash == kept.oof_hash
                and candidate.test_hash == kept.test_hash
            )
            rho = _spearman(candidate.oof_series, kept.oof_series)
            if exact:
                duplicate = DuplicateDecision(candidate.config, kept.config, "배열 해시 정확 중복", rho)
                break
            if rho >= DUPLICATE_SPEARMAN:
                duplicate = DuplicateDecision(candidate.config, kept.config, "순위 상관 중복", rho)
                break
        if duplicate is None:
            retained.append(candidate)
        else:
            decisions.append(duplicate)
    retained.sort(key=lambda check: order[check.config])
    return retained, decisions


def _segment_masks(train: pd.DataFrame) -> dict[str, np.ndarray]:
    feature_columns = [column for column in train if column not in (ID, TARGET)]
    missing = train[feature_columns].isna().sum(axis=1).to_numpy()
    return {
        "결측 0": missing == 0,
        "결측 1-2": (missing >= 1) & (missing <= 2),
        "결측 3-5": (missing >= 3) & (missing <= 5),
        "결측 6+": missing >= 6,
    }


def _auc_or_none(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0 or len(np.unique(y[mask])) < 2:
        return None
    return float(roc_auc_score(y[mask], pred[mask]))


def measure_null_controls(
    matrix: pd.DataFrame,
    y: np.ndarray,
    single_aucs: dict[str, float],
    *,
    seed: int = NULL_SEED,
    random_count: int = NULL_RANDOM_COUNT,
) -> NullControls:
    """독립 난수 순위와 구성원 정확 복제가 만드는 기여 변화 대역을 측정한다."""
    if random_count < 1:
        raise ValueError("난수 대조 수는 1 이상이어야 한다.")
    # 기존 구성원 순위는 대조마다 변하지 않는다.
    # 매번 DataFrame 전체를 다시 rank하면 실제 75만 행 풀에서 불필요하게 비싸므로
    # 한 번 계산한 백분위 순위 합에 대조 열 하나만 더한다.
    ranked = matrix.rank(pct=True).to_numpy(dtype=np.float64)
    rank_sum = ranked.sum(axis=1)
    baseline = float(roc_auc_score(y, rank_sum / len(matrix.columns)))
    rng = np.random.default_rng(seed)
    random_changes = []
    denominator = len(matrix.columns) + 1
    for _ in range(random_count):
        random_rank = (rng.permutation(len(matrix)) + 1).astype(np.float64) / len(matrix)
        random_changes.append(
            float(roc_auc_score(y, (rank_sum + random_rank) / denominator)) - baseline
        )
    clone_changes = {}
    for column, member in enumerate(matrix):
        clone_changes[member] = (
            float(roc_auc_score(y, (rank_sum + ranked[:, column]) / denominator)) - baseline
        )
    all_changes = random_changes + list(clone_changes.values())
    return NullControls(
        seed=seed,
        random_count=random_count,
        random_changes=tuple(random_changes),
        clone_changes=clone_changes,
        best_member=max(single_aucs, key=single_aucs.get),
        lower=float(min(all_changes)),
        upper=float(max(all_changes)),
    )


def audit_pool(
    candidates: list[CandidateArtifacts],
    context: AuditContext,
    *,
    null_seed: int = NULL_SEED,
    random_count: int = NULL_RANDOM_COUNT,
) -> PoolAudit:
    checks = [verify_candidate(candidate, context) for candidate in candidates]
    retained, duplicate_decisions = deduplicate(checks)
    if not retained:
        raise PoolAuditError("무결성·중복 검사 뒤 남은 후보가 없다.")

    index = pd.Index(context.train[ID], name=ID)
    initial_matrix = pd.DataFrame(
        {check.config: check.oof_series.reindex(index) for check in retained},
        index=index,
        dtype=np.float64,
    )
    if initial_matrix.isna().any().any():
        raise PoolAuditError("중복 제거 뒤 후보 행렬에 정렬되지 않은 OOF가 있다.")
    y = context.train.set_index(ID)[TARGET].reindex(index).to_numpy()
    all_ranked = initial_matrix.rank(pct=True).to_numpy(dtype=np.float64)
    active = list(range(len(initial_matrix.columns)))

    def subset_auc(columns: list[int]) -> float:
        # ensemble.rank_mean과 같은 np.mean 덧셈 순서를 매 단계 유지한다.
        # 합에서 한 열을 빼는 대수적 단축은 부동소수점 동률을 바꿀 수 있다.
        return float(roc_auc_score(y, all_ranked[:, columns].mean(axis=1)))

    initial_auc = subset_auc(active)
    matrix = initial_matrix.iloc[:, active]
    ranked = all_ranked[:, active]
    rank_sum = ranked.sum(axis=1)
    single_aucs = {member: float(roc_auc_score(y, matrix[member])) for member in matrix}
    full_auc = float(roc_auc_score(y, rank_sum / len(matrix.columns)))
    controls = measure_null_controls(
        matrix, y, single_aucs, seed=null_seed, random_count=random_count
    )
    segments = _segment_masks(context.train)

    qualities: list[CandidateQuality] = []
    for member in initial_matrix:
        others = [name for name in initial_matrix if name != member]
        correlations = {
            name: _spearman(initial_matrix[member], initial_matrix[name]) for name in others
        }
        nearest = max(correlations, key=correlations.get) if correlations else None
        residual_correlation = None
        if nearest is not None:
            residual_correlation = float(
                np.corrcoef(
                    initial_matrix[member].to_numpy() - y,
                    initial_matrix[nearest].to_numpy() - y,
                )[0, 1]
            )
        final_others = [name for name in matrix if name != member]
        contribution = (
            full_auc
            - float(
                roc_auc_score(
                    y,
                    ranked[
                        :,
                        [matrix.columns.get_loc(name) for name in final_others],
                    ].mean(axis=1),
                )
            )
            if final_others
            else None
        )
        segment_aucs = {
            name: _auc_or_none(y, initial_matrix[member].to_numpy(), mask)
            for name, mask in segments.items()
        }
        action = "유지"
        qualities.append(
            CandidateQuality(
                config=member,
                auc=float(roc_auc_score(y, initial_matrix[member])),
                nearest=nearest,
                nearest_spearman=correlations.get(nearest) if nearest else None,
                residual_correlation=residual_correlation,
                contribution=contribution,
                segment_aucs=segment_aucs,
                action=action,
            )
        )

    return PoolAudit(
        checks=tuple(checks),
        duplicate_decisions=tuple(duplicate_decisions),
        retained_configs=tuple(matrix.columns),
        initial_ensemble_auc=initial_auc,
        full_ensemble_auc=full_auc,
        quality=tuple(qualities),
        controls=controls,
    )


def _fmt(value: float | None, digits: int = 8) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_markdown(audit: PoolAudit) -> str:
    """감사 결과를 이슈와 저장소에 남길 수 있는 자기완결 Markdown으로 만든다."""
    valid_count = sum(check.valid for check in audit.checks)
    full_count = sum(check.seed_mean_status == "완전 확인" for check in audit.checks)
    legacy_count = sum(check.seed_mean_status == "기존 기록 부분 확인" for check in audit.checks)
    failed = [check.config for check in audit.checks if not check.valid]
    lines = [
        "# OOF 후보 풀 품질·다양성 감사",
        "",
        "## 결론",
        "",
        f"현재 장부 {len(audit.checks)}개 중 {valid_count}개가 계보, 정렬, fold, float64와 유한성 검사를 통과했다.",
        f"시드별 OOF 평균은 {full_count}개에서 재계산해 확인했고, #98 이전 실행 {legacy_count}개는 시드별 파일이 없어 기존 기록 부분 확인으로 남는다.",
        f"정확·순위 중복 제거 뒤 {len(audit.retained_configs)}개가 남았고, 모두 nested OOF 평가 후보로 유지한다.",
        f"전체 후보의 균등 순위 평균 OOF AUC는 `{audit.full_ensemble_auc:.12f}`다.",
        f"난수 {audit.controls.random_count}개와 구성원 복제 대조의 영점 대역은 `{audit.controls.lower:+.12f}`에서 `{audit.controls.upper:+.12f}`다.",
        "균등 순위 평균의 제외 기여와 영점 대역은 참고값이며, 후보 진입이나 제거에 쓰지 않는다.",
        "",
    ]
    if failed:
        lines.append(f"무결성 실패 후보: {', '.join(failed)}.")
    if not failed:
        lines.append("무결성 실패 후보가 없다.")

    lines.extend(
        [
            "",
            "## 무결성 및 배열 해시",
            "",
            "| 후보 | run | 판정 | 시드 평균 | OOF 결측 | 시험 결측 | OOF SHA-256 | 시험 SHA-256 |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for check in audit.checks:
        verdict = "통과" if check.valid else "실패: " + "; ".join(check.failures)
        lines.append(
            f"| {check.config} | `{check.run_id}` | {verdict} | {check.seed_mean_status} | "
            f"{check.oof_missing} | {check.test_missing} | `{check.oof_hash or '-'}` | "
            f"`{check.test_hash or '-'}` |"
        )
    warnings = [(check.config, warning) for check in audit.checks for warning in check.warnings]
    if warnings:
        lines.extend(["", "기존 기록 부분 확인 사유:", ""])
        lines.extend(f"- `{config}`: {warning}." for config, warning in warnings)

    lines.extend(
        [
            "",
            "## 중복 판정",
            "",
        ]
    )
    if audit.duplicate_decisions:
        lines.extend(
            [
                "| 제거 후보 | 유지 후보 | 근거 | 스피어만 |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for decision in audit.duplicate_decisions:
            lines.append(
                f"| {decision.dropped} | {decision.kept} | {decision.reason} | "
                f"{decision.spearman:.12f} |"
            )
    else:
        lines.append("OOF와 시험 예측이 모두 같은 정확 중복 및 스피어만 0.998 이상 중복이 없다.")

    lines.extend(["", "## 후보 유지 정책", ""])
    lines.append("무결성과 중복 검사를 통과한 후보는 균등 순위 평균 기여의 부호와 관계없이 모두 유지한다.")
    lines.append("구성원 선택과 가중치는 outer fold 안에서 학습하는 nested OOF 평가가 결정한다.")
    lines.extend(
        [
            "",
            f"고정점 후보: {', '.join(f'`{config}`' for config in audit.retained_configs)}.",
        ]
    )

    lines.extend(
        [
            "",
            "## 품질과 다양성",
            "",
            "제외 기여는 전체 후보의 균등 순위 평균에서 각 후보 하나를 제외한 참고값이다.",
            "잔차 상관은 최근접 순위 상관 후보와의 피어슨 상관이다.",
            "",
            "| 후보 | 단독 OOF | 최근접 후보 | 스피어만 | 잔차 상관 | 제외 기여 | 판정 |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for quality in audit.quality:
        lines.append(
            f"| {quality.config} | {quality.auc:.12f} | {quality.nearest or '-'} | "
            f"{_fmt(quality.nearest_spearman, 12)} | {_fmt(quality.residual_correlation, 12)} | "
            f"{_fmt(quality.contribution, 12)} | {quality.action} |"
        )

    lines.extend(
        [
            "",
            "## 주요 신호 구간",
            "",
            "구간은 원시 12개 입력 열의 행별 결측 개수로 고정했다.",
            "",
            "| 후보 | 결측 0 | 결측 1-2 | 결측 3-5 | 결측 6+ |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for quality in audit.quality:
        segment = quality.segment_aucs
        lines.append(
            f"| {quality.config} | {_fmt(segment['결측 0'])} | {_fmt(segment['결측 1-2'])} | "
            f"{_fmt(segment['결측 3-5'])} | {_fmt(segment['결측 6+'])} |"
        )

    random_values = np.asarray(audit.controls.random_changes)
    best_clone = audit.controls.clone_changes[audit.controls.best_member]
    lines.extend(
        [
            "",
            "## 기여 영점 대조",
            "",
            f"난수 대조는 고정 seed `{audit.controls.seed}`로 독립 순위 열 {audit.controls.random_count}개를 각각 하나씩 추가했다.",
            f"난수 변화는 최소 `{random_values.min():+.12f}`, 중앙값 `{np.median(random_values):+.12f}`, 95백분위 `{np.quantile(random_values, 0.95):+.12f}`, 최대 `{random_values.max():+.12f}`다.",
            f"단독 OOF 최고 후보 `{audit.controls.best_member}`의 정확 복제 변화는 `{best_clone:+.12f}`다.",
            "",
            "| 복제 후보 | 기여 변화 |",
            "| --- | ---: |",
        ]
    )
    for member, change in audit.controls.clone_changes.items():
        lines.append(f"| {member} | {change:+.12f} |")
    lines.extend(
        [
            "",
            f"두 대조를 합친 영점 대역은 `{audit.controls.lower:+.12f}`에서 `{audit.controls.upper:+.12f}`다.",
            "",
            "## 판정 경계",
            "",
            "OOF와 시험 예측 양쪽 배열 해시가 같은 후보는 정확 중복으로 제거한다.",
            "정확 중복 제거 뒤 OOF 스피어만 순위 상관이 0.998 이상인 후보끼리는 단독 OOF가 높은 후보만 유지한다.",
            "균등 순위 평균의 제외 기여와 영점 대역은 후보 제거 기준이 아니다.",
            "무결성과 중복 검사를 통과한 후보는 모두 nested OOF 평가에 넘긴다.",
            "",
        ]
    )
    return "\n".join(lines)


def _git_file(commit: str, path: str) -> bytes | None:
    shown = subprocess.run(
        ["git", "show", f"{commit}:{path}"], capture_output=True, check=False
    )
    return shown.stdout if shown.returncode == 0 else None


def load_mlflow_candidates(pool: Pool, tracking_uri: str = TRACKING_URI) -> list[CandidateArtifacts]:
    """현재 MLflow 실행 저장소에서 풀 장부의 감사 산출물을 읽는다."""
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    candidates = []
    for member in pool.members:
        try:
            run = client.get_run(member.run_id)
            root = client.list_artifacts(member.run_id)
            names = {item.path for item in root}
            config_names = sorted(
                name for name in names if name.endswith((".yaml", ".yml"))
            )
            if len(config_names) != 1:
                raise PoolAuditError(
                    f"run {member.run_id}의 설정 YAML이 하나가 아니다: {config_names}"
                )
            config_name = config_names[0]
            paths = {
                name: Path(client.download_artifacts(member.run_id, name))
                for name in (config_name, "oof.parquet", "test_pred.parquet")
            }
            seed_oofs = {}
            for seed in CONFIRM_SEEDS:
                name = f"oof_seed_{seed}.parquet"
                if name in names:
                    seed_oofs[seed] = pd.read_parquet(
                        client.download_artifacts(member.run_id, name)
                    )
            commit = run.data.tags.get("git_commit", "")
            candidates.append(
                CandidateArtifacts(
                    member=member,
                    status=run.info.status,
                    params=dict(run.data.params),
                    metrics=dict(run.data.metrics),
                    tags=dict(run.data.tags),
                    config_bytes=paths[config_name].read_bytes(),
                    committed_config_bytes=_git_file(commit, f"configs/{config_name}"),
                    oof=pd.read_parquet(paths["oof.parquet"]),
                    test_pred=pd.read_parquet(paths["test_pred.parquet"]),
                    seed_oofs=seed_oofs,
                )
            )
        except (MlflowException, OSError) as exc:
            raise PoolAuditError(f"run {member.run_id} 산출물을 읽지 못했다: {exc}") from exc
    return candidates


def load_context() -> AuditContext:
    for path in (TRAIN_PATH, TEST_PATH, FOLDS_PATH):
        if not path.exists():
            raise PoolAuditError(f"필수 입력 없음: {path}")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    folds = pd.read_parquet(FOLDS_PATH)
    if train[ID].duplicated().any() or test[ID].duplicated().any() or folds[ID].duplicated().any():
        raise PoolAuditError("train, test 또는 folds에 id 중복이 있다.")
    if train[ID].map(folds.set_index(ID)["fold"]).isna().any():
        raise PoolAuditError("train id 일부가 folds.parquet에 없다.")
    return AuditContext(
        train=train,
        test=test,
        folds=folds,
        input_hashes={
            "train": file_sha256(TRAIN_PATH),
            "test": file_sha256(TEST_PATH),
            "folds": file_sha256(FOLDS_PATH),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF 후보 풀 품질·다양성 감사 (#63)")
    parser.add_argument("--output", type=Path, help=f"Markdown 저장 경로 (권장: {DEFAULT_OUTPUT})")
    parser.add_argument("--random-controls", type=int, default=NULL_RANDOM_COUNT)
    parser.add_argument("--null-seed", type=int, default=NULL_SEED)
    args = parser.parse_args()
    try:
        pool = Pool.load()
        if not pool.members:
            raise PoolAuditError("후보 풀이 비어 있다.")
        audit = audit_pool(
            load_mlflow_candidates(pool),
            load_context(),
            null_seed=args.null_seed,
            random_count=args.random_controls,
        )
        report = render_markdown(audit)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n")
            print(f"감사 보고서 저장: {args.output}")
        else:
            print(report)
    except (PoolAuditError, ValueError, KeyError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
