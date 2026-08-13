"""개선 판정 module. ADR 0001의 판정 규칙을 한곳에 모은다. (지도 #91, #93)

compare·pool CLI는 이 module의 caller다. 판정 함수는 통과 여부와 근거 값을 담은
구조화된 Verdict를 돌려주고, 한국어 리포트 문장은 CLI renderer가 근거 값에서 만든다.
증거 부족이나 전제 위반으로 판정을 수행할 수 없으면 JudgmentError(판정 불가)를
던지고, CLI가 종료 메시지로 번역한다.

판정 규칙은 ADR 0001 계열 1(특성·단일 모델 challenger)의 2단계 판정이다.

- 스크리닝(seed 42 단일): champion의 같은 시드(seed 42) OOF AUC(champion.yaml의
  seed_aucs[42]) 대비 개선이 0 이상이면 확정 재검증 자격을 얻는다.
  시드 평균본 OOF AUC를 기준선으로 쓰면 시드 평균 이득(약 +0.0003)이 문턱에 섞여
  같은 시드의 실재 개선을 걸러내므로 짝지은 비교여야 한다. (#74 개정)
  통과는 채택이 아니다.
- 확정 재검증(3시드 평균본):
  - 시드 평균본 OOF AUC가 champion 대비 +0.0001 이상.
  - 3시드 중 2시드 이상에서 같은 시드의 champion 대비 시드별 OOF AUC 개선이 0보다 크다.
  - 개선 폭이 +0.0001 이상 +0.0002 미만인 경계 구간이면 시드 평균 fold 점수 5개 중
    3개 이상 승리를 추가로 요구한다. 그 외 구간에서 fold 승리 수는 보조 증거로 기록만 한다.
- 새 피처는 fold별 gain importance 평균이 플라시보 평균보다 높아야 한다.
  확정 재검증의 게이트이며, 스크리닝에서는 참고로만 쓴다.
- 플라시보 파생 카나리아(placebo_noise_te 등)의 중요도가 플라시보 원본보다 높으면
  누수로 보고 그 run은 어느 단계에서도 판정에 쓰지 않는다. (#33)
- 대리 스크리닝: 느린 champion 모델 계열에 앞서 빠른 모델 계열의 동일 조건
  기준 실행과 짝지어 후보를 거른다. 기준 실행과 challenger 모두 seed 42의 동일한
  모델 설정과 입력 자료여야 하고 challenger에는 새 특성만 추가돼야 한다.
- 단일 시드 개선 폭이 +0.0003을 넘으면 그대로 채택하던 SINGLE_SEED_MARGIN 규칙은
  폐기됐다. champion은 항상 3시드 평균본이라는 불변식을 지킨다. (ADR 0001)

metric 이름 규약(auc_oof_seed_*, auc_fold_*)의 의미 해석도 이 module 소관이다.
실행 저장소(runs)는 기록 원형을 그대로 돌려주고, 여기의 파싱 helper가 해석한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .features import PLACEBO
from .runs import RunStore

AUC_THRESHOLD = 0.0001  # 확정 문턱. 이 미만의 개선은 CV 잡음으로 본다. (#15, ADR 0001)
BOUNDARY_UPPER = 0.0002  # 이 미만의 개선 폭은 경계 구간으로 fold 승리 게이트를 추가한다.
SCREENING_SEEDS = [42]  # 스크리닝 시드. 고정. (ADR 0001)
CONFIRM_SEEDS = [42, 43, 44]  # 확정 재검증 시드. 고정. (ADR 0001)
SEED_WIN_MIN = 2  # 3시드 중 시드별 개선이 필요한 최소 시드 수.
FOLD_WIN_MIN = 3  # 경계 구간에서 5개 fold 중 필요한 최소 승리 수.
CHAMPION_PATH = Path("artifacts/champion.yaml")


class JudgmentError(Exception):
    """판정 불가: 증거 부족이나 전제 위반. CLI가 sys.exit로 번역한다."""


_SEED_AUC_PREFIX = "auc_oof_seed_"
_FOLD_AUC_PREFIX = "auc_fold_"


def seed_auc_metric(seed: int) -> str:
    return f"{_SEED_AUC_PREFIX}{seed}"


def _suffixed_ints(metrics: dict[str, float], prefix: str) -> dict[int, float]:
    return {
        int(k.rsplit("_", 1)[1]): v for k, v in metrics.items() if k.startswith(prefix)
    }


def seed_aucs_of(metrics: dict[str, float]) -> dict[int, float]:
    """auc_oof_seed_<seed> metric들을 시드별 OOF AUC로 해석한다. 구 버전 실행에는 없다."""
    return _suffixed_ints(metrics, _SEED_AUC_PREFIX)


def fold_aucs_of(metrics: dict[str, float]) -> dict[int, float]:
    """auc_fold_<fold> metric들을 fold별 AUC로 해석한다(시드 평균본 기준)."""
    return _suffixed_ints(metrics, _FOLD_AUC_PREFIX)


@dataclass(frozen=True)
class RunFacts:
    """판정에 필요한 MLflow run의 단면."""

    run_id: str
    experiment: str
    auc_oof: float
    features: set[str]
    seeds: list[int]
    seed_aucs: dict[int, float]  # auc_oof_seed_* metric. 구 버전 실행에는 없다.
    fold_aucs: dict[int, float]  # auc_fold_* metric (시드 평균본 기준).
    git_commit: str
    importance: pd.DataFrame  # feature, fold, seed, gain
    model_params: dict[str, str] = field(default_factory=dict)
    input_hashes: dict[str, str] = field(default_factory=dict)


def load_run_facts(run_id: str, store: RunStore) -> RunFacts:
    meta = store.facts_of(run_id)
    config = store.config_of(run_id)
    model_params = {"model.kind": str(config["model"]["kind"])}
    model_params.update(
        {f"model.params.{key}": str(value) for key, value in config["model"]["params"].items()}
    )
    model_params.update(
        {f"model.fit.{key}": str(value) for key, value in config["model"]["fit"].items()}
    )
    return RunFacts(
        run_id=run_id,
        experiment=meta.params["experiment"],
        auc_oof=meta.metrics["auc_oof"],
        features=set(meta.params["features"].split(",")),
        seeds=[int(s) for s in meta.params["seeds"].split(",")],
        seed_aucs=seed_aucs_of(meta.metrics),
        fold_aucs=fold_aucs_of(meta.metrics),
        git_commit=meta.tags["git_commit"],
        importance=store.importance_of(run_id),
        model_params=model_params,
        input_hashes={
            key: value for key, value in meta.tags.items() if key.startswith("sha256.")
        },
    )


def _is_canary(feature: str) -> bool:
    """플라시보에서 파생된 카나리아 피처인지. 예: placebo_noise_te. (#33)"""
    return feature.startswith(f"{PLACEBO}_")


def _mean_gain(importance: pd.DataFrame) -> pd.Series:
    """feature별 fold×seed 평균 gain."""
    return importance.groupby("feature")["gain"].mean()


@dataclass(frozen=True)
class GainCheck:
    """피처 하나의 평균 gain을 플라시보 기준값과 비교한 결과."""

    feature: str
    gain: float | None  # 미기록이면 None. 카나리아는 미기록을 0.0으로 본다.
    ok: bool


@dataclass(frozen=True)
class CanaryReport:
    """placebo 카나리아 검사: 파생 피처가 플라시보 원본보다 중요해지면 누수다. (#33)"""

    placebo_gain: float | None  # 플라시보 원본의 평균 gain. 미기록이면 None(전부 실패).
    checks: list[GainCheck]
    ok: bool


@dataclass(frozen=True)
class NewFeatureReport:
    """기준에 없던 새 피처의 평균 gain이 플라시보보다 높은지. 새 피처가 없으면 항상 참."""

    new_features: list[str]
    placebo_gain: float | None  # 새 피처가 있는데 None이면 판정 근거 자체가 없다(실패).
    checks: list[GainCheck]
    ok: bool


def check_canaries(challenger: RunFacts) -> CanaryReport:
    mean_gain = _mean_gain(challenger.importance)
    placebo_gain = mean_gain.get(PLACEBO)
    placebo_gain = float(placebo_gain) if placebo_gain is not None else None
    checks = []
    for canary in sorted(f for f in challenger.features if _is_canary(f)):
        gain = float(mean_gain.get(canary, 0.0))
        checks.append(
            GainCheck(canary, gain, placebo_gain is not None and gain < placebo_gain)
        )
    return CanaryReport(placebo_gain, checks, all(c.ok for c in checks))


def check_new_features(base_features: set[str], challenger: RunFacts) -> NewFeatureReport:
    new_features = sorted(
        f for f in challenger.features - base_features - {PLACEBO} if not _is_canary(f)
    )
    if not new_features:
        return NewFeatureReport([], None, [], ok=True)
    mean_gain = _mean_gain(challenger.importance)
    if PLACEBO not in mean_gain.index:
        return NewFeatureReport(new_features, None, [], ok=False)
    placebo_gain = float(mean_gain[PLACEBO])
    checks = []
    for feature in new_features:
        gain = mean_gain.get(feature)
        gain = float(gain) if gain is not None else None
        checks.append(GainCheck(feature, gain, gain is not None and gain > placebo_gain))
    return NewFeatureReport(new_features, placebo_gain, checks, all(c.ok for c in checks))


@dataclass(frozen=True)
class ScreeningVerdict:
    """스크리닝 판정의 근거 값. 통과는 확정 재검증 자격이지 채택이 아니다."""

    seed: int  # 짝지은 비교의 시드.
    baseline_auc: float  # champion의 같은 시드 OOF AUC.
    challenger_auc: float
    delta: float
    auc_ok: bool
    canary: CanaryReport
    new_features: NewFeatureReport  # 스크리닝에서는 게이트가 아니라 참고다.
    passed: bool


@dataclass(frozen=True)
class ProxyScreeningVerdict:
    """대리 스크리닝 판정의 근거 값. 통과는 공식 스크리닝 진입 자격이다."""

    baseline_auc: float
    challenger_auc: float
    delta: float
    auc_ok: bool
    canary: CanaryReport
    new_features: NewFeatureReport
    passed: bool


@dataclass(frozen=True)
class SeedComparison:
    """확정 재검증의 시드별 짝지은 비교 한 건."""

    seed: int
    champion_auc: float
    challenger_auc: float
    delta: float
    win: bool


@dataclass(frozen=True)
class ConfirmationVerdict:
    """확정 재검증 판정의 근거 값."""

    champion_auc: float
    challenger_auc: float
    delta: float
    auc_ok: bool
    seed_comparisons: list[SeedComparison]
    seed_wins: int
    seed_ok: bool
    fold_wins: int
    fold_total: int
    boundary: bool  # 경계 구간이면 fold 승리가 게이트, 아니면 보조 증거다.
    fold_ok: bool
    canary: CanaryReport
    new_features: NewFeatureReport
    passed: bool


def judge_screening(champion: dict, challenger: RunFacts) -> ScreeningVerdict:
    """스크리닝: 같은 시드끼리 짝지어 개선 >= 0이면 확정 재검증 자격. (ADR 0001, #74 개정)"""
    seed = SCREENING_SEEDS[0]
    if "seed_aucs" not in champion or seed not in {int(k) for k in champion["seed_aucs"]}:
        raise JudgmentError(
            f"champion.yaml에 seed_aucs[{seed}]가 없어 짝지은 스크리닝 비교를 할 수 없다. "
            "동일 설정·시드로 champion을 재실행해 시드별 지표를 백필할 것."
        )
    baseline = {int(k): v for k, v in champion["seed_aucs"].items()}[seed]
    delta = challenger.auc_oof - baseline
    auc_ok = delta >= 0.0
    canary = check_canaries(challenger)
    new_features = check_new_features(set(champion["features"].split(",")), challenger)
    return ScreeningVerdict(
        seed=seed,
        baseline_auc=baseline,
        challenger_auc=challenger.auc_oof,
        delta=delta,
        auc_ok=auc_ok,
        canary=canary,
        new_features=new_features,
        passed=auc_ok and canary.ok,
    )


def judge_proxy_screening(
    baseline: RunFacts, challenger: RunFacts
) -> ProxyScreeningVerdict:
    """동일한 빠른 모델 계열 안에서 새 특성의 공식 스크리닝 진입 자격을 판정한다."""
    if baseline.seeds != SCREENING_SEEDS or challenger.seeds != SCREENING_SEEDS:
        raise JudgmentError(
            f"대리 스크리닝은 기준 실행과 challenger 모두 시드가 {SCREENING_SEEDS}여야 한다. "
            f"(기준 실행: {baseline.seeds}, challenger: {challenger.seeds})"
        )
    if not baseline.features < challenger.features:
        raise JudgmentError(
            "대리 스크리닝 challenger는 기준 실행의 모든 특성을 유지하고 새 특성을 "
            "하나 이상 추가해야 한다."
        )
    if baseline.model_params != challenger.model_params:
        raise JudgmentError(
            "대리 스크리닝의 모델 설정이 기준 실행과 다르다. "
            f"기준 실행={baseline.model_params}, challenger={challenger.model_params}"
        )
    if baseline.input_hashes != challenger.input_hashes:
        raise JudgmentError(
            "대리 스크리닝의 입력 자료 해시가 기준 실행과 다르다. "
            f"기준 실행={baseline.input_hashes}, challenger={challenger.input_hashes}"
        )

    delta = challenger.auc_oof - baseline.auc_oof
    auc_ok = delta >= 0.0
    canary = check_canaries(challenger)
    new_features = check_new_features(baseline.features, challenger)
    return ProxyScreeningVerdict(
        baseline_auc=baseline.auc_oof,
        challenger_auc=challenger.auc_oof,
        delta=delta,
        auc_ok=auc_ok,
        canary=canary,
        new_features=new_features,
        passed=auc_ok and canary.ok and new_features.ok,
    )


def require_confirmation_facts(champion: dict, challenger: RunFacts) -> None:
    """확정 재검증에 필요한 시드별·fold별 기준값이 양쪽에 있는지 검증한다."""
    missing = [s for s in CONFIRM_SEEDS if s not in challenger.seed_aucs]
    if missing:
        names = ", ".join(seed_auc_metric(s) for s in missing)
        raise JudgmentError(
            f"challenger run에 시드별 OOF AUC 지표({names})가 없다. "
            "판정 계약(#70) 이전 실행이므로 갱신된 파이프라인으로 재실행할 것."
        )
    if "seed_aucs" not in champion or "fold_aucs" not in champion:
        raise JudgmentError(
            "champion.yaml에 seed_aucs/fold_aucs가 없다. 판정 계약(#70) 이전 champion이므로 "
            "동일 설정·시드로 champion을 재실행해 시드별 지표를 백필한 뒤 판정할 것."
        )


def judge_confirmation(champion: dict, challenger: RunFacts) -> ConfirmationVerdict:
    """확정 재검증: 시드 평균본 문턱 + 2/3 시드 개선 + 경계 구간 fold 승리 게이트. (ADR 0001)"""
    require_confirmation_facts(champion, challenger)
    delta = challenger.auc_oof - champion["oof_auc"]
    auc_ok = delta >= AUC_THRESHOLD

    champion_seed_aucs = {int(k): v for k, v in champion["seed_aucs"].items()}
    seed_comparisons = []
    for seed in CONFIRM_SEEDS:
        seed_delta = challenger.seed_aucs[seed] - champion_seed_aucs[seed]
        seed_comparisons.append(
            SeedComparison(
                seed=seed,
                champion_auc=champion_seed_aucs[seed],
                challenger_auc=challenger.seed_aucs[seed],
                delta=seed_delta,
                win=seed_delta > 0,
            )
        )
    seed_wins = sum(c.win for c in seed_comparisons)
    seed_ok = seed_wins >= SEED_WIN_MIN

    champion_fold_aucs = {int(k): v for k, v in champion["fold_aucs"].items()}
    fold_wins = sum(
        challenger.fold_aucs[f] > champion_fold_aucs[f] for f in sorted(champion_fold_aucs)
    )
    boundary = AUC_THRESHOLD <= delta < BOUNDARY_UPPER
    fold_ok = fold_wins >= FOLD_WIN_MIN if boundary else True

    canary = check_canaries(challenger)
    new_features = check_new_features(set(champion["features"].split(",")), challenger)
    return ConfirmationVerdict(
        champion_auc=champion["oof_auc"],
        challenger_auc=challenger.auc_oof,
        delta=delta,
        auc_ok=auc_ok,
        seed_comparisons=seed_comparisons,
        seed_wins=seed_wins,
        seed_ok=seed_ok,
        fold_wins=fold_wins,
        fold_total=len(champion_fold_aucs),
        boundary=boundary,
        fold_ok=fold_ok,
        canary=canary,
        new_features=new_features,
        passed=auc_ok and seed_ok and fold_ok and canary.ok and new_features.ok,
    )
