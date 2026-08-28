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
  - 시드 평균본 OOF AUC가 champion 대비 +0.00002 이상.
  - 3시드 중 2시드 이상에서 같은 시드의 champion 대비 시드별 OOF AUC 개선이 0보다 크다.
  - 개선 폭이 +0.00002 이상 +0.0002 미만인 경계 구간이면 시드 평균 fold 점수 5개 중
    3개 이상 승리를 추가로 요구한다. 그 외 구간에서 fold 승리 수는 보조 증거로 기록만 한다.
- 새 피처는 fold별 gain importance 평균이 플라시보 평균보다 높아야 한다.
  확정 재검증의 게이트이며, 스크리닝에서는 참고로만 쓴다.
- 플라시보 파생 카나리아(placebo_noise_te 등)의 중요도가 플라시보 원본과 0 중
  큰 값보다 높으면 누수로 보고 그 run은 어느 단계에서도 판정에 쓰지 않는다. (#33)
  음수 permutation importance는 0 중요도보다 엄격한 영가설 상한으로 쓰지 않는다.
- 플라시보 게이트의 기준값은 플라시보 원본의 평균 gain 하나뿐이고, 기준값 미기록은
  실패다. 판정 게이트는 이 module의 _gate가 유일한 소스다(#94). tracking의 경고와
  summary의 요약표는 게이트가 아니므로 평균 gain·기준값 helper만 공유한다.
- 대리 스크리닝: 느린 champion 모델 계열에 앞서 빠른 모델 계열의 동일 조건
  기준 실행과 짝지어 후보를 거른다. 기준 실행과 challenger 모두 seed 42의 동일한
  모델 설정과 입력 자료여야 하고 challenger에는 새 특성만 추가돼야 한다.
- 단일 시드 개선 폭이 +0.0003을 넘으면 그대로 채택하던 SINGLE_SEED_MARGIN 규칙은
  폐기됐다. champion은 항상 3시드 평균본이라는 불변식을 지킨다. (ADR 0001)

계열 3(앙상블)의 판정도 이 module 소관이다(#104).

- 채택 가능: 결합 전략의 nested OOF AUC가 champion(3시드 평균본) OOF AUC 대비
  +0.00002 이상.
- 확정: 채택 가능한 전략 중 nested OOF AUC가 가장 높은 전략을 추천한다.
  채택 가능 전략이 없으면 "채택 없음, 단독 champion 유지"다.
- 전략 간 fold별 승리 수는 보조 증거로 기록만 한다. (ADR 0001)

가중 OOF(test 결측 패턴 구성비로 재채점한 OOF AUC)도 이 module 소관이다(#383).

- train과 test는 어느 칸이 비는지가 다르고(열별 결측률 최대 3.4%p 차이), 결측이
  난이도를 지배한다. 그래서 OOF는 test보다 어려운 표본에서 잰 값이다.
- weighted_oof_auc는 12개 설명변수의 결측 마스크를 행 키로 삼아
  `w = P_test(패턴) / P_train(패턴)`로 재가중해 OOF AUC를 다시 잰다.
- 추가 눈금이며 기존 판정 경로의 수치를 바꾸지 않는다. 구성원 채택·교체 판정에는
  쓰지 않고(32구성원이 균일 상승해 선택을 바꾸지 못한다), 최종 결합 전략(#337),
  제출 2장(#69), 결측 처리 방식 자체를 바꾸는 후보(#360)의 판정에만 쓴다.

계열 2(다양성 구성원)의 풀 진입 판정도 이 module 소관이다(#95).

- 진입 하한: 시드 평균본 OOF AUC가 진입 시점 champion − 0.01 이상. 하한은 진입
  시점에만 적용하고 champion 갱신 때마다 재심사하지 않는다.
- 중복 게이트: 풀 내 최근접 구성원과 OOF 예측의 스피어만 순위 상관이 0.998 이상이면
  중복으로 보고 성능이 높은 쪽만 유지한다. 상관은 중복 제거 전용이다.
- 기여 참고값: 표준 평가 앙상블(풀 전체의 순위 평균)의 OOF AUC에서 해당 구성원을
  제외했을 때의 변화를 기록한다. 특정 결합 방식의 한계만 보여 주므로 진입이나 제거
  게이트로 쓰지 않는다. 풀이 비어 있으면 계산하지 않는다.

채택 자격(장부에 오르는 실행의 기록 조건)도 여기가 단일 소스다: 3시드 평균본이고,
git_dirty가 아니고, 커밋된 folds와 sha256이 일치해야 한다. (#14 관행)
compare --adopt와 pool --admit이 같은 검사를 공유하고, submit도 같은 함수를 쓸 수
있는 시그니처지만 대회 막판이라 전환하지 않았다(설계 호환만 확보, 지도 #91).

metric 이름 규약(auc_oof_seed_*, auc_fold_*)의 의미 해석도 이 module 소관이다.
실행 저장소(runs)는 기록 원형을 그대로 돌려주고, 여기의 파싱 helper가 해석한다.
판정의 기준이 되는 장부(champion·후보 풀)의 원본과 YAML 해석은 ledger module
소관이며(#96), 판정 함수는 그 타입을 읽기만 한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .data import ID, TARGET, TRAIN_PATH
from .ensemble import (
    CANDIDATE_POOL_CORE_COMBINER_NAMES,
    DEFAULT_COMBINER_NAMES,
    MISSINGNESS_TEST_PATH,
    rank_mean,
)
from .features import PLACEBO
from .ledger import POOL_PATH, Champion, Pool
from .runs import RunStore, RunStoreError
from .training_state_manifest import (
    MANIFEST_NAME as TRAINING_STATE_MANIFEST_NAME,
    RUN_KIND as TRAINING_STATE_RUN_KIND,
    TrainingStateManifestError,
    validate_candidate_parent_lineage,
    validate_candidate_manifest,
)

AUC_THRESHOLD = 0.00002  # 계열 1·3 공통 채택 문턱. (#15, #64, ADR 0001)
BOUNDARY_UPPER = 0.0002  # 이 미만의 개선 폭은 경계 구간으로 fold 승리 게이트를 추가한다.
SCREENING_SEEDS = [42]  # 스크리닝 시드. 고정. (ADR 0001)
CONFIRM_SEEDS = [42, 43, 44]  # 확정 재검증 시드. 고정. (ADR 0001)
SEED_WIN_MIN = 2  # 3시드 중 시드별 개선이 필요한 최소 시드 수.
FOLD_WIN_MIN = 3  # 경계 구간에서 5개 fold 중 필요한 최소 승리 수.
ENTRY_FLOOR_MARGIN = 0.01  # champion − 0.01이 풀 진입 하한. (ADR 0001)
DUPLICATE_SPEARMAN = 0.998  # 이 이상이면 중복으로 본다. (ADR 0001)
FOLDS_PATH = Path("artifacts/folds.parquet")
POOL_JUDGMENT_CONTRACT_VERSION = "candidate-pool-v2"
POOL_JUDGMENT_LEGACY_CONTRACT_VERSIONS = ("candidate-pool-v1",)
POOL_JUDGMENT_SUPPORTED_CONTRACT_VERSIONS = (
    *POOL_JUDGMENT_LEGACY_CONTRACT_VERSIONS,
    POOL_JUDGMENT_CONTRACT_VERSION,
)
POOL_EQUIVALENCE_BAND_UPPER = 0.000027669802


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
    git_dirty: bool = False  # 채택 자격 검사용. load_run_facts는 항상 태그에서 채운다.
    folds_sha256: str = ""


_RESULT_HASH_TAGS = {
    "sha256.fold_feature_reuse",
    "sha256.model_training_diagnostics",
    "sha256.oof_prediction",
    "sha256.submission",
    "sha256.training_row_evidence",
    "sha256.training_state_manifest",
    "sha256.training_state_recovery",
}
_RESULT_HASH_PREFIXES = ("sha256.observability.",)


def _input_hashes_of(tags: dict[str, str]) -> dict[str, str]:
    """입력 계보 해시만 고른다. 실행 결과물 해시는 짝비교 입력 동일성에서 제외한다."""
    return {
        key: value
        for key, value in tags.items()
        if key.startswith("sha256.")
        and key not in _RESULT_HASH_TAGS
        and not key.startswith(_RESULT_HASH_PREFIXES)
    }


def load_run_facts(run_id: str, store: RunStore) -> RunFacts:
    meta = store.facts_of(run_id)
    _require_publishable_run(meta, store)
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
        input_hashes=_input_hashes_of(meta.tags),
        git_dirty=meta.tags["git_dirty"] == "True",
        folds_sha256=meta.tags["sha256.folds"],
    )


def _is_canary(feature: str) -> bool:
    """플라시보에서 파생된 카나리아 피처인지. 예: placebo_noise_te. (#33)"""
    return feature.startswith(f"{PLACEBO}_")


@dataclass(frozen=True)
class GainCheck:
    """피처 하나의 평균 gain을 플라시보 기준값과 비교한 결과."""

    feature: str
    gain: float | None  # 미기록이면 None. 카나리아는 미기록을 0.0으로 본다.
    ok: bool


def mean_gain_of(importance: pd.DataFrame) -> pd.Series:
    """feature별 fold×seed 평균 gain. 게이트·경고·요약표가 공유하는 유일한 정의다."""
    return importance.groupby("feature")["gain"].mean()


def placebo_gain_of(mean_gain: pd.Series) -> float | None:
    """플라시보 게이트의 기준값: 플라시보 원본의 평균 gain. 미기록이면 None."""
    gain = mean_gain.get(PLACEBO)
    return float(gain) if gain is not None else None


def _gate(
    feature: str, gain: float | None, placebo_gain: float | None, *, above: bool
) -> GainCheck:
    """플라시보 게이트 한 건. 기준값 미기록(None)이나 gain 미기록(None)은 실패다.

    above=True면 기준값 초과(새 피처의 기여 증거)가 통과다.
    False면 플라시보와 0 중 큰 상한 이하(카나리아 무해)가 통과다.
    """
    ok = (
        placebo_gain is not None
        and gain is not None
        and (gain > placebo_gain if above else gain <= max(placebo_gain, 0.0))
    )
    return GainCheck(feature, gain, ok)


@dataclass(frozen=True)
class CanaryReport:
    """placebo 카나리아 검사: 파생 피처가 영가설 상한보다 중요해지면 누수다. (#33)"""

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


def check_canaries(features: set[str], mean_gain: pd.Series) -> CanaryReport:
    placebo_gain = placebo_gain_of(mean_gain)
    checks = []
    for canary in sorted(f for f in features if _is_canary(f)):
        # 카나리아 미기록은 0.0으로 본다: 모델이 완전히 무시했다는 뜻이라 무해하다.
        gain = float(mean_gain.get(canary, 0.0))
        checks.append(_gate(canary, gain, placebo_gain, above=False))
    return CanaryReport(placebo_gain, checks, all(c.ok for c in checks))


def check_new_features(
    base_features: set[str], features: set[str], mean_gain: pd.Series
) -> NewFeatureReport:
    new_features = sorted(
        f for f in features - base_features - {PLACEBO} if not _is_canary(f)
    )
    if not new_features:
        return NewFeatureReport([], None, [], ok=True)
    placebo_gain = placebo_gain_of(mean_gain)
    if placebo_gain is None:
        return NewFeatureReport(new_features, None, [], ok=False)
    checks = []
    for feature in new_features:
        gain = mean_gain.get(feature)
        gain = float(gain) if gain is not None else None
        checks.append(_gate(feature, gain, placebo_gain, above=True))
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


def judge_screening(champion: Champion, challenger: RunFacts) -> ScreeningVerdict:
    """스크리닝: 같은 시드끼리 짝지어 개선 >= 0이면 확정 재검증 자격. (ADR 0001, #74 개정)"""
    seed = SCREENING_SEEDS[0]
    if seed not in champion.seed_aucs:
        raise JudgmentError(
            f"champion.yaml에 seed_aucs[{seed}]가 없어 짝지은 스크리닝 비교를 할 수 없다. "
            "동일 설정·시드로 champion을 재실행해 시드별 지표를 백필할 것."
        )
    baseline = champion.seed_aucs[seed]
    delta = challenger.auc_oof - baseline
    auc_ok = delta >= 0.0
    mean_gain = mean_gain_of(challenger.importance)
    canary = check_canaries(challenger.features, mean_gain)
    new_features = check_new_features(champion.features, challenger.features, mean_gain)
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
    mean_gain = mean_gain_of(challenger.importance)
    canary = check_canaries(challenger.features, mean_gain)
    new_features = check_new_features(baseline.features, challenger.features, mean_gain)
    return ProxyScreeningVerdict(
        baseline_auc=baseline.auc_oof,
        challenger_auc=challenger.auc_oof,
        delta=delta,
        auc_ok=auc_ok,
        canary=canary,
        new_features=new_features,
        passed=auc_ok and canary.ok and new_features.ok,
    )


def require_confirmation_facts(champion: Champion, challenger: RunFacts) -> None:
    """확정 재검증에 필요한 시드별·fold별 기준값이 양쪽에 있는지 검증한다."""
    missing = [s for s in CONFIRM_SEEDS if s not in challenger.seed_aucs]
    if missing:
        names = ", ".join(seed_auc_metric(s) for s in missing)
        raise JudgmentError(
            f"challenger run에 시드별 OOF AUC 지표({names})가 없다. "
            "판정 계약(#70) 이전 실행이므로 갱신된 파이프라인으로 재실행할 것."
        )
    if not champion.seed_aucs or not champion.fold_aucs:
        raise JudgmentError(
            "champion.yaml에 seed_aucs/fold_aucs가 없다. 판정 계약(#70) 이전 champion이므로 "
            "동일 설정·시드로 champion을 재실행해 시드별 지표를 백필한 뒤 판정할 것."
        )


def judge_confirmation(champion: Champion, challenger: RunFacts) -> ConfirmationVerdict:
    """확정 재검증: 시드 평균본 문턱 + 2/3 시드 개선 + 경계 구간 fold 승리 게이트. (ADR 0001)"""
    require_confirmation_facts(champion, challenger)
    delta = challenger.auc_oof - champion.oof_auc
    auc_ok = delta >= AUC_THRESHOLD

    seed_comparisons = []
    for seed in CONFIRM_SEEDS:
        seed_delta = challenger.seed_aucs[seed] - champion.seed_aucs[seed]
        seed_comparisons.append(
            SeedComparison(
                seed=seed,
                champion_auc=champion.seed_aucs[seed],
                challenger_auc=challenger.seed_aucs[seed],
                delta=seed_delta,
                win=seed_delta > 0,
            )
        )
    seed_wins = sum(c.win for c in seed_comparisons)
    seed_ok = seed_wins >= SEED_WIN_MIN

    fold_wins = sum(
        challenger.fold_aucs[f] > champion.fold_aucs[f] for f in sorted(champion.fold_aucs)
    )
    boundary = AUC_THRESHOLD <= delta < BOUNDARY_UPPER
    fold_ok = fold_wins >= FOLD_WIN_MIN if boundary else True

    mean_gain = mean_gain_of(challenger.importance)
    canary = check_canaries(challenger.features, mean_gain)
    new_features = check_new_features(champion.features, challenger.features, mean_gain)
    return ConfirmationVerdict(
        champion_auc=champion.oof_auc,
        challenger_auc=challenger.auc_oof,
        delta=delta,
        auc_ok=auc_ok,
        seed_comparisons=seed_comparisons,
        seed_wins=seed_wins,
        seed_ok=seed_ok,
        fold_wins=fold_wins,
        fold_total=len(champion.fold_aucs),
        boundary=boundary,
        fold_ok=fold_ok,
        canary=canary,
        new_features=new_features,
        passed=auc_ok and seed_ok and fold_ok and canary.ok and new_features.ok,
    )


@dataclass(frozen=True)
class PoolCandidate:
    """풀 진입 판정에 필요한 실행의 단면."""

    run_id: str
    experiment: str
    auc_oof: float
    seeds: list[int]
    git_dirty: bool
    folds_sha256: str
    oof: pd.Series  # id 인덱스의 OOF 예측.


def load_candidate(run_id: str, store: RunStore) -> PoolCandidate:
    meta = store.facts_of(run_id)
    _require_publishable_run(meta, store)
    return PoolCandidate(
        run_id=run_id,
        experiment=meta.params["experiment"],
        auc_oof=meta.metrics["auc_oof"],
        seeds=[int(s) for s in meta.params["seeds"].split(",")],
        git_dirty=meta.tags["git_dirty"] == "True",
        folds_sha256=meta.tags["sha256.folds"],
        oof=store.oof_of(run_id),
    )


def _require_publishable_run(meta, store: RunStore) -> None:
    if meta.status != "FINISHED":
        raise ValueError(f"run {meta.run_id}이 완료 상태가 아니다: {meta.status}")
    try:
        config = store.config_of(meta.run_id)
    except RunStoreError:
        config = None
    if (
        isinstance(config, dict)
        and config.get("training_state") is not None
        and meta.tags.get("run.kind") != TRAINING_STATE_RUN_KIND
    ):
        raise ValueError(
            f"run {meta.run_id}은 training_state config지만 snapshot 실행 정체성이 없다."
        )
    if meta.tags.get("run.kind") == TRAINING_STATE_RUN_KIND:
        try:
            training_state_document = validate_candidate_manifest(
                manifest_bytes=store.artifact_bytes_of(
                    meta.run_id, TRAINING_STATE_MANIFEST_NAME
                ),
                tags=meta.tags,
                params=meta.params,
                artifact_bytes_of=lambda name: store.artifact_bytes_of(
                    meta.run_id, name
                ),
            )
            validate_candidate_parent_lineage(
                child_run_id=meta.run_id,
                child_document=training_state_document,
                child_tags=meta.tags,
                facts_of=store.facts_of,
                artifact_bytes_of=store.artifact_bytes_of,
            )
        except TrainingStateManifestError as exc:
            raise ValueError(
                f"run {meta.run_id}의 학습 시점 게시 계약이 불완전하다: {exc}"
            ) from exc
    if meta.tags.get("judgment.eligible") == "false":
        raise ValueError(f"run {meta.run_id}은 판정 불가 부모 실행이다.")


def spearman(a: pd.Series, b: pd.Series) -> float:
    """OOF 예측 두 벌의 스피어만 순위 상관. 동순위는 평균 순위로 처리한다."""
    return float(np.corrcoef(a.rank().to_numpy(), b.rank().to_numpy())[0, 1])


def rank_ensemble_auc(preds: list[pd.Series], y: pd.Series) -> float:
    """표준 평가 앙상블: 구성원별 예측을 순위(백분위)로 바꿔 평균한 뒤 채점한다.

    수식은 ensemble의 균등 순위 평균 adapter가 소유한다(#104). 기여 참고값은
    "블록=전체 OOF"인 특수 사례라 같은 수식을 그대로 쓴다.
    """
    return float(roc_auc_score(y.to_numpy(), rank_mean(pd.concat(preds, axis=1))))


@dataclass(frozen=True)
class DuplicateCheck:
    """중복 게이트: 최근접(상관 최대) 구성원 하나와만 비교한다. (ADR 0001)"""

    nearest_run_id: str
    nearest_spearman: float
    nearest_auc: float
    duplicate: bool  # 상관이 문턱 이상인가.
    replace: bool  # 중복이되 후보가 더 높아 기존 구성원을 교체하는가.


@dataclass(frozen=True)
class ContributionCheck:
    """기여 참고값: 표준 평가 앙상블에 후보를 넣었을 때의 OOF AUC 변화."""

    auc_without: float
    auc_with: float
    contribution: float
    ok: bool


@dataclass(frozen=True)
class EntryVerdict:
    """풀 진입 판정의 근거 값."""

    champion_run_id: str
    champion_auc: float
    candidate_auc: float
    floor: float  # champion − ENTRY_FLOOR_MARGIN.
    floor_ok: bool
    duplicate: DuplicateCheck | None  # 풀이 비어 있으면 None.
    drop_run_id: str | None  # 중복 교체로 탈락시킬 기존 구성원.
    contribution: ContributionCheck | None  # 묻지 않으면 None.
    admit: bool


def judge_entry(
    pool: Pool, candidate: PoolCandidate, champion: Champion, store: RunStore, y: pd.Series
) -> EntryVerdict:
    """풀 진입 판정: 진입 하한 + 중복 게이트. (ADR 0001 계열 2)

    중복 게이트는 최종 논리곱이 아니라 탈락의 조기 확정이다: 중복인데 기존 구성원이
    더 높으면 그 자리에서 탈락이고, 후보가 더 높으면 교체 대상만 정한다.
    균등 순위 평균 기여는 참고값으로 계산하되 진입 여부에는 영향을 주지 않는다.
    """
    floor = champion.oof_auc - ENTRY_FLOOR_MARGIN
    floor_ok = candidate.auc_oof >= floor
    base = {
        "champion_run_id": champion.run_id,
        "champion_auc": champion.oof_auc,
        "candidate_auc": candidate.auc_oof,
        "floor": floor,
        "floor_ok": floor_ok,
    }

    members = pool.members
    if not members:
        return EntryVerdict(
            **base, duplicate=None, drop_run_id=None, contribution=None, admit=floor_ok
        )

    cand_pred = candidate.oof
    member_preds = {
        m.run_id: store.oof_of(m.run_id).reindex(cand_pred.index) for m in members
    }
    for run_id, pred in member_preds.items():
        assert pred.notna().all(), f"구성원 {run_id}의 OOF id가 후보와 일치하지 않는다."

    corrs = {run_id: spearman(cand_pred, pred) for run_id, pred in member_preds.items()}
    nearest_id = max(corrs, key=corrs.get)
    nearest = next(m for m in members if m.run_id == nearest_id)
    is_duplicate = corrs[nearest_id] >= DUPLICATE_SPEARMAN
    replace = is_duplicate and candidate.auc_oof > nearest.oof_auc
    duplicate = DuplicateCheck(
        nearest_run_id=nearest_id,
        nearest_spearman=float(corrs[nearest_id]),
        nearest_auc=float(nearest.oof_auc),
        duplicate=is_duplicate,
        replace=replace,
    )
    if is_duplicate and not replace:
        return EntryVerdict(
            **base, duplicate=duplicate, drop_run_id=None, contribution=None, admit=False
        )
    drop_run_id = nearest_id if replace else None

    # 기여 참고값: 교체로 빠질 구성원은 제외한 풀 기준으로 잰다.
    base_preds = [p for run_id, p in member_preds.items() if run_id != drop_run_id]
    if not base_preds:
        return EntryVerdict(
            **base, duplicate=duplicate, drop_run_id=drop_run_id, contribution=None,
            admit=floor_ok,
        )

    auc_without = rank_ensemble_auc(base_preds, y)
    auc_with = rank_ensemble_auc(base_preds + [cand_pred], y)
    delta = auc_with - auc_without
    contribution = ContributionCheck(
        auc_without=auc_without, auc_with=auc_with, contribution=delta, ok=delta > 0
    )
    return EntryVerdict(
        **base, duplicate=duplicate, drop_run_id=drop_run_id, contribution=contribution,
        admit=floor_ok,
    )


@dataclass(frozen=True)
class PoolAdmissionAuthorization:
    """후보 풀 판정 기록이 허용한 장부 변경 한 건."""

    judgment_id: str
    contract_version: str
    action: str
    candidate_run_id: str
    candidate_config: str
    replaced_run_id: str | None
    nested_oof_delta: float
    boundary_contribution: bool
    evidence_path: Path
    evidence_sha256: str
    record_path: Path
    record_sha256: str


def canonical_name_list_sha256(names: tuple[str, ...] | list[str]) -> str:
    """순서를 보존한 이름 목록의 정규 JSON SHA-256."""
    payload = json.dumps(
        list(names), ensure_ascii=False, separators=(",", ":")
    ).encode() + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_pool_record(condition: bool, message: str) -> None:
    if not condition:
        raise JudgmentError(f"후보 풀 판정 기록 거부: {message}")


def load_pool_admission_authorization(
    path: Path,
    *,
    candidate_run_id: str,
    candidate_config: str,
    pool_path: Path = POOL_PATH,
    folds_path: Path = FOLDS_PATH,
    registered_combiner_names: tuple[str, ...] | None = None,
) -> PoolAdmissionAuthorization:
    """지원하는 후보 풀 판정 기록을 검증해 장부 변경 권한으로 좁힌다.

    판정 기록 생성은 별도 절차의 소관이다.
    이 함수는 현재 장부와 판정 입력이 그대로일 때만 이미 끝난 nested OOF 판정을
    소비하도록 쓰기 경계를 닫는다.
    """
    _require_pool_record(
        not path.is_absolute() and ".." not in path.parts,
        "판정 기록은 저장소 상대 경로여야 한다.",
    )
    _require_pool_record(path.is_file(), f"파일이 없다: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise JudgmentError(f"후보 풀 판정 기록을 읽을 수 없다: {exc}") from exc
    _require_pool_record(isinstance(raw, dict), "최상위 값은 사전이어야 한다.")
    _require_pool_record(raw.get("schema_version") == 1, "schema_version은 1이어야 한다.")
    contract_version = raw.get("contract_version")
    _require_pool_record(
        contract_version in POOL_JUDGMENT_SUPPORTED_CONTRACT_VERSIONS,
        "지원하는 계약 판본이 아니다: "
        f"{', '.join(POOL_JUDGMENT_SUPPORTED_CONTRACT_VERSIONS)}",
    )
    judgment_id = raw.get("judgment_id")
    _require_pool_record(isinstance(judgment_id, str) and judgment_id, "judgment_id가 없다.")

    change = raw.get("change")
    _require_pool_record(isinstance(change, dict), "change가 없다.")
    action = change.get("action")
    _require_pool_record(
        action in {"admission", "replacement", "restoration"},
        "--admit이 소비할 수 있는 action이 아니다.",
    )
    candidate = change.get("candidate")
    _require_pool_record(isinstance(candidate, dict), "change.candidate가 없다.")
    _require_pool_record(
        candidate.get("run_id") == candidate_run_id,
        "후보 run_id가 명령 인자와 다르다.",
    )
    _require_pool_record(
        candidate.get("config") == candidate_config,
        "후보 config가 실행 기록과 다르다.",
    )
    lineage = candidate.get("model_lineage_group")
    _require_pool_record(
        isinstance(lineage, str) and lineage,
        "후보의 모델 계보 묶음이 없다.",
    )
    replaced_run_id = change.get("replaces_run_id")
    if action == "replacement":
        _require_pool_record(
            isinstance(replaced_run_id, str) and replaced_run_id,
            "replacement에는 replaces_run_id가 필요하다.",
        )
    else:
        _require_pool_record(
            replaced_run_id is None,
            f"{action}에는 replaces_run_id를 쓸 수 없다.",
        )

    selection = raw.get("selection")
    _require_pool_record(isinstance(selection, dict), "selection이 없다.")
    _require_pool_record(
        selection.get("kind") in {"precommitted_single", "nested_selection"},
        "selection.kind가 올바르지 않다.",
    )
    _require_pool_record(
        isinstance(selection.get("description"), str)
        and bool(selection["description"].strip()),
        "후보 선택 경위가 없다.",
    )
    if contract_version == POOL_JUDGMENT_CONTRACT_VERSION:
        combiner_scope = selection.get("combiner_scope")
        _require_pool_record(
            combiner_scope in {"core", "full"},
            "candidate-pool-v2 selection.combiner_scope가 올바르지 않다.",
        )
        contract_combiner_names = (
            CANDIDATE_POOL_CORE_COMBINER_NAMES
            if combiner_scope == "core"
            else DEFAULT_COMBINER_NAMES
        )
    else:
        combiner_scope = "full"
        contract_combiner_names = DEFAULT_COMBINER_NAMES

    frozen = raw.get("frozen_input")
    _require_pool_record(isinstance(frozen, dict), "frozen_input이 없다.")
    frozen_pool = frozen.get("candidate_pool")
    _require_pool_record(isinstance(frozen_pool, dict), "동결 후보 풀이 없다.")
    _require_pool_record(pool_path.is_file(), f"현재 후보 풀 파일이 없다: {pool_path}")
    _require_pool_record(
        frozen_pool.get("sha256") == _file_sha256(pool_path),
        "동결 후보 풀 해시가 현재 장부와 다르다.",
    )
    frozen_folds = frozen.get("folds")
    _require_pool_record(isinstance(frozen_folds, dict), "동결 folds가 없다.")
    _require_pool_record(folds_path.is_file(), f"현재 folds 파일이 없다: {folds_path}")
    _require_pool_record(
        frozen_folds.get("sha256") == _file_sha256(folds_path),
        "동결 folds 해시가 현재 파일과 다르다.",
    )
    combiners = frozen.get("registered_combiners")
    _require_pool_record(isinstance(combiners, dict), "동결 등록 결합 방식 집합이 없다.")
    if contract_version == POOL_JUDGMENT_CONTRACT_VERSION:
        _require_pool_record(
            combiners.get("scope") == combiner_scope,
            "동결 등록 결합 방식의 평가 범위가 selection과 다르다.",
        )
    current_combiner_names = (
        contract_combiner_names
        if registered_combiner_names is None
        else registered_combiner_names
    )
    _require_pool_record(
        current_combiner_names == contract_combiner_names,
        "계약 판본과 평가 범위에 맞는 등록 결합 방식 집합이 아니다.",
    )
    current_names = list(current_combiner_names)
    current_names_sha256 = canonical_name_list_sha256(current_combiner_names)
    _require_pool_record(
        combiners.get("names") == current_names,
        "등록 결합 방식의 이름이나 순서가 현재 등록부와 다르다.",
    )
    _require_pool_record(
        combiners.get("names_sha256") == current_names_sha256,
        "등록 결합 방식 이름 목록 해시가 다르다.",
    )

    comparison = raw.get("nested_oof_comparison")
    _require_pool_record(isinstance(comparison, dict), "nested OOF 대조가 없다.")
    before = comparison.get("before")
    after = comparison.get("after")
    _require_pool_record(
        isinstance(before, dict) and isinstance(after, dict),
        "nested OOF 대조의 before 또는 after가 없다.",
    )
    _require_pool_record(
        before.get("strategy") in current_names and after.get("strategy") in current_names,
        "대조의 최선 결합 방식이 현재 기본 등록부에 없다.",
    )
    try:
        before_auc = float(before["auc"])
        after_auc = float(after["auc"])
        nested_oof_delta = float(comparison["delta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgmentError("후보 풀 판정 기록 거부: nested OOF 수치가 올바르지 않다.") from exc
    _require_pool_record(
        all(math.isfinite(value) for value in (before_auc, after_auc, nested_oof_delta)),
        "nested OOF 수치는 유한해야 한다.",
    )
    _require_pool_record(
        math.isclose(
            nested_oof_delta,
            after_auc - before_auc,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "nested OOF 전체 차이가 before와 after의 차이와 맞지 않는다.",
    )
    _require_pool_record(nested_oof_delta > 0.0, "nested OOF 전체 차이가 양수가 아니다.")
    fold_delta = comparison.get("outer_fold_delta")
    _require_pool_record(isinstance(fold_delta, dict), "바깥쪽 분할별 차이가 없다.")
    try:
        normalized_fold_delta = {
            int(key): float(value) for key, value in fold_delta.items()
        }
    except (TypeError, ValueError) as exc:
        raise JudgmentError(
            "후보 풀 판정 기록 거부: 바깥쪽 분할별 차이가 올바르지 않다."
        ) from exc
    _require_pool_record(
        set(normalized_fold_delta) == set(range(5)),
        "바깥쪽 분할별 차이는 0부터 4까지 모두 있어야 한다.",
    )
    _require_pool_record(
        all(math.isfinite(value) for value in normalized_fold_delta.values()),
        "바깥쪽 분할별 차이는 유한해야 한다.",
    )
    fold_wins = sum(value > 0.0 for value in normalized_fold_delta.values())
    _require_pool_record(
        comparison.get("outer_fold_wins") == fold_wins,
        "바깥쪽 분할 승수가 분할별 차이와 맞지 않는다.",
    )
    expected_boundary = nested_oof_delta <= POOL_EQUIVALENCE_BAND_UPPER
    _require_pool_record(
        comparison.get("boundary_contribution") == expected_boundary,
        "경계 기여 표시가 성능 동등 대역과 맞지 않는다.",
    )

    evidence = raw.get("evidence")
    _require_pool_record(isinstance(evidence, dict), "근거 산출물이 없다.")
    evidence_value = evidence.get("path")
    _require_pool_record(
        isinstance(evidence_value, str) and evidence_value,
        "근거 산출물 경로가 없다.",
    )
    evidence_path = Path(evidence_value)
    _require_pool_record(
        not evidence_path.is_absolute() and ".." not in evidence_path.parts,
        "근거 산출물은 저장소 상대 경로여야 한다.",
    )
    _require_pool_record(evidence_path.is_file(), f"근거 산출물이 없다: {evidence_path}")
    evidence_sha256 = evidence.get("sha256")
    _require_pool_record(
        evidence_sha256 == _file_sha256(evidence_path),
        "근거 산출물 해시가 다르다.",
    )
    result = raw.get("result")
    _require_pool_record(
        isinstance(result, dict) and result.get("state") == "adopted",
        "최종 판정 상태가 adopted가 아니다.",
    )
    expected_decision = "replace" if action == "replacement" else "admit"
    _require_pool_record(
        result.get("decision") == expected_decision,
        "최종 판정과 변경 종류가 맞지 않는다.",
    )

    return PoolAdmissionAuthorization(
        judgment_id=judgment_id,
        contract_version=contract_version,
        action=action,
        candidate_run_id=candidate_run_id,
        candidate_config=candidate_config,
        replaced_run_id=replaced_run_id,
        nested_oof_delta=nested_oof_delta,
        boundary_contribution=expected_boundary,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        record_path=path,
        record_sha256=_file_sha256(path),
    )


WEIGHTED_OOF_AUC_METRIC = "auc_oof_weighted"
WEIGHTED_OOF_ESS_METRIC = "weighted_oof_effective_sample_size"
WEIGHTED_OOF_ESS_FRACTION_METRIC = "weighted_oof_effective_sample_fraction"
WEIGHTED_OOF_ZERO_WEIGHT_ROWS_METRIC = "weighted_oof_zero_weight_rows"
WEIGHTED_OOF_TEST_ONLY_PATTERNS_METRIC = "weighted_oof_test_only_patterns"


def _effective_sample_size(weight: np.ndarray) -> float:
    """Kish 유효 표본 수 (Σw)² / Σw². 가중이 고를수록 표본 수에 가깝다."""
    total = float(weight.sum())
    square_total = float(np.square(weight).sum())
    if square_total == 0.0:
        raise JudgmentError("가중치가 전부 0이라 유효 표본 수를 잴 수 없다.")
    return total * total / square_total


@dataclass(frozen=True)
class MissingnessReweighting:
    """train 행을 test의 결측 패턴 구성비로 옮기는 표본 가중치와 그 계보. (#383)

    행 키는 12개 설명변수의 결측 마스크다. 목표값과 id는 키에서 뺀다. 가중치는
    `w = P_test(패턴) / P_train(패턴)`이고, test에 없는 패턴의 train 행은 0이다.
    train에 없는 test 패턴은 실을 행이 없어 재채점에서 빠지므로 그 수를 남긴다.
    """

    weight: pd.Series  # id 인덱스, train 전 행. float64.
    train_pattern_count: int
    test_pattern_count: int
    test_only_pattern_count: int
    zero_weight_rows: int
    # 패턴 하나당 한 행의 가중치 원본(결측 열, train 행 수, 양쪽 구성비, 가중치).
    # 행 가중치 69만 개를 그대로 남기는 대신 이 표를 실행 기록에 붙인다.
    # 합성 가중치를 쓰는 테스트에서는 비어 있다.
    patterns: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def effective_sample_size(self) -> float:
        return _effective_sample_size(self.weight.to_numpy(dtype=np.float64))


def _missingness_pattern(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """결측 마스크를 id 인덱스의 정수 키로 접는다. 열 순서가 키의 의미를 정한다."""
    mask = frame[columns].isna().to_numpy()
    bits = np.left_shift(np.int64(1), np.arange(len(columns), dtype=np.int64))
    return pd.Series(
        (mask * bits).sum(axis=1),
        index=pd.Index(frame[ID], name=ID),
        dtype=np.int64,
    )


def missingness_reweighting(
    train_path: Path = TRAIN_PATH,
    test_path: Path = MISSINGNESS_TEST_PATH,
) -> MissingnessReweighting:
    """train/test 결측 패턴 구성비 차이를 표본 가중치로 만든다. (#383)

    구성비는 자료에서 읽기만 하고 목표값을 보지 않으므로 fold 안팎을 나눌 필요가 없다.
    """
    return _missingness_reweighting_cached(
        Path(train_path).resolve(),
        Path(test_path).resolve(),
    )


@lru_cache(maxsize=4)
def _missingness_reweighting_cached(
    train_path: Path,
    test_path: Path,
) -> MissingnessReweighting:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    columns = [column for column in train.columns if column not in {ID, TARGET}]
    if [column for column in test.columns if column not in {ID, TARGET}] != columns:
        raise JudgmentError("train과 test의 결측 패턴 대상 열이 다르다.")

    train_pattern = _missingness_pattern(train, columns)
    test_pattern = _missingness_pattern(test, columns)
    train_share = train_pattern.value_counts(normalize=True)
    test_share = test_pattern.value_counts(normalize=True)
    weight = (
        train_pattern.map(test_share).fillna(0.0) / train_pattern.map(train_share)
    ).astype(np.float64)
    return MissingnessReweighting(
        weight=weight.rename("weight"),
        train_pattern_count=int(len(train_share)),
        test_pattern_count=int(len(test_share)),
        test_only_pattern_count=int(len(set(test_share.index) - set(train_share.index))),
        zero_weight_rows=int((weight == 0.0).sum()),
        patterns=_pattern_table(train_pattern, train_share, test_share, columns),
    )


def _pattern_table(
    train_pattern: pd.Series,
    train_share: pd.Series,
    test_share: pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    """가중치 원본 표. train에 있는 패턴만 담는다(다른 패턴은 실을 행이 없다).

    빈 칸이 없는 패턴의 missing_columns는 "-"다. 빈 문자열은 CSV 왕복에서 결측이 되어
    "빈 칸 없음"과 "값 없음"이 구분되지 않는다.
    """
    keys = sorted(train_share.index)
    counts = train_pattern.value_counts()
    return pd.DataFrame(
        {
            "pattern": keys,
            "missing_columns": [
                ";".join(
                    column
                    for position, column in enumerate(columns)
                    if key >> position & 1
                )
                or "-"
                for key in keys
            ],
            "train_rows": [int(counts[key]) for key in keys],
            "train_share": [float(train_share[key]) for key in keys],
            "test_share": [float(test_share.get(key, 0.0)) for key in keys],
            "weight": [
                float(test_share.get(key, 0.0)) / float(train_share[key])
                for key in keys
            ],
        }
    )


@dataclass(frozen=True)
class WeightedOof:
    """test 결측 패턴 구성비로 재채점한 OOF AUC와 그 표본 계보. (#383)

    ADR 0001의 구성원 채택·교체 판정에는 쓰지 않는다. 최종 결합 전략(#337), 제출
    2장(#69), 결측 처리 방식 자체를 바꾸는 후보(#360)의 판정에만 쓰는 추가 눈금이다.
    """

    auc: float
    effective_sample_size: float
    effective_sample_fraction: float
    zero_weight_rows: int
    test_only_pattern_count: int

    def metrics(self) -> dict[str, float]:
        """실행 기록에 남길 metric 이름과 값. 단독 실행과 중첩 평가가 함께 쓴다."""
        return {
            WEIGHTED_OOF_AUC_METRIC: self.auc,
            WEIGHTED_OOF_ESS_METRIC: self.effective_sample_size,
            WEIGHTED_OOF_ESS_FRACTION_METRIC: self.effective_sample_fraction,
            WEIGHTED_OOF_ZERO_WEIGHT_ROWS_METRIC: float(self.zero_weight_rows),
            WEIGHTED_OOF_TEST_ONLY_PATTERNS_METRIC: float(self.test_only_pattern_count),
        }


def weighted_oof_auc(
    prediction: pd.Series,
    y: pd.Series,
    reweighting: MissingnessReweighting,
) -> WeightedOof:
    """OOF 예측을 test의 결측 패턴 구성비로 재채점한다. (#383)

    유효 표본 수와 0 가중 행 수는 재채점한 행에서 다시 세므로, 풀 부분집합이나
    일부 fold만 건네도 그 표본의 계보가 남는다. 가중치가 전부 1이면 값은 기존
    OOF AUC와 같다.
    """
    weight = reweighting.weight.reindex(prediction.index)
    if weight.isna().any():
        raise JudgmentError("가중 OOF 재채점에 결측 패턴 가중치가 없는 id가 들어왔다.")
    if not prediction.index.equals(y.index):
        raise JudgmentError("가중 OOF 재채점의 예측과 목표값 id 순서가 다르다.")
    weight_values = weight.to_numpy(dtype=np.float64)
    scored = weight_values > 0.0
    if len(set(y.to_numpy()[scored])) < 2:
        raise JudgmentError("가중치가 0이 아닌 행에 목표값 두 값이 다 있어야 한다.")
    effective = _effective_sample_size(weight_values)
    return WeightedOof(
        auc=float(
            roc_auc_score(
                y.to_numpy(),
                prediction.to_numpy(dtype=np.float64),
                sample_weight=weight_values,
            )
        ),
        effective_sample_size=effective,
        effective_sample_fraction=effective / len(weight_values),
        zero_weight_rows=int((~scored).sum()),
        test_only_pattern_count=reweighting.test_only_pattern_count,
    )


@dataclass(frozen=True)
class StrategyOutcome:
    """계열 3 판정의 평문 입력 한 건: 결합 전략 하나의 nested 평가 결과. (#104)

    check_adoption_eligibility가 기록 원형을 평문으로 받는 무늬 그대로, ensemble의
    평가 타입이 아닌 값(전략 이름, nested OOF AUC, outer fold별 AUC)만
    받는다.
    """

    name: str
    nested_auc: float
    fold_aucs: dict[int, float]  # outer fold별 AUC.


@dataclass(frozen=True)
class StrategyAssessment:
    """결합 전략 하나의 계열 3 판정 근거 값."""

    name: str
    nested_auc: float
    delta: float  # champion 대비.
    eligible: bool  # 채택 가능한가(delta >= AUC_THRESHOLD).
    fold_wins: int  # 전략 간 fold별 승리 수. 보조 증거로 기록만 한다. (ADR 0001)


@dataclass(frozen=True)
class EnsembleVerdict:
    """계열 3 판정의 근거 값."""

    champion_auc: float
    assessments: list[StrategyAssessment]  # nested OOF AUC 내림차순.
    recommended: str | None  # None이면 채택 없음, 단독 champion 유지.


def judge_ensemble(
    outcomes: list[StrategyOutcome], champion_auc: float
) -> EnsembleVerdict:
    """계열 3 판정: 채택 문턱을 넘은 nested OOF AUC 최고 전략 확정. (ADR 0001)"""
    if not outcomes:
        raise JudgmentError("판정할 결합 전략이 없다.")
    names = [outcome.name for outcome in outcomes]
    if len(set(names)) != len(names):
        raise JudgmentError(f"결합 전략 이름이 중복된다: {names}")
    folds = set(outcomes[0].fold_aucs)
    if any(set(outcome.fold_aucs) != folds for outcome in outcomes):
        raise JudgmentError(
            "전략 간 outer fold 구성이 다르다: 같은 fold 배정으로 평가한 결과만 비교한다."
        )

    # fold 승리: 그 fold에서 유일한 최고 AUC인 전략만 승리로 센다.
    wins = dict.fromkeys(names, 0)
    for fold in folds:
        best = max(outcome.fold_aucs[fold] for outcome in outcomes)
        winners = [o.name for o in outcomes if o.fold_aucs[fold] == best]
        if len(winners) == 1:
            wins[winners[0]] += 1

    assessments = sorted(
        (
            StrategyAssessment(
                name=outcome.name,
                nested_auc=outcome.nested_auc,
                delta=outcome.nested_auc - champion_auc,
                eligible=outcome.nested_auc - champion_auc >= AUC_THRESHOLD,
                fold_wins=wins[outcome.name],
            )
            for outcome in outcomes
        ),
        key=lambda a: (-a.nested_auc, a.name),
    )

    top = assessments[0]
    if not top.eligible:
        return EnsembleVerdict(champion_auc, assessments, None)
    return EnsembleVerdict(champion_auc, assessments, top.name)


@dataclass(frozen=True)
class AdoptionEligibility:
    """채택 자격 검사의 근거 값: 장부(champion·후보 풀)에 오르는 실행의 기록 조건."""

    seeds: list[int]
    seeds_ok: bool  # 3시드 평균본인가.
    git_dirty: bool
    folds_sha256: str
    committed_folds_sha256: str
    folds_ok: bool  # 커밋된 folds와 sha256이 일치하는가.
    ok: bool


def check_adoption_eligibility(
    *, seeds: list[int], git_dirty: bool, folds_sha256: str, committed_folds_sha256: str
) -> AdoptionEligibility:
    """채택 자격 검사: 3시드 평균본, git_dirty 아님, folds sha256 일치. (#14 관행)

    compare --adopt와 pool --admit이 공유한다. 인자를 기록 원형으로 받으므로 submit도
    태그만으로 같은 검사를 쓸 수 있지만, 대회 막판이라 전환하지 않았다(지도 #91).
    """
    seeds_ok = seeds == CONFIRM_SEEDS
    folds_ok = folds_sha256 == committed_folds_sha256
    return AdoptionEligibility(
        seeds=seeds,
        seeds_ok=seeds_ok,
        git_dirty=git_dirty,
        folds_sha256=folds_sha256,
        committed_folds_sha256=committed_folds_sha256,
        folds_ok=folds_ok,
        ok=seeds_ok and not git_dirty and folds_ok,
    )
