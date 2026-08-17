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
- 플라시보 파생 카나리아(placebo_noise_te 등)의 중요도가 플라시보 원본보다 높으면
  누수로 보고 그 run은 어느 단계에서도 판정에 쓰지 않는다. (#33)
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
- 동률 그룹: 1위 고정 기준으로, 1위와의 차이가 0.00002 미만인 채택 가능 전략만
  포함한다(연쇄 확장 없음).
- 확정: 동률 그룹 안에서 복잡도 서열이 가장 낮은 1개를 추천 전략으로 확정한다.
  복잡도 서열은 ADR 0001의 "구성원 수와 선택 자유도가 적은 더 단순한 방식"을
  각 결합 전략 adapter의 선언으로 기계화한 것이다.
  채택 가능 전략이 없으면 "채택 없음, 단독 champion 유지"다.
- 전략 간 fold별 승리 수는 보조 증거로 기록만 한다. (ADR 0001)

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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .ensemble import rank_mean
from .features import PLACEBO
from .ledger import Champion, Pool
from .runs import RunStore

AUC_THRESHOLD = 0.00002  # 계열 1·3 공통 채택 문턱. (#15, #64, ADR 0001)
BOUNDARY_UPPER = 0.0002  # 이 미만의 개선 폭은 경계 구간으로 fold 승리 게이트를 추가한다.
SCREENING_SEEDS = [42]  # 스크리닝 시드. 고정. (ADR 0001)
CONFIRM_SEEDS = [42, 43, 44]  # 확정 재검증 시드. 고정. (ADR 0001)
SEED_WIN_MIN = 2  # 3시드 중 시드별 개선이 필요한 최소 시드 수.
FOLD_WIN_MIN = 3  # 경계 구간에서 5개 fold 중 필요한 최소 승리 수.
ENTRY_FLOOR_MARGIN = 0.01  # champion − 0.01이 풀 진입 하한. (ADR 0001)
DUPLICATE_SPEARMAN = 0.998  # 이 이상이면 중복으로 본다. (ADR 0001)
FOLDS_PATH = Path("artifacts/folds.parquet")


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

    above=True면 기준값 초과(새 피처의 기여 증거), False면 기준값 미만(카나리아 무해)이
    통과다.
    """
    ok = (
        placebo_gain is not None
        and gain is not None
        and (gain > placebo_gain if above else gain < placebo_gain)
    )
    return GainCheck(feature, gain, ok)


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
    return PoolCandidate(
        run_id=run_id,
        experiment=meta.params["experiment"],
        auc_oof=meta.metrics["auc_oof"],
        seeds=[int(s) for s in meta.params["seeds"].split(",")],
        git_dirty=meta.tags["git_dirty"] == "True",
        folds_sha256=meta.tags["sha256.folds"],
        oof=store.oof_of(run_id),
    )


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
class StrategyOutcome:
    """계열 3 판정의 평문 입력 한 건: 결합 전략 하나의 nested 평가 결과. (#104)

    check_adoption_eligibility가 기록 원형을 평문으로 받는 무늬 그대로, ensemble의
    평가 타입이 아닌 값(전략 이름, 복잡도 서열, nested OOF AUC, outer fold별 AUC)만
    받는다.
    """

    name: str
    complexity: int  # 복잡도 서열(선택 자유도 순위). 낮을수록 단순하다.
    nested_auc: float
    fold_aucs: dict[int, float]  # outer fold별 AUC.


@dataclass(frozen=True)
class StrategyAssessment:
    """결합 전략 하나의 계열 3 판정 근거 값."""

    name: str
    complexity: int
    nested_auc: float
    delta: float  # champion 대비.
    eligible: bool  # 채택 가능한가(delta >= AUC_THRESHOLD).
    fold_wins: int  # 전략 간 fold별 승리 수. 보조 증거로 기록만 한다. (ADR 0001)


@dataclass(frozen=True)
class EnsembleVerdict:
    """계열 3 판정의 근거 값."""

    champion_auc: float
    assessments: list[StrategyAssessment]  # nested OOF AUC 내림차순.
    tie_group: list[str]  # 채택 가능 전략이 없으면 빈 목록.
    recommended: str | None  # None이면 채택 없음, 단독 champion 유지.


def judge_ensemble(
    outcomes: list[StrategyOutcome], champion_auc: float
) -> EnsembleVerdict:
    """계열 3 판정: 채택 문턱 + 동률 그룹 + 복잡도 서열 확정. (ADR 0001, #104)

    복잡도 서열 최저 선택은 ADR 0001의 "동률이면 구성원 수와 선택 자유도가 적은
    더 단순한 방식" 규정을 adapter 선언 값으로 기계 판정한 것이다.
    """
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
                complexity=outcome.complexity,
                nested_auc=outcome.nested_auc,
                delta=outcome.nested_auc - champion_auc,
                eligible=outcome.nested_auc - champion_auc >= AUC_THRESHOLD,
                fold_wins=wins[outcome.name],
            )
            for outcome in outcomes
        ),
        key=lambda a: (-a.nested_auc, a.complexity, a.name),
    )

    top = assessments[0]
    if not top.eligible:
        return EnsembleVerdict(champion_auc, assessments, [], None)
    tie_group = [
        a.name
        for a in assessments
        if a.eligible and top.nested_auc - a.nested_auc < AUC_THRESHOLD
    ]
    recommended = min(
        (a for a in assessments if a.name in tie_group),
        key=lambda a: (a.complexity, a.name),
    ).name
    return EnsembleVerdict(champion_auc, assessments, tie_group, recommended)


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
