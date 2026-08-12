"""개선 판정. champion 또는 대리 기준 실행과 challenger를 비교한다. (#19, #70, #87)

사용법:
    uv run python -m pipeline.compare <challenger_run_id>
    uv run python -m pipeline.compare <challenger_run_id> --adopt --reason "채택 사유"
    uv run python -m pipeline.compare <challenger_run_id> --proxy-baseline <baseline_run_id>

판정 규칙은 ADR 0001 계열 1(특성·단일 모델 challenger)의 2단계 판정이다.
challenger run의 시드가 어느 단계인지 결정한다: [42]면 스크리닝, [42, 43, 44]면 확정 재검증.

- 스크리닝(seed 42 단일): champion의 같은 시드(seed 42) OOF AUC(champion.yaml의
  seed_aucs[42]) 대비 개선이 0 이상이면 확정 재검증 자격을 얻는다.
  시드 평균본 OOF AUC를 기준선으로 쓰면 시드 평균 이득(약 +0.0003)이 문턱에 섞여
  같은 시드의 실재 개선을 걸러내므로 짝지은 비교여야 한다. (#74 개정)
  통과는 채택이 아니므로 --adopt는 항상 거부된다.
- 확정 재검증(3시드 평균본):
  - 시드 평균본 OOF AUC가 champion 대비 +0.0001 이상.
  - 3시드 중 2시드 이상에서 같은 시드의 champion 대비 시드별 OOF AUC 개선이 0보다 크다.
  - 개선 폭이 +0.0001 이상 +0.0002 미만인 경계 구간이면 시드 평균 fold 점수 5개 중
    3개 이상 승리를 추가로 요구한다. 그 외 구간에서 fold 승리 수는 보조 증거로 기록만 한다.
- 새 피처는 fold별 gain importance 평균이 플라시보 평균보다 높아야 한다.
  확정 재검증의 게이트이며, 스크리닝에서는 참고로만 출력한다(미달이면 3시드 재실행 낭비 경고).
- 플라시보 파생 카나리아(placebo_noise_te 등)의 중요도가 플라시보 원본보다 높으면
  누수로 보고 그 run은 어느 단계에서도 판정에 쓰지 않는다. (#33)
- 단일 시드 개선 폭이 +0.0003을 넘으면 그대로 채택하던 SINGLE_SEED_MARGIN 규칙은 폐기됐다.
  champion은 항상 3시드 평균본이라는 불변식을 지킨다. (ADR 0001)

champion의 원본은 커밋되는 artifacts/champion.yaml이다. mlflow.db는 로컬 전용이므로
"무엇이 champion인가"라는 결정은 git 이력에 남긴다. 시드별·fold별 AUC도 함께 기록해
확정 재검증이 mlflow.db 없이도 판정 가능하게 한다. --adopt가 이 파일을 고쳐 쓰고
사용자는 커밋만 한다. 파일이 없으면 --adopt는 첫 champion 부트스트랩으로 동작한다.

대리 스크리닝은 느린 champion 모델 계열에서 특성을 판정하기 전에 빠른 모델 계열의
동일 조건 기준 실행과 짝지어 후보를 거르는 선별 절차다. 공식 스크리닝이나 확정
재검증을 대신하지 않는다. 기준 실행과 challenger 모두 seed 42의 동일한 모델 설정과
입력 자료를 사용해야 하며, challenger에는 새 특성만 추가돼야 한다. OOF AUC가 악화되지
않고 모든 새 특성의 importance가 플라시보보다 높을 때만 공식 스크리닝으로 넘긴다.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .features import PLACEBO

AUC_THRESHOLD = 0.0001  # 확정 문턱. 이 미만의 개선은 CV 잡음으로 본다. (#15, ADR 0001)
BOUNDARY_UPPER = 0.0002  # 이 미만의 개선 폭은 경계 구간으로 fold 승리 게이트를 추가한다.
SCREENING_SEEDS = [42]  # 스크리닝 시드. 고정. (ADR 0001)
CONFIRM_SEEDS = [42, 43, 44]  # 확정 재검증 시드. 고정. (ADR 0001)
SEED_WIN_MIN = 2  # 3시드 중 시드별 개선이 필요한 최소 시드 수.
FOLD_WIN_MIN = 3  # 경계 구간에서 5개 fold 중 필요한 최소 승리 수.
CHAMPION_PATH = Path("artifacts/champion.yaml")
TRACKING_URI = "sqlite:///mlflow.db"


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


def load_run_facts(run_id: str) -> RunFacts:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(run_id)
    importance_path = client.download_artifacts(run_id, "feature_importance.parquet")
    config_artifacts = [
        item.path for item in client.list_artifacts(run_id) if item.path.endswith((".yaml", ".yml"))
    ]
    if len(config_artifacts) != 1:
        sys.exit(
            f"run {run_id}의 루트에서 설정 YAML 하나를 찾지 못했다: {config_artifacts}"
        )
    config_path = client.download_artifacts(run_id, config_artifacts[0])
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)
    model_params = {"model.kind": str(config["model"]["kind"])}
    model_params.update(
        {f"model.params.{key}": str(value) for key, value in config["model"]["params"].items()}
    )
    model_params.update(
        {f"model.fit.{key}": str(value) for key, value in config["model"]["fit"].items()}
    )
    metrics = run.data.metrics
    return RunFacts(
        run_id=run_id,
        experiment=run.data.params["experiment"],
        auc_oof=metrics["auc_oof"],
        features=set(run.data.params["features"].split(",")),
        seeds=[int(s) for s in run.data.params["seeds"].split(",")],
        seed_aucs={
            int(k.rsplit("_", 1)[1]): v
            for k, v in metrics.items()
            if k.startswith("auc_oof_seed_")
        },
        fold_aucs={
            int(k.rsplit("_", 1)[1]): v
            for k, v in metrics.items()
            if k.startswith("auc_fold_")
        },
        git_commit=run.data.tags["git_commit"],
        importance=pd.read_parquet(importance_path),
        model_params=model_params,
        input_hashes={
            key: value for key, value in run.data.tags.items() if key.startswith("sha256.")
        },
    )


def _is_canary(feature: str) -> bool:
    """플라시보에서 파생된 카나리아 피처인지. 예: placebo_noise_te. (#33)"""
    return feature.startswith(f"{PLACEBO}_")


def check_canaries(challenger: RunFacts, lines: list[str]) -> bool:
    """placebo 카나리아: 파생 피처가 플라시보 원본보다 중요해지면 누수다. (#33)"""
    mean_gain = challenger.importance.groupby("feature")["gain"].mean()
    ok_all = True
    for canary in sorted(f for f in challenger.features if _is_canary(f)):
        placebo_gain = mean_gain.get(PLACEBO)
        gain = mean_gain.get(canary, 0.0)
        ok = placebo_gain is not None and gain < placebo_gain
        ok_all &= ok
        shown = f"{placebo_gain:.1f}" if placebo_gain is not None else "기록 없음"
        lines.append(
            f"카나리아 {canary}: 평균 gain {gain:.1f} vs 플라시보 {shown} "
            f"→ {'통과' if ok else '누수 의심(판정 불가)'}"
        )
    return ok_all


def check_new_features(champion: dict, challenger: RunFacts, lines: list[str]) -> bool:
    """champion에 없던 새 피처의 평균 gain이 플라시보보다 높은지. 새 피처가 없으면 항상 참."""
    champion_features = set(champion["features"].split(","))
    new_features = sorted(
        f for f in challenger.features - champion_features - {PLACEBO} if not _is_canary(f)
    )
    if not new_features:
        lines.append("새 피처 없음: importance 조건은 묻지 않는다.")
        return True
    mean_gain = challenger.importance.groupby("feature")["gain"].mean()
    if PLACEBO not in mean_gain.index:
        lines.append("challenger run에 플라시보 피처가 없어 새 피처의 importance를 판정할 수 없다.")
        return False
    ok_all = True
    placebo_gain = mean_gain[PLACEBO]
    for feature in new_features:
        gain = mean_gain.get(feature)
        ok = gain is not None and gain > placebo_gain
        ok_all &= ok
        shown = f"{gain:.1f}" if gain is not None else "기록 없음"
        lines.append(
            f"새 피처 {feature}: 평균 gain {shown} vs 플라시보 {placebo_gain:.1f} "
            f"→ {'통과' if ok else '미달'}"
        )
    return ok_all


def judge_screening(champion: dict, challenger: RunFacts) -> tuple[bool, list[str]]:
    """스크리닝: 같은 시드끼리 짝지어 개선 >= 0이면 확정 재검증 자격. (ADR 0001, #74 개정)"""
    lines: list[str] = []
    seed = SCREENING_SEEDS[0]
    if "seed_aucs" not in champion or seed not in {int(k) for k in champion["seed_aucs"]}:
        sys.exit(
            f"champion.yaml에 seed_aucs[{seed}]가 없어 짝지은 스크리닝 비교를 할 수 없다. "
            "동일 설정·시드로 champion을 재실행해 시드별 지표를 백필할 것."
        )
    baseline = {int(k): v for k, v in champion["seed_aucs"].items()}[seed]
    delta = challenger.auc_oof - baseline
    auc_ok = delta >= 0.0
    lines.append(
        f"스크리닝(seed {seed} 짝지은 비교) OOF AUC: champion {baseline:.5f} → "
        f"challenger {challenger.auc_oof:.5f} (delta {delta:+.5f}, 문턱 개선 >= 0) "
        f"→ {'통과' if auc_ok else '미달'}"
    )
    canary_ok = check_canaries(challenger, lines)
    if not check_new_features(champion, challenger, lines):
        lines.append("참고: 새 피처 importance는 확정 재검증의 게이트다. 이대로면 3시드 재실행이 낭비된다.")
    passed = auc_ok and canary_ok
    if passed:
        lines.append(
            "확정 재검증 자격 획득: 설정의 cv.seeds를 [42, 43, 44]로 바꿔 재실행한 뒤 "
            "그 run으로 다시 판정할 것. 스크리닝 통과는 채택이 아니다."
        )
    return passed, lines


def judge_proxy_screening(
    baseline: RunFacts, challenger: RunFacts
) -> tuple[bool, list[str]]:
    """동일한 빠른 모델 계열 안에서 새 특성의 공식 스크리닝 진입 자격을 판정한다."""
    if baseline.seeds != SCREENING_SEEDS or challenger.seeds != SCREENING_SEEDS:
        sys.exit(
            f"대리 스크리닝은 기준 실행과 challenger 모두 시드가 {SCREENING_SEEDS}여야 한다. "
            f"(기준 실행: {baseline.seeds}, challenger: {challenger.seeds})"
        )
    if not baseline.features < challenger.features:
        sys.exit(
            "대리 스크리닝 challenger는 기준 실행의 모든 특성을 유지하고 새 특성을 "
            "하나 이상 추가해야 한다."
        )
    if baseline.model_params != challenger.model_params:
        sys.exit(
            "대리 스크리닝의 모델 설정이 기준 실행과 다르다. "
            f"기준 실행={baseline.model_params}, challenger={challenger.model_params}"
        )
    if baseline.input_hashes != challenger.input_hashes:
        sys.exit(
            "대리 스크리닝의 입력 자료 해시가 기준 실행과 다르다. "
            f"기준 실행={baseline.input_hashes}, challenger={challenger.input_hashes}"
        )

    lines: list[str] = []
    delta = challenger.auc_oof - baseline.auc_oof
    auc_ok = delta >= 0.0
    lines.append(
        f"대리 스크리닝(seed 42 짝지은 비교) OOF AUC: 기준 실행 {baseline.auc_oof:.5f} → "
        f"challenger {challenger.auc_oof:.5f} (delta {delta:+.5f}, 문턱 개선 >= 0) "
        f"→ {'통과' if auc_ok else '미달'}"
    )
    canary_ok = check_canaries(challenger, lines)
    importance_ok = check_new_features(
        {"features": ",".join(sorted(baseline.features))}, challenger, lines
    )
    passed = auc_ok and canary_ok and importance_ok
    if passed:
        lines.append(
            "공식 스크리닝 진입 자격 획득: champion 모델 계열의 seed 42 설정으로 "
            "같은 특성을 실행한 뒤 공식 개선 판정을 수행할 것. 대리 스크리닝 통과는 채택이 아니다."
        )
    return passed, lines


def judge_confirmation(champion: dict, challenger: RunFacts) -> tuple[bool, list[str]]:
    """확정 재검증: 시드 평균본 문턱 + 2/3 시드 개선 + 경계 구간 fold 승리 게이트. (ADR 0001)"""
    lines: list[str] = []
    delta = challenger.auc_oof - champion["oof_auc"]
    auc_ok = delta >= AUC_THRESHOLD
    lines.append(
        f"시드 평균본 OOF AUC: champion {champion['oof_auc']:.5f} → "
        f"challenger {challenger.auc_oof:.5f} (delta {delta:+.5f}, 문턱 +{AUC_THRESHOLD}) "
        f"→ {'통과' if auc_ok else '미달'}"
    )

    champion_seed_aucs = {int(k): v for k, v in champion["seed_aucs"].items()}
    seed_wins = 0
    for seed in CONFIRM_SEEDS:
        seed_delta = challenger.seed_aucs[seed] - champion_seed_aucs[seed]
        win = seed_delta > 0
        seed_wins += win
        lines.append(
            f"시드 {seed}: champion {champion_seed_aucs[seed]:.5f} → "
            f"challenger {challenger.seed_aucs[seed]:.5f} (delta {seed_delta:+.5f}) "
            f"→ {'개선' if win else '악화'}"
        )
    seed_ok = seed_wins >= SEED_WIN_MIN
    lines.append(f"시드별 개선 {seed_wins}/{len(CONFIRM_SEEDS)} (최소 {SEED_WIN_MIN}) → {'통과' if seed_ok else '미달'}")

    champion_fold_aucs = {int(k): v for k, v in champion["fold_aucs"].items()}
    fold_wins = sum(
        challenger.fold_aucs[f] > champion_fold_aucs[f] for f in sorted(champion_fold_aucs)
    )
    boundary = AUC_THRESHOLD <= delta < BOUNDARY_UPPER
    if boundary:
        fold_ok = fold_wins >= FOLD_WIN_MIN
        lines.append(
            f"경계 구간(+{AUC_THRESHOLD} 이상 +{BOUNDARY_UPPER} 미만): fold 승리 "
            f"{fold_wins}/{len(champion_fold_aucs)} (최소 {FOLD_WIN_MIN}) → {'통과' if fold_ok else '미달'}"
        )
    else:
        fold_ok = True
        lines.append(f"보조 증거: fold 승리 {fold_wins}/{len(champion_fold_aucs)} (경계 구간이 아니라 게이트 아님)")

    canary_ok = check_canaries(challenger, lines)
    importance_ok = check_new_features(champion, challenger, lines)
    return auc_ok and seed_ok and fold_ok and canary_ok and importance_ok, lines


def write_champion(challenger: RunFacts, reason: str) -> None:
    record = {
        "run_id": challenger.run_id,
        # 판정이 +0.0001 단위 비교이므로 반올림 없이 전체 정밀도로 남긴다.
        "oof_auc": float(challenger.auc_oof),
        # 확정 재검증의 시드별 비교와 경계 구간 fold 승리 게이트의 기준값. (ADR 0001)
        "seed_aucs": {s: float(challenger.seed_aucs[s]) for s in sorted(challenger.seed_aucs)},
        "fold_aucs": {f: float(challenger.fold_aucs[f]) for f in sorted(challenger.fold_aucs)},
        "config": challenger.experiment,
        "features": ",".join(sorted(challenger.features)),
        "git_commit": challenger.git_commit,
        "adopted_at": datetime.date.today().isoformat(),
        "reason": reason,
    }
    with CHAMPION_PATH.open("w") as f:
        yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)


def _require_confirmation_facts(champion: dict, challenger: RunFacts) -> None:
    """확정 재검증에 필요한 시드별·fold별 기준값이 양쪽에 있는지 검증한다."""
    missing = [s for s in CONFIRM_SEEDS if s not in challenger.seed_aucs]
    if missing:
        names = ", ".join(f"auc_oof_seed_{s}" for s in missing)
        sys.exit(
            f"challenger run에 시드별 OOF AUC 지표({names})가 없다. "
            "판정 계약(#70) 이전 실행이므로 갱신된 파이프라인으로 재실행할 것."
        )
    if "seed_aucs" not in champion or "fold_aucs" not in champion:
        sys.exit(
            "champion.yaml에 seed_aucs/fold_aucs가 없다. 판정 계약(#70) 이전 champion이므로 "
            "동일 설정·시드로 champion을 재실행해 시드별 지표를 백필한 뒤 판정할 것."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="champion 대 challenger 개선 판정 (ADR 0001 계열 1)")
    parser.add_argument("run_id", help="challenger의 MLflow run_id")
    parser.add_argument(
        "--proxy-baseline",
        help="대리 스크리닝에 사용할 동일 모델·입력·seed 42 기준 실행의 MLflow run_id",
    )
    parser.add_argument("--adopt", action="store_true", help="확정 재검증 통과 시 champion.yaml을 갱신")
    parser.add_argument("--reason", help="--adopt에 기록할 한 줄 채택 사유")
    args = parser.parse_args()

    if args.adopt and not args.reason:
        sys.exit("--adopt에는 --reason \"한 줄 사유\"가 필요하다.")
    if args.proxy_baseline and (args.adopt or args.reason):
        sys.exit("대리 스크리닝은 champion 채택이 아니므로 --adopt와 --reason을 쓸 수 없다.")

    challenger = load_run_facts(args.run_id)

    if args.proxy_baseline:
        baseline = load_run_facts(args.proxy_baseline)
        print(f"대리 기준 실행: {baseline.experiment} run {baseline.run_id}")
        print(f"challenger   : {challenger.experiment} run {challenger.run_id}")
        passed, lines = judge_proxy_screening(baseline, challenger)
        for line in lines:
            print(line)
        print(f"판정: {'대리 스크리닝 통과' if passed else '대리 스크리닝 미달'}")
        return

    if not CHAMPION_PATH.exists():
        print(f"{CHAMPION_PATH} 없음: 첫 champion 부트스트랩 모드.")
        if challenger.seeds != CONFIRM_SEEDS:
            sys.exit(f"champion은 항상 3시드 평균본이다: cv.seeds를 {CONFIRM_SEEDS}로 재실행할 것.")
        if args.adopt:
            write_champion(challenger, args.reason)
            print(f"champion 기록: run {challenger.run_id} (auc_oof {challenger.auc_oof:.5f}). 커밋할 것.")
        else:
            print("--adopt --reason \"...\"으로 이 run을 첫 champion으로 기록한다.")
        return

    with CHAMPION_PATH.open() as f:
        champion = yaml.safe_load(f)
    if champion["run_id"] == challenger.run_id:
        sys.exit("challenger가 현재 champion과 같은 run이다.")

    print(f"champion  : {champion['config']} run {champion['run_id']}")
    print(f"challenger: {challenger.experiment} run {challenger.run_id}")

    if challenger.seeds == SCREENING_SEEDS:
        passed, lines = judge_screening(champion, challenger)
        for line in lines:
            print(line)
        print(f"판정: {'스크리닝 통과' if passed else '스크리닝 미달'}")
        if args.adopt:
            sys.exit("채택 거부: 스크리닝 통과는 채택이 아니다. 3시드 확정 재검증 run으로 --adopt할 것.")
        return

    if challenger.seeds != CONFIRM_SEEDS:
        sys.exit(
            f"시드 고정 위반: 스크리닝은 {SCREENING_SEEDS}, 확정 재검증은 {CONFIRM_SEEDS}만 판정한다. "
            f"(이 run의 시드: {challenger.seeds})"
        )

    _require_confirmation_facts(champion, challenger)
    improved, lines = judge_confirmation(champion, challenger)
    for line in lines:
        print(line)
    print(f"판정: {'개선(확정)' if improved else '개선 아님'}")

    if not args.adopt:
        return
    if not improved:
        sys.exit("채택 거부: 확정 재검증이 개선이 아니다.")
    write_champion(challenger, args.reason)
    print(f"champion 갱신: run {challenger.run_id} (auc_oof {challenger.auc_oof:.5f}). 커밋할 것.")


if __name__ == "__main__":
    main()
