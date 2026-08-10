"""개선 판정. champion(artifacts/champion.yaml)과 challenger(MLflow run)를 비교한다. (#19)

사용법:
    uv run python -m pipeline.compare <challenger_run_id>
    uv run python -m pipeline.compare <challenger_run_id> --adopt --reason "채택 사유"

판정 규칙 (#15, #19):
- OOF AUC 개선이 +0.0001 이상이어야 한다.
- champion에 없던 새 피처는 fold별 gain importance 평균이 플라시보 평균보다 높아야 한다.
  피처 변화가 없는 실험에는 이 조건을 묻지 않는다.
- 개선 폭이 +0.0003 이하인데 단일 시드면 채택을 거부한다(3시드 재실행 안내).

champion의 원본은 커밋되는 artifacts/champion.yaml이다. mlflow.db는 로컬 전용이므로
"무엇이 champion인가"라는 결정은 git 이력에 남긴다. --adopt가 이 파일을 고쳐 쓰고
사용자는 커밋만 한다. 파일이 없으면 --adopt는 첫 champion 부트스트랩으로 동작한다.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .features import PLACEBO

AUC_THRESHOLD = 0.0001  # 이 미만의 개선은 CV 잡음으로 본다. (#15)
SINGLE_SEED_MARGIN = 0.0003  # 이하의 미세 개선은 3시드 평균으로만 채택한다. (#15)
CHAMPION_PATH = Path("artifacts/champion.yaml")
TRACKING_URI = "sqlite:///mlflow.db"


@dataclass(frozen=True)
class RunFacts:
    """판정에 필요한 MLflow run의 단면."""

    run_id: str
    experiment: str
    auc_oof: float
    features: set[str]
    n_seeds: int
    git_commit: str
    importance: pd.DataFrame  # feature, fold, seed, gain


def load_run_facts(run_id: str) -> RunFacts:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(run_id)
    importance_path = client.download_artifacts(run_id, "feature_importance.parquet")
    return RunFacts(
        run_id=run_id,
        experiment=run.data.params["experiment"],
        auc_oof=run.data.metrics["auc_oof"],
        features=set(run.data.params["features"].split(",")),
        n_seeds=len(run.data.params["seeds"].split(",")),
        git_commit=run.data.tags["git_commit"],
        importance=pd.read_parquet(importance_path),
    )


def judge(champion: dict, challenger: RunFacts) -> tuple[bool, bool, list[str]]:
    """(개선 여부, 단일 시드 채택 거부 여부, 판정 로그)를 돌려준다."""
    lines: list[str] = []
    delta = challenger.auc_oof - champion["oof_auc"]
    auc_ok = delta >= AUC_THRESHOLD
    lines.append(
        f"OOF AUC: champion {champion['oof_auc']:.5f} → challenger {challenger.auc_oof:.5f} "
        f"(delta {delta:+.5f}, 문턱 +{AUC_THRESHOLD}) → {'통과' if auc_ok else '미달'}"
    )

    champion_features = set(champion["features"].split(","))
    new_features = sorted(challenger.features - champion_features - {PLACEBO})
    importance_ok = True
    if not new_features:
        lines.append("새 피처 없음: importance 조건은 묻지 않는다.")
    else:
        mean_gain = challenger.importance.groupby("feature")["gain"].mean()
        if PLACEBO not in mean_gain.index:
            importance_ok = False
            lines.append("challenger run에 플라시보 피처가 없어 새 피처의 importance를 판정할 수 없다.")
        else:
            placebo_gain = mean_gain[PLACEBO]
            for feature in new_features:
                gain = mean_gain.get(feature)
                ok = gain is not None and gain > placebo_gain
                importance_ok &= ok
                shown = f"{gain:.1f}" if gain is not None else "기록 없음"
                lines.append(
                    f"새 피처 {feature}: 평균 gain {shown} vs 플라시보 {placebo_gain:.1f} "
                    f"→ {'통과' if ok else '미달'}"
                )

    seed_blocked = auc_ok and delta <= SINGLE_SEED_MARGIN and challenger.n_seeds == 1
    if seed_blocked:
        lines.append(
            f"개선 폭 {delta:+.5f}이 +{SINGLE_SEED_MARGIN} 이하인데 단일 시드다: "
            "채택하려면 설정의 cv.seeds를 [42, 43, 44]로 바꿔 3시드 평균으로 재실행할 것."
        )

    return auc_ok and importance_ok, seed_blocked, lines


def write_champion(challenger: RunFacts, reason: str) -> None:
    record = {
        "run_id": challenger.run_id,
        "oof_auc": round(challenger.auc_oof, 5),
        "config": challenger.experiment,
        "features": ",".join(sorted(challenger.features)),
        "git_commit": challenger.git_commit,
        "adopted_at": datetime.date.today().isoformat(),
        "reason": reason,
    }
    with CHAMPION_PATH.open("w") as f:
        yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="champion 대 challenger 개선 판정")
    parser.add_argument("run_id", help="challenger의 MLflow run_id")
    parser.add_argument("--adopt", action="store_true", help="판정 통과 시 champion.yaml을 갱신")
    parser.add_argument("--reason", help="--adopt에 기록할 한 줄 채택 사유")
    args = parser.parse_args()

    if args.adopt and not args.reason:
        sys.exit("--adopt에는 --reason \"한 줄 사유\"가 필요하다.")

    challenger = load_run_facts(args.run_id)

    if not CHAMPION_PATH.exists():
        print(f"{CHAMPION_PATH} 없음: 첫 champion 부트스트랩 모드.")
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

    improved, seed_blocked, lines = judge(champion, challenger)
    print(f"champion  : {champion['config']} run {champion['run_id']}")
    print(f"challenger: {challenger.experiment} run {challenger.run_id}")
    for line in lines:
        print(line)
    print(f"판정: {'개선' if improved else '개선 아님'}")

    if not args.adopt:
        return
    if not improved:
        sys.exit("채택 거부: 판정이 개선이 아니다.")
    if seed_blocked:
        sys.exit("채택 거부: 미세 개선은 3시드 평균이 필요하다.")
    write_champion(challenger, args.reason)
    print(f"champion 갱신: run {challenger.run_id} (auc_oof {challenger.auc_oof:.5f}). 커밋할 것.")


if __name__ == "__main__":
    main()
