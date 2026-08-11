"""후보 풀 장부(artifacts/pool.yaml)와 다양성 구성원 진입 판정. (ADR 0001 계열 2, #70)

사용법:
    uv run python -m pipeline.pool <run_id>                         # 진입 판정 리포트
    uv run python -m pipeline.pool <run_id> --admit --reason "..."  # 통과 시 장부 등록

판정 규칙(ADR 0001 계열 2):
- 진입 하한: 시드 평균본 OOF AUC가 진입 시점 champion − 0.01 이상.
  하한은 진입 시점에만 적용하고 champion 갱신 때마다 재심사하지 않는다.
- 스크리닝은 단일 시드로 허용하되(리포트만), 등록(--admit)은 3시드 평균본만 가능하다.
  후보 풀에는 시드 평균본만 올린다.
- 중복 게이트: 풀 내 최근접 구성원과 OOF 예측의 스피어만 순위 상관이 0.998 이상이면
  중복으로 보고 성능이 높은 쪽만 유지한다. 상관은 중복 제거 전용이다.
- 기여 판정: 표준 평가 앙상블(풀 전체의 순위 평균)의 OOF AUC에서 해당 구성원을 제외했을 때
  AUC가 하락해야 유지된다. 풀이 비어 있으면 기여 판정은 묻지 않는다.
- 탈락(중복 교체 포함)은 장부에서 지우고 MLflow 태그(pool.dropped_*)와 티켓 코멘트로만 남긴다.
- git_dirty 실행과 커밋된 folds와 sha256이 다른 실행은 등록하지 않는다. (#14 관행)

풀 장부의 원본은 커밋되는 artifacts/pool.yaml이다. champion.yaml과 같은 이유로
"무엇이 풀에 있는가"라는 결정과 진입 근거를 git 이력에 남긴다. 일괄 재심사는
P3 풀 점검(#63)과 P4 앙상블 구성 단계의 소관이다. nested OOF 평가기는 P4에서 만든다.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .compare import CHAMPION_PATH, CONFIRM_SEEDS, TRACKING_URI
from .data import ID, TARGET, file_sha256

ENTRY_FLOOR_MARGIN = 0.01  # champion − 0.01이 진입 하한. (ADR 0001)
DUPLICATE_SPEARMAN = 0.998  # 이 이상이면 중복으로 본다. (ADR 0001)
POOL_PATH = Path("artifacts/pool.yaml")
FOLDS_PATH = Path("artifacts/folds.parquet")
TRAIN_PATH = Path("data/train.csv")


@dataclass(frozen=True)
class PoolCandidate:
    """진입 판정에 필요한 MLflow run의 단면."""

    run_id: str
    experiment: str
    auc_oof: float
    seeds: list[int]
    git_dirty: bool
    folds_sha256: str
    oof: pd.DataFrame  # columns: id, fold, pred


def _client():
    import mlflow

    return mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)


def load_candidate(run_id: str) -> PoolCandidate:
    client = _client()
    run = client.get_run(run_id)
    oof_path = client.download_artifacts(run_id, "oof.parquet")
    return PoolCandidate(
        run_id=run_id,
        experiment=run.data.params["experiment"],
        auc_oof=run.data.metrics["auc_oof"],
        seeds=[int(s) for s in run.data.params["seeds"].split(",")],
        git_dirty=run.data.tags["git_dirty"] == "True",
        folds_sha256=run.data.tags["sha256.folds"],
        oof=pd.read_parquet(oof_path),
    )


def load_pool() -> dict:
    if not POOL_PATH.exists():
        return {"members": []}
    with POOL_PATH.open() as f:
        return yaml.safe_load(f)


def save_pool(pool: dict) -> None:
    with POOL_PATH.open("w") as f:
        yaml.safe_dump(pool, f, allow_unicode=True, sort_keys=False)


def _member_pred(run_id: str) -> pd.Series:
    """구성원 run의 OOF 예측을 id 인덱스 Series로 돌려준다."""
    oof = pd.read_parquet(_client().download_artifacts(run_id, "oof.parquet"))
    return oof.set_index(ID)["pred"]


def _load_labels(index: pd.Index) -> pd.Series:
    """train 라벨을 OOF의 id 순서로 정렬해 돌려준다. id가 어긋나면 즉시 실패한다."""
    labels = pd.read_csv(TRAIN_PATH, usecols=[ID, TARGET]).set_index(ID)[TARGET]
    aligned = labels.reindex(index)
    assert aligned.notna().all(), "OOF의 id가 train과 일치하지 않는다."
    return aligned


def spearman(a: pd.Series, b: pd.Series) -> float:
    """OOF 예측 두 벌의 스피어만 순위 상관. 동순위는 평균 순위로 처리한다."""
    return float(np.corrcoef(a.rank().to_numpy(), b.rank().to_numpy())[0, 1])


def rank_ensemble_auc(preds: list[pd.Series], y: pd.Series) -> float:
    """표준 평가 앙상블: 구성원별 예측을 순위(백분위)로 바꿔 평균한 뒤 채점한다."""
    ranks = np.mean([p.rank(pct=True).to_numpy() for p in preds], axis=0)
    return float(roc_auc_score(y.to_numpy(), ranks))


@dataclass(frozen=True)
class EntryVerdict:
    admit: bool
    drop_run_id: str | None  # 중복 교체로 탈락시킬 기존 구성원.
    evidence: dict
    lines: list[str]


def judge_entry(pool: dict, candidate: PoolCandidate, champion: dict) -> EntryVerdict:
    lines: list[str] = []
    evidence: dict = {
        "champion_run_id": champion["run_id"],
        "champion_oof_auc": float(champion["oof_auc"]),
    }

    floor = champion["oof_auc"] - ENTRY_FLOOR_MARGIN
    floor_ok = candidate.auc_oof >= floor
    evidence["floor_margin"] = float(candidate.auc_oof - floor)
    lines.append(
        f"진입 하한: OOF AUC {candidate.auc_oof:.5f} vs champion − {ENTRY_FLOOR_MARGIN} = "
        f"{floor:.5f} → {'통과' if floor_ok else '미달'}"
    )

    members = pool["members"]
    if not members:
        evidence.update(nearest_run_id=None, nearest_spearman=None,
                        ensemble_auc_with=None, ensemble_auc_without=None, contribution=None)
        lines.append("풀이 비어 있다: 중복 게이트와 기여 판정은 묻지 않는다.")
        return EntryVerdict(floor_ok, None, evidence, lines)

    cand_pred = candidate.oof.set_index(ID)["pred"]
    y = _load_labels(cand_pred.index)
    member_preds = {m["run_id"]: _member_pred(m["run_id"]).reindex(cand_pred.index) for m in members}
    for run_id, pred in member_preds.items():
        assert pred.notna().all(), f"구성원 {run_id}의 OOF id가 후보와 일치하지 않는다."

    # 중복 게이트: 최근접(상관 최대) 구성원 하나와만 비교한다. (ADR 0001)
    corrs = {run_id: spearman(cand_pred, pred) for run_id, pred in member_preds.items()}
    nearest_id = max(corrs, key=corrs.get)
    nearest_rho = corrs[nearest_id]
    evidence.update(nearest_run_id=nearest_id, nearest_spearman=float(nearest_rho))
    duplicate = nearest_rho >= DUPLICATE_SPEARMAN
    drop_run_id = None
    if duplicate:
        nearest = next(m for m in members if m["run_id"] == nearest_id)
        if candidate.auc_oof > nearest["oof_auc"]:
            drop_run_id = nearest_id
            lines.append(
                f"중복 게이트: 최근접 {nearest_id} 스피어만 {nearest_rho:.5f} >= {DUPLICATE_SPEARMAN}, "
                f"후보가 더 높다({candidate.auc_oof:.5f} > {nearest['oof_auc']:.5f}) → 기존 구성원 교체"
            )
        else:
            lines.append(
                f"중복 게이트: 최근접 {nearest_id} 스피어만 {nearest_rho:.5f} >= {DUPLICATE_SPEARMAN}, "
                f"기존 구성원이 더 높다({nearest['oof_auc']:.5f} >= {candidate.auc_oof:.5f}) → 진입 탈락"
            )
            return EntryVerdict(False, None, evidence, lines)
    else:
        lines.append(
            f"중복 게이트: 최근접 {nearest_id} 스피어만 {nearest_rho:.5f} < {DUPLICATE_SPEARMAN} → 통과"
        )

    # 기여 판정: 교체로 빠질 구성원은 제외한 풀 기준으로 잰다.
    base_preds = [p for run_id, p in member_preds.items() if run_id != drop_run_id]
    if not base_preds:
        evidence.update(ensemble_auc_with=None, ensemble_auc_without=None, contribution=None)
        lines.append("교체 후 풀에 다른 구성원이 없다: 기여 판정은 묻지 않는다.")
        return EntryVerdict(floor_ok, drop_run_id, evidence, lines)

    auc_without = rank_ensemble_auc(base_preds, y)
    auc_with = rank_ensemble_auc(base_preds + [cand_pred], y)
    contribution = auc_with - auc_without
    contribution_ok = contribution > 0
    evidence.update(
        ensemble_auc_with=float(auc_with),
        ensemble_auc_without=float(auc_without),
        contribution=float(contribution),
    )
    lines.append(
        f"기여 판정: 표준 평가 앙상블 OOF AUC {auc_without:.5f} → 포함 시 {auc_with:.5f} "
        f"(기여 {contribution:+.5f}) → {'유지' if contribution_ok else '탈락 후보'}"
    )

    return EntryVerdict(floor_ok and contribution_ok, drop_run_id, evidence, lines)


def _drop_member(pool: dict, run_id: str, reason: str) -> None:
    """장부에서 지우고 MLflow 태그로만 탈락을 남긴다. (ADR 0001)"""
    pool["members"] = [m for m in pool["members"] if m["run_id"] != run_id]
    client = _client()
    client.set_tag(run_id, "pool.dropped_at", datetime.date.today().isoformat())
    client.set_tag(run_id, "pool.dropped_reason", reason)
    print(f"구성원 탈락: run {run_id} ({reason}) - 티켓 코멘트로도 남길 것.")


def main() -> None:
    parser = argparse.ArgumentParser(description="후보 풀 진입 판정 (ADR 0001 계열 2)")
    parser.add_argument("run_id", help="후보의 MLflow run_id")
    parser.add_argument("--admit", action="store_true", help="판정 통과 시 pool.yaml에 등록")
    parser.add_argument("--reason", help="--admit에 기록할 한 줄 진입 사유")
    args = parser.parse_args()

    if args.admit and not args.reason:
        sys.exit("--admit에는 --reason \"한 줄 사유\"가 필요하다.")
    if not CHAMPION_PATH.exists():
        sys.exit(f"{CHAMPION_PATH} 없음: 진입 하한의 기준 champion이 필요하다.")

    with CHAMPION_PATH.open() as f:
        champion = yaml.safe_load(f)
    pool = load_pool()
    candidate = load_candidate(args.run_id)

    if any(m["run_id"] == candidate.run_id for m in pool["members"]):
        sys.exit(f"run {candidate.run_id}는 이미 풀 구성원이다.")

    print(f"후보: {candidate.experiment} run {candidate.run_id} "
          f"(auc_oof {candidate.auc_oof:.5f}, 시드 {candidate.seeds})")
    verdict = judge_entry(pool, candidate, champion)
    for line in verdict.lines:
        print(line)
    print(f"판정: {'진입' if verdict.admit else '진입 아님'}")

    if not args.admit:
        if verdict.admit and candidate.seeds != CONFIRM_SEEDS:
            print(f"참고: 등록은 3시드 평균본({CONFIRM_SEEDS})만 가능하다. 재실행 후 --admit할 것.")
        return
    if not verdict.admit:
        sys.exit("등록 거부: 판정이 진입이 아니다.")
    if candidate.seeds != CONFIRM_SEEDS:
        sys.exit(f"등록 거부: 풀에는 3시드 평균본({CONFIRM_SEEDS})만 올린다. (이 run의 시드: {candidate.seeds})")
    if candidate.git_dirty:
        sys.exit("등록 거부: git_dirty 실행은 앙상블 후보에서 제외한다. (#14)")
    if candidate.folds_sha256 != file_sha256(FOLDS_PATH):
        sys.exit("등록 거부: 이 run의 folds sha256이 커밋된 artifacts/folds.parquet과 다르다.")

    if verdict.drop_run_id is not None:
        _drop_member(
            pool, verdict.drop_run_id,
            f"중복 교체: run {candidate.run_id}와 스피어만 {verdict.evidence['nearest_spearman']:.5f}",
        )

    pool["members"].append({
        "run_id": candidate.run_id,
        "config": candidate.experiment,
        # 판정이 +0.0001 단위 비교이므로 반올림 없이 전체 정밀도로 남긴다.
        "oof_auc": float(candidate.auc_oof),
        "seeds": ",".join(map(str, candidate.seeds)),
        "entered_at": datetime.date.today().isoformat(),
        "reason": args.reason,
        "evidence": verdict.evidence,
    })
    save_pool(pool)
    print(f"풀 등록: run {candidate.run_id} → {POOL_PATH}. 커밋할 것.")


if __name__ == "__main__":
    main()
