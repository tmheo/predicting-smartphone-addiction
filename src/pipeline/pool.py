"""후보 풀 장부(artifacts/pool.yaml)와 다양성 구성원 진입 CLI. (ADR 0001 계열 2, #70)

사용법:
    uv run python -m pipeline.pool <run_id>                         # 진입 판정 리포트
    uv run python -m pipeline.pool <run_id> --admit --reason "..."  # 통과 시 장부 등록

판정 규칙(진입 하한·중복 게이트·기여 판정)과 채택 자격 검사는 judgment module
소관이다(ADR 0001 계열 2). 이 module은 EntryVerdict의 근거 값을 한국어 리포트로
그려내고, --admit에서 장부 기록과 탈락 태그 주석을 처리하는 caller다.

- 스크리닝은 단일 시드로 허용하되(리포트만), 등록(--admit)은 채택 자격 검사를
  통과한 3시드 평균본만 가능하다. 후보 풀에는 시드 평균본만 올린다.
- 탈락(중복 교체 포함)은 장부에서 지우고 MLflow 태그(pool.dropped_*)와 티켓
  코멘트로만 남긴다.

풀 장부의 원본은 커밋되는 artifacts/pool.yaml이다. champion.yaml과 같은 이유로
"무엇이 풀에 있는가"라는 결정과 진입 근거를 git 이력에 남긴다. 장부의 타입과
YAML 해석은 ledger module 소관이며(#96), 진입 근거는 진입 시점 스냅샷이라
champion이 교체되어도 갱신하지 않는다. 일괄 재심사는 P3 풀 점검(#63)과
P4 앙상블 구성 단계의 소관이다. nested OOF 평가는 ensemble module 소관이다(#104).
"""

from __future__ import annotations

import argparse
import datetime
import sys

from .data import file_sha256, labels
from .judgment import (
    CONFIRM_SEEDS,
    DUPLICATE_SPEARMAN,
    ENTRY_FLOOR_MARGIN,
    FOLDS_PATH,
    EntryVerdict,
    check_adoption_eligibility,
    judge_entry,
    load_candidate,
)
from .ledger import CHAMPION_PATH, POOL_PATH, Champion, EntryEvidence, Pool, PoolMember
from .runs import MlflowRunStore, RunStore, RunStoreError


def render_entry(verdict: EntryVerdict) -> list[str]:
    lines = [
        (
            f"진입 하한: OOF AUC {verdict.candidate_auc:.5f} vs champion − {ENTRY_FLOOR_MARGIN} = "
            f"{verdict.floor:.5f} → {'통과' if verdict.floor_ok else '미달'}"
        )
    ]
    if verdict.duplicate is None:
        lines.append("풀이 비어 있다: 중복 게이트와 기여 판정은 묻지 않는다.")
        return lines

    dup = verdict.duplicate
    if dup.replace:
        lines.append(
            f"중복 게이트: 최근접 {dup.nearest_run_id} 스피어만 {dup.nearest_spearman:.5f} >= {DUPLICATE_SPEARMAN}, "
            f"후보가 더 높다({verdict.candidate_auc:.5f} > {dup.nearest_auc:.5f}) → 기존 구성원 교체"
        )
    elif dup.duplicate:
        lines.append(
            f"중복 게이트: 최근접 {dup.nearest_run_id} 스피어만 {dup.nearest_spearman:.5f} >= {DUPLICATE_SPEARMAN}, "
            f"기존 구성원이 더 높다({dup.nearest_auc:.5f} >= {verdict.candidate_auc:.5f}) → 진입 탈락"
        )
        return lines
    else:
        lines.append(
            f"중복 게이트: 최근접 {dup.nearest_run_id} 스피어만 {dup.nearest_spearman:.5f} < {DUPLICATE_SPEARMAN} → 통과"
        )

    if verdict.contribution is None:
        lines.append("교체 후 풀에 다른 구성원이 없다: 기여 판정은 묻지 않는다.")
        return lines
    con = verdict.contribution
    lines.append(
        f"기여 판정: 표준 평가 앙상블 OOF AUC {con.auc_without:.5f} → 포함 시 {con.auc_with:.5f} "
        f"(기여 {con.contribution:+.5f}) → {'유지' if con.ok else '탈락 후보'}"
    )
    return lines


def entry_evidence(verdict: EntryVerdict) -> EntryEvidence:
    """진입 근거 조립: 판정의 근거 값(EntryVerdict)을 진입 시점 스냅샷 기록으로 옮긴다."""
    dup = verdict.duplicate
    con = verdict.contribution
    return EntryEvidence(
        champion_run_id=verdict.champion_run_id,
        champion_oof_auc=verdict.champion_auc,
        floor_margin=float(verdict.candidate_auc - verdict.floor),
        nearest_run_id=dup.nearest_run_id if dup else None,
        nearest_spearman=dup.nearest_spearman if dup else None,
        ensemble_auc_with=con.auc_with if con else None,
        ensemble_auc_without=con.auc_without if con else None,
        contribution=con.contribution if con else None,
    )


def _drop_member(pool: Pool, run_id: str, reason: str, store: RunStore) -> None:
    """장부에서 지우고 실행 저장소의 태그로만 탈락을 남긴다. (ADR 0001)"""
    pool.members = [m for m in pool.members if m.run_id != run_id]
    store.annotate(
        run_id,
        tags={
            "pool.dropped_at": datetime.datetime.now(datetime.UTC).date().isoformat(),
            "pool.dropped_reason": reason,
        },
    )
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

    champion = Champion.load()
    pool = Pool.load()
    store = MlflowRunStore()
    try:
        candidate = load_candidate(args.run_id, store)
    except RunStoreError as exc:
        sys.exit(str(exc))

    if any(m.run_id == candidate.run_id for m in pool.members):
        sys.exit(f"run {candidate.run_id}는 이미 풀 구성원이다.")

    print(f"후보: {candidate.experiment} run {candidate.run_id} "
          f"(auc_oof {candidate.auc_oof:.5f}, 시드 {candidate.seeds})")
    verdict = judge_entry(pool, candidate, champion, store, labels(candidate.oof.index))
    for line in render_entry(verdict):
        print(line)
    print(f"판정: {'진입' if verdict.admit else '진입 아님'}")

    if not args.admit:
        if verdict.admit and candidate.seeds != CONFIRM_SEEDS:
            print(f"참고: 등록은 3시드 평균본({CONFIRM_SEEDS})만 가능하다. 재실행 후 --admit할 것.")
        return
    if not verdict.admit:
        sys.exit("등록 거부: 판정이 진입이 아니다.")
    eligibility = check_adoption_eligibility(
        seeds=candidate.seeds,
        git_dirty=candidate.git_dirty,
        folds_sha256=candidate.folds_sha256,
        committed_folds_sha256=file_sha256(FOLDS_PATH),
    )
    if not eligibility.seeds_ok:
        sys.exit(f"등록 거부: 풀에는 3시드 평균본({CONFIRM_SEEDS})만 올린다. (이 run의 시드: {candidate.seeds})")
    if eligibility.git_dirty:
        sys.exit("등록 거부: git_dirty 실행은 앙상블 후보에서 제외한다. (#14)")
    if not eligibility.folds_ok:
        sys.exit("등록 거부: 이 run의 folds sha256이 커밋된 artifacts/folds.parquet과 다르다.")

    if verdict.drop_run_id is not None:
        _drop_member(
            pool, verdict.drop_run_id,
            f"중복 교체: run {candidate.run_id}와 스피어만 {verdict.duplicate.nearest_spearman:.5f}",
            store,
        )

    pool.members.append(PoolMember(
        run_id=candidate.run_id,
        config=candidate.experiment,
        oof_auc=candidate.auc_oof,
        seeds=candidate.seeds,
        entered_at=datetime.datetime.now(datetime.UTC).date().isoformat(),
        reason=args.reason,
        evidence=entry_evidence(verdict),
    ))
    pool.save()
    print(f"풀 등록: run {candidate.run_id} → {POOL_PATH}. 커밋할 것.")


if __name__ == "__main__":
    main()
