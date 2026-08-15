"""개선 판정 CLI. champion 또는 대리 기준 실행과 challenger를 비교한다. (#19, #70, #87)

사용법:
    uv run python -m pipeline.compare <challenger_run_id>
    uv run python -m pipeline.compare <challenger_run_id> --adopt --reason "채택 사유"
    uv run python -m pipeline.compare <challenger_run_id> --proxy-baseline <baseline_run_id>

판정 규칙과 문턱은 judgment module 소관이다(ADR 0001 계열 1). 이 module은
challenger run의 시드로 단계를 결정하고([42]면 스크리닝, [42, 43, 44]면 확정
재검증, --proxy-baseline이 있으면 대리 스크리닝), Verdict의 근거 값을 한국어
리포트로 그려낸 뒤 --adopt를 처리하는 caller다.

champion의 원본은 커밋되는 artifacts/champion.yaml이다. mlflow.db는 로컬 전용이므로
"무엇이 champion인가"라는 결정은 git 이력에 남긴다. 시드별·fold별 AUC도 함께 기록해
확정 재검증이 mlflow.db 없이도 판정 가능하게 한다. 장부의 타입과 YAML 해석은
ledger module 소관이고(#96), --adopt가 채택 기록을 조립해 save를 부르면 사용자는
커밋만 한다. 파일이 없으면 --adopt는 첫 champion 부트스트랩으로 동작한다.
"""

from __future__ import annotations

import argparse
import datetime
import sys

from .data import file_sha256
from .judgment import (
    AUC_THRESHOLD,
    BOUNDARY_UPPER,
    CONFIRM_SEEDS,
    FOLD_WIN_MIN,
    FOLDS_PATH,
    SCREENING_SEEDS,
    SEED_WIN_MIN,
    CanaryReport,
    ConfirmationVerdict,
    JudgmentError,
    NewFeatureReport,
    ProxyScreeningVerdict,
    RunFacts,
    ScreeningVerdict,
    check_adoption_eligibility,
    judge_confirmation,
    judge_proxy_screening,
    judge_screening,
    load_run_facts,
)
from .ledger import CHAMPION_PATH, Champion
from .runs import MlflowRunStore, RunStoreError


def _canary_lines(report: CanaryReport) -> list[str]:
    shown = f"{report.placebo_gain:.1f}" if report.placebo_gain is not None else "기록 없음"
    return [
        f"카나리아 {check.feature}: 평균 gain {check.gain:.1f} vs 플라시보 {shown} "
        f"→ {'통과' if check.ok else '누수 의심(판정 불가)'}"
        for check in report.checks
    ]


def _new_feature_lines(report: NewFeatureReport) -> list[str]:
    if not report.new_features:
        return ["새 피처 없음: importance 조건은 묻지 않는다."]
    if report.placebo_gain is None:
        return ["challenger run에 플라시보 피처가 없어 새 피처의 importance를 판정할 수 없다."]
    lines = []
    for check in report.checks:
        shown = f"{check.gain:.1f}" if check.gain is not None else "기록 없음"
        lines.append(
            f"새 피처 {check.feature}: 평균 gain {shown} vs 플라시보 {report.placebo_gain:.1f} "
            f"→ {'통과' if check.ok else '미달'}"
        )
    return lines


def render_screening(verdict: ScreeningVerdict) -> list[str]:
    lines = [
        (
            f"스크리닝(seed {verdict.seed} 짝지은 비교) OOF AUC: champion {verdict.baseline_auc:.5f} → "
            f"challenger {verdict.challenger_auc:.5f} (delta {verdict.delta:+.5f}, 문턱 개선 >= 0) "
            f"→ {'통과' if verdict.auc_ok else '미달'}"
        )
    ]
    lines += _canary_lines(verdict.canary)
    lines += _new_feature_lines(verdict.new_features)
    if not verdict.new_features.ok:
        lines.append("참고: 새 피처 importance는 확정 재검증의 게이트다. 이대로면 3시드 재실행이 낭비된다.")
    if verdict.passed:
        lines.append(
            "확정 재검증 자격 획득: 같은 config를 --stage confirm으로 재실행한 뒤 "
            "그 run으로 다시 판정할 것. 스크리닝 통과는 채택이 아니다."
        )
    return lines


def render_proxy_screening(verdict: ProxyScreeningVerdict) -> list[str]:
    lines = [
        (
            f"대리 스크리닝(seed 42 짝지은 비교) OOF AUC: 기준 실행 {verdict.baseline_auc:.5f} → "
            f"challenger {verdict.challenger_auc:.5f} (delta {verdict.delta:+.5f}, 문턱 개선 >= 0) "
            f"→ {'통과' if verdict.auc_ok else '미달'}"
        )
    ]
    lines += _canary_lines(verdict.canary)
    lines += _new_feature_lines(verdict.new_features)
    if verdict.passed:
        lines.append(
            "공식 스크리닝 진입 자격 획득: champion 모델 계열의 seed 42 설정으로 "
            "같은 특성을 실행한 뒤 공식 개선 판정을 수행할 것. 대리 스크리닝 통과는 채택이 아니다."
        )
    return lines


def render_confirmation(verdict: ConfirmationVerdict) -> list[str]:
    lines = [
        (
            f"시드 평균본 OOF AUC: champion {verdict.champion_auc:.5f} → "
            f"challenger {verdict.challenger_auc:.5f} (delta {verdict.delta:+.5f}, 문턱 +{AUC_THRESHOLD}) "
            f"→ {'통과' if verdict.auc_ok else '미달'}"
        )
    ]
    for comparison in verdict.seed_comparisons:
        lines.append(
            f"시드 {comparison.seed}: champion {comparison.champion_auc:.5f} → "
            f"challenger {comparison.challenger_auc:.5f} (delta {comparison.delta:+.5f}) "
            f"→ {'개선' if comparison.win else '악화'}"
        )
    lines.append(
        f"시드별 개선 {verdict.seed_wins}/{len(verdict.seed_comparisons)} (최소 {SEED_WIN_MIN}) "
        f"→ {'통과' if verdict.seed_ok else '미달'}"
    )
    if verdict.boundary:
        lines.append(
            f"경계 구간(+{AUC_THRESHOLD} 이상 +{BOUNDARY_UPPER} 미만): fold 승리 "
            f"{verdict.fold_wins}/{verdict.fold_total} (최소 {FOLD_WIN_MIN}) → {'통과' if verdict.fold_ok else '미달'}"
        )
    else:
        lines.append(f"보조 증거: fold 승리 {verdict.fold_wins}/{verdict.fold_total} (경계 구간이 아니라 게이트 아님)")
    lines += _canary_lines(verdict.canary)
    lines += _new_feature_lines(verdict.new_features)
    return lines


def require_adoption_eligibility(challenger: RunFacts) -> None:
    """채택 자격 검사(#14 관행)의 미달을 종료 메시지로 번역한다. pool --admit과 규칙을 공유한다."""
    eligibility = check_adoption_eligibility(
        seeds=challenger.seeds,
        git_dirty=challenger.git_dirty,
        folds_sha256=challenger.folds_sha256,
        committed_folds_sha256=file_sha256(FOLDS_PATH),
    )
    if not eligibility.seeds_ok:
        sys.exit(f"채택 거부: champion은 항상 3시드 평균본이다. (이 run의 시드: {challenger.seeds})")
    if eligibility.git_dirty:
        sys.exit("채택 거부: git_dirty 실행은 champion으로 채택하지 않는다. 커밋 후 재실행할 것. (#14)")
    if not eligibility.folds_ok:
        sys.exit("채택 거부: 이 run의 folds sha256이 커밋된 artifacts/folds.parquet과 다르다.")


def write_champion(challenger: RunFacts, reason: str) -> None:
    Champion(
        run_id=challenger.run_id,
        oof_auc=challenger.auc_oof,
        seed_aucs=challenger.seed_aucs,
        fold_aucs=challenger.fold_aucs,
        config=challenger.experiment,
        features=challenger.features,
        git_commit=challenger.git_commit,
        adopted_at=datetime.datetime.now(datetime.UTC).date().isoformat(),
        reason=reason,
    ).save()


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

    store = MlflowRunStore()
    try:
        challenger = load_run_facts(args.run_id, store)
        baseline = load_run_facts(args.proxy_baseline, store) if args.proxy_baseline else None
    except RunStoreError as exc:
        sys.exit(str(exc))

    try:
        if baseline is not None:
            print(f"대리 기준 실행: {baseline.experiment} run {baseline.run_id}")
            print(f"challenger   : {challenger.experiment} run {challenger.run_id}")
            verdict = judge_proxy_screening(baseline, challenger)
            for line in render_proxy_screening(verdict):
                print(line)
            print(f"판정: {'대리 스크리닝 통과' if verdict.passed else '대리 스크리닝 미달'}")
            return

        if not CHAMPION_PATH.exists():
            print(f"{CHAMPION_PATH} 없음: 첫 champion 부트스트랩 모드.")
            if challenger.seeds != CONFIRM_SEEDS:
                sys.exit("champion은 항상 3시드 평균본이다: --stage confirm으로 재실행할 것.")
            if args.adopt:
                require_adoption_eligibility(challenger)
                write_champion(challenger, args.reason)
                print(f"champion 기록: run {challenger.run_id} (auc_oof {challenger.auc_oof:.5f}). 커밋할 것.")
            else:
                print("--adopt --reason \"...\"으로 이 run을 첫 champion으로 기록한다.")
            return

        champion = Champion.load()
        if champion.run_id == challenger.run_id:
            sys.exit("challenger가 현재 champion과 같은 run이다.")

        print(f"champion  : {champion.config} run {champion.run_id}")
        print(f"challenger: {challenger.experiment} run {challenger.run_id}")

        if challenger.seeds == SCREENING_SEEDS:
            verdict = judge_screening(champion, challenger)
            for line in render_screening(verdict):
                print(line)
            print(f"판정: {'스크리닝 통과' if verdict.passed else '스크리닝 미달'}")
            if args.adopt:
                sys.exit("채택 거부: 스크리닝 통과는 채택이 아니다. 3시드 확정 재검증 run으로 --adopt할 것.")
            return

        if challenger.seeds != CONFIRM_SEEDS:
            sys.exit(
                f"시드 고정 위반: 스크리닝은 {SCREENING_SEEDS}, 확정 재검증은 {CONFIRM_SEEDS}만 판정한다. "
                f"(이 run의 시드: {challenger.seeds})"
            )

        verdict = judge_confirmation(champion, challenger)
    except JudgmentError as exc:
        sys.exit(str(exc))

    for line in render_confirmation(verdict):
        print(line)
    print(f"판정: {'개선(확정)' if verdict.passed else '개선 아님'}")

    if not args.adopt:
        return
    if not verdict.passed:
        sys.exit("채택 거부: 확정 재검증이 개선이 아니다.")
    require_adoption_eligibility(challenger)
    write_champion(challenger, args.reason)
    print(f"champion 갱신: run {challenger.run_id} (auc_oof {challenger.auc_oof:.5f}). 커밋할 것.")


if __name__ == "__main__":
    main()
