"""재현 구성원 중첩 결합 판정, 자체 36개 기준 회차. (#624, 지도 #619; 스펙 준비 #632)

제약 파생 사다리 재현 구성원 12개(동결 명세 rpf-v1)를 자체 36개 풀에 누적 사다리
3단계(raw4 4개, cats_te 8개, ratio_round 12개)로 더한 구성이 현행 등록 문턱을
넘는지 JudgmentRound로 판정한다. 기준 팔 값은 이슈 514 최종 확정의 pool36_full
nested OOF(shrunk_rank_logit_logistic, MLflow 223055f4)이며 동결 명세에 복사돼 있다.
자기 검사는 전 분할 재현(36 열이라 분할당 1분 안팎)이다.

실행 순서(저장소 루트, 커밋된 상태, data/·mlruns/ 있음):

    uv run python scripts/round_members_smoke.py scripts/round_issue624_own36.py --replay-fold 0
    uv run python scripts/round_issue624_own36.py precommit
    uv run python scripts/round_issue624_own36.py run --workers 3 --threads 4
    uv run python scripts/round_issue624_own36.py compare
    uv run python scripts/round_issue624_own36.py report
    uv run python scripts/round_issue624_own36.py publish
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import round as round_module
from pipeline.judgment import StackGate
from pipeline.member_sources import reproduction_pool_members
from pipeline.members import HASH_VERIFIED, MemberSource, MemberSpec
from pipeline.round import (
    SELF_CHECK_ALL_FOLDS_REPLAY,
    CandidateArm,
    ExpectedFold,
    ReferenceArm,
    ReferenceValues,
    ReportSection,
    RoundRecords,
    RoundSpec,
    SelfCheckSpec,
)

FREEZE_SPEC = Path("docs/research/reproduction-pool-freeze/rpf-v1-PLACEHOLDER.json")
POOL_PATH = Path("artifacts/pool.yaml")
FREEZE = json.loads(FREEZE_SPEC.read_text(encoding="utf-8"))
REFERENCE = FREEZE["reference_arms"]["own36"]
STAGES = [rung["stage"] for rung in FREEZE["ladder"]]
ARM_OF_STAGE = {stage: f"own36-{stage.replace('_', '-')}" for stage in STAGES}


def _reference_source() -> MemberSource:
    """동결 명세에 복사된 풀 36개(진입 순서, OOF 해시)를 그대로 선언한다."""
    specs = tuple(
        MemberSpec(
            member_id=row["config"],
            origin="own",
            verification=HASH_VERIFIED,
            run_id=row["run_id"],
            oof_sha256=row["oof_sha256"],
            expected_auc=row["auc"],
        )
        for row in REFERENCE["members"]
    )
    return MemberSource(
        name=f"{FREEZE_SPEC}#reference_arms/own36",
        members=specs,
        train_rows=FREEZE["inputs"]["train"]["rows"],
    )


def _candidate_source(stage: str) -> MemberSource:
    """기준 36개 뒤에 사다리 단계의 재현 구성원을 동결 순서로 잇는다."""
    reference = _reference_source()
    ladder = reproduction_pool_members(FREEZE_SPEC, stage=stage)
    return MemberSource(
        name=f"{reference.name}+ladder/{stage}",
        members=reference.members + ladder.members,
        train_rows=reference.train_rows,
    )


def _render_ladder(records: RoundRecords) -> list[str]:
    arms = records.comparison.payload["arms"]
    reference_auc = float(records.precommit.payload["reference"]["nested_auc"])
    lines = [
        "누적 사다리 단계별 nested AUC와 직전 단계 대비 증분. 제안 규칙은 동결 명세 judgment_rules를 따른다.",
        "",
        "| 팔 | 구성원 | nested AUC | 기준 대비 | 직전 단계 대비 | 분할 양수 | 판정 |",
        "| --- | ---: | ---: | ---: | ---: | :-: | :-: |",
    ]
    previous = reference_auc
    passing: list[tuple[float, int, str]] = []
    for stage in STAGES:
        arm = ARM_OF_STAGE[stage]
        body = arms[arm]
        verdict = body["verdict"]
        nested = float(body["nested_auc"])
        lines.append(
            f"| {arm} | {body['member_count']} | `{nested:.10f}` | `{verdict['delta']:+.7f}` "
            f"| `{nested - previous:+.7f}` | {verdict['folds_positive']}/{len(verdict['fold_deltas'])} "
            f"| {'통과' if verdict['passes_gate'] else '미달'} |"
        )
        previous = nested
        if verdict["passes_gate"]:
            passing.append((nested, int(body["member_count"]), arm))
    lines.append("")
    if passing:
        best = max(passing, key=lambda item: (item[0], -item[1]))
        lines.append(f"- 제안 구성: **{best[2]}** (통과 {len(passing)}개 가운데 nested AUC 최고, 동률이면 구성원이 적은 쪽).")
    else:
        lines.append("- 통과 구성 없음: 자체 36개 기준에서는 현재 풀 유지가 결론이다.")
    return lines


SPEC = RoundSpec(
    round_id="reproduction-pool-own36/issue624",
    title="재현 구성원 중첩 결합 판정: 자체 36개 기준",
    question=(
        "제약 파생 사다리 재현 구성원(raw4 4개, cats_te 누적 8개, ratio_round 누적 12개)을 "
        "자체 36개 풀에 더한 구성이 shrunk_rank_logit_logistic nested AUC에서 "
        "현행 등록 문턱(+0.00002, 바깥 분할 5/5 양수)을 넘는가."
    ),
    contract="reproduction-pool-judgment",
    reference=ReferenceArm(
        name=REFERENCE["name"],
        values=ReferenceValues(
            source=(
                f"{FREEZE_SPEC}#reference_arms/own36 "
                f"(MLflow {REFERENCE['values_source']['mlflow_run_id'][:8]} "
                f"{REFERENCE['values_source']['artifact']} proposal, 이슈 514 pool36_full)"
            ),
            nested_auc=REFERENCE["nested_auc"],
            fold_aucs={int(fold): auc for fold, auc in REFERENCE["fold_aucs"].items()},
        ),
        source=_reference_source(),
    ),
    candidates=tuple(
        CandidateArm(name=ARM_OF_STAGE[stage], source=_candidate_source(stage)) for stage in STAGES
    ),
    combiner=REFERENCE["combiner"],
    gate=StackGate(),
    selfcheck=SelfCheckSpec(
        grade=SELF_CHECK_ALL_FOLDS_REPLAY,
        source=(
            f"{FREEZE_SPEC}#reference_arms/own36/fold_aucs "
            "(MLflow 223055f4 근거 산출물의 분할별 AUC, 예측 해시는 기록에 없어 AUC만 대조)"
        ),
        expected={int(fold): ExpectedFold(auc=auc) for fold, auc in REFERENCE["fold_aucs"].items()},
    ),
    sealed_inputs=(FREEZE_SPEC, POOL_PATH),
    report_sections=(ReportSection(title="누적 사다리 단계별 요약", render=_render_ladder),),
)

if __name__ == "__main__":
    round_module.main(SPEC)
