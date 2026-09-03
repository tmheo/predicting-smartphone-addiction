"""재현 구성원 중첩 결합 판정, 314 확장 기준 회차. (#624, 지도 #619; 스펙 준비 #632)

제약 파생 사다리 재현 구성원 12개(동결 명세 rpf-v1)를 314 확장 스택(이슈 513
재조립 구성, 자체 36 + 외부 278)에 누적 사다리 3단계로 더한 구성이 현행 등록
문턱을 넘는지 JudgmentRound로 판정한다. 기준 팔 값과 314 구성(순서·OOF 해시)은
이슈 513 판정 기록에서 동결 명세로 복사됐고, 외부 구성원의 OOF 경로는 두 번째
제출 manifest(#457)로 푼다. 자기 검사는 봉인 분할 0 재현(분할당 14분 안팎)이다.

실행 순서(저장소 루트, 커밋된 상태, data/·data/external/·mlruns/ 있음, 동시 3개 상한):

    uv run python scripts/round_members_smoke.py scripts/round_issue624_ext314.py
    uv run python scripts/round_issue624_ext314.py precommit
    uv run python scripts/round_issue624_ext314.py run --workers 3 --threads 4
    uv run python scripts/round_issue624_ext314.py compare
    uv run python scripts/round_issue624_ext314.py report
    uv run python scripts/round_issue624_ext314.py publish
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import round as round_module
from pipeline.judgment import StackGate
from pipeline.member_sources import reproduction_pool_members
from pipeline.members import HASH_VERIFIED, MemberSource, MemberSpec
from pipeline.round import (
    SELF_CHECK_SEALED_FOLD_REPLAY,
    CandidateArm,
    ExpectedFold,
    ReferenceArm,
    ReferenceValues,
    ReportSection,
    RoundRecords,
    RoundSpec,
    SelfCheckSpec,
)

FREEZE_SPEC = Path("docs/research/reproduction-pool-freeze/rpf-v1-6fa08f3da327.json")
FREEZE = json.loads(FREEZE_SPEC.read_text(encoding="utf-8"))
REFERENCE = FREEZE["reference_arms"]["ext314"]
PRECOMMIT_PATH = Path(REFERENCE["values_source"]["precommit"]["path"])
COMPARISON_PATH = Path(REFERENCE["values_source"]["comparison"]["path"])
MANIFEST_PATH = Path(REFERENCE["values_source"]["manifest"]["path"])
SEALED_FOLD = 0
SEALED_FOLD_PATH = Path(REFERENCE["sealed_folds"][str(SEALED_FOLD)]["path"])
STAGES = [rung["stage"] for rung in FREEZE["ladder"]]
ARM_OF_STAGE = {stage: f"ext314-{stage.replace('_', '-')}" for stage in STAGES}


def _reference_source() -> MemberSource:
    """이슈 513 precommit의 314 구성(순서·해시)을 그대로 선언한다. 시험 예측은 안 쓴다."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    oof_path_of = {row["column"]: row.get("oof_path") for row in manifest["members"]}
    recorded = json.loads(PRECOMMIT_PATH.read_text(encoding="utf-8"))["reassembled"]["members"]
    specs = []
    for row in recorded:
        own = row["origin"] == "own"
        specs.append(
            MemberSpec(
                member_id=row["column"],
                origin=row["origin"],
                verification=HASH_VERIFIED,
                run_id=row["run_id"] if own else None,
                oof_path=None if own else oof_path_of[row["column"]],
                oof_sha256=row["oof_sha256"],
            )
        )
    return MemberSource(
        name=f"{PRECOMMIT_PATH}#reassembled",
        members=tuple(specs),
        train_rows=FREEZE["inputs"]["train"]["rows"],
    )


def _candidate_source(stage: str) -> MemberSource:
    """314 구성 뒤에 사다리 단계의 재현 구성원을 동결 순서로 잇는다."""
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
        lines.append("- 통과 구성 없음: 314 확장 기준에서는 현재 확장 스택 유지가 결론이다.")
    return lines


SPEC = RoundSpec(
    round_id="reproduction-pool-ext314/issue624",
    title="재현 구성원 중첩 결합 판정: 314 확장 기준",
    question=(
        "제약 파생 사다리 재현 구성원(raw4 4개, cats_te 누적 8개, ratio_round 누적 12개)을 "
        "314 확장 스택에 더한 구성이 c_selected_shrunk_rank_logit_logistic nested AUC에서 "
        "현행 등록 문턱(+0.00002, 바깥 분할 5/5 양수)을 넘는가."
    ),
    contract="reproduction-pool-judgment",
    reference=ReferenceArm(
        name=REFERENCE["name"],
        values=ReferenceValues(
            source=f"{FREEZE_SPEC}#reference_arms/ext314 (이슈 513 {COMPARISON_PATH.name} reassembled)",
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
        grade=SELF_CHECK_SEALED_FOLD_REPLAY,
        source=f"{FREEZE_SPEC}#reference_arms/ext314/sealed_folds/{SEALED_FOLD} (이슈 513 {SEALED_FOLD_PATH})",
        expected={
            SEALED_FOLD: ExpectedFold(
                auc=REFERENCE["sealed_folds"][str(SEALED_FOLD)]["auc"],
                prediction_sha256=REFERENCE["sealed_folds"][str(SEALED_FOLD)]["prediction_sha256"],
            )
        },
    ),
    sealed_inputs=(FREEZE_SPEC, PRECOMMIT_PATH, COMPARISON_PATH, MANIFEST_PATH, SEALED_FOLD_PATH),
    report_sections=(ReportSection(title="누적 사다리 단계별 요약", render=_render_ladder),),
)

if __name__ == "__main__":
    round_module.main(SPEC)
