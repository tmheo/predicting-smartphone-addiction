"""판정 회차 파일럿: 이슈 513 재조립 판정을 JudgmentRound로 재현한다. (#553, 지도 #550)

기록된 docs/research/extended-stack-pool-reassembly/issue513 판정을 새 회차 id
judgment-round-pilot/issue553으로 재실행해 module을 실전 검증한다.
기준 팔 값(313 nested·분할별 AUC)과 재조립 314 구성(순서·OOF 해시)은 그 기록에서
동결 그대로 가져오고, 외부 구성원의 OOF 경로는 현재 두 번째 제출 manifest로 푼다.
자기 검사 등급은 원 회차와 같은 해시 동일성이다.

실행 순서:

    uv run python scripts/round_issue553_pilot.py precommit
    uv run python scripts/round_issue553_pilot.py run --workers 3 --threads 4
    uv run python scripts/round_issue553_pilot.py compare
    uv run python scripts/round_issue553_pilot.py report
    uv run python scripts/round_issue553_pilot.py publish
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import round as round_module
from pipeline.judgment import StackGate
from pipeline.members import HASH_VERIFIED, MemberSource, MemberSpec
from pipeline.round import (
    SELF_CHECK_HASH_IDENTITY,
    CandidateArm,
    ReferenceArm,
    ReferenceValues,
    ReportSection,
    RoundRecords,
    RoundSpec,
    SelfCheckSpec,
)

RECORD_DIR = Path("docs/research/extended-stack-pool-reassembly/issue513")
MANIFEST_PATH = Path("docs/research/extended-stack-submission-2-manifest.json")
RECORDED_FOLD_PATHS = {
    fold: RECORD_DIR / "reassembled" / f"fold-{fold}" / "reassembled.json"
    for fold in range(5)
}
RECORDED_PRECOMMIT = json.loads((RECORD_DIR / "precommit.json").read_text("utf-8"))
RECORDED_COMPARISON = json.loads((RECORD_DIR / "comparison.json").read_text("utf-8"))
ARM = "reassembled-314"


def _member_source() -> MemberSource:
    """이슈 513 precommit의 314 구성(순서·해시)을 그대로 선언한다. 시험 예측은 안 쓴다."""
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
    oof_path_of = {row["column"]: row.get("oof_path") for row in manifest["members"]}
    specs = []
    for row in RECORDED_PRECOMMIT["reassembled"]["members"]:
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
    return MemberSource(name=str(RECORD_DIR / "precommit.json"), members=tuple(specs))


def _replay_rows(records: RoundRecords) -> list[tuple[str, object, object]]:
    ours = records.comparison.payload["arms"][ARM]
    recorded = RECORDED_COMPARISON["reassembled"]
    rows: list[tuple[str, object, object]] = [
        ("구성 해시", RECORDED_PRECOMMIT["reassembled"]["composition_sha256"], ours["composition_sha256"]),
        ("nested AUC", recorded["nested_auc"], ours["nested_auc"]),
        ("nested 예측 해시", recorded["prediction_sha256"], ours["prediction_sha256"]),
        ("기준 대비 delta", RECORDED_COMPARISON["delta_vs_current_submission"], ours["verdict"]["delta"]),
        ("게이트 통과", RECORDED_COMPARISON["passes_gate"], ours["verdict"]["passes_gate"]),
    ]
    for fold in sorted(RECORDED_FOLD_PATHS):
        frozen = json.loads(RECORDED_FOLD_PATHS[fold].read_text("utf-8"))
        mine = records.folds[ARM][fold]
        rows.append((f"분할 {fold} AUC", frozen["auc"], mine["auc"]))
        rows.append((f"분할 {fold} 예측 해시", frozen["prediction_sha256"], mine["prediction_sha256"]))
    return rows


def _render_replay(records: RoundRecords) -> list[str]:
    rows = _replay_rows(records)
    lines = [
        "이슈 513에 기록된 판정 수치와 이번 재실행 결과를 항목별로 대조한다.",
        "",
        "| 항목 | 이슈 513 기록 | 이번 재현 | 일치 |",
        "| --- | --- | --- | :-: |",
    ]
    for label, frozen, mine in rows:
        lines.append(f"| {label} | `{frozen}` | `{mine}` | {'O' if frozen == mine else 'X'} |")
    verdict = "**전 항목 일치: 재현 성공**" if all(f == m for _, f, m in rows) else "**불일치 항목 있음: 재현 실패**"
    return [*lines, "", f"- {verdict} (항목 {len(rows)}개)."]


SPEC = RoundSpec(
    round_id="judgment-round-pilot/issue553",
    title="판정 회차 파일럿: 이슈 513 재조립 판정 재현",
    question=(
        "JudgmentRound로 재실행한 재조립 314 팔이 이슈 513에 기록된 판정과 "
        "같은 nested·분할별 AUC와 예측 해시를 내는가."
    ),
    contract="judgment-round-pilot",
    reference=ReferenceArm(
        name="current-submission-313",
        values=ReferenceValues(
            source=str(RECORD_DIR / "comparison.json"),
            nested_auc=RECORDED_COMPARISON["baseline"]["nested_auc"],
            fold_aucs={
                int(fold): auc
                for fold, auc in RECORDED_COMPARISON["baseline"]["fold_aucs"].items()
            },
        ),
    ),
    candidates=(CandidateArm(name=ARM, source=_member_source()),),
    combiner="c_selected_shrunk_rank_logit_logistic",
    gate=StackGate(),
    selfcheck=SelfCheckSpec(
        grade=SELF_CHECK_HASH_IDENTITY,
        source="이슈 513 원 회차와 같은 등급이며 기대 해시는 이슈 513 precommit에서 왔다.",
    ),
    sealed_inputs=(
        RECORD_DIR / "precommit.json",
        RECORD_DIR / "comparison.json",
        MANIFEST_PATH,
        *(RECORDED_FOLD_PATHS[fold] for fold in sorted(RECORDED_FOLD_PATHS)),
    ),
    report_sections=(ReportSection(title="이슈 513 기록 재현 대조", render=_render_replay),),
)

if __name__ == "__main__":
    round_module.main(SPEC)
