"""신원 출처 adapter 4종. (#531, #632)

파일 형식 해석만 하고 검증은 members module에 맡기는 얇은 층이다.
다음 대회에서 출처 형식이 바뀌면 adapter만 새로 쓴다.

1. manifest_members: 불변 manifest 출처(#457 스타일,
   docs/research/extended-stack-submission-2-manifest.json). 자체 구성원은
   run_id + RunStore로 OOF를 해석하고 시험 예측은 prediction_sha256 대조. hash-verified.
2. freeze_spec_members: 동결 명세 출처(ecf-v3 스타일,
   docs/research/external-candidate-freeze/ecf-v3-*.json). 4중 해시 완비. hash-verified.
3. pool_members: 풀 장부 출처(artifacts/pool.yaml). 해시가 없으므로 identity-only
   (labels를 주면 AUC 재채점으로 auc-verified까지 오른다). 비판정 용도 전용.
4. reproduction_pool_members: 재현 전용 풀 동결 명세 출처(rpf-v1 스타일,
   docs/research/reproduction-pool-freeze/rpf-v1-*.json). 대회 기록을 동결한 뒤
   재현 실험 구성원을 run_id + RunStore로 읽고 OOF 배열 해시를 대조한다.
   hash-verified. 명세의 누적 사다리(ladder) 단계로 부분집합을 고를 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

from .ledger import Pool
from .members import (
    HASH_VERIFIED,
    IDENTITY_ONLY,
    MemberSource,
    MemberSourceInvalid,
    MemberSpec,
)


def manifest_members(path: Path | str) -> MemberSource:
    """#457 스타일 불변 manifest의 members 목록을 동결 순서 그대로 읽는다."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for entry in payload["members"]:
        column = entry["column"]
        test = entry["test"]
        if entry["origin"] == "own":
            specs.append(
                MemberSpec(
                    member_id=column,
                    origin="own",
                    verification=HASH_VERIFIED,
                    run_id=entry["run_id"],
                    test_path=f"{test['test_path']}[{column}]",
                    test_sha256=test["prediction_sha256"],
                )
            )
        else:
            specs.append(
                MemberSpec(
                    member_id=column,
                    origin=entry["origin"],
                    verification=HASH_VERIFIED,
                    oof_path=entry["oof_path"],
                    test_path=test["test_path"],
                    test_sha256=test["prediction_sha256"],
                )
            )
    return MemberSource(name=str(path), members=tuple(specs))


def freeze_spec_members(path: Path | str) -> MemberSource:
    """ecf-v3 스타일 동결 명세의 후보를 동결 순서(order 필드)대로 읽는다."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload["row_contract"]
    specs = []
    for position, candidate in enumerate(payload["candidates"], start=1):
        if candidate["order"] != position:
            raise MemberSourceInvalid(
                f"{path}: 동결 후보 순서가 연속적이지 않다"
                f"(자리 {position}, order {candidate['order']})."
            )
        specs.append(
            MemberSpec(
                member_id=candidate["member_id"],
                origin="candidate",
                verification=HASH_VERIFIED,
                oof_path=candidate["oof_path"],
                test_path=candidate["test_path"],
                oof_sha256=candidate["oof_sha256"],
                test_sha256=candidate["test_sha256"],
                pair_sha256=candidate["pair_sha256"],
                expected_auc=candidate["rescored_auc"],
            )
        )
    return MemberSource(
        name=str(path),
        members=tuple(specs),
        train_rows=contract["train_rows"],
        test_rows=contract["test_rows"],
    )


def pool_members(pool: Pool | None = None) -> MemberSource:
    """풀 장부의 (config, run_id) 신원을 진입 순서 그대로 읽는다. 비판정 용도 전용."""
    pool = Pool.load() if pool is None else pool
    specs = tuple(
        MemberSpec(
            member_id=member.config,
            origin="pool",
            verification=IDENTITY_ONLY,
            run_id=member.run_id,
            expected_auc=member.oof_auc,
        )
        for member in pool.members
    )
    return MemberSource(name="artifacts/pool.yaml", members=specs)


def reproduction_pool_members(path: Path | str, *, stage: str | None = None) -> MemberSource:
    """rpf-v1 스타일 재현 전용 풀 동결 명세의 구성원을 동결 순서(order 필드)대로 읽는다.

    stage를 주면 명세의 누적 사다리 `ladder[stage]`에 든 구성원만 그 순서대로 남긴다.
    OOF는 run_id + RunStore로 해석하고 시험 예측은 선언하지 않는다(판정 전용 출처).
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload["inputs"]["train"]
    specs = []
    for position, member in enumerate(payload["members"], start=1):
        if member["order"] != position:
            raise MemberSourceInvalid(
                f"{path}: 재현 구성원 순서가 연속적이지 않다"
                f"(자리 {position}, order {member['order']})."
            )
        specs.append(
            MemberSpec(
                member_id=member["config"],
                origin="reproduction",
                verification=HASH_VERIFIED,
                run_id=member["run_id"],
                oof_sha256=member["oof"]["array_sha256"],
                expected_auc=member["oof"]["auc"],
            )
        )
    name = str(path)
    if stage is not None:
        ladder = {rung["stage"]: rung["members"] for rung in payload["ladder"]}
        if stage not in ladder:
            raise MemberSourceInvalid(
                f"{path}: 사다리 단계 {stage!r}가 없다(있는 단계 {sorted(ladder)})."
            )
        by_id = {spec.member_id: spec for spec in specs}
        unknown = [config for config in ladder[stage] if config not in by_id]
        if unknown:
            raise MemberSourceInvalid(f"{path}: 사다리 단계 {stage}의 구성원이 명세에 없다: {unknown}")
        specs = [by_id[config] for config in ladder[stage]]
        name = f"{path}#ladder/{stage}"
    return MemberSource(name=name, members=tuple(specs), train_rows=contract["rows"])
