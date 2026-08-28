"""판본 3 외부 구성원 장부 색인에서 외부 후보 동결 명세를 만든다. (#486, 판정 계약은 #491 ADR-0006)

입력은 `docs/research/external-member-ledger-v3/index.json`의
`eligible_current_records_in_order`다. 순서와 예측 쌍 SHA-256을 그대로 받아
변경 불가 명세(`외부 후보 동결 명세`, #482)를 만든다. 동결 전에 전체 OOF 성능이나
근접 중복을 보고 후보를 빼지 않는다(#481). 사용자가 동결 전에 별도 판단으로 뺀
후보는 `--exclude "구성원=사유"`로 명세의 `user_exclusions`에 사유와 함께 남긴다.

검사 항목(하나라도 어긋나면 명세를 만들지 않는다):

- 색인의 자격 있는 현행 기록 목록이 `current_records`의 `자격 있음`·`감사 완료`
  기록과 순서까지 같고, 대체된 기록을 참조하지 않는다.
- 감사 기록 파일의 `record_sha256`이 그 필드를 뺀 정규 JSON에서 다시 계산한 값과
  색인의 값에 모두 맞는다(제자리 수정 없음).
- 정규화 배열이 OOF 691,369행·시험 296,302행의 유한 float64이고, 배열 SHA-256과
  예측 쌍 SHA-256이 색인과 맞으며, 재채점 AUC가 기록값과 1e-9 안에서 맞는다.
- 후보 사이에 같은 예측 쌍 SHA-256(정확 중복)이 없다.

사용법:
    uv run python scripts/freeze_external_candidates.py --verify-only
    uv run python scripts/freeze_external_candidates.py \\
        --survey-cutoff 2026-08-30T10:00:00Z [--exclude "구성원=사유"]...

명세는 기본으로 `docs/research/external-candidate-freeze/<후보 집합 식별자>.json`에
쓰며, 같은 경로가 이미 있으면 덮어쓰지 않는다. 색인이 `rehearsal: true`면
예행 명세로 표시하고 `run-logs/` 아래에 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.data import ID, TARGET, TRAIN_PATH

SCHEMA = "external-candidate-freeze/1"
DEFAULT_INDEX = Path("docs/research/external-member-ledger-v3/index.json")
DEFAULT_OUT_DIR = Path("docs/research/external-candidate-freeze")
REHEARSAL_OUT_DIR = Path("run-logs/strict-external-selection/freeze")
SELECTION_ADR = Path("docs/adr/0006-strict-external-candidate-ladder.md")
AUDIT_CONTRACT_REF = "https://github.com/tmheo/predicting-smartphone-addiction/issues/482"
ELIGIBLE = "자격 있음"
AUDIT_DONE = "감사 완료"
N_TRAIN = 691369
N_TEST = 296302
AUC_TOLERANCE = 1e-9

SELECTION_POLICY = [
    "자격 있는 현행 외부 구성원 감사 기록을 색인 순서대로 모두 동결한다(전체 OOF 성능·근접 중복으로 미리 빼지 않는다).",
    "선별 단위는 검증된 개별 OOF·시험 예측 쌍이며 단독 AUC는 진단값이다.",
    "판정은 ADR-0006의 사전 고정 사다리(현재 313개 비교 팔 위에 정확 중복을 뺀 후보를 더한 전체·출처 절제·주의 사항 부류 절제 구성, 고정 shrunk_rank_logit_logistic, 313 대비 +0.00002·바깥 분할 5/5 문턱, 통과 중 nested 최고·잡음 바닥 안이면 구성원 적은 쪽)로만 한다.",
    "예측 쌍 SHA-256이 313 구성원과 같은 후보는 정확 중복으로 판정 도구가 자동 제외하고 precommit에 기록한다.",
    "공개 점수는 어느 단계에도 쓰지 않는다.",
    "이 명세는 변경 불가이며 후보가 늘거나 공개 판본이 바뀌면 새 명세를 만든다.",
]


class FreezeError(RuntimeError):
    """동결 명세를 만들 수 없는 색인·기록·배열 불일치."""


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def pair_sha256(oof: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(oof, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(test, dtype=np.float64).tobytes())
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def load_array(path: Path, rows: int, label: str) -> np.ndarray:
    _require(path.is_file(), f"{label}: 정규화 배열이 없다: {path}")
    values = np.load(path)
    _require(
        values.dtype == np.float64 and values.ndim == 1 and values.shape == (rows,),
        f"{label}: 정규화 배열 형태가 {values.dtype}{values.shape}이다(기대 float64 ({rows},))",
    )
    _require(bool(np.isfinite(values).all()), f"{label}: 비유한값이 있다")
    return values


def verify_index(index_path: Path, labels: np.ndarray | None) -> tuple[dict, list[dict]]:
    """색인·기록·배열을 대조하고 (색인, 검증된 후보 행 목록)을 돌려준다."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _require(index.get("ledger_version") == 3, f"장부 판본이 {index.get('ledger_version')}이다")
    records_dir = index_path.parent / "records"
    rows = index["current_records"]
    by_id = {row["audit_record_id"]: row for row in rows}
    _require(len(by_id) == len(rows), "현행 기록 식별자가 중복된다")
    superseded = set(index.get("superseded_record_ids", []))
    superseding = {
        row["supersedes_audit_record_id"]
        for row in rows
        if row.get("supersedes_audit_record_id")
    }
    eligible_rows = [row for row in rows if row["eligibility"] == ELIGIBLE]
    listed = index["eligible_current_records_in_order"]
    _require(
        [row["audit_record_id"] for row in listed]
        == [row["audit_record_id"] for row in eligible_rows],
        "eligible_current_records_in_order가 current_records의 자격 있음 기록 순서와 다르다",
    )
    verified: list[dict] = []
    seen_pairs: dict[str, str] = {}
    for order, (entry, row) in enumerate(zip(listed, eligible_rows, strict=True), start=1):
        member = row["member_id"]
        _require(entry["member_id"] == member, f"{member}: 목록과 현행 기록의 구성원이 다르다")
        _require(entry["pair_sha256"] == row["pair_sha256"], f"{member}: 목록과 현행 기록의 쌍 해시가 다르다")
        _require(row["audit_state"] == AUDIT_DONE, f"{member}: 감사 진행 상태가 {row['audit_state']}이다")
        _require(row["audit_record_id"] not in superseded, f"{member}: 대체된 기록이 목록에 있다")
        _require(row["audit_record_id"] not in superseding, f"{member}: 다른 현행 기록이 대체한 기록이 목록에 있다")
        _require(not row["exclusion_reason_codes"] and not row["insufficiency_reasons"], f"{member}: 자격 있음인데 제외·근거 부족 사유가 있다")
        record_path = records_dir / f"{row['audit_record_id']}.json"
        _require(record_path.is_file(), f"{member}: 감사 기록 파일이 없다: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        recomputed = text_sha256(
            canonical_json({k: v for k, v in record.items() if k != "record_sha256"})
        )
        _require(
            recomputed == record["record_sha256"] == row["record_sha256"],
            f"{member}: record_sha256 불일치(제자리 수정 의심)",
        )
        _require(record["identity"]["member_id"] == member, f"{member}: 기록의 구성원 식별자가 다르다")
        _require(record["audit"]["eligibility"] == ELIGIBLE, f"{member}: 기록의 자격 판정이 {record['audit']['eligibility']}이다")
        normalized = record["predictions"]["normalized"]
        _require(normalized is not None, f"{member}: 정규화 예측이 없다")
        for key in ("oof_path", "test_path", "oof_sha256", "test_sha256", "pair_sha256"):
            _require(normalized[key] == row[key] and row[key] == entry.get(key, row[key]), f"{member}: {key}가 기록·색인·목록에서 다르다")
        oof = load_array(Path(row["oof_path"]), N_TRAIN, member)
        test = load_array(Path(row["test_path"]), N_TEST, member)
        _require(array_sha256(oof) == row["oof_sha256"], f"{member}: OOF 배열 SHA-256 불일치")
        _require(array_sha256(test) == row["test_sha256"], f"{member}: 시험 배열 SHA-256 불일치")
        _require(pair_sha256(oof, test) == row["pair_sha256"], f"{member}: 예측 쌍 SHA-256 불일치")
        _require(row["pair_sha256"] not in seen_pairs, f"{member}: {seen_pairs.get(row['pair_sha256'])}와 예측 쌍이 정확히 같다")
        seen_pairs[row["pair_sha256"]] = member
        auc_delta = None
        if labels is not None and row.get("rescored_auc") is not None:
            auc = float(roc_auc_score(labels, oof))
            auc_delta = auc - float(row["rescored_auc"])
            _require(abs(auc_delta) <= AUC_TOLERANCE, f"{member}: 재채점 AUC가 기록과 {auc_delta:+.2e} 다르다")
        verified.append(
            {
                "order": order,
                "member_id": member,
                "display_name": row.get("display_name"),
                "audit_record_id": row["audit_record_id"],
                "audit_revision": row["audit_revision"],
                "supersedes_audit_record_id": row.get("supersedes_audit_record_id"),
                "record_sha256": row["record_sha256"],
                "evidence_manifest_sha256": row.get("evidence_manifest_sha256"),
                "kernel_ref": row.get("kernel_ref"),
                "script_version_id": row.get("script_version_id"),
                "pair_sha256": row["pair_sha256"],
                "oof_sha256": row["oof_sha256"],
                "test_sha256": row["test_sha256"],
                "oof_path": row["oof_path"],
                "test_path": row["test_path"],
                "caveat_codes": list(row.get("caveat_codes", [])),
                "rescored_auc": row.get("rescored_auc"),
                "rescored_auc_delta": auc_delta,
            }
        )
    return index, verified


def parse_exclusions(items: list[str]) -> dict[str, str]:
    exclusions: dict[str, str] = {}
    for item in items:
        _require("=" in item, f"--exclude는 '구성원=사유' 꼴이어야 한다: {item}")
        member, reason = item.split("=", 1)
        _require(bool(reason.strip()), f"{member}: 제외 사유가 비어 있다")
        _require(member not in exclusions, f"{member}: 제외가 중복된다")
        exclusions[member.strip()] = reason.strip()
    return exclusions


def build_spec(
    index_path: Path,
    index: dict,
    verified: list[dict],
    survey_cutoff: str,
    exclusions: dict[str, str],
) -> dict:
    rehearsal = bool(index.get("rehearsal", False))
    unknown = set(exclusions) - {row["member_id"] for row in verified}
    _require(not unknown, f"제외 대상이 자격 있는 후보에 없다: {sorted(unknown)}")
    candidates = []
    excluded = []
    for row in verified:
        if row["member_id"] in exclusions:
            excluded.append(
                {
                    "member_id": row["member_id"],
                    "audit_record_id": row["audit_record_id"],
                    "pair_sha256": row["pair_sha256"],
                    "reason": exclusions[row["member_id"]],
                }
            )
            continue
        candidates.append(row)
    for order, row in enumerate(candidates, start=1):
        row["ledger_order"] = row.pop("order")
        row["order"] = order
    spec: dict = {
        "schema": SCHEMA,
        "rehearsal": rehearsal,
        "ledger_version": index["ledger_version"],
        "contract_version": index["contract_version"],
        "contract_ref": index.get("contract_ref", AUDIT_CONTRACT_REF),
        "selection_contract": {
            "path": str(SELECTION_ADR),
            "sha256": file_sha256(SELECTION_ADR) if SELECTION_ADR.is_file() else None,
        },
        "index": {
            "path": str(index_path),
            "sha256": file_sha256(index_path),
            "generated_at": index.get("generated_at"),
            "tool": index.get("tool"),
            "eligible_count": len(verified),
        },
        "fold_spec": index["fold_spec"],
        "row_contract": index["row_contract"],
        "survey_cutoff": survey_cutoff,
        "selection_policy": SELECTION_POLICY,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "user_exclusions": excluded,
    }
    spec["content_sha256"] = text_sha256(canonical_json(spec))
    prefix = "ecf-rehearsal" if rehearsal else f"ecf-v{index['ledger_version']}"
    spec["candidate_set_id"] = f"{prefix}-{spec['content_sha256'][:12]}"
    spec["frozen_at"] = now_iso()
    spec["spec_sha256"] = text_sha256(
        canonical_json({k: v for k, v in spec.items() if k != "spec_sha256"})
    )
    return spec


def verify_spec_file(path: Path) -> dict:
    """명세 파일의 자체 SHA-256을 확인하고 읽는다. 판정 도구가 같은 검사를 쓴다."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    _require(spec.get("schema") == SCHEMA, f"{path}: 명세 schema가 {spec.get('schema')}이다")
    expected = text_sha256(canonical_json({k: v for k, v in spec.items() if k != "spec_sha256"}))
    _require(spec["spec_sha256"] == expected, f"{path}: spec_sha256 불일치(제자리 수정 의심)")
    content = {
        k: v
        for k, v in spec.items()
        if k not in ("spec_sha256", "candidate_set_id", "frozen_at", "content_sha256")
    }
    _require(spec["content_sha256"] == text_sha256(canonical_json(content)), f"{path}: content_sha256 불일치")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="외부 후보 동결 명세 생성기 (#486)")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--survey-cutoff", help="증분 조사 기준 시각(ISO 8601, UTC)")
    parser.add_argument("--out", type=Path, help="명세 파일 경로(기본: 출력 폴더/<후보 집합 식별자>.json)")
    parser.add_argument("--exclude", action="append", default=[], help="'구성원=사유' 꼴의 사용자 제외(반복 가능)")
    parser.add_argument("--verify-only", action="store_true", help="검사만 하고 명세를 쓰지 않는다")
    parser.add_argument("--skip-auc", action="store_true", help="재채점 AUC 대조를 건너뛴다(예행 색인용)")
    args = parser.parse_args()

    labels = None
    if not args.skip_auc:
        train = pd.read_csv(TRAIN_PATH, usecols=[ID, TARGET])
        labels = train[TARGET].to_numpy()
    try:
        index, verified = verify_index(args.index, labels)
        print(f"검사 통과: 자격 있는 현행 기록 {len(verified)}개, 색인 {args.index}")
        for row in verified:
            print(f"  {row['order']:>2} {row['member_id']:<64} {row['pair_sha256'][:12]}…")
        if args.verify_only:
            return
        if not args.survey_cutoff:
            parser.error("--survey-cutoff가 필요하다(검사만 하려면 --verify-only).")
        datetime.fromisoformat(args.survey_cutoff.replace("Z", "+00:00"))
        spec = build_spec(args.index, index, verified, args.survey_cutoff, parse_exclusions(args.exclude))
    except FreezeError as exc:
        sys.exit(f"동결 실패: {exc}")

    out_dir = REHEARSAL_OUT_DIR if spec["rehearsal"] else DEFAULT_OUT_DIR
    out = args.out or out_dir / f"{spec['candidate_set_id']}.json"
    if out.exists():
        sys.exit(f"동결 명세는 변경 불가다. 이미 있다: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    verify_spec_file(out)
    print(
        f"동결 명세 저장: {out}\n  후보 집합 식별자 {spec['candidate_set_id']}, 후보 {spec['candidate_count']}개, "
        f"사용자 제외 {len(spec['user_exclusions'])}개, spec_sha256 {spec['spec_sha256']}"
    )


if __name__ == "__main__":
    main()
