"""35개 후보 풀 재심사 사전 고정 장부의 내부 일관성을 검사한다."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml

from pipeline.ensemble import DEFAULT_COMBINER_NAMES, PRECISION_COMBINER_NAMES


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "artifacts/pool-rereview-precommit-2026-08-22.yaml"
HISTORICAL_SOURCE_SHA256 = {
    "pool": "e6f093c08af4d09a70e2ee9a7cc99f9d099b06b7505116005464b5ae1240712a",
    "full_refit_plan": "cb42b27f01abecdc51784e224d3346b27910d29b106171d8cdd471e1246b403f",
}


class PrecommitValidationError(RuntimeError):
    """실제 제거 대조 전에 고쳐야 하는 사전 고정 장부 오류."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecommitValidationError(message)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict),
        "사전 고정 장부의 최상위 값은 키-값 구조여야 한다.",
    )
    return payload


def validate(
    path: Path = DEFAULT_LEDGER, *, require_private: bool = False
) -> list[str]:
    """장부, 동결 기준과 현재 결합 전략 계약이 정확히 맞는지 검사한다."""
    ledger = _load(path)
    baseline_path = REPO_ROOT / ledger["sources"]["baseline_ledger"]["path"]
    baseline = _load(baseline_path)
    notes: list[str] = []

    _require(ledger["schema_version"] == 1, "지원하지 않는 schema_version이다.")
    _require(ledger["ticket_issue"] == 341, "ticket_issue는 341이어야 한다.")
    _require(ledger["map_issue"] == 338, "map_issue는 338이어야 한다.")

    missing_private: list[str] = []
    for name, source in ledger["sources"].items():
        source_path = REPO_ROOT / source["path"]
        if name in HISTORICAL_SOURCE_SHA256:
            actual = source["sha256"]
            _require(
                actual == HISTORICAL_SOURCE_SHA256[name],
                f"동결 출처 내용 해시 불일치: {source['path']} ({actual})",
            )
            continue
        if not source_path.is_file():
            if name in {"train", "test", "external_original"}:
                missing_private.append(name)
                continue
            raise PrecommitValidationError(f"동결 출처 파일이 없다: {source['path']}")
        actual = _sha256(source_path)
        _require(
            actual == source["sha256"],
            f"동결 출처 내용 해시 불일치: {source['path']} ({actual})",
        )
    if missing_private:
        _require(
            not require_private,
            "실행에 필요한 비공개 입력이 없다: " + ", ".join(missing_private),
        )
        notes.append(
            "현재 작업 폴더에 없는 비공개 입력은 실행 환경에서 검사한다: "
            + ", ".join(missing_private)
        )

    members = ledger["candidate_pool"]["members"]
    baseline_members = [member["config"] for member in baseline["members"]]
    _require(
        len(members) == ledger["candidate_pool"]["count"] == 35,
        "후보 수가 35가 아니다.",
    )
    _require(len(set(members)) == len(members), "후보 이름이 중복되었다.")
    _require(members == baseline_members, "후보와 순서가 동결 기준 장부와 다르다.")

    baseline_refits = {
        member["config"]: member["full_refit_count"] for member in baseline["members"]
    }
    default_refits = ledger["candidate_pool"]["full_refit_count"]["default"]
    overrides = ledger["candidate_pool"]["full_refit_count"]["overrides"]
    ledger_refits = {member: overrides.get(member, default_refits) for member in members}
    _require(
        ledger_refits == baseline_refits,
        "전체 자료 재학습 횟수가 동결 기준과 다르다.",
    )

    baseline_lineages = {
        name: entry
        for name, entry in baseline["lineage_groups"].items()
        if entry["size"] > 1
    }
    lineages = ledger["lineage_groups"]
    _require(
        set(lineages) == set(baseline_lineages),
        "다구성원 모델 계보 묶음 목록이 다르다.",
    )
    for name, entry in lineages.items():
        baseline_entry = baseline_lineages[name]
        _require(entry["size"] == len(entry["members"]), f"계보 묶음 크기 불일치: {name}")
        _require(
            entry["members"] == baseline_entry["members"],
            f"계보 묶음 구성원 불일치: {name}",
        )
        _require(
            set(entry["predecessor_depth"]) == set(entry["members"]),
            f"계보 깊이가 모든 구성원을 덮지 않는다: {name}",
        )

    perspectives = ledger["information_perspectives"]
    baseline_perspectives = baseline["information_perspectives"]
    _require(set(perspectives) == set(baseline_perspectives), "정보 관점 목록이 다르다.")
    for name, entry in perspectives.items():
        baseline_entry = baseline_perspectives[name]
        _require(entry["size"] == len(entry["members"]), f"정보 관점 크기 불일치: {name}")
        _require(
            entry["members"] == baseline_entry["members"],
            f"정보 관점 구성원 불일치: {name}",
        )
        _require(
            set(entry["members"]) <= set(members),
            f"후보 밖 정보 관점 구성원: {name}",
        )

    strategies = ledger["strategies"]
    _require(
        tuple(strategies["included"]) == DEFAULT_COMBINER_NAMES,
        "기본 결합 전략의 이름 또는 등록 순서가 현재 계약과 다르다.",
    )
    _require(
        tuple(strategies["excluded_precision"]) == PRECISION_COMBINER_NAMES,
        "제외한 정밀 결합 전략 목록이 현재 계약과 다르다.",
    )
    _require(
        strategies["required_success_count"] == len(DEFAULT_COMBINER_NAMES) == 19,
        "필수 성공 전략 수가 19가 아니다.",
    )

    null_band = ledger["null_band"]
    _require(
        null_band["clone_sources"] == "candidate_pool.members",
        "영점 복제 목록 참조가 다르다.",
    )
    _require(
        [scale["blocks"] * scale["contrasts_per_block"] for scale in null_band["scales"]]
        == [35, 175],
        "영점 대조 두 척도는 각각 35건과 175건이어야 한다.",
    )
    _require(
        ledger["randomness"]["bootstrap_replicates"] == 2000,
        "부트스트랩 반복 수는 2000이어야 한다.",
    )
    _require(ledger["randomness"]["bootstrap_seed"] == 342342, "BOOTSTRAP_SEED가 다르다.")

    stages = ledger["procedure"]["stages"]
    _require(
        [stage["stage"] for stage in stages] == [1, 2, 3, 4],
        "단계가 1부터 4까지가 아니다.",
    )
    _require(len(lineages) == 9, "1단계 모델 계보 묶음은 9개여야 한다.")
    _require(len(perspectives) == 14, "3단계 정보 관점 묶음은 14개여야 한다.")
    _require(
        ledger["procedure"]["split_order"]
        == ["outer-0", "outer-1", "outer-2", "outer-3", "outer-4", "final"],
        "분할 실행 순서가 다르다.",
    )

    decision_fields = set(ledger["outputs"]["decision"]["contrast_required"])
    required_decision_fields = {
        "removed",
        "working_members_before",
        "delta_vs_working",
        "verdict",
        "delta_vs_anchor",
        "before_strategy",
        "after_strategy",
        "strategy_changed",
        "fold_delta",
        "negative_folds",
    }
    _require(
        required_decision_fields <= decision_fields,
        "선행 판정 규칙이 요구한 대조 산출물 필드가 빠졌다.",
    )

    notes.append(
        "후보 35개, 모델 계보 묶음 9개와 정보 관점 14개가 동결 기준과 같다."
    )
    notes.append(
        "등록 결합 전략 19개와 제외 전략 3개의 이름과 순서가 현재 계약과 같다."
    )
    notes.append(
        "영점 대조 210건, 네 단계와 결정 산출물 계약이 완결되어 있다."
    )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--allow-missing-private",
        action="store_true",
        help="장부 구조만 검사할 때 비공개 입력 누락을 허용한다.",
    )
    args = parser.parse_args()
    for note in validate(
        args.path.resolve(), require_private=not args.allow_missing_private
    ):
        print(f"- {note}")


if __name__ == "__main__":
    main()
