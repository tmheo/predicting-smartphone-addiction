"""이슈 366의 독립 분할 결과를 최종 판정 산출물 하나로 합친다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline.pool_rereview import load_inputs

from fast_single_removal import (
    FIXED_STRATEGY,
    SCREEN_STRATEGY,
    _member_metadata,
    _write_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        label: args.input_dir / f"screened-{label}.json"
        for label in ("0", "1", "2", "3", "4", "final")
    }
    results = {
        label: json.loads(path.read_text()) for label, path in paths.items()
    }
    input_hashes = {
        result["input"]["prediction_sha256"] for result in results.values()
    }
    if len(input_hashes) != 1:
        raise ValueError("독립 분할 결과의 입력 예측 해시가 서로 다르다.")
    for label in ("0", "1", "2", "3", "4"):
        if str(results[label]["execution"]["fold"]) != label:
            raise ValueError(f"분할 결과 이름과 내부 번호가 다르다: {label}")
        if "held_out" not in results[label]["split"]:
            raise ValueError(f"바깥쪽 채점 결과가 없다: {label}")
    final_split = results["final"]["split"]
    if "all_strategies" not in final_split:
        raise ValueError("전체 OOF의 19개 전략 재평가 결과가 없다.")

    context = load_inputs(Path(results["final"]["input"]["prediction_path"]))
    selected_members = set(final_split["selected_members"])
    removed_members = [
        member for member in context.members if member not in selected_members
    ]
    held_out = [results[str(fold)]["split"]["held_out"] for fold in range(5)]
    wins = sum(item["winner"] == "selected" for item in held_out)
    anchor_wins = sum(item["winner"] == "anchor" for item in held_out)
    ties = sum(item["winner"] == "tie" for item in held_out)
    full_delta = float(final_split["all_strategies"]["delta_vs_anchor"])
    accepted = full_delta > 0.0 and wins >= 3

    return {
        "schema_version": 1,
        "issue": 366,
        "question": "빠른 단일 제거 후보가 35개 기준보다 반복적으로 높은가?",
        "input": {
            "prediction_path": str(
                Path(results["final"]["input"]["prediction_path"]).resolve()
            ),
            "prediction_sha256": input_hashes.pop(),
            "members": len(context.members),
        },
        "procedure": {
            "screen_strategy": SCREEN_STRATEGY,
            "screen_limit": 1,
            "screen_role": "단계별 정확 평가 후보의 순서만 정한다.",
            "fixed_acceptance_strategy": FIXED_STRATEGY,
            "max_steps": 5,
            "required_held_out_wins": 3,
            "final_registered_strategy_count": 19,
        },
        "source_results": {
            label: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "elapsed_seconds": results[label]["execution"]["elapsed_seconds"],
                "strategy_fits": results[label]["execution"]["strategy_fits"],
            }
            for label, path in paths.items()
        },
        "splits": {
            f"outer-{label}" if label != "final" else "final": result["split"]
            for label, result in results.items()
        },
        "final_removed_members": _member_metadata(context, removed_members),
        "validation": {
            "held_out_wins": wins,
            "held_out_anchor_wins": anchor_wins,
            "held_out_ties": ties,
            "requires_wins": 3,
            "full_oof_improved": full_delta > 0.0,
            "full_oof_delta": full_delta,
            "accepted": accepted,
        },
        "decision": {
            "pool": "selected" if accepted else "anchor_35",
            "strategy": (
                final_split["all_strategies"]["selected_best_strategy"]
                if accepted
                else final_split["all_strategies"]["anchor_best_strategy"]
            ),
            "reason": (
                "전체 OOF 개선과 바깥쪽 검증 3/5를 모두 충족했다."
                if accepted
                else "전체 OOF는 개선됐지만 바깥쪽 검증 승리가 3/5에 못 미쳤다."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    sha256 = _write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256,
        "validation": payload["validation"],
        "decision": payload["decision"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
