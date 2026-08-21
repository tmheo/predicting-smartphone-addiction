"""#340 원형 실행기.

    uv run python prototypes/issue-340/runner.py verify
    uv run python prototypes/issue-340/runner.py cost

`verify`는 누출 경계, 결정적 재현, 중간 저장 재개, 산출물 모양을 싼 설정으로 검사한다.
`cost`는 등록 전략 전부를 쓰는 설정으로 절차를 한 번 돌려 결합 전략 적합 횟수를 세고,
실제 규모(691,369행 · 35구성원 · 19전략)로 외삽한 비용을 보고한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fixture import build_fixture  # noqa: E402
import procedure as proc  # noqa: E402

OUT_DIR = HERE / "out"

# #347 11절이 대조마다 요구한 항목.
REQUIRED_CONTRAST_FIELDS = (
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
)

# 실제 규모. 외삽의 분모다.
REAL_ROWS = 691_369
REAL_MEMBERS = 35
REAL_STRATEGIES = len(proc.FULL_STRATEGIES)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def _tiny() -> tuple[Any, dict[str, Any]]:
    fixture = build_fixture(rows=2000, wide=False)
    registry = proc.build_registry(fixture, proc.TINY_STRATEGIES)
    return fixture, registry


def check_shape(result: proc.ProcedureResult) -> list[str]:
    """산출물 모양: 대조마다 #347 11절 항목이 실제로 남는가."""
    notes = []
    splits = list(result.outer.values()) + [result.final_run]
    contrasts = 0
    for outcome in splits:
        for record in outcome.trajectory:
            contrasts += 1
            payload = record.__dict__
            missing = [f for f in REQUIRED_CONTRAST_FIELDS if f not in payload]
            if missing:
                raise AssertionError(f"{outcome.label} step {record.step}: 누락 {missing}")
    notes.append(f"대조 {contrasts}건 전부가 11절 항목 {len(REQUIRED_CONTRAST_FIELDS)}개를 남겼다.")
    return notes


def check_leak(fixture, registry, result: proc.ProcedureResult) -> list[str]:
    """누출 경계: 채점 분할의 목표값을 뒤섞어도 그 분할의 결정이 바뀌지 않아야 한다.

    fold k의 목표값만 무작위로 섞고 fold k의 학습 분할 절차를 다시 돌린다.
    풀, 전략, 궤적의 어느 숫자든 달라지면 채점 분할이 결정에 새어든 것이다.
    """
    notes = []
    rng = np.random.default_rng(11_340)
    for fold, outcome in sorted(result.outer.items()):
        score = (fixture.fold_of == fold).to_numpy()
        shuffled = fixture.y.copy()
        block = shuffled.to_numpy()[score]
        shuffled.iloc[np.flatnonzero(score)] = rng.permutation(block)
        train = ~score
        replayed = proc.run_split(
            f"outer-{fold}",
            tuple(fixture.configs),
            fixture.preds[train],
            fixture.fold_of[train],
            shuffled[train],
            registry,
            fixture.ledger,
            proc.FitCounter(),
        )
        before = _dump(proc._split_payload(outcome))
        after = _dump(proc._split_payload(replayed))
        if before != after:
            raise AssertionError(f"outer-{fold}: 채점 분할 목표값이 결정에 닿았다.")
        notes.append(f"outer-{fold}: 채점 분할 목표값 치환에도 결정이 글자 단위로 같다.")
    return notes


def check_determinism(fixture, registry, result: proc.ProcedureResult) -> list[str]:
    """결정적 재현: 같은 입력을 두 번 돌리면 산출물이 글자 단위로 같아야 한다."""
    again = proc.run_procedure(fixture, registry)
    if _dump(result.decision_payload()) != _dump(again.decision_payload()):
        raise AssertionError("같은 입력의 두 실행이 다른 산출물을 냈다.")
    return ["같은 입력의 두 실행이 글자 단위로 같은 산출물을 냈다."]


def check_resume(fixture, registry, result: proc.ProcedureResult) -> list[str]:
    """중간 저장: 절반에서 끊고 이어받아도 결과가 같아야 한다."""
    path = OUT_DIR / "resume-checkpoint.jsonl"
    path.unlink(missing_ok=True)
    partial = proc.run_procedure(fixture, registry, checkpoint=path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise AssertionError("중간 저장 단위가 너무 적어 재개를 시험할 수 없다.")
    # 절차가 도중에 죽은 상황을 흉내낸다. 앞 2단위만 남기고 잘라낸다.
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    resumed = proc.run_procedure(fixture, registry, checkpoint=path)
    if _dump(partial.decision_payload()) != _dump(resumed.decision_payload()):
        raise AssertionError("재개한 실행이 처음부터 돈 실행과 다른 결정 산출물을 냈다.")
    if _dump(result.decision_payload()) != _dump(resumed.decision_payload()):
        raise AssertionError("중간 저장을 쓴 실행이 쓰지 않은 실행과 다르다.")
    if resumed.fits >= partial.fits:
        raise AssertionError("재개가 끝난 단위를 다시 계산했다.")
    path.unlink(missing_ok=True)
    return [
        f"중간 저장 {len(lines)}단위 중 2단위만 남기고 재개해도 결정 산출물이 글자 단위로 같다.",
        f"재개 실행의 결합 전략 적합은 {resumed.fits}회로 처음부터 돈 {partial.fits}회보다 적다.",
        "실행 계측은 재현 대상이 아니라 결정 산출물과 분리해 기록해야 한다.",
    ]


def check_divergence(result: proc.ProcedureResult) -> list[str]:
    """분할별 축소안이 서로 얼마나 다른가. 지도의 미확정 항목에 넘길 사실이다."""
    pools = {fold: set(o.final_pool) for fold, o in sorted(result.outer.items())}
    final = set(result.final_run.final_pool)
    union = set().union(*pools.values())
    common = set.intersection(*pools.values()) if pools else set()
    counts = {
        member: sum(member in pool for pool in pools.values()) for member in union
    }
    unstable = sorted(m for m, c in counts.items() if 0 < c < len(pools))
    notes = [
        f"분할별 축소 풀 크기: "
        + ", ".join(f"fold {k}={len(v)}" for k, v in sorted(pools.items())),
        f"5분할 전부에 남은 구성원 {len(common)}개, 어느 한 분할에라도 남은 구성원 {len(union)}개.",
        f"분할마다 갈린 구성원 {len(unstable)}개: {', '.join(unstable) if unstable else '없음'}",
        f"전체 OOF 최종 실행 풀 {len(final)}개: {', '.join(sorted(final))}",
        f"분할별 최선 전략: "
        + ", ".join(f"fold {k}={o.final_strategy}" for k, o in sorted(result.outer.items()))
        + f", 최종={result.final_run.final_strategy}",
    ]
    return notes


def verify() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture, registry = _tiny()
    started = time.monotonic()
    result = proc.run_procedure(fixture, registry)
    elapsed = time.monotonic() - started

    report: dict[str, list[str]] = {}
    report["산출물 모양"] = check_shape(result)
    report["누출 경계"] = check_leak(fixture, registry, result)
    report["결정적 재현"] = check_determinism(fixture, registry, result)
    report["중간 저장 재개"] = check_resume(fixture, registry, result)
    report["분할별 갈림"] = check_divergence(result)

    (OUT_DIR / "verify.json").write_text(
        _dump(result.payload()), encoding="utf-8"
    )
    print(f"# 원형 검증 (행 {len(fixture.preds)}, 구성원 {len(fixture.configs)}, "
          f"전략 {len(registry)}, {elapsed:.1f}초, 적합 {result.fits}회)")
    print(f"절차 nested OOF AUC {result.nested_auc:.12f} / "
          f"35개 앵커 대응값 {result.anchor_nested_auc:.12f} / "
          f"Δ {result.nested_auc - result.anchor_nested_auc:+.12f}")
    for title, notes in report.items():
        print(f"\n## {title}")
        for note in notes:
            print(f"- {note}")


def cost() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture(rows=4000, wide=True)
    registry = proc.build_registry(fixture, proc.FULL_STRATEGIES)
    started = time.monotonic()
    result = proc.run_procedure(
        fixture, registry, progress=lambda line: print(line, flush=True)
    )
    elapsed = time.monotonic() - started
    (OUT_DIR / "cost.json").write_text(_dump(result.payload()), encoding="utf-8")

    members = len(fixture.configs)
    contrasts = sum(
        len(o.trajectory) for o in list(result.outer.values()) + [result.final_run]
    )
    # 대조 수는 구성원 수에 대체로 비례한다. 실제 35개 풀로 선형 외삽한다.
    scale_contrasts = REAL_MEMBERS / members
    scale_strategies = REAL_STRATEGIES / len(registry)
    scale_rows = REAL_ROWS / len(fixture.preds)
    projected_fits = result.fits * scale_contrasts * scale_strategies
    per_fit = elapsed / result.fits

    print(f"\n# 비용 (행 {len(fixture.preds)}, 구성원 {members}, 전략 {len(registry)})")
    print(f"- 소요 {elapsed:.1f}초, 결합 전략 적합 {result.fits}회, 대조 {contrasts}건")
    print(f"- 이 설정의 적합 1회 평균 {per_fit:.3f}초")
    print(f"- 실제 35개 풀 외삽 적합 횟수 약 {projected_fits:,.0f}회")
    print(f"- 행 배율 {scale_rows:.0f}배는 결합 전략마다 다르므로 여기서 곱하지 않는다.")
    print(f"- 분할별 축소 풀 크기: "
          + ", ".join(f"fold {k}={len(o.final_pool)}" for k, o in sorted(result.outer.items()))
          + f", 최종={len(result.final_run.final_pool)}")


def scale() -> None:
    """실제 규모 한 점 측정.

    합성이지만 행 수와 구성원 수를 실제와 같게 두고 `score_pool` 한 번의 비용을 잰다.
    `cost`의 적합 횟수 외삽에 곱할 적합 1회 단가를 여기서 얻는다.
    """
    fixture = build_fixture(rows=REAL_ROWS, wide=True)
    # 실제 35개 풀 크기를 맞추려고 합성 구성원을 되풀이해 늘린다.
    #   비용의 지배 요인은 행 수와 열 수이므로 열의 정보 내용은 단가에 영향이 없다.
    base = fixture.preds
    widened = base.copy()
    index = 0
    while widened.shape[1] < REAL_MEMBERS:
        source = base.columns[index % base.shape[1]]
        widened[f"{source}__pad{index}"] = base[source].to_numpy()
        index += 1
    pool = tuple(widened.columns[:REAL_MEMBERS])
    registry = proc.build_registry(fixture, proc.FULL_STRATEGIES)

    # 바깥쪽 학습 분할 하나(4/5 행)에서 leave-one-fold-out을 도는 실제 모양 그대로 잰다.
    train = (fixture.fold_of != 0).to_numpy()
    counter = proc.FitCounter()
    started = time.monotonic()
    choice, _ = proc.score_pool(
        pool,
        widened[train],
        fixture.fold_of[train],
        fixture.y[train],
        registry,
        counter,
    )
    elapsed = time.monotonic() - started
    per_fit = elapsed / counter.fits
    print(f"# 실제 규모 단가 (행 {REAL_ROWS:,}, 구성원 {REAL_MEMBERS}, 전략 {len(registry)})")
    print(f"- 학습 분할 하나의 score_pool 1회: {elapsed:.1f}초, 적합 {counter.fits}회")
    print(f"- 적합 1회 평균 {per_fit:.3f}초")
    print(f"- 고른 전략 {choice.name}, 수렴 실패 {len(choice.failures)}개")
    for scale_name, fits in (("절차 1회", 34_212), ("영점 대조 36팔", 36 * 385)):
        hours = fits * per_fit / 3600.0
        print(f"- {scale_name}: 적합 {fits:,}회 → 단일 처리 약 {hours:.1f}시간")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "cost", "scale"))
    args = parser.parse_args()
    {"verify": verify, "cost": cost, "scale": scale}[args.command]()


if __name__ == "__main__":
    main()
