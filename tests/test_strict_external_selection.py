"""ADR-0005 검색 규칙의 특성화 시험: 이동 집합, 동률, 충돌·교체, 쌍 추가, 정지. (#486)"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/judge_strict_external_selection.py")
SPEC = importlib.util.spec_from_file_location("judge_strict_external_selection", SCRIPT)
assert SPEC and SPEC.loader
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)

OWN = 2  # 자체 열 0, 1
OPEN = frozenset({1, 2, 3, 4})


class TableEngine:
    """열 부분집합 → 점수 표. 없는 부분집합은 기본값."""

    def __init__(self, table: dict[tuple[int, ...], float], default: float = 0.5) -> None:
        self.table = table
        self.default = default
        self.calls: list[tuple[int, ...]] = []

    def pool_score(self, open_folds, subset):
        self.calls.append(subset)
        return self.table.get(subset, self.default), {}


def search(table, candidates, excluded=(), conflicts=None):
    engine = TableEngine(table)
    result = JUDGE.Search(engine, OPEN, OWN, list(candidates), set(excluded), {k: set(v) for k, v in (conflicts or {}).items()}, None)
    reason = result.run()
    return result, reason, engine


def test_forward_accepts_largest_positive_and_stops_without_positive():
    table = {(0, 1): 0.5, (0, 1, 2): 0.51, (0, 1, 3): 0.52, (0, 1, 2, 3): 0.52}
    result, reason, _ = search(table, [2, 3])
    assert result.selected == [3]
    assert reason == "no_positive_pair"
    stages = [s["stage"] for s in result.stages]
    assert stages[0] == "phase1:forward:1" and result.stages[0]["accepted"]["incoming"] == 3
    assert result.stages[1]["accepted"] is None  # 3 뒤에 2를 더해도 양수가 아니다


def test_tie_resolves_by_frozen_order_and_add_before_swap():
    table = {(0, 1): 0.5, (0, 1, 2): 0.51, (0, 1, 3): 0.51}
    result, _, _ = search(table, [3, 2])  # 동결 순서: 3이 먼저
    assert result.stages[0]["accepted"]["incoming"] == 3


def test_excluded_candidate_never_moves():
    table = {(0, 1): 0.5, (0, 1, 2): 0.9}
    result, _, engine = search(table, [2, 3], excluded=[2])
    assert result.selected == []
    assert all(2 not in subset for subset in engine.calls)


def test_conflict_allows_single_atomic_swap_only():
    # 3은 2와 충돌. 2가 먼저 들어가면 3은 추가가 아니라 교체로만 평가된다.
    table = {(0, 1): 0.5, (0, 1, 2): 0.51, (0, 1, 3): 0.505, (0, 1, 2, 3): 0.99}
    result, _, engine = search(table, [2, 3], conflicts={2: [3], 3: [2]})
    assert result.selected == [2]
    assert (0, 1, 2, 3) not in engine.calls
    step2 = result.stages[1]
    assert [m["move"] for m in step2["evaluated"]] == ["swap"]
    assert step2["evaluated"][0]["incoming"] == 3 and step2["evaluated"][0]["outgoing"] == 2


def test_swap_is_accepted_when_strictly_better():
    table = {(0, 1): 0.5, (0, 1, 2): 0.51, (0, 1, 3): 0.505, (0, 1, 3, 4): 0.6, (0, 1, 2, 4): 0.52, (0, 1, 4): 0.5}
    # 4는 2와 함께면 0.52, 3과 함께면 0.6. 2→3 교체는 (2,4)=0.52 → (3,4)=0.6로 양수.
    result, _, _ = search(table, [2, 3, 4], conflicts={2: [3], 3: [2]})
    assert sorted(result.selected) == [3, 4]


def test_multi_conflict_blocks_swap():
    # 4는 2, 3 둘 다와 충돌. 2, 3이 선택된 뒤에는 단일 원자 교체로 불변식을 회복할 수 없어 이동이 없다.
    table = {(0, 1): 0.5, (0, 1, 2): 0.51, (0, 1, 2, 3): 0.52, (0, 1, 3): 0.505, (0, 1, 4): 0.45, (0, 1, 3, 4): 0.99, (0, 1, 2, 4): 0.99}
    result, _, engine = search(table, [2, 3, 4], conflicts={4: [2, 3], 2: [4], 3: [4]})
    assert sorted(result.selected) == [2, 3]
    assert all(subset == (0, 1, 4) for subset in engine.calls if 4 in subset)


def test_backward_removes_when_removal_improves():
    # 2 단독 +, 3 단독 +, 둘 다면 더 좋지만 2를 빼면 최고.
    table = {(0, 1): 0.5, (0, 1, 2): 0.52, (0, 1, 3): 0.51, (0, 1, 2, 3): 0.53, (0, 1, 3, 4): 0.9}
    result, _, _ = search(table, [2, 3])
    assert sorted(result.selected) == [2, 3]
    # 제거 평가가 있었고 양수 제거가 없어 승인되지 않았다.
    backward = next(s for s in result.stages if "backward" in s["stage"])
    assert [m["move"] for m in backward["evaluated"]] == ["remove", "remove"]
    assert backward["accepted"] is None


def test_pair_sweep_accepts_complementary_pair_then_reconverges():
    # 2와 3은 단독으로 도움이 안 되지만 함께면 도움이 된다.
    table = {(0, 1): 0.5, (0, 1, 2): 0.5, (0, 1, 3): 0.5, (0, 1, 2, 3): 0.6, (0, 1, 2, 3, 4): 0.61, (0, 1, 4): 0.5}
    result, reason, _ = search(table, [2, 3, 4])
    assert reason == "pair_accepted_then_converged"
    assert sorted(result.selected) == [2, 3, 4]
    pair_stage = next(s for s in result.stages if s["stage"] == "pair_sweep")
    assert pair_stage["accepted"]["incoming"] == [2, 3]
    after = [s["stage"] for s in result.stages]
    after = after[after.index("pair_sweep") + 1:]
    assert after and all(stage.startswith("phase2") for stage in after)


def test_pair_sweep_skips_conflicting_pairs():
    table = {(0, 1): 0.5, (0, 1, 2, 3): 0.9, (0, 1, 3, 4): 0.9}
    result, _, engine = search(table, [2, 3, 4], conflicts={2: [3], 3: [2]})
    assert sorted(result.selected) == [3, 4]
    assert (0, 1, 2, 3) not in engine.calls
    assert (0, 1, 3, 4) in engine.calls


def test_pair_sweep_skips_pairs_blocked_by_selected_conflict():
    # 2가 선택된 뒤 3은 2와 충돌하므로 쌍 추가 후보가 될 수 없다.
    table = {(0, 1): 0.5, (0, 1, 2): 0.6, (0, 1, 2, 3, 4): 0.99, (0, 1, 2, 4): 0.6}
    result, _, engine = search(table, [2, 3, 4], conflicts={2: [3], 3: [2]})
    assert result.selected == [2]
    assert (0, 1, 2, 3, 4) not in engine.calls
