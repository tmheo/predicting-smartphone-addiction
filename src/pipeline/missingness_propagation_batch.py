"""결측 증강 전파 일괄 판정의 사전 고정 계약과 결정적 검색.

이 모듈은 예측을 만들거나 후보 풀 장부를 쓰지 않는다.
이슈 515의 변경 불가 사전 기록을 검증하고, 후속 판정이 써야 하는
원본과 결측 증강판의 원자 교체 검색 순서를 한 곳에서 정의한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import file_sha256
from .judgment import DUPLICATE_SPEARMAN

CONTRACT_VERSION = "missingness-propagation-batch-v1"
PRECOMMIT_SCHEMA = f"{CONTRACT_VERSION}/precommit/1"
INPUT_BUNDLE_SCHEMA = f"{CONTRACT_VERSION}/input/1"
PAIR_COUNT = 34
POOL_MEMBER_COUNT = 36
FIXED_MEMBERS = (
    "exp067_tabpfn3",
    "exp208_issue500_ag25_missingness_augmented",
)
CHAMPION_FLOOR_MARGIN = 0.01
SEARCH_STRATEGY = "shrunk_rank_logit_logistic"
ADOPTION_STRATEGIES = (
    "missing_segmented_rank_logit",
    "missing_interaction_rank_logit",
    "shrunk_rank_logit_logistic",
)
OUTER_FOLDS = (0, 1, 2, 3, 4)


SEARCH_RULES = {
    "state": (
        "현재 36개 풀의 고정 구성원 2개와 34개 상호 배타 자리로 구성하며, "
        "각 자리는 원본과 결측 증강판 가운데 정확히 하나만 가진다."
    ),
    "score": (
        "검색 범위의 고정 OOF에서 shrunk_rank_logit_logistic의 nested OOF AUC를 쓴다."
    ),
    "eligibility": (
        "결측 증강판의 시드 평균 OOF AUC가 동결 champion OOF AUC보다 0.01 이상 "
        "낮으면 원본에서 결측 증강판으로 가는 검색 이동 자격이 없다."
    ),
    "forward": (
        "자격 있는 미선택 결측 증강판으로 바꾸는 모든 단일 원자 교체를 정확히 평가하고, "
        "중복 불변식을 지키는 엄격 양수 이동 가운데 증가가 가장 큰 하나를 받아 수렴시킨다."
    ),
    "backward": (
        "선택된 결측 증강판을 원본으로 되돌리는 모든 단일 원자 교체를 정확히 평가하고, "
        "중복 불변식을 지키는 엄격 양수 이동 가운데 증가가 가장 큰 하나를 받아 수렴시킨다."
    ),
    "pair": (
        "남은 자격 있는 결측 증강판의 순서 없는 두 개 원자 교체를 한 번 전수 평가하고, "
        "엄격 양수인 최선 묶음 하나만 받는다."
    ),
    "sequence": (
        "순방향 수렴, 역방향 수렴, 순방향 수렴, 두 개 묶음 전수 평가 순서다. "
        "묶음을 받으면 순방향, 역방향, 순방향을 다시 수렴시킨다."
    ),
    "tie": (
        "AUC 증가가 정확히 같을 때만 후보 풀과 짝의 동결 순서가 앞선 이동을 고른다. "
        "별도 부동소수점 허용 오차는 쓰지 않는다."
    ),
    "direct_pair_delta": (
        "3배 행 대조군과 결측 증강군의 직접 OOF 차이는 진단으로만 기록하고 검색 전 제외에 쓰지 않는다."
    ),
}

DUPLICATE_RULES = {
    "threshold": DUPLICATE_SPEARMAN,
    "full_oof": (
        "전체 OOF 제안 풀의 모든 구성원 쌍은 스피어만 순위 상관이 0.998 미만이어야 한다."
    ),
    "outer_open_folds": (
        "바깥쪽 채점 분할을 뺀 네 분할 검색은 시작 풀의 기존 위반만 보존할 수 있고 새 위반을 만들 수 없다."
    ),
}

CONDITIONAL_RULES = {
    "outer_search": (
        "바깥쪽 채점 분할마다 나머지 네 분할에서 전체 검색을 다시 수행하고, "
        "선택 풀에 결합기를 한 번 맞춰 빠진 분할을 예측한다."
    ),
    "procedure_score": (
        "다섯 바깥쪽 예측을 원래 행 순서로 이어 붙인 AUC가 동결 OOF 조건부 절차 점수다."
    ),
    "proposal": (
        "전체 다섯 분할 OOF 검색의 단일 제안 풀만 공식화 후보이며, "
        "분할별 풀은 절차 점수와 선택 안정성 진단에만 쓴다."
    ),
    "no_fold_pool_aggregation": (
        "분할별 풀의 투표, 교집합 또는 합집합으로 제안 풀을 만들지 않는다."
    ),
}

ADOPTION_RULES = {
    "conditional_gate": (
        "제안 절차의 동결 OOF 조건부 절차 점수가 현재 고정 36개 풀의 같은 절차 점수보다 엄격히 높아야 한다."
    ),
    "direct_gate": (
        "단일 제안 풀과 현재 36개 풀을 핵심 결합 방식 세 가지로 각각 nested 평가하고, "
        "각 풀의 최선 방식끼리 비교한 전체 nested OOF 차이가 엄격히 양수여야 한다."
    ),
    "diagnostics_only": (
        "방식별 차이, 바깥 분할 승수와 성능 동등 대역은 경고와 진단이며 채택 문턱이 아니다."
    ),
    "all_required": "두 채택 관문은 모두 완결되고 통과해야 한다.",
}

FAILURE_RULES = {
    "pair_completion": (
        "중앙 반입 마감까지 같은 공급자와 실행 환경 등급의 두 팔 및 모든 무결성 관문을 완결한 짝만 입력한다."
    ),
    "all_completed_pairs": (
        "완결된 짝은 직접 OOF 차이의 부호와 관계없이 모두 입력하며, 미완결 짝은 상태와 사유만 기록한다."
    ),
    "no_partial_recovery": (
        "한 팔만 완주한 짝이나 서로 다른 공급자 결과를 이어 붙여야 하는 짝은 입력하지 않는다."
    ),
    "judgment_failure": (
        "입력 묶음을 동결한 뒤 전체 검색이나 두 관문 가운데 하나라도 실패하거나 미완결이면 "
        "부분 결과를 채택하지 않고 현재 36개 풀을 유지한다."
    ),
}

FORMALIZATION_RULES = {
    "proposal_only": "검색 결과는 제안이며 후보 풀 장부를 직접 바꾸지 않는다.",
    "refit_rehearsal": (
        "새로 선택된 모든 결측 증강판이 전체 자료 training_rows를 지원하고, "
        "전체 자료 재학습 스모크 예행과 계보 및 예측 산출물 검증을 통과해야 한다."
    ),
    "atomic_ledgers": (
        "모든 관문이 통과한 같은 제안 커밋과 공식화 경로에서 artifacts/pool.yaml과 "
        "artifacts/full-refit-plan.yaml을 함께 바꾼다."
    ),
    "failure_keeps_both": "하나라도 실패하거나 미완결이면 두 장부를 모두 현재 상태로 유지한다.",
    "public_score": "Public 점수는 구현, 선택, 판정과 채택에 사용하지 않는다.",
}


class MissingnessPropagationBatchError(RuntimeError):
    """사전 고정 계약, 판정 입력 또는 결정적 검색이 유효하지 않다."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MissingnessPropagationBatchError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity_sha256(identity: Mapping[str, Any]) -> str:
    """신원 본문을 키 순서와 공백에 무관한 내용 SHA-256으로 식별한다."""
    return canonical_sha256(dict(identity))


def self_hashed_payload(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = canonical_sha256(result)
    return result


def verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    actual = payload.get(field)
    without_hash = {key: value for key, value in payload.items() if key != field}
    require(actual == canonical_sha256(without_hash), f"{field}가 본문 내용과 다르다.")


@dataclass(frozen=True)
class SearchMove:
    kind: str
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]
    selected_after: tuple[int, ...]
    score: float
    delta: float


@dataclass(frozen=True)
class SearchStage:
    name: str
    start_selected: tuple[int, ...]
    start_score: float
    evaluated: tuple[SearchMove, ...]
    accepted: SearchMove | None


@dataclass(frozen=True)
class SearchResult:
    selected: tuple[int, ...]
    score: float
    stages: tuple[SearchStage, ...]
    evaluated_state_count: int
    termination: str

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


class MaximumGainSearch:
    """34개 상호 배타 자리의 결정적 최대 상승 원자 교체 검색."""

    def __init__(
        self,
        *,
        pair_order: Sequence[int],
        eligible: Iterable[int],
        score: Callable[[tuple[int, ...]], float],
        allowed: Callable[[tuple[int, ...]], bool],
        score_many: Callable[[tuple[tuple[int, ...], ...]], Sequence[float]] | None = None,
        allow_invalid_start: bool = False,
    ) -> None:
        self.order = tuple(pair_order)
        require(len(self.order) == len(set(self.order)), "짝 동결 순서에 중복이 있다.")
        self.rank = {pair: position for position, pair in enumerate(self.order)}
        self.eligible = frozenset(eligible)
        require(
            self.eligible <= set(self.order), "검색 이동 자격에 알 수 없는 짝이 있다."
        )
        self._score_callback = score
        self._score_many_callback = score_many
        self._allowed = allowed
        self._scores: dict[tuple[int, ...], float] = {}
        self._stages: list[SearchStage] = []
        self.selected: tuple[int, ...] = ()
        require(
            self._allowed(self.selected) or allow_invalid_start,
            "검색 시작 풀은 현재 범위의 중복 규칙을 만족하거나 기존 위반 예외가 명시돼야 한다.",
        )

    def _state(self, values: Iterable[int]) -> tuple[int, ...]:
        selected = set(values)
        require(selected <= set(self.order), "검색 상태에 알 수 없는 짝이 있다.")
        return tuple(pair for pair in self.order if pair in selected)

    def score(self, state: tuple[int, ...]) -> float:
        state = self._state(state)
        if state not in self._scores:
            value = float(self._score_callback(state))
            require(math.isfinite(value), f"검색 점수가 유한하지 않다: {state}")
            self._scores[state] = value
        return self._scores[state]

    def _prefetch(self, states: Iterable[tuple[int, ...]]) -> None:
        ordered = tuple(dict.fromkeys(self._state(state) for state in states))
        missing = tuple(state for state in ordered if state not in self._scores)
        if not missing:
            return
        if self._score_many_callback is None:
            values = tuple(float(self._score_callback(state)) for state in missing)
        else:
            values = tuple(float(value) for value in self._score_many_callback(missing))
            require(
                len(values) == len(missing),
                "일괄 검색 점수 수가 요청 상태 수와 다르다.",
            )
        for state, value in zip(missing, values, strict=True):
            require(math.isfinite(value), f"검색 점수가 유한하지 않다: {state}")
            self._scores[state] = value

    def _move(
        self,
        *,
        kind: str,
        incoming: tuple[int, ...],
        outgoing: tuple[int, ...],
        selected_after: Iterable[int],
        current_score: float,
    ) -> SearchMove | None:
        state = self._state(selected_after)
        if not self._allowed(state):
            return None
        score = self.score(state)
        return SearchMove(
            kind=kind,
            incoming=incoming,
            outgoing=outgoing,
            selected_after=state,
            score=score,
            delta=score - current_score,
        )

    def _tie_key(self, move: SearchMove) -> tuple[int, ...]:
        relevant = move.incoming if move.incoming else move.outgoing
        return tuple(self.rank[pair] for pair in relevant)

    def _step(self, name: str, moves: Iterable[SearchMove | None]) -> SearchMove | None:
        current = self.selected
        current_score = self.score(current)
        evaluated = tuple(move for move in moves if move is not None)
        positive = [move for move in evaluated if move.delta > 0.0]
        accepted = (
            min(positive, key=lambda move: (-move.delta, self._tie_key(move)))
            if positive
            else None
        )
        if accepted is not None:
            self.selected = accepted.selected_after
        self._stages.append(
            SearchStage(
                name=name,
                start_selected=current,
                start_score=current_score,
                evaluated=evaluated,
                accepted=accepted,
            )
        )
        return accepted

    def _forward(self, label: str) -> None:
        step = 0
        while True:
            step += 1
            current_score = self.score(self.selected)
            selected = set(self.selected)
            candidates = [
                (pair, self._state((*self.selected, pair)))
                for pair in self.order
                if pair in self.eligible and pair not in selected
            ]
            self._prefetch(
                state for _pair, state in candidates if self._allowed(state)
            )
            moves = [
                self._move(
                    kind="original_to_missingness_augmented",
                    incoming=(pair,),
                    outgoing=(pair,),
                    selected_after=state,
                    current_score=current_score,
                )
                for pair, state in candidates
            ]
            if self._step(f"{label}:forward:{step}", moves) is None:
                return

    def _backward(self, label: str) -> None:
        step = 0
        while True:
            step += 1
            current_score = self.score(self.selected)
            candidates = [
                (
                    pair,
                    self._state(value for value in self.selected if value != pair),
                )
                for pair in self.selected
            ]
            self._prefetch(
                state for _pair, state in candidates if self._allowed(state)
            )
            moves = [
                self._move(
                    kind="missingness_augmented_to_original",
                    incoming=(),
                    outgoing=(pair,),
                    selected_after=state,
                    current_score=current_score,
                )
                for pair, state in candidates
            ]
            if self._step(f"{label}:backward:{step}", moves) is None:
                return

    def _pair_sweep(self) -> bool:
        current_score = self.score(self.selected)
        selected = set(self.selected)
        remaining = [
            pair
            for pair in self.order
            if pair in self.eligible and pair not in selected
        ]
        candidates: list[tuple[int, int, tuple[int, ...]]] = []
        for first_position, first in enumerate(remaining):
            for second in remaining[first_position + 1 :]:
                candidates.append(
                    (first, second, self._state((*self.selected, first, second)))
                )
        self._prefetch(
            state for _first, _second, state in candidates if self._allowed(state)
        )
        moves: list[SearchMove | None] = [
            self._move(
                kind="two_atomic_replacements",
                incoming=(first, second),
                outgoing=(first, second),
                selected_after=state,
                current_score=current_score,
            )
            for first, second, state in candidates
        ]
        return self._step("pair_sweep", moves) is not None

    def run(self) -> SearchResult:
        self.score(self.selected)
        self._forward("phase1")
        self._backward("phase1")
        self._forward("phase1b")
        if self._pair_sweep():
            self._forward("phase2")
            self._backward("phase2")
            self._forward("phase2b")
            termination = "pair_accepted_then_converged"
        else:
            termination = "no_positive_pair"
        return SearchResult(
            selected=self.selected,
            score=self.score(self.selected),
            stages=tuple(self._stages),
            evaluated_state_count=len(self._scores),
            termination=termination,
        )


def spearman_violations(
    predictions: pd.DataFrame,
    members: Sequence[str],
    *,
    threshold: float = DUPLICATE_SPEARMAN,
) -> frozenset[tuple[str, str]]:
    """주어진 행 범위에서 중복 문턱 이상인 구성원 쌍을 돌려준다."""
    require(len(members) == len(set(members)), "중복 검사 구성원 이름이 겹친다.")
    require(set(members) <= set(predictions.columns), "중복 검사 예측 열이 빠졌다.")
    correlation = predictions.loc[:, list(members)].corr(method="spearman")
    require(
        np.isfinite(correlation.to_numpy(dtype=np.float64)).all(),
        "스피어만 상관 행렬에 유한하지 않은 값이 있다.",
    )
    violations: set[tuple[str, str]] = set()
    for left_position, left in enumerate(members):
        for right in members[left_position + 1 :]:
            if float(correlation.loc[left, right]) >= threshold:
                violations.add((left, right))
    return frozenset(violations)


def validate_precommit(
    payload: Mapping[str, Any],
    repo_root: Path,
    *,
    allow_contract_module_correction: bool = False,
) -> None:
    """커밋된 사전 기록이 현재 고정 파일과 정확히 맞는지 확인한다."""
    require(payload.get("schema") == PRECOMMIT_SCHEMA, "사전 기록 스키마가 다르다.")
    require(
        payload.get("contract_version") == CONTRACT_VERSION, "판정 계약 판본이 다르다."
    )
    verify_self_hash(payload, "precommit_sha256")

    for name, record in payload["inputs"].items():
        path = repo_root / record["path"]
        require(path.is_file(), f"동결 입력이 없다: {name} {record['path']}")
        require(
            file_sha256(path) == record["sha256"],
            f"동결 입력 내용 해시가 다르다: {name}",
        )

    pool_path = repo_root / payload["inputs"]["candidate_pool"]["path"]
    pool = yaml.safe_load(pool_path.read_text(encoding="utf-8"))
    champion_path = repo_root / payload["inputs"]["champion"]["path"]
    champion = yaml.safe_load(champion_path.read_text(encoding="utf-8"))
    pair_freeze_path = repo_root / payload["inputs"]["pair_execution_freeze"]["path"]
    pair_freeze = json.loads(pair_freeze_path.read_text(encoding="utf-8"))
    length_evidence_path = (
        repo_root / payload["inputs"]["paired_training_length_evidence"]["path"]
    )
    length_evidence = json.loads(length_evidence_path.read_text(encoding="utf-8"))
    capacity_path = repo_root / payload["inputs"]["parallel_capacity_freeze"]["path"]
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    pool_names = [member["config"] for member in pool["members"]]
    pool_run_ids = [member["run_id"] for member in pool["members"]]
    require(len(pool_names) == POOL_MEMBER_COUNT, "동결 후보 풀이 36개가 아니다.")
    require(len(pool_names) == len(set(pool_names)), "동결 후보 풀 이름이 중복된다.")
    require(
        len(pool_run_ids) == len(set(pool_run_ids)),
        "동결 후보 풀 실행 신원이 중복된다.",
    )
    require(
        pool_names == payload["scope"]["pool_member_order"], "후보 풀 순서가 바뀌었다."
    )

    fixed = payload["fixed_members"]
    require(
        [record["member"] for record in fixed] == list(FIXED_MEMBERS),
        "고정 구성원이 다르다.",
    )
    for record in fixed:
        require(
            identity_sha256(record["identity"]) == record["identity_sha256"],
            f"{record['member']}: 고정 구성원 신원 해시가 다르다.",
        )
        member = pool["members"][record["pool_position"] - 1]
        require(
            member["config"] == record["member"]
            and member["run_id"] == record["identity"]["run_id"],
            f"{record['member']}: 후보 풀 신원 또는 위치가 다르다.",
        )
        config_path = repo_root / record["identity"]["source_config_path"]
        require(config_path.is_file(), f"{record['member']}: 고정 구성원 설정이 없다.")
        require(
            file_sha256(config_path) == record["identity"]["source_config_sha256"],
            f"{record['member']}: 고정 구성원 설정 해시가 다르다.",
        )

    pairs = payload["pairs"]
    require(len(pairs) == PAIR_COUNT, "원본과 결측 증강판 대응이 34개가 아니다.")
    require(
        [pair["ordinal"] for pair in pairs] == list(range(1, PAIR_COUNT + 1)),
        "짝 순서가 연속적이지 않다.",
    )
    pair_members = [pair["original"]["member"] for pair in pairs]
    require(len(pair_members) == len(set(pair_members)), "원본 짝 이름이 중복된다.")
    expected_pair_members = [name for name in pool_names if name not in FIXED_MEMBERS]
    require(pair_members == expected_pair_members, "34개 짝이 후보 풀 순서와 다르다.")
    pair_freeze_by_name = {pair["member"]: pair for pair in pair_freeze["pairs"]}
    evidence_by_name = {
        member["member"]: member for member in length_evidence["members"]
    }
    require(
        set(pair_freeze_by_name) == set(expected_pair_members),
        "짝 실행 동결 명세가 현재 34개 원본을 정확히 덮지 않는다.",
    )
    require(
        set(evidence_by_name) == set(expected_pair_members),
        "학습 길이 근거가 현재 34개 원본을 정확히 덮지 않는다.",
    )
    execution_contract_sha256 = canonical_sha256(pair_freeze["pair_contract"])
    require(
        payload["pair_execution_contract"]
        == {
            "sha256": execution_contract_sha256,
            "value": pair_freeze["pair_contract"],
        },
        "짝 실행 계약이 이슈 510의 동결 명세와 다르다.",
    )

    all_identity_hashes: list[str] = [record["identity_sha256"] for record in fixed]
    augmented_names: list[str] = []
    for pair in pairs:
        original = pair["original"]
        augmented = pair["missingness_augmented"]
        member = original["member"]
        source_identity = evidence_by_name[member]["source_identity"]
        pool_member = pool["members"][original["identity"]["pool_position"] - 1]
        expected_original_identity = {
            "role": "current_pool_original",
            "member": member,
            "pool_position": original["identity"]["pool_position"],
            "run_id": pool_member["run_id"],
            "pool_entry_sha256": canonical_sha256(pool_member),
            "source_identity_sha256": canonical_sha256(source_identity),
            "source_git_commit": source_identity["git_commit"],
            "source_config_artifact_sha256": source_identity["config_artifact_sha256"],
            "normalized_config_sha256": source_identity["normalized_config_sha256"],
            "oof_artifact_sha256": source_identity["oof_artifact_sha256"],
            "input_sha256": source_identity["input_sha256"],
        }
        require(
            original["identity"] == expected_original_identity,
            f"{member}: 원본 신원이 동결 출처 근거와 다르다.",
        )
        require(
            identity_sha256(original["identity"]) == original["identity_sha256"],
            f"{member}: 원본 신원 해시가 다르다.",
        )
        generated = pair_freeze_by_name[member]
        require(
            pair["comparison_arms"] == generated["arms"],
            f"{member}: 짝비교 두 팔의 실행 신원이 동결 명세와 다르다.",
        )
        augmented_arm = next(
            arm for arm in generated["arms"] if arm["arm"] == "missingness_augmented"
        )
        expected_augmented_identity = {
            "role": "missingness_augmented_replacement",
            "member": augmented_arm["name"],
            "slot_original_member": member,
            "ordinal": pair["ordinal"],
            "config_path": augmented_arm["path"],
            "config_sha256": augmented_arm["sha256"],
            "common_config_semantic_sha256": generated["common_config_semantic_sha256"],
            "source_original_identity_sha256": original["identity_sha256"],
            "pair_execution_contract_sha256": execution_contract_sha256,
            "required_seeds": list(pair_freeze["pair_contract"]["seeds"]),
            "required_outer_folds": list(pair_freeze["pair_contract"]["outer_folds"]),
            "runtime_class": generated["runtime_class"],
        }
        require(
            augmented["identity"] == expected_augmented_identity,
            f"{member}: 결측 증강판 신원이 실행 동결 명세와 다르다.",
        )
        require(
            identity_sha256(augmented["identity"]) == augmented["identity_sha256"],
            f"{member}: 결측 증강판 신원 해시가 다르다.",
        )
        require(
            augmented["identity"]["source_original_identity_sha256"]
            == original["identity_sha256"],
            f"{member}: 결측 증강판이 원본 신원을 가리키지 않는다.",
        )
        config_path = repo_root / augmented["identity"]["config_path"]
        require(config_path.is_file(), f"{member}: 결측 증강 설정이 없다.")
        require(
            file_sha256(config_path) == augmented["identity"]["config_sha256"],
            f"{member}: 결측 증강 설정 해시가 다르다.",
        )
        augmented_names.append(augmented["member"])
        all_identity_hashes.extend(
            [original["identity_sha256"], augmented["identity_sha256"]]
        )
    require(
        len(augmented_names) == len(set(augmented_names)),
        "결측 증강판 이름이 중복된다.",
    )
    require(
        len(all_identity_hashes) == len(set(all_identity_hashes)),
        "구성원 신원 해시가 중복된다.",
    )
    require(
        payload["identity_manifest_sha256"] == canonical_sha256(all_identity_hashes),
        "구성원 신원 목록 해시가 다르다.",
    )
    require(
        payload["collection_contract"]["central_import_cutoff_utc"]
        == capacity["deadlines_utc"]["central_import"],
        "중앙 반입 마감이 자원 동결 명세와 다르다.",
    )
    require(
        payload["collection_contract"]["execution_source_commit"]
        == payload["source_commit_before_contract"],
        "짝비교 실행 출처 커밋이 계약 출발 커밋과 다르다.",
    )

    require(payload["search"] == SEARCH_RULES, "검색 규칙이 구현과 다르다.")
    require(
        payload["duplicate_invariant"] == DUPLICATE_RULES, "중복 규칙이 구현과 다르다."
    )
    require(
        payload["conditional_procedure"] == CONDITIONAL_RULES,
        "조건부 절차 규칙이 구현과 다르다.",
    )
    require(payload["adoption_gates"] == ADOPTION_RULES, "채택 관문이 구현과 다르다.")
    require(
        payload["failure_handling"] == FAILURE_RULES,
        "미완료 처리 규칙이 구현과 다르다.",
    )
    require(
        payload["formalization"] == FORMALIZATION_RULES, "공식화 규칙이 구현과 다르다."
    )
    require(
        payload["scope"]["fixed_members"] == list(FIXED_MEMBERS),
        "범위의 고정 구성원이 다르다.",
    )
    require(payload["scope"]["pair_count"] == PAIR_COUNT, "범위의 짝 수가 다르다.")
    require(
        payload["search_parameters"]["strategy"] == SEARCH_STRATEGY,
        "검색 결합 방식이 다르다.",
    )
    require(
        payload["search_parameters"]["champion_floor_margin"] == CHAMPION_FLOOR_MARGIN,
        "진입 하한 여유 폭이 다르다.",
    )
    require(
        payload["search_parameters"]["champion"]
        == {
            "config": champion["config"],
            "run_id": champion["run_id"],
            "oof_auc": float(champion["oof_auc"]),
        },
        "검색 진입 하한의 champion 신원이 다르다.",
    )
    require(
        payload["search_parameters"]["missingness_augmented_entry_threshold"]
        == float(champion["oof_auc"]) - CHAMPION_FLOOR_MARGIN,
        "결측 증강판 검색 진입 하한이 다르다.",
    )
    require(
        payload["search_parameters"]["duplicate_spearman"] == DUPLICATE_SPEARMAN,
        "중복 문턱이 다르다.",
    )
    require(
        payload["search_parameters"]["adoption_strategies"]
        == list(ADOPTION_STRATEGIES),
        "채택 관문의 핵심 결합 방식이 다르다.",
    )
    require(
        payload["search_parameters"]["outer_folds"] == list(OUTER_FOLDS),
        "바깥 분할이 다르다.",
    )

    module_path = repo_root / payload["code"]["contract_module"]["path"]
    freeze_script_path = repo_root / payload["code"]["freeze_script"]["path"]
    if not allow_contract_module_correction:
        require(
            file_sha256(module_path) == payload["code"]["contract_module"]["sha256"],
            "판정 계약 모듈 해시가 다르다.",
        )
    require(
        file_sha256(freeze_script_path) == payload["code"]["freeze_script"]["sha256"],
        "사전 기록 생성기 해시가 다르다.",
    )


def validate_input_bundle(
    bundle: Mapping[str, Any],
    *,
    precommit: Mapping[str, Any],
    allowed_source_commits: Mapping[str, str] | None = None,
) -> None:
    """후속 중앙 반입 묶음의 완결성과 전부 포함 조건을 확인한다."""
    require(
        bundle.get("schema") == INPUT_BUNDLE_SCHEMA, "판정 입력 묶음 스키마가 다르다."
    )
    verify_self_hash(bundle, "input_bundle_sha256")
    require(
        bundle.get("precommit_sha256") == precommit["precommit_sha256"],
        "사전 기록 신원이 다르다.",
    )
    pair_order = precommit["scope"]["pair_member_order"]
    expected_pairs = {pair["slot"]: pair for pair in precommit["pairs"]}
    collection_contract = precommit["collection_contract"]
    cutoff = datetime.fromisoformat(collection_contract["central_import_cutoff_utc"])
    records = bundle["collection"]
    require(
        [record["member"] for record in records] == list(pair_order),
        "입력 수집 순서가 34개 짝 순서와 다르다.",
    )
    require(len(records) == PAIR_COUNT, "입력 수집 상태가 34개를 모두 덮지 않는다.")
    seen_run_ids: set[str] = set()
    source_commits = dict(allowed_source_commits or {})
    for record in records:
        status = record["status"]
        require(
            status in {"complete", "incomplete"},
            f"{record['member']}: 수집 상태가 잘못됐다.",
        )
        if status == "complete":
            expected_pair = expected_pairs[record["member"]]
            require(
                len(record.get("arms", [])) == 2,
                f"{record['member']}: 완결 짝의 두 팔이 없다.",
            )
            expected_arms = {
                arm["arm"]: arm for arm in expected_pair["comparison_arms"]
            }
            providers = {arm["provider"] for arm in record["arms"]}
            runtime_classes = {arm["runtime_class"] for arm in record["arms"]}
            container_images = {arm["container_image_digest"] for arm in record["arms"]}
            dependency_locks = {arm["dependency_lock_sha256"] for arm in record["arms"]}
            require(
                len(providers) == 1, f"{record['member']}: 두 팔의 공급자가 다르다."
            )
            require(
                len(runtime_classes) == 1,
                f"{record['member']}: 두 팔의 실행 환경 등급이 다르다.",
            )
            require(
                runtime_classes
                == {
                    expected_pair["missingness_augmented"]["identity"]["runtime_class"]
                },
                f"{record['member']}: 두 팔의 실행 환경 등급이 사전 기록과 다르다.",
            )
            require(
                len(container_images) == 1 and next(iter(container_images)),
                f"{record['member']}: 두 팔의 컨테이너 신원이 다르거나 비었다.",
            )
            require(
                len(dependency_locks) == 1 and next(iter(dependency_locks)),
                f"{record['member']}: 두 팔의 의존성 잠금 신원이 다르거나 비었다.",
            )
            require(
                {arm["arm"] for arm in record["arms"]}
                == {"tripled", "missingness_augmented"},
                f"{record['member']}: 완결 짝의 팔 구성이 다르다.",
            )
            for arm in record["arms"]:
                require(
                    arm["run_id"] not in seen_run_ids,
                    f"{record['member']}: 실행 신원이 다른 팔과 중복된다.",
                )
                seen_run_ids.add(arm["run_id"])
                require(
                    arm["integrity_verdict"] == "pass",
                    f"{record['member']}: 반입 무결성 관문을 통과하지 못했다.",
                )
                require(
                    arm["status"] == "FINISHED",
                    f"{record['member']}: 완료 실행이 아니다.",
                )
                expected_arm = expected_arms[arm["arm"]]
                require(
                    arm["experiment"] == expected_arm["name"],
                    f"{record['member']}: 실행 이름이 사전 기록과 다르다.",
                )
                require(
                    arm["config_sha256"] == expected_arm["sha256"],
                    f"{record['member']}: 실행 설정 해시가 사전 기록과 다르다.",
                )
                expected_source_commit = source_commits.get(
                    record["member"], collection_contract["execution_source_commit"]
                )
                require(
                    arm["git_commit"] == expected_source_commit,
                    f"{record['member']}: 실행 출처 커밋이 허용된 교정 계보와 다르다.",
                )
                require(
                    arm["git_dirty"] is False,
                    f"{record['member']}: 깨끗하지 않은 코드 상태에서 실행됐다.",
                )
                require(
                    arm["seeds"]
                    == list(precommit["pair_execution_contract"]["value"]["seeds"]),
                    f"{record['member']}: 세 시드가 사전 기록과 다르다.",
                )
                require(
                    arm["outer_folds"]
                    == list(precommit["search_parameters"]["outer_folds"]),
                    f"{record['member']}: 바깥 분할이 사전 기록과 다르다.",
                )
                require(
                    arm["input_sha256"]
                    == expected_pair["original"]["identity"]["input_sha256"],
                    f"{record['member']}: 자료 입력 해시가 원본 신원과 다르다.",
                )
                imported_at = datetime.fromisoformat(arm["central_imported_at_utc"])
                require(
                    imported_at <= cutoff,
                    f"{record['member']}: 중앙 반입 마감 뒤에 들어온 실행이다.",
                )
                require(
                    bool(arm["oof_sha256"]),
                    f"{record['member']}: OOF 신원 해시가 없다.",
                )
                require(
                    bool(arm["oof_prediction_sha256"]),
                    f"{record['member']}: OOF 예측 배열 신원 해시가 없다.",
                )
                require(
                    math.isfinite(float(arm["oof_auc"])),
                    f"{record['member']}: OOF 재채점값이 유한하지 않다.",
                )
                require(
                    bool(arm["required_diagnostics_sha256"]),
                    f"{record['member']}: 필수 진단 신원 해시가 없다.",
                )
                require(
                    bool(arm["execution_record_bundle_sha256"]),
                    f"{record['member']}: 실행 기록 묶음 해시가 없다.",
                )
            arms_by_name = {arm["arm"]: arm for arm in record["arms"]}
            direct_delta = float(
                arms_by_name["missingness_augmented"]["oof_auc"]
            ) - float(arms_by_name["tripled"]["oof_auc"])
            require(
                record["direct_oof_delta"] == direct_delta,
                f"{record['member']}: 직접 짝비교 진단값을 OOF에서 재계산할 수 없다.",
            )
        else:
            require(
                not record.get("arms"),
                f"{record['member']}: 미완결 짝에 판정 팔이 들어갔다.",
            )
            require(
                bool(record.get("reason")), f"{record['member']}: 미완결 사유가 없다."
            )
    complete_members = [
        record["member"] for record in records if record["status"] == "complete"
    ]
    require(
        bundle["complete_pair_members"] == complete_members,
        "완결 짝 목록이 수집 상태와 다르다.",
    )


def verify_search_mechanics() -> dict[str, Any]:
    """실제 예측을 읽지 않고 최대 상승, 동률과 두 개 묶음 규칙을 점검한다."""
    scores = {
        (): 0.0,
        (1,): 1.0,
        (2,): 2.0,
        (3,): -1.0,
        (1, 2): 2.5,
        (1, 3): 3.0,
        (2, 3): 2.0,
        (1, 2, 3): 3.5,
    }
    result = MaximumGainSearch(
        pair_order=(1, 2, 3),
        eligible=(1, 2, 3),
        score=scores.__getitem__,
        allowed=lambda _state: True,
    ).run()
    first = result.stages[0].accepted
    require(
        first is not None and first.incoming == (2,),
        "최대 상승 검색이 첫 양수 이동을 잘못 골랐다.",
    )

    tie_scores = {(): 0.0, (1,): 1.0, (2,): 1.0, (2, 1): 1.0}
    tied = MaximumGainSearch(
        pair_order=(2, 1),
        eligible=(1, 2),
        score=tie_scores.__getitem__,
        allowed=lambda _state: True,
    ).run()
    tie_first = tied.stages[0].accepted
    require(
        tie_first is not None and tie_first.incoming == (2,),
        "동률 해소가 동결 순서를 따르지 않는다.",
    )

    pair_scores = {
        (): 0.0,
        (1,): -1.0,
        (2,): -1.0,
        (1, 2): 1.0,
    }
    paired = MaximumGainSearch(
        pair_order=(1, 2),
        eligible=(1, 2),
        score=pair_scores.__getitem__,
        allowed=lambda _state: True,
    ).run()
    pair_stage = next(stage for stage in paired.stages if stage.name == "pair_sweep")
    require(
        pair_stage.accepted is not None and pair_stage.accepted.incoming == (1, 2),
        "단독 음수인 두 원자 교체 묶음을 전수 평가하지 않았다.",
    )

    batched_calls: list[tuple[tuple[int, ...], ...]] = []

    def score_many(
        states: tuple[tuple[int, ...], ...],
    ) -> tuple[float, ...]:
        batched_calls.append(states)
        return tuple(scores[state] for state in states)

    batched = MaximumGainSearch(
        pair_order=(1, 2, 3),
        eligible=(1, 2, 3),
        score=scores.__getitem__,
        score_many=score_many,
        allowed=lambda state: bool(state),
        allow_invalid_start=True,
    ).run()
    require(
        batched.selected == result.selected and any(len(call) > 1 for call in batched_calls),
        "기존 중복 위반 시작점의 일괄 점수 검색 결과가 단일 점수 검색과 다르다.",
    )
    return {
        "maximum_gain_first": list(first.incoming),
        "frozen_order_tie_first": list(tie_first.incoming),
        "pair_sweep_selected": list(pair_stage.accepted.incoming),
        "batched_invalid_start_selected": list(batched.selected),
        "batched_call_count": len(batched_calls),
    }
