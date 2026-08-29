"""결측 증강 짝비교가 출처 실행의 학습 노출량을 그대로 적용하는 계약."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .config import PairedTrainingLengthConfig
from .data import file_sha256


SCHEMA_VERSION = 1
CONTRACT = "paired-training-length-v1"


_NEURAL_BATCH_SIZE_DEFAULTS = {
    "lookup_transformer": 2048,
    "contextualized_spline_transformer": 4096,
    "tabm": 128,
    "realmlp": 256,
}


@dataclass(frozen=True)
class PairedOptimizerStepPlan:
    """복제 행에서도 출처와 같은 epoch별 최적화 갱신 수를 만드는 계획."""

    model_kind: str
    row_multiplier: int
    original_row_count: int
    training_row_count: int
    source_batch_size: int
    paired_batch_size: int
    source_steps_per_epoch: int
    paired_steps_per_epoch: int

    def apply(self, params: dict) -> dict:
        adjusted = dict(params)
        adjusted["batch_size"] = self.paired_batch_size
        return adjusted

    def evidence(self) -> dict[str, object]:
        return {
            "contract": "paired-optimizer-step-v1",
            "model_kind": self.model_kind,
            "row_multiplier": self.row_multiplier,
            "original_row_count": self.original_row_count,
            "training_row_count": self.training_row_count,
            "source_batch_size": self.source_batch_size,
            "paired_batch_size": self.paired_batch_size,
            "source_steps_per_epoch": self.source_steps_per_epoch,
            "paired_steps_per_epoch": self.paired_steps_per_epoch,
            "optimizer_steps_per_epoch_preserved": True,
        }


def build_optimizer_step_plan(
    model_kind: str,
    params: dict,
    *,
    original_row_count: int,
    training_row_count: int,
) -> PairedOptimizerStepPlan | None:
    """신경망 복제 행의 물리 배치를 늘려 epoch별 갱신 수를 보존한다."""
    default_batch_size = _NEURAL_BATCH_SIZE_DEFAULTS.get(model_kind)
    if default_batch_size is None:
        return None
    if (
        isinstance(original_row_count, bool)
        or isinstance(training_row_count, bool)
        or not isinstance(original_row_count, int)
        or not isinstance(training_row_count, int)
        or original_row_count < 1
        or training_row_count < 1
    ):
        raise ValueError("짝비교 학습 행 수는 양의 정수여야 한다.")
    row_multiplier, remainder = divmod(training_row_count, original_row_count)
    if remainder or row_multiplier < 1:
        raise ValueError(
            "신경망 짝비교 학습 행 수는 원본 학습 행 수의 양의 정수배여야 한다."
        )
    source_batch_size = params.get("batch_size", default_batch_size)
    if (
        isinstance(source_batch_size, bool)
        or not isinstance(source_batch_size, int)
        or source_batch_size < 1
    ):
        raise ValueError(f"{model_kind}: batch_size는 양의 정수여야 한다.")
    paired_batch_size = source_batch_size * row_multiplier
    source_steps = math.ceil(original_row_count / source_batch_size)
    paired_steps = math.ceil(training_row_count / paired_batch_size)
    if source_steps != paired_steps:
        raise AssertionError(
            f"{model_kind}: 복제 행의 epoch별 최적화 갱신 수가 보존되지 않았다: "
            f"{source_steps} != {paired_steps}"
        )
    return PairedOptimizerStepPlan(
        model_kind=model_kind,
        row_multiplier=row_multiplier,
        original_row_count=original_row_count,
        training_row_count=training_row_count,
        source_batch_size=source_batch_size,
        paired_batch_size=paired_batch_size,
        source_steps_per_epoch=source_steps,
        paired_steps_per_epoch=paired_steps,
    )


@dataclass(frozen=True)
class PairedTrainingLengths:
    """후보 하나의 시드, 바깥쪽 분할, 내부 구성원별 관측 학습 길이."""

    member: str
    model_kind: str
    source_identity: dict[str, object]
    coordinates: dict[tuple[int, int], tuple[int, ...] | None]
    source_sha256: str

    def for_coordinate(self, seed: int, outer_fold: int) -> tuple[int, ...] | None:
        key = (int(seed), int(outer_fold))
        if key not in self.coordinates:
            raise ValueError(
                f"짝비교 학습 길이에서 좌표를 찾지 못했다: "
                f"member={self.member!r}, seed={seed}, outer_fold={outer_fold}"
            )
        return self.coordinates[key]

    def evidence(self, seed: int, outer_fold: int) -> dict[str, object]:
        lengths = self.for_coordinate(seed, outer_fold)
        return {
            "contract": CONTRACT,
            "member": self.member,
            "model_kind": self.model_kind,
            "seed": int(seed),
            "outer_fold": int(outer_fold),
            "observed_training_lengths": None if lengths is None else list(lengths),
            "source_identity": dict(self.source_identity),
            "source_sha256": self.source_sha256,
        }


def load(config: PairedTrainingLengthConfig | None) -> PairedTrainingLengths | None:
    """해시로 고정한 증거 파일에서 후보 하나의 학습 길이를 읽고 완전성을 검증한다."""
    if config is None:
        return None
    actual_sha256 = file_sha256(config.source)
    if actual_sha256 != config.sha256:
        raise ValueError(
            f"짝비교 학습 길이 증거 해시가 다르다: {config.source}\n"
            f"기대 {config.sha256}\n실제 {actual_sha256}"
        )
    with config.source.open() as stream:
        raw = json.load(stream)
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("contract") != CONTRACT:
        raise ValueError(
            f"지원하지 않는 짝비교 학습 길이 증거 형식이다: {config.source}"
        )
    members = raw.get("members")
    if not isinstance(members, list):
        raise ValueError("짝비교 학습 길이 증거의 members가 목록이 아니다.")
    matches = [item for item in members if item.get("member") == config.member]
    if len(matches) != 1:
        raise ValueError(
            f"짝비교 학습 길이 증거의 후보 신원이 유일하지 않다: {config.member!r}"
        )
    item = matches[0]
    model_kind = item.get("model_kind")
    source_identity = item.get("source_identity")
    observations = item.get("observations")
    status = item.get("status")
    if not isinstance(model_kind, str) or not isinstance(source_identity, dict):
        raise ValueError(f"{config.member}: 학습 길이 출처 신원이 잘못됐다.")
    seeds = raw.get("seeds")
    outer_folds = raw.get("outer_folds")
    if not isinstance(seeds, list) or not isinstance(outer_folds, list):
        raise ValueError("짝비교 학습 길이 증거의 좌표 축이 목록이 아니다.")
    expected = {(int(seed), int(fold)) for seed in seeds for fold in outer_folds}
    coordinates: dict[tuple[int, int], tuple[int, ...] | None] = {}
    if status == "not_applicable":
        if model_kind != "logistic_onehot" or observations not in ([], None):
            raise ValueError(
                f"{config.member}: 학습 길이 비적용은 logistic_onehot에만 허용된다."
            )
        coordinates = {key: None for key in expected}
    elif status == "confirmed":
        if not isinstance(observations, list):
            raise ValueError(f"{config.member}: 확정 학습 길이 관측 목록이 없다.")
        grouped: dict[tuple[int, int], dict[int, int]] = {}
        for observation in observations:
            try:
                key = (int(observation["seed"]), int(observation["outer_fold"]))
                inner_member = int(observation["inner_member"])
                length = int(observation["observed_training_length"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{config.member}: 학습 길이 관측값이 잘못됐다.") from exc
            if key not in expected or inner_member < 0 or length < 1:
                raise ValueError(f"{config.member}: 학습 길이 관측 좌표가 범위를 벗어났다.")
            by_member = grouped.setdefault(key, {})
            if inner_member in by_member:
                raise ValueError(f"{config.member}: 학습 길이 관측 좌표가 중복됐다.")
            by_member[inner_member] = length
        if set(grouped) != expected:
            missing = sorted(expected - set(grouped))
            extra = sorted(set(grouped) - expected)
            raise ValueError(
                f"{config.member}: 학습 길이 좌표가 완전하지 않다. "
                f"누락={missing}, 초과={extra}"
            )
        inner_counts: set[int] = set()
        for key, values in grouped.items():
            indices = sorted(values)
            if indices != list(range(len(indices))):
                raise ValueError(
                    f"{config.member}: 내부 구성원 번호가 0부터 연속되지 않는다: {key}"
                )
            inner_counts.add(len(indices))
            coordinates[key] = tuple(values[index] for index in indices)
        if len(inner_counts) != 1:
            raise ValueError(f"{config.member}: 좌표마다 내부 구성원 수가 다르다.")
    else:
        raise ValueError(f"{config.member}: 알 수 없는 학습 길이 상태 {status!r}")
    return PairedTrainingLengths(
        member=config.member,
        model_kind=model_kind,
        source_identity=source_identity,
        coordinates=coordinates,
        source_sha256=actual_sha256,
    )
