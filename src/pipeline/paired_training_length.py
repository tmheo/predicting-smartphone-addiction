"""결측 증강 짝비교가 출처 실행의 학습 노출량을 그대로 적용하는 계약."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .config import PairedTrainingLengthConfig
from .data import file_sha256


SCHEMA_VERSION = 1
CONTRACT = "paired-training-length-v1"


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
