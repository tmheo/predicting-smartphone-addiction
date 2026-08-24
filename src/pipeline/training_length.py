"""관측 학습 길이와 재학습 예산의 공통 계약. (#371)

세 값을 서로 다른 것으로 구분한다.

- 원시 선택값: 모델 연결부가 학습 엔진에서 그대로 읽은 수. 0부터 세는 위치일 수도,
  1부터 세는 횟수일 수도, 검증 없이 고정한 횟수일 수도 있다.
- 관측 학습 길이: 한 바깥쪽 분할에서 실제로 돈 반복 횟수. 항상 1 이상의 정수다.
- 재학습 예산: 전체 자료 재학습 하나에 넘길 시드별 최종 반복 횟수.

책임 경계는 다음과 같다. 원시 선택값의 뜻은 **모델 계열 연결부**가 안다.
이 모듈은 계열 이름도 `best_epoch` 같은 필드 이름도 해석하지 않고,
호출자가 명시한 원시 의미와 원시 값만 검증해 관측 학습 길이로 바꾼다.
그래서 새 계열이 들어와도 이 모듈은 바뀌지 않고, 어떤 계열이 `+1`을 받을지는
연결부가 선언한 원시 의미 하나로만 결정된다.

재학습 예산 산정은 `derive_refit_budgets`가 소유한다. 확정된 관측 학습 길이만 받아
시드별 중앙값, 배수, 사사오입을 적용하고 중간값을 전부 드러낸 파생 결과를 돌려준다.
이 모듈은 장부를 읽거나 파일을 저장하거나 실행 저장소를 조회하거나 모델을 돌리지 않는다.
근거 계보 검증과 실행 관문은 재학습 계획 장부가 소유한다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

# 원시 의미. 연결부가 자기 원시 선택값에 대해 셋 중 하나를 선언한다.
ZERO_BASED_POSITION = "zero_based_position"  # 0부터 세는 위치. 실제 횟수는 +1이다.
ONE_BASED_COUNT = "one_based_count"  # 학습 엔진이 이미 센 실제 반복 횟수.
FIXED_COUNT = "fixed_count"  # 검증으로 고른 값이 아니라 설정이 고정한 실제 횟수.
RAW_MEANINGS = (ZERO_BASED_POSITION, ONE_BASED_COUNT, FIXED_COUNT)

MEDIAN_STATISTIC = "median"
HALF_UP_ROUNDING = "half_up"


class TrainingLengthError(ValueError):
    """원시 의미 변환이나 재학습 예산 산정의 계약을 어겼다."""


@dataclass(frozen=True)
class ObservedTrainingLength:
    """한 좌표에서 확정한 관측 학습 길이와 그 원시 근거.

    이 값이 존재한다는 것 자체가 `observe_training_length`의 검증을 통과했다는 뜻이다.
    좌표는 시드와 바깥쪽 분할이며, 내부 구성원이 있는 계열만 `inner_member`를 채운다.
    """

    seed: int
    outer_fold: int
    raw_field: str
    raw_value: int
    raw_meaning: str
    value: int
    inner_member: int | None = None

    def __post_init__(self) -> None:
        if self.raw_meaning not in RAW_MEANINGS:
            raise TrainingLengthError(f"알 수 없는 원시 의미다: {self.raw_meaning!r}")
        if _require_int(self.value, "관측 학습 길이") < 1:
            raise TrainingLengthError(
                f"관측 학습 길이는 양의 정수여야 한다: {self.value!r}"
            )


@dataclass(frozen=True)
class RefitBudgetPolicy:
    """재학습 예산 계산 규약. 기본값이 ADR 0002가 확정한 규약이다."""

    statistic: str = MEDIAN_STATISTIC
    multiplier: float = 1.25
    rounding: str = HALF_UP_ROUNDING

    def __post_init__(self) -> None:
        if self.statistic != MEDIAN_STATISTIC:
            raise TrainingLengthError(
                f"통계량은 {MEDIAN_STATISTIC!r}만 지원한다: {self.statistic!r}"
            )
        if self.rounding != HALF_UP_ROUNDING:
            raise TrainingLengthError(
                f"사사오입 방식은 {HALF_UP_ROUNDING!r}만 지원한다: {self.rounding!r}"
            )
        if isinstance(self.multiplier, bool) or not isinstance(
            self.multiplier, (int, float)
        ):
            raise TrainingLengthError(f"배수는 수여야 한다: {self.multiplier!r}")
        if not math.isfinite(self.multiplier) or self.multiplier <= 0:
            raise TrainingLengthError(f"배수는 양의 유한값이어야 한다: {self.multiplier!r}")


@dataclass(frozen=True)
class SeedBudgetDerivation:
    """시드 하나의 재학습 예산과 그 계산 중간값 전부."""

    seed: int
    observed_lengths: tuple[int, ...]
    median: float
    scaled: float
    budget: int


@dataclass(frozen=True)
class RefitBudgetDerivation:
    """구성원 하나의 시드별 재학습 예산 파생 결과."""

    policy: RefitBudgetPolicy
    seeds: tuple[SeedBudgetDerivation, ...]

    def budgets(self) -> dict[int, int]:
        """시드 -> 재학습 예산. 실행 경로가 읽는 최종 숫자다."""
        return {seed.seed: seed.budget for seed in self.seeds}


def _require_int(value: object, label: str) -> int:
    """불리언과 정수가 아닌 값을 거부한다.

    `bool`은 `int`의 하위형이라 `isinstance` 하나로는 걸러지지 않는다.
    `True`가 위치 `1`로 조용히 통과하는 경로를 막는다.
    """
    if isinstance(value, bool):
        raise TrainingLengthError(f"{label}는 불리언일 수 없다: {value!r}")
    if not isinstance(value, int):
        raise TrainingLengthError(f"{label}는 정수여야 한다: {value!r}")
    return value


def observed_length_from_raw(raw_value: object, raw_meaning: str) -> int:
    """원시 의미가 정한 대로 원시 값 하나를 관측 학습 길이로 바꾼다.

    0부터 세는 위치만 `+1`한다. 1부터 세는 횟수와 고정 횟수는 값이 바뀌지 않는다.
    모델 계열이나 원시 필드 이름은 보지 않는다.
    """
    if isinstance(raw_value, ObservedTrainingLength):
        raise TrainingLengthError(
            "이미 관측 학습 길이로 바꾼 값은 다시 변환할 수 없다: "
            f"{raw_value.raw_field}={raw_value.raw_value} -> {raw_value.value}"
        )
    if raw_meaning not in RAW_MEANINGS:
        raise TrainingLengthError(
            f"알 수 없는 원시 의미다: {raw_meaning!r} (가능한 값: {list(RAW_MEANINGS)})"
        )

    value = _require_int(raw_value, "원시 값")
    if raw_meaning == ZERO_BASED_POSITION:
        if value < 0:
            raise TrainingLengthError(f"0부터 세는 위치는 음수일 수 없다: {value}")
        return value + 1
    if value < 1:
        raise TrainingLengthError(f"횟수는 1 이상이어야 한다: {value}")
    return value


def observe_training_length(
    *,
    seed: int,
    outer_fold: int,
    raw_field: str,
    raw_value: object,
    raw_meaning: str,
    inner_member: int | None = None,
) -> ObservedTrainingLength:
    """연결부가 선언한 원시 근거 하나를 검증된 관측 학습 길이로 만든다."""
    if not raw_field:
        raise TrainingLengthError("원시 필드 이름이 비었다")
    seed = _require_int(seed, "시드")
    outer_fold = _require_int(outer_fold, "바깥쪽 분할")
    if outer_fold < 0:
        raise TrainingLengthError(f"바깥쪽 분할은 0 이상이어야 한다: {outer_fold}")
    if inner_member is not None:
        inner_member = _require_int(inner_member, "내부 구성원")
        if inner_member < 0:
            raise TrainingLengthError(f"내부 구성원은 0 이상이어야 한다: {inner_member}")

    value = observed_length_from_raw(raw_value, raw_meaning)
    return ObservedTrainingLength(
        seed=seed,
        outer_fold=outer_fold,
        raw_field=raw_field,
        raw_value=_require_int(raw_value, "원시 값"),
        raw_meaning=raw_meaning,
        value=value,
        inner_member=inner_member,
    )


def _seed_groups(
    evidence: Iterable[ObservedTrainingLength],
) -> list[tuple[int, tuple[int, ...]]]:
    """근거를 시드 -> 관측 학습 길이 목록으로 모은다. 시드 순서는 처음 등장 순서다.

    검증을 통과한 `ObservedTrainingLength`만 받는다. 원시 값이나 맨 정수를 직접 받으면
    변환하지 않은 위치가 예산 계산에 그대로 들어갈 수 있어 그 입구를 열지 않는다.
    """
    ordered: dict[int, list[int]] = {}
    for observation in evidence:
        if not isinstance(observation, ObservedTrainingLength):
            raise TrainingLengthError(
                f"근거는 검증된 관측 학습 길이여야 한다: {observation!r}"
            )
        ordered.setdefault(observation.seed, []).append(observation.value)
    return [(seed, tuple(values)) for seed, values in ordered.items()]


def derive_refit_budgets(
    evidence: Iterable[ObservedTrainingLength],
    policy: RefitBudgetPolicy | None = None,
) -> RefitBudgetDerivation:
    """확정된 관측 학습 길이에서 시드별 재학습 예산을 계산한다.

    시드마다 관측 학습 길이의 중앙값을 구하고 배수를 곱한 뒤 양수 사사오입한다.
    입력 관측값, 중앙값, 배수 적용값과 최종 정수를 모두 담은 파생 결과를 돌려준다.
    """
    policy = RefitBudgetPolicy() if policy is None else policy
    groups = _seed_groups(evidence)
    if not groups:
        raise TrainingLengthError("관측 학습 길이가 하나도 없다")

    derivations: list[SeedBudgetDerivation] = []
    for seed, lengths in groups:
        seed_median = float(median(lengths))
        scaled = seed_median * policy.multiplier
        derivations.append(
            SeedBudgetDerivation(
                seed=seed,
                observed_lengths=lengths,
                median=seed_median,
                scaled=scaled,
                budget=round_half_up(scaled),
            )
        )
    return RefitBudgetDerivation(policy=policy, seeds=tuple(derivations))


def round_half_up(value: float) -> int:
    """양수 `value`를 `floor(value + 0.5)`로 사사오입한다.

    파이썬 기본 `round`는 짝수로 붙이므로 `12.5`가 `12`가 된다.
    재학습 예산은 위로 붙는 사사오입이 규약이라 이 함수를 쓴다.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingLengthError(f"사사오입 대상은 수여야 한다: {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise TrainingLengthError(f"사사오입은 양의 유한값에만 정의한다: {value!r}")
    return math.floor(value + 0.5)
