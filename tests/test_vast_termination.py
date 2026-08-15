from datetime import UTC, datetime

import pytest

from remote_ops.vast_termination import (
    Schedule,
    TargetState,
    TerminationError,
    encode_registry,
    parse_registry,
    terminate_schedule,
)


class FakeVast:
    def __init__(self, states: list[TargetState]) -> None:
        self.states = states
        self.index = 0
        self.deleted_instances: list[int] = []
        self.deleted_volumes: list[int] = []

    def target_state(self, schedule: Schedule) -> TargetState:
        state = self.states[min(self.index, len(self.states) - 1)]
        self.index += 1
        return state

    def delete_instance(self, instance_id: int) -> None:
        self.deleted_instances.append(instance_id)

    def delete_volume(self, volume_id: int) -> None:
        self.deleted_volumes.append(volume_id)


def schedule() -> Schedule:
    return Schedule(
        job_id="acceptance-001",
        terminate_at=datetime(2026, 8, 15, 3, tzinfo=UTC),
        instance_id=123,
        volume_id=456,
    )


def test_registry_round_trip_is_canonical() -> None:
    value = [schedule()]

    assert parse_registry(encode_registry(value)) == value


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '[{"job_id":"bad id","terminate_at":"2026-08-15T03:00:00Z","instance_id":1,"volume_id":2}]',
        '[{"job_id":"ok","terminate_at":"2026-08-15T03:00:00+00:00","instance_id":1,"volume_id":2}]',
        '[{"job_id":"ok","terminate_at":"2026-08-15T03:00:00Z","instance_id":0,"volume_id":2}]',
        '[{"job_id":"ok","terminate_at":"2026-08-15T03:00:00Z","instance_id":1}]',
    ],
)
def test_registry_rejects_unsafe_shapes(raw: str) -> None:
    with pytest.raises(TerminationError):
        parse_registry(raw)


def test_absent_targets_are_idempotent_success() -> None:
    fake = FakeVast([TargetState(False, False)])

    result = terminate_schedule(fake, schedule())

    assert result.final_state.absent
    assert result.attempts == 0
    assert fake.deleted_instances == []
    assert fake.deleted_volumes == []


def test_present_targets_are_deleted_and_confirmed_absent() -> None:
    fake = FakeVast([TargetState(True, True), TargetState(False, False)])
    sleeps: list[float] = []

    result = terminate_schedule(fake, schedule(), sleep=sleeps.append)

    assert result.final_state.absent
    assert result.attempts == 1
    assert not result.used_final_rest_retry
    assert fake.deleted_instances == [123]
    assert fake.deleted_volumes == []
    assert sleeps == [10]


def test_attached_volume_is_deleted_after_instance_disappears() -> None:
    fake = FakeVast(
        [
            TargetState(True, True),
            TargetState(False, True),
            TargetState(False, False),
        ]
    )
    sleeps: list[float] = []

    result = terminate_schedule(fake, schedule(), sleep=sleeps.append)

    assert result.final_state.absent
    assert result.attempts == 2
    assert fake.deleted_instances == [123]
    assert fake.deleted_volumes == [456]
    assert sleeps == [10, 10]


def test_final_rest_retry_runs_after_poll_deadline() -> None:
    fake = FakeVast(
        [
            TargetState(True, True),
            TargetState(True, True),
            TargetState(False, True),
            TargetState(False, False),
        ]
    )
    times = iter([0.0, 0.0, 300.0])
    sleeps: list[float] = []

    result = terminate_schedule(
        fake,
        schedule(),
        poll_timeout_seconds=300,
        sleep=sleeps.append,
        monotonic=lambda: next(times),
    )

    assert result.final_state.absent
    assert result.used_final_rest_retry
    assert fake.deleted_instances == [123, 123]
    assert fake.deleted_volumes == [456]
    assert sleeps == [10, 10, 10]


def test_polling_cannot_be_faster_than_ten_seconds() -> None:
    with pytest.raises(TerminationError, match="at least 10"):
        terminate_schedule(FakeVast([TargetState(False, False)]), schedule(), poll_interval_seconds=9)
