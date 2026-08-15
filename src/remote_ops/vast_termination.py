"""GitHub Actions에서 실행하는 Vast.ai 독립 종료 안전장치."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REGISTRY_VARIABLE = "VAST_TERMINATION_SCHEDULES"
ALERT_LABEL = "ready-for-human"
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TRANSIENT_STATUS = {429, 502, 503, 504}


class TerminationError(RuntimeError):
    """종료 안전장치가 완료되지 못했음을 나타낸다."""


@dataclass(frozen=True)
class Schedule:
    job_id: str
    terminate_at: datetime
    instance_id: int
    volume_id: int | None

    def public_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "terminate_at": self.terminate_at.isoformat().replace("+00:00", "Z"),
            "instance_id": self.instance_id,
            "volume_id": self.volume_id,
        }


@dataclass(frozen=True)
class TargetState:
    instance_present: bool
    volume_present: bool

    @property
    def absent(self) -> bool:
        return not self.instance_present and not self.volume_present


@dataclass(frozen=True)
class TerminationResult:
    attempts: int
    final_state: TargetState
    used_final_rest_retry: bool


class VastControl(Protocol):
    def target_state(self, schedule: Schedule) -> TargetState: ...

    def delete_instance(self, instance_id: int) -> None: ...

    def delete_volume(self, volume_id: int) -> None: ...


def _positive_int(value: object, field: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TerminationError(f"{field} must be a positive integer")
    return value


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise TerminationError("terminate_at must be an ISO 8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise TerminationError("terminate_at is not a valid timestamp") from error
    if parsed.tzinfo != UTC:
        raise TerminationError("terminate_at must use UTC")
    return parsed


def parse_registry(raw: str) -> list[Schedule]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise TerminationError("schedule registry is not valid JSON") from error
    if not isinstance(value, list):
        raise TerminationError("schedule registry must be a JSON array")

    schedules: list[Schedule] = []
    seen: set[str] = set()
    expected = {"job_id", "terminate_at", "instance_id", "volume_id"}
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != expected:
            raise TerminationError(f"schedule at index {index} has an invalid shape")
        job_id = item["job_id"]
        if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
            raise TerminationError(f"schedule at index {index} has an invalid job_id")
        if job_id in seen:
            raise TerminationError(f"duplicate job_id: {job_id}")
        seen.add(job_id)
        schedules.append(
            Schedule(
                job_id=job_id,
                terminate_at=_parse_utc(item["terminate_at"]),
                instance_id=int(_positive_int(item["instance_id"], "instance_id")),
                volume_id=_positive_int(item["volume_id"], "volume_id", nullable=True),
            )
        )
    return schedules


def encode_registry(schedules: list[Schedule]) -> str:
    return json.dumps(
        [schedule.public_dict() for schedule in schedules],
        separators=(",", ":"),
        sort_keys=True,
    )


class JsonApi:
    def __init__(self, base_url: str, token: str, user_agent: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.user_agent = user_agent

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, object] | None = None,
        payload: dict[str, object] | list[object] | None = None,
        retry: int = 3,
        allow_not_found: bool = False,
    ) -> object | None:
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)
        data = None if payload is None else json.dumps(payload).encode()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        delay = 1.0
        for attempt in range(retry):
            try:
                with urlopen(Request(url, data=data, headers=headers, method=method), timeout=30) as response:
                    body = response.read()
                    return json.loads(body) if body else None
            except HTTPError as error:
                error.read()
                if allow_not_found and error.code == 404:
                    return None
                if error.code not in TRANSIENT_STATUS or attempt + 1 == retry:
                    raise TerminationError(f"API request failed with HTTP {error.code}") from error
            except (URLError, TimeoutError) as error:
                if attempt + 1 == retry:
                    raise TerminationError("API request failed due to a transport error") from error
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")


class VastApi:
    def __init__(self, token: str) -> None:
        if not token:
            raise TerminationError("VAST_TERMINATION_API_KEY is missing")
        self.api = JsonApi("https://console.vast.ai", token, "s6e8-vast-termination/1")

    def _instance_ids(self) -> set[int]:
        result: set[int] = set()
        after_token: str | None = None
        while True:
            params: dict[str, object] = {
                "limit": 25,
                "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
                "select_filters": json.dumps({}),
            }
            if after_token:
                params["after_token"] = after_token
            response = self.api.request("GET", "/api/v1/instances/", query=params)
            if not isinstance(response, dict) or not isinstance(response.get("instances"), list):
                raise TerminationError("Vast.ai returned an invalid instance list")
            for item in response["instances"]:
                identifier = _response_id(item)
                if identifier is not None:
                    result.add(identifier)
            token = response.get("next_token")
            if not token:
                return result
            if not isinstance(token, str):
                raise TerminationError("Vast.ai returned an invalid pagination token")
            after_token = token

    def _volume_ids(self) -> set[int]:
        response = self.api.request(
            "GET",
            "/api/v0/volumes",
            query={"owner": "me", "type": "all_volume"},
        )
        if not isinstance(response, dict) or not isinstance(response.get("volumes"), list):
            raise TerminationError("Vast.ai returned an invalid volume list")
        return {
            identifier
            for item in response["volumes"]
            if (identifier := _response_id(item)) is not None
        }

    def target_state(self, schedule: Schedule) -> TargetState:
        instance_present = schedule.instance_id in self._instance_ids()
        volume_present = schedule.volume_id is not None and schedule.volume_id in self._volume_ids()
        return TargetState(instance_present, volume_present)

    def delete_instance(self, instance_id: int) -> None:
        self.api.request(
            "DELETE",
            f"/api/v0/instances/{instance_id}/",
            payload={},
            allow_not_found=True,
        )

    def delete_volume(self, volume_id: int) -> None:
        self.api.request(
            "DELETE",
            "/api/v0/volumes/",
            query={"id": volume_id},
            payload={},
            allow_not_found=True,
        )


def _response_id(item: object) -> int | None:
    if not isinstance(item, dict):
        return None
    value = item.get("id")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _try_delete_instance(client: VastControl, instance_id: int) -> None:
    try:
        client.delete_instance(instance_id)
    except TerminationError:
        pass


def _try_delete_volume(client: VastControl, volume_id: int) -> None:
    try:
        client.delete_volume(volume_id)
    except TerminationError:
        pass


def terminate_schedule(
    client: VastControl,
    schedule: Schedule,
    *,
    poll_interval_seconds: float = 10,
    poll_timeout_seconds: float = 300,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> TerminationResult:
    if poll_interval_seconds < 10:
        raise TerminationError("poll interval must be at least 10 seconds")
    if poll_timeout_seconds > 300:
        raise TerminationError("poll timeout must not exceed 300 seconds")

    state = client.target_state(schedule)
    if state.absent:
        return TerminationResult(0, state, False)

    attempts = 0
    volume_delete_requested = False
    if state.instance_present:
        attempts += 1
        _try_delete_instance(client, schedule.instance_id)
    elif state.volume_present and schedule.volume_id is not None:
        attempts += 1
        volume_delete_requested = True
        _try_delete_volume(client, schedule.volume_id)

    deadline = monotonic() + poll_timeout_seconds
    while deadline - monotonic() >= poll_interval_seconds:
        sleep(poll_interval_seconds)
        state = client.target_state(schedule)
        if state.absent:
            return TerminationResult(attempts, state, False)
        if not state.instance_present and state.volume_present and not volume_delete_requested:
            if schedule.volume_id is None:
                raise AssertionError("volume presence requires a fixed volume id")
            attempts += 1
            volume_delete_requested = True
            _try_delete_volume(client, schedule.volume_id)

    if state.instance_present:
        attempts += 1
        _try_delete_instance(client, schedule.instance_id)
    elif state.volume_present and schedule.volume_id is not None:
        attempts += 1
        _try_delete_volume(client, schedule.volume_id)
    sleep(poll_interval_seconds)
    state = client.target_state(schedule)
    if not state.instance_present and state.volume_present and schedule.volume_id is not None:
        attempts += 1
        _try_delete_volume(client, schedule.volume_id)
        sleep(poll_interval_seconds)
        state = client.target_state(schedule)
    return TerminationResult(attempts, state, True)


class GitHubApi:
    def __init__(self, token: str, repository: str, assignee: str) -> None:
        if not token or not repository or not assignee:
            raise TerminationError("GitHub runtime identity is incomplete")
        self.repository = repository
        self.assignee = assignee
        self.api = JsonApi("https://api.github.com", token, "s6e8-vast-termination/1")

    def alert(
        self,
        schedule: Schedule,
        reason: str,
        run_url: str,
        *,
        started_at: str,
        attempts: int | None,
        final_state: TargetState | None,
    ) -> None:
        marker = f"<!-- vast-termination-alert:{schedule.job_id} -->"
        title = f"Vast.ai 종료 경보: {schedule.job_id}"
        body = "\n".join(
            [
                marker,
                "## 필요한 조치",
                "",
                "Vast.ai 독립 종료 안전장치가 대상 부재를 확인하지 못했습니다.",
                "공식 명령줄 도구, 직접 REST API, 브라우저 긴급 삭제 순서로 복구하고 모든 과금 자원의 부재를 확인해야 합니다.",
                "",
                f"- 원격 실행 작업: `{schedule.job_id}`",
                f"- 인스턴스 식별자: `{schedule.instance_id}`",
                f"- 저장 공간 식별자: `{schedule.volume_id}`",
                f"- 종료 예정 시각: `{schedule.terminate_at.isoformat().replace('+00:00', 'Z')}`",
                f"- 실제 시작 시각: `{started_at}`",
                f"- 삭제 시도 횟수: `{attempts if attempts is not None else 'unknown'}`",
                (
                    "- 최종 인스턴스 존재 여부: "
                    f"`{final_state.instance_present if final_state is not None else 'unknown'}`"
                ),
                (
                    "- 최종 저장 공간 존재 여부: "
                    f"`{final_state.volume_present if final_state is not None else 'unknown'}`"
                ),
                f"- 실패 분류: `{reason}`",
                f"- 실행 기록: {run_url}",
            ]
        )
        response = self.api.request(
            "GET",
            f"/repos/{self.repository}/issues",
            query={"state": "all", "labels": ALERT_LABEL, "per_page": 100},
        )
        issues = response if isinstance(response, list) else []
        existing = next(
            (
                issue
                for issue in issues
                if isinstance(issue, dict) and marker in str(issue.get("body", ""))
            ),
            None,
        )
        if existing is None:
            self.api.request(
                "POST",
                f"/repos/{self.repository}/issues",
                payload={
                    "title": title,
                    "body": body,
                    "labels": [ALERT_LABEL],
                    "assignees": [self.assignee],
                },
            )
            return
        number = existing.get("number")
        if not isinstance(number, int):
            raise TerminationError("existing alert issue has no number")
        self.api.request(
            "PATCH",
            f"/repos/{self.repository}/issues/{number}",
            payload={"state": "open", "assignees": [self.assignee]},
        )
        self.api.request(
            "POST",
            f"/repos/{self.repository}/issues/{number}/comments",
            payload={"body": body},
        )


def _run_url() -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "unknown/unknown")
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    return f"{server}/{repository}/actions/runs/{run_id}"


def _github_from_environment() -> GitHubApi:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    return GitHubApi(
        os.environ.get("GITHUB_TOKEN", ""),
        repository,
        os.environ.get("ALERT_ASSIGNEE", repository.partition("/")[0]),
    )


def _observe(schedules: list[Schedule], expected_job_id: str) -> None:
    if not expected_job_id:
        raise TerminationError("expected_job_id is required in observe mode")
    schedule = next((item for item in schedules if item.job_id == expected_job_id), None)
    if schedule is None:
        raise TerminationError("expected schedule was not found")
    print(
        "schedule observed",
        f"job_id={schedule.job_id}",
        f"instance_id={schedule.instance_id}",
        f"volume_id={schedule.volume_id}",
        f"terminate_at={schedule.terminate_at.isoformat().replace('+00:00', 'Z')}",
    )


def run(mode: str, expected_job_id: str = "") -> int:
    github = _github_from_environment()
    schedules = parse_registry(os.environ.get(REGISTRY_VARIABLE, ""))
    run_url = _run_url()

    if mode == "observe":
        _observe(schedules, expected_job_id)
        return 0

    if mode == "force-alert":
        schedule = next((item for item in schedules if item.job_id == expected_job_id), None)
        if schedule is None:
            raise TerminationError("expected schedule was not found")
        github.alert(
            schedule,
            "intentional-acceptance-test",
            run_url,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            attempts=0,
            final_state=None,
        )
        raise TerminationError("intentional acceptance-test failure")

    if mode != "process":
        raise TerminationError(f"unsupported mode: {mode}")

    now = datetime.now(UTC)
    due = [schedule for schedule in schedules if schedule.terminate_at <= now]
    if not due:
        print(f"no due schedules registry_count={len(schedules)}")
        return 0

    try:
        vast = VastApi(os.environ.get("VAST_TERMINATION_API_KEY", ""))
    except TerminationError as error:
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        for schedule in due:
            github.alert(
                schedule,
                str(error),
                run_url,
                started_at=started_at,
                attempts=0,
                final_state=None,
            )
        return 1

    completed: set[Schedule] = set()
    failed = False
    for schedule in due:
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        result: TerminationResult | None = None
        try:
            result = terminate_schedule(vast, schedule)
            print(
                "termination checked",
                f"job_id={schedule.job_id}",
                f"started_at={started_at}",
                f"attempts={result.attempts}",
                f"final_rest_retry={str(result.used_final_rest_retry).lower()}",
                f"instance_present={str(result.final_state.instance_present).lower()}",
                f"volume_present={str(result.final_state.volume_present).lower()}",
            )
            if not result.final_state.absent:
                raise TerminationError("targets remain after the final REST retry")
            completed.add(schedule)
        except TerminationError as error:
            failed = True
            print(f"termination failed job_id={schedule.job_id} reason={error}", file=sys.stderr)
            github.alert(
                schedule,
                str(error),
                run_url,
                started_at=started_at,
                attempts=result.attempts if result is not None else None,
                final_state=result.final_state if result is not None else None,
            )

    if completed:
        completed_ids = ",".join(sorted(schedule.job_id for schedule in completed))
        print(f"completed schedules require local registry removal job_ids={completed_ids}")
    if failed:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["observe", "process", "force-alert"], required=True)
    parser.add_argument("--expected-job-id", default="")
    args = parser.parse_args()
    try:
        return run(args.mode, args.expected_job_id)
    except TerminationError as error:
        print(f"termination workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
