# Vast.ai 독립 종료 안전장치

GitHub Actions의 `Vast.ai termination guard` 실행 흐름은 Vast.ai 계산 자원 밖에서 5분마다 만료된 고정 대상을 확인하고 삭제한다.
GitHub 예약 실행은 지연되거나 누락될 수 있으므로 이 안전장치는 절대 시각을 보장하지 않는다.

## 최초 설정

GitHub Environment `vast-termination`을 만든다.
Vast.ai에서 `instance_read`와 `instance_write` 권한만 가진 별도 키를 발급하고 Environment 비밀 값 `VAST_TERMINATION_API_KEY`로 저장한다.
키를 명령 인수, 구성 변수, 셸 기록, 실행 로그 또는 저장소 파일에 넣지 않는다.

저장소 구성 변수 `VAST_TERMINATION_SCHEDULES`는 다음 필드만 가진 JSON 배열이다.
프로젝트용 별도 저장 공간을 만들지 않은 작업은 `volume_id`를 `null`로 고정한다.

```json
[
  {
    "instance_id": 12345,
    "job_id": "screening-exp000-seed42",
    "terminate_at": "2026-08-15T05:00:00Z",
    "volume_id": 67890
  }
]
```

## 유료 작업 시작 전 확인

종료 예정 시각은 스크리닝 1시간 45분, 확정 재검증 7시간 45분 뒤로 등록한다.
구성 변수를 다시 읽어 작업 식별자, 계산 자원, 저장 공간과 종료 시각이 일치하는지 확인한다.
기본 브랜치에서 실행 흐름이 활성 상태인지 확인한다.
GitHub Actions 실행 목록에서 최근 10분 안에 `schedule`로 시작해 성공한 예약 감시와 실행 주소가 있는지 확인한다.
`workflow_dispatch`의 `observe` 방식과 등록한 작업 식별자로 즉시 실행하고 로그의 `schedule observed`를 확인한다.
하나라도 실패하면 유료 작업을 시작하지 않는다.

## 실패 복구

실패하면 안전장치는 같은 작업 식별자의 `ready-for-human` 경보 이슈를 만들거나 기존 이슈에 새 실행 기록을 추가하고 저장소 소유자에게 할당한다.
공식 명령줄 도구, 직접 REST API, Vast.ai 브라우저 긴급 삭제 순서로 복구한다.
인스턴스와 프로젝트용 저장 공간이 모두 목록에서 사라진 것을 확인하기 전에는 새 유료 작업을 시작하지 않는다.

실행 로그에서 대상 부재를 확인한 뒤 로컬 제어 경로로 완료한 일정을 구성 변수에서 제거한다.
같은 대상을 다시 실행하거나 대상이 이미 없는 경우도 성공으로 처리한다.
