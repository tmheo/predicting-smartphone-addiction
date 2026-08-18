# Vast.ai API 키 기반 원격 자원 제어 기능과 한계

조사 기준일은 2026년 8월 15일이다.
이 문서는 Vast.ai 공식 문서와 공식 `vast-ai/vast-cli` 저장소의 v1.5.4 소스만 근거로 삼았다.
실제 계정과 API 키를 사용한 호출은 하지 않았으므로, 계정별 정책이나 문서에 없는 동작은 검증 범위에 포함하지 않는다.

## 결론

Vast.ai의 공식 명령줄 도구와 API 키만으로 매물 검색, 조건 필터링, 인스턴스 생성, 상태 확인, SSH 접속 정보 조회, 자료 전송, 비용 조회, 정지, 삭제, 별도 저장 공간 삭제까지 브라우저 없이 처리할 수 있다.
Vast.ai는 공식 API 입문 문서에서 REST API만으로 인증, 검색, 임대, 접속, 정리까지의 전체 인스턴스 생명주기를 설명하고, 공식 명령줄 도구를 같은 API 위에 구축된 기본 사용자 경로로 권한다([API 소개](https://docs.vast.ai/api-reference/introduction), [API 전체 절차](https://docs.vast.ai/api-reference/hello-world)).

평상시 원격 자원 제어 경로는 공식 `vastai` 명령줄 도구 v1.5.4 이상과 `--raw` 출력을 기본으로 정하는 것이 적합하다.
공식 문서도 대부분의 사용자에게 인증, 재시도, 요청 구조를 처리하는 명령줄 도구나 SDK를 권하고, 직접 REST API 호출은 다른 언어와의 연동이나 더 세밀한 제어가 필요할 때 사용하라고 안내한다([API 소개](https://docs.vast.ai/api-reference/introduction), [공식 명령줄 도구 저장소](https://github.com/vast-ai/vast-cli/tree/v1.5.4)).

브라우저는 최초 계정과 결제 수단 설정 및 첫 API 키 발급에만 필요하며, 이후 API 키 생성, 조회, 교체, 폐기도 권한이 있는 기존 키를 사용해 명령줄 도구 또는 API로 처리할 수 있다([API 키 관리](https://docs.vast.ai/guides/reference/api-keys), [API 키 생성](https://docs.vast.ai/api-reference/accounts/create-api-key)).
평상시 키에는 `misc`, `instance_read`, `instance_write`, `user_read`, `billing_read`만 부여하고, 키와 SSH 공개 키를 새로 관리하는 `user_write`, 결제 수단을 움직이는 `billing_write`, 공급자용 `machine_write`, 팀 관리 권한은 제외하는 구성이 적합하다([권한 범주와 API 대응표](https://docs.vast.ai/api-reference/permissions)).

중요한 한계는 독립적인 강제 삭제 예약이다.
공식 명령줄 도구의 인스턴스 삭제 명령에는 예약 인수가 없고, 예약 기능이 공개된 인스턴스 명령은 제한된 원격 명령 실행과 재부팅 등이며 시간 단위, 일 단위, 주 단위 반복만 지원한다([인스턴스 삭제 명령 소스](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/cli/commands/instances.py#L288-L349), [예약 실행 문서](https://docs.vast.ai/cli/reference/execute), [예약 재부팅 문서](https://docs.vast.ai/cli/reference/reboot-instance)).
매물의 `end_date`와 `duration`은 공급자가 보장하는 임대 가능 기간이지 사용자가 정한 자동 삭제 시각이 아니며, 수명이 끝난 인스턴스도 정지 상태로 남아 삭제 전까지 저장 공간 비용이 계속될 수 있다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers), [인스턴스 자주 묻는 질문](https://docs.vast.ai/guides/reference/faq/instances)).
따라서 Vast.ai 내부에 의존하지 않는 실행기나 자동 실행 서비스가 마감 시각에 `destroy instance`를 호출하고 삭제 결과를 재확인해야 완전한 비용 안전장치가 된다.

## 기능별 확인 결과

| 작업 | 공식 명령줄 도구 | 직접 API | 판정과 주의점 |
| --- | --- | --- | --- |
| 계정과 잔액 확인 | `vastai show user --raw` | `GET /api/v0/users/current/` | 브라우저 없이 가능하며 응답에 잔액이 포함된다([계정 조회](https://docs.vast.ai/api-reference/accounts/show-user), [명령줄 계정 조회](https://docs.vast.ai/cli/reference/show-user)). |
| 매물 검색 | `vastai search offers '<조건>' --type on-demand --raw` | `POST /api/v0/bundles/` | 검증 여부, 신뢰도, 임대 가능 기간, 임대 가능 상태, GPU 종류와 메모리, 총 시간당 가격을 모두 거를 수 있다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers)). |
| 인스턴스 생성 | `vastai create instance <offer_id> ... --cancel-unavail --raw` | `PUT /api/v0/asks/{offer_id}/` | 검색과 생성 사이에 매물이 사라질 수 있으므로 `cancel_unavail`로 즉시 실패시키고 다음 매물을 검색해야 한다([API로 인스턴스 만들기](https://docs.vast.ai/api-reference/creating-instances-with-api), [인스턴스 생성 API](https://docs.vast.ai/api-reference/instances/create-instance)). |
| 상태와 접속 정보 확인 | `vastai show instance <id> --raw`, `vastai ssh-url <id>` | `GET /api/v0/instances/{id}/` | `actual_status`, `ssh_host`, `ssh_port`, GPU 정보와 비용 필드를 조회할 수 있다([인스턴스 조회](https://docs.vast.ai/api-reference/instances/show-instance), [SSH 주소 조회](https://docs.vast.ai/cli/reference/ssh-url)). |
| 자료 전송 | `vastai copy local:<경로> C.<id>:<경로>`와 반대 방향 | 인스턴스 복사 API 또는 SSH/SCP | 로컬과 인스턴스 사이 전송을 지원하며 `/root`와 `/`를 대상으로 쓰면 SSH 권한이 깨질 수 있다([자료 복사 명령](https://docs.vast.ai/cli/reference/copy)). |
| 현재 상태와 시간당 가격 | `vastai show instances --raw` | `GET /api/v1/instances/` | 인스턴스 응답의 `actual_status`, `dph_total`, `search`와 `instance` 비용 세부 항목으로 현재 계약의 가격을 확인할 수 있다([인스턴스 목록 API](https://docs.vast.ai/api-reference/instances/show-instances)). |
| 발생 비용 확인 | `vastai show invoices-v1 --charges ...` | `GET /api/v0/charges/` | GPU, 저장 공간, 전송량을 인스턴스별로 나눈 실제 발생 비용을 조회할 수 있다([비용 내역 API](https://docs.vast.ai/api-reference/billing/show-charges)). |
| 일시 정지 | `vastai stop instance <id>` | 인스턴스 관리 API에서 `stopped` 상태 요청 | GPU 비용만 멈추고 임시 저장 공간 비용은 계속되므로 최종 정리 수단이 아니다([인스턴스 관리](https://docs.vast.ai/guides/instances/manage-instances), [저장 방식](https://docs.vast.ai/guides/instances/storage/types)). |
| 인스턴스 삭제 | `vastai destroy instance <id> -y --raw` | `DELETE /api/v0/instances/{id}/` | 인스턴스와 임시 저장 공간을 영구 삭제하며 성공 응답을 받을 수 있다([인스턴스 삭제 API](https://docs.vast.ai/api-reference/instances/destroy-instance), [비대화형 삭제 구현](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/cli/commands/instances.py#L305-L327)). |
| 별도 저장 공간 정리 | `vastai show volumes --type all --raw`, `vastai delete volume <id>` | `GET`, `DELETE /api/v0/volumes/` | 연결된 인스턴스를 먼저 삭제해야 하며, 별도 저장 공간은 인스턴스 삭제 후에도 남아 독립적으로 과금되므로 따로 삭제해야 한다([저장 공간 목록](https://docs.vast.ai/api-reference/volumes/list-volumes), [저장 공간 삭제](https://docs.vast.ai/api-reference/volumes/delete-volume), [저장 방식](https://docs.vast.ai/guides/instances/storage/types)). |
| 최종 과금 중지 확인 | 인스턴스 목록과 모든 저장 공간 목록을 각각 `--raw`로 재조회 | 인스턴스와 저장 공간 목록 API를 재조회 | 숫자 형태의 현재 과금 속도 `0`을 직접 반환하는 공식 API 필드는 확인되지 않았으므로, 소유한 인스턴스와 저장 공간이 없다는 사실로 지속 과금 자원이 없음을 판정하고 비용 내역은 별도로 보존한다([인스턴스 목록 API](https://docs.vast.ai/api-reference/instances/show-instances), [저장 공간 목록](https://docs.vast.ai/api-reference/volumes/list-volumes), [요금 구조](https://docs.vast.ai/guides/reference/billing)). |
| 독립적인 강제 삭제 예약 | 공식 지원을 확인하지 못함 | 공개 API 명세에서 확인하지 못함 | 외부 실행기가 마감 시각에 삭제 API를 호출하고 결과를 재조회해야 한다([인스턴스 삭제 명령 소스](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/cli/commands/instances.py#L288-L349), [예약 작업이 공개된 명령들](https://docs.vast.ai/guides/instances/storage/cloud-backups)). |

## 매물 선택 규칙의 구현 가능성

공식 검색 API는 `verified`, `reliability`, `duration`, `rentable`, `gpu_name`, `gpu_ram`, `dph_total`, `host_id`, `machine_id`를 필터로 받는다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers)).
API의 `duration`은 지금부터 남은 최소 임대 가능 시간을 초 단위로 받고 `gpu_ram`은 MB 단위로 받지만, 명령줄 도구는 검색식의 `duration`을 일 단위로, `gpu_ram`을 GB 단위로 변환한다([공식 명령줄 도구의 단위 변환](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vast.py#L979-L1052), [매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers)).
일반 임대는 명령줄에서 `--type on-demand`, API에서 `type`의 온디맨드 값을 명시하면 되고, 입찰 가격을 넣지 않은 생성은 일반 임대로 처리된다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers), [공식 명령줄 도구 안내](https://github.com/vast-ai/vast-cli/blob/v1.5.4/vastai/SKILL.md#interruptible-spot-rentals)).

RTX A4000을 우선하는 1차 검색은 다음 구조로 표현할 수 있다.
`REQUIRED_DAYS`는 예상 실행 시간에 준비와 회수 여유를 더한 시간을 일 단위로 바꾼 값이고, `PRICE_LIMIT`은 해당 실행의 승인된 시간당 상한이다.

```bash
vastai search offers \
  'gpu_name=RTX_A4000 num_gpus=1 verified=true reliability>=0.995 duration>=REQUIRED_DAYS rentable=true gpu_ram>=16 direct_port_count>=1 dph_total<=PRICE_LIMIT' \
  --type on-demand --raw
```

RTX A4000 매물이 없으면 `gpu_name` 조건만 제거해 16GB 이상 매물을 찾고 가격과 성능 순서를 적용할 수 있으며, 검색 API는 필터와 정렬을 모두 지원한다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers), [명령줄 전체 절차](https://docs.vast.ai/cli/hello-world)).
기준 실행과 후보 실행을 같은 공급자와 같은 GPU 등급에서 수행하려면 첫 계약의 `host_id` 또는 `machine_id`와 `gpu_name`을 실행 기록에 저장한 뒤 후속 검색에 같은 값을 넣을 수 있다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers)).
직접 SSH를 기본으로 쓰려면 `direct_port_count>=1`을 검색 조건에 넣고 생성 시 `--ssh --direct`를 지정해야 한다([명령줄 전체 절차](https://docs.vast.ai/cli/hello-world), [접속 방식](https://docs.vast.ai/api-reference/creating-instances-with-api)).

## 브라우저 없는 기본 절차

다음은 구현할 자동 실행 도구가 따라야 할 순서이며, 실제 인스턴스를 만드는 명령이 아니라 작업 경계를 정하는 절차다.

1. `vastai show user --raw`로 인증과 잔액을 확인한다([계정 조회](https://docs.vast.ai/api-reference/accounts/show-user)).
2. 필수 조건과 실행별 가격 상한으로 일반 임대 매물을 검색하고, 후보의 `id`, `host_id`, `machine_id`, `gpu_name`, `gpu_ram`, `reliability`, `duration`, `dph_total`, 저장 공간과 전송 가격을 기록한다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers), [요금 구조](https://docs.vast.ai/guides/reference/billing)).
3. 선택한 매물 ID로 `create instance`를 호출하며 `--cancel-unavail`, `--ssh`, `--direct`, 명시적인 이미지와 저장 공간 크기, 실행 식별용 `--label`을 사용한다([API로 인스턴스 만들기](https://docs.vast.ai/api-reference/creating-instances-with-api)).
4. 반환된 `new_contract`를 인스턴스 ID로 저장하고, `actual_status`를 10초 이상 간격으로 조회하되 제한 시간과 `exited`, `unknown`, `offline` 실패 분기를 둔다([API 전체 절차](https://docs.vast.ai/api-reference/hello-world)).
5. `running`이 되면 `ssh-url` 또는 `ssh_host`와 `ssh_port`로 접속하고 `vastai copy`로 자료를 올린다([SSH 주소 조회](https://docs.vast.ai/cli/reference/ssh-url), [자료 복사 명령](https://docs.vast.ai/cli/reference/copy)).
6. 실행 중에는 인스턴스 상태와 가격, 계정 잔액을 제한된 주기로 확인하고, 실제 발생 비용은 비용 내역 API로 확인한다([인스턴스 목록 API](https://docs.vast.ai/api-reference/instances/show-instances), [비용 내역 API](https://docs.vast.ai/api-reference/billing/show-charges), [계정 조회](https://docs.vast.ai/api-reference/accounts/show-user)).
7. 결과를 로컬 또는 외부 저장 공간으로 회수하고 무결성을 확인한 뒤 인스턴스를 삭제한다([자료 복사 명령](https://docs.vast.ai/cli/reference/copy), [인스턴스 삭제 API](https://docs.vast.ai/api-reference/instances/destroy-instance)).
8. 인스턴스 목록에서 해당 실행이 사라졌는지 확인하고, `show volumes --type all --raw`에 남은 프로젝트용 저장 공간이 있으면 삭제한 뒤 두 목록을 다시 확인한다([인스턴스 목록 API](https://docs.vast.ai/api-reference/instances/show-instances), [저장 공간 관리](https://docs.vast.ai/guides/instances/storage/volumes)).
9. 실행 기간의 비용 내역과 삭제 성공 응답을 실행 기록에 남긴다([비용 내역 API](https://docs.vast.ai/api-reference/billing/show-charges)).

삭제 요청의 성공 응답만으로 절차를 끝내면 안 된다.
삭제 API의 성공 응답은 해당 삭제 요청의 성공만 나타내므로 대상 인스턴스가 목록에서 사라졌는지 제한 시간을 둬 재확인해야 하고, 별도 저장 공간은 독립적으로 남을 수 있으므로 저장 공간 목록도 반드시 확인해야 한다([인스턴스 삭제 API](https://docs.vast.ai/api-reference/instances/destroy-instance), [인스턴스 목록 API](https://docs.vast.ai/api-reference/instances/show-instances), [저장 방식](https://docs.vast.ai/guides/instances/storage/types)).

## API 키와 비밀 값 관리

API 요청은 `Authorization: Bearer <키>` 헤더를 사용하며, 명령줄 도구와 SDK도 같은 키를 사용한다([인증](https://docs.vast.ai/api-reference/authentication)).
API 키는 기본적으로 만료되지 않으며 언제든 즉시 폐기할 수 있으므로, 자동 만료에 기대지 말고 프로젝트 전용 키를 정기적으로 교체해야 한다([인증](https://docs.vast.ai/api-reference/authentication), [명령줄 인증](https://docs.vast.ai/cli/authentication)).
Vast.ai는 운영 키의 90일 주기 교체를 합리적인 기준으로 제시하고, 용도별 키 분리와 최소 권한을 권장한다([API 키 관리](https://docs.vast.ai/guides/reference/api-keys)).
키 재설정은 기존 값을 즉시 무효화하며 겹침 기간이 없으므로, 중단 없는 교체가 필요하면 새 프로젝트 키를 만들고 비밀 저장소의 참조를 바꾼 뒤 이전 키를 삭제하는 순서가 안전하다([API 키 관리](https://docs.vast.ai/guides/reference/api-keys)).
고정된 외부 IP에서만 호출한다면 API 키 생성의 `key_params.ip_whitelist`로 출발지 IP를 제한할 수 있다([API 키 생성](https://docs.vast.ai/api-reference/accounts/create-api-key)).

일상 실행 키에 권장하는 범주는 다음과 같다.

- `misc`: 매물 검색에 필요하다([권한 범주와 API 대응표](https://docs.vast.ai/api-reference/permissions)).
- `instance_read`: 인스턴스, 로그, SSH 키 연결 상태와 저장 공간 조회에 필요하다([권한 범주와 API 대응표](https://docs.vast.ai/api-reference/permissions)).
- `instance_write`: 인스턴스 생성, 정지, 삭제, 자료 복사와 저장 공간 삭제에 필요하다([권한 범주와 API 대응표](https://docs.vast.ai/api-reference/permissions)).
- `user_read`: 계정 잔액과 등록된 SSH 키 확인에 필요하다([권한 범주와 API 대응표](https://docs.vast.ai/api-reference/permissions)).
- `billing_read`: 비용 내역 확인에 사용한다([권한 범주와 API 대응표](https://docs.vast.ai/api-reference/permissions), [비용 내역 API](https://docs.vast.ai/api-reference/billing/show-charges)).

일상 실행 키에는 `user_write`와 `billing_write`를 부여하지 않는다.
SSH 공개 키는 최초 설정 때 별도로 등록하고, 일상 실행 키가 API 키, SSH 키, 결제 관련 설정을 바꿀 수 없게 한다([권한 범주와 API 대응표](https://docs.vast.ai/api-reference/permissions), [명령줄 전체 절차](https://docs.vast.ai/cli/hello-world)).

키 값은 운영체제의 비밀 저장소에 보관하고 명령을 실행하는 자식 프로세스에만 `VAST_API_KEY` 환경 변수로 주입한다.
공식 문서도 자동 실행과 스크립트에서 `VAST_API_KEY` 환경 변수를 권장하며, 이 값은 로컬 키 파일보다 우선한다([명령줄 인증](https://docs.vast.ai/cli/authentication)).
`vastai set api-key <키>`는 키를 명령 인수로 받고 설정 파일에 평문으로 쓰므로, 이번 운영 정책에서는 사용하지 않는다([공식 구현](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/cli/commands/auth.py#L199-L220)).
같은 이유로 `--api-key <키>`도 셸 기록과 프로세스 인수에 노출될 수 있으므로 사용하지 않는다.
키가 들어간 셸 명령을 기록하거나 저장소의 `.env`, 구성 파일, 노트북, 로그에 남기지 않는다.

자격 증명이 설정된 실행에서 `--explain`을 사용하면 안 된다.
공식 v1.5.4 구현은 `--explain` 출력에 준비된 요청의 `Authorization` 헤더 전체를 포함하므로 로그에 API 키가 노출된다([공식 HTTP 클라이언트 구현](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/api/client.py#L88-L111)).
`--raw`는 자동 처리에 필요한 JSON 출력만 요청하므로 기본으로 사용하되, 응답에 `instance_api_key`, `jupyter_token` 같은 비밀 필드가 포함될 수 있어 원문 전체를 일반 로그에 남기지 않는다([API 전체 절차](https://docs.vast.ai/api-reference/hello-world), [인스턴스 조회](https://docs.vast.ai/api-reference/instances/show-instance)).

## 호출 제한과 실패 처리

Vast.ai는 API 호출 제한을 API 경로와 요청자 식별 정보별 최소 호출 간격으로 적용하고, 일부 경로에는 요청 방식별 제한과 짧은 구간의 최대 호출 수도 적용한다([호출 제한과 오류](https://docs.vast.ai/api-reference/rate-limits-and-errors)).
정확한 경로별 수치는 공개 문서에 제시되지 않으며, 제한에 걸리면 HTTP `429`와 일반 텍스트 또는 JSON 오류가 돌아오고 표준 `Retry-After` 헤더는 제공되지 않는다([호출 제한과 오류](https://docs.vast.ai/api-reference/rate-limits-and-errors)).
따라서 상태 조회는 공식 예시의 10초보다 빠르게 반복하지 않고, 지수형 대기와 무작위 지연을 적용하며, 최대 재시도 횟수와 전체 제한 시간을 둬야 한다([API 전체 절차](https://docs.vast.ai/api-reference/hello-world), [호출 제한과 오류](https://docs.vast.ai/api-reference/rate-limits-and-errors)).

오류 응답은 경로마다 조금 다르지만 흔한 JSON 형태는 `success`, `error`, `msg`이고, 인증 오류는 `401`, 권한 부족은 `403`, 호출 제한은 `429`로 구분해야 한다([호출 제한과 오류](https://docs.vast.ai/api-reference/rate-limits-and-errors), [인증](https://docs.vast.ai/api-reference/authentication)).
공식 v1.5.4 HTTP 클라이언트는 `429`, `502`, `503`, `504`와 연결 오류 및 제한 시간 초과를 재시도하며, 기본 재시도 횟수는 3회이고 요청 제한 시간은 120초다([공식 HTTP 클라이언트 구현](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/api/client.py#L40-L107)).
명령줄 도구의 재시도만으로 전체 작업의 안정성을 보장할 수는 없으므로, 자동 실행 도구는 검색, 생성, 상태 대기, 결과 회수, 삭제를 각각 멱등성 관점에서 다뤄야 한다.
특히 생성 요청이 시간 초과되면 같은 요청을 바로 반복하지 말고 인스턴스 목록과 실행 식별용 `label`을 먼저 조회해 중복 생성 여부를 판정해야 한다.
삭제 요청이 실패하거나 시간 초과되면 대상이 이미 사라졌는지 먼저 조회하고, 남아 있으면 제한된 횟수로 다시 삭제해야 한다.

## 공식 명령줄 도구와 직접 API 비교

| 기준 | 공식 명령줄 도구 | 직접 REST API |
| --- | --- | --- |
| 기능 범위 | 이 조사에 필요한 평상시 작업 전체를 제공한다([공식 명령줄 도구 저장소](https://github.com/vast-ai/vast-cli/tree/v1.5.4)). | 같은 기능을 원시 경로와 JSON 요청으로 제공한다([API 소개](https://docs.vast.ai/api-reference/introduction)). |
| 인증 | `VAST_API_KEY`를 읽어 Bearer 헤더를 구성한다([명령줄 인증](https://docs.vast.ai/cli/authentication)). | 호출자가 매 요청에 Bearer 헤더를 넣어야 한다([인증](https://docs.vast.ai/api-reference/authentication)). |
| 형식과 단위 | 사람이 쓰기 쉬운 검색식과 단위 변환, `--raw` JSON 출력을 제공한다([공식 단위 변환 구현](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vast.py#L979-L1052)). | API의 MB와 초 단위 및 경로별 요청 구조를 호출자가 정확히 맞춰야 한다([매물 검색 API](https://docs.vast.ai/api-reference/search/search-offers)). |
| 재시도 | 일시적인 서버 오류와 전송 오류의 기본 재시도를 제공한다([공식 HTTP 클라이언트 구현](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/api/client.py#L88-L107)). | 호출 제한 헤더가 없으므로 호출자가 모든 대기와 재시도를 구현해야 한다([호출 제한과 오류](https://docs.vast.ai/api-reference/rate-limits-and-errors)). |
| 비밀 값 안전성 | 환경 변수 주입을 사용할 수 있지만 `set api-key`, `--api-key`, `--explain`을 금지해야 한다([명령줄 인증](https://docs.vast.ai/cli/authentication), [공식 HTTP 클라이언트 구현](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/api/client.py#L107-L111)). | 비밀 저장소에서 읽은 값을 헤더에만 넣도록 구현할 수 있지만, 요청과 응답 로그 가림 처리를 직접 책임져야 한다([인증](https://docs.vast.ai/api-reference/authentication)). |
| 변경 대응 | Vast.ai가 API 변화에 맞춰 배포한 도구 버전으로 흡수할 수 있다. | 경로, 필드, 페이지 처리 변화에 자체 코드가 직접 대응해야 한다. |
| 적합한 용도 | 사람이 실행하는 명령과 초기 자동 실행 구현의 기본 경로다. | 명령줄 도구에 없는 제어, 다른 언어 연동, 세밀한 오류와 응답 처리가 필요할 때 쓰는 보조 경로다. |

기본 경로를 명령줄 도구로 정하더라도 버전을 고정하고 정기적으로 갱신해야 한다.
자동 실행은 모든 목록과 상태 조회에 `--raw`를 사용하고, 사람이 읽는 표 출력이나 문구를 해석하지 않아야 한다([공식 명령줄 도구 안내](https://github.com/vast-ai/vast-cli/blob/v1.5.4/vastai/SKILL.md#global-flags)).

## 독립적인 강제 종료의 판정

공식 자료에서 사용자가 지정한 한 번의 시각에 Vast.ai 제어면이 인스턴스를 자동으로 삭제하는 공개 명령이나 API는 확인하지 못했다.
공식 예약 작업은 시간 단위, 일 단위, 주 단위 작업을 다루고, 공개된 인스턴스 삭제 명령에는 예약 인수가 없다([예약 실행 문서](https://docs.vast.ai/cli/reference/execute), [인스턴스 삭제 명령 소스](https://github.com/vast-ai/vast-cli/blob/f7171feb403fd037d2d9b9dcc379aebed80311a7/vastai/cli/commands/instances.py#L305-L327)).
인스턴스가 종료되거나 잔액이 없어 자동 정지돼도 저장 공간 비용은 계속될 수 있으므로, 작업 컨테이너 내부의 종료 명령과 Vast.ai의 자동 정지는 비용 상한을 보장하지 못한다([결제 자주 묻는 질문](https://docs.vast.ai/guides/reference/faq/billing), [저장 방식](https://docs.vast.ai/guides/instances/storage/types)).

필요한 안전장치는 Vast.ai 인스턴스 밖에서 실행되는 독립된 삭제 작업이다.
이 작업은 최대 종료 시각에 인스턴스 삭제 API를 호출하고, 인스턴스 목록에서 사라짐을 확인한 뒤, 프로젝트가 만든 별도 저장 공간을 삭제하고, 두 목록이 비었는지 다시 확인해야 한다.
외부 삭제 작업 자체가 실패하면 Runpod 전환 조건으로 처리할 수 있도록 실패 알림과 수동 복구 절차도 필요하다.

## 운영 결정을 위한 요약

1. Vast.ai는 API 키와 공식 명령줄 도구만으로 평상시 원격 자원 제어 전체를 브라우저 없이 수행할 수 있으므로 주 실행 환경의 기본 제어 경로로 채택할 수 있다.
2. 기본 도구는 `vastai` v1.5.4 이상, 출력은 `--raw`, 인증은 운영체제 비밀 저장소에서 실행별로 주입한 `VAST_API_KEY`로 고정한다.
3. 일상 실행 키에는 `misc`, `instance_read`, `instance_write`, `user_read`, `billing_read`만 부여하고 90일을 기본 교체 주기로 삼는다.
4. `set api-key`, `--api-key`, `--explain`과 원문 응답 전체 기록을 금지하고, 로그에서 토큰과 세션 정보를 가린다.
5. 삭제 성공은 인스턴스 목록과 모든 저장 공간 목록의 재조회로 확인하며, 숫자 형태의 과금 속도 `0` 필드가 없다는 점을 명세에 명시한다.
6. Vast.ai 내부에는 독립적인 일회성 강제 삭제 예약이 확인되지 않았으므로 외부 실행기에서 삭제와 재확인을 예약해야 한다.
7. 외부 강제 삭제 장치를 설정할 수 없거나 삭제 후 자원 부재를 확인할 수 없으면 Runpod으로 전환한다.
