# 외부 독립 종료 실행 서비스의 보장과 제약 조사

## 결론

2026-08-15 현재 **Amazon EventBridge Scheduler와 AWS Lambda 조합을 독립 종료 예약의 기본 실행 환경으로 권장한다**.
EventBridge Scheduler는 단발 `at(...)` 예약, 예약 이름과 ARN, `GetSchedule` 재조회, 비활성화한 유연 실행 구간에서의 60초 정밀도, 실패 전달 재시도와 SQS 실패 보관함을 공식 기능으로 제공한다 ([예약 종류](https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html), [GetSchedule](https://docs.aws.amazon.com/scheduler/latest/APIReference/API_GetSchedule.html), [Scheduler 개요](https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html)).
Lambda는 최대 15분 실행할 수 있으므로, Vast.ai 삭제를 즉시 요청한 뒤 10초 이상 간격으로 최대 5분 동안 인스턴스와 프로젝트용 저장 공간의 부재를 확인하는 기존 규약을 한 번의 실행 안에서 처리할 수 있다 ([Lambda 실행 제한](https://docs.aws.amazon.com/lambda/latest/dg/configuration-timeout.html)).

다만 이 조합도 외부 API까지 포함한 실시간 종료를 보장하지는 않는다.
공식적으로 보장된 정밀도는 예약 시각부터 60초 안에 대상 API를 호출하는 데까지이고, Lambda 비동기 대기 시간과 Vast.ai API 응답 시간에는 공개된 최대 지연 보장이 없다 ([예약 종류](https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html), [Lambda 비동기 지표](https://docs.aws.amazon.com/lambda/latest/dg/monitoring-metrics-types.html)).
따라서 절대 실행 상한보다 5분 앞서 첫 삭제를 예약하고, 실행하지 못한 경우의 경보와 긴급 복구를 별도로 두는 조건에서 운영 요구를 충족한다고 판정한다.
예약을 절대 상한 시각과 같게 잡으면서 1초도 늦지 않게 삭제해야 한다는 뜻이라면 조사한 후보 가운데 충족하는 서비스는 없다.

Google Cloud Tasks와 Cloud Run은 이름 있는 단발 작업, 사전 재조회, 안전한 비밀 값 저장과 자동 재시도를 제공하므로 예비 후보가 될 수 있다.
그러나 Google은 Cloud Tasks가 전달 시각에 강한 보장을 제공하지 않는다고 명시하고, 서비스 수준 협약도 작업 생성 성공률만 보장하므로 이 지도의 기본 독립 종료 실행 환경으로는 채택하지 않는다 ([Cloud Tasks 개요](https://docs.cloud.google.com/tasks/docs/dual-overview), [Cloud Tasks 서비스 수준 협약](https://cloud.google.com/tasks/sla)).

GitHub Actions는 계산 자원과 개인 컴퓨터가 꺼져 있어도 실행되지만, 예약 실행이 늦어질 수 있고 부하가 크면 대기 중 작업이 누락될 수도 있다.
또한 `schedule`은 반복 cron 일정일 뿐 단발 예약 자원과 자동 재시도를 제공하지 않으므로 운영 조건을 충족하지 않는다 ([예약 실행 제약](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows), [수동 재실행](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)).

## 판정 요약

| 조건 | GitHub Actions | EventBridge Scheduler + Lambda | Cloud Tasks + Cloud Run |
| --- | --- | --- | --- |
| 로컬과 GPU 인스턴스가 꺼져도 실행 | 충족 | 충족 | 충족 |
| 단발 예약 자원 | 불충족, 반복 cron만 지원 | 충족, `at(...)`와 실행 뒤 자동 삭제 지원 | 충족, 이름 있는 작업 하나를 지정 시각에 전달 |
| 예약 식별자, 대상, 시각 사전 재조회 | 부분 충족, 실행 흐름과 cron 원문은 조회할 수 있으나 예약 건별 식별자가 없음 | 충족, `GetSchedule`이 이름, ARN, 일정 식, 시간대, 상태, 대상 ARN과 입력을 반환 | 충족, `tasks.get`이 이름, `scheduleTime`, HTTP 대상을 반환하고 전체 조회 권한으로 본문도 확인 가능 |
| 공식 시각 정확도와 최대 지연 | 불충족, 지연과 누락 가능 | 조건부 충족, 유연 실행 구간을 끄면 60초 정밀도이나 종단 간 최대 지연은 없음 | 불충족, 전달 시각에 강한 보장이 없고 지연 상한도 없음 |
| 삭제와 자원 부재 확인 재시도 | 실행 코드로 가능하나 실행 흐름 자체의 자동 재시도 없음 | 충족, 함수 안의 5분 확인과 Scheduler 및 Lambda의 전달 재시도를 함께 사용 | 충족, 실행 코드와 작업 대기열의 지수 간격 재시도 사용 |
| 중복 실행 의미 | 예약 반복과 수동 재실행을 직접 막아야 함 | 최소 한 번 전달이며 Lambda 비동기 중복 가능, 고정 대상에 대한 멱등 처리 필수 | 최소 한 번 전달이며 드물게 중복 가능, 멱등 처리 필수 |
| 최소 권한 Vast.ai 키 저장 | 환경 비밀 값으로 가능하나 저장소의 실행 흐름 수정 권한과 결합됨 | 충족, Secrets Manager의 특정 비밀 값 읽기만 함수 역할에 허용 가능 | 충족, Secret Manager의 특정 비밀 값 접근만 전용 서비스 계정에 허용 가능 |
| 실패 알림 | 실행이 시작된 뒤 실패하면 알림 가능, 예약 누락은 실행 기록이 없어 감지하기 어려움 | 충족, Scheduler 전달 실패와 Lambda 코드 실패를 분리해 경보 가능 | 조건부 충족, 작업 시도 로그와 지표에 별도 경보를 구성해야 하고 기본 실패 보관함은 없음 |
| 감사 기록 | 실행 기록과 로그가 기본 90일 보존되지만 누락된 예약은 기록이 없음 | 충족, CloudTrail 제어 기록과 Lambda 실행 로그를 결합 | 충족, 작업 감사 로그와 Cloud Run 실행 로그를 결합하되 자료 접근 감사 로그는 별도 활성화 필요 |
| 소규모 사용 비용 | 공개 저장소의 표준 실행기는 무료 | 대체로 월 0.40달러 수준부터 시작, 비밀 값 하나의 고정비가 주 비용 | 무료 사용량 안에 들 가능성이 높지만 결제 계정 연결은 필수 |
| 최종 판정 | 불충족 | **기본 후보로 충족** | 시간 보장 부족으로 불충족 |

## 공통 실행 계약

어떤 관리형 서비스를 쓰더라도 독립 종료 예약은 삭제 명령 하나가 아니라 다음 계약을 실행해야 한다.

1. 예약 입력에는 원격 실행 작업 식별자, Vast.ai 인스턴스 식별자 목록, 프로젝트용 저장 공간 식별자 목록, 절대 실행 상한과 예약 시각을 넣는다.
2. Vast.ai API 키는 예약 입력에 넣지 않고 실행 시점에 별도 비밀 저장소에서 읽는다.
3. 실행기는 먼저 대상의 현재 존재 여부를 조회하고, 존재하는 대상에만 삭제를 요청한다.
4. 삭제 요청 뒤 10초 이상의 간격으로 최대 5분 동안 인스턴스 목록과 저장 공간 목록에서 대상이 사라졌는지 확인한다.
5. 대상이 이미 없다는 응답과 삭제 뒤 목록에서 사라진 상태를 성공으로 동일하게 취급한다.
6. 삭제 요청 성공만으로 성공 처리하지 않고, 인스턴스와 프로젝트용 저장 공간이 모두 없을 때만 성공 처리한다.
7. 성공과 실패 로그에는 예약 식별자, 대상 식별자, 예정 시각, 실제 시작 시각, 삭제 시도 횟수, 최종 존재 여부만 남긴다.
8. API 키, `instance_api_key`, Jupyter 토큰, SSH 비밀 값과 Vast.ai 원문 응답은 기록하지 않는다.
9. 전달과 실행은 중복될 수 있다고 전제하고, 고정된 대상 식별자에 대한 삭제와 부재 확인을 멱등하게 만든다.
10. 예약 생성 직후 같은 제어 경로에서 예약을 다시 읽어 모든 필드를 대조한 뒤에만 유료 원격 실행 작업을 시작한다.

대상을 `현재 계정의 모든 자원`처럼 넓게 표현하면 늦게 도착한 중복 실행이 다음 원격 실행 작업의 자원을 삭제할 수 있다.
모든 삭제 실행은 예약 당시 고정한 자원 식별자만 다루어야 한다.

## 권장 후보: Amazon EventBridge Scheduler와 AWS Lambda

### 예약과 사전 재조회

EventBridge Scheduler는 `at(yyyy-mm-ddThh:mm:ss)` 일정 식으로 대상을 한 번만 호출하는 예약을 지원한다 ([CreateSchedule](https://docs.aws.amazon.com/scheduler/latest/APIReference/API_CreateSchedule.html)).
`CreateSchedule`은 예약 이름을 받고 예약 ARN을 반환하며, 같은 요청에 `ClientToken`을 넣어 예약 생성 요청 자체를 멱등하게 만들 수 있다 ([CreateSchedule](https://docs.aws.amazon.com/scheduler/latest/APIReference/API_CreateSchedule.html)).
`GetSchedule`은 이름과 예약 묶음 이름으로 예약을 조회하고, 예약 ARN, `ScheduleExpression`, 시간대, 상태, 대상 ARN, 대상 입력, 재시도 정책과 완료 뒤 처리 방식을 반환한다 ([GetSchedule](https://docs.aws.amazon.com/scheduler/latest/APIReference/API_GetSchedule.html)).
따라서 원격 실행 장부의 예약 식별자는 `예약 묶음 이름/예약 이름`과 반환된 ARN으로 고정하고, 실행 전 재조회에서 다음을 모두 대조할 수 있다.

- 상태가 `ENABLED`인지 확인한다.
- 일정 식이 기대한 단발 `at(...)`인지 확인한다.
- 시간대가 `UTC`인지 확인한다.
- 유연 실행 구간이 `OFF`인지 확인한다.
- 대상 ARN이 승인된 종료 Lambda 함수인지 확인한다.
- 대상 입력의 원격 실행 작업, 인스턴스, 저장 공간과 시각이 원격 실행 장부와 같은지 확인한다.
- `ActionAfterCompletion`이 `DELETE`인지 확인한다.
- Scheduler 전달 실패용 SQS 실패 보관함과 재시도 정책이 붙었는지 확인한다.

`ActionAfterCompletion=DELETE`를 사용하면 단발 예약은 대상을 한 번 호출한 뒤 자동 삭제되므로 해마다 다시 실행될 여지가 없고 예약 수 한도도 계속 차지하지 않는다 ([완료 뒤 자동 삭제](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule-delete.html)).
자동 삭제는 Lambda 함수가 작업을 성공적으로 마쳤다는 뜻이 아니라 Scheduler가 Lambda 호출을 완료했다는 뜻으로 해석해야 한다.
EventBridge Scheduler가 Lambda를 비동기로 호출하면 함수 코드의 성공이나 실패를 기다리지 않기 때문이다 ([Scheduler의 Lambda 호출](https://docs.aws.amazon.com/lambda/latest/dg/with-eventbridge-scheduler.html), [Lambda 비동기 호출](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html)).

### 시각 정확도

유연 실행 구간을 끄면 모든 EventBridge Scheduler 예약은 60초 정밀도로 호출되고, 예를 들어 01:00 예약은 01:00:00부터 01:00:59 사이에 대상 API를 호출한다 ([예약 종류](https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html)).
이는 분 단위 cron과 지연 가능성만 설명하는 다른 후보보다 가장 분명한 공식 시간 조건이다.
그러나 Lambda 비동기 대기열의 `AsyncEventAge`가 늘어날 수 있고 Lambda 실행 시작의 최대 지연은 이 60초 조건에 포함되지 않는다 ([Lambda 비동기 지표](https://docs.aws.amazon.com/lambda/latest/dg/monitoring-metrics-types.html)).

절대 실행 상한을 `T`라고 할 때 권장 예약 시각은 `T - 5분`이다.
예약 자체의 최악 정밀도 60초 뒤에도 첫 삭제 요청을 준비할 여유가 약 4분 남고, 삭제 뒤 부재 확인은 자원 삭제 요청 이후에 계속 수행할 수 있다.
이 5분은 공식적인 종단 간 보장이 아니라 공개된 60초 정밀도와 서버리스 실행 지연에 대한 운영 여유다.
예약 재조회에 실패하거나 5분 앞선 종료를 받아들일 수 없는 작업은 Vast.ai에서 시작하지 않고 기존 결정대로 Runpod으로 전환해야 한다.

### 실행 제한과 삭제 재시도

Lambda의 기본 실행 제한은 3초이고 설정 가능한 최대는 900초이므로, 종료 함수는 7분에서 10분 사이의 제한을 명시적으로 설정할 수 있다 ([Lambda 실행 제한](https://docs.aws.amazon.com/lambda/latest/dg/configuration-timeout.html)).
Vast.ai 삭제를 즉시 요청하고 10초 간격으로 5분 확인한 뒤 마지막 직접 REST API 삭제를 재시도하는 규약은 15분 안에 들어간다.
각 Vast.ai HTTP 호출에는 별도의 짧은 연결 및 읽기 제한을 두어 한 호출이 Lambda 전체 제한을 소진하지 않게 해야 한다.
Lambda 제한에 도달하면 함수는 오류가 되고 비동기 재시도와 실패 목적지 규칙이 적용된다 ([Lambda 비동기 오류 처리](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-error-handling.html)).

### 전달 재시도와 중복 실행

EventBridge Scheduler는 대상 전달을 최소 한 번 보장하며, 대상 호출 실패 시 지연 재시도를 수행한다 ([Scheduler 개요](https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html)).
재시도 정책은 이벤트 보관 기간을 60초부터 24시간까지, 최대 재시도 횟수를 0회부터 185회까지 설정할 수 있다 ([RetryPolicy](https://docs.aws.amazon.com/scheduler/latest/APIReference/API_RetryPolicy.html)).
이 재시도는 Lambda 호출 API가 거절되거나 제한되는 등 **Scheduler가 Lambda에 전달하지 못한 실패**를 다룬다.

Scheduler가 Lambda 비동기 대기열에 넣는 데 성공한 뒤 함수 코드가 실패하면 Scheduler는 그 실패를 알지 못한다.
이 경우 Lambda의 별도 비동기 정책이 작동하며 기본값은 함수 오류 뒤 1분과 2분 간격의 두 번 추가 실행이다 ([Lambda 비동기 오류 처리](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-error-handling.html)).
Lambda는 오류가 없어도 같은 이벤트를 드물게 두 번 전달할 수 있다고 명시하므로, 고정한 Vast.ai 대상 식별자의 부재를 성공으로 취급하는 멱등성이 필수다 ([Lambda 비동기 오류 처리](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-error-handling.html)).

권장 설정은 Scheduler 전달 재시도와 Lambda 비동기 재시도를 모두 켜되 한 시간 안에서 끝나게 제한하는 것이다.
지연된 삭제도 지속 과금 중단에는 가치가 있지만, 다음 작업의 다른 자원을 건드리지 않도록 예약 당시 식별자만 처리해야 한다.
중복 방지를 이유로 재시도를 없애는 것보다 삭제와 부재 확인을 멱등하게 만드는 편이 안전하다.

### Vast.ai API 키와 최소 권한

종료 전용 Vast.ai API 키에는 앞선 [Vast.ai 원격 자원 제어 경로와 자격 증명 규약](https://github.com/tmheo/predicting-smartphone-addiction/issues/125)에서 정한 `instance_read`와 `instance_write`만 부여한다.
키는 Scheduler 대상 입력이나 Lambda 환경 변수 평문이 아니라 AWS Secrets Manager의 전용 비밀 값 하나에 저장한다.
Secrets Manager는 저장 값을 KMS로 암호화하고 TLS로 전달하며, 특정 비밀 값에 대한 최소 권한 정책을 권장한다 ([Secrets Manager 권장 사항](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)).

역할은 다음처럼 분리한다.

- Scheduler 실행 역할은 승인된 종료 Lambda 함수 하나에 대한 `lambda:InvokeFunction`만 허용한다.
- Lambda 실행 역할은 종료 키 비밀 값 하나에 대한 `secretsmanager:GetSecretValue`와 자신의 로그 쓰기만 허용한다.
- 예약을 만드는 운영 주체는 전용 예약 묶음 안의 `CreateSchedule`, `GetSchedule`, `DeleteSchedule`만 허용한다.
- 사람의 비밀 값 생성과 교체 권한은 Lambda 실행 역할과 분리한다.

`GetSchedule` 응답에는 대상 `Input`이 들어가므로 API 키를 예약 입력에 넣으면 예약 조회 권한자에게 그대로 노출된다 ([GetSchedule](https://docs.aws.amazon.com/scheduler/latest/APIReference/API_GetSchedule.html)).
예약 입력에는 비밀이 아닌 자원 식별자와 시각만 넣어야 한다.

### 실패 알림

실패 경로가 두 층이므로 알림도 두 층으로 구성해야 한다.

1. Scheduler가 Lambda를 호출하지 못한 사건은 Scheduler의 SQS 실패 보관함과 `TargetErrorCount`, `InvocationDroppedCount`, `InvocationsSentToDeadLetterCount` 지표로 감시한다 ([Scheduler CloudWatch 지표](https://docs.aws.amazon.com/scheduler/latest/UserGuide/monitoring-cloudwatch.html)).
2. Lambda가 시작된 뒤 삭제나 부재 확인에 실패한 사건은 Lambda 비동기 `OnFailure` 목적지와 `Errors`, `AsyncEventsDropped`, `DestinationDeliveryFailures` 지표로 감시한다 ([Lambda 실패 기록 보존](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-retain-records.html), [Lambda 지표](https://docs.aws.amazon.com/lambda/latest/dg/monitoring-metrics-types.html)).
3. 각 지표의 1회 이상 발생을 CloudWatch 경보로 만들고 SNS 전자우편 알림에 연결한다 ([CloudWatch 경보 알림](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Notify_Users_Alarm_Changes.html)).
4. SNS 전자우편 구독은 사용자가 확인해야 실제 알림이 전달되므로 최초 설정 때 확인과 시험 알림을 완료한다 ([SNS 전자우편 구독](https://docs.aws.amazon.com/sns/latest/dg/sns-email-notifications.html)).

Scheduler 실패 보관함만 두면 Lambda 코드 실패가 빠지고, Lambda 오류 경보만 두면 Scheduler가 함수를 한 번도 호출하지 못한 사건이 빠진다.
두 경로를 모두 실제 오류로 한 번씩 시험해야 운영 합격이다.

### 로그 가림과 감사 기록

Lambda의 기본 로그 목적지는 CloudWatch Logs이고, 오류 수와 실행 시간 지표도 자동 수집된다 ([Lambda 로그](https://docs.aws.amazon.com/lambda/latest/dg/monitoring-logs.html), [Lambda 감시](https://docs.aws.amazon.com/lambda/latest/dg/lambda-monitoring.html)).
종료 함수는 Vast.ai 원문 응답을 출력하지 않고 앞에서 정한 허용 필드만 구조화해 기록해야 한다.
CloudWatch Logs 자료 보호 정책은 사용자 정규식으로 추가 식별자를 정의해 표시 시 민감 값을 가릴 수 있으므로 Vast.ai 키 형식에 대한 방어 정책을 추가한다 ([자료 보호 정책](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/mask-sensitive-log-data-start.html), [사용자 식별자](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL-custom-data-identifiers.html)).
이 가림은 잘못 출력한 비밀 값의 사후 방어선이지 원문 응답을 기록해도 된다는 허가가 아니다.

EventBridge Scheduler의 모든 API 호출은 CloudTrail에 기록되며 `CreateSchedule`, `UpdateSchedule`, `DeleteSchedule`도 포함된다 ([Scheduler CloudTrail 기록](https://docs.aws.amazon.com/scheduler/latest/UserGuide/logging-using-cloudtrail.html)).
CloudTrail의 예약 생성과 재조회 기록, Lambda의 실제 시작과 종료 로그, 삭제 시도와 최종 부재 결과를 결합하면 예약과 실행 양쪽의 감사 기록이 된다.
예약은 완료 뒤 자동 삭제되므로 성공 로그와 CloudTrail 기록의 보존 기간을 명시적으로 설정하고, 나중에 로컬 장비가 켜지면 원격 실행 장부에 요약을 반영해야 한다.

### 비용과 계정 준비

EventBridge Scheduler는 월 1,400만 번 호출까지 무료이고 Lambda 무료 사용량은 월 100만 요청과 40만 GB-초이므로 이 프로젝트의 단발 종료 호출은 일반적으로 무료 사용량 안에 든다 ([EventBridge 요금](https://aws.amazon.com/eventbridge/pricing/), [Lambda 요금](https://aws.amazon.com/lambda/pricing/)).
Secrets Manager는 비밀 값 하나당 월 0.40달러이고 API 호출 1만 번당 0.05달러이므로 종료 키 하나를 계속 보관할 때 약 0.40달러가 최소 고정비다 ([Secrets Manager 요금](https://aws.amazon.com/secrets-manager/pricing/)).
CloudWatch 무료 사용량에는 로그 5GB와 표준 경보 지표 10개가 포함되지만, 자료 보호 검색은 별도 사용량으로 계산될 수 있다 ([CloudWatch 요금](https://aws.amazon.com/cloudwatch/pricing/)).
실제 비용은 리전, 로그 양, SQS, SNS와 자료 전송을 포함해 배포 전에 다시 계산해야 한다.

최초 준비에는 결제 가능한 AWS 계정, 관리용 IAM 주체, Scheduler 대상 실행 역할, Lambda 실행 역할, 종료 함수, Secrets Manager 비밀 값, Scheduler와 Lambda용 실패 목적지, CloudWatch 경보와 확인된 SNS 구독이 필요하다.
AWS는 계정과 관리 사용자를 먼저 설정한 뒤 필요한 권한만 가진 별도 역할을 만들 것을 권장한다 ([Scheduler 설정](https://docs.aws.amazon.com/scheduler/latest/UserGuide/setting-up.html)).
현재 저장소와 GitHub 계정만으로 바로 사용할 수 있는 GitHub Actions보다는 최초 준비가 크지만, 독립 종료의 시간 조건과 실패 분리를 위해 감수할 가치가 있다.

## 불충족 후보: GitHub Actions

### 예약과 시각 보장

GitHub Actions의 `schedule`은 다섯 필드 POSIX cron이며 최소 실행 간격은 5분이다 ([예약 실행](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)).
연도 필드가 없어서 특정 날짜와 시각에 한 번만 실행되는 예약을 표현할 수 없고, 실행 뒤 실행 흐름 파일을 고치거나 별도 상태로 재실행을 막아야 한다.
실행 흐름 파일과 cron 문자열은 조회할 수 있지만 예약 건마다 서버가 발급한 식별자, 대상과 시각을 한 번에 돌려주는 단발 예약 자원은 없다.

예약 실행은 부하가 클 때 늦어질 수 있고 특히 매시 정각은 부하가 높으며, 부하가 충분히 크면 대기 중 작업이 누락될 수 있다 ([실행 흐름 문제 해결](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows)).
공개 저장소에 60일 동안 활동이 없으면 예약 실행 흐름이 자동 비활성화되고, 예약 실행은 기본 브랜치의 최신 커밋에서만 동작한다 ([예약 실행](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)).
이는 대회 중 활동이 잦더라도 독립 종료 장치가 의존할 보장으로는 부족하다.

### 재시도, 제한과 알림

작업 제한의 기본값은 360분이라 5분 삭제 확인에는 충분하다 ([실행 흐름 문법](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)).
그러나 실패한 실행의 재실행은 쓰기 권한자가 30일 안에 수동으로 수행하는 기능이고 한 실행을 최대 50번 다시 실행할 수 있을 뿐, 예약 작업의 자동 재시도 정책은 아니다 ([수동 재실행](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)).
실행이 시작된 뒤 실패하면 전자우편이나 웹 알림을 받을 수 있지만, 부하 때문에 예약 자체가 누락되면 실패한 실행 기록이 생기지 않으므로 이 알림만으로 누락을 감지할 수 없다 ([Actions 알림](https://docs.github.com/en/subscriptions-and-notifications/how-tos/managing-github-actions-notifications)).

### 비밀 값, 로그와 비용

환경 비밀 값은 그 환경을 참조하는 작업에만 제공되고, 출력된 GitHub 비밀 값은 로그에서 자동 가려진다 ([GitHub 비밀 값 종류](https://docs.github.com/en/code-security/reference/secret-security/secret-types)).
그러나 변환된 비밀 값의 가림은 보장되지 않으므로 추가 값은 출력 전에 `add-mask`로 등록해야 한다 ([GitHub 비밀 값](https://docs.github.com/en/enterprise-cloud@latest/actions/concepts/security/secrets), [로그 가림 명령](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands)).
공개 저장소의 표준 GitHub 제공 실행기는 무료이고 실행 로그와 산출물은 기본 90일 보존된다 ([GitHub 제공 실행기](https://docs.github.com/en/actions/reference/runners/github-hosted-runners), [로그 보존](https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization)).

비용과 기존 계정 활용은 가장 유리하지만 지연, 누락, 단발 예약 부재와 자동 재시도 부재가 독립 종료의 핵심을 어긴다.
따라서 보조 감시나 수동 실행 경로로는 쓸 수 있어도 기본 독립 종료 예약으로는 사용하지 않는다.

## 예비 후보: Google Cloud Tasks와 Cloud Run

### Cloud Scheduler 대신 Cloud Tasks를 비교한 이유

Google Cloud Scheduler는 다섯 필드 unix-cron으로 반복 일정을 만들며 단발 일정 형식을 제공하지 않는다 ([Cloud Scheduler 일정 형식](https://docs.cloud.google.com/scheduler/docs/configuring/cron-job-schedules)).
Cloud Scheduler도 최소 한 번 전달이라 중복 가능성이 있고 대상은 멱등해야 한다 ([Cloud Scheduler 개요](https://docs.cloud.google.com/scheduler/docs/overview)).
실행 뒤 스스로 일정을 삭제하는 추가 제어가 필요하므로 단발 종료에는 이름과 `scheduleTime`을 가진 Cloud Tasks 작업이 더 단순하다.

### 예약, 재조회와 단발 의미

Cloud Tasks 작업은 호출자가 정한 고유 이름, RFC 3339 `scheduleTime`, HTTP 대상, 본문과 실행 제한을 가질 수 있다 ([Task 자원](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks)).
`tasks.get`으로 실행 전에 작업을 읽을 수 있고, 기본 조회에는 큰 값이나 민감할 수 있는 본문이 빠지며 전체 조회에는 `cloudtasks.tasks.fullView` 권한이 추가로 필요하다 ([tasks.get](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks/get)).
예약 식별자, 대상 자원 식별자와 시각을 모두 재조회하려면 대상 식별자를 비밀이 아닌 작업 본문에 넣고 전체 조회 권한으로 대조하거나 작업 이름에 원격 실행 작업 식별자를 포함해야 한다.
API 키는 작업 본문에 넣지 않는다.

작업이 2xx 응답으로 성공하거나 재시도를 모두 소진하면 작업 자원은 삭제되므로 본질적으로 단발이다 ([tasks.get](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks/get)).
작업은 생성 뒤 최대 31일이면 실행 여부와 관계없이 삭제되지만 이 지도의 2시간 또는 8시간 예약에는 충분하다 ([Queue 생성 API](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues/create)).

### 시간 보장과 실행 제한

Google은 Cloud Tasks가 전달 자체는 보장하지만 전달 시각에는 강한 보장을 제공하지 않는다고 명시한다 ([Cloud Tasks 개요](https://docs.cloud.google.com/tasks/docs/dual-overview)).
Cloud Tasks 서비스 수준 협약의 99.95% 목표도 `CreateTask` 요청의 성공률을 대상으로 하고 예약 시각의 전달 지연은 대상으로 하지 않는다 ([Cloud Tasks 서비스 수준 협약](https://cloud.google.com/tasks/sla)).
`scheduleTime`은 예약 시도 시각을 마이크로초 단위로 저장하지만 저장 정밀도가 최대 지연 보장을 뜻하지 않는다 ([Task 자원](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks)).

HTTP 작업의 전달 제한은 15초부터 30분까지이고 기본값은 10분이다 ([Task 자원](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks)).
Cloud Run 서비스 요청 제한은 최대 60분이지만 Cloud Tasks가 호출하는 경로에서는 더 짧은 30분 전달 제한이 실질 상한이다 ([Cloud Run 요청 제한](https://docs.cloud.google.com/run/docs/configuring/request-timeout)).
5분 부재 확인에는 충분하므로 7분에서 10분 사이로 두 실행 제한을 맞출 수 있다.

### 재시도와 중복 실행

Cloud Tasks는 최소 한 번 전달을 제공하며 드물게 같은 작업이 여러 번 실행될 수 있으므로 실행기는 멱등해야 한다 ([Cloud Tasks 개요](https://docs.cloud.google.com/tasks/docs/dual-overview)).
2xx가 아닌 응답이나 전달 제한 안에 응답하지 못한 작업은 작업 대기열의 최소 및 최대 간격, 배수 증가 횟수, 최대 시도 횟수와 최대 재시도 기간에 따라 다시 시도된다 ([작업 대기열 재시도](https://docs.cloud.google.com/tasks/docs/configuring-queues)).
2026-08 현재 작업별 재시도 설정은 시험 제공 단계이므로, 운영에서는 종료 작업 전용 대기열 하나에 안정된 대기열 수준 재시도 정책을 적용하는 편이 안전하다 ([작업별 재시도](https://docs.cloud.google.com/tasks/docs/configure-retry-task)).
대상이 이미 없으면 성공으로 처리하고 예약 당시 식별자만 삭제해야 늦은 중복 실행이 안전하다.

### 비밀 값, 권한, 알림과 감사

Cloud Run은 API 키 같은 민감 값을 Secret Manager에 저장할 것을 권장한다 ([Cloud Run 비밀 값](https://docs.cloud.google.com/run/docs/configuring/services/secrets)).
종료 전용 Cloud Run 서비스 계정에 비밀 값 하나의 `roles/secretmanager.secretAccessor`만 부여하고, 작업 전달용 서비스 계정은 Cloud Run 서비스 하나의 호출 권한만 갖게 할 수 있다 ([Secret Manager 접근 관리](https://docs.cloud.google.com/secret-manager/docs/manage-access-to-secrets), [Cloud Run 서비스 계정](https://docs.cloud.google.com/run/docs/configuring/services/service-identity)).
Cloud Tasks는 OIDC 토큰을 붙여 비공개 Cloud Run 대상을 호출할 수 있다 ([Task HTTP 요청](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks)).

Cloud Tasks는 작업 시도 횟수와 응답 코드, 예약 시도와 실제 시도 사이의 지연 지표를 Cloud Monitoring으로 보내고 지표 경보를 만들 수 있다 ([Cloud Tasks 관찰](https://docs.cloud.google.com/tasks/docs/monitor)).
작업 대기열 로그의 `AttemptDispatch`와 `AttemptResponse`를 활성화하고 비정상 응답과 큰 전달 지연에 로그 기반 알림을 구성할 수 있다 ([Cloud Tasks 관찰](https://docs.cloud.google.com/tasks/docs/monitor), [로그 기반 알림](https://cloud.google.com/logging/docs/logs-based-metrics/charts-and-alerts)).
AWS 조합의 SQS 실패 보관함처럼 재시도 소진 사건을 별도 자원으로 보존하는 기본 기능은 없으므로, 실패 시도 로그와 Cloud Run의 명시적 실패 로그를 함께 경보해야 한다.

Cloud Tasks의 `GetTask` 자료 접근 감사 로그는 기본 비활성화되어 있으므로 명시적으로 켜야 한다 ([Cloud Tasks 감사 로그](https://docs.cloud.google.com/tasks/docs/audit-logging)).
`CreateTask`는 감사 로그를 만들지 않으므로, 예약 생성 증거에는 작업 대기열의 `CreateTask` 운영 로그를 별도로 활성화해 사용해야 한다 ([Cloud Tasks 관찰](https://docs.cloud.google.com/tasks/docs/monitor), [Cloud Tasks 감사 로그](https://docs.cloud.google.com/tasks/docs/audit-logging)).
Secret Manager의 `AccessSecretVersion`도 자료 접근 감사 로그이므로 키 접근 감사가 필요하면 함께 활성화한다 ([Secret Manager 감사 로그](https://docs.cloud.google.com/secret-manager/docs/audit-logging)).
Cloud Run은 요청, 컨테이너와 시스템 로그를 Cloud Logging에 자동 전송한다 ([Cloud Run 로그](https://docs.cloud.google.com/run/docs/logging)).
Cloud Logging이 임의의 Vast.ai API 키를 자동으로 가린다고 전제하지 말고, 원문 응답을 출력하지 않는 허용 필드 방식으로 로그를 만든다.

### 비용과 계정 준비

Cloud Tasks는 월 100만 작업 단위까지 무료이고 Cloud Run 요청 기반 요금은 월 200만 요청과 일정 CPU 및 메모리 사용량까지 무료다 ([Cloud Tasks 요금](https://cloud.google.com/tasks/pricing), [Cloud Run 요금](https://cloud.google.com/run/pricing)).
Secret Manager도 활성 비밀 값 판본 6개와 월 1만 접근까지 무료이므로 이 프로젝트의 종료 호출은 일반적으로 무료 사용량 안에 든다 ([Secret Manager 요금](https://cloud.google.com/secret-manager/pricing)).
무료 사용량만 쓰더라도 프로젝트는 활성 결제 계정에 연결되어 있어야 한다 ([Cloud 결제 상태](https://docs.cloud.google.com/billing/docs/how-to/verify-billing-enabled)).

최초 준비에는 Google Cloud 프로젝트, 활성 결제 계정, Cloud Tasks, Cloud Run, Secret Manager와 Logging API, 전용 작업 대기열, 비공개 Cloud Run 서비스, 두 서비스 계정, 감사 로그와 알림 경로가 필요하다.
AWS와 비슷한 준비가 들면서 공식 시간 보장은 더 약하므로 AWS 계정을 사용할 수 없을 때만 예비 후보로 남긴다.

## 최종 운영 권고

기본 독립 종료 예약은 다음 AWS 자원 묶음으로 고정한다.

- EventBridge Scheduler의 전용 예약 묶음과 단발 예약
- 최대 10분 제한의 종료 전용 Lambda 함수
- `instance_read`, `instance_write`만 가진 Vast.ai 종료 키와 Secrets Manager 비밀 값 하나
- Scheduler 전달 실패용 SQS 실패 보관함
- Lambda 실행 실패용 `OnFailure` 목적지
- Scheduler와 Lambda 오류 지표의 CloudWatch 경보
- 사용자가 확인한 SNS 전자우편 구독
- CloudTrail과 비밀 값을 제외한 구조화 Lambda 로그

실제 Vast.ai 유료 실행 전에 저비용 대상 하나로 다음 합격 검사를 수행한다.

1. `T - 5분` 단발 예약을 만들고 반환된 ARN을 원격 실행 장부에 기록한다.
2. `GetSchedule`로 이름, ARN, `at(...)`, UTC, 상태, 대상 ARN, 입력 대상, 재시도, 실패 보관함과 자동 삭제를 대조한다.
3. 로컬 장비와 대상 계산 자원의 관찰 프로세스를 모두 끈 상태에서 Lambda가 독립 실행되는지 확인한다.
4. 존재하는 시험 자원 삭제와 삭제 뒤 목록 부재를 확인한다.
5. 같은 예약 입력을 수동으로 두 번 실행해 두 번째 실행이 대상 부재를 성공으로 처리하는지 확인한다.
6. Scheduler 대상 권한을 일부러 거부해 Scheduler 실패 보관함과 사용자 알림을 확인한다.
7. Lambda에서 의도적으로 오류를 내 Scheduler 실패 경로와 별개인 Lambda `OnFailure`와 사용자 알림을 확인한다.
8. CloudTrail, CloudWatch 로그와 원격 실행 장부에 비밀 값 없이 필요한 감사 필드가 남는지 확인한다.

이 전체 검사를 통과하기 전에는 AWS 조합을 설정한 것으로 보지 않는다.
예약 등록이나 재조회, 두 실패 알림 중 하나라도 확인할 수 없으면 Vast.ai 작업을 시작하지 않고 [Vast.ai 원격 자원 제어 경로와 자격 증명 규약 확정](https://github.com/tmheo/predicting-smartphone-addiction/issues/125)의 결정대로 Runpod으로 전환한다.

## 조사 범위

이 문서는 2026-08-15에 공개된 GitHub, Amazon Web Services와 Google Cloud의 공식 문서와 공식 요금표만 사용했다.
실제 AWS나 Google Cloud 계정, 결제 수단, API 키와 Vast.ai 유료 자원은 만들거나 호출하지 않았다.
Azure Functions의 반복 시간 트리거와 Google Cloud Scheduler의 반복 cron은 단발 예약 자원이 아니라서 상세 후보 비교에서 제외했다.
