# MLflow 3.15.1 실험 관찰 기능 조사

[MLflow 3.15에서 진행 중 실험을 기록하고 표시하는 기능과 한계 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/38)의 조사 결과다.
질문은 학습 전에 만든 실행을 로컬 MLflow UI에서 진행 중으로 표시하고 실제 경과 시간, 단계와 fold 진행률, 60초 생존 신호, 실패 상태, 실행 중 갱신되는 기록, 완료 후 텍스트 로그와 결과 표 및 그림을 제공할 때 사용할 수 있는 기능과 한계를 확인하는 것이다.
MLflow 공식 문서, MLflow 저장소의 `v3.15.1` 태그 소스, 이 저장소에 설치된 MLflow 3.15.1만 근거로 삼았다.

## 결론

MLflow 실행을 데이터 적재와 학습이 끝난 뒤가 아니라 설정 검증 직후에 시작하면 실행이 즉시 `RUNNING` 상태로 저장되고 로컬 UI의 실행 목록에서 관찰할 수 있다.
MLflow UI의 기본 `Duration`은 종료 시각에서 시작 시각을 뺀 값이며, 진행 중 실행은 종료 시각이 없어서 `Duration`이 비어 있다.
따라서 실제 경과 시간과 60초 생존 신호는 단조 증가하는 사용자 지표로 기록하고, 현재 단계처럼 문자열인 상태는 갱신 가능한 태그로 기록해야 한다.
실행 중 로그는 MLflow 산출물에 덧붙이는 방식보다 로컬 `run.log`에 계속 이어 쓰고 `tail -f`로 읽게 하는 편이 적합하다.
완료 또는 정상적으로 처리 가능한 실패 때 그 파일을 MLflow 산출물로 올리면 UI에서 바로 읽을 수 있다.
예외는 `FAILED`로 기록할 수 있지만 `SIGKILL` 같은 강제 종료는 MLflow가 감지하지 못하므로 실행이 `RUNNING`으로 영구히 남는다.
이 경우를 판별하고 `KILLED`로 정리하려면 마지막 생존 신호와 별도의 정리 절차가 필요하다.
Parquet는 원본 보존에는 적합하지만 일반 산출물 화면에서 미리 볼 수 없으므로 사람이 확인할 결과에는 CSV와 PNG를 함께 제공해야 한다.

## 검증된 사실

### 실행 생성, 표시와 시간 의미

`mlflow.start_run()`은 새 실행을 만들 때 추적 저장소의 `create_run()`을 즉시 호출하며, 만들어진 실행의 상태는 `RUNNING`, 시작 시각은 생성 시각, 종료 시각은 `None`이다 ([MLflow 3.15.1 `start_run` 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/fluent.py#L450-L716), [실행 생성 시각 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/_tracking_service/client.py#L158-L191), [SQL 저장소 실행 생성 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/store/tracking/sqlalchemy_store.py#L949-L998)).
따라서 학습 전에 `start_run()`을 호출하면 학습이 끝나기 전에도 검색 API와 UI 실행 목록이 해당 실행을 반환한다.
MLflow UI는 `startTime`과 `endTime`이 모두 있을 때만 두 값의 차이를 `Duration`으로 표시하고, 둘 중 하나가 없으면 빈 값을 표시한다 ([MLflow 3.15.1 UI 시간 계산 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/common/utils/Utils.tsx#L237-L250), [진행 중 실행 UI 시험](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/run-page/RunPage.test.tsx#L343-L394)).
`MlflowClient.set_terminated()`은 종료 시각을 지정하지 않으면 현재 시각을 기록하므로 완료된 실행의 기본 `Duration`은 MLflow 실행 생성부터 종료 처리까지의 벽시계 시간이다 ([MLflow 3.15.1 종료 처리 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/_tracking_service/client.py#L788-L810)).
현재 저장소는 데이터 적재와 전체 교차 검증을 마친 뒤 `tracking.log_run()` 안에서 MLflow 실행을 시작하므로 UI에 보인 `445ms`는 학습 시간이 아니라 결과를 기록하기 위해 실행을 열어 둔 시간이다 ([현재 실험 실행 흐름](../../src/pipeline/run.py), [현재 MLflow 기록 흐름](../../src/pipeline/tracking.py)).

실험 목록의 자동 새로고침은 기본으로 켜져 있고, 해당 화면이 활성 상태일 때 30초마다 실행 검색을 다시 수행한다 ([MLflow 3.15.1 목록 새로고침 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/experiment-page/hooks/useExperimentRunsAutoRefresh.tsx#L11-L133), [30초 간격 정의](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/experiment-page/utils/experimentPage.fetch-utils.ts#L52-L52), [기본값 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/experiment-page/models/ExperimentPageUIState.tsx#L195-L212)).
실행 상세 화면의 개요 자료에는 자동 조회 간격이 없지만, 지표 차트의 자동 새로고침은 기본으로 켜져 있고 30초 간격으로 지표 이력을 다시 가져온다 ([실행 상세 조회 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/run-page/hooks/useGetRunQuery.tsx#L108-L135), [지표 새로고침 간격 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/utils/MetricsUtils.ts#L359-L359), [지표 차트 기본값 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/run-page/RunViewMetricCharts.tsx#L340-L365)).
사용자가 수동으로 새로고침해도 모든 최신 실행 자료를 다시 읽을 수 있으므로 별도 실행 주소나 별도 실시간 화면은 필요하지 않다.

### 진행 단계, 지표 이력과 생존 신호

`mlflow.log_metric()`은 숫자 값과 함께 기록 시각과 정수 `step`을 저장하고, 같은 이름의 여러 기록은 `MlflowClient.get_metric_history()`로 모두 조회할 수 있다 ([MLflow 3.15.1 지표 기록 API](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/fluent.py#L1145-L1235), [지표 이력 API](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/client.py#L509-L565)).
실행 목록과 개요가 보여 주는 최신 지표는 가장 나중에 호출된 값이 아니라 `step`, 기록 시각, 값의 순서로 가장 큰 항목이므로 진행 지표의 `step`은 반드시 단조 증가해야 한다 ([MLflow 3.15.1 최신 지표 선택 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/store/tracking/sqlalchemy_store.py#L1385-L1504)).
태그는 문자열을 저장할 수 있고 같은 키에 다시 기록하면 현재 값이 바뀌지만, 이전 값의 이력은 보존하지 않는다 ([MLflow 3.15.1 태그 API](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/fluent.py#L1079-L1117), [태그 갱신 저장소 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/store/tracking/sqlalchemy_store.py#L1978-L1996)).
따라서 `data_load`, `seed_1`, `fold_2`, `evaluation`, `artifacts` 같은 현재 단계는 태그에 적합하고, 경과 시간과 진행률처럼 시간에 따른 변화를 보려는 값은 지표에 적합하다.

MLflow Tracking에는 사용자 실행을 위한 별도 heartbeat 필드, 기한이 지난 `RUNNING` 실행 판정, 자동 상태 정리 기능이 없다.
MLflow 3.15.1의 추적 API와 저장소 소스에서 heartbeat 기반 실행 상태 전이는 제공하지 않으며, 실행 상태 전이는 명시적인 `set_terminated()` 또는 실행 문맥 종료로만 일어난다 ([실행 상태 API](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/client.py#L3784-L3830), [실행 상태 종류](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/entities/run_status.py#L4-L36)).
그러므로 60초 생존 신호는 사용자 지표와 태그로 구현해야 한다.

시스템 지표 기록을 켜면 CPU, 메모리, 디스크, 네트워크와 가능한 경우 GPU 사용량을 기본 10초 간격으로 지표 이력에 남기고 UI에서 볼 수 있다 ([MLflow 3.15.1 시스템 지표 문서](https://mlflow.org/docs/3.15.1/ml/tracking/system-metrics/), [MLflow 3.15.1 감시기 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/system_metrics/system_metrics_monitor.py#L19-L154)).
현재 프로젝트 환경에는 필요한 `psutil`이 설치되어 있어 이 기능을 바로 켤 수 있다.
다만 시스템 지표는 자원 사용량일 뿐 현재 단계나 작업 단위를 나타내지 않으므로, 60초 생존 신호를 대신하는 필수 기록으로 삼기에는 의미가 불분명하다.

### 로그와 산출물의 실행 중 갱신

`mlflow.log_text()`와 `mlflow.log_artifact()`는 실행 ID를 받아 실행 중에도 텍스트와 파일을 올릴 수 있다 ([MLflow 3.15.1 Python API 문서](https://mlflow.org/docs/3.15.1/api_reference/python_api/mlflow.html#mlflow.log_text), [텍스트 기록 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/client.py#L2904-L2942)).
로컬 산출물 저장소에서 같은 경로로 파일을 다시 올리면 숨은 임시 파일에 먼저 쓴 뒤 `os.replace()`로 기존 파일 전체를 원자적으로 바꾼다 ([MLflow 3.15.1 로컬 산출물 저장소 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/store/artifact/local_artifact_repo.py#L90-L170)).
`log_text()`를 같은 경로에 여러 번 호출해도 내용이 덧붙지 않고 이전 파일 전체가 새 내용으로 교체된다.
설치된 3.15.1에서 실행 중 `run.log`에 `first`를 올린 뒤 같은 경로에 `second`를 올리는 재현 시험도 최종 내용이 `second`만 남는 것을 확인했다.
따라서 실행 로그를 줄마다 MLflow로 보내는 방식은 적합하지 않다.
로컬 파일에 계속 이어 쓰고 필요할 때 전체 파일의 시점을 같은 산출물 경로로 교체할 수는 있지만, 산출물 화면 자체에는 자동 새로고침 절차가 없으므로 실행 중 열어 둔 내용이 자동으로 따라오지는 않는다 ([MLflow 3.15.1 산출물 화면 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/ArtifactPage.tsx#L113-L176)).

`mlflow.log_table()`은 같은 JSON 또는 Parquet 경로가 이미 있으면 기존 표를 내려받아 새 행을 붙인 뒤 파일 전체를 다시 올린다 ([MLflow 3.15.1 표 기록 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/client.py#L3342-L3524)).
이 기능은 구조화된 표의 누적에는 쓸 수 있지만 자유 형식 로그에 맞지 않고 매번 전체 표를 다시 쓰므로 60초 로그 전달 수단으로는 부적합하다.

### 완료, 실패, 중단과 남은 `RUNNING` 실행

`with mlflow.start_run()` 문맥은 본문이 정상 종료되면 `FINISHED`, 예외가 문맥 밖으로 나가면 `FAILED`로 종료한다 ([MLflow 3.15.1 `ActiveRun` 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/fluent.py#L375-L393)).
이 자동 처리는 상태와 종료 시각만 기록하며 오류 종류, 오류 메시지와 traceback을 산출물이나 태그로 자동 저장하지 않는다.
따라서 실패 원인을 보존하려면 실행 코드가 예외를 잡아 로컬 로그와 traceback을 먼저 올린 뒤 `FAILED`로 종료해야 한다.

활성 실행을 문맥 관리자 없이 열어 둔 채 Python이 정상 종료되면 MLflow의 `atexit` 처리가 기본 상태인 `FINISHED`로 종료를 시도한다 ([MLflow 3.15.1 종료 시 처리 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/fluent.py#L719-L769)).
반대로 프로세스가 `SIGKILL`로 끝나면 이 코드가 실행될 기회가 없다.
설치된 MLflow 3.15.1과 임시 SQLite 저장소를 사용한 재현 시험에서 실행 생성과 생존 지표 기록 직후 자식 프로세스를 `SIGKILL`로 끝냈더니 상태는 `RUNNING`, 종료 시각은 `None`으로 그대로 남았다.
MLflow 서버나 UI는 이 실행을 자동으로 `KILLED` 또는 `FAILED`로 바꾸지 않았다.
남은 실행은 외부 절차가 `MlflowClient.set_terminated(run_id, status="KILLED")`를 호출해야 종료 상태와 종료 시각을 얻는다 ([MLflow 3.15.1 종료 API](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/tracking/client.py#L3784-L3830)).

### 사람이 바로 읽을 수 있는 결과 산출물

MLflow 3.15.1 UI의 일반 산출물 미리보기는 PNG를 포함한 그림, `.txt`와 `.log`를 포함한 텍스트, Markdown, HTML, PDF, CSV와 TSV 등을 지원하지만 Parquet는 지원 확장자에 들어 있지 않다 ([MLflow 3.15.1 지원 확장자 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/common/utils/FileUtils.ts#L28-L68), [미리보기 선택 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/artifact-view-components/ShowArtifactPage.tsx#L55-L127)).
CSV와 TSV 미리보기는 첫 500행을 읽어 정렬 가능한 표로 보여 준다 ([MLflow 3.15.1 표 미리보기 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/artifact-view-components/ShowArtifactTableView.tsx#L31-L131)).
HTML은 격리된 iframe 안에서 표시되고, 그림은 UI 안에서 바로 표시된다 ([MLflow 3.15.1 HTML 미리보기 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/artifact-view-components/ShowArtifactHtmlView.tsx#L46-L104), [그림 미리보기 선택 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/artifact-view-components/ShowArtifactPage.tsx#L87-L105)).
일반 미리보기 파일은 50MiB를 넘으면 UI가 내용을 표시하지 않는다 ([MLflow 3.15.1 미리보기 크기 제한](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/artifact-view-components/ShowArtifactPage.tsx#L36-L78)).
`mlflow.log_table()`에 JSON을 사용하면 MLflow의 표 보기 기능을 쓸 수 있지만, Parquet 표는 3.15.1 UI의 JSON 기반 표 보기에서 직접 해석되지 않는다 ([MLflow 3.15.1 기록 표 보기 소스](https://github.com/mlflow/mlflow/blob/v3.15.1/mlflow/server/js/src/experiment-tracking/components/artifact-view-components/ShowArtifactLoggedTableView.tsx#L360-L470)).
그러므로 `feature_importance.parquet` 원본을 유지하면서 정렬된 CSV와 상위 특성 PNG를 함께 기록하는 구성이 가장 확실하다.
하나의 요약 화면이 나중에 필요해지면 HTML도 추가할 수 있지만, 현재 합의한 CSV와 PNG를 확인하는 데 필수는 아니다.

## 권장 기록 규약

다음 내용은 위에서 검증한 기능과 한계를 바탕으로 한 권장안이며 MLflow가 자동으로 제공하는 동작이 아니다.

### 실행 수명

- 설정 파일을 읽고 검증한 직후 `with mlflow.start_run(run_name=cfg.name)`을 열고, 데이터 적재부터 모든 시드와 fold 학습, 평가, 산출물 기록을 그 문맥 안에서 수행한다.
- 이 경계로 바꾸면 종료 뒤 기본 `Duration`이 사용자가 생각하는 전체 실험 시간과 일치한다.
- 실행 중 기본 `Duration`은 비어 있으므로 별도의 경과 시간 지표를 사용한다.

### 진행 상태와 60초 생존 신호

- `progress.stage` 태그에는 `data_load`, `feature_build`, `training`, `evaluation`, `artifacts` 가운데 현재 단계를 기록한다.
- `progress.seed_index`, `progress.seed_total`, `progress.fold_index`, `progress.fold_total`, `progress.completed_units`, `progress.total_units`는 숫자 지표로 기록한다.
- `progress.elapsed_seconds`는 실행 시작에 사용한 단조 시계 기준 경과 초를 값으로 삼아 단계 경계와 최소 60초마다 기록한다.
- `progress.elapsed_seconds`의 `step`은 실행 안에서 0부터 하나씩 늘리는 생존 신호 순번으로 사용한다.
- `progress.last_activity_at` 태그는 같은 시점의 UTC ISO 8601 시각으로 갱신해 목록 화면에서도 마지막 활동을 읽을 수 있게 한다.
- fold가 끝날 때 기존 fold AUC도 즉시 기록해 최종 기록 시점까지 기다리지 않는다.
- 시스템 지표는 CPU, 메모리와 GPU 사용량이 실제로 유용할 때 선택적으로 켜고, 진행 판정의 필수 신호로 사용하지 않는다.

### 로컬 로그와 MLflow 보존본

- 실행마다 충돌하지 않는 로컬 경로에 `run.log`를 만들고 각 줄을 즉시 flush해 실행 중 `tail -f`로 읽을 수 있게 한다.
- 단계 시작과 종료, 시드와 fold 번호, fold AUC, 단계별 소요 시간, 경고, 오류와 전체 traceback을 같은 파일에 기록한다.
- 정상 완료 또는 처리 가능한 실패 때 `logs/run.log`로 MLflow에 한 번 올리고, 실행 상태를 바꾸기 전에 업로드를 마친다.
- 실행 중 MLflow에서 로그 시점이 꼭 필요해질 때만 전체 파일을 같은 경로에 주기적으로 교체하고, 이를 터미널식 실시간 스트리밍으로 간주하지 않는다.
- `SIGKILL`에서는 마지막 업로드가 불가능하므로 로컬 로그는 실행 정리와 별개로 남겨야 한다.

### 실패와 강제 종료 정리

- 일반 예외는 실패 단계, 오류 종류와 오류 메시지를 태그로 남기고 전체 traceback이 포함된 로컬 로그를 올린 뒤 `FAILED`로 종료한다.
- 사용자 중단이나 운영체제 종료 신호를 처리할 수 있는 경우에는 로그를 올린 뒤 `KILLED`로 종료한다.
- 다음 실험 시작 전 또는 별도 점검 명령에서 오래된 `RUNNING` 실행을 찾고, 마지막 활동 시각이 기준을 넘었으며 해당 로컬 프로세스가 없음을 확인한 뒤 `KILLED`로 정리한다.
- 시간만으로 자동 정리하면 실제로 오래 계산 중인 fold를 잘못 종료할 수 있으므로 마지막 활동 시각과 프로세스 존재 여부를 함께 확인한다.

### 결과 산출물

- 원본 자료인 `feature_importance.parquet`는 현재 스키마 그대로 유지한다.
- 사람이 읽는 `feature_importance_summary.csv`에는 특성별 평균 gain, 표준편차, 순위와 플라시보 대비 상태를 평균 gain 내림차순으로 기록한다.
- `feature_importance_top30.png`에는 평균 gain 상위 30개와 플라시보 기준을 알아보기 쉽게 표시한다.
- `run.log`, CSV와 PNG는 모두 50MiB 아래로 유지해 MLflow UI 미리보기가 확실히 동작하게 한다.

## 조사로 구체화된 다음 결정

이 조사로 진행 기록과 실패 처리 규약을 하나의 결정 티켓으로 구체화할 수 있다.
그 티켓에서는 정확한 태그와 지표 이름, 총 작업 단위 계산법, 로컬 로그 경로, 생존 신호를 보내는 실행 주체, `FAILED`와 `KILLED` 구분, 오래된 실행 판정 시간을 확정하면 된다.
그 결정 뒤 구현 인수 조건과 이전 실행 기록의 호환 범위를 별도 결정으로 구체화할 수 있다.

## 출처

- [MLflow 3.15.1 실험 추적 문서](https://mlflow.org/docs/3.15.1/ml/tracking/)
- [MLflow 3.15.1 Python API 문서](https://mlflow.org/docs/3.15.1/api_reference/python_api/mlflow.html)
- [MLflow 3.15.1 시스템 지표 문서](https://mlflow.org/docs/3.15.1/ml/tracking/system-metrics/)
- [MLflow `v3.15.1` 소스](https://github.com/mlflow/mlflow/tree/v3.15.1)
