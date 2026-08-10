# 실험 기록과 비교 방법 조사

GitHub issue [#14](https://github.com/tmheo/predicting-smartphone-addiction/issues/14)의 조사 결과다.
질문은 "로컬 uv 기반 단독 개발 Kaggle 프로젝트에서 실험을 기록하고 비교하는 방법으로 무엇이 적합한가"이다.
공식 문서, 소스 저장소, GitHub API 등 1차 출처만 근거로 삼았고 각 주장에 출처 링크를 달았다.

## 요구 사항 정리

각 실험 실행에서 다음을 남겨야 한다.

- 설정: feature 목록, 모델 파라미터, fold 정의.
- 점수: fold별 CV ROC AUC와 전체 OOF ROC AUC.
- 예측 파일: OOF 예측(약 75만 행)과 테스트 예측(약 25만 행).
- 계보: 입력 파일 해시, 행 단위 fold ID, 코드 커밋.

제약 조건은 단독 개발자, 로컬 실행, uv 기반 Python 3.13, 노트북 중심 저장소다.
나중에 앙상블 코드가 여러 실행의 OOF를 `id` + fold ID 기준으로 정렬해 소비할 수 있어야 한다는 점이 특히 중요하다.
`docs/research/code-notebook-insights.md`의 34번 노트북(OOF Signal Forge)이 행별 fold ID, 구성원별 OOF, 파일 해시를 함께 저장하는 선례다.

## 후보별 조사

### MLflow (로컬 file store)

- 서버나 데이터베이스를 따로 설정하지 않으면 MLflow tracking은 기본값으로 로컬 `mlruns` 디렉터리에 기록한다 ([MLflow Tracking 문서](https://mlflow.org/docs/latest/ml/tracking/)).
- artifact도 기본값은 로컬 `mlruns` 디렉터리이고, 필요하면 S3나 Azure Blob 같은 원격 저장소로 바꿀 수 있다 ([MLflow Tracking 문서](https://mlflow.org/docs/latest/ml/tracking/)).
- 메타데이터 저장소를 SQLite 등 SQLAlchemy 호환 데이터베이스로 바꿀 수도 있는데, 이 경우에도 로컬 파일 하나(`mlflow.db`)면 되므로 서버는 필요 없다 ([MLflow Tracking 문서](https://mlflow.org/docs/latest/ml/tracking/)).
- 각 run은 파라미터, 지표, 시작과 종료 시각 같은 메타데이터와 함께 출력 파일(artifact)을 기록한다 ([MLflow Tracking 문서](https://mlflow.org/docs/latest/ml/tracking/)).
- git 저장소 안에서 실행하면 시스템 태그 `mlflow.source.git.commit`에 커밋 해시가 자동으로 기록된다 ([MLflow Tracking API 문서](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)).
- `mlflow.data` 모듈은 데이터셋마다 내용 기반의 고유 digest(해시)를 자동 계산하고, `mlflow.log_input()`으로 run에 기록한다 ([MLflow Dataset 문서](https://mlflow.org/docs/latest/ml/dataset/)).
- `mlflow.search_runs()`는 조건에 맞는 run들을 pandas DataFrame으로 돌려주고, `metrics.auc > 0.97 AND params.model LIKE 'lgbm%'` 같은 SQL 유사 필터를 지원한다 ([MLflow Search Runs 문서](https://mlflow.org/docs/latest/ml/search/search-runs/)).
- 필터는 `metrics.` `params.` `tags.` `attributes.` `datasets.` 접두사를 지원하며, `AND`만 되고 `OR`는 지원하지 않는다 ([MLflow Search Runs 문서](https://mlflow.org/docs/latest/ml/search/search-runs/)).

정리하면 계정, 서버, 네트워크 없이 pip 패키지 하나로 요구 사항의 기록 항목을 전부 담을 수 있고, 비교는 로컬 UI(`mlflow ui`)와 pandas DataFrame 양쪽으로 할 수 있다.

### Weights & Biases (클라우드 SaaS)

- W&B는 계정을 만들고 클라우드 서비스에 기록하는 구조이고, artifact는 `wandb.Artifact`로 만들어 `log_artifact()`로 올린 뒤 나중에 `use_artifact().download()`로 내려받는다 ([W&B Artifacts 문서](https://docs.wandb.ai/guides/artifacts/)).
- 무료 요금제의 저장 용량은 월 5GB다 ([W&B 요금 안내](https://wandb.ai/site/pricing/)).
- 오프라인 모드(`wandb offline`)를 켜면 지표와 artifact를 로컬 디스크에 쓰고 서버 동기화를 하지 않으며, 나중에 `wandb sync`로 올릴 수 있다 ([W&B CLI 문서](https://docs.wandb.ai/models/ref/cli/wandb-offline)).

UI 품질은 좋지만 이 저장소의 조건과는 안 맞는 부분이 많다.
OOF 75만 행 + 테스트 25만 행 예측을 실행마다 올리면 실행당 수십 MB가 쌓여 무료 한도 5GB가 수십~수백 회 실행 안에 소진된다.
앙상블 코드는 클라우드에서 파일을 다시 내려받아야 하므로, 어차피 로컬에서 만들어 로컬에서 소비하는 파일에 네트워크 왕복과 계정 의존이 끼어든다.
서비스 종속이 생기고 기록이 로컬 git 저장소 밖에 남는다는 점도 단독 로컬 개발 조건과 어긋난다.

### Aim (로컬 자체 호스팅 추적기)

- Aim은 수만 건의 실행을 다루도록 설계된 자체 호스팅 오픈 소스 실험 추적기로, 비교용 UI와 프로그래매틱 조회 SDK를 제공한다 ([Aim 저장소](https://github.com/aimhubio/aim)).
- 저장소 활동은 이어지고 있지만 마지막 정식 릴리스는 v3.29.1(2025년 5월)로 릴리스 주기가 뜸하다 ([Aim releases](https://github.com/aimhubio/aim/releases)).

지표와 파라미터 추적 자체는 로컬에서 잘 되지만, 핵심 설계가 지표 시계열 비교에 맞춰져 있어 대용량 예측 파일 관리가 주 기능이 아니다.
MLflow 대비 커뮤니티와 문서 규모가 작고, 이 저장소의 요구 사항에서 MLflow보다 나은 축이 없어 굳이 선택할 이유가 없다.

### sacred (설정 캡처 프레임워크)

- sacred는 실험의 설정, 조직, 기록, 재현을 돕는 프레임워크이고 주 저장 경로가 MongoDB observer이며, 시각화는 Omniboard 같은 별도 프론트엔드에 의존한다 ([sacred 저장소](https://github.com/IDSIA/sacred)).
- 마지막 릴리스는 0.8.7(2024년 11월)이다 ([sacred releases](https://github.com/IDSIA/sacred/releases)).

단독 로컬 개발에 MongoDB와 별도 대시보드를 얹는 것은 운영 부담이 과하고, 개발 활력도 낮아 후보에서 제외한다.

### DVC (데이터 버전 관리 + 실험 관리)

- DVC 실험 관리는 실험을 `.git/refs/exps` 아래의 숨은 git 참조로 추적하고, 기본적으로 로컬에 저장되며 원격 저장소 없이 동작한다 ([DVC Experiment Management 문서](https://doc.dvc.org/user-guide/experiment-management)).
- `dvc exp show`로 실험 간 파라미터, 지표, 플롯을 비교하고, 지표는 YAML/JSON/CSV 같은 구조화 파일에서 읽는다 ([DVC Experiment Management 문서](https://doc.dvc.org/user-guide/experiment-management)).
- 대용량 데이터는 캐시와 원격 저장소 연동으로 관리한다 ([DVC Experiment Management 문서](https://doc.dvc.org/user-guide/experiment-management)).

데이터 파일 버전 관리 자체는 강점이지만, 실험 관리가 `dvc.yaml` 파이프라인 정의를 전제로 하므로 노트북 중심 저장소를 파이프라인 구조로 재편해야 효과가 난다.
지금 단계에서는 도입 비용이 얻는 것보다 크다.
나중에 훈련 코드를 스크립트 파이프라인으로 옮기는 시점이 오면 재검토할 가치는 있다.

### 파일 기반 자체 규약 (JSON/CSV + git)

외부 도구 없이 실행마다 디렉터리를 만들고 `config.json`, `metrics.json`, `oof.parquet`, `test_pred.parquet`을 저장하는 방식이다.
의존성이 없고 모든 파일이 그대로 보이므로 앙상블 코드가 소비하기는 가장 쉽다.
반면 단점이 뚜렷하다.

- 수십 개 실행을 걸러 보고 정렬하는 조회 계층을 결국 직접 만들어야 한다.
- 규약을 강제하는 장치가 없어 항목 누락이나 이름 불일치가 시간이 지나며 쌓인다.
- git 커밋 해시, 입력 파일 해시 기록도 전부 수작업 코드다.

즉 MLflow 로컬 file store가 공짜로 주는 것(기록 스키마, UI, DataFrame 조회, git 커밋 자동 태그)을 손으로 다시 만드는 셈이다.

## 비교 축별 정리

| 축 | MLflow(로컬) | W&B | 파일 자체 규약 | Aim | sacred | DVC |
|---|---|---|---|---|---|---|
| 대용량 OOF/테스트 예측 | 로컬 `mlruns`에 저장, 경로로 바로 소비 | 클라우드 업로드 후 다운로드, 무료 5GB/월 한도 | 로컬 파일 그대로 | 주 기능 아님 | MongoDB 중심 | 캐시로 관리 가능 |
| 운영 부담 | 패키지 설치만, 서버/계정 불필요 | 계정 필수, 서비스 종속 | 없음, 대신 직접 구현 | 패키지 + 로컬 UI 서버 | MongoDB 필요 | 파이프라인 재편 필요 |
| 재현성(설정/커밋/해시) | params + git 커밋 자동 태그 + dataset digest | 지원 | 전부 수작업 | params 지원 | 설정 캡처가 강점 | git 기반이라 강함 |
| 수십 실행 조회성 | UI + `search_runs()` DataFrame 필터 | UI 강함 | 직접 구현 | UI + 조회 SDK | 별도 프론트엔드 필요 | `dvc exp show` |

## 권장안: MLflow 로컬 file store + 예측 파일 artifact 규약

이 저장소에는 MLflow를 기본값 그대로(로컬 `mlruns` file store) 쓰는 것을 권장한다.
근거는 다음과 같다.

- 요구 사항의 네 가지 기록 항목(설정, fold별/전체 점수, 예측 파일, 계보)이 params, metrics, artifacts, tags/digest에 1:1로 대응한다.
- 서버, 계정, 네트워크가 전혀 필요 없어 단독 로컬 개발의 운영 부담이 사실상 0이다 ([MLflow Tracking 문서](https://mlflow.org/docs/latest/ml/tracking/)).
- 예측 파일이 로컬 디스크의 `mlruns` 아래에 그대로 남으므로, 앙상블 코드가 네트워크 없이 경로만으로 소비할 수 있다.
- 수십 개 실행 비교는 `mlflow ui`로 훑어보고, 앙상블 후보 선별은 `mlflow.search_runs()`가 돌려주는 pandas DataFrame으로 코드에서 처리한다 ([MLflow Search Runs 문서](https://mlflow.org/docs/latest/ml/search/search-runs/)).
- 기록은 전부 로컬 파일이므로 서비스 종속이 없고, 최악의 경우에도 `mlruns` 디렉터리를 직접 읽으면 된다.

W&B는 무료 한도와 네트워크/계정 의존 때문에, Aim과 sacred는 이점 없이 부담만 더해서, DVC는 파이프라인 재편 비용 때문에 제외했다.
파일 자체 규약은 MLflow가 이미 주는 것을 다시 만드는 일이라 제외했다.

### 구체 규약

도입은 `uv add mlflow` 하나로 끝나고, `.gitignore`에 `mlruns/`를 추가한다.
실행마다 다음을 기록한다.

- params: feature 목록(정렬된 문자열), 모델 이름과 하이퍼파라미터, fold 수와 분할 시드.
- metrics: `auc_fold_0` ~ `auc_fold_4`와 `auc_oof`.
- artifacts: `config.json`(params 원본 전체), `oof.parquet`, `test_pred.parquet`.
- 계보: `mlflow.data.from_pandas()`의 digest를 `mlflow.log_input()`으로 기록하거나, 입력 CSV의 sha256을 태그로 남긴다 ([MLflow Dataset 문서](https://mlflow.org/docs/latest/ml/dataset/)).

행 단위 fold ID는 별도 파일로 두지 말고 OOF 파일 스키마에 포함한다.

- `oof.parquet`: `id`, `fold`, `pred` 세 컬럼.
- `test_pred.parquet`: `id`, `pred` 두 컬럼.

이렇게 하면 앙상블 코드가 파일 하나로 fold 정렬 검증과 예측 병합을 함께 할 수 있어, 34번 노트북(OOF Signal Forge)의 계보 관리 선례를 그대로 따르게 된다.
75만 행 규모에서는 CSV보다 parquet이 용량과 읽기 속도 모두 유리하므로 예측 파일 형식은 parquet으로 통일한다.

코드 커밋 기록에는 주의점이 하나 있다.
`mlflow.source.git.commit` 자동 태그는 git 저장소에서 실행될 때 붙는데 ([MLflow Tracking API 문서](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)), 노트북 실행 환경에서는 소스 인식이 어긋날 수 있고 커밋하지 않은 변경이 있으면 해시가 실제 코드를 대표하지 못한다.
그래서 실행 시작 시 `git rev-parse HEAD`와 작업 트리의 변경 여부를 직접 태그(`git_commit`, `git_dirty`)로 남기는 코드를 공통 헬퍼에 넣고, `git_dirty`가 참인 실행은 앙상블 후보에서 제외하는 관행을 권장한다.

앙상블 시 소비 절차는 다음과 같다.

1. `mlflow.search_runs(filter_string="metrics.auc_oof > 0.9 AND tags.git_dirty = 'False'")`로 후보 run을 DataFrame으로 뽑는다.
2. 각 run의 artifact 경로에서 `oof.parquet`을 읽어 `id`로 병합하고, `fold` 컬럼이 모든 구성원에서 일치하는지 검증한다.
3. 채택한 blend의 구성원 run_id 목록을 새 run의 params로 기록해 앙상블 자체의 계보도 남긴다.

기록량이 늘어 file store 조회가 느려지면 tracking URI를 `sqlite:///mlflow.db`로 바꾸는 로컬 업그레이드 경로가 있다 ([MLflow Tracking 문서](https://mlflow.org/docs/latest/ml/tracking/)).
이 전환도 로컬 파일 하나 추가일 뿐이므로 지금 결정을 되돌릴 필요는 없다.

## 출처

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Tracking API (시스템 태그)](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)
- [MLflow Search Runs](https://mlflow.org/docs/latest/ml/search/search-runs/)
- [MLflow Dataset Tracking](https://mlflow.org/docs/latest/ml/dataset/)
- [W&B Artifacts](https://docs.wandb.ai/guides/artifacts/)
- [W&B 요금 안내](https://wandb.ai/site/pricing/)
- [W&B 오프라인 모드 CLI](https://docs.wandb.ai/models/ref/cli/wandb-offline)
- [Aim 저장소](https://github.com/aimhubio/aim)
- [sacred 저장소](https://github.com/IDSIA/sacred)
- [DVC Experiment Management](https://doc.dvc.org/user-guide/experiment-management)
