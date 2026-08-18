# Predicting Smartphone Addiction

Kaggle [Playground Series - Season 6 Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8) 참가 레포.

## 대회 개요

- **과제**: 스마트폰 중독 여부(`addicted_label`)의 확률을 예측하는 이진 분류
- **평가 지표**: ROC AUC
- **일정**: 2026-08-01 ~ 2026-08-31 23:59 UTC
- **제출 형식**: `id,addicted_label` 헤더에 각 테스트 id별 예측 확률

## 데이터

| 파일 | 크기 | 설명 |
| --- | --- | --- |
| `train.csv` | 691,369행 x 14컬럼 | 학습 데이터, 타깃 `addicted_label` 포함 |
| `test.csv` | 296,302행 x 13컬럼 | 예측 대상 |
| `sample_submission.csv` | - | 제출 형식 예시 |

피처는 수치형 9개(age, daily_screen_time_hours, social_media_hours, gaming_hours, work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day, weekend_screen_time)와 범주형 3개(gender, stress_level, academic_work_impact)로 구성된다.
타깃 양성 비율은 약 70.9%이고, 대부분의 피처에 결측치가 있다(최대 약 19%).
데이터는 실제 데이터에서 합성 생성된 것으로, `data/` 아래에 두며 git에는 커밋하지 않는다.

## 개발 환경

[uv](https://docs.astral.sh/uv/)로 Python 3.13 환경과 의존성을 관리한다.

```bash
# 의존성 설치 (.venv 자동 생성)
uv sync
```

### Kaggle 인증 및 데이터 다운로드

```bash
# 브라우저 OAuth 로그인 (최초 1회)
uv run kaggle auth login

# 데이터 다운로드 및 압축 해제
uv run kaggle competitions download -c playground-series-s6e8 -p data
unzip -o data/playground-series-s6e8.zip -d data && rm data/playground-series-s6e8.zip
```

### 새 작업 폴더의 환경 관문 사전 확인

새 Git 작업 폴더에서 시험이나 원격 실행 작업을 시작하기 전에 다음 명령을 한 번 실행한다.
이 명령이 통과하기 전에는 모델 계산이나 유료 원격 자원 생성을 시작하지 않는다.

```bash
cd /absolute/path/to/new-worktree
scripts/verify_environment_gates.sh \
  --source-root /absolute/path/to/verified-worktree
```

명령은 검증 목록의 비커밋 입력을 준비하고 다시 확인한 뒤, 전체 시험 수집에서 PyTorch 조기 적재가 없는지 검사한다.
이어서 Docker의 외부 관리 Python 환경에서 잠긴 가상환경과 `pipeline.entry_diagnostic` 시작 경계를 검증하고 전체 시험을 실행한다.
Docker를 사용할 수 없거나 어느 관문이든 실패하면 후속 단계로 진행하지 않는다.
이미 검증된 `data/`가 있는 작업 폴더에서 다시 실행할 때는 `--source-root`를 생략할 수 있다.

실제 Vast.ai 또는 Runpod 계산 자원에서는 이 사전 확인 명령을 다시 실행하지 않는다.
입력 전송과 해시 검증을 마친 뒤 아래의 원격 Python 실행 관문으로 진입하며, 해당 관문이 실패하면 모델 명령을 시작하지 않고 자원을 정리한다.

### 새 Git 작업 폴더의 비커밋 입력 준비

새 Git 작업 폴더에서 자료가 필요한 명령을 실행하기 전에 저장소의 검증 목록으로 비커밋 입력을 준비한다.
원본은 이미 검증된 입력을 가진 기존 작업 폴더의 절대 경로로 지정한다.
대상 작업 폴더에는 `data/`가 없어야 하며, 기존 `data/`를 덮어쓰는 방식은 허용하지 않는다.

```bash
cd /absolute/path/to/new-worktree
uv run --frozen python -m pipeline.private_inputs prepare \
  --source-root /absolute/path/to/verified-worktree
uv run --frozen python -m pipeline.private_inputs check
```

`prepare`는 `private-inputs.sha256`에 선언된 파일만 복사한다.
원본과 임시 사본의 SHA-256을 확인한 뒤 완성된 `data/`를 대상에 옮기고, 준비된 파일의 권한을 읽기 전용으로 고정한다.
누락, 해시 불일치, 일반 파일이 아닌 입력, 심볼릭 링크, 쓰기 가능 상태 또는 목록 밖 입력이 있으면 명령은 종료 코드가 0이 아닌 값으로 끝난다.
새 입력이 필요하면 `data/` 전체를 연결하거나 임의로 복사하지 말고 검증된 SHA-256과 함께 목록을 변경한다.

### 시험 실행

```bash
uv run pytest
```

`test_model_*.py`에 있는 모형 시험 파일은 macOS OpenMP 충돌을 막기 위해 파일마다 별도 Python 프로세스에서 실행된다.
시험 모듈을 수집하는 동안 `torch`를 적재하면 전체 수집이 명확한 오류로 중단된다.
PyTorch와 XGBoost 또는 LightGBM을 한 시험에서 함께 확인해야 하면 위험한 실행 순서 전체를 명시적인 자식 프로세스 안에 둔다.

### 원격 Python 실행

원격 실행 명세에서 컨테이너 이미지를 고른 뒤 유료 자원을 만들기 전에 실제 이미지의 가상환경 구성 요소를 검사한다.

```bash
scripts/verify_remote_image_python.sh \
  --platform linux/amd64 \
  registry.example/image@sha256:fixed-digest
```

로컬에서 대상 구조나 공급자 전용 이미지를 실행할 수 없는 경우에만 원격 SSH 인증 직후 입력 전송 전에 같은 검사를 수행한다.
자세한 실패 처리와 운영체제 패키지 준비 규칙은 `docs/agents/remote-gpu-transfer.md`를 따른다.

Vast.ai와 Runpod에서 모형 명령을 실행할 때는 시스템 Python에 pip 패키지를 설치하지 않고 공통 실행 명령을 사용한다.
입력 전송 묶음에는 `pyproject.toml`, `uv.lock`, `src/`, `scripts/run_remote_python.sh`와 `scripts/record_remote_python.py`를 함께 넣는다.

```bash
scripts/run_remote_python.sh \
  --system-python python3 \
  --project /workspace/job/input \
  --venv /workspace/job/python-env \
  --evidence /workspace/job/results/python-environment.json \
  -- \
  -m pipeline.entry_diagnostic configs/expNNN.yaml \
  --out-dir /workspace/job/results/entry-expNNN \
  --reference \
  --expected-baseline-auc 0.968294911389327
```

`--` 뒤에는 `python` 명령 자체가 아니라 가상환경 Python에 전달할 인수만 둔다.
실행 명령은 존재하지 않는 작업 전용 가상환경 경로만 받아들이고, 그 안에 `uv==0.11.7`을 설치한 뒤 저장소의 의존성 선언과 `uv.lock`이 일치하는지 확인하며 정확히 잠긴 판본을 설치한다.
준비가 끝나면 Python 실행 파일, Python 판본, 설치 도구 판본과 설치된 모든 패키지 판본을 지정한 JSON 파일에 기록한다.
가상환경 생성, 잠금 확인, 의존성 설치 또는 증거 기록이 실패하면 사용자 명령에는 진입하지 않는다.

### 노트북 실행

```bash
uv run jupyter lab
```

## 프로젝트 구조

```
├── data/                  # 대회 데이터 (gitignore)
├── notebooks/
│   └── eda.ipynb          # 탐색적 데이터 분석
├── scripts/
│   ├── run_remote_python.sh          # 원격 Python 준비 및 실행 관문
│   ├── verify_remote_image_python.sh # 대상 원격 이미지의 가상환경 구성 요소 검사
│   └── verify_environment_gates.sh  # 유료 자원 생성 전 환경 관문 사전 확인
├── docs/
│   ├── adr/               # 아키텍처 결정 기록
│   └── agents/            # 에이전트 스킬 설정 (이슈 트래커, 트리아지 라벨 등)
├── pyproject.toml
└── uv.lock
```

## 실험 기록 조회

실험 기록은 MLflow에 남는다.
메타데이터(시작 시각, metric, params, tag)는 `mlflow.db`(SQLite)에, 예측 파일 등 artifact는 `mlruns/<experiment_id>/<run_id>/artifacts/`에 저장된다.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

브라우저에서 <http://127.0.0.1:5000> 에 접속하면 run 목록을 시작 시각 순으로 보고 metric·artifact를 확인할 수 있다.

## 모델 진입 진단

새 모델 계열은 정식 스크리닝 전에 공통 fold 진입 진단을 실행한다.
기본값은 커밋된 fold 0과 seed 42이며, 정식 실행과 같은 설정 파일, 피처 계획과 모델 adapter를 사용한다.
먼저 현재 champion 설정으로 동등 단계 기준 실행을 저장한다.

```bash
uv run python -m pipeline.entry_diagnostic configs/exp067_lookup_xgb_impute_comps5.yaml \
  --out-dir artifacts/entry-exp067-fold0-seed42 \
  --reference \
  --expected-baseline-auc 0.968294911389327
```

`--expected-baseline-auc`에는 저장된 같은 단계 champion의 fold 0·seed 42 AUC를 전체 정밀도로 넣는다.
기준 재실행 값이 이 값과 `1e-9`보다 크게 다르면 기준 산출물은 중단 상태로 저장되어 challenger에 사용할 수 없다.

challenger는 기준 진단 JSON과 검증 예측을 모두 명시적으로 입력받는다.
같은 모델 계열의 개선 후보는 비교 대상인 모델 설정 축만 `--allow-model-diff`로 허용하고 짝지은 AUC 차이 0 이상을 승격 문턱으로 쓴다.

```bash
uv run python -m pipeline.entry_diagnostic configs/expNNN.yaml \
  --out-dir artifacts/entry-expNNN \
  --baseline-diagnostic artifacts/entry-exp067-fold0-seed42/entry_diagnostic.json \
  --baseline-predictions artifacts/entry-exp067-fold0-seed42/validation_predictions.parquet \
  --comparison-mode champion-improvement \
  --allow-model-diff params.learning_rate
```

새 모델 계열은 `new-model-family` 모드를 사용하며 기존 `champion - 0.01` 진입 하한을 유지한다.
모델 계열과 설정 묶음 전체가 비교 축이면 각각 명시적으로 허용한다.

```bash
uv run python -m pipeline.entry_diagnostic configs/expNNN.yaml \
  --out-dir artifacts/entry-expNNN \
  --baseline-diagnostic artifacts/entry-exp067-fold0-seed42/entry_diagnostic.json \
  --baseline-predictions artifacts/entry-exp067-fold0-seed42/validation_predictions.parquet \
  --comparison-mode new-model-family \
  --allow-model-diff kind \
  --allow-model-diff params \
  --allow-model-diff fit
```

결과 디렉터리에는 공통 JSON, 목표값을 포함한 검증 예측과 피처 중요도가 저장된다.
JSON에는 입력 해시, fold와 시드, 피처 계획, 의존성 판본, 허용 모델 차이, 행 정렬과 목표값 검사, 같은 저장 예측에서 다시 계산한 두 AUC와 차이, 단계별 시간, CUDA 최고 메모리, seed 42 5-fold 예상 시간, 모델별 assertion과 통과 또는 중단 근거가 들어간다.
기준 저장 AUC가 기준 예측 재채점과 다르거나 실행 정체성과 검증 행이 짝을 이루지 않으면 challenger 판정을 시작하지 않는다.
진입 진단은 MLflow 실행을 만들지 않으며 `artifacts/champion.yaml`과 `artifacts/pool.yaml`을 변경하지 않는다.

## 정식 CV 실행 복구

정식 CV 실행은 각 시드의 fold가 끝날 때 검증 예측, 테스트 예측, 중요도, AUC와 실행 정체성을 `run-recovery/<실험>-<단계>/`에 원자적으로 저장한다.
같은 커밋, 설정 원문, 입력 해시, fold 파일, 시드, 실행 의존성 판본과 fold 번호가 모두 일치할 때만 완료된 fold를 다시 사용한다.

```bash
uv run python -m pipeline.run configs/expNNN.yaml --stage screen

# 한 원격 실행 작업에 별도 복구 위치를 고정할 때
uv run python -m pipeline.run configs/expNNN.yaml --stage confirm \
  --recovery-dir /workspace/recovery/expNNN-confirm
```

불완전한 임시 상태, manifest 또는 산출물 해시 불일치, 행과 열 순서 불일치, 중복 fold가 있으면 재사용하지 않고 실행을 중단한다.
완료 실행은 `fold_recovery.json`을 MLflow 산출물로 남기며, 실행 기록 묶음에도 최종 예측 및 중요도와 함께 포함된다.
복구 디렉터리 자체와 임시 상태는 실행 기록 묶음에 포함하지 않는다.

## 제출

제출은 마일스톤 단위 건전성 점검 용도다.
판단 기준은 CV(OOF)이고, public 점수는 CV와 같은 방향인지 확인하는 데만 쓴다.

```bash
uv run python -m pipeline.submit <run_id>
```

해당 MLflow run의 submission artifact를 Kaggle에 제출하고, public 점수를 그 run에 metric `public_auc`로 기록한다.
제출 메시지는 run 이름, run_id 앞 8자리, 커밋 해시, OOF AUC로 자동 생성된다.
`git_dirty=True`로 기록된 run은 제출할 수 없고(우회 없음), 이미 제출된 run의 재제출은 `--force`로만 허용된다.

MLflow 밖에서 이미 제출한 CSV는 기존 실행을 덮어쓰지 않고 별도 파생 실행으로 사후 등록한다.

```bash
uv run python -m pipeline.submit \
  --record-existing <submission_ref> \
  --submission <submitted.csv> \
  --run-name <run_name> \
  --source-run-id <source_run_id> \
  --git-commit <40-character-commit> \
  --artifact <manifest.json> \
  --param ensemble.cv_model_weight=5 \
  --tag source.issue=66
```

명령은 Kaggle 제출 번호, 완료 상태, 제출 시각, 설명, 파일 이름과 공개 점수를 조회하고 CSV 스키마와 SHA-256을 검증한다.
새 실행에는 제출 CSV를 `submission.csv`로, 원격 제출 정보와 해시를 `submission_record.json`으로 기록한다.
같은 제출 번호와 같은 CSV를 다시 등록하면 기존 실행을 돌려주며, 해시가 다르면 중단한다.
