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

### 노트북 실행

```bash
uv run jupyter lab
```

## 프로젝트 구조

```
├── data/                  # 대회 데이터 (gitignore)
├── notebooks/
│   └── eda.ipynb          # 탐색적 데이터 분석
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

```bash
uv run python -m pipeline.entry_diagnostic configs/expNNN.yaml \
  --out-dir artifacts/entry-expNNN
```

결과 디렉터리에는 공통 JSON, 검증 예측과 피처 중요도가 저장된다.
JSON에는 행 정렬과 유한성 검사, fold AUC, 단계별 시간, CUDA 최고 메모리, seed 42 5-fold 예상 시간, 모델별 assertion과 통과 또는 중단 근거가 들어간다.
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
