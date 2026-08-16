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

## 제출

제출은 마일스톤 단위 건전성 점검 용도다.
판단 기준은 CV(OOF)이고, public 점수는 CV와 같은 방향인지 확인하는 데만 쓴다.

```bash
uv run python -m pipeline.submit <run_id>
```

해당 MLflow run의 submission artifact를 Kaggle에 제출하고, public 점수를 그 run에 metric `public_auc`로 기록한다.
제출 메시지는 run 이름, run_id 앞 8자리, 커밋 해시, OOF AUC로 자동 생성된다.
`git_dirty=True`로 기록된 run은 제출할 수 없고(우회 없음), 이미 제출된 run의 재제출은 `--force`로만 허용된다.
