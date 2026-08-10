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

## 제출

제출은 마일스톤 단위 건전성 점검 용도다.
판단 기준은 CV(OOF)이고, public 점수는 CV와 같은 방향인지 확인하는 데만 쓴다.

```bash
uv run python -m pipeline.submit <run_id>
```

해당 MLflow run의 submission artifact를 Kaggle에 제출하고, public 점수를 그 run에 metric `public_auc`로 기록한다.
제출 메시지는 run 이름, run_id 앞 8자리, 커밋 해시, OOF AUC로 자동 생성된다.
`git_dirty=True`로 기록된 run은 제출할 수 없고(우회 없음), 이미 제출된 run의 재제출은 `--force`로만 허용된다.
