# Playground Series S6E8 상위 득표 노트북 코드 분석: 1위부터 13위

## 조사 범위와 방법

이 문서는 [`code-notebook-inventory.md`](code-notebook-inventory.md)의 고정 목록 가운데 1위부터 13위까지를 분석한다.
득표 수와 순서는 고정 목록의 조사 시점인 2026-08-10 JST를 따른다.
각 고정 주소에서 Kaggle CLI 2.2.4로 2026-08-10에 내려받은 최신 공개 `.ipynb`와 `kernel-metadata.json`을 끝까지 읽었다.
고정 주소는 별도 판본 번호가 없으면 현재 최신 공개 판본을 가리키므로, 아래 링크와 마지막 변경 시각을 함께 적어 조사한 판본을 식별했다.
공개 점수는 제목이나 본문이 `LB` 또는 `public leaderboard`라고 직접 밝힌 수치만 기록했다.
노트북이 출력 없이 저장된 수치를 설명하거나 외부 예측 배열을 읽는 경우에는 실행 가능한 코드 근거와 저자의 서술 근거를 구분했다.
셀 번호는 내려받은 최신 공개 `.ipynb`에서 위에서부터 센 번호다.

## 한눈에 보는 결론

| 순위 | 노트북 | 득표 | 중심 접근 | 검증 설계 | 명시된 공개 점수 | 코드 근거 강도 |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 1 | [S6E8 Addiction LB 0.97092 🔥](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092) | 53 | 이미 만들어진 제출 파일 배포와 선택적 선형 결합 | 이 노트북에는 없음 | 0.97092 | 낮음 |
| 2 | [NoMobilePHOne(Nomophobia) Optuna XGB](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb) | 45 | 단일 80:20 분할에서 정확도로 XGBoost 탐색 | 단일 보류 집합, 정확도 | 없음 | 낮음 |
| 3 | [S6E8 honest OOF blend](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend) | 43 | 74개 공통 OOF 예측의 로짓 공간 선형 결합과 결측 구간 상호작용 | 5겹 바깥 교차 검증, 공통 분할 | 0.97084 | 높음 |
| 4 | [TPS S6E8: EDA, Advanced Feats & Weighted Ensemble](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble) | 40 | LightGBM, CatBoost, XGBoost의 가중 순위 평균 | 5겹 계층 교차 검증 | 없음 | 보통 |
| 5 | [Mobile Addiction \|\| LGBM](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm) | 27 | 비율 변수와 결측 표시를 넣은 단일 LightGBM | 10겹 계층 교차 검증 | 없음 | 보통 |
| 6 | [Hill Climbing Ensemble for Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction) | 26 | 9개 외부 OOF 예측의 탐욕적 가중 결합 | 결합 가중치와 점수를 같은 OOF에서 계산 | 없음 | 보통 이하 |
| 7 | [🥇 #1 Public LB 0.97068 \| Honest 55-Model Stack](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack) | 25 | 전역 선형 결합과 결측 구간별 상호작용 결합 비교 | 5겹 바깥 교차 검증 | 0.97068 | 보통 이상 |
| 8 | [S6E8: Elite Rank Average Ensemble [0.97092]](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092) | 22 | 상위 공개 제출 파일의 단순 순위 평균 | OOF 검증 없음 | 0.97092 | 낮음 |
| 9 | [S6E8: mix the meta-models, then fix the weak bands](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands) | 22 | 결측 구간 결합과 전역 결합의 혼합, 구간 전용 FM 보정 | 3개 메타 분할의 15개 겹을 보고하나 2개 분할은 저장 수치 | 없음 | 높음, 일부 저장 결과 |
| 10 | [S6E8 \| Lookup-Transformer + Insights lb 0.97041](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041) | 20 | 정확한 값 조회 임베딩, 수치 임베딩, Transformer와 두 나무 모델 결합 | 현재 코드는 11겹, 본문은 10겹이라고 서술 | 0.97041 | 높음, 서술 충돌 있음 |
| 11 | [S6E8: What Moved the Score, and What Didn't](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t) | 20 | 값 단위 목표 부호화, 소수 격자 변수, 로짓 공간 선형 결합 | 반복 5겹 실험과 중첩 목표 부호화 | 0.96990 | 높음 |
| 12 | [S6E8: why gaming_hours helps but adds nothing new](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new) | 20 | 조건부 진단, 반복 제거 실험, 생성 규칙과 결합 편향 분석 | 핵심 제거 실험은 3개 씨앗의 5겹 검증 | 없음 | 높음, 일부 시점 고정 자료 |
| 13 | [S6E8: CatBoost](https://www.kaggle.com/code/donmarch14/s6e8-catboost) | 18 | 생성 규칙, 숫자 자리, 중첩 목표 부호화를 넣은 CatBoost | 5겹 바깥 검증과 4겹 안쪽 목표 부호화 | 없음 | 높음 |

## 노트북별 분석

### 1위: S6E8 Addiction LB 0.97092 🔥

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092), 고정 목록 득표 53개, 마지막 변경 2026-08-09T21:11:29.740000Z다.
- 접근: 공개 자료의 `17_blend_submission.csv`를 그대로 읽어 `submission.csv`로 저장하고, 사용자가 별도 제출 파일을 지정하면 두 확률을 선형 결합한다.
- 검증 설계: 셀 1은 5겹 OOF 예측을 공개한다고 설명하지만, 셀 2에서 OOF 파일 읽기가 주석 처리되어 있고 이 노트북 자체에는 AUC 계산, 교차 검증, 원천 모델 훈련이 없다.
- 공개 점수: 제목이 공개 순위표 점수 0.97092를 직접 밝힌다.
- 핵심 코드: [셀 2와 셀 5](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092)의 핵심은 `submission = pd.read_csv(..."17_blend_submission.csv")`와 고정 가중치 선형 결합이다.
- 재사용 가치: 강한 공개 제출 파일을 즉시 재현하거나 자기 예측과 결합하는 입력 자료로 쓸 수 있다.
- 주의점: 원천 변수 생성과 훈련 코드는 비공개라고 셀 1이 명시하므로, 0.97092의 원인이나 일반화 성능을 이 코드만으로 감사할 수 없다.
- 주의점: `YOUR_BLEND_WEIGHT`를 평가 자료 없이 정하도록 되어 있어, 사용자가 공개 순위표를 보며 값을 고르면 공개 순위표 과적합이 생길 수 있다.

### 2위: NoMobilePHOne(Nomophobia) Optuna XGB

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb), 고정 목록 득표 45개, 마지막 변경 2026-08-01T02:04:10.653000Z다.
- 접근: 범주형 세 변수를 `LabelEncoder`로 정수화하고, 80:20 무작위 분할에서 XGBoost의 트리 수, 깊이, 학습률, 양성 가중치를 Optuna 50회로 찾는다.
- 검증 설계: [셀 37부터 셀 46](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb)는 하나의 보류 집합을 탐색과 최종 평가에 함께 사용하며, 목적 함수는 대회 지표 ROC AUC가 아니라 `model.score`의 정확도다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 38](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb)의 `score = model.score(x_test, y_test)`가 모든 탐색 선택을 결정한다.
- 재사용 가치: 빠른 XGBoost 분류 실습과 기본 탐색 예시로는 쓸 수 있다.
- 주의점: 같은 보류 집합을 50회 탐색한 뒤 다시 성능 보고에 사용하므로 선택 편향이 있다.
- 주의점: 숫자 결측값 평균 채우기와 범주 부호화를 훈련 자료에만 적용하고 시험 자료에는 같은 변환을 적용하지 않는다.
- 주의점: 시험 예측과 `submission.csv` 생성 코드가 없어 대회 제출 절차가 완성되지 않는다.

### 3위: S6E8 honest OOF blend

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), 고정 목록 득표 43개, 마지막 변경 2026-08-04T12:42:13.953000Z다.
- 접근: 공통 분할로 만든 74개 OOF와 시험 예측을 `float64`로 읽고, 확률을 로짓으로 바꾼 뒤 선형 분류기로 결합한다.
- 접근: 전역 결합 외에도 완전 관측 행, 4개 이상 결측 행, 구성원 간 불일치를 각 구성원 로짓과 곱한 결측 구간 설계를 비교한다.
- 검증 설계: [셀 8, 16, 18](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 `StratifiedKFold(5, shuffle=True, random_state=42)`의 각 바깥 겹에서 표준화와 선형 분류기를 다시 맞춰 보지 않은 행만 예측한다.
- 검증 설계: 기초 OOF도 모두 같은 5겹 분할이어야 한다고 명시하며, 다른 겹 수나 씨앗별 분할을 섞은 OOF는 겉모양이 정상이어도 2단 결합에서는 누출을 만든다고 경고한다.
- 공개 점수: 본문은 같은 방식의 72개 구성원 제출이 공개 순위표 0.97084였다고 직접 밝힌다.
- 핵심 코드: [셀 4](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)는 각 배열의 길이를 확인하고 `np.column_stack`으로 정렬하며, [셀 18](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 각 바깥 겹 안에서 `StandardScaler`와 `LogisticRegression`을 다시 맞춘다.
- 핵심 근거: `lookup` 구성원은 OOF AUC 0.96853, 기존 묶음과의 최대 상관 0.9869, 결합 기여 약 +0.000109로 보고되며, 넓은 값 격자 나무 세 개의 합보다 기여가 컸다.
- 재사용 가치: 다른 사람의 OOF를 결합할 때 분할 식별자를 먼저 확인하고, 구성원 단독 점수보다 잔차 상관과 제외 전후 차이를 보라는 가장 강한 절차적 근거를 제공한다.
- 재사용 가치: 결측 여부가 목표를 직접 예측하지 않더라도 결측량에 따라 기초 모델의 신뢰도가 달라질 수 있다는 구분이 유용하다.
- 주의점: 74개 기초 예측의 훈련 코드는 별도 자료의 `src/`와 `manifest.csv`에 있으므로 이 노트북 하나만으로 모든 구성원을 다시 훈련하지는 않는다.
- 주의점: 일부 장시간 진단은 `RUN_LOO = False`이고 저장된 결과를 출력하므로, 제외 전후 기여 표는 현재 실행에서 다시 계산되지 않는다.
- 주의점: 결측 구간 설계의 개선 폭 약 0.000029는 노트북이 추정한 공개 순위표 분해능 약 ±0.00014보다 작다.

### 4위: TPS S6E8: EDA, Advanced Feats & Weighted Ensemble

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), 고정 목록 득표 40개, 마지막 변경 2026-08-07T02:40:56.977000Z다.
- 접근: 소수 부분, 결측 표시, 여가 시간, 화면 시간 대비 수면, 생산성, 알림 대비 앱 열기, 주말과 평일 차이를 만든다.
- 접근: LightGBM, CatBoost, XGBoost를 5겹으로 훈련하고 각 겹의 예측을 순위로 바꾼 뒤 0.30, 0.45, 0.25로 가중 평균한다.
- 검증 설계: [셀 11](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)은 계층 5겹 OOF 예측을 모아 전체 ROC AUC를 계산한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 7](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)의 `col % 1`이 소수 격자 변수를 만들고, [셀 11](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)의 `rankdata`와 세 고정 가중치가 최종 결합을 만든다.
- 재사용 가치: 세 주요 나무 계열을 같은 분할에서 비교하고 순위 단위로 결합하는 간결한 기준선이다.
- 주의점: 가중치가 검증 실험이나 중첩 선택으로 정해진 것이 아니라 `historical reliability`라는 주석만 있어 근거가 약하다.
- 주의점: XGBoost에는 조기 종료를 설정하지 않으면서 1,500개 트리를 쓰므로 다른 두 모델과 훈련 정지 조건이 다르다.
- 주의점: 결측 표시와 여러 비율 변수의 개별 기여를 제거 실험으로 확인하지 않아 어떤 변수가 실제로 이득인지 분리할 수 없다.

### 5위: Mobile Addiction || LGBM

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), 고정 목록 득표 27개, 마지막 변경 2026-08-08T15:14:10.530000Z다.
- 접근: 범주 결측값을 별도 수준으로 두고 숫자 결측 표시를 추가한 뒤, 화면 구성 비율, 수면 대비 화면 시간, 화면 사용 중 설명되지 않은 나머지, 여가 대비 일, 시간당 앱 열기와 알림을 만든다.
- 검증 설계: [셀 30](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)은 계층 10겹 OOF ROC AUC와 로그 손실을 계산하고 시험 예측은 열 겹 평균을 사용한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 28](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)의 `unaccounted_screen_time`과 여러 비율 변수, [셀 30](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)의 GPU LightGBM이 중심이다.
- 재사용 가치: 목표를 쓰지 않는 변수 생성과 훈련 및 시험 자료의 열 일치 검사가 간단하고 재현 가능하다.
- 주의점: 이 10겹 OOF는 3위와 7위가 사용하는 고정 5겹 OOF 묶음과 바로 결합하면 안 되며, 공통 분할로 다시 훈련해야 한다.
- 주의점: 설명되지 않은 화면 시간은 음수를 0으로 잘라 생성 규칙 위반을 숨기므로, 규칙 위반 자체를 진단하려면 자르기 전 값을 따로 보존해야 한다.
- 주의점: 파생 변수와 결측 표시의 제거 실험이 없어 10겹 계산량에 비해 변수별 증거는 약하다.

### 6위: Hill Climbing Ensemble for Smartphone Addiction

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction), 고정 목록 득표 26개, 마지막 변경 2026-08-05T07:36:48.630000Z다.
- 접근: XGBoost, RealMLP, TabM, ResNet, TabNet, FT-Transformer, CatBoost, FLAML LightGBM, FLAML XGBoost의 OOF와 시험 예측을 외부 노트북 산출물에서 읽는다.
- 접근: `hillclimbers.climb_hill`로 ROC AUC를 최대화하는 가중 결합을 찾으며 음수 가중치도 허용한다.
- 검증 설계: [셀 9](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction)은 결합 가중치를 찾은 동일한 OOF 행에서 `hc_oof`를 반환한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 7](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction)은 9개 공개 산출물을 위치 기준으로 묶고, [셀 9](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction)은 `precision=0.001`, `negative_weights=True`로 탐색한다.
- 재사용 가치: 다양한 나무와 신경망 예측을 한 표로 모으고 OOF와 시험 예측을 함께 결합하는 최소 예시다.
- 주의점: 기초 OOF의 겹 수, 씨앗, 행 정렬을 이 노트북 안에서 검사하지 않는다.
- 주의점: 가중치 선택과 성능 계산을 중첩하지 않아 보고되는 OOF 점수는 결합 가중치 선택 편향을 포함할 수 있다.
- 주의점: 각 CSV를 `id`로 합치지 않고 행 위치로만 묶으므로 외부 산출물 하나의 행 순서가 달라져도 탐지하지 못한다.

### 7위: 🥇 #1 Public LB 0.97068 | Honest 55-Model Stack

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack), 고정 목록 득표 25개, 마지막 변경 2026-08-03T03:52:07.067000Z다.
- 접근: 공개 OOF 묶음과 선택적 Naji 예측을 읽어 전역 로짓 선형 결합과 결측 구간 상호작용 결합을 비교한다.
- 접근: 완전 관측, 1개부터 3개 결측, 4개 이상 결측의 세 구간을 두고 구성원별 신뢰도가 달라지는지 검증한다.
- 검증 설계: [셀 16과 셀 18](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)은 계층 5겹 바깥 검증에서 전역 결합과 `C` 0.01, 0.03, 0.10의 구간 결합을 비교하고, 개선 폭이 0.00002 이상일 때만 복잡한 쪽을 선택한다.
- 공개 점수: 제목이 공개 순위표 0.97068을 직접 밝힌다.
- 핵심 코드: [셀 15](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)의 설계 행렬은 원래 로짓, 완전 관측 상호작용, 심한 결측 상호작용, 불일치 상호작용, 다섯 요약 열을 결합한다.
- 재사용 가치: 복잡한 결합이 작은 안전 여유를 넘지 못하면 단순 결합으로 되돌아가는 규칙과 결측 구간별 성능 표가 좋다.
- 주의점: [셀 9와 셀 15](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)는 OOF 배열을 `float32`로 낮추지만, 3위 노트북은 높은 상관의 로짓 결합에서 `float64`가 시험 행 순위를 지키는 데 필요하다는 반대 실험을 제시한다.
- 주의점: 구간 설계를 표준화하지 않고 `lbfgs` 반복 횟수 도달 여부도 확인하지 않으며, 3위와 9위 노트북은 같은 종류의 넓은 설계에 표준화와 수렴 검사가 필요하다고 보고한다.
- 주의점: 제목은 55개 모델이라고 남아 있지만 현재 코드는 첨부 자료에서 모든 `oof_*.npy`를 동적으로 읽으므로 실제 구성원 수는 입력 자료 판본에 따라 달라진다.
- 주의점: 선택적 Naji 예측은 파일명 `10_blend`에 고정되어 현재 1위 노트북의 `17_blend`와 판본이 다르다.

### 8위: S6E8: Elite Rank Average Ensemble [0.97092]

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092), 고정 목록 득표 22개, 마지막 변경 2026-08-08T19:41:01.897000Z다.
- 접근: 네 개의 상위 공개 제출 파일을 각각 백분위 순위로 바꾸고 동일 가중치 평균을 낸다.
- 검증 설계: OOF 예측, 교차 검증, 가중치 검증이 없으며 시험 제출 파일끼리의 상관만 그린다.
- 공개 점수: [셀 1 본문](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)이 `LB: 0.97092`라고 직접 밝힌다.
- 핵심 코드: [셀 5와 셀 10](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)은 `/kaggle/input` 아래에서 이름에 `submission`이 든 CSV를 자동 수집하고 `rankdata(...).mean(axis=1)`로 결합한다.
- 재사용 가치: ROC AUC용 제출 파일을 확률 척도 차이에 덜 민감한 순위 평균으로 합치는 가장 짧은 예시다.
- 주의점: 코드는 정확히 네 파일을 지정하지 않고 조건에 맞는 모든 CSV를 넣으므로, 입력이 하나만 늘어도 본문이 설명한 결합과 달라진다.
- 주의점: 첫 파일의 `id`만 가져오고 다른 파일을 `id`로 정렬하거나 일치 여부를 확인하지 않는다.
- 주의점: 네 제출과 가중치를 공개 순위표 성과로 골랐으므로 비공개 평가 자료에 대한 일반화 근거가 없다.

### 9위: S6E8: mix the meta-models, then fix the weak bands

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands), 고정 목록 득표 22개, 마지막 변경 2026-08-05T14:24:58.670000Z다.
- 접근: 74개 OOF 묶음에 세 개의 factorization machine 예측을 더한 전역 선형 결합과, 원래 74개만 쓰는 결측 구간 결합을 각각 만든다.
- 접근: 두 결합의 백분위 순위를 1:2로 섞고, `daily_screen_time_hours`가 3시간 초과 6시간 이하 및 6시간 초과 7.8시간 이하인 두 구간 안에서만 전용 FM으로 행 순서를 조금 바꾼다.
- 검증 설계: 본문은 메타 교차 검증 씨앗 42, 2026, 314159의 15개 겹에서 최종 개선 방향이 모두 같고 평균 OOF AUC가 0.969721이라고 보고한다.
- 검증 설계: [셀 17부터 셀 22](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)에서 씨앗 42는 실행되지만 다른 두 씨앗의 표는 저장된 결과이며 현재 코드가 다시 계산하지 않는다.
- 공개 점수: 이 노트북 자체의 공개 점수는 명시되지 않는다.
- 핵심 코드: [셀 7부터 셀 11](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)은 희귀 값 묶음, 값별 저차원 벡터, 선형항, 쌍별 내적, 선택적 심층 신경망으로 FM을 정의한다.
- 핵심 코드: [셀 22](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)의 `band_adjust`는 구간 밖 행과 예측값 전체의 다중집합을 보존하면서 구간 안 순서만 바꾼다.
- 재사용 가치: 강한 구성원 하나를 더 만드는 것보다 기존 묶음과 다른 함수 계열을 추가하고, 두 편향이 다른 단순 결합을 섞는 편이 낫다는 실험 근거가 풍부하다.
- 재사용 가치: 구간 전용 모델이 전역 결합보다 너무 약하면 아무리 상관이 낮아도 해가 되며, 이 노트북에서는 구간 내부 AUC 차이가 약 0.015 이하일 때만 작은 보정이 유효했다.
- 주의점: 기본 설정 `RUN_TRAINING = False`는 공개 배열을 읽으므로 세 전역 FM과 구간 FM의 훈련 결과가 현재 실행에서 재현되지는 않는다.
- 주의점: 구간 경계와 가중치 0.05가 여러 실험 뒤 선택되었고 다른 두 메타 분할은 저장 수치이므로, 15개 겹 양성이라는 표를 완전히 독립적인 사전 등록 검증으로 읽으면 안 된다.
- 주의점: 최종 개선 폭 0.000027은 이 노트북도 공개 순위표로 분해할 수 없다고 명시한다.

### 10위: S6E8 | Lookup-Transformer + Insights lb 0.97041

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), 고정 목록 득표 20개, 마지막 변경 2026-08-04T19:42:12.647000Z다.
- 접근: 각 변수의 정확한 관측값을 조회표의 키로 삼는 임베딩과, 순위 정규화한 수치의 주기 함수를 이용한 임베딩을 더한다.
- 접근: 설명되지 않은 화면 시간 등 시간 예산 파생 토큰을 만들고 Transformer가 변수 토큰 사이 상호작용을 학습하게 하며, CatBoost와 목표 부호화 LightGBM을 순위 결합한다.
- 검증 설계: [셀 2](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)의 현재 코드는 `N_FOLDS = 3 if QUICK else 11`이므로 기본 실행은 계층 11겹이다.
- 검증 설계: 제목 설명과 표는 계속 10겹이라고 쓰므로 최신 공개 판본 안에서 서술과 실행 코드가 충돌한다.
- 공개 점수: 제목이 공개 순위표 0.97041을 직접 밝힌다.
- 핵심 코드: [셀 10](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 훈련과 시험 자료를 합쳐 목표를 쓰지 않는 정확값 어휘와 `QuantileTransformer` 변환을 만들고, [셀 12](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 조회 임베딩과 주기 수치 임베딩을 더한 토큰을 정의한다.
- 핵심 코드: [셀 18](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)의 LightGBM 목표 부호화는 바깥 겹 훈련 행 안에서 다시 5겹 OOF 값을 만들므로 행 자기 목표 누출을 막는다.
- 재사용 가치: 정확한 값이 연속 크기가 아니라 합성 생성기의 반복 키처럼 작동할 수 있다는 가설을 조회 임베딩과 분할 절반 잔차 재현성으로 연결한다.
- 재사용 가치: 무작위 추가 마스킹과 변수별 학습 가능한 결측 임베딩은 다양한 결측 조합을 신경망이 보게 하는 실용적인 방법이다.
- 주의점: [셀 20](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 같은 OOF 전체에서 0.05 간격 결합 가중치를 찾고 그 OOF AUC를 그대로 보고하므로 결합 단계는 중첩되지 않았다.
- 주의점: 어휘와 수치 변환을 훈련과 시험 자료를 합쳐 만들기 때문에 목표 누출은 아니지만 시험 분포를 사용하는 전이형 전처리다.
- 주의점: 현재 11겹 OOF는 고정 5겹 묶음에 그대로 넣으면 누출 위험이 있으므로, 5겹 묶음과 결합하려면 아키텍처만 옮겨 같은 분할로 다시 훈련해야 한다.

### 11위: S6E8: What Moved the Score, and What Didn't

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), 고정 목록 득표 20개, 마지막 변경 2026-08-08T10:43:10.820000Z다.
- 접근: 원래 결측 열과 모형으로 채운 열을 함께 두고 시간 구성 비율을 만든 뒤, 모든 원시 변수를 문자열 수준으로 본 목표 평균과 빈도를 추가한다.
- 접근: 소수 부분과 첫째 소수 자리 숫자를 별도 변수로 만들어 정확값 목표 부호화가 놓치는 생성기 격자 채널을 포착하고, 여러 예측은 로짓 공간 선형 분류기로 결합한다.
- 검증 설계: [셀 5](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)는 세 교차 검증 씨앗의 5겹 평균 변동을 잡음 기준으로 삼는다.
- 검증 설계: [셀 16과 셀 17](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 바깥 5겹 안의 훈련 행에도 다시 안쪽 5겹 목표 부호화를 적용해 어느 행도 자기 목표를 보지 않게 한다.
- 공개 점수: 본문이 최종 공개 순위표 0.96990을 직접 밝힌다.
- 핵심 코드: [셀 13](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)의 `keep_raw=True`는 모형으로 채운 값이 NaN 원본을 대체하지 않고 나란히 존재하게 한다.
- 핵심 코드: [셀 22](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 `frac_*`와 `d1_*`을 만들며, 저자는 이 채널이 목표 부호화 단독보다 OOF AUC 약 +0.00011이었다고 보고한다.
- 핵심 코드: [셀 25](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 OOF를 절반씩 나눠 가중 평균과 로짓 결합의 차이를 다섯 번 짝지어 비교한다.
- 재사용 가치: 결측값을 자연스럽게 처리하는 나무에는 채운 값이 원본 NaN 열을 대체하지 말고 추가 정보로 들어가야 한다는 제거 실험이 설득력 있다.
- 재사용 가치: 목표 부호화, 소수 격자, 로그 오즈 결합의 이득뿐 아니라 쌍별 목표 부호화, 다중 해상도 부호화, 단조 제약, 원본 자료 합치기, 의사 라벨의 실패도 같은 문서에 남아 있다.
- 주의점: 일부 표의 수치는 현재 셀이 아니라 더 큰 외부 실험 묶음에서 왔다고 셀 30이 명시하므로, 현재 실행 결과와 충돌하면 실행 셀을 우선해야 한다.
- 주의점: 공개 점수와 예상 오프셋으로 목표 부호화 누출 여부를 사후 확인한 설명은 보조 증거일 뿐이며, 핵심 누출 방지는 중첩 코드 자체다.
- 주의점: 훈련과 시험 자료를 함께 사용한 숫자 결측 대체는 목표를 쓰지 않지만 전이형 전처리이며, 다른 대회 규칙에서는 허용 여부를 확인해야 한다.

### 12위: S6E8: why gaming_hours helps but adds nothing new

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), 고정 목록 득표 20개, 마지막 변경 2026-08-10T04:14:48.077000Z다.
- 접근: 단일 변수 AUC, 강한 두 변수를 고정한 구간별 조건부 AUC, 변수를 하나씩 더한 LightGBM 제거 실험을 서로 비교한다.
- 접근: 후보 원본 자료와 대회 자료의 시간 예산 제약을 비교하고, `other_screen = daily - social - gaming - work`가 조건부 분석이 놓친 다변수 생성 규칙임을 보인다.
- 검증 설계: [셀 15와 셀 16](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 씨앗 0, 1, 2마다 계층 5겹을 새로 그려 같은 씨앗 안에서 변수 집합을 짝지어 비교한다.
- 검증 설계: [셀 43](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 목표 부호화 없이 범주와 NaN을 그대로 처리하는 5겹 LightGBM 기준선과 OOF를 제공한다.
- 공개 점수: 이 노트북 자체의 공개 점수는 명시되지 않는다.
- 핵심 근거: [셀 11, 16, 18, 20](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)에 따르면 `gaming_hours`는 고정 구간 안 평균 AUC 약 0.499이지만 세 화면 변수에 더하면 약 +0.00322이며, 강한 두 변수가 모두 관측된 행만 쓰면 그 이득의 약 46%가 사라진다.
- 핵심 근거: 같은 노트북은 `other_screen`이 완전 관측 행에서 `gaming_hours`와 `work_study_hours` 이득의 약 3분의 2를 설명하고, 전체 12개 변수 모델에도 씨앗 변동의 약 열 배에 해당하는 이득을 준다고 보고한다.
- 핵심 코드: [셀 48](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 약한 기준선과 값 단위 목표 부호화를 넣은 강한 기준선 모두에 같은 파생 변수를 더해 기준선 강도에 따른 이득 부풀림을 측정한다.
- 핵심 코드: [셀 50](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 가중치 선택에 쓰지 않은 바깥 겹에서 탐욕 결합을 평가해 동일 OOF에서 선택하고 평가한 낙관 폭을 따로 추정한다.
- 재사용 가치: 단일 변수의 조건부 무신호가 다변수 조합의 무가치를 뜻하지 않으며, 변수 제거는 여러 씨앗과 강한 기준선에서 반복해야 한다는 교훈이 가장 잘 입증되어 있다.
- 재사용 가치: 결측 형태는 목표와 거의 무관하지만 훈련과 시험을 구분하며, 이 경우 높은 적대적 검증 AUC를 곧바로 재가중치 필요성으로 해석하면 안 된다는 구분이 유용하다.
- 주의점: 후보 원본 자료는 실제 생성 원본으로 증명되지 않았다고 저자도 명시하므로, 원본 비교는 생성 메커니즘의 설명 보조 자료일 뿐 대회 자료에서 다시 확인한 사실을 대신하지 않는다.
- 주의점: 후반 공개 순위표 분석에는 2026-08-09의 수동 입력값과 종료된 일곱 대회의 외부 자료가 섞여 있어 가장 빨리 낡는 부분이며, 예측 파이프라인 근거와 분리해서 읽어야 한다.
- 주의점: 74개 OOF 결합에서 구성원 후보 선정은 바깥 겹 안에 중첩하지 않았고 가중치만 중첩했으므로, 저자가 말하듯 결합 낙관 폭을 작게 보는 쪽으로 치우칠 수 있다.

### 13위: S6E8: CatBoost

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/donmarch14/s6e8-catboost), 고정 목록 득표 18개, 마지막 변경 2026-08-04T08:39:51.397000Z다.
- 접근: 원시 NaN과 결측 표시, 시간 합계와 비율, `other_screen`, 숫자의 일의 자리와 첫째 및 둘째 소수 자리, 낮은 해상도 범주와 두 쌍 범주를 CatBoost에 넣는다.
- 접근: 원래 범주 세 변수와 그 쌍, 숫자 20분위 구간에 목표 평균을 추가하되 바깥 겹 훈련 행은 다시 안쪽 4겹으로 부호화한다.
- 검증 설계: [셀 5](https://www.kaggle.com/code/donmarch14/s6e8-catboost)는 계층 5겹 바깥 검증과 계층 4겹 안쪽 목표 부호화를 사용하며, 각 바깥 겹의 시험 예측을 평균한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 4](https://www.kaggle.com/code/donmarch14/s6e8-catboost)의 `other_screen`, 자리 숫자, 값 반올림 범주, 활동 요약과 쌍 범주가 변수 생성의 중심이다.
- 핵심 코드: [셀 5](https://www.kaggle.com/code/donmarch14/s6e8-catboost)의 `add_fold_safe_te`는 안쪽 OOF 훈련 부호화와 바깥 검증 및 시험용 전체 바깥 훈련 매핑을 분리한다.
- 재사용 가치: 11위에서 확인한 목표 부호화와 소수 격자, 12위에서 확인한 `other_screen`을 하나의 누출 방지 CatBoost 절차로 합친다.
- 재사용 가치: GPU 실패 시 같은 겹을 CPU로 다시 시작하고 이후 겹도 CPU로 유지하는 장치가 공개 실행 환경에서 실용적이다.
- 주의점: 결측 표시가 유용하다는 설명은 제거 실험이 없고, 12위는 결측 형태의 목표 AUC가 약 0.502라고 보고하므로 근거가 충돌한다.
- 주의점: 정확값 목표 부호화가 아니라 범주 세 변수와 숫자 20분위 부호화여서 11위의 값 단위 부호화 이득을 그대로 재현하는 절차는 아니다.
- 주의점: 실제 저장 파일은 `oof_preds.csv`인데 셀 1과 마지막 출력은 `oof.csv`라고 써서 산출물 이름이 일치하지 않는다.

## 주제별 종합

### 누출과 검증

가장 반복적이고 강한 결론은 2단 결합에 들어가는 OOF가 같은 행 분할에서 만들어져야 한다는 점이다.
[3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)는 겹 수와 `random_state`가 다른 OOF를 섞으면 한 구성원의 훈련 행이 다른 결합 검증 행을 포함해 조용한 누출이 생긴다고 코드와 반례 설명으로 강조한다.
[5위](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)는 10겹이고 [10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)의 현재 코드는 11겹이므로, 둘 다 5겹 공개 묶음과 결합하려면 같은 5겹으로 다시 훈련해야 한다.

목표 부호화는 [10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [13위](https://www.kaggle.com/code/donmarch14/s6e8-catboost)가 모두 바깥 훈련 행 안에서 다시 안쪽 OOF 값을 만들 때 강한 코드 근거를 갖는다.
반대로 [2위](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb)는 같은 단일 보류 집합을 50회 탐색과 최종 평가에 쓰고, [6위](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction)와 [10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)의 최종 결합은 같은 OOF에서 가중치를 찾고 점수를 보고해 선택 편향이 남는다.
[12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 탐욕 결합 가중치를 바깥 겹 안에서 다시 맞춰 이 편향을 직접 재며, 현재 자료 크기에서는 아주 작지만 0은 아니라고 결론낸다.

[1위](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092)와 [8위](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)는 공개 제출 파일만 합치며 OOF 검증이 없다.
두 노트북의 높은 공개 점수는 제출 자산의 유용성을 보여 주지만, 가중치와 구성원 선택이 비공개 평가 자료에도 유지된다는 증거는 아니다.

훈련과 시험 자료를 함께 사용하되 목표를 쓰지 않는 전이형 전처리는 [9위](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands), [10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)에 나타난다.
이는 목표 누출과는 다르지만 시험 분포를 쓰는 선택이므로 다른 대회에 옮길 때 규칙을 확인해야 한다.

### 합성 자료의 규칙

가장 강하게 반복된 생성 규칙은 `daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours`다.
[10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 대회 자료에서 위반이 없음을 각각 계산하고, [13위](https://www.kaggle.com/code/donmarch14/s6e8-catboost)는 그 나머지를 `other_screen` 변수로 실제 사용한다.
[11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)와 [12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 후보 원본 자료에서는 이 제약이 자주 깨짐을 보여 주어, 사람의 행동 법칙이 아니라 합성 생성 과정의 흔적으로 해석한다.

두 번째 반복 결론은 수치가 부드러운 연속량인 동시에 정확한 값과 소수 자리가 반복되는 격자라는 점이다.
[3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)는 정확값과 값 쌍의 목표 부호화가 여러 나무 계열에서 같은 방향으로 좋아졌다고 보고하고, [10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)는 값별 잔차가 서로 다른 자료 절반에서 재현되는지 검사한 뒤 조회 임베딩으로 연결한다.
[11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)는 정확값 목표 부호화가 소수 첫째 자리 전체에 공통된 신호를 볼 수 없다는 차이를 보이고, [13위](https://www.kaggle.com/code/donmarch14/s6e8-catboost)는 자리 숫자와 반올림 범주를 함께 사용한다.

후보 원본 자료를 추가 훈련 행으로 쓰는 것은 [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)에서 약 -0.00008로 해가 되었고, [12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)도 후보 자료의 계보를 증명할 수 없다고 선을 긋는다.
따라서 원본 후보의 가장 큰 가치는 행 추가보다 생성기가 새로 만든 규칙과 왜곡을 식별하는 진단 자료에 있다.

### 결측값과 전처리

결측 형태가 목표를 직접 예측하는지와 결측으로 모델 신뢰도가 떨어지는지는 다른 질문이다.
[10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 결측 표시의 이득을 약 +0.00001로 보고하고, [12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 결측 개수 단독 AUC 약 0.502와 열별 최대 목표율 차이 약 0.0042를 계산한다.
반면 [3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)와 [7위](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)는 완전 관측 행의 결합 AUC가 4개 이상 결측 행보다 훨씬 높아, 결측 구간에 따라 구성원 가중치를 달리할 작은 여지가 있다고 본다.
두 결과는 충돌하지 않으며, 결측은 목표의 원인이 아니어도 예측에 남은 정보량을 바꾼다.

[11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)의 가장 직접적인 전처리 제거 실험은 NaN을 자연스럽게 처리하는 나무에서 모형으로 채운 값이 원래 열을 대체하면 해가 되고, 원래 NaN 열 옆에 추가될 때만 도움이 된다는 것이다.
[10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)와 [9위](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)는 신경망에서 변수별 학습 가능한 결측 벡터와 무작위 추가 마스킹을 써서 같은 문제를 다르게 푼다.

[12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 훈련과 시험의 관측값 분포는 비슷하지만 결측률은 달라 결측 형태만으로 둘을 구분하는 적대적 검증 AUC가 약 0.57이라고 보고한다.
결측 형태가 목표와 무관하다는 별도 측정 때문에 이 분포 차이를 곧바로 행 재가중치 근거로 삼지 않는다.

### 변수 생성

재현 근거가 가장 강한 파생 변수는 시간 예산의 나머지 `other_screen`이다.
[10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 단독 AUC 약 0.765를 보고하고, [12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 여러 씨앗의 제거 실험으로 이득을 확인하며, [13위](https://www.kaggle.com/code/donmarch14/s6e8-catboost)는 최종 절차에 넣는다.

값 단위 목표 부호화는 [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)에서 약 +0.0023 CV로 가장 큰 단일 개선이며, [3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)의 공개 묶음에서도 여러 나무 계열에 걸쳐 반복된다.
다만 [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)의 36개 쌍을 32구간으로 만든 목표 부호화는 -0.00040이었고, [3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)의 정확값 쌍 격자는 여러 나무에서 개선되었다.
이는 쌍 변수 자체의 찬반이 아니라 해상도, 표본 수, 평활화, 기준선이 결과를 바꾼다는 충돌이다.

일반적인 비율 변수의 증거는 약하다.
[4위](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)와 [5위](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)는 많은 비율을 쓰지만 제거 실험이 없고, [10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)는 일반 비율이 음수였으며 로그 비율은 3겹의 +0.00036이 10겹에서 +0.00001로 사라졌다고 보고한다.
반면 생성 규칙을 직접 나타내는 `other_screen`은 겹 수를 늘려도 남으므로, 의미가 그럴듯한 비율보다 생성 과정을 식별한 변수를 우선할 근거가 더 강하다.

### 모델 계열

LightGBM, XGBoost, CatBoost는 4위, 5위, 10위, 11위, 12위, 13위에서 반복되는 강한 기준 계열이다.
그러나 상위 결합에서 새 구성원의 가치는 단독 AUC보다 기존 예측과 다른 오차에 더 크게 좌우된다.
[3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)는 OOF 0.94085인 약한 신경망도 낮은 상관 때문에 가중치를 얻는 반면 OOF 0.96749인 XGBoost는 상관 0.998 때문에 기여가 거의 없다고 보고한다.

[10위](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)의 Lookup-Transformer는 정확값 조회, 부드러운 수치 추세, 변수 간 주의 연산을 한 모델에 담아 나무와 다른 오차를 만든다.
[3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)의 공통 5겹 재훈련에서는 이 계열 하나가 8개 다른 변수 관점 모델보다 약 15배 큰 결합 기여를 보였다는 코드 기반 비교가 있다.

[9위](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)의 FM은 값 쌍마다 독립 목표 평균을 만들지 않고 값별 저차원 벡터의 내적으로 모든 변수 쌍이 정보를 공유하게 한다.
단독 점수와 비상관성 모두 Lookup-Transformer보다 약해 전역 결합 기여는 약 +0.000006에 그쳤지만, 함수 계열 다양성을 실험하는 좋은 음성 및 양성 대조군을 제공한다.

### 결합 방식

단순 순위 평균은 [4위](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)와 [8위](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)처럼 확률 척도가 다른 모델을 ROC AUC 기준으로 합칠 때 간단하고 안정적이다.
그러나 [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)와 [12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 약하지만 다른 모델을 음수 계수의 보정항으로 쓸 수 있는 로짓 공간 선형 결합이 비음수 평균보다 낫다는 짝지은 검증을 제시한다.

모든 구성원을 같은 가중치로 평균하는 것은 구성원 수가 많을수록 좋아진다는 보장이 없다.
[12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)는 상위 10개 동일 가중 평균보다 74개 전체 동일 가중 평균이 나쁘지만, 음수 계수를 허용한 중첩 선형 결합은 74개 전체가 상위 10개보다 좋다고 보고한다.
결론은 작은 묶음이 좋다는 것이 아니라 큰 묶음을 무차별 평균하는 방식이 나쁘다는 것이다.

결측 구간 상호작용은 [7위](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)가 제안하고 [3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)와 [9위](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)가 다시 측정한다.
세 노트북 모두 개선 방향을 보고하지만 폭은 대체로 0.00002부터 0.00003이며 공개 순위표 분해능보다 작다.
따라서 이 방식은 공개 점수 한 번으로 선택할 대상이 아니라 큰 OOF에서 짝지어 검증하고 작은 안전 여유와 단순 결합 복귀 규칙을 둬야 한다.

### 반복 주장, 충돌, 근거 판정

| 주장 | 반복 여부 | 충돌 또는 제한 | 판정 |
| --- | --- | --- | --- |
| 공통 OOF 분할이 2단 결합의 전제다 | 3위, 7위, 9위가 직접 사용하고 10위와의 분할 차이를 3위가 지적 | 6위는 기초 OOF 분할을 검사하지 않음 | 매우 강함 |
| 정확값 격자와 목표 부호화가 핵심이다 | 3위, 10위, 11위, 13위 | 쌍 부호화는 해상도와 기준선에 따라 부호가 바뀜 | 단일 값은 강함, 쌍은 조건부 |
| 시간 예산 나머지가 유용하다 | 10위, 11위, 12위, 13위 | 결측 행에서는 직접 계산되지 않음 | 강함 |
| 결측 표시가 목표 신호다 | 4위, 5위, 13위는 사용 또는 주장 | 10위와 12위는 직접 이득이 거의 없다고 측정 | 약함 |
| 결측량에 따라 모델 신뢰도가 달라진다 | 3위, 7위, 9위 | 개선 폭이 매우 작고 시험 OOF 분산 척도 차이도 있음 | 보통 |
| 구성원 수보다 오차 다양성이 중요하다 | 3위, 7위, 9위, 11위, 12위 | 낮은 상관만으로는 부족하고 단독 성능 하한도 필요 | 강함 |
| 공개 순위표의 1e-5 개선을 읽을 수 있다 | 일부 제목과 결합 선택이 암묵적으로 기대 | 3위, 9위, 12위는 분해능이 부족하다고 정량화 | 부정 근거가 강함 |
| 원본 후보 자료를 훈련에 더하면 좋다 | 흔한 대회 관행으로 언급 | 11위에서 음수, 12위는 계보 불확실 | 약하거나 부정적 |
| 더 많은 겹은 언제나 변수 이득을 줄인다 | 10위의 일부 비율 변수에서 큰 감소 | 12위의 `other_screen` 계열은 5겹과 10겹에서 유지 | 변수별로 다름 |

## 실행 우선순위 제안

첫째, 모든 자체 모델과 공개 구조 재훈련에 `StratifiedKFold(5, shuffle=True, random_state=42)`를 고정하고 행별 겹 식별자를 산출물에 저장하는 것이 우선이다.
이 결정은 [3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)의 누출 반례와 [9위](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)의 공통 묶음 설계가 가장 강하게 뒷받침한다.

둘째, 강한 단일 모델 기준선에는 원시 NaN, 원시 범주, `other_screen`, 정확값 목표 및 빈도 부호화, 첫째 소수 자리 채널을 하나씩 짝지어 제거하는 실험이 적합하다.
이 조합은 [11위](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12위](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), [13위](https://www.kaggle.com/code/donmarch14/s6e8-catboost)의 겹치는 양성 근거를 최소 변수 집합으로 압축한다.

셋째, Lookup-Transformer는 현재 11겹 코드를 그대로 쓰지 말고 공통 5겹으로 다시 훈련해 OOF 묶음에 넣을 가치가 가장 크다.
[3위](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)가 같은 분할 재훈련 뒤 측정한 결합 기여가 다른 새 계열보다 뚜렷하게 크다.

넷째, 최종 결합은 모든 OOF를 로짓으로 바꾼 선형 분류기를 각 바깥 겹 안에서 다시 맞추는 방식을 먼저 두고, 결측 구간 설계와 순위 평균은 같은 바깥 겹에서 짝지어 비교해야 한다.
공개 순위표나 동일 OOF에서 가중치를 고르지 말고, 작은 개선에는 단순 결합으로 되돌아가는 안전 여유를 둬야 한다.

다섯째, 1위와 8위의 공개 제출 파일은 성능 상한 참고와 마지막 별도 후보로는 유용하지만, 자체 검증 경로와 섞어 모델 선택 근거로 쓰지 않는 편이 안전하다.
이 둘은 원천 훈련이나 OOF 검증보다 공개 점수에 기대는 자산이기 때문이다.

## 접근 제한과 재현 한계

13개 고정 주소의 최신 공개 노트북 원문과 메타데이터는 모두 정상적으로 내려받았으며 접근 차단은 없었다.
다만 1위는 원천 변수 생성과 훈련 코드를 비공개로 두고, 6위는 9개 외부 노트북 산출물을 읽으며, 3위와 7위는 별도 OOF 자료에 기초 모델 훈련을 위임한다.
9위는 기본 설정에서 FM 예측 배열을 읽고 두 개 메타 분할의 결과를 저장 표로 제시하며, 12위 후반의 현재 공개 순위표 수치는 노트북 안에서 다시 내려받지 못해 날짜가 적힌 수동 입력값이다.
따라서 이 문서는 공개된 최신 코드 전체를 분석했지만, 모든 장시간 훈련과 외부 산출물을 현지에서 다시 실행해 수치까지 재현한 것은 아니다.
