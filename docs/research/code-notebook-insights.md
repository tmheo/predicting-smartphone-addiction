# Playground Series S6E8 분석 대상 노트북 종합

## 조사 범위와 판정 기준

이 문서는 2026-08-10 KST 기준 대회 Code 탭에 공개되어 있고 득표 수가 10개 이상인 최신 버전 37개를 종합한다.
이 범위를 이 저장소의 용어인 **분석 대상 노트북**으로 부른다.
득표 순서, 버전 공개 시각, 작성자와 원문 주소는 공식 Kaggle CLI로 확정한 목록을 따랐다.
세 선행 분석에서는 각 노트북의 최신 공개 `.ipynb`와 메타데이터를 내려받아 모든 코드 셀을 확인했다.
26위부터 37위까지는 [공식 대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)의 일부 사실도 다시 계산했다.
저장된 실행 결과가 없는 노트북의 수치는 재현된 결과로 취급하지 않았다.
실행 가능한 코드, 작성자가 본문에 보고한 결과, 제목에 적힌 public leaderboard 점수도 구분했다.

이 문서에서는 기술 용어를 다음과 같이 사용한다.

- **동일한 fold 분할**은 모든 모델에서 같은 행에 같은 fold ID를 부여한다는 뜻이다.
- **OOF(out-of-fold) 예측**은 각 행을 학습에 쓰지 않은 모델이 그 행에 대해 만든 예측이다.
- **nested CV**에서는 outer CV로 최종 성능을 평가하고 inner CV에서 target encoding, hyperparameter tuning, 모델 선택을 수행한다.

근거 강도는 다음 기준으로 판정했다.

- **강함**은 같은 데이터 분할에서 한 요소씩 빼 본 실험, 여러 시드를 사용한 반복 실험, nested CV 또는 서로 다른 노트북의 독립적인 재현으로 뒷받침된 주장이다.
- **보통**은 실행 가능한 OOF(out-of-fold) 예측 절차가 있으나 ablation, nested selection 또는 저장된 실행 결과 가운데 일부가 없는 주장이다.
- **약함**은 단일 holdout set, public leaderboard, 저장된 외부 예측, 그림이나 서술만으로 뒷받침되는 주장이다.
- Public leaderboard 점수는 제출 파일의 성능을 보여 줄 수 있다.
  그러나 그 점수를 만든 base model과 선택 절차가 공개되지 않았다면 재사용 가능한 인사이트를 뒷받침하는 강한 근거로 보지 않았다.

## 전체 목록

아래 표에는 분석 대상 노트북 37개를 정확히 한 번씩 싣는다.

| 순위 | 분석 대상 노트북 | 작성자 | 득표 | 중심 접근 | 근거 판정 |
| ---: | --- | --- | ---: | --- | --- |
| 1 | [S6E8 Addiction LB 0.97092 🔥](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092) | `najiama` | 53 | 외부 제출 파일 배포와 선택적 linear blend | 약함 |
| 2 | [NoMobilePHOne(Nomophobia) Optuna XGB](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb) | `mpwolke` | 45 | 단일 holdout set에서 정확도를 기준으로 XGBoost 탐색 | 약함 |
| 3 | [S6E8 honest OOF blend](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend) | `szymonkapiski` | 43 | 모든 모델이 동일한 fold 분할로 만든 OOF의 nested logistic blend | 강함 |
| 4 | [TPS S6E8: EDA, Advanced Feats & Weighted Ensemble](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble) | `koushikkumardinda` | 40 | 세 tree-based model의 가중 순위 평균 | 보통 |
| 5 | [Mobile Addiction \|\| LGBM](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm) | `vh10935cse20` | 27 | 비율 및 결측 특성을 사용한 10-fold LightGBM | 보통 |
| 6 | [Hill Climbing Ensemble for Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction) | `omidbaghchehsaraei` | 26 | 외부 OOF 예측 9개를 사용한 greedy weighted blend | 보통 이하 |
| 7 | [🥇 #1 Public LB 0.97068 \| Honest 55-Model Stack](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack) | `riponce` | 25 | global blend와 결측 개수 구간별 blend 비교 | 보통 이상 |
| 8 | [S6E8: Elite Rank Average Ensemble [0.97092]](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092) | `amanatar` | 22 | 공개 제출 네 개의 순위 평균 | 약함 |
| 9 | [S6E8: mix the meta-models, then fix the weak bands](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands) | `raykkretzschmar` | 22 | global blend, 결측 개수 구간별 blend, 구간 전용 FM 보정 | 강함, 일부 저장 결과 |
| 10 | [S6E8 \| Lookup-Transformer + Insights lb 0.97041](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041) | `tamerlanomralinov` | 20 | exact-value lookup embedding Transformer와 트리 모델의 blend | 강함, 서술 충돌 있음 |
| 11 | [S6E8: What Moved the Score, and What Didn't](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t) | `tomasa2` | 20 | 반복 ablation과 value-level target encoding | 강함 |
| 12 | [S6E8: why gaming_hours helps but adds nothing new](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new) | `georgymamarin` | 20 | 데이터 생성 규칙, 조건부 신호와 nested blend 진단 | 강함 |
| 13 | [S6E8: CatBoost](https://www.kaggle.com/code/donmarch14/s6e8-catboost) | `donmarch14` | 18 | 시간 나머지, 숫자 자리와 nested target encoding | 강함 |
| 14 | [Complete EDA: Predicting Smartphone Addition](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition) | `sarveshchhetri` | 17 | 효과 크기와 분포 탐색 | 약함 |
| 15 | [PlaygroundS6E8\|Public\|L2Stack\|V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1) | `ravi20076` | 17 | 공개 OOF를 입력으로 사용한 2단 Ridge stacking | 보통 이하 |
| 16 | [S6E8: LGBM](https://www.kaggle.com/code/donmarch14/s6e8-lgbm) | `donmarch14` | 17 | 여러 파생 특성을 사용한 5-fold LightGBM | 보통 이상 |
| 17 | [S6E8 \| 13 FE Features + XGBoost + Optuna \| 0.96602](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602) | `rugvedbane` | 16 | 13개 파생 특성과 XGBoost | 보통 이하 |
| 18 | [📱 Smartphone Addiction Prediction \| ANN](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann) | `hamidrana` | 16 | 전처리 뒤 단일 검증 세트를 쓰는 완전 연결 신경망 | 약함 |
| 19 | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction) | `zhenruiweng` | 16 | 각 fold 안에서 수행한 target encoding과 RealMLP | 보통 이상 |
| 20 | [Feature-Engineered GBDT: Smartphone Addiction AUC](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc) | `avikdas567` | 16 | 네 tree-based model의 OOF weighted blend | 보통 이하 |
| 21 | [S6E8 Single LGB](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb) | `evgendvorkin` | 16 | 확장 특성을 사용한 10-fold LightGBM | 보통 |
| 22 | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction) | `omidbaghchehsaraei` | 15 | 미리 계산한 target encoding과 RealMLP | 약함, 목표 누출 있음 |
| 23 | [Predicting smartphone addiction](https://www.kaggle.com/code/jek1wantaufik/predicting-smartphone-addiction) | `jek1wantaufik` | 14 | 비공개로 학습한 LightGBM ensemble의 추론 | 약함 |
| 24 | [TabM for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction) | `omidbaghchehsaraei` | 14 | 미리 계산한 target encoding과 TabM | 약함, 목표 누출 있음 |
| 25 | [predicting-smartphone-addiction](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction) | `devashish001` | 13 | 기본 분류기와 F1 중심 탐색 | 약함 |
| 26 | [S6E8: HistGradientBoosting \| LB 0.96945](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945) | `redamountassir` | 12 | exact-value target encoding을 사용한 HistGradientBoosting | 강함, 실행 출력 없음 |
| 27 | [Smartphone Addiction](https://www.kaggle.com/code/cv13j0/smartphone-addiction) | `cv13j0` | 12 | 중복된 훈련 데이터에서 base model 비교 | 약함, 중복 누출 있음 |
| 28 | [Smartphone addiction GBM rank blend nb01](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01) | `danushkumarv` | 12 | 자체 rank blend와 공개 OOF를 사용한 stacking | 보통 이상, 최종 경로 모호 |
| 29 | [S6E8: LGBM \| LB 0.96965](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965) | `redamountassir` | 11 | exact-value target encoding을 사용한 LightGBM | 강함, 실행 출력 없음 |
| 30 | [S6E8 \| Continuous Blender](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender) | `anhadmahajan06` | 11 | 파일명에 적힌 public leaderboard 점수로 제출 파일을 골라 rank blend | 약함 |
| 31 | [PS:S6E8 EDA+ XGB LGBM Ensemble](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble) | `bhaskarmishra44796` | 11 | 전체 데이터로 결측값을 대체한 뒤 두 트리 모델의 예측을 평균 | 보통 이하 |
| 32 | [📱 Predicting Smartphone Addiction - EDA](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda) | `pavloivanin` | 11 | 분포와 행동 비율 탐색 | 약함 |
| 33 | [S6E8 XGBoost \| Public Score 0.96983](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983) | `byerscrip` | 10 | exact-value target encoding을 사용한 XGBoost | 강함, 실행 출력 없음 |
| 34 | [🧠⚡ SmartAddict - OOF Signal Forge](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge) | `lucifer19` | 10 | 세 base model의 nested OOF stacking | 강함, 실행 출력 없음 |
| 35 | [S6:E8\|EDA](https://www.kaggle.com/code/santosh1974/s6-e8-eda) | `santosh1974` | 10 | KS 통계량과 단일 변수 AUC 탐색 | 보통 이하 |
| 36 | [Smartphone Addiction - EDA](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda) | `tuannm3812` | 10 | 결측, 중복과 다변수 분포 차이 진단 | 강함, 목표 예측 없음 |
| 37 | [🚀 Baseline](https://www.kaggle.com/code/pavloivanin/baseline) | `pavloivanin` | 10 | 세 tree-based model의 확률을 고정 비율로 평균 | 보통 |

## 개별 분석

### 1. S6E8 Addiction LB 0.97092 🔥

[원문](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092)은 외부 데이터의 `17_blend_submission.csv`를 그대로 제출하거나 사용자가 지정한 예측과 linear blend한다.
노트북 자체에는 base model 훈련, OOF 계산 또는 가중치 검증이 없으므로 제목의 public leaderboard 점수 0.97092를 원인 분석의 근거로 쓸 수 없다.
강한 외부 제출 파일을 마지막 후보로 보관하는 용도는 있다.
다만 public leaderboard를 보며 가중치를 고르면 공개 평가 데이터에 과도하게 맞을 위험이 있다.

### 2. NoMobilePHOne(Nomophobia) Optuna XGB

[원문](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb)은 범주를 정수로 바꾸고 한 번 나눈 80:20 holdout set에서 XGBoost 설정을 50회 탐색한다.
탐색과 최종 성능 보고에 같은 holdout set을 사용하고 대회 지표인 ROC AUC 대신 정확도를 최적화하므로 선택 편향이 있다.
테스트 데이터에 같은 전처리를 적용하거나 제출 파일을 만드는 코드도 없어 교육용 예시 이상의 재사용 가치는 낮다.

### 3. S6E8 honest OOF blend

[원문](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 모든 모델이 공유하는 5-fold 분할에서 만든 74개 OOF를 logit으로 바꾸고 각 outer fold 안에서 표준화와 linear blend를 다시 학습한다.
fold 수나 시드가 다른 OOF를 섞으면 한 모델의 학습 행이 다른 모델의 blend 검증 행에 들어가는 눈에 띄지 않는 누출이 생긴다는 점을 코드와 설명으로 보여 준다.
base model 단독 AUC보다 기존 모델과의 잔차 상관과 제외 전후 기여를 함께 보라는 절차가 이 조사에서 가장 재사용 가치가 높다.
결측 개수 구간에 따라 가중치를 달리하는 blend의 개선 폭 약 0.000029는 노트북이 추정한 public leaderboard의 점수 구분 폭보다 작으므로 단순 blend로 돌아가는 규칙이 필요하다.

### 4. TPS S6E8: EDA, Advanced Feats & Weighted Ensemble

[원문](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)은 소수부, 결측 여부와 여러 행동 비율을 특성으로 만든다.
그런 다음 LightGBM, CatBoost, XGBoost의 5-fold 예측을 순위로 바꿔 0.30, 0.45, 0.25의 가중치로 평균한다.
같은 stratified split에서 세 tree-based model을 비교하는 틀은 타당하다.
그러나 고정 가중치와 각 파생 변수의 이득은 ablation으로 뒷받침되지 않는다.
따라서 세 모델의 OOF를 동일한 분할에서 만드는 구조는 재사용하되 특성 묶음과 가중치는 다시 검증해야 한다.

### 5. Mobile Addiction || LGBM

[원문](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)은 결측 여부, 화면 사용 구성 비율, 합계에 잡히지 않는 화면 사용 시간을 특성으로 넣은 10-fold LightGBM이다.
목표 변수를 사용하지 않는 특성 생성과 열 일치 검사는 재사용할 만하다.
그러나 화면 사용 시간의 나머지가 음수이면 0으로 잘라 데이터 생성 규칙 위반을 숨기며 특성 ablation도 없다.
이 10-fold OOF를 모든 모델이 공유하는 5-fold blend 데이터에 그대로 넣으면 안 된다.
같은 5-fold 분할로 다시 훈련해야 한다.

### 6. Hill Climbing Ensemble for Smartphone Addiction

[원문](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction)은 아홉 외부 모델의 OOF 예측을 모아 음수 가중치까지 허용하는 greedy blend를 찾는다.
가중치를 찾는 데 사용한 OOF에서 blend 점수를 다시 계산하므로 보고된 점수에 선택 편향이 있다.
base model OOF의 fold ID와 행 `id`도 검사하지 않아 정렬이 어긋날 위험이 있다.
여러 모델의 예측을 한 표로 모으는 구조만 재사용해야 한다.
blend 평가는 별도의 outer fold에서 수행해야 한다.

### 7. 🥇 #1 Public LB 0.97068 | Honest 55-Model Stack

[원문](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)은 전체 데이터에 하나의 가중치를 적용하는 global logistic blend와 결측 개수에 따라 가중치를 달리하는 blend를 비교한다.
성능은 5-fold nested CV의 outer fold에서 평가한다.
개선 폭이 0.00002를 넘지 않으면 단순 blend로 돌아가는 규칙은 작은 이득을 다룰 때 유용한 안전장치다.
다만 입력을 `float32`로 낮추고 넓은 행렬을 표준화하지 않는 선택은 [3번 노트북](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)의 반대 실험 및 수렴 점검 결과와 충돌한다.

### 8. S6E8: Elite Rank Average Ensemble [0.97092]

[원문](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)은 네 공개 제출의 예측을 백분위 순위로 바꿔 같은 비중으로 평균한다.
ROC AUC에서 예측 확률의 척도 차이를 없애는 짧은 예시지만 OOF, 모델 선택 검증과 `id` 정렬 검사가 없다.
제목의 0.97092는 공개 제출 파일에서 나온 결과이며 이 방식이 private leaderboard에서도 일반화된다는 근거는 아니다.

### 9. S6E8: mix the meta-models, then fix the weak bands

[원문](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)은 74개 OOF를 사용한 global blend, 결측 개수 구간별 blend, FM을 이용한 특정 화면 사용 시간 구간의 순서 보정을 함께 적용한다.
세 가지 meta split의 15개 fold에서 개선 방향이 같았다고 보고하지만 현재 코드는 한 split만 다시 계산한다.
나머지는 저장된 수치다.
기존 ensemble과 다른 model family라도 특정 구간에서 단독 성능이 너무 약하면 오히려 해가 된다는 진단은 재현할 가치가 있다.
동점인 예측값의 순서는 유지하면서 특정 구간 안에서만 순서를 보정하는 방식도 마찬가지다.
최종 개선 폭 0.000027은 매우 작고 경계와 가중치는 결과를 본 뒤 선택했다.
독립적인 반복 실험 없이는 채택 근거가 약하다.

### 10. S6E8 | Lookup-Transformer + Insights lb 0.97041

[원문](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 exact-value lookup embedding, 순위를 정규화한 수치 embedding과 Transformer interaction을 CatBoost 및 LightGBM과 blend한다.
관측값의 정확한 값이 연속형 수치일 뿐 아니라 합성 데이터 생성기의 반복 키로도 작동할 수 있다는 가설을 제시한다.
서로 겹치지 않는 두 데이터 절반에서 잔차 패턴이 재현된다는 결과로 이 가설을 뒷받침한다.
현재 코드는 기본값이 11-fold인데 본문에는 10-fold라고 적혀 있다.
최종 가중치도 같은 OOF에서 선택하고 평가한다.
Lookup-Transformer를 모든 모델이 공유하는 5-fold 분할로 다시 훈련한 뒤 트리 모델과 잔차가 얼마나 다른지 측정할 가치가 높다.

### 11. S6E8: What Moved the Score, and What Didn't

[원문](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 세 교차 검증 시드 사이의 점수 변동을 잡음 기준으로 삼고 value-level target encoding과 소수 자리 채널을 nested CV로 검증한다.
value-level target encoding으로 OOF가 약 +0.0023 개선된 결과, 원래 NaN인 열을 유지하면서 대체값을 새 열로 추가했을 때의 이득, logistic blend의 우위는 직접 재현할 가치가 높다.
반면 pair target encoding, 원본 후보 데이터 추가와 pseudo-labeling은 이 노트북의 조건에서 실패했다.
일부 표는 더 큰 외부 실험 묶음에서 가져온 저장 수치다.
현재 실행 코드와 충돌하면 실행 코드를 우선해야 한다.

### 12. S6E8: why gaming_hours helps but adds nothing new

[원문](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 조건부 단일 변수 AUC, 여러 시드에서 반복한 ablation, 데이터 생성 규칙과 nested blend의 편향을 각각 측정한다.
`gaming_hours`는 강한 두 변수로 나눈 구간 안에서는 거의 신호가 없었다.
그러나 세 화면 사용 변수에 더하면 약 +0.00322의 이득이 있었고 `other_screen`이 그 이득의 큰 부분을 설명했다.
따라서 조건부 단일 변수에서 신호가 없다고 해서 다변수 조합에서도 가치가 없다고 단정할 수 없다.
강한 baseline에서 각 특성을 넣고 뺀 결과를 짝지어 비교해야 한다는 강한 근거다.
후반의 public leaderboard 분석과 후보 원본 데이터의 출처 및 생성 이력은 특정 시점에 수동으로 모은 자료다.
예측 절차의 근거와 분리해야 한다.

### 13. S6E8: CatBoost

[원문](https://www.kaggle.com/code/donmarch14/s6e8-catboost)은 원시 NaN, `other_screen`, 숫자 자리, 반올림 범주와 inner OOF target encoding을 CatBoost에 넣는다.
5-fold outer CV의 각 훈련 구간 안에서 4-fold inner CV로 target encoding을 만든다.
11번과 12번에서 확인된 후보를 하나의 누출 방지 절차로 합친 셈이다.
결측 여부의 이득은 ablation으로 검증하지 않았고 exact-value target encoding 대신 20분위 target encoding을 사용했다.
따라서 각 구성 요소의 가치까지 입증한 것은 아니다.

### 14. Complete EDA: Predicting Smartphone Addition

[원문](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition)은 결측률, 목표 비율, 효과 크기, Welch 검정과 훈련 및 테스트 데이터의 분포를 살펴보는 탐색 전용 노트북이다.
효과 크기와 결측률 점검은 데이터 감사에 유용하지만 모델, OOF와 제출이 없어 예측 성능을 뒷받침하지는 않는다.
분포 변화가 없다는 결론은 그림을 눈으로 확인한 결과다.
[36번 노트북](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)처럼 훈련 데이터와 테스트 데이터를 다변수로 구분해 본 검증보다 근거가 약하다.

### 15. PlaygroundS6E8|Public|L2Stack|V1

[원문](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1)은 여러 공개 OOF 예측을 `PredefinedSplit`에 맞춘 2단 Ridge meta-model로 stacking한다.
base model의 OOF와 테스트 예측을 같은 열 구조로 관리하는 설계는 좋다.
그러나 데이터를 모으는 단계에서 `id`를 버리고 행 위치가 같다고 가정한다.
입력에는 target encoding 누출이 있는 22번과 24번의 예측도 포함되어 있다.
따라서 현재 stacking OOF를 정직한 성능 추정으로 볼 수 없다.

### 16. S6E8: LGBM

[원문](https://www.kaggle.com/code/donmarch14/s6e8-lgbm)은 결측 개수, 시간 나머지, 주말 차이, 행동 비율, 숫자 자리와 값 조합을 특성으로 만들어 5-fold LightGBM으로 평가한다.
훈련 데이터와 테스트 데이터에 같은 특성 생성 함수를 사용하고 OOF와 테스트 예측을 함께 저장한다.
실용적인 단일 모델 baseline으로 쓸 수 있는 구조다.
파생 특성이 많지만 개별 ablation은 없다.
또한 훈련 데이터와 테스트 데이터를 합쳐 범주 번호를 만들기 때문에 특성군별 기여와 transductive preprocessing의 영향은 다시 측정해야 한다.

### 17. S6E8 | 13 FE Features + XGBoost + Optuna | 0.96602

[원문](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602)은 임계값, 시간 차이, 비율과 구간을 바탕으로 13개 특성을 만든 XGBoost다.
현재 실행 코드는 한 번 나눈 80:20 holdout set만 사용하고 Optuna 탐색 코드는 주석 처리되어 있다.
따라서 각 특성이 성능을 높였다는 결과를 재현할 수 없다.
제목의 public leaderboard 점수 0.96602는 빠른 제출의 기준으로 참고할 수 있지만 특성 묶음의 일반화를 뒷받침하는 근거는 약하다.

### 18. 📱 Smartphone Addiction Prediction | ANN

[원문](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann)은 결측값 대체, one-hot encoding과 표준화를 거친 5층 완전 연결 신경망이다.
전체 훈련 데이터로 전처리를 학습한 뒤 `validation_split=0.2`를 적용하므로 검증 데이터의 분포가 전처리에 미리 반영된다.
stratified OOF를 만들지 않으며 무작위 시드도 설정하지 않는다.
트리 모델과 다른 잔차를 만드는지 확인할 비교 후보는 될 수 있다.
그러나 현재 검증값을 모델 선택에 사용하면 안 된다.

### 19. RealMLP for Predicting Smartphone Addiction

[원문](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction)은 5-fold outer CV에서 `TargetEncoder.fit_transform`을 매번 다시 호출해 각 행의 target이 그 행의 encoding에 직접 들어가는 누출을 막은 RealMLP 구현이다.
신경망의 다양성과 target encoding을 적용한 위치는 참고할 만하지만 중앙값 대체, 범주형 값의 label encoding과 분위수 구간화는 outer CV로 나누기 전에 전체 학습 데이터에 맞춘다.
데이터에서 값을 학습하는 모든 전처리를 각 outer fold 안에서 수행하고 새로운 범주는 기존 첫 범주와 구분한 뒤 다시 평가해야 한다.

### 20. Feature-Engineered GBDT: Smartphone Addiction AUC

[원문](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc)은 HistGradientBoosting, LightGBM, XGBoost와 CatBoost가 모두 같은 5-fold 분할을 사용하도록 OOF 예측을 만든다.
전체 OOF에서 음이 아닌 blend 가중치를 찾고 같은 OOF로 blend AUC까지 계산하므로 가중치 선택에 따른 낙관 편향이 있다.
여러 트리 모델 계열을 동일한 분할에서 비교하는 구성은 재사용하되 blend 선택과 평가는 별도의 outer fold로 분리해야 한다.

### 21. S6E8 Single LGB

[원문](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb)은 그룹별 결측값 대체, 빈도, 시간 비율, 숫자 자리와 값 조합을 feature로 만들어 10-fold LightGBM에 넣는다.
16번보다 폭넓은 feature 후보를 제시하지만 그룹 중앙값과 빈도는 outer CV를 시작하기 전에 전체 데이터에서 계산한다.
결측값 대체를 각 fold 안에서 수행하고 16번과 동일한 5-fold 분할로 feature 그룹별 제거 실험을 해야 두 결과를 직접 비교할 수 있다.

### 22. RealMLP for Predicting Smartphone Addiction

[원문](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)은 원시 값의 target encoding을 전체 데이터에서 한 번 만든 뒤 동일한 5-fold 분할로 RealMLP를 학습한다.
outer validation fold의 target이 다른 fold에 속한 학습 행의 encoding 통계에 들어가므로 fold 간 target leakage가 발생한다.
RealMLP 설정과 target을 사용하지 않는 파생 feature는 후보로 남길 수 있지만 현재 OOF와 이를 사용한 ensemble 결과는 폐기하고 다시 만들어야 한다.

### 23. Predicting smartphone addiction

[원문](https://www.kaggle.com/code/jek1wantaufik/predicting-smartphone-addiction)은 직렬화된 LightGBM ensemble과 feature 목록을 불러와 테스트 데이터의 예측 확률만 평균한다.
학습 데이터, 분할, 검증 점수와 개별 모델 구성을 현재 코드에서 확인할 수 없어 재현 가능한 baseline은 아니다.
모델 ensemble과 정확한 feature 목록을 함께 배포하는 방식만 참고할 수 있다.

### 24. TabM for Predicting Smartphone Addiction

[원문](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction)은 22번과 마찬가지로 target encoding을 전체 데이터에 미리 적용한 뒤 TabM을 5-fold로 학습한다.
22번과 같은 이유로 outer validation fold의 target이 학습 행의 feature에 간접적으로 들어가는 fold 간 target leakage가 있다.
TabM이 다른 모델과 얼마나 다른 오차를 내는지 시험하려면 각 outer fold의 train split만 사용해 모든 target encoding을 다시 만들어야 한다.

### 25. predicting-smartphone-addiction

[원문](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction)은 기본 분류기, XGBoost와 LightGBM 탐색 결과를 한 번 나눈 80:20 holdout set에서 비교한다.
결측값 대체와 범주형 값의 label encoding을 데이터를 나누기 전에 맞추며 모델 탐색과 평가에는 대회 지표가 아닌 F1과 정확도를 사용하고 제출 코드도 없다.
교육용 모델 비교의 틀로는 쓸 수 있지만 대회 실험 후보로 삼으려면 검증 방식과 지표를 전면 교체해야 한다.

### 26. S6E8: HistGradientBoosting | LB 0.96945

[원문](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945)은 시간 구성 제약에서 파생한 feature와 모든 원시 값에 대한 exact-value target encoding을 HistGradientBoosting에 넣는다.
5개의 outer fold마다 train split에서 5-fold inner CV로 target encoding을 다시 만들어 target leakage를 막은 점이 강점이다.
본문은 서로 다른 exact value가 4,062개라고 적었지만 현재 공식 훈련 데이터에서는 1,389개만 확인된다.
두 수치가 충돌하고 실행 출력도 없으므로 생성 데이터에 대한 해석과 OOF 수치를 다시 확인해야 한다.

### 27. Smartphone Addiction

[원문](https://www.kaggle.com/code/cv13j0/smartphone-addiction)은 같은 공식 `train.csv`를 원본 데이터라고 잘못 지정한 뒤 자기 자신과 이어 붙인다.
모든 훈련 행이 정확히 두 번 존재하며 기본 5-fold CV에서는 validation fold에 있는 행의 복제본이 train split에 들어가므로 교차 검증 결과에 누출이 발생한다.
대회 지표인 AUC도 측정하지 않으므로 이 노트북의 모델 비교 결과는 재사용하지 않아야 한다.

### 28. Smartphone addiction GBM rank blend nb01

[원문](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 직접 학습한 세 트리 모델을 동일한 5-fold 분할에서 rank blend하며 공개 `fold_nb`를 따르는 2단 stacking도 구현한다.
공개 OOF의 fold ID를 그대로 따르는 절차는 좋지만 자체 blend 가중치는 같은 전체 OOF에서 선택하고 평가한다.
현재 최종 제출 코드는 외부 파일이 있으면 자체 stacking 예측을 제외하며 본문 설명과도 다르므로 blend 연구와 제출 경로를 분리해 다시 구현해야 한다.

### 29. S6E8: LGBM | LB 0.96965

[원문](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965)은 26번과 같은 제약 feature와 nested exact-value target encoding을 LightGBM에 적용한다.
제목의 public leaderboard 점수는 0.96965이고 본문은 OOF 0.968259를 보고하지만 저장된 실행 출력은 없다.
26번 및 33번과 동일한 분할에서 다시 실행하면 모델 계열별 단독 성능과 잔차 상관을 비교하기 좋은 후보가 된다.

### 30. S6E8 | Continuous Blender

[원문](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 파일명에 적힌 public leaderboard 점수로 제출 파일을 고르고 다섯 종류의 rank blend를 만든다.
OOF와 target을 사용한 검증이 전혀 없고 public leaderboard 점수가 가장 높은 제출 파일에 95%를 주는 blend도 있어 public leaderboard에 과적합할 위험이 크다.
순위 정규화 구현만 참고하고 blend 구성원과 가중치를 고르는 근거로는 쓰지 않아야 한다.

### 31. PS:S6E8 EDA+ XGB LGBM Ensemble

[원문](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)은 반복적인 결측값 대체를 마친 뒤 XGBoost와 LightGBM이 동일한 5-fold 분할에서 만든 OOF 예측을 반반 평균한다.
결측값 대체기는 교차 검증을 시작하기 전에 전체 학습 데이터에 한 번 맞추며 공식 데이터에 결측값이 있는데도 없다는 주석을 남겨 코드와 데이터가 충돌한다.
모델이 같은 분할을 공유하는 구성만 유지하고 결측값 대체, 조기 종료의 예측 범위와 고정 가중치는 고쳐야 한다.

### 32. 📱 Predicting Smartphone Addiction - EDA

[원문](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)은 target별 분포와 상관관계를 살펴보고 여러 행동 비율을 탐색하는 EDA 전용 노트북이다.
비율 feature 후보는 제시하지만 테스트 데이터 변환, target 예측 모델과 제거 실험이 없어 추가 예측 가치를 입증하지 않는다.
후속 실험의 후보 목록으로만 사용해야 한다.

### 33. S6E8 XGBoost | Public Score 0.96983

[원문](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 26번 및 29번과 같은 제약 feature와 exact-value target encoding을 XGBoost에 넣는다.
5개의 outer fold마다 train split에서 5-fold inner CV로 target encoding을 다시 만들며 제목에는 public leaderboard 점수 0.96983이 적혀 있다.
세 노트북 중 제목에 적힌 점수는 가장 높지만 저장된 OOF 출력은 없으므로 같은 분할에서 다시 실행하기 전에는 특정 모델 계열이 더 낫다고 해석할 수 없다.

### 34. 🧠⚡ SmartAddict - OOF Signal Forge

[원문](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)은 XGBoost 하나와 LightGBM 둘이 동일한 5-fold 분할에서 만든 OOF를 준비한 뒤 각 blend 평가 fold를 제외한 나머지 OOF에서만 탐욕적으로 가중치를 찾는다.
평균 개선 폭이 0.00002를 넘고 다섯 개 fold 가운데 세 개 이상에서 성능이 좋아질 때만 blend를 채택하며 그렇지 않으면 성능이 가장 좋은 단일 모델로 돌아간다.
행별 fold ID, 구성원별 OOF와 파일 해시를 저장해 데이터 계보까지 관리하므로 자체 blend의 기준으로 재현 가치가 매우 높다.
저장된 실행 출력이 없어 실제로 선택된 모델과 점수는 다시 실행해 확인해야 한다.

### 35. S6:E8|EDA

[원문](https://www.kaggle.com/code/santosh1974/s6-e8-eda)은 숫자 feature별 AUC와 훈련 및 테스트 분포 사이의 KS 통계량을 계산한다.
간단한 데이터 진단의 틀은 유용하지만 약 99만 행에서는 작은 차이도 매우 낮은 p값을 만들 수 있으므로 통계량의 크기도 함께 봐야 한다.
결측 패턴, 범주 분포와 다변수 분포 이동은 다루지 않는다.

### 36. Smartphone Addiction - EDA

[원문](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)은 효과 크기, feature별 AUC, 상호 정보량, 결측 여부에 따른 target 비율, 중복 행과 다변수 분포 차이를 단계적으로 검사한다.
원시 feature, 결측 여부를 나타내는 feature, 두 feature 그룹의 조합으로 각각 훈련 데이터와 테스트 데이터를 구분하는 3-fold OOF를 계산해 결측값의 target 신호와 분포 이동을 구분한 설계가 좋다.
target 예측 baseline은 없지만 데이터 점검 절차로는 37개 가운데 가장 재현 가치가 높다.

### 37. 🚀 Baseline

[원문](https://www.kaggle.com/code/pavloivanin/baseline)은 동일한 stratified 5-fold 분할에서 LightGBM, XGBoost와 CatBoost를 학습하고 0.4, 0.3, 0.3의 가중치로 평균한다.
간결한 baseline이지만 범주 빈도는 훈련 데이터와 테스트 데이터를 합쳐 계산하며 고정 가중치와 단위가 서로 다른 숫자 feature의 행별 요약에는 제거 실험에 따른 근거가 없다.
모든 모델이 동일한 OOF 분할을 공유하는 구성은 재사용하고 feature 요약과 가중치는 별도로 검증해야 한다.

## 최종 인사이트

### 데이터 누수

가장 먼저 피해야 할 사례는 중복 행이 train split과 validation fold로 나뉘는 경우다.
[27번](https://www.kaggle.com/code/cv13j0/smartphone-addiction)은 공식 훈련 데이터를 두 번 이어 붙여 validation fold에 있는 모든 행의 복제본을 train split에도 넣으므로 해당 교차 검증 결과를 폐기해야 한다.

target encoding은 OOF 값처럼 보이기만 해서는 충분하지 않으며 최종 outer CV 안에 inner CV로 중첩해야 한다.
[22번](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)과 [24번](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction)은 outer validation fold의 target이 다른 fold에 속한 학습 행의 encoding 통계에 들어간다.
반면 [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 각 outer fold의 train split에서 inner CV를 수행해 OOF target encoding을 새로 만든다.

2단 stacking에 쓰는 base model OOF도 행마다 동일한 fold 분할로 만들어야 한다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 fold 수나 시드가 다른 OOF를 섞을 때 눈에 띄지 않는 누출이 생길 수 있다고 설명한다.
따라서 10-fold인 [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)과 현재 11-fold인 [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 모든 모델이 공유하는 5-fold 분할로 다시 훈련해야 한다.

target을 사용하지 않은 채 훈련 데이터와 테스트 데이터를 합쳐 빈도나 어휘를 만드는 transductive preprocessing은 target leakage와 구분해야 한다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983), [37번](https://www.kaggle.com/code/pavloivanin/baseline)은 이 방식을 사용한다.
대회에서는 허용할 수 있지만 새 데이터가 미리 주어지지 않는 배포 환경에는 그대로 적용할 수 없다.

### 검증 설계

모든 자체 모델이 동일한 `StratifiedKFold(5, shuffle=True, random_state=42)` 분할과 행별 fold ID를 공유하도록 맞추는 것이 가장 안전하다.
동일한 OOF 분할을 전제로 하는 [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), 공개된 fold ID를 따르는 [28번](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01), 산출물의 계보를 저장하는 [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)이 이 선택을 뒷받침한다.

feature, hyperparameter 또는 blend 가중치를 고른 데이터에서 같은 선택의 성능을 다시 측정하면 낙관 편향이 생긴다.
[2번](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb)은 동일한 holdout set을 탐색과 평가에 모두 사용한다.
[6번](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction), [20번](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc), [28번](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 같은 OOF에서 가중치를 찾고 성능도 평가한다.

blend를 선택할 때는 [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)처럼 각 outer fold를 평가용으로 따로 두고 나머지 OOF에서 가중치를 학습해야 한다.
평균 개선 폭과 성능이 좋아진 fold 개수를 함께 보는 이 방식이 가장 재현하기 쉽다.
[7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)이 사용하는 최소 개선 폭과 단순 blend로 돌아가는 규칙도 같은 목적에 맞는 안전장치다.

단일 holdout set의 결과, public leaderboard 점수와 실행 출력으로 확인할 수 없는 본문 수치는 후보를 찾는 데는 쓸 수 있지만 결론을 확정할 근거로는 부족하다.
[17번](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602)의 단일 holdout 검증과 [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)의 제목에 적힌 점수는 모든 모델이 공유하는 5-fold 분할에서 다시 실행해 확인해야 한다.

### 합성 데이터 규칙

가장 일관되게 확인된 생성 규칙은 `daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours`다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 대회 데이터에 이 규칙을 어긴 행이 없다고 계산했다.
26위부터 37위까지의 선행 분석도 [공식 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)에서 네 값을 모두 관측한 421,427행을 확인해 위반이 0개임을 재확인했다.

이 규칙은 사람 행동의 보편적인 법칙이라기보다 합성 데이터 생성기의 흔적으로 보는 편이 타당하다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)과 [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 후보 원본 데이터에서 같은 제약을 어긴 사례가 자주 나온다고 보고한다.

수치 열은 부드러운 연속량의 성격과 함께 정확히 같은 값과 소수 자리가 반복되는 격자 구조도 지닌다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 여러 트리 모델 계열에서 exact-value target encoding을 적용했을 때 이득이 반복해서 나타났다고 보고한다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 정확한 값을 조회하는 embedding을 사용하며 [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 소수 첫째 자리 channel에서 추가 이득을 얻었다고 보고한다.

후보 원본 데이터를 대회 훈련 행에 추가해야 한다는 근거는 부정적이다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 약 -0.00008의 손실을 보고한다.
[12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 해당 데이터가 실제 생성 원본이라는 점이 입증되지 않았다고 명시한다.

### 전처리

NaN을 자체적으로 처리하는 트리 모델에서는 원래 열을 대체값으로 덮지 않고, 원시 NaN 열을 그대로 둔 채 대체값 열을 보조 특성으로 추가하는 방안이 가장 근거가 강하다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 대체값이 원래 열을 대신하면 성능이 나빠지고 두 열을 나란히 넣을 때만 도움이 된다는 ablation 결과를 제시한다.

결측 패턴 자체가 target을 예측하는 정도와 결측으로 관측 정보가 줄어드는 효과는 구분해야 한다.
[12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 결측 개수만 사용했을 때 AUC가 약 0.502라고 보고한다.
반면 [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)과 [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)은 결측이 많은 행에서 ensemble 성능이 낮아지는 현상을 확인한다.

훈련 데이터와 테스트 데이터는 관측값 분포가 비슷해도 결측 패턴은 다를 수 있다.
[12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 결측 패턴만으로 데이터 출처를 구분했을 때 AUC가 약 0.57이라고 보고한다.
[36번](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)은 원시 값과 결측 표시가 훈련 데이터와 테스트 데이터를 얼마나 잘 구분하는지 각각 측정한다.

중앙값 계산, 표준화, 분위수 구간화, 반복 결측 대체는 모두 각 outer fold의 훈련 데이터에서만 학습해야 한다.
[18번](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann), [19번](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction), [21번](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb), [25번](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction), [31번](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)은 데이터로부터 학습하는 일부 변환을 fold 분할 전에 적용해 검증 데이터의 분포를 미리 반영한다.

### 특성 생성

재현 우선순위가 가장 높은 파생 특성은 `other_screen = daily - social - gaming - work`다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 이 특성 하나만으로 약 0.765의 AUC를 얻었다고 보고한다.
[12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 여러 시드와 강한 baseline을 사용한 ablation으로 성능 향상을 확인하며 [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost)은 누출을 방지한 CatBoost에 이 특성을 넣는다.

각 값에 적용하는 target encoding은 단일 특성 가운데 반복해서 가장 큰 개선을 보인 후보지만 반드시 nested CV 안에서 만들어야 한다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 OOF 점수가 약 +0.0023 개선됐다고 보고한다.
[26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 세 가지 트리 모델 계열에 같은 구조를 구현한다.

소수 자리, 반올림값, 정확값을 조합한 특성은 조건부 후보로 남겨야 한다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 소수 첫째 자리 특성에서 작은 추가 이득을 얻었지만 36개 특성 쌍에 32개 구간으로 target encoding을 적용했을 때는 -0.00040을 기록했다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 해상도를 달리한 정확값 특성 쌍이 여러 트리 모델에서 성능을 높였다고 보고한다.

일반적인 행동 비율은 여러 노트북에서 반복해서 사용하지만 근거는 약하다.
[4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [16번](https://www.kaggle.com/code/donmarch14/s6e8-lgbm)은 많은 비율 특성을 쓰지만 각각의 효과를 확인한 ablation이 없다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 3-fold CV에서 나타난 로그 비율의 개선이 10-fold CV에서는 거의 사라졌다고 보고한다.

### 모형 구성

LightGBM, XGBoost, CatBoost는 여러 실험에서 반복해서 검증된 강한 baseline 모델 계열이다.
[4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)이 서로 다른 특성 구성으로 이 모델 계열을 사용한다.

동일한 값 단위 target encoding을 적용한 HistGradientBoosting, LightGBM, XGBoost 비교 실험은 우선순위가 높다.
[26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 제목에 각각 public leaderboard 점수 0.96945, 0.96965, 0.96983을 제시한다.
하지만 동일한 fold 분할에서 저장한 OOF 예측이 없으므로 같은 분할로 다시 실행해야 한다.

Lookup-Transformer는 트리 모델과 다른 오차 패턴을 만드는 새로운 ensemble 구성원으로 가장 유망하다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 정확값 lookup과 완만한 수치 추세를 함께 표현한다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 모든 모델이 공유하는 5-fold 분할로 이 모델 계열을 다시 훈련했을 때 다른 파생 변수 기반 모델보다 blend 기여가 뚜렷하게 컸다고 보고한다.

RealMLP와 TabM은 서로 다른 함수 계열을 추가할 후보지만 공개 구현을 그대로 비교해서는 안 된다.
[19번](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction)은 각 outer fold 안에서 target encoding을 만들지만 다른 전처리는 fold 분할 전에 학습한다.
[22번](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)과 [24번](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction)은 target leakage가 있으므로 모두 수정한 뒤 다시 평가해야 한다.

### 앙상블

구성원 수보다 기존 ensemble과 다른 오차를 내는지가 더 중요하다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 단독 OOF 점수가 낮은 신경망도 기존 모델과의 상관이 낮으면 가중치를 얻는다고 보고한다.
반대로 단독 OOF 점수가 높은 XGBoost라도 기존 ensemble과의 상관이 0.998이면 기여가 거의 없다.

첫 blend baseline으로는 모든 base model이 공유하는 5-fold 분할에서 OOF 예측을 만든 뒤 logit으로 변환한다.
그다음 각 outer fold의 훈련 데이터에서 선형 meta-model을 다시 학습하는 방식이 적합하다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 음수 보정 계수를 허용한 logit blend의 장점을 반복해서 보여 준다.

순위 평균은 확률 척도가 다른 소수의 모델을 결합하기에 간단한 baseline이다.
다만 OOF 검증 없이 public leaderboard 제출 결과만 보고 구성원을 고르면 근거가 약하다.
[4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)은 같은 fold 분할에서 만든 트리 모델 예측을 순위로 변환해 결합한다.
반면 [8번](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)과 [30번](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 공개 제출 파일과 public leaderboard 점수만으로 구성원을 선택한다.

결측 개수 구간별 blend는 반복 실험에서 매번 조금씩 개선됐지만 우선순위는 낮다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack), [9번](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)이 약 0.00002부터 0.00003의 개선을 보고한다.
이 차이는 public leaderboard가 구분할 수 있는 점수 차이보다 작다.
따라서 더 큰 OOF 데이터에서 paired validation을 하고 개선이 확인되지 않으면 단순 blend로 되돌리는 기준이 필요하다.

외부 제출 파일을 결합하는 작업은 OOF 기반 연구용 blend와 분리해야 한다.
[1번](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092), [8번](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092), [30번](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 public leaderboard 점수가 높거나 사용하기 쉽다.
하지만 원본 모델의 훈련 과정, 동일한 fold 분할에서 만든 OOF 예측, 구성원 선택에 대한 검증이 없다.

## 반복 주장, 충돌 결과와 근거가 약한 주장

### 여러 노트북이 반복한 주장

| 주장 | 반복 근거 | 종합 판정 |
| --- | --- | --- |
| 2단 stacking에는 모든 base model이 공유하는 OOF 분할이 필요하다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [28번](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01), [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge) | 매우 강함 |
| target encoding은 각 outer fold 안에서 만들어야 한다 | [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983) | 매우 강함 |
| 하루 사용 시간의 합계 제약과 `other_screen`이 유용하다 | [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [공식 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data) | 강함 |
| 정확값 격자와 값 단위 target encoding이 중요하다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945) | 단일 값은 강함 |
| ensemble에서는 단독 점수보다 오차의 다양성이 중요하다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [9번](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new) | 강함 |

### 서로 충돌하거나 조건에 따라 달라진 결과

| 쟁점 | 한쪽 결과 | 반대 또는 제한 결과 | 해석 |
| --- | --- | --- | --- |
| 결측 표시 자체의 예측력 | [4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost)이 사용한다 | [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 약 +0.00001, [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 결측 개수만 사용한 AUC가 약 0.502라고 보고한다 | target을 직접 예측하는 힘은 약하지만 행마다 남아 있는 정보량이나 데이터 출처를 나타내는 신호일 수 있다 |
| 특성 쌍 target encoding | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 정확값 특성 쌍에서 반복해서 개선됐다고 보고한다 | [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 32개 구간으로 만든 특성 쌍에서 -0.00040을 보고한다 | 해상도, smoothing, baseline에 따라 개선 여부가 달라진다 |
| 일반적인 비율 특성 | [4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [17번](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602)이 널리 사용한다 | [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 3-fold CV에서 보인 개선이 10-fold CV에서 사라졌다고 보고한다 | 의미가 그럴듯하다는 이유만으로 한꺼번에 넣지 말고 각 특성을 ablation으로 검증해야 한다 |
| OOF 수치 정밀도 | [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)은 입력을 `float32`로 낮춘다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 상관이 높은 예측을 logit blend할 때 `float64`가 필요하다고 보고한다 | blend 산출물은 `float64`로 통일해 직접 비교하는 편이 안전하다 |
| 원본 후보 데이터 추가 | 일부 노트북은 원본 데이터를 추가하는 실험 경로를 준비한다 | [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 성능이 나빠졌다고 보고하고 [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 후보 데이터가 원본이라는 주장을 부정한다 | 훈련 행을 늘리는 용도로 쓰지 말고 데이터 생성 규칙을 진단하는 데만 사용한다 |

### 근거가 약한 주장

- 공개 제출 파일만 blend해 얻은 높은 점수가 private leaderboard에서도 유지된다는 주장에는 근거가 부족하다.
  [1번](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092), [8번](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092), [30번](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender) 모두 OOF 검증을 제시하지 않는다.
- 많은 행동 비율과 임계값이 점수를 높인다는 주장에는 개별 ablation이 없다.
  해당 노트북은 [4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [17번](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602), [32번](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)이다.
- 훈련 데이터와 테스트 데이터 사이에 분포 차이가 없다는 주장을 판단할 때는 그림만 확인한 [14번](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition)과 [32번](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)보다 여러 변수를 함께 사용해 두 데이터를 구분한 [36번](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)의 검증을 우선해야 한다.
- 제목에 적힌 public leaderboard 점수만으로 XGBoost가 HistGradientBoosting과 LightGBM보다 낫다고 확정할 수 없다.
  [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 동일한 fold 분할에서 저장한 OOF 예측을 제공하지 않는다.
- 결측 개수 구간별 ensemble이 다른 데이터에도 일반화된다는 주장은 추가 반복 검증이 필요하다.
  [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack), [9번](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)이 보고한 개선 폭은 0.00002부터 0.00003 수준이다.

## 재현 가치가 높은 후보

우선순위는 예상 점수보다 검증 가능성, 여러 실험에서 반복된 근거, 후속 실험에서 얻을 수 있는 정보량을 기준으로 정했다.

1. [36번의 데이터 검사](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)를 재현한다.
   원시 값, 결측 표시, 두 정보를 함께 사용했을 때 훈련 데이터와 테스트 데이터를 구분하는 AUC와 정확한 중복 행 수를 baseline 자료로 남긴다.
2. 모든 실험에서 동일한 5-fold 분할을 사용하고 각 행의 fold ID를 고정한다.
   [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)처럼 OOF 예측, 테스트 예측, 파일 해시, 각 구성원의 출처와 생성 이력을 함께 저장한다.
3. 원시 NaN을 유지한 강한 트리 모델 baseline에 `other_screen`, 값 단위 target encoding과 frequency encoding, 소수 첫째 자리 특성을 하나씩 추가하는 ablation을 실행한다.
   [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost)에서 공통으로 성능이 개선된 후보들이다.
4. [26번 HistGradientBoosting](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번 LightGBM](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번 XGBoost](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)를 같은 5-fold 분할, 같은 특성, 같은 nested target encoding으로 다시 실행한다.
5. [10번 Lookup-Transformer](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)를 모든 모델이 공유하는 5-fold 분할로 다시 훈련한다.
   트리 모델 baseline과의 OOF 순위 상관, 잔차 상관, 이 모델을 blend에서 제외하기 전후의 기여도를 측정한다.
6. 첫 blend baseline으로 [3번의 nested logit blend](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)를 사용한다.
   [34번이 제시한 fold별 승리 횟수와 단순 모델로 되돌리는 조건](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)을 적용한다.
7. RealMLP와 TabM은 [19번처럼 각 outer fold 안에서 target encoding을 만들고](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction) 나머지 전처리도 모두 해당 fold 안으로 옮긴다.
   두 모델은 기존 모델과 다른 오차를 만드는 후보로만 평가한다.
8. 공개 제출 파일을 blend하는 작업은 자체 OOF 연구와 분리한다.
   출처, `id`, fold ID, 원본 OOF 예측이 모두 확인된 경우에만 마지막 제출 후보로 사용한다.

## 한계

이 문서는 2026-08-10에 고정한 최신 공개 판본을 정적으로 분석한 결과를 종합한다.
조사 이후 득표 수와 판본은 바뀔 수 있다.
37개 원문을 모두 확인했지만 여러 노트북이 외부 OOF, 직렬화 모델, 실행 출력이 제거된 코드를 사용한다.
따라서 모든 훈련 수치를 로컬에서 다시 실행해 검증한 것은 아니다.
제목의 public leaderboard 점수는 작성자가 밝힌 값이며 제출 기록을 별도로 검증한 값은 아니다.
이 문서의 재사용 우선순위는 leaderboard 점수 재현을 보장하기 위한 것이 아니라 엄격한 후속 실험의 출발점을 고르기 위한 것이다.
