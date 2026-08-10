# Playground Series S6E8 분석 대상 노트북 종합

## 조사 범위와 판정 기준

이 문서는 2026-08-10 JST에 대회 Code 탭에서 공개 중이고 득표 수가 10개 이상이었던 최신 공개 판본 37개를 종합한다.
이 범위를 이 저장소의 용어인 **분석 대상 노트북**으로 부른다.
득표 순서, 판본 시각, 작성자와 원문 주소는 공식 Kaggle CLI로 확정한 목록을 따랐다.
세 선행 분석은 각 노트북의 최신 공개 `.ipynb`와 메타데이터를 내려받아 모든 코드 셀을 확인했고, 26위부터 37위까지는 [공식 대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)의 일부 사실도 다시 계산했다.
이 문서는 저장된 실행 출력이 없는 노트북의 수치를 재현된 결과로 취급하지 않고, 실행 가능한 코드, 작성자의 본문 보고, 제목의 공개 점수를 구분한다.

근거 강도는 다음 기준으로 판정했다.

- **강함**은 같은 분할의 제거 실험, 여러 씨앗 반복, 중첩 검증 또는 서로 다른 노트북의 독립된 재현이 있는 주장이다.
- **보통**은 실행 가능한 OOF(겹 밖 예측) 절차가 있으나 제거 실험, 중첩 선택 또는 저장 출력 가운데 일부가 없는 주장이다.
- **약함**은 단일 보류 집합, 공개 순위표, 저장된 외부 예측, 그림이나 서술만으로 뒷받침되는 주장이다.
- 공개 점수는 제출 자산의 성능을 보여 줄 수 있지만, 그 점수를 만든 원천 모형과 선택 절차가 공개되지 않으면 재사용 가능한 인사이트의 강한 근거로 보지 않았다.

## 전체 목록

아래 표에는 분석 대상 노트북 37개를 정확히 한 번씩 싣는다.

| 순위 | 분석 대상 노트북 | 작성자 | 득표 | 중심 접근 | 근거 판정 |
| ---: | --- | --- | ---: | --- | --- |
| 1 | [S6E8 Addiction LB 0.97092 🔥](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092) | `najiama` | 53 | 외부 제출 파일 배포와 선택적 선형 결합 | 약함 |
| 2 | [NoMobilePHOne(Nomophobia) Optuna XGB](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb) | `mpwolke` | 45 | 단일 보류 집합에서 정확도로 XGBoost 탐색 | 약함 |
| 3 | [S6E8 honest OOF blend](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend) | `szymonkapiski` | 43 | 공통 OOF의 중첩 로짓 결합 | 강함 |
| 4 | [TPS S6E8: EDA, Advanced Feats & Weighted Ensemble](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble) | `koushikkumardinda` | 40 | 세 나무 계열의 가중 순위 평균 | 보통 |
| 5 | [Mobile Addiction \|\| LGBM](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm) | `vh10935cse20` | 27 | 비율 및 결측 특성의 10겹 LightGBM | 보통 |
| 6 | [Hill Climbing Ensemble for Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction) | `omidbaghchehsaraei` | 26 | 외부 OOF 9개의 탐욕 가중 결합 | 보통 이하 |
| 7 | [🥇 #1 Public LB 0.97068 \| Honest 55-Model Stack](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack) | `riponce` | 25 | 전역 결합과 결측 구간 결합 비교 | 보통 이상 |
| 8 | [S6E8: Elite Rank Average Ensemble [0.97092]](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092) | `amanatar` | 22 | 공개 제출 네 개의 순위 평균 | 약함 |
| 9 | [S6E8: mix the meta-models, then fix the weak bands](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands) | `raykkretzschmar` | 22 | 전역 및 결측 구간 결합과 구간 전용 FM 보정 | 강함, 일부 저장 결과 |
| 10 | [S6E8 \| Lookup-Transformer + Insights lb 0.97041](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041) | `tamerlanomralinov` | 20 | 정확값 조회 임베딩 Transformer와 나무 결합 | 강함, 서술 충돌 있음 |
| 11 | [S6E8: What Moved the Score, and What Didn't](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t) | `tomasa2` | 20 | 반복 제거 실험과 값 단위 목표 부호화 | 강함 |
| 12 | [S6E8: why gaming_hours helps but adds nothing new](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new) | `georgymamarin` | 20 | 생성 규칙, 조건부 신호와 중첩 결합 진단 | 강함 |
| 13 | [S6E8: CatBoost](https://www.kaggle.com/code/donmarch14/s6e8-catboost) | `donmarch14` | 18 | 시간 나머지, 숫자 자리와 중첩 목표 부호화 | 강함 |
| 14 | [Complete EDA: Predicting Smartphone Addition](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition) | `sarveshchhetri` | 17 | 효과 크기와 분포 탐색 | 약함 |
| 15 | [PlaygroundS6E8\|Public\|L2Stack\|V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1) | `ravi20076` | 17 | 공개 OOF의 Ridge 2단 결합 | 보통 이하 |
| 16 | [S6E8: LGBM](https://www.kaggle.com/code/donmarch14/s6e8-lgbm) | `donmarch14` | 17 | 폭넓은 파생 특성의 5겹 LightGBM | 보통 이상 |
| 17 | [S6E8 \| 13 FE Features + XGBoost + Optuna \| 0.96602](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602) | `rugvedbane` | 16 | 13개 파생 특성과 XGBoost | 보통 이하 |
| 18 | [📱 Smartphone Addiction Prediction \| ANN](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann) | `hamidrana` | 16 | 전처리 뒤 단일 검증을 쓰는 완전 연결 신경망 | 약함 |
| 19 | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction) | `zhenruiweng` | 16 | 겹 안 목표 부호화와 RealMLP | 보통 이상 |
| 20 | [Feature-Engineered GBDT: Smartphone Addiction AUC](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc) | `avikdas567` | 16 | 네 나무 계열의 OOF 가중 결합 | 보통 이하 |
| 21 | [S6E8 Single LGB](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb) | `evgendvorkin` | 16 | 확장 특성의 10겹 LightGBM | 보통 |
| 22 | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction) | `omidbaghchehsaraei` | 15 | 사전 목표 부호화와 RealMLP | 약함, 목표 누출 있음 |
| 23 | [Predicting smartphone addiction](https://www.kaggle.com/code/jek1wantaufik/predicting-smartphone-addiction) | `jek1wantaufik` | 14 | 비공개 학습 LightGBM 묶음의 추론 | 약함 |
| 24 | [TabM for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction) | `omidbaghchehsaraei` | 14 | 사전 목표 부호화와 TabM | 약함, 목표 누출 있음 |
| 25 | [predicting-smartphone-addiction](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction) | `devashish001` | 13 | 기본 분류기와 F1 중심 탐색 | 약함 |
| 26 | [S6E8: HistGradientBoosting \| LB 0.96945](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945) | `redamountassir` | 12 | 정확값 목표 부호화 HistGradientBoosting | 강함, 실행 출력 없음 |
| 27 | [Smartphone Addiction](https://www.kaggle.com/code/cv13j0/smartphone-addiction) | `cv13j0` | 12 | 중복 훈련 자료의 기본 모형 비교 | 약함, 중복 누출 있음 |
| 28 | [Smartphone addiction GBM rank blend nb01](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01) | `danushkumarv` | 12 | 자체 순위 결합과 공개 OOF 적층 | 보통 이상, 최종 경로 모호 |
| 29 | [S6E8: LGBM \| LB 0.96965](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965) | `redamountassir` | 11 | 정확값 목표 부호화 LightGBM | 강함, 실행 출력 없음 |
| 30 | [S6E8 \| Continuous Blender](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender) | `anhadmahajan06` | 11 | 파일명 공개 점수로 고른 제출 순위 결합 | 약함 |
| 31 | [PS:S6E8 EDA+ XGB LGBM Ensemble](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble) | `bhaskarmishra44796` | 11 | 전체 자료 결측 대체 뒤 두 나무 평균 | 보통 이하 |
| 32 | [📱 Predicting Smartphone Addiction - EDA](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda) | `pavloivanin` | 11 | 분포와 행동 비율 탐색 | 약함 |
| 33 | [S6E8 XGBoost \| Public Score 0.96983](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983) | `byerscrip` | 10 | 정확값 목표 부호화 XGBoost | 강함, 실행 출력 없음 |
| 34 | [🧠⚡ SmartAddict - OOF Signal Forge](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge) | `lucifer19` | 10 | 세 기초 모형의 중첩 OOF 결합 | 강함, 실행 출력 없음 |
| 35 | [S6:E8\|EDA](https://www.kaggle.com/code/santosh1974/s6-e8-eda) | `santosh1974` | 10 | KS 통계량과 단일 변수 AUC 탐색 | 보통 이하 |
| 36 | [Smartphone Addiction - EDA](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda) | `tuannm3812` | 10 | 결측, 중복과 다변수 분포 차이 진단 | 강함, 목표 예측 없음 |
| 37 | [🚀 Baseline](https://www.kaggle.com/code/pavloivanin/baseline) | `pavloivanin` | 10 | 세 나무 계열의 고정 확률 평균 | 보통 |

## 개별 분석

### 1. S6E8 Addiction LB 0.97092 🔥

[원문](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092)은 외부 자료의 `17_blend_submission.csv`를 그대로 제출하거나 사용자가 지정한 예측과 선형 결합한다.
노트북 자체에는 원천 모형 훈련, OOF 계산 또는 가중치 검증이 없으므로 제목의 공개 점수 0.97092를 원인 분석 근거로 쓸 수 없다.
강한 외부 제출을 마지막 후보로 보관하는 용도는 있지만, 공개 순위표를 보며 가중치를 고르면 공개 평가 자료에 과도하게 맞을 위험이 있다.

### 2. NoMobilePHOne(Nomophobia) Optuna XGB

[원문](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb)은 범주를 정수로 바꾸고 단일 80:20 분할에서 XGBoost 설정을 50회 탐색한다.
탐색과 최종 보고에 같은 보류 집합을 쓰고 대회 지표 ROC AUC가 아닌 정확도를 최적화하므로 선택 편향이 있다.
시험 자료에 같은 전처리를 적용하거나 제출을 만드는 코드도 없어 교육용 예시 이상의 재사용 가치는 낮다.

### 3. S6E8 honest OOF blend

[원문](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 공통 5겹에서 만든 74개 OOF를 로짓으로 바꾸고, 각 바깥 겹 안에서 표준화와 선형 결합을 다시 학습한다.
겹 수나 씨앗이 다른 OOF를 섞으면 한 구성원의 학습 행이 다른 구성원의 결합 검증 행에 들어가는 조용한 누출이 생긴다는 점을 코드와 설명으로 보여 준다.
구성원 단독 AUC보다 기존 구성원과의 잔차 상관과 제외 전후 기여를 함께 보라는 절차가 이 조사에서 가장 재사용 가치가 높다.
결측 구간 상호작용의 개선 폭 약 0.000029는 노트북이 추정한 공개 순위표 분해능보다 작으므로 단순 결합 복귀 규칙이 필요하다.

### 4. TPS S6E8: EDA, Advanced Feats & Weighted Ensemble

[원문](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)은 소수부, 결측 표시와 여러 행동 비율을 만든 뒤 LightGBM, CatBoost, XGBoost의 5겹 예측 순위를 0.30, 0.45, 0.25로 결합한다.
같은 계층 분할에서 세 나무 계열을 비교하는 골격은 타당하지만, 고정 가중치와 파생 변수별 이득은 제거 실험으로 뒷받침되지 않는다.
따라서 세 모형 공통 OOF 뼈대는 재사용하고 변수 묶음과 가중치는 다시 검증해야 한다.

### 5. Mobile Addiction || LGBM

[원문](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)은 결측 표시, 화면 구성 비율과 설명되지 않은 화면 시간을 사용한 10겹 LightGBM이다.
목표를 쓰지 않는 변수 생성과 열 일치 검사는 재사용할 만하지만, 음수 화면 나머지를 0으로 잘라 생성 규칙 위반을 숨기고 변수 제거 실험도 없다.
10겹 OOF를 공통 5겹 결합 자료에 그대로 넣으면 안 되며 같은 5겹으로 다시 훈련해야 한다.

### 6. Hill Climbing Ensemble for Smartphone Addiction

[원문](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction)은 아홉 외부 모형의 OOF를 모아 음수 가중치까지 허용한 탐욕 결합을 찾는다.
가중치를 찾은 OOF에서 결합 점수를 다시 계산하고 기초 OOF의 겹 식별자와 행 `id`를 검사하지 않아 보고 점수에 선택 편향과 정렬 위험이 있다.
다양한 모형 산출물을 한 표로 모으는 구조만 재사용하고 결합 평가는 별도 바깥 겹으로 분리해야 한다.

### 7. 🥇 #1 Public LB 0.97068 | Honest 55-Model Stack

[원문](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)은 전역 로짓 결합과 결측량별 상호작용 결합을 5겹 바깥 검증에서 비교한다.
개선 폭이 0.00002를 넘지 않으면 단순 결합으로 돌아가는 규칙은 작은 결합 이득을 다루는 좋은 안전장치다.
다만 입력을 `float32`로 낮추고 넓은 설계를 표준화하지 않는 선택은 [3번 노트북](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)의 반대 실험 및 수렴 점검과 충돌한다.

### 8. S6E8: Elite Rank Average Ensemble [0.97092]

[원문](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)은 네 공개 제출을 백분위 순위로 바꿔 같은 비중으로 평균한다.
ROC AUC에서 확률 척도 차이를 없애는 짧은 예시지만 OOF, 구성원 선택 검증과 `id` 정렬 검사가 없다.
제목의 0.97092는 공개 제출 자산의 결과이며 이 방식의 비공개 평가 일반화 근거는 아니다.

### 9. S6E8: mix the meta-models, then fix the weak bands

[원문](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)은 74개 OOF 전역 결합, 결측 구간 결합, FM을 이용한 특정 화면 시간 구간의 순서 보정을 섞는다.
세 메타 분할 15개 겹에서 같은 개선 방향을 보고하지만 현재 코드는 한 분할만 다시 계산하고 나머지는 저장 수치다.
기존 묶음과 다른 함수 계열도 구간 내 단독 성능이 너무 약하면 해가 된다는 진단과 다중집합을 보존한 국소 순서 보정은 재현 가치가 있다.
최종 개선 폭 0.000027과 사후 선택된 경계 및 가중치는 독립 반복 없이는 채택 근거가 약하다.

### 10. S6E8 | Lookup-Transformer + Insights lb 0.97041

[원문](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 정확한 관측값 조회 임베딩, 순위 정규화 수치 임베딩과 Transformer 상호작용을 CatBoost 및 LightGBM과 결합한다.
정확한 값이 연속 크기뿐 아니라 합성 생성기의 반복 키로 작동할 수 있다는 가설을 서로 다른 자료 절반의 잔차 재현성과 연결한다.
현재 코드는 기본 11겹인데 본문은 10겹이라고 쓰며, 최종 가중치도 같은 OOF에서 선택하고 평가한다.
이 구조는 공통 5겹으로 다시 훈련한 뒤 나무 모형과의 잔차 다양성을 측정할 가치가 높다.

### 11. S6E8: What Moved the Score, and What Didn't

[원문](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 세 교차 검증 씨앗의 변동을 잡음 기준으로 삼고, 값 단위 목표 부호화와 소수 자리 채널을 중첩 검증한다.
값 단위 목표 부호화의 약 +0.0023 OOF 개선, 원래 NaN 열 옆에 대체값을 추가할 때의 이득, 로짓 결합의 우위는 직접 재현할 가치가 높다.
반면 쌍 목표 부호화, 원본 후보 자료 추가와 의사 라벨은 이 노트북의 조건에서 실패했다.
일부 표는 더 큰 외부 실험 묶음의 저장 수치이므로 현재 실행 코드와 충돌하면 실행 코드를 우선해야 한다.

### 12. S6E8: why gaming_hours helps but adds nothing new

[원문](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 조건부 단일 변수 AUC, 여러 씨앗의 제거 실험, 생성 규칙과 중첩 결합 편향을 따로 측정한다.
`gaming_hours`는 강한 두 변수의 구간 안에서는 거의 무신호지만 세 화면 변수에 더하면 약 +0.00322였고, `other_screen`이 그 이득의 큰 부분을 설명했다.
이 결과는 조건부 단일 변수 무신호가 다변수 조합의 무가치를 뜻하지 않으며 강한 기준선에서 짝지은 제거 실험이 필요하다는 강한 근거다.
후반의 공개 순위표 분석과 후보 원본 자료 계보는 시점 고정 수동 자료이므로 예측 파이프라인 근거와 분리해야 한다.

### 13. S6E8: CatBoost

[원문](https://www.kaggle.com/code/donmarch14/s6e8-catboost)은 원시 NaN, `other_screen`, 숫자 자리, 반올림 범주와 안쪽 OOF 목표 부호화를 CatBoost에 넣는다.
5겹 바깥 검증과 4겹 안쪽 목표 부호화가 분리되어 있어 11번과 12번에서 확인된 후보를 하나의 누출 방지 절차로 합친다.
결측 표시의 이득은 제거 실험이 없고 정확값이 아닌 20분위 목표 부호화를 쓰므로 각 구성요소의 가치까지 입증한 것은 아니다.

### 14. Complete EDA: Predicting Smartphone Addition

[원문](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition)은 결측률, 목표 비율, 효과 크기, Welch 검정과 훈련 및 시험 분포를 살펴보는 탐색 전용 노트북이다.
효과 크기와 결측률 점검은 자료 감사에 유용하지만 모형, OOF와 제출이 없어 예측 성능 근거는 아니다.
분포 이동이 없다는 결론은 그림을 눈으로 본 결과여서 [36번 노트북](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)의 다변수 분포 구분 검증보다 근거가 약하다.

### 15. PlaygroundS6E8|Public|L2Stack|V1

[원문](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1)은 여러 공개 OOF를 `PredefinedSplit`에 맞춘 Ridge 2단 모형으로 결합한다.
기초 OOF와 시험 예측을 같은 열 구조로 관리하는 설계는 좋지만 자료 취합 단계가 `id`를 버리고 행 위치를 가정한다.
입력에는 목표 부호화 누출이 있는 22번과 24번 산출물도 포함되어 있어 현재 결합 OOF를 정직한 성능 추정으로 볼 수 없다.

### 16. S6E8: LGBM

[원문](https://www.kaggle.com/code/donmarch14/s6e8-lgbm)은 결측 개수, 시간 나머지, 주말 차이, 행동 비율, 숫자 자리와 값 조합을 5겹 LightGBM으로 평가한다.
같은 함수로 훈련과 시험 변수를 만들고 OOF 및 시험 예측을 함께 저장하는 구조는 실용적인 단일 모형 기준선이다.
파생 변수가 많지만 개별 제거 실험이 없고 훈련과 시험을 합쳐 범주 번호를 만들기 때문에, 변수군별 기여와 전이형 전처리의 영향은 다시 측정해야 한다.

### 17. S6E8 | 13 FE Features + XGBoost + Optuna | 0.96602

[원문](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602)은 임계값, 시간 차이, 비율과 구간으로 13개 특성을 만든 XGBoost다.
현재 실행 코드는 단일 80:20 보류 집합만 쓰고 Optuna 탐색 코드는 주석 처리되어 있어 특성별 향상 경로가 재현되지 않는다.
제목의 공개 점수 0.96602는 빠른 제출 기준으로 참고할 수 있지만 특성 묶음의 일반화 근거는 약하다.

### 18. 📱 Smartphone Addiction Prediction | ANN

[원문](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann)은 결측 대체, 범주 펼치기와 표준화를 거친 5층 완전 연결 신경망이다.
전체 학습 자료로 전처리를 맞춘 뒤 `validation_split=0.2`를 적용해 검증 분포를 미리 보고, 층화 OOF와 무작위 시드도 없다.
나무와 다른 잔차를 만드는지 보는 비교 후보는 될 수 있지만 현재 검증값을 모형 선별에 쓰면 안 된다.

### 19. RealMLP for Predicting Smartphone Addiction

[원문](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction)은 5겹 바깥 반복 안에서 `TargetEncoder.fit_transform`을 다시 호출해 행 자기 목표 누출을 막은 직접 구현 RealMLP다.
신경망 다양성과 목표 부호화 위치는 재사용 가치가 있지만 중앙값, 범주 번호와 분위 구간을 바깥 분할 전에 전체 학습 자료에 맞춘다.
모든 자료 기반 전처리를 겹 안으로 옮기고 미지 범주를 기존 첫 범주와 구분한 뒤 다시 평가해야 한다.

### 20. Feature-Engineered GBDT: Smartphone Addiction AUC

[원문](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc)은 HistGradientBoosting, LightGBM, XGBoost와 CatBoost의 공통 5겹 OOF를 만든다.
전체 OOF에서 음이 아닌 가중치를 찾고 같은 OOF로 결합 AUC를 계산하므로 가중치 선택 편향이 있다.
공통 분할의 여러 나무 계열 비교 골격은 재사용하되 결합 선택은 별도 바깥 겹으로 옮겨야 한다.

### 21. S6E8 Single LGB

[원문](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb)은 그룹별 결측 대체, 빈도, 시간 비율, 숫자 자리와 값 조합을 10겹 LightGBM에 넣는다.
16번보다 넓은 변수 후보를 제공하지만 그룹 중앙값과 빈도를 바깥 검증 전에 전체 자료에서 계산한다.
결측 대체를 겹 안으로 옮기고 16번과 같은 5겹에서 변수군별 제거 실험을 해야 직접 비교할 수 있다.

### 22. RealMLP for Predicting Smartphone Addiction

[원문](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)은 원시 값 목표 부호화를 전체 자료에 한 번 만든 뒤 같은 5겹으로 RealMLP를 학습한다.
바깥 검증 겹의 목표값이 다른 겹 학습 행의 부호화 통계에 들어가므로 교차 겹 목표 누출이 생긴다.
RealMLP 설정과 목표를 쓰지 않는 파생 변수는 후보로 남길 수 있지만 현재 OOF와 그 산출물을 쓴 결합은 폐기하고 다시 만들어야 한다.

### 23. Predicting smartphone addiction

[원문](https://www.kaggle.com/code/jek1wantaufik/predicting-smartphone-addiction)은 직렬화된 LightGBM 묶음과 특성 목록을 불러와 시험 확률만 평균한다.
학습 자료, 분할, 검증 점수와 개별 모형 구성을 현재 코드에서 확인할 수 없어 재현 가능한 기준선은 아니다.
모형 묶음과 정확한 특성 목록을 함께 배포하는 방식만 참고할 수 있다.

### 24. TabM for Predicting Smartphone Addiction

[원문](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction)은 22번과 같은 사전 목표 부호화 뒤 TabM을 5겹으로 학습한다.
22번과 동일하게 바깥 검증 목표가 학습 행 특성에 간접적으로 들어가는 교차 겹 목표 누출이 있다.
TabM의 오차 다양성을 시험하려면 각 바깥 겹의 학습 부분만으로 모든 목표 부호화를 다시 만들어야 한다.

### 25. predicting-smartphone-addiction

[원문](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction)은 기본 분류기와 XGBoost 및 LightGBM 탐색을 단일 80:20 분할에서 비교한다.
결측 대체와 범주 번호화를 분할 전에 맞추고, 탐색과 평가를 대회 지표가 아닌 F1 및 정확도로 수행하며 제출 코드도 없다.
교육용 모형 비교 골격은 될 수 있지만 대회 실험 후보로는 검증과 지표를 전면 교체해야 한다.

### 26. S6E8: HistGradientBoosting | LB 0.96945

[원문](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945)은 시간 구성 제약 변수와 모든 원시 값의 정확값 목표 부호화를 HistGradientBoosting에 넣는다.
각 5겹 바깥 훈련 부분에서 5겹 안쪽 목표 부호화를 다시 만들어 목표 누출을 막은 절차가 강점이다.
본문의 정확값 수준 4,062개는 현재 공식 훈련 자료의 1,389개와 충돌하고 실행 출력도 없으므로 생성기 해석과 OOF 수치는 다시 확인해야 한다.

### 27. Smartphone Addiction

[원문](https://www.kaggle.com/code/cv13j0/smartphone-addiction)은 같은 공식 `train.csv`를 원본 자료라고 잘못 지정한 뒤 자기 자신과 이어 붙인다.
모든 훈련 행이 정확히 두 번 존재하고 기본 5겹에서 검증 행의 복제본이 학습 부분에 들어가므로 교차 검증 결과는 누출되어 있다.
대회 AUC도 측정하지 않으므로 이 노트북의 모형 비교 결과는 재사용하지 않아야 한다.

### 28. Smartphone addiction GBM rank blend nb01

[원문](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 세 자체 나무 모형의 공통 5겹 순위 결합과 공개 `fold_nb`를 따른 2단 결합을 함께 구현한다.
공개 OOF의 겹 식별자를 그대로 따르는 절차는 좋지만 자체 가중치는 같은 전체 OOF에서 선택하고 평가한다.
현재 최종 제출 코드는 외부 파일이 있으면 자체 적층 예측을 빼며 본문 설명과도 달라, 결합 연구와 제출 경로를 분리해 다시 구현해야 한다.

### 29. S6E8: LGBM | LB 0.96965

[원문](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965)은 26번과 같은 제약 변수 및 중첩 정확값 목표 부호화를 LightGBM에 적용한다.
제목의 공개 점수는 0.96965이고 본문은 OOF 0.968259를 보고하지만 저장 실행 출력이 없다.
26번 및 33번과 같은 분할에서 다시 실행하면 함수 계열별 단독 성능과 잔차 상관을 비교하기 좋은 후보가 된다.

### 30. S6E8 | Continuous Blender

[원문](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 파일명에서 읽은 공개 점수로 제출을 고르고 순위 결합 다섯 종류를 만든다.
OOF와 목표값 검증이 전혀 없고 가장 높은 공개 제출에 95%를 주는 결합도 있어 공개 순위표 과적합 위험이 크다.
순위 정규화 구현만 참고하고 구성원 및 가중치 선택 근거로는 쓰지 않아야 한다.

### 31. PS:S6E8 EDA+ XGB LGBM Ensemble

[원문](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)은 반복 결측 대체 뒤 XGBoost와 LightGBM의 공통 5겹 예측을 반반 평균한다.
결측 대체기를 교차 검증 전에 전체 학습 자료에 한 번 맞추고, 공식 자료에 결측이 있는데 없다는 주석을 남겨 코드와 자료가 충돌한다.
공통 분할 뼈대만 유지하고 결측 대체, 조기 종료 예측 범위와 고정 가중치를 고쳐야 한다.

### 32. 📱 Predicting Smartphone Addiction - EDA

[원문](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)은 목표별 분포, 상관과 여러 행동 비율을 살펴보는 탐색 전용 노트북이다.
비율 후보는 제공하지만 시험 자료 변환, 목표 예측 모형과 제거 실험이 없어 추가 예측 가치를 입증하지 않는다.
후속 실험의 후보 목록으로만 사용해야 한다.

### 33. S6E8 XGBoost | Public Score 0.96983

[원문](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 26번 및 29번과 같은 제약 변수와 정확값 목표 부호화를 XGBoost에 넣는다.
5겹 바깥 훈련 부분마다 5겹 안쪽 목표 부호화를 다시 만들고 제목은 공개 점수 0.96983을 밝힌다.
세 노트북 가운데 제목 점수는 가장 높지만 저장 OOF 출력이 없으므로 같은 분할의 재실행 전에는 모형 계열 우위로 해석할 수 없다.

### 34. 🧠⚡ SmartAddict - OOF Signal Forge

[원문](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)은 공통 5겹의 XGBoost와 두 LightGBM을 만든 뒤 각 결합 검증 겹을 제외한 OOF에서만 탐욕 가중치를 찾는다.
평균 개선이 0.00002를 넘고 다섯 겹 중 적어도 세 겹에서 이길 때만 결합을 채택하며, 그렇지 않으면 최고 단일 모형으로 돌아간다.
행별 겹, 구성원 OOF와 파일 해시를 저장하는 계보 관리까지 포함해 자체 결합 기준으로 재현 가치가 매우 높다.
저장 출력이 없어 실제 선택 모형과 점수는 다시 실행해야 한다.

### 35. S6:E8|EDA

[원문](https://www.kaggle.com/code/santosh1974/s6-e8-eda)은 숫자 변수의 단일 변수 AUC와 훈련 및 시험 분포의 KS 통계량을 계산한다.
짧은 자료 진단 뼈대는 유용하지만 약 99만 행에서는 작은 차이도 매우 작은 p값을 만들 수 있으므로 통계량 크기를 함께 봐야 한다.
결측 형태, 범주 분포와 다변수 분포 이동은 다루지 않는다.

### 36. Smartphone Addiction - EDA

[원문](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)은 효과 크기, 단일 변수 AUC, 상호 정보량, 결측 목표율, 중복 행과 다변수 분포 구분을 단계적으로 검사한다.
원시 변수, 결측 표시, 둘의 결합으로 훈련 및 시험을 구분하는 3겹 OOF를 따로 계산해 결측의 목표 신호와 분포 이동을 구분한 설계가 좋다.
목표 예측 기준선은 없지만 자료 감사 절차로는 37개 가운데 가장 재현 가치가 높다.

### 37. 🚀 Baseline

[원문](https://www.kaggle.com/code/pavloivanin/baseline)은 같은 계층 5겹에서 LightGBM, XGBoost와 CatBoost를 학습하고 0.4, 0.3, 0.3으로 평균한다.
간결한 기준선이지만 범주 빈도를 훈련과 시험을 합쳐 만들고 고정 가중치 및 단위가 다른 숫자 전체의 행 요약에 제거 근거가 없다.
공통 OOF 골격은 재사용하고 변수 요약과 가중치는 별도 검증해야 한다.

## 최종 인사이트

### 데이터 누수

가장 중요한 금지 사례는 중복 행이 학습과 검증으로 갈라지는 경우다.
[27번](https://www.kaggle.com/code/cv13j0/smartphone-addiction)은 공식 훈련 자료를 두 번 이어 붙여 모든 검증 행의 복제본을 학습 부분에 보내므로 그 교차 검증 결과를 폐기해야 한다.

목표 부호화는 OOF 값처럼 보이는 것만으로 충분하지 않고 최종 바깥 검증 경계 안에 다시 중첩되어야 한다.
[22번](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)과 [24번](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction)은 바깥 검증 목표가 다른 겹 학습 행의 부호화 통계에 들어가지만, [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 바깥 훈련 부분마다 안쪽 OOF 부호화를 다시 만든다.

2단 결합의 기초 OOF도 같은 행 분할에서 만들어져야 한다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 겹 수나 씨앗이 다른 OOF를 섞을 때 생기는 조용한 누출을 설명하며, 10겹인 [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm)과 현재 11겹인 [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 공통 5겹으로 다시 훈련해야 한다.

목표를 쓰지 않고 훈련과 시험을 합쳐 빈도나 어휘를 만드는 전이형 전처리는 목표 누출과 구분해야 한다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983), [37번](https://www.kaggle.com/code/pavloivanin/baseline)은 이 방식을 사용하며, 대회 안에서는 허용 가능하지만 새 자료가 미리 없는 배포 상황으로 일반화할 수는 없다.

### 검증 설계

모든 자체 모형은 같은 `StratifiedKFold(5, shuffle=True, random_state=42)`와 행별 겹 식별자를 공유하는 것이 가장 안전하다.
이 선택은 공통 OOF를 전제로 하는 [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), 공개 겹 식별자를 따르는 [28번](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01), 산출물 계보를 저장하는 [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)이 함께 뒷받침한다.

특성, 초매개변수 또는 결합 가중치를 고른 자료에서 같은 선택의 성능을 다시 보고하면 낙관 편향이 생긴다.
[2번](https://www.kaggle.com/code/mpwolke/nomobilephone-nomophobia-optuna-xgb)은 같은 보류 집합을 탐색과 평가에 쓰고, [6번](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction), [20번](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc), [28번](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 같은 OOF에서 가중치를 찾고 점수를 계산한다.

결합 선택은 [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)처럼 바깥 결합 겹으로 분리하고 평균 개선과 겹별 승수 조건을 함께 보는 방식이 가장 재현 가능하다.
[7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)의 최소 개선 폭과 단순 결합 복귀 규칙도 같은 목적의 유용한 안전장치다.

단일 보류 집합, 저장된 공개 점수와 실행 출력이 없는 본문 수치는 후보 발굴에는 쓸 수 있지만 결론 확정에는 부족하다.
[17번](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602)의 단일 보류 검증과 [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)의 제목 점수는 공통 5겹 재실행으로 확인해야 한다.

### 합성 데이터 규칙

가장 강하게 반복된 생성 규칙은 `daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours`다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 대회 자료에서 위반이 없음을 계산했고, 26위부터 37위까지의 선행 분석도 [공식 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)의 네 값 완전 관측 421,427행에서 위반 0개를 재확인했다.

이 규칙은 사람 행동의 보편 법칙보다 합성 생성기의 흔적으로 보는 편이 타당하다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)과 [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 후보 원본 자료에서 같은 제약이 자주 깨진다고 보고한다.

수치 열은 부드러운 연속량인 동시에 정확한 값과 소수 자리가 반복되는 격자 구조를 가진다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 여러 나무 계열에서 정확값 부호화의 반복 이득을 보고하고, [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 정확값 조회 임베딩을 사용하며, [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 소수 첫째 자리 채널의 추가 이득을 보고한다.

후보 원본 자료를 대회 훈련 행에 더하는 근거는 부정적이다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 약 -0.00008의 손실을 보고하고, [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 그 자료가 실제 생성 원본인지 입증되지 않았다고 명시한다.

### 전처리

NaN을 자연스럽게 처리하는 나무 모형에서는 대체값으로 원래 열을 덮지 않고 원시 NaN 열 옆에 보조 열로 추가하는 후보가 가장 근거가 강하다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 대체값이 원래 열을 대신하면 해가 되고 나란히 있을 때만 도움이 된다는 제거 실험을 제시한다.

결측 형태가 목표를 직접 예측하는 정도와 결측 때문에 남은 정보량이 줄어드는 현상은 구분해야 한다.
[12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 결측 개수 단독 AUC 약 0.502를 보고하지만, [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)과 [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)은 결측이 많은 행에서 결합 성능이 낮아지는 현상을 확인한다.

훈련과 시험은 관측값 분포가 가까워도 결측 형태가 다를 수 있다.
[12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 결측 형태만으로 자료 출처를 구분하는 AUC 약 0.57을 보고하고, [36번](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)은 원시 값과 결측 표시의 분포 구분력을 따로 측정한다.

모든 중앙값, 표준화, 분위 구간과 반복 결측 대체는 바깥 겹의 학습 부분에서만 맞춰야 한다.
[18번](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann), [19번](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction), [21번](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb), [25번](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction), [31번](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)은 일부 자료 기반 변환을 분할 전에 맞춰 검증 분포를 미리 본다.

### 특성 생성

재현 우선순위가 가장 높은 파생 특성은 `other_screen = daily - social - gaming - work`다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 단독 AUC 약 0.765를 보고하고, [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 여러 씨앗과 강한 기준선의 제거 실험으로 이득을 확인하며, [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost)은 누출 방지 CatBoost에 넣는다.

값 단위 목표 부호화는 단일 특성 가운데 가장 큰 반복 이득 후보지만 반드시 중첩해야 한다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 약 +0.0023 OOF 개선을 보고하고, [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 세 나무 계열에 같은 구조를 구현한다.

소수 자리, 반올림값과 정확값 쌍은 조건부 후보로 남겨야 한다.
[11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 소수 첫째 자리의 작은 추가 이득을 보고하지만 36개 쌍의 32구간 목표 부호화는 -0.00040이었고, [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 다른 해상도의 정확값 쌍이 여러 나무에서 좋아졌다고 보고한다.

일반적인 행동 비율은 여러 노트북에 반복해서 등장하지만 근거는 약하다.
[4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [16번](https://www.kaggle.com/code/donmarch14/s6e8-lgbm)은 많은 비율을 쓰지만 개별 제거 실험이 없고, [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 로그 비율의 3겹 이득이 10겹에서 거의 사라졌다고 보고한다.

### 모형 구성

LightGBM, XGBoost와 CatBoost는 가장 반복적으로 검증된 강한 기준 모형 계열이다.
[4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)이 서로 다른 특성 구성에서 이 계열을 사용한다.

동일한 정확값 목표 부호화 설계를 HistGradientBoosting, LightGBM과 XGBoost로 비교하는 실험은 우선순위가 높다.
[26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 제목 공개 점수 0.96945, 0.96965, 0.96983을 각각 보고하지만 공통 저장 OOF가 없어 같은 분할 재실행이 필요하다.

Lookup-Transformer는 나무와 다른 오차를 만드는 새 구성원으로 가장 유망하다.
[10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 정확값 조회와 부드러운 수치 추세를 함께 표현하고, [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 공통 5겹 재훈련 뒤 이 계열의 결합 기여가 다른 새 변수 관점 모형보다 뚜렷하게 컸다고 보고한다.

RealMLP와 TabM은 함수 계열 다양성 후보지만 공개 구현을 그대로 비교하면 안 된다.
[19번](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction)은 목표 부호화를 바깥 겹 안에서 만들지만 다른 전처리를 겹 밖에서 맞추고, [22번](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)과 [24번](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction)은 목표 누출이 있어 모두 수정 후 재평가가 필요하다.

### 앙상블

구성원 수보다 기존 묶음과 다른 오차가 더 중요하다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 단독 OOF가 약한 신경망도 낮은 상관 때문에 가중치를 얻고, 단독 OOF가 높은 XGBoost도 기존 묶음과 상관이 0.998이면 기여가 거의 없다고 보고한다.

첫 결합 기준은 공통 5겹 OOF를 로짓으로 바꾸고 각 바깥 결합 겹에서 선형 모형을 다시 맞추는 방식이 적합하다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 음수 보정 계수를 허용한 로짓 결합의 장점을 반복해서 보여 준다.

순위 평균은 확률 척도가 다른 소수의 구성원을 결합하는 간단한 기준선이지만 OOF 없이 공개 제출만 고르면 근거가 약하다.
[4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble)은 같은 분할의 나무 예측을 순위 결합하지만, [8번](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092)과 [30번](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 공개 제출 및 공개 점수만으로 구성원을 선택한다.

결측 구간 상호작용은 반복 방향은 양성이지만 우선순위는 낮다.
[3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack), [9번](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)이 약 0.00002부터 0.00003의 개선을 보고하지만 공개 순위표 분해능보다 작아 큰 OOF의 짝지은 검증과 단순 결합 복귀 규칙이 필요하다.

외부 제출 파일은 연구용 OOF 결합과 분리해야 한다.
[1번](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092), [8번](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092), [30번](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 공개 점수는 높거나 활용이 쉽지만 원천 훈련, 공통 OOF와 구성원 선택 검증이 없다.

## 반복 주장, 충돌 결과와 근거가 약한 주장

### 여러 노트북이 반복한 주장

| 주장 | 반복 근거 | 종합 판정 |
| --- | --- | --- |
| 2단 결합에는 공통 OOF 분할이 필요하다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [28번](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01), [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge) | 매우 강함 |
| 목표 부호화는 바깥 겹 안에 중첩해야 한다 | [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983) | 매우 강함 |
| 시간 예산 제약과 `other_screen`이 유용하다 | [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost), [공식 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data) | 강함 |
| 정확값 격자와 값 단위 목표 부호화가 중요하다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041), [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945) | 단일 값은 강함 |
| 앙상블에는 단독 점수보다 오차 다양성이 중요하다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [9번](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new) | 강함 |

### 서로 충돌하거나 조건에 따라 달라진 결과

| 쟁점 | 한쪽 결과 | 반대 또는 제한 결과 | 해석 |
| --- | --- | --- | --- |
| 결측 표시의 직접 예측 가치 | [4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost)이 사용한다 | [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 약 +0.00001, [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 결측 개수 AUC 약 0.502를 보고한다 | 직접 목표 신호는 약하지만 행별 정보량과 자료 출처 신호일 수 있다 |
| 쌍 목표 부호화 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 정확값 쌍의 반복 이득을 보고한다 | [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 32구간 쌍에서 -0.00040을 보고한다 | 해상도, 평활화와 기준선에 따라 부호가 바뀐다 |
| 일반 비율 변수 | [4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [17번](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602)이 널리 사용한다 | [10번](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)은 3겹 이득이 10겹에서 사라졌다고 보고한다 | 의미가 그럴듯하다는 이유만으로 묶어 넣지 말고 하나씩 제거해야 한다 |
| OOF 정밀도 | [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)은 입력을 `float32`로 낮춘다 | [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)은 높은 상관의 로짓 결합에 `float64`가 필요하다고 보고한다 | 결합 산출물은 `float64`로 통일해 직접 비교하는 편이 안전하다 |
| 원본 후보 자료 추가 | 일부 노트북은 원본 자료 분기를 준비한다 | [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 손실을 보고하고 [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new)은 계보를 부정한다 | 행 추가보다 생성 규칙 진단에만 사용한다 |

### 근거가 약한 주장

- 공개 제출 파일만 결합해 얻은 높은 점수가 비공개 평가에서도 유지된다는 주장은 [1번](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97092), [8번](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092), [30번](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)에 공통으로 OOF 근거가 없다.
- 많은 행동 비율과 임계값이 점수를 높인다는 주장은 [4번](https://www.kaggle.com/code/koushikkumardinda/tps-s6e8-eda-advanced-feats-weighted-ensemble), [5번](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm), [17번](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602), [32번](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)에 개별 제거 실험이 없다.
- 훈련과 시험 분포 이동이 없다는 주장은 그림만 본 [14번](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition)과 [32번](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)보다 다변수 구분 검증을 한 [36번](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)의 근거를 우선해야 한다.
- 제목 공개 점수만으로 XGBoost가 HistGradientBoosting과 LightGBM보다 낫다는 주장은 [26번](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)의 공통 저장 OOF가 없어 확정할 수 없다.
- 결측 구간별 앙상블이 일반화된다는 주장은 [3번](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend), [7번](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack), [9번](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)의 개선 폭이 0.00002부터 0.00003 수준이라 추가 반복이 필요하다.

## 재현 가치가 높은 후보

우선순위는 예상 점수보다 검증 가능성, 반복 근거와 후속 실험의 정보량을 기준으로 정했다.

1. [36번의 자료 감사](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)를 재현해 원시 값, 결측 표시, 둘의 결합이 만드는 훈련 및 시험 구분 AUC와 정확한 중복 수를 기준 자료로 남긴다.
2. 모든 실험에 공통 5겹과 행별 겹 식별자를 고정하고, [34번](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)처럼 OOF, 시험 예측, 파일 해시와 구성원 계보를 함께 저장한다.
3. 원시 NaN을 유지한 강한 나무 기준선에 `other_screen`, 값 단위 목표 및 빈도 부호화, 소수 첫째 자리 채널을 하나씩 더하는 제거 실험을 [11번](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t), [12번](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new), [13번](https://www.kaggle.com/code/donmarch14/s6e8-catboost)의 공통 양성 후보로 실행한다.
4. [26번 HistGradientBoosting](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29번 LightGBM](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33번 XGBoost](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)를 같은 5겹, 같은 특성, 같은 중첩 목표 부호화로 다시 실행한다.
5. [10번 Lookup-Transformer](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)를 공통 5겹으로 다시 훈련해 나무 기준선과의 OOF 순위 상관, 잔차 상관과 제외 전후 결합 기여를 측정한다.
6. 결합은 [3번의 중첩 로짓 결합](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend)을 첫 기준으로 두고, [34번의 겹별 승수 및 단순 모형 복귀 조건](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)을 적용한다.
7. RealMLP와 TabM은 [19번의 바깥 겹 안 목표 부호화](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction)를 바탕으로 모든 전처리까지 겹 안으로 옮긴 뒤 오차 다양성 후보로만 평가한다.
8. 공개 제출 결합은 자체 OOF 연구와 분리하고, 출처, `id`, 겹 식별자와 원천 OOF가 모두 확인된 경우에만 마지막 제출 후보로 사용한다.

## 한계

이 문서는 2026-08-10에 고정한 최신 공개 판본의 정적 코드 분석을 종합하며, 조사 이후 득표 수와 판본은 바뀔 수 있다.
37개 원문은 모두 확인되었지만 여러 노트북이 외부 OOF, 직렬화 모형 또는 실행 출력이 제거된 코드를 사용하므로 모든 훈련 수치를 현지에서 다시 실행한 것은 아니다.
제목의 공개 점수는 작성자가 밝힌 값이며 제출 기록을 별도로 검증한 값은 아니다.
따라서 이 문서의 재사용 우선순위는 순위표 점수의 재현 보장보다 엄격한 후속 실험의 출발점을 고르는 데 목적이 있다.
