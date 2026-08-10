# 유사 Playground Series 에피소드 목록과 상위 솔루션 소재

S6E8 (Predicting Smartphone Addiction, 이진 분류, ROC AUC, 원본 데이터셋 기반 합성 tabular)과 문제 형태가 같은 과거 에피소드를 선별하고, 각 대회의 상위 솔루션 공개 스레드를 정리한 문서다.
스냅샷 기준일은 2026-08-10이다.
관련 이슈: [#25](https://github.com/tmheo/predicting-smartphone-addiction/issues/25) (맵 [#24](https://github.com/tmheo/predicting-smartphone-addiction/issues/24)의 자식).

## 선별 기준

- Kaggle 공개 API로 Playground Series 전 에피소드의 평가 지표를 조회해, 단일 타깃 이진 분류 + ROC AUC 에피소드만 추렸다.
- 범위는 S4E1 ~ S6E7을 우선으로 했고, 해당 조건을 만족하는 에피소드는 10개다.
- 그 이전 시즌은 도메인 유사도가 특히 높은 S3E24 (건강 바이오시그널 이진 AUC) 하나만 포함했다.
- S4E3 (Steel Plate Defect)은 지표가 다중 레이블 평균 AUC라 제외했다.
- S3E3, S3E7 등 시즌 3의 다른 이진 AUC 에피소드와 2021~2022 Tabular Playground의 AUC 대회들은 데이터 생성 방식과 메타가 지금과 달라 제외했다.
- 문서의 모든 스레드 링크는 실제 접근으로 존재를 확인했고, 순위 표기는 farid.one의 Kaggle Solutions 집계와 교차 확인했다.
- 1위 writeup이 공개되지 않은 대회는 확인된 최상위 공개 writeup을 대신 실었다.

## 요약 표

| 에피소드 | 대회명 | 시기 | 원본 데이터셋 (모두 Kaggle 공개) | 최상위 공개 writeup | 원본/생성기 아티팩트가 순위를 갈랐나 |
| --- | --- | --- | --- | --- | --- |
| S4E1 | Bank Churn | 2024-01 | Bank Customer Churn Prediction | 2위 (1위 미공개) | **예** (원본 정확값 매칭 피처, 원본 이어붙이기) |
| S4E7 | Insurance Cross Selling | 2024-07 | Health Insurance Cross Sell | 1위 | 부분 (라벨 뒤집기 후처리 트릭) |
| S4E10 | Loan Approval | 2024-10 | Loan Approval Prediction | 1위 | 부분 (원본 병합 + 수치의 범주 취급) |
| S5E3 | Rainfall | 2025-03 | Rainfall Prediction using ML | 2위 (1위 미공개) | **예** (극소 데이터, LB 프로빙으로 public AUC 1.0) |
| S5E8 | Bank (정기예금) | 2025-08 | Bank Marketing Dataset | 1위 | 아니오 (OOF 앙상블 물량전) |
| S5E11 | Loan Payback | 2025-11 | Loan Prediction dataset 2025 | 1위 | 아니오 (피처 엔지니어링 + 앙상블) |
| S5E12 | Diabetes | 2025-12 | Diabetes Health Indicators | 1위 | **예** (의도적 분포 왜곡, ID 위치 기반 shift 분석) |
| S6E2 | Heart Disease | 2026-02 | Heart disease prediction | 1위 | 부분 (원본 데이터 타깃 통계 피처) |
| S6E3 | Customer Churn | 2026-03 | Telco Customer Churn (IBM) | 1위 | **예** (원본 스냅 피처, 정확값/자릿수 아티팩트) |
| S6E5 | F1 Pit Stops | 2026-05 | F1 Strategy Dataset | 1위 | 부분 (원본 대비 분포 변화 분석이 피처 선택을 결정) |
| S3E24 | Smoker Status | 2023-11 | Smoker Status Prediction using Bio-Signals | 3위 (1~2위 미공개) | 아니오 (원본 병합은 보조적) |

## 에피소드별 상세

### S4E1: Binary Classification with a Bank Churn Dataset

- 대회: <https://www.kaggle.com/competitions/playground-series-s4e1>, 지표 ROC AUC.
- 원본: [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction) (공개).
- 상위 솔루션:
  - [2위 솔루션](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472496): 1~10개 피처 부분집합을 전부 만들어 "이 조합이 원본 데이터에 그대로 존재하는가"를 이진 지표 피처로 넣었다. 합성 데이터가 원본 행 값을 재조합해 만들어진다는 성질을 직접 공략한 사례다. LGBM + AutoGluon 앙상블에 라벨 뒤집기 트릭을 더했다.
  - [3위 솔루션](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472413): 원본 데이터셋을 대회 train 앞에 이어붙이되 두 번 이어붙이는 편이 private 점수가 가장 좋았다고 밝혔다. TF-IDF+SVD, 다중 인코더, 7모델 Ridge 가중 앙상블.
  - 1위 writeup은 공개 스레드를 확인하지 못했다.
- 특이점: 상위권이 모두 원본 데이터셋을 직접 활용했고, 특히 2위의 정확값 부분집합 매칭은 S6E8 디스커션에서 확인된 "정확값 인코딩" 계열의 원형이다.

### S4E7: Binary Classification of Insurance Cross Selling

- 대회: <https://www.kaggle.com/competitions/playground-series-s4e7>, 지표 ROC AUC.
- 원본: [Health Insurance Cross Sell Prediction Data](https://www.kaggle.com/datasets/annantkumarsingh/health-insurance-cross-sell-prediction-data) (공개).
- 상위 솔루션:
  - [1위 솔루션](https://www.kaggle.com/c/playground-series-s4e7/discussion/523404) (Ravi Ramakrishnan 팀): 78개 약학습기의 3단 스택. previously_insured, vehicle_damage 같은 핵심 피처로 세그먼트를 나눠 세그먼트별 모델을 따로 학습했다. 공개적으로 알려져 있던 라벨 반전(target reversal) 후처리를 전 제출에 적용했다.
  - [AutoML Grand Prix 1위](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/516475) (Vopani): CatBoost 단일 모델.
- 특이점: 1,100만 행 규모의 대형 합성 데이터라 물량전 성격이 강했다. train/test 간 라벨이 뒤집힌 표본을 찾아 후처리하는 트릭이 통했다.

### S4E10: Loan Approval Prediction

- 대회: <https://www.kaggle.com/competitions/playground-series-s4e10>, 지표 ROC AUC.
- 원본: [Loan Approval Prediction](https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction) (공개).
- 상위 솔루션:
  - [1위 솔루션 "CatBoost All The Way Down"](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/543725) (Hardy Xu): 원본 데이터셋 포함, 수치 피처를 수치와 범주 양쪽 복사본으로 취급, 다른 모델 예측을 baseline으로 넣은 CatBoost 재학습. "플레이그라운드에서 피처 엔지니어링은 별 소용이 없다"고 명시했다.
  - [4위 접근](https://www.kaggle.com/competitions/playground-series-s4e10/writeups/ravi-ramakrishnan-rank-4-approach-thoughtful-model) (Ravi Ramakrishnan).
- 특이점: 수치 피처의 범주 취급이 통한 것은 합성 데이터 값이 눈금 위에 몰려 있기 때문으로, S6E8의 정확값 인사이트와 같은 계열이다.

### S5E3: Binary Prediction with a Rainfall Dataset

- 대회: <https://www.kaggle.com/competitions/playground-series-s5e3>, 지표 ROC AUC.
- 원본: [Rainfall Prediction using Machine Learning](https://www.kaggle.com/datasets/subho117/rainfall-prediction-using-machine-learning) (공개, 366행).
- 상위 솔루션:
  - [2위 솔루션 "GBDT + NN + SVR + Original Data"](https://www.kaggle.com/competitions/playground-series-s5e3/discussion/571176) (Chris Deotte): 원본 366행을 행으로 이어붙이거나 열로 병합하는 두 방식 모두 사용. 연 단위 GroupKFold 6폴드, 동등 가중 블렌드, 피처 엔지니어링 없음.
  - [18위 솔루션](https://www.kaggle.com/competitions/playground-series-s5e3/writeups/spiritmilk-18th-place-solution-single-xgboost-with) (Spiritmilk): 커스텀 AUC 손실의 단일 XGBoost.
  - 1위 writeup은 공개 스레드를 확인하지 못했다.
- 특이점 스레드:
  - [LB 프로빙 해설](https://www.kaggle.com/competitions/playground-series-s5e3/discussion/568718): public test가 146행뿐이라 AUC의 순위 정의를 이용한 프로빙으로 public 라벨을 복원할 수 있었고, 실제로 public AUC 1.0 제출이 10개 나왔다.
  - [원본은 홍콩 2015~2016 실측 데이터](https://www.kaggle.com/competitions/playground-series-s5e3/discussion/566908): 원본 데이터셋의 출처가 실제 기상 관측 자료로 역추적됐다.
- 특이점: train 2,190행짜리 극소 데이터라 public LB가 사실상 무의미해졌고, private는 CV를 믿은 사람들이 가져갔다. 데이터가 작을 때 생기는 병리 현상의 표본 같은 대회다.

### S5E8: Binary Classification with a Bank Dataset

- 대회: <https://www.kaggle.com/competitions/playground-series-s5e8>, 지표 ROC AUC.
- 원본: [Bank Marketing Dataset](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full) (공개, UCI 포르투갈 은행 텔레마케팅).
- 상위 솔루션:
  - [1위 솔루션 "JAPE: Just Another Proper Ensemble"](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/603210) (Optimistix): 136개 이상의 OOF를 AutoGluon으로 앙상블. 원본 데이터도 활용했으나 결정적이지는 않았고, 라벨 뒤집기 트릭은 +0.00003 수준의 미세 효과였다.
  - [2위 "Yet another ensemble"](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/2nd-place-yet-another-ensemble) (Mahog).
  - [3위 "OOF Stacking + AutoGluon"](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon).
- 특이점: 트릭 없이 OOF 물량과 스태킹 완성도가 순위를 갈랐다. S6E8과 가장 비슷한 "표준" 승리 공식의 최신 표본이다.

### S5E11: Predicting Loan Payback

- 대회: <https://www.kaggle.com/competitions/playground-series-s5e11>, 지표 ROC AUC.
- 원본: [Loan Prediction dataset 2025](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025) (공개).
- 상위 솔루션:
  - [1위 "A lot of features, a lot of models, and a little bit of luck"](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an) (Mahog): 자릿수 조합, 타깃 인코딩, 범주 상호작용 등 대규모 피처 엔지니어링. 단일 XGBoost만으로 2위감이었고, 100개 모델을 Ridge와 힐클라이밍으로 결합했다.
  - [2위 "7 models, but 1 was also enough"](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/2nd-place-solution-7-models-but-1-was-also-enou).
- 특이점: 이 대회에서는 피처 엔지니어링이 크게 통했다. "자릿수(digit) 피처"는 합성 데이터의 값 생성 습관을 공략하는 기법이라 S6E8에서도 시도 가치가 있다.

### S5E12: Diabetes Prediction Challenge

- 대회: <https://www.kaggle.com/competitions/playground-series-s5e12>, 지표 ROC AUC.
- 원본: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset) (공개).
- 상위 솔루션:
  - [1위 "Hill Climbing + Ridge Ensemble"](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl) (wind1234it): 대부분의 모델에서 원본 데이터셋을 train에 이어붙여 표본을 늘렸다. 힐클라이밍 선택 + Ridge 스택.
  - [2위 "Winning based on ID Shift Analysis"](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an) (DaylightH): 적대적 검증으로 train 뒷부분(ID가 큰 쪽)일수록 test 분포에 가깝다는 것을 발견하고, 뒷부분 16배, 원본 8배, 앞부분 1배의 표본 가중치를 줬다. 생성기가 만든 위치 기반 분포 이동 자체가 결정적 신호였다.
- 특이점 스레드:
  - ["Kaggle messed up this dataset on purpose"](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/652262) (Tilii): 핵심 피처(HbA1c, 혈당)가 제거되고 남은 피처도 이산화되거나 범위가 좁혀지는 등 의도적 왜곡이 있었다는 분석.
  - [원본이 concept shift 대응에 도움이 될 수 있다](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/663033).
- 특이점: 생성기 아티팩트(행 위치에 따른 분포 이동) 진단이 우승을 갈랐다. S6E8의 "생성기 산술 오류 잔차" 발견과 같은 부류의 접근이다.

### S6E2: Predicting Heart Disease

- 대회: <https://www.kaggle.com/competitions/playground-series-s6e2>, 지표 ROC AUC.
- 원본: [Heart disease prediction dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease) (공개).
- 상위 솔루션:
  - [1위 "Diversity, Selection, and Trusting the CV-LB Relation"](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t) (Masaya Kawamata): 약 150개의 다양한 예측(비닝, 자릿수 추출, 유전 프로그래밍 피처, DVAE 임베딩 등)을 만들고 Optuna로 부분집합을 골라 Ridge로 결합. 원본 데이터셋에서 타깃 평균, WoE, 엔트로피 통계를 피처로 추출했다.
  - [2위 "Avoid leaks and overfitting"](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/2nd-place-solution-avoid-leaks-and-overfitting) (satokin13m): 105개 모델을 상관 기반으로 6개까지 줄여 NN 메타러너로 스택. 제목의 leak은 대회 리크가 아니라 타깃 인코딩을 CV 루프 안에서 계산해야 한다는 방법론 얘기다.
- 특이점: 건강 도메인 + 이진 AUC + 원본 타깃 통계 활용이라는 점에서 S6E8과 문제 형태가 가장 비슷한 최근 대회 중 하나다.

### S6E3: Predict Customer Churn

- 대회: <https://www.kaggle.com/competitions/playground-series-s6e3>, 지표 ROC AUC.
- 원본: [Telco Customer Churn (IBM sample)](https://www.kaggle.com/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm) (공개, 7,032행).
- 상위 솔루션:
  - [1위 "GPT5.4, Gemini3.1, ClaudeOpus4.6 - KGMON Playbook!"](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm) (Chris Deotte): 합성 값을 원본 데이터의 최근접 값으로 되돌리는 스냅(snap) 피처, cKDTree 최근접 이웃 조회, 자릿수 추출로 생성기 아티팩트를 신호로 썼다. 원본 데이터만으로 계산한 타깃 인코딩 prior도 사용. 4단 스택에 GBDT 90개 + 딥러닝 60개, 코드는 전부 LLM 에이전트가 작성했다.
  - [3위 "An Ensemble of 100 OOFs"](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/3rd-place-solution-an-ensemble-of-100-oofs) (Traiko Dinev).
  - [5위 "149 Models -> 6 Meta Models -> 3 Blends"](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/5th-place-solution-149-models-6-meta-models).
- 특이점: "합성 데이터 생성을 노이즈가 아니라 구조화된 신호로 취급한다"는 원칙을 가장 철저히 실행한 대회다. S6E8의 정확값/생성기 아티팩트 인사이트와 직결된다.

### S6E5: Predicting F1 Pit Stops

- 대회: <https://www.kaggle.com/competitions/playground-series-s6e5>, 지표 ROC AUC.
- 원본: [F1 Strategy Dataset](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction) (공개). 주최 측이 예측이 자명해지는 것을 막으려고 Normalized_TyreLife 피처를 의도적으로 뺐다.
- 상위 솔루션:
  - [1위 "By the skin of my teeth"](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth) (Optimistix): 0.00001 차이 승리. XGB/LGBM/CatBoost/RealMLP/TabM/FT-Transformer 등 대규모 다양화 + AutoML(AutoGluon, LightAutoML, FLAML). 원본과 대회 데이터 사이 Driver 피처의 분포 차이를 보고 일부 모델은 Driver를 빼고 학습했다.
  - [4위 "5 day rush"](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/4th-place-5-day-rush) (Mahog).
  - [5위 "a 99-model logit stack"](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/5th-place-solution-a-99-model-logit-stack).
- 특이점: 원본 대비 분포 변화 분석이 피처 채택 결정을 이끌었다. 초박빙 승부라 앙상블 마무리 품질이 곧 순위였다.

### S3E24: Binary Prediction of Smoker Status using Bio-Signals

- 대회: <https://www.kaggle.com/competitions/playground-series-s3e24>, 지표 ROC AUC.
- 원본: [Smoker Status Prediction using Bio-Signals](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals) (공개).
- 상위 솔루션:
  - [3위 솔루션](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/455248) (Ravi Ramakrishnan): CatBoost/LGBM/XGB/RF/NN의 Optuna 가중 앙상블. 원본 데이터셋을 함께 썼지만 초반에는 오히려 손해였다고 밝혔다.
  - [4위 "robust Hill Climbing"](https://www.kaggle.com/competitions/playground-series-s3e24/writeups/aldparis-4-th-place-solution-robust-hill-climbing) (aldparis).
  - 1~2위 writeup은 공개 스레드를 확인하지 못했다.
- 특이점: 건강/생활습관 이진 AUC라 도메인은 S6E8과 가장 가깝지만, 승부는 평범한 앙상블 완성도로 갈렸다.

## 원본/생성기 아티팩트가 순위를 가른 에피소드

다음 4개는 원본 데이터셋 분석이나 생성기 아티팩트 활용이 실제로 최상위 순위를 결정했다.

1. **S6E3**: 원본 스냅 피처, 최근접 이웃 조회, 자릿수 아티팩트가 1위 솔루션의 핵심.
2. **S5E12**: 생성기가 만든 행 위치 기반 분포 이동을 진단한 표본 가중치가 2위, 원본 이어붙이기가 1위의 기반.
3. **S4E1**: 원본과의 정확값 부분집합 매칭 피처(2위), 원본 이중 이어붙이기(3위).
4. **S5E3**: 극소 데이터에서 LB 프로빙으로 public AUC 1.0, 원본 행/열 병합이 2위의 재료.

다만 S6E8은 디스커션 분석 결과 train/test 중복 행이 0개라 라벨 룩업형 트릭은 배제됐고, 정확값 눈금과 생성기 산술 오류 잔차가 유효한 신호로 확인된 상태다.
따라서 위 4개 중에서도 "룩업"이 아니라 "아티팩트를 피처로 바꾼" 사례(S6E3, S5E12, S4E1의 2위)가 직접 참고 대상이다.

## 정독 우선순위

1. **S6E3 1위 (Deotte, KGMON Playbook)**: 정확값 스냅, 원본 최근접 이웃, 자릿수 추출 등 S6E8 디스커션에서 확인된 신호들과 정확히 같은 계열의 기법을 최고 완성도로 실행했다. 가장 최근 메타(4단 스택, LLM 에이전트 활용)까지 한 번에 배울 수 있다.
2. **S5E12 1~2위**: S6E8도 생성기의 의도적 변형(임계 규칙의 완만화, 산술 오류)이 확인된 대회라, 적대적 검증으로 생성기 왜곡을 진단하고 표본 가중치로 바꾸는 사고 과정이 그대로 이식 가능하다.
3. **S6E2 1~2위**: 건강 도메인 + 같은 시즌 + 원본 타깃 통계 피처. 150개 예측 풀에서 Optuna로 고르는 앙상블 구성법과 CV-LB 관계를 믿는 제출 선택 기준이 실전 지침으로 유용하다.
4. **S4E1 2위**: 원본 정확값 부분집합 매칭 피처의 원형. S6E8의 정확값 타깃 인코딩(+0.0032 확인됨)을 확장할 아이디어 원천이다.
5. **S5E8 1~3위**: 트릭이 없을 때의 표준 승리 공식(OOF 물량 + 스태킹). S6E8 후반전에서 순위를 지키는 기본기를 여기서 잡는다.
6. 나머지는 필요할 때 선택적으로 본다: S5E11 1위 (자릿수 피처와 대규모 FE가 통하는 조건), S4E10 1위 (수치의 범주 취급), S6E5 1위 (원본 대비 분포 변화 분석), S4E7 1위 (라벨 반전 후처리, 다만 S6E8은 중복 0이라 적용 불가), S5E3 (극소 데이터 병리 사례), S3E24 (도메인 참고용).
