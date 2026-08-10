# 디스커션 배치 B 리딩 노트: 피처 엔지니어링 / 모델링 / 커뮤니티

Kaggle Playground Series S6E8 (스마트폰 중독 예측, 이진 분류, ROC AUC) 디스커션 9개 스레드를 본문과 코멘트까지 전부 읽고 정리한 노트다.
읽은 날짜: 2026-08-10.
페이지가 JS 렌더링이라 Jina Reader와 Playwright 헤드리스 브라우저를 병행해 본문과 코멘트 전문을 수집했다.

## 1. XGBoost + Optuna on GPU | 0.96514 LB - sharing what worked (732985)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985
작성자: Rugved Bane (502위), 코멘트 8개.

### 핵심 주장

- LightGBM에서 XGBoost로 바꾸면 작지만 일관된 이득이 있었다 (0.96488 -> 0.96514).
- XGBoost를 device='cuda' + tree_method='hist'로 돌리면 Optuna 200 트라이얼이 현실적으로 가능해진다 (T4 GPU 기준 약 1.5시간).
- 가장 강한 엔지니어링 피처는 screen_time_bin 구간화였고, weekend_gap과 leftover_screen 추가로 0.96514 -> 0.96578, 비율 피처(work_ratio, social_ratio, gaming_ratio 등)로 0.96602까지 올렸다.
- 중요도 낮은 피처를 제거하면 안 된다: 저중요도 피처를 빼자 0.965 -> 0.894로 폭락했다고 주장.
- 결론적으로 하이퍼파라미터 튜닝보다 피처 엔지니어링이 점수를 더 움직였다.

### 코멘트의 반박과 보강

- Zih-Chen Hung (516위): 같은 실험을 LGBM으로 재현했더니 약한 피처(gender, stress_level, academic_work)를 빼도 0.9491 -> 0.9489로 거의 안 움직였다.
  본문이 주장한 7포인트 폭락은 재현되지 않았고, 임계값 설정에 의문을 제기했다.
  또 Optuna 25 트라이얼만으로 LB 0.96572를 얻어, 튜닝은 일찍 수확 체감에 도달한다고 지적했다.
- 작성자도 인정: 100 -> 200 트라이얼 확장은 0.003~0.004% 수준의 미미한 이득이라 다시는 안 하겠다고 답했다.
- Tilii (30위): n_estimators는 튜닝 대상이 아니다.
  아주 큰 값(예: 100,000)을 넣고 early stopping을 쓰는 게 맞다.
  learning_rate도 탐색 단계에서는 0.02~0.05로 고정하고, 최종 프로덕션 런에서만 0.01 또는 0.005로 낮추라고 조언했다.
- GurSimran: Kaggle 환경의 LightGBM은 CUDA 빌드가 아니므로 GPU는 XGBoost에서만 쓸 수 있다.

### 우리 모델링에 주는 시사점

- 튜닝 예산은 소규모(25~50 트라이얼)로 잡고, 남는 시간은 피처 검증에 쓴다.
- n_estimators 대신 early stopping, learning_rate 고정 전략을 기본으로 채택한다.
- "저중요도 피처 제거 시 폭락" 주장은 재현 실패 사례가 있으므로 그대로 믿지 말고, 피처 제거는 자체 CV로 직접 검증한다.
- weekend_gap, leftover_screen(잔여 스크린타임), 비율 피처는 우리도 후보로 실험할 가치가 있다.

## 2. Single Model Feature Engineering technique... (733023)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023
작성자: hamzah (182위), 코멘트 4개.

### 핵심 주장

- 공개 노트북들의 OOF 예측과 테스트 예측을 메타 피처로 단일 XGBoost에 넣고, 그 메타 피처들의 다항 상호작용 피처까지 만들면 단일 모델로 CV 0.96947 / LB 0.97059가 나온다.
- Chris Deotte의 OOF 메타 피처 아이디어에 다항 상호작용을 얹은 것이 본인의 변형이며, S6E3에서 16위 솔루션에 썼던 기법이다.

### 코멘트의 반박과 보강

- Optimistix (3위): 다른 모델 M2의 OOF/테스트 예측을 M1에 넣는 순간 그것은 단일 모델이 아니라 앙상블이다.
- siukeitin: 정확히는 스태킹이다.
  sklearn의 StackingClassifier에서 passthrough=True를 켠 것과 동일하고, 이 "단일 모델"은 스태킹 용어로 final estimator다.
- tcspecialist: 실전 검증 보강.
  본인의 과거 6개 런의 OOF/제출 예측을 같은 방식으로 추가했더니 0.96699 -> 0.96717로 올랐다.

### 우리 모델링에 주는 시사점

- 이 대회 상위권 점수(0.970+)는 사실상 스태킹 없이는 어렵다.
- 우리도 모든 실험 런에서 OOF 예측과 테스트 예측을 반드시 저장하는 파이프라인을 처음부터 갖춰야 한다.
- 메타 피처의 다항 상호작용은 저비용 추가 실험 거리다.
- 다만 "단일 모델"이라는 프레이밍에 속지 말 것: 이는 앙상블이고, OOF 생성 시 폴드 누수 관리가 핵심이다.

## 3. Feature Engineering: What Works, What Fails, and the Math Behind It (733541)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541
작성자: Muhammad Faheem (390위), 코멘트 2개.

### 핵심 주장

- 10-fold 층화 ablation 하네스로 피처 그룹을 누적 추가하며 OOF AUC 변화를 측정했다.
  베이스라인(원본 피처) 0.96394, 행동 비율 피처 추가 -0.00023, 구조적 아티팩트 추가 +0.00088, 복잡한 행동 수식 피처 추가 -0.00043.
- 이긴 피처는 심리학이 아니라 생성기의 결함을 역공학한 것들이다.
  other_screen: daily_screen_time_hours - (social + gaming + work) 잔차로, 생성기 내부 산술이 안 맞는 행을 노출한다.
  _decimals: 각 시간 컬럼 소수부 문자열 길이로, 생성기의 부동소수점 반올림 흔적을 잡는다.
- 적대적 검증(train vs test 분류기) AUC는 0.56441로 약한 시프트가 있으나, 시프트는 원본 컬럼(app_opens_per_day, notifications_per_day)이 주도하고 새 구조적 피처는 상위 10위에도 없어 안전하다.
- 널 임포턴스 검사에서 daily_screen_time_hours_decimals는 노이즈 기준선의 11.68배, other_screen_abs는 8.83배 게인으로 진짜 신호임을 확인했다.
- 결론: 손으로 만든 행동 비율은 버리고, 결측 플래그도 공변량 시프트 함정이므로 버리고, 생성기의 산술/반올림 실수를 파라.

### 코멘트의 반박과 보강

- Trish Cornelissen: 결측 플래그를 버리면 결측치는 어떻게 채우는가?
- 작성자: 아예 안 채운다.
  XGBoost/LightGBM은 NaN을 네이티브로 처리하며 결측 행의 분기 방향을 게인 기준으로 스스로 학습한다.

### 우리 모델링에 주는 시사점

- 우리 피처 실험도 단발 CV가 아니라 누적 ablation + 적대적 검증 + 널 임포턴스의 3중 검증 체계를 갖추는 것이 좋다.
- other_screen 잔차와 _decimals 피처는 이 대회에서 수치로 검증된 몇 안 되는 유효 피처이므로 최우선 채택 후보다.
- 결측치는 채우지 말고 NaN 그대로 트리 모델에 넘기는 것을 기본값으로 한다.

## 4. I Injected Real-World Data Into My Model and Every Single Fold Got Worse (733552)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552
작성자: Muhammad Faheem (398위), 코멘트 2개.

### 핵심 주장

- 합성 데이터의 원본으로 보이는 실데이터 7,500행(결측 0, 양성 비율 70.77%로 합성 train의 70.94%와 거의 동일)을 찾아 50배 샘플 가중치로 훈련에 주입했더니 10개 폴드 전부가 베이스라인(0.963~0.967) 아래로 떨어졌다 (약 0.002~0.005 AUC 하락).
- "실데이터 논리가 합성 라벨에 페널티를 받는다"는 해석보다, 620,000행 폴드에서 7,500행에 50배 가중치를 주면 그 작은 표본이 손실을 지배하게 되는 가중치 과잉 문제라는 해석이 더 그럴듯하다고 분석했다.
- 단일 실험 결과는 여러 스토리를 동시에 지지하므로, 대안 가설을 테스트하기 전에는 결론을 내리지 말라는 방법론적 교훈으로 마무리했다.

### 코멘트의 반박과 보강

- Tilii (32위): 원본 데이터셋은 "거의" 안 주어지는 게 아니라 절대 같이 배포되지 않으며, 탐정 놀이가 필요한 것도 아니고 링크가 Overview 섹션에 이미 있다.
  이 데이터셋은 합성 후 값을 무작위로 제거하는 변형이 가해졌다 (관련 스레드 732428, 732434 참조).
  변형이 없던 과거 대회에서도 50배 가중치는 통하지 않았을 것이다.
  원본 데이터는 훈련이 아니라 검증에만 쓰는 것이 맞으므로, 2x/5x/10x 가중치 스윕 계획은 접으라고 조언했다.
  "ARDIS - Always Read the DIScussions"라는 격언으로 마무리.
- 작성자는 계획을 접었다고 수긍했다.

### 우리 모델링에 주는 시사점

- 원본 실데이터는 훈련 데이터에 섞지 않는다.
  쓰더라도 검증 보조나 분포 참고 용도로만 쓴다.
- Overview 페이지에서 원본 데이터셋 링크를 확인해 둔다.
- 실험 결과 해석 시 "가중치/사용법 오류" 가설을 먼저 배제한 뒤에 데이터 자체를 탓하는 순서를 지킨다.

## 5. LightGBM Gain Importance: What the Model Actually Cares About (732256)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256
작성자: Muhammad Faheem (398위), 코멘트 2개.

### 핵심 주장

- 5-fold 층화 LightGBM에서 Gain Importance(Split이 아니라)를 뽑으면 원본 5개 피처가 압도한다.
  daily_screen_time_hours (~180만 게인), social_media_hours (~81.3만), weekend_screen_time (~78.6만), notifications_per_day (~31.4만), app_opens_per_day (~31.0만).
- EDA에서 좋아 보이던 비율 피처(sleep_deficit, dopamine_ratio)는 트리가 이미 원본 컬럼으로 최적 경계를 찾고 있어서 feature fraction만 희석시켰고, OOF AUC가 0.96312 -> 0.96305로 떨어졌다.
- 명시적 결측 플래그와 범주형(gender, stress_level, academic_work_impact)은 중요도 최하위의 죽은 무게였다.

### 코멘트의 반박과 보강

- Dariush Afshar (21위): 결측 플래그가 죽은 무게라는 관찰에 수학적 근거를 추가했다.
  n_missing의 단독 AUC는 0.50172로 타겟 신호가 전무하다.
  그런데 12개 전 컬럼의 결측률이 train과 test에서 다르다 (social_media_hours 19.38% vs 16.00%, app_opens_per_day 11.67% vs 8.68%, academic_work_impact 6.40% vs 8.68% 등).
  행 수가 커서 이 차이는 최소 13 SE, 최대 40 SE 이상으로 통계적으로 확실하다.
  즉 is_missing 플래그는 타겟 신호는 없고 train/test 소속 정보만 담으므로 버려야 할 이유가 둘이며, 적대적 검증에서 이 플래그들이 분리력을 보여도 피처 드리프트로 오독하면 안 된다.
- 작성자: 이 설명이 본인의 LB 하락을 정확히 설명한다.
  missing_count 피처가 로컬 OOF는 +0.00009 (0.96452 -> 0.96461) 올렸지만 Public LB는 0.96568 -> 0.96567로 떨어졌다.
  전형적인 공변량 시프트다.

### 우리 모델링에 주는 시사점

- 결측 관련 피처(is_missing, missing_count)는 전부 배제한다.
  로컬 CV가 올라도 LB에서 배신하는 것이 수치로 확인된 함정이다.
- 적대적 검증을 돌릴 때 결측 지표가 만드는 분리력은 미리 예상하고 해석에서 걸러낸다.
- 피처 중요도는 Split이 아니라 Gain 기준으로 본다.

## 6. Plot Twist: Why My "Golden" EDA Features Dropped My CV Score (732223)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223
작성자: Muhammad Faheem (398위), 코멘트 4개.

### 핵심 주장

- EDA에서 아름다워 보인 3가지 피처가 전부 CV에서 실패했다.
  is_missing 플래그: 게인 중요도 거의 0 (네이티브 NaN 처리가 이미 최적화).
  sleep_deficit: KDE 플롯에서는 클래스가 갈리지만 트리는 직교 분할자라 이미 screen_time > X, sleep < Y로 쪼개고 있었고 CV만 떨어졌다.
  total_weekly_screen_time: 0.80 상관의 두 컬럼을 합쳤지만 원본 daily_screen_time_hours가 여전히 압도적 1위였고 합성 피처는 무시됐다.
- 원본 신호가 이미 강해서 (베이스라인 CV ~0.96312) 산술 조합은 희석만 일으킨다.

### 코멘트의 반박과 보강

- Dariush Afshar (21위): 선형 조합 실패에는 동의하지만 모든 피처 엔지니어링이 실패하는 것은 아니며, 구분이 핵심이다.
  생성기의 산술 제약을 인코딩한 other_screen = daily - (social + gaming + work) 잔차는 단독 ROC AUC 약 0.765다.
  5-fold에서 베이스라인 OOF 0.964048 -> 잔차 + 소수점 자릿수 피처 추가 시 0.964792 (+0.00074, 폴드 표준편차의 약 1.5배라 노이즈가 아님).
  주의: "소수점 1자리 행은 타겟 비율이 약 3%p 낮다"는 소문은 한 컬럼에서만 성립했다.
  결론: 트리가 이미 도달 가능한 경계를 재표현하는 피처는 실패하고, 데이터 생성 방식의 구조를 인코딩하는 피처만 성공한다.
- Georgy Mamarin (197위): 정밀 재측정으로 보강.
  결측 컬럼 개수의 단독 순위력은 0.502로 전무하고, 단일 컬럼의 결측/비결측 간 최대 타겟 비율 차이도 0.0042 (전체 비율 0.709 대비)로 미미하다.
  Dariush의 잔차 단독 AUC 0.7649로 재현 확인.
  other_screen 단독은 +0.00064 (시드 간 산포 0.00006)로, Dariush의 +0.00074 중 거의 전부가 잔차 몫이고 소수점 피처는 거의 아무것도 얹지 못한다.
  흥미로운 발견: gaming_hours와 work_study_hours는 강한 두 컬럼을 조건으로 놓으면 자체 신호가 없는데도 +0.00380을 기여하며, 그중 +0.00242를 잔차 하나가 회수한다.
  즉 이 두 컬럼은 중독 신호가 아니라 강한 컬럼에 대한 산술적 사실을 운반하는 통로다.
- 작성자: missing_count의 로컬 +0.00009가 LB에서 증발한 실험을 여기서도 공유하며, 결측 구조 기반 피처는 전부 생성기 노이즈에 과적합하는 것이라고 결론지었다.

### 우리 모델링에 주는 시사점

- EDA 시각화가 예쁜 피처와 모델에 유효한 피처는 다르다.
  선형 조합/차이/비율 피처는 기본적으로 기각 후보로 놓고 시작한다.
- other_screen 잔차는 복수의 독립 재측정(0.765 단독 AUC, +0.0006~0.0007 OOF)으로 검증된 이 대회 최고의 단일 엔지니어링 피처다.
- 소수점 자릿수 피처는 잔차와 함께 넣으면 한계 기여가 거의 없다는 반증이 있으므로, 넣더라도 단독 기여를 따로 측정한다.

## 7. Handling Class Imbalance & Missing Values in This Dataset (731764)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764
작성자: BhuvanNR (155위), 코멘트 5개.

### 핵심 주장

- 타겟 addicted_label은 약 71/29 (양성 490,474 / 음성 200,895)로 경미한 불균형이다.
  소수 클래스가 20만 행이나 되고 지표가 순위 품질을 보는 AUC이므로 영향은 제한적이다.
  가벼운 재가중(scale_pos_weight, class_weight='balanced')이면 충분하고 SMOTE는 불필요할 것이다.
- 거의 모든 피처에 결측이 있고 social_media_hours(약 13.4만), gaming_hours(약 12.7만)는 15~20% 수준이라 처리 방식이 점수를 움직일 것이다.
  is_missing 플래그, 그룹별 중앙값 대치, 트리 모델의 네이티브 NaN 처리 비교, gender는 결측을 별도 범주로 두기를 제안했다.
- Stratified K-Fold는 필수다.

### 코멘트의 반박과 보강

- broccoli beef: 결측 여부와 타겟의 독립성을 카이제곱 검정으로 전수 확인했다.
  12개 컬럼 중 app_opens_per_day만 p=0.025로 유의했으나 Cramer's V가 0.0027로 효과 크기는 사실상 0이다.
  total_missing도 p=0.32, V=0.0043.
  결측 구조에 설계된 신호가 있을 가능성은 통계적으로 매우 낮다.
- Tilii (32위): 합성 데이터 대회에서 결측값을 과도하게 고민하지 말라.
  원본에 결측이 없었다면 (이 대회가 그 경우) 합성 후 무작위로 제거됐을 가능성이 높고, 그렇다면 결측 컬럼에 의미 있는 신호는 없다.
  클래스 균형 작업은 9:1 비율 전까지는 필요 없다.
- Ravi Ramakrishnan (73위): 여기에 클래스 불균형 문제는 없다고 본다.
- 작성자도 "경미한 수준이라 걱정할 정도는 아니다"라고 수긍했다.

### 우리 모델링에 주는 시사점

- 클래스 불균형 대응(SMOTE, 리샘플링, 재가중)에 시간을 쓰지 않는다.
  71/29 + AUC 조합에서는 상위권 누구도 이걸 신경 쓰지 않는다.
- 결측은 신호가 아니라 무작위 제거 노이즈라는 것이 통계 검정으로 확인됐으므로, NaN 그대로 두는 전략을 기본으로 한다.
- Stratified K-Fold는 당연히 기본 채택.

## 8. As a Beginner, What's the First Thing You Check in a Tabular Competition? (733495)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495
작성자: Mayank Garg (925위), 코멘트 9개.

### 핵심 주장

- 초보자의 질문 스레드: EDA vs 베이스라인 우선순위, FE/CV/튜닝 중 어디서 점수가 나오는가, 피해야 할 초보 실수는 무엇인가.

### 코멘트의 반박과 보강

- Georgy Mamarin (195위)의 답이 이 스레드의 백미다.
  같은 베이스라인 위에서 각 변화의 가치를 5-fold OOF로 전부 재측정했다.
  모델 용량 증가 (63 leaves/400 rounds -> 255/1500 + 낮은 학습률): +0.0012 (정정 후 수치).
  각 컬럼을 크기가 아니라 정확한 값으로 타겟 인코딩: +0.0032 (용량 변화의 약 2.5배).
  공개된 최강 단일 피처인 daily - (social + gaming + work) 잔차: +0.00058.
  tamerlanomralinov의 slack + 관측 성분 개수 피처: +0.00071.
  본인이 직접 발명한 피처: +0.0002 (목록에서 최소).
  공개 OOF 라이브러리 상위 10개 모델 평균: +0.0005, 가중치 최적화: +0.0001.
- 정확한 값 타겟 인코딩이 통하는 이유는 스마트폰과 무관하다: 데이터가 그리드 위에 생성/반올림되어 있어 정확한 값 인코딩이 그 그리드를 집어내는 것이다 (OMID BAGHCHEH SARAEI 아이디어, tomasa2가 +0.0023으로 독립 측정).
- 단일 피처 AUC 순위표를 믿지 말라는 경고: work_study_hours가 단독 0.65로 sleep_hours(0.53)보다 높아 보였지만, 강한 컬럼을 고정하고 슬라이스 안에서 재보면 work_study_hours는 12개 셀에서 부호조차 유지하지 못했고 sleep_hours는 전부 같은 부호였다.
  "모델 안에서 어떤 컬럼도 혼자가 아니다."
- Public LB는 테스트의 20%로 채점되므로 이 정도 미세 차이는 공개 보드에서 분간되지 않으며, 모든 수치는 OOF 기준이라고 명시했다.
- 본인 첫 코멘트의 수치 3개를 스스로 정정하는 후속 코멘트를 남겼다 (ablation 사다리 중간 단계에 피처가 끼어 있어 +0.0019/+0.0027이 +0.0012/+0.0032로 바뀜).
  측정 사다리의 각 단계가 같은 조건인지 확인하라는 교훈.
- butlerc9: 이 대회 스레드에는 AI 생성 저품질 답변 스팸이 많으니, 지난 플레이그라운드 대회 상위 솔루션과 Chris Deotte의 글을 공부하라.
- Shiv Satyam (125위): 가장 단순한 모델부터 만들어 CV-LB 관계를 먼저 파악한다.
- Rugved Bane: FE가 가장 큰 수익을 주며, Public LB보다 CV를 믿어라.

### 우리 모델링에 주는 시사점

- 우선순위가 수치로 정해졌다: 1) 정확한 값 타겟 인코딩 (+0.0032), 2) 모델 용량 확장 (+0.0012), 3) other_screen/slack 계열 잔차 피처 (+0.0006~7), 4) 앙상블 평균 (+0.0005), 나머지는 미미.
- 타겟 인코딩은 반드시 폴드 내부에서 적합해야 하며 전체 train에 적합하면 검증 점수가 가짜로 뛴다.
- 플라시보 피처(무작위 값 + 동일 결측 패턴) 하나를 상시로 넣어 +0.0003 수준 이득이 진짜인지 폴드 노이즈인지 판별하는 관행을 도입한다 (733730 스레드의 Georgy 코멘트에서 온 기법).
- 단일 피처 AUC 순위는 참고만 하고 조건부 슬라이스 검증 없이는 믿지 않는다.

## 9. Nomophobia: No Mobile Phone Phobia on Kaggle Playground : ) (731755)

URL: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731755
작성자: Marília Prata, 코멘트 4개.

### 핵심 주장

- 모델링 스레드가 아니라 도메인 배경 스레드다.
  노모포비아(휴대폰 연결 상실 공포)에 관한 학술 논문 (Bhattacharya 외, PMC6510111) 인용 모음.
- 노모포비아는 DSM-IV의 특정 공포증 정의에 기반하며, 다른 정신 질환(사회 불안, 공황 장애)과 증상이 겹쳐 배제 진단이 필요하다는 내용.

### 코멘트의 반박과 보강

- 전부 개인 경험담이다 (2010년대 초 플립폰으로 회귀한 사람, 휴대폰을 꺼서 뽁뽁이에 싸 두는 작성자 본인 등).
- 기술적 논의는 없다.

### 우리 모델링에 주는 시사점

- 직접적인 모델링 가치는 없다.
- 간접 교훈 하나: 이 대회 데이터의 "중독" 라벨은 실제 임상 진단이 아니라 합성 생성물이며, 배치 B의 다른 스레드들이 보여주듯 도메인 심리학 기반 피처보다 생성기 아티팩트 기반 피처가 점수를 낸다.
  도메인 지식에 기반한 FE 투자는 이 대회에서는 우선순위가 낮다.

## 배치 B 종합 결론

1. 이 대회의 점수 사다리는 수치로 검증된 순서가 있다: 정확한 값 타겟 인코딩 (+0.0032) > 모델 용량/튜닝 (+0.0012) > other_screen 잔차 계열 (+0.0006~7) > 앙상블 평균 (+0.0005) > 손수 만든 행동 피처 (~+0.0002 이하 또는 음수).
2. 결측 관련 피처(is_missing, missing_count)는 로컬 CV를 올리고 LB를 떨어뜨리는 검증된 함정이다.
   결측률이 train/test에서 다르므로 (전 컬럼, 13~40 SE) 이 피처들은 split 소속 정보만 담는다.
   결측은 NaN 그대로 트리에 넘긴다.
3. 선형 조합/비율/차이 피처는 트리가 이미 도달 가능한 경계의 재표현이라 실패한다.
   성공하는 피처는 생성기의 구조적 결함(산술 제약 위반 잔차, 반올림 그리드)을 인코딩한 것뿐이다.
4. 클래스 불균형(71/29)은 AUC 지표 하에서 무시해도 된다.
5. 원본 실데이터는 훈련에 섞지 않는다 (50배 가중치 실험에서 전 폴드 하락, 상위권의 "검증 용도만" 조언).
6. OOF/테스트 예측 저장을 상시화해야 상위권 진입 수단인 스태킹(메타 피처 + 다항 상호작용)이 가능해진다.
7. Public LB는 테스트의 20%라 미세 차이를 분간하지 못하므로 OOF를 기준으로 의사결정한다.
