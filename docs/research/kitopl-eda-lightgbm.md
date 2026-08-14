# kitopl EDA LightGBM 노트북 신규 실험 단서 조사

## 결론

2026-08-14 JST 기준 [kitopl EDA LightGBM 노트북의 저장 실행본](https://www.kaggle.com/code/kitopl/eda-lightgbm?scriptVersionId=342289114)을 현재 실험 지도, 연구 문서, 설정과 파이프라인에 대조했다.
대부분의 내용은 정확값 신호, 화면 시간 산술 잔차, 결측 표시 배제, 원시 범주 열 제거, `max_bin` 확대와 CV·Public 관계에 관한 기존 결정으로 이미 다뤘다.
새로 실험할 가치가 있는 단서는 `max_bin`을 뺀 LightGBM 저용량·규제·표본추출 설정 묶음 하나다.
노트북의 원시 12열 LightGBM에서 `max_bin=2047`만 적용한 D1은 OOF AUC 0.96559였고, 여기에 나머지 설정을 함께 적용한 D2는 0.96667로 약 `+0.00108` 더 높았다.
현재 저장소의 LightGBM은 `num_leaves=255`, `learning_rate=0.05`와 기본 잎·표본추출·L2 설정만 사용하므로 이 묶음은 아직 검증하지 않은 모델 설정 축이다.
따라서 [LightGBM 열별 구간 해상도의 성능·다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/109)이 확정한 구간 설정을 고정한 뒤, 이 묶음의 이전 가능성을 묻는 새 실험 티켓을 여는 것을 권장한다.
다만 D2는 같은 OOF를 본 Optuna 탐색의 선택 결과이고 단일 모델 시드만 보고하므로, 공개 `+0.00108`은 후보 우선순위 근거일 뿐 채택 근거가 아니다.

## 조사한 판본과 근거 범위

Kaggle 화면에서 고정해 읽은 판본은 `scriptVersionId=342289114`이며 화면에는 `Version 5 of 6`으로 표시됐다.
이 판본의 화면에는 코드 셀의 표준 출력과 표가 저장돼 있어 fold별 점수, 최적 반복 수, 설정 비교와 제거 실험 수치를 직접 확인할 수 있었다.
같은 화면의 댓글 탭은 `Comments (0)`이므로 판단을 보강하거나 반박하는 작성자 또는 독자 댓글은 없다.
Kaggle CLI로 내려받은 최신 `.ipynb`는 코드 셀 35개의 `outputs` 배열이 모두 비어 있었지만, 버전 5 웹 화면에는 저장 출력이 렌더링됐다.
조사 도중 `kaggle kernels status`는 더 최신 판본을 `RUNNING`으로 보고했으므로, 이 문서는 실행 중인 후속 초안의 결과를 추정하지 않고 버전 5의 코드와 저장 출력만 수치 근거로 삼는다.
내려받은 후속 소스는 D2 테스트 예측과 제출 파일 생성 코드를 추가했지만, D1과 D2의 학습 설정은 버전 5에 저장된 설정과 같았다.

## 노트북에서 직접 확인한 사실

### 검증과 입력

노트북은 대회 훈련 자료 691,369행의 원시 특성 12개만 사용한다.
수치 열 9개는 원시 값으로 두고 범주 열 3개만 Pandas `category`로 바꾼다.
정확값 목표값 인코딩, 화면 시간 잔차, 제약 결측 재구성, 조건부 결측 복원과 조성 특성은 사용하지 않는다.
검증 분할은 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`이며, 이는 저장소의 [공유 fold 생성 코드](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/scripts/make_folds.py)와 같은 분할 사양이다.
모든 모델 비교는 같은 다섯 분할의 OOF AUC를 사용한다.
모델의 `seed`는 42 하나이며, 저장소 판정 계약의 확정 시드 `[42, 43, 44]` 반복은 없다.
튜닝 설정은 작성자가 이 노트북을 쓴 뒤 Optuna로 찾았다고 설명하지만, 탐색 범위, 시행 횟수, nested 선택 또는 독립 확인 자료는 공개 코드에 없다.
따라서 D2의 OOF는 탐색에 사용한 분할에서 다시 보고한 선택 후 점수로 봐야 한다.

### 단계별 LightGBM 수치

노트북은 학습 없는 `daily_screen_time_hours` 단일 열의 AUC를 0.86537로 저장했다.
fold 안에서 중앙값 대체, 표준화와 one-hot을 학습한 로지스틱 회귀의 OOF AUC는 0.91148이었다.
원시 12열 LightGBM C의 OOF AUC는 0.96349였다.
C의 fold별 AUC는 0.96254, 0.96333, 0.96386, 0.96433, 0.96338이었다.
C의 fold별 최적 반복 수 가운데 최댓값은 1,850이었다.
C의 설정은 `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`과 최대 2,000회, 100회 조기 종료였다.
C에는 `subsample_freq`가 없으므로 `subsample=0.8`은 실제 행 표본추출을 켜지 않는다.
LightGBM 공식 문서는 행 표본추출이 `bagging_freq > 0`이면서 `bagging_fraction < 1.0`일 때만 유효하고 `subsample_freq`가 `bagging_freq`의 별칭이라고 명시한다 ([LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)).
D1은 C에서 `max_bin`만 2,047로 바꿨고 OOF AUC 0.96559, C 대비 `+0.00210`, 최대 최적 반복 수 2,236을 저장했다.
D2는 OOF AUC 0.96667, C 대비 `+0.00318`, D1 대비 약 `+0.00108`, 최대 최적 반복 수 5,222를 저장했다.
노트북 첫 표는 D2를 반올림해 0.96668로 표기하므로 0.96667과 0.96668의 차이는 표시 자릿수 차이다.
D2의 전체 설정은 `max_bin=2047`, `num_leaves=45`, `min_child_samples=229`, `learning_rate=0.03172214617330919`, `colsample_bytree=0.4654152088234655`, `subsample=0.9225106784455195`, `subsample_freq=1`, `reg_lambda=0.9179636700368584`와 최대 12,000회, 100회 조기 종료다.
공식 문서에 따르면 이 묶음은 잎 수와 잎의 최소 표본 수, 열 표본추출, 행 표본추출과 L2 규제를 동시에 바꾼다 ([LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)).
따라서 D1과 D2의 차이는 한 매개변수의 제거 실험이 아니라 여섯 축을 함께 바꾼 묶음의 결과다.

### 제거 실험과 값별 지문

원시 12열 C에서 `notifications_per_day`를 빼면 OOF가 0.95461로 `-0.00888` 낮아졌다.
원시 12열 C에서 `app_opens_per_day`를 빼면 OOF가 0.95614로 `-0.00735` 낮아졌다.
두 열을 함께 빼면 OOF가 0.94451로 `-0.01898` 낮아졌고, 두 열만 쓰면 OOF가 0.82087이었다.
대조로 `daily_screen_time_hours`를 빼면 OOF가 0.95342로 `-0.01007` 낮아졌다.
`sleep_hours`와 `age`를 각각 빼면 변화는 `-0.00033`과 `-0.00027`이었다.
원시 범주 열 3개를 함께 빼면 OOF는 0.96353으로 `+0.00004`였고, 결측 표시 12개를 더하면 변화는 반올림 전 약 `-0.000003`이었다.
작성자는 `notifications_per_day`와 `app_opens_per_day`의 선형 상관이 작아도 값별 목표값 비율이 비단조로 크게 흔들리기 때문에 나무가 두 열을 활용한다고 해석한다.
매끄러운 성분을 뺀 값별 목표값 비율의 무작위 반분 상관은 `notifications_per_day` 0.993, `app_opens_per_day` 0.994, `daily_screen_time_hours` 0.931, `sleep_hours` 0.890이었다.
`notifications_per_day`와 `app_opens_per_day`는 훈련과 테스트에서 각각 231개와 166개의 값을 모두 공유했고, 훈련에 없던 값을 가진 테스트 행은 0개였다.
`daily_screen_time_hours`는 훈련 값 1,389개와 테스트 값 1,349개 가운데 1,341개를 공유했고, 훈련에 없던 값을 가진 테스트 행은 9개였다.
작성자도 이 반분 상관은 다른 특성의 구성 효과를 제거하지 않은 통제되지 않은 수치라고 명시하고, 더 엄격한 잔차 반분 검사는 [Lookup-Transformer 노트북](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)의 결과를 인용한다.

### 결측과 Public 점수 주장

노트북의 각 열 결측 여부와 목표값 비율을 비교한 z 검정에서 가장 작은 p 값은 `app_opens_per_day`의 0.0252였다.
작성자는 12회 Bonferroni 문턱 0.0042를 넘지 않는다는 점과 결측 표시 제거 실험을 근거로 전체 결측을 MCAR라고 부른다.
이 검사는 결측 여부가 목표값과 연관되는지만 검사하므로, 결측 표시들끼리 또는 결측과 다른 관측 열 사이의 의존성을 배제하지 않는다.
노트북 본문은 C의 Public 점수를 0.96511이라고 적고, Kaggle 화면 상단은 별도 버전의 최고 점수를 0.96515로 표시한다.
어느 값도 D2 설정의 Public 점수라고 연결할 저장 근거는 없다.

## 현재 저장소와의 대조

### 이미 반영된 내용

`notifications_per_day`와 `app_opens_per_day`가 비단조 정확값 신호를 가진다는 관찰은 새롭지 않다.
[정확값 신호 표현 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/36)은 두 열을 포함한 정확값 목표값 인코딩을 3시드 OOF로 채택했고, [Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58)은 fold 학습 부분 전용 어휘로 같은 값 키 가설을 더 직접 구현했다.
따라서 두 열의 큰 제거 손실은 기존 정확값 표현의 우선순위를 재확인하지만 새 특성 티켓을 열지는 않는다.

`daily_screen_time_hours - social_media_hours - gaming_hours - work_study_hours` 산술 잔차도 이미 반영됐다.
[산술 잔차 표현의 최적 구성 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/46)은 엄격한 `other_screen`과 부분 관측 행을 덮는 `screen_slack` 병행을 3시드 `+0.00081`로 채택했다.
노트북이 다음 단계로 제안한 `weekend / daily`와 일반 비율은 이미 대조 결과가 있다.
[복원 행렬 기반 비율·차이 피처의 한계 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/90)은 조성 12열을 실제로 검사해 `social_frac`, `work_frac`, `leisure_frac`, `resid_frac`, `week_total`만 채택하고 `wk_ratio` 등 7열을 플라시보 미달로 기각했다.

원시 범주 열 3개 제거도 이미 같은 방향으로 더 엄격하게 확인했다.
[플라시보 미달 피처 제거의 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/79)은 세 열 제거가 3시드에서 `+0.00001`에 그쳐 채택 문턱 미달이라고 판정했다.
노트북의 단일 시드 `+0.00004`는 이 결론을 바꾸지 않는다.

결측 표시가 목표값 예측에 도움이 되지 않는다는 제거 실험도 기존 문서와 일치한다.
[Kaggle 디스커션 종합](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/discussion-insights.md)은 여러 공개 제거 실험을 대조해 결측 표시를 기각했고, [남은 실험 공간 전수 재점검](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/remaining-experiment-space-audit.md)은 원시 NaN 유지와 결측 표시 배제 경계를 재확인했다.
[실험 지도](https://github.com/tmheo/predicting-smartphone-addiction/issues/44)도 결측 표시와 결측 개수의 기본 특성 추가를 범위 밖으로 확정했다.

`max_bin` 확대는 [LightGBM 구간 수 확대의 성능·다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/100)이 현재 피처 계획에서 이미 다시 실행했다.
우리 결과는 전역 1,023이 255보다 3시드 약 `+0.000052`였지만 전역 1,439는 255보다 약 `-0.000008`이어서, 원시 12열에서 2,047이 크게 이긴 노트북 결과가 강화된 피처 계획에 그대로 이전되지 않음을 보여 준다.
노트북의 `max_bin=2047`은 별도 티켓을 다시 열 근거가 아니며, 두 고유값 많은 원시 열에만 해상도를 주는 후속 질문은 이미 [LightGBM 열별 구간 해상도의 성능·다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/109)이 맡고 있다.

조기 종료 상한을 충분히 넓혀야 한다는 운영 교훈도 현재 파이프라인에 반영돼 있다.
현재 [대리 기준 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp063_lgb_max_bin_1023.yaml)은 최대 10,000회와 200회 조기 종료를 쓰므로 노트북 D2의 최대 최적 반복 수 5,222를 수용한다.
[LightGBM 어댑터](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/model.py)는 설정의 임의 모델 매개변수를 `LGBMClassifier`에 전달하므로 새 설정 묶음 자체에는 파이프라인 구현 변경이 필요 없다.

### 그대로 받아들일 수 없는 내용

노트북의 `MCAR` 결론은 현재 저장소 근거와 충돌한다.
[숨은 제약과 결측 주입 구조 측정](https://github.com/tmheo/predicting-smartphone-addiction/issues/88)은 66쌍 결측 표시 상관이 모두 양수이고 행별 결측 수 분산비가 독립 모형 대비 1.78임을 훈련과 테스트에서 재현해 열별 독립 결측 주입을 반증했다.
현재 근거가 허용하는 결론은 결측 구조가 목표값과 사실상 독립이라 기본 결측 표시가 쓸모없다는 것이며, 전체 결측 과정이 MCAR라는 것은 아니다.

훈련과 테스트의 값 범위가 비슷하므로 분포 이동이 없다는 주장도 범위 비교만으로는 성립하지 않는다.
기존 적대적 검증은 결측률 차이로 두 자료를 어느 정도 구분할 수 있으나 결측 표시를 통제한 값 분포 차이는 거의 없다고 분해했다 ([Kaggle 디스커션 종합](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/discussion-insights.md)).
노트북의 범위 표는 값 분포 이동이 없다는 새 증거가 아니라 기존 진단의 약한 부분 확인이다.

Public 점수와 CV 차이가 작다는 이유로 개별 실험을 채택해서는 안 된다.
[실험 채택 판정 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)은 Public 점수를 모든 계열에서 판정 근거로 금지하고 마일스톤 건전성 점검에만 쓴다.
[특성 마일스톤의 CV·Public 관계 확인](https://github.com/tmheo/predicting-smartphone-addiction/issues/57)은 네 자체 실행에서 방향과 오프셋을 이미 더 강하게 확인했다.
노트북 화면과 본문의 0.96515와 0.96511 표기 차이도 Public 수치를 새 실험 단서로 해석하지 말아야 할 이유다.

## 새 실험으로 남는 단서

현재 저장소에는 LightGBM의 구간 수만 바꾼 실험과 CatBoost의 깊이·학습률 비교는 있지만, LightGBM의 잎 수, 잎 최소 표본 수, 열 표본추출, 실제 행 표본추출과 L2 규제를 함께 제한한 실험은 없다.
노트북 D2는 D1과 같은 `max_bin=2047`에서 나머지 묶음만으로 약 `+0.00108`을 더했으므로, `max_bin`과 구분되는 후보 신호를 제공한다.
현재 대리 기준과 노트북의 핵심 차이는 `num_leaves` 255 대 45, `min_child_samples` 기본 20 대 229, 열 표본추출 1.0 대 약 0.465, 행 표본추출 비활성 대 약 0.923을 매 반복 활성화, L2 0 대 약 0.918이다.
이 변화들은 강화된 현재 피처 계획에서 255개 잎이 과도한 자유도를 갖는지와 표본추출이 다른 잔차를 만드는지를 함께 묻는다.
공개 수치는 원시 12열과 단일 시드에서 선택된 결과라 절대 개선 폭을 이전할 수 없지만, 정확한 설정과 저장된 동일 fold 결과가 있어 제한된 한 번의 이전 검사는 정당화된다.

## 권장 새 티켓

권장 이름은 `P3 보강: LightGBM 저용량·규제·표본추출 묶음의 성능·다양성 기여 결정`이다.
질문은 현재 최선 LightGBM 피처 계획과 이슈 109가 고른 구간 설정을 고정한 채, kitopl D2의 `max_bin` 외 설정 묶음이 단독 OOF와 후보 풀 기여를 개선하는지로 한정한다.
이 티켓은 이슈 109에 의해 막히게 연결해 구간 해상도와 나머지 설정을 동시에 바꾸지 않게 한다.

첫 후보는 `num_leaves=45`, `min_child_samples=229`, `learning_rate=0.03172214617330919`, `colsample_bytree=0.4654152088234655`, `subsample=0.9225106784455195`, `subsample_freq=1`, `reg_lambda=0.9179636700368584`로 고정한다.
`max_bin` 또는 `max_bin_by_feature`는 이슈 109의 우승 설정을 그대로 유지한다.
`n_estimators=10000`, `early_stopping_rounds=200`과 `force_row_wise=true`는 현재 대리 기준을 유지한다.
첫 실행은 현재 LightGBM 기준과 같은 커밋, 같은 피처, 같은 seed 42 OOF로 짝비교한다.
개선이 0 이상일 때만 ADR 0001에 따라 `[42, 43, 44]` 확정 재검증으로 보낸다.
확정 재검증이 단일 모델 문턱에 미달해도 3시드 평균본이 후보 풀 진입 하한을 넘으면 중복 게이트와 표준 평가 앙상블 기여를 측정한다.
전체 묶음이 통과할 때만 `subsample_freq=0` 대조 하나를 추가해 실제 행 표본추출이 이득 또는 다양성의 핵심인지 분리한다.
전체 묶음이 seed 42에서 미달하면 개별 매개변수 탐색이나 새 Optuna 탐색으로 확장하지 않고 닫는다.
이 제한은 같은 OOF를 보고 고른 외부 설정에서 다시 넓은 탐색을 시작해 선택 자유도를 키우지 않기 위한 것이다.

## 권장 이슈 121 결의

버전 5의 저장 출력에서 `max_bin=2047` 단독 0.96559와 전체 튜닝 0.96667의 차이 약 `+0.00108`을 확인했다.
이 차이를 만든 비구간 설정 묶음은 현재 지도에서 아직 시험하지 않았으므로 새 LightGBM 설정 이전 티켓을 연다.
정확값 지문, 산술 잔차, 일반 비율, 원시 범주 열 제거, 결측 표시와 `max_bin` 자체는 각각 기존 채택, 기각 또는 열린 이슈와 중복되므로 새 티켓을 열지 않는다.
MCAR, 값 범위만으로 분포 이동 없음, Public 점수로 CV를 신뢰한다는 일반화는 현재 검증 계약이나 더 강한 저장소 진단과 맞지 않으므로 채택하지 않는다.
새 티켓을 만든 뒤 이 조사 문서를 연결하고 이슈 121을 닫는 것을 권장한다.

## 한계

이 조사는 공개 코드와 저장 출력의 근거 강도를 판정했으며 모델을 새로 학습하지 않았다.
D2의 Optuna 탐색 기록과 독립 확인 결과가 없으므로 각 매개변수의 개별 기여와 선택 편향 크기는 알 수 없다.
조사 시점에 더 최신 Kaggle 판본이 실행 중이었으므로 후속 판본이 완주하면 코드, 출력과 결론이 달라졌는지 다시 확인해야 한다.
