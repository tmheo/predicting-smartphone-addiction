# 스칼라 token Transformer 공개판과 자체 이식판의 성능 격차 분석

## 결론

가장 강한 결론은 `exp115_scalar_token_transformer_m0`가 공개 노트북을 같은 입력으로 이식한 실행이 아니라는 점이다.
공개판은 원시 수치와 파생값 35개에 원래 12개 열의 빈도와 목표값 평균 24개를 더한 59개 수치 스칼라를 사용한다.
`exp115`는 공개판의 파생값, 빈도와 목표값 평균을 모두 제거하고 현재 champion용 33개 피처 계획으로 교체했다.
그 결과 마지막 MLP에 들어가는 폭과 전체 학습 가능 매개변수도 `1,182,801`개에서 `747,249`개로 `435,552`개 줄었다.

같은 seed 42와 같은 fold 0에서 공개 저장 OOF는 `0.9670351852`, `exp115`는 `0.9551717921`이다.
직접 비교 격차는 `0.0118633931`이며, 사용자가 출발점으로 든 약 `0.006`보다 크다.
저장소의 `0.9620748339`는 스칼라 token Transformer가 아니라 표 합성곱망의 3시드 OOF다.
`exp115`는 fold 0 진입 진단만 수행했으며 5-fold OOF와 Kaggle 제출을 만들지 않았다.
따라서 공개 점수와 `exp115`의 5-fold OOF를 비교한 것처럼 읽을 수 있는 직접 측정값은 현재 없다.

신경망 골격을 잘못 옮긴 증거는 찾지 못했다.
공개 클래스와 자체 `_ScalarTokenModel`을 같은 PyTorch에서 같은 seed로 만들었을 때 33열과 59열 모두 매개변수 모양, 초기값, 평가 모드 출력과 같은 난수 상태의 학습 모드 출력이 정확히 일치했다.
손실, mixup, EMA, AdamW, 학습률 일정, epoch 상한, 조기 종료와 AMP도 동작상 같다.

가장 가능성 높은 원인은 공개 구조가 의존한 입력 표현 전체를 바꾸면서 생긴 모형과 입력의 부조화다.
특히 정확값 목표 평균은 알림 수와 앱 실행 수처럼 원시값의 단조 순서만으로 읽기 어려운 생성 규칙 신호를 바로 제공한다.
반면 `exp115`에는 정확값 lookup도 목표값 평균도 없고, 연속 ReLU·주기 기저만 이 신호를 배워야 한다.
공개판과 같은 59열 계약을 보존해야 한다고 했던 선행 조사 권고와 달리 실제 구현 티켓은 목표값·빈도 인코딩을 빼고 champion 피처 계획을 쓰도록 범위를 바꿨다.

## 조사 대상과 판본 고정

조사일은 2026-08-20 JST다.
Kaggle 공식 API의 현재 `current_version_number`는 `1`이다.
현재 페이지의 notebook id는 `130940978`이고 oEmbed 주소가 가리키는 실행 식별자는 `scriptVersionId=342815072`다.
현재 판본의 마지막 실행 시각은 Kaggle 목록 기준 `2026-08-16 17:05:34 UTC`다.

| 증거 | 2026-08-20 재확인 값 |
| --- | --- |
| [현재 공개 페이지](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction) | version `1` |
| [고정 판본 주소](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/versions/1) | `scriptVersionId=342815072` |
| 노트북 SHA-256 | `eeb3e1cccbaab29c71ef946876f7042509f6ef537df4a9b04ced36e3c424e46c` |
| Kaggle 메타데이터 SHA-256 | `3b9ab90b326ac4cf9cc6ce45f6c76ef04980fb2c44eb7d9f1e1e604bcb32470c` |
| 실행 로그 SHA-256 | `21e24aa8ad869aaeb87d2d72f40c3e59be3b37fd3f666fd3dab58b1a222bd657` |
| OOF SHA-256 | `1594f8e7f72ee8c6bf5dacbdddc56fb29d8998c24c43ad9424b39abd65e80cb4` |
| 제출 예측 SHA-256 | `6228dfc18fe458c6f061f684daff8daef7e2a4aed39acf245be5ec0a190877a9` |
| 학습 자료 SHA-256 | `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c` |
| 시험 자료 SHA-256 | `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e` |
| 공통 fold SHA-256 | `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4` |

현재 내려받은 소스, 메타데이터, 로그와 출력의 해시는 [선행 판본 조사](omid-tabtransformer-notebook.md)가 2026-08-17에 version 1에서 고정한 값과 모두 같다.
따라서 조사 사이에 공개 판본이 바뀌지 않았고 판본 차이는 이번 격차의 원인이 아니다.
Kaggle CLI에서 주소 끝에 `/1`을 붙인 직접 pull은 API가 403을 반환했지만, 현재 API가 version 1이라고 응답하고 현재 소스와 모든 기존 version 1 해시가 일치하므로 판본 동일성에는 모순이 없다.

공개 노트북 소스에는 Apache License 2.0이 적용된다.
자체 구현은 원문 주소, 판본, 원문 SHA-256, 변경 사실과 사용 조건 원문을 [`scalar_token_transformer.py`](../../src/pipeline/scalar_token_transformer.py)와 [`scalar_token_transformer.LICENSE`](../../src/pipeline/scalar_token_transformer.LICENSE)에 보존했다.
대회 자료와 NumPy, pandas, scikit-learn, PyTorch의 사용 조건은 노트북 소스 사용 조건과 별개다.

## 비교 척도부터 바로잡기

현재 공개 OOF를 공식 학습 자료와 커밋된 fold에 다시 결합해 채점한 값은 다음과 같다.

| 비교값 | AUC | 비교 가능성 |
| --- | ---: | --- |
| 공개판 전체 5-fold OOF | `0.9674689212` | 공개판 자체의 전체 OOF |
| 공개판 fold 0 | `0.9670351852` | `exp115`와 직접 비교 가능 |
| `exp115` M0 fold 0 | `0.9551717921` | seed 42, 같은 fold 0 |
| `exp116` A0 fold 0 | `0.9564251321` | M0의 attention 제거 대조 |
| 표 합성곱망 3시드 전체 OOF | `0.9620748339` | 다른 모델이며 scalar 수치가 아님 |

공개판 fold 0과 `exp115`의 직접 격차는 `0.0118633931`이다.
공개 전체 OOF와 `exp115` 단일 fold를 빼면 `0.0122971291`이지만 서로 다른 집계이므로 원인 귀속값으로 쓰지 않는다.
공개판 다섯 fold는 `0.9670351852`에서 `0.9682952311` 사이여서 `exp115`는 공개판의 가장 낮은 fold보다도 `0.01186` 낮다.

현재 Kaggle 페이지와 출력에는 이 노트북 제출의 Public 점수가 기계 판독 가능한 값으로 들어 있지 않다.
공개 `submission.csv`는 내려받을 수 있지만 숨은 시험 라벨이 없으므로 로컬에서 Public 점수를 재채점할 수 없다.
Public 점수가 OOF보다 약 `+0.001` 높다는 일반 관찰을 이 모델의 상수 보정값으로 쓰면 안 된다.
공개판은 5개 fold 확률 평균으로 시험 예측을 만들고 `exp115`는 fold 0에서 중단됐으므로 제출 평균화도 정렬되지 않았다.

## 끝에서 끝까지의 동작 차이

| 축 | 공개 version 1 | `exp115` | 판독 |
| --- | --- | --- | --- |
| 원시 입력 | 수치 9열과 범주 3열 | 같은 원시 12열 | 자료와 fold는 같다 |
| 행 단위 피처 | 결측 개수 1열과 합·비율·잔차·로그·제곱근 25열 | 화면 관계 6열 | 공개 26열을 보존하지 않았다 |
| 결측 복원 | 파생값과 최종 행렬에서 0 대체 | 학습 fold 전용 제약 복원 4열, XGBoost 복원 5열 | 자체 쪽이 더 엄격하지만 다른 표현이다 |
| 복원 조성 | 없음 | XGBoost 복원값 기반 비율·주간 합 5열 | 공개판에 없는 자체 피처다 |
| 범주 표현 | 원래 12열 각각의 문자열 빈도와 목표값 평균 | 범주 3열을 학습 fold 어휘의 순서 ID로 바꿈 | 목표값 표면이 사라졌다 |
| 정확값 표현 | 수치값도 문자열 exact key로 빈도·목표값 평균을 만듦 | exact lookup도 목표값 평균도 없음 | 가장 중요한 표현 차이다 |
| 플라시보 | 없음 | `placebo_noise` 1열 | 중요도는 31위로 낮았다 |
| 총 입력 수 | 35개 수치 기반 + 24개 인코딩 = 59 | 원시 12 + 파생 6 + 플라시보 1 + 복원 14 = 33 | 26개 token 감소 |
| 변환 적합 경계 | 빈도·목표값 평균과 분위 변환을 outer 학습 부분에 맞춤 | 복원기, 범주 어휘와 분위 변환을 outer 학습 부분에 맞춤 | 검증 라벨은 양쪽 모두 배제 |
| 목표값 평균의 학습 행 | outer 학습 행 자신의 라벨을 포함 | 목표값 평균 자체가 없음 | 공개판은 내부 OOF가 아님 |
| 분위 변환 | 정규분포 출력, 1,000 분위, seed 42 | 같은 출력·분위 수, 최대 10,000행 표본, fold seed | fold 0의 실질 설정은 같다 |
| ReLU 기저 | 열별 `x*w+b`, 폭 16, ReLU | 같다 | 동작 일치 |
| 주기 기저 | 열별 학습 주파수, 폭 16, sigma 2.33 | 같다 | 동작 일치 |
| token 투영 | 32에서 64로 공유 선형 투영 후 열 식별자 추가 | 같다 | 동작 일치 |
| attention | 폭 64, 머리 4, FFN 256, post-norm 3층 | 같다 | 동작 일치 |
| readout | 59개 token 전부와 원래 59개 스칼라를 펼쳐 결합 | 33개 token 전부와 원래 33개 스칼라를 펼쳐 결합 | 골격은 같지만 입력 폭이 다르다 |
| MLP | `3835 -> 256 -> 128 -> 64 -> 1` | `2145 -> 256 -> 128 -> 64 -> 1` | 첫 선형층에서 큰 용량 차이가 난다 |
| dropout head | 같은 선형층을 공유하는 8개 dropout 표본 평균 | 같다 | 동작 일치 |
| 손실 | 이진 logit 손실, 라벨 평활 0.005 | 같다 | 동작 일치 |
| mixup | batch마다 Beta(0.2, 0.2) 한 값을 뽑아 입력과 손실을 섞음 | 같다 | 수식 일치 |
| EMA | 매 batch 뒤 decay 0.999로 갱신 | 같다 | 동작 일치 |
| 최적화 | AdamW, lr 0.001, weight decay 0.03, 기울기 norm 1.0 | 같다 | 동작 일치 |
| 학습률 | 20 epoch 주기 cosine restart, 최저 lr 1e-6 | 같다 | 동작 일치 |
| 학습 길이 | 최대 120 epoch, patience 18 | 같다 | 설정 일치 |
| fold 0 최고 epoch | 15 | 38 | 입력과 난수 흐름 변화 뒤 학습 궤적은 크게 달라졌다 |
| seed | 노트북 시작에서 한 번 42, 이후 fold가 난수 상태를 이어받음 | fold마다 seed를 독립 재설정 | fold 0도 batch 난수 흐름은 정확히 같지 않다 |
| 결정론 | NumPy·Torch seed와 CuDNN deterministic만 설정 | Python·NumPy·Torch, CuBLAS, deterministic algorithms까지 설정 | 자체 쪽이 더 엄격하다 |
| 정밀도 | float32 입력, GPU AMP와 GradScaler | 같다 | 장치와 패키지 판본은 다르다 |
| 검증 추론 | 최고 EMA 한 개를 batch 512로 추론 | 같다 | sigmoid 전후 AUC 채점은 순위상 같다 |
| 시험 추론 | 5개 fold 확률의 산술 평균 | fold 0 진입 진단만 수행 | Public 비교를 직접 할 수 없다 |

## 신경망 이식 동등성 진단

공개 노트북에서 클래스 정의를 직접 읽어 같은 환경에서 자체 `_ScalarTokenModel`과 대조했다.
33열과 59열 각각에서 같은 seed로 두 구현을 초기화했다.

| 진단 | 33열 | 59열 |
| --- | ---: | ---: |
| 공개 매개변수 수 | `747,249` | `1,182,801` |
| 자체 매개변수 수 | `747,249` | `1,182,801` |
| 매개변수 tensor 수 | `57` | `57` |
| 최대 초기값 절대 차이 | `0` | `0` |
| 평가 모드 최대 출력 절대 차이 | `0` | `0` |
| 같은 난수 상태 학습 모드 최대 출력 절대 차이 | `0` | `0` |

이 결과는 attention block, 열 식별자, token 펼침, 원시 스칼라 skip, MLP와 다중 dropout head의 이식이 구조적으로 같다는 강한 증거다.
같은 방식으로 공개 mixup 손실과 자체 손실의 수식도 같다.
공개 `BCEWithLogitsLoss`의 두 항에 `lam`과 `1-lam`을 곱하는 방식과 자체 구현의 두 항 합은 동일하다.

[RealMLP 이식 발산 진단](realmlp-port-divergence.md)이 찾은 float64 어휘를 float32 값으로 조회하는 결함은 이 구현에 없다.
스칼라 token Transformer는 범주 어휘를 원래 dtype 값에서 만들고 같은 dtype으로 먼저 조회한 뒤 수치 행렬을 float32로 바꾼다.
연속 열은 exact 어휘 조회를 하지 않으며 분위 변환의 적합과 변환 입력 dtype도 서로 같다.
따라서 RealMLP의 `-0.0045`를 만든 결함을 이번 격차에 재사용해 설명할 수 없다.

## 원인 후보 순위

### 1위: 공개 59입력 계약을 33입력 계약으로 바꾼 모형·표현 부조화

증거 강도는 높고 예상 방향은 공개판 우세다.
이 원인은 공개 파생값, exact 빈도·목표값 평균, token 수와 펼친 readout 용량을 함께 포함하는 묶음 원인이다.
현재 산출물만으로 이 네 하위 원인의 효과를 서로 더해 독립 귀속할 수는 없다.

fold 0의 outer 학습 부분으로 공개 목표값 평균표를 만들고 검증 부분에 적용한 단일 열 AUC는 다음과 같다.
이 계산은 모형을 새로 학습하지 않고 공개 전처리 식만 재현한 진단이다.

| 열 | 원시값 방향 보정 AUC | 공개식 목표값 평균 AUC | 차이 |
| --- | ---: | ---: | ---: |
| age | `0.500724392` | `0.552135221` | `+0.051410830` |
| daily_screen_time_hours | `0.865769095` | `0.877569036` | `+0.011799941` |
| social_media_hours | `0.816071362` | `0.822253432` | `+0.006182070` |
| gaming_hours | `0.610896766` | `0.636758208` | `+0.025861442` |
| work_study_hours | `0.650536095` | `0.669034184` | `+0.018498089` |
| sleep_hours | `0.525410132` | `0.592732690` | `+0.067322558` |
| notifications_per_day | `0.508280386` | `0.746974411` | `+0.238694025` |
| app_opens_per_day | `0.534140954` | `0.735758004` | `+0.201617050` |
| weekend_screen_time | `0.848510194` | `0.863387000` | `+0.014876806` |

알림 수와 앱 실행 수는 원시값의 단조 순서 AUC가 거의 0.5지만 exact 목표값 평균은 각각 약 0.747과 0.736이다.
공개판은 이 두 강한 스칼라를 직접 받고 `exp115`는 받지 않는다.
`exp115`에는 원시 exact lookup도 없으므로 연속 ReLU·주기 기저가 이 비단조 목표 표면을 처음부터 배워야 한다.

기존 MLflow 산출물도 목표값 평균이 실제로 사용되는 신호임을 뒷받침한다.
`exp068_lookup_exact_te`의 중요도에서 `daily_screen_time_hours_te`, `social_media_hours_te`, `weekend_screen_time_te`, `notifications_per_day_te`, `app_opens_per_day_te`가 각각 4, 5, 6, 9, 10위였다.
다만 exact lookup을 이미 가진 Lookup-Transformer에서는 안전한 목표값 평균 블록의 OOF 한계 변화가 약 `+0.00002`였고, 복원 피처와 결합해도 약 `+0.000062`였다.
이 작은 한계값을 exact lookup이 없는 스칼라 기저 모형에 그대로 적용하면 안 된다.

빈도만 추가한 과거 LightGBM 실행은 `-0.00021`, zhenrui 계열 파생 16열의 최근 LightGBM 짝비교는 `-0.0000640`이었다.
따라서 빈도나 일반 파생값 하나가 단독으로 `0.01186`을 설명한다는 증거는 없다.
현재 근거가 지지하는 것은 59입력 표현 전체와 그에 맞춰 커진 readout의 결합 효과다.

### 2위: 33입력에서 attention이 오히려 손해인 현상

증거 강도는 높고 `exp115`에 대한 방향과 크기는 직접 측정됐다.
같은 33개 입력과 같은 학습 설정에서 attention만 열별 잔차 MLP로 바꾼 A0는 `0.9564251321`로 M0보다 `+0.0012533400` 높았다.
즉 공개 모델 이름의 핵심처럼 보이는 attention은 자체 입력 계약에서는 성능을 만들지 않았고 오히려 약 `0.00125`를 잃었다.

A0와 공개 fold 0 사이에도 `0.0106100531`이 남는다.
따라서 attention 구현 오류 하나가 전체 격차의 원인일 수 없다.
이 결과는 공개 입력 표현이 attention과 펼친 readout이 작동하는 조건의 일부였다는 해석과 맞는다.

같은 33개 champion 피처 계획을 쓴 contextualized spline Transformer M0는 같은 fold 0에서 `0.9667574340`을 기록했다.
이는 `exp115`보다 `0.0115856419` 높다.
33개 피처 자체가 약해서 `0.955`에 머문 것이 아니라, 스칼라 ReLU·주기 token 구조와 바꾼 입력 계약의 조합이 맞지 않았다는 간접 증거다.

### 3위: token 수 감소와 펼친 readout 용량 감소

증거 강도는 중간이고 예상 방향은 공개판 우세다.
59개 token을 33개로 줄이면 MLP 첫 입력은 `3,835`에서 `2,145`로 줄고 전체 매개변수는 `435,552`개 감소한다.
감소분은 전체가 token 수에 따라 달라지는 열별 ReLU·주기 기저, 열 식별자와 첫 readout 선형층에서 생긴다.
attention과 ReLU·주기 block의 열당 구조는 그대로다.

입력 수와 피처 내용이 동시에 바뀌었으므로 현재 실행만으로 용량 효과를 따로 잴 수 없다.
공개 파생값과 인코딩이 중복이어도 token을 추가하면 펼친 MLP에 열별 독립 가중치와 비선형 용량이 생긴다.
이 효과는 1위 원인의 일부이며 별도 용량 맞춤 대조가 없어서 수치 귀속은 보류한다.

### 4위: 난수 흐름, GPU와 패키지 판본

증거 강도는 낮고 방향은 불명이다.
공개판은 노트북 시작에서 seed 42를 한 번 설정하고 DataLoader의 기본 난수원, mixup 순열, dropout과 attention dropout이 난수 상태를 공유한다.
자체판은 fold마다 seed를 다시 설정하고 학습 행 순열과 mixup 짝에 전용 생성기를 사용하며 모형 dropout과 분리한다.
따라서 fold 0도 초기 가중치는 같지만 batch 순서와 이후 난수 소비 순서는 같지 않다.

공개판은 Tesla T4와 Kaggle 컨테이너를 사용했고 자체 진단은 RTX A4000, PyTorch `2.13.0+cu130`, scikit-learn `1.9.0`을 사용했다.
두 실행 모두 GPU AMP를 사용하지만 장치별 attention·행렬곱 kernel과 라이브러리 판본이 다르다.
이 차이는 재현의 마지막 자릿수에는 영향을 줄 수 있으나 구조 동등성 진단과 `0.01186`의 큰 체계적 격차를 뒤집을 1차 근거는 없다.

### 5위: 공개 목표값 평균의 자기 라벨 포함과 checkpoint 선택

증거 강도는 중간이고 Public OOF에 대한 방향은 작거나 불명이다.
공개 목표값 평균은 outer 검증 라벨을 사용하지 않으므로 공개 OOF를 직접 무효화하는 누출은 아니다.
다만 outer 학습 행 표현에는 자기 라벨이 들어가 학습과 검증의 생성 규칙이 다르다.

fold 0 학습 부분에서 평활 10을 쓸 때 daily와 weekend exact key의 약 `2.67%` 행은 자기 라벨 계수가 `0.01` 이상이다.
두 열의 희귀 key에서 최대 계수는 `1/11 = 0.0909091`이다.
반면 행 기준 중앙 계수는 daily `0.0011765`, weekend `0.0012804`이고 다른 주요 열은 더 작다.
따라서 자기 포함은 채택할 수 없는 절차지만 `0.01186` 전체를 설명할 정도로 공개 검증 AUC를 직접 부풀렸다는 증거는 없다.

공개 fold 0은 epoch 15의 최고 EMA AUC를 선택했고 epoch 33에서 조기 종료했다.
종료 직전 검증 AUC는 약 `0.96614`로 최고점보다 약 `0.00090` 낮았다.
그러나 자체판도 같은 patience와 같은 최고 EMA 선택을 쓰므로 checkpoint 선택 방식은 양쪽의 구성 차이가 아니다.

## 차이별 귀속 가능 범위

| 후보 | 증거 | 예상 방향 | 현재 귀속 가능한 크기 |
| --- | --- | --- | --- |
| 59입력에서 33입력으로 바꾼 전체 계약 | 소스 직접 diff, exact 목표값 평균 단일 열 진단, 매개변수 수 | 공개 우세 | 전체 `0.01186`의 가장 큰 몫으로 추정하지만 하위 분리는 불가 |
| 33입력 attention | M0와 A0 직접 대조 | 자체 M0 불리 | `-0.00125334` 직접 관찰 |
| 펼친 readout 용량 감소 | 매개변수 산술 | 공개 우세 가능 | 59입력 묶음에 포함, 독립 귀속 불가 |
| 빈도만의 효과 | 과거 LightGBM 짝비교 | 자체 자료에서는 비양수 | `-0.00021`은 다른 모형의 한계값이라 참고만 가능 |
| 안전한 exact 목표값 평균의 효과 | Lookup MLflow 산출물 | 작은 양수 | 약 `+0.00002` 또는 복원 결합 뒤 `+0.000062`, scalar로 전이 불가 |
| 일반 파생값 블록 | 최근 LightGBM 대리 | 비양수 | zhenrui 16열 `-0.0000640`, 공개 26열과 동일하지 않음 |
| 난수·장치·판본 | 설정 diff만 있음 | 불명 | 반복 실행이 없어 정량 불가 |
| 자기 포함 목표값 평균 | 코드와 희귀 key 계수 | 학습 왜곡, 검증 방향 불명 | 독립 대조가 없어 정량 불가 |
| 5-fold 시험 평균과 Public 척도 | 공개 추론 코드, 자체 제출 없음 | Public 우세 가능 | 같은-fold 격차에는 `0`, Public 격차는 별도 척도 티켓 대상 |

이 표의 수치를 서로 더하면 안 된다.
서로 다른 모형에서 얻은 한계값과 59입력 묶음 안의 중첩 원인을 함께 담고 있기 때문이다.

## 다음 통제 실험의 최소 설계

이번 조사에서는 새 GPU 실행이나 유료 자원을 만들지 않았다.
다음 단계가 실행을 열기로 결정하면 같은 GPU 등급, 같은 컨테이너, seed 42, 공통 fold 0에서 아래 순서로 짝지어야 한다.

1. 공개 version 1의 59입력과 학습 절차를 그대로 재현한 `R0`로 출처 재현성을 확인한다.
2. `R0`가 공개 fold 0 `0.9670351852` 근처에 오지 않으면 피처 제거 실험으로 넘어가지 말고 난수 흐름, pandas 문자열 결측 처리, scikit-learn 분위 표본과 PyTorch kernel을 먼저 맞춘다.
3. `R0`에서 학습 행 목표값 평균만 내부 OOF로 바꾼 `R1`을 만들어 자기 포함의 비용을 잰다.
4. `R1`에서 목표값 평균 12열만 제거한 `R2`와 빈도 12열만 제거한 `R3`를 각각 만들어 두 exact 표현을 분리한다.
5. `R1`에서 공개 파생 26열을 제거한 `R4`를 만들어 행 단위 파생값의 효과를 분리한다.
6. 33입력 `exp115`에 첫 readout의 총 매개변수만 `R1`과 맞춘 `R5`를 만들어 피처 내용과 용량을 분리한다.
7. 각 실행에서 attention 제거판은 해당 입력판 하나에만 짝지어 attention의 조건부 효과를 확인한다.

`R0`는 공개 동작 확인용이며 자기 라벨 포함 목표값 평균 때문에 채택 후보가 아니다.
후보 자격 판정은 `R1` 이후의 안전한 실행만 대상으로 해야 한다.
모든 변형은 결과를 보기 전에 열 목록, 순서, 결측 문자열 처리, 내부 OOF 분할과 중단 문턱을 고정해야 한다.

## 최종 판정

이번 성능 차이는 원본 코드보다 자체 구현이 같은 모델을 `0.006` 덜 재현한 사례로 분류하면 안 된다.
실제 같은-fold 격차는 `0.01186`이고, 그 전에 입력 표현 59개를 33개로 바꾼 별도 모델을 실행했다.
신경망 본체의 코드 이식은 현재 진단 범위에서 정확하다.
가장 먼저 확인할 대상은 attention 구현이나 optimizer가 아니라 공개 exact 목표값 평균·빈도·파생값과 59-token readout을 함께 복원한 출처 충실 기준선이다.

원본 충실 재현을 열 가치는 있다.
목적은 공개 점수를 후보 풀에 넣는 것이 아니라, 59입력 계약을 되살렸을 때 fold 0의 약 `0.01186`이 회복되는지 확인해 입력 표현과 실행 환경을 분리하는 것이다.
다만 자기 라벨 포함판은 진단 기준선으로만 쓰고, 실제 후보 여부는 내부 OOF 목표값 평균을 쓴 안전판에서 다시 판단해야 한다.

## 근거

- [TabTransformer 공개 version 1](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/versions/1)
- [공개 출력 페이지](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/output)
- [선행 공개 노트북 조사](omid-tabtransformer-notebook.md)
- [스칼라 token Transformer 진입 진단](scalar-token-transformer-entry-diagnostic.md)
- [`exp115` 설정](../../configs/exp115_scalar_token_transformer_m0.yaml)
- [`scalar_token_transformer` 구현](../../src/pipeline/scalar_token_transformer.py)
- [공개 구조 조사 결의](https://github.com/tmheo/predicting-smartphone-addiction/issues/169#issuecomment-5312786803)
- [자체 진입 진단 결의](https://github.com/tmheo/predicting-smartphone-addiction/issues/178#issuecomment-5329577950)
- [RealMLP 이식 발산 지점 진단](realmlp-port-divergence.md)
- [정확값 추가 표현 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/49#issuecomment-5249149977)
- [zhenrui 파생 수치 16열 대리 결과](https://github.com/tmheo/predicting-smartphone-addiction/issues/258#issuecomment-5344549552)
