# 167-model 앙상블과 단변량 spline Transformer 노트북 검토

이 문서는 [리서치: 167-model 앙상블과 contextualized deep univariate spline 노트북의 신규 실험 단서 확인](https://github.com/tmheo/predicting-smartphone-addiction/issues/148)의 근거다.
조사 시점은 2026-08-16이며, Kaggle API에서 받은 최신 공개 판본의 소스, 실행 기록, 출력 파일과 이 저장소의 현재 판정 근거만 사용했다.

## 결론

167-model 앙상블은 참고할 부분이 있지만 새 진입 진단이나 실험 티켓을 열 근거는 없다.
단변량 spline Transformer는 최초 검토에서 관찰 대상으로만 남겼으나, 아래 재검토에서는 공개 구현의 결함을 고쳐 자체 공통 절차로 재구현한다는 전제 아래 저비용 진입 진단 한 건을 열 가치가 있다고 결론을 고쳤다.
[S6E8 167 Models Diversity Beats Strength 판본 7](https://www.kaggle.com/code/adarsh1077/s6e8-167-models-diversity-beats-strength/versions/7)의 배열 해시 중복 제거와 선형 결합 수렴 확인은 각각 열린 [OOF 후보 풀의 품질과 다양성 진입 기준 점검](https://github.com/tmheo/predicting-smartphone-addiction/issues/63)과 [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)에 흡수할 수 있다.
rank-gauss 표현도 이슈 64에서 비교할 값어치는 있지만, 노트북의 `+0.00008`은 최신 판본에서 실행되는 제거 실험이 아니라 서술로만 남아 있으므로 채택 근거가 아니다.
외부 OOF와 제출 예측은 이 지도의 명시 규칙에 따라 후보 풀과 최종 제출에서 제외되므로 177개 외부 구성원이나 최종 제출 파일을 가져오지 않는다.

[Contextualized Deep Univariate Spline Transformer 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)은 단변량 다중 해상도 spline, 보조 가산 경로와 얕은 attention을 결합한 새로운 구현 사례다.
저장 출력에서 다시 계산한 단일 시드 OOF AUC는 `0.9665204982`이고, 저장소의 고정 seed 42 fold가 아니라 seed 21 fold를 쓰며, 핵심 구성 요소별 제거 실험과 현재 후보 풀 대비 같은-fold 다양성 측정이 없다.
이 한계는 공개 OOF를 채택 근거로 쓰지 못하게 하지만, 문제를 고친 자체 구현의 실험 가치까지 없애지는 않는다.

## 확인한 공개 판본과 산출물

Kaggle API는 첫 노트북의 최신 공개 판본을 7, 두 번째 노트북을 3으로 반환했다.
Kaggle 목록 API의 최신 실행 시각은 각각 2026-08-15 16:43:01 UTC와 2026-08-15 17:47:16 UTC였다.
최초 조사 시점 득표 수는 각각 13개와 4개였고, 재검토 시점에 두 번째 노트북은 7개로 늘었다.

첫 노트북은 CPU, 인터넷 비활성 설정이며 18개 공개 자료와 대회 자료를 입력으로 선언한다.
내려받은 소스의 SHA-256은 `b74cb8168c949a5d4b38bd7421bc0a3f7655529f8b57d523559b47234707e95e`다.
공개 실행 기록의 SHA-256은 `d61ea2869cd60d2f9a528ea34a832f38630242d467a191f0f11215453ac63728`이며, 마지막 기록 시각은 실행 시작 뒤 약 879초다.
내려받은 `.ipynb`의 코드 셀 7개는 모두 `execution_count=null`이고 저장 출력이 비어 있다.
따라서 수치는 노트북 본문만 보지 않고 별도로 공개된 최신 실행 기록과 `submission.csv`까지 대조했다.

두 번째 노트북은 Kaggle API에서 Tesla T4 GPU를 선언하고, 실행 기록도 `Device: cuda`와 `GPU: Tesla T4`를 남긴다.
내려받은 소스의 SHA-256은 `c308b69cfeabad223a1e147fa174f78d1ddaccc09991b2075eecaf757f4781a2`다.
공개 실행 기록의 SHA-256은 `c683bef73188b3cf5f8362b4394518faa2ede9c4f3a3102603a1354f05b681d6`이다.
내려받은 `.ipynb`의 코드 셀 1개도 `execution_count=null`이고 저장 출력이 비어 있다.
대신 [공개 출력](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/output)에는 691,369행 OOF, fold별 시험 예측, fold 지표와 두 제출 파일이 실제로 남아 있다.

두 노트북 소스는 `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`처럼 실행 환경에 이미 있는 패키지만 가져오고 별도 설치나 버전 고정을 하지 않는다.
두 노트북의 내려받기 메타데이터와 소스에는 노트북 코드 자체의 명시적 라이선스가 없다.
첫 노트북은 작성자의 OOF 자료 하나를 CC0라고 본문에 명시하지만, 나머지 17개 입력 자료와 노트북 코드 전체의 사용 조건을 대신하지 않는다.
따라서 구현을 그대로 복사하는 후속 작업은 명시적 사용 조건을 먼저 확인해야 한다.

## 167-model 앙상블의 정확한 방법

이름의 167은 최신 실행의 실제 구성원 수가 아니다.
최신 실행은 18개 공개 자료에서 OOF와 시험 예측 쌍 182개를 모으고, OOF 바이트 해시가 같은 2개를 제거한 뒤 3개를 추가로 제외해 177개를 유지한다.
제외 대상은 비유한 값이나 단독 AUC `0.90` 미만 예측과, OOF와 시험 예측을 각각 순위화한 뒤 40,000개씩 표본 추출해 KS 통계가 `0.05`를 넘는 예측이다.
다만 각 배열을 따로 순위화하면 연속 예측의 주변분포가 거의 균등해지므로 이 KS는 일반적인 분포 이동보다 동점과 지지집합 변화에 민감한 검사다.
이 검사를 이식한다면 원시 또는 잘린 logit 분포와 고유값·동점률 차이를 별도로 기록해야 한다.

각 구성원 OOF와 시험 예측은 배열 안에서 백분위 순위로 바꾼 뒤 표준정규 분위수로 옮긴다.
노트북은 커뮤니티 공통 `StratifiedKFold(5, shuffle=True, random_state=42)`의 각 평가 fold를 한 번씩 제외하고, 나머지 네 fold의 OOF에 `StandardScaler`와 L2 로지스틱 회귀를 맞춰 제외한 fold를 예측한다.
이 절차는 저장소 ADR 0001이 정의한 outer-fold 결합 평가의 형태와 맞지만, rank-gauss 변환 자체는 outer 분할 전에 OOF 전체에서 한 번 계산한다.
순위 변환은 타깃을 보지 않지만 평가 fold의 예측 분포를 보는 전이식 변환이므로, 이슈 64에서 시험할 때는 outer 학습 부분의 경험적 누적분포만 맞추고 평가 부분에는 적용만 해야 한다.

최종 제출은 177개 전체 OOF와 목표값에 표준화와 로지스틱 회귀를 다시 맞추고 시험 예측에 적용한다.
전체 OOF로 최종 2단 모델을 맞추는 것 자체는 시험 예측 생성 절차지만, 같은 OOF에서 표현, 규제, 구성원 집합과 여러 실패안을 반복 비교한 선택 편향까지 제거하지는 않는다.

## 167-model 수치의 신뢰 범위

최신 실행 기록은 최고 단일 구성원 `0.969856`, 균등 순위 평균 `0.968267`, 177개 nested rank-gauss 결합 `0.970093`을 직접 출력한다.
같은 공개 구성원으로 다시 맞춘 blend를 이름 접두사로 제외하면 168개가 남고, 그 nested OOF는 `0.970083`으로 전체보다 `0.000010` 낮다.
작성자가 본문에서 보고한 Public LB는 `0.97113`이지만 공개 출력에는 Kaggle 제출 채점 증거가 없으므로 작성자 보고치로만 취급한다.

작성자는 자체 모델 5개 추가가 nested OOF를 `0.970043`에서 `0.970081`로, Public LB를 `0.97106`에서 `0.97113`으로 높였다고 쓴다.
또한 구성원 79개에서 132개로 늘린 효과 `+0.00032`, rank-gauss의 logit 대비 효과 `+0.00008`, 여러 모델 변형과 결측 구간 결합의 효과를 표로 제시한다.
그러나 최신 판본의 실행 코드는 이 제거 실험들을 다시 계산하지 않고, 공개 실행 기록에도 그 중간 결과가 없다.
따라서 이 수치는 재현된 최신 출력이 아니라 작성자 보고치다.

노트북은 같은 OOF에서 이미 맞춘 `naji_blend*` 구성원이 전체 수치를 낙관적으로 만든다고 정확히 경고한다.
그 구성원을 뺀 수치를 honest라고 부르지만, 다른 공개 OOF의 학습 소스, fold 안 전처리, 하이퍼파라미터 선택과 전체 OOF 반복 선택까지 전수 검증하지는 않는다.
최종 제출은 경고한 blend 구성원을 다시 포함한 177개 전체를 사용한다.
따라서 `0.970083`도 우리 판정 계약에서 검증된 자체 OOF가 아니며, `0.97113`은 더구나 모델 채택 근거가 아니다.

## 167-model에서 재사용할 것과 재사용하지 않을 것

배열 해시로 정확 중복을 제거하는 절차는 이슈 63 본문에 이미 명시되어 있으므로 새 발견이 아니다.
원본 행 순서와 fold 식별자를 확인해야 한다는 지적도 이슈 63의 정렬·fold 일치 검사와 겹친다.
외부 OOF는 지도에서 szymonkapiski 25구성원판의 읽기 전용 진입 진단만 예외로 허용했고, 이 177개 묶음은 그 예외에 포함되지 않는다.

선형 결합에서 `max(n_iter_) < max_iter`를 확인하고 미수렴 실행을 거부하는 검사는 이슈 64 구현에 넣을 가치가 있다.
rank-gauss도 백분위 순위와 잘린 logit을 비교하는 이슈 64 안에서 추가 표현 하나로 시험할 수 있다.
단, 표현 선택과 정규화는 outer 학습 부분 안에서만 수행하고 ADR 0001의 `+0.0001` 미만 단순 방식 우선 규칙을 그대로 적용해야 한다.

서로 다른 피처 관점이 시드 변형보다 낫다는 방향은 현재 후보 풀이 모델 계열과 피처 계획을 분리해 측정하는 이유를 재확인한다.
그러나 이를 시드 평균을 금지하는 규칙으로 옮기면 안 된다.
현재 [Lookup-Transformer fold 내 초기화 평균 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/127)은 fold 내 초기화 3개 평균이 단일 OOF를 `+0.00009792` 높이고 표준 평가 앙상블을 `+0.00016654` 높였음을 자체 고정 fold에서 확인했다.
외부 노트북의 서술보다 이 자체 짝비교가 이 저장소의 시드 평균 결정에 우선한다.

stack 잔차를 목표로 새 특성을 검사하라는 제안은 한계 기여를 보라는 원칙으로는 맞다.
그러나 최신 판본에는 35개 생성 지문 후보가 합쳐서 `-0.000182`였다는 계산 코드와 출력이 없고, 이 저장소도 남은 특성 공간을 이미 별도 조사와 대리 스크리닝으로 통제한다.
따라서 새 잔차 특성 탐색 티켓을 열지 않는다.

## 단변량 spline Transformer의 정확한 방법

이 모델은 원시 수치 9개, 행 단위 파생 수치 10개, 정확값 목표 인코딩 12개와 빈도 인코딩 12개를 합친 수치 43개를 사용한다.
범주 입력은 원래 설명변수 12개의 정확값 문자열과 원시 수치 9개의 결측 표시를 합친 21개다.

각 수치 열은 학습 fold 분위수로 만든 여러 해상도의 학습 가능 조각선형 spline, 거친 spline, 작은 MLP와 원시값 경로를 거친다.
동적 gate가 네 경로와 여러 spline 해상도를 합치고, 열별 잔차 블록이 64차원 token을 만든다.
43개 열 전체를 보는 작은 `43 -> 32 -> 43` SiLU 망이 spline 앞에서 열별 보정값을 만들며, 마지막 층은 0으로 초기화한다.

각 열 token의 점수를 합한 가산 logit은 보조 손실로 학습된다.
그 뒤 64차원 8-head self-attention 한 층이 열 상호작용을 만들고, 최종 MLP가 표준화 수치, 원래 token, attention token, 범주 embedding, 열별 가산 점수와 가산 확률을 모두 받아 최종 logit을 직접 예측한다.
실행 기록의 fold별 학습 가능 파라미터 수는 약 710만 개다.

목표 인코딩은 outer 학습 부분 안에서 다시 5-fold로 나눠 학습 행 자체를 제외해 만든다.
outer 검증과 시험 인코딩도 inner 학습 부분에서 만든 다섯 mapping의 평균이므로 목표값 직접 누출은 확인되지 않았다.
표준화와 spline knot도 outer 학습 부분만으로 맞춘다.

다만 범주 어휘는 각 outer fold에서 `fold_train`, `fold_val`, `fold_test`를 합친 뒤 구성한다.
타깃을 보지는 않지만 평가 fold와 시험의 정확값이 embedding 크기와 무작위 초기화 순서에 영향을 주고, 학습 fold에 없는 평가 전용 값은 학습되지 않은 별도 embedding으로 남는다.
이는 저장소 Lookup-Transformer가 학습 fold 전용 어휘와 조회 어휘 미등록값을 쓰기로 한 결정과 충돌한다.
재검토하더라도 어휘는 outer 학습 부분만으로 만들고 검증·시험 미등록값을 하나의 식별자로 보내야 한다.

모델은 seed 21의 `StratifiedKFold(5, shuffle=True)`를 쓰고 fold마다 seed를 21부터 25까지 바꾼다.
이는 커밋된 `artifacts/folds.parquet`의 seed 42 fold와 달라 현재 실행들과 OOF를 직접 짝비교할 수 없다.
각 fold는 같은 검증 fold AUC로 최대 35 epoch 중 checkpoint를 고르고 그 예측을 최종 fold 점수에도 사용하므로, 보고 OOF에는 early stopping 선택의 낙관성이 조금 포함될 수 있다.

## 단변량 spline 수치의 신뢰 범위

공개 OOF 파일은 id 0부터 691,368까지 중복 없이 691,369행을 담고 결측값이 없다.
저장 파일에서 다시 계산한 최종 OOF AUC는 `0.9665204982`이고 가산 경로 OOF AUC는 `0.9621312273`이다.
실행 기록의 반올림 출력 `0.966520`과 `0.962131`에 일치한다.

fold별 최종 AUC는 `0.9676373`, `0.9680005`, `0.9673817`, `0.9676256`, `0.9667868`이고 평균은 `0.9674864`다.
전체 OOF AUC는 fold별 확률 눈금 차이까지 함께 순위를 매기므로 fold AUC 평균보다 낮으며, 판정에는 전체 OOF `0.9665205`를 써야 한다.
저장 출력에는 Public LB가 없고 본문도 LB를 주장하지 않는다.

현재 champion `exp067_lookup_xgb_impute_comps5`의 3시드 OOF는 `0.9690978395`다.
후보 풀의 `exp081_lookup_fold_initialization_avg3`는 `0.9691957618`, TabM은 `0.9683261182`, TabPFN-3은 `0.9672432267`이다.
fold가 달라 엄밀한 차이는 아니지만 공개 spline OOF는 이 값들보다 각각 약 `0.00268`, `0.00181`, `0.00072` 낮다.

낮은 단독 OOF만으로 다양성 가치를 기각할 수는 없다.
실제로 후보 풀에는 단독 OOF `0.9596584`인 정확값 one-hot 로지스틱 회귀도 작은 양수 기여로 남아 있다.
그러나 spline 공개 OOF는 고정 fold가 아니고 외부 OOF 예외에도 속하지 않아, 현재 풀과의 상관이나 제외 전후 기여를 ADR 0001의 증거로 계산할 수 없다.

실행 기록은 T4에서 5-fold 단일 시드에 124.4분이 걸렸고 fold당 약 22분에서 30분이 걸렸다고 보고한다.
별도 컴파일 확장이나 유료 가중치는 없어서 계산 가능성은 높다.
하지만 3시드 확정은 단순 직렬 기준 약 6.2 GPU 시간이 필요하고, 현재 증거는 그 비용을 들일 만큼 강하지 않다.

최종 경로와 가산 경로만 비교하므로 `+0.004389` 차이를 spline, 사전 문맥 보정, attention, 범주 embedding, 정확값 목표·빈도 인코딩이나 큰 최종 MLP 중 어느 하나에 귀속할 수 없다.
기존 Lookup-Transformer가 이미 정확값 lookup embedding, 주기 수치 표현과 attention을 쓰고 TabM도 후보 풀에 있으므로, 제거 실험 없는 전체 구조는 독립 기제의 근거가 아니다.

## 최초 검토 당시 후속 결정

새 티켓은 만들지 않는다.
167-model 노트북에서 재사용할 검사는 이슈 63과 이슈 64의 기존 질문 안에 들어가며 외부 예측 자체는 지도 규칙으로 제외된다.
단변량 spline 모델은 공개 실행이 완주되고 계산 가능하지만, 고정 fold 불일치, 전이식 범주 어휘, 낮은 단일 OOF, 사용 조건 불명확과 핵심 제거 실험 부재가 동시에 남는다.

향후 원저자나 다른 1차 자료가 학습 fold 전용 어휘와 seed 42 공통 fold로 만든 id 정렬 OOF를 공개하고, 단변량 spline 핵심을 뺀 대조 대비 전체 OOF `+0.0001` 이상과 현재 Lookup·TabM 계열 대비 중복 문턱 `0.998` 미만을 함께 보일 때만 새 진입 진단을 다시 검토한다.
그 전에는 외부 OOF를 가져오거나 전체 모델을 재구현하지 않는다.

## 재검토: 자체 공통 절차로 고쳐 실행할 가치

### 갱신된 결론

공개 노트북 코드는 명시적 사용 허가가 없으므로 그대로 복사하지 않는다.
그러나 누출 경계와 공통 fold를 바로잡아 전체 모델 구조를 독립 구현한 fold 0 진입 진단은 열 가치가 있다.
공개 실행의 약 710만 학습 매개변수는 모델 한 개 안에서 학습되는 숫자값의 개수이며, 69만 행 자료와 T4 fold 0 실행 시간 25.2분을 고려하면 그 규모 자체는 탈락 사유가 아니다.
최초 검토의 "원저자가 공통 fold 제거 실험을 공개할 때까지 기다린다"는 조건은 지나치게 강했다.
우리 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)과 [공통 fold](../../artifacts/folds.parquet) 안에서 직접 대조하면 외부 자료가 대신 해결해 줄 필요가 없는 질문이기 때문이다.
재검토에서 최신 공개 소스와 출력을 다시 내려받았고, [판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)의 소스 SHA-256 `c308b69c...f4781a2`와 [공개 실행 기록](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/output)의 SHA-256 `c683bef7...b681d6`은 최초 조사와 같았다.

이 결론은 공개 OOF나 제출 예측을 후보 풀에 넣자는 뜻이 아니다.
공개 결과는 구조가 실제로 학습되고 계산 가능한지를 보여 주는 참고 근거로만 쓰고, 모든 개선 판정은 자체 구현의 공통 fold OOF로 새로 만든다.

### 최초 기각 근거에서 유지할 것과 고칠 것

- seed 21 fold, 학습·검증·시험을 합친 범주 어휘, 구성 요소 제거 실험 부재와 코드 사용 조건 불명확은 [판본 3 코드](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)를 그대로 실행하거나 공개 OOF를 채택할 수 없다는 근거로는 타당하다.
- 이 네 항목은 자체 구현에서 공통 fold, 학습 fold 전용 어휘, 사전 고정 제거 대조와 독립 구현으로 고칠 수 있으므로 구조 자체를 기각하는 근거로 쓰면 안 된다.
- 같은 검증 fold에서 checkpoint를 고르고 그 fold AUC를 보고하는 방식은 약한 선택 편향을 남기지만, 현재 [Lookup-Transformer 구현](../../src/pipeline/lookup_transformer.py)도 검증 fold early stopping을 사용하므로 후보와 기준을 같은 규약으로 맞추는 문제이지 이 구조만의 탈락 사유가 아니다.
- 공개 최종 OOF `0.9665204982`는 현재 champion `0.9690978395`보다 낮지만 정식 판정이 아닌 수치 비교로는 [후보 풀](../../artifacts/pool.yaml)의 진입 하한인 champion 대비 `-0.01` 안에 있고, 단독 OOF `0.9596584`인 정확값 one-hot 모델도 작은 양수 기여로 유지된 전례가 있어 낮은 단독 점수만으로 다양성 가치를 닫을 수 없다.
- 공개 최종 경로와 가산 경로의 차이 `+0.004389`는 여러 구성 요소가 한꺼번에 바뀐 결과라 spline의 효과량으로 읽을 수 없다는 최초 지적은 그대로 유효하다.

### 원 논문과 공식 구현까지 거슬러 올라간 결과

확인한 [Kaggle 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)에는 원 논문, 외부 공식 구현이나 모형 자료를 가리키는 인용이 없고 입력도 대회 자료 하나뿐이다.
따라서 이 정확한 전체 구조는 원 논문이 있는 모형의 공식 재현판으로 취급하지 않고, 공개 노트북 저자의 대회용 조합으로 취급한다.

코드의 `UnivariateSpline`은 cubic B-spline 기저를 계산하지 않는다.
학습 fold 분위수 knot 사이에서 학습 가능한 벡터 값을 선형 보간하므로, 수학적으로는 학습 가능한 연속 조각선형 수치 임베딩에 가깝다.
[On Embeddings for Numerical Features in Tabular Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)과 [공식 구현](https://github.com/yandex-research/rtdl-num-embeddings)은 조각선형 인코딩과 주기 인코딩 모두가 수치를 벡터로 바꾼 뒤 다른 열과 섞는 강한 일반 방법임을 보여 준다.
[From Uniform to Learned Knots](https://arxiv.org/abs/2604.05635)은 25개 표 자료의 MLP, ResNet과 FT-Transformer에서 spline 계열을 직접 비교했고, 분류에서는 조각선형 인코딩이 가장 견고하며 spline은 경쟁 가능하지만 일관되게 우월하지 않다고 보고한다.
이 논문은 cubic B-, M-, I-spline 기저와 knot 배치를 연구하므로 노트북의 선형 보간 구현과 같은 모형은 아니지만, 수치 표현을 분리 대조해야 한다는 근거와 무제한 전체 복제를 막는 반증을 함께 제공한다.

### Lookup-Transformer와 TabM에 비해 실제로 새로운 부분

현재 [Lookup-Transformer](../../src/pipeline/lookup_transformer.py)는 정확값 lookup embedding과 학습 가능한 주기 수치 임베딩을 더한 뒤 4층 Transformer에서 모든 열을 섞는다.
노트북은 다중 해상도 조각선형 경로, 거친 경로, 작은 MLP와 원시값 경로를 열마다 먼저 결합하고 열별 잔차 블록을 통과시킨 뒤, 가산 보조 손실과 1층 attention으로 상호작용을 늦춘다.
따라서 Lookup 대비 다른 귀납 편향은 spline이라는 이름 하나가 아니라 "열별 비선형 학습에 용량을 먼저 쓰고, 가산 예측을 강제한 뒤, 얕게 상호작용한다"는 순서다.

반면 spline 표현만 떼어 보면 이미 [TabM 설정](../../configs/exp065_tabm.yaml)의 `num_emb_type: pwl`과 크게 겹친다.
수치 정확값의 문자열 범주 embedding, attention과 최종 공동 예측도 Lookup 계열과 겹친다.
하지만 네 전문가 경로를 열별로 먼저 결합하고, 사전 문맥 보정과 가산 보조 손실을 거쳐 얕은 attention과 최종 공동 예측으로 가는 전체 순서는 어느 기존 자체 모델도 시험하지 않았다.
따라서 전체 구조를 주 후보로 실행하고, spline 경로만 현재 Lookup의 주기 수치 표현으로 바꾼 대조군을 붙여 겹치는 부분의 기여를 분리하는 편이 질문에 맞다.

### 완료된 결정들과의 충돌 여부

- [전체 실험 지도](https://github.com/tmheo/predicting-smartphone-addiction/issues/44)의 외부 예측 제외 규칙은 그대로 지키며, 이번 후보는 공개 OOF가 아니라 자체 공통 fold 실행만 후보 풀과 최종 제출에 쓴다.
- [Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58)은 정확값 lookup과 주기 수치 표현을 함께 쓰는 계열이 트리와 다른 오차를 만든다는 사실을 확정했지만, 열별 깊은 조각선형 표현을 먼저 학습하는 순서는 시험하지 않았다.
- [Lookup 복원 특성 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/106)은 exp067 피처 계획을 현재 champion으로 만들었으므로, 새 구조도 같은 피처 계획을 써서 학습기 차이만 비교해야 한다.
- [Lookup 용량·규제 재검증](https://github.com/tmheo/predicting-smartphone-addiction/issues/108)은 같은 Lookup 계열의 폭, 깊이와 규제 조정을 소진했지만, 수치 표현과 상호작용 순서를 함께 바꾸는 이번 가설을 반증하지 않는다.
- [Lookup fold 내 초기화 평균 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/127)은 exp081이 단독 개선 문턱에는 아주 조금 못 미쳐도 후보 풀 기여 `+0.00016654`를 만들 수 있음을 보여 주므로, 새 구조도 champion 교체와 다양성 기여를 따로 판단해야 한다.
- [조회 어휘 미등록값 진단](https://github.com/tmheo/predicting-smartphone-addiction/issues/128)은 미등록값 손실의 낙관적 상한이 `+0.000006839`에 불과하다고 확정했으므로, spline 후보의 근거를 미등록값 일반화로 설명하면 안 된다.
- [새 딥러닝 아키텍처 후보 지도](https://github.com/tmheo/predicting-smartphone-addiction/issues/135)는 TabR-S, TabICLv2, AMFormer와 Trompt라는 전체 모형 후보를 이미 골랐지만, 이번 후보는 S6E8에서 직접 완주한 전체 구조를 약 1 GPU 시간의 fold 0 진입 진단으로 먼저 거르므로 그 포트폴리오를 다시 여는 일이 아니다.
- [최초 노트북 검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/148)의 공개 구현 감사 결과는 유지하되, 공개 구현의 결함을 자체 구현 가능한 구조의 탈락 사유로 확장한 결론만 고친다.

### 누출, 재현성과 독립 구현 경계

자체 구현은 [exp067 피처 계획](../../configs/exp067_lookup_xgb_impute_comps5.yaml), [공통 fold](../../artifacts/folds.parquet)와 seed 42를 그대로 사용해야 한다.
표준화, knot, 범주 어휘와 모든 전처리는 outer 학습 부분에서만 맞추고, 검증·시험의 미등록값은 학습되지 않은 개별 embedding이 아니라 공통 미등록 식별자로 보낸다.
첫 진입 진단에서는 노트북의 12개 정확값 목표 인코딩과 12개 빈도 인코딩을 빼야 한다.
정확값 목표 인코딩은 [Lookup 복원·정확값 목표 인코딩 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/106)에서 카나리아가 실패했고, 빈도 인코딩은 앞선 [단일 열 빈도와 추가 정확값 표현 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/49)에서 기각됐으므로 핵심 구조의 효과와 섞을 이유가 없다.
노트북 코드에는 명시적 사용 허가가 없으므로 코드를 복사하지 않고, 공개된 알고리즘 설명과 사용 조건이 분명한 [수치 임베딩 공식 구현](https://github.com/yandex-research/rtdl-num-embeddings)을 참고해 독립 구현해야 한다.

### 전체 5-fold 전에 할 저비용 진입 진단

진입 진단의 기준은 exp067의 seed 42, fold 0 저장 예측으로 고정하고 다시 학습하지 않는다.
첫 후보 `M0`는 exp067의 피처 계획을 사용하면서 노트북의 전체 모델 순서를 보존한 독립 구현이다.
각 연속 열은 다중 해상도 조각선형, 거친 조각선형, 작은 MLP와 원시값 전문가를 동적으로 결합하고 열별 잔차 블록을 통과한다.
전체 수치 행의 작은 사전 문맥 보정, 열별 가산 보조 손실, 1층 attention, 정확값 범주 embedding과 원래 token·attention token·가산 점수를 함께 받는 최종 MLP도 유지한다.
노트북의 목표·빈도 인코딩만 이전 자체 실험에서 이미 기각된 입력 피처이므로 제외하며, 이는 모델 구조를 줄이는 변경이 아니라 학습기 비교를 위해 입력 피처 계획을 exp067로 고정하는 변경이다.

제거 대조 `A0`도 같은 fold 0 작업에서 함께 실행한다.
`A0`는 `M0`의 전문가 결합, 사전 문맥 보정, 열별 블록, 가산 손실, attention과 최종 MLP를 그대로 두고 다중 해상도 조각선형 전문가만 매개변수 규모를 맞춘 주기 수치 표현으로 바꾼다.
따라서 `M0`와 `A0`의 차이는 조각선형 표현의 기여를, `A0`와 exp067의 차이는 열별 선행 학습과 얕은 상호작용의 기여를 가른다.

- champion 교체 목표는 fold 0 AUC가 같은 seed의 exp067보다 낮지 않은 경우다.
- 다양성 구성원 목표는 fold 0 AUC가 exp067 대비 `-0.01` 진입 하한 안에 있고, 기존 풀 최근접 스피어만 상관이 `0.998` 미만이며, 고정 순위 평균 기여가 독립 난수와 기존 구성원 복제 대조의 상단보다 큰 경우다.
- 두 목표를 모두 놓치면 전체 5-fold로 확장하지 않고 즉시 닫는다.
- 둘 중 하나를 통과하면 `M0`의 seed 42 전체 5-fold를 실행하고, 그 뒤의 3시드 확정과 풀 진입은 [ADR 0001](../adr/0001-experiment-adoption-contract.md)을 그대로 따른다.

[공개 출력](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/output)의 fold별 시간은 22.1분에서 29.8분이고 fold 0은 25.2분이며, 43개 수치 token과 21개 범주 embedding을 가진 전체 모형의 5-fold가 T4에서 124.4분 걸렸다.
문제를 고친 전체 구조 `M0`과 대조 `A0`의 fold 0 합계는 약 1 GPU 시간을 예상하며, 중요도 계산을 포함한 예산 상한은 1.5 GPU 시간으로 잡는다.
seed 42 전체 실행은 2시간에서 3시간, 3시드 확정은 단순 직렬 기준 6시간에서 8시간을 예산 상한으로 잡되, fold 0 관문을 통과하기 전에는 이 비용을 쓰지 않는다.

공개 OOF는 정식 판정 자료가 아니지만 구조적 다양성이 완전히 없는지 확인하는 참고 진단에는 쓸 수 있다.
id와 목표값을 맞춘 뒤 현재 후보 풀 16개와 비교한 결과, 공개 최종 예측의 가장 가까운 스피어만 상관은 `0.975128`이었고 고정 순위 평균에 추가한 AUC 변화는 `+0.00008088`이었다.
분할과 어휘 구성이 다르므로 이 수치로 후보 풀에 진입시킬 수는 없지만, 자체 공통-fold 진입 진단조차 생략해야 한다는 주장에는 반대되는 근거다.

### 최종 판정

[P3 보강: contextualized deep univariate spline Transformer의 성능·다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/149)을 실제 실험 티켓으로 열었다.
티켓의 질문은 "문제를 고쳐 독립 구현한 전체 contextualized deep univariate spline Transformer가 공통 fold에서 현재 Lookup보다 강하거나 후보 풀에 다른 오차를 더하며, 그 효과에서 조각선형 경로가 실제로 기여하는가"여야 한다.
champion 교체 가능성은 낮음에서 중간이고, 다양성 구성원 가능성은 중간으로 본다.
공개 OOF가 약하지만 풀 하한 안에 있고 Lookup과 처리 순서가 다르기 때문에 다양성 가능성은 남아 있으며, TabM의 PWL과 겹치고 최신 spline 비교가 분류에서 일관된 우위를 보이지 않기 때문에 champion 교체 가능성은 더 낮다.
전체 구조 진입 진단이 실패하면 해상도 수, 동적 결합 방식, 목표·빈도 인코딩이나 최종 MLP 크기를 따로 조정하는 후속 탐색은 열지 않는다.

## 출처

- [S6E8 167 Models Diversity Beats Strength 최신 공개 페이지](https://www.kaggle.com/code/adarsh1077/s6e8-167-models-diversity-beats-strength)는 코드, 서술, 입력 자료와 Public LB 보고치의 1차 출처다.
- [Contextualized Deep Univariate Spline Transformer 최신 공개 페이지](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer)는 구조, 학습 코드와 fold 규율의 1차 출처다.
- [Contextualized Deep Univariate Spline Transformer 공개 출력](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/output)은 OOF, fold 지표, 시험 예측과 실행 시간의 1차 출처다.
- [On Embeddings for Numerical Features in Tabular Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)과 [공식 구현](https://github.com/yandex-research/rtdl-num-embeddings)은 조각선형·주기 수치 임베딩과 후단 모형을 고정한 분리 대조의 1차 출처다.
- [From Uniform to Learned Knots](https://arxiv.org/abs/2604.05635)은 spline 수치 표현의 분류·회귀별 효과와 계산량 차이를 비교한 1차 출처다.
- [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)은 이 저장소의 OOF, 다양성 구성원과 nested 결합 판정 기준이다.
- [기존 Code 조사](code-notebook-insights.md), [후속 Code 조사](code-notebook-insights-2.md), [남은 실험 공간 전수 재점검](remaining-experiment-space-audit.md)은 중복 여부와 현재 우선순위의 저장소 근거다.
- [`artifacts/champion.yaml`](../../artifacts/champion.yaml)과 [`artifacts/pool.yaml`](../../artifacts/pool.yaml)은 현재 champion과 후보 풀 수치의 기록 원본이다.
