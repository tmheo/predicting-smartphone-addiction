# 검색·메모리·행 관계 학습 계열의 딥러닝 실행 후보 조사

## 결론

이 계열에서 지금 실행 지도로 넘길 후보는 1순위 `TabR-S + context freeze`, 2순위 `ModernNCA` 두 개다.
TabR-S는 69만 행에서 시간과 메모리를 통제할 공개 대규모 실행 근거가 있고, 현재 champion이 참조하지 않는 학습 fold의 행별 라벨 문맥을 직접 사용하므로 단일 모델 교체와 후보 풀 다양성 양쪽에서 가치가 있다.
ModernNCA는 더 단순한 학습 거리와 soft nearest-neighbor 예측으로 TabR 및 Lookup-Transformer와 다른 오류를 만들 여지가 있지만, 69만 행의 모든 학습 후보를 반복 참조하는 비용 때문에 1 fold 1 epoch 실측 게이트를 먼저 통과해야 한다.
PTaRL, TabPTM과 ModernNCA 다중 구성 앙상블은 관찰 후보로 남긴다.
Hopular, NPT, PET, TabGSL은 현재 규모와 실행 한도에서 제외하고, BiSHop은 이름과 달리 외부 행 메모리를 쓰지 않으므로 이 계열 후보에서 제외한다.

## 조사 질문과 범위

이 문서는 [리서치: 검색·메모리·행 관계 학습 계열의 실행 후보 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/139)의 답이다.
조사 기준일은 2026-08-15이며 논문, 공식 구현, 공식 저장소의 라이선스와 이 저장소의 자체 OOF 판정만 근거로 썼다.
대상 자료는 691,369개 학습 행과 12개 설명변수로 구성되고, 변수별 관측값 종류 수는 2개에서 1,437개라 열별 정확값 반복이 강하다.
현재 champion은 `exp067_lookup_xgb_impute_comps5`이고 3시드 평균 OOF AUC는 0.96909784다.
현재 champion은 각 행의 정확값 lookup embedding과 PLR 수치 embedding을 feature-token Transformer로 결합하지만 예측할 때 다른 학습 행이나 그 라벨을 참조하지 않는다.
따라서 이 조사에서 말하는 새로운 정보는 열별 정확값 기억이 아니라, 입력 행마다 달라지는 학습 fold 이웃의 피처와 라벨을 직접 참조하거나 학습한 행 구조를 압축한 전역 메모리를 참조하는 정보다.

## 저장소의 판정 경계

[실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)에 따라 단일 모델 교체 후보는 seed 42의 공통 5-fold OOF가 같은 시드 champion 이상이어야 3시드 확정으로 넘어간다.
3시드 확정은 평균 OOF AUC가 champion보다 0.0001 이상 높고 세 시드 중 두 시드 이상이 같은 시드 champion을 이겨야 한다.
개선 폭이 0.0001 이상 0.0002 미만이면 다섯 fold 중 세 fold 이상 승리도 필요하다.
다양성 구성원은 3시드 평균 OOF가 진입 시점 champion보다 0.01 이상 낮지 않아야 하고, 최근접 풀 구성원과의 스피어만 상관이 0.998 이상이면 더 강한 쪽 하나만 남긴다.
중복 게이트를 통과한 후보도 표준 순위 평균 앙상블의 OOF AUC를 실제로 높여야 유지한다.
Public 점수는 어느 판정에도 쓰지 않는다.
이 지도는 16GB에서 24GB 단일 GPU로 약 24 GPU 시간 안에 seed 42 초기 선별을 끝낼 수 있는 후보만 실행 후보로 인정한다.

## 현재 모델들과 구조 차이

| 기준 모델 | 행 사이 정보를 쓰는 방식 | 이번 후보와의 핵심 차이 |
| --- | --- | --- |
| Lookup-Transformer champion | 다른 행을 보지 않고 열마다 학습한 정확값 embedding을 현재 행 안의 Transformer로 결합한다. | TabR와 ModernNCA는 예측할 행마다 학습 fold의 다른 행과 라벨을 직접 참조한다. |
| TabM | 같은 행을 여러 parameter-efficient MLP 구성원이 처리하고 그 예측을 함께 학습한다. | 검색 후보는 parameter ensemble이 아니라 입력별 이웃 집합을 바꾸며 라벨 문맥을 만든다. |
| TabPFN-3 | 사전학습한 in-context model이 제한된 문맥 안의 학습 예를 소비한다. | TabR와 ModernNCA는 이 과제 fold에서 검색 표현 자체를 학습하고 수십만 학습 행을 후보 저장소로 쓴다. |
| 기존 약한 MLP·RealMLP·TabNet·FT-Transformer·범용 ResNet | 기본적으로 한 행씩 처리하거나 이 과제에서 독립 정보 기여가 이미 미미했다. | 이번 실행 후보는 다른 행의 라벨을 비매개 방식으로 입력에 가져온다. |

[Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58)은 원본 프록시 10-이웃 평균을 넣은 변형이 기본 Lookup-Transformer보다 OOF AUC 0.00013 낮았다고 판정했다.
이 결과는 외부 원본 프록시의 고정 이웃 신호가 중복이라는 뜻이지, 각 outer 학습 fold 안에서 현재 과제 라벨로 이웃 표현을 학습하는 TabR와 ModernNCA를 반증하지 않는다.
[RealMLP·TabM의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/61)은 TabM이 champion 교체에는 실패했지만 Lookup-Transformer와 스피어만 상관이 약 0.979이고 표준 앙상블 기여가 약 0.00008이라 다양성 구성원으로 남았다고 판정했다.
[TabPFN-3의 스모크 게이트 통과 시 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/102)은 TabPFN-3이 단독 OOF 0.96724323으로 champion보다 낮았지만 최신 후보 풀에서도 약 0.000028의 양수 기여를 냈다고 판정했다.
이 두 결과는 단독 점수가 조금 낮은 새로운 신경망도 오류 구조가 다르면 가치가 있음을 보여주므로, 검색 후보는 champion 교체와 후보 풀 기여를 별도로 판정해야 한다.

## 우선순위 요약

| 우선순위 | 후보 | 분류 | 69만 행 실행성 | 단일 교체 가치 | 다양성 가치 |
| ---: | --- | --- | --- | --- | --- |
| 1 | TabR-S + context freeze | 실행 | 조건부 높음 | 높음 | 높음 |
| 2 | ModernNCA | 실행 | 실측 게이트 필요 | 중간 이상 | 높음 |
| 3 | ModernNCA 다중 구성 앙상블 | 관찰 | 단일 구성 통과 뒤에만 가능 | 중간 이상 | 중간 |
| 4 | PTaRL | 관찰 | 높음 | 낮음 | 중간 |
| 5 | TabPTM | 관찰 및 사전학습 계열로 이관 | 조건부 높음 | 불명 | 중간 이상 |
| 6 | Hopular | 제외 | 낮음 | 불명 | 불명 |
| 7 | NPT | 제외 | 매우 낮음 | 불명 | 불명 |
| 8 | PET | 제외 | 낮음 | 불명 | 중간 |
| 9 | TabGSL | 제외 | 매우 낮음 | 불명 | 중간 |
| 10 | BiSHop | 이 계열에서 제외 | 높음일 수 있음 | 별도 조사 대상 | 이 계열 근거 없음 |

## 실행 후보 1: TabR-S + context freeze

### 연구 근거와 작동 원리

TabR는 feed-forward encoder 중간에 kNN과 비슷한 검색 분기를 넣고, 학습 후보 행의 embedding, 라벨, 목표 행과 후보 행의 차이를 attention과 비슷한 방식으로 합쳐 예측한다 ([ICLR 2024 논문](https://openreview.net/forum?id=rhgIgTSSxW)).
검색은 모든 후보에 대해 target-to-candidate 점수를 계산한 뒤 상위 문맥만 쓰므로 후보끼리 self-attention을 수행하는 NPT와 달리 후보 수에 대한 이차 attention 복잡도가 없다.
논문은 기본 문맥 크기로 96을 사용했고, 중간 규모 43개 분류·회귀 과제에서 기존 표 자료 딥러닝과 XGBoost를 비교했다.
논문은 최대 1,200,192행의 기본 benchmark와 300만 행 이상의 Weather 자료를 포함해 확장성을 따로 검증했다.
300만 행 전체의 기본 TabR-S는 한 번 학습하는 데 18시간 9분이 걸렸지만, 첫 epoch 뒤 문맥을 고정한 `CF-1`은 3시간 15분으로 약 7배 빨라졌고 예측 오차 증가는 작았다.
논문은 새 학습 행을 재학습 없이 검색 후보에 더할 수 있음을 보여 주고, 큰 자료에서는 일부 행으로 학습한 뒤 전체 자료를 후보로 참조하는 경로도 예비 검증했다.
공식 구현은 context freeze와 학습 뒤 후보 추가 경로를 모두 제공한다 ([공식 구현](https://github.com/yandex-research/tabular-dl-tabr)).

### 이 과제에서 새로운 정보가 되는 이유

Lookup-Transformer는 열별 값 ID를 기억하지만 `age=21`과 `daily_screen_time_hours=8.4`를 함께 가진 특정 학습 행의 라벨을 예측 시점에 읽지 않는다.
TabR는 현재 행마다 학습한 metric으로 96개 문맥 행을 고르고 문맥 라벨과 목표-문맥 표현 차이를 함께 사용한다.
따라서 정확값이 여러 열에서 동시에 반복되는 이 과제에서는 같은 값 셀의 독립 합보다 공동 반복 행 주변의 국소 라벨 표면을 만들 수 있다.
원본 프록시 kNN은 외부 자료의 고정 거리와 고정 라벨 평균이지만 TabR는 현재 과제의 outer 학습 fold 안에서 검색 metric과 라벨 결합을 함께 학습하므로 정보 원천과 오류 경로가 다르다.
TabM의 여러 MLP 구성원이나 일반 Transformer와도 달리 목표 행마다 다른 라벨 문맥을 사용하므로 후보 풀의 새 오류 계열 가능성이 높다.

### 확장성, 메모리와 추론 비용

추론 때 각 fold의 학습 행 전체가 후보 저장소이며 실제 결합은 상위 96개 문맥에 한정된다.
후보 embedding과 라벨은 미리 보관할 수 있지만 목표-후보 차이에 의존하는 value 표현은 실행 중 계산해야 한다.
논문의 GPU 실험 대부분은 A100에서 수행되었고 일부는 2080 Ti에서 수행되었으므로, 16GB에서 24GB 환경 적합성은 논문 수치만으로 단정하지 않고 사전 실측해야 한다.
우리 자료는 논문의 120만 행 benchmark보다 작고 피처 수도 훨씬 적지만, 5-fold와 permutation importance가 단일 holdout 논문 실행보다 비싸다는 점을 시간 예산에 반영해야 한다.

### 사용 조건

공식 TabR 코드는 MIT 라이선스라 대회용 수정, 학습과 예측에 사용할 수 있다 ([라이선스](https://github.com/yandex-research/tabular-dl-tabr/blob/main/LICENSE)).
이 후보는 공개 사전학습 가중치를 쓰지 않고 각 fold에서 처음부터 학습하므로 외부 가중치 계보 검사가 필요하지 않다.
Faiss는 별도 설치가 필요하고 공식 저장소도 conda 계열 환경을 권장하므로 원격 실행 명세에서 CUDA와 Faiss 판본을 고정해야 한다.

### 최소 사전 진단

1. 현재 champion과 같은 `exp067_lookup_xgb_impute_comps5` 피처 계획을 쓰고, 전처리와 후보 저장소는 outer 학습 fold에만 적합한다.
2. 논문 기본 `TabR-S`, 문맥 크기 96, 첫 epoch 뒤 context freeze를 고정하고 하이퍼파라미터 탐색은 하지 않는다.
3. fold 0에서 seed 42로 두 epoch만 실행해 첫 epoch와 둘째 epoch의 문맥 ID 변화율, peak GPU memory, epoch 시간, 검증 AUC와 예측 유한성을 기록한다.
4. 같은 fold에서 검증 행이 후보 ID에 들어가지 않고 학습 행 자체가 자기 문맥에서 빠지는지 assertion과 저장된 표본으로 검증한다.
5. fold 0 실행이 4 GPU 시간을 넘거나 peak memory가 장비 용량의 90%를 넘거나 아래의 제한된 permutation importance까지 포함한 전체 5-fold 예상 시간이 20 GPU 시간을 넘으면 정식 seed 42 실행을 시작하지 않는다.
6. 비용 게이트를 통과하면 공통 5-fold seed 42 OOF를 완주하고 champion의 같은 시드 OOF 0.96876256과 비교한다.
7. champion을 이기지 못해도 OOF가 champion보다 0.01 안에 있으면 최근접 스피어만 상관과 표준 순위 평균 기여를 계산해 다양성 경로를 판정한다.

### 즉시 중단 조건

후보 저장소에 검증 행이나 검증 라벨이 들어가면 결과를 판정 불가로 폐기한다.
메모리 또는 시간 게이트를 한 번 낮은 batch size로 재측정해도 통과하지 못하면 24시간 실행 후보에서 제외한다.
seed 42 OOF가 champion보다 낮고, 동시에 최근접 상관이 0.998 이상이거나 표준 순위 평균 기여가 0 이하이면 3시드로 확장하지 않는다.
문맥의 대부분이 열별 정확값 동일 행으로 고정되고 champion과의 상관이 0.998 이상이면서 성능도 낮으면 검색 너비나 encoder를 추가 탐색하지 않는다.

## 실행 후보 2: ModernNCA

### 연구 근거와 작동 원리

ModernNCA는 입력을 선형층과 작은 MLP로 metric 공간에 투영하고, 같은 라벨 행은 가깝고 다른 라벨 행은 멀어지도록 soft nearest-neighbor 손실을 직접 최적화한다 ([ICLR 2025 논문](https://openreview.net/forum?id=JytL2MrlLT)).
학습 때는 매 mini-batch마다 학습 후보의 일부를 무작위로 뽑는 Stochastic Neighborhood Sampling을 사용하고, 추론 때는 학습 행 전체에 대한 거리와 라벨 가중 평균을 사용한다.
논문은 1,000행에서 100만 행 사이의 300개 자료를 평가했고 ModernNCA가 CatBoost와 비슷한 평균 순위를 보였다고 보고했다.
같은 논문의 300개 자료 평균은 한 seed 학습 87.5초, GPU memory 5.36GB였지만 평균값이라 100만 행의 최대 비용을 나타내지는 않는다.
공식 TALENT 구현은 `torch.cdist`로 query batch와 후보 행의 모든 거리를 계산하고, 학습 때 후보를 표본추출하며, 검증과 추론 때 전체 학습 행을 후보로 전달한다 ([모델 코드](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py), [실행 코드](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/methods/modernNCA.py)).

### 이 과제에서 새로운 정보가 되는 이유

ModernNCA의 최종 예측은 각 열의 독립 lookup이 아니라 학습된 전체 행 거리 위의 라벨 확률 질량이다.
TabR처럼 별도 label embedding과 target-candidate 차이 network를 두지 않으므로, 더 부드러운 metric과 전체 이웃 라벨 평균이라는 다른 귀납 편향을 가진다.
현재 champion, TabM과 TabPFN-3 어느 것도 이 과제 전체 학습 fold에 대해 task-fitted soft nearest-neighbor 확률을 직접 계산하지 않는다.
저차원 12열 자료는 ModernNCA 논문이 한계로 든 `d`가 행 수보다 훨씬 큰 고차원 조건과 반대라 metric 학습에 유리한 쪽이다.

### 확장성, 메모리와 추론 비용

학습 비용은 query batch 크기와 표본추출한 후보 수의 곱에 비례하고, 추론 비용은 query 수와 학습 fold 행 수의 곱에 비례한다.
한 outer fold에는 약 55만 개 학습 후보와 약 14만 개 검증 query가 있으므로, 공식 exact 추론은 fold마다 약 760억 개 query-candidate 거리 원소를 계산해야 한다.
거리 행렬은 batch 단위로 버릴 수 있어 전체 행렬을 메모리에 둘 필요는 없지만, 계산량은 그대로 남는다.
논문은 학습 후보 30%에서 50%가 평균 성능과 일반화에 좋았다고 보고했으므로 비용 때문에 5% 이하로 임의 축소한 설정을 논문 재현으로 간주해서는 안 된다.
따라서 ModernNCA는 69만 행에서 작동 가능하다는 연구 근거는 있지만, 24 GPU 시간 안에 5-fold seed 42를 끝낼지는 장비별 실측으로만 확정할 수 있다.

### 사용 조건

논문이 지정한 공식 구현인 TALENT는 MIT 라이선스이며 ModernNCA 구현과 수정·배포 권한을 함께 제공한다 ([라이선스](https://github.com/LAMDA-Tabular/TALENT/blob/main/LICENSE)).
공개 사전학습 가중치를 사용하지 않고 각 fold에서 처음부터 학습하므로 코드와 종속 패키지의 라이선스만 원격 명세에 고정하면 된다.

### 최소 사전 진단

1. 현재 champion의 피처 계획을 고정하고 수치 표준화, 범주 one-hot, 결측 처리는 outer 학습 fold 안에서만 적합한다.
2. 논문의 가장 작은 공개 구조인 embedding 차원 64, residual block 0개, 후보 표본 비율 0.30과 batch size 512를 첫 실측점으로 고정한다.
3. fold 0에서 seed 42로 한 epoch와 전체 검증 추론을 실행해 peak memory, 학습 epoch 시간, 검증 추론 시간과 AUC를 기록한다.
4. 아래의 제한된 permutation importance까지 포함한 전체 5-fold의 seed 42 예상 시간이 20 GPU 시간 이하이고 peak memory가 장비 용량의 90% 이하면 같은 설정으로 공통 5-fold를 완주한다.
5. 비용만 초과하면 batch size를 한 번 낮춰 memory와 처리량을 재측정하되, 후보 표본 비율과 exact 전체 후보 추론은 바꾸지 않는다.
6. 공통 seed 42 OOF 뒤에는 TabR, champion, TabM과 TabPFN-3 각각의 스피어만 상관과 표준 순위 평균 기여를 함께 계산한다.

### 즉시 중단 조건

낮은 batch size 재측정 뒤에도 5-fold 예상 시간이 20 GPU 시간을 넘으면 approximate nearest-neighbor 변형을 새 모델처럼 만들지 않고 관찰 후보로 내린다.
fold 0 검증에 학습 fold 밖 라벨이 섞이거나 자기 행 제거가 실패하면 결과를 폐기한다.
seed 42 OOF가 champion보다 낮고, 후보 풀 하한도 넘지 못하면 3시드로 확장하지 않는다.
후보 풀 하한을 넘더라도 최근접 상관이 0.998 이상이면서 더 강한 기존 구성원이 있거나 표준 순위 평균 기여가 0 이하이면 중단한다.

## 관찰 후보

### ModernNCA 다중 구성 앙상블

2026년 갱신된 300개 자료 benchmark는 ModernNCA 다중 구성 앙상블이 단일 ModernNCA보다 안정적이고 여러 과제에서 상위권이라고 보고한다 ([공식 benchmark 논문](https://www.lamda.nju.edu.cn/caihr/file/tabular_benchmark/paper.pdf)).
그러나 이것은 새로운 행 관계 아키텍처가 아니라 여러 ModernNCA 설정과 시드의 비용을 더하는 방식이다.
단일 ModernNCA가 seed 42 성능, 시간과 다양성 게이트를 모두 통과한 뒤에만 3시드 확정 자체를 앙상블로 보고 검토하며, 별도 대규모 설정 앙상블은 24시간 초기 선별 범위에서 열지 않는다.

### PTaRL

PTaRL은 첫 단계에서 전체 학습 표현을 k-means로 압축한 전역 prototype을 만들고, 둘째 단계에서 각 행 표현을 prototype 공간으로 투영하면서 optimal transport, 좌표 다양화와 prototype 직교 제약을 함께 학습한다 ([ICLR 2024 논문](https://openreview.net/forum?id=G32oY4Vnm8)).
추론 때 개별 학습 행을 다시 읽지 않고 피처 수의 로그 정도인 작은 prototype 행렬만 쓰므로 69만 행 확장성은 TabR와 ModernNCA보다 좋다.
논문은 최대 108,000행에서 여섯 backbone 모두의 정확도 또는 RMSE가 개선됐다고 보고했고, prototype 수 `K`에 대한 행별 비용을 `O(K² log K)`로 설명했다.
그러나 이후 300개 자료의 통일 benchmark에서 PTaRL은 TabR, ModernNCA, CatBoost와 강한 MLP 계열보다 낮은 평균 순위를 보였으므로 지금 바로 24 GPU 시간을 배정할 근거는 약하다 ([공식 benchmark 논문](https://www.lamda.nju.edu.cn/caihr/file/tabular_benchmark/paper.pdf)).
원 저자 저장소에는 최상위 LICENSE 파일이 없지만 ([원 저자 구현](https://github.com/HangtingYe/PTaRL)), TALENT의 재구현은 MIT 라이선스 아래 제공된다 ([TALENT PTaRL 코드](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/ptarl.py)).
다른 조사 계열에서 실행 후보가 두 개 미만으로 줄거나, PTaRL을 현재 Lookup backbone에 붙일 논문 수준 근거가 새로 생길 때만 관찰 상태를 다시 연다.

### TabPTM과 검색형 사전학습 model

TabPTM은 행을 여러 거리 함수에서 라벨별 최근접 이웃까지의 거리 벡터로 바꾸고, 여러 자료에서 사전학습한 MLP로 그 meta-representation을 예측에 사용한다 ([논문](https://openreview.net/forum?id=a06UO11IrQ)).
현재 TALENT 구현은 클래스당 최대 10,000개 중심과 32개 이웃, 함께 배포한 사전학습 checkpoint를 사용하므로 전체 55만 학습 행을 매 query마다 참조하는 ModernNCA보다 계산을 제한할 수 있다 ([공식 구현](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/methods/tabptm.py)).
다만 실행 여부는 checkpoint의 사전학습 자료 계보, 대회 외부 자료 조건과 사전학습 계열 후보 사이 성능을 함께 비교해야 한다.
따라서 TabPTM, LoCalPFN과 TabDPT는 [사전학습·자기지도·생성형 표 자료 계열의 실행 후보 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/138)에서 함께 판정하는 편이 중복이 없다.
LoCalPFN은 이웃 검색과 TabPFN fine-tuning을 결합하지만 공식 저장소에 명시적 오픈소스 라이선스 파일이 없고 TabPFN 계열과 구조가 겹친다 ([NeurIPS 2024 논문](https://arxiv.org/abs/2406.05207), [공식 구현](https://github.com/layer6ai-labs/LoCalPFN)).
TabDPT는 Faiss로 검색한 최대 1,024개 안팎의 행 문맥을 공개 pretrained model이 처리하고 코드·가중치·사전학습 자료를 공개했다 ([NeurIPS 2025 논문](https://proceedings.neurips.cc/paper_files/paper/2025/file/fc0e3f908a2116ba529ad0a1530a3675-Paper-Conference.pdf), [Apache-2.0 구현](https://github.com/layer6ai-labs/TabDPT-inference)).
2026년 8월 공개된 TabDPT-Turbo는 오히려 검색을 없애고 긴 행 문맥 attention으로 속도를 높였으므로 이 문서의 검색 후보가 아니라 사전학습 계열의 최신 비교 대상이다 ([TabDPT-Turbo 논문](https://arxiv.org/abs/2608.01400)).

## 제외 후보

### Hopular

Hopular는 각 층이 학습 집합 전체를 고정 associative memory로 사용하고 sample-sample, feature-feature와 feature-target 관계를 Modern Hopfield update로 반복 정제한다 ([논문](https://openreview.net/forum?id=V4Pa9B8zRk), [공식 설명](https://ml-jku.github.io/hopular/)).
연구 범위는 1,000행 이하 소형 자료와 약 10,000행 중형 자료였고, 중형 자료에서도 전체 학습 집합을 memory에 보관했다.
논문의 약 10,000행 실험은 A100에서 단일 표본 경로도 대략 2.4GB에서 3.9GB를 썼고 full-batch 경로는 최대 18.2GB를 썼다.
69만 행에서 같은 전체 memory attention을 5-fold로 실행할 확장 근거가 없고, 후속 TabR가 같은 계열보다 효율이 좋음을 직접 비교했으므로 제외한다.
공식 코드는 MIT 라이선스지만 사용 권한이 계산 한계를 바꾸지는 않는다 ([라이선스](https://github.com/ml-jku/hopular/blob/main/LICENSE)).

### NPT

NPT는 행 전체를 입력으로 받아 행 사이 self-attention을 학습하며, 전체 attention은 행 수에 대해 시간과 공간이 이차로 증가한다 ([논문](https://arxiv.org/abs/2106.02584)).
논문은 약 8,000행에서 표준 크기 model이 24GB GPU memory에 도달해 큰 자료에서는 무작위 mini-batch로 전체 행 attention을 근사했다.
45,730행 Protein 자료의 선택된 한 설정도 약 11시간 51분이 걸렸고, 1,100만 행 Higgs 실행은 mini-batch 근사에서도 약 5일 22시간이 걸렸다.
공식 구현은 CUDA 10.2 이하를 권장하고 최근 대규모 attention 최적화를 포함하지 않으므로 2026년 원격 환경으로 옮기는 비용도 크다 ([공식 구현](https://github.com/OATML/non-parametric-transformers)).
Apache-2.0 라이선스는 허용되지만 69만 행 5-fold를 24시간 안에 선별할 근거가 없어 제외한다 ([라이선스](https://github.com/OATML/non-parametric-transformers/blob/main/LICENSE)).

### PET

PET는 검색한 이웃 행을 hypergraph로 만들고 라벨·피처 message propagation과 고차 피처 상호작용으로 목표 행 표현을 바꾸므로 작동 원리 자체는 이 과제의 다양성 축에 맞는다 ([NeurIPS 2022 논문](https://openreview.net/forum?id=JJCnsgk4OIS)).
그러나 공식 구현은 모든 열을 정수 범주로 만든 CTR·추천 자료를 전제로 Elasticsearch로 검색 pool을 미리 만들며, 연속 12열의 현재 과제용 일반 분류 경로를 제공하지 않는다 ([공식 구현](https://github.com/KounianhuaDu/PET)).
공식 저장소에 명시적 LICENSE가 없고, 의존성이 DGL 0.7과 PyTorch 1.8에 묶여 있으며, 대규모 일반 표 분류 benchmark도 없으므로 제외한다.

### TabGSL

TabGSL은 각 행을 node로 보고 학습한 adjacency에서 kNN graph를 만든 뒤 contrastive graph structure learning과 GCN을 함께 최적화한다 ([논문](https://arxiv.org/abs/2305.15843)).
논문 스스로 평가 자료가 5,000행 미만과 50피처 미만이며 큰 자료의 시간·공간 비용은 미해결이라고 명시한다.
69만 행 전체 adjacency 또는 반복 graph 갱신을 24시간 안에 수행할 공식 희소 구현과 라이선스가 확인되지 않아 제외한다.

### BiSHop

BiSHop은 generalized sparse modern Hopfield 층을 row-wise와 column-wise로 쌓지만, 여기서 row-wise는 한 표본 안의 피처와 embedding 방향을 뜻하고 학습 자료의 다른 행을 외부 memory로 읽는다는 뜻이 아니다 ([ICML 2024 논문](https://arxiv.org/abs/2404.03830)).
논문은 external memory 기능을 사용하지 않은 점을 한계이자 미래 연구로 명시했고 큰 자료도 학습 행을 10,000개로 잘라 평가했다.
따라서 BiSHop은 Apache-2.0 구현이 있는 유효한 별도 아키텍처지만 ([공식 구현](https://github.com/MAGICS-LAB/BiSHop), [라이선스](https://github.com/MAGICS-LAB/BiSHop/blob/main/LICENSE)), 검색·메모리·행 관계 후보로 세면 이름 때문에 범위를 잘못 넓히게 된다.
특성 상호작용 후보로서의 가치는 [특성 상호작용·조건부 계산 신경망 계열의 실행 후보 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/136)에 맡긴다.

## 공통 누출 방지와 비교 산출물

모든 검색 후보의 fold별 후보 저장소는 outer 학습 fold 행과 라벨만 포함해야 하고, 검증 행과 테스트 행은 query로만 들어가야 한다.
학습 손실에서 query가 학습 행이면 자기 자신을 후보에서 제거해야 한다.
수치 표준화, 분위 변환, 범주 어휘, 결측 대체, prototype, 검색 index와 metric network는 모두 outer 학습 fold 안에서만 적합해야 한다.
각 seed 42 실행은 OOF AUC 외에 최근접 기존 구성원, champion·TabM·TabPFN-3과의 스피어만 상관, 표준 순위 평균 포함 전후 AUC, fold별 실행 시간, peak memory와 query당 후보 수를 남겨야 한다.
검색 문맥의 exact-match 비율, 라벨 entropy와 fold별 이웃 거리 분포를 함께 남기면 현재 champion의 정확값 기억을 되풀이했는지 판별할 수 있다.
후보가 새 피처를 추가하지 않으면 단일 model 교체 판정에서 새 피처 gate는 묻지 않지만, 공통 placebo importance는 저장소 계약대로 계산한다.
전체 검증 fold를 피처 수만큼 다시 검색하면 importance 비용이 학습보다 커질 수 있으므로, adapter가 seed로 고정한 검증 행 8,192개와 반복 1회를 정밀도 설정으로 소유하고 모든 피처와 placebo에 같은 permutation importance 절차를 적용한다.
이 제한된 importance의 예상 시간도 20 GPU 시간 실행 게이트에 포함하고, 표본 수가 너무 작아 placebo 비교가 불안정하면 재실행보다 후보를 판정 불가로 보수적으로 처리한다.

## 최종 권고

먼저 TabR-S의 fold 0 비용·누출 진단을 하고 통과하면 seed 42 전체 OOF를 실행한다.
그 다음 ModernNCA의 exact 전체 후보 추론 비용을 한 fold에서 재고 20 GPU 시간 예상 한도 안에서만 seed 42 전체 OOF를 실행한다.
두 후보 모두 단일 champion 교체와 다양성 구성원 판정을 같은 OOF로 각각 수행한다.
PTaRL이나 ModernNCA 다중 구성 앙상블은 이 두 후보가 모두 실패하거나 단일 ModernNCA가 강하게 통과하기 전에는 열지 않는다.
Hopular, NPT, PET와 TabGSL은 현재 지도에서 다시 열지 않는다.

## 한계

이 문서는 모델을 실제로 학습하지 않았으므로 16GB에서 24GB GPU의 시간과 peak memory는 사전 진단에서 확정해야 한다.
논문의 평균 benchmark 순위는 이 과제의 AUC를 보장하지 않으며 최종 채택 근거는 오직 저장소의 공통 5-fold OOF다.
TabR와 ModernNCA 모두 학습 행의 라벨을 예측 때 참조하므로 split 경계와 후보 ID 감사를 일반 한 행 model보다 엄격하게 적용해야 한다.
