# S6E8 남은 실험 공간 전수 재점검

## 결론

2026-08-14 JST에 이슈 [지도: 이슈 37 이후 최종 제출까지의 실험 프로그램](https://github.com/tmheo/predicting-smartphone-addiction/issues/44)의 닫힌 결정, 열린 자식 이슈, 현재 `main` 구현, Kaggle S6E8 공개 코드와 디스커션을 다시 대조했다.
새 모델 계열을 더 여는 것보다 현재 최고 모델과 이미 채택된 특성 사이의 조합을 확인하는 편이 기대 정보와 근거가 모두 강하다.
추가할 가치가 있는 독립 실험 축은 네 개이고, 조건부 후속 축은 한 개이며, 열린 이슈에 흡수할 변형과 후보 풀 진입 전에 보강할 검사가 각각 한 개다.
그 밖의 자료, 원본-합성 변환, 결측, 목적 함수, 표본 가중, 보정, 앙상블, 전체 자료 재학습과 의사 라벨링 후보는 이미 열려 있거나 반증되었거나 근거가 약하다.

우선순위는 다음과 같다.

| 순위 | 결정 | 처리 제안 |
| ---: | --- | --- |
| 1 | 현재 최고 Lookup-Transformer에 채택된 결측 복원 특성 블록을 이식한다 | 새 task 티켓 |
| 2 | CatBoost에 수치 정확값 범주 복제를 병행한다 | 새 task 티켓 |
| 3 | Lookup-Transformer 용량과 규제를 제한된 설정으로 재검증한다 | 새 task 티켓 |
| 4 | LightGBM의 열별 구간 해상도를 값 종류 수에 맞춰 분리한다 | 닫힌 구간 수 티켓의 새 근거를 잇는 task 티켓 |
| 5 | CatBoost 규제와 표본추출을 제한적으로 재검증한다 | 앞선 CatBoost 변형 통과 때만 별도 task 검토 |
| 6 | 순위와 잘린 로짓을 함께 쓰는 2단 입력 표현을 비교한다 | 열린 [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)에 흡수 |
| 필수 검사 | 무정보 예측과 중복 예측으로 후보 풀 기여의 영점 대역을 측정한다 | 열린 [OOF 후보 풀의 품질과 다양성 진입 기준 점검](https://github.com/tmheo/predicting-smartphone-addiction/issues/63)에 흡수 |

## 범위와 방법

저장소에서는 지도 본문, 자식 이슈 전체, `CONTEXT.md`, ADR 0001, 연구 문서, 현재 설정과 모델 및 결합기 구현을 읽었다.
Kaggle에서는 공식 API로 2026-08-14 현재 S6E8 코드 탭의 최신 실행 200개와 Public Score 정렬 상단 200개를 다시 열거했다.
2026-08-10 장부 이후 새 디스커션 8개와 갱신된 재방문 글을 Kaggle 공식 API의 원문과 코멘트까지 읽었다.
우선순위가 높은 최신 공개 노트북은 Kaggle CLI로 최신 공개 소스를 내려받아 Markdown, 코드 셀과 저장 출력을 직접 확인했다.
LightGBM과 반복 교차검증의 동작은 각각 공식 문서와 공식 소스 코드로 확인했다.

자료 시점 때문에 대회 종료 후 상위권 해법은 아직 존재하지 않는다.
공개 점수만 있는 주장, 저장 출력이나 제거 실험이 없는 주장, 외부 제출 파일을 섞은 결과는 채택 근거에서 제외했다.
아래에서 `근거`는 원문이나 저장소에서 직접 확인한 사실이고, `추론`은 그 사실을 우리 고정 fold와 파이프라인에 옮긴 판단이다.

## 1순위: 현재 최고 모델에 채택된 결측 복원 특성 이식

### 질문

현재 최고인 Lookup-Transformer에 제약 결측 재구성, XGBoost 조건부 결측 복원, 복원 행렬 조성 특성을 추가하면 단독 OOF와 후보 풀 기여가 개선되는가?

### 근거

- 현재 최고 실행 `exp059_lookup_transformer`는 원시 12열과 화면 시간 파생 6열만 읽으며, 정확값 TE, 제약 결측 재구성, XGBoost 복원 보조 열과 복원 행렬 조성 열은 읽지 않는다 ([설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp059_lookup_transformer.yaml)).
- [Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58)은 공개 레시피 재현과 원본 최근접 이웃 열 하나만 비교했고, 이후 CatBoost 쪽에서 채택된 복원 특성 블록은 비교하지 않았다.
- 제약 결측 재구성은 우리 CatBoost에서 3시드 OOF를 약 `+0.00015` 개선했고 이득이 결측 행에 집중됐다 ([제약 기반 결측 재구성 표현의 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/74)).
- XGBoost 조건부 결측 복원 보조 열은 우리 CatBoost에서 약 `+0.00015`를 더했고, 채택한 다섯 복원 열은 대리 검사와 공식 검사 방향이 같았다 ([XGBoost 조건부 결측 복원 보조 열의 한계 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/86)).
- 복원 행렬 조성 다섯 열은 다시 약 `+0.00019`를 더해 당시 최고 모델을 갱신했다 ([복원 행렬 기반 비율·차이 피처의 한계 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/90)).
- 공개 [TabM with constrained imputation](https://www.kaggle.com/code/szymonkapiski/s6e8-tabm-with-constrained-imputation)은 신경망에 원시 열과 제약 재구성 보조 열을 함께 주고, 결측 복원 MAE와 OOF 개선이 모두 나아졌다고 보고한다.
- 현재 Lookup-Transformer는 결측값을 별도 토큰으로 표현하고 값 가리기 증강을 하지만, 빠진 값의 조건부 추정치나 산술 실현 가능 구간을 입력으로 받지는 않는다 ([구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/lookup_transformer.py)).

### 추론

정확값 TE는 Lookup 임베딩이 이미 학습하는 값별 목표값 표면과 중복될 가능성이 높다.
반면 제약 재구성과 조건부 복원은 관측되지 않은 값을 다른 열에서 추정해 주므로, 별도 결측 토큰만으로는 직접 주어지지 않는 정보다.
따라서 TE와 복원 특성을 한꺼번에 넣으면 실패 원인을 알 수 없고, 복원 블록을 먼저 재는 편이 낫다.

### 최소 실험 설계

1. `exp059`와 같은 커밋, fold와 seed 42로 `복원 블록`만 추가한다.
2. 복원 블록은 이미 채택된 제약 재구성 열, XGBoost 복원 열 다섯 개, 조성 열 다섯 개로 제한하고 새 파생 열을 만들지 않는다.
3. 복원 블록이 seed 42에서 기준 이상이고 추가 열이 모두 플라시보 permutation importance를 넘을 때만 3시드 확정 재검증으로 간다.
4. 정확값 TE는 별도 실행으로 재서 Lookup 임베딩과의 중복 여부를 분리한다.
5. 둘 중 하나가 단독으로 통과할 때만 결합 실행을 연다.

누출 위험은 기존 fold-fit 제공자를 그대로 쓰면 새로 생기지 않는다.
계산 가능성은 현재 Kaggle T4 두 장의 시드 병렬 경로가 이미 검증되어 있어 높다.

## 2순위: CatBoost 수치 정확값의 native 범주 복제

### 질문

현재 CatBoost 구성에 수치 원본을 그대로 유지하면서 같은 값을 손실 없는 범주 키로 복제해 주면, 수동 정확값 TE와 다른 ordered CTR 신호가 단독 OOF와 후보 풀 기여를 개선하는가?

### 근거

- 현재 CatBoost 실행은 수치 원본과 fold-fit 정확값 TE를 쓰지만, 수치 정확값을 CatBoost native categorical 열로 병행하지 않는다 ([채택 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp057_cat_xgb_impute_comps5.yaml)).
- 공개 [Value-Lookup CatBoost](https://www.kaggle.com/code/funnybishop/s6e8-value-lookup-catboost)는 수치 아홉 열을 원본 수치와 정확값 문자열 범주로 함께 주고, 같은 CatBoost 대비 OOF가 0.96690에서 0.96839로 올랐다고 보고한다.
- 이 공개 노트북의 최신 판본에는 완전한 학습 출력이 없어 위 개선 폭은 저자 보고치다.
- 공개 [fixed-schedule exact-value CatBoost artifacts](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts)는 원시 12열과 `float64` 왕복 가능한 정확값 키 아홉 열로 학습한 5-fold OOF, fold ID, fold별 점수와 해시를 제공한다.
- 이 산출물의 fold ID에서 문서와 배열의 1-based 차이를 바로잡으면 우리 `artifacts/folds.parquet`과 행별로 일치한다.
- 외부 OOF를 읽기 전용으로 현재 12구성원 풀과 대조한 결과, 단독 AUC는 0.96729798, 최근접은 `exp057` CatBoost와 Spearman 0.992204였고, 표준 순위 평균은 0.96838565에서 0.96845097로 약 `+0.0000653` 올랐다.
- 외부 예측은 채택 근거가 아니지만, 지도 규칙이 허용한 진입 진단으로는 현재 풀에 독립 잔차가 남았다는 우선순위 근거다.
- CatBoost 공식 문서는 범주 열에서 여러 종류의 CTR을 계산하며, 수치 열은 사전에 구간화한다고 설명한다 ([범주 특성](https://catboost.ai/docs/en/features/categorical-features), [CTR 설정](https://catboost.ai/docs/en/references/training-parameters/ctr)).
- CatBoost 원 논문은 순서 있는 목표 통계가 일반 목표 통계의 목표값 누출을 줄이는 핵심 절차라고 설명한다 ([CatBoost 논문](https://arxiv.org/abs/1706.09516)).

### 추론

수동 fold-fit TE는 각 값의 평균 목표값을 하나의 수치로 주고, native CTR은 학습 순서에 따른 통계와 CatBoost 내부 범주 처리를 사용하므로 완전히 같은 함수 표현이 아니다.
최근접 상관 0.992204와 외부 풀 기여 양수는 새 모델 계열을 만들지 않고도 CatBoost 내부 표현을 다양화할 가능성을 뒷받침한다.

### 최소 실험 설계

1. 현재 채택 CatBoost 피처 계획과 seed 42를 고정한다.
2. 수치 아홉 열을 `float.hex`처럼 `float64` 왕복이 가능한 문자열 키로 복제해 CatBoost 범주 열로 추가한다.
3. 첫 실행은 기존 수동 TE를 유지한 채 범주 복제만 더하고 `max_ctr_complexity=1`로 고정한다.
4. 추가 열의 permutation importance가 플라시보를 넘지 못하면 수동 TE를 native 정확값 범주로 교체한 변형을 같은 트랙에서 한 번만 비교한다.
5. 단일 범주 CTR이 살아남을 때만 `max_ctr_complexity=2`를 열며, 그렇지 않으면 범주 조합을 확장하지 않는다.
6. 통과 시 3시드 확정, 최근접 상관과 표준 평가 앙상블 기여를 다시 측정한다.

정확값 범주 복제는 타깃을 읽지 않으므로 제공자 자체의 누출 위험은 낮고, CatBoost 목표 통계는 outer fold 학습 부분 안에서만 적합해야 한다.

## 3순위: Lookup-Transformer의 제한적 용량과 규제 재검증

### 질문

공개 레시피 한 점만 실행한 현재 최고 모델에서 작은 용량과 규제 변화가 같은 기능 계열 안의 성능을 더 올리는가?

### 근거

- `exp059`는 공개 레시피의 `d_model=128`, 4층, 8헤드, dropout 0.1, 값 가리기 0.1, 32 epoch, EMA 0.999를 그대로 쓴다 ([설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp059_lookup_transformer.yaml), [기본값 구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/lookup_transformer.py)).
- [Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58)은 이 한 점과 원본 최근접 이웃 특성 변형만 비교했다.
- 공개 [mix the meta-models, then fix the weak bands](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)는 Lookup-Transformer의 readout, 깊이와 폭을 바꿔 재학습해도 기존 Lookup과 순위 상관 0.9926인 같은 계열 예측이 됐다고 보고한다.
- 같은 공개 노트북의 결과는 아키텍처 변경이 새 다양성 축이라는 주장을 약화하지만, 동일 계열 최고 점수를 찾는 작은 검사는 반증하지 않는다.
- 최신 [model capacity 실험](https://www.kaggle.com/code/wowtimwow/model-capacity-beat-my-feature-engineering-by-18x)은 약한 LightGBM 기준에서 용량 하나가 파생 특성 15개보다 훨씬 큰 차이를 냈고, 용량 변경 뒤 특성 제거 실험을 다시 해야 한다는 재현 가능한 경고를 제공한다.

### 추론

현재 최고 모델을 공개 설정 한 점에서 멈춘 것은 지역적 탐색 공백이다.
다만 공개 재학습이 같은 예측 계열로 수렴했으므로 대규모 구조 탐색이나 새 다양성 구성원 티켓은 정당화되지 않는다.

### 최소 실험 설계

1. `exp059`를 기준으로 seed 42 전체 5-fold를 유지한다.
2. 용량 후보는 기준, 폭 증가 한 개, 깊이 증가 한 개로 제한한다.
3. 규제 후보는 값 가리기와 임베딩 weight decay를 각각 한 단계만 높이거나 낮춘 두 설정으로 제한한다.
4. 한 번에 한 축만 바꾸고, 용량과 규제의 우승 설정이 각각 기준을 넘을 때만 둘을 결합한다.
5. ADR 0001의 단일 모델 문턱과 3시드 확정 재검증을 그대로 적용한다.
6. 기존 `exp059`와 상관이 0.998 이상이면서 단독 OOF가 낮은 변형은 후보 풀에도 넣지 않는다.

이 검사는 새 구현이 아니라 이미 설정으로 노출된 값을 바꾸는 작업이라 재현성과 계산 가능성이 높다.

## 4순위: LightGBM 열별 구간 해상도

### 질문

값 종류가 많은 원시 화면 시간 두 열에만 더 촘촘한 구간을 주고 나머지 열은 1023 이하로 유지하면, 전역 1439의 하락 없이 값별 신호를 더 보존할 수 있는가?

### 근거

- [LightGBM 구간 수 확대의 성능·다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/100)은 현재 특성 계획에서 전역 `max_bin=1023`이 255보다 약 `+0.000052`였지만 전역 1439는 255보다 약 `-0.000008`이었다고 확정했다.
- 같은 결정은 전역 1439의 계산 비용이 감당 가능하다는 이유로 `max_bin_by_feature` 축소안을 열지 않았다.
- 그 결정 뒤 최신 실행된 공개 [max_bin: raise it to your distinct value count](https://www.kaggle.com/code/kitopl/max-bin-raise-it-to-your-distinct-value-count)는 원시 12열에서 255 이상의 이득 중 91%가 값 종류 수 1389와 1437인 `daily_screen_time_hours`, `weekend_screen_time` 두 열에서 나온다고 보고한다.
- 같은 노트북은 값 종류가 255 이하인 열에 2047개 구간을 줘도 AUC와 나무 수가 기준과 정확히 같고, 큰 두 열과 중간 네 열의 이득이 가산적이라는 대조 실험을 코드로 공개한다.
- LightGBM 공식 문서는 `max_bin_by_feature`가 열마다 최대 구간 수를 정하며, 지정하지 않으면 전 열이 `max_bin`을 쓴다고 명시한다 ([Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)).
- LightGBM 공식 소스는 열별 목록 길이를 전체 열 수와 맞추고 각 값을 자료 집합의 구간 설정으로 저장한다 ([dataset.cpp](https://github.com/lightgbm-org/LightGBM/blob/main/src/io/dataset.cpp)).

### 추론

우리 전역 1439의 하락과 공개 열별 분해는 모순되지 않는다.
필요한 두 원시 열만 1439로 올리고 TE, 복원과 파생 열을 1023 이하로 두면 전역 고해상도에서 생긴 불필요한 분할 자유도를 피할 수 있다.
이것은 이슈 100이 계산 비용 대안으로만 검토한 열별 설정과 다른 정확도 가설이며, 그 티켓이 닫힌 뒤 공개된 열별 대조가 새 근거다.

### 최소 실험 설계

1. 현재 대리 기준인 전역 1023을 같은 seed 42 비교 기준으로 쓴다.
2. `daily_screen_time_hours`와 `weekend_screen_time`만 1439, 나머지는 1023인 한 설정을 실행한다.
3. 두 열만 2047인 설정을 결과 동일성 대조로 한 번 실행한다.
4. 개선이 없으면 `min_data_in_bin`이나 다른 열 조합으로 확장하지 않고 닫는다.
5. 개선되면 3시드 확정 뒤 대리 기준 교체 여부만 판단하고, 기존 대리 결과는 소급 변경하지 않는다.

## 5순위 조건부 후보: CatBoost 규제와 표본추출

### 근거

- 현재 CatBoost 계열 비교는 depth 6과 8, 학습률 0.05와 0.03을 비교했으며 `l2_leaf_reg`, `random_strength`, `bagging_temperature`를 독립적으로 재검증하지 않았다 ([트리 모델 계열별 OOF 성능과 다양성 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/59)).
- 공개 [PS-S06E08: CatBoost](https://www.kaggle.com/code/stephentarter/ps-s06e08-catboost)은 5-fold 탐색 뒤 depth 7, 학습률 약 0.098, 별도 L2, 무작위 강도와 배깅 온도 설정이 기본보다 약 `+0.00363`이었다고 보고한다.
- 이 공개 결과의 절대 OOF는 0.964273으로 우리 CatBoost보다 낮고, 본문 L2 탐색 범위와 최종 보고값도 서로 맞지 않아 직접 근거는 약하다.
- 공개 [CatBoost ordered vs plain](https://www.kaggle.com/code/dariushafshar/s6e8-catboost-ordered-vs-plain)은 같은 5-fold에서 ordered boosting이 plain보다 약 `-0.000155`이고 4배에서 5배 느렸다고 보고하므로 boosting 방식 변경은 후보가 아니다.

### 추론과 처리

규제와 표본추출은 저장소에 남은 실제 공백이지만, 공개 근거의 품질은 별도 티켓을 지금 열 정도로 강하지 않다.
2순위 정확값 범주 복제가 통과해 CatBoost가 다시 핵심 후보가 될 때만, 그 우승 피처를 고정하고 기본 설정과 사전 고정한 규제 묶음 한 개를 seed 42에서 짝비교한다.
대규모 Optuna 탐색, 단일 split 탐색과 ordered boosting은 열지 않는다.

## 열린 결합 티켓에 흡수할 변형: 순위와 잘린 로짓의 이중 표현

### 근거

- 현재 [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)는 균등 및 가중 순위 평균, 로짓 로지스틱 회귀와 Ridge를 비교하지만, 한 구성원의 순위와 로짓을 동시에 입력하는 설계는 명시하지 않는다.
- 현재 구현된 결합기 목록도 `rank_mean`과 `ridge_logit`뿐이다 ([ensemble.py](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)).
- 공개 [The strongest fully-reproducible stack](https://www.kaggle.com/code/dariushafshar/the-strongest-fully-reproducible-stack-lb-0-9708)은 구성원마다 백분위 순위와 잘린 로짓을 함께 주는 2단 입력이 둘 중 하나만 쓰는 입력보다 낫다고 nested 계산으로 보고한다.
- 공개 [mix the meta-models, then fix the weak bands](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)는 서로 다른 편향을 가진 두 선형 2단 설계의 순위 혼합이 3개 meta 분할의 15개 fold 모두에서 같은 방향이었다고 보고한다.

### 추론과 처리

이중 표현은 새 앙상블 단계가 아니라 이슈 64의 입력 표현 한 줄을 늘리는 변형이다.
각 outer 학습 부분에서만 로짓 자르기, 표준화와 계수 학습을 수행하고 outer 평가 부분에는 적용만 해야 한다.
평균 개선이 `+0.0001` 미만이면 ADR 0001에 따라 더 단순한 결합기로 돌아가므로, 탐색 자유도는 제한된다.
결측 구간별 2단 설계와 두 설계의 혼합은 이미 열린 [비선형·구간별 2단 결합의 추가 가치 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)의 범위이므로 새 티켓을 만들 필요가 없다.

## 후보 풀 티켓에 흡수할 필수 검사: 무정보 대조

### 근거

- 현재 [OOF 후보 풀의 품질과 다양성 진입 기준 점검](https://github.com/tmheo/predicting-smartphone-addiction/issues/63)은 계보, 정렬, fold, 정밀도와 중복 후보 제거를 다루지만 영점 측정 대조를 명시하지 않는다.
- 공개 [Price a new stack member vs 74 OOFs](https://www.kaggle.com/code/dariushafshar/price-a-new-stack-member-vs-74-oofs-lb-0-97077)는 순수 난수 예측을 넣었을 때 `+0.000001`, 기존 최강 구성원을 복제했을 때 `-0.00004`를 측정해, 그 대역 안의 기여는 새 정보라는 근거가 아니라고 판정했다.
- 공개 [12 OOF Members Survive a Fold and Hash Audit](https://www.kaggle.com/code/dariushafshar/12-oof-members-survive-a-fold-and-hash-audit)는 파일 이름이 달라도 바이트가 같은 예측 두 쌍을 찾아 14개 파일을 12개 고유 구성원으로 줄였다.
- 같은 감사 노트북은 fold ID와 AUC가 맞아도 상위 학습이 검증 라벨을 보지 않았다는 사실까지 증명하지는 못한다고 명시한다.

### 처리 제안

1. 모든 OOF와 테스트 예측의 배열 해시를 기록해 정확 중복을 먼저 제거한다.
2. 고정 seed의 독립 난수 순위 열 여러 개를 표준 평가 앙상블에 넣어 기여 변화의 영점 분포를 만든다.
3. 최고 구성원의 정확 복제와 각 구성원의 복제를 넣어 단순 재가중이 만드는 변화 대역을 잰다.
4. 실제 후보의 기여가 두 영점 대조의 상단을 넘는지 함께 기록한다.
5. 영점 대역 안의 양수 기여는 정보가 있는 구성원으로 해석하지 않고 이슈 63에서 제거 후보로 표시한다.

이 검사는 모델을 다시 학습하지 않으며 후보 풀 확정 전에 한 번만 수행하면 된다.
후보 풀 규약의 `기여 > 0`을 바꾸는 판정으로 쓸 경우에는 ADR 0001에 근거를 함께 남겨야 한다.

## 영역별 전수 판정

### 자료와 검증

- 고정 Stratified 5-fold는 ID 위치 신호가 없다는 [ID 구간 진단](https://github.com/tmheo/predicting-smartphone-addiction/issues/55)과 train/test 값 분포 이동이 결측률 차이뿐이라는 기존 진단 때문에 유지한다.
- 최신 [Private-LB-robust validation](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734628)은 grouped, ID-block, 결측 층화와 적대적 검증을 질문하지만 S6E8 측정 결과를 제시하지 않는다.
- meta 결합의 아주 작은 차이는 고정 nested OOF가 정식 판정이고, 추가 분할 반복은 민감도 검사로만 쓸 수 있다.
- scikit-learn 공식 문서는 `RepeatedStratifiedKFold`가 서로 다른 무작위화로 Stratified K-Fold를 반복한다고 정의한다 ([문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)).
- 후보 풀과 결합에서 값이 비슷할 때 반복 meta 분할을 보조 표로 남길 수 있지만, 새 모델의 공통 고정 fold OOF를 대신하지 않는다.

### 특성

- 다항식, 임계값, 일반 비율과 상호작용은 최신 [Non-ratio Feature Engineered columns](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735000)에서도 제거 실험 없이 public 0.96646만 보고되어 기존 배제 결정을 뒤집지 못한다.
- 최신 [CatBoost Feature Ablation](https://www.kaggle.com/code/vladstud716373618/s6e8-catboost-feature-ablation-600-11-best)은 600개 이상 후보에서 삼각함수와 두 열의 TE만 남겼지만 최종 OOF가 약 0.962이고, 불규칙 값 신호는 이미 정확값 TE와 Lookup 임베딩이 더 직접 표현한다.
- 최신 [rank transformation and missing indicators](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734636)은 제거 실험 수치를 제시하지 않으며, 결측 표시가 목표값 신호 없이 train/test 출처만 구분한다는 다중 통제 결과와 충돌한다.
- 수치 순위 변환은 나무가 쓰는 순서를 바꾸지 않고 구간화만 바꿀 수 있으므로, LightGBM에서는 위 열별 구간 해상도 검사가 더 직접적이다.
- 중간 화면 시간 구간에서 세 범주 열을 다시 쓰자는 가설은 최신 [model capacity 디스커션](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734990)에서 in-sample TE AUC조차 0.5040에서 0.5132에 그쳐 기각됐다.

### 원본-합성 변환

- 최신 [The Generator Didn't Just Smooth the Labels](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734501)은 원본의 산술 위반이 합성 자료에서 사라졌다는 기존 예산 제약 진단을 재확인할 뿐 새 제약을 제시하지 않는다.
- 원본 프록시 행 주입, 원본 목표값 prior, 최근접 이웃, CDF 차이, KDE 밀도비와 후보 생성기 재현은 각각 기존 티켓에서 채택 또는 기각이 끝났다.
- 원본 행 가중과 adversarial reweighting은 자체 및 공개 실험에서 손해였고 train/test 값 분포 이동도 확인되지 않아 다시 열지 않는다.

### 결측

- 원시 NaN 유지, 결측 표시 배제, 제약 재구성과 XGBoost 조건부 복원이라는 현재 경계는 유지한다.
- 고결측 구간 전용 FM과 결측별 isotonic 보정은 공개 [mix the meta-models, then fix the weak bands](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)에서 각각 `-0.000091`, `-0.000080`으로 실패했다.
- 결측 구간별 2단 가중만 열린 이슈 67에서 nested 방식으로 다루면 충분하다.

### 모델 계열

- RealMLP와 TabM은 열린 [RealMLP·TabM의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/61)이 진행 중이다.
- TabPFN-3은 열린 [TabPFN-3의 스모크 게이트 통과 시 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/102)이 이미 조건부 진입을 다룬다.
- Factorization Machine은 공개 S6E8에서 전역 단독 OOF가 약 0.9666에서 0.9674이고 기존 74구성원 묶음 기여가 약 `+0.000006`에 그쳤으며, 이미 채택한 Lookup 계열이 더 강하고 더 비상관이었다 ([원문](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)).
- 화면 시간 구간 전용 FM은 작은 양수 결과가 있지만 이슈 67이 이미 약한 구간 전용 FM 보정을 명시한다.
- 공개 `LGB + DART Tuner` 노트북의 실제 공개 코드에는 `boosting_type="dart"`가 없고 표준 GBDT를 학습하므로 DART 근거로 쓸 수 없다 ([원문](https://www.kaggle.com/code/mohankrishnathalla/s6e8-lgb-dart-tuner-oof-saver)).
- TabNet, FT-Transformer, 범용 ResNet 및 약한 MLP는 외부 OOF 진입 진단에서 단독 성능과 한계 기여가 모두 낮아 지도에서 이미 제외됐다.
- 따라서 현 시점에 새 티켓을 정당화하는 새 모델 계열은 없다.

### 목적 함수와 표본 가중

- 학급 불균형 재가중, 재표본화와 SMOTE는 기존 공개 통제와 지도 범위에서 이미 제외됐다.
- Factorization Machine에 pairwise AUC surrogate를 더한 공개 변형은 BCE 대응 모델과 순위 상관 0.9965로, 새 다양성 축이 아니었다 ([원문](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands)).
- focal loss, hard-example 가중과 불확실성 가중은 S6E8에서 누출 없는 양수 제거 실험이 없고, 구간 전용 학습과 2단 가중이 열린 이슈 67에서 더 직접적으로 다뤄진다.
- 교사 예측 증류는 최신 [Private-LB-robust validation](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734628)이 누출 위험과 nested teacher OOF 필요성을 제기했지만 측정 결과가 없다.
- 최종 목표가 압축 모델이 아니라 최고 앙상블 두 개이므로, 증류는 정보 손실 가능성과 추가 nested 비용을 감수할 독립 이득 근거가 없어 열지 않는다.

### 보정과 불확실성

- ROC AUC는 예측 순서로 평가되므로 한 모델에 같은 단조 보정을 적용해도 단독 AUC는 바뀌지 않는다.
- 모델 사이 눈금 차이는 순위 평균, 로짓 입력과 위 이중 표현이 이미 다룬다.
- 결측 구간별 isotonic 보정은 15개 fold 모두에서 실패했고, 구간별 목표값 비율도 거의 같아 새 보정 티켓을 열 근거가 없다.
- public 점수 차이의 불확실성은 이미 마일스톤과 최종 후보 선택 이슈가 다루며, 구성원 선택이나 가중치 학습에는 쓰지 않는다.

### 앙상블

- 균등 및 가중 순위 평균, nested 로지스틱과 Ridge, 탐욕 및 배깅 탐욕, Optuna 부분집합, 비선형 및 구간별 2단 결합은 열린 이슈 64, 62, 67이 모두 다룬다.
- 최신 94구성원 공개 묶음은 새 12구성원이 5개 fold 모두에서 약 `+0.00010`을 더했다고 보고하지만 외부 예측은 지도 규칙상 읽기 전용 진입 진단 외에는 쓸 수 없다 ([94 Verified OOFs](https://www.kaggle.com/code/dariushafshar/94-verified-oofs-honest-cv-0-96985-lb-0-97097)).
- public 0.97099 제출들은 서로 다른 외부 제출 파일을 public 점수에 맞춰 순위 평균한 결과라 자체 OOF 실험 후보가 아니다.
- 새로 필요한 것은 이중 표현 한 변형과 무정보 대조뿐이다.

### 전체 자료 재학습과 의사 라벨링

- fold별 최적 반복 수 집계, 1.25배 규칙, 시드 평균, 전체 자료 학습본과 CV 테스트 예측 결합은 열린 [선택 모델의 시드 평균과 전체 자료 재학습 규약 확정](https://github.com/tmheo/predicting-smartphone-addiction/issues/66)이 모두 다룬다.
- 높은 확신 표본의 소규모 의사 라벨링과 즉시 중단 조건은 열린 [의사 라벨링의 마지막 단계 진입 여부 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/68)이 다룬다.
- 공개 해법에서 원본 추가와 의사 라벨링이 실패했다는 기존 결과를 뒤집는 새 S6E8 통제 결과는 찾지 못했다.
- 외부 제출이나 공개 2단 예측을 교사 라벨로 쓰는 방식은 외부 예측 배제 규칙과 누출 위험 때문에 열지 않는다.

## 이슈 102 의존 관계 확인

[TabPFN-3의 스모크 게이트 통과 시 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/102)은 현재 GitHub 의존 관계에서 `blocked_by=0`, `blocking=1`로 조회된다.
따라서 이슈 102가 후속 후보 풀 확정을 막는 방향의 연결은 반영되었고, 이 조사에서 추가 의존 관계 수정은 필요하지 않다.

## 최종 결정표

| 후보 | 기존 중복 | 누출 위험 | 예상 독립 정보 | 계산 가능성 | 판정 |
| --- | --- | --- | --- | --- | --- |
| Lookup에 복원 특성 이식 | 미검증 조합 | 낮음, 기존 fold-fit 제공자 | 높음, 빠진 값의 조건부 추정 | 높음 | 새 티켓 권장 |
| CatBoost 정확값 범주 복제 | 수동 TE와 일부 중복 | 낮음, CatBoost CTR은 fold 안 적합 | 중간 이상, 외부 진입 진단 양수 | 높음 | 새 티켓 권장 |
| Lookup 제한 설정 검사 | 공개 설정 한 점만 실행 | 낮음 | 중간, 같은 계열 성능 개선 | 높음 | 새 티켓 권장 |
| LightGBM 열별 구간 수 | 이슈 100은 비용 대안만 종결 | 낮음 | 중간, 두 원시 열의 값별 신호 | 높음 | 새 티켓 권장 |
| CatBoost 규제·표본추출 | depth와 학습률만 비교 | 낮음 | 불명, 공개 근거 약함 | 높음 | CatBoost 새 변형 통과 때만 조건부 |
| 순위와 로짓 이중 입력 | 이슈 64에 미명시 | 낮음, nested면 안전 | 중간 이하 | 매우 높음 | 이슈 64에 흡수 |
| 무정보 및 중복 대조 | 이슈 63에 미명시 | 없음 | 성능이 아니라 측정 신뢰도 | 매우 높음 | 이슈 63에 흡수 |
| 새 FM 계열 | Lookup이 지배 | 낮음 | 매우 낮음 | 높음 | 기각 |
| DART | 공개 코드가 실제 DART가 아님 | 낮음 | 근거 없음 | 높음 | 기각 |
| 증류와 새 손실 | 측정 근거 없음, 열린 구간·의사 라벨과 겹침 | 높음 | 불명 | 낮음 | 기각 |
| 새 결측 표시와 보정 | 기존 반증과 충돌 | 낮음 | 없음 | 높음 | 기각 |
| 원본 재가중과 재생성 | 기존 반증과 충돌 | 중간 | 없음 | 낮음 | 기각 |

## 한계

이 문서는 공개 코드와 현재 저장소를 근거로 실험 진입 여부를 판단했으며 새 후보를 실제로 학습하지 않았다.
Kaggle 공개 노트북의 정밀 수치 가운데 저장 출력이 없는 값은 작성자 보고치이며, 우리 fold의 채택 근거는 ADR 0001에 따른 자체 실행만 될 수 있다.
대회가 진행 중이므로 2026-08-14 이후 새 코드와 디스커션은 증분 업데이트 절차로 다시 확인해야 한다.
