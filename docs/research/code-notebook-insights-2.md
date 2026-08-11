# Playground Series S6E8 코드 노트북 추가 조사: Public Score 정렬 상단

## 조사 범위와 판정 기준

이 문서는 2026-08-11 KST 기준 대회 Code 탭을 Public Score 내림차순으로 정렬했을 때 상단에 오는 노트북 가운데,
[선행 조사](code-notebook-insights.md)의 분석 대상 37개에 포함되지 않은 노트북을 종합한다.
목록은 공식 Kaggle CLI의 `scoreDescending` 정렬로 확정했고, 각 노트북의 최신 공개본을 CLI로 내려받아 모든 코드 셀을 확인했다.

다음 두 부류는 실질 내용이 없다고 판정해 개별 분석에서 제외했다.

- 외부 제출 파일이나 공개 OOF 예측을 단순 결합만 하는 노트북.
- 기존 분석 대상 노트북을 거의 그대로 복사한 사본.

근거 판정 기준은 선행 조사와 같다.
저장된 실행 출력이 없는 본문 수치는 작성자 보고 수치로 구분했다.

## 요약

| 노트북 | 작성자 | 중심 내용 | 근거 판정 |
| --- | --- | --- | --- |
| [S6E8 full lattice target encoding with XGBoost](https://www.kaggle.com/code/szymonkapiski/s6e8-full-lattice-target-encoding-with-xgboost) | `szymonkapiski` | 전체 36쌍 격자 TE와 개수 열, 단일 XGBoost OOF 0.96780 | 강함 |
| [S6E8 TabM with constrained imputation](https://www.kaggle.com/code/szymonkapiski/s6e8-tabm-with-constrained-imputation) | `szymonkapiski` | 제약 결측 재구성과 fold 내 3시드 평균, 단일 TabM OOF 0.96867 | 강함 |
| [S6E8 Smartphone Addiction full solution](https://www.kaggle.com/code/szymonkapiski/s6e8-smartphone-addiction-full-solution) | `szymonkapiski` | 25구성원 OOF 라이브러리와 결측 층별 오류 프로파일 | 강함 |
| [Everything above 0.970 is inside the noise floor](https://www.kaggle.com/code/dariushafshar/everything-above-0-970-is-inside-the-noise-floor) | `dariushafshar` | Public LB 노이즈 플로어 약 0.00015 산정과 제출 선택 규칙 | 강함 |
| [Price a new stack member vs 74 OOFs](https://www.kaggle.com/code/dariushafshar/price-a-new-stack-member-vs-74-oofs-lb-0-97077) | `dariushafshar` | 플라시보 대조 기반 앙상블 구성원 한계 기여 측정 | 강함 |
| [The strongest fully-reproducible stack](https://www.kaggle.com/code/dariushafshar/the-strongest-fully-reproducible-stack-lb-0-9708) | `dariushafshar` | rank와 logit 이중 표현, 고정 1/3 결측 regime 혼합 | 강함 |
| [Feature Engineering: What Worked and What Didn't](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t) | `kodaifukuda0311` | 특성 계열별 채택 기각 표와 원본 참조 분포 특성 | 보통, 표는 저장 수치 |
| [S6E8 XGB: The Power of Exact-Value TE](https://www.kaggle.com/code/kodaifukuda0311/s6e8-xgb-the-power-of-exact-value-te) | `kodaifukuda0311` | 12개 원본 열 전체의 정확값 TE와 5시드 CV | 보통 |
| [S6E8 Public LB 0.97009 Single Model RealMLP](https://www.kaggle.com/code/zhenruiweng/s6e8-public-lb-0-97009-single-model-realmlp) | `zhenruiweng` | PBLD 주기 임베딩 RealMLP 단일 모델 LB 0.97009 | 보통, 실행 출력 없음 |
| [RealMLP 0.97014](https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014) | `nawfeelrahman1124444` | 유사 RealMLP 변형에 파생 특성 추가 | 약함 |
| [Single LightGBM Target Encoding No Model Blend](https://www.kaggle.com/code/boltuzamaki/single-lightgbm-target-encoding-no-model-blend) | `boltuzamaki` | 전체 데이터 학습 단일 LightGBM, 결측 패턴 비트마스크 | 약함, 자기 행 포함 TE |
| [PS6E8 EDA + Feature Engineering Pipeline](https://www.kaggle.com/code/zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline) | `zhukovoleksiy` | NaN이 숨은 0이 아니라는 검정과 값별 타깃률 재현성 검사 | 보통 |

## 개별 분석

### S6E8 full lattice target encoding with XGBoost

[원문](https://www.kaggle.com/code/szymonkapiski/s6e8-full-lattice-target-encoding-with-xgboost)은 우리와 동일한 `StratifiedKFold(5, shuffle=True, random_state=42)` 분할에서 단일 XGBoost로 OOF 0.96780을 보고한다.
핵심 기제는 손으로 고른 4쌍이 아니라 **숫자 열 전체 36쌍의 2차원 격자 셀을 모두 TE**하고, 정수 내림에 더해 0.1 해상도 셀을 추가하며, 모든 TE 열 옆에 **셀 개수(CT) 열**을 함께 넣어 모델이 얇은 셀을 스스로 불신하게 만드는 것이다.
훈련 구간 TE는 내부 4-fold OOF로 만들어 자기 행이 자기 인코딩에 들어가지 않는다.
같은 변경이 LightGBM 0.96740에서 0.96763, XGBoost 0.96749에서 0.96780, CatBoost 0.96701에서 0.96718로 세 학습기 모두 같은 방향으로 움직였고, 0.1 해상도는 XGBoost에서만 +0.00011 유효했다.
화면 사용 블록의 3중 격자는 LightGBM에서 소폭(0.96768) 유효했다고 보고한다.
저장 예측을 로드해 점수를 재현하는 구조이고 훈련 코드가 동일 셀에 공개되어 있다.

### S6E8 TabM with constrained imputation

[원문](https://www.kaggle.com/code/szymonkapiski/s6e8-tabm-with-constrained-imputation)은 같은 5-fold에서 단일 TabM(pytabkit)으로 OOF 0.96867, 단일 모델 public LB 0.96967을 보고한다.
세 요소가 결합되어 있다.

1. **제약 결측 재구성**: 생성 규칙 `daily >= social + gaming + work`를 특성이 아니라 산술 제약으로 사용한다.
   daily가 결측이면 관측된 성분 합이 하한, 한 성분만 결측이면 `[0, slack]` 양측 경계가 되며, iterative imputer 추정치를 이 실현 가능 구간으로 잘라 넣고 구간 폭을 별도 열로 준다.
   가린 값 복원 MAE는 0.679로 무제약 imputer 0.691, 중앙값 대체 1.089보다 낫다.
   원시 열은 그대로 두고 재구성 열을 병행 추가한다.
2. **fold 내 시드 평균**: 아키텍처 5종의 편차가 0.00039에 불과해 구조 탐색은 소진 상태였고, fold 안에서 3시드 예측을 평균한 것이 +0.00019로 전 fold 승리했다.
   앙상블에 3개를 넣는 것이 아니라 fold를 떠나기 전에 평균해 잡음이 적은 모델 하나를 만드는 방식이다.
3. **신경망 전용 비율 특성**: 트리에서는 -0.0003으로 측정된 비율 및 합성 특성을 신경망에만 준다.

트리 모델과의 상관은 약 0.984로 유일하게 다른 오류 계열이지만, 본인 49구성원 blend에는 기존 단일 시드 TabM들과 상관 0.9985라 **blend 기여가 0.00000**이었다는 점도 명시한다.
단일 모델 0.96967 대비 49구성원 blend가 0.97059로, 전체 스택의 가치가 0.00092라는 보정 수치도 제공한다.

### S6E8 Smartphone Addiction full solution

[원문](https://www.kaggle.com/code/szymonkapiski/s6e8-smartphone-addiction-full-solution)은 blend OOF 0.96782, public LB 0.96900의 전체 해법을 25구성원 [OOF 라이브러리](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-25-models)로 공개한다.
라이브러리에는 행 순서 그대로의 OOF와 테스트 예측, manifest, 전체 hyperparameter, 훈련 소스가 포함되며 fold 스펙이 우리와 동일하다고 명시한다.
재사용 가치가 높은 측정 결과는 다음과 같다.

- 결측 개수 층별 AUC 프로파일: 0개 0.974에서 6개 이상 0.887까지 떨어지며, 행의 61%가 결측을 가지므로 남은 오류가 거의 전부 거기 있다.
- 제약 결측 재구성은 전체 +0.00041인데 이득이 결측 있는 층에서 2배에서 5배로 집중된다.
  기제가 작동했는지 층별로 확인하는 검증 형태 자체가 전이 가능하다.
- 부정 결과: 비율 및 합성 특성 -0.0003, 결측 지표 무효(MCAR), age 범주화 -0.00012.
- 튜닝 교훈: 단일 split 튜닝은 XGBoost와 LightGBM 모두에서 잘못된 구성을 골랐고, CatBoost 탐색 공간에 기본 설정을 빼먹어 튜닝본이 미튜닝본보다 나빠졌다.
  기본 설정을 trial 0으로 넣으라는 조언이 실용적이다.
- blend 기여는 단독 점수 순이 아니다: 상관 0.997인 hgb는 0.00000, 상관 0.932인 realmlp는 +0.00018을 더했다.

### dariushafshar의 앙상블 방법론 3부작

**[Everything above 0.970 is inside the noise floor](https://www.kaggle.com/code/dariushafshar/everything-above-0-970-is-inside-the-noise-floor)**는 Hanley-McNeil 표준오차로 public LB의 최소 판별 가능 차이를 계산하고, 근사 동일 제출 사이에서 측정한 노이즈 플로어 약 0.00015를 제시한다.
결론으로 나온 제출 선택 규칙: 최고 LB에서 0.00015 이내인 후보는 전부 동급으로 보고, 고정 fold nested CV, 시드 안정성, 단순성으로 결정한다.

**[Price a new stack member vs 74 OOFs](https://www.kaggle.com/code/dariushafshar/price-a-new-stack-member-vs-74-oofs-lb-0-97077)**는 새 앙상블 구성원의 가치를 74구성원 공개 라이브러리 위에서 fold별 한계 기여로 측정하되, **영정보 대조 2종**을 함께 돌린다.
순수 난수 열은 +0.000001, 기존 최강 구성원의 정확한 복제는 -0.00004를 만들었고, 이 대역 안의 측정치는 측정이 아니라는 기준을 세운다.
본인의 7개 모델(OOF 0.962에서 0.965)은 전부 합쳐 +0.000007로, 74구성원 스택이 포화 상태라는 진단이다.
자기 모델끼리만 비교하면 +0.0005로 보였다는 분모 선택 오류 고백도 유익하다.

**[The strongest fully-reproducible stack](https://www.kaggle.com/code/dariushafshar/the-strongest-fully-reproducible-stack-lb-0-9708)**은 노트북 안에서 nested CV를 실제로 재계산한다.
확인된 두 지렛대는 다음과 같다.

1. 구성원마다 정규화 순위와 잘린 logit **두 표현을 함께** meta 모델에 주면 어느 한쪽만보다 낫다.
2. 결측 regime 상호작용 스택은 단독으로는 불안정하지만, 사전에 고정한 1/3 가중치 순위 혼합으로는 전 fold에서 이겨 nested CV +0.00007을 만든다.

부정 결과 표도 값지다: GBM meta 학습기 -0.000116, logreg C 튜닝 무효, NNLS와 greedy 열세, 이 규모 풀에 단일 구성원 추가는 중앙값 +0.000003, 구간별 순위 보정은 fold 간 부호 불안정.
CV와 LB 정합 절에서는 nested CV가 public LB를 5회 중 4회 반올림 이내로 예측했고, nested CV 수준이 public LB보다 약 0.0011 낮은 수준 이동이 있으며, **public LB 차이 0.00007 미만은 해석 불가**라고 보고한다.

### kodaifukuda0311의 특성 실험 2종

[Feature Engineering: What Worked and What Didn't](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)와 [S6E8 XGB: The Power of Exact-Value TE](https://www.kaggle.com/code/kodaifukuda0311/s6e8-xgb-the-power-of-exact-value-te)는 LightGBM과 XGBoost에서 특성 계열별 채택 기각 표를 제공한다.
채택: 소수의 비율 특성, `screen_minus_work`, `screen_share_of_awake`, 12개 원본 열 전체의 정확값 TE(sklearn `TargetEncoder(cv=5)`를 각 outer fold 안에서 적용, 5시드).
기각: 대량 비율 및 차이 모음, 잔차 특성, 쌍 상호작용, 순서형 상호작용, 범주형 열만의 TE와 빈도 인코딩.

독자적인 부분은 **원본 프록시 데이터(`jayjoshi37` 판본)를 참조 분포로 쓰는 특성**이다.
훈련 데이터와 정확히 겹치는 행을 제거한 뒤, 원본 분포에서의 경험적 CDF 위치, 계급 조건부 CDF 격차, 중앙값까지 거리, 분위 구간별 타깃 평균, KDE 로그 우도비를 계산해 행별 특성으로 넣는다.
전체 CDF와 계급 조건부 CDF 격차를 "강한 개선"으로 보고하지만 근거는 표의 저장 수치이며 ablation 코드나 실행 출력은 노트북에 없다.
쌍 단위 원본 타깃 평균과 다변량 kNN 타깃률은 오히려 하락으로 보고해, 우리의 이슈 53 및 54 기각 결과와 방향이 일치한다.

### 단일 RealMLP 계열

[zhenruiweng의 Single Model RealMLP](https://www.kaggle.com/code/zhenruiweng/s6e8-public-lb-0-97009-single-model-realmlp)는 자체 구현 RealMLP(PBLD 주기 임베딩, 내부 앙상블 10, 3시드 x 5-fold)에 fold 안에서 sklearn `TargetEncoder(cv=5)`를 적용해 단일 모델 계열로 public LB 0.97009를 보고한다.
결측 지표와 중앙값 대체, 비율 특성을 함께 쓰며, 중앙값은 fold 분할 전에 전체 훈련 데이터에서 계산하는 약점이 남아 있다.
[nawfeelrahman1124444의 RealMLP 0.97014](https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014)는 같은 계열에 로그, 반올림 범주, 분위 구간, 조합 특성을 더한 변형으로 제목 점수는 더 높지만 저장 출력이 없어 근거가 약하다.
두 노트북 모두 신경망 단일 모델이 트리 단일 모델의 제목 점수(0.96983)를 넘는다고 주장하는 사례로, TabM 결과와 함께 신경망 계열의 우선순위를 높이는 방향의 증거다.

### 나머지

[boltuzamaki의 Single LightGBM](https://www.kaggle.com/code/boltuzamaki/single-lightgbm-target-encoding-no-model-blend)은 public LB 0.96949를 제시하지만 TE에 자기 행이 포함되고 전체 데이터로 고정 반복수 학습이라 검증 근거가 없으며 작성자도 이를 인정한다.
결측 패턴을 정수 비트마스크 하나로 인코딩하는 아이디어와 타깃 없는 쌍 빈도(pair support) 특성만 참고 후보다.
[zhukovoleksiy의 EDA 파이프라인](https://www.kaggle.com/code/zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline)은 "NaN이 숨은 0"이라는 가설을 관측된 0.00 값의 존재와 왼쪽 꼬리의 매끄러운 밀도로 기각하고, 훈련 데이터를 반으로 나눠 값별 타깃률이 재현되는지 확인하는 절차를 보여 준다.

## 기존 결론과의 충돌과 보강

| 쟁점 | 우리 결정 | 이번 조사 결과 | 해석 |
| --- | --- | --- | --- |
| 결측 대체값 보조 열 | [이슈 49](https://github.com/tmheo/predicting-smartphone-addiction/issues/49)에서 스크리닝 미달로 기각 | szymonkapiski는 제약 클리핑 재구성 + 경계 폭 열로 +0.00041, 이득이 결측 층에 집중 | 우리가 시험한 것은 무제약 대체였다. 생성 규칙을 산술 경계로 쓰는 변형은 미검증이다 |
| 쌍 결합 TE | [이슈 51](https://github.com/tmheo/predicting-smartphone-addiction/issues/51)에서 개별 쌍 채택 0으로 종료 | 전체 36쌍을 개수 열과 함께 한 블록으로 넣으면 세 학습기 모두 +0.0002 안팎 | 우리는 쌍을 개별 선별했고, 전 쌍 일괄 블록 + CT 열 + 0.1 해상도 조합은 미검증이다 |
| 원본 기준 타깃 통계 | [이슈 53](https://github.com/tmheo/predicting-smartphone-addiction/issues/53)에서 값별 라벨 평균 prior 기각 | kodaifukuda도 값 키 및 kNN 계열은 하락 보고로 일치하나, CDF 위치와 계급 CDF 격차는 개선 주장 | 분포 위치 특성은 값 키 prior와 기제가 다르다. 다만 근거가 저장 수치뿐이라 약하다 |
| 결측 지표 특성 | 지도 범위 밖으로 배제 | szymonkapiski도 MCAR 확인 후 무효 보고 | 배제 결정 재확인 |
| 앙상블 다양성 우선 | 선행 조사 결론 | 상관 0.997 구성원 0.00000 vs 상관 0.932 구성원 +0.00018, TabM 개선판도 blend 기여 0 | 재확인. 더 좋은 단일 모델이 자동으로 더 좋은 구성원이 아니다 |

## 검증 설계와 최종 제출에 주는 시사점

- **Public LB 노이즈 플로어 약 0.00015, 판별 한계 약 0.00007**은 마일스톤별 CV와 Public 관계 확인([이슈 57](https://github.com/tmheo/predicting-smartphone-addiction/issues/57), [60](https://github.com/tmheo/predicting-smartphone-addiction/issues/60), [65](https://github.com/tmheo/predicting-smartphone-addiction/issues/65))과 최종 제출 선택([이슈 69](https://github.com/tmheo/predicting-smartphone-addiction/issues/69))의 판단 기준으로 직접 쓸 수 있다.
- **영정보 플라시보 대조(난수 열, 기존 구성원 복제)로 앙상블 구성원 기여의 측정 대역을 먼저 정하는 절차**는 ADR 0001의 앙상블 판정을 보강할 후보다.
- 커뮤니티에는 우리와 동일한 fold 스펙을 명시한 **CC0 OOF 라이브러리**(szymonkapiski 25구성원판은 훈련 소스와 hyperparameter 포함, 47구성원판 별도)가 존재한다.
  지도의 현행 규칙은 외부 예측을 후보 풀에서 제외하므로, 채택 여부는 별도 결정이 필요하다.
- GBM meta 학습기, NNLS, greedy가 모두 선형 logistic보다 나빴다는 측정은 [이슈 64](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)와 [67](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)의 사전 정보로 유용하다.

## 스킵한 노트북과 사유

- `anhadmahajan06`의 두 노트북(U Smart Phone Addict, Ceiling Breaker): 파일명 점수 기반 제출물 순위 결합으로, 선행 조사 30번과 같은 계열이다.
- `rauffauzanrambe`, `daniilkrasnovvv`: 선행 조사 1번(najiama) 사본.
- `makthanithin`: 비공개 모델의 OOF와 제출물 배포로, 선행 조사 1번 및 23번과 같은 계열이다.
- `navazshfathi`: 공개 제출물 결합.
- `boltuzamaki`의 47모델 스택, `dynamo14324`의 50모델 스택, `nikita7364777`: 공개 OOF 라이브러리 위의 스태킹으로, 방법론적 신규성은 dariushafshar 3부작이 상회한다.
  다만 `nikita7364777`이 LGB, XGB, CatBoost, LogReg, greedy 다섯 meta 학습기를 같은 라이브러리에서 비교한 표는 이슈 67에서 참조할 수 있다.
- `mhamza0810`: 원시 특성에 공개 OOF 8종을 열로 붙여 XGBoost를 학습하는 비선형 스태킹인데, 선행 조사 22번 및 24번의 누출 있는 OOF가 입력에 포함된다.
- `amanatar`의 TabM Advanced Feature Engineering: szymonkapiski TabM 노트북의 무표기 사본.
- `mikhailnaumov`, `shashwat1729`, `daoviet`, `echloeprice` 등: 범용 템플릿 또는 표준 GBDT 구성으로 신규 내용 없음.

## 한계

이 문서는 2026-08-11에 내려받은 최신 공개본의 정적 분석이다.
szymonkapiski와 dariushafshar 노트북의 핵심 수치는 저장 예측 로드 또는 노트북 내 재계산 구조라 상대적으로 신뢰할 수 있으나, kodaifukuda0311과 RealMLP 계열의 개선 주장은 저장 수치와 제목 점수에 의존한다.
충돌 표의 항목은 우리 파이프라인에서 같은 fold로 재실행하기 전에는 채택 근거가 되지 않는다.
