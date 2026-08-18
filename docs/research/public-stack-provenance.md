# 상위 공개 스택 구성원의 출처 추적과 재현 후보 (이슈 #174)

## 결론: 재현 후보 우선순위

이 조사는 public 0.97077~0.97097 스택 계열의 정점인
[94 Verified OOFs (dariushafshar)](https://www.kaggle.com/code/dariushafshar/94-verified-oofs-honest-cv-0-96985-lb-0-97097)의 구성원 94개 전부의 계보를 복원했다.
공급원은 정확히 네 곳이고, 구성원 94개 가운데 85개는 공개 코드 또는 완전한 레시피 계약으로 재현 가능하며, 5개는 재현 불가, 4개는 부분 재현만 가능하다.

지도 #172의 규칙에 따라, 아래의 외부 OOF 수치는 전부 **읽기 전용 진입 진단**이다.
외부 예측 파일은 채택 근거로도, 앙상블 구성원으로도 쓰지 않으며, 편입은 자체 파이프라인에서 재현·재학습한 것만 허용된다.

| 순위 | 레시피 | 출처 | 라이선스 | 예상 신규성 | 재현 난이도 |
| ---: | --- | --- | --- | --- | --- |
| 1 | 독립 전처리 고정 스케줄 Lookup-Transformer 2호: train+test 결합 정확값 어휘, quantile-normal 수치 채널, 파생 토큰 6개, 24 epoch 고정, EMA, 조기 종료 없음 | [beicicc lookup 계약](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts), 아키텍처는 [tamerlanomralinov](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041) | 계약 CC BY 4.0, 노트북 Apache 2.0 | 높음: 진단 기여 +0.000162, 우리 exp081과 스피어만 0.975 | 낮음: 기존 `lookup_transformer.py`의 전처리·스케줄 변형 |
| 2 | RealMLP-TD: hidden 512x512x512, PL 임베딩, outer fold 안 5-fold 내부 TE, 4 epoch 고정, 2시드 평균 | [beicicc RealMLP 계약](https://www.kaggle.com/datasets/beicicc/s6e8-fixed4-realmlp-two-seed-artifacts), 코드 참고는 [szymon src/train_realmlp.py](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models)와 [zhenruiweng](https://www.kaggle.com/code/zhenruiweng/s6e8-public-lb-0-97009-single-model-realmlp) | CC BY 4.0, CC0, Apache 2.0 | 높음: 우리 풀에 RealMLP 계열 없음, 진단 기여 +0.000074, 최근접 0.986 | 중간: pytabkit 의존, GPU 필요 |
| 3 | CatBoost 정확값 범주 + 화면 관계 7특성 블록(차이 3, 안전 비율 4, NaN 전파·무보정) | [beicicc CatBoost 화면 관계 계약](https://www.kaggle.com/datasets/beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts) | other(예측 파일 재배포 불가, 레시피 서술은 참조 가능) | 중간: 짝 ablation +0.00044(5/5 fold), 진단 기여 +0.000053, exp070과 0.993 | 낮음: exp070 계열 설정에 7열 추가 |
| 4 | no-TE 뷰 3종(LGB·XGB·CatBoost에서 인코더 제거)과 CatBoost native ordered 범주(catnative) | [adarsh1077 라이브러리 README](https://www.kaggle.com/datasets/adarsh1077/s6e8-adarsh-oof-library), 특성 레시피는 [tomasa2](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t) | CC0 | 중간: 178구성원 스택 상위 계수 4개 중 3개가 no-TE 뷰, catnative 단독 0.96860 | 낮음: 기존 트리 설정에서 TE 열 제거 또는 범주 전환 |
| 5 | 정체성·자리수 특성 블록: `other_screen` + 수치 9열별 round0/1/2, absdiff, is_round0/1, tenths, hundredths | [beicicc LGBM 계약](https://www.kaggle.com/datasets/beicicc/s6e8-fixed900-identity-digit-lightgbm-artifacts)과 [XGB 계약](https://www.kaggle.com/datasets/beicicc/s6e8-fixed1500-xgb-identity-digit-artifacts) | CC BY 4.0, CC0 | 중간: 짝 ablation +0.0023~0.0024, 진단 기여 +0.000030~+0.000063 | 낮음: LightGBM 대리 스크리닝 먼저 |
| 6 | 교사-학생 anti-residual 보정: 교사 LGBM 순위를 매끄러운 학생 LGBM이 회귀, 부호 제곱 잔차를 nested로 0.10 가중 | [raykkretzschmar 08-14 판](https://www.kaggle.com/code/raykkretzschmar/mix-the-meta-models-then-learn-what-they-miss) | Apache 2.0 | 낮음-중간: 앵커 4종에서 +0.000018~+0.000036, 60/60 fold 양수 | 중간-높음: nested 이중 학습 구현 |
| 7 | 3-arm 메타 구조: 순위+로짓 이중 표현, 결측·완전성 상호작용 두 팔, 0.5/0.25/0.25 순위 혼합 | [94 Verified OOFs 노트북](https://www.kaggle.com/code/dariushafshar/94-verified-oofs-honest-cv-0-96985-lb-0-97097) | Apache 2.0 | 이슈 #64·#67 범위에 흡수 | 낮음: 코드 전문 공개 |

FM(factorization machine) 계열은 코드가 전부 공개되어 있음에도 후보에서 제외를 유지한다.
제작자 스스로 74구성원 풀 기여를 +0.000006으로 측정했고, 2026-08-14 전수 재점검의 기각 결론을 뒤집을 새 근거가 없다.

재현 불가 판정 구성원은 5개(naji01~naji05)이고, 부분 재현만 가능한 구성원은 4개(golem a·d·e·f)다.

## 조사 범위와 방법

2026-08-17 KST에 Kaggle 공식 API와 CLI로 다음을 수행했다.

- 대상 노트북과 이전 판본 계열(74 OOFs, 82 OOFs)을 `kaggle kernels pull`로 내려받아 코드 셀과 마크다운 셀을 전부 읽었다.
- 각 노트북의 kernel 메타데이터에서 입력 데이터셋 12개를 확정하고, 데이터셋마다 메타데이터(라이선스), 파일 목록, README, manifest, contract를 내려받아 구성원 계보를 복원했다.
- 2026-08-14 전수 재점검 이후 갱신된 상위 노트북을 Public Score 정렬과 최신 정렬 양쪽으로 훑어 5개를 추가로 정독했다.
- 지도 규칙이 허용하는 읽기 전용 진입 진단으로, beicicc 후보 6개의 OOF를 내려받아 fold 일치, 단독 AUC, 풀 최근접 스피어만 상관, 표준 순위 평균 기여를 측정했다.

라이선스와 출처 기록은 `docs/agents/kaggle-public-notebook-licensing.md` 절차를 따랐다.
공개 노트북 소스는 Apache License 2.0이고, 데이터셋 라이선스는 아래 계보 표에 개별 기록했다.

## 94 Verified OOFs 스택의 구조

노트북은 다음을 수행한다.

1. 82개 기준 구성원과 12개 신규 구성원의 OOF·테스트 예측 쌍을 이름 고정 목록(`SCORED_82`)과 해시 검사로 로드한다.
2. 5-fold nested 비교로 82구성원 기준 대비 94구성원 풀을 짝지어 평가한다.
   보고된 개선은 평균 +0.00010, 5/5 fold 양수다.
3. 최종 예측은 3-arm 구조다.
   arm 1은 정규화 순위와 잘린 로짓을 함께 넣은 전역 로지스틱 회귀다.
   arm 2와 arm 3은 같은 입력에 완전 관측 여부, 고결측 여부, 구성원 표준편차, (arm 3만) 화면 시간 4열 완전성 지시자를 곱한 상호작용 특성을 더한 로지스틱 회귀다.
   세 팔의 백분위 순위를 0.5/0.25/0.25로 혼합한다.
4. 공개 제출 55370638이 public LB 0.97097을 기록했다는 사실을 역사적 기록으로 명시하고, LB 개선 폭 0.00011이 측정된 노이즈 플로어 0.00015 안이라고 스스로 판정한다.

같은 저자의 게이트 노트북
[12 OOF Members Survive a Fold and Hash Audit](https://www.kaggle.com/code/dariushafshar/12-oof-members-survive-a-fold-and-hash-audit)는 신규 12개의 fold 벡터가 표준 분할과 정확히 일치함을 검사하고, 14개 파일명이 바이트 중복 2쌍 때문에 12개 고유 구성원임을 해시로 보였다.
같은 노트북은 fold 일치와 AUC 재현이 정렬을 증명할 뿐 상류 학습이 검증 라벨을 보지 않았다는 사실까지 증명하지 못한다고 명시한다.
이 한계는 우리의 자체 재현 원칙을 그대로 지지한다.

## 구성원 계보: 94개 = 73 + 5 + 4 + 12

`SCORED_82`는 szymonkapiski 라이브러리 73개, raykkretzschmar FM 5개, dariushafshar golem 4개(a, d, e, f)로 구성되고, 신규 12개는 전부 beicicc 계약 데이터셋에서 온다.

| 공급원 | 구성원 수 | 라이선스 | (a) 생성 코드 공개 | (b) 저장 출력·fold 검증 | (c) 레시피 재현 가능성 |
| --- | ---: | --- | --- | --- | --- |
| [szymonkapiski/s6e8-oof-library-47-models](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models) (실제 74모델) | 73 | CC0-1.0 | `mine` 구성원은 데이터셋 안 `src/`에 훈련 코드 전문 공개, `pub_*`는 원 공개 노트북 링크 명시, `naji*`는 코드 없음 | float64 OOF·test 쌍, manifest.csv(구성원별 AUC·특성·출처·주석), hyperparameters.json, `train_keys.parquet`로 id 정렬 검증 가능, fold 스펙 명시 | `mine` 계열 높음, `pub_*` 계열은 원 노트북 코드로 가능하나 일부는 TE 누출 수정 필요, `naji*` 불가 |
| [raykkretzschmar/s6e8-fm-lattice-blend-members](https://www.kaggle.com/datasets/raykkretzschmar/s6e8-fm-lattice-blend-members) | 5 (fmplr, fmnum, fmdeep, fmwide, fmpure) | Apache 2.0 | `train_fm.py`, `train_fm2.py`, `band_fm.py` 전문 공개, 방법 서술 노트북 별도 | float64 npy와 id 포함 parquet 병행, 구성원별 AUC와 최근접 상관 README 기록 | 높음, 다만 기여 측정치가 낮아 후보 제외 유지 |
| [dariushafshar/s6e8-golem-oof-library](https://www.kaggle.com/datasets/dariushafshar/s6e8-golem-oof-library) | 4 (a, d, e, f) | CC0-1.0 | 훈련 코드 비공개, README와 manifest.csv에 모델 계열·특성 계열·조기 종료 방식 서술만 존재 | manifest.csv에 fold별 AUC·범위·상관을 패키징 시점에 재계산해 기록 | 부분: 하이퍼파라미터 부재, a와 f는 검증 fold 조기 종료로 OOF가 약간 낙관적이라고 저자가 명시 |
| beicicc 계약 데이터셋 8종 (아래 상세) | 12 | CC0 2종, CC BY 4.0 4종, other 2종 | 훈련 코드 비공개, 대신 contract/manifest에 특성 정의·전체 하이퍼파라미터·고정 스케줄·시드·fold 해시·자료 해시를 완전 기록 | fold_id.npy 동봉(1-based), fold별 AUC CSV, 모든 산출물 SHA-256 | 높음: 계약만으로 재현 가능 수준, 코드 부재는 계약 완전성으로 상쇄 |

szymonkapiski 라이브러리의 `pub_*` 12개는 공개 노트북 재현이고 원문 링크가 manifest에 있다.
그 가운데 `pub_rmlp`, `pub_tabm`(omidbaghchehsaraei 계열)은 우리 선행 조사에서 확인한 전체 자료 TE 누출이 있는 원본을 그대로 실행한 것이므로, 재현한다면 fold-fit TE로 교정해야 한다.
`lookup` 구성원은 tamerlanomralinov의 아키텍처를 szymonkapiski가 5-fold로 재훈련한 것으로, 원 노트북이 10-fold라 그대로 스택할 수 없다는 이유가 manifest에 기록되어 있다.

### 재현 불가 판정

- `naji01`~`naji05` 5개는 najiama의 [OOF 데이터셋](https://www.kaggle.com/datasets/najiama/predicting-smartphone-addiction-oof-submission-csv)에서 오며 생성 코드가 없고 manifest에도 "author's own"으로만 기록된다.
  94 OOFs 노트북 스스로 이 파일들의 라이선스가 불명(unknown licence)이라 재배포하지 않는다고 명시한다.
  najiama는 2026-08-17 판 노트북에서도 핵심 코드를 비공개로 유지한다고 밝혔다.
  이 5개는 재현 불가로 표시하고 편입 후보에서 제외한다.
- golem `a`, `d`, `e`, `f` 4개는 특성 계열 서술은 있으나 하이퍼파라미터가 없고, `a`와 `f`는 검증 fold 조기 종료 낙관이 공표되어 있다.
  정확 재현은 불가하고 계열 수준 재구현만 가능하므로 직접 편입 후보로 삼지 않는다.
  다만 `a`의 "identity-residual + quantisation-digit" 특성 방향은 beicicc 정체성·자리수 계약(우선순위 5)이 더 정밀하게 문서화했으므로 그 경로로 흡수한다.

### beicicc 계약 데이터셋 상세

beicicc의 8개 데이터셋은 모두 동일한 형식이다.
훈련 코드는 없지만 contract 또는 manifest JSON에 자료 SHA-256, fold 계약(스펙과 fold별 검증 id 해시), 특성 정의(수식 포함), 전체 하이퍼파라미터, 고정 스케줄(조기 종료와 체크포인트 선택 금지 명시), fold별 AUC, 산출물 해시가 기록된다.

| 데이터셋 | 구성원 | 단독 OOF | 요지 |
| --- | --- | ---: | --- |
| [fixed-schedule-lookup-transformer](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts) | lookup_fixed24 | 0.96605 | Tamerlan 아키텍처(d=128, 4층, 8헤드), 24 epoch 고정, EMA, 조기 종료 없음, train+test 결합 정확값 어휘와 quantile-normal 수치 채널 |
| [second-seed-fixed-schedule-lookup](https://www.kaggle.com/datasets/beicicc/s6e8-second-seed-fixed-schedule-lookup-artifacts) | lookup_fixed24_seed1042 | 0.96606 | 같은 계약에서 시드만 1042~1046, 두 판의 등순위 평균이 독립 시드 잔차로 사용됨 |
| [fixed-schedule-exact-value-catboost](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts) | (fixed4000, 아래와 바이트 중복) | 0.96730 | 원시 12열 + `float.hex` 왕복 정확값 키 9열, depth 8, lr 0.05, 4000 iter 고정 |
| [fixed4000-catboost-screen-relation](https://www.kaggle.com/datasets/beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts) | baseline, screen_relations | 0.96730, 0.96773 | 위 baseline에 화면 관계 7특성만 추가한 짝 ablation, +0.00044, 5/5 fold |
| [fixed4-realmlp-two-seed](https://www.kaggle.com/datasets/beicicc/s6e8-fixed4-realmlp-two-seed-artifacts) | realmlp_seed01_fixed4 | 0.96826 | RealMLP 512x512x512, PL 임베딩, outer 학습부 안 5-fold 내부 TE, 4 epoch 고정, 2시드 평균 |
| [fixed900-identity-digit-lightgbm](https://www.kaggle.com/datasets/beicicc/s6e8-fixed900-identity-digit-lightgbm-artifacts) | raw12, enhanced103 | 0.96339, 0.96575 | 정체성·자리수 블록 짝 ablation, +0.00235, 5/5 fold |
| [fixed1500-xgb-identity-digit](https://www.kaggle.com/datasets/beicicc/s6e8-fixed1500-xgb-identity-digit-artifacts) | raw12 (enhanced103은 중복 제외) | 0.96340 | 같은 블록의 XGBoost 판, +0.00245, 5/5 fold |
| [fixed1500-xgb-screen-relation](https://www.kaggle.com/datasets/beicicc/s6e8-fixed1500-xgb-screen-relation-artifacts) | baseline103, treatment110 | 0.96585, 0.96603 | 103특성 위 화면 관계 7특성 짝 ablation, +0.00018, 5/5 fold |
| [fixed900-structural-lgbm](https://www.kaggle.com/datasets/beicicc/s6e8-fixed900-structural-lgbm-artifacts) | raw12, structural | 0.96251, 0.96372 | `other_screen` + 소수 자리 개수 좌표 9열 짝 ablation, +0.00121, 5/5 fold |

화면 관계 7특성의 정의는 다음과 같고, 결측 정책이 명시적이다.
차이 3개(`gaming_minus_work`, `screen_minus_work`, `weekend_minus_daily`)는 NaN을 자연 전파한다.
비율 4개(`social_share_screen`, `gaming_share_screen`, `work_share_screen`, `screen_to_sleep`)는 분자·분모가 유한하고 분모가 양수일 때만 정의하며, epsilon·대체·클리핑·정의 여부 플래그를 전혀 쓰지 않는다.

정체성·자리수 블록은 `other_screen`과 수치 9열별 10개 열이다.
반올림 3해상도(round0/1/2)와 그 절대 편차, 정수·소수1자리 여부 지시자, 소수 첫째·둘째 자리 값이다.

beicicc 데이터셋 2종(fixed4000-catboost-screen-relation, fixed900-structural-lgbm)의 라이선스는 `other`다.
이 두 데이터셋의 예측 파일은 읽기 전용 진단 외 용도로 쓰지 않고 재배포하지 않으며, README와 manifest에 서술된 레시피 사실만 참조한다.

## 읽기 전용 진입 진단 결과

2026-08-17에 우리 `artifacts/folds.parquet`(id 순서가 공식 train.csv와 일치함을 확인)과 현재 후보 풀 16구성원(champion exp081, OOF 0.96920)으로 측정했다.
기준은 16구성원 균등 순위 평균(AUC 0.9686780)이고, 기여는 후보 1개를 17번째로 더했을 때의 변화다.
반복해서 강조하면, 이 수치는 외부 예측의 채택 근거가 아니라 어떤 레시피를 자체 재현할지 고르는 우선순위 근거로만 쓴다.

| 후보 | fold 일치 | 단독 AUC | 풀 최근접 (스피어만) | 순위 평균 기여 |
| --- | --- | ---: | --- | ---: |
| lookup_fixed24 | 일치(1-based 보정) | 0.966051 | exp081_lookup (0.97499) | +0.0001620 |
| realmlp_seed01_fixed4 | 일치 | 0.968258 | exp074_lgb_kitopl (0.98574) | +0.0000737 |
| xgb_screen_relations_treatment110 | fold 파일은 같은 계약 해시 | 0.966029 | exp074_lgb_kitopl (0.98947) | +0.0000625 |
| catboost 정확값+화면7 (screen_relations) | 일치 | 0.967734 | exp070_cat_exact_cats (0.99288) | +0.0000534 |
| identity_digit_enhanced103 (LGBM) | 일치 | 0.965746 | exp074_lgb_kitopl (0.98859) | +0.0000302 |
| structural_decimal (LGBM) | 일치 | 0.963724 | exp026_constrained (0.98733) | -0.0000052 |

주목할 결과는 lookup_fixed24다.
우리 champion과 같은 아키텍처 계열인데도 스피어만 상관이 0.975에 그치고 진단 기여가 여섯 후보 중 가장 크다.
차이는 아키텍처가 아니라 전처리와 스케줄이다.
우리 exp081은 자체 특성 33열(복원 블록 포함)을 읽지만, beicicc 판은 원시 12열의 정확값 어휘(train+test 결합, 타깃 미사용)와 quantile-normal 수치 채널, 파생 토큰 6개만 쓰고 24 epoch 고정에 조기 종료가 없다.
raykkretzschmar가 "readout·깊이·폭을 바꾼 재학습은 기존 lookup과 상관 0.9926인 쌍둥이"라고 보고한 것과 결합하면, Lookup 계열의 다양성 축은 구조가 아니라 입력 표현과 학습 스케줄이라는 일관된 그림이 된다.

structural_decimal은 진단 기여가 음수이므로 소수 자리 개수 좌표 단독 블록은 우선순위에서 제외한다.
정체성·자리수 블록(우선순위 5)이 같은 방향의 신호를 더 강하게 포함한다.

## 재현 후보 상세

### 1순위: 독립 전처리 고정 스케줄 Lookup 2호

- 레시피: 기존 `src/pipeline/lookup_transformer.py`를 유지하되 입력을 원시 12열로 줄이고, 정확값 어휘를 train+test 결합으로 만들고(타깃 미사용), 수치 채널을 quantile-normal로 바꾸고, 파생 토큰 6개(계약의 `derived_tokens`)를 추가하고, 24 epoch 고정과 EMA 0.999, 조기 종료 없음으로 학습한다.
- 근거: 진단 기여 +0.000162는 이번 측정에서 가장 크고, 우리 기존 Lookup과의 상관 0.975는 풀 내부 최근접 상관들(0.98 이상)보다 낮다.
- 위험: 정확한 파생 토큰 6개의 정의가 계약에 이름으로만 있고 수식이 없다.
  화면 시간 파생(`other_screen` 계열)으로 추정되나, 재현 시 우리 기존 파생 6열로 대체하고 seed 42 한 점에서 먼저 확인한다.
- 판정 경로: ADR 0001 단일 모델 문턱과 3시드 확정, 풀 진입은 이슈 #63 규약.

### 2순위: RealMLP-TD 2시드 평균

- 레시피: pytabkit `RealMLP_TD_Classifier`, hidden 512x512x512, piecewise-linear 임베딩(hidden 20, output 5), 배치 256, 4 epoch 고정(스케줄 지평 8), outer 학습부 안 5-fold 내부 TE, 2시드 산술 평균.
- 근거: 우리 풀에 RealMLP 계열이 없고, 단독 0.96826은 우리 exp065 TabM(0.96833)과 대등하며, 진단 기여 +0.000074다.
  szymonkapiski의 74구성원 blend에서도 상관 0.932의 realmlp가 +0.00018을 더한 기록이 있다.
- 참고 코드: szymon `src/train_realmlp.py`(CC0)와 zhenruiweng 공개 노트북(Apache 2.0), 다만 zhenruiweng 판은 중앙값 대체를 fold 밖에서 맞추므로 우리 fold-fit 규율로 교정한다.
- 이슈 #61(RealMLP·TabM 다양성)이 이미 열려 있으므로 그 트랙의 구체 레시피로 쓴다.

### 3순위: CatBoost 정확값 + 화면 관계 7특성

- 레시피: 우리 exp070 계열(정확값 native 범주)에 위 7특성을 한 블록으로 추가한다.
  결측 정책(NaN 전파, 무보정 비율)을 계약 그대로 따른다.
- 근거: 동일 설정 짝 ablation에서 +0.00044(5/5 fold)이고, 이 짝의 baseline은 우리 exp070과 같은 정확값 키 접근이다.
  진단 기여 +0.000053, exp070과 상관 0.993이므로 단독 편입보다 exp070 대체 후보로 잰다.
- 우리 이슈 #62·#51 계열에서 일반 비율 특성이 기각된 적이 있으나, 그 실험들은 이 7열 고정 블록과 결측 정책이 다르므로 재판정 대상이다.

### 4순위: no-TE 뷰와 catnative

- 레시피: 현재 채택된 트리 구성(LGB, XGB, CatBoost)에서 TE 계열 열만 제거한 no-TE 뷰 3종과, 수치 원시값을 문자열 범주로 CatBoost에 직접 주는 catnative 판.
- 근거: adarsh1077의 178구성원 rank-gauss 로지스틱 스택에서 상위 계수 4개 중 3개가 no-TE 뷰였고(+0.334, +0.278, +0.176), catnative는 단독 0.96860에 최대 계수 +0.749였다.
  같은 문서는 시드·하이퍼파라미터 변형 구성원 대부분이 0 또는 음수 계수였다고 보고한다.
- 한계: 생성 코드가 없고 README 서술 수치라 근거 강도는 보통이며, catnative는 우리 exp070(수동 TE 병행)과 기제가 겹치므로 상관 측정을 먼저 한다.
- 비용이 매우 낮으므로(설정에서 열 제거) LightGBM 대리 스크리닝으로 시작한다.

### 5순위: 정체성·자리수 특성 블록

- 레시피: `other_screen` + 수치 9열별 round0/1/2, absdiff_round0/1/2, is_round0, is_round1, tenths, hundredths(총 91열)를 우리 트리 기준 모델에 한 블록으로 추가한다.
- 근거: LGBM과 XGB 양쪽에서 짝 ablation +0.0023~+0.0024(각 5/5 fold, 짝수·홀수 id 슬라이스에서도 재현).
  우리 champion의 특성 목록에는 자리수 계열이 없다.
- 주의: 우리 풀의 exp074(kitopl 구간 해상도 계열)와 잔차가 겹칠 수 있어 진단 기여가 +0.000030에 그쳤다.
  단독 구성원보다 기존 트리 모델의 특성 증분으로 재는 편이 맞다.

### 6순위: 교사-학생 anti-residual 보정

- 레시피: 교사는 원시 특성 + 정확값 빈도 + 결측 지시자 + 생성기 항등식의 LightGBM이고, 학생은 교사의 백분위 순위를 회귀하는 더 매끄러운 LightGBM이다.
  각 outer fold에서 내부 4모델이 outer 학습행의 OOF 교사 타깃을 만들고, 보정은 교사-학생 순위 잔차의 부호 제곱을 표준화해 0.10 가중으로 더한다.
- 근거: 서로 다른 앵커 4종(94구성원 스택, naji 스택, 자체 스택, 3원 blend)에서 +0.000018~+0.000036, 60/60 fold 양수.
- 판정: 개선 폭이 ADR 0001 앙상블 문턱(+0.00002) 부근이므로, 이슈 #67의 결합 비교가 끝난 뒤 남는 잔차가 있을 때만 연다.

### 7순위: 3-arm 메타 구조

- 94 OOFs 노트북의 메타 구조(순위+로짓 이중 표현, 완전성·고결측 상호작용 팔, 고정 0.5/0.25/0.25 순위 혼합)는 코드 전문이 Apache 2.0으로 공개되어 있다.
- 이중 표현은 2026-08-14 전수 재점검이 이미 이슈 #64에 흡수했고, 상호작용 팔과 혼합은 이슈 #67 범위이므로 새 티켓 없이 그 비교의 한 설계로 넣는다.
- 같은 저자의 부정 결과(GBM 메타 -0.000116, NNLS·greedy 열세)와 raykkretzschmar의 반복 측정(메타 용량 확장 3종 모두 로지스틱보다 나쁨)이 선형 메타 유지의 사전 근거다.

## 2026-08-14 이후 증분

### adarsh1077: S6E8 Diversity Beats Strength (public 0.97113, 현 공개 최고)

[원문](https://www.kaggle.com/code/adarsh1077/s6e8-diversity-beats-strength)은 공개 라이브러리 6종 + 자체 22모델(CC0 공개)로 178구성원 rank-gauss nested 로지스틱 스택을 만든다.
재사용 가치가 높은 측정은 다음과 같다.

- 전 train 행 691,369개의 12열 키가 전부 서로 달라 정확 일치 검색 채널이 없고, test 행의 0.00%만 train과 일치한다.
  따라서 공개 `lookup` 계열의 신호는 행 검색이 아니라 열별 정확값 목표 통계라는 해석을 코드로 뒷받침한다.
- 등화된 스택의 Bayes 최적 AUC 추정은 OOF 약 0.97006으로, 남은 여지가 5e-5 안팎이라는 상한 산정 절차를 공개한다.
- 6회 제출에서 public LB가 nested OOF보다 +0.00098~+0.00115 위에 안정적으로 앉았다.
  test 예측이 5개 fold 모델 평균이라 OOF보다 유리하다는 설명이며, 우리 마일스톤 제출의 OOF 대비 격차 해석에 직접 쓸 수 있다.
- leave-one-author-out: 저자별 배열 전부를 빼고 재적합하면 boltuzamaki 라이브러리(45개)가 +0.000189로 가장 크고, 나머지 공급원 대부분은 1e-6 급 노이즈 대역이다.
- 35개 생성기 지문 후보(둘째 소수, 나머지 연산, 교차 자리수 등)를 스택 잔차에 회귀해 -0.000182로 일괄 기각했다.
  새 특성은 타깃이 아니라 현재 스택의 잔차에 대고 검정하라는 절차가 전이 가능하다.

### najiama의 출처 감식과 0.97113 계열

[Where does the 0.97101 NN score really come from](https://www.kaggle.com/code/najiama/where-does-the-0-97101-nn-score-really-come-from)는 화제가 된 "NN Residual Network" 노트북(anthonytherrien, public 0.97101)의 최종 제출이 NN 가중 1e-4의 껍데기이고, 실제로는 공개 제출 파일 두 개의 바이트 복사임을 스피어만 1.0과 최대 절대 차 0.0으로 증명했다.
[S6E8 Addiction LB 0.97113](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97113)은 비공개 코드의 blend 산출물 재배포라 재현 대상이 아니다.
교훈은 기존 결론의 재확인이다.
공개 상위권 점수는 파일 재배포 계열이 많고, 출처는 항상 예측 파일 해시·상관으로 검증해야 한다.

### raykkretzschmar 08-14 판: 교사-학생 잔차

위 6순위에 기술했다.
같은 판의 부정 결과 표(메타 용량 3종 실패, 구간 isotonic -0.00008, 목적 함수 변형은 상관 0.9965로 다양성 축 아님, `daily<=3h`와 `n_missing>=4` 구간 전용 모델 음수)는 우리 이슈 #67의 사전 정보로 유효하다.

### 기타

- yadoy666의 "94 Verified OOF GPU-Accelerated Meta Stack"과 "177-Member Diversity Stack"은 각각 dariushafshar와 adarsh1077 구조의 사본 계열로 신규 내용이 없다.
- stephentarter의 PS-S06E08 연작(08-15~16)은 표준 구성 학습 노트북으로, CatBoost 판은 2026-08-14 재점검에서 이미 다뤘다.
- boltuzamaki의 [s6e8-oof-prediction-library](https://www.kaggle.com/datasets/boltuzamaki/s6e8-oof-prediction-library)(45배열)는 adarsh 측정에서 저자 단위 가치가 가장 컸다.
  내부에 바이트 중복 쌍(`xgb_d7_alt1`/`xgb_d7_alt2`)이 있으므로 참조 시 해시 제거가 필요하다.
  훈련 코드 동봉 여부는 이번 조사에서 확인하지 못했으므로, 후속으로 이 라이브러리의 manifest를 같은 절차로 감사할 가치가 있다.

## 라이선스와 출처 기록

- 이 문서가 인용한 모든 공개 노트북 소스는 Kaggle 공개 규약에 따라 Apache License 2.0이다.
- 데이터셋 라이선스는 계보 표와 beicicc 상세 표에 개별 기록했다.
  CC0과 CC BY 4.0 데이터셋은 출처 표기와 함께 참조했고, `other` 라이선스 2종은 읽기 전용 진단 외 사용과 재배포를 하지 않는다.
- najiama OOF 데이터셋은 라이선스 불명이므로 어떤 용도로도 쓰지 않는다.
- 코드를 실제로 복사·수정해 저장소에 들여올 때는 `docs/agents/kaggle-public-notebook-licensing.md`의 고지·변경 표시 절차를 그 시점에 다시 적용한다.

## 한계

- 이 문서는 2026-08-17 KST에 내려받은 판본의 정적 분석과 읽기 전용 진단이다.
  대회 종료 전이므로 라이브러리와 노트북은 계속 갱신될 수 있다.
- beicicc 계약과 golem manifest의 수치는 저장 출력이 있으나, 상류 학습이 검증 라벨을 보지 않았다는 사실은 어떤 감사로도 증명되지 않는다.
  채택은 오직 자체 재현 실행의 ADR 0001 판정으로만 한다.
- 진입 진단의 기준(16구성원 균등 순위 평균)은 표준화된 근사이며, 실제 편입 판정은 이슈 #63·#64의 결합 규약을 따른다.
- adarsh1077과 golem의 레시피 서술 수치는 코드 없는 저자 보고치라 근거 강도가 보통이다.
