# 확장 스택용 외부 구성원 장부 (이슈 #442)

## 결론

공개 OOF 라이브러리 17개 데이터셋(공급원 11곳)에서 후보 226개를 반입해 검증하니 **209개가 확장 스택에 넣을 수 있는 구성원**이다.
라이선스는 CC0 198개, CC BY 4.0 6개, Apache 2.0 5개다.
17개는 제외했고 사유를 아래에 남겼다.

검증을 통과한 209개의 단독 AUC는 `0.91692`에서 `0.96869` 사이이고, 우리 champion `0.9693397`보다 높은 구성원은 없다.
`0.968` 이상은 39개, `0.95` 미만은 54개다.

#386이 잰 재현 가능 85구성원 가운데 이 장부에 있는 것은 81개이고 그중 79개가 통과했다.
빠진 6개는 pub_evg(10분할), xgb_screen_relations_baseline103(같은 바이트의 xgb_identity_digit_enhanced103이 대신 통과), 라이선스 other인 beicicc 4개다.
즉 #386 기준선과 겹치는 통과 구성원은 실질 80개이고, 그 바깥에 새로 통과한 구성원이 130개다.

지도 #441의 목표(nested 약 0.9702, 자체 35 + 외부 약 200)에 필요한 폭은 이 장부로 확보된다.
품질은 별개다: 통과 209개 가운데 107개가 주의 사항을 달고 있고, 그중 49개는 float32 저장, 50개는 레시피 비공개다.

## 산출물

- 기계가 읽는 장부: `docs/research/external-member-ledger.json`
  구성원 226개 전부가 `status`(accepted/excluded), `exclusion_reason`, `caveats`, `in_ext85`, 파일 경로, dtype, 재채점 AUC, 선언 AUC, 해시, 분할 검사 결과를 가진다.
- 생성 도구: `scripts/build_external_member_ledger.py`
  읽기 전용이며 MLflow 실행을 만들지 않고 `artifacts/pool.yaml`을 건드리지 않는다.
- 외부 파일: `data/external/ext94/`(#386 반입분)와 `data/external/ext442/`(이번 반입분).
  `data/`는 커밋 제외 경로이므로 재현하려면 장부의 `sources`에 적힌 데이터셋을 같은 경로에 내려받는다.

## 조사 범위와 방법

2026-08-27 KST에 Kaggle API로 데이터셋 메타데이터(라이선스)와 파일 목록을 확인하고, 로컬에 없던 8개 데이터셋을 내려받았다.
분할 근거는 각 데이터셋의 README와 저자 노트북 코드(`kaggle kernels pull`)에서 읽었다.

검증 항목은 #386과 같다.

- 행 수 691,369(OOF)와 296,302(test), 유한값
- 우리 라벨로 재채점한 AUC와 저자 선언 AUC의 차이가 `1e-5` 이내
- OOF+test 배열(float64 바이트)의 SHA-256으로 중복 제거, 먼저 적재된 쪽을 남김
- 분할 벡터가 동봉된 구성원은 `artifacts/folds.parquet`와 일치

여기에 지도 #441의 자격 규칙을 더했다.

- 라이선스가 CC0·Apache 2.0·CC BY 4.0·MIT 가운데 하나로 표시된 데이터셋만 반입
- 2단계 산출물과 10분할 배열은 제외
- 계보 조사(#174)가 재현 불가·부분 재현으로 판정한 구성원은 제외

## 행 순서와 분할 대조

외부 라이브러리는 전부 id 없이 위치로 정렬한다.
다음이 전부 성립한다.

- `artifacts/folds.parquet`의 id 순서 = `train.csv` 파일 순서
- szymonkapiski `train_keys.parquet`의 id와 라벨 = 우리 `train.csv`, `test_keys.parquet`의 id = `test.csv`
- boltuzamaki `train_labels.parquet`의 라벨 = 우리 라벨, parquet의 `id` 열 = `train.csv`·`test.csv` 순서
- dariushafshar 측정 팩의 `folds_seed42.npy` = 우리 5분할과 위치별로 정확히 일치
- beicicc 7개 데이터셋의 `fold_id.npy`(1부터 셈)는 보정 뒤 우리 5분할과 일치

분할 벡터가 있는 구성원은 9개뿐이고 나머지 200개는 위치 정렬과 저자의 분할 서술에 의존한다.
장부의 `fold_evidence`가 근거의 종류를 구분한다.

| 근거 | 통과 | 뜻 |
| --- | ---: | --- |
| `fold_vector` | 9 | 분할 벡터 동봉, 우리 분할과 일치 확인 |
| `published_code` | 61 | 고정 5분할을 쓰는 훈련 코드가 공개됨 |
| `sibling_code` | 1 | 같은 저자의 다른 노트북이 같은 골격으로 고정 5분할 사용 |
| `author_statement` | 138 | README 또는 노트북 서술만 있음 |

## 공급원별 결과

| 공급원 | 데이터셋 | 라이선스 | 후보 | 통과 | 통과 AUC 범위 |
| --- | --- | --- | ---: | ---: | --- |
| szymon74 | szymonkapiski/s6e8-oof-library-47-models | CC0 | 74 | 67 | 0.91880 ~ 0.96867 |
| szymon_weak50 | szymonkapiski/s6e8-50-weakest-oof-models | CC0 | 50 | 50 | 0.91692 ~ 0.95676 |
| bolt47 | boltuzamaki/s6e8-oof-prediction-library | CC0 | 47 | 44 | 0.93799 ~ 0.96834 |
| adarsh22 | adarsh1077/s6e8-adarsh-oof-library | CC0 | 22 | 22 | 0.94209 ~ 0.96860 |
| beicicc7 | beicicc 계약 데이터셋 7종 | CC0 2종, CC BY 4.0 5종 | 10 | 9 | 0.96339 ~ 0.96826 |
| hboyang6 | hboyang/s6e8-catstrall-member | CC0 | 6 | 6 | 0.96555 ~ 0.96869 |
| fm5 | raykkretzschmar/s6e8-fm-lattice-blend-members | Apache 2.0 | 7 | 5 | 0.96455 ~ 0.96739 |
| golem | dariushafshar/s6e8-golem-oof-library | CC0 | 7 | 3 | 0.93438 ~ 0.94216 |
| mohan_cat/lgb/xgb | mohankrishnathalla/s6e8-{cat-mlp,lgb-dart,xgb}-oof | CC0 | 3 | 3 | 0.96503 ~ 0.96616 |
| 합계 | | | 226 | 209 | 0.91692 ~ 0.96869 |

선언 AUC가 있는 통과 구성원 200개는 재채점과 최대 `4.96e-06` 차이다.
원본이 소수 5자리인 szymonkapiski 라이브러리의 반올림 오차 크기이고 나머지는 전부 `1e-6` 안이다.
선언 AUC가 없는 9개(mohankrishnathalla 3, hboyang 6)는 재채점값만 남겼다.

통과 상위 10개는 다음과 같다.

| 구성원 | AUC | dtype | 분할 근거 |
| --- | ---: | --- | --- |
| hboyang6:kirill_o1 | 0.968691 | float32 | author_statement |
| szymon74:tabm_seed3 | 0.968673 | float64 | published_code |
| adarsh22:catnative | 0.968601 | float64 | author_statement |
| szymon74:lookup | 0.968526 | float64 | author_statement |
| szymon74:tabm_x12 | 0.968487 | float64 | published_code |
| szymon74:tabm_deeper | 0.968460 | float64 | published_code |
| adarsh22:gxgbcs4 | 0.968451 | float64 | author_statement |
| szymon74:pub_rmlp | 0.968436 | float64 | author_statement |
| adarsh22:gcatnote | 0.968405 | float64 | author_statement |
| hboyang6:koda_exact_te | 0.968404 | float64 | author_statement |

## 제외 17개

| 구성원 | AUC | 사유 |
| --- | ---: | --- |
| szymon74:naji01 ~ naji05 | 0.96367 ~ 0.96881 | 원출처 najiama 데이터셋의 라이선스 불명(szymonkapiski가 CC0로 재게시했으나 원저자 허가가 없음) |
| szymon74:pub_evg | 0.96587 | 10분할 배열(evgendvorkin 단일 LGBM) |
| szymon74:pub_ravi | 0.96651 | 2단계 산출물(ravi20076 L2 스택) |
| bolt47:foldsafe_te_xgb_10f | 0.96843 | 10분할 배열(저자가 5분할→10분할 이동 실험이라고 서술) |
| bolt47:xgb_te_4fold | 0.96791 | 분할 수가 5가 아닌 것으로 이름이 명시, 보수적으로 제외 |
| bolt47:xgb_d7_alt1 | 0.96810 | bolt47:xgb_d7_alt2와 바이트 중복 |
| golem:a, golem:f | 0.96479, 0.96405 | 부분 재현 판정(#174): 검증 fold 조기 종료 낙관 공표 |
| golem:d, golem:e | 0.96260, 0.96486 | 부분 재현 판정(#174): 하이퍼파라미터 부재 |
| fm5:band_band_mid, band_bandfm2 | - | 구간 한정 구성원, 전체 길이가 아니며 저자가 혼합 구성원이 아니라고 명시 |
| beicicc xgb_screen_relations_baseline103 | 0.96585 | xgb_identity_digit_enhanced103과 바이트 중복(같은 저자의 짝 실험 기준선) |

반입하지 않은 출처는 다음과 같다.

| 데이터셋 | 라이선스 | 사유 |
| --- | --- | --- |
| najiama/predicting-smartphone-addiction-oof-submission-csv | unknown | 라이선스 불명, 생성 코드 없음 |
| hboyang/s6e8-150-fusion-local-members | unknown | 라이선스 불명, `candidate_naji16_*` 2단계 산출물 포함 |
| beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts | other | 저자 명시 허가 문구 없음 |
| beicicc/s6e8-fixed900-structural-lgbm-artifacts | other | 저자 명시 허가 문구 없음 |
| beicicc/s6e8-sixmember-crossfit-logitlr-artifacts | other | 2단계 산출물이며 라이선스 other |
| dariushafshar/s6e8-measured-findings-pack | CC0 | 구성원 없음, 분할 대조에만 사용 |

## 통과 구성원의 주의 사항

장부의 `caveats`에 구성원별로 남겼다.
판정 티켓이 절제 실험으로 확인할 수 있게 묶음으로 정리한다.

- **float32 저장 49개**: boltuzamaki 44개 전부와 hboyang 5개.
  szymonkapiski는 float32 하향 변환이 test 행 28%의 순위를 뒤집고 공개 점수 `0.00001`을 잃게 했다고 측정했다.
  단독 AUC는 소수 5자리까지 보존되므로 정합 검증에는 영향이 없다.
- **레시피 비공개 50개**: szymonkapiski의 약한 50개.
  저자가 자체 학습과 고정 5분할을 명시했고 hboyang·szymonkapiski의 공개 스택이 실제로 썼다.
  단독 AUC 0.917~0.957로 전부 약하고, 저자 스스로 처음 30개까지는 스택에 음수 기여였다고 보고했다.
- **전체 자료 TE 누출 의심 2개**: szymon74:pub_rmlp, pub_tabm.
  #174가 원 노트북의 전체 자료 TE를 확인했고 szymonkapiski가 그대로 재실행했다.
  OOF가 낙관적일 수 있으므로 결합기가 과대 가중할 위험이 있다.
- **검증 fold 조기 종료 3개**: mohankrishnathalla cat_v3, lgb_v3, xgb_v3.
  golem a·f와 같은 종류의 낙관이지만 코드가 공개돼 있어 부분 재현 판정은 받지 않았다.
- **이름과 서술이 어긋나는 2개**: hboyang6:kirill_o1, koda_exact_te.
  이름이 다른 공개 노트북 레시피를 가리키지만 README는 독립 학습이라고 서술한다.
  kirill_o1은 통과 구성원 가운데 단독 AUC가 가장 높다.
- **boltuzamaki 44개의 분할 근거**: 스택 노트북이 "5분할 또는 10분할"이라고만 서술한다.
  10분할이 명시된 1개와 4fold가 명시된 1개를 뺐지만, 나머지가 전부 5분할이라는 보장은 저자 서술뿐이다.
  hboyang·adarsh1077·szymonkapiski의 공개 스택은 이 44개를 같은 전제로 쓴다.

## 한계

- 분할 벡터가 있는 9개를 빼면 분할 안전성은 저자 서술에 의존한다.
  분할이 어긋난 배열은 OOF가 실제보다 좋아 보여 결합기가 과대 가중하므로, 판정 티켓은 `fold_evidence`가 `author_statement`인 138개를 뺀 구성을 절제 실험으로 함께 재야 한다.
- 하류 학습이 검증 라벨을 보지 않았다는 사실은 어떤 검사로도 증명되지 않는다.
  이는 #174와 #386이 같은 문장으로 남긴 한계다.
- 재채점 AUC가 선언과 일치하는 것은 정렬을 증명할 뿐 분할을 증명하지 않는다.
- 이 장부는 구성원을 더하는 축만 다룬다.
  결합기 규제 강도와 rank-gauss 변환은 지도의 미정 항목이고 판정 티켓의 사다리 뒤에 정한다.

## 후속으로 넘기는 사실

- 판정 티켓(#443)의 사다리 (2) "자체 35 + #386 재현가능 85"는 장부의 `in_ext85` 플래그로 79개를 고르고, 기준선 재현에 필요한 나머지 6개(pub_evg, baseline103 중복분, beicicc other 4개)는 `scripts/diagnose_external94_width.py`가 이미 읽는 로컬 파일로 채운다.
  other 라이선스 4개는 기준선 재현에만 쓰고 제출 구성에는 넣지 않는다.
- 사다리 (3) "자체 35 + 검증 구성원 전체"는 통과 209개다.
  결합기가 244개 열을 받는다.
- 절제 실험 후보는 float32 49개, 레시피 비공개 50개, TE 누출 의심 2개, 저자 서술만 있는 138개다.
- 조립 티켓(#444)에 넘길 출처·라이선스 표기는 장부의 `dataset`·`license`·`upstream` 필드에 있다.
  CC BY 4.0 6개(beicicc)와 Apache 2.0 5개(raykkretzschmar)는 제출 manifest에 저작자 표기가 필요하다.
