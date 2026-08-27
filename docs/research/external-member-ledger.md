# 확장 스택용 외부 구성원 장부 (판본 2, 이슈 #442·#454)

## 결론

판본 2(2026-08-27, 이슈 #454)는 판본 1(이슈 #442)의 17개 데이터셋 226후보에 #452가 찾은 새 데이터셋 4곳, beicicc의 라이선스 other 데이터셋 2곳, 공개 노트북 30개의 출력물을 더해 **후보 425개 가운데 400개가 확장 스택에 넣을 수 있는 구성원**이다.
판본 1의 통과 209개는 status와 caveats를 그대로 두었고, 판본 2가 새로 통과시킨 구성원은 191개다.
신규 191개는 데이터셋 146개(paiky1995 6, nhtquyn 120, hboyang 150-fusion 11, najiama 재게시 5, beicicc other 3, masayakawamata 1)와 노트북 출력물 45개다.
신규 후보 204개 가운데 13개는 제외했다(10분할 5, 2단계 6, 바이트 중복 2).

라이선스는 판본 2부터 검증 항목이 아니라 기록 항목이다.
통과 400개의 라이선스는 CC0 325개, CC BY 4.0 6개, Apache 2.0 5개, other 3개, unknown 61개이고, 라이선스를 확인할 수 없는 64개(unknown·other 데이터셋 19개, 노트북 출력물 45개)는 `license_unknown` 주의 사항을 달고 사용 한정으로만 쓴다.
사용 한정의 규칙과 근거는 `docs/agents/kaggle-public-notebook-licensing.md`의 "공개 예측 배열의 사용 한정"에 있다.

통과 400개의 단독 AUC는 `0.85317`(nhtquyn gmm)에서 `0.96899`(kodaifukuda0311 RealMLP) 사이다.
kodaifukuda0311 RealMLP는 판본 1 최고 `0.96869`(hboyang6:kirill_o1)보다 높지만 우리 champion `0.9693397`보다는 낮다.
`0.968` 이상은 52개, `0.95` 미만은 181개이며 그중 120개가 nhtquyn의 약한 고전 확률 모델이다.

품질은 폭과 별개다.
통과 400개 가운데 295개가 주의 사항을 달고 있고, 절제 부류는 float32 저장 175개, 레시피 비공개 50개, TE 누출 의심 2개, 저자 서술만 154개, `license_unknown` 64개, `fold_evidence_none` 3개, `near_duplicate_cluster` 67개다.

## 산출물

- 기계가 읽는 장부: `docs/research/external-member-ledger.json` (`version: 2`, `issue: 454`, `history`에 판본 1 기록)
  구성원 425개 전부가 `status`(accepted/excluded), `exclusion_reason`, `caveats`, `in_ext85`, `added_in`(442 또는 454), 파일 경로, dtype, 재채점 AUC, 선언 AUC, #452 조사 대조값(`reference_auc`), 해시, 분할 검사 결과, 다른 통과 구성원과의 스피어만 최댓값(`spearman_max`, `spearman_closest`)을 가진다.
- 생성 도구: `scripts/build_external_member_ledger.py`
  읽기 전용이며 MLflow 실행을 만들지 않고 `artifacts/pool.yaml`을 건드리지 않는다.
  csv·npz로 저장된 노트북 출력물은 하류 도구가 읽을 수 있게 `data/external/ext454/notebooks/<노트북>/normalized/` 아래 float64 npy로 정규화해 두고 장부의 경로가 그 파일을 가리킨다.
  로컬 M-시리즈 14코어에서 약 5분이 걸린다.
- 외부 파일: `data/external/ext94/`(#386 반입분), `data/external/ext442/`(판본 1 반입분), `data/external/ext454/`(판본 2 반입분, 데이터셋 4개와 노트북 출력물 30개, 약 2.0 GB).
  `data/`는 커밋 제외 경로이므로 재현하려면 장부의 `sources`에 적힌 데이터셋과 노트북을 같은 경로에 내려받는다.
  노트북 출력물은 `kaggle kernels output <owner/slug> -p data/external/ext454/notebooks/<owner>_<slug> -o`로 받고, 연속 호출에는 429가 나므로 호출 사이에 4초, 429 뒤에는 45초를 쉰다.

## 조사 범위와 방법

판본 1은 2026-08-27 KST에 Kaggle API로 데이터셋 메타데이터와 파일 목록을 확인하고 로컬에 없던 8개 데이터셋을 내려받았다.
판본 2는 같은 날 #452의 전수 조사 결과를 따라 데이터셋 4개와 노트북 출력물 34개를 내려받았고, 노트북 4개는 두 번 시도해도 배열을 받지 못해 장부에 사유만 남겼다.
분할 근거는 각 데이터셋의 README·설명과 저자 노트북 코드(`kaggle kernels pull`)에서 읽었고, 노트북 출력물의 근거는 #452가 읽은 코드다.

검증 항목은 #386과 같다.

- 행 수 691,369(OOF)와 296,302(test), 유한값
- 우리 라벨로 재채점한 AUC와 저자 선언 AUC의 차이가 `1e-5` 이내
- OOF+test 배열(float64 바이트)의 SHA-256으로 중복 제거, 먼저 적재된 쪽(판본 1이 먼저)을 남김
- 분할 벡터가 동봉된 구성원은 `artifacts/folds.parquet`와 일치

판본 2가 더한 항목은 다음과 같다.

- 선언 AUC가 없는 배열은 재채점 AUC가 `0.8` 이상이어야 정렬된 것으로 본다(정렬이 어긋난 배열은 0.5 근처가 나온다).
- #452 조사가 독립적으로 내려받아 잰 재채점 AUC를 `reference_auc`로 두고 이번 반입분과 대조한다.
  신규 통과 191개 전부가 `3.1e-06` 안에서 같아 조사 뒤 재실행된 노트북은 없다.
- 통과 구성원끼리 OOF 스피어만 순위 상관이 `0.998`을 넘는 쌍을 연결 군집으로 묶고, 대표가 아닌 판본 2 구성원에 `near_duplicate_cluster`를 단다.
  대표는 군집에 판본 1 구성원이 있으면 그중 AUC가 가장 높은 것, 없으면 군집에서 AUC가 가장 높은 판본 2 구성원이다.
  판본 1 구성원의 주의 사항은 바꾸지 않는다.
- csv에 `id` 열이 있는 노트북 출력물은 `train.csv`·`test.csv` id 순서와 대조하고 순열이면 id로 재정렬한다(이번 반입분은 전부 순서가 같아 재정렬한 것이 없다).

지도 #451의 자격 규칙은 지도 #441과 라이선스만 다르다.

- 라이선스 unknown·other 데이터셋과 라이선스 표시가 없는 노트북 출력물도 반입하되 `license_unknown` 주의 사항을 단다.
- 2단계 산출물과 10분할 배열은 제외한다.
- 분할 근거가 없는 구성원은 제외하지 않고 `fold_evidence_none` 주의 사항으로 표시한다.
- 계보 조사(#174)가 재현 불가·부분 재현으로 판정한 구성원은 제외한다.

## 행 순서와 분할 대조

외부 라이브러리는 전부 id 없이 위치로 정렬한다.
다음이 전부 성립한다.

- `artifacts/folds.parquet`의 id 순서 = `train.csv` 파일 순서
- szymonkapiski `train_keys.parquet`의 id와 라벨 = 우리 `train.csv`, `test_keys.parquet`의 id = `test.csv`
- boltuzamaki `train_labels.parquet`의 라벨 = 우리 라벨, parquet의 `id` 열 = `train.csv`·`test.csv` 순서
- dariushafshar 측정 팩의 `folds_seed42.npy` = 우리 5분할과 위치별로 정확히 일치
- beicicc 9개 데이터셋의 `fold_id.npy`(1부터 셈)는 보정 뒤 우리 5분할과 일치
- nhtquyn의 `fold_id.npy`(0부터 셈)는 우리 5분할과 위치별로 정확히 일치
- paiky1995 `oof_predictions.csv`의 `id` = `train.csv` 순서, `test_predictions.csv`의 `id`는 0부터 세는 행 위치이며 npy와 같은 순서

분할 벡터가 있는 통과 구성원은 132개(beicicc 12, nhtquyn 120)이고 나머지 268개는 위치 정렬과 저자의 분할 서술에 의존한다.
장부의 `fold_evidence`가 근거의 종류를 구분한다.

| 근거 | 통과 | 판본 1 | 판본 2 | 뜻 |
| --- | ---: | ---: | ---: | --- |
| `fold_vector` | 132 | 9 | 123 | 분할 벡터 동봉, 우리 분할과 일치 확인 |
| `published_code` | 98 | 61 | 37 | 고정 5분할을 쓰는 훈련 코드가 공개됨 |
| `sibling_code` | 13 | 1 | 12 | 같은 저자의 다른 노트북이 같은 골격으로 고정 5분할 사용 |
| `author_statement` | 154 | 138 | 16 | README 또는 노트북 서술만 있음 |
| `none` | 3 | 0 | 3 | 근거 없음, `fold_evidence_none` 절제 부류 |

## 공급원별 결과

판본 1의 17개 데이터셋 결과는 그대로다.

| 공급원 | 데이터셋 | 라이선스 | 후보 | 통과 | 통과 AUC 범위 |
| --- | --- | --- | ---: | ---: | --- |
| szymon74 | szymonkapiski/s6e8-oof-library-47-models | CC0 (naji 5개는 unknown) | 74 | 72 | 0.91880 ~ 0.96881 |
| szymon_weak50 | szymonkapiski/s6e8-50-weakest-oof-models | CC0 | 50 | 50 | 0.91692 ~ 0.95676 |
| bolt47 | boltuzamaki/s6e8-oof-prediction-library | CC0 | 47 | 44 | 0.93799 ~ 0.96834 |
| adarsh22 | adarsh1077/s6e8-adarsh-oof-library | CC0 | 22 | 22 | 0.94209 ~ 0.96860 |
| beicicc7 | beicicc 계약 데이터셋 7종 | CC0 2종, CC BY 4.0 5종 | 10 | 9 | 0.96339 ~ 0.96826 |
| hboyang6 | hboyang/s6e8-catstrall-member | CC0 | 6 | 6 | 0.96555 ~ 0.96869 |
| fm5 | raykkretzschmar/s6e8-fm-lattice-blend-members | Apache 2.0 | 7 | 5 | 0.96455 ~ 0.96739 |
| golem | dariushafshar/s6e8-golem-oof-library | CC0 | 7 | 3 | 0.93438 ~ 0.94216 |
| mohan_cat/lgb/xgb | mohankrishnathalla/s6e8-{cat-mlp,lgb-dart,xgb}-oof | CC0 | 3 | 3 | 0.96503 ~ 0.96616 |

szymon74의 통과가 67에서 72로 늘어난 것은 판본 1이 라이선스 불명으로 제외했던 najiama 재게시분 naji01~05(0.96367 ~ 0.96881)를 판본 2가 사용 한정으로 통과시켰기 때문이다.
장부에서는 `added_in: 454`, `license: unknown`으로 구분된다.

판본 2가 더한 공급원은 다음과 같다.

| 공급원 | 출처 | 라이선스 | 분할 근거 | 후보 | 통과 | 통과 AUC 범위 |
| --- | --- | --- | --- | ---: | ---: | --- |
| paiky6 | paiky1995/s6e8-oof-library-11-members | CC0 | sibling_code | 11 | 6 | 0.96801 ~ 0.96873 |
| nhtquyn | nhtquyn/s6e8-addiction | CC0 | fold_vector | 120 | 120 | 0.85317 ~ 0.92996 |
| hboyang150 | hboyang/s6e8-150-fusion-local-members | unknown | author_statement | 17 | 11 | 0.96349 ~ 0.96877 |
| masaya | masayakawamata/s6e8-catstr-aug16 | CC0 | none | 1 | 1 | 0.96704 |
| beicicc other 2종 | beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts, s6e8-fixed900-structural-lgbm-artifacts | other | fold_vector | 4 | 3 | 0.96251 ~ 0.96773 |
| szymon74 naji | najiama 재게시분(szymonkapiski 라이브러리 안) | unknown | author_statement | 5 | 5 | 0.96367 ~ 0.96881 |
| nb_* 30개 | 공개 노트북 출력물 | unknown(출력물 표시 없음) | published_code 25, sibling_code 6, none 2 | 46 | 45 | 0.91135 ~ 0.96899 |
| 합계 | | | | 204 | 191 | 0.85317 ~ 0.96899 |

선언 AUC가 있는 신규 통과 구성원(paiky1995 6, nhtquyn 120, zhukovoleksiy 3, rv1922 4)은 재채점과 최대 `1e-6` 차이다.
나머지 58개는 선언값이 없어 재채점값과 #452 조사값의 일치만으로 정렬을 확인했다.

### 노트북 출력물 45개

| 노트북 | 장부 키 | 구성원과 재채점 AUC | 분할 근거 | 비고 |
| --- | --- | --- | --- | --- |
| kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only | nb_kodaifukuda | realmlp `0.96899` | published_code | 통과 400개 가운데 최고, 원자료 분포 통계 참조, 공개 점수 0.97016 |
| omidbaghchehsaraei ft-transformer, cnn, tabtransformer, fastai, xgboost-v2, catboost | nb_omid_* | `0.96657`, `0.96771`, `0.96747`, `0.96676`, `0.96873`, `0.96715` | sibling_code | cnn·tabtransformer는 고유값 4,000대 |
| zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline | nb_zhukov | cat_base `0.96799`, lgb02 `0.96836`, xgb_base `0.96786` | published_code | manifest.csv 선언값과 일치 |
| redamountassir s6e8-lgbm-lb-0-96965, s6e8-histgradientboosting-lb-0-96945 | nb_reda_lgbm, nb_reda_hgb | lgbm `0.96826`, hgb `0.96803` | published_code | kirill0212 스택의 e-* 입력 |
| yaminh/smartphone-addiction-prediction-strong-eda-cv-eble | nb_yaminh | lgbm_te `0.96751`, xgb_te `0.96773`, catboost `0.96193` | published_code | 앙상블 열 제외 |
| sidhaarthshree/lightgbm-ensemble-based-on-eda | nb_sidhaarth | lgb_a `0.96765`, lgb_b `0.96756`, xgb `0.96774` | published_code | lgb_b는 lgb_a와 0.99835 |
| yekenot/ps-s6-e8-trompt-pytorch-frame | nb_yekenot | trompt `0.96667` | published_code | 장부에 없는 계열 |
| lucymlai32 phase-2-xgboost-and-model-blending, smartphone-addiction-prediction | nb_lucy_xgb, nb_lucy_cat | xgboost `0.96580`, catboost_v2 `0.96438` | published_code | |
| cdeotte simple-xgb-starter, simple-cat-starter, simple-nn-starter | nb_cdeotte_* | `0.96481`, `0.96291`, `0.93974` | published_code | xgb는 dariushafshar 변형과 0.99876 |
| dariushafshar/0-97184-leader-xgb-feature-ablation | nb_darius_ablation | xgb `0.96488` | published_code | cdeotte XGB 변형 |
| rv1922/smartphone-addiction | nb_rv1922 | lgbm v1 `0.96326`, v2 `0.96374`, v3 `0.96403`, xgb seed42 `0.96482` | published_code | manifest.json 선언값과 일치, lgbm v1·v2는 v3와 군집 |
| yadoy666/predicting-smartphone-addiction | nb_yadoy | catboost `0.96370`, xgboost `0.96471` | published_code | fmdeep은 fm5:fmdeep과 같은 배열이라 반입 안 함 |
| danushkumarv/smartphone-addiction-gbm-rank-blend-nb01 | nb_danush | lgb `0.96395`, xgb `0.96447`, cb `0.96271` | published_code | 스택 출력 제외 |
| harwindersingh766/ps-s6e8-xgboost-te-lb-0-96548 | nb_harwinder | xgb `0.96410` | published_code | `_sb` 판 제외 |
| dynamo14324/smartphone-addiction-championship-v11 | nb_dynamo | lgb `0.96337`, xgb `0.96322` | published_code | 의사 라벨 판 제외 |
| mohankrishnathalla s6e8-realmlp-oof-saver, s6e8-tabm-oof-saver | nb_mohan_realmlp, nb_mohan_tabm | realmlp `0.95813`, mlp `0.94142` | published_code | 제목이 TabM인 노트북의 코드는 MLP |
| kava1/predicting-smartphone-addiction-resnet-fe | nb_kava1 | resnet `0.95687` | published_code | |
| lopure/hdviz-pca-parallel-with-linear-svm | nb_lopure | linear `0.91135`, poly `0.92880`, rbf `0.92217` | published_code | rbf는 최근접 상관 0.924로 가장 다름 |
| shamanthakreddymallu/s6e8-baseline | nb_shaman_baseline | lgb_fe `0.96377`, lr `0.93661` | none | 코드에서 시드를 찾지 못함, `fold_evidence_none` |
| lavanyabacche/xgb-starter-01 | nb_lavanya | xgb_starter `0.96481` | published_code | cdeotte XGB와 바이트 중복이라 제외 |

통과 상위 10개는 다음과 같다.

| 구성원 | AUC | dtype | 분할 근거 | 판본 |
| --- | ---: | --- | --- | --- |
| nb_kodaifukuda:realmlp | 0.968993 | float64 | published_code | 2 |
| szymon74:naji05 | 0.968815 | float64 | author_statement | 2 |
| szymon74:naji03 | 0.968814 | float64 | author_statement | 2 (naji05와 0.99999) |
| hboyang150:local_tabm_rich_alt | 0.968768 | float64 | author_statement | 2 (szymon74:tabm_seed3와 0.99885) |
| nb_omid_xgb2:xgboost_v2 | 0.968733 | float64 | sibling_code | 2 |
| paiky6:v14_lookup_bag | 0.968727 | float64 | sibling_code | 2 |
| szymon74:naji04 | 0.968722 | float64 | author_statement | 2 |
| hboyang6:kirill_o1 | 0.968691 | float32 | author_statement | 1 |
| szymon74:tabm_seed3 | 0.968673 | float64 | published_code | 1 |
| hboyang150:local_tabm_rich_seed3 | 0.968669 | float64 | author_statement | 2 (szymon74:tabm_seed3와 0.99859) |

## 제외 25개

판본 1의 제외 12개는 그대로다(naji 5개는 통과로 바뀜).

| 구성원 | AUC | 사유 |
| --- | ---: | --- |
| szymon74:pub_evg | 0.96587 | 10분할 배열(evgendvorkin 단일 LGBM) |
| szymon74:pub_ravi | 0.96651 | 2단계 산출물(ravi20076 L2 스택) |
| bolt47:foldsafe_te_xgb_10f | 0.96843 | 10분할 배열(저자가 5분할→10분할 이동 실험이라고 서술) |
| bolt47:xgb_te_4fold | 0.96791 | 분할 수가 5가 아닌 것으로 이름이 명시, 보수적으로 제외 |
| bolt47:xgb_d7_alt1 | 0.96810 | bolt47:xgb_d7_alt2와 바이트 중복 |
| golem:a, golem:f | 0.96479, 0.96405 | 부분 재현 판정(#174): 검증 fold 조기 종료 낙관 공표 |
| golem:d, golem:e | 0.96260, 0.96486 | 부분 재현 판정(#174): 하이퍼파라미터 부재 |
| fm5:band_band_mid, band_bandfm2 | - | 구간 한정 구성원, 전체 길이가 아니며 저자가 혼합 구성원이 아니라고 명시 |
| beicicc xgb_screen_relations_baseline103 | 0.96585 | xgb_identity_digit_enhanced103과 바이트 중복(같은 저자의 짝 실험 기준선) |

판본 2의 제외 13개는 다음과 같다.

| 구성원 | AUC | 사유 |
| --- | ---: | --- |
| paiky6:v19_lookup_bag_10f, v21_realmlp_10f, v22_mlp_enc_10f, v23_lookup_aug_10f, v24_cat_enc_10f | 0.96551 ~ 0.96891 | 10분할 배열(저자가 `_10f` 접미사와 설명으로 명시) |
| hboyang150:candidate_naji16_* 6개 | 0.96981 ~ 0.96987 | 2단계 산출물(najiama 16구성원 융합 후보) |
| beicicc fixed4000 baseline | 0.96730 | exact_value_catboost_fixed4000(CC BY 4.0 데이터셋)과 바이트 중복 |
| nb_lavanya:xgb_starter | 0.96481 | nb_cdeotte_xgb:xgb와 바이트 중복(포크) |

반입하지 않은 데이터셋과 노트북은 장부의 `sources_not_imported`와 `notebooks_not_imported`에 사유와 함께 있다.
요지는 다음과 같다.

| 출처 | 사유 |
| --- | --- |
| najiama/predicting-smartphone-addiction-oof-submission-csv (unknown) | 단일 5개는 szymon74:naji01~05로 같은 배열을 보유, 2단계 11개 제외 |
| szymonkapiski/s6e8-oof-library-25-models (CC0) | 25쌍 전부 장부 szymon74와 float32 바이트 동일 |
| tamerlanomralinov/s6e8-full-best-blend-npy (CC0) | 10분할 9개 |
| raykkretzschmar/s6e8-transductive-anti-student-signals (CC0) | 교사 스택 증류(2단계 의심), 시드·행 순서 미명시 |
| atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124 (unknown), wellkilo (other), beicicc sixmember (other) | 2단계 |
| kenchanhodgkin 16개, thisray, 제출 파일만 있는 7개 | 시험 예측 또는 OOF 없음 |
| omidbaghchehsaraei lookup-transformer, yadoy666 fmdeep | 장부 hboyang6:kirill_o1, fm5:fmdeep과 같은 배열 |
| najiama single-lgbm, kirill0212 public-ensemble, 2단계 스택 노트북 14개 | 의사 라벨·재수출·2단계 |
| stephentarter 4개, ern711, darkmatternet, lavanyabacche catboost, dranilkumardubey, yusufmurtaza01, destroyer123787, magurodataanalysis, rv1922 xgb seed777·2026, harwindersingh766 `_sb`, kodaifukuda0311 xgb, zhenruiweng | 커뮤니티 5분할이 아님(다른 시드, 반복 분할, 시드 평균) |
| factualexplorer, tamerlanomralinov insights, evgendvorkin, echloeprice | 10분할 |
| vladstud716373618 | 재채점 AUC 0.49997로 정렬 불명 |
| shashwat1729/s6e8-lookup-pair-transformer | 출력 내려받기 권한 거부(403), #452·#454에서 두 번 시도 |
| mhamza0810 xgb, udaken10 xgb, shamanthakreddymallu lightgbm v2 | 파일 목록에는 OOF·시험 파일이 있으나 `kernels output`이 실행 기록만 내려줌, 두 번 시도 |

## 통과 구성원의 주의 사항

장부의 `caveats`에 구성원별로 남겼다.
판정 티켓이 절제 실험으로 확인할 수 있게 묶음으로 정리한다.
판본 2 부류는 주의 사항 문자열의 첫 낱말(`license_unknown:`, `fold_evidence_none:`, `near_duplicate_cluster:`)이 부류 이름이다.

- **`license_unknown` 64개**: hboyang 150-fusion 11, najiama 재게시 5, beicicc other 3, 노트북 출력물 45.
  결합 입력으로만 쓰고 재배포·커밋·자체 산출물 첨부는 하지 않는다.
  hboyang의 같은 저자 catstrall 데이터셋은 CC0이므로 표기를 요청하면 풀릴 가능성이 있다.
- **`fold_evidence_none` 3개**: masayakawamata cat_str(0.96704), shamanthakreddymallu lgb_fe(0.96377)·lr(0.93661).
  분할 근거가 없으므로 분할이 어긋났다면 OOF가 낙관적일 수 있다.
  판정 티켓은 이 3개를 뺀 구성을 절제로 함께 재고, 포함 여부는 #456이 정한다.
- **`near_duplicate_cluster` 67개**: 통과 구성원끼리 스피어만 `0.998`을 넘는 쌍이 179개, 연결 군집이 38개다.
  nhtquyn 120개 안에서는 138쌍이 62개 군집으로 묶여 대표 62개를 빼고 58개가 표시됐다(#452 조사는 float32 계산으로 64개 군집이라고 적었다).
  나머지 9개는 naji02(대표 naji04, 0.99862), naji03(대표 naji05, 0.99999), hboyang150 local_tabm_rich_alt·seed3·seed909(대표 szymon74:tabm_seed3, 0.99833 ~ 0.99885), sidhaarth lgb_b(대표 lgb_a), cdeotte xgb(대표 dariushafshar 변형, 0.99876), rv1922 lgbm v1·v2(대표 v3)다.
  판본 1 구성원끼리도 0.998을 넘는 쌍이 있어 31개가 해당하지만(szymon_weak50 m23~m31, adarsh22 gcat·gxgb 계열, bolt47 xgb_dd 계열 등) 판본 1의 주의 사항은 바꾸지 않았고 `spearman_max`·`spearman_closest` 필드로만 볼 수 있다.
- **float32 저장 175개**: 판본 1의 49개(boltuzamaki 44, hboyang 5)에 nhtquyn 120개와 hboyang 150-fusion 6개가 더해졌다.
  szymonkapiski는 float32 하향 변환이 test 행 28%의 순위를 뒤집고 공개 점수 `0.00001`을 잃게 했다고 측정했다.
- **양자화 흔적 5개**: paiky1995 v13·v15·v16은 고유값이 약 560개로 float16으로 표현 가능한 값만 있고, omidbaghchehsaraei cnn·tabtransformer는 고유값이 4,000대다.
  순위 기반 결합기는 이 구성원에서 대량의 동점을 본다.
- **nhtquyn 120개의 약함**: 단독 AUC `0.853 ~ 0.930`으로 판본 1의 가장 약한 구성원(`0.91692`)보다 약하고 설명·코드가 없어 학습 자료 범위를 확인할 수 없다.
  분할 벡터는 우리 분할과 정확히 일치한다.
- 판본 1의 부류는 그대로다: 레시피 비공개 50개(szymon_weak50), 전체 자료 TE 누출 의심 2개(szymon74:pub_rmlp·pub_tabm, #444가 제외), 검증 fold 조기 종료 3개(mohankrishnathalla), 이름과 서술이 어긋나는 2개(hboyang6:kirill_o1·koda_exact_te), boltuzamaki 44개의 분할 근거 서술.

## 한계

- 분할 벡터가 있는 132개를 빼면 분할 안전성은 저자 서술이나 코드 읽기에 의존한다.
  분할이 어긋난 배열은 OOF가 실제보다 좋아 보여 결합기가 과대 가중하므로, 판정 티켓은 `fold_evidence`가 `author_statement`인 154개와 `none`인 3개를 뺀 구성을 절제 실험으로 함께 재야 한다.
- 하류 학습이 검증 라벨을 보지 않았다는 사실은 어떤 검사로도 증명되지 않는다.
  이는 #174와 #386이 같은 문장으로 남긴 한계다.
- 재채점 AUC가 선언·조사값과 일치하는 것은 정렬을 증명할 뿐 분할을 증명하지 않는다.
- omidbaghchehsaraei 6개는 코드를 읽지 않고 같은 저자의 두 노트북 코드로 분할을 추정했다(`sibling_code`).
- 노트북 출력물은 저자가 노트북을 다시 실행하면 바뀔 수 있다.
  장부의 `reference_auc`·`sha256`으로 이번 반입분을 고정했고, 다시 내려받을 때는 두 값이 같은지 확인해야 한다.
- 이 장부는 구성원을 더하는 축만 다룬다.
  191개를 더한 사다리의 값어치는 판정 티켓이 절제 실험과 함께 재야 한다.

## 후속으로 넘기는 사실

- 판정 티켓(#455)의 현재 판 재현 `own35_ext207`은 `added_in == 442`이고 `status == "accepted"`인 209개에서 TE 누출 의심 2개(szymon74:pub_rmlp·pub_tabm)를 뺀 207개다.
  `scripts/judge_extended_stack.py`의 `load_ledger`는 통과 수를 209로 단언하므로 판본 2를 읽으려면 그 단언을 `added_in` 기준 선택으로 바꿔야 한다.
- 신규 전체는 `added_in == 454`인 통과 191개이고 결합기가 자체 35 + 외부 400 = 435개 열을 받는다.
- 절제 부류는 `caveats` 문자열의 첫 낱말로 고른다: `license_unknown`(64), `fold_evidence_none`(3), `near_duplicate_cluster`(67), 판본 1 부류 `float32 저장`(175)·`레시피·모델 종류 비공개`(50)·TE 누출(2)과 `fold_evidence == author_statement`(154).
  nhtquyn을 대표 62개로 줄인 구성은 `near_duplicate_cluster`가 없는 nhtquyn 구성원으로 만든다.
- 공급원별 절제는 `source` 필드로 고른다(paiky6, nhtquyn, hboyang150, masaya, beicicc other 2종, szymon74 naji, nb_*).
- 조립 티켓(#457)에 넘길 출처·라이선스 표기는 장부의 `dataset`·`license`·`upstream` 필드에 있다.
  CC BY 4.0 6개(beicicc)와 Apache 2.0 5개(raykkretzschmar)는 제출 manifest에 저작자 표기가 필요하고, `license_unknown` 구성원은 manifest에 출처를 적되 배열은 첨부하지 않는다.
- nhtquyn 120개와 paiky1995 6개는 `data/external/ext454/`의 원본 npy를, 노트북 출력물의 csv·npz는 `normalized/` 아래 npy를 장부 경로가 가리킨다.
  `load_ledger_array`가 그대로 읽는다.
