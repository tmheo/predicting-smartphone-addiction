# 공개 OOF 라이브러리 재조사 (2026-08-27)

이 문서는 [이슈 #452](https://github.com/tmheo/predicting-smartphone-addiction/issues/452)의 답이다.
외부 구성원 장부(이슈 #442, `docs/research/external-member-ledger.md`)가 반입한 17개 데이터셋 밖에 어떤 공개 OOF·시험 예측 쌍이 더 있는지, 그중 무엇이 확장 스택 제출의 외부 구성원이 될 수 있는지를 2026-08-27 KST에 조사한 기록이다.
조사는 읽기 전용이며 `artifacts/`, 장부, `data/external/`을 바꾸지 않았고 Kaggle 제출도 하지 않았다.
내려받은 파일은 저장소 밖 임시 경로에만 두었으므로 반입하려면 아래 재실행 레시피로 다시 받아야 한다.

자격 규칙은 장부와 한 가지가 다르다.
외부 구성원은 재배포하지 않고 결합기 입력으로만 쓰므로, 이번 조사는 라이선스가 unknown·other인 데이터셋과 라이선스 표시가 없는 노트북 출력물도 후보에 넣고 "사용 한정"으로 표시한다.
나머지 조건은 장부와 같다.
OOF 691,369행과 시험 예측 296,302행이 모두 있고, 커뮤니티 5분할(`StratifiedKFold(5, shuffle=True, random_state=42)`, `train.csv` 원본 행 순서)의 근거가 있으며, 2단계 산출물과 10분할 배열이 아니어야 한다.

## 결론

장부 밖에서 지금 바로 쓸 수 있는 신규 외부 구성원은 **189개**이고, 그중 데이터셋 출처가 146개, 노트북 출력물이 43개다.
알려진 중복 3개를 빼면 실질 186개다.

데이터셋 146개는 다섯 출처에서 나온다.

- `paiky1995/s6e8-oof-library-11-members`(CC0, 08-25)의 5분할 신경망 6개는 단독 AUC `0.96801`~`0.96873`으로 장부 상위권과 같은 급이다.
  분할 근거는 데이터셋 설명의 명시 서술과 같은 저자 노트북의 고정 5분할 코드다.
- `nhtquyn/s6e8-addiction`(CC0, 08-22)의 고전 확률 모델 120개는 동봉된 분할 벡터가 `artifacts/folds.parquet`와 위치별로 완전히 일치한다.
  다만 단독 AUC가 `0.85317`~`0.92996`으로 장부의 가장 약한 구성원(`0.91692`)보다 더 약하고, 스피어만 순위 상관 0.998 기준 내부 중복 군집이 64개뿐이라 기여는 절제 실험으로 따로 재야 한다.
- `hboyang/s6e8-150-fusion-local-members`(unknown, 사용 한정)의 단일 모델 11개는 `0.96349`~`0.96877`이고 README가 커뮤니티 분할을 명시한다.
  `local_tabm_rich_*` 3개는 장부의 `szymon74:tabm_seed3`와 스피어만 순위 상관이 0.998을 넘어 실질 신규는 8개다.
- najiama의 단일 모델 5개(`0.96367`~`0.96881`, unknown, 사용 한정)는 장부에 `szymon74:naji01~05`로 이미 반입돼 있고 제외 사유가 라이선스뿐이었으므로 파일을 새로 받을 필요가 없다.
- beicicc의 other 라이선스 4개(`0.96251`~`0.96773`, 사용 한정)는 분할 벡터가 일치하는 구성원으로 `data/external/ext94/beicicc/`에 이미 있고 #386 기준선 재현에만 쓰였다.

노트북 출력물 43개는 공개 노트북 500개의 출력 파일을 훑어 찾은 단일 모델 OOF·시험 쌍 가운데, 코드가 커뮤니티 5분할을 쓰고 배열을 내려받아 재채점까지 통과한 것이다.
단독 AUC는 `0.91135`(lopure 선형 SVM)~`0.96899`(kodaifukuda0311 RealMLP)이고, kodaifukuda0311의 RealMLP는 장부 최고 `0.96869`보다 높은 외부 구성원이다.
출력 파일에는 라이선스 표시가 없으므로 전부 사용 한정이다.

분할 근거가 없어 막힌 것은 `masayakawamata/s6e8-catstr-aug16` 1개(CC0, `0.96704`)와 shamanthakreddymallu 기준선 노트북 2개다.
`raykkretzschmar/s6e8-transductive-anti-student-signals`의 `soft_student`(CC0, `0.96736`)와 najiama의 단일 LightGBM 노트북(`0.96862`)은 각각 교사 스택 증류와 혼합 제출에서 수확한 의사 라벨에 기대므로 2단계로 본다.
10분할 14개(tamerlanomralinov 9, paiky1995 5), 시험 예측이 없는 kenchanhodgkin 16개 데이터셋, 제출 파일만 있는 7개 데이터셋은 규칙상 쓸 수 없다.
노트북 4개(shashwat1729 lookup-pair, mhamza0810 XGB, udaken10 XGB, shamanthakreddymallu LightGBM v2)는 출력 파일을 내려받지 못해 미확인이다.

2026-08-27 이후 갱신된 S6E8 데이터셋은 없다.

후속으로 넘길 판단은 두 가지다.
데이터셋 146개와 노트북 출력물 43개를 장부에 사용 한정으로 반입하고 사다리(자체 35 + 209 + 146, + 43)로 nested OOF를 다시 재는 일, 그리고 라이선스가 없는 구성원을 확장 스택 제출에 넣을지에 대한 사용 정책을 확정하는 일이다.

## 조사 범위와 방법

Kaggle CLI 2.2.4로 2026-08-27 KST에 조사했다.

- 데이터셋 열거는 `kaggle datasets list -s <검색어> --sort-by updated --page-size 100 -p <쪽> -v`로 했다.
  검색어 `s6e8`은 3쪽 57개에서 끝났고, `oof`, `smartphone addiction`, `playground s6e8`, `playground-series-s6e8`, `blend members`, `oof library`, `stack members`, `smartphone oof`, `addiction oof`, `s6e8 stack`, `s6e8 oof`, `s6e8 members`, `s6e8 predictions`, `s6e8 test preds`, `s6e8 artifacts`, `s6e8 submission`을 각 3쪽씩 더해 고유 ref 124개를 모았다.
- 파일 목록과 라이선스는 `kaggle datasets files <ref> --page-size 200 --format json`과 `kaggle datasets metadata <ref> -p <경로>`로 읽었다.
- 노트북은 `kaggle kernels list --competition playground-series-s6e8 --sort-by dateRun --page-size 100 -p 1..5 -v`(500개)와 저자별 `kaggle kernels list --user <저자>`로 모았다.
  출력 파일 목록은 `kaggle kernels files <ref> --format json`으로 500개 전부 읽었고(shashwat1729의 6개는 403으로 실패), 코드는 `kaggle kernels pull <ref> -p <경로> -m`으로 60여 개를 읽었으며, 배열은 `kaggle kernels output <ref> -p <경로> -o`로 46개 노트북에서 내려받았다.
- 12개 데이터셋(nhtquyn, paiky1995, masayakawamata, tamerlanomralinov, raykkretzschmar transductive, hboyang 150-fusion, szymonkapiski 25-models, atakanaldemir, thisray, wellkilo, masha6574, giorgosi)을 내려받아 검증했다.
- 검증 항목은 장부와 같다.
  행 수 691,369(OOF)와 296,302(test), 유한값, 우리 라벨 재채점 AUC와 선언 AUC의 차이, OOF+test float64 바이트 SHA-256의 장부 226개 해시와의 중복, 동봉 분할 벡터의 `artifacts/folds.parquet` 대조다.
  여기에 장부 통과 209개와의 스피어만 순위 상관 최댓값(0.998 중복 기준)과, 가장 가까운 장부 구성원과의 시험 예측 순위 상관(시험 행 순서 방증)을 더했다.
- `artifacts/folds.parquet`는 `StratifiedKFold(5, shuffle=True, random_state=42)`를 `train.csv` 순서로 다시 만든 벡터와 일치함을 먼저 확인했다.

## 후보표

| 데이터셋 | 라이선스 | 갱신 | 구성원 | OOF·test | dtype | 분할 근거 | 2단계·10분할 | 판정 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| paiky1995/s6e8-oof-library-11-members | CC0 | 08-25 | 11 | 둘 다 | float64 | author_statement + sibling_code | 10분할 5 | **5분할 6개 사용 가능**, 10분할 5개 제외 |
| nhtquyn/s6e8-addiction | CC0 | 08-22 | 120 | 둘 다 | float32 | fold_vector | 없음 | **120개 사용 가능**, 약하고 내부 중복 많음 |
| hboyang/s6e8-150-fusion-local-members | unknown | 08-18 | 17 | 둘 다 | float64 11, float32 6 | author_statement | 2단계 6 | **단일 11개 사용 가능(사용 한정)**, 3개는 szymon74:tabm_seed3와 중복 |
| najiama/predicting-smartphone-addiction-oof-submission-csv | unknown | 08-23 | 16 | 둘 다 | csv | author_statement | 2단계 11 | **단일 5개 사용 가능(사용 한정)**, 장부 szymon74:naji01~05 파일 그대로 |
| beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts, s6e8-fixed900-structural-lgbm-artifacts | other | 08-08 | 4 | 둘 다 | float64 | fold_vector | 없음 | **4개 사용 가능(사용 한정)**, `data/external/ext94/beicicc/`에 있음 |
| masayakawamata/s6e8-catstr-aug16 | CC0 | 08-16 | 1 | 둘 다 | float64 | none | 없음 | 불가, 분할 근거 없음 |
| raykkretzschmar/s6e8-transductive-anti-student-signals | CC0 | 08-19 | 1 (+시험 전용 신호 7) | 둘 다 | float64 | author_statement, 시드 미명시 | 교사 증류(2단계) | 보류, 2단계 의심 |
| tamerlanomralinov/s6e8-full-best-blend-npy | CC0 | 08-19 | 9 | 둘 다 | float64 | published_code | 10분할 9 | 불가, 10분할 |
| atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124 | unknown (README는 CC0 표기) | 08-24 | 1 | 둘 다 | csv float64 | author_statement | 2단계(244구성원 메타모델) | 불가, 2단계 |
| wellkilo/s6e8-evidence-first-soft-student-assets | other | 08-20 | 0 | 순위 벡터만 | int | - | 2단계 | 불가 |
| szymonkapiski/s6e8-oof-library-25-models | CC0 | 08-01 | 25 | 둘 다 | float32 | published_code | pub_ravi 2단계 | 신규 없음, 25개 전부 장부 szymon74와 float32 바이트 동일 |
| kenchanhodgkin/pg-s6e8-exp000~exp012 child (16개) | CC0 | 08-11~08-16 | 각 1 | OOF만 | csv | - | - | 불가, 시험 예측 없음 |
| anhadmahajan06/ps-s6e8predicting-smartphone-addiction-submission | Apache 2.0 | 08-10 | 0 | 제출 20 | csv | - | - | 불가, OOF 없음 |
| thisray/s6e8-our-component | CC0 | 08-18 | 0 | 시험 전용 1 | csv | - | - | 불가, OOF 없음 |
| anthonytherrien/predicting-smartphone-addiction-vault | MIT | 08-25 | 0 | 제출 2(타인 제출 재게시) | csv | - | - | 불가 |
| najiama/s6e8-psa | unknown | 08-16 | 0 | 제출 4 | csv | - | - | 불가 |
| souvikdbiswas/smartphone-addiction-best-ot-datas | Apache 2.0 | 08-25 | 0 | 제출 2 | csv | - | - | 불가 |
| qamrodz/prediction-smartphone-addiction-submission | Apache 2.0 | 08-18 | 0 | 제출 1 + 원자료 | csv | - | - | 불가 |

구성원이 없는 나머지는 다음과 같다.
`dariushafshar/kaggle-competition-leaderboard-intelligence`(Apache 2.0, 08-26)와 `georgymamarin/playground-series-s6-leaderboards`는 리더보드 패널, `dynamo14324/s6e8-engineered-features-pre-processed-data`(CC0)는 특성 표, `deveshkadam969/smartphone-addiction`(Apache 2.0)은 노트북 4개, `zaidshaikh203/predicting-smartphone-addiction`(MIT)은 `sample_submission.csv`뿐이다.
`abbasi1214`, `sachin1228`, `nh2nam`, `prakharszn`, `navazshfathi`의 데이터셋은 대회 원자료 복제(20.7 MB)다.
`masha6574/cimt-final-oof`(08-27)는 2,903행짜리 다른 대회 OOF이고 `giorgosi/xgb-meta-model-oof-json`은 XGBoost 모델 JSON이다.

## 검증 결과

### nhtquyn/s6e8-addiction

- `fold_id.npy`는 int8 (691,369,)이고 값은 0~4, 분포는 138,274 x 4 + 138,273이다.
  `artifacts/folds.parquet`와 위치별로 완전히 일치하며 0부터 세므로 beicicc처럼 보정할 필요가 없다.
- `oof.npy`는 (691,369, 120) float32, `test.npy`는 (296,302, 120) float32이고 `members.csv` 120행과 열 수가 맞으며 전부 유한값이다.
- 120열 전부 재채점하니 선언 `solo_oof_auc`와의 차이가 최대 `4.87e-08`이다.
- 단독 AUC는 `0.85317`(g094_gmm_k2_covdiag_rg_all)~`0.92996`(g015_binned_nb_bins64_all)이고, 0.92 이상이 24개, 0.90 미만이 10개다.
- 계열은 qda 39, gmm 36, gnb 20, binned_nb 15, lda 10이고, 관점은 core·core_cat·imp_ratio·screen·all 5종, 변환은 none·std·rg다.
- 장부 통과 209개와의 스피어만 순위 상관 최댓값은 `0.98603`(g037_lda_std_core_cat 대 szymon74:logreg)이고 바이트 중복은 없다.
- 내부에서는 0.998을 넘는 쌍이 135개이고, 0.998 기준 연결 군집은 64개(0.995 기준 50개, 0.99 기준 39개)다.
  예를 들어 core와 core_cat 관점, GNB의 smoothing 두 값은 사실상 같은 구성원이다.
- 분할별 AUC 예시는 g001이 `[0.91453, 0.91411, 0.91483, 0.91488, 0.91449]`로 분할 간 편차가 작다.
- 시험 예측은 g037의 경우 szymon74:logreg의 시험 예측과 순위 상관 `0.98737`로 OOF 쪽(`0.98603`)과 같은 수준이라 시험 행 순서가 `test.csv` 순서라는 방증이 된다.
- 설명·README·코드가 없고 저자의 공개 노트북도 없다.
  float32 저장이므로 장부의 float32 주의 사항 부류에 들어간다.

### paiky1995/s6e8-oof-library-11-members

- 11쌍 전부 float64 (691,369,)와 (296,302,)이고 유한값이다.
  `oof_predictions.csv`와 `test_predictions.csv`의 `id`는 `train.csv`·`test.csv` 순서와 같고 npy와의 최대 차이는 `5e-10`(csv는 소수 9자리)이다.
- 5분할 6개의 재채점 AUC는 선언과 `1e-6` 안에서 같다.
  v10_tabm `0.9680063`, v13_lookup `0.9682934`, v14_lookup_bag `0.9687267`, v15_lookup_wide `0.9681470`, v16_lookup_aug `0.9681545`, v17_realmlp `0.9682819`다.
- 10분할 5개는 v19_lookup_bag_10f `0.9689077`, v21_realmlp_10f `0.9685624`, v22_mlp_enc_10f `0.9655108`, v23_lookup_aug_10f `0.9688452`, v24_cat_enc_10f `0.9684448`이며 규칙상 제외한다.
- 고유값 수가 v13 563, v15 560, v16 564로 float16으로 표현 가능한 값만 있어 저장 과정에서 float16 양자화를 거친 흔적이 있고, v14는 11,605(3시드 평균), v10은 466,437, v17은 521,546이다.
  순위 기반 결합기는 이 세 구성원에서 대량의 동점을 보게 된다.
- 장부 대비 스피어만 순위 상관 최댓값은 v10 `0.99380`(adarsh22:gxgbcs4), v13 `0.96809`, v14 `0.97256`, v15 `0.96622`, v16 `0.96492`(모두 szymon74:lookup), v17 `0.98293`(adarsh22:gxgbcs4)이고 내부 최댓값은 `0.99517`이다.
  v10의 시험 예측 순위 상관은 `0.99613`으로 OOF와 일관된다.
- 분할 근거는 두 가지다.
  데이터셋 설명이 "`StratifiedKFold(shuffle=True, random_state=42)` on that order, `_10f`는 10분할이고 나머지는 5분할, TE와 분위 변환은 학습 fold 안에서 재적합, float64"라고 명시한다.
  같은 저자의 노트북 `paiky1995/s6e8-correlation-does-not-predict-contribution`(08-25)이 같은 Lookup-Transformer 골격을 `StratifiedKFold(5, shuffle=True, random_state=42)`로 학습해 OOF·시험 csv를 출력하므로 장부 분류로는 `sibling_code`다.
- 노트북은 Lookup-Transformer 구조가 `tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041`에서 왔고 자기 분할로 다시 학습했다고 밝힌다.

### hboyang/s6e8-150-fusion-local-members

- 17쌍이고 README는 float64라고 하지만 6개(cat_fresh_d9_s606, lookup_fresh_d256_l8_s5150, lookup_fresh_d384_l6_s2718, xgb_fresh_d6_s606, xgb_fresh_d7_s314159, local_lookup_d384_l4)는 float32다.
- 2단계 6개(`candidate_naji16_*`)는 AUC `0.96981`~`0.96987`이다.
- 단일 11개의 재채점 AUC는 realmlp_fresh_s2026 `0.96349`, xgb_fresh_d6_s606 `0.96567`, xgb_fresh_d7_s314159 `0.96605`, lookup_fresh_d256_l8_s5150 `0.96732`, lookup_fresh_d384_l6_s2718 `0.96750`, local_lookup_d384_l4 `0.96757`, cat_fresh_d9_s606 `0.96766`, tabm_fresh_rich_s2026 `0.96827`, local_tabm_rich_seed909 `0.96863`, local_tabm_rich_seed3 `0.96867`, local_tabm_rich_alt `0.96877`이다.
- 장부 대비 스피어만 순위 상관은 local_tabm_rich_alt `0.99885`, seed3 `0.99859`, seed909 `0.99833`(모두 szymon74:tabm_seed3)로 3개가 0.998을 넘고, tabm_fresh는 `0.99675`(szymon74:tabm_deeper), cat_d9는 `0.99659`(bolt47:cat_dual_view)다.
  local_tabm_rich_alt의 시험 예측 순위 상관은 `0.99967`이다.
- 장부의 hboyang6(catstrall 6개)와 바이트 중복은 없다.
- README가 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`와 원본 행 순서를 명시하므로 근거는 `author_statement`다.
- Kaggle 메타데이터 라이선스가 unknown이라 사용 한정이다.
  같은 저자가 08-24에 낸 `hboyang/s6e8-catstrall-member`는 CC0이므로 표기를 요청하면 풀릴 가능성이 있다.

### najiama 단일 5개와 beicicc other 4개

- najiama 01~05는 장부에 `szymon74:naji01~05`(float64, `0.963674`, `0.968619`, `0.968814`, `0.968722`, `0.968815`)로 이미 반입돼 있고 제외 사유는 원출처 라이선스 불명뿐이다.
  분할 근거는 szymonkapiski README의 정렬 검증 서술과 najiama 설명의 "5-Fold K-Fold Cross-Validation" 서술이며, 공개 5분할 스택(thisray, hboyang, wellkilo, kirill0212, raykkretzschmar)이 전부 같은 전제로 쓴다.
  이번에 새로 내려받지 않았다.
- beicicc의 other 라이선스 구성원 4개는 `s6e8-fixed4000-catboost-screen-relation-artifacts`의 baseline `0.96730`, screen_relations `0.96773`과 `s6e8-fixed900-structural-lgbm-artifacts`의 raw12 `0.96251`, structural `0.96372`이다.
  `docs/research/external94-width-evidence.json`이 분할 벡터 일치(1부터 셈 보정)와 유한값을 기록했고 장부의 `ext85_not_in_ledger`에 사유가 라이선스로 적혀 있다.
  같은 저자의 2단계 데이터셋 `s6e8-sixmember-crossfit-logitlr-artifacts`는 그대로 제외한다.

### masayakawamata/s6e8-catstr-aug16

- `oof_cat_str.npy` float64 (691,369,), `tep_cat_str.npy` (296,302,), 재채점 AUC `0.9670419`, 고유값 691,369다.
- 우리 분할로 나눈 분할별 AUC는 `[0.96645, 0.96717, 0.96718, 0.96773, 0.96669]`이지만 이것은 분할 일치의 증거가 아니다.
- 장부 대비 스피어만 순위 상관 최댓값은 `0.99767`(beicicc exact_value_catboost_fixed4000)이고 hboyang6:cat_strall_d8과는 `0.99185`다.
- 설명·README·코드가 없고, 저자의 S6E8 노트북이 없으며, thisray·hboyang·wellkilo 스택 노트북의 입력 목록에도 없다.
  저자 서술 한 줄만 있으면 풀리는 상태다.

### tamerlanomralinov/s6e8-full-best-blend-npy

- 9쌍 float64이고 재채점 AUC는 lgb_cat `0.96413`부터 lookup_transformer `0.96876`까지다.
- `blend_config.json`이 `"folds": 10`을 적고, 저자 노트북 `tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041`은 "Measured OOF, 10-fold"라고 서술하며 코드는 `N_FOLDS = 3 if QUICK else 11`이다.
  10분할 규칙으로 제외한다.
- 신경망 4개(lookup_transformer, dl_s23, dl_s7, mlp)는 로짓으로 저장됐고 고유값이 약 3,000개라 float16 흔적이 있다.

### raykkretzschmar/s6e8-transductive-anti-student-signals

- `transductive_signals.npz`의 키 10개 가운데 OOF·시험 쌍은 `oof_soft_student`(691,369)와 `test_soft_student`(296,302) 하나뿐이다.
  나머지는 시험 전용 신호(test_teacher, test_student, global_control, global_reconstructed, retrieval_signal)거나 부분 길이(reference_contrast 240,000, specialist_* 92,775)다.
- soft_student의 재채점 AUC는 `0.9673637`, 분할별 `[0.96676, 0.96747, 0.96751, 0.96789, 0.96719]`, 장부 대비 스피어만 최댓값 `0.99255`(bolt47:xgb_dd_d4)다.
- README는 "strict five-fold OOF prediction from the seven-variable distillation student"라고만 적고 시드와 행 순서를 밝히지 않는다.
  저자 노트북 `raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits`(08-19)도 "seven-variable, five-fold OOF distillation model"이라고만 서술한다.
- 교사(najiama 스택)의 소프트 라벨로 증류한 학생이므로 교사가 본 라벨 정보가 학생 OOF로 전이될 수 있다.
  2단계 산출물로 보고 보류한다.

### 나머지 데이터셋

- `szymonkapiski/s6e8-oof-library-25-models`의 25쌍은 float32이고, 전부 장부 szymon74의 같은 이름 구성원을 float32로 바꾼 값과 바이트가 같다(float64 대비 최대 차 `2.98e-08`).
  pub_ravi는 장부에서 2단계로 이미 제외됐다.
- `atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124`의 `v13_diversity_anchor_oof.csv`는 (691,369, 2)이고 id가 `train.csv` 순서이며 재채점 AUC `0.9701665`가 audit JSON의 선언과 같다.
  194개 고정 구성원과 szymon weak50을 합친 244구성원 로지스틱 메타모델 출력에 보정을 더한 것이라 2단계다.
- `thisray/s6e8-our-component`의 csv는 296,302행, id 691369~987670, 열 `our_component`로 시험 전용이다.
- kenchanhodgkin 16개 데이터셋의 파일은 `feature_columns.json`, `model_fold0~4`(.joblib 또는 .pt), `oof_predictions.csv`, `results.json`(exp009는 `history_fold*.json` 추가)뿐이고 시험 예측 파일은 하나도 없다.
- `wellkilo/s6e8-evidence-first-soft-student-assets`는 0.97117 champion 제출의 정수 순위 벡터와 감사 계약만 담는다.

## 노트북 출력물

공개 노트북 500개의 출력 파일 목록을 읽으니 OOF와 시험 예측을 함께 내놓는 노트북이 재수출·2단계를 포함해 약 90개다.
그중 단일 모델 노트북 46개의 배열을 내려받아 데이터셋과 같은 항목(행 수, 유한값, 재채점 AUC, 장부 대비 스피어만 최댓값, 시험 예측 순위 상관)으로 검증했다.
분할은 코드에서 읽었고, 표의 실행일은 Kaggle 목록의 마지막 실행 시각이다.
검증 원자료는 임시 경로의 `notebook_outputs_validation.csv`에 있으며 저장소에는 넣지 않았다.

### 사용 가능 43개

| 노트북 | 실행일 | 구성원과 재채점 AUC | 분할 근거 | 비고 |
| --- | --- | --- | --- | --- |
| kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only | 08-27 | realmlp `0.96899` | published_code | 원자료(jayjoshi37) 분포 통계를 참조, 공개 점수 0.97016, 장부 최댓값보다 높음 |
| omidbaghchehsaraei ft-transformer, cnn, tabtransformer, fastai, xgboost-v2, catboost | 08-02~08-23 | `0.96657`, `0.96771`, `0.96747`, `0.96676`, `0.96873`, `0.96715` | sibling_code(같은 저자의 lookup·realmlp 노트북이 5분할 시드 42) | cnn·tabtransformer는 고유값 4,000대로 양자화 흔적 |
| zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline | 08-24 | cat_base `0.96799`, lgb02 `0.96836`, xgb_base `0.96786` | published_code(manifest에 분할 명시) | 출력이 `oof/` 하위 경로 |
| redamountassir s6e8-lgbm-lb-0-96965, s6e8-histgradientboosting-lb-0-96945 | 08-06 | lgbm `0.96826`, hgb `0.96803` | published_code | kirill0212 스택의 e-* 입력 |
| yaminh/smartphone-addiction-prediction-strong-eda-cv-eble | 08-21 | lgbm_te `0.96751`, xgb_te `0.96773`, catboost `0.96193` | published_code | 기반 모델 3개만, 앙상블 열은 제외 |
| sidhaarthshree/lightgbm-ensemble-based-on-eda | 08-04 | lgb_a `0.96765`, lgb_b `0.96756`, xgb `0.96774` | published_code | npz 한 파일 |
| yekenot/ps-s6-e8-trompt-pytorch-frame | 08-25 | trompt `0.96667` | published_code | Trompt(pytorch-frame), 장부에 없는 계열 |
| lucymlai32 phase-2-xgboost-and-model-blending, smartphone-addiction-prediction | 08-21 | xgboost `0.96580`, catboost_v2 `0.96438` | published_code | |
| cdeotte simple-xgb-starter, simple-cat-starter, simple-nn-starter | 08-20~08-21 | `0.96481`, `0.96291`, `0.93974` | published_code | 시작 노트북 3종 |
| dariushafshar/0-97184-leader-xgb-feature-ablation | 08-26 | xgb `0.96488` | published_code | cdeotte XGB 변형 |
| rv1922/smartphone-addiction | 08-11 | lgbm v1 `0.96326`, v2 `0.96374`, v3 `0.96403`, xgb seed42 `0.96482` | published_code | xgb seed777·2026은 다른 분할이라 제외, 출력이 `model_outputs/` 하위 경로 |
| yadoy666/predicting-smartphone-addiction | 08-26 | catboost `0.96370`, xgboost `0.96471` | published_code | fmdeep은 fm5:fmdeep과 순위 상관 0.99936이라 중복, 앙상블 출력은 2단계 |
| danushkumarv/smartphone-addiction-gbm-rank-blend-nb01 | 08-04 | lgb `0.96395`, xgb `0.96447`, cb `0.96271` | published_code | 스택 출력은 najiama 입력이라 2단계 |
| harwindersingh766/ps-s6e8-xgboost-te-lb-0-96548 | 08-06 | xgb `0.96410` | published_code | `_sb` 판은 시드 3개 분할 평균이라 제외 |
| dynamo14324/smartphone-addiction-championship-v11 | 08-24 | lgb `0.96337`, xgb `0.96322` | published_code | 의사 라벨 판은 시험 예측만 있어 제외 |
| mohankrishnathalla s6e8-realmlp-oof-saver, s6e8-tabm-oof-saver | 08-12~08-14 | realmlp `0.95813`, mlp `0.94142` | published_code | 제목이 TabM인 노트북의 코드는 MLP |
| kava1/predicting-smartphone-addiction-resnet-fe | 08-20 | resnet `0.95687` | published_code | |
| lopure/hdviz-pca-parallel-with-linear-svm | 08-14 | linear `0.91135`, poly `0.92880`, rbf `0.92217` | published_code | SVM 계열, rbf는 장부 최근접 상관 0.895로 가장 다름 |

시험 예측 순위 상관은 43개 전부 OOF 쪽 최댓값과 같거나 높아 시험 행 순서가 `test.csv` 순서라는 방증이 된다.
`lavanyabacche/xgb-starter-01`은 cdeotte XGB와 AUC와 고유값 수가 같아 같은 배열로 보고 세지 않았다.

### 장부 구성원과 같은 배열 2개

- `omidbaghchehsaraei/lookup-transformer-predicting-smartphone-addiction`(08-22)의 출력은 장부 `hboyang6:kirill_o1`과 OOF·시험 순위 상관이 정확히 1.0이다.
  kirill0212 노트북이 omid 노트북 출력을 `o-*`로 재수출했고 hboyang이 그것을 다시 게시한 것이므로, 장부의 "이름이 다른 공개 노트북 레시피를 가리킨다"는 주의 사항은 이 출처로 확정된다.
- `yadoy666/predicting-smartphone-addiction`의 fmdeep은 `fm5:fmdeep`과 `0.99936`이다.

### 분할이 다르거나 섞인 것

| 노트북 | 출력 | 사유 |
| --- | --- | --- |
| kodaifukuda0311/s6e8-xgb-the-power-of-exact-value-te-fe | oof_xgb.npy, pred_xgb.npy | 시드 5개(42, 202, 2026, 777, 4946)마다 다른 분할을 평균 |
| zhenruiweng/s6e8-public-lb-0-97009-single-model-realmlp | oof.csv, test_pred.csv | 시드 42·789·1011별 분할 평균 |
| stephentarter ps-s06e08-catboost, histgradientboosting, lightgbm, xgboost | *_oof_probs.csv, *_test_probs.csv | 설정 스크립트의 첫 시드가 10301(재채점 `0.96421`~`0.96453`) |
| ern711/multi-level-deep-univariate-spline-transformer | multilevel_output_heads_oof.csv 등 | `OUTER_SPLIT_SEED = 21`, `docs/research/multilevel-spline-notebook.md` 참고 |
| factualexplorer/baseline-lgbm-xgb-cb-rank-averaged-oof-tuned | preds.npz | 전체 모드 10분할 |
| tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041 | oof·test npy 3쌍 | 10분할 서술, 코드 11분할 |
| evgendvorkin/s6e8-single-lgb | oof_preds.npy, test_preds.npy | 10분할(장부 pub_evg와 같은 사유) |
| echloeprice/phone-addict | oof_*.csv, submission_*.csv | 10분할 |
| destroyer123787/predicting-smartphone-addiction | oof_predictions.csv, test_predictions.csv | RepeatedStratifiedKFold, 혼합 열 |
| dranilkumardubey/nova-sap | oof_predictions.csv | 시드 2026 반복 분할 |
| darkmatternet/s6e8-rules-eda-catboost-guide | s6e8_catboost_guide_oof.csv | 시드 20260821 |
| lavanyabacche/predicting-smartphone-addiction-catboost-fe | oof_v12_*.npy | 시드 2026, 혼합 출력 |
| magurodataanalysis/s6e8-linear-vs-trees | oof.csv | 3분할 |
| yusufmurtaza01/s6e8-training | oof_*.npy, test_*.npy | 시드 2025 반복 분할 |
| vladstud716373618/baseline-5-fold-cv-catboost-deep-fe | oof.csv, test_pred.csv | 코드는 시드 42지만 내려받은 OOF의 재채점 AUC가 `0.49997`로 정렬 불명 |
| shamanthakreddymallu/s6e8-baseline | oof_lgb_fe.npy, oof_lr.npy 등 | 코드에서 시드를 찾지 못함(재채점 `0.96377`, `0.93661`), 저자 서술 있으면 풀림 |

### 2단계와 재수출

- `kirill0212/s6e8-public-ensemble`(08-24)의 `oof_*`·`test_*` 230쌍은 szymon 47·weak50, golem, adarsh, beicicc, boltuzamaki, mohankrishnathalla 데이터셋과 omid·donmarch14·zhenruiweng·nawfeelrahman·ravi·redamountassir 노트북 출력을 로짓으로 재저장한 것이며 신규 학습이 없다.
- `yadoy666/94-verified-oof-gpu-accelerated-meta-stack`의 union94 행렬은 기존 94개 재수출이다.
- 2단계 출력은 dariushafshar s6e8-177-member-stack-oof-export·s6e8-pool125-nested-oof-export·0-97125-rank-logit-fusion-forkable, hboyang/s6e8-150-member-fusion, nikita7364777/rank-gauss-logit-rank-blending, darkmatternet/s6e8-oof-meta-ensemble-guide, beicicc/s6e8-realmlp-seed01-strict-meta-20260805, beicicc/s6e8-strict-neural-residual-audit, anthonytherrien/s6e8-lgbm-xgb-catboost-stack, ravi20076/playgrounds6e8-public-l2stack-v1, lucifer19/smartaddict-oof-signal-forge, georgymamarin/s6e8-will-your-0-971-survive-the-private-split, funguscakehead/da-thig, wesleyhuan/s6e8-multi-model-compare(oof_blend 열), stephentarter/ps-s06e08-model-ensembling-stacking이다.
- `najiama/single-lgbm-model-lb-0-96990-cv-0-96862`(08-22, 재채점 `0.96862`)는 5분할 시드 42지만 혼합 제출에서 수확한 의사 라벨로 학습해 2단계에 기댄다.
- `rafanikitas/s6e8-multi-level-stacking`은 level1·level2 OOF csv만 있고 시험 예측이 없다.
- `donmarch14/s6e8-catboost`·`s6e8-lgbm`(5분할 시드 42)은 장부에 szymonkapiski의 재실행분 pub_cat·pub_donlgbm이 있어 다시 검증하지 않았다.

### 미확인

- `shashwat1729/s6e8-lookup-pair-transformer`(08-07, 코드는 5분할 시드 42)는 출력 내려받기가 권한 거부(kernels.get)로 막혔고, 같은 저자의 다른 노트북 6개는 파일 목록 조회도 403이다.
- `mhamza0810/s6e8-single-model-fe-cv-0-96947`, `udaken10/xgboost-improved`(08-27), `shamanthakreddymallu/s6e8-lightgbm`(08-27)은 코드가 5분할 시드 42이고 파일 목록에 OOF·시험 파일이 있지만 `kernels output`이 실행 기록만 내려주고 배열은 주지 않았다.
  표에는 미확인(내려받기 실패)으로 남긴다.

### 라이선스

Kaggle 공개 노트북 소스는 Apache 2.0이지만 출력 파일에는 기본적으로 라이선스 표시가 없다(`docs/agents/kaggle-public-notebook-licensing.md`).
따라서 위 43개는 결합기 입력으로만 쓰는 사용 한정 구성원이며, 산출물이나 데이터셋으로 재배포하지 않는다.
장부 규칙(데이터셋 라이선스 4종)으로 되돌리면 이 43개와 unknown·other 데이터셋 20개는 전부 빠지고 paiky1995 6개와 nhtquyn 120개만 남는다.

## 08-27 이후 신규분과 재실행 레시피

2026-08-27 이후 갱신된 S6E8 데이터셋은 없다.
검색어 `s6e8`의 최신 갱신은 08-26 `dariushafshar/kaggle-competition-leaderboard-intelligence`이고, 다른 검색어에서 08-27로 잡힌 `masha6574/cimt-final-oof`는 다른 대회 자료다.

08-27에 실행된 대회 노트북은 9개다.
`kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only`는 검증을 통과했고, `udaken10/xgboost-improved`와 `shamanthakreddymallu/s6e8-lightgbm`은 OOF·시험 파일이 있으나 내려받지 못해 미확인이며, `mikhailnaumov/smartphone-addiction-xgb`, `vinay24baghira/s6e8-xgb-lgbm-cb-voting-stack-full-guide`, `nicolepatterson8910/smartphoneaddict`, `junkonno/fork-of-fork-of-fork-of-fork-of-notebookd25-ca7ba4`는 제출 파일이나 그림만 출력하고, `atifkhan12/pca-smart-phone-addiction`과 `udaken10/xgboost-s6e8-stepwise-tuning`은 출력이 없다.

같은 조사를 다시 돌리는 명령은 다음과 같다.

```bash
for p in 1 2 3; do kaggle datasets list -s s6e8 --sort-by updated --page-size 100 -p $p -v; done
kaggle datasets files <owner/slug> --page-size 200 --format json
kaggle datasets metadata <owner/slug> -p <dir>
kaggle datasets download -d <owner/slug> -p <scratch-dir> --unzip
for p in 1 2 3 4 5; do kaggle kernels list --competition playground-series-s6e8 --sort-by dateRun --page-size 100 -p $p -v; done
kaggle kernels files <owner/slug> --page-size 200 --format json
kaggle kernels pull <owner/slug> -p <dir> -m
kaggle kernels output <owner/slug> -p <dir> -o
```

`kaggle datasets files`와 `kaggle kernels files`는 `-p`를 받지 않고 `--page-token`으로만 넘긴다.
`kernels files`와 `kernels output`은 수백 번 연달아 부르면 429가 나므로 호출 사이에 1.5초 이상 쉬고 429가 나면 45초 뒤 다시 시도해야 하며, 출력이 하위 경로에 있으면 그대로 하위 경로로 내려오고 `--file-pattern`은 파일 이름 앞부분에 걸린다.
검증 스크립트는 저장소에 넣지 않았고, 반입할 때는 `scripts/build_external_member_ledger.py`에 nhtquyn(행렬 npy + members.csv, weak50과 같은 모양), paiky1995(구성원별 npy), hboyang 150-fusion(구성원별 npy), 노트북 출력물(npy·csv·npz 혼재, id 열이 있으면 id로 재정렬) 적재기를 더하면 된다.

## 한계

- 노트북 출력물 46개 가운데 4개는 배열을 내려받지 못해 미확인으로 남았다.
- 출력 파일 목록 조회는 처음 시도에서 338개가 429로 실패해 느린 속도로 재시도했고, 최종 미해결은 shashwat1729의 403 6개뿐이다.
  노트북 목록 자체는 dateRun 상위 500개다.
- omid의 6개 노트북은 코드를 읽지 않고 같은 저자의 두 노트북 코드로 분할을 추정했다(`sibling_code`).
- 데이터셋 열거는 검색어 기반이라 제목과 설명에 `s6e8`, `oof`, `smartphone`, `addiction` 같은 낱말이 없는 데이터셋은 놓칠 수 있다.
- `author_statement`와 `sibling_code`는 저자 서술에 기대며, 재채점 AUC 일치는 정렬을 증명할 뿐 분할을 증명하지 않는다.
  분할 벡터가 동봉된 nhtquyn과 beicicc만 분할이 확인된다.
- 노트북 출력물끼리의 중복은 장부 대비로만 걸렀고 서로 간 스피어만 순위 상관은 재지 않았다.
  반입할 때 바이트 해시와 0.998 규칙으로 다시 걸러야 한다.
- nhtquyn은 문서가 없어 학습 자료 범위(원자료 사용 여부)와 전처리를 확인할 수 없고, kodaifukuda0311 RealMLP는 원자료 분포 통계를 특성으로 쓴다.
- 이 조사는 구성원을 더하는 축만 다루고 nested OOF 기여는 재지 않았다.
  189개를 더한 사다리의 값어치는 판정 티켓이 절제 실험과 함께 재야 한다.
