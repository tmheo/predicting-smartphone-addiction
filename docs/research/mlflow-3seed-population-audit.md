# MLflow 3시드 분석 모집단과 제외 사유 고정

감사 기준일은 2026년 8월 18일이다.
대상은 저장소 루트의 `mlflow.db`(sqlite)이며, `mode=ro` 읽기 전용 연결로만 조회했고 어떤 기록도 수정하지 않았다.
이 문서는 지도 이슈 #208의 하위 티켓 #209의 산출물로, 이후 결정(#210, #211)과 노트북 명세(#214)가 그대로 사용할 모집단 정의와 제외 사유를 고정한다.

## 결론: 고정된 모집단

| 모집단 | 정의 | 건수 |
| --- | --- | --- |
| P0 원자료 | 실험 `predicting-smartphone-addiction`의 모든 실행 | 153 |
| P1 전체 OOF | 완료(FINISHED)이고 `seeds=42,43,44`인 단일 모델 실행 | 42 |
| P2 엄격 시드 | P1 중 `auc_oof_seed_42/43/44` metric을 모두 보유한 실행 | 39 |
| P3 대표 | P1에서 동일 구성마다 대표 1건을 고른 실행 | 32 |
| 전이 분석 | P1 중 `public_auc` metric 보유 실행 | 10 |
| 앙상블 구획 | `ensemble.*` param을 가진 완료 실행(별도 구획) | 7 |

P0의 상태 분포는 FINISHED 139, KILLED 12, FAILED 2다.
FINISHED 139는 3시드 단일 모델 42, 단일 시드(`seeds=42`) 단일 모델 90, 앙상블 7로 나뉜다.
지도 Notes의 "완료 실행 42개"와 P1이 정확히 일치함을 확인했다.

## 판별 규칙

이후 결정과 노트북 명세는 실행 id 목록이 아니라 아래 규칙으로 모집단을 재도출한다.
이 문서의 표는 감사 기준일 스냅샷에 규칙을 적용한 결과이자 노트북 구현의 검증 기준이다.

- 완료: `runs.status == "FINISHED"`.
- 3시드: param `seeds == "42,43,44"`.
- 엄격 시드 근거: metric `auc_oof_seed_42`, `auc_oof_seed_43`, `auc_oof_seed_44`가 모두 존재.
- 앙상블: `ensemble.`으로 시작하는 param이 하나라도 존재.
  앙상블 실행에는 `seeds` param이 없으므로 3시드 필터와 자연히 분리된다.
- dirty: tag `git_dirty == "True"`.
- Public 보유: metric `public_auc` 존재.
- 대표 선정: 동일 구성의 완료 실행 중 깨끗한(clean) 커밋에서 나온 가장 최근 실행 1건.
  깨끗한 실행이 하나도 없으면 가장 최근 dirty 실행을 대표로 삼고 dirty 예외로 표시한다.
- 동일 구성 판정: 기본은 param `experiment`가 같으면 동일 구성.
  단, `experiment`가 달라도 `experiment`를 제외한 모든 param과 AUC 계열 metric이 완전히 일치하면 이름만 바뀐 동일 구성으로 본다(아래 exp033/exp035 사례).

## 제외와 표시 사유 코드

| 코드 | 뜻 | 건수 | 처리 |
| --- | --- | --- | --- |
| NOT_FINISHED | KILLED 또는 FAILED 실행 | 14 | 모든 분석에서 제외, 원자료에만 보존 |
| SINGLE_SEED | 완료됐으나 `seeds=42` 단일 시드 스크리닝 | 90 | 3시드 분석에서 제외, 원자료에만 보존 |
| LEGACY_FORMAT | 3시드 완료이나 시드별 metric이 없는 이전 기록 형식 | 3 | P1에 포함, P2에서 제외, 표에 표시 |
| DUPLICATE | 동일 구성의 비대표 재실행 | 10 | P3에서 제외, 민감도 분석에는 포함 |
| DIRTY | `git_dirty=True` 커밋에서 나온 실행 | 8 | 제외가 아니라 표시. 대표 선정에서 후순위, 앙상블 후보 제외 관행(#14) 유지 |
| ENSEMBLE_DERIVED | 단일 모델이 아닌 파생 앙상블 | 7 | 단일 모델 비교에서 제외, 별도 구획에서 비교 |
| NO_OOF | `auc_oof` metric이 없어 OOF 분석이 불가능한 실행 | 1 | OOF 분석 제외, Public 건전성 점검에만 사용 |

## 세부 발견

### 이전 기록 형식(LEGACY_FORMAT) 3건

시드별 metric 기록은 2026년 8월 11일 오전에 도입됐고, 그 전의 3시드 실행 3건은 `auc_fold_0..4`와 `auc_oof`만 남겼다.

- `f4e77ff3` exp005_te_replacement (08-11 08:36): legacy이면서 dirty이고 해당 구성의 유일한 실행이다.
- `01a466bd` exp006_te_drop_gaming (08-11 08:57): legacy이면서 dirty.
- `264f7e6f` exp006_te_drop_gaming (08-11 09:25): legacy이지만 clean이고 `public_auc=0.96795`를 보유한다.

세 건 모두 `auc_oof`는 있으므로 P1(전체 OOF)과 전이 분석에는 참여하고, 시드 안정성 분석(P2)에서만 빠진다.

### 동일 구성 다중 실행과 대표

`experiment` 이름이 같은 다중 실행은 5개 구성 12건이고, 이름이 다른 동일 구성 1쌍이 추가로 있다.
같은 구성의 재실행은 `auc_oof`가 소수점 아래 9자리까지 같아(결정적 파이프라인) 어느 실행을 대표로 골라도 지표는 사실상 동일하다.

| 구성 | 실행 수 | 대표 | 비대표(제외 사유 DUPLICATE) |
| --- | --- | --- | --- |
| exp006_te_drop_gaming | 3 | `4aaddd50` | `01a466bd`(dirty, legacy), `264f7e6f`(legacy, public 보유) |
| exp011_resid_pair | 4 | `8236e35f` | `e21d19af`(public 보유), `0b44644d`(dirty), `aa523c49`(dirty) |
| exp032_recon_orig_mean_top3 | 2 | `b1bd4b08` | `0e957245`(dirty) |
| exp033_recon_orig_mean_top3_raw | 2 | `c34f1da1` | `f5805b9b`(dirty) |
| exp058_logreg_onehot | 3 | `8198a001` | `e2b76edd`, `7ce4bc3a` |
| exp033_lattice_te = exp035_lattice_te | 2 | `c62a9ad3`(exp035) | `5fdb7c26`(exp033_lattice_te) |

exp033_lattice_te(`5fdb7c26`)와 exp035_lattice_te(`c62a9ad3`)는 `experiment` param을 제외한 모든 param과 AUC 계열 metric이 완전히 일치한다.
exp033 번호가 exp033_recon_orig_mean_top3_raw와 충돌해 exp035로 다시 번호를 붙인 재실행으로 판단하고, 뒤의 exp035를 대표로 삼는다.
따라서 P3 대표는 실험 이름 33개가 아니라 구성 32개다.

### dirty 실행 8건

- 비대표 dirty 5건: `01a466bd`, `0b44644d`, `aa523c49`, `0e957245`, `f5805b9b`. 대표 선정에서 자연히 빠진다.
- 대표가 dirty뿐인 구성 3건: `f4e77ff3` exp005_te_replacement, `3f7d735f` exp025_constrained_impute, `737f4dae` exp043_cat_depth6.
  이 3건은 dirty 예외 표시를 달고 P3에 포함한다.
  제외하면 해당 구성 자체가 분석에서 사라져 표본 선택 편향이 생기기 때문이다.

### Public AUC 보유 실행 15건

`public_auc`는 15건에 있고 소속은 다음과 같다.

- 3시드 단일 모델 10건(기본 전이 분석 모집단): `264f7e6f` exp006(0.96795, legacy), `e21d19af` exp011(0.96869), `62f57ea7` exp026(0.96898), `737f4dae` exp043(0.96942, dirty), `2c615036` exp052(0.96955), `3d5239b0` exp057(0.96979), `b951fac5` exp059(0.97019), `6238d8c5` exp070(0.96982), `2bd55026` exp067_lookup(0.97030), `d55d1cd4` exp081(0.97033).
- 단일 시드 1건(참고용): `ce66e16b` exp001_lgbm_baseline(0.96450).
- 앙상블 4건(별도 구획): 0.97055, 0.97056, 0.97057, 0.97063.

전이 분석 짝짓기 규칙: `public_auc`가 기록된 실행 자체의 `auc_oof`를 짝으로 쓴다.
Public 점수는 그 실행의 제출 파일에서 나왔으므로, 대표 실행이 따로 있어도(exp006, exp011) 대표의 OOF로 바꿔 짝짓지 않는다.

### 앙상블 구획 7건의 특이점

| run | 이름 | auc_oof | public_auc | 비고 |
| --- | --- | --- | --- | --- |
| `1a1a6b42` | ensemble_rank_mean_issue63_pool3 | 0.96945 | 0.97055 | |
| `d7c53c28` | ensemble_rank_logit_logistic_issue64_pool16 | 0.96948 | 0.97056 | |
| `455c5aad` | ensemble_missing_segmented_rank_logit_issue65_pool16 | 0.96951 | 0.97057 | |
| `c2171fa9` | submission_issue66_full_refit_cv_full | 없음 | 0.97063 | NO_OOF. full refit 제출 실행이라 OOF가 없다 |
| `d845b5d1` | ensemble_missing_interaction_rank_logit_issue183_pool19 | 0.96951 | 없음 | git 태그 없음 |
| `7fbe590b` | ensemble_missing_segmented_rank_logit_issue179_pool20 | 0.96961 | 없음 | git 태그 없음 |
| `86d87d91` | ensemble_missing_interaction_rank_logit_issue202_pool22 | 0.96961 | 없음 | git 태그 없음 |

기록 격차 두 가지를 표시한다.
첫째, `c2171fa9`는 `auc_oof`가 없어 OOF 분석에 참여할 수 없고 Public 건전성 점검에만 쓴다.
둘째, 8월 18일의 앙상블 3건(`d845b5d1`, `7fbe590b`, `86d87d91`)에는 `git_commit`, `git_dirty` 태그가 없다.
기존 기록 수정은 이 지도의 범위 밖이므로 태그를 소급 기록하지 않고, 앙상블 구획 분석에서 커밋 기반 판별을 쓰지 않는 근거로만 남긴다.

### 입력 동일성

P1의 42건 전부에 `sha256.*` 입력 해시 태그가 있고 누락은 없다.

## 3시드 완료 42건 전체 표

시작 시각 순이다.
대표 열의 "예외"는 dirty뿐인 구성이라 dirty 대표를 쓴 경우다.

| experiment | run | 시작 | auc_oof | 엄격 | public_auc | dirty | 대표 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp005_te_replacement | `f4e77ff3` | 08-11 08:36 | 0.96659 | N | | Y | 예외 |
| exp006_te_drop_gaming | `01a466bd` | 08-11 08:57 | 0.96659 | N | | Y | |
| exp006_te_drop_gaming | `264f7e6f` | 08-11 09:25 | 0.96659 | N | 0.96795 | | |
| exp006_te_drop_gaming | `4aaddd50` | 08-11 11:58 | 0.96659 | Y | | | 대표 |
| exp011_resid_pair | `e21d19af` | 08-11 12:50 | 0.96740 | Y | 0.96869 | | |
| exp011_resid_pair | `8236e35f` | 08-11 16:01 | 0.96740 | Y | | | 대표 |
| exp011_resid_pair | `0b44644d` | 08-11 16:22 | 0.96740 | Y | | Y | |
| exp011_resid_pair | `aa523c49` | 08-11 17:04 | 0.96740 | Y | | Y | |
| exp022_orig_knn | `52e9c12b` | 08-11 18:25 | 0.96733 | Y | | | 대표 |
| exp023_orig_proxy_residual | `202b7d47` | 08-11 21:16 | 0.96737 | Y | | | 대표 |
| exp025_constrained_impute | `3f7d735f` | 08-11 22:04 | 0.96757 | Y | | Y | 예외 |
| exp026_constrained_impute_nowidth | `62f57ea7` | 08-11 22:36 | 0.96755 | Y | 0.96898 | | 대표 |
| exp032_recon_orig_mean_top3 | `0e957245` | 08-11 23:55 | 0.96765 | Y | | Y | |
| exp032_recon_orig_mean_top3 | `b1bd4b08` | 08-12 00:16 | 0.96765 | Y | | | 대표 |
| exp033_recon_orig_mean_top3_raw | `f5805b9b` | 08-12 08:16 | 0.96762 | Y | | Y | |
| exp033_recon_orig_mean_top3_raw | `c34f1da1` | 08-12 08:48 | 0.96762 | Y | | | 대표 |
| exp033_lattice_te | `5fdb7c26` | 08-12 08:48 | 0.96729 | Y | | | |
| exp035_lattice_te | `c62a9ad3` | 08-12 09:26 | 0.96729 | Y | | | 대표 |
| exp037_drop_floor_cats | `007cab1f` | 08-12 10:51 | 0.96756 | Y | | | 대표 |
| exp043_cat_depth6 | `737f4dae` | 08-12 13:58 | 0.96820 | Y | 0.96942 | Y | 예외 |
| exp045_xgb_depth8 | `e2c432b4` | 08-12 16:13 | 0.96794 | Y | | | 대표 |
| exp052_cat_xgb_impute_pass5 | `2c615036` | 08-13 00:09 | 0.96836 | Y | 0.96955 | | 대표 |
| exp057_cat_xgb_impute_comps5 | `3d5239b0` | 08-13 10:18 | 0.96854 | Y | 0.96979 | | 대표 |
| exp058_logreg_onehot | `e2b76edd` | 08-13 15:05 | 0.95966 | Y | | | |
| exp058_logreg_onehot | `7ce4bc3a` | 08-13 20:30 | 0.95966 | Y | | | |
| exp058_logreg_onehot | `8198a001` | 08-13 20:32 | 0.95966 | Y | | | 대표 |
| exp059_lookup_transformer | `b951fac5` | 08-14 08:02 | 0.96892 | Y | 0.97019 | | 대표 |
| exp060_lookup_transformer_nn10 | `e0cb50f3` | 08-14 08:27 | 0.96879 | Y | | | 대표 |
| exp063_lgb_max_bin_1023 | `4e2b2f56` | 08-14 10:28 | 0.96760 | Y | | | 대표 |
| exp065_tabm | `df2023d4` | 08-14 20:59 | 0.96833 | Y | | | 대표 |
| exp070_cat_exact_cats | `6238d8c5` | 08-15 03:03 | 0.96858 | Y | 0.96982 | | 대표 |
| exp067_lookup_xgb_impute_comps5 | `2bd55026` | 08-15 10:05 | 0.96910 | Y | 0.97030 | | 대표 |
| exp067_tabpfn3 | `85b09132` | 08-15 12:56 | 0.96724 | Y | | | 대표 |
| exp074_lgb_kitopl_d2_bundle | `446a90be` | 08-15 16:30 | 0.96840 | Y | | | 대표 |
| exp080_lookup_emb_wd_1e3 | `f9bcc589` | 08-16 00:31 | 0.96909 | Y | | | 대표 |
| exp081_lookup_fold_initialization_avg3 | `d55d1cd4` | 08-16 07:43 | 0.96920 | Y | 0.97033 | | 대표 |
| exp110_lgb_kitopl_no_te | `ae829ae3` | 08-18 15:21 | 0.96733 | Y | | | 대표 |
| exp107_logreg_onehot_nn10 | `c4c4c780` | 08-18 15:46 | 0.95999 | Y | | | 대표 |
| exp111_xgb_depth8_no_te | `3cbc2ccc` | 08-18 15:48 | 0.96483 | Y | | | 대표 |
| exp071_cat_exact_no_te | `521b4924` | 08-18 15:59 | 0.96816 | Y | | | 대표 |
| exp106_lookup_fixed24_train_test_preprocessing | `547e7bc9` | 08-18 17:59 | 0.96787 | Y | | | 대표 |
| exp108_logreg_onehot_nn10_l1 | `83f7977a` | 08-18 18:24 | 0.96023 | Y | | | 대표 |

## 재현 방법

노트북 명세는 다음 절차로 이 감사를 재현할 수 있어야 한다.

1. MLflow 데이터베이스 경로를 명시적으로 받고, 파일이 없으면 새 데이터베이스를 만들지 말고 즉시 중단한다.
   sqlite `file:<path>?mode=ro` 연결이 이 요구를 그대로 만족한다.
2. `runs`, `params`, `latest_metrics`, `tags` 네 표만 조회해 실행별 기록 원형을 모은다.
3. 위 판별 규칙을 적용해 P0에서 P3, 전이 분석, 앙상블 구획을 재도출한다.
4. 결과 건수(153, 42, 39, 32, 10, 7)와 이 문서의 표가 일치하는지 검증한다.
   감사 기준일 이후 실행이 추가되면 건수는 커질 수 있으나, 이 문서에 실린 실행의 소속과 표시는 변하지 않아야 한다.
