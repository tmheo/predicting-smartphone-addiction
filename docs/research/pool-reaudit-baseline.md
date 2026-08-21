# 후보 풀 소급 재심사 기준 장부

## 결론

현재 후보 풀 35개를 모두 다시 검증했고 35개가 무결성 검사를 통과했다.
정확 중복과 순위 중복 제거 뒤 35개가 남는다.
모델 계열은 11개, 모델 계보 묶음은 17개, 정보 관점은 14개다.
현재 구성으로 전체 자료 재학습은 99회다.

기준 장부 파일은 `artifacts/pool-baseline-2026-08-21.yaml`이고 SHA-256은 `cef5c08efad104580dc9fab7a3c7605d1e5f95ce5f9b825caa66206dc50ff96f`다.
재심사가 끝날 때까지 이 파일과 아래 표를 바꾸지 않는다.

## 동결한 입력

| 입력 | SHA-256 |
| --- | --- |
| `artifacts/pool.yaml` | `e6f093c08af4d09a70e2ee9a7cc99f9d099b06b7505116005464b5ae1240712a` |
| `artifacts/full-refit-plan.yaml` | `cb42b27f01abecdc51784e224d3346b27910d29b106171d8cdd471e1246b403f` |
| `data/train.csv` | `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c` |
| `data/test.csv` | `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e` |
| `artifacts/folds.parquet` | `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4` |
| 원본 프록시 CSV | `2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074` |

기준 커밋은 `15a36ad65fd61993518c8e713e7729261d57fde9`다.

## 구성원 분류

| 구성원 | 모델 계열 | 모델 계보 묶음 | 계보 역할 | 정보 관점 | 전처리 기준 범위 | 전체 자료 재학습 | 무결성 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| `exp006_te_drop_gaming` | `lightgbm` | `exp006_te_drop_gaming` | 묶음 시작 | 목표값 부호화 | fold_train | 3 | 통과 |
| `exp011_resid_pair` | `lightgbm` | `exp006_te_drop_gaming` | champion 개선판 | 목표값 부호화, 산술 파생 조합 | fold_train | 3 | 통과 |
| `exp022_orig_knn` | `lightgbm` | `exp022_orig_knn` | 묶음 시작 | 목표값 부호화, 산술 파생 조합, 원본 프록시 최근접 라벨 | fold_train | 3 | 통과 |
| `exp023_orig_proxy_residual` | `lightgbm` | `exp023_orig_proxy_residual` | 묶음 시작 | 목표값 부호화, 산술 파생 조합 | fold_train | 3 | 통과 |
| `exp025_constrained_impute` | `lightgbm` | `exp025_constrained_impute` | 품질 개선판 교체 | 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원 구간 폭, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp032_recon_orig_mean_top3` | `lightgbm` | `exp025_constrained_impute` | 특성 묶음 추가판 | 목표값 부호화, 산술 파생 조합, 원본 프록시 통계 사전, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp033_recon_orig_mean_top3_raw` | `lightgbm` | `exp025_constrained_impute` | 설정값 변형판 | 목표값 부호화, 산술 파생 조합, 원본 프록시 통계 사전, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp035_lattice_te` | `lightgbm` | `exp035_lattice_te` | 묶음 시작 | 격자 이변수 목표값 부호화, 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp058_logreg_onehot` | `logistic_onehot` | `exp058_logreg_onehot` | 묶음 시작 | 원시 관측 열 전용 | fold_train | 1 | 통과 |
| `exp059_lookup_transformer` | `lookup_transformer` | `exp059_lookup_transformer` | 묶음 시작 | 산술 파생 조합, 정확값 어휘 조회 | fold_train | 3 | 통과 |
| `exp070_cat_exact_cats` | `catboost` | `exp070_cat_exact_cats` | 중복 교체판 | 목표값 부호화, 산술 파생 조합, 정확값 범주 복제, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp067_tabpfn3` | `tabpfn3` | `exp067_tabpfn3` | 묶음 시작 | 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp081_lookup_fold_initialization_avg3` | `lookup_transformer` | `exp059_lookup_transformer` | 학습 설정 개선판 | 산술 파생 조합, 정확값 어휘 조회, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp110_lgb_kitopl_no_te` | `lightgbm` | `exp110_lgb_kitopl_no_te` | 특성 묶음 제거판 | 산술 파생 조합, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp111_xgb_depth8_no_te` | `xgboost` | `exp111_xgb_depth8_no_te` | 특성 묶음 제거판 | 산술 파생 조합, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp071_cat_exact_no_te` | `catboost` | `exp070_cat_exact_cats` | 특성 묶음 제거판 | 산술 파생 조합, 정확값 범주 복제, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp106_lookup_fixed24_train_test_preprocessing` | `lookup_transformer` | `exp106_lookup_fixed24_train_test_preprocessing` | 묶음 시작 | 산술 파생 조합, 정확값 어휘 조회 | train_test | 3 | 통과 |
| `exp107_logreg_onehot_nn10` | `logistic_onehot` | `exp058_logreg_onehot` | 특성 묶음 추가판 | 원본 프록시 최근접 라벨 | fold_train | 1 | 통과 |
| `exp108_logreg_onehot_nn10_l1` | `logistic_onehot` | `exp058_logreg_onehot` | 특성 묶음 추가판 | 원본 프록시 최근접 라벨 | fold_train | 1 | 통과 |
| `exp117_ag25_gbm_r21` | `lightgbm` | `exp110_lgb_kitopl_no_te` | 중복 교체판 | 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp113_tab_cnn_m0` | `tab_cnn` | `exp113_tab_cnn_m0` | 묶음 시작 | 산술 파생 조합, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp085_contextual_spline_m0` | `contextualized_spline_transformer` | `exp085_contextual_spline_m0` | 묶음 시작 | 산술 파생 조합, 정확값 어휘 조회, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp124_realmlp_dtype_fix` | `realmlp` | `exp124_realmlp_dtype_fix` | 결함 수정판 교체 | 원시 관측 열 전용 | fold_train | 3 | 통과 |
| `exp127_lookup_muon` | `lookup_transformer` | `exp059_lookup_transformer` | 학습 설정 개선판 | 산술 파생 조합, 정확값 어휘 조회, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp027_recon_ce` | `lightgbm` | `exp025_constrained_impute` | 특성 묶음 추가판 | 목표값 부호화, 산술 파생 조합, 정확값 빈도 부호화, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp048_lgb_orig_cdf_diff` | `lightgbm` | `exp025_constrained_impute` | 특성 묶음 추가판 | 목표값 부호화, 산술 파생 조합, 원본 프록시 분포 좌표, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp134_realmlp_muon` | `realmlp` | `exp124_realmlp_dtype_fix` | 학습 설정 개선판 | 원시 관측 열 전용 | fold_train | 3 | 통과 |
| `exp135_xgb_hpo_trial30` | `xgboost` | `exp111_xgb_depth8_no_te` | 중복 교체판 | 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원값 | fold_train | 3 | 통과 |
| `exp131_lookup_bivariate_plr5` | `lookup_transformer` | `exp059_lookup_transformer` | 특성 묶음 추가판 | 산술 파생 조합, 정확값 어휘 조회, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp136_realmlp_muon_recon_widths` | `realmlp` | `exp124_realmlp_dtype_fix` | 특성 묶음 추가판 | 제약 기반 결측 복원 구간 폭 | fold_train | 3 | 통과 |
| `exp137_tabm_recon_widths` | `tabm` | `exp137_tabm_recon_widths` | 특성 묶음 추가판 교체 | 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원 구간 폭, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp133_scalar_token_transformer_oof_te` | `scalar_token_transformer` | `exp133_scalar_token_transformer_oof_te` | 묶음 시작 | 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp131_tab_cnn_oof_target_mean` | `tab_cnn` | `exp113_tab_cnn_m0` | 단일 변경 개선판 | 목표값 부호화, 산술 파생 조합, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp132_tab_cnn_epochs100` | `tab_cnn` | `exp113_tab_cnn_m0` | 단일 변경 개선판 | 산술 파생 조합, 제약 기반 결측 복원값, 학습 기반 결측 복원 | fold_train | 3 | 통과 |
| `exp139_realmlp_reference_qnormal_train_test` | `realmlp` | `exp124_realmlp_dtype_fix` | 전처리 기준 범위 변형판 | 전처리 기준 집합 값 좌표, 제약 기반 결측 복원 구간 폭 | train_test | 3 | 통과 |

## 모델 계보 묶음

| 묶음 | 구성원 수 | 구성원 | 풀 밖 이전판 |
| --- | ---: | --- | --- |
| `exp006_te_drop_gaming` | 2 | `exp006_te_drop_gaming`, `exp011_resid_pair` | 없음 |
| `exp022_orig_knn` | 1 | `exp022_orig_knn` | 없음 |
| `exp023_orig_proxy_residual` | 1 | `exp023_orig_proxy_residual` | 없음 |
| `exp025_constrained_impute` | 5 | `exp025_constrained_impute`, `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw`, `exp027_recon_ce`, `exp048_lgb_orig_cdf_diff` | `exp026_constrained_impute_nowidth` |
| `exp035_lattice_te` | 1 | `exp035_lattice_te` | 없음 |
| `exp058_logreg_onehot` | 3 | `exp058_logreg_onehot`, `exp107_logreg_onehot_nn10`, `exp108_logreg_onehot_nn10_l1` | 없음 |
| `exp059_lookup_transformer` | 4 | `exp059_lookup_transformer`, `exp081_lookup_fold_initialization_avg3`, `exp127_lookup_muon`, `exp131_lookup_bivariate_plr5` | 없음 |
| `exp070_cat_exact_cats` | 2 | `exp070_cat_exact_cats`, `exp071_cat_exact_no_te` | `exp057_cat_xgb_impute_comps5` |
| `exp067_tabpfn3` | 1 | `exp067_tabpfn3` | 없음 |
| `exp110_lgb_kitopl_no_te` | 2 | `exp110_lgb_kitopl_no_te`, `exp117_ag25_gbm_r21` | `exp074_lgb_kitopl_d2_bundle` |
| `exp111_xgb_depth8_no_te` | 2 | `exp111_xgb_depth8_no_te`, `exp135_xgb_hpo_trial30` | `exp045_xgb_depth8` |
| `exp106_lookup_fixed24_train_test_preprocessing` | 1 | `exp106_lookup_fixed24_train_test_preprocessing` | 없음 |
| `exp113_tab_cnn_m0` | 3 | `exp113_tab_cnn_m0`, `exp131_tab_cnn_oof_target_mean`, `exp132_tab_cnn_epochs100` | 없음 |
| `exp085_contextual_spline_m0` | 1 | `exp085_contextual_spline_m0` | 없음 |
| `exp124_realmlp_dtype_fix` | 4 | `exp124_realmlp_dtype_fix`, `exp134_realmlp_muon`, `exp136_realmlp_muon_recon_widths`, `exp139_realmlp_reference_qnormal_train_test` | `exp121_realmlp_fixed4_two_init` |
| `exp137_tabm_recon_widths` | 1 | `exp137_tabm_recon_widths` | `exp065_tabm` |
| `exp133_scalar_token_transformer_oof_te` | 1 | `exp133_scalar_token_transformer_oof_te` | 없음 |

## 정보 관점

| 정보 관점 | 구성원 수 | 구성원 |
| --- | ---: | --- |
| 격자 이변수 목표값 부호화 | 1 | `exp035_lattice_te` |
| 목표값 부호화 | 17 | `exp006_te_drop_gaming`, `exp011_resid_pair`, `exp022_orig_knn`, `exp023_orig_proxy_residual`, `exp025_constrained_impute`, `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw`, `exp035_lattice_te`, `exp070_cat_exact_cats`, `exp067_tabpfn3`, `exp117_ag25_gbm_r21`, `exp027_recon_ce`, `exp048_lgb_orig_cdf_diff`, `exp135_xgb_hpo_trial30`, `exp137_tabm_recon_widths`, `exp133_scalar_token_transformer_oof_te`, `exp131_tab_cnn_oof_target_mean` |
| 산술 파생 조합 | 27 | `exp011_resid_pair`, `exp022_orig_knn`, `exp023_orig_proxy_residual`, `exp025_constrained_impute`, `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw`, `exp035_lattice_te`, `exp059_lookup_transformer`, `exp070_cat_exact_cats`, `exp067_tabpfn3`, `exp081_lookup_fold_initialization_avg3`, `exp110_lgb_kitopl_no_te`, `exp111_xgb_depth8_no_te`, `exp071_cat_exact_no_te`, `exp106_lookup_fixed24_train_test_preprocessing`, `exp117_ag25_gbm_r21`, `exp113_tab_cnn_m0`, `exp085_contextual_spline_m0`, `exp127_lookup_muon`, `exp027_recon_ce`, `exp048_lgb_orig_cdf_diff`, `exp135_xgb_hpo_trial30`, `exp131_lookup_bivariate_plr5`, `exp137_tabm_recon_widths`, `exp133_scalar_token_transformer_oof_te`, `exp131_tab_cnn_oof_target_mean`, `exp132_tab_cnn_epochs100` |
| 원본 프록시 분포 좌표 | 1 | `exp048_lgb_orig_cdf_diff` |
| 원본 프록시 최근접 라벨 | 3 | `exp022_orig_knn`, `exp107_logreg_onehot_nn10`, `exp108_logreg_onehot_nn10_l1` |
| 원본 프록시 통계 사전 | 2 | `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw` |
| 원시 관측 열 전용 | 3 | `exp058_logreg_onehot`, `exp124_realmlp_dtype_fix`, `exp134_realmlp_muon` |
| 전처리 기준 집합 값 좌표 | 1 | `exp139_realmlp_reference_qnormal_train_test` |
| 정확값 범주 복제 | 2 | `exp070_cat_exact_cats`, `exp071_cat_exact_no_te` |
| 정확값 빈도 부호화 | 1 | `exp027_recon_ce` |
| 정확값 어휘 조회 | 6 | `exp059_lookup_transformer`, `exp081_lookup_fold_initialization_avg3`, `exp106_lookup_fixed24_train_test_preprocessing`, `exp085_contextual_spline_m0`, `exp127_lookup_muon`, `exp131_lookup_bivariate_plr5` |
| 제약 기반 결측 복원 구간 폭 | 4 | `exp025_constrained_impute`, `exp136_realmlp_muon_recon_widths`, `exp137_tabm_recon_widths`, `exp139_realmlp_reference_qnormal_train_test` |
| 제약 기반 결측 복원값 | 22 | `exp025_constrained_impute`, `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw`, `exp035_lattice_te`, `exp070_cat_exact_cats`, `exp067_tabpfn3`, `exp081_lookup_fold_initialization_avg3`, `exp110_lgb_kitopl_no_te`, `exp111_xgb_depth8_no_te`, `exp071_cat_exact_no_te`, `exp117_ag25_gbm_r21`, `exp113_tab_cnn_m0`, `exp085_contextual_spline_m0`, `exp127_lookup_muon`, `exp027_recon_ce`, `exp048_lgb_orig_cdf_diff`, `exp135_xgb_hpo_trial30`, `exp131_lookup_bivariate_plr5`, `exp137_tabm_recon_widths`, `exp133_scalar_token_transformer_oof_te`, `exp131_tab_cnn_oof_target_mean`, `exp132_tab_cnn_epochs100` |
| 학습 기반 결측 복원 | 13 | `exp070_cat_exact_cats`, `exp067_tabpfn3`, `exp081_lookup_fold_initialization_avg3`, `exp071_cat_exact_no_te`, `exp117_ag25_gbm_r21`, `exp113_tab_cnn_m0`, `exp085_contextual_spline_m0`, `exp127_lookup_muon`, `exp131_lookup_bivariate_plr5`, `exp137_tabm_recon_widths`, `exp133_scalar_token_transformer_oof_te`, `exp131_tab_cnn_oof_target_mean`, `exp132_tab_cnn_epochs100` |

## 재심사에 넘기는 제약

시드별 OOF 산출물이 없어 시드 평균을 독립 재계산하지 못한 구성원이 8개다: `exp006_te_drop_gaming`, `exp011_resid_pair`, `exp022_orig_knn`, `exp023_orig_proxy_residual`, `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw`, `exp035_lattice_te`, `exp058_logreg_onehot`.
이 구성원들은 #98 이전 실행이라 시드 단위로 짝지은 대조를 만들 수 없다.
성능 동등 대역을 시드 단위로 잴 계획이면 이 제약을 먼저 반영해야 한다.

설정 스키마가 하나가 아니다.
`exp006_te_drop_gaming`, `exp011_resid_pair`는 `features.providers` 이전의 `include`/`fold_fit`/`derived` 스키마를 쓴다.
저장소 현재 `configs/` 파일이 실행 당시 설정과 다른 사례도 있어, 이 장부는 실행 산출물의 설정만 읽는다.

설정 차이로는 보이지 않는 계보가 있다.
`exp124_realmlp_dtype_fix`는 이전판과 설정 잎이 같고 차이가 코드 수정에만 있다.
따라서 계보 판단을 설정 차이만으로 대신할 수 없다.

## 분류 규칙

모델 계열은 실행 산출물 설정의 `model.kind` 하나로 정한다.

모델 계보 묶음은 후보 풀 장부의 진입 근거나 그 근거가 가리키는 이슈가 이전판을 이름으로 지목한 간선만 모아 만든 연결 성분이다.
같은 모델 계열이라는 사실만으로는 간선을 만들지 않는다.
이전판이 현재 풀 밖이어도 간선을 남기므로, 같은 풀 밖 이전판을 공유하는 구성원은 한 묶음이 된다.
설정 잎 단위 포함 관계는 각 구성원의 `config_relations`에 사실로만 남기고 묶음을 만들지 않는다.
생략된 기본값은 잎으로 나타나지 않아 포함 관계가 의미 동일성을 뜻하지 않기 때문이다.

정보 관점은 컬럼 제공자 종류와, CONTEXT.md가 정보 관점으로 이름 붙인 학습기 설정 두 가지에서 유도한다.
학습기 설정 가운데 정보 관점으로 세는 항목은 정확값 어휘(`lookup_cols`, `exact_cols`)와 전처리 기준 집합 값 좌표(`reference_qnormal_columns`)뿐이다.
나머지 `model.params` 항목은 용량·최적화·실행 설정으로 보고 모델 계열 축이 진다.
컬럼 제공자를 쓰지 않고 이 두 설정도 없는 구성원은 `원시 관측 열 전용`으로 적는다.
`preprocessing_scope`는 정보 관점이 아니라 전처리 기준 집합의 범위 사실로 따로 적는다.

## 재현

```
uv run python scripts/freeze_pool_baseline.py
```

이 명령은 모델을 다시 학습하지 않고 장부와 실행 산출물만 읽는다.
