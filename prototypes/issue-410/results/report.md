# 이슈 #410 시제품 결과: 전진 추가·역방향 제거의 동결 OOF 조건부 절차

버리는 시제품 `prototype_pool_rebuild_search.py`가 `candidate-pool-rebuild-v1` 계약(ADR 0003)의 정확 검색을 현재 33개 OOF에 적용한 결과다.
후보 풀과 재학습 계획은 바꾸지 않았다.
동결 신원 `4bdee56d5b2379d115328f86419b8556962639a34d77346f9d4fe93464c988f3`, 코드 해시 `f0d2364598fbbcf445ef1026cf7bd2e5632ffe56b0ad7d9c356368efd1a462ed`, 실행 커밋 `4d1e8eaeb11a1165369d6fd357ac6ce3acd85145`.

## 답

저비용 시제품이 정확한 전진 추가, 원자 교체, 역방향 제거와 2개 묶음 구제를 실용적인 시간 안에 수행했다.
범위 6개(전체 + 바깥 분할 5개)의 검색 벽시계 합은 83분(평가 3383회, 로지스틱 적합 36885회)이고, 드라이버 2개를 작업자 7개씩으로 동시에 돌려 실제 경과는 약 50분이었다.
준비 33초, 마무리(held-out 예측·앵커·핵심 전략 3개 직접 대조·동일성 재확인) 573초가 더 든다.
참조 구현 `ensemble.evaluate_nested`와 AUC 절대 차이 0.0으로 일치했고, 강제 종료 뒤 재개는 부분 완료 단계의 평가를 캐시에서 재사용해 정상 진행됐다.

다만 이 자료에서 재구축 제안 풀은 계약의 채택 관문을 통과하지 못한다.
전체 동결 OOF 제안 풀(13개)은 현재 32개 풀보다 nested AUC가 +5.945e-06 높고 핵심 전략 3개 최선끼리 차이도 +5.945e-06로 양수지만, 동결 OOF 조건부 절차 점수는 -2.504e-06(분할 승수 2/5)로 낮다.
세 차이 모두 성능 동등 대역(±2.2e-05) 안이다.

## 제안 풀(전체 동결 OOF)

13개: `exp156_lookup_bivariate_plr5_initavg8`, `exp022_orig_knn`, `exp025_constrained_impute`, `exp035_lattice_te`, `exp070_cat_exact_cats`, `exp067_tabpfn3`, `exp081_lookup_fold_initialization_avg3`, `exp111_xgb_depth8_no_te`, `exp106_lookup_fixed24_train_test_preprocessing`, `exp117_ag25_gbm_r21`, `exp085_contextual_spline_m0`, `exp134_realmlp_muon`, `exp139_realmlp_reference_qnormal_train_test`.

현재 32개에서 빠진 20개: `exp006_te_drop_gaming`, `exp011_resid_pair`, `exp023_orig_proxy_residual`, `exp032_recon_orig_mean_top3`, `exp058_logreg_onehot`, `exp059_lookup_transformer`, `exp110_lgb_kitopl_no_te`, `exp071_cat_exact_no_te`, `exp113_tab_cnn_m0`, `exp027_recon_ce`, `exp048_lgb_orig_cdf_diff`, `exp135_xgb_hpo_trial30`, `exp131_lookup_bivariate_plr5`, `exp136_realmlp_muon_recon_widths`, `exp137_tabm_recon_widths`, `exp133_scalar_token_transformer_oof_te`, `exp131_tab_cnn_oof_target_mean`, `exp132_tab_cnn_epochs100`, `exp140_realmlp_orig_cdf_diff`, `exp157_lookup_muon_initavg8`.

| 항목 | 제안 풀 | 현재 32개 풀 | 차이 |
|---|---|---|---|
| 전체 nested AUC(`shrunk_rank_logit_logistic`) | 0.9697509427 | 0.9697449975 | +5.945e-06 |
| 동결 OOF 조건부 절차 점수 | 0.9697424939 | 0.9697449975 | -2.504e-06 |
| 직접 대조 `shrunk_rank_logit_logistic` | 0.9697509427 | 0.9697449975 | +5.945e-06 |
| 직접 대조 `missing_segmented_rank_logit` | 0.9697436159 | 0.9697358811 | +7.735e-06 |
| 직접 대조 `missing_interaction_rank_logit` | 0.9697450268 | 0.9697355898 | +9.437e-06 |
| 직접 대조 최선끼리 | 0.9697509427 (shrunk_rank_logit_logistic) | 0.9697449975 (shrunk_rank_logit_logistic) | +5.945e-06 |

조건부 절차 점수의 분할별 held-out AUC:

| 분할 | 제안(분할 풀 크기) | 현재 32개 | 차이 |
|---|---|---|---|
| 0 | 0.9691745 (12) | 0.9691638 | +1.07e-05 |
| 1 | 0.9698356 (15) | 0.9698415 | -5.92e-06 |
| 2 | 0.9698695 (13) | 0.9698888 | -1.93e-05 |
| 3 | 0.9703013 (13) | 0.9703130 | -1.17e-05 |
| 4 | 0.9695316 (12) | 0.9695179 | +1.36e-05 |

선택된 λ는 모든 분할과 두 풀에서 1.0(수축 없음)이었다.

## 검색 경과

| 범위 | 단계 | 평가 | 적합 | 벽시계(초) | 평가당 평균/최대(초) | 최종 크기 | 최종 AUC | 묶음 채택 | 재개 |
|---|---|---|---|---|---|---|---|---|---|
| full | 16 | 611 | 9165 | 1579 | 17.1/31.8 | 13 | 0.9697509427 | exp025_constrained_impute + exp117_ag25_gbm_r21 | 0 |
| 0 | 15 | 519 | 5190 | 675 | 8.6/13.2 | 12 | 0.9699001022 | 없음 | 1 |
| 1 | 18 | 570 | 5700 | 500 | 5.8/9.0 | 15 | 0.9697346081 | exp117_ag25_gbm_r21 + exp131_tab_cnn_oof_target_mean | 1 |
| 2 | 16 | 568 | 5680 | 767 | 8.9/20.1 | 13 | 0.9697210965 | exp025_constrained_impute + exp117_ag25_gbm_r21 | 1 |
| 3 | 16 | 568 | 5680 | 791 | 9.2/14.1 | 13 | 0.9696106658 | exp025_constrained_impute + exp131_tab_cnn_oof_target_mean | 0 |
| 4 | 15 | 547 | 5470 | 656 | 7.9/11.8 | 12 | 0.9698076552 | exp025_constrained_impute + exp117_ag25_gbm_r21 | 0 |

단계별 벽시계(초)는 전진 1차가 가장 길고 2개 묶음 단계가 그다음이다.

| 범위 | forward1 | backward1 | forward2 | pair | forward3 | backward2 |
|---|---|---|---|---|---|---|
| full | 538 (252) | 31 (8) | 0 (24) | 758 (275) | 209 (63) | 44 (12) |
| 0 | 336 (318) | 21 (11) | 0 (21) | 319 (190) | - | - |
| 1 | 224 (318) | 13 (11) | 0 (21) | 201 (190) | 45 (37) | 17 (14) |
| 2 | 306 (275) | 20 (9) | 0 (23) | 333 (231) | 75 (41) | 33 (12) |
| 3 | 331 (275) | 18 (9) | 0 (23) | 355 (231) | 70 (41) | 17 (12) |
| 4 | 268 (275) | 17 (9) | 0 (23) | 316 (231) | 33 (21) | 21 (11) |

괄호는 평가한 이동 수다.
전진 재수렴(forward2)은 직전 전진 수렴과 같은 풀에서 같은 이동을 보므로 전부 캐시에서 읽혀 0초다.

### 수락된 이동

- 범위 full: add exp139_realmlp_reference_qnormal_train_test (+2.20e-04, 5/5) → add exp070_cat_exact_cats (+6.36e-05, 5/5) → add exp106_lookup_fixed24_train_test_preprocessing (+3.00e-05, 5/5) → add exp085_contextual_spline_m0 (+1.79e-05, 5/5) → add exp081_lookup_fold_initialization_avg3 (+5.24e-06, 4/5) → add exp111_xgb_depth8_no_te (+9.75e-06, 4/5) → add exp035_lattice_te (+4.88e-06, 2/5) → add exp022_orig_knn (+4.11e-06, 4/5) → pair exp025_constrained_impute+exp117_ag25_gbm_r21 (+7.13e-06, 4/5) → add exp134_realmlp_muon (+3.15e-06, 3/5) → add exp067_tabpfn3 (+1.30e-06, 3/5)
- 범위 0: add exp139_realmlp_reference_qnormal_train_test (+2.26e-04, 4/4) → add exp070_cat_exact_cats (+6.97e-05, 4/4) → add exp106_lookup_fixed24_train_test_preprocessing (+2.20e-05, 4/4) → add exp085_contextual_spline_m0 (+1.81e-05, 4/4) → add exp081_lookup_fold_initialization_avg3 (+7.29e-06, 4/4) → add exp035_lattice_te (+1.32e-05, 4/4) → add exp157_lookup_muon_initavg8 (+3.71e-06, 3/4) → add exp025_constrained_impute (+4.28e-06, 3/4) → add exp117_ag25_gbm_r21 (+1.82e-06, 3/4) → add exp111_xgb_depth8_no_te (+5.49e-06, 3/4) → add exp006_te_drop_gaming (+4.77e-06, 4/4)
- 범위 1: add exp139_realmlp_reference_qnormal_train_test (+2.23e-04, 4/4) → add exp070_cat_exact_cats (+6.25e-05, 4/4) → add exp106_lookup_fixed24_train_test_preprocessing (+2.71e-05, 4/4) → add exp085_contextual_spline_m0 (+1.72e-05, 4/4) → add exp111_xgb_depth8_no_te (+5.84e-06, 4/4) → add exp081_lookup_fold_initialization_avg3 (+5.64e-06, 3/4) → add exp131_lookup_bivariate_plr5 (+1.04e-05, 3/4) → add exp035_lattice_te (+3.95e-06, 2/4) → add exp006_te_drop_gaming (+4.30e-06, 3/4) → add exp071_cat_exact_no_te (+1.94e-06, 2/4) → add exp025_constrained_impute (+6.44e-06, 4/4) → pair exp117_ag25_gbm_r21+exp131_tab_cnn_oof_target_mean (+3.25e-06, 3/4) → add exp048_lgb_orig_cdf_diff (+1.88e-06, 3/4)
- 범위 2: add exp139_realmlp_reference_qnormal_train_test (+2.19e-04, 4/4) → add exp070_cat_exact_cats (+6.86e-05, 4/4) → add exp106_lookup_fixed24_train_test_preprocessing (+2.94e-05, 4/4) → add exp085_contextual_spline_m0 (+1.90e-05, 4/4) → add exp071_cat_exact_no_te (+1.09e-06, 3/4) → add exp081_lookup_fold_initialization_avg3 (+1.14e-05, 4/4) → add exp035_lattice_te (+3.24e-06, 2/4) → add exp006_te_drop_gaming (+8.03e-06, 3/4) → add exp131_lookup_bivariate_plr5 (+7.63e-06, 3/4) → pair exp025_constrained_impute+exp117_ag25_gbm_r21 (+7.02e-06, 3/4) → add exp111_xgb_depth8_no_te (+1.75e-06, 3/4)
- 범위 3: add exp139_realmlp_reference_qnormal_train_test (+2.12e-04, 4/4) → add exp070_cat_exact_cats (+6.39e-05, 4/4) → add exp106_lookup_fixed24_train_test_preprocessing (+2.49e-05, 4/4) → add exp085_contextual_spline_m0 (+1.90e-05, 4/4) → add exp081_lookup_fold_initialization_avg3 (+4.73e-06, 4/4) → add exp035_lattice_te (+1.04e-05, 4/4) → add exp132_tab_cnn_epochs100 (+3.80e-06, 2/4) → add exp157_lookup_muon_initavg8 (+2.35e-06, 2/4) → add exp006_te_drop_gaming (+9.11e-06, 3/4) → pair exp025_constrained_impute+exp131_tab_cnn_oof_target_mean (+3.97e-06, 2/4) → add exp110_lgb_kitopl_no_te (+3.06e-06, 3/4)
- 범위 4: add exp139_realmlp_reference_qnormal_train_test (+2.19e-04, 4/4) → add exp070_cat_exact_cats (+6.29e-05, 4/4) → add exp106_lookup_fixed24_train_test_preprocessing (+3.44e-05, 4/4) → add exp085_contextual_spline_m0 (+1.86e-05, 4/4) → add exp131_lookup_bivariate_plr5 (+4.01e-06, 3/4) → add exp081_lookup_fold_initialization_avg3 (+7.59e-06, 3/4) → add exp035_lattice_te (+1.06e-05, 4/4) → add exp006_te_drop_gaming (+6.71e-06, 3/4) → add exp071_cat_exact_no_te (+4.00e-06, 4/4) → pair exp025_constrained_impute+exp117_ag25_gbm_r21 (+3.96e-06, 2/4)

역방향 제거는 6개 범위 어디서도 양수 이동이 없었다.
원자 교체(exp131 Lookup ↔ exp157 Lookup)는 한쪽이 풀에 있는 모든 단계에서 평가됐고 전부 음수였다.
2개 묶음 단계는 6개 범위 중 5개에서 양수 묶음을 찾아 채택했으며, `exp025_constrained_impute`는 단독 추가로는 양수가 아니었지만 `exp117_ag25_gbm_r21` 또는 `exp131_tab_cnn_oof_target_mean`과 묶일 때 양수가 됐다.

### 경고

수락된 이동 67개 가운데 50개가 성능 동등 대역 절대값 2.207e-05 안의 차이였고, 8개는 분할 승수가 과반 이하였다.
각 범위의 첫 세 이동(exp139, exp070, exp106 추가)만 대역 밖이다.

### 대리 순위와의 비교

역방향 단계마다 균등 순위 평균(#366의 대리 선별)으로 제거 후보를 줄 세운 1위가 정확 판정 1위와 같은지 비교했다.

| 범위 | 단계 | 대리 1위 | 정확 1위 | 대리 1위의 정확 순위 |
|---|---|---|---|---|
| full | backward1 | exp022_orig_knn | exp022_orig_knn | 1 |
| full | backward2 | exp025_constrained_impute | exp067_tabpfn3 | 5 |
| 0 | backward1 | exp006_te_drop_gaming | exp006_te_drop_gaming | 1 |
| 1 | backward1 | exp006_te_drop_gaming | exp006_te_drop_gaming | 1 |
| 1 | backward2 | exp025_constrained_impute | exp048_lgb_orig_cdf_diff | 6 |
| 2 | backward1 | exp006_te_drop_gaming | exp071_cat_exact_no_te | 5 |
| 2 | backward2 | exp006_te_drop_gaming | exp111_xgb_depth8_no_te | 6 |
| 3 | backward1 | exp132_tab_cnn_epochs100 | exp006_te_drop_gaming | 2 |
| 3 | backward2 | exp132_tab_cnn_epochs100 | exp110_lgb_kitopl_no_te | 2 |
| 4 | backward1 | exp006_te_drop_gaming | exp071_cat_exact_no_te | 2 |
| 4 | backward2 | exp006_te_drop_gaming | exp070_cat_exact_cats | 4 |

제거가 양수인 단계가 없어 최종 풀은 달라지지 않지만, 대리 1위만 정확 판정하는 방식은 9회 중 6회에서 정확 최선 후보를 보지 못했을 것이다.

## 선택 안정성

제안 풀 구성원이 바깥 분할 5개의 선택 풀에 남은 횟수:

| 구성원 | 잔존 |
|---|---|
| exp025_constrained_impute | 5/5 |
| exp035_lattice_te | 5/5 |
| exp070_cat_exact_cats | 5/5 |
| exp081_lookup_fold_initialization_avg3 | 5/5 |
| exp085_contextual_spline_m0 | 5/5 |
| exp106_lookup_fixed24_train_test_preprocessing | 5/5 |
| exp139_realmlp_reference_qnormal_train_test | 5/5 |
| exp156_lookup_bivariate_plr5_initavg8 | 5/5 |
| exp117_ag25_gbm_r21 | 4/5 |
| exp111_xgb_depth8_no_te | 3/5 |
| exp022_orig_knn | 0/5 |
| exp067_tabpfn3 | 0/5 |
| exp134_realmlp_muon | 0/5 |

분할 풀에는 선택됐지만 제안 풀에는 없는 후보:

- 분할 0: `exp006_te_drop_gaming`, `exp157_lookup_muon_initavg8`
- 분할 1: `exp006_te_drop_gaming`, `exp071_cat_exact_no_te`, `exp048_lgb_orig_cdf_diff`, `exp131_lookup_bivariate_plr5`, `exp131_tab_cnn_oof_target_mean`
- 분할 2: `exp006_te_drop_gaming`, `exp071_cat_exact_no_te`, `exp131_lookup_bivariate_plr5`
- 분할 3: `exp006_te_drop_gaming`, `exp110_lgb_kitopl_no_te`, `exp131_tab_cnn_oof_target_mean`, `exp132_tab_cnn_epochs100`, `exp157_lookup_muon_initavg8`
- 분할 4: `exp006_te_drop_gaming`, `exp071_cat_exact_no_te`, `exp131_lookup_bivariate_plr5`

`exp006_te_drop_gaming`은 분할 풀 5개 모두에 있으면서 제안 풀에는 없고, `exp022_orig_knn`, `exp067_tabpfn3`, `exp134_realmlp_muon`은 제안 풀에만 있다.

## 중복 불변식

| 풀 | 크기 | 최대 스피어만 | 쌍 | 통과 |
|---|---|---|---|---|
| current32 | 32 | 0.998144 | exp131_lookup_bivariate_plr5 / exp157_lookup_muon_initavg8 | 아니오 |
| proposal | 13 | 0.996898 | exp134_realmlp_muon / exp139_realmlp_reference_qnormal_train_test | 예 |
| scope-0 | 12 | 0.995583 | exp025_constrained_impute / exp117_ag25_gbm_r21 | 예 |
| scope-1 | 15 | 0.997682 | exp156_lookup_bivariate_plr5_initavg8 / exp131_lookup_bivariate_plr5 | 예 |
| scope-2 | 13 | 0.997682 | exp156_lookup_bivariate_plr5_initavg8 / exp131_lookup_bivariate_plr5 | 예 |
| scope-3 | 13 | 0.995427 | exp006_te_drop_gaming / exp025_constrained_impute | 예 |
| scope-4 | 12 | 0.997682 | exp156_lookup_bivariate_plr5_initavg8 / exp131_lookup_bivariate_plr5 | 예 |

검색 결과 풀 6개는 모두 `0.998` 미만이다.
현재 32개 풀은 exp131 Lookup과 exp157 Lookup 쌍(0.998144) 때문에 불변식을 만족하지 않으며, 이는 #409에서 이미 알던 사실이다.
전체 쌍별 상관은 `report.json`의 `duplicate_invariants.*.all_pairs`에 있다.

## 동일성 확인

| 풀 크기 | 범위 | 참조 AUC | 시제품 AUC | 차이 | 참조(초) | 시제품(초) | 적합 수 |
|---|---|---|---|---|---|---|---|
| 5 | 2 | 0.9693878177248236 | 0.9693878177248236 | 0.0 | 18.6 | 3.6 | 10 |
| 7 | 4 | 0.9697246207450595 | 0.9697246207450595 | 0.0 | 24.4 | 5.1 | 10 |
| 2 | 전체 | 0.9693829421071594 | 0.9693829421071594 | 0.0 | 16.3 | 4.7 | 15 |
| 5 | 전체 | 0.9694277875780096 | 0.9694277875780096 | 0.0 | 34.4 | 6.4 | 15 |
| 11 | 전체 | 0.9696630637224702 | 0.9696630637224702 | 0.0 | 71.2 | 8.7 | 15 |

마무리 단계의 핵심 전략 직접 대조에서도 참조 구현의 `shrunk_rank_logit_logistic` 값이 시제품 검색 점수와 차이 0.0(제안), 0.0(현재 32개)로 같았다.

## 자원과 병목

- 입력: 동결 순위 변환 25개 파일 4.7GB(제외 fold 집합마다 691,369 × 33 float64), 준비 32초.
- 메모리: 작업자 RSS 최대 약 6GB이나 대부분 공유 메모리 매핑 파일 페이지이며, 동시 작업자 14개에서 시스템 여유 65% 유지. 마무리 드라이버 최대 RSS 6.9GB.
- 병목: 로지스틱 적합(lbfgs)이 평가 시간의 대부분이다. 전체 범위 평가는 적합 15회, 바깥 분할 제외 범위는 10회이며 풀이 커질수록 적합당 시간이 는다(전체 범위 평가 평균 17초, 최대 32초).
- 2개 묶음 단계는 범위당 190~275개 평가로 단일 단계 중 가장 길다(201~758초).
- 중단·재개: 단계 단위 상태 파일과 평가 단위 캐시(jsonl)를 동결 신원 해시로 묶어, 해시가 같을 때만 이어서 실행한다. 분할 2 5단계 중간(27개 중 14개 완료)과 분할 1 0단계 중간(32개 중 21개 완료)에서 강제 종료 뒤 재시작해 캐시 재사용을 확인했다.

## 산출물

- `precommit.json`: 33개 실행 신원과 순서, OOF 배열 해시, folds·목표값·장부 해시, 전략, 절차, 코드 해시.
- `scope-*/state.json`, `scope-*/steps/*.json`: 단계별 현재 풀, 평가한 모든 이동의 점수·분할 AUC·λ, 선택 결과, 경고.
- `scope-*/evaluations.jsonl`: 모든 평가의 원자료(풀, AUC, 분할 AUC, λ별 AUC, 적합 수, 반복 횟수, 시간).
- `report.json`: 절차 점수, 직접 대조, 안정성, 불변식, 시간, 동일성.
- `equivalence.json`, `driver-*.log`, `finish.log`, `memory-samples.log`.
