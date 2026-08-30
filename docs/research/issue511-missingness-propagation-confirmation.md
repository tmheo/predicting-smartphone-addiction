# 결측 증강 전파 선별 짝비교 최종 실행 기록

이 문서는 GitHub 이슈 [#511](https://github.com/tmheo/predicting-smartphone-addiction/issues/511)의 완결된 3시드 짝비교와 중앙 반입 결과를 기록한다.
결과 확인 뒤 사용자가 GPU 후보를 선별했으므로 사전 동결 34짝 전체가 아니라 최종 선택된 24짝만 완결했다.
TabCNN 계열 3짝은 실행 범위에서 제외했고 GPU 후보 7짝은 비용 검토 뒤 사용자가 선택하지 않았다.

## 결론

- 완결 짝은 24개이며 결측 증강군이 3배 대조군보다 높은 짝은 19개, 낮은 짝은 5개다.
- 가장 높은 결측 증강 OOF AUC는 `exp131_lookup_bivariate_plr5`의 `0.9694062694`다.
- 가장 큰 직접 개선은 `exp106_lookup_fixed24_train_test_preprocessing`의 `+0.0010767164`다.
- 신경망 5짝은 첫 실행과 두 번째 교정을 무효화한 뒤 부모별 학습 경로를 보존하는 세 번째 교정으로 다시 실행했다.
- Vast.ai 계산 자원과 별도 저장 공간 목록은 모두 비어 있으며 추가 과금 자원은 남아 있지 않다.
- 이 기록은 직접 짝비교 실행과 중앙 반입의 완료 근거이며 후보 풀 변경이나 중첩 선별 채택을 수행하지 않는다.

## 완결 결과

| 짝 | 후보 | 공급자 | 3배 대조군 | 결측 증강군 | 차이 |
| ---: | --- | --- | ---: | ---: | ---: |
| 01 | `exp006_te_drop_gaming` | local | 0.9664985589 | 0.9668758281 | +0.0003772691 |
| 02 | `exp011_resid_pair` | local | 0.9673305442 | 0.9676794857 | +0.0003489416 |
| 03 | `exp022_orig_knn` | kaggle | 0.9672731827 | 0.9676698099 | +0.0003966272 |
| 04 | `exp023_orig_proxy_residual` | vast | 0.9672724172 | 0.9675915373 | +0.0003191202 |
| 05 | `exp025_constrained_impute` | kaggle | 0.9675273735 | 0.9678206259 | +0.0002932523 |
| 06 | `exp032_recon_orig_mean_top3` | kaggle | 0.9675635361 | 0.9678601016 | +0.0002965655 |
| 07 | `exp035_lattice_te` | local | 0.9671330571 | 0.9674748763 | +0.0003418193 |
| 08 | `exp058_logreg_onehot` | local | 0.9596852254 | 0.9596153717 | -0.0000698537 |
| 10 | `exp070_cat_exact_cats` | local | 0.9685734289 | 0.9687455118 | +0.0001720829 |
| 12 | `exp110_lgb_kitopl_no_te` | kaggle | 0.9671940316 | 0.9677639451 | +0.0005699134 |
| 13 | `exp111_xgb_depth8_no_te` | local | 0.9646583321 | 0.9655900749 | +0.0009317428 |
| 14 | `exp071_cat_exact_no_te` | vast | 0.9682027652 | 0.9684607649 | +0.0002579996 |
| 15 | `exp106_lookup_fixed24_train_test_preprocessing` | vast | 0.9678089144 | 0.9688856308 | +0.0010767164 |
| 17 | `exp085_contextual_spline_m0` | vast | 0.9681562677 | 0.9683130514 | +0.0001567838 |
| 18 | `exp027_recon_ce` | local | 0.9676715613 | 0.9679660267 | +0.0002944654 |
| 19 | `exp048_lgb_orig_cdf_diff` | kaggle | 0.9675550982 | 0.9678390141 | +0.0002839160 |
| 21 | `exp135_xgb_hpo_trial30` | local | 0.9682973365 | 0.9685052720 | +0.0002079355 |
| 22 | `exp131_lookup_bivariate_plr5` | vast | 0.9693371781 | 0.9694062694 | +0.0000690913 |
| 24 | `exp137_tabm_recon_widths` | vast | 0.9683928038 | 0.9683260724 | -0.0000667315 |
| 28 | `exp139_realmlp_reference_qnormal_train_test` | vast | 0.9685463700 | 0.9681423913 | -0.0004039787 |
| 31 | `exp168_issue413_lgb_no_te_fixed20` | kaggle | 0.9649215109 | 0.9652515060 | +0.0003299951 |
| 32 | `exp197_issue419_lgb_recon_ce_fixed20` | kaggle | 0.9654478861 | 0.9652054887 | -0.0002423974 |
| 33 | `exp183_issue419_cat_exact_fixed10` | vast | 0.9668387323 | 0.9666845501 | -0.0001541822 |
| 34 | `exp209_issue505_lgb_lr_onehot_init` | kaggle | 0.9680530961 | 0.9683458713 | +0.0002927752 |

## 무결성과 교정

모든 완결 짝은 같은 공급자와 실행 환경 등급에서 두 팔을 함께 끝냈다.
각 중앙 실행은 고정된 세 시드, 입력 해시, 깨끗한 출처, 15개 학습 좌표, OOF 재채점과 중앙 묶음 산출물 해시를 통과했다.
`13-exp111`과 `21-exp135`만 XGBoost 고정 학습 길이 진단 수정 출처를 사용했고 설정 해시는 원래 고정 출처와 동일하다.
신경망 5짝은 원본 물리 배치 크기, 부모 행 노출 순서, 최적화 갱신 수, 학습률 일정 위치와 원본 전처리 범위를 보존했다.
교정 3배 대조군이 각 역사적 원본 성능을 재현했으므로 세 번째 신경망 짝비교만 유효하다.

## 실행하지 않은 짝

- `09-exp059_lookup_transformer`: 비용 검토 뒤 GPU 선별 대상에서 제외.
- `11-exp081_lookup_fold_initialization_avg3`: 비용 검토 뒤 GPU 선별 대상에서 제외.
- `16-exp113_tab_cnn_m0`: TabCNN 계열 사전 제외.
- `20-exp134_realmlp_muon`: 비용 검토 뒤 GPU 선별 대상에서 제외.
- `23-exp136_realmlp_muon_recon_widths`: 비용 검토 뒤 GPU 선별 대상에서 제외.
- `25-exp133_scalar_token_transformer_oof_te`: 비용 검토 뒤 GPU 선별 대상에서 제외.
- `26-exp131_tab_cnn_oof_target_mean`: TabCNN 계열 사전 제외.
- `27-exp132_tab_cnn_epochs100`: TabCNN 계열 사전 제외.
- `29-exp140_realmlp_orig_cdf_diff`: 비용 검토 뒤 GPU 선별 대상에서 제외.
- `30-exp157_lookup_muon_initavg8`: 비용 검토 뒤 GPU 선별 대상에서 제외.

## 자원 정리

Vast.ai 계정은 2026-08-30T07:40:13Z에 다시 조회했다.
활성 인스턴스 0개, 별도 저장 공간 0개이며 잔액은 `$20.743657`다.
실행별 비용과 실패·재시도 자원 정산은 이슈 댓글과 로컬 `run-logs/issue511` 장부에 보존한다.

## 근거

- 최종 기계 판독 기록: `artifacts/issue511-missingness-propagation-confirmation.json`
- 실행 전 고정 기록: `artifacts/issue510-missingness-propagation-precommit.json`
- 학습 길이 고정 기록: `artifacts/issue510-paired-training-lengths.json`
- 신경망 교정 계약: `docs/adr/0007-preserve-neural-optimizer-steps-in-replicated-row-comparisons.md`
- 중앙 실행 식별자, OOF 해시와 묶음 manifest 해시는 기계 판독 기록의 각 짝 항목에 있다.
