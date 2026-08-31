# 결측 증강 전파 일괄 판정

이 문서는 GitHub 이슈 [결측 증강 전파 후보를 동결 OOF 조건부로 일괄 판정해 공식 풀을 확정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/512)의 변경 불가 종결 기록이다.

## 결론

교정 실행을 포함한 정확 검색의 제안이 두 OOF 관문과 재학습 계획의 정적 준비 상태 검증을 통과해 후보 풀과 전체 자료 재학습 계획을 함께 바꿨다.
전체 OOF 검색은 현재 풀 AUC `0.969835989200`에서 `0.969882875814`로 `+0.000046886614` 개선되는 5개 원자 교체를 선택했다.
선택된 원본 자리는 다음과 같다.

- `exp035_lattice_te`
- `exp058_logreg_onehot`
- `exp070_cat_exact_cats`
- `exp110_lgb_kitopl_no_te`
- `exp131_lookup_bivariate_plr5`

부분 결과와 Public 점수는 판정에 사용하지 않았다.

## 기존 판정 정정

앞선 종결 기록은 이슈 511에서 유효성이 확인된 교정 실행 7개를 사전 기록의 오래된 출처 커밋과 다르다는 이유로 제외했다.
그 결과 실제 완결 짝 24개를 17개로 줄여 읽었고, 현재 풀의 기존 중복 위반을 해소하는 `exp131_lookup_bivariate_plr5` 교정판도 제외했다.
검색 상태를 한 건도 평가하지 않은 채 제안 풀에 도달할 수 없다고 결론 내린 것은 잘못이었다.
교정 계약을 적용한 이번 입력은 완결 24짝과 미완결 10짝이며, 완결 짝 중 직접 OOF 차이가 양수인 짝은 19개다.
직접 짝비교 차이의 부호는 검색 입력 포함이나 최종 교체를 단독으로 결정하지 않았다.
허용한 교정 실행은 다음과 같다.

- `exp111_xgb_depth8_no_te`
- `exp106_lookup_fixed24_train_test_preprocessing`
- `exp085_contextual_spline_m0`
- `exp135_xgb_hpo_trial30`
- `exp131_lookup_bivariate_plr5`
- `exp137_tabm_recon_widths`
- `exp139_realmlp_reference_qnormal_train_test`

## 도달 가능성과 정확 검색

이슈 511에서 유효 판정된 exp131 교정 결측 증강판은 현재 exp131과 exp157의 기존 중복 위반을 해소하면서 새 위반을 만들지 않는다. 따라서 현재 풀을 점수 기준점으로 두고 첫 채택 이동부터 전체 OOF 중복 불변식을 만족하는 정확 검색을 시작할 수 있다.
전체 OOF와 바깥 분할 검색에서 중복 불변식을 지킨 채 총 1658개 고유 상태를 정확 채점했다.
최종 전체 OOF 제안의 선택 번호는 `[7, 8, 10, 12, 22]`이고 중복 위반은 `[]`다.

## 채택 관문

동결 OOF 조건부 절차 점수 차이는 `+0.000044152982`이며 관문 통과 여부는 `true`다.
핵심 결합 방식 세 가지에서 각 풀의 최선 방식끼리 비교한 직접 중첩 OOF 차이는 `+0.000046886614`이며 관문 통과 여부는 `true`다.
현재 풀의 최선 방식은 `shrunk_rank_logit_logistic`, 제안 풀의 최선 방식은 `shrunk_rank_logit_logistic`다.
직접 중첩 비교의 바깥 분할 승수는 `5/5`다.

## 재학습 준비 상태

새로 선택된 결측 증강판 5개가 제안 풀과 같은 해시의 검증된 재학습 계획에 포함되는지 정적으로 확인했다.
이슈 512에서는 모델 학습과 시험 예측 생성을 실행하지 않았으며 실제 전체 자료 재학습은 후속 생산 단계로 넘겼다.

- `mpv1_exp035_lattice_te_missingness_augmented`: 계획 예산 `{"42": 258, "43": 239, "44": 255}`, 항목 해시 `ed2647ad001feea26bb444b7c894bda3af2b3c6422d21a0b4795c411f0737c7c`
- `mpv1_exp058_logreg_onehot_missingness_augmented`: 계획 예산 `{"42": null}`, 항목 해시 `d14085e599351ee1fe8b77506a10f8db0f207a375b7d40a538aa48b1efc03529`
- `mpv1_exp070_cat_exact_cats_missingness_augmented`: 계획 예산 `{"42": 5021, "43": 4960, "44": 5268}`, 항목 해시 `6b700439156830b3996a72276360bb4c2fd6232235072a5271c9ed040b6c8eb5`
- `mpv1_exp110_lgb_kitopl_no_te_missingness_augmented`: 계획 예산 `{"42": 5774, "43": 5655, "44": 6043}`, 항목 해시 `bdee5c036a6debf8d325ab007f015810d37c8d8f3bf3aa1427b80f153baa4755`
- `mpv1_exp131_lookup_bivariate_plr5_missingness_augmented`: 계획 예산 `{"42": 15, "43": 15, "44": 15}`, 항목 해시 `7a6f2ef2c279aeed614c397d74fae9ae329412d291b758ff5f341bbf0be4451a`

## 공식 장부

- `artifacts/pool.yaml`: `40947563a00cab8212498c7e339517e387979b14c6477c6ce8e196036e02044c`
- `artifacts/full-refit-plan.yaml`: `89edf321b1821f1de645799f2353705c461065a275263375d5479e3edd6b006c`

두 파일은 같은 공식화 경로에서 함께 바뀌었다.

## 근거 파일

- 입력 묶음: `input-bundle.json` (`54a1fab1b1408791a04b5c415b9c2c4edff9d972eb6c7571ee5671ff828e280d`)
- 도달 가능성 기록: `preflight.json` (`aae096fa319f8c43d90a225d4e3effa50ad19c2f4f9d7e908caf0350bc732ad6`)
- 정확 검색: `search.json` (`e4bf476f51a7e105a1e809633460d8d6ff32c3bdf81c8c2ab745fd89735b5021`)
- 조건부 절차 관문: `conditional-gate.json` (`87d5d0aa900a05ded721d3d27b22cb728266d83e876504b28eae8b1c5e926e89`)
- 직접 중첩 관문: `direct-nested-gate.json` (`51f93addc0d35d34e9c96fbc4e34d426f3bd5ecd114a13ae5870f9b8eca9af12`)
- 선택 근거: `selection-evidence.json` (`a1e09244d1f8c814ec94d48c9f10551fe6906ba5c6fc9ec949c01f2bf2db8b28`)
- 재학습 준비 상태: `full-refit-readiness.json` (`e1fb052594d0a25479838025990a750380d72a4575c8bfdfd7bfc7bd18dc61a8`)
- 최종 판정: `judgment.json` (`eb69ddf211dbd3c9242fb6e4ea63349c7c5173af2c421cb814c29621635b430a`)
- 파일 목록: `manifest.sha256`
