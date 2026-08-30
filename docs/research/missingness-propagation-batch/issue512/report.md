# 결측 증강 전파 일괄 판정

이 문서는 GitHub 이슈 [결측 증강 전파 후보를 동결 OOF 조건부로 일괄 판정해 공식 풀을 확정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/512)의 변경 불가 종결 기록이다.

## 결론

현재 후보 풀과 전체 자료 재학습 계획을 그대로 유지한다.
허용된 원자 교체로 현재 풀의 중복 위반을 해소할 수 없어 전체 OOF 제안 풀 자체가 존재하지 않는다.
따라서 검색 점수, 동결 OOF 조건부 절차 관문, 핵심 결합 방식 세 가지의 직접 중첩 관문과 전체 자료 재학습 스모크 예행은 시작하지 않았다.
부분 결과와 Public 점수는 판정에 사용하지 않았다.

## 판정 입력

사전 고정한 34개 짝 가운데 일괄 판정 입력 묶음의 완결 짝은 17개이고 미완결 짝은 17개다.
중앙 반입에서 완결로 기록됐지만 사전 기록 출처 커밋과 달라 미완결로 분류한 짝은 7개다.
해당 짝은 다음과 같다.

- `exp111_xgb_depth8_no_te`
- `exp106_lookup_fixed24_train_test_preprocessing`
- `exp085_contextual_spline_m0`
- `exp135_xgb_hpo_trial30`
- `exp131_lookup_bivariate_plr5`
- `exp137_tabm_recon_widths`
- `exp139_realmlp_reference_qnormal_train_test`

그 밖의 미완결 짝은 이슈 511에서 TabCNN 계열 제외 또는 비용 검토 뒤 미실행으로 확정한 짝이다.
완결 짝의 직접 OOF 차이 부호는 입력 포함 여부에 사용하지 않았다.

## 도달 가능성 판정

- `exp131_lookup_bivariate_plr5`와 `exp157_lookup_muon_initavg8`의 스피어만 순위 상관은 `0.9981438822`로 문턱 `0.998` 이상이다.
- 바꿀 수 있는 자리는 `exp131_lookup_bivariate_plr5`, `exp157_lookup_muon_initavg8`이지만 판정 입력을 통과해 검색 이동 자격을 얻은 결측 증강판은 없다.

원자 교체는 해당 자리의 예측만 바꾸므로 다른 자리를 바꾸어 이 중복 관계를 없앨 수 없다.
모든 도달 가능한 상태가 같은 위반을 보존하므로 전체 OOF 제안 풀의 모든 구성원 쌍이 `0.998` 미만이어야 한다는 조건을 만족할 수 없다.
이 판정은 검색 결과를 본 뒤 문턱을 바꾼 것이 아니라 사전 기록의 출처와 중복 조건을 그대로 적용한 결과다.

## 공식 장부

- `artifacts/pool.yaml`: `c513443b6d1cc8af348dc06f8c547ed2728a659261cf7d78dc4e17a27ca668d9`
- `artifacts/full-refit-plan.yaml`: `b40c091ee2998f90c1916eeb8498b210176f01040a15f80b578df01c53f65770`

두 파일은 판정 전후에 바이트 단위로 같고 이번 이슈에서 수정하지 않았다.

## 근거 파일

- 입력 묶음: `input-bundle.json` (`942821c80551bcad0ddd0be23e63e56bfab6e4de4f2e5b222ca3422a6085de09`)
- 도달 가능성 기록: `preflight.json` (`e8fd6188212b858c9c522c8da014c4330c933b9b86d83fb7835515deff016e10`)
- 최종 판정: `judgment.json` (`b2390ea543aee999fad9198a85f54739413d7f0416ecf7d2007fa4d7deb292ae`)
- 파일 목록: `manifest.sha256`
