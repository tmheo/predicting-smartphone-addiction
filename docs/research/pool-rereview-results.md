# 35개 후보 풀 사전 고정 재심사 결과

이 문서는 GitHub 이슈 [#339](https://github.com/tmheo/predicting-smartphone-addiction/issues/339)의 실제 규모 실행 결과와 재현 근거를 기록한다.
후보, 순서, 문턱, 결합 전략과 난수는 결과를 보기 전에 동결했으며 실행 중 변경하지 않았다.

## 동결 입력

| 항목 | 값 |
| --- | --- |
| 사전 고정 장부 | `artifacts/pool-rereview-precommit-2026-08-22.yaml` |
| 사전 고정 장부 SHA-256 | `978b77916e4e7bc00edd1c4f3e72186726ed8ba5ee102f0e550446c9dd89a8bf` |
| 기준 후보 풀 장부 | `artifacts/pool-baseline-2026-08-21.yaml` |
| 기준 후보 풀 장부 SHA-256 | `cef5c08efad104580dc9fab7a3c7605d1e5f95ce5f9b825caa66206dc50ff96f` |
| 단계화 예측 SHA-256 | `80dba40bbdfad07c08607864b0430787533dd710b850f19a9e72a51208c4a60c` |
| 실행 코드 커밋 | `97550d0c5844ff5a14ae02e3e291ae553c103aba` |
| 실행 코드 SHA-256 | `8657cca25a3d9b9dc2ff8e634f20c96359fc87e197e01b7d9fc596430e0de196` |
| `pyproject.toml` SHA-256 | `d9c7186959cdfc43a5383d9909755ae52c6b3f087ad275e6c39a25cf779236a9` |
| `uv.lock` SHA-256 | `27527bf7a3094af0c9fa85613216f097a9d18b4cf75ab4c5b4c73b2ad3af25b4` |

## 판정

전체 OOF 궤적의 점추정 규칙은 35개 후보 가운데 12개를 가장 작은 성능 동등 후보로 선택했다.
선택안의 AUC는 `0.9697082595858664`이며 35개 앵커의 `0.9697358490699927`보다 `0.000027589484126289143` 낮다.
이 차이는 채택한 성능 동등 하한 `-0.000027669802`보다 `0.00000008031787371085728`만큼 높다.
다만 쌍체 부트스트랩 2.5 백분위수가 하한보다 낮아 경계 표시는 참이다.

사전 고정 절차의 바깥쪽 검증에서는 선택안이 앵커를 이긴 분할이 없었고 앵커가 5개 분할 모두 이겼다.
분할별 원시 예측을 이어 붙인 절차 nested OOF AUC는 선택안 `0.9526077862185631`, 앵커 `0.9697358490699927`이며 차이는 `-0.017128062851429537`이다.
따라서 이 이슈에서는 후보 풀 장부를 바꾸지 않고 12개 선택안과 직전 13개 선택안을 이슈 [#346](https://github.com/tmheo/predicting-smartphone-addiction/issues/346)에 함께 넘긴다.

## 바깥쪽 검증 분할

| 분할 | 안쪽 학습 선택 전략 | 선택 구성원 수 | 보류 선택 AUC | 보류 앵커 AUC | 차이 | 승자 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `missing_segmented_rank_logit` | 4 | `0.969094190647066` | `0.969171529960641` | `-0.0000773393135750089` | 앵커 |
| 1 | `shrunk_rank_logit_logistic` | 4 | `0.9697467334620536` | `0.9698191225795225` | `-0.000072389117468874` | 앵커 |
| 2 | `shrunk_rank_logit_logistic` | 4 | `0.9698122905451999` | `0.9698584859056756` | `-0.00004619536047567596` | 앵커 |
| 3 | `missing_segmented_rank_logit` | 4 | `0.9702603462787069` | `0.9703021927585102` | `-0.00004184647980320921` | 앵커 |
| 4 | `missing_segmented_rank_logit` | 4 | `0.9694012117963073` | `0.9695279120258615` | `-0.00012670022955418858` | 앵커 |

분할 0, 3, 4에서는 `exp067_tabpfn3`, `exp106_lookup_fixed24_train_test_preprocessing`, `exp131_lookup_bivariate_plr5`, `exp139_realmlp_reference_qnormal_train_test`가 선택됐다.
분할 1, 2에서는 `exp106_lookup_fixed24_train_test_preprocessing`, `exp131_lookup_bivariate_plr5`, `exp133_scalar_token_transformer_oof_te`, `exp139_realmlp_reference_qnormal_train_test`가 선택됐다.

## 전체 OOF 최종 선택

최종 선택은 14단계의 `missing_interaction_rank_logit` 전략이며 전체 재학습 수를 기준 99회에서 36회로 63회 줄인다.

1. `exp022_orig_knn`
2. `exp023_orig_proxy_residual`
3. `exp035_lattice_te`
4. `exp067_tabpfn3`
5. `exp106_lookup_fixed24_train_test_preprocessing`
6. `exp085_contextual_spline_m0`
7. `exp134_realmlp_muon`
8. `exp131_lookup_bivariate_plr5`
9. `exp136_realmlp_muon_recon_widths`
10. `exp137_tabm_recon_widths`
11. `exp133_scalar_token_transformer_oof_te`
12. `exp139_realmlp_reference_qnormal_train_test`

직전 12단계 후보는 위 12개에 `exp124_realmlp_dtype_fix`를 더한 13개 풀이며 전략은 `missing_segmented_rank_logit`이다.
직전 후보의 AUC는 `0.9697115508329799`, 앵커 대비 차이는 `-0.000024298237012776447`이다.

제거 후보 23개는 다음과 같다.

- `exp006_te_drop_gaming`
- `exp011_resid_pair`
- `exp025_constrained_impute`
- `exp027_recon_ce`
- `exp032_recon_orig_mean_top3`
- `exp033_recon_orig_mean_top3_raw`
- `exp048_lgb_orig_cdf_diff`
- `exp058_logreg_onehot`
- `exp059_lookup_transformer`
- `exp070_cat_exact_cats`
- `exp071_cat_exact_no_te`
- `exp081_lookup_fold_initialization_avg3`
- `exp107_logreg_onehot_nn10`
- `exp108_logreg_onehot_nn10_l1`
- `exp110_lgb_kitopl_no_te`
- `exp111_xgb_depth8_no_te`
- `exp113_tab_cnn_m0`
- `exp117_ag25_gbm_r21`
- `exp124_realmlp_dtype_fix`
- `exp127_lookup_muon`
- `exp131_tab_cnn_oof_target_mean`
- `exp132_tab_cnn_epochs100`
- `exp135_xgb_hpo_trial30`

## 선택 안정성

| 구성원 | 바깥쪽 5개 분할 선택 횟수 |
| --- | ---: |
| `exp106_lookup_fixed24_train_test_preprocessing` | 5 |
| `exp131_lookup_bivariate_plr5` | 5 |
| `exp139_realmlp_reference_qnormal_train_test` | 5 |
| `exp067_tabpfn3` | 3 |
| `exp133_scalar_token_transformer_oof_te` | 2 |

나머지 30개 후보는 바깥쪽 검증에서 한 번도 선택되지 않았다.

## 영점 대조와 성능 동등 하한

| 출처 | 하한 |
| --- | ---: |
| 기존 기준 | `-0.000027669802` |
| 전체 OOF 영점 대조 | `-0.000019662813742216123` |
| 안쪽 학습 영점 대조 | `-0.000027666345507215695` |
| 채택값 | `-0.000027669802` |

영점 대조 산출물 SHA-256은 `ecc159377756ed74a8709322ecc73ca682eff097dc30ba49c35571adc0063115`이다.
최종 선택안의 쌍체 부트스트랩 차이는 2,000회에서 최솟값 `-0.00006398009260544235`, 2.5 백분위수 `-0.000046347203389796035`, 중앙값 `-0.000027745514381893877`, 97.5 백분위수 `-0.000008023316433078414`, 최댓값 `0.0000045096618237483455`였다.
2.5 백분위수가 채택 하한보다 낮으므로 사전 고정한 경계 조건을 만족한다.

## 실행, 중간 저장과 재개

Vast.ai 인스턴스 `48364969`에서 작업자 88개로 실행했으며 최종 실행은 2026-08-22 14:15:30 KST에 시작해 22:01:45 KST에 끝났다.
측정된 실행 시간은 `27931.26161726797`초이며 최상위 전략 적합 32,370회와 후보 풀 평가 390회를 수행했다.
이 실행은 앞선 장비에서 완료한 영점 대조 블록 1개를 중간 저장 파일로 이어받았다.
마지막 장비를 시작할 때 완료된 결과 분할은 없었으므로 바깥쪽 5개 분할과 전체 OOF 분할은 이 장비에서 계산했다.
완료 후 재개 검증에서는 결과 분할 6개를 중간 저장 파일에서 읽고 바깥쪽 적합을 0회 재계산했으며 판정 파일이 바이트 단위로 같음을 확인했다.
실패와 재시도 목록은 비어 있다.

마지막 장비의 추정 비용은 5.55달러이고 전체 시도의 추정 합계는 6.74달러다.
결과 회수 후 인스턴스를 삭제했고 프로젝트 볼륨이 0개이며 종료 안전 예약도 제거됐음을 확인했다.

## 무결성과 재현 확인

| 산출물 | SHA-256 |
| --- | --- |
| 판정 파일 | `ac98d1a8906a7cfe7b29693f3c1414daa0da41b768aa588783cd78ac25320a91` |
| 실행 계측 | `9194b810d9815e8aef6222162068429b74ba5efa7cd454956ba7dd4a181f46cc` |
| 내부 명세 | `29d7baba4824a708538f89741d1348cb192a85a36dcb37a52e63c5faa528bea0` |
| 결과 명세 | `1bc2318fcf6b3f90fbbebb5b8e5930fb31d33fe70eb3972b8c911ee1293dd875` |
| 회수 압축 파일 | `44b851e729db560b875c0bddb430e6345b90e76620ba976d4600d5f26387d3d5` |
| 최종 중간 저장 파일 | `4ab62c2e21f9d164e9cc661160459fe31b6cc542dbe5816eae47700c90377efe` |

결과 명세와 내부 명세의 모든 파일 해시를 통과했다.
691,369개 행의 식별자가 일대일로 맞고 중복이 없음을 확인했다.
독립 계산한 전체 및 분할별 AUC와 선택 구성원이 판정 파일과 정확히 일치했다.
같은 입력으로 판정 절차를 세 번 다시 실행했으며 세 결과가 같은 SHA-256 `47717b4e0a9cbfb118cc65fcbb5a95db6aa57484ce09753fe699df197e3e2015`를 냈다.

## 해석과 인계

분할별 선택 AUC가 모두 약 0.969인데 절차 nested OOF AUC가 0.9526인 원인을 별도로 재현하고 확인했다.
분할 0, 3, 4가 고른 `missing_segmented_rank_logit`의 예측 평균은 약 0.709이고 분할 1, 2가 고른 `shrunk_rank_logit_logistic`의 예측 평균은 약 0.500이다.
사전 고정 계약은 각 분할의 원시 보류 예측을 이어 붙여 하나의 AUC를 계산하므로 분할 사이 출력 척도 차이가 전체 순위를 흐트러뜨렸다.
식별자 정렬, 예측 파일 무결성, AUC 재계산과 반복 실행이 모두 일치하므로 손상된 산출물이나 우발적 구현 차이로 보지 않는다.

원인 확인용으로 각 분할 안에서 순위 보정한 뒤 이어 붙인 AUC는 `0.9696629548156505`였다.
그러나 결과를 본 뒤 이 값을 판정량으로 쓰면 사전 고정 절차를 바꾸게 되므로 판정에는 사용하지 않았다.

전체 OOF 점추정만 보면 12개 풀이 가장 작은 동등 후보지만 여유가 `0.00000008031787371085728`에 불과하고 부트스트랩 경계가 참이다.
바깥쪽 검증에서도 앵커가 5개 분할 모두 이겼으며 절차 출력 척도 문제가 별도로 드러났다.
따라서 12개 풀과 직전 13개 풀을 함께 이슈 [#346](https://github.com/tmheo/predicting-smartphone-addiction/issues/346)에 인계하고, 최종 유지와 제거 및 결합 전략은 그 이슈에서 정한다.
이 결과만으로 후보 풀 장부는 변경하지 않는다.
