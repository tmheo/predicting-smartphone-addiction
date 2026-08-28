# 엄격 외부 후보 사다리 판정 보고 (`ecf-v3-1154a775f8a8`)

## 판정

- 결과: **미달: 현재 두 장(e88f706e + 443b3a71) 유지**
- 문턱: 313 대비 이어붙인 nested AUC 차이 `+2e-05` 이상, 바깥 분할 5/5 엄격 양수. 결과 확인 뒤 바꾸지 않는다.
- 선택 규칙: 통과 구성이 여럿이면 nested 최고. 최고와의 차이가 잡음 바닥 5.7e-06 안이면 그 가운데 구성원 수가 가장 적은 구성, 구성원 수까지 같으면 사다리 순서가 앞선 구성.
- 동결 후보 19개 가운데 정확 중복 11개를 뺀 사다리 후보 8개, 구성 6개.
- 비교 팔 자기 검사: 313 이어붙인 AUC 0.9703509, #455 기준 0.9703509, 차이 `+0.00e+00`, 분할 최대 차이 `0.00e+00`, 잡음 바닥 `5.7e-06` → **통과**

## 사다리

| 순서 | 구성 | 구성원 | nested AUC | 가중 OOF(진단) | 313 대비 | 분할 0 | 분할 1 | 분할 2 | 분할 3 | 분할 4 | 양수 | 통과 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `ext313_strict_all` | 321 | 0.9703572 | 0.9712228 | +0.0000062 | -0.0000028 | -0.0000070 | +0.0000132 | +0.0000254 | +0.0000023 | 3/5 | - |
| 1 | `ablate_source_beicicc` | 319 | 0.9703566 | 0.9712227 | +0.0000057 | +0.0000094 | -0.0000032 | +0.0000026 | +0.0000137 | +0.0000059 | 4/5 | - |
| 2 | `ablate_source_busyaprime` | 318 | 0.9703563 | 0.9712230 | +0.0000054 | -0.0000110 | +0.0000083 | +0.0000208 | +0.0000220 | -0.0000130 | 3/5 | - |
| 3 | `ablate_source_ravi20076` | 318 | 0.9703483 | 0.9712146 | -0.0000027 | -0.0000097 | -0.0000002 | +0.0000027 | -0.0000007 | -0.0000055 | 1/5 | - |
| 4 | `ablate_caveat_float32_storage` | 316 | 0.9703447 | 0.9712111 | -0.0000063 | -0.0000096 | -0.0000115 | -0.0000029 | -0.0000050 | -0.0000024 | 0/5 | - |
| 5 | `ablate_caveat_near_duplicate_cluster` | 317 | 0.9703559 | 0.9712223 | +0.0000049 | -0.0000023 | -0.0000044 | +0.0000150 | +0.0000192 | -0.0000031 | 2/5 | - |

구성별 후보:

- `ext313_strict_all`: 313 + 정확 중복을 뺀 사다리 후보 8개 전부. 후보 8개, λ {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
- `ablate_source_beicicc`: 전체 구성에서 출처 `beicicc`의 후보 제외. 후보 6개, 뺀 후보 `beicicc/s6e8-fold-safe-tabnet:tabnet`, `beicicc/s6e8-fold-safe-realmlp:realmlp`, λ {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
- `ablate_source_busyaprime`: 전체 구성에서 출처 `busyaprime`의 후보 제외. 후보 5개, 뺀 후보 `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb`, `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:hgb`, `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:xgb`, λ {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
- `ablate_source_ravi20076`: 전체 구성에서 출처 `ravi20076`의 후보 제외. 후보 5개, 뺀 후보 `ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`, `ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C`, `ravi20076/playgrounds6e8-public-baseline-v1:CB1C`, λ {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
- `ablate_caveat_float32_storage`: 전체 구성에서 주의 사항 부류 `float32_storage`의 후보 제외. 후보 3개, 뺀 후보 `beicicc/s6e8-fold-safe-tabnet:tabnet`, `beicicc/s6e8-fold-safe-realmlp:realmlp`, `ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`, `ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C`, `ravi20076/playgrounds6e8-public-baseline-v1:CB1C`, λ {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
- `ablate_caveat_near_duplicate_cluster`: 전체 구성에서 주의 사항 부류 `near_duplicate_cluster`의 후보 제외. 후보 4개, 뺀 후보 `beicicc/s6e8-fold-safe-realmlp:realmlp`, `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb`, `ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`, `ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C`, λ {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}

비교 팔 313 분할별 AUC(자기 검사 기준): 분할 0 0.9697543, 분할 1 0.9705149, 분할 2 0.9703991, 분할 3 0.9709869, 분할 4 0.9700996, λ {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}

## 정확 중복(자동 제외)

| 동결 순서 | 후보 | 313 구성원 |
| ---: | --- | --- |
| 1 | `nb_zhukov:cat_base` | `ext_nb_zhukov:cat_base` |
| 2 | `nb_zhukov:lgb02` | `ext_nb_zhukov:lgb02` |
| 3 | `nb_zhukov:xgb_base` | `ext_nb_zhukov:xgb_base` |
| 4 | `nb_reda_lgbm:lgbm` | `ext_nb_reda_lgbm:lgbm` |
| 5 | `nb_reda_hgb:hgb` | `ext_nb_reda_hgb:hgb` |
| 6 | `nb_yekenot:trompt` | `ext_nb_yekenot:trompt` |
| 7 | `nb_mohan_realmlp:realmlp` | `ext_nb_mohan_realmlp:realmlp` |
| 8 | `nb_lopure:linear_svm` | `ext_nb_lopure:linear_svm` |
| 9 | `nb_lopure:poly_svm` | `ext_nb_lopure:poly_svm` |
| 10 | `nb_lopure:rbf_svm` | `ext_nb_lopure:rbf_svm` |
| 11 | `nb_shaman_baseline:lr` | `ext_nb_shaman_baseline:lr` |

## 사다리 후보(동결 순서, 단독 AUC는 진단값)

| 동결 순서 | 후보 | 출처 | 주의 사항 | 전체 OOF 단독 AUC |
| ---: | --- | --- | --- | ---: |
| 12 | `beicicc/s6e8-fold-safe-tabnet:tabnet` | beicicc | license_unknown_use_limited, float32_storage | 0.965657 |
| 13 | `beicicc/s6e8-fold-safe-realmlp:realmlp` | beicicc | license_unknown_use_limited, float32_storage, near_duplicate_cluster | 0.968156 |
| 14 | `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` | busyaprime | license_unknown_use_limited, near_duplicate_cluster | 0.962558 |
| 15 | `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:hgb` | busyaprime | license_unknown_use_limited | 0.962048 |
| 16 | `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:xgb` | busyaprime | license_unknown_use_limited | 0.962314 |
| 17 | `ravi20076/playgrounds6e8-public-baseline-v1:XGB1C` | ravi20076 | license_unknown_use_limited, float32_storage, rehosted_training_data_private_notebook, near_duplicate_cluster | 0.964201 |
| 18 | `ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` | ravi20076 | license_unknown_use_limited, float32_storage, rehosted_training_data_private_notebook, near_duplicate_cluster | 0.964173 |
| 19 | `ravi20076/playgrounds6e8-public-baseline-v1:CB1C` | ravi20076 | license_unknown_use_limited, float32_storage, rehosted_training_data_private_notebook | 0.963944 |

생략한 구성: `ablate_caveat_license_unknown_use_limited` (사다리 후보 전원이 공통으로 가진 부류라 절제하면 313과 같다); `ablate_caveat_rehosted_training_data_private_notebook` (구성원 집합이 `ablate_source_ravi20076`와 같다)

## 근접 중복 진단(스피어만 0.998 이상, 열린 4분할, 제외 없음)

| 봉인 분할 | 쌍 수 | 종류별 | 후보가 낀 쌍 |
| ---: | ---: | --- | --- |
| 0 | 47 | {'own-own': 1, 'own-ext313': 1, 'ext313-ext313': 42, 'ext313-candidate': 2, 'candidate-candidate': 1} | `ext_beicicc:s6e8-fixed4-realmlp-two-seed-artifacts:realmlp_seed01_fixed4`·`cand_beicicc/s6e8-fold-safe-realmlp:realmlp` 0.999124, `ext_beicicc:s6e8-fixed900-structural-lgbm-artifacts:raw12`·`cand_busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` 0.998119, `cand_ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`·`cand_ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` 0.998963 |
| 1 | 46 | {'own-own': 2, 'ext313-ext313': 41, 'ext313-candidate': 2, 'candidate-candidate': 1} | `ext_beicicc:s6e8-fixed4-realmlp-two-seed-artifacts:realmlp_seed01_fixed4`·`cand_beicicc/s6e8-fold-safe-realmlp:realmlp` 0.999080, `ext_beicicc:s6e8-fixed900-structural-lgbm-artifacts:raw12`·`cand_busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` 0.998134, `cand_ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`·`cand_ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` 0.998977 |
| 2 | 52 | {'own-own': 1, 'ext313-ext313': 48, 'ext313-candidate': 2, 'candidate-candidate': 1} | `ext_beicicc:s6e8-fixed4-realmlp-two-seed-artifacts:realmlp_seed01_fixed4`·`cand_beicicc/s6e8-fold-safe-realmlp:realmlp` 0.999099, `ext_beicicc:s6e8-fixed900-structural-lgbm-artifacts:raw12`·`cand_busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` 0.998105, `cand_ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`·`cand_ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` 0.998982 |
| 3 | 41 | {'ext313-ext313': 38, 'ext313-candidate': 2, 'candidate-candidate': 1} | `ext_beicicc:s6e8-fixed4-realmlp-two-seed-artifacts:realmlp_seed01_fixed4`·`cand_beicicc/s6e8-fold-safe-realmlp:realmlp` 0.999082, `ext_beicicc:s6e8-fixed900-structural-lgbm-artifacts:raw12`·`cand_busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` 0.998113, `cand_ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`·`cand_ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` 0.998965 |
| 4 | 47 | {'own-own': 1, 'ext313-ext313': 43, 'ext313-candidate': 2, 'candidate-candidate': 1} | `ext_beicicc:s6e8-fixed4-realmlp-two-seed-artifacts:realmlp_seed01_fixed4`·`cand_beicicc/s6e8-fold-safe-realmlp:realmlp` 0.999127, `ext_beicicc:s6e8-fixed900-structural-lgbm-artifacts:raw12`·`cand_busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` 0.998178, `cand_ravi20076/playgrounds6e8-public-baseline-v1:XGB1C`·`cand_ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` 0.998971 |

## 실행 인계 완결 조건(#481, ADR-0006 개정 항목) 대조

- 변경 불가 감사 기록·자격 판정: 동결 명세 `docs/research/external-candidate-freeze/ecf-v3-1154a775f8a8.json` (spec_sha256 `f83a633dab2f2130a28577c3eeca257775774ec8e5bdbdab79631744949256f8`)의 후보 19개, 사용자 제외 0개, 조사 기준 시각 `2026-08-28T00:34:34Z`.
- 313개와 사다리 후보의 입력 명세: 비교 팔 313 구성 해시 `28680c46db7d7c6301c75e81da346f5fbb01ef5ef359989b34e27795bca4c562`, 후보 구성 해시 `b50cf7e615c91dbea408b143d4d4480636948f1db114363d5edef30dbab0ce32`, 자체 35 `ae446f2cc00f391f34a0298403490245336351b31de15d4b7a2fb2ae67e881ce`.
- 사다리 구성 목록·선택 규칙: `precommit.json`의 `ladder`·`selection_rules`.
- 교체 문턱: `precommit.json`의 `gate` (+2e-05, 5/5), 잡음 바닥 5.7e-06.
- 비교 팔 자기 검사·구성별 봉인 예측·사다리 비교: `fold-<k>/baseline-predictions.parquet`·`baseline.json`, `ladder/<구성>/fold-<k>/predictions.parquet`·`nested.json`, `ladder-comparison.json`.
- 실패·재개 규칙: `precommit.json`의 `rules` (모든 하위 명령이 입력 해시와 코드 상태를 다시 확인).
- 실행 경계: 조립·업로드와 최종 두 장 수동 고정은 사용자 승인 뒤에만(#488).

## 코드 상태

- git `7c90de9082193176e1f75e14893b4e740b4e812b` (dirty False), 판정 도구 sha256 `2994d2d1b1dbca7b4609d7c1a9f00ac585a564263263925b1e23015171e7e393`, 동결 생성기 sha256 `d29f02d5ac8b9e00e2ce0452bd3ab90598e265e59f55ba391688b3c3f64287ed`, 결합기 module sha256 `a8ea753e598c8564baa3ec13c5e78958543eb931124304b652a73caf1d35f3e3`, sklearn 1.9.0, numpy 2.5.2.
- precommit_sha256 `d634d06e90a570c326f408a9f2a5e88ccb666a56119b118c0d57682bc977a0ee`, 비교 2026-08-28T08:10:16Z, 보고 작성 2026-08-28T08:10:17Z.
