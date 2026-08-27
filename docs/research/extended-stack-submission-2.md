# 넓힌 확장 스택 두 번째 최종 제출물 조립·업로드 (이슈 #456, #457)

[넓힌 확장 스택의 교체 여부와 조립 구성을 정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/456)의 결정과 [교체 판을 조립·업로드하고 최종 두 장을 수동 고정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/457)의 실행 기록이다.
판정은 [#455](https://github.com/tmheo/predicting-smartphone-addiction/issues/455)(`docs/research/extended-stack-ladder-2.md`)에서 끝났고, 조립 규칙은 [#444](https://github.com/tmheo/predicting-smartphone-addiction/issues/444)(`docs/research/extended-stack-submission.md`)에서 정한 것을 이어 썼다.
조립 프로그램은 `scripts/assemble_extended_stack.py`, 기계가 읽는 기록은 `docs/research/extended-stack-submission-2-manifest.json`이다.
제출 파일 `artifacts/submissions/issue457-extended-stack-2.csv`는 커밋 제외 경로라 SHA-256으로만 남긴다.

## 결정 (#456)

2026-08-28 사용자 결정: 사다리 규칙대로 문턱을 넘은 구성 가운데 nested가 가장 높은 것을 그대로 조립해 두 번째 장을 교체한다.

| 질문 | 결정 | 근거 |
| --- | --- | --- |
| 교체 여부 | 교체한다 | `ablate_new_nhtquyn`(313구성원)이 현재 판 대비 `+0.0000633`, 분할 5/5로 유일하게 교체 문턱을 넘었다 |
| TE 누출 2개(`pub_rmlp`, `pub_tabm`) | 제외 유지 | #444에서 코드 수준 누출로 확인했고, 사다리의 모든 구성이 이미 뺀 상태다 |
| `fold_evidence_none` 3개, `license_unknown` 64개 | 포함 | 절제에서 `license_unknown`을 빼면 현재 판보다 낮아지고(`-0.0000515`), `fold_evidence_none`은 잡음 안(`+6e-06`)이다. 사용 한정 규칙(지도 #451, 2026-08-27)대로 결합 입력으로만 쓴다 |
| 자체 35의 시험 예측 | #444대로 5:1 혼합판 | 안전판 `e88f706e`와 같은 검증된 산출물이고, 순위 공간 결합이라 외부와 눈금이 달라도 문제없다 |
| 최종 두 장 수동 고정 | 사용자가 직접 한다 | Kaggle 자동 선택이 Public 상위 2개로 바뀌면 안전판이 빠진다 |

## 구성

- OOF 행렬: 자체 35 + 외부 278 = 313구성원.
  #455 사다리 구성 `ablate_new_nhtquyn`과 열 순서까지 같아 nested `0.9703509`(가중 `0.9712170`, 현재 판 `0.9702876` 대비 `+0.0000633`, 분할 5/5 양수)가 이 구성의 판정값이다.
  외부 278 = 판본 1의 207(TE 누출 2개 제외) + 신규 71(공개 노트북 출력물 45, hboyang 150-fusion 11, paiky1995 6, najiama 재게시 5, beicicc other 3, masayakawamata 1).
  신규 전체 433 대비 nhtquyn 고전 확률 모델 120개를 뺐다(절제 `-5.7e-05`).
- 시험 행렬: 자체 35열은 5:1 혼합판(`artifacts/full-refit/member_test_cv_full.parquet`, 재학습 계획·풀 해시는 manifest), 외부 278열은 장부 `test_path`(CV 분할 평균).
  행 수 296,302, `test.csv` id 순서, 유한값을 확인했다.
- 결합 전략 `shrunk_rank_logit_logistic`, 격자 {0.25, 0.5, 0.75, 1.0}에서 λ = 1.0.
  #337·#386·#444의 관찰대로 λ = 1이라 시험 예측 순위는 `rank_logit_logistic`과 같다.
  전체 OOF 적합 590초(로컬 M-시리즈 14코어).

## 제출 파일 확인

| 확인 | 값 |
| --- | --- |
| 행 수 · id 순서 | 296,302 · `test.csv`와 동일 |
| 유한값 · 범위 | 전부 유한 · [3.4e-06, 1.0] |
| 서로 다른 값 수 | 296,302(동률 없음) |
| 현재 두 번째 장 `4f2466f8`(242구성원)와 스피어만 | 0.999451 |
| 안전판 `e88f706e`(후보 1, 5:1 혼합판)와 스피어만 | 0.996790 |
| CV 전용판 `b24e5ba7`(후보 2)와 스피어만 | 0.996687 |
| in-sample OOF AUC(참고치, 판정에 쓰지 않음) | 0.9705308 |
| 제출 CSV SHA-256 | `a4d9c5dbcc90f4f63a972ddd885f64f10fcab23a99106c6118d4b1f6665456df` |

재현: 작업 폴더 변경이 있던 첫 조립과 커밋 `246645a`의 깨끗한 트리에서 돌린 두 번째 조립의 CSV SHA-256이 바이트 단위로 같았다.
manifest는 두 번째 조립의 것이다(`git.dirty = false`).
현재 두 번째 장과 스피어만 0.99945로 순위가 거의 같고 71개를 더한 만큼만 달라진 장이다.

## 출처와 라이선스

외부 278구성원의 출처와 라이선스는 manifest의 `members`(구성원별)와 `external_summary`(집계)에 있고, 원본 근거는 판본 2 장부 `docs/research/external-member-ledger.json`(sha256 `e34d01f3…`)이다.
집계는 CC0 1.0 203, unknown 61, CC BY 4.0 6, Apache 2.0 5, other 3이다.
판본 1의 207구성원 표는 `docs/research/extended-stack-submission.md`에 있고, 신규 71개의 출처는 다음과 같다.

| 소유자 | 데이터셋 또는 노트북 | 라이선스 | 구성원 |
| --- | --- | --- | ---: |
| hboyang | `hboyang/s6e8-150-fusion-local-members` | unknown | 11 |
| paiky1995 | `paiky1995/s6e8-oof-library-11-members` | CC0-1.0 | 6 |
| szymonkapiski | `szymonkapiski/s6e8-oof-library-47-models`(najiama 재게시 naji01~05) | unknown | 5 |
| beicicc | `beicicc/s6e8-fixed900-structural-lgbm-artifacts`, `s6e8-fixed4000-catboost-screen-relation-artifacts` | other | 3 |
| masayakawamata | `masayakawamata/s6e8-catstr-aug16` | CC0-1.0 | 1 |
| 공개 노트북 30개의 출력물 | omidbaghchehsaraei 6, rv1922 4, zhukovoleksiy·yaminh·sidhaarthshree·danushkumarv·lopure 각 3, beicicc 외 나머지 각 1~2 | unknown(출력물에 라이선스 표시 없음) | 45 |

`license_unknown` 부류 64개(노트북 출력물 45, hboyang150 11, najiama 5, beicicc other 3)는 사용 한정 구성원이다.
예측 배열은 결합 입력으로만 쓰고 재배포, 저장소 커밋, 자체 산출물 첨부를 하지 않는다.
CC BY 4.0 6개(beicicc)와 Apache 2.0 5개(raykkretzschmar)의 저작자 표시는 manifest가 유지한다.

## 업로드와 기록 (#457)

업로드 직전에 제출 파일을 다시 확인했다: 296,302행, `test.csv` id 순서 동일, 열 `id`·목표, 전부 유한, SHA-256 `a4d9c5db…`(manifest와 동일).
당일(UTC 2026-08-27) 제출 횟수는 1회였다.

2026-08-27T17:14:34Z(KST 08-28 02:14)에 올렸다(Kaggle ref 55823369, 파일 `issue457-extended-stack-2.csv`).
**Public 0.97135**로 계정 최고이며 현재 두 번째 장 `4f2466f8`(0.97134) 대비 `+0.00001`, 안전판 `e88f706e`(0.97099) 대비 `+0.00036`이다.
이 값은 사후 확인값이고 판정에는 쓰지 않는다(ADR 0002).
nested `0.9703509`와의 오프셋은 `+0.00100`으로 현재 두 번째 장의 `+0.00105`와 같은 크기다.
nested 증분 `+0.0000633`이 Public에서 `+0.00001`로 나타난 것은 Public 눈금(소수 5자리, 시험의 일부)의 해상도 안이다.

MLflow 기록은 `pipeline.submit --record-existing 55823369`로 남겼다.
파생 실행 `443b3a71a2b045ba9052fbb3d821255d`(실행 이름 `ensemble_shrunk_rank_logit_logistic_issue457_extended_stack_2_own35_ext278`, `source.kind=derived_submission`, `source.run_id=b24e5ba7`, `git_commit=246645a`, `git_dirty=False`)에 제출 CSV, 조립 manifest, 판본 2 장부, #455 사다리 근거를 첨부했다.
param에 구성원 수(313 = 자체 35 + 외부 278), 판정 구성 `ablate_new_nhtquyn`, λ = 1.0, 5:1 혼합 계획, 장부 판본·해시를, tag에 라이선스 집계(CC0 203, unknown 61, CC BY 4.0 6, Apache 2.0 5, other 3), 주의 사항 부류 집계, 제외한 TE 누출 2개와 절제로 뺀 nhtquyn 120개를 적었다.
metric `public_auc = 0.97135`는 기록 도구가 Kaggle에서 회수했고, 확장 스택 자체의 판정값은 manifest에서 따로 올렸다: `auc_oof = 0.9703509`(nested), `weighted_oof_auc = 0.9712170`, `auc_fold_0..4`, `delta_vs_pool35_source = +0.000540`, `delta_vs_current_plate = +0.0000633`, `auc_oof_insample = 0.9705308`(참고치).

## 최종 두 장

| 장 | 실행 | Kaggle ref | Public | nested |
| --- | --- | --- | --- | --- |
| 1 (안전판) | `e88f706e` | 55795055 | 0.97099 | 0.9698106 |
| 2 (넓힌 확장 스택) | `443b3a71` | 55823369 | 0.97135 | 0.9703509 |

새 제출이 계정 Public 최고가 되어 Kaggle의 자동 선택은 "Public 상위 2개"인 `443b3a71`(0.97135)과 `4f2466f8`(0.97134)이 되고 안전판 `e88f706e`가 빠진다.
그래서 www.kaggle.com에서 최종 두 장을 안전판 `e88f706e`(55795055) + 넓힌 확장 스택 `443b3a71`(55823369)로 수동 고정해야 하며, 이 단계는 사용자가 직접 확인한다.
이전 두 번째 장 `4f2466f8`(55810100)은 최종 선택에서 빠진다.
