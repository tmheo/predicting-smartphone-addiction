# 동결 후보 풀의 결합 전략 최종 재확정 결과

이 문서는 GitHub 이슈 [#337](https://github.com/tmheo/predicting-smartphone-addiction/issues/337)의 동결 입력, 등록 전략 22개 비교, 두 눈금 판정, 값 좌표 관점 제거 대조와 재학습 계획 일치 확인을 기록한다.
판정 기록은 `artifacts/judgments/issue337-final-combiner.yaml`이고, 비교 원본은 MLflow 파생 앙상블 실행 `b24e5ba7b7eb4e3a9e10788005896328`의 `ensemble_evaluation.json`과 `strategy_oof.parquet`이다.

## 실행 시점

티켓의 달력 시작 조건은 2026-08-28T00:00:00Z였으나, 사용자 지시로 2026-08-26 08:33 UTC에 실행했다.
실행 시점에 후보 풀을 바꿀 수 있는 열린 실험 티켓이 없어 후보 풀 35개를 동결 상태로 보았다.
2026-08-28 전에 후보 풀이나 등록 전략이 바뀌면 이 판정은 오래된 판정이 되고 같은 계약으로 다시 실행해야 한다.

## 동결 입력

| 입력 | SHA-256 |
| --- | --- |
| 코드 커밋 | `42be08012b9df09104404b45db4fa84f3e1a0315` (깨끗한 작업 폴더) |
| `artifacts/pool.yaml` (35개) | `caa1b90769720a4accbe07074dbc7efe0335ab6657fea80c6839b60121dc39d3` |
| `artifacts/champion.yaml` | `aa012114107c06532cf51c0fa9c741f5949146428cf266cf4bedded783d20e09` |
| `artifacts/folds.parquet` | `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4` |
| `artifacts/full-refit-plan.yaml` | `2c56c63f7c09f1a2c887a7c2b958090fd329db80916677f800fa8ee9bb996a36` |
| `data/train.csv` | `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c` |
| `data/test.csv` | `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e` |

`pipeline.pool_audit`로 35개 구성원 OOF를 실행 ID와 행 ID로 다시 조회해 계보, 정렬, fold, `float64`, 유한성 검사를 전부 통과했고 결측 행과 중복 행은 0이다.
감사 도구는 exp157과 exp131의 순위 상관 `0.99814`를 참고 중복으로 표시했지만, 이 티켓은 구성원을 추가하거나 제거하지 않으므로 풀은 그대로 두었다.

## 등록 전략 22개 비교

`pipeline.ensemble`에 등록 전략 22개 전부를 `--only`로 명시해 같은 5개 바깥쪽 검증 분할에서 한 번에 비교했다.
정밀 결합 전략 3개(`bagged_greedy_rank_mean`, `optuna_subset_rank_mean`, `optuna_subset_ridge_logit`)도 같은 실행에 포함했다.
결과 확인 뒤 전략을 추가하거나 설정을 바꾸지 않았고, 실패하거나 제외한 전략은 없다.
총 경과 시간은 1,745초다.

| 순위(가중) | 전략 | 가중 OOF | nested OOF | nested 순위 | 초 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `shrunk_rank_logit_logistic` | 0.9706942946 | 0.9698105828 | 1 | 225 |
| 2 | `missing_segmented_rank_logit` | 0.9706847707 | 0.9698052341 | 2 | 29 |
| 3 | `missing_interaction_rank_logit` | 0.9706819592 | 0.9698029524 | 3 | 41 |
| 4 | `missing_4plus_rank_logit` | 0.9706819393 | 0.9698002428 | 4 | 33 |
| 5 | `rank_logistic` | 0.9706789813 | 0.9697938043 | 6 | 28 |
| 6 | `rank_logit_logistic` | 0.9706788148 | 0.9697948474 | 5 | 33 |
| 7 | `logit_logistic` | 0.9706724553 | 0.9697891828 | 7 | 3 |
| 8 | `rank_gauss_logistic` | 0.9706705200 | 0.9697845846 | 8 | 30 |
| 9 | `bagged_greedy_rank_mean` | 0.9705976491 | 0.9697111350 | 9 | 757 |
| 10 | `greedy_rank_mean` | 0.9705831551 | 0.9696981232 | 10 | 117 |
| 11 | `optuna_subset_rank_mean` | 0.9705831551 | 0.9696981232 | 11 | 181 |
| 12 | `xgb_rank_logit` | 0.9705200697 | 0.9696332941 | 12 | 38 |
| 13 | `nnls_rank` | 0.9701376977 | 0.9692449346 | 13 | 27 |
| 14 | `performance_weighted_rank_mean` | 0.9700191462 | 0.9691324618 | 14 | 17 |
| 15 | `rank_mean` | 0.9700173346 | 0.9691305960 | 15 | 3 |
| 16 | `ridge_logit_alpha_100` | 0.9680041733 | 0.9670809035 | 16 | 1 |
| 17 | `ridge_logit_alpha_10` | 0.9679979113 | 0.9670742954 | 17 | 1 |
| 18 | `ridge_logit` | 0.9679972450 | 0.9670736015 | 18 | 1 |
| 19 | `ridge_logit_alpha_0p1` | 0.9679971840 | 0.9670735379 | 19 | 1 |
| 20 | `ridge_logit_alpha_0p01` | 0.9679971773 | 0.9670735310 | 20 | 1 |
| 21 | `optuna_subset_ridge_logit` | 0.9674395246 | 0.9665183021 | 21 | 174 |
| 22 | `nnls_logit` | 0.9673001809 | 0.9663651079 | 22 | 2 |

가중 OOF는 유효 표본 636,745행(92.10%), 0 가중 행 1,311, test 전용 패턴 276으로 쟀다.

## 판정

최종 결합 전략은 가중 OOF 기준 최고인 `shrunk_rank_logit_logistic`이다.
nested OOF 기준 최고도 같은 전략이라 두 눈금의 1위는 갈리지 않았고, 상위 22개 순위에서 갈린 자리는 5위와 6위(`rank_logistic`과 `rank_logit_logistic`)뿐이다.

2위 `missing_segmented_rank_logit`과의 차이는 가중 `+0.0000095`, nested `+0.0000053`이고 상위 5개의 폭은 가중 `0.0000153`, nested `0.0000157`이다.
모두 성능 동등 대역 `±0.0000277`([pool-reduction-judgment-rule.md](pool-reduction-judgment-rule.md)) 안이므로, 계약대로 점추정 최고를 고르되 이 선택이 잡음 폭 안의 선택임을 명시한다.

바깥쪽 검증 분할별 1위는 분할 0·3이 `missing_segmented_rank_logit`, 분할 1·4가 `missing_interaction_rank_logit`, 분할 2가 `rank_logistic`이고, 선택 전략은 어느 분할에서도 1위가 아니다.
선택 전략은 `missing_segmented_rank_logit`, `missing_interaction_rank_logit`, `missing_4plus_rank_logit`에 각각 1/5 분할에서만 앞선다.

이전 파생 기록(33구성원, `575940e9`) 대비 같은 전략 증분은 `+0.0000276`이며 새 구성원은 exp197과 exp183이다.

## 선택 전략의 수축 계수와 눈금 관찰

선택 전략의 수축 계수 λ는 5개 바깥쪽 분할 모두 1.0이었다.
분할 안에서 `rank_logit_logistic`과 스피어만 상관이 1.0이라, 선택 전략의 예측은 `rank_logit_logistic` 예측의 분할 안 백분위 순위와 같다.
따라서 두 전략의 nested 차이(`+0.0000157`)는 결합 방식의 차이가 아니라 바깥쪽 분할 블록마다 백분위 순위를 내면서 분할 간 눈금 차이가 사라진 결과이고, 단일 전체 적합으로 만드는 시험 예측에서는 두 전략의 순위가 같다.

이 관찰을 확인하려고 22개 전략의 OOF 예측을 모두 같은 바깥쪽 분할 백분위 순위로 바꿔 다시 채점했다.
결과 확인 뒤 만든 눈금이라 선택 기준으로 쓰지 않으며 판정 기록의 `fold_rank_normalized_diagnostic`에 보조 진단으로만 남긴다.

| 전략 | 가중 OOF(분할 순위) | nested OOF(분할 순위) |
| --- | ---: | ---: |
| `missing_segmented_rank_logit` | 0.9706993213 | 0.9698196825 |
| `missing_interaction_rank_logit` | 0.9706982598 | 0.9698192377 |
| `missing_4plus_rank_logit` | 0.9706964041 | 0.9698147148 |
| `shrunk_rank_logit_logistic` = `rank_logit_logistic` | 0.9706942946 | 0.9698105828 |
| `rank_logistic` | 0.9706924550 | 0.9698076404 |

같은 발판에서는 `missing_segmented_rank_logit`이 가중 `+0.0000050`, nested `+0.0000091` 앞서지만 역시 동등 대역 안이다.
읽기는 두 가지다.
첫째, 결합 전략 상위 5개는 어느 눈금으로 재도 서로 구분되지 않으며 최종 점수는 전략 선택이 아니라 구성원 풀이 결정한다.
둘째, 제출 2장의 두 번째 장을 고르는 [P5: 최종 제출 후보 두 개 확정](https://github.com/tmheo/predicting-smartphone-addiction/issues/69)은 `missing_segmented_rank_logit`을 같은 발판의 1위이자 분할 2/5 승자로 참고할 수 있다.

## 값 좌표 관점 조건부 진단

선택 전략을 고정하고 네 풀을 같은 nested 절차로 짝지어 평가했다(`scripts/diagnose_value_coordinate_ablation.py`).
전체 풀 재평가는 비교 실행의 값을 열 자리까지 재현했다.

| 풀 | 구성원 | nested OOF | 전체 대비 | 가중 OOF | 전체 대비 | 전체 풀 분할 승 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 동결 전체 | 35 | 0.9698105828 | - | 0.9706942946 | - | - |
| exp106 제외 | 34 | 0.9697899527 | -0.0000206 | 0.9706751993 | -0.0000191 | 5/5 |
| exp139 제외 | 34 | 0.9698105079 | -0.0000001 | 0.9706937175 | -0.0000006 | 3/5 |
| 둘 다 제외 | 33 | 0.9697815319 | -0.0000291 | 0.9706658001 | -0.0000285 | 5/5 |

- Lookup-Transformer 구현(exp106)은 RealMLP 구현이 함께 있어도 단독 제거 손실이 `-0.0000206`, 5/5 분할로 양의 기여를 유지한다.
  제거하면 계수가 같은 계보의 exp059(`-0.161` → `+0.164`), exp131(`+0.877` → `+1.008`), exp157(`+0.587` → `+0.699`)로 옮겨간다.
- RealMLP 구현(exp139)은 exp106이 있는 조건에서 단독 제거 손실이 `-0.0000001`, 3/5 분할로 조건부 기여가 0에 가깝다.
  계수 `+0.393`은 같은 계보의 exp136(`+0.265` → `+0.470`)과 exp134(`+0.135` → `+0.288`)로 거의 그대로 옮겨간다.
- 묶음 제거 손실 `-0.0000291`은 단독 손실의 합 `-0.0000207`보다 `0.0000084` 크다.
  즉 exp139의 값어치는 exp106이 없을 때 그 관점을 대신 담당하는 데 있고, 두 구현이 함께 있으면 관점 하나가 여유분이 된다.
- 이 진단은 소급 제거 관문이 아니라 조건부 기여의 설명이며, 세 번째 학습기 계열로의 이식은 이 결과만으로 열지 않는다.

## 재학습 계획 일치

선택 전략이 `artifacts/full-refit-plan.yaml`의 `protocol.combiner`(`shrunk_rank_logit_logistic`)와 같아 계획을 바꾸지 않았다.
`pipeline.refit_plan artifacts/full-refit-plan.yaml --validate-only`가 구성원 35개, 재학습 103회, 후보 풀 `caa1b907…`으로 통과했다.
후보 풀과 champion 장부도 바꾸지 않았다.

## 검증

- MLflow 파생 앙상블 실행 `b24e5ba7`을 다시 조회해 `auc_oof 0.9698105828`, `auc_oof_weighted 0.9706942946`, 분할별 AUC 5개, 기준 실행 `575940e9` 대비 증분, 입력 SHA-256 태그 3개, 산출물 5개(`ensemble_evaluation.json`, `strategy_oof.parquet`, `member_weights.csv`, `missingness_weights.csv`, `oof.parquet`)를 확인했다.
- `strategy_oof.parquet`에서 22개 전략의 전체·가중 OOF를 다시 채점해 비교 산출물과 `1e-12` 안에서 일치함을 판정 기록 생성 때 확인했다.
- `git diff --check`와 `scripts/verify_environment_gates.sh`를 통과했다.

## 인계

- [P5: 최종 제출 후보 두 개 확정](https://github.com/tmheo/predicting-smartphone-addiction/issues/69)은 선택 전략 `shrunk_rank_logit_logistic`으로 전체 자료 재학습과 제출 조립을 진행한다.
- 두 번째 장 후보로 `missing_segmented_rank_logit`을 검토할 때는 위의 같은 발판 진단과 분할 승수를 참고한다.
- 실행 원자료는 `run-logs/issue337/`에 있다.
