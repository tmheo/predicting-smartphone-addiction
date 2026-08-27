# 확장 스택 두 번째 최종 제출물 조립 (이슈 #444)

[통과한 확장 스택 구성의 두 번째 최종 제출물 조립 방식을 정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/444)의 결정과 실행 기록이다.
판정은 [#443](https://github.com/tmheo/predicting-smartphone-addiction/issues/443)(`docs/research/extended-stack-ladder.md`)에서 끝났고, 여기서는 사용자와 정한 조립 규칙을 기계 적용했다.
조립 프로그램은 `scripts/assemble_extended_stack.py`, 기계가 읽는 기록은 `docs/research/extended-stack-submission-manifest.json`이다.
제출 파일 `artifacts/submissions/issue444-extended-stack.csv`는 커밋 제외 경로라 SHA-256으로만 남긴다.
업로드와 최종 두 장 선택은 [#445](https://github.com/tmheo/predicting-smartphone-addiction/issues/445)에서 사용자 확인 뒤에 한다.

## 결정

네 가지를 사용자와 함께 정했다.

| 질문 | 결정 | 근거 |
| --- | --- | --- |
| 자체 35의 시험 예측 | 5:1 혼합판(`artifacts/full-refit/member_test_cv_full.parquet`) | 안전판 제출 `e88f706e`와 같은 판이라 검증된 산출물을 재사용하고, 혼합 효과가 #66 +0.00006, #226 +0.00005, #69 +0.00003으로 세 번 다 양수였다. 결합기가 순위 공간에서 재므로 외부 구성원(CV 분할 평균)과 눈금이 달라도 문제없다. |
| 결합기 적합 | 전체 OOF 1회 적합, λ는 5분할 leave-one-fold-out | 기존 두 장(`b24e5ba7`, `e88f706e`)과 같은 계약(`ensemble.full_fit_predictions`)이라 새 코드가 없고, 두 장의 차이가 "구성원 폭" 하나로 좁혀진다. 분할 모델 5개 평균은 코드가 없고 잡음 바닥 수준의 차이를 위해 5배 시간을 쓴다. |
| TE 누출 의심 2개(szymon74 pub_rmlp, pub_tabm) | 뺀다: 242구성원 | 의심이 아니라 코드 수준에서 확인된 누출이다(아래). 제거 비용 -0.000029는 문턱과 0.9702 목표를 모두 유지한다. |
| MLflow·git 태그·출처 기록 | 조립은 CSV + manifest만 만들고, MLflow 기록은 #445가 업로드 뒤 `pipeline.submit --record-existing`으로 남긴다 | 외부 구성원은 MLflow 실행이 없어 `pipeline.ensemble --record-issue`로는 파생 앙상블 실행을 만들 수 없다. `--record-existing`은 #416 계약(`git_commit`·`git_dirty` 태그)을 채우고 `--artifact`·`--param`·`--tag`로 manifest·장부·라이선스 집계를 붙일 수 있으며, `e88f706e`가 정확히 이 경로로 기록됐다. `derived_ensemble` 종류의 실행을 만들지 않으므로 champion 판정 도구가 외부 예측을 볼 일이 없다. |

### TE 누출 2개를 뺀 이유

`docs/research/code-notebook-insights.md`의 22번·24번 리뷰가 omidbaghchehsaraei의 RealMLP·TabM 노트북 코드를 직접 읽고 확인한 사실이다.
두 노트북은 모든 열의 target encoding을 전체 학습 자료에서 한 번 만든 뒤 5분할 학습을 하므로, 검증 분할의 목표값이 학습 행의 특성 통계에 들어가는 분할 간 목표 누출이 있다.
szymonkapiski의 manifest는 두 구성원을 "author's own"(저자 노트북 그대로 재실행)으로 적고 OOF 0.96844·0.96751이 선언값과 일치하므로 누출이 그대로 들어 있다.
불확실한 것은 누출 여부가 아니라 OOF 낙관의 크기뿐이고, 그 낙관은 nested가 잴 수 없는 방향(결합기가 둘을 과대 가중해 시험에서 손해)이라 지도 #441의 "분할 안전성 확인" 자격에 엄밀히는 미달로 보고 뺐다.
#442 장부의 `status`는 바꾸지 않았고 manifest의 `assembled.excluded_members`에 사유를 적었다.

## 구성

- OOF 행렬: 자체 35 + 외부 207 = 242구성원.
  #443 절제 구성 `ablate_te_leak`과 열 순서까지 같아 nested `0.9702876`(35개 풀 대비 +0.000477, 분할 5/5 양수)이 이 구성의 판정값이다.
  판정 구성 own35_ext209(244, nested `0.9703167`)와의 차이 -0.000029는 절제 표의 값 그대로다.
- 시험 행렬: 자체 35열은 5:1 혼합판(재학습 계획 `2c56c63f...`, 풀 `caa1b907...`), 외부 207열은 장부 `test_path`(CV 분할 평균).
  행 수 296,302, `test.csv` id 순서, 유한값을 확인했다.
- 결합 전략 `shrunk_rank_logit_logistic`, 격자 {0.25, 0.5, 0.75, 1.0}에서 λ = 1.0.
  #337·#386의 관찰대로 λ = 1이라 시험 예측 순위는 `rank_logit_logistic`과 같다.
  전체 OOF 적합 444초(로컬 M-시리즈 14코어).

## 제출 파일 눈금 확인

`shrunk_rank_logit_logistic`의 출력은 `λ·백분위순위(메타) + (1-λ)·순위평균`이라 이미 순위 공간이고, 시험 블록 안에서 순위를 매기므로 값 자체는 (0, 1] 안의 백분위다.
AUC는 순위만 보므로 눈금은 점수에 영향이 없고, 확인할 것은 순위가 깨지지 않았는가다.

| 확인 | 값 |
| --- | --- |
| 행 수 · id 순서 | 296,302 · `test.csv`와 동일 |
| 유한값 · 범위 | 전부 유한 · [3.4e-06, 1.0] |
| 서로 다른 값 수 | 296,302(동률 없음) |
| 안전판 `e88f706e`(후보 1, 5:1 혼합판)와 스피어만 | 0.997704 |
| CV 전용판 `b24e5ba7`(후보 2)와 스피어만 | 0.997590 |
| in-sample OOF AUC(참고치, 판정에 쓰지 않음) | 0.9704159 |
| 제출 CSV SHA-256 | `c4262346a2abfb0578055e1753d07f76585012ec2d185ea5d9e6319ed8b248ca` |

#69의 두 후보끼리는 스피어만 0.99987이었는데 확장 스택은 둘과 0.9976~0.9977이다.
안전판과 실제로 다른 장이라는 뜻이고, 첫 번째 장과 두 번째 장이 서로 다른 위험을 진다는 지도의 의도에 맞는다.

가중치 분포도 판정과 같은 그림이다.
절댓값 합이 자체 35에서 6.35, 외부 207에서 23.16이며, 양쪽 모두 음수 계수가 섞여 있다(#443의 "약한 구성원이 많은 풀에서는 학습된 음수 가중이 필요하다").
절댓값이 큰 자체 구성원은 exp131 0.64, exp157 0.46, exp106 0.45이고 외부는 bolt47 foldsafe_te_wide 1.05, adarsh22 catnative 0.79다.

## 출처와 라이선스

외부 207구성원의 출처와 라이선스는 manifest의 `members`(구성원별)와 `external_summary`(집계)에 있고, 원본 근거는 #442 장부 `docs/research/external-member-ledger.json`이다.
집계는 CC0 1.0 196, CC BY 4.0 6, Apache 2.0 5다.

| 소유자 | 데이터셋 | 라이선스 | 구성원 |
| --- | --- | --- | ---: |
| szymonkapiski | `szymonkapiski/s6e8-oof-library-47-models` | CC0-1.0 | 65 |
| szymonkapiski | `szymonkapiski/s6e8-50-weakest-oof-models` | CC0-1.0 | 50 |
| boltuzamaki | `boltuzamaki/s6e8-oof-prediction-library` | CC0-1.0 | 44 |
| adarsh1077 | `adarsh1077/s6e8-adarsh-oof-library` | CC0-1.0 | 22 |
| hboyang | `hboyang/s6e8-catstrall-member` | CC0-1.0 | 6 |
| raykkretzschmar | `raykkretzschmar/s6e8-fm-lattice-blend-members` | Apache-2.0 | 5 |
| dariushafshar | `dariushafshar/s6e8-golem-oof-library` | CC0-1.0 | 3 |
| mohankrishnathalla | `mohankrishnathalla/s6e8-cat-mlp-oof`, `s6e8-lgb-dart-oof`, `s6e8-xgb-oof` | CC0-1.0 | 3 |
| beicicc | `beicicc/s6e8-fixed900-identity-digit-lightgbm-artifacts` | CC-BY-4.0 | 2 |
| beicicc | `beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts` | CC-BY-4.0 | 1 |
| beicicc | `beicicc/s6e8-second-seed-fixed-schedule-lookup-artifacts` | CC-BY-4.0 | 1 |
| beicicc | `beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts` | CC-BY-4.0 | 1 |
| beicicc | `beicicc/s6e8-fixed4-realmlp-two-seed-artifacts` | CC-BY-4.0 | 1 |
| beicicc | `beicicc/s6e8-fixed1500-xgb-identity-digit-artifacts` | CC0-1.0 | 2 |
| beicicc | `beicicc/s6e8-fixed1500-xgb-screen-relation-artifacts` | CC0-1.0 | 1 |

CC BY 4.0 구성원 6개(beicicc)와 Apache 2.0 구성원 5개(raykkretzschmar)는 저작자 표시 의무가 있어 이 표와 manifest에 소유자·데이터셋을 남긴다.
예측 배열만 결합하고 코드나 가중치를 재배포하지 않으므로 Apache 2.0 원문 사본이나 NOTICE 보존 의무는 발생하지 않는다.
szymonkapiski 라이브러리의 `pub_*` 구성원은 다른 저자 노트북의 재실행이며, 상류 저자는 장부의 `upstream` 필드에 있다.

## #445로 넘기는 것

- 제출 파일: `artifacts/submissions/issue444-extended-stack.csv`, SHA-256 위 표.
  업로드 전에 행 수·id 순서·유한성·SHA-256을 다시 확인한다.
- MLflow 기록: `pipeline.submit --record-existing <ref> --submission artifacts/submissions/issue444-extended-stack.csv --run-name ensemble_shrunk_rank_logit_logistic_issue444_own35_ext207 --source-run-id b24e5ba7b7eb4e3a9e10788005896328 --git-commit <조립 커밋> --artifact docs/research/extended-stack-submission-manifest.json --artifact docs/research/external-member-ledger.json` 형태로, `e88f706e`와 같은 경로.
  param에 `issue=444`, `ensemble.member_count=242`, `external.member_count=207`, `external.ledger_sha256`, `refit.mix=cv5_full1`, tag에 `external.licenses=CC0-1.0:196,CC-BY-4.0:6,Apache-2.0:5`를 남긴다.
- 최종 두 장: 안전판 `e88f706e`(Public 0.97099)와 이 확장 스택.
  Public 점수는 기록만 하고 판정에 쓰지 않는다.
- 재현: `uv run python scripts/judge_extended_stack.py --prepare`(캐시가 없을 때) 뒤 `uv run python scripts/assemble_extended_stack.py`.
  같은 입력이면 제출 CSV SHA-256이 같아야 한다.
