# 외부 후보 동결 명세 생성기와 중첩 선별 판정 도구 예행 (이슈 #486)

## 결론

[#482](https://github.com/tmheo/predicting-smartphone-addiction/issues/482) 계약의 외부 후보 동결 명세 생성기 `scripts/freeze_external_candidates.py`와 [ADR-0005](../adr/0005-strict-external-member-nested-selection.md)의 중첩 선별 판정 도구 `scripts/judge_strict_external_selection.py`를 만들고, 자체 35개에서 파생한 합성 후보 6개로 바깥 분할 0을 봉인하는 예행을 끝냈다.
실제 판본 3 후보 19개의 동결 명세는 만들지 않았고 실제 판정도 하지 않았다(증분 조사 [#487](https://github.com/tmheo/predicting-smartphone-addiction/issues/487) 뒤에 한다).
동결 생성기의 검사(`--verify-only`)는 실제 색인의 19개 기록·배열·해시·재채점 AUC를 모두 통과했다.

예행 도중 두 가지를 발견해 고쳤다.

1. 후보 절차 팔의 빠른 결합기 구현이 학습 행을 분할 순서로 이어붙이자 등록 결합기 `shrunk_rank_logit_logistic`과 봉인 예측이 `7.96e-04`까지 달랐다.
   원인은 행 순서다. rank_logit 이중 표현은 열이 서로 강하게 상관돼 lbfgs가 행 순서에 따라 다른 계수에 닿는다.
   원래 행 순서를 지키게 고친 뒤 차이는 `0.0`(비트 단위 동일)이다.
   등록 결합기 자체도 행 순서에 이만큼 민감하다는 뜻이므로, 어떤 판정 도구든 행 순서를 `evaluate_nested`와 같게 유지해야 한다.
2. ADR-0005의 후보 풀 중복 불변식을 "명단의 모든 쌍"으로 읽으면 자체 35개 안의 쌍 `exp131_lookup_bivariate_plr5`·`exp157_lookup_muon_initavg8`가 열린 4분할(분할 0 봉인)에서 스피어만 `0.998102`라 시작 구성부터 위반이다.
   전체 OOF에서는 0.998 미만이라 풀 진입을 통과했지만 4분할 부분집합에서는 넘는다.
   검색이 자체 구성원을 바꿀 수 없으므로 도구는 외부 후보가 낀 쌍에만 불변식을 적용하고 자체끼리의 쌍은 진단값으로 기록한다.
   계약 문구 정정은 [#490](https://github.com/tmheo/predicting-smartphone-addiction/issues/490)에서 다룬다.

또 하나, 도구를 만들면서 계약 자체의 문제가 드러났다.
ADR-0005의 후보 절차 팔(자체 35 필수 시작 + 엄격 후보 선별)은 비교 팔 313을 구조적으로 넘을 수 없다(자체 35 nested `0.9698106` vs 313 `0.9703509`, 외부 209 전체 기여 `+0.00051`, 엄격 19개 중 11개는 313 안에 이미 있음).
2026-08-28 사용자 결정으로 08-30 판정은 정확 검색 대신 [#455](https://github.com/tmheo/predicting-smartphone-addiction/issues/455)식 사전 고정 nested 사다리로 바꾼다([#490](https://github.com/tmheo/predicting-smartphone-addiction/issues/490) 계약 개정, [#491](https://github.com/tmheo/predicting-smartphone-addiction/issues/491) 사다리 도구).
이 문서의 도구 가운데 동결 명세 생성기, `precommit`(동결 입력·313 구성·자체 35 구성·코드 상태 고정), `baseline`(313 등록 결합기 봉인 예측)은 사다리 판정에서도 그대로 쓰고, 정확 검색(`select`·`full`)은 계약이 되살아날 때를 위해 보존한다.

## 도구

| 도구 | 역할 |
| --- | --- |
| `scripts/freeze_external_candidates.py` | 판본 3 색인 → 외부 후보 동결 명세. 기록 `record_sha256` 재계산, 정규화 배열 형태·SHA-256·쌍 SHA-256·재채점 AUC 대조, 정확 중복 검사, `--exclude "구성원=사유"` 사용자 제외 기록, 명세 자체 SHA-256과 후보 집합 식별자(`ecf-v3-<내용 해시 12자>`). 같은 경로는 덮어쓰지 않는다. |
| `scripts/judge_strict_external_selection.py precommit` | 동결 명세, folds, 자체 35(풀 장부 + MLflow OOF), 비교 팔 313(#457 manifest + 판본 2 장부), 후보 배열을 적재·검증해 `cache/`와 `precommit.json`(구성 해시, 결합기, 검색 규칙, 문턱, 코드 상태)을 만든다. 실제 판정은 git dirty와 부분 분할을 거부한다. |
| `... run` | `baseline k`(313 등록 결합기)와 `select k`(후보 선별 + 봉인 예측) 하위 프로세스 실행. 무거운 작업 동시 상한, 이미 있는 산출물 건너뛰기, 다른 드라이버의 작업 감지. |
| `... select` | 열린 4분할에서 단독 AUC·스피어만·제외·충돌을 진단하고 ADR-0005 검색(순방향 추가·단일 원자 교체 → 후방 제거 → 순방향 → 쌍 추가 1회 → 재수렴)을 실행한다. 잠근 명단으로 봉인 예측을 만들고 등록 결합기와 대조한다. |
| `... compare` | 두 팔의 봉인 예측을 원래 행 순서로 이어붙여 AUC 차이, 분할별 차이, 5/5 부호, 문턱 판정, 비교 팔 재현(#455 대비)을 `nested-comparison.json`에 남기고 `selection-stability.json`을 만든다. |
| `... full` | 통과했을 때만 전체 OOF에 같은 검색을 한 번 적용해 `full-selection.json`(제안 명단, 시험 열 순서). |
| `... report`, `... assemble` | `report.md`·`manifest.sha256`, 통과·승인 뒤 제출 CSV와 조립 manifest. |
| `... rehearsal-index` | 자체 35개 파생 합성 후보로 판본 3 모양의 색인을 만든다(예행 전용, 실제 후보 배열을 읽지 않는다). |
| `tests/test_strict_external_selection.py` | 검색 규칙 특성화 시험 10개(최대 양수 이동, 동결 순서 동률, 제외, 단일 원자 교체만 허용, 다중 충돌 차단, 후방 제거, 쌍 추가와 재수렴, 충돌 쌍 제외). |

모든 하위 명령은 시작할 때 동결 명세, train·folds·pool·manifest·캐시 해시와 판정 도구·동결 생성기·`pipeline/ensemble.py`의 SHA-256, git commit을 `precommit.json`과 대조하고 하나라도 다르면 `판정 불가`로 멈춘다.

## 예행 설정

- 합성 색인: `run-logs/strict-external-selection/rehearsal-index/` (`rehearsal-index --count 4 --sigma 0.6 --informative 2 --shift 0.3 --seed 486`).
  `syn00`·`syn01`은 목표값 쪽으로 로짓을 0.3 밀어 승인 경로를 열었고, `syn_own_dup`은 자체 구성원의 근접 복제(자체 충돌 제외 경로), `syn_cand_dup`은 `syn00`의 근접 복제(후보끼리 충돌·교체 경로)다.
- 동결 명세: `ecf-rehearsal-5520002d7172` (spec_sha256 `f97a5388…`), 후보 6개.
- precommit: `run-logs/strict-external-selection/ecf-rehearsal-5520002d7172/precommit.json` (`fdd2bfaa…`), 바깥 분할 `[0]`, 비교 팔 313 구성 해시와 자체 35 구성 해시 기록.
- 실행: `run --workers 2 --heavy-workers 1 --threads 4` (로컬 CPU).

## 예행 결과(분할 0 봉인)

| 항목 | 값 |
| --- | --- |
| 열린 행 / 봉인 행 | 553,095 / 138,274 |
| 자체 충돌 제외 | `syn_own_dup` (자체 `exp133`과 스피어만 0.999996) |
| 후보끼리 충돌 | `syn00` ↔ `syn_cand_dup` |
| 자체끼리 0.998 이상 쌍(진단) | `exp131_lookup_bivariate_plr5` · `exp157_lookup_muon_initavg8` 0.998102 |
| 시작 점수(자체 35) | 0.9699535 |
| 검색 | 순방향 1: `syn00` 추가 +0.0081229 → 2: `syn01` 추가 +0.0056964 → 3: `syn00`→`syn_cand_dup` 교체 +4.7e-06 → 4: `syn02` 추가 +1.6e-06 → 5: 양수 없음 → 후방 3개 평가 양수 없음 → 순방향 양수 없음 → 허용 쌍 0개 → 종료 |
| 평가 / 메타 적합 / 소요 | 17회 / 277회 / 565초(첫 평가는 순위 캐시 생성 포함 약 80초, 이후 평가당 약 24초) |
| 최종 명단 | 자체 35 + `syn01`, `syn02`, `syn_cand_dup` (점수 0.9837791, 외부가 낀 쌍 최대 스피어만 0.993573) |
| 봉인 예측 λ | 1.0 (LOFO λ별 AUC 0.25: 0.9759919, 0.5: 0.9803641, 0.75: 0.9829454, 1.0: 0.9837791) |
| 등록 결합기 대조 | 최대 절대 차이 **0.0**, λ 일치 |
| 비교 팔 313 봉인 예측 | λ 1.0, 473초, 분할 0 AUC 0.9697543 = #455 `ablate_new_nhtquyn` 분할 0 `0.9697543` (재현) |
| compare | 후보 팔 0.9834575 vs 비교 팔 0.9697543, `판정 불가(부분 분할, 예행)` |

후보 팔이 크게 높은 것은 목표값을 밀어 넣은 합성 후보 때문이며 실제 의미는 없다.
산출물 12개의 해시는 실행 폴더의 `manifest.sha256`에 있다(`precommit.json` `80c089fd…`, `fold-0/selection.json` `7f637626…`, `nested-comparison.json` `a9ce2677…`).

## 실측 비용

- 로지스틱 적합 1회(553k행, 108특성, 4스레드): 0.6초. 검색 풀 점수 1회: 메타 적합 10회 + AUC 17회 ≈ 24초(후보 41열 기준).
- 후보 19개면 분할당 약 300회 평가로 1시간 안팎, 분할 5곳 병렬(프로세스당 약 4GB)로 두 시간 안, 통과 뒤 전체 OOF 검색 두 시간 안팎이라는 추정은 유지된다.
- 시작 구성을 313으로 바꾸면 적합당 수십 초, 프로세스당 15GB대라 같은 창에 넣기 어렵다. 이것이 사다리로 바꾼 이유 가운데 하나다.

## 남은 일

- [#490](https://github.com/tmheo/predicting-smartphone-addiction/issues/490): 계약 개정(사다리 구성, 자체끼리 0.998 쌍 처리, ADR-0005·#478·#481 정정).
- [#491](https://github.com/tmheo/predicting-smartphone-addiction/issues/491): 사다리 판정 도구(이 도구의 동결 생성기·precommit·baseline 재사용).
- [#488](https://github.com/tmheo/predicting-smartphone-addiction/issues/488): 실제 동결·판정·조립·업로드.
