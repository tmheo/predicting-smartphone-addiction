# 엄격 외부 후보 동결·중첩 선별 실행 절차

2026-08-30 실행 회차에서 외부 후보 동결 명세를 만들고 [ADR-0005](../adr/0005-strict-external-member-nested-selection.md)의 중첩 선별 판정을 실행할 때 따르는 절차다.
도구는 `scripts/freeze_external_candidates.py`(동결 명세 생성기)와 `scripts/judge_strict_external_selection.py`(판정 도구)이며 둘 다 로컬 CPU에서만 돈다.
예행 결과와 도구 검증은 [docs/research/strict-external-selection-rehearsal.md](../research/strict-external-selection-rehearsal.md)에 있다.

**2026-08-28 결정으로 판정 방식이 바뀐다.**
ADR-0005의 정확 검색(자체 35 시작 + 엄격 후보 선별)은 비교 팔 313을 구조적으로 넘을 수 없어 [#490](https://github.com/tmheo/predicting-smartphone-addiction/issues/490)에서 사전 고정 nested 사다리로 계약을 개정하고 [#491](https://github.com/tmheo/predicting-smartphone-addiction/issues/491)에서 사다리 도구를 만든다.
그 전까지 이 문서의 1·2단계(동결 명세, precommit)와 `baseline`은 그대로 유효하고, 3~5단계의 `select`·`full`은 개정 뒤 사다리 명령으로 바뀐다.

## 시작 조건

- 증분 조사([#487](https://github.com/tmheo/predicting-smartphone-addiction/issues/487))가 `외부 구성원 조사 완결`이고 색인 `docs/research/external-member-ledger-v3/index.json`이 갱신돼 있다.
- 조사 도중 하나라도 끝내지 못했으면 동결과 판정을 시작하지 않는다.
- ravi20076 v1 세 후보의 `rehosted_training_data_private_notebook` 주의 사항을 어떻게 볼지 사용자가 정했다([#488](https://github.com/tmheo/predicting-smartphone-addiction/issues/488)).
  `근거 부족`으로 보면 색인 도구로 새 감사 기록을 만들어 `supersedes`로 잇는 것이 정식 경로다.
  시간이 없으면 동결 생성기의 `--exclude "구성원=사유"`로 명세의 `user_exclusions`에 사유와 함께 남길 수 있다.
- 작업 폴더가 커밋된 상태다.
  실제 판정의 `precommit`은 git dirty를 거부한다.
- `data/`, `mlflow.db`, `mlruns/`, `artifacts/full-refit/`가 작업 폴더에 있다.
  워크트리에서는 메인 체크아웃으로 심볼릭 링크를 건다.

## 순서

모든 명령은 저장소 루트에서 실행한다.

1. 색인 검사와 동결 명세 생성.

   ```bash
   uv run python scripts/freeze_external_candidates.py --verify-only
   uv run python scripts/freeze_external_candidates.py --survey-cutoff <증분 조사 기준 시각, 예 2026-08-30T10:00:00Z>
   ```

   명세는 `docs/research/external-candidate-freeze/<후보 집합 식별자>.json`에 생기며 커밋 대상이다.
   같은 경로가 있으면 덮어쓰지 않는다.
   후보가 늘거나 공개 판본이 바뀌면 새 명세를 만든다.

2. 동결 명세를 커밋한 뒤 사전 고정.

   ```bash
   uv run python scripts/judge_strict_external_selection.py precommit --spec docs/research/external-candidate-freeze/<식별자>.json
   ```

   `run-logs/strict-external-selection/<식별자>/precommit.json`에 동결 입력 해시, 비교 팔 313과 자체 35의 구성 해시, 고정 결합기, 검색 규칙, 문턱, 코드 상태가 고정된다.
   여기까지가 결과를 보기 전에 고정해야 하는 전부다.
   이 단계에서 자체 35, 비교 팔 313, 후보 OOF 행렬이 `cache/`에 저장되고 해시가 precommit에 기록된다.

3. 분할별 작업 실행.

   ```bash
   uv run python scripts/judge_strict_external_selection.py run --run-dir run-logs/strict-external-selection/<식별자> --workers 5 --heavy-workers 2 --threads 2
   ```

   `baseline k`(비교 팔 313의 봉인 분할 예측, 작업당 10GB대 메모리)와 `select k`(후보 선별과 봉인 예측, 작업당 3GB대) 열 개가 하위 프로세스로 돈다.
   진행은 `fold-<k>/progress.jsonl`과 `logs/`로 본다.
   중단되면 같은 명령으로 이어 달린다.
   산출물이 있는 작업은 건너뛰고, 모든 하위 명령은 시작할 때 입력 해시와 코드 상태를 precommit과 다시 대조한다.
   코드나 입력이 바뀌었으면 전체가 `판정 불가`이며 precommit부터 다시 한다.

4. 비교와 안정성.

   ```bash
   uv run python scripts/judge_strict_external_selection.py compare --run-dir run-logs/strict-external-selection/<식별자>
   ```

   `nested-comparison.json`에 두 팔의 이어붙인 AUC 차이, 분할별 차이, 5/5 부호, 문턱 판정, 비교 팔 재현(#455의 `0.9703509` 대비)이 남고 `selection-stability.json`이 함께 생긴다.
   결과를 본 뒤 문턱과 규칙을 바꾸지 않는다.

5. 통과했을 때만 전체 OOF 제안 명단.

   ```bash
   uv run python scripts/judge_strict_external_selection.py full --run-dir run-logs/strict-external-selection/<식별자>
   ```

   `full-selection.json`의 `proposal`이 엄격 외부 제안 구성이다.
   미달이면 이 단계를 건너뛰고 현재 두 장(`e88f706e` + `443b3a71`) 유지가 결론이다.

6. 보고와 해시 묶음.

   ```bash
   uv run python scripts/judge_strict_external_selection.py report --run-dir run-logs/strict-external-selection/<식별자>
   ```

   `report.md`와 `manifest.sha256`이 생긴다.
   `run-logs/`는 커밋 제외 경로이므로 판정 기록으로 남길 파일(`precommit.json`, `fold-*/selection.json`, `fold-*/baseline.json`, `nested-comparison.json`, `selection-stability.json`, `full-selection.json`, `report.md`, `manifest.sha256`)은 `docs/research/strict-external-selection/<식별자>/`로 복사해 커밋한다.
   예측 parquet는 복사하지 않는다.

7. 통과했고 사용자가 승인했을 때만 조립.

   ```bash
   uv run python scripts/judge_strict_external_selection.py assemble --run-dir run-logs/strict-external-selection/<식별자>
   ```

   제출 CSV는 `artifacts/submissions/strict-external-<식별자>.csv`, manifest는 실행 폴더의 `assembly-manifest.json`이다.
   자체 35의 시험 예측은 5:1 혼합판, 외부 후보는 명세의 정규화 시험 배열이며 결합기는 전체 OOF에 한 번 적합한다.
   Kaggle 업로드와 최종 두 장 수동 고정은 사용자가 다시 승인해야 하며 마지막 업로드는 2026-08-31 12:00 UTC 전이다.

## 실행 인계 완결 조건과 산출물 대응

[#481](https://github.com/tmheo/predicting-smartphone-addiction/issues/481)의 항목은 다음 산출물이 채운다.

| 조건 | 산출물 |
| --- | --- |
| 변경 불가 감사 기록, 자격 판정과 보증 한계 | 색인과 `records/`(증분 조사 결과), 동결 명세의 `candidates[].record_sha256` |
| 후보 집합 식별자, 계약 판본, 조사 기준 시각, 순서 있는 기록 식별자와 예측 쌍 SHA-256, 선택 정책, 명세 자체 SHA-256 | 동결 명세(`candidate_set_id`, `contract_version`, `survey_cutoff`, `candidates[]`, `selection_policy`, `spec_sha256`) |
| 비교 팔 313과 자체 35의 입력 명세와 내용 해시 | `precommit.json`의 `comparison_arm.composition_sha256`, `own_start.composition_sha256`, `caches` |
| 고정 결합기, 결정적 검색, 동률, 충돌, 중단 규칙 | `precommit.json`의 `combiner`, `search_rules` |
| 교체 문턱 | `precommit.json`의 `gate` |
| 변경 불가 산출물 묶음 | `fold-<k>/selection.json`, `fold-<k>/predictions.parquet`, `fold-<k>/baseline-predictions.parquet`, `nested-comparison.json`, `selection-stability.json`, `full-selection.json`, `report.md`, `manifest.sha256` |
| 실패·재개 규칙 | `precommit.json`의 `rules`와 모든 하위 명령의 사전 대조 |
| 업로드·고정의 사용자 승인 경계 | `precommit.json`의 `rules.upload`, 이 문서 7단계 |

## 시간과 자원

예행(합성 후보 6개, 분할 0 하나, 4스레드)에서 검색 풀 점수 한 번은 약 24초(메타 적합 10회)이고 봉인 분할 하나의 선별은 평가 횟수에 비례한다.
후보 19개면 분할 하나에 대략 300회 평가, 두 시간 안팎이고 분할 5곳을 병렬(프로세스당 약 4GB)로 돌리면 전체 판정은 세 시간 안에 끝난다.
비교 팔 313의 봉인 분할 예측은 분할당 8분 안팎이며 메모리 때문에 동시 2개까지만 둔다.
통과 뒤 전체 OOF 검색은 열린 분할이 5곳이라 평가당 약 40초, 세 시간 안팎이다.

## 하지 않는 일

- 결과를 본 뒤 문턱, 검색 규칙, 후보 순서를 바꾸는 일.
- 분할별 명단을 투표·교집합·합집합으로 합쳐 제안 명단을 만드는 일.
- 공개 점수를 어느 단계에든 쓰는 일.
- 외부 예측을 `artifacts/pool.yaml`, champion 판정, 안전판 제출에 넣는 일.
- 외부 배열을 저장소에 커밋하거나 재배포하는 일.
