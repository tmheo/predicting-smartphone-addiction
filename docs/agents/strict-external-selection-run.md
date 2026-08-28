# 엄격 외부 후보 동결·사다리 판정 실행 절차

외부 후보 동결 명세를 만들고 [ADR-0006](../adr/0006-strict-external-candidate-ladder.md)의 사전 고정 사다리 판정을 실행할 때 따르는 절차다.
도구는 `scripts/freeze_external_candidates.py`(동결 명세 생성기)와 `scripts/judge_strict_external_selection.py`(판정 도구)이며 둘 다 로컬 CPU에서만 돈다.
도구의 예행 결과는 [docs/research/strict-external-selection-rehearsal.md](../research/strict-external-selection-rehearsal.md)에, 1회차 실제 판정 기록은 `docs/research/strict-external-selection/<후보 집합 식별자>/`에 있다.

ADR-0005의 정확 검색은 [#490](https://github.com/tmheo/predicting-smartphone-addiction/issues/490)에서 접었다.
그 명령(`adr0005-select`, `adr0005-compare`, `adr0005-full`)은 보존만 하며 이 절차에서 쓰지 않는다.

## 시작 조건

- 증분 조사([#487](https://github.com/tmheo/predicting-smartphone-addiction/issues/487))가 `외부 구성원 조사 완결`이고 색인 `docs/research/external-member-ledger-v3/index.json`이 갱신돼 있다.
  조사 도중 하나라도 끝내지 못했으면 동결과 판정을 시작하지 않는다.
- 자격 판단이 필요한 주의 사항(예: ravi20076 v1의 `rehosted_training_data_private_notebook`)을 어떻게 볼지 사용자가 동결 전에 정했다.
  `근거 부족`으로 보면 색인 도구로 새 감사 기록을 만들어 `supersedes`로 잇는 것이 정식 경로다.
  시간이 없으면 동결 생성기의 `--exclude "구성원=사유"`로 명세의 `user_exclusions`에 사유와 함께 남길 수 있다.
  결과를 본 뒤에는 그 주의 사항을 이유로 구성을 고르거나 빼지 않는다.
- 작업 폴더가 커밋된 상태다.
  실제 판정의 `precommit`은 git dirty를 거부한다.
- `data/`, `mlruns/`, `artifacts/full-refit/`, 외부 예측 배열(`data/external/`)이 작업 폴더에 있다.
  워크트리에서는 메인 체크아웃으로 심볼릭 링크를 건다.
- 다른 400열대 shrunk 작업이 이 기계에서 돌고 있지 않다.
  판정 작업은 동시 3개가 상한이다(#455: 5개는 커널 패닉).

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
   자격 있는 현행 기록은 전체 OOF 성능이나 근접 중복을 이유로 빼지 않는다.

2. 동결 명세를 커밋한 뒤 사전 고정.

   ```bash
   uv run python scripts/judge_strict_external_selection.py precommit --spec docs/research/external-candidate-freeze/<식별자>.json
   ```

   `run-logs/strict-external-selection/<식별자>/precommit.json`에 다음이 고정된다.
   동결 입력 해시, 자체 35·비교 팔 313·후보의 구성 해시, 313 구성원의 예측 쌍 SHA-256, 정확 중복 후보와 대응 구성원, 사다리 구성 목록(구성마다 열 순서), 고정 결합기, 문턱, 자기 검사 기준값, 선택 규칙, 코드 상태.
   여기까지가 결과를 보기 전에 고정해야 하는 전부이며, 결과를 본 뒤 더하거나 빼지 않는다.
   자체 35, 비교 팔 313, 후보 OOF 행렬은 `cache/`에 저장되고 해시가 precommit에 기록된다.
   사다리 후보가 0개면 precommit에 그 사실이 남고 3단계는 실행할 작업이 없다.

3. 작업 실행.

   ```bash
   uv run python scripts/judge_strict_external_selection.py run --run-dir run-logs/strict-external-selection/<식별자> --workers 3 --threads 4
   ```

   `baseline k`(비교 팔 313의 봉인 분할 예측 5개, 분할당 8분 안팎)와 `ladder <구성>`(구성 하나의 nested 5분할, 300열대는 40분에서 1시간)이 하위 프로세스로 돈다.
   모두 작업당 10GB대 메모리라 동시 상한은 3이다.
   진행은 `logs/`로 본다.
   중단되면 같은 명령으로 이어 달린다.
   산출물이 있는 작업은 건너뛰고, 모든 하위 명령은 시작할 때 입력 해시와 코드 상태를 precommit과 다시 대조한다.
   코드나 입력이 바뀌었으면 전체가 `판정 불가`이며 precommit부터 다시 한다.
   계산 하나라도 실패하면 완료한 일부 결과를 쓰지 않는다.

4. 비교와 선택.

   ```bash
   uv run python scripts/judge_strict_external_selection.py compare --run-dir run-logs/strict-external-selection/<식별자>
   ```

   `ladder-comparison.json`에 다음이 남는다.
   비교 팔 자기 검사(313 봉인 예측을 이어붙인 AUC와 분할별 AUC가 #455 `0.9703509`와 잡음 바닥 `5.7e-06` 안인지), 구성별 nested AUC·가중 OOF AUC(진단)·313 대비 이어붙인 차이·분할별 차이·5/5 부호·문턱 판정, 선택 규칙 적용 경로와 제안 구성.
   자기 검사가 실패하면 판정 불가다.
   통과 구성이 없으면 현재 두 장(`e88f706e` + `443b3a71`) 유지가 결론이다.
   결과를 본 뒤 문턱과 규칙을 바꾸지 않는다.

5. 보고와 해시 묶음.

   ```bash
   uv run python scripts/judge_strict_external_selection.py report --run-dir run-logs/strict-external-selection/<식별자>
   ```

   `report.md`와 `manifest.sha256`이 생긴다.
   `run-logs/`는 커밋 제외 경로이므로 판정 기록으로 남길 파일(`precommit.json`, `fold-*/baseline.json`, `ladder/*/nested.json`, `ladder-comparison.json`, `report.md`, `manifest.sha256`)은 `docs/research/strict-external-selection/<식별자>/`로 같은 상대 경로에 복사해 커밋한다.
   예측 parquet는 복사하지 않는다(해시는 `manifest.sha256`에 있다).

6. 통과했고 사용자가 승인했을 때만 조립([#488](https://github.com/tmheo/predicting-smartphone-addiction/issues/488)).

   ```bash
   uv run python scripts/judge_strict_external_selection.py assemble --run-dir run-logs/strict-external-selection/<식별자>
   ```

   제출 CSV는 `artifacts/submissions/strict-external-<식별자>-<구성>.csv`, manifest는 실행 폴더의 `assembly-manifest.json`이다.
   자체 35의 시험 예측은 5:1 혼합판, 313의 외부는 장부 시험 배열(#457 manifest 해시 대조), 후보는 명세의 정규화 시험 배열이며 결합기는 전체 OOF에 한 번 적합한다.
   Kaggle 업로드와 최종 두 장 수동 고정은 사용자가 다시 승인해야 하며 마지막 업로드는 2026-08-31 12:00 UTC 전이다.

## 실행 인계 완결 조건과 산출물 대응

[#481](https://github.com/tmheo/predicting-smartphone-addiction/issues/481)의 항목을 ADR-0006의 개정대로 읽으면 다음 산출물이 채운다.

| 조건 | 산출물 |
| --- | --- |
| 변경 불가 감사 기록, 자격 판정과 보증 한계 | 색인과 `records/`(증분 조사 결과), 동결 명세의 `candidates[].record_sha256` |
| 후보 집합 식별자, 계약 판본, 조사 기준 시각, 순서 있는 기록 식별자와 예측 쌍 SHA-256, 선택 정책, 명세 자체 SHA-256 | 동결 명세(`candidate_set_id`, `contract_version`, `survey_cutoff`, `candidates[]`, `selection_policy`, `spec_sha256`) |
| 313개와 사다리 후보의 입력 명세와 내용 해시 | `precommit.json`의 `comparison_arm`(구성 해시, 예측 쌍 해시), `candidate_arm`, `exact_duplicates`, `caches` |
| 사다리 구성 목록·선택 규칙 | `precommit.json`의 `ladder.configs`, `ladder.omitted`, `selection_rules` |
| 고정 결합기 | `precommit.json`의 `combiner` |
| 교체 문턱 | `precommit.json`의 `gate`, `noise_floor` |
| 비교 팔 자기 검사·구성별 봉인 예측·사다리 비교 | `fold-<k>/baseline-predictions.parquet`·`baseline.json`, `ladder/<구성>/fold-<k>/predictions.parquet`·`nested.json`, `ladder-comparison.json` |
| 변경 불가 산출물 묶음 | `report.md`, `manifest.sha256` |
| 실패·재개 규칙 | `precommit.json`의 `rules`와 모든 하위 명령의 사전 대조 |
| 업로드·고정의 사용자 승인 경계 | `precommit.json`의 `rules.upload`, 이 문서 6단계 |

## 시간과 자원

- 비교 팔 313의 봉인 분할 예측은 분할당 8분 안팎(#486 예행 473초).
- 사다리 구성 하나(313 + 후보 수 열)의 nested 5분할은 #455의 313열 실측 2,402초를 기준으로 40분에서 1시간이다.
  전체 구성 작업은 근접 중복 진단(열린 4분할 스피어만 5회)이 몇 분 더 든다.
- 후보 8개·6구성이면 작업 11개, 동시 3개로 두 시간 안팎이다.

## 하지 않는 일

- 결과를 본 뒤 문턱, 사다리 구성, 선택 규칙, 후보 순서를 바꾸는 일.
- 단독 성능이나 근접 중복으로 후보를 동결 전에 빼는 일.
- 후보 하나씩 더한 구성을 사다리에 넣는 일.
- 가중 OOF AUC나 공개 점수를 판정에 쓰는 일.
- 외부 예측을 `artifacts/pool.yaml`, champion 판정, 안전판 제출에 넣는 일.
- 외부 배열을 저장소에 커밋하거나 재배포하는 일.
