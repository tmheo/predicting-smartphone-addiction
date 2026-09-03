# 중첩 결합 판정 도구의 재현 전용 풀 파일 입력 지원 조사 (#620)

지도 [#619](https://github.com/tmheo/predicting-smartphone-addiction/issues/619)는 대회 기록(자체 36개 풀 파일 `artifacts/pool.yaml`, 314 확장 스택, 재학습 장부)을 동결하고 재현 구성원을 재현 전용 풀 파일에 두기로 했다.
이 문서는 중첩 결합 판정 도구 세 갈래(동결 명세 생성, 사다리 판정, 등록 결합기 실행)가 `artifacts/pool.yaml` 대신 그 재현 전용 풀 파일을 입력으로 받을 수 있는지, 받을 수 없다면 어느 module의 어느 진입점을 얼마나 바꿔야 하는지, 그리고 [#624](https://github.com/tmheo/predicting-smartphone-addiction/issues/624)의 두 기준 판정(자체 36개 기준, 314 확장 기준)에 필요한 입력이 지금 어디에 있는지를 조사한 결과다.
근거는 저장소의 `src/pipeline/`, `scripts/`, `docs/adr/`, `docs/agents/strict-external-selection-run.md`와 커밋된 판정 기록이며, 인용한 행 번호는 커밋 `2aeb887` 기준이다.

## 결론

- 어느 도구도 명령행 옵션으로 `artifacts/pool.yaml` 대신 다른 풀 파일을 받지 않는다.
  `pipeline.ensemble`, `pipeline.pool_judgment`, `pipeline.pool`의 명령행 진입점은 `Pool.load()`를 인자 없이 불러 `ledger.POOL_PATH`에 고정돼 있다.
- 그러나 스택 교체 판정(사다리 판정)의 정본인 `pipeline.round`의 `JudgmentRound`는 풀 파일을 아예 읽지 않는다.
  구성원은 스펙 스크립트가 `MemberSource`로 선언하며, 구성원 출처 adapter(`pipeline.member_sources`)가 파일 형식을 해석한다.
  따라서 재현 구성원을 함께 판정하는 데 기존 판정 도구의 진입점을 바꿀 필요가 없고, 재현 전용 풀 파일을 `MemberSource`로 바꾸는 adapter 하나와 회차 스펙 스크립트만 새로 쓰면 된다.
- 엄격 외부 사다리 도구(`scripts/judge_strict_external_selection.py`)도 자체 구성원을 풀에서 읽지 않고 #457 manifest에서 읽으며, `artifacts/pool.yaml`은 동결 입력 해시로만 봉인한다.
  비교 팔이 313개로 고정된 이 도구는 재현 판정에 맞지 않고, ADR-0009대로 새 회차는 `RoundSpec`으로 쓰는 것이 맞다.
- 동결 명세 생성기(`scripts/freeze_external_candidates.py`)는 외부 장부 색인만 읽고 풀 파일과 무관하다.
- 판정에 쓰이려면 구성원 출처가 `hash-verified`여야 하는데, 현재 풀 장부 형식(`Pool` YAML)에는 예측 배열 해시가 없어 `member_sources.pool_members`는 `identity-only`로만 읽힌다.
  그러므로 재현 전용 풀 파일은 `Pool` YAML이 아니라 실행 식별자와 OOF·시험 배열 해시를 함께 담은 동결 명세 형식이어야 하며, 그 해시를 결과 확인 전에 고정하는 동결 단계가 하나 필요하다.
- 구성원 하나씩의 등록 판정(`pipeline.pool_judgment`)과 등록(`pipeline.pool --admit`)을 재현 전용 풀에 대해 돌리려면 두 명령행 진입점에 `--pool` 옵션을 더하는 작은 변경이 필요하다.
  라이브러리 함수(`generate_pool_judgment`, `load_pool_admission_authorization`, `Pool.load`, `Pool.save`)는 이미 경로 인자를 받는다.
- 314 확장 기준의 입력(314 구성과 해시, 기준값, 분할별 AUC)은 이슈 513 판정 기록에 전부 커밋돼 있어 파일럿 #553의 스펙 스크립트를 거의 그대로 재사용할 수 있다.
  자체 36개 기준은 결합기와 기준값 출처를 사용자가 정해야 한다.
  36개 단독 nested 값은 `shrunk_rank_logit_logistic`(0.9698828758140019)로만 기록돼 있고 분할별 AUC는 로컬 MLflow에만 있다.

## 도구별 풀 경로 입력 방식

| 도구 | 진입점 | 풀 경로 출처 | 별도 파일 가능 여부 |
| --- | --- | --- | --- |
| 풀 장부 module | `src/pipeline/ledger.py` `Pool.load(path=POOL_PATH)` 120행, `Pool.save(path=POOL_PATH)` 145행 | 기본값 `POOL_PATH = Path("artifacts/pool.yaml")` 24행 | 가능하다. 인자로 어떤 경로든 받는다. |
| 등록 결합기 nested 평가(계열 3 판정) | `src/pipeline/ensemble.py` `main()` 1992행 | 2045행 `pool = Pool.load()` 고정 | 불가하다. 1996~2027행의 옵션은 `--only`, `--output`, `--strategy-oof-output`, `--submission`, `--record-issue`, `--baseline-run`뿐이다. |
| 후보 풀 판정(candidate-pool-v2) | `src/pipeline/pool_judgment.py` `main()` 1694행 | 명령행에서는 고정. 라이브러리 `generate_pool_judgment(pool_path=POOL_PATH)` 1547행과 `_freeze()` 383~401행은 경로 인자를 받아 `Pool.load(pool_absolute)`로 읽는다. | 라이브러리로는 가능, 명령행으로는 불가하다. 1696~1727행 옵션에 `--pool`이 없다. |
| 풀 진입·등록 | `src/pipeline/pool.py` `main()` 144행 | 166행 `pool = Pool.load()`, 247행 `POOL_PATH`에 등록 | 불가하다. 등록 권한 검증 `judgment.load_pool_admission_authorization(pool_path=POOL_PATH)`(`src/pipeline/judgment.py` 737~742행)는 경로 인자를 받지만 명령행이 넘기지 않는다. |
| 구성원 출처 adapter | `src/pipeline/member_sources.py` `pool_members(pool)` 96~109행 | `Pool` 객체를 받으므로 `Pool.load(다른 경로)`를 넘길 수 있다. 다만 출처 이름이 109행에 `"artifacts/pool.yaml"`로 고정돼 있다. | 읽기는 가능하나 검증 수준이 `IDENTITY_ONLY`(103행)라 판정용이 아니다. docstring 10~11행이 "비판정 용도 전용"으로 못박는다. |
| 판정 회차(JudgmentRound) | `src/pipeline/round.py` `main(spec)` 1042행, `RoundSpec` 222~236행 | 풀 파일을 읽지 않는다. `ReferenceArm`(143~150행)과 `CandidateArm`(155~162행)이 `MemberSource`를 직접 받는다. | 해당 없음. 스펙 스크립트가 어떤 출처든 `MemberSource`로 선언하면 된다. precommit이 465~466행에서 `load_members` 뒤 `require(HASH_VERIFIED)`를 요구한다. |
| 엄격 외부 사다리 판정 | `scripts/judge_strict_external_selection.py` `precommit --spec` 1768행, `--run-dir` 1769행 | 자체 35개는 213~225행 `load_own()`이 `COMPARISON_MANIFEST_PATH`(#457 manifest, 100행)에서 읽고 docstring이 "풀을 읽지 않는다"고 명시한다. `artifacts/pool.yaml`은 469~470행 `own_start.pool_sha256`으로 봉인하고 541행 재개 검사와 1656행 조립 manifest에서 해시만 대조한다. | 불가하다. 비교 팔이 313개(`COMPARISON_MEMBER_COUNT` 102행), 자체 35개(`OWN_MEMBER_COUNT` 103행)로 상수 고정이라 재현 판정의 틀이 아니다. |
| 외부 후보 동결 명세 생성 | `scripts/freeze_external_candidates.py` `main()` 294~300행 | 풀 파일을 읽지 않는다. `--index`(기본 `docs/research/external-member-ledger-v3/index.json`, 47행), `--survey-cutoff`, `--out`, `--exclude`, `--verify-only`, `--skip-auc`만 받는다. | 해당 없음. 외부 후보 전용이며 자체 구성원과 무관하다. |
| 재사용 적격 자체 후보 동결 명세 생성 | `scripts/freeze_reusable_own_candidates.py` `main()` 786~790행 | 53행 `POOL_PATH` 고정, 595행 `Pool.load(POOL_PATH)`로 현재 풀 구성을 제외 대상으로 읽는다. 588행에서 시작 시각이 `--run-cutoff` 이하인 풀 밖 실행 전부를 후보로 고른다. | 부분적으로 가능하다. `--out`으로 명세 경로는 바꿀 수 있으나 실행 식별자 부분집합을 고르는 옵션이 없어 재현 12개만 담으려면 필터가 필요하다. |

`JudgmentRound`의 재개 검사(`src/pipeline/round.py` 527~560행)는 `sealed_inputs`로 선언한 파일 해시, 팔별 캐시 parquet 해시, git commit, `uv.lock`, 관련 소스 module 해시(287~313행 `default_code_state`)를 precommit과 대조한다.
관련 소스 목록(94~105행)에 `member_sources`와 `members`가 들어 있으므로 adapter를 새로 더한 뒤에는 그 커밋에서 precommit부터 시작해야 한다.

## 최소 변경안

기존 도구의 진입점을 바꾸지 않는 경로가 있으므로 그 경로를 기본으로 두고, 등록 판정을 재현 풀에 대해 돌릴 때만 명령행 옵션을 더한다.

### 1. 사다리 판정: 새 adapter 하나와 회차 스펙 스크립트 두 개

- 재현 전용 풀 파일은 구성원마다 `config`, `run_id`, `oof_sha256`, `test_sha256`(과 `pair_sha256`)을 담은 동결 명세 형식으로 둔다.
  `Pool` YAML 형식은 해시 필드가 없어 `hash-verified`를 만족할 수 없고, `MemberSpec`(`src/pipeline/members.py` 70~90행)은 `HASH_VERIFIED` 선언에 해시 근거를 요구한다(123~126행).
  형식의 본보기는 `docs/research/reusable-own-candidate-freeze/rocf-v1-b42e02ea2e2b.json`의 `candidates[]`(`config`, `run_id`, `seeds`, `git_commit`, `git_dirty`, `oof.array_sha256`, `test.array_sha256`, `prediction_pair_sha256`, `order`)다.
- `src/pipeline/member_sources.py`에 재현 풀 명세를 읽는 adapter를 하나 더한다.
  `manifest_members`(31~68행)의 자체 구성원 분기처럼 `run_id`와 `oof_sha256`를 `MemberSpec`에 넣고 `verification=HASH_VERIFIED`, `origin="reproduction"`으로 선언하면 되며, `freeze_spec_members`(71~93행)의 순서 검사와 `row_contract`를 그대로 옮기면 20~30행이다.
  `pool_members`의 고정 이름(109행)은 이 기회에 `path` 인자로 바꾸는 편이 맞지만 판정 경로에는 영향이 없다.
- 해시를 결과 확인 전에 고정하는 동결 단계는 `scripts/freeze_reusable_own_candidates.py`에 `--run-id`(반복) 필터를 더해 재현 실행 12개만 감사하게 하거나, 같은 검사(3시드 평균본, 깨끗한 코드 상태, 입력 해시 일치, 행 수와 유한성)를 하는 작은 생성기를 새로 두는 두 갈래가 있다.
  후보 선택 루프는 565~600행이고 현재 풀 제외는 595행이다.
- 회차 스펙 스크립트는 `scripts/round_issue553_pilot.py`를 본떠 기준마다 하나씩 둔다.
  회차 id는 ADR-0009대로 `<주제 슬러그>/<이슈 번호>`이므로 예를 들어 `reproduction-pool-own36/issue624`와 `reproduction-pool-ext314/issue624`처럼 슬러그를 달리 하면 같은 이슈 번호로 두 회차를 열 수 있다.
  기준 팔 값은 `ReferenceValues(source, nested_auc, fold_aucs)`(`src/pipeline/round.py` 124~141행)로, 평가 팔은 기준 구성원 뒤에 재현 구성원을 이은 `MemberSource`로 선언한다.
  사다리(원시 4열, 범주 복제와 정확값 TE, 비율과 반올림의 세 단계)는 `CandidateArm`을 여러 개 나열하면 된다.
- 결합기는 `RoundSpec.combiner`가 `ensemble.COMBINER_REGISTRY` 또는 `CSelectedShrunkRankLogitCombiner.name` 가운데 하나여야 한다(`src/pipeline/round.py` 260~265행).
- 게이트는 `StackGate()` 기본값이 `delta_required=AUC_THRESHOLD(0.00002)`, `folds_required_positive=5`다(`src/pipeline/judgment.py` 118행, 1299~1309행).

### 2. 구성원별 등록 판정과 등록: 명령행에 `--pool` 옵션

지도 #619가 말한 "현행 등록 문턱"과 "잔차 기여와 분할 부호"는 후보 풀 판정 `pipeline.pool_judgment`의 규칙이다.
이 도구는 현재 풀(before)과 풀에 후보를 더한 구성(after)을 핵심 결합기 3개로 nested 평가해 전체 nested OOF 차이가 엄격히 양수면 `admit`으로 판정하고(1401~1413행 `_result_for`), 바깥 분할별 차이와 양수 분할 수를 기록에 남긴다(931~946행).
재현 전용 풀에 대해 이 판정을 돌리려면 다음이 최소 변경이다.

- `src/pipeline/pool_judgment.py` `main()`(1694행)에 `--pool` 옵션을 더하고 1746행의 `generate_pool_judgment(request, store=...)` 호출에 `pool_path=args.pool`을 넘긴다.
  `_freeze()`(383~401행)가 이미 그 경로로 `Pool.load`를 하고 풀 해시를 evidence에 봉인한다.
- `src/pipeline/pool.py` `main()`(144행)에 `--pool` 옵션을 더해 166행 `Pool.load()`, 247행 등록 저장, 그리고 `load_pool_admission_authorization(pool_path=...)`(`src/pipeline/judgment.py` 742행) 호출에 같은 경로를 넘긴다.
  이렇게 하면 `--admit`이 `artifacts/pool.yaml`이 아니라 재현 전용 풀 파일에 기록한다.
- `src/pipeline/ensemble.py` `main()`은 champion 대비 계열 3 판정용이라 필수는 아니지만, 재현 풀의 등록 결합기 보고서를 보려면 2045행 `Pool.load()`에 같은 `--pool` 옵션을 붙인다.

이 경로의 재현 전용 풀 파일은 `Pool` YAML 형식이어야 하며, `Pool.save(path)`(`src/pipeline/ledger.py` 145행)로 만들 수 있다.
사다리 판정(1번)과 등록 판정(2번)을 모두 쓴다면 재현 풀 파일이 두 형식(해시 있는 동결 명세, `Pool` YAML)으로 갈리므로, #624에서 어느 경로를 주 판정으로 삼을지 먼저 정해야 한다.

### 3. 바꾸지 않는 것

- `scripts/judge_strict_external_selection.py`와 `scripts/freeze_external_candidates.py`는 손대지 않는다.
  ADR-0006 27행이 비교 팔을 313개로 고정했고, `docs/agents/strict-external-selection-run.md`의 "하지 않는 일"이 외부 예측을 `artifacts/pool.yaml`에 넣지 않는다고 못박는다.
- `artifacts/pool.yaml`(SHA-256 `40947563a00cab8212498c7e339517e387979b14c6477c6ce8e196036e02044c`, 36개)은 그대로 둔다.
  이 해시는 이슈 514 기록의 `full_refit.pool_sha256`과 이슈 513 precommit의 `reassembled.official_pool_file_sha256`과 같아 대회 최종 상태임을 확인했다.

## 두 기준 판정의 입력 목록

| 기준 | 필요 입력 | 현재 값과 위치 |
| --- | --- | --- |
| 자체 36개 기준 | 기준 팔 구성원 36개의 `config`, `run_id`, OOF 해시 | `artifacts/pool.yaml` `members[*]`의 `config`·`run_id`(36개, `Pool.load`로 읽음). OOF 해시는 `docs/research/extended-stack-pool-reassembly/issue513/precommit.json`의 `reassembled.members[0:36]`(`origin: own`, `run_id`, `oof_sha256`)에 hash-verified로 있다. OOF 배열 자체는 로컬 MLflow(`mlruns/`)에서 `RunStore.oof_of(run_id)`로 읽는다. |
| 자체 36개 기준 | 기준값(nested AUC, 분할별 AUC)과 결합기 | 커밋된 값은 `docs/research/extended-stack-final-assembly/issue514/submission-record.json` `candidates.pool36_full`의 nested `0.9698828758140019`(`shrunk_rank_logit_logistic`, λ=1.0, MLflow `223055f44dc9427da588a141bc3b1ca3`)뿐이다. 분할별 AUC는 `src/pipeline/ensemble.py` 1758~1759행이 MLflow metric `auc_fold_<k>`로만 남기므로 로컬 MLflow에서 꺼내야 한다. `c_selected_shrunk_rank_logit_logistic`으로 36개 기준값을 잰 기록은 없다. 결합기를 314 기준과 맞출지, 그 경우 기준값을 어느 회차에서 만들지는 미정이다. |
| 자체 36개 기준 | 재현 구성원 12개의 동결 명세 | 아직 없다. 재현 실행이 끝난 뒤 최소 변경안 1의 동결 단계가 만든다. |
| 자체 36개 기준 | 고정 분할, 학습 자료 | `artifacts/folds.parquet`, `data/train.csv`(`RoundSpec` 기본 경로, `src/pipeline/round.py` 234~235행). 이 워크트리에는 `data/`가 없어 실행 절차대로 메인 체크아웃에 심볼릭 링크를 걸어야 한다. |
| 314 확장 기준 | 기준 팔 구성원 314개(자체 36 + 외부 278)의 순서와 OOF 해시 | `docs/research/extended-stack-pool-reassembly/issue513/precommit.json` `reassembled.members`(314개, `composition_sha256` `e3208ed9…`). `scripts/round_issue553_pilot.py` 49~66행이 이 목록을 그대로 `MemberSource`로 만든 선례다. |
| 314 확장 기준 | 외부 278개의 OOF 경로 | `docs/research/extended-stack-submission-2-manifest.json`(#457, SHA-256 `3d9a205c…`) `members[*].oof_path`. 경로는 `data/external/...` 아래 npy·parquet이며 저장소에 커밋하지 않는다. 판본 2 장부 `docs/research/external-member-ledger.json`(`scripts/judge_extended_stack.py` 75행)이 같은 경로의 원장이다. |
| 314 확장 기준 | 기준값(nested AUC, 분할별 AUC)과 결합기 | `docs/research/extended-stack-pool-reassembly/issue513/comparison.json` `reassembled.nested_auc` `0.9703843058098193`, `reassembled.fold_aucs`(5개), 결합기 `c_selected_shrunk_rank_logit_logistic`. 분할별 예측 해시는 같은 폴더 `reassembled/fold-<k>/reassembled.json`의 `auc`·`prediction_sha256`에 있다. ADR-0009와 파일럿 #553 기록(`docs/research/judgment-round-pilot/issue553/`)이 이 값을 비트 단위로 재현했다. |
| 314 확장 기준 | 재현 구성원 12개의 동결 명세 | 자체 36개 기준과 같은 명세를 공유한다. |
| 314 확장 기준 | 외부 후보 동결 명세와 외부 장부 색인 | 재현 판정에는 필요 없다. 외부 후보를 더하지 않으므로 `scripts/freeze_external_candidates.py`를 돌리지 않는다. 참고로 마지막 명세는 `docs/research/external-candidate-freeze/ecf-v3-b18bc301d500.json`(후보 24개, 조사 기준 2026-08-30T12:00:00Z, SHA-256 `8cf8ac53…`)이고 색인은 `docs/research/external-member-ledger-v3/index.json`(자격 있는 현행 기록 24개, SHA-256 `4a2a8af6…`)이다. |
| 두 기준 공통 | 코드 상태 | 커밋된 상태에서만 precommit할 수 있다(`src/pipeline/round.py` 434행). adapter와 스펙 스크립트를 커밋한 뒤 시작한다. |

## 근거 파일 목록

- `src/pipeline/ledger.py` 24행, 120행, 145행: `POOL_PATH` 기본값과 `Pool.load`·`Pool.save`의 경로 인자.
- `src/pipeline/ensemble.py` 1992~2045행: 등록 결합기 nested 평가 명령행과 고정된 `Pool.load()`, 1758~1759행: MLflow에 남기는 분할별 AUC.
- `src/pipeline/pool_judgment.py` 383~401행, 1401~1413행, 1547행, 1694~1727행: 풀 동결, 등록 판정 규칙, 라이브러리 경로 인자, 명령행 옵션.
- `src/pipeline/pool.py` 144~166행, 247행: 풀 진입 명령행과 고정 경로.
- `src/pipeline/judgment.py` 118행, 737~742행, 1299~1309행: 채택 문턱, 등록 권한 검증의 경로 인자, `StackGate`.
- `src/pipeline/member_sources.py` 1~13행, 31~68행, 71~93행, 96~109행: adapter 3종과 검증 수준.
- `src/pipeline/members.py` 39~42행, 70~126행, 236~310행: 검증 수준, `MemberSpec`·`MemberSource` 불변식, `load_members`.
- `src/pipeline/round.py` 94~105행, 124~162행, 174~201행, 222~265행, 287~313행, 434행, 465~466행, 527~560행, 1042행: 관련 소스 목록, 팔·기준값·자기 검사 타입, `RoundSpec`, 코드 상태, dirty 거부, `HASH_VERIFIED` 요구, 재개 검사, 명령행 진입점.
- `scripts/round_issue553_pilot.py` 36~66행, 101~134행: 이슈 513 기록에서 314 구성과 기준값을 `MemberSource`·`ReferenceValues`로 선언한 선례.
- `scripts/judge_strict_external_selection.py` 84행, 95~107행, 213~262행, 469~470행, 541행, 1656행, 1762~1815행: 풀 해시 봉인, 상수 고정, 자체·비교 팔 적재, 명령행.
- `scripts/freeze_external_candidates.py` 46~48행, 294~300행: 색인 입력과 명령행.
- `scripts/freeze_reusable_own_candidates.py` 53행, 565~600행, 786~790행: 풀 밖 자체 후보 동결의 현재 풀 제외와 시각 기준 선택, 명령행.
- `scripts/judge_issue526_ext327.py` 73~85행, 127~150행: 이슈 513 precommit을 314 팔의 원천으로 쓰는 또 다른 선례.
- `docs/adr/0006-strict-external-candidate-ladder.md` 27~34행: 비교 팔 313 고정과 동결 항목.
- `docs/adr/0009-judgment-round-contract.md`: 회차 id 규약, 자기 검사 등급, 게시 manifest, 파일럿 재현 결과.
- `docs/agents/strict-external-selection-run.md`: 시작 조건(`data/`·`mlruns/` 심볼릭 링크, 커밋 상태), "하지 않는 일".
- `docs/research/extended-stack-pool-reassembly/issue513/precommit.json`, `comparison.json`, `reassembled/fold-<k>/reassembled.json`: 314 구성 해시와 기준값.
- `docs/research/extended-stack-final-assembly/issue514/submission-record.json`: 36개 단독 nested 값과 풀 해시.
- `docs/research/extended-stack-submission-2-manifest.json`: 외부 278개 OOF 경로와 자체 35개 실행 식별자.
- `docs/research/reusable-own-candidate-freeze/rocf-v1-b42e02ea2e2b.json`: 자체 후보 동결 명세 형식의 본보기.
- `docs/research/external-candidate-freeze/ecf-v3-b18bc301d500.json`, `docs/research/external-member-ledger-v3/index.json`: 외부 후보 동결 명세와 장부 색인의 현재 값.
