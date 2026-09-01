# 재사용 후보 자산 전체 분류 (이슈 #575)

이 문서는 wayfinder 지도 [#572](https://github.com/tmheo/predicting-smartphone-addiction/issues/572)의 research 티켓 [#575](https://github.com/tmheo/predicting-smartphone-addiction/issues/575)의 사실 조사 결과다.
S6E8 대회 종료 시점 저장소의 자산을 다음 영상 대회(RSNA 무릎, DICOM/MRI)로의 이관 가치 기준으로 세 부류로 나눈다.
이 분류는 사실 조사이고, 실제 추출 범위와 방식의 확정은 후속 grilling 티켓 #576과 #578의 몫이다.

조사 방법은 다음과 같다.
`src/pipeline`의 module 69개와 `src/remote_ops` 1개는 파일별 docstring, 내부 import 문, 최상위 정의를 스크립트로 추출해 읽었다.
scripts 74개는 이름과 docstring 첫 줄로 군집화했고 개별 정독은 하지 않았다.
docs/agents 11건, docs/adr 9건, 루트 CLAUDE.md(AGENTS.md), CONTEXT.md는 머리글과 핵심 절을 읽었다.
줄 수는 `wc -l` 기준이다.

## 부류 정의

- 부류 1 (범용): 다음 영상 대회에서 그대로 또는 거의 그대로 쓸 수 있다.
- 부류 2 (tabular 전용): 구조와 개념은 재사용 가능하나 구현이 tabular 자료 형태에 묶여 있다.
- 부류 3 (이번 대회 전용): S6E8 자료나 특정 이슈에 묶여 이관 가치가 없다.

## 요약

| 대상 | 부류 1 (범용) | 부류 2 (tabular 전용) | 부류 3 (대회 전용) |
|---|---|---|---|
| src/pipeline + src/remote_ops (70개) | 28 | 32 | 10 |
| scripts (74개) | 5 | 5 | 64 |
| docs/agents (11건) | 9 | 0 | 2 |
| docs/adr (9건) | 6 | 0 | 3 |

범용 코어의 중심은 판정·기록 계열이다: identity, sealed, members, judgment, round, ledger, runs, jobs, config, cpu_budget, bundle, tracking, observe, fold_observability와 Vast.ai 운영 절차 문서군.
범용으로 분류한 module이 tabular 전용 module을 import하는 역전이 4곳(직접)과 2계열(전이)에서 확인됐다.
상세는 마지막 절에 있다.

## 1. src/pipeline (69개) + src/remote_ops (1개)

### 부류 1: 범용 (28개, 약 10,900줄)

| module | 줄 수 | 근거 |
|---|---|---|
| identity.py | 50 | 예측 배열의 값 내용 기반 의미 해시로 저장 형식과 무관하고 자료 형태도 가리지 않는다 |
| sealed.py | 140 | canonical JSON 자기 해시 봉인 기록의 정본으로 어떤 판정에도 쓸 수 있다 |
| jobs.py | 238 | subprocess 작업 실행, 동시 상한, 재진입 탐지의 정본 구현으로 자료 형태와 무관하다 |
| cpu_budget.py | 72 | cgroup 할당량 기준 시드 병렬 워커 CPU 분배 규약으로 어느 머신에서나 유효하다 |
| config.py | 387 | YAML 설정 로딩과 "실험 하나 = 설정 하나 = run 하나" 구조는 범용이나, 스키마 필드(FeatureConfig 등)는 새 대회에서 재정의가 필요한 경계 항목이다 |
| ledger.py | 166 | champion.yaml과 pool.yaml을 git에 커밋하는 판정 장부 소유 타입으로 개념이 대회 독립적이다 |
| runs.py | 482 | 완료된 실험 실행의 사실·산출물을 읽는 유일한 통로(실행 저장소)로 MLflow 기반 범용 구조다 |
| members.py | 328 | 구성원 행렬을 해시 대조로 적재·검증하는 정본으로 예측 배열이면 형태를 가리지 않는다 |
| member_sources.py | 109 | manifest·동결 명세·풀 장부를 members에 넘기는 얇은 adapter 3종이다 |
| judgment.py | 1,417 | ADR-0001 판정 규칙(선별·확정 관문, 시드 AUC 눈금)의 단일 정본이나, 지표가 AUC로 굳어 있고 아래 역전 2건이 있다 |
| round.py | 1,097 | 판정 회차(봉인, 실행, 비교, 보고, 게시)의 정본으로 스크립트 5곳의 재구현을 대체했고 구조가 대회 독립적이나 아래 역전 1건이 있다 |
| compare.py | 269 | champion 대 challenger 개선 판정 CLI로 judgment의 얇은 caller다 |
| bundle.py | 600 | 로컬 밖 실행을 실행 저장소로 반입하는 실행 기록 묶음으로 원격·Kaggle 실행 반입 어디서나 필요하다 |
| tracking.py | 383 | 실험 기록 내용 규약(MLflow SQLite, git 상태, 입력 해시)으로 범용이나 PLACEBO 역전 1건이 있다 |
| observe.py | 451 | MLflow 실행 수명주기 소유와 선행 규약 집행 지점으로 자료 형태와 무관하다 |
| summary.py | 195 | 결과 요약(CSV, PNG, HTML) 생성기로 범용이나 PLACEBO 역전 1건이 있다 |
| cleanup.py | 113 | 스테일 실행 판정·정리로 실행 저장소만 안다 |
| run.py | 279 | 실행 진입점 골격은 범용이나 plan을 거쳐 features에 전이 의존한다 |
| private_inputs.py | 177 | 새 작업 폴더의 비커밋 입력 준비·검증 진입점으로 어떤 대회에도 그대로 쓴다 |
| seed_parallel.py | 216 | 시드 반복의 순차·GPU별 프로세스 병렬 실행 계약으로 범용이다 |
| seed_reuse.py | 250 | 검증된 단일 시드 실행의 계산 재사용 계약으로 범용이다 |
| fold_observability.py | 1,197 | fold 실행의 판본화 시간·자원 관측 원본으로 저장 위치를 모르는 설계라 범용이다 |
| recovery.py | 459 | 예측만 저장하는 fold 복구 경계로 모델 내부 상태를 저장하지 않아 형태 독립적이다 |
| training_length.py | 461 | 관측 학습 길이와 재학습 예산의 공통 계약으로 학습기 종류를 가리지 않는다 |
| muon.py | 125 | Muon 혼성 optimizer로 PyTorch 신경망이면 어디서나 쓴다(tabm_parameter_groups 함수 이름만 tabular 흔적) |
| pytest_openmp_guard.py | 89 | macOS OpenMP 충돌에서 torch 시험을 격리하는 pytest 플러그인으로 범용이다 |
| submit.py | 398 | Kaggle 제출과 public 점수 기록으로 Kaggle 대회면 그대로 쓴다 |
| remote_ops/vast_termination.py | 542 | GitHub Actions에서 도는 Vast.ai 독립 종료 안전장치로 완전 범용이다 |

### 부류 2: tabular 전용 (32개, 약 24,200줄)

| module | 줄 수 | 근거 |
|---|---|---|
| data.py | 60 | CSV 로딩, 범주 정렬, fold 부여로 구조는 단순하나 구현이 CSV·DataFrame에 묶인다(경계: DICOM 로더로 교체할 자리) |
| cv.py | 90 | CV 루프와 실행당 산출물 규약의 원천으로 루프 자체는 얇지만 plan을 통해 tabular에 묶인다 |
| cv_seed_execution.py | 621 | 한 시드의 CV 실행 소유자로 features·plan에 직접 의존한다 |
| features.py | 1,904 | 컬럼 제공자 구현(TE, CE, 결측 지시자 등)으로 tabular 특성 공학 그 자체다 |
| plan.py | 791 | 피처 계획의 선언·검증·조율 개념은 범용이나 provider 타입 체계가 DataFrame에 묶인다 |
| training_rows.py | 256 | 바깥쪽 분할의 학습 행 구성과 결정성 증거로 행 복제·증강이 tabular 전제다 |
| fold_fit_reuse.py | 922 | 내용 기반 fold-fit 재사용 저장소 개념은 범용이나 DataFrame 스키마·값 해시에 묶인다 |
| model.py | 2,557 | kind에서 adapter 팩토리로 가는 레지스트리 골격은 범용 후보이나 등록된 adapter와 계약이 tabular다 |
| ensemble.py | 2,123 | nested OOF 평가기와 결합 전략 계약은 OOF가 있는 대회 어디서나 유효하나 구현이 이진 AUC·rank logit에 묶인다 |
| initial_score.py | 685 | 잔차 부스팅의 행별 초기 로짓 생성기로 tabular 부스팅 전제다 |
| pool.py | 251 | 후보 풀 장부와 다양성 진입 CLI로 개념은 범용이나 판정 눈금이 AUC다 |
| pool_audit.py | 758 | 풀의 무결성·중복·다양성 감사로 구조는 재사용 가능하나 지표가 tabular 이진 분류다 |
| pool_judgment.py | 1,763 | candidate-pool-v2 판정 기록 생성 경로로 풀 계열 구조에 묶인다 |
| refit.py | 776 | 전체 자료 재학습과 최종 조립(ADR-0002)으로 규약은 범용이나 구현이 풀·plan에 묶인다 |
| refit_plan.py | 1,279 | 재학습 계획 장부(문법, 계보 검증, 예산 재계산)로 개념은 범용이나 training_state·풀 구조에 묶인다 |
| entry_diagnostic.py | 944 | 정식 실행과 같은 계획·adapter를 쓰는 진입 진단으로 개념은 범용이나 구현이 tabular 경로다 |
| catboost_hpo.py | 609 | CatBoost GPU fold 0 탐색으로 tabular 부스팅 전용이다 |
| xgb_hpo.py | 355 | XGBoost 설정값 탐색 진입점으로 tabular 부스팅 전용이다 |
| denoising_autoencoder.py | 569 | 분할 적합 비지도 잠재 표현 제공자로 tabular DAE 구현이다(풀 미등록으로 끝남) |
| training_state_contract.py | 324 | 한 학습 궤적의 여러 고정 시점 계약(ADR-0004)으로 개념은 신경망 학습 범용이나 frame 해시가 DataFrame에 묶인다(경계) |
| training_state_cv.py | 263 | 여러 고정 시점 후보를 만드는 전용 CV 경로로 plan에 묶인다 |
| training_state_manifest.py | 814 | 학습 시점 후보 child manifest의 단일 검증기로 개념은 범용이다(경계) |
| training_state_recovery.py | 1,041 | 사전 고정 시점 집합의 fold 복구 경계로 개념은 범용이다(경계) |
| training_state_run.py | 556 | 여러 학습 시점을 독립 후보 실행으로 게시하는 진입점이다 |
| tabm.py | 599 | TabM 학습기(pytabkit) fold 구현으로 다른 tabular 대회에서 재사용 가능하다 |
| tabpfn3.py | 204 | TabPFN-3 fold 구현으로 tabular 기반 모델이다 |
| tabiclv2.py | 310 | TabICLv2 공식 추론기 경계로 tabular 기반 모델이다 |
| tabr.py | 681 | 전체판 TabR fold 학습기(Yandex 공식 MIT 판 이식)로 tabular 검색 기반 모델이다 |
| tabr_s.py | 579 | TabR-S와 문맥 고정 fold 학습기로 tabular 모델이다 |
| trompt.py | 667 | TALENT 판 기준 Trompt fold 학습기로 tabular 모델이다 |
| xrfm_fold.py | 416 | xRFM 재귀 특성 커널 머신 경계로 tabular 모델이다 |
| amformer.py | 724 | AAAI 2024 AMFormer 논문 수식의 독립 구현으로 tabular 주의 모델이다 |

### 부류 3: 이번 대회 전용 (10개, 약 11,000줄)

| module | 줄 수 | 근거 |
|---|---|---|
| lookup_transformer.py | 1,553 | S6E8 공개 노트북(tamerlanomralinov Lookup-Transformer)의 재현으로 이 자료의 조회 어휘에 묶인다 |
| tab_cnn.py | 823 | S6E8 공개 노트북 표 합성곱망의 재현이다 |
| scalar_token_transformer.py | 730 | S6E8 공개 노트북 TabTransformer의 재현이다 |
| contextualized_spline_transformer.py | 1,152 | S6E8 공개 노트북 스플라인 Transformer의 재현이다 |
| realmlp.py | 1,753 | S6E8 노트북의 고정 4 epoch RealMLP fold 안전 재현으로 설정이 이 자료에 맞춰져 있다 |
| paired_training_length.py | 299 | 결측 증강 짝비교가 출처의 학습 노출량을 보존하는 계약(ADR-0007)으로 S6E8 결측 증강 서사에 묶인다 |
| missingness_propagation_batch.py | 968 | 결측 증강 전파 일괄 판정의 사전 고정 계약으로 이 대회의 합성 결측 구조 전용이다 |
| teacher_student_residual.py | 895 | 교사-학생 순위 잔차 보정 평가기로 이 대회의 실험 경로였고 채택되지 않았다 |
| teacher_student_residual_selection.py | 1,268 | 위 기법의 중첩 OOF 선택기로 같은 이유다 |
| pool_rereview.py | 1,582 | 2026-08-22 사전 고정 장부에 묶인 35개 풀 재심사 실행기다 |

## 2. scripts (74개)

이름과 docstring 첫 줄 기준의 군집 분류다.

| 군집 | 개수 | 부류 | 근거 |
|---|---|---|---|
| diagnose_* | 14 | 3 | 전부 특정 이슈(지문, 결측 구조, 폭 가설 등)의 일회성 진단이다 |
| judge_* | 12 | 3 | 이슈별 스택·짝 판정으로 ADR-0009에 따라 round.py가 골격을 흡수했다 |
| record_* (원격 제외) | 9 | 3 | 이슈별 판정·제출 기록 생성이다 |
| freeze_* | 8 | 3 | 이슈별 사전 동결 명세 생성으로 봉인 개념은 sealed.py에 이미 정본이 있다 |
| screen_* | 5 | 3 | 이슈별 fold 0 약식 선별이다 |
| assemble_* | 4 | 3 | 이슈별 최종 제출물 조립이다 |
| smoke_* | 4 | 2 | tabular 모델(TabPFN-3, TabR, Trompt, xRFM) 스모크 게이트로 모델과 함께 간다 |
| build_external_member_ledger*, analyze_*, measure_*, decompose_*, calibrate_*, estimate_*, reproduce_*, select_*, validate_*, judgment_golden, round_issue553_pilot, diagnose 외 기타 | 13 | 3 | 외부 장부, carry-over 추정, 재현, 골든 박제 등 전부 S6E8 이슈에 묶인다(단 public 보드 carry-over 사전 추정 기법 자체는 Kaggle 범용 아이디어다) |
| make_folds.py | 1 | 2 | 공유 fold를 한 번 생성해 커밋하는 규약은 범용이나 분할 로직이 tabular다 |
| record_remote_python.py, run_remote_python.sh, verify_remote_image_python.sh, verify_environment_gates.sh, preflight_fold_gpus.py | 5 | 1 | 원격 실행 기록·검증과 다중 GPU 사전 검사로 어느 대회에서나 그대로 쓴다 |

## 3. docs/agents (11건)

| 문서 | 부류 | 근거 |
|---|---|---|
| issue-tracker.md | 1 | GitHub 이슈 운영 규약으로 대회 독립적이다 |
| triage-labels.md | 1 | 5개 정본 triage 라벨 대응표로 대회 독립적이다 |
| domain.md | 1 | 단일 CONTEXT.md와 docs/adr 소비 규약으로 어느 저장소에나 이식 가능하다 |
| kaggle-public-notebook-licensing.md | 1 | Kaggle 공개 노트북 Apache 2.0 절차로 Kaggle 대회 범용이다 |
| remote-gpu-transfer.md | 1 | scp 차단 환경의 SSH 표준 스트림 전송과 SHA-256 검증 절차로 범용이다 |
| vast-resource-control.md | 1 | Vast.ai 자격 증명, 수명주기, 정리, 증거 규칙으로 범용이다 |
| vast-termination.md | 1 | GitHub Actions 종료 안전장치 운영 문서로 범용이다 |
| vast-control-acceptance-2026-08-15.md | 1 | Vast.ai 제어 합격 기준의 증거 기록으로 다음 대회의 수용 기준선이다 |
| vast-control-permission-revalidation-2026-08-16.md | 1 | 작업용 키 권한 재검증 증거 기록으로 위와 같다 |
| discussion-update.md | 3 | S6E8 디스커션 증분 반영 절차로 대상이 이 대회다(절차 골격은 새 대회용으로 복제 가능) |
| strict-external-selection-run.md | 3 | ADR-0006 사다리 판정 전용 실행 절차다 |

## 4. 루트 규약 문서와 ADR

### CLAUDE.md(AGENTS.md)

빠른 실험 반복 기본값, 시험 실행 정책, 원격 공급자 선택 규약은 범용이다.
Kaggle CPU 반입 조건과 GPU 공급자 정책 문단은 이 대회의 이슈(123, 126, 414) 결정이 박혀 있어 새 대회에서 재결정이 필요하다.
범용 골격과 대회 정책이 한 파일에 섞여 있으므로 이관 시 분리가 필요하다.

### CONTEXT.md

합성 생성 포렌식, 노트북 조사, 결측 증강 용어는 이번 대회 전용이다.
그러나 예측 신원, 구성원 행렬, 실행 저장소, 봉인 기록, 판정 회차 같은 판정·기록 계열 용어는 부류 1 module의 정의 원천이라 해당 절만 분리 이관할 가치가 있다.

### docs/adr (9건)

| ADR | 부류 | 근거 |
|---|---|---|
| 0001 실험 채택 판정 계약 | 1 | 특성·다양성·앙상블 세 계열의 채택 관문 구조는 대회 독립적이다(눈금 상수는 재조정 필요) |
| 0002 전체 자료 재학습 규약 | 1 | refit과 시험 예측 혼합 규약은 OOF 대회 범용이다(경계: 혼합 비율 근거는 이 대회 실측) |
| 0003 풀 재구축 평가 경계 | 1 | 고정 OOF 재선택과 신규 모델 선택의 누출 경계 분리는 범용 원칙이다 |
| 0004 학습 시점 후보 실행 | 1 | 한 궤적의 여러 시점을 독립 후보로 보존하는 결정은 신경망 학습 범용이다 |
| 0005 엄격 외부 선별 | 3 | superseded 상태이고 외부 OOF 라이브러리라는 S6E8 상황 전용이다 |
| 0006 엄격 외부 후보 사다리 | 3 | 313 구성원 기준의 사다리 판정 계약으로 이 대회의 스택에 묶인다 |
| 0007 복제 행 최적화 갱신 보존 | 3 | 결측 증강 행 복제 상황 전용 결정이다 |
| 0008 공통 module 기록 규약 | 1 | digest 호환, 통일 봉인 키, 구성원별 검증 수준 결정은 부류 1 코어의 설계 근거다 |
| 0009 판정 회차 계약 | 1 | round.py의 계약 원천으로 부류 1 코어의 설계 근거다 |

## 5. 의존 관계 역전 (범용 module이 tabular 전용 module을 import)

실제 import 문으로 확인한 결과다.

직접 역전 4곳:

- `judgment.py:106`이 `from .features import PLACEBO`로 tabular 특성 module의 상수를 가져온다.
- `judgment.py:100-105`가 `from .ensemble import CANDIDATE_POOL_CORE_COMBINER_NAMES, DEFAULT_COMBINER_NAMES, MISSINGNESS_TEST_PATH, rank_mean`으로 결합기 이름 상수와 rank_mean 함수를 가져온다.
- `round.py:59,67`이 `from . import ensemble`과 `from .ensemble import CombinerConvergenceError, evaluate_outer_fold`로 nested OOF 평가기를 직접 가져온다.
- `tracking.py:44`와 `summary.py:27`이 각각 `from .features import PLACEBO`를 가져온다.

전이 역전 2계열:

- bundle.py와 run.py는 cv.py를 거치고, cv.py는 plan.py를, plan.py는 features.py와 denoising_autoencoder.py를 import하므로, 범용 반입·진입점이 tabular 특성 구현에 전이 의존한다.
- members.py는 runs.py를 거쳐 data.py의 CSV 로딩에 닿는다(구성원 행렬 자체는 형태 독립이므로 약한 역전이다).

역전의 성격은 두 갈래다.
PLACEBO 계열은 상수 하나의 위치 문제라 상수를 범용 쪽으로 옮기면 끊어진다.
ensemble 계열(judgment, round)은 결합기 평가 로직에 대한 실질 의존이라, 추출 시 결합기 계약(Combiner protocol)을 범용 인터페이스로 남기고 구현만 tabular 쪽에 두는 분리가 필요하다.
이 분리 설계의 확정은 #576과 #578에서 다룬다.
