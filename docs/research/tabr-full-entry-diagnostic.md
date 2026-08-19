# 전체판 TabR 스모크 게이트 진입 진단

이 문서는 GitHub 이슈 [#199](https://github.com/tmheo/predicting-smartphone-addiction/issues/199)의 구현과 Vast.ai 스모크·fold 0 진단 결과를 기록한다.

## 결론

전체판 TabR(공식 기본 설정, 조기 종료까지 학습)는 fold 0 seed 42에서 AUC `0.9441070604`를 기록했다.
사전 고정한 게이트 2(fold 0 단독 AUC 0.9660 이상)에 `-0.0219` 미달이므로 5-fold로 확장하지 않고 이 계열을 닫는다.
게이트 1(축소 표본 정상 수렴)과 게이트 3(풀 최근접 스피어만 0.96777 < 0.98)은 통과했다.
즉 TabR는 기존 풀과 상관이 낮은 새 정보 축이 맞지만, 단독 성능이 공개 스택 관측([#191](https://github.com/tmheo/predicting-smartphone-addiction/issues/191)의 tomasa2 하한)대로 기여 가능 수준(0.966대)에 크게 못 미친다.
champion과 후보 풀은 변경하지 않는다.

## #142와의 관계

- 이슈 [#142](https://github.com/tmheo/predicting-smartphone-addiction/issues/142)는 TabR-S(두 epoch, 첫 epoch 뒤 문맥 고정)를 fold 0에서 AUC `0.9419956232`로 측정하고 닫았다.
- #191의 공개 증분 조사는 "이 대회에서 한 번도 검토된 적 없는 계열"이라고 서술했으나, 이는 #142 선례를 놓친 것이다.
- 이 티켓의 측정은 다른 질문이다: "절단 측정이 아니라 공식 학습 방식(조기 종료 patience 16, 매 배치 문맥 재계산)으로 제대로 학습하면 어디까지 가는가."
- 답: 45 epoch 학습(최고 epoch 27)으로 `0.9420 → 0.9441`, `+0.0021`에 그친다. #142의 낮은 수치는 절단 탓이 아니라 이 자료에서 TabR 표현력의 한계다.

## 구현

`src/pipeline/tabr.py`에 공식 `bin/tabr.py`의 기본(default-evaluation) 구조를 이식했다.
기준 구현은 `yandex-research/tabular-dl-tabr`의 MIT 판본 커밋 `17baa9082506f8e7a0f8d11bb1e08212926a1507`이다.
TabR-S(#142의 `tabr_s.py`)와 같은 선형 encoder, 한 블록 predictor 구조이고, 차이는 학습 방식(문맥 고정 없음, 검증 AUC 조기 종료)이다.
후보 키 계산은 공식 `memory_efficient=True` 경로(후보 전체 키는 기울기 없이, 뽑힌 문맥 행만 기울기와 함께 재계산)를 따른다.
선택 항목으로 공식 튜닝판의 PLR(lite) 수치 임베딩을 지원한다.
`TabRAdapter`와 `MODEL_REGISTRY`의 `tabr` 항목으로 기존 실행 계약에 연결했고, 설정 `configs/exp123_tabr.yaml`은 champion exp081과 같은 33개 피처를 사용한다.

후보 저장소는 outer 학습 fold의 행과 라벨만 보유하고 학습 query는 자기 행을 문맥에서 제외한다.
`candidate_store_training_only`, `validation_labels_excluded_from_context`, `self_rows_excluded_from_candidates` assertion은 스모크와 fold 0 모두에서 통과했다.

## 고정 원격 환경

| 항목 | 값 |
| --- | --- |
| 공급자 | Vast.ai |
| GPU | NVIDIA RTX A5000 1장, 24,564MiB, driver 590.48.01 |
| 컨테이너 | `nvcr.io/nvidia/pytorch:26.01-py3` |
| Python | `3.12.3` |
| 의존성 | `uv 0.11.7` + `uv.lock` 고정(`faiss-gpu-cu12==1.14.1` 포함) |
| 검색 | exact `faiss.GpuIndexFlatL2` |
| 실행 커밋 | `759c273740900ad2dff8d5e611db7ffa52a0a6cd` |
| 인스턴스 | 48096471 (매물 39247976, machine 30333, $0.1589/h) |

## 게이트 1: 축소 표본 스모크 (통과)

fold 0 outer 학습 부분의 층화 20% 표본(110,619행)을 후보 저장소·학습 자료로, fold 0 검증 전체(138,274행)를 검증으로 사용했다. seed 42, patience 16, max 48 epoch.

| 변형 | 최고 검증 AUC | 최고 epoch | 종료 | epoch 평균 시간 |
| --- | ---: | ---: | --- | ---: |
| default(공식 기본 TabR) | 0.941462 | 14 | 조기 종료(32 epoch) | 8.2초 |
| plr_lite(PLR 임베딩 탐침) | 0.942695 | 13 | 조기 종료(31 epoch) | 23.6초 |

사전 등록한 선택 규칙(plr_lite 우위 +0.002 미만이면 default)에 따라 fold 0은 default 변형으로 실행했다.
plr_lite 우위는 +0.00123이었다.
전량 외삽 epoch 시간 120.5초로 `max_epochs = clamp(floor(12600/120.5), 8, 64) = 64`를 고정했다.

## fold 0 결과 (게이트 2 탈락)

기준 exp081 재현과 exp123 challenger를 같은 인스턴스에서 짝지어 실행했다.

| 실행 | fold 0 AUC | 비고 |
| --- | ---: | --- |
| exp081 기준 재현 | 0.9685268985 | 저장값 0.9684993008 대비 +0.0000276, 명시 허용 2e-4 안 |
| exp123 전체판 TabR | 0.9441070604 | 최고 epoch 27, 조기 종료 45 epoch, 학습 4,453초 |

- 게이트 2: `0.9441070604 < 0.9660` → 탈락(-0.0219).
- 공통 진단의 새 모델 계열 하한(champion - 0.01 = 0.9585)에도 미달해 진단 판정도 `stop`이다.
- epoch당 약 98초, PyTorch 최고 예약 메모리는 장치의 35.9%로 시간·메모리 한도는 통과했다.

## 게이트 3: 풀 최근접 상관 (통과, 참고)

fold 0 검증 예측과 풀 22구성원 OOF의 fold 0 조각의 스피어만 상관을 쟀다(`scripts/diagnose_tabr_pool_correlation.py`).
최근접은 `exp111_xgb_depth8_no_te`로 0.96777이며 0.98 미만이다.
상관은 낮지만 단독 성능이 기여 하한에 못 미쳐, "낮은 상관 + 단독 0.966 미만 → 기여 0"이라는 tomasa2 관측과 정합한다.

## 원격 실행과 비용

- 작업 A(스모크)와 작업 B2(기준+challenger)를 같은 인스턴스의 분리된 작업 루트로 실행했고, 입력·결과 묶음은 SSH 표준 스트림과 SHA-256 검증으로 전송했다.
- 작업 B(1차)는 exp081 학습 완료 후 `tracking.git_state()`가 git 저장소 부재로 실패해 산출물 없이 종료됐다.
  원인은 입력 묶음을 `git archive`로 만들어 `.git`이 없던 것이다.
  작업 B2는 커밋 해시를 보존한 얕은 git 클론을 올려 해결했다.
  교훈: `pipeline.entry_diagnostic` 원격 실행 묶음은 git 체크아웃이어야 한다.
- 그 외 실행 실수 두 건(작업 루트에서 실행해 상대 경로 미해결, PLR 탐침의 평가 배치 CUDA OOM)도 각각 재실행과 스크립트 보강으로 해결했고 원격 실행 장부에 기록했다.
- 인스턴스와 저장 공간은 결과 회수 후 삭제했고 계정 목록 부재를 확인했다. GitHub 보조 종료 예약에서 이 작업 항목을 제거했다.
- 이 인스턴스의 청구 합계는 약 `$0.686`이다(마감 정산 소액 추가 가능).

## 재현 자료

- 스모크 결과 묶음 SHA-256: `33a5a1d732756edc9d1e19ee8d55747eea7bea0a4d9d4c8ba3d6dd8d30f2c80d`
- fold 0 결과 묶음 SHA-256: `b8cb679a3a64326691eac11748ce9a85848a947318ab7bd297c311c358f0ccd2`
- 작업 A2 입력 묶음 SHA-256: `70e91ed2e122c521acd5fb211eb51371cd19619fc3eb8fdbfdf7308c19123f55`
- 작업 B2 입력 묶음 SHA-256: `46b57cc79447e71f8dd92bd7d999b76675cbb7d15a28b23eeb77ace0b714bfb6`
- 원자료는 기본 `main` 작업 폴더의 `run-logs/issue-199-tabr/`에 보존한다(진입 진단은 MLflow 실행을 만들지 않는다).
