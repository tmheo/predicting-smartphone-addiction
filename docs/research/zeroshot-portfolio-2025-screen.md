# AutoGluon zeroshot portfolio 2025 이식과 약식 검증

이슈 [#197](https://github.com/tmheo/predicting-smartphone-addiction/issues/197)의 실행 기록이다.
[#192 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/192)의 1순위 후보인 AutoGluon zeroshot portfolio 2025의 모델 설정을 기존 adapter로 이식해, 고정 fold 0에서 약식 검증했다.

## 출처와 라이선스

- 저장소: [autogluon/autogluon](https://github.com/autogluon/autogluon), Apache License 2.0.
- 파일: `tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2025.py`.
- 마지막 변경 커밋 `2d7e6056b8b64dc44114faf652d4c99ec3c3770f`(2026-01-08), 원본 파일 SHA-256 `e2ffbe42850c6aa8cbd5c30df84c77a91f7bfe679af8a1806d54c30775241b27`.
- 하이퍼파라미터 dict만 `scripts/screen_zeroshot_portfolio.py`로 옮겼고, 출처·변경 사실을 그 파일 머리에 명시했다.
- AutoGluon 기본값 설정 두 개(CAT `_default`, TABM `_default`)는 AutoGluon 소스의 기본값을 명시값으로 풀어 적었다(`catboost/hyperparameters/parameters.py`, `tabm/_tabm_internal.py`).

## 이식 범위

원본 포트폴리오는 19설정이고, 우선순위 순서는 원본의 `priority`다.

| 계열 | 설정 수 | 처리 |
| --- | ---: | --- |
| GBM(LightGBM) | 3 | 이식(`ag25_gbm_r33`, `_r21`, `_r11`) |
| CAT(CatBoost) | 5 | 이식(`ag25_cat_default`, `_r51`, `_r10`, `_r24`, `_r91`) |
| XGB | 2 | 이식(`ag25_xgb_r171`, `_r40`) |
| TABM | 6 | 이식(`ag25_tabm_r184`, `_r69`, `_r52`, `_default`, `_r191`, `_r49`) |
| REALTABPFN-V2 | 1 | 제외: adapter 없음. TabPFN 축은 zero-shot 구성원(exp067_tabpfn3)으로 이미 대표되고, fine-tuning 축은 #194에서 라이선스 사유로 닫았다 |
| TABICL | 1 | 제외: TabICLv2 진입 진단 fold 0 AUC 0.9417로 이미 탈락한 계열([진단 기록](tabiclv2-entry-diagnostic.md)) |
| MITRA | 1 | 제외: adapter 없는 foundation model 계열, fine-tuning 필요 |

이식하며 바꾼 것.

- `ag_args`(이름·우선순위 메타데이터)를 제거했다.
- XGB의 `enable_categorical`은 adapter가 소유하므로 설정에서 뺐다.
- TabM의 `amp`는 adapter가 지원하지 않아 뺐다(원본도 전부 `False`).
- TabM `batch_size: "auto"`는 AutoGluon의 표본 수 규칙(108,000행 이상 → 1024)으로 1024를 대입했다(학습 fold 553,095행).
- TabM `n_epochs`는 원본 무제한 대신 상한 200을 뒀다(patience 16이 먼저 멈춘다).
- `n_estimators`/`iterations` 10000과 early stopping 200은 저장소 규약을 따랐다.
- 원본 파일 머리의 주석대로 이 포트폴리오는 "표본 1만 개 이하, 특성 500개 이하, GPU 보유" 조건에 맞춰 학습된 소규모 자료용이다.
  우리 자료(69만 행)와 체제가 다르다는 사실을 알고 이식했고, 판정은 아래 약식 검증이 맡는다.

## 약식 검증 프로토콜

- 도구: `scripts/screen_zeroshot_portfolio.py`. `scripts/screen_pairs.py`(#48 규약)의 선례를 따른 선별 전용 실행으로, MLflow 실행을 만들지 않는다.
- 분할: 커밋된 `artifacts/folds.parquet`의 fold 0을 검증에, 나머지 4개 fold 전체 행을 학습에 쓴다. 시드 42.
- 피처 계획: champion 계열(exp065_tabm과 동일) 전 설정 공통. fold-fit 변환기는 설정과 무관하므로 한 번 학습해 전 설정이 공유한다.
- 기준 실행: 같은 프로토콜의 `base_lgb`(exp001 계열), `base_xgb`(exp045), `base_cat`(exp041), `base_tabm`(exp065, 초기화 평균만 1로 축소).
- 판정 앵커의 적용: 티켓의 "풀 하한(0.969대) 근접"을 fold 0 눈금으로 옮기면, 풀 0.968대 구성원들의 fold 0(seed 평균) 0.9673~0.9679와 champion fold 0 0.9686 사이다.
  통과 규칙은 "fold 0 AUC가 같은 계열 기준 실행 이상이고 0.9672(풀 0.968밴드 최저 구성원 exp045의 fold 0) 이상"으로 정했다.
- 실행 환경: GBDT 계열은 로컬 CPU(14코어), TabM 계열은 Vast.ai RTX A4000 2대(장부 `run-logs/vast-issue197/ledger.md`).

## 결과

fold 0, seed 42의 검증 AUC다.
원자료는 로컬 `run-logs/zeroshot_portfolio_screen.csv`(GBDT)와 `run-logs/vast-issue197/`(TabM 회수분)에 있다.
`유효`는 TE 카나리아(placebo_noise_te gain < placebo_noise gain) 판정이다.

### GBDT 계열 (로컬 CPU)

| 이름 | fold 0 AUC | 계열 기준 대비 Δ | best_iteration | 유효 | 통과 |
| --- | ---: | ---: | ---: | --- | --- |
| base_lgb (exp001 계열 기준) | 0.9668705 | - | 307 | True | - |
| base_xgb (exp045 기준) | 0.9673533 | - | 839 | True | - |
| base_cat (exp041 기준) | 0.9675828 | - | 2238 | True | - |
| ag25_xgb_r40 | 0.9678245 | +0.00047 | 8252 | True | 승격 |
| ag25_gbm_r21 | 0.9678059 | +0.00094 | 9991(상한) | True | 승격 |
| ag25_gbm_r33 | 0.9677621 | +0.00089 | 4586 | True | 승격 |
| ag25_cat_default | 0.9676937 | +0.00011 | 3325 | True | 계열 최고로 승격 |
| ag25_cat_r91 | 0.9676624 | +0.00008 | 6879 | True | 탈락(계열 내 차선) |
| ag25_cat_r10 | 0.9675425 | -0.00004 | 9999(상한) | True | 탈락 |
| ag25_cat_r24 | 0.9675400 | -0.00004 | 2562 | True | 탈락 |
| ag25_cat_r51 | 0.9673439 | -0.00024 | 9999(상한) | True | 탈락 |
| ag25_xgb_r171 | 0.9673047 | -0.00005 | 9999(상한) | True | 탈락 |
| ag25_gbm_r11 | 0.9628472 | -0.00402 | 9999(상한) | False | 탈락(절대 하한 미달) |

### TabM 계열 (Vast.ai RTX A4000 2대)

TabM 약식은 permutation importance를 생략해 카나리아 항목이 비어 있다(피처 계획은 정식 실행들이 이미 검증한 champion 계열 판이다).
원격 실행·정리 근거는 로컬 `run-logs/vast-issue197/ledger.md`에 있다(총 비용 약 $0.28).

| 이름 | fold 0 AUC | 계열 기준 대비 Δ | 통과 |
| --- | ---: | ---: | --- |
| base_tabm (exp065에서 fold 내 초기화 평균만 1로 축소) | 0.9675792 | - | - |
| ag25_tabm_r49 | 0.9676405 | +0.00006 | 탈락(아래 판정) |
| ag25_tabm_default | 0.9675237 | -0.00006 | 탈락 |
| ag25_tabm_r191 | 0.9674157 | -0.00016 | 탈락 |
| ag25_tabm_r52 | 0.9673018 | -0.00028 | 탈락 |
| ag25_tabm_r69 | 0.9672493 | -0.00033 | 탈락 |
| ag25_tabm_r184 | 0.9666267 | -0.00095 | 탈락 |

## 판정

통과 규칙: fold 0 AUC가 같은 계열 기준 실행 이상이고 절대 하한 0.9672 이상.
계열 기준 대비 +0.0002 미만의 개선은 계열 최고 하나만 승격한다(지도 190의 공격적 선별 기준).

- GBDT 승격 4건을 seed 42 5-fold 스크리닝 설정으로 만들었다:
  `exp117_ag25_gbm_r21`, `exp118_ag25_gbm_r33`, `exp119_ag25_xgb_r40`, `exp120_ag25_cat_default`.
- LightGBM 두 설정(+0.0009급)은 우리 장수 기본값(lr 0.05, num_leaves 255) 대비 낮은 학습률과
  강한 정칙화·부분표본 조합이 이 자료에서 크게 유리함을 보여 준다.
  fold 0 눈금에서 kitopl 특성 묶음 없이도 풀의 LightGBM 최고 구성원(exp074 fold 0 0.96776, 시드 평균)과 같은 수준이다.
- CatBoost 5설정은 전부 base_cat ±0.0002 안이라 포트폴리오의 이득이 이 계열에는 없다.
- 원본 포트폴리오가 1만 행 이하 자료에 최적화된 점을 감안하면, 69만 행 체제에서도
  낮은 학습률 축의 이득이 이전된 것이 주요 관찰이다.

- TabM 계열은 승격 없이 닫는다.
  계열 최고 ag25_tabm_r49의 개선 폭 +0.00006은 잡음 수준이고, 풀 재직 구성원 exp065의 원 구성(fold 내 3판 평균)의 fold 0 0.96772에도 못 미쳐 GPU 스크리닝을 열 근거가 없다.
  AutoGluon TabM 포트폴리오의 큰 블록(d_block 512~1024, batch 1024)은 이 자료에서 exp065의 작은 블록 깊은 구성(d_block 160 × 10)보다 일관되게 낮았다.

## seed 42 5-fold 스크리닝 결과 (승격 4건)

실행은 이 워크트리의 MLflow 저장소에 있다(반입 전).
비교용 풀 수치는 3시드 평균본 OOF다.

| 설정 | run_id | seed 42 OOF | 같은 계열 풀 최고 |
| --- | --- | ---: | --- |
| exp117_ag25_gbm_r21 | `b22fac86c6a64d819e474e8da86504f3` | 0.9686372328 | exp074 0.96840 (초과) |
| exp118_ag25_gbm_r33 | `1e61fd4ac13749eea20148c668e016e2` | 0.9684384262 | exp074 0.96840 (근소 초과) |
| exp119_ag25_xgb_r40 | `8cf5a3cc1dac46acb3326c7276d6abba` | 0.9684403694 | exp045 0.96794 (크게 초과) |
| exp120_ag25_cat_default | `8dcd46641a8d4d40bd0e197b542e3303` | 0.9683330157 | exp070 0.96858 (미달) |

### OOF 스피어만 상관 (참고: 신규는 seed 42 단일본, 풀은 3시드 평균본)

- exp117: 최근접 exp074 `0.99805` (중복 게이트 0.998 경계 위. 진입 시 성능 높은 쪽 유지 규칙으로 exp074 교체 가능성).
- exp118: 최근접 exp074 `0.99738`. 단 exp117과 `0.99887`이라 exp117이 들어가면 중복 탈락 대상.
- exp119: 최근접 exp074 `0.99745`. 단 exp117과 `0.99890`이라 같은 처지. exp117의 3시드 확정이 흔들릴 때의 예비.
- exp120: 최근접 exp070 `0.99915`로 중복 게이트 초과이며 성능도 낮아 진입 불가 전망. 3시드 확정을 열지 않는다.

## 다음 단계

1. exp117과 exp119의 3시드 확정 재검증(로컬 CPU, --stage confirm)을 이 워크트리에서 실행한다.
2. 실행 기록 묶음(pipeline.bundle export)을 만들어 본 체크아웃 반입 후, ADR 0001 계열 2의 진입 판정(pipeline.pool)을 수행한다.
   exp117이 exp074를 교체 진입하면 kitopl 자릿수 특성 축이 풀에서 사라지므로, 교체 판정 결과를 #187 재평가에 전달한다.
3. 보강된 풀의 nested 재평가와 결합 전략 재선정은 #187이 소유한다.
4. 중단 조건 판정: 2023년판 100설정 확장은 열지 않는다.
   이번 회차에서 이득이 확인된 축은 "낮은 학습률 + 강한 정칙화 GBDT" 하나였고, 그 축의 상위 설정 세 개가 서로 0.998 이상 상관이라 설정 수를 늘려도 다양성 이득이 없다.
