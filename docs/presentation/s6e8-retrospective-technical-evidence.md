이 문서는 발표 본문 화면에서 정확한 실행 기록과 원본 근거로 이동하는 단일 기술 근거 부록이다.
A부터 I까지 아홉 구획은 화면에서 생략한 기술 정의, 실행 식별자, 설정, 해시, 분할별 진단값, 라이선스 한계와 원본 링크를 맡는다.
상세 내용을 다시 복제하기보다 각 사실을 기계 판독 기록, 실행 산출물, 저장소 근거 문서와 종결 결정으로 추적할 수 있게 연결한다.

## A. 공식 결과와 자료 범위

### 공식 결과와 제출

| 항목 | 값 | 원본 |
|---|---:|---|
| 최종 순위 | 14위 | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/issues/547#issuecomment-5487179345) |
| Public 점수 | `0.97135` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| Private 점수 | `0.97109` | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/issues/547#issuecomment-5487179345) |
| Kaggle 제출 식별자 | `55907610` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |
| 전체 자료 재학습 실행 식별자 | `3279e114ef444cfeaff4232bc401d7b4` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |
| 실행 소스 커밋 | `43045c1fabce9f35ebf86c4edf7900a4253f30fe` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |

제출 파일은 `artifacts/submissions/issue514-extended314-own-full.csv`이며 SHA-256은 `cbb0419a8b34b54ed11ece481d5927da3d98f2aa574839756eb8e965d3ecceaf`다.
Public 점수와 Private 점수는 같은 제출을 서로 다른 시험 표본에서 채점한 값이므로 두 값의 차이를 개별 실험의 효과로 해석하지 않는다.
Kaggle 공식 순위표와 제출 목록은 2026-09-01에 다시 조회했다.
마지막 업로드인 327열 제출 `55920131`은 Private 점수 `0.97108`이므로 최종 14위 성적을 만든 제출이 아니다.
313열 C 선택판과 최종 314열 판의 Public 점수는 모두 `0.97135`이고 Private 점수도 표시 단위에서 모두 `0.97109`다.

### 자료 범위

| 항목 | 값 | 원본 |
|---|---:|---|
| 학습 행 | `691,369` | [첫 기준 실행 종결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/18#issuecomment-5239693077) |
| 시험 행 | `296,302` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 입력 | 식별자와 목표값을 제외한 12개 열 | [첫 기준 실행 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml) |
| 목표값 | 이진 열 `addicted_label` | [자료 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/data.py) |

| 입력 파일 | SHA-256 | 원본 |
|---|---|---|
| `data/train.csv` | `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c` | [Trompt 입력 검증 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/trompt-entry-diagnostic.md) |
| `data/test.csv` | `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e` | [Trompt 입력 검증 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/trompt-entry-diagnostic.md) |
| `artifacts/folds.parquet` | `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4` | [분할 파일](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/artifacts/folds.parquet) |

| 종류 | 입력 열 |
|---|---|
| 수치 9개 | `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time` |
| 범주 3개 | `gender`, `stress_level`, `academic_work_impact` |

화면 03의 위험도는 대회 목표값의 순서를 만들기 위한 예측이며 개인의 실제 중독 여부를 진단하는 임상 값이 아니다.

## B. 점수와 검증 경계

### 화면 04부터 06의 ROC AUC와 교육용 예시

이진 목표값의 ROC AUC는 임의로 고른 양성 한 행이 음성 한 행보다 높은 예측값을 받을 확률로 해석할 수 있다.
예측값이 같으면 그 쌍은 절반을 맞힌 것으로 계산한다.
실제 실행은 `src/pipeline/cv.py`의 `score_predictions`에서 scikit-learn `roc_auc_score`로 전체 OOF를 재채점한다.

| 중독-비중독 쌍 | A 순서 | B 순서 |
|---|---|---|
| 가와 나 | 올바름 | 올바름 |
| 가와 라 | 올바름 | 올바름 |
| 다와 나 | 틀림 | 올바름 |
| 다와 라 | 올바름 | 올바름 |
| 합계 | `3/4` | `4/4` |

이 표는 설명을 위해 만든 네 사람 예시이며 실제 대회 행이나 실제 모델 점수가 아니다.
동점이 없는 이 예시만 놓고 보면 A의 쌍 순서 비율은 `0.75`, B는 `1.0`이다.

- 점수 구현: [`src/pipeline/cv.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/cv.py)
- 설명 장면 결정: [비전문가용 핵심 개념 설명 장면](https://github.com/tmheo/predicting-smartphone-addiction/issues/570#issuecomment-5488821634)
- 용어 결정: [비전문가 발표의 기술 용어와 표기 원칙](https://github.com/tmheo/predicting-smartphone-addiction/issues/579#issuecomment-5489318781)

### 화면 08에서 처음 설명하는 OOF AUC

고정 분할은 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`로 한 번 만들고 커밋한 `artifacts/folds.parquet`이다.
파일 SHA-256은 `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4`다.
고정한 다섯 fold를 차례로 검증 부분으로 사용한다.
각 학습 행은 자기 목표값으로 학습하지 않은 모델에서 예측 하나를 받고, 다섯 예측 조각을 원래 행 순서로 이어 OOF를 만든다.
OOF AUC는 이 OOF 예측 전체를 ROC AUC로 채점한 값이다.

| 점수 | 입력과 선택 경계 | 해석 한계 |
|---|---|---|
| 일반 OOF AUC | 한 구성의 고정 5분할 검증 예측을 원래 행 순서로 이어 채점 | 실험 선택 이력 전체의 편향을 제거하지 않음 |
| nested OOF AUC | 바깥 fold를 봉인하고 나머지 fold의 OOF에서 결합기와 내부 설정을 고른 뒤 봉인 fold를 채점 | 기초 구성 생성 이전의 모든 선택까지 다시 수행한 완전 중첩 평가는 아님 |
| Public AUC | 시험 자료의 공개 채점 부분을 Kaggle이 채점 | 후보 선택이나 결합기 선택에 사용하지 않음 |
| Private AUC | 대회 종료 뒤 최종 채점 부분을 Kaggle이 채점 | 개별 기법의 인과 효과를 증명하지 않음 |

- 분할 생성: [`scripts/make_folds.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/scripts/make_folds.py)
- OOF 생성과 채점: [`src/pipeline/cv.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/cv.py)
- 판정 계약: [`docs/adr/0001-experiment-adoption-contract.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)

### 화면 12의 nested OOF 경계

결합 입력은 구성원별 float64 OOF 열, 고정 fold와 목표값이며 세 입력은 같은 식별자 인덱스와 같은 행 순서를 가져야 한다.
바깥 fold 하나를 봉인하고 나머지 네 fold의 OOF에서 결합기와 내부 설정을 고른다.
선택한 결합기를 봉인한 fold에 적용해 예측을 만들고, 다섯 바깥 fold의 예측을 원래 행 순서로 이어 nested OOF를 만든다.
최종 시험 예측은 판정이 끝난 뒤 전체 OOF와 전체 목표값에 결합기를 한 번 다시 맞추고, 같은 열 순서의 시험 예측 행렬에 적용한다.
외부 구성원 상류의 OOF 생성 절차가 모두 공개된 것은 아니므로 이 중첩 경계가 외부 예측 생성 단계의 모든 선택 편향까지 제거하지는 않는다.

- 계약: [`docs/adr/0001-experiment-adoption-contract.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)
- 바깥 fold 평가와 전체 조립 구현: [`src/pipeline/ensemble.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)
- 분할 생성: [`scripts/make_folds.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/scripts/make_folds.py)

## C. 같은 값을 표현한 실험

화면 07부터 09는 같은 수치 열을 표현하는 방식을 바꾼 사전 고정 직접 비교다.
세 실행은 같은 자료 분할, 같은 LightGBM 주요 설정과 난수 42를 사용했고 표현 방식만 바꿨다.

| 표현 | 실행 식별자 | 일반 OOF AUC | 비교 기준과 차이 | 판정 |
|---|---|---:|---:|---|
| 수치 유지 | `ce66e16b12fd43b4bc95fdcf2972555c` | `0.962759` | 비교 기준 | 기준 |
| 12개 입력을 전부 범주로 처리 | `73d5cac128eb4b429c348aefdc42dc86` | `0.958586` | 화면 표시 `-0.00417` | 중단 |
| 수치 유지와 정확값 범주 복제 병행 | `77217687c0514dab9f693fd4aa50c741` | `0.966046` | 화면 표시 `+0.00329` | 채택 |

전부 범주로 처리한 실행은 숫자의 정확값 묶기 신호를 얻는 대신 수치의 순서 정보를 버렸다.
수치 유지와 범주 복제를 병행한 실행은 원래 수치 아홉 개를 남기고 같은 값의 범주 복제 아홉 개를 추가했다.
세 실행에는 같은 플라시보 피처가 포함되어 있어 이 피처는 비교군 사이의 차이가 아니다.

범주 복제 대상은 `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`이다.

| 설정 | 고정한 차이 | 설정 파일 SHA-256 |
|---|---|---|
| `exp001_lgbm_baseline` | 수치 9개 유지, 범주 3개, 플라시보 포함 | `f3d02faab85b527ac2edfb182c733cad573b0ddd22e714c8989f8f0420775113` |
| `exp002_all_categorical` | 원시 입력 12개를 모두 범주로 처리 | `029d4eb7c018e4a4ec737bad8439f5822ed9222054788df73c75137ff0a63074` |
| `exp003_categorical_copies` | 수치 9개 유지, 같은 값의 범주 복제 9개 추가 | `73eaa6fc4dde479c4f8c951ad5f47cfb94667476b432ee3403d0cba9e2cb2def` |

세 설정은 LightGBM의 `objective=binary`, `metric=auc`, `learning_rate=0.05`, `num_leaves=255`, `n_estimators=10000`, `early_stopping_rounds=200`, 난수 42를 공유한다.
공유 분할 SHA-256은 B 구획에 기록한 값과 같다.

- 비교 기준 설정: [`configs/exp001_lgbm_baseline.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml)
- 전부 범주화 설정: [`configs/exp002_all_categorical.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp002_all_categorical.yaml)
- 수치 유지와 범주 복제 설정: [`configs/exp003_categorical_copies.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp003_categorical_copies.yaml)
- 원 판정: [전 피처 범주형 challenger 실험: 실행과 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)

## D. 결측 증강 판정

### 증강과 상태 계산 경계

결측 증강판은 바깥 학습 fold의 원본 행에 관측 셀을 확률 `0.25`로 독립 결측 처리한 복제본 두 개를 더한 세 배 학습 행 구성이다.
기존 결측은 그대로 유지하고 복제본은 부모 행의 fold 관계와 목표값을 물려받는다.
피처 제공자와 모형 내부 전처리 상태는 원본 부모 행에서만 맞춘 뒤 복제본에 적용하므로 복제본이 상태 계산 범위를 넓히지 않는다.
짝비교의 학습 노출량은 원본과 증강판이 같은 난수, fold, 부모 행 순서와 학습률 일정 위치에서 끝나도록 고정했다.

- 대표 설정: [`configs/missingness-propagation/07_exp035_lattice_te_missingness_augmented.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/missingness-propagation/07_exp035_lattice_te_missingness_augmented.yaml)
- 학습 길이 근거: [`artifacts/issue510-paired-training-lengths.json`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/artifacts/issue510-paired-training-lengths.json)
- 상태 경계와 용어: [`CONTEXT.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/CONTEXT.md)

### 후보 동결과 정확 검색

| 입력 상태 | 값 |
|---|---:|
| 짝 전체 | 34개 |
| 완결 짝 | 24개 |
| 미완결 짝 | 10개 |
| 허용한 교정 실행 | 7개 |
| 정확히 채점한 고유 상태 | 1,658개 |
| 결과를 본 뒤 고른 부분 결과 | 없음 |
| 판정에 사용한 Public 점수 | 없음 |

정확 검색은 풀 전체 중복 불변식을 지키면서 원본 자리와 결측 증강판을 상호 배타인 한 자리로 취급했다.
직접 짝비교의 부호는 후보의 검색 입력 포함이나 최종 교체를 단독으로 결정하지 않았다.

| 선택 번호 | 원본 자리 | 선택한 결측 증강판 | 전체 자료 재학습 시드별 예산 | 구성원 항목 SHA-256 |
|---:|---|---|---|---|
| 7 | `exp035_lattice_te` | `mpv1_exp035_lattice_te_missingness_augmented` | `42:258`, `43:239`, `44:255` | `ed2647ad001feea26bb444b7c894bda3af2b3c6422d21a0b4795c411f0737c7c` |
| 8 | `exp058_logreg_onehot` | `mpv1_exp058_logreg_onehot_missingness_augmented` | `42:null` | `d14085e599351ee1fe8b77506a10f8db0f207a375b7d40a538aa48b1efc03529` |
| 10 | `exp070_cat_exact_cats` | `mpv1_exp070_cat_exact_cats_missingness_augmented` | `42:5021`, `43:4960`, `44:5268` | `6b700439156830b3996a72276360bb4c2fd6232235072a5271c9ed040b6c8eb5` |
| 12 | `exp110_lgb_kitopl_no_te` | `mpv1_exp110_lgb_kitopl_no_te_missingness_augmented` | `42:5774`, `43:5655`, `44:6043` | `bdee5c036a6debf8d325ab007f015810d37c8d8f3bf3aa1427b80f153baa4755` |
| 22 | `exp131_lookup_bivariate_plr5` | `mpv1_exp131_lookup_bivariate_plr5_missingness_augmented` | `42:15`, `43:15`, `44:15` | `7a6f2ef2c279aeed614c397d74fae9ae329412d291b758ff5f341bbf0be4451a` |

### 두 관문과 분할별 값

| 관문 | 현재 풀 | 제안 풀 | 차이 | 결과 |
|---|---:|---:|---:|---|
| 동결 OOF 조건부 절차 | 별도 기록 | 별도 기록 | `+0.00004415298240634247` | 통과 |
| 최선 방식끼리의 직접 nested OOF | `0.9698359892003905` | `0.9698828758140019` | `+0.00004688661361140767` | 통과, 바깥 fold `5/5` 양수 |

두 풀의 직접 비교에서 선택된 결합 방식은 모두 `shrunk_rank_logit_logistic`이다.

| 바깥 fold | 현재 풀 AUC | 제안 풀 AUC | 차이 |
|---:|---:|---:|---:|
| 0 | `0.9692733377379816` | `0.9693299877411192` | `+0.00005665000313759272` |
| 1 | `0.9699449758700680` | `0.9699927220915519` | `+0.00004774622148384644` |
| 2 | `0.9699337513153030` | `0.9699737362544572` | `+0.00003998493915413626` |
| 3 | `0.9704320162532365` | `0.9704601210261992` | `+0.00002810477296266711` |
| 4 | `0.9695958623774630` | `0.9696578096623132` | `+0.00006194728485020562` |

### 변경 불가 기록

| 기록 | 내용 SHA-256 |
|---|---|
| 입력 묶음 | `54a1fab1b1408791a04b5c415b9c2c4edff9d972eb6c7571ee5671ff828e280d` |
| 도달 가능성 기록 | `aae096fa319f8c43d90a225d4e3effa50ad19c2f4f9d7e908caf0350bc732ad6` |
| 정확 검색 | `e4bf476f51a7e105a1e809633460d8d6ff32c3bdf81c8c2ab745fd89735b5021` |
| 조건부 절차 관문 | `87d5d0aa900a05ded721d3d27b22cb728266d83e876504b28eae8b1c5e926e89` |
| 직접 중첩 관문 | `51f93addc0d35d34e9c96fbc4e34d426f3bd5ecd114a13ae5870f9b8eca9af12` |
| 선택 근거 | `a1e09244d1f8c814ec94d48c9f10551fe6906ba5c6fc9ec949c01f2bf2db8b28` |
| 재학습 준비 상태 | `e1fb052594d0a25479838025990a750380d72a4575c8bfdfd7bfc7bd18dc61a8` |
| 최종 판정 | `eb69ddf211dbd3c9242fb6e4ea63349c7c5173af2c421cb814c29621635b430a` |

판정은 깨끗한 소스 커밋 `23cf8b2119060a9fffc4db659998ef135a5f475f`에서 실행했다.
이 회차는 재학습 계획의 정적 준비 상태만 확인했으며 실제 전체 자료 재학습과 시험 예측 생성은 H 구획의 최종 생산 단계에서 수행했다.

- 판정 보고: [`docs/research/missingness-propagation-batch/issue512/report.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/report.md)
- 기계 판독 요약: [`judgment.json`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/judgment.json)
- 분할별 직접 중첩 결과: [`direct-nested-gate.json`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/direct-nested-gate.json)
- 원 판정: [결측 증강 전파 일괄 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/512#issuecomment-5472767484)

## E. 실험 실행 체계

| 실행 장소 | 이 회고에서 맡은 역할 | 정식 판정에 들어오는 조건 |
|---|---|---|
| 로컬 | 개발, 소규모 실행, 반입, 재채점, 판정, 최종 조립 | 원본 실행 또는 검증된 실행 기록 묶음 |
| Kaggle CPU | 고정한 CPU 비교 짝의 병렬 실행 | 같은 공급자와 실행 환경 등급의 두 비교군 완결 |
| Kaggle GPU | 초반 정식 실행, 후반 호환성 확인과 진단 | 정식 판정 범위에 맞는 실행만 사용 |
| Vast.ai | 주 GPU 실행 장소 | 해시 대조, 원본 상태와 입력 경계 확인, 로컬 재채점 통과 |
| Runpod | Vast.ai 전환 조건을 충족할 때 쓰는 예비 GPU 실행 장소 | Vast.ai와 같은 반입 및 재채점 계약 통과 |

### 공통 실행과 중앙 검수

1. 결과를 보기 전에 설정, 실행 단계, 난수, 분할, 소스 커밋, 입력 파일과 의존성 잠금 내용을 고정한다.
2. 어느 환경이든 별도 학습 반복문을 만들지 않고 같은 `pipeline.run <config> --stage <screen|confirm>`을 실행한다.
3. 실행 기록, 예측, 지표와 진단을 실행 저장소에 남긴다.
4. 원격 실행은 manifest를 포함한 ZIP 실행 기록 묶음을 내보낸다.
5. 원격과 로컬에서 묶음 SHA-256이 같은지 확인한다.
6. 로컬 반입은 입력 해시, 커밋 존재, 커밋 시점 설정과 묶음 설정의 일치, 깨끗한 코드 상태를 검사한다.
7. 난수별 OOF 평균과 주장 지표를 로컬 입력으로 다시 채점하고 통과한 실행만 정상 실행으로 재생한다.
8. 판정 뒤 원격 계산 자원과 저장 공간을 삭제하고 과금이 멈췄는지 다시 확인한다.

한 비교 짝의 대조군과 후보군은 같은 공급자와 같은 실행 환경 등급에서 완결해야 한다.
서로 다른 공급자에서 끝낸 두 팔이나 실패 뒤 남은 일부 결과를 이어 붙이지 않는다.

### 실제 전환과 비용 범위

| 사건 | 확인된 값 | 판단 범위 |
|---|---:|---|
| Runpod 초기 선별 | 모형 실행 26분 24초, `$0.24` | 초기 공급자 실제 비교 |
| Vast.ai 초기 선별 | 모형 실행 31분 45초, `$0.12` | 초기 공급자 실제 비교 |
| Vast.ai 실패 뒤 Runpod 전환 | Vast.ai `$0.089`, Runpod `$3.91` | 정해 둔 예비 환경 전환이 실제로 작동 |
| Kaggle CPU 5개와 Vast.ai CPU 13개 | Vast.ai `$1.801`, 18개 실행과 270개 진단 반입 통과 | 비교 짝을 쪼개지 않은 CPU 병렬 처리 |
| 최종 신경망 전체 자료 재학습 | Vast.ai GPU 3장, `$0.393844836990070` | 마지막에 바뀐 신경망 한 구성의 세 난수 작업만 포함 |

초기 결정은 Runpod 우선이었고 실제 비용, 재고와 운영 경험이 쌓인 뒤 Vast.ai 우선으로 바뀌었다.
최종 전환 조건은 적합한 Vast.ai 매물 확보 실패, 서로 다른 두 호스트의 SSH 및 사전 검사 실패, 공급 환경 장애 또는 독립 종료 설정 불가였다.
프로그램 오류, 자료 오류, 설정 불일치와 GPU 메모리 부족은 공급자 전환 사유가 아니었다.

최종 전체 자료 재학습의 첫 입력 묶음은 macOS Python 3.13의 `pathlib._local`이 들어간 pickle 때문에 원격 Python 3.12에서 학습 전에 실패했다.
실패 결과 압축 파일 SHA-256은 `a4e3f5b7ba8bac10953d7217c43137973899062d704908b679258932dbd9da81`다.
경로를 문자열로 바꾼 두 번째 결과 압축 파일 SHA-256은 `a574cf92a7452f3a7f9b3bb297b0c8806d26b928afd5396f0d93ef6183c3625c`이고 원격과 로컬 값이 일치했다.
결과 회수와 로컬 검수 뒤 활성 Vast.ai 인스턴스와 별도 저장 공간은 모두 0개였다.

- 역할과 전환 근거: [`docs/research/presentation-environment-evidence.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md)
- 실행 기록 묶음 반입 구현: [`src/pipeline/bundle.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/bundle.py)
- Kaggle 실행 절차: [`docs/kaggle-gpu-run.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/kaggle-gpu-run.md)
- 원격 파일 전달: [`docs/agents/remote-gpu-transfer.md`](https://github.com/tmheo/kagglekit/blob/main/docs/agents/remote-gpu-transfer.md)
- Vast.ai 자원 제어: [`docs/agents/vast-resource-control.md`](https://github.com/tmheo/kagglekit/blob/main/docs/agents/vast-resource-control.md)

## F. 성공과 중단 사례

| 사례 | 점추정 또는 진입 결과 | 반복 근거 | 사전 관문 | 결론 |
|---|---|---|---|---|
| RealMLP 자료형 정합 복원 | `+0.0046091491` | 난수 42, 43, 44 | 같은 조건 짝비교 | 채택 |
| Lookup-Transformer 설정 17개 | 모두 기준 미달 | fold 0, 난수 42 | 진입 진단 | 중단 |
| 새 신경망 네 종류 | champion 대비 `-0.02653`부터 `-0.24351` | fold 0, 난수 42 | 진입 하한 `-0.01` | 모두 중단 |
| 약한 외부 예측 120개 계열 | 한계 기여 `-0.000057` | 전체 결합 판정 | 양의 기여 | 미채택 |
| 327열 결합 | `+0.0000046619547824` | 바깥 fold 3/5 양수 | 사전 교체 문턱 | 미채택 |

### RealMLP 자료형 정합 복원

- 수정판 실행 식별자: `c41c6a4deae04e1fbd8a75193eaaa32c`
- 결함판 출처 실행 식별자: `dbe1f8cccca4458889265eb0d0f45273`
- 수정 설정: `configs/exp124_realmlp_dtype_fix.yaml`, SHA-256 `3e0fa92985c3ebce4bd67129ccf83c96ca287fd1aa29a14b05610ba5a7eed4e9`
- 미등록값: `800,896`에서 `23`
- 3시드 평균 OOF AUC: `0.9637131967`에서 `0.9683223458`
- 수정판 시드별 OOF AUC: `0.9682564`, `0.9682484`, `0.9682668`
- 원 판정: [자료형 정합 복원 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/243#issuecomment-5343200265)

### Lookup-Transformer 설정 17개

기준은 `exp067_lookup_xgb_impute_comps5`의 fold 0, 난수 42 AUC `0.968294911389327`이다.

| 설정 축 | 구성 | 기준 대비 범위 | 결과 |
|---|---|---:|---|
| 최고 학습률 | `exp087`부터 `exp090`, `1e-3`, `1.5e-3`, `3e-3`, `4e-3` | `-0.0000447742`부터 `-0.0002923532` | 기존 `2e-3` 유지 |
| 학습률 일정 | `exp091`부터 `exp095`, 고정 momentum OneCycle, cosine, 선형, 상수, 검증 기반 감소 | `-0.0000074166`부터 `-0.0000437297` | 기존 OneCycle 유지 |
| 최적화 알고리즘 | `exp096`부터 `exp103`, AdamW, RAdam, NAdam과 세 학습률 | `-0.0000356111`부터 `-0.0018451317` | 기존 AdamW 유지 |

모든 설정 파일은 [`configs/`](https://github.com/tmheo/predicting-smartphone-addiction/tree/main/configs) 아래 `exp087`부터 `exp103`까지 고정돼 있다.
전체 5분할과 세 난수로 승격한 후보는 없었다.
원 판정은 [Lookup-Transformer 제한 탐색 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/160#issuecomment-5308772959)에 각 설정의 정확한 fold 0 AUC와 진단값을 기록한다.

### 새 신경망 네 종류

| 계열 | 설정 | fold 0 AUC | champion 대비 | 추가 중단 조건 | 핵심 결과 SHA-256 | 원본 |
|---|---|---:|---:|---|---|---|
| TabR-S | `exp082_tabr_s` | `0.9419956232` | `-0.0265329259` | 없음 | 진단 JSON `7d2f109158d1926a61abc90cfd12e9e24673b7e9f51c176e96120a3fed3c8d4a` | [진단 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/tabr-s-entry-diagnostic.md) |
| TabICLv2 | `exp083_tabiclv2_probe` | `0.9417453063` | `-0.0267832428` | 없음 | `8640ab509b12d082bf463991fb77dd4bf8b2d0c4a3be2d2de54be2f1c6d0434a` | [진단 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/tabiclv2-entry-diagnostic.md) |
| AMFormer | `exp085_amformer` | `0.7250206215` | `-0.2435079275` | 없음 | `b62899e5db6a45340df4ee63a9621acebacbb22c4ca9e64e5a67a88775b6038a` | [진단 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/amformer-entry-diagnostic.md) |
| Trompt | `exp086_trompt` | `0.9401445165` | `-0.0283840326` | 5분할 예상 `39.73`시간이 `24`시간 한도 초과 | `413f45e5a28e1c5b4ba7630ac111689972a3001a600f661c04cb18744d48dd1c` | [진단 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/trompt-entry-diagnostic.md) |

네 진입 진단은 같은 fold 0, 난수 42 champion AUC `0.9685285491`과 승격 하한 `0.9585285491`을 사용했다.
설정 파일 SHA-256은 TabR-S `a8553bb03d2b610fd1f2dddbbd5acd91470c3ea45c882b0cd4581c1cac414050`, TabICLv2 `e02a7f23173cc826c2f134b9e6854242279880c8378fbb4aedd89cf6cf4d1ca5`, AMFormer `deb768fb068a61108e350f4134288897846aa3117a7aa5bec2e279193f5c68ba`, Trompt `7aaea8039520e43b06ef2646d1dc8a7cbf6fb557309196f0c8c4cf602d7e1baf`다.
모두 행 수, 순서, 유한성 및 학습 자료 경계 검사를 통과했지만 성능 하한을 통과하지 못해 5분할과 세 난수 확정 실행으로 넓히지 않았다.

### 외부 120열과 327열 미채택

약한 고전 확률 모형 120열을 포함한 433열 결합은 242열 기준보다 `+0.0000063`, 바깥 fold `3/5` 양수에 그쳤다.
120열을 제외한 313열 결합은 같은 기준보다 `+0.0000633`, 바깥 fold `5/5` 양수였으므로 120열 계열의 한계 기여는 `-0.000057`이다.

327열 결합은 314열 기준 `0.9703843058098193`보다 `+0.0000046619547824` 높은 `0.9703889677646016`이었지만 요구 차이 `+0.00002`보다 작았다.
바깥 fold도 3개만 양수였고 예측 배열 SHA-256은 `c4d13d390371a261b0e41a96521a27667d4927fc6b1eb6c2a6c2f487e65b4264`다.

- 약한 외부 예측 120개 계열: [`docs/research/extended-stack-ladder-2.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ladder-2.md)
- 327열 결합: [`docs/research/extended-stack-ext327/issue526/comparison.json`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ext327/issue526/comparison.json)

## G. 서로 다른 오차와 Lookup-Transformer

### 작동 원리와 실행 설정

Lookup-Transformer는 각 열의 정확한 값을 학습 fold 전용 어휘에서 조회하는 표현과, 수치의 부드러운 추세를 나타내는 PLR 표현을 Transformer에서 함께 섞는다.
검증과 시험에만 있는 정확값은 결측 식별자와 다른 미등록 식별자로 보낸다.
이 설명은 나무 계열과 다른 예측 표현을 사용했다는 작동 원리이며, 특정 행이 왜 틀렸는지에 대한 인과 설명은 아니다.

| 항목 | 값 |
|---|---|
| 설정 | `configs/exp059_lookup_transformer.yaml` |
| 설정 파일 SHA-256 | `d8999d9512f798027fcfaa9cb3ef042d3d8766a525556ac2e25ad99a9a766503` |
| 실행 식별자 | `b951fac51b6b44298f7fdb0b543caba6` |
| 구현 커밋 | `dbeb5fe` |
| 검증 | 고정 5분할, 난수 42, 43, 44 평균 |
| 주요 구조 | 폭 128, 4층, 8개 머리, 정확값 조회와 PLR 표현 |
| 학습 주요값 | 32 epoch, batch 2,048, AdamW, OneCycle, 최고 학습률 `2e-3`, EMA와 값 dropout |
| 어휘와 분위 적합 범위 | 각 바깥 학습 fold만 사용 |

- 전체 설정: [`configs/exp059_lookup_transformer.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp059_lookup_transformer.yaml)
- 구현: [`src/pipeline/lookup_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/lookup_transformer.py)

### 혼자 잘하는가

| 측정값 | 값 |
|---|---:|
| 3시드 평균 일반 OOF AUC | `0.96892` |
| 당시 champion 일반 OOF AUC | `0.96854` |
| 차이 | `+0.00038` |
| 시드별 개선 | `+0.00019`, `+0.00022`, `+0.00019` |
| 양수 fold | `5/5` |

### 함께할 때 돕는가

두 예측의 스피어만 순위 상관은 같은 행의 위험 순서가 얼마나 비슷한지 재는 중복 지표다.
순위 상관이 낮다는 사실만으로 좋은 구성원은 아니며 단독 성능, 중복 관문과 실제 결합 기여를 함께 통과해야 한다.

| 측정값 | 값 | 판정 |
|---|---:|---|
| 최근접 구성 | `exp045_xgb_depth8` | 비교 대상 |
| 최근접 스피어만 순위 상관 | `0.9814932318570025` | 중복 기준 `0.998`보다 낮음 |
| 풀 전체 순위 상관 범위 | `0.952`부터 `0.981` | 나무 계열과 다른 순서 |
| 풀 전체 잔차 상관 범위 | `0.915`부터 `0.981` | 오차 중복 보조 진단 |
| 표준 평가 결합 | 표시값 `0.96813`에서 `0.96839` | 상승 |
| 정확한 한계 기여 | `+0.00025324160365325366` | 후보 풀 진입 |

화면 28의 두 줄 오차 표식은 이 판정을 설명하기 위한 교육용 예시이며 실제 행별 오차 측정값이 아니다.
실제 채택 근거는 전체 OOF 순위 상관, 잔차 상관과 결합 전후 AUC다.

- 원 판정: [Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965)
- 후보 풀 장부: [`artifacts/pool.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/artifacts/pool.yaml)
- 발표 수치 경계: [`docs/research/presentation-score-evidence.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md)

## H. 최종 314개 예측 열

### 화면 30의 결합 내부 점수 계보

| 단계 | 구성 | nested OOF AUC | 원본 |
|---|---|---:|---|
| 자체 출발점 | 자체 35열 | `0.9698106` | [엄격 외부 후보 사다리 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0006-strict-external-candidate-ladder.md) |
| 기존 외부 예측 포함 | 자체 35열과 외부 207열, 합계 242열 | `0.9702876097776773` | [확장 사다리 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ladder-2.md) |
| 해로운 120열 제외 | 자체 35열과 외부 278열, 합계 313열 | `0.9703509` | [확장 사다리 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ladder-2.md) |
| 결합 규제 내부 선택 | 같은 313열과 C 선택 결합 | `0.9703608940404231` | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md) |
| 최종 재조립 | 자체 36열과 외부 278열, 합계 314열 | `0.9703843058098193` | [314열 재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md) |

화면 30의 `+0.00057`은 첫 단계 표시값 `0.96981`과 마지막 단계 표시값 `0.97038`의 시간순 차이다.
중간에 자체 풀, 외부 예측 범위와 결합 설정이 함께 바뀌었으므로 한 변경의 직접 효과로 해석하지 않는다.

### 화면 31의 최종 조립 입력

| 항목 | 값 | 원본 |
|---|---:|---|
| 자체 전체 자료 재학습 예측 | 36열 | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 외부 예측 | 278열 | [외부 구성원 장부](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/external-member-ledger-v3.md) |
| 최종 결합 입력 | 314열 | [314열 재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md) |
| nested OOF AUC | `0.9703843058098193` | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md) |
| 최종 314열 실행 식별자 | `3279e114ef444cfeaff4232bc401d7b4` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 최종 제출 식별자 | `55907610` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |

### 313열과 314열의 직접 판정

비교 팔은 자체 35열과 외부 278열의 313열 C 선택 결합이고, 재조립 팔은 D 구획에서 확정한 자체 36열과 같은 외부 278열의 314열 결합이다.
교체 문턱은 nested OOF 차이 `+0.00002` 이상과 바깥 fold `5/5` 양수로 결과 확인 전에 고정했다.

| 바깥 fold | 313열 AUC | 314열 AUC | 차이 | 313열 C | 314열 C | 수축 계수 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.9697722263` | `0.9697850148` | `+0.0000127885` | `0.03` | `0.03` | `1.0` |
| 1 | `0.9705318909` | `0.9705601888` | `+0.0000282979` | `0.03` | `0.03` | `1.0` |
| 2 | `0.9704121756` | `0.9704303417` | `+0.0000181661` | `0.03` | `0.03` | `1.0` |
| 3 | `0.9709798514` | `0.9710118898` | `+0.0000320384` | `0.01` | `0.03` | `1.0` |
| 4 | `0.9701083233` | `0.9701340914` | `+0.0000257680` | `0.03` | `0.03` | `1.0` |

전체 nested OOF 차이는 `+0.0000234117693961311`이고 바깥 fold 다섯 곳이 모두 양수여서 두 관문을 통과했다.
비교 팔 구성 해시는 `28680c46db7d7c6301c75e81da346f5fbb01ef5ef359989b34e27795bca4c562`이고 재조립 팔 구성 해시는 `e3208ed93ee2912699aa0a0a02839479489faf69ec65f9dd3b0dd8f82823035e`다.
재조립 팔 nested 예측 SHA-256은 `fc6a8a3208957fc0dc4fa883535a3e3df3be5ec79676719a5be4b0d2bcbc02c8`다.

- 판정 보고: [`docs/research/extended-stack-pool-reassembly/issue513/report.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md)

### 최종 결합 설정

각 예측 열의 원시 확률에서 경험적 순위와 로짓을 함께 만들고 표준화한 뒤 L2 로지스틱 회귀를 맞춘다.
메타 예측의 순위와 단순 구성원 순위 평균 사이의 수축 계수 `lambda`와 로지스틱 규제 강도 `C`는 각 바깥 학습 부분 안에서 분할 하나 제외 방식으로 함께 고른다.

| 항목 | 값 |
|---|---|
| 결합 방식 | `c_selected_shrunk_rank_logit_logistic` |
| C 격자 | `0.001`, `0.003`, `0.01`, `0.03`, `0.1`, `0.3`, `1.0` |
| 수축 계수 격자 | `0.25`, `0.5`, `0.75`, `1.0` |
| 동률 규칙 | 더 작은 C, 이어서 더 작은 수축 계수 |
| 최종 선택 C | `0.03` |
| 최종 선택 수축 계수 | `1.0` |
| 최종 적합 입력 | 전체 314열 OOF와 전체 목표값 |
| 최종 추론 입력 | 같은 열 순서의 314열 시험 예측 |

수축 계수가 `1.0`이므로 최종 예측에는 단순 순위 평균이 섞이지 않고 규제한 메타 로지스틱 예측의 순위가 사용됐다.
로지스틱 계수는 음수도 허용되므로 비음수 평균 결합이 아니다.
구현은 [`CSelectedShrunkRankLogitCombiner`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)와 [`full_fit_predictions`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)에 있다.

### 전체 자료 재학습과 결정적 조립

최종 자체 풀 36열 가운데 항목이 바뀌지 않은 29열은 이미 검증한 전체 자료 예측을 구성원 항목 해시로 확인해 재사용했다.
새로 들어오거나 교체된 다음 7열만 학습 자료 전체로 다시 적합했다.

- `mpv1_exp035_lattice_te_missingness_augmented`
- `mpv1_exp058_logreg_onehot_missingness_augmented`
- `mpv1_exp070_cat_exact_cats_missingness_augmented`
- `mpv1_exp110_lgb_kitopl_no_te_missingness_augmented`
- `mpv1_exp131_lookup_bivariate_plr5_missingness_augmented`
- `exp208_issue500_ag25_missingness_augmented`
- `exp209_issue505_lgb_lr_onehot_init`

자체 36열 실행 식별자는 `223055f44dc9427da588a141bc3b1ca3`이고 최종 314열 실행 식별자는 `3279e114ef444cfeaff4232bc401d7b4`다.
신경망인 `mpv1_exp131_lookup_bivariate_plr5_missingness_augmented`의 난수 42, 43, 44만 Vast.ai GPU 세 장에서 학습했고 나머지 새 구성은 로컬에서 학습했다.

| 고정 기록 또는 산출물 | SHA-256 |
|---|---|
| 공식 자체 풀 `artifacts/pool.yaml` | `40947563a00cab8212498c7e339517e387979b14c6477c6ce8e196036e02044c` |
| 전체 자료 재학습 계획 `artifacts/full-refit-plan.yaml` | `89edf321b1821f1de645799f2353705c461065a275263375d5479e3edd6b006c` |
| 전체 자료 재학습 manifest | `6d109f4e20f4929a26af11eedb13c8748f9ea31a1bbcb42092e5709d3aa42553` |
| 최종 조립 manifest | `7f3179e577748dda1ea5b36a498d07a7a01a9b120f8e00e64a10099502e51495` |
| 조립 체크섬 파일 | `e324b3f333868a12952b0bfc803e37f1be61fce6703b5c8aa06095ab353b165d` |
| 자체 36열 제출 파일 | `5c41f1b8a3780e034fc79fcdaff055924737ef8ce390c289d09b3920aeed6f67` |
| 최종 314열 제출 파일 | `cbb0419a8b34b54ed11ece481d5927da3d98f2aa574839756eb8e965d3ecceaf` |

조립은 서로 다른 새 과정과 새 출력 폴더에서 두 번 실행했고 두 CSV, 전체 조립 manifest와 체크섬 파일이 바이트 단위로 같았다.
최종 314열 제출은 296,302행, `id`와 `addicted_label` 두 열, 유한한 `[0, 1]` 값, 동률 없음과 시험 자료 ID 순서 일치를 통과했다.

- 최종 조립 실행 기록: [`docs/research/extended-stack-final-assembly/issue514/report.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md)
- 기계 판독 제출 기록: [`submission-record.json`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json)
- 자체 구성 설정, 실행 계보와 예산: [`artifacts/full-refit-plan.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/artifacts/full-refit-plan.yaml)

### 외부 278열의 무결성과 라이선스 한계

외부 278열은 우리가 다시 학습한 278개 모형이 아니라 공개 OOF와 시험 예측 쌍이다.
저장소가 확인한 범위는 구성원 이름, 출처, 행 수와 순서, OOF 및 시험 예측 의미 해시, 재채점 AUC, 분할 근거 종류, 사용 조건과 주의 사항이다.

| 분할 근거 | 열 수 |
|---|---:|
| 저자 서술 | 152 |
| 공개 코드 | 98 |
| 같은 저자의 형제 코드 | 13 |
| 분할 벡터 | 12 |
| 근거 없음 | 3 |

| 사용 조건 | 열 수 | 처리 |
|---|---:|---|
| CC0 | 203 | 장부와 조립 manifest에 출처 유지 |
| CC BY 4.0 | 6 | 저작자 표시 유지 |
| Apache 2.0 | 5 | 저작자 표시 유지 |
| unknown | 61 | 결합 입력 전용, 배열 재배포 금지 |
| other | 3 | 결합 입력 전용, 배열 재배포 금지 |

unknown과 other를 합친 64열은 사용 한정 구성원이다.
이 배열은 저장소에 커밋하거나 독립 해법 묶음에 첨부하지 않으며 조립 입력으로만 사용한다.
외부 배열의 완전 재현은 Kaggle 공개 자료의 장기 보존, 고정 노트북 판본과 일부 저자 서술에 의존하므로 자체 36열보다 근거가 약하다.

- 외부 구성원과 해시 장부: [`docs/research/external-member-ledger.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/external-member-ledger.md)
- 증분 공개 후보 감사: [`docs/research/external-member-ledger-v3.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/external-member-ledger-v3.md)
- 선택 구성의 절제와 한계: [`docs/research/extended-stack-ladder-2.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ladder-2.md)

### 제출 계보

| 실행 식별자 | Kaggle 제출 | 구성 | nested OOF AUC | Public | Private | 최종 선택 |
|---|---:|---|---:|---:|---:|---|
| `e88f706e` | `55795055` | 자체 35열 안전판 | `0.9698106` | `0.97099` | `0.97063` | 예 |
| `4f2466f8` | `55810100` | 자체 35열과 외부 207열, 합계 242열 | `0.9702876` | `0.97134` | `0.97106` | 아니요 |
| `443b3a71` | `55823369` | 자체 35열과 외부 278열, 합계 313열 | `0.9703509` | `0.97135` | `0.97108` | 아니요 |
| `30b6f97c` | `55844886` | 같은 313열과 C 선택 결합 | `0.9703609` | `0.97135` | `0.97109` | 아니요 |
| `3279e114` | `55907610` | 자체 36열 전체 자료 재학습과 외부 278열, 합계 314열 | `0.9703843` | `0.97135` | `0.97109` | 예 |
| `0e423c9a` | `55920131` | 314열과 엄격 외부 후보 13열, 합계 327열 | `0.9703890` | `0.97133` | `0.97108` | 아니요, 기록용 제출 |

313열 C 선택판과 최종 314열 판의 Public 및 Private 점수는 표시 단위에서 같다.
314열 갱신이 내부 판정 관문을 통과했다는 사실과 Kaggle 점수가 표시 단위에서 오르지 않았다는 사실을 함께 유지한다.
전체 제출 계보와 원본 링크는 [`docs/research/s6e8-our-final-solution.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md)에 있다.

## I. 1등과의 비교 및 다음 원칙

### 화면 33에서 확인된 사실과 비교 한계

| 항목 | 확인된 값 또는 사실 | 근거 수준 |
|---|---|---|
| 우리 최고 자체 단일 구성 | OOF AUC `0.9694062694182052` | 저장된 후보 풀과 판정 기록으로 재현 가능 |
| 1등 글의 RealMLP | 바깥쪽 예측 AUC `0.970706453` | 1등 글의 그림에서 확인되지만 전체 검증 명세는 미공개 |
| 1등 최종 결합 | 449개 모델 표시 | 구성원 장부, 선택 기준과 결합식은 미공개 |
| 1등 공식 결과 | Private 점수 `0.97176`, 최종 1위 | 공식 최종 순위표에서 확인 |
| 우리 공식 결과 | Private 점수 `0.97109`, 최종 14위 | 공식 최종 순위표와 제출 목록에서 확인 |

우리 단일 구성의 OOF AUC와 1등 RealMLP의 바깥쪽 예측 AUC는 분할표, 전처리 경계와 선택 이력이 같다고 확인할 수 없어 직접 차이를 계산하지 않는다.
1등 RealMLP의 최종 Private 점수와 449개 결합의 구성원 선택 및 가중치도 공개되지 않았다.
따라서 우승권과의 남은 차이를 더 강한 단일 모델을 더 빨리 찾는 탐색 역량으로 보는 것은 확인된 사실에서 도출한 회고 해석이며 검증된 인과 결론이 아니다.

- 1등 원문과 재현 가능성 판정: [1등 해법 원문 조사](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-first-place-writeup.md)
- 우리 단일 구성과 점수 계보: [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md)

### 화면 34의 다음 대회 권고

1. 서로 다른 작동 원리의 강한 단일 모델 후보를 대회 초반부터 넓게 탐색한다.
2. 결과를 보기 전에 고정한 fold와 중단 관문으로 작은 근거에서 확장 여부를 결정한다.
3. 혼자 잘하는가와 함께할 때 돕는가를 나눠 검수한 뒤 서로 다른 오차만 조립한다.

이 권고는 더 많은 실험을 무조건 수행하자는 뜻이 아니다.
더 이른 탐색, 더 빠른 중단과 기존 검수 원칙의 유지를 함께 요구한다.
구체적인 후보 범위와 자원 배분은 다음 대회의 자료와 제약을 확인한 뒤 별도 결정한다.
