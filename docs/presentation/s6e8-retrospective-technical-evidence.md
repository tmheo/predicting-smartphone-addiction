이 문서는 발표 본문 화면에서 정확한 실행 기록과 원본 근거로 이동하는 단일 기술 근거 부록이다.
현재 화면 01부터 09에 대응하는 A부터 C까지를 완성했고, 대표 화면 시제품에서 먼저 작성한 E, F, H를 함께 유지한다.

## A. 공식 결과와 자료 범위

### 공식 결과와 제출

| 항목 | 값 | 원본 |
|---|---:|---|
| 최종 순위 | 14위 | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/issues/547#issuecomment-5487179345) |
| Public 점수 | `0.97135` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| Private 점수 | `0.97109` | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/issues/547#issuecomment-5487179345) |
| Kaggle 제출 식별자 | `55907610` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |
| 전체 자료 재학습 실행 식별자 | `3279e114ef444cfeaff4232bc401d7b4` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |

제출 파일은 `artifacts/submissions/issue514-extended314-own-full.csv`이며 SHA-256은 `cbb0419a8b34b54ed11ece481d5927da3d98f2aa574839756eb8e965d3ecceaf`다.
Public 점수와 Private 점수는 같은 제출을 서로 다른 시험 표본에서 채점한 값이므로 두 값의 차이를 개별 실험의 효과로 해석하지 않는다.

### 자료 범위

| 항목 | 값 | 원본 |
|---|---:|---|
| 학습 행 | `691,369` | [첫 기준 실행 종결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/18#issuecomment-5239693077) |
| 시험 행 | `296,302` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 입력 | 식별자와 목표값을 제외한 12개 열 | [첫 기준 실행 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml) |
| 목표값 | 이진 열 `addicted_label` | [자료 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/data.py) |

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

고정한 다섯 fold를 차례로 검증 부분으로 사용한다.
각 학습 행은 자기 목표값으로 학습하지 않은 모델에서 예측 하나를 받고, 다섯 예측 조각을 원래 행 순서로 이어 OOF를 만든다.
OOF AUC는 이 OOF 예측 전체를 ROC AUC로 채점한 값이다.

- 분할 생성: [`scripts/make_folds.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/scripts/make_folds.py)
- OOF 생성과 채점: [`src/pipeline/cv.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/cv.py)
- 판정 계약: [`docs/adr/0001-experiment-adoption-contract.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)

### 화면 12의 nested OOF 경계

구성원과 결합 방식은 봉인한 바깥 fold를 제외한 OOF에서 고른다.
선택한 구성을 봉인한 fold에 적용해 예측을 만들고, 다섯 바깥 fold의 예측을 원래 행 순서로 이어 nested OOF를 만든다.

- 계약: [`docs/adr/0001-experiment-adoption-contract.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)
- 구현: [`src/pipeline/ensemble.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)
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

- 비교 기준 설정: [`configs/exp001_lgbm_baseline.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml)
- 전부 범주화 설정: [`configs/exp002_all_categorical.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp002_all_categorical.yaml)
- 수치 유지와 범주 복제 설정: [`configs/exp003_categorical_copies.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp003_categorical_copies.yaml)
- 원 판정: [전 피처 범주형 challenger 실험: 실행과 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)

## E. 실험 실행 체계

| 실행 장소 | 이 회고에서 맡은 역할 | 정식 판정에 들어오는 조건 |
|---|---|---|
| 로컬 | 개발, 소규모 실행, 반입, 재채점, 판정, 최종 조립 | 원본 실행 또는 검증된 실행 기록 묶음 |
| Kaggle CPU | 고정한 CPU 비교 짝의 병렬 실행 | 같은 공급자와 실행 환경 등급의 두 비교군 완결 |
| Kaggle GPU | 초반 정식 실행, 후반 호환성 확인과 진단 | 정식 판정 범위에 맞는 실행만 사용 |
| Vast.ai | 주 GPU 실행 장소 | 해시 대조, 원본 상태와 입력 경계 확인, 로컬 재채점 통과 |
| Runpod | Vast.ai 전환 조건을 충족할 때 쓰는 예비 GPU 실행 장소 | Vast.ai와 같은 반입 및 재채점 계약 통과 |

- 역할과 전환 근거: [`docs/research/presentation-environment-evidence.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md)
- 원격 파일 전달: [`docs/agents/remote-gpu-transfer.md`](https://github.com/tmheo/kagglekit/blob/main/docs/agents/remote-gpu-transfer.md)

## F. 성공과 중단 사례

| 사례 | 점추정 또는 진입 결과 | 반복 근거 | 사전 관문 | 결론 |
|---|---|---|---|---|
| RealMLP 자료형 정합 복원 | `+0.0046091491` | 난수 42, 43, 44 | 같은 조건 짝비교 | 채택 |
| Lookup-Transformer 설정 17개 | 모두 기준 미달 | fold 0, 난수 42 | 진입 진단 | 중단 |
| 약한 외부 예측 120개 계열 | 한계 기여 `-0.000057` | 전체 결합 판정 | 양의 기여 | 미채택 |
| 327열 결합 | `+0.0000046619547824` | 바깥 fold 3/5 양수 | 사전 교체 문턱 | 미채택 |

### RealMLP 자료형 정합 복원

- 수정판 실행 식별자: `c41c6a4deae04e1fbd8a75193eaaa32c`
- 결함판 출처 실행 식별자: `dbe1f8cccca4458889265eb0d0f45273`
- 미등록값: `800,896`에서 `23`
- 3시드 평균 OOF AUC: `0.9637131967`에서 `0.9683223458`
- 원 판정: [자료형 정합 복원 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/243#issuecomment-5343200265)

### 중단 및 미채택 사례

- Lookup-Transformer 설정 17개: [제한 탐색 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/160#issuecomment-5308772959)
- 약한 외부 예측 120개 계열: [`docs/research/extended-stack-ladder-2.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ladder-2.md)
- 327열 결합: [`docs/research/extended-stack-ext327/issue526/comparison.json`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ext327/issue526/comparison.json)

## H. 최종 314개 예측 열

| 항목 | 값 | 원본 |
|---|---:|---|
| 자체 전체 자료 재학습 예측 | 36열 | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 외부 예측 | 278열 | [외부 구성원 장부](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/external-member-ledger-v3.md) |
| 최종 결합 입력 | 314열 | [314열 재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md) |
| nested OOF AUC | `0.9703843058098193` | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md) |
| 전체 자료 재학습 실행 식별자 | `3279e114ef444cfeaff4232bc401d7b4` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 최종 제출 식별자 | `55907610` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |

최종 자체 예측 파일의 SHA-256은 `5c41f1b8a3780e034fc79fcdaff055924737ef8ce390c289d09b3920aeed6f67`이다.
최종 314열 예측 파일의 SHA-256은 `cbb0419a8b34b54ed11ece481d5927da3d98f2aa574839756eb8e965d3ecceaf`이다.
