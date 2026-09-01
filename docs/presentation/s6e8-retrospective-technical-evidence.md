이 문서는 35개 발표 화면의 핵심 수치와 기술 정의를 원본 기록으로 연결하는 단일 기술 근거 부록이다.
화면 제작이 완료된 구획부터 A부터 I까지의 고정 구조에 맞춰 채운다.

## A. 공식 결과와 자료 범위

### 자료 범위

| 항목 | 값 | 원본 |
|---|---:|---|
| 원시 입력 | 12개 생활 습관 변수 | [첫 기준 실행 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml) |
| 학습 행 | 691,369행 | [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md) |
| 예측 행 | 296,302행 | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 목표값 | 이진 목표값 `addicted_label` | [첫 기준 실행 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml) |

식별자와 목표값은 12개 예측 입력에 포함하지 않는다.
발표 화면에서는 행 수를 각각 약 69만 건과 약 30만 건으로 줄여 말한다.

### 화면 32의 공식 결과

| 항목 | 값 | 원본 |
|---|---:|---|
| 최종 주 제출 식별자 | `55907610` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |
| Public 점수 | `0.97135` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| Private 점수 | `0.97109` | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md) |
| 최종 순위 | 14위 | [최종 해법 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md) |

Kaggle 공식 순위표와 제출 목록은 2026-09-01에 다시 조회했다.
마지막 업로드인 327열 제출 `55920131`은 Private 점수 `0.97108`이므로 최종 14위 성적을 만든 제출이 아니다.
313열 C 선택판과 최종 314열 판의 Public 점수는 모두 `0.97135`이고 Private 점수도 표시 단위에서 모두 `0.97109`다.

## B. 점수와 검증 경계

### 화면 08의 직접 비교

| 항목 | 비교 기준 | 후보 | 차이 |
|---|---:|---:|---:|
| OOF AUC | `0.962759` | `0.966046` | 표시값 `+0.00329` |

- 비교 기준 실행 식별자: `ce66e16b12fd43b4bc95fdcf2972555c`
- 후보 실행 식별자: `77217687c0514dab9f693fd4aa50c741`
- 기준 설정: [`configs/exp001_lgbm_baseline.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml)
- 후보 설정: [`configs/exp003_categorical_copies.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp003_categorical_copies.yaml)
- 원 판정: [전 피처 범주형 challenger 실험: 실행과 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)

### 화면 12의 nested OOF 경계

구성원과 결합 방식은 봉인한 바깥 fold를 제외한 OOF에서 고른다.
선택한 구성을 봉인한 fold에 적용해 예측을 만들고, 다섯 바깥 fold의 예측을 원래 행 순서로 이어 nested OOF를 만든다.

- 계약: [`docs/adr/0001-experiment-adoption-contract.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)
- 구현: [`src/pipeline/ensemble.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)
- 분할 생성: [`scripts/make_folds.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/scripts/make_folds.py)

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
| 전체 자료 재학습 실행 식별자 | `3279e114ef444cfeaff4232bc401d7b4` | [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md) |
| 최종 제출 식별자 | `55907610` | [제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |

외부 278열 가운데 라이선스가 unknown 또는 other로 분류된 64열은 결합 입력으로만 사용하고 재배포하지 않는 사용 한정 구성원이다.
최종 자체 예측 파일의 SHA-256은 `5c41f1b8a3780e034fc79fcdaff055924737ef8ce390c289d09b3920aeed6f67`이다.
최종 314열 예측 파일의 SHA-256은 `cbb0419a8b34b54ed11ece481d5927da3d98f2aa574839756eb8e965d3ecceaf`이다.

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
