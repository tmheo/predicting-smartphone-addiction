이 문서는 대표 화면 시제품에서 기술 근거 부록의 정보 밀도와 원본 연결 방식을 검토하기 위한 부분 초안이다.
전체 35개 화면 제작 때 A부터 I까지의 고정 구획을 모두 채운다.

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
