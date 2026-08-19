# RealMLP 격차의 원인별 정량 귀속 종합

이 문서는 GitHub 이슈 [격차 귀속 종합 문서 작성과 연구 문서 main 반입](https://github.com/tmheo/predicting-smartphone-addiction/issues/233)의 산출물이며, 지도 [공개 RealMLP 0.970x와 exp121 격차의 원인 분해](https://github.com/tmheo/predicting-smartphone-addiction/issues/227)의 목적지를 완결한다.
질문은 "공개 RealMLP 노트북의 LB 0.97009~0.97014와 우리 exp121의 3시드 평균 OOF 0.9637131967 사이 표면 격차 약 0.0064는 어디에서 오는가"이다.
개별 조사의 전체 내용은 [realmlp-gap-metric.md](realmlp-gap-metric.md)(척도 분해), [realmlp-gap-diff.md](realmlp-gap-diff.md)(구성 요소 diff), [realmlp-port-divergence.md](realmlp-port-divergence.md)(발산 지점 진단)에 있고, 이 문서는 그 결과들을 원인별 귀속 하나로 종합한다.

## 결론 요약

표면 격차 0.0064는 세 몫으로 분해되고, 각 몫의 원인은 전부 특정됐다.

| 몫 | 크기 | 원인 |
| --- | ---: | --- |
| 측정 척도 차이 | `0.0011~0.0015` | public LB가 nested OOF보다 높게 앉는 분포 오프셋(실측 `+0.0010~+0.0012`)과 잔여 평균화 효과(`+0.0000~+0.0003`) |
| 진짜 재현 격차 | `-0.0045452012` | 우리 `src/pipeline/realmlp.py` 이식의 동작 차이. 발산 지점은 `transform`의 어휘 매핑 이전 float32 형 변환 한 곳으로 특정됨 |
| 노트북 구성 차이 | 약 `+0.0007` | zhenruiweng 0.97009 판이 beicicc 계약 대비 추가로 가진 파생 특성 16열, PBLD 임베딩 확장, 내부 앙상블 10 |

합산 `0.0011 + 0.0001 + 0.00455 + 0.0007 = 0.00645`는 표면 격차 `0.00638~0.00643`(LB 0.97009 또는 0.97014 기준)과 LB 노이즈 플로어 `0.00015` 이내에서 정합한다.
공개 노트북의 목표 누출(omid 판의 전체 데이터 TE 등)은 그들의 CV 주장 수치만 부풀릴 뿐, 숨겨진 test로 채점되는 LB 수치에는 귀속분이 없다.

## 표면 격차의 척도 분해

[realmlp-gap-metric.md](realmlp-gap-metric.md)([이슈 229](https://github.com/tmheo/predicting-smartphone-addiction/issues/229))가 저장소 내부 실측만으로 층위를 나눴다.

- public LB는 nested OOF보다 약 `+0.0010~+0.0012` 위에 앉는다.
  adarsh1077의 6회 제출 실측이 `+0.00098~+0.00115`, 우리 상위 점수대 제출 실측(이슈 57, 60, 65)이 `+0.00106~+0.00122`다.
  따라서 LB 0.97009는 OOF 척도로 약 `0.9689~0.9691`에 해당한다.
- 시드·초기화 평균의 잔여 효과는 `+0.0000~+0.0003`으로 유계다(자체 3시드 평균 이득 `+0.0000581`, beicicc 2초기화 평균 이득 `+0.0000944`).
- 동일 척도 비교의 기준점은 beicicc 공개 계약 OOF `0.9682583979`이며, exp121 `0.9637131967`과의 차이 `-0.0045452012`가 어떤 척도 보정으로도 설명되지 않는 진짜 재현 격차다.
- LB 0.97009에서 오프셋을 뺀 `0.96899`와 beicicc 계약 `0.96826`의 잔차 약 `+0.0007`은 재현 격차가 아니라 서로 다른 모델(zhenruiweng 변형의 구성 확장)을 비교한 데서 오는 차이다.

## 진짜 재현 격차 -0.0045의 귀속 경로

### 후보 순위화와 기각 ([이슈 230](https://github.com/tmheo/predicting-smartphone-addiction/issues/230))

[realmlp-gap-diff.md](realmlp-gap-diff.md)가 세운 레시피 수준 후보들은 이식 원본인 beicicc 공개 노트북(kernel 129554888, SHA-256 `60a0bd05...58ef`)의 재대조로 전부 기각됐다.

| 기각된 후보 | 기각 근거 |
| --- | --- |
| 바깥쪽 fold 분할 상이 | beicicc 계약의 분할이 우리 `artifacts/folds.parquet`와 일치함이 실측됨([public-stack-provenance.md](public-stack-provenance.md)) |
| 반쪽 LR 스케줄(고정 4 epoch, 지평 8) | 원본 노트북 자체가 `FIXED_EPOCHS=4, SCHEDULE_HORIZON_EPOCHS=8`로 동일 |
| fold 안 어휘의 unknown 손실 | 원본도 fold 안 어휘 적합에 unknown 코드 0을 쓰는 같은 구조 |
| TE 구현 차이 | 원본도 sklearn `TargetEncoder(cv=5, smooth=auto, shuffle, random_state=시드+fold)`로 동일, 53열 assert까지 같음 |
| `placebo_noise` 1열 | exp121 실측 순열 중요도 `0.0000398`로 잡음 수준 |
| epoch 선택 낙관 | 계약이 epoch 선택 금지를 명시 |

남은 배타적 두 후보는 (1) 우리 이식의 동작 차이, (2) 계약 OOF가 이 노트북 레시피의 값이 아닐 가능성이었고, 원본 그대로 실행 한 번으로 동시 판별하기로 했다.

### 판별: 이식 격차로 확정 ([이슈 231](https://github.com/tmheo/predicting-smartphone-addiction/issues/231))

beicicc 원본 노트북을 한 글자도 바꾸지 않고 Vast.ai GPU에서 실행한 결과, OOF ROC-AUC `0.9681533377`로 계약 수치대에 정확히 들어왔다.

| 수치 | 값 | 차이 |
| --- | ---: | ---: |
| 계약 OOF (beicicc 공개 명세) | 0.9682583979 | 기준 |
| 원본 그대로 재실행 | 0.9681533377 | -0.0001051 |
| exp121 (우리 이식, 3시드 평균) | 0.9637131967 | -0.0045452 |

계약 대비 `-0.000105`는 GPU 비결정성과 실행 환경 차이 수준이므로 "계약 OOF가 다른 출처" 가설은 기각됐고, `-0.0045452`는 우리 이식의 동작 차이로 확정됐다.
원본 결과 묶음은 main `run-logs/vast-issue231/`에 보존돼 있다.

### 발산 지점: 어휘 매핑 이전의 float32 형 변환 ([이슈 235](https://github.com/tmheo/predicting-smartphone-addiction/issues/235))

[realmlp-port-divergence.md](realmlp-port-divergence.md)가 발산 지점을 `src/pipeline/realmlp.py`의 `_FoldFeatureEngineer.transform` 한 곳으로 특정했다.
수치 열을 float32로 형 변환한 뒤 float64로 만들어진 어휘에 매핑하기 때문에, 소수 두 자리 값 대부분이 어휘 키와 일치하지 않아 소수 6개 열의 정확값 범주 코드가 거의 전부 unknown 0으로 무너진다.

핵심 대조 앵커가 이 진단을 양방향 정확값으로 재현한다.

| 측정 (fold 1, 같은 분할·같은 적합 상태) | outer valid unknown 합계 |
| --- | ---: |
| 우리 이식 그대로 (float32 변환 후 매핑) | 800,896 |
| 같은 적합 상태에서 float64 유지 매핑 | 23 |
| 원본 노트북 GPU 실행 실측 | 23 |
| exp121 확인 실행 실측 (run `56701722`) | 800,896 |

exp121 확인 실행의 `validation_unknown_category_values` 실측은 fold별 800,730~801,006, 검증 행당 평균 5.79채널, fold 간 편차 0.03% 미만이다.
반면 원본 실행의 같은 지표는 fold별 23~27이다.
학습 부분에서도 소수 6개 열 코드의 96.5%가 0이므로, 이 채널들과 그에 붙은 TE 6열은 검증에서만 죽는 게 아니라 학습 단계부터 죽어 있다.
LightGBM 대리 스크리닝은 float64 유지 매핑이 `+0.0029654`를 회복함을 보였고, 정확값 범주를 임베딩으로 소비하는 RealMLP에서는 회복 폭이 더 클 것으로 추정돼 1순위 수정(dtype 정합 복원)의 추정 효과는 `+0.003~+0.0045`로 격차의 대부분이다.
원본 실행 대비 exp121의 fold별 격차가 다섯 fold 모두 균일하게 약 `-0.0045`인 양상도 잡음이 아닌 체계적 특성 손실과 일치한다.

## 어닐링 완주 델타 실측 ([이슈 232](https://github.com/tmheo/predicting-smartphone-addiction/issues/232))

반쪽 LR 스케줄이 원본과 동일함이 확인된 뒤에도, 어닐링 완주가 단독 개선 후보인지는 단일 델타 짝비교로 실측했다.
exp121에서 `schedule_epochs`만 8에서 4로 바꾼 exp122(커밋 9e024ce)의 3시드 확인 짝비교 결과다.

| 측정값 | exp122 (완주) | exp121 (반쪽) | 델타 |
| --- | ---: | ---: | ---: |
| 3시드 평균 OOF AUC | 0.9637326202 | 0.9637131967 | +0.0000194234 |

방향은 3/3 시드에서 양수로 일관되지만 크기가 풀 마일스톤 문턱 `+0.0002`의 1/10, 재현 격차의 0.4%에 그친다.
어닐링 완주는 격차 원인도 아니고 단독 개선 후보로도 의미 없는 수준이다.
이 델타는 죽은 채널 상태의 이식 위에서 측정된 값이므로, dtype 정합 수정판 기준으로는 근사값으로만 해석해야 한다.
main MLflow run은 `5137f3cde57147e083d76653f509ad42`, 결과 묶음은 `run-logs/vast-issue232/`에 있다.

## 판정

- 표면 격차 0.0064 중 척도 차이(약 5분의 1)와 노트북 구성 차이(약 1할)를 걷어내면, 실재하는 재현 격차는 `-0.0045452012`이고 그 원인은 이식 코드의 float32 형 변환 한 곳이다.
- 레시피(분할, 스케줄, 어휘 규율, TE)는 원본과 동일함이 확인됐으므로, 공개 RealMLP와 우리 사이에 방법론 격차는 없다.
- 후속의 실행 개폐(dtype 정합 수정 재실행, 파생 16열·용량 확장 흡수 여부)는 [exp121 개선 실험 개폐 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/234)이 정하며, 수정 시 재실험 파급 목록은 [realmlp-port-divergence.md](realmlp-port-divergence.md)에 있다.
