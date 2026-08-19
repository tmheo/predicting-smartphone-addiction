# 공개 RealMLP 3종과 exp121 구현의 구성 요소 diff

이 문서는 GitHub 이슈 [공개 RealMLP 3종과 exp121 구현의 구성 요소 diff](https://github.com/tmheo/predicting-smartphone-addiction/issues/228)의 답이며, 지도 이슈 #227의 연구 티켓이다.
질문은 세 공개 RealMLP 노트북과 우리 `exp121_realmlp_fixed4_two_init` 사이에 성능에 영향을 줄 수 있는 구성 요소 차이가 정확히 무엇인가다.

## 비교 대상과 판본 고정

| 구현 | 출처 | 판본 고정 | 보고 수치 |
| --- | --- | --- | --- |
| omid | [omidbaghchehsaraei RealMLP](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction) | 기존 분석 재활용: `code-notebook-insights.md` 22번, `code-notebook-analysis-14-25.md` 22위 절 | 공개 점수 미표기, OOF 출력 없음 |
| nawfeel | [nawfeelrahman1124444 RealMLP 0.97014](https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014) | kernel id `129460789`, `realmlp-0-97014.ipynb` SHA-256 `f79643a83851c6804a39a09036b0700fefa7e767aa620c5a7ea5d3f7ec11f406` | 제목 기준 공개 LB 0.97014 |
| zhenrui | [zhenruiweng Single Model RealMLP](https://www.kaggle.com/code/zhenruiweng/s6e8-public-lb-0-97009-single-model-realmlp) | kernel id `129907249`, `s6e8-public-lb-0-97009-single-model-realmlp.ipynb` SHA-256 `ef16bb88782581ad4e880f295903db88278428464c1aa31643a0511d5197b116` | 제목 기준 공개 LB 0.97009 |
| exp121 (우리) | `src/pipeline/realmlp.py`, `configs/exp121_realmlp_fixed4_two_init.yaml` | 저장소 커밋 기준, 확인 실행 `56701722ba224b98b5ddf799ec5e55db` | 3시드 평균 OOF `0.9637131967` |

nawfeel과 zhenrui 두 판은 2026-08-19에 `kaggle kernels pull`로 받았고, Kaggle API 메타데이터에는 판본 번호 필드가 없어 kernel id와 받은 소스의 SHA-256으로 판본을 고정한다.
두 노트북 소스는 Kaggle 공개 노트북 규칙에 따라 Apache License 2.0이며, 이 문서는 분석 목적의 열람과 인용만 하고 코드를 `src/`에 들여오지 않았다.
두 노트북의 `competition_sources`는 `playground-series-s6e8` 하나뿐이고 외부 dataset 입력은 없다.

우리 쪽 비교 기준에는 exp121의 이식 원본인 [beicicc fold-safe RealMLP 계약](https://www.kaggle.com/datasets/beicicc/s6e8-fixed4-realmlp-two-seed-artifacts)도 참고로 포함한다.
계약의 두 초기화 평균 AUC는 `0.9682583979`이고 우리 3시드 평균과의 차이는 `-0.0045452012`다.
`public-stack-provenance.md`에 따르면 beicicc 계약 자체가 zhenrui 노트북과 szymonkapiski `train_realmlp.py` 계보에서 나온 레시피다.

## 축별 diff 표

| 축 | omid | nawfeel | zhenrui | exp121 (우리) |
| --- | --- | --- | --- | --- |
| 특성 집합 | 원시 12열 전부 문자열 범주 + 파생 수치(결측 개수, 활동 합, 화면 비율, 미설명 화면 시간, 수면·알림 비율) + 원시 12열 TE·빈도 | 원시 12열 전부 정확값 범주 + 파생 수치 10열(결측 수, 비율·합) + 로그 2열 + 자릿수 수치 27열 + 자릿수·반올림·조합·activity 범주 약 50열 + quantile bin 5열(40~50구간) + 모든 원시·파생 열(약 100열)에 TE(mean)와 빈도, 총 약 300열 | 원시 12열 + 결측 지시 9열 + 결측 수 1열 + 상호작용·비율·로그 파생 수치 16열 + 정확값 범주 9열 + quantile bin 2열(10구간) + bin 제외 범주 21열 TE, 총 70열 | 원시 12열 + placebo 1열 + 결측 지시 9열 + 정확값 범주 9열 + quantile bin 2열(10구간) + bin 제외 범주 21열 TE, 총 54열 |
| 파생 수치 특성 | 몇 열 있음 | 많음(비율, 합, 로그, 자릿수) | 16열(비율, 합, 로그, 스트레스 상호작용, 수면 부족) | 없음(beicicc 계약의 53열 구성을 그대로 따름) |
| 결측 처리 | 기존 분석에 상세 기록 없음 | 결측을 문자열 "nan" 범주로 유지, 파생 수치는 0 대체 | 중앙값 대체(전체 train 기준) + 결측 지시 열 | 중앙값 대체(outer 학습부 기준) + 결측 지시 열 |
| 전처리 적합 범위 | TE를 전체 자료에서 CV 전에 적합 | 어휘·bin·TE 전부 전체 train에서 CV 전에 적합 | 어휘·중앙값·bin은 전체 train에서 fold 밖 적합, TE만 fold 안 | 중앙값·어휘·bin·TE 전부 outer 학습부 안 적합, 검증의 미지 범주는 전용 unknown 코드 0 |
| target encoding 구현 | 5-fold OOF 모양이지만 모형 CV 전에 한 번 계산, mean+빈도 | 모형 CV 전에 한 번 계산, 모형 CV와 같은 시드·같은 5-fold, smoothing 10, mean+빈도, 모든 열 대상 | 각 outer fold 안에서 sklearn `TargetEncoder(cv=5, smooth="auto")` 적합 | 각 outer fold 안에서 sklearn `TargetEncoder(StratifiedKFold 5, smooth="auto")` 적합, fold별 파생 시드 |
| 수치 임베딩 | pytabkit RealMLP-TD 기본 piecewise-linear | PBLD 주기(cos), hidden 20, out 5, freq 5.0, PReLU | PBLD 주기, hidden 32, out 6, freq 10.0, PReLU | PBLD 주기, hidden 20, out 5, freq 5.0, PReLU |
| 은닉 구조 | pytabkit 기본(기존 분석에 별도 기록 없음) | [512, 512, 512], GELU, embed 16, dropout 0.06, wd 0.013, lr 0.01 | [768, 512, 512], SiLU, embed 8, dropout 0.07, wd 0.015, lr 0.008 | [512, 512, 512], SiLU, embed 8, dropout 0.06, wd 0.013, lr 0.01 |
| epoch과 LR 스케줄 | 최대 100 epoch + 검증 조기 종료 | 4 epoch에 flat_cos(flat 0.35)를 완주, epoch별 검증 AUC로 최고 상태 선택 | 10 epoch에 flat_cos(flat 0.3)를 완주, epoch별 검증 AUC로 최고 상태 선택 | 고정 4 epoch에 스케줄 지평 8 epoch, 진행률 0.5에서 종료라 LR이 최고치의 약 81%인 채 어닐링 없이 끝남, epoch 선택 없음 |
| 내부 앙상블 수 | 8 | 16 | 10 | 8, 여기에 fold 안 초기화 2개 평균 |
| 시드 수 | 1 | 1(63) | 3(42, 789, 1011), 시드마다 다른 CV 분할 | 파이프라인 시드 3(42, 43, 44), 커밋된 공통 fold 고정 |
| test 예측 생성 | fold 모형 5개 평균 | fold 모형 5개(최고 epoch 상태) 평균 | 시드 3 x fold 5 = 15개 모형 평균 | OOF는 fold 학습, 제출은 전체 자료 재학습(`fit_full`, 규약 1.25배로 5 epoch) 경로이며 이 실험에서는 미실행 |
| 원본/proxy 자료 | 미사용 | 미사용 | 미사용 | 미사용 |
| 검증 절차 누출 | TE 교차 폴드 누수(큼) + 조기 종료의 검증 선택(작음) | TE 교차 폴드 누수(중간) + 최고 epoch 선택(작음) + cat 차원 계산에 test 사용(무해) | TE는 안전, 최고 epoch 선택(작음) + 시드별 다른 분할의 OOF를 평균한 뒤 AUC 계산(작음) + fold 밖 어휘·중앙값·bin(목표 미사용) | 없음, `placebo_noise` 통제와 fold별 적합 검증 포함 |

zhenrui의 시드별 CV 분할이 서로 다르므로, 보고된 다중 시드 OOF AUC는 서로 다른 분할의 예측 평균에 대한 값이라 우리 고정 fold 3시드 평균과 정의가 다르다.
nawfeel의 test용 어휘 매핑은 미지 값을 -1로 만든 뒤 +1 보정으로 0 코드에 넣지만, zhenrui의 test 매핑은 미지 값 -1을 clip으로 코드 0에 겹쳐 실제 첫 범주와 충돌시키는 사소한 결함이 있다.

## 차이별 예상 효과 방향

### 공개 쪽 수치를 부풀리는 요소

1. omid와 nawfeel의 CV 전 일괄 target encoding은 학습 행의 인코딩 통계에 검증 fold 목표값이 들어가는 교차 폴드 누수다.
   nawfeel은 인코딩 fold와 모형 fold가 같은 시드라 검증 행 자체의 인코딩은 fold 정합이지만, 학습 행 경유의 누수는 남아 보고 OOF를 부풀린다.
   두 판의 공개 LB는 test 목표값을 쓰지 않으므로 그 자체는 실측이지만, 누수 특성으로 학습된 모형의 OOF는 우리 fold-safe 기준과 비교할 수 없다.
2. nawfeel과 zhenrui의 epoch별 검증 AUC 최고 상태 선택은 fold 검증값으로 checkpoint를 고르는 선택 낙관이며, OOF로 저장하는 값도 그 최고 epoch의 검증 예측이라 OOF가 소폭 부풀려진다.
   omid의 검증 조기 종료도 같은 계열이다.
3. zhenrui의 어휘·중앙값·bin fold 밖 적합은 목표값을 쓰지 않으므로 target 누수는 아니지만, 정확값 범주의 어휘를 전체 train에서 만들기 때문에 검증 행의 정확값이 항상 어휘 안에 있게 되어 fold 엄격 CV보다 OOF가 유리해진다.
4. zhenrui의 시드별 상이 분할 OOF 평균 후 AUC 계산은 분할 앙상블 효과가 섞인 낙관적 요약이다.

### 우리가 놓친 진짜 개선 후보

1. 파생 수치 특성 블록이 가장 뚜렷한 차이다.
   zhenrui는 beicicc 계약(우리 특성 집합과 동일한 53열)에 없는 상호작용·비율·로그 16열을 더해 LB 0.97009에 도달했고, nawfeel은 자릿수·반올림·조합·빈도까지 훨씬 넓게 쓴다.
   beicicc 자신의 다른 계약에서 정체성·자리수 블록이 +0.0023~0.0024, 화면 관계 7특성이 +0.00044를 기록한 전례가 있어, 수천분의 1 단위의 진짜 개선 여지가 있다.
2. LR 스케줄 완주가 우리 쪽 특이점이다.
   세 공개 판 모두 스케줄을 끝까지 어닐링하는데, exp121만 고정 4 epoch에 지평 8 epoch라 LR이 최고치의 약 81%인 중간 상태에서 끝난다.
   이 반쪽 스케줄은 beicicc 계약 레시피의 서술을 따른 것이지만, 계약 대비 `-0.0045` 차이의 원인 후보이므로 `schedule_epochs=4`로 완주시키는 짝비교가 저렴한 1순위 확인이다.
3. 정확값 어휘의 적합 범위 차이는 개선 후보이자 비교가능성 문제다.
   우리 fold 안 어휘에서는 outer 학습부에 없던 검증 행의 정확값이 unknown 0으로 떨어져 정확값 정체성 신호를 잃는다.
   확인 실행의 관측값 `validation_unknown_category_values`로 손실 규모를 바로 잴 수 있고, train 전체 어휘(목표 미사용 transductive)는 저장소 규율 판단이 필요한 별도 결정이다.
4. 용량과 내부 앙상블 확장은 작은 개선 후보다.
   zhenrui의 n_ens 10, 첫 층 768, PBLD hidden 32·out 6·freq 10.0, nawfeel의 n_ens 16과 embed 16은 우리 값보다 크며, 효과는 수만분의 1에서 수천분의 1 사이로 추정된다.
5. 빈도(count) 인코딩은 nawfeel과 omid만 쓰고 우리와 zhenrui에는 없으므로 대리 스크리닝 후보로 남는다.

### 차이가 아닌 것

원본/proxy 자료는 네 구현 모두 쓰지 않으므로 이 축에서는 차이가 없다.
class weight balanced, label smoothing 0.04 cos, EMA 0.997875, robust scale 전처리, NTP 선형층과 softmax 출력 같은 골격은 nawfeel·zhenrui·exp121이 사실상 동일하다.

## 결론

공개 RealMLP의 표기 점수와 exp121의 차이는 하나의 원인이 아니라 세 층으로 나뉜다.
첫째, omid·nawfeel의 TE 누수와 세 판 공통의 epoch 선택 낙관은 공개 쪽 OOF를 부풀리는 평가 인공물이라 채택 대상이 아니다.
둘째, 파생 수치 특성 블록, 스케줄 완주, 용량 확장은 우리가 누출 없이 흡수할 수 있는 진짜 개선 후보다.
셋째, 정확값 어휘의 fold 안 적합은 우리 규율의 의도된 비용이며, `validation_unknown_category_values` 측정과 `schedule_epochs=4` 완주 짝비교가 beicicc 계약 대비 `-0.0045`를 분해하는 가장 저렴한 다음 실험이다.
