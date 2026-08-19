# RealMLP 이식 발산 지점 진단

이 문서는 GitHub 이슈 [src/pipeline/realmlp.py와 beicicc 원본 노트북의 코드 수준 diff로 -0.0045 발산 지점 특정](https://github.com/tmheo/predicting-smartphone-addiction/issues/235)의 분석 결과를 기록한다.
비교 대상은 `src/pipeline/realmlp.py`(설정 `configs/exp121_realmlp_fixed4_two_init.yaml`)와 beicicc 원본 노트북(SHA-256 `60a0bd05332e8932468d9cc796855013be3c3798344fd75c15c016764eba58ef`, 사본은 main `run-logs/vast-issue231/input-root/notebook/`)이다.
이슈 231에서 원본 그대로 GPU 실행이 OOF `0.9681533377`로 계약 수치대에 들어와, 재현 격차 `-0.0045452012`는 우리 이식의 동작 차이로 확정된 상태였다.

## 결론

발산 지점은 `src/pipeline/realmlp.py`의 `_FoldFeatureEngineer.transform` 한 곳이다.
수치 열을 float32로 형 변환한 뒤에 수치-범주 어휘 매핑과 quantile bin 변환을 수행하는데, 어휘와 bin 경계는 `fit`에서 float64 값으로 만들어졌다.
소수 두 자리 값 대부분은 float32로 정확히 표현되지 않아 float64 어휘 키와 일치하지 않고, 소수 6개 열의 정확값 범주 코드가 학습·검증·시험 전부에서 거의 전부 0(unknown)으로 무너진다.
이 한 곳이 티켓의 대조 앵커(원본 23~27 vs 우리 약 800,700~801,000)를 정확히 재현하며, 수정 시 재현 격차 `-0.0045`의 대부분을 회복할 것으로 추정한다.

## 발산 지점: 어휘 매핑 이전의 float32 형 변환

원본 노트북의 `FoldFeatureEngineer.transform`은 수치 열을 float64로 유지한 채 결측만 중앙값으로 채우고 곧바로 어휘 매핑을 수행한다.

```python
# 원본 (cell 7)
out[column] = out[column].fillna(self.medians_[column])          # float64 유지
out[name] = out[column].map(self.category_maps_[name]).fillna(0)
```

우리 이식의 `transform`은 어휘 매핑 이전에 수치 열을 float32로 형 변환한다(`src/pipeline/realmlp.py:222-236`).

```python
# 우리 이식
output[column] = (
    pd.to_numeric(output[column], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(self.medians[column])
    .astype("float32")                                            # 어휘 매핑 이전의 형 변환
)
...
output[name] = output[column].map(self.category_maps[name]).fillna(0)
```

`fit`은 float64 값으로 어휘를 만들었으므로, float32로 변환된 값(예: 1.83 -> 1.8300000429...)은 float64 키 1.83과 일치하지 않아 unknown 코드 0이 된다.
값이 정수 계열인 `age`, `notifications_per_day`, `app_opens_per_day` 세 열은 float32로 정확히 표현되므로 생존하고, 소수 두 자리 값을 갖는 나머지 여섯 열(`daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `weekend_screen_time`)은 x.00, x.25, x.50, x.75 형태(전체의 약 4%)만 살아남는다.
같은 이유로 두 quantile bin 채널도 float64 경계에 float32 값이 들어가 경계에 걸친 값들이 이웃 bin으로 밀린다.

## 대조 앵커의 실측 재현

두 지표의 집계 기준은 동일함을 먼저 확인했다.
원본 `unknown_value_count`와 우리 `unknown_value_count`는 둘 다 매핑 대상 12개 열(원시 범주 3개 + 수치-범주 9개)에서 코드가 0인 값의 개수 합이며, 값 단위로 같다.
따라서 규모 차이는 집계 기준 차이가 아니라 실제 동작 차이다.

원본과 동일한 분할(StratifiedKFold 5, shuffle, seed 42)의 fold 1에서 우리 `_FoldFeatureEngineer`를 그대로 적합하고, transform만 두 방식으로 실행해 비교했다.

| 측정 | outer valid unknown 합계 |
| --- | --- |
| 우리 이식 그대로 (float32 변환 후 매핑) | 800,896 |
| 같은 적합 상태에서 float64 유지 매핑 | 23 |
| 원본 노트북 GPU 실행 실측 (fold 1) | 23 |
| exp121 확인 실행 실측 (run `56701722`, fold 1) | 800,896 |

양방향 모두 정확값 일치다.
exp121의 fold 1 실측 800,896이 재현값과 자리까지 일치하므로, 저장소의 커밋된 fold 분할과 원본 노트북의 분할도 같은 분할임이 함께 확인된다.
학습 부분에서도 소수 6개 열의 코드 3,318,570개 중 3,203,433개(96.5%)가 0이므로, 이 채널들은 검증에서 unknown이 되는 수준이 아니라 학습 단계부터 죽어 있다.

fold 1의 열별 unknown 분포는 다음과 같다.

| 열 | 우리 이식 | float64 유지 |
| --- | --- | --- |
| gender, stress_level, academic_work_impact | 0 | 0 |
| age_cat_, notifications_per_day_cat_, app_opens_per_day_cat_ | 0 | 0 |
| daily_screen_time_hours_cat_ | 133,617 | 10 |
| social_media_hours_cat_ | 134,095 | 2 |
| gaming_hours_cat_ | 134,111 | 0 |
| work_study_hours_cat_ | 133,078 | 0 |
| sleep_hours_cat_ | 132,672 | 0 |
| weekend_screen_time_cat_ | 133,323 | 11 |

## 손상 범위

- 소수 6개 열의 정확값 범주 임베딩 채널이 죽는다. 원본 실행 기준 이 채널들의 어휘 차원은 232~1,434로, 이 노트북 설계에서 용량이 가장 큰 범주 채널들이다.
- 같은 6개 열의 목표 인코딩 열(`_*_cat_TE`)이 상수에 가깝게 퇴화한다. 학습 행의 96.5%가 코드 0 한 범주에 몰리기 때문이다.
- 두 quantile bin 채널(`daily_screen_time_hours`, `social_media_hours`)은 fold 1 검증에서 각각 19,934행(14.4%)과 28,273행(20.4%)이 float64 유지 대비 다른 bin 코드를 받는다. 학습과 검증이 같은 방식으로 밀리므로 내부적으로는 일관된 인코딩이고, 단독 효과는 작다.

원시 수치 값 자체(PBLD 임베딩 입력)와 결측 표지 9개 열, 원시 범주 3개 열은 손상되지 않는다.

## 효과 크기 추정

LightGBM 대리 스크리닝(fold 1, 동일 특성 계약 53열 + 각 변형의 자체 목표 인코딩, 고정 초매개변수)으로 두 변형을 비교했다.

| 변형 | valid AUC |
| --- | --- |
| A: 우리 이식 그대로 | 0.9625786 |
| B: float64 유지 매핑 | 0.9655440 |
| B - A | +0.0029654 |

GBM은 원시 수치 열을 직접 분할해 정확값 정보를 부분 복원할 수 있으므로, 이 대리 추정은 회복 폭의 하한에 가깝다.
정확값 범주를 임베딩 채널로 소비하도록 설계된 RealMLP에서는 회복 폭이 더 클 것으로 본다.

실제 RealMLP의 fold별 격차도 이 추정과 정합한다(원본 GPU 실행 vs exp121 확인 실행의 파이프라인 시드 42, 같은 분할).

| fold | 원본 실행 | exp121 | 차이 |
| --- | --- | --- | --- |
| 1 | 0.9675323 | 0.9630376 | -0.0044947 |
| 2 | 0.9682674 | 0.9635799 | -0.0046875 |
| 3 | 0.9683223 | 0.9639027 | -0.0044196 |
| 4 | 0.9687628 | 0.9645222 | -0.0042406 |
| 5 | 0.9679113 | 0.9632027 | -0.0047086 |

다섯 fold 모두에서 균일하게 약 -0.0045가 나는 양상은 잡음이 아니라 체계적 특성 손실과 일치한다.

## 수정 후보 순위

1. dtype 정합 복원: `transform`에서 수치-범주 어휘 매핑과 bin 변환을 float64 값으로 수행하고, float32 형 변환은 매핑이 끝난 뒤로 옮긴다(또는 `fit`의 어휘 구축까지 float32로 통일한다). 추정 효과 +0.003 ~ +0.0045로 격차의 대부분이며, bin 경계 이동도 같은 수정으로 함께 해소된다.
2. `placebo_noise` 수치 입력 1열: 저장소 필수 열이며 exp121 실측 순열 중요도가 +-0.00003 수준이라 추정 효과 |Δ| <= 0.0002다. 유지한다.
3. 두 초기화 평균(원본은 fold당 1개 초기화): 우리 쪽이 유리한 방향의 차이라 음의 격차를 설명할 수 없다. 유지한다.
4. 목표 인코딩 내부 시드 정체성(원본 SEED+fold vs 우리 해시 기반), 결정론 설정(TF32 차단, 결정론 알고리즘, cudnn benchmark 끔), 배치 순서 난수원 차이: 전부 잡음 수준(+-0.0002 이하)으로 추정한다.

## 기각된 후보

- 목표 인코딩 cv 전달 방식: 원본은 `cv=5`(정수)와 `shuffle`/`random_state` 인자를 쓰고 우리는 `StratifiedKFold` 객체를 넘기는데, sklearn 1.9.0에서 같은 시드일 때 두 방식의 출력이 동일함을 확인했다.
- 아키텍처와 학습 절차: PBLD 임베딩(차원·주파수·PReLU), 한 판 8모형 병렬 구조, 매개변수 그룹 5종과 배율, flat_cos 반쪽 스케줄(4/8), 배치 구성과 진행도 계산, 라벨 평활 손실과 클래스 가중, EMA 갱신·epoch 말 적재, 기울기 절단, softmax 평균 예측까지 줄 단위 대조로 원본과 동일함을 확인했다. 이는 이슈 230의 레시피 재대조 결론과도 일치한다.

## 파급 범위: 다른 구현은 같은 결함이 없다

같은 결함 계열(어휘를 만든 dtype과 조회하는 dtype의 불일치)이 다른 경로에도 있는지 저장소 전체를 확인했다.

- 공용 특성 경로 `src/pipeline/features.py`의 정확값 TE/CE/PairCE는 값을 문자열 키로 바꿔(`_exact_keys`의 `astype(str)`) 매핑하므로 dtype 불일치가 구조적으로 불가능하다. 이 경로를 쓰는 CatBoost·LightGBM 계열 실험(exp070, exp071, exp117 등)은 재실험이 필요 없다.
- `lookup_transformer`는 원시 dtype 그대로 어휘를 만들고 같은 dtype 값으로 조회하며, 이슈 128 실측(exp067 기준 미등록률 0.0174%)으로 정상 동작이 이미 검증돼 있다.
- `contextualized_spline_transformer`, `tab_cnn`, `tabr_s`, `scalar_token_transformer`, `amformer`, `trompt`, `tabpfn3`, `logistic_onehot`은 어휘 구성과 조회가 모두 같은 dtype의 원시 값에서 이뤄지고, float32 형 변환은 매핑이 끝난 뒤 수치 채널에만 적용된다.
- `tabm`은 pytabkit 변환기가 pandas category dtype 열만 범주로 다루므로 동일하게 안전하다.
- realmlp 모듈의 소비자는 `model.py`의 realmlp adapter(`kind: realmlp` 설정은 `exp121_realmlp_fixed4_two_init` 하나)와 테스트뿐이다.

bin 밀림 쪽도 같은 방식으로 전수 확인했다.
경계에서 불연속인 이산 구간화(`KBinsDiscretizer`)를 쓰는 곳은 realmlp뿐이다.
다른 경계 연산은 결함이 성립하지 않는다.
`QuantileTransformer`를 쓰는 네 adapter(tab_cnn, scalar_token_transformer, lookup_transformer, tabr_s)는 적합과 변환의 dtype이 서로 일치하고 출력도 경계에서 연속이다.
원본 prior의 경험적 CDF 차(`features.py`)와 앙상블의 경험적 CDF 표현(`ensemble.py`)은 양쪽 모두 float64로 통일돼 있다.
spline의 `torch.searchsorted`는 마디에서 연속인 조각별 선형 기저라 경계 배정이 바뀌어도 출력이 달라지지 않는다.

## 수정 시 재실험 필요 항목

수정의 파급은 exp121 계열에 한정되며, 실행 개폐와 순서는 [exp121 개선 실험 개폐 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/234)이 정한다.

1. exp121 재실행: dtype 정합 수정만 반영한 단일 델타 짝비교(3 파이프라인 시드, GPU).
2. 후보 풀 갱신: `artifacts/pool.yaml`의 exp121 항목(run `56701722`)을 수정판으로 교체 판정하고 최근접 구성원 중복 관문을 재확인한다.
3. 전체 자료 재학습 계획 갱신: `artifacts/full-refit-plan.yaml`의 exp121 항목(run id와 fold_median 예산 `{42: 5, 43: 5, 44: 5}`)을 수정판 실행 기준으로 재산정한다.
4. [보강 풀의 nested 재평가와 결합 전략 재선정](https://github.com/tmheo/predicting-smartphone-addiction/issues/187)은 아직 열려 있으므로, exp121 수정판을 먼저 반영한 뒤 한 번만 실행하는 순서가 중복 실행을 피한다.
5. [schedule_epochs=4 완주 짝비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/232)의 어닐링 델타는 죽은 채널 상태의 이식 위에서 측정되므로, 수정판 기반으로 재측정하거나 근사값으로만 해석해야 한다.

제출물 영향은 없다.
exp121은 2026-08-19에 풀에 들어왔고, 이슈 180은 exp121로 새 제출을 만들지 않았다.

## 재현 방법

fold 1 실증은 다음 절차로 재현한다.

```python
from sklearn.model_selection import StratifiedKFold
from pipeline.realmlp import _FoldFeatureEngineer

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, valid_idx = next(iter(outer.split(np.zeros(len(y)), y)))
engineer = _FoldFeatureEngineer().fit(X_raw.iloc[train_idx].copy())
engineer.unknown_value_count(engineer.transform(X_raw.iloc[valid_idx].copy()))
# -> 800896; transform의 float32 형 변환을 매핑 뒤로 옮기면 -> 23
```

수정 실험의 실행 여부와 설계는 [exp121 개선 실험 개폐 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/234)이 정한다.
