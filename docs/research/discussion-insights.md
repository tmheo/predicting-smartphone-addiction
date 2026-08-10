# S6E8 디스커션 인사이트 종합

Kaggle Playground Series S6E8 (Predicting Smartphone Addiction) 대회 디스커션 25개 스레드 전체를 읽고, 모델링에 활용할 수 있는 인사이트를 주제별로 정리한 문서다.
스냅샷 기준일은 2026-08-10이고, 대회는 2026-08-31까지 진행되므로 이후 올라온 스레드는 반영되어 있지 않다.

원자료는 세 개의 리딩 노트다.

- 전수 목록과 미식별 스레드 11개: [이슈 #2](https://github.com/tmheo/predicting-smartphone-addiction/issues/2), `research/discussion-inventory` 브랜치의 `docs/research/discussion-inventory.md`
- 배치 A (합성 데이터 포렌식 / 결측치 / 리더보드 분석 5개): [이슈 #3](https://github.com/tmheo/predicting-smartphone-addiction/issues/3), `research/discussion-batch-a` 브랜치의 `docs/research/discussion-batch-a.md`
- 배치 B (피처 엔지니어링 / 모델링 / 커뮤니티 9개): [이슈 #4](https://github.com/tmheo/predicting-smartphone-addiction/issues/4), `research/discussion-batch-b` 브랜치의 `docs/research/discussion-batch-b.md`

상충하는 주장은 양쪽을 병기하고 어느 쪽 근거가 강한지 표시했다.
정리 마지막의 [상충 주장 판정표](#상충-주장-판정표)에 모아 두었다.

## 1. 합성 데이터 생성기 특성

### 원본의 타깃 생성 룰과 이론적 상한

- 사라진 "원본" 7,500행 데이터의 타깃은 사실상 하드 룰이다: `daily_screen_time_hours > 8` 또는 `social_media_hours > 4`면 p=1, `daily <= 6`이고 `social <= 4`면 p=0, 중간 밴드(6 < daily <= 8, social <= 4)는 p=0.5의 순수 노이즈다 ([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428)).
- 중간 밴드 1,025행에 XGBoost를 돌리면 AUC 0.510 ± 0.033으로, 그 영역에는 원리적으로 학습할 신호가 없다 ([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428)).
- 이 베이즈 최적 모델은 원본 전체에서 AUC 0.9888이 나온다.
  이 대회 점수의 이론적 상한 구조를 보여주는 수치다 ([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428)).

### 생성기가 하드 룰을 매끄러운 확률장으로 바꿨다

- 합성 생성기는 원본의 하드 룰을 매끄럽고 보정된(calibrated) 확률장으로 바꿔 놓았다.
  같은 룰이 원본에서는 AUC 0.9888이지만 합성 데이터에서는 0.835로 떨어지고, 원본에서 노이즈였던 중간 구간이 합성에서는 AUC 0.896짜리 신호 영역이 됐다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).
- 따라서 원본의 하드 룰을 그대로 이식하거나 룰 기반 후처리를 하면 상한 0.835에 걸린다.
  모델링 대상은 원본 세계가 아니라 생성기가 만든 확률장이다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).
- OOF 예측이 대각선에 놓이므로 합성 라벨은 매끄러운 확률장에서의 베르누이 추출이다.
  중복 행이 0개라 중복 매칭류 누수 트릭은 시간 낭비다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).

### 값 격자가 이산적이라 "정확한 값"이 강한 키다

- `sleep_hours`, `notifications_per_day`, `app_opens_per_day` 등의 분포는 매끄러운 연속 분포가 아니라 이산 격자 패턴이며, 생성기가 원본의 이산 아티팩트를 그대로 복제했다 ([734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)).
- 같은 방향의 독립 증거가 셋 더 있다: 전 피처를 범주형으로 취급한 Keras 베이스라인의 임베딩 공간 클래스 분리 ([732358](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732358)), 전 피처 one-hot + 결합 피처 로지스틱 회귀의 OOF 0.9601 ([733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708) 코멘트), 정확값 타깃 인코딩 +0.0032 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).
- 정확값 인코딩이 통하는 이유는 스마트폰 도메인과 무관하다.
  데이터가 격자 위에 생성/반올림되어 있어 정확한 값이 그 격자를 집어내는 것뿐이다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495)).

### 생성기의 산술 결함이 유일하게 돈이 되는 구조다

- 대회 데이터의 26%는 원본의 `weekend_screen_time` 제약(평일 스크린타임의 1.044~1.965배)을 위반하는, 존재할 수 없는 조합이다.
  다만 train/test에서 비율이 동일해 누수로 쓸 거리는 없고, ratio 피처를 LightGBM에 줘도 시드 편차 수준의 널 결과다.
  트리가 이미 그 영역을 스스로 파내고 있다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).
- 반면 생성기 내부 산술이 안 맞는 행을 노출하는 `other_screen = daily - (social + gaming + work)` 잔차는 단독 AUC 0.765의 실질 피처다 (자세한 수치는 [4장](#4-피처-엔지니어링)).
- 데이터의 addiction 경계는 비단조다.
  저 social 밴드에서 addicted 비율이 내려갔다 올라가는 dip이 Wilson 신뢰구간으로 유의하고 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트), envelope 안팎의 타깃 발화율이 81% 대 25%로 혹(hump) 형태다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).
  단조 모델과 monotone 제약이 손해 보는 구조적 이유다.

### 원본 데이터의 정체와 활용

- 대회가 가리키는 원본 데이터셋은 접속 불가지만, 그 원본조차 합성 데이터다.
  Kaggle에 남은 7,500행 사본(jayjoshi37 등)이 df.describe() 통계 일치로 원본 프록시로 확인됐다 ([731719](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731719)).
- 원본 실데이터를 훈련에 섞는 것은 실패가 확인됐다.
  7,500행을 50배 가중치로 주입하자 10개 폴드 전부가 베이스라인 아래로 떨어졌다 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)).
  상위권(Tilii)의 조언도 원본은 훈련이 아니라 검증 참고 용도로만 쓰라는 것이다.

## 2. 결측치 신호

이 주제는 디스커션에서 가장 많이 다뤄졌고, 결론이 여러 스레드의 교차 검증으로 수렴했다.

### 결측은 타깃과 독립이다

- 결측 컬럼 개수(n_missing)의 단독 AUC는 0.502로 타깃 신호가 전무하다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983), [732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256) 코멘트, [732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트에서 각각 독립 측정).
- 카이제곱 검정 전수 확인에서도 12개 컬럼 중 `app_opens_per_day`만 p=0.025로 유의했으나 Cramer's V가 0.0027로 효과 크기는 사실상 0이다 ([731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764) 코멘트).
- 원본에 결측이 없었으므로 합성 후 무작위로 제거된 것이며, 결측 구조에 설계된 신호가 있을 가능성은 통계적으로 매우 낮다 ([731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764) 코멘트).

### 그러나 결측률은 train/test에서 다르다

- 12개 전 컬럼에서 train과 test의 결측률이 다르다.
  최대 3.4pp(`social_media_hours` 19.38% 대 16.00%), z 값 13~44로 우연이 아니고, 방향도 섞여 있어 단일 삭제율로는 설명되지 않는다 ([732427](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732427)).
- 따라서 is_missing 플래그는 타깃 신호 없이 split 소속 정보만 인코딩한다.
  로컬 CV를 미세하게 올리고 LB에서 배신하는 전형적 공변량 시프트 함정이다.
  실측 사례: missing_count가 로컬 OOF +0.00009, Public LB는 하락 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256) 코멘트).

### train/test 이동은 결측이 전부다

- 987,671행 풀링 adversarial validation에서 원시 피처의 분리 AUC는 0.564지만, 결측을 대치하면 0.503, 완전한 행만 쓰면 0.498(95% CI가 0.5 포함)로 붕괴한다.
  값의 분포 이동은 없고, 존재하는 이동은 전부 결측 패턴이다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214), 코멘트에서 독립 재현됨).
- 따라서 drift 보정 기법(adversarial reweighting 등)은 불필요하고, adversarial validation을 돌릴 때 결측 지표가 만드는 분리력을 피처 드리프트로 오독하면 안 된다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214), [732427](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732427)).

### 실무 결론: NaN은 그대로 둔다

- 통제된 ablation에서 타깃 모델 OOF AUC는 플래그 없음 0.962806, 수치 플래그 9개 추가 0.962804, 범주 플래그 3개 추가 0.962761이다.
  어떤 결측 플래그도 도움이 안 되고 미세하게 해롭다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) 코멘트).
- 결측은 채우지 말고 NaN 그대로 트리 모델에 넘긴다.
  XGBoost/LightGBM은 결측 행의 분기 방향을 게인 기준으로 스스로 학습한다 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541), [731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764) 코멘트).
- 상충 주장: "결측 자체가 시그널이므로 is_missing 지표로 LB 0.965+를 얻었다"는 스레드가 있다 ([732955](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732955)).
  그러나 ablation 수치가 없는 정성적 주장이고, 코멘트에서 상위권(Tilii)이 정반대 결과들을 지적했다.
  통제된 ablation을 제시한 [733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) 쪽 근거가 압도적으로 강하다.

## 3. CV-LB 안정성과 리더보드 해석

### 리더보드의 분해능

- 공개 LB에 같은 점수로 표시되는 팀들은 동점이 아니다.
  Kaggle은 full precision으로 랭킹하고 소수점 5자리는 표시용 반올림이며, 0.97086 부근에서 랭크 1계단의 비용은 약 4e-07 AUC다 ([733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618)).
- 팀 간 비교의 paired sigma는 0.00009~0.00011로, 진리값 대비 표준오차(약 0.00066)보다 6~7배 작다.
  1위 대 50위는 약 3 시그마로 실제 차이지만, 같은 모델의 시드 두 개는 절반 확률로 순위가 뒤집히고, 10~100위 구간에서는 순수 시드 노이즈 ±1 시그마가 약 60팀 범위다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)).
- 리더보드 분해능 공식: sd(gap) = sd(move) * sqrt(2(1 - rho)).
  비슷한 블렌드끼리의 95% 분해 가능 차이는 ~0.00015 수준까지 좁아진다.
  예측 벡터의 상관이 아니라 AUC 추정치의 상관을 넣어야 하며, 전자를 넣으면 낙관적으로 치우친다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) 코멘트).

### best-of-N 함정과 제출 규율

- Kaggle은 best 공개 점수를 유지하므로 개선 없는 재제출도 잃을 것 없는 코인 플립이고, 겉보기 랭크 상승의 절반은 실력이 아니라 best-of-N 효과다 ([733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618)).
- 실용 규칙: 어떤 변경이 순위를 40계단 올렸어도 public 점수 변화가 0.0001 미만이면 아무것도 측정한 게 아니다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)).
- Public LB는 테스트의 20%로 채점되므로 미세 차이를 분간하지 못한다.
  의사결정은 OOF 기준으로 한다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).

### private 재편에 대한 전망

- private split은 public의 약 4배 크기라 진짜 CV 우위가 살아남을 확률이 높다 ([733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618)).
- 값의 분포 이동이 없으므로(2장), private 재편 위험은 분포 이동이 아니라 노이즈와 public 과적합에서 온다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214)).
- 과거 AUC 대회(S6E3 등)에서 public LB는 private 배치를 대체로 예측했다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005) 코멘트).
- 종합하면: 최종 제출은 CV 기준으로 고르되, CV와 public이 함께 오르는 변경만 채택하는 보수적 운영이 합리적이다.

## 4. 피처 엔지니어링

### 수치로 검증된 효과 사다리

같은 베이스라인 위에서 5-fold OOF로 재측정된 개선 폭 순위다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트, Georgy Mamarin).

1. 각 컬럼을 크기가 아니라 정확한 값으로 타깃 인코딩: **+0.0032** (tomasa2가 +0.0023으로 독립 측정).
2. 모델 용량 확장 (63 leaves/400 rounds에서 255/1500 + 낮은 학습률): **+0.0012**.
3. slack + 관측 성분 개수 피처: **+0.00071** ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트).
4. `other_screen = daily - (social + gaming + work)` 잔차: **+0.00058~0.00074**, 단독 AUC 0.765 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트에서 복수 독립 재측정, [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
5. 공개 OOF 라이브러리 상위 10개 모델 평균: **+0.0005**.
6. 손수 만든 행동 피처: **+0.0002 이하 또는 음수**.

- 타깃 인코딩은 반드시 폴드 안에서 적합해야 한다.
  전체 train으로 적합하면 검증 점수가 가짜로 뛴다 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트).
  10-fold를 쓰면 조회 테이블이 데이터의 90%로 계산되어 분포 꼬리의 희소 격자점 추정이 안정된다 ([734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)).

### 실패하는 피처와 성공하는 피처의 구분

- 트리가 이미 도달 가능한 경계를 재표현하는 피처(threshold, 선형 결합, 비율, 차이)는 실패한다.
  성공하는 피처는 트리가 한 번의 분할로 못 하는 컬럼 간 산술, 즉 데이터 생성 방식의 구조를 인코딩한 것뿐이다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트, [733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트, [732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256)).
- EDA 시각화가 예쁜 피처와 모델에 유효한 피처는 다르다.
  KDE에서 클래스가 갈려 보인 sleep_deficit, 0.80 상관 컬럼을 합친 total_weekly_screen_time 모두 CV를 떨어뜨렸다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223)).
- `gaming_hours`와 `work_study_hours`는 자체 중독 신호가 없는데도 강한 컬럼을 조건으로 +0.00380을 기여한다.
  중독 신호가 아니라 강한 컬럼에 대한 산술적 사실을 운반하는 통로다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트).
- 상충 주장: screen_time_bin 구간화, weekend_gap, 비율 피처로 LB를 0.96514에서 0.96602까지 올렸다는 보고가 있다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985)).
  그러나 같은 스레드의 "저중요도 피처 제거 시 0.965에서 0.894로 폭락" 주장이 코멘트의 독립 재현에서 실패했고(0.9491에서 0.9489로 거의 무변화), 비율 피처 무익은 통제된 ablation 다수가 확인했다 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256), [732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223), [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
  ablation 쪽 근거가 강하므로 비율 피처는 기각 후보로 놓고 시작한다.
- 소수점 자릿수(_decimals) 피처는 판정이 갈린다.
  널 임포턴스 검사에서 노이즈 기준선의 11.68배 게인으로 유효하다는 측정 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541))과, other_screen 잔차와 함께 넣으면 한계 기여가 거의 0이라는 정밀 재측정 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트)이 있다.
  후자가 한계 기여를 직접 쟀으므로 더 강하다.
  넣더라도 잔차 대비 단독 기여를 따로 측정한다.

### 금지 목록

- 결측 관련 피처(is_missing, missing_count)는 전부 배제한다 (2장 참조).
- monotone 제약은 이 데이터에서 금지에 가깝다.
  스크린 컬럼 3개에 걸었더니 OOF -0.0034였고 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트), 데이터가 실제로 비단조라는 구조적 근거도 있다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).
- 원본의 하드 룰 이식과 룰 기반 후처리는 상한 0.835짜리 함정이다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).

### 신호의 집중

- 신호는 사실상 3~5개 피처(`daily_screen_time_hours`, `social_media_hours`, `weekend_screen_time` 중심)에 집중되어 있다.
  베이지안 네트워크, 조건부 상호정보량, 트리 실험 세 갈래가 교차 확인했고, 3피처 세트가 12피처 전체와 CV가 거의 같다 ([733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708)).
- LightGBM Gain 중요도에서도 원본 5개 피처가 압도한다.
  `stress_level`, `academic_work_impact`, `gender`, `age`, `sleep_hours` 등은 어떤 조건에서도 정보량이 0 근처다 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256), [733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708)).
- 단, 단일 피처 AUC 순위표는 믿지 않는다.
  `work_study_hours`가 단독 0.65로 높아 보였지만 강한 컬럼을 고정한 슬라이스 안에서는 부호조차 유지하지 못했다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).

## 5. 모델 선택과 앙상블

### 단일 모델

- GBM 계열(LightGBM/XGBoost)이 기본이고, 용량을 키우는 것이 피처 하나보다 크다 (+0.0012, 4장 사다리 참조).
- n_estimators는 튜닝 대상이 아니다.
  아주 큰 값을 넣고 early stopping을 쓴다.
  learning_rate는 탐색 단계에서 0.02~0.05로 고정하고 최종 런에서만 0.01 이하로 낮춘다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트, Tilii).
- 하이퍼파라미터 튜닝은 일찍 수확 체감에 도달한다.
  Optuna 25 트라이얼로 LB 0.96572가 나왔고, 100에서 200으로 늘려도 0.003~0.004% 수준이다.
  튜닝 예산은 소규모로 잡고 남는 시간을 피처 검증에 쓴다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트).
- Kaggle 환경의 LightGBM은 CUDA 빌드가 아니므로 GPU 가속은 XGBoost에서만 가능하다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트).
- 클래스 불균형(71/29)은 AUC 지표 하에서 무시해도 된다.
  SMOTE, 리샘플링, 재가중 모두 불필요하다 ([731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764)).
- 전 피처 one-hot 로지스틱 회귀도 0.96까지 나오므로, 해석 가능한 서브모델이나 스태킹의 다양성 소스로 쓸 수 있다 ([733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708) 코멘트).
- 단조 모델(로지스틱 회귀, monotone 제약 부스터)은 데이터의 비단조 혹 구조를 표현할 수 없다는 한계를 감안하고 쓴다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).

### 앙상블과 스태킹

- 상위권 점수(0.970+)는 사실상 스태킹 없이는 어렵다.
  공개 노트북들의 OOF/테스트 예측을 메타 피처로 쓰고 다항 상호작용까지 얹은 "단일" XGBoost가 CV 0.96947 / LB 0.97059를 냈다.
  용어상 이것은 passthrough 스태킹이다 ([733023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023)).
- 따라서 모든 실험 런에서 OOF 예측과 테스트 예측을 처음부터 저장하는 파이프라인이 필수다 ([733023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023)).
- AUC 지표에서 모델 블렌딩은 순위 평균(rank averaging)이 산술 평균이나 SLSQP 가중 최적화보다 안전한 기본값이다.
  확률 스케일 차이를 무력화한다 ([734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)).
- 시드 앙상블(여러 시드 평균)은 시드 노이즈(순위 ±60계단 수준)를 줄여 순위 안정화에 실질적으로 기여한다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)).

## 6. 검증 위생과 실험 방법론

- 피처 실험은 단발 CV가 아니라 누적 ablation + 적대적 검증 + 널 임포턴스의 3중 검증 체계로 한다 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
- 플라시보 피처(무작위 값 + 동일 결측 패턴) 하나를 상시로 넣어, +0.0003 수준의 이득이 진짜인지 폴드 노이즈인지 판별한다.
  실측: placebo는 -0.00002였다 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트).
- ablation 사다리의 각 단계가 같은 조건인지 확인한다.
  중간 단계에 피처가 끼어 있어 수치가 틀렸다가 스스로 정정된 사례가 있다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).
- 실험 결과 해석 시 "가중치/사용법 오류" 가설을 먼저 배제한 뒤에 데이터 자체를 탓한다 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)).
- 단일 실험 결과는 여러 스토리를 동시에 지지하므로, 대안 가설을 테스트하기 전에는 결론을 내리지 않는다 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)).

## 7. 기타

- 제출은 확률(predict_proba)로 한다.
  AUC는 랭킹 지표라 0/1로 이진화하면 점수를 잃고, 확률이 [0, 1] 스케일일 필요도 없다 ([732503](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732503)).
- 도메인 심리학(노모포비아 등) 기반 피처 엔지니어링은 이 대회에서 우선순위가 낮다.
  라벨은 임상 진단이 아니라 합성 생성물이고, 점수를 내는 것은 생성기 아티팩트 기반 피처다 ([731755](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731755), [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
- 이 대회 디스커션에는 AI 생성 저품질 답변 스팸이 많다.
  지난 플레이그라운드 상위 솔루션과 Chris Deotte의 글이 더 나은 학습 자료다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).

## 상충 주장 판정표

| 쟁점 | 주장 A | 주장 B | 판정 |
| --- | --- | --- | --- |
| 결측 플래그 | 시그널이므로 추가하면 LB 상승 ([732955](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732955)) | 타깃 신호 없음, split 정보만 인코딩, 무익 내지 해로움 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214), [732427](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732427), [732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256)) | **B**. A는 ablation 없는 정성 주장, B는 통제된 ablation 복수 재현 |
| 비율/구간화 피처 | LB 0.96514에서 0.96602로 상승 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985)) | 트리가 이미 아는 경계의 재표현이라 무익 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256), [732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223), [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)) | **B**. A 스레드는 다른 주장(피처 제거 폭락)의 재현도 실패했고, B는 통제 실험 다수 |
| 저중요도 피처 제거 | 제거 시 0.965에서 0.894로 폭락 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985)) | 재현 시 0.9491에서 0.9489로 거의 무변화 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트) | **B**. 다만 피처 제거는 자체 CV로 직접 검증 후 결정 |
| _decimals 피처 | 널 임포턴스 기준선의 11.68배로 진짜 신호 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)) | 잔차 피처와 함께 넣으면 한계 기여 거의 0 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트) | **B 우세**. 한계 기여를 직접 측정. 단독 기여는 따로 재측정 가치 있음 |
| 원본 데이터 활용 | 훈련 주입 (가중치 스윕 계획) ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)) | 검증/분포 참고 용도만 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552) 코멘트, [732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)) | **B**. 50배 가중치 실험에서 전 폴드 하락, 작성자도 계획 철회 |

## 실행 요약: 파이프라인 기본값

디스커션 전체에서 수렴한, 우리 첫 파이프라인의 기본값이다.

1. 데이터: NaN 그대로, 대치 없음, 결측 피처 없음, 원본 데이터 훈련 미사용.
2. 검증: Stratified K-Fold (10-fold 권장), 플라시보 피처 상시 포함, 모든 런에서 OOF/테스트 예측 저장.
3. 피처: 정확값(문자열화) 타깃 인코딩(폴드 내 적합)을 최우선으로, other_screen 잔차와 slack 계열을 추가.
   선형 결합/비율/결측 피처는 만들지 않는다.
4. 모델: LightGBM/XGBoost 고용량(255 leaves급, early stopping, 학습률 고정 후 최종만 하향), monotone 제약 금지, 불균형 대응 없음.
5. 앙상블: 시드 앙상블 + 순위 평균 블렌딩, 여력이 되면 OOF 메타 피처 스태킹.
6. 제출: 확률 제출, 최종 선택은 CV 기준, public 0.0001 미만 변화는 노이즈로 취급.
