# S6E8 디스커션 인사이트 종합

Kaggle Playground Series S6E8 (Predicting Smartphone Addiction) 대회 디스커션 25개 스레드 전체를 읽고, 모델링에 활용할 수 있는 인사이트를 주제별로 정리한 문서다.
스냅샷 기준일은 2026-08-10이고, 대회는 2026-08-31까지 진행되므로 이후 올라온 스레드는 반영되어 있지 않다.
이후 올라오는 스레드는 증분 업데이트 절차(`docs/agents/discussion-update.md`)에 따라 이 문서에 반영한다.

원자료는 세 개의 리딩 노트다.

- 전수 목록과 미식별 스레드 11개: [이슈 #2](https://github.com/tmheo/predicting-smartphone-addiction/issues/2), `research/discussion-inventory` 브랜치의 `docs/research/discussion-inventory.md`
- 배치 A (합성 데이터 포렌식 / 결측치 / 리더보드 분석 5개): [이슈 #3](https://github.com/tmheo/predicting-smartphone-addiction/issues/3), `research/discussion-batch-a` 브랜치의 `docs/research/discussion-batch-a.md`
- 배치 B (피처 엔지니어링 / 모델링 / 커뮤니티 9개): [이슈 #4](https://github.com/tmheo/predicting-smartphone-addiction/issues/4), `research/discussion-batch-b` 브랜치의 `docs/research/discussion-batch-b.md`

상충하는 주장은 양쪽을 병기하고 어느 쪽 근거가 강한지 표시했다.
정리 마지막의 [상충 주장 판정표](#상충-주장-판정표)에 모아 두었다.

## 1. 합성 데이터 생성기 특성

### 원본의 타깃 생성 룰과 이론적 상한

- 사라진 "원본" 7,500행 데이터의 타깃은 사실상 임계값 몇 개로 기계적으로 정해진다: `daily_screen_time_hours > 8` 또는 `social_media_hours > 4`면 p=1, `daily <= 6`이고 `social <= 4`면 p=0, 중간 구간(6 < daily <= 8, social <= 4)은 p=0.5의 순수한 동전 던지기다 ([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428)).
- 중간 구간 1,025행에 XGBoost를 돌리면 AUC 0.510 ± 0.033으로, 그 영역에는 원리적으로 학습할 신호가 없다 ([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428)).
- 이 규칙을 그대로 예측기로 쓰는 것이 이론상 가능한 최선의 모델인데, 원본 전체에서 AUC 0.9888이 나온다.
  이 대회 점수가 어디까지 오를 수 있는지 상한을 보여주는 수치다 ([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428)).

### 생성기가 딱 떨어지던 규칙을 완만한 확률 변화로 바꿨다

- 원본에서는 임계값을 넘느냐 마느냐로 라벨이 거의 결정됐지만, 합성 데이터에서는 피처 값에 따라 addicted 확률이 0과 1 사이를 완만하게 오르내리는 구조로 바뀌었다.
  그 증거로, 같은 규칙이 원본에서는 AUC 0.9888이지만 합성 데이터에서는 0.835로 떨어지고, 원본에서 순수 동전 던지기였던 중간 구간이 합성에서는 AUC 0.896짜리 신호 영역이 됐다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).
- 따라서 원본의 임계값 규칙을 그대로 가져오거나 규칙 기반 후처리를 하면 상한 0.835에 걸린다.
  모델이 배워야 할 대상은 원본의 규칙이 아니라 생성기가 만든 이 완만한 확률 구조다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).
- OOF 예측의 보정 곡선이 대각선 위에 놓인다.
  즉 합성 라벨은 행마다 정해진 확률로 동전을 던져 뽑은 결과이고, 모델이 그 확률 자체는 이미 잘 맞히고 있다는 뜻이다.
  또 중복 행이 0개라, train과 test에서 똑같은 행을 찾아 라벨을 베끼는 식의 누수 트릭은 시간 낭비다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).

### 값이 정해진 눈금 위에만 있어, 정확한 값 자체가 강력한 단서다

- `sleep_hours`, `notifications_per_day`, `app_opens_per_day` 등의 값은 연속적으로 고르게 퍼져 있지 않고 특정 값들에만 몰려 있으며, 생성기가 원본의 이런 값 패턴을 그대로 복제했다 ([734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)).
- 같은 방향의 독립 증거가 셋 더 있다: 전 피처를 범주형으로 취급한 Keras 베이스라인의 임베딩 공간 클래스 분리 ([732358](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732358)), 전 피처 one-hot + 결합 피처 로지스틱 회귀의 OOF 0.9601 ([733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708) 코멘트), 정확값 타깃 인코딩 +0.0032 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).
- 정확값 인코딩이 통하는 이유는 스마트폰 도메인과 무관하다.
  데이터가 특정 값들 위에서 생성되거나 반올림되어 있어, 정확한 값이 같은 값에서 나온 행들을 묶어 주는 것뿐이다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495)).

### 점수로 이어지는 것은 사실상 생성기의 산술 오류뿐이다

- 대회 데이터의 26%는 원본의 `weekend_screen_time` 제약(평일 스크린타임의 1.044~1.965배)을 위반하는, 존재할 수 없는 조합이다.
  다만 train/test에서 비율이 동일해 누수로 쓸 거리는 없고, ratio 피처를 LightGBM에 줘도 시드를 바꿀 때 생기는 오차 범위 안이라 효과가 없다.
  트리 모델이 분기 과정에서 이미 그 영역을 스스로 구분해 내고 있기 때문이다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).
- 반면 생성기의 내부 계산이 안 맞는 행을 드러내는 `other_screen = daily - (social + gaming + work)` 잔차는 단독 AUC 0.765의 실질 피처다 (자세한 수치는 [4장](#4-피처-엔지니어링)).
- 이 데이터에서 addicted 비율은 사용량이 늘어난다고 한 방향으로만 움직이지 않는다.
  social_media_hours가 낮은 구간에서 addicted 비율이 한 번 내려갔다가 다시 올라가는 굴곡이 있고, 이 굴곡은 Wilson 신뢰구간 기준으로 통계적으로 유의하다 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트).
  또 주말/평일 비율이 weekend_screen_time 제약 범위(1.044~1.965배) 안이면 addicted 비율이 81%인데, 2.5배를 넘는 구간에서는 25%까지 떨어진다.
  비율이 커질수록 addicted 비율이 계속 오르는 게 아니라 중간에서 솟았다가 다시 내려오는 모양이다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).
  예측값이 피처와 같은 방향으로만 움직이도록 강제하는 모델(monotone 제약 등)이 이 데이터에서 불리한 이유가 여기에 있다.

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
  최대 3.4%p(`social_media_hours` 19.38% 대 16.00%), z 값 13~44로 우연이 아니고, 컬럼마다 높아지는 쪽과 낮아지는 쪽이 섞여 있어 전체에 삭제율 하나를 적용한 것으로는 설명되지 않는다 ([732427](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732427)).
- 따라서 is_missing 플래그에는 타깃 정보가 없고, 그 행이 train과 test 중 어디서 왔는지에 대한 정보만 담긴다.
  이런 피처는 로컬 CV를 미세하게 올려 놓고 LB에서는 점수를 깎는 전형적인 함정이다.
  실측 사례: missing_count가 로컬 OOF +0.00009, Public LB는 하락 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256) 코멘트).

### train과 test의 차이는 결측 패턴이 전부다

- train과 test를 합친 987,671행으로 "이 행이 train인지 test인지" 맞히는 분류기를 학습시키면(adversarial validation), 피처를 그대로 쓸 때는 AUC 0.564로 어느 정도 구분이 된다.
  그런데 결측을 채우면 0.503, 결측 없는 행만 쓰면 0.498(95% 신뢰구간이 0.5 포함)로 구분 능력이 사라진다.
  즉 채워져 있는 값들의 분포는 train과 test가 같고, 둘을 구분하게 해 주던 것은 전부 "어느 칸이 비어 있는가"였다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214), 코멘트에서 독립 재현됨).
- 따라서 분포 차이를 보정하는 기법(adversarial reweighting 등)은 불필요하고, adversarial validation을 돌릴 때 결측 패턴이 만드는 구분 능력을 값의 분포 차이로 오해하면 안 된다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214), [732427](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732427)).

### 실무 결론: NaN은 그대로 둔다

- 통제된 ablation에서 타깃 모델 OOF AUC는 플래그 없음 0.962806, 수치 플래그 9개 추가 0.962804, 범주 플래그 3개 추가 0.962761이다.
  어떤 결측 플래그도 도움이 안 되고 미세하게 해롭다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) 코멘트).
- 결측은 채우지 말고 NaN 그대로 트리 모델에 넘긴다.
  XGBoost/LightGBM은 결측 행의 분기 방향을 게인 기준으로 스스로 학습한다 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541), [731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764) 코멘트).
- 상충 주장: "결측 자체가 시그널이므로 is_missing 지표로 LB 0.965+를 얻었다"는 스레드가 있다 ([732955](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732955)).
  그러나 ablation 수치가 없는 정성적 주장이고, 코멘트에서 상위권(Tilii)이 정반대 결과들을 지적했다.
  통제된 ablation을 제시한 [733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) 쪽 근거가 압도적으로 강하다.

## 3. CV-LB 안정성과 리더보드 해석

### 리더보드는 얼마나 작은 점수 차이까지 구분할 수 있나

- 공개 LB에 같은 점수로 표시되는 팀들은 동점이 아니다.
  Kaggle은 반올림 전의 전체 자릿수 점수로 순위를 매기고 소수점 5자리는 표시용 반올림일 뿐이라, 0.97086 부근에서는 AUC 약 4e-07 차이만으로도 순위가 한 계단 갈린다 ([733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618)).
- 점수 오차는 두 종류를 구분해야 한다.
  한 팀의 표시 점수가 진짜 AUC(테스트가 무한히 많다면 나올 값)에서 벗어나는 폭은 표준오차 약 0.00066이다.
  그런데 모든 팀이 같은 20% 표본으로 채점되므로, 표본이 우연히 쉽게/어렵게 뽑혀 생기는 공통 오차는 두 팀의 점수를 빼면 상쇄된다.
  그래서 두 팀의 점수 차이가 흔들리는 폭(paired sigma)은 0.00009~0.00011로, 절대 오차보다 6~7배 작다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)).
  같은 자로 두 사람의 키를 재면 자가 부정확해도 누가 더 큰지는 정확히 알 수 있는 것과 같은 원리다.
- 이 paired sigma를 자로 삼아 리더보드를 읽으면: 1위와 50위의 차이는 약 3 시그마라 노이즈로 보기 어려운 실제 실력 차이다.
  반면 같은 모델을 시드만 바꾼 두 제출은 어느 쪽이 위에 갈지 동전 던지기(50%)이고, 팀이 빽빽하게 몰린 10~100위 구간에서는 시드 노이즈 1 시그마만큼의 점수 흔들림만으로 순위가 약 60계단 움직인다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)).
- 두 제출물의 점수 차이가 흔들리는 폭은 sd(gap) = sd(move) * sqrt(2(1 - rho))로 계산한다.
  sd(move)는 제출물 하나의 점수 오차(위의 약 0.00066), rho는 테스트 표본이 바뀔 때 두 점수(AUC 추정치)가 함께 오르내리는 정도다.
  두 제출물이 비슷할수록 rho가 1에 가까워져 차이의 오차가 작아지고, 비슷한 블렌드끼리는 95% 신뢰 수준에서 구분 가능한 최소 차이가 약 0.00015까지 좁아진다.
  즉 비슷한 블렌드 두 개의 점수 차이가 0.00015보다 작으면 노이즈와 구분되지 않는다.
  주의: rho 자리에는 AUC 추정치의 상관을 넣어야 한다.
  예측 벡터끼리의 상관(보통 0.999 수준)을 넣으면 rho를 과대평가해서, 리더보드가 실제보다 미세한 차이까지 구분해 준다고 착각하게 된다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) 코멘트).

### best-of-N 함정과 제출 규율

- 리더보드에 표시되는 점수는 최신 제출이 아니라 지금까지 제출한 것 중 가장 좋은 public 점수이고, 한 번 좋은 점수가 나오면 그 뒤에 나쁜 제출을 해도 내려가지 않는다.
  각 제출 점수에는 표본 노이즈가 섞여 있으므로, 모델 개선 없이 같은 제출을 반복해도 우연히 잘 나온 만큼 표시 점수가 올라간다.
  점수가 나쁘면 표시는 그대로고 좋으면 오르니, 재제출은 잃을 것 없는 동전 던지기다.
  이렇게 N번 뽑은 것 중 최고 기록은 진짜 실력보다 좋게 나오기 마련이고(best-of-N 효과), 겉보기 순위 상승의 절반은 실력이 아니라 이 효과다 ([733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618)).
- 실용 규칙: 어떤 변경이 순위를 40계단 올렸어도 public 점수 변화가 0.0001 미만이면 아무것도 측정한 게 아니다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)).
- Public LB는 테스트의 20%로 채점되므로 미세 차이를 분간하지 못한다.
  의사결정은 OOF 기준으로 한다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).

### private 리더보드에서 순위가 얼마나 뒤집힐까

- private 채점은 public의 약 4배 크기 표본이라, 진짜 CV 우위는 살아남을 확률이 높다 ([733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618)).
- train과 test의 값 분포가 같으므로(2장), private에서 순위가 뒤집힌다면 그 원인은 분포 차이가 아니라 노이즈와 public 점수에 대한 과적합이다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214)).
- 과거 AUC 대회(S6E3 등)에서 public LB 순위는 private 최종 순위를 대체로 잘 예측했다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005) 코멘트).
- 종합하면: 최종 제출은 CV 기준으로 고르되, CV와 public이 함께 오르는 변경만 채택하는 보수적 운영이 합리적이다.

## 4. 피처 엔지니어링

### 수치로 검증된 개선 폭 순위

5-fold OOF로 잰 개선 폭 순위다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트, Georgy Mamarin).
단, 측정 조건이 항목마다 다르다.
피처 항목(3, 4, 6번)은 같은 기준 모델에 하나씩 얹어 쟀고, 용량 확장과 정확값 인코딩(1, 2번)은 누적으로(인코딩은 용량 확장 위에서) 쟀으며, OOF 라이브러리 평균(5번)은 다른 베이스에서 측정되어 나머지와 직접 비교할 수 없다고 원문에 명시돼 있다.

1. 각 컬럼을 크기가 아니라 정확한 값으로 타깃 인코딩: **+0.0032** (tomasa2가 +0.0023으로 독립 측정).
2. 모델 용량 확장 (63 leaves/400 rounds에서 255/1500 + 낮은 학습률): **+0.0012**.
3. slack + 관측 성분 개수 피처: **+0.00071** (tamerlanomralinov 작, [733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트에서 재측정).
   slack은 daily 스크린타임을 예산으로 보고 관측된(결측 아닌) 성분들의 합을 뺀 여유분을 뜻하며, 관측 성분 개수와 짝지어 성분이 일부 결측인 행에서도 산술 관계를 활용한다.
   정확한 수식은 원 노트북 미확인이라 이 해석은 스레드 맥락(ryota517의 예산 제약 틀) 기반이다.
4. `other_screen = daily - (social + gaming + work)` 잔차: **+0.00058~0.00074**, 단독 AUC 0.765 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트에서 복수 독립 재측정, [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
5. 공개 OOF 라이브러리(참가자들이 공유한 공개 노트북 모델들의 OOF/테스트 예측 모음) 상위 10개 모델 평균: **+0.0005**.
   가장 좋은 모델 하나만 쓰는 것 대비 수치이고, 균등 평균 대신 가중치를 최적화해도 약 +0.0001 더 얻는 데 그친다.
6. 손수 만든 행동 피처(도메인 직관으로 컬럼을 조합한 것): **+0.0002 이하 또는 음수**.

- slack 피처(3번)와 other_screen 잔차(4번)는 같은 예산 제약 산술의 변형이다.
  성분 셋이 모두 채워진 행에서는 두 값이 동일하고, slack + 관측 성분 개수 쪽이 성분이 일부 결측인 행까지 커버하는 확장판이다.
  위 순위는 피처를 한 번에 하나씩 얹어 잰 것이라, 둘을 같이 넣었을 때 효과가 합산된다는 보장이 없다.
  실험에서는 둘을 따로 넣어 보고 같이도 넣어 봐서 기여가 겹치는지 측정한다.
- 타깃 인코딩은 반드시 폴드 안에서 적합해야 한다.
  전체 train으로 적합하면 검증 점수가 가짜로 뛴다 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트).
  10-fold를 쓰면 값별 타깃 평균을 담는 조회 테이블이 데이터의 90%로 계산되어, 드물게 나타나는 값의 추정이 안정된다 ([734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)).

### 실패하는 피처와 성공하는 피처의 구분

- 트리가 분할을 거듭하면 스스로 만들 수 있는 경계를 다른 형태로 다시 써 주는 것에 불과한 피처(임계값, 선형 결합, 비율, 차이)는 실패한다.
  성공하는 피처는 트리가 한 번의 분할로는 만들 수 없는 컬럼 간 산술, 즉 데이터가 생성된 방식의 구조를 담은 것뿐이다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트, [733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트, [732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256)).
- EDA 시각화가 예쁜 피처와 모델에 유효한 피처는 다르다.
  KDE에서 클래스가 갈려 보인 sleep_deficit, 0.80 상관 컬럼을 합친 total_weekly_screen_time 모두 CV를 떨어뜨렸다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223)).
- `gaming_hours`와 `work_study_hours`는 단독으로는 중독 신호가 없는데도, 강한 컬럼과 조합되면 +0.00380을 기여한다.
  그 자체가 중독 신호라서가 아니라, 강한 컬럼과 얽힌 산술 관계(생성기의 계산 흔적)를 모델에 전달해 주는 역할이다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트).
- 상충 주장: screen_time_bin 구간화, weekend_gap, 비율 피처로 LB를 0.96514에서 0.96602까지 올렸다는 보고가 있다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985)).
  그러나 같은 스레드의 "저중요도 피처 제거 시 0.965에서 0.894로 폭락" 주장이 코멘트의 독립 재현에서 실패했고(0.9491에서 0.9489로 거의 무변화), 비율 피처 무익은 통제된 ablation 다수가 확인했다 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256), [732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223), [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
  ablation 쪽 근거가 강하므로 비율 피처는 기각 후보로 놓고 시작한다.
- 소수점 자릿수(_decimals) 피처는 판정이 갈린다.
  타깃을 무작위로 섞어 만든 노이즈 기준선(널 임포턴스)보다 11.68배 큰 게인이라 진짜 신호라는 측정 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541))과, other_screen 잔차가 이미 들어 있으면 추가로 주는 기여가 거의 0이라는 정밀 재측정 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트)이 있다.
  후자가 "이미 있는 피처 위에 얹었을 때 얼마나 더 주는가"를 직접 쟀으므로 더 강하다.
  넣더라도 잔차 피처와 기여가 겹치는지 따로 측정한다.

### 금지 목록

- 결측 관련 피처(is_missing, missing_count)는 전부 배제한다 (2장 참조).
- monotone 제약은 이 데이터에서 금지에 가깝다.
  스크린 컬럼 3개에 걸었더니 OOF -0.0034였고 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트), addicted 비율이 실제로 값에 따라 오르내린다는 구조적 근거도 있다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).
- 원본의 임계값 규칙을 그대로 가져오는 것과 규칙 기반 후처리는 상한 0.835짜리 함정이다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).

### 신호의 집중

- 신호는 사실상 3~5개 피처(`daily_screen_time_hours`, `social_media_hours`, `weekend_screen_time` 중심)에 집중되어 있다.
  베이지안 네트워크, 조건부 상호정보량, 트리 실험 세 갈래가 교차 확인했고, 3피처 세트가 12피처 전체와 CV가 거의 같다 ([733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708)).
- LightGBM Gain 중요도에서도 원본 5개 피처가 압도한다.
  `stress_level`, `academic_work_impact`, `gender`, `age`, `sleep_hours` 등은 어떤 조건에서도 정보량이 0 근처다 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256), [733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708)).
- 단, 피처 하나씩 잰 단독 AUC 순위표는 믿지 않는다.
  `work_study_hours`가 단독 0.65로 높아 보였지만, 강한 컬럼의 값을 고정해 놓고 그 안에서 다시 보면 타깃과의 관계가 방향조차 유지되지 않았다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).

## 5. 모델 선택과 앙상블

### 단일 모델

- GBM 계열(LightGBM/XGBoost)이 기본이고, 모델 용량(잎 수와 트리 수)을 키우는 효과가 피처 하나를 추가하는 것보다 크다 (+0.0012, 4장 개선 폭 순위 참조).
- n_estimators는 튜닝 대상이 아니다.
  아주 큰 값을 넣고 early stopping을 쓴다.
  learning_rate는 탐색 단계에서 0.02~0.05로 고정하고 최종 런에서만 0.01 이하로 낮춘다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트, Tilii).
- 하이퍼파라미터 튜닝은 조금만 해도 금방 추가 이득이 사라진다.
  Optuna 25 트라이얼로 LB 0.96572가 나왔고, 100에서 200으로 늘려도 0.003~0.004% 수준이다.
  튜닝 예산은 소규모로 잡고 남는 시간을 피처 검증에 쓴다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트).
- Kaggle 환경의 LightGBM은 CUDA 빌드가 아니므로 GPU 가속은 XGBoost에서만 가능하다 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트).
- 클래스 불균형(71/29)은 AUC 지표 하에서 무시해도 된다.
  SMOTE, 리샘플링, 재가중 모두 불필요하다 ([731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764)).
- 전 피처 one-hot 로지스틱 회귀도 0.96까지 나오므로, 해석용 보조 모델로 쓰거나 스태킹에 트리 모델과 다른 관점을 보태는 재료로 쓸 수 있다 ([733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708) 코멘트).
- 예측이 피처 값과 같은 방향으로만 움직이는 모델(로지스틱 회귀, monotone 제약을 건 부스팅)은 addicted 비율이 중간에서 솟았다가 내려오는 이 데이터의 구조를 표현할 수 없다는 한계를 감안하고 쓴다 ([733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)).

### 앙상블과 스태킹

- 상위권 점수(0.970+)는 사실상 스태킹 없이는 어렵다.
  공개 노트북들의 OOF/테스트 예측을 메타 피처로 쓰고 다항 상호작용까지 얹은 "단일" XGBoost가 CV 0.96947 / LB 0.97059를 냈다.
  원래 피처를 유지한 채 다른 모델들의 예측을 입력으로 함께 쓰는 방식이라, 용어상 이것은 passthrough 스태킹이다 ([733023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023)).
- 따라서 모든 실험 런에서 OOF 예측과 테스트 예측을 처음부터 저장하는 파이프라인이 필수다 ([733023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023)).
- AUC 지표에서 모델 블렌딩은 순위 평균(rank averaging)이 산술 평균이나 SLSQP 가중 최적화보다 안전한 기본값이다.
  모델마다 확률값의 눈금이 달라서 생기는 문제를 순위로 바꿔 없애 주기 때문이다 ([734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)).
- 시드 앙상블(여러 시드 평균)은 시드 노이즈(순위 ±60계단 수준)를 줄여 순위 안정화에 실질적으로 기여한다 ([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)).

## 6. 검증 위생과 실험 방법론

- 피처 실험은 CV 한 번의 점수로 판단하지 않는다.
  피처를 하나씩 더하고 빼며 비교하는 누적 ablation, train/test 구분 검사(adversarial validation), 타깃을 무작위로 섞어 노이즈 기준선을 만드는 널 임포턴스의 세 겹 검증으로 한다 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
- 플라시보 피처(무작위 값 + 동일 결측 패턴) 하나를 상시로 넣어, +0.0003 수준의 이득이 진짜인지 폴드 노이즈인지 판별한다.
  실측: placebo는 -0.00002였다 ([733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) 코멘트).
- 피처를 단계적으로 쌓아 가며 비교할 때는 각 단계가 정말 같은 조건에서 비교되는지 확인한다.
  중간 단계에 의도치 않은 피처가 끼어 있어 수치가 틀렸다가 나중에 정정된 사례가 있다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).
- 실험 결과 해석 시 "가중치/사용법 오류" 가설을 먼저 배제한 뒤에 데이터 자체를 탓한다 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)).
- 하나의 실험 결과는 서로 다른 여러 해석과 동시에 들어맞을 수 있으므로, 다른 가설들을 검증해 보기 전에는 결론을 내리지 않는다 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)).

## 7. 기타

- 제출은 확률(predict_proba)로 한다.
  AUC는 예측값의 순서만 보는 지표라 0/1로 이진화하면 점수를 잃고, 순서만 맞으면 제출값이 [0, 1] 범위를 벗어나도 상관없다 ([732503](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732503)).
- 도메인 심리학(노모포비아 등) 기반 피처 엔지니어링은 이 대회에서 우선순위가 낮다.
  라벨은 임상 진단이 아니라 합성 생성물이고, 점수를 내는 것은 생성기 아티팩트 기반 피처다 ([731755](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731755), [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)).
- 이 대회 디스커션에는 AI 생성 저품질 답변 스팸이 많다.
  지난 플레이그라운드 상위 솔루션과 Chris Deotte의 글이 더 나은 학습 자료다 ([733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트).

## 상충 주장 판정표

| 쟁점 | 주장 A | 주장 B | 판정 |
| --- | --- | --- | --- |
| 결측 플래그 | 시그널이므로 추가하면 LB 상승 ([732955](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732955)) | 타깃 신호 없음, split 정보만 인코딩, 무익 내지 해로움 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214), [732427](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732427), [732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256)) | **B**. A는 ablation 없는 정성 주장, B는 통제된 ablation 복수 재현 |
| 비율/구간화 피처 | LB 0.96514에서 0.96602로 상승 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985)) | 트리가 스스로 만들 수 있는 경계를 다시 쓴 것이라 무익 ([732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256), [732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223), [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)) | **B**. A 스레드는 다른 주장(피처 제거 폭락)의 재현도 실패했고, B는 통제 실험 다수 |
| 저중요도 피처 제거 | 제거 시 0.965에서 0.894로 폭락 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985)) | 재현 시 0.9491에서 0.9489로 거의 무변화 ([732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) 코멘트) | **B**. 다만 피처 제거는 자체 CV로 직접 검증 후 결정 |
| _decimals 피처 | 널 임포턴스 기준선의 11.68배로 진짜 신호 ([733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)) | 잔차 피처와 함께 넣으면 한계 기여 거의 0 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트) | **B 우세**. 한계 기여를 직접 측정. 단독 기여는 따로 재측정 가치 있음 |
| 원본 데이터 활용 | 훈련 주입 (가중치 스윕 계획) ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)) | 검증/분포 참고 용도만 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552) 코멘트, [732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)) | **B**. 50배 가중치 실험에서 전 폴드 하락, 작성자도 계획 철회 |

## 실행 요약: 파이프라인 기본값

디스커션 전체에서 수렴한, 우리 첫 파이프라인의 기본값이다.

1. 데이터: NaN 그대로, 대치 없음, 결측 피처 없음, 원본 데이터 훈련 미사용.
2. 검증: Stratified K-Fold (10-fold 권장), 플라시보 피처 상시 포함, 모든 런에서 OOF/테스트 예측 저장.
3. 피처: 정확값(문자열화) 타깃 인코딩(폴드 내 적합)을 최우선으로, other_screen 잔차와 slack 계열을 추가.
   둘은 완전한 행에서 값이 같아 기여가 겹칠 수 있으므로 따로/같이 넣어 비교한다.
   선형 결합/비율/결측 피처는 만들지 않는다.
4. 모델: LightGBM/XGBoost 고용량(255 leaves급, early stopping, 학습률 고정 후 최종만 하향), monotone 제약 금지, 불균형 대응 없음.
5. 앙상블: 시드 앙상블 + 순위 평균 블렌딩, 여력이 되면 OOF 메타 피처 스태킹.
6. 제출: 확률 제출, 최종 선택은 CV 기준, public 0.0001 미만 변화는 노이즈로 취급.

## 부록: 읽은 스레드 장부

이 문서에 반영된 디스커션 스레드의 전수 목록이다.
증분 업데이트(`docs/agents/discussion-update.md`)에서 새 스레드 식별의 기준 장부로 쓴다.
코멘트 수는 확인 시점(2026-08-10) 목록 페이지 기준이고, 재방문 표시가 있는 스레드는 회차마다 코멘트 수가 늘었는지 다시 확인한다.

| id | 제목 | 코멘트 수 | 재방문 |
| --- | --- | --- | --- |
| [734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063) | Decoding the Synthetic Generator: 0.9689+ via Stringified Target Encoding and Rank Averaging | 1 | O |
| [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005) | Changing the random seed moves you 60 places - what this leaderboard can and can't resolve | 1 | |
| [733983](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983) | A quarter of the rows describe people who can't exist - and LightGBM already knew | 0 | O |
| [733908](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733908) | An interesting competition | 1 | |
| [733730](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733730) | My finding on the data and some questions | 2 | |
| [733708](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733708) | 5 Features, Logistic Regression, ~0.945 ROC AUC | 1 | O |
| [733619](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733619) | Question about Top 3 Prizes for Teams | 0 | |
| [733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618) | 27 teams show 0.97086. None of them are tied. | 0 | |
| [733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552) | I Injected Real-World Data Into My Model and Every Single Fold Got Worse | 2 | |
| [733541](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541) | Feature Engineering: What Works, What Fails, and the Math Behind It | 2 | |
| [733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) | As a Beginner, What's the First Thing You Check in a Tabular Competition? | 9 | O |
| [733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) | There is no distribution shift in this data - only missingness. What that means for the private LB | 2 | |
| [733023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023) | Single Model Feature Engineering technique... | 4 | |
| [732985](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985) | XGBoost + Optuna on GPU \| 0.96514 LB - sharing what worked | 8 | |
| [732955](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732955) | The Signal in the Noise: Treating Missing Values as Features in S6E8 | 3 | |
| [732503](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732503) | Example Submission and Expected outputs are different | 5 | |
| [732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434) | The generator turned a hard rule into a smooth field: a forensic look at the synthetic layer | 3 | O |
| [732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428) | Generation model of the missing "original" dataset | 0 | O |
| [732427](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732427) | Train and test have different missingness rates - in all twelve columns | 0 | |
| [732358](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732358) | Baseline Keras training, all categoricals | 0 | |
| [732256](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732256) | LightGBM Gain Importance: What the Model Actually Cares About (and What it Ignores) | 2 | |
| [732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) | Plot Twist: Why My "Golden" EDA Features Dropped My CV Score (EDA vs LightGBM) | 4 | |
| [731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764) | Handling Class Imbalance & Missing Values in This Dataset | 5 | |
| [731755](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731755) | Nomophobia: No Mobile Phone Phobia on Kaggle Playground : ) | 4 | |
| [731719](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731719) | Original Dataset not available | 10 | |
