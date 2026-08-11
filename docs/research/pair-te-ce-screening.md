# 쌍 결합 TE·CE 후보 탐색 결과

이슈 [#51](https://github.com/tmheo/predicting-smartphone-addiction/issues/51)의 실행 기록이다.
탐색 규약은 [#48의 결의](https://github.com/tmheo/predicting-smartphone-addiction/issues/48)를 그대로 따랐다.

## 결론

- 12개 컬럼의 66쌍 전부를 약식 검증했고, 규약 문턱(약식 기준 실행 대비 Δ ≥ +0.0001)을 넘은 쌍은 `daily_screen_time_hours × weekend_screen_time` 하나였다.
- 이 쌍의 정식 seed 42 5-fold 스크리닝(exp017_pair_dst_wst, run `7f53d1f9a19a40eaa0abc7f490068db1`)은 OOF AUC 0.96712로 champion 0.96740 대비 -0.00028 미달이었다.
- 정식 스크리닝 통과자가 없으므로 결합 세트와 3시드 확정 재검증은 열리지 않았고, champion은 exp011_resid_pair를 유지한다.
- 삼중 결합과 자릿수 결합의 개방 조건(쌍 정식 채택 ≥ 1)이 충족되지 않아 두 트랙 모두 닫는다.

## 약식 검증 구성

- `scripts/screen_pairs.py`가 실행한다.
  커밋된 `artifacts/folds.parquet`의 fold 0을 검증 fold로, 나머지 4개 fold 전체 행을 학습에 쓴다.
- 가벼운 학습 설정은 learning_rate 0.1, early stopping 100, num_leaves 255, 내부 TE 5-fold이고 그 외는 champion 설정과 같다.
- 후보 실행 하나는 champion 피처 + 해당 쌍 TE + 해당 쌍 CE(훈련+테스트 합산 빈도 log1p) + 쌍 카나리아 TE다.
- 쌍 카나리아는 `placebo_noise × weekend_screen_time` 쌍 TE로, 66개 후보 실행 전부에서 gain이 플라시보보다 낮아 전 실행이 유효했다.

### 기준 실행 두 개

규약의 약식 기준 실행(`base_plain`, champion 피처 구성 그대로)에 더해, 쌍 카나리아를 포함한 보조 기준(`base_canary`)을 하나 더 실행했다.

| 기준 실행 | fold-0 AUC |
| --- | --- |
| base_plain (규약 기준) | 0.965778 |
| base_canary (쌍 카나리아 포함) | 0.965842 |

잡음 컬럼인 쌍 카나리아 하나를 추가한 것만으로 fold-0 점수가 +0.000064 움직였다.
이 폭이 약식 문턱(+0.0001)과 같은 자릿수라서, 후보와 피처 차이가 후보 쌍뿐인 base_canary 대비 Δ를 보조 지표로 함께 기록했다.
선별 게이트는 규약대로 base_plain 대비 Δ로 판정했다.

## 약식 결과 요약

- 66쌍 중 규약 문턱 통과 1쌍: `daily_screen_time_hours × weekend_screen_time` (Δ +0.000108).
- base_canary 대비로는 어떤 쌍도 +0.0001을 넘지 못했다(최고 +0.000043).
  약식 통과 1건도 카나리아 잡음 편차 수준이라는 신호였고, 정식 스크리닝 미달로 확인됐다.
- placebo 파생 카나리아 누수 검출은 66회 전부 통과였다.

## 정식 seed 42 스크리닝 (exp017_pair_dst_wst)

- OOF AUC 0.96712, champion(3시드 평균본) 0.96740 대비 -0.00028으로 스크리닝 미달.
- 같은 seed 42끼리 비교하면 champion 0.96705 대비 +0.00007로, 확정 문턱 +0.0001에도 못 미친다.
- 새 피처 게이트도 쌍 CE의 평균 gain(21923)이 플라시보(30298)보다 낮아 미달이었다.
  쌍 TE의 gain(103031)은 플라시보를 넘었지만 점수 기여로 이어지지 않았다.
- 보조 기록: 약식 선별에 쓴 fold 0을 제외한 4개 fold에서도 champion(3시드 평균) fold 점수 대비 전패였다(fold 1 -0.00028, fold 2 -0.00012, fold 3 -0.00029, fold 4 -0.00036).

## 부록: 66쌍 약식 전체 결과

Δ 내림차순(카나리아 포함 기준) 정렬이다.
원본 수치는 로컬 `run-logs/pair_screen.csv`에 남는다.

| 쌍 | fold-0 AUC | Δ vs 기준(카나리아 포함) | Δ vs 기준(규약) | 쌍 TE gain | 쌍 CE gain | 유효 |
| --- | --- | --- | --- | --- | --- | --- |
| daily_screen_time_hours × weekend_screen_time | 0.965886 | +0.000043 | +0.000108 | 20451 | 11683 | True |
| weekend_screen_time × stress_level | 0.965872 | +0.000030 | +0.000094 | 31503 | 12706 | True |
| social_media_hours × academic_work_impact | 0.965869 | +0.000027 | +0.000091 | 26767 | 8375 | True |
| work_study_hours × stress_level | 0.965861 | +0.000019 | +0.000083 | 11738 | 54076 | True |
| sleep_hours × stress_level | 0.965861 | +0.000019 | +0.000083 | 8765 | 5893 | True |
| work_study_hours × app_opens_per_day | 0.965857 | +0.000015 | +0.000079 | 8925 | 9947 | True |
| sleep_hours × academic_work_impact | 0.965853 | +0.000010 | +0.000075 | 12041 | 9558 | True |
| social_media_hours × stress_level | 0.965848 | +0.000006 | +0.000070 | 18158 | 9786 | True |
| notifications_per_day × weekend_screen_time | 0.965847 | +0.000004 | +0.000069 | 94354 | 8775 | True |
| daily_screen_time_hours × sleep_hours | 0.965844 | +0.000002 | +0.000067 | 6293 | 7809 | True |
| social_media_hours × sleep_hours | 0.965836 | -0.000006 | +0.000059 | 4986 | 5644 | True |
| gaming_hours × academic_work_impact | 0.965835 | -0.000007 | +0.000057 | 11530 | 7754 | True |
| age × app_opens_per_day | 0.965834 | -0.000008 | +0.000056 | 76710 | 8522 | True |
| sleep_hours × gender | 0.965831 | -0.000011 | +0.000053 | 11819 | 9850 | True |
| daily_screen_time_hours × app_opens_per_day | 0.965827 | -0.000015 | +0.000049 | 45801 | 8048 | True |
| gaming_hours × stress_level | 0.965825 | -0.000017 | +0.000047 | 13803 | 10014 | True |
| age × academic_work_impact | 0.965823 | -0.000019 | +0.000045 | 8910 | 4303 | True |
| social_media_hours × gaming_hours | 0.965823 | -0.000019 | +0.000045 | 5808 | 6347 | True |
| age × gender | 0.965818 | -0.000024 | +0.000041 | 13888 | 6379 | True |
| age × notifications_per_day | 0.965815 | -0.000027 | +0.000037 | 68155 | 9404 | True |
| gaming_hours × work_study_hours | 0.965812 | -0.000030 | +0.000035 | 3140 | 5487 | True |
| social_media_hours × gender | 0.965810 | -0.000032 | +0.000032 | 23406 | 7568 | True |
| sleep_hours × weekend_screen_time | 0.965809 | -0.000033 | +0.000031 | 5699 | 5750 | True |
| work_study_hours × academic_work_impact | 0.965804 | -0.000038 | +0.000026 | 8978 | 8348 | True |
| gender × stress_level | 0.965799 | -0.000043 | +0.000021 | 7587 | 3398 | True |
| work_study_hours × sleep_hours | 0.965798 | -0.000045 | +0.000020 | 2657 | 3791 | True |
| age × sleep_hours | 0.965797 | -0.000045 | +0.000019 | 10802 | 5942 | True |
| app_opens_per_day × gender | 0.965796 | -0.000046 | +0.000018 | 82928 | 9822 | True |
| daily_screen_time_hours × notifications_per_day | 0.965792 | -0.000050 | +0.000014 | 50152 | 6415 | True |
| gaming_hours × gender | 0.965791 | -0.000051 | +0.000013 | 8477 | 5326 | True |
| notifications_per_day × stress_level | 0.965788 | -0.000054 | +0.000010 | 37962 | 8354 | True |
| work_study_hours × gender | 0.965787 | -0.000055 | +0.000009 | 6010 | 5572 | True |
| app_opens_per_day × stress_level | 0.965782 | -0.000060 | +0.000005 | 36239 | 7184 | True |
| daily_screen_time_hours × work_study_hours | 0.965774 | -0.000068 | -0.000004 | 9208 | 7993 | True |
| notifications_per_day × gender | 0.965773 | -0.000069 | -0.000005 | 41824 | 8622 | True |
| sleep_hours × app_opens_per_day | 0.965771 | -0.000072 | -0.000007 | 10663 | 5665 | True |
| gaming_hours × sleep_hours | 0.965758 | -0.000084 | -0.000020 | 5929 | 6470 | True |
| gaming_hours × app_opens_per_day | 0.965755 | -0.000087 | -0.000023 | 7202 | 6031 | True |
| sleep_hours × notifications_per_day | 0.965752 | -0.000091 | -0.000026 | 6744 | 5478 | True |
| notifications_per_day × app_opens_per_day | 0.965749 | -0.000093 | -0.000029 | 404511 | 10369 | True |
| social_media_hours × weekend_screen_time | 0.965745 | -0.000098 | -0.000033 | 19621 | 5988 | True |
| daily_screen_time_hours × gender | 0.965744 | -0.000098 | -0.000034 | 92655 | 8146 | True |
| stress_level × academic_work_impact | 0.965744 | -0.000098 | -0.000034 | 12166 | 2318 | True |
| app_opens_per_day × weekend_screen_time | 0.965743 | -0.000099 | -0.000035 | 169653 | 5421 | True |
| age × stress_level | 0.965742 | -0.000100 | -0.000035 | 8871 | 6177 | True |
| weekend_screen_time × academic_work_impact | 0.965737 | -0.000105 | -0.000041 | 18001 | 5392 | True |
| gaming_hours × weekend_screen_time | 0.965731 | -0.000112 | -0.000047 | 4984 | 4825 | True |
| age × social_media_hours | 0.965727 | -0.000116 | -0.000051 | 38330 | 13296 | True |
| age × daily_screen_time_hours | 0.965725 | -0.000117 | -0.000053 | 13693 | 8341 | True |
| daily_screen_time_hours × stress_level | 0.965721 | -0.000121 | -0.000057 | 36635 | 8887 | True |
| app_opens_per_day × academic_work_impact | 0.965716 | -0.000126 | -0.000062 | 126046 | 82618 | True |
| daily_screen_time_hours × social_media_hours | 0.965707 | -0.000135 | -0.000071 | 19597 | 5974 | True |
| social_media_hours × work_study_hours | 0.965702 | -0.000140 | -0.000076 | 6092 | 9094 | True |
| age × work_study_hours | 0.965701 | -0.000141 | -0.000077 | 8622 | 9270 | True |
| notifications_per_day × academic_work_impact | 0.965697 | -0.000145 | -0.000081 | 33097 | 5803 | True |
| age × gaming_hours | 0.965695 | -0.000147 | -0.000083 | 767930 | 13476 | True |
| social_media_hours × notifications_per_day | 0.965690 | -0.000153 | -0.000088 | 201953 | 6438 | True |
| work_study_hours × notifications_per_day | 0.965682 | -0.000160 | -0.000096 | 4530 | 4616 | True |
| daily_screen_time_hours × gaming_hours | 0.965668 | -0.000174 | -0.000110 | 8207 | 6249 | True |
| social_media_hours × app_opens_per_day | 0.965662 | -0.000180 | -0.000115 | 213198 | 5632 | True |
| daily_screen_time_hours × academic_work_impact | 0.965662 | -0.000180 | -0.000116 | 15945 | 8243 | True |
| work_study_hours × weekend_screen_time | 0.965661 | -0.000181 | -0.000116 | 4121 | 3299 | True |
| age × weekend_screen_time | 0.965652 | -0.000190 | -0.000126 | 19071 | 5766 | True |
| gender × academic_work_impact | 0.965648 | -0.000194 | -0.000130 | 3975 | 1733 | True |
| gaming_hours × notifications_per_day | 0.965645 | -0.000197 | -0.000132 | 4516 | 3883 | True |
| weekend_screen_time × gender | 0.965639 | -0.000203 | -0.000139 | 21429 | 8184 | True |
