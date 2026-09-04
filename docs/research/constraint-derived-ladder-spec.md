# 제약 파생 4열 승격 사다리 설정 12개 명세 초안 (#621)

지도 [#619](https://github.com/tmheo/predicting-smartphone-addiction/issues/619)의 첫 실행 티켓 [#621](https://github.com/tmheo/predicting-smartphone-addiction/issues/621)이 확정할 설정 12개의 명세다.
설정 파일 초안은 `configs/constraint-derived/`에 있고 이 문서는 그 초안이 왜 그렇게 생겼는지와 사용자가 정할 항목을 적는다.
구현은 [#622](https://github.com/tmheo/predicting-smartphone-addiction/issues/622)가 하므로 초안은 아직 `pipeline.run`의 설정 파싱을 통과하지 않는다.
자료 수치는 `data/train.csv` 기준이고 코드 행 번호는 커밋 `2aeb887` 기준이다.

## 요지

- 4열은 새 파생 열로 두고 `sgw_sum`을 재사용하지 않는다.
  2위의 정의는 성분 하나라도 결측이면 결측이고, `sgw_sum`은 결측을 0으로 보고 더하므로 정의가 다르다.
- 4열은 성분이 모두 관측된 행에서만 정의하고 소수 둘째 자리에서 반올림한다.
  원시 열이 0.01 격자에 있어 차이도 0.01 격자에 있어야 하는데, 부동소수 뺄셈은 60%의 행에서 격자를 벗어난 값을 만들어 정확값 키를 깨뜨린다.
- 세 단계 사다리는 계열마다 같은 제공자 목록을 쓴다.
  1단계는 4열 원시 추가, 2단계는 그 위에 계열 고유의 정확값 키 표현과 정확값 TE, 3단계는 그 위에 비율 7열과 자리수 표현 8종이다.
  2단계의 계열 고유 표현은 CatBoost가 범주 복제, RealMLP가 정확값 임베딩이고 LightGBM과 XGBoost는 고유 표현 없이 정확값 TE만 둔다(#634 결정, 판본 `cdv2`).
- 기준 4개는 원본 행 계열 최고 설정을 그대로 다시 돌린다.
  LightGBM `exp117_ag25_gbm_r21`, XGBoost `exp135_xgb_hpo_trial30`, CatBoost `exp070_cat_exact_cats`, RealMLP `exp139_realmlp_reference_qnormal_train_test`다.
- 코드 변경은 세 곳이다.
  `DERIVED_REGISTRY`에 이름 43개를 등록하고, `categorical_copies`가 파생 열을 복제할 수 있게 `derived` 인자를 더하며, RealMLP에 `extra_raw_numeric_columns` 인자를 더한다.

## 4열의 정의와 결측 처리

| 이름 | 정의 | 정의되는 행 | 비고 |
| --- | --- | --- | --- |
| `fake_daily` | `social_media_hours + gaming_hours + work_study_hours` | 세 성분 모두 관측(65.4%) | `sgw_sum`과 다르다. `sgw_sum`은 결측 성분을 0으로 보고 더해 전 행에서 정의된다(`features.py` 48~50행). |
| `fake_social` | `daily_screen_time_hours - gaming_hours - work_study_hours` | daily와 두 성분 관측(68.5%) | social이 관측된 행에서는 `social + other`, social이 결측인 행에서는 social과 other의 합을 복원한다. |
| `fake_work` | `daily_screen_time_hours - social_media_hours - gaming_hours` | daily와 두 성분 관측(65.3%) | 위와 같은 구조다. |
| `fake_game` | `daily_screen_time_hours - social_media_hours - work_study_hours` | daily와 두 성분 관측(67.7%) | 이름은 2위 원문의 `fake_game`을 따르고 우리 열 이름 `gaming_hours`와 맞추지 않는다. |

결측 처리는 `pandas`의 산술이 하는 대로 성분 하나라도 결측이면 결측이다.
이 규약은 2위 원문의 정의와 같고, 우리의 엄격 잔차 `other_screen`과 같은 결측 규약이다.
네 열 모두 음수는 없다(자료 검사 결과 음수 비율 0.0%).

반올림은 `np.round(x, 2)`다.
원시 네 열은 관측값 전부가 0.01 격자에 있고(0.1 격자는 약 10%, 정수는 약 1%), 두 값의 차이도 0.01 격자에 있어야 한다.
그러나 `daily - gaming - work`를 그대로 두면 473,383행 가운데 284,848행이 격자를 벗어난 부동소수 값을 갖고, 정확값 TE 키와 범주 복제 키가 같은 값을 다른 키로 보게 된다.
`_first_decimal`(`features.py` 206~210행)이 같은 이유로 소수 6자리에서 끊는다.

기존 열과의 관계는 다음과 같다.
`screen_slack`(daily가 관측된 행에서 `daily - 관측 성분 합`)은 성분 하나만 결측인 행에서 그 성분의 `fake_*` 열과 값이 같다.
그러나 `screen_slack`은 어느 성분이 결측이냐에 따라 뜻이 바뀌는 한 열이고, `fake_*`는 뜻이 고정된 세 열이다.
이 차이가 실험의 본체이므로 기존 열은 그대로 두고 4열을 더한다.

## 세 단계 사다리

세 단계는 누적이다.
2단계는 1단계의 제공자를 전부 포함하고 3단계는 2단계를 전부 포함한다.
계열마다 기준 설정의 제공자 목록에 아래를 더하며 기준 설정의 기존 제공자와 모형 설정은 바꾸지 않는다.

| 단계 | 이름 접미 | 더하는 제공자 | 새 열 수 |
| --- | --- | --- | --- |
| 1 | `raw4` | `derived`에 `fake_daily, fake_social, fake_work, fake_game` | 4 |
| 2 | `cats_te` | 계열 고유의 정확값 키 표현(아래 표)과 `target_encoding`의 cols에 4열 추가 | 계열별 4~8 |
| 3 | `ratio_round` | `derived`에 비율 7열과 자리수 32열, `key_digits: 1`인 `target_encoding`(`_te_r1`) 4열 | 43 |

### 2단계의 계열별 정확값 키 표현(#634)

2단계 접미 `cats_te`는 "계열이 정확값 키를 다루는 고유 표현 + 정확값 TE 4열"이라는 사다리 단의 이름이다.
고유 표현은 계열마다 다르며 첫 회차(`cdv2`)의 구성은 다음과 같다.

| 계열 | 고유 표현 | 2단계 새 열 수 |
| --- | --- | --- |
| CatBoost | `categorical_copies`의 `derived` 인자로 4열의 `<col>_cat`(ordered target statistics) | 8 |
| RealMLP | `extra_raw_numeric_columns`의 정확값 임베딩 `<col>_cat_`과 결측 지시자 | 4(+adapter 내부 열) |
| LightGBM | 없음. 정확값 TE 4열만 | 4 |
| XGBoost | 없음. 정확값 TE 4열만 | 4 |

LightGBM과 XGBoost에 고유 표현을 두지 않는 근거는 아래 "스모크에서 드러난 관찰" 절과 "#634 결정" 절에 있다.
두 계열의 기준 설정은 고카디널리티 범주 없이 튜닝돼 범주 규제가 사실상 없고(`cat_smooth` 0.001, `min_data_per_group` 10, XGBoost는 `max_cat_threshold` 기본값), 지도가 기준 모형 설정 불변을 정했으므로 1,100~1,300개 정확값 범주를 분할 기반으로 넘기면 재는 것은 표현의 폭이 아니라 튜닝 불일치다.
이 절의 결정 전(`cdv1`)에는 두 계열도 CatBoost와 같은 범주 복제를 두었다.

### 2단계의 제공자 순서 문제와 해결

`categorical_copies`는 dataset-wide 단계이고 `derived`는 row-wise 단계라서(`plan.py` 123~140행, 적용 순서 base -> dataset-wide -> row-wise), 현재 코드로는 파생 열의 범주 복제를 만들 수 없다.
해결은 `CategoricalCopies`에 `derived: [이름...]` 인자를 더해 `DERIVED_REGISTRY`의 파생 함수를 train과 test에 각각 적용한 뒤 합집합 범주로 복제하게 하는 것이다.
파생 4열은 행 단위 결정적 함수라 dataset-wide 단계에서 계산해도 누출이 없다.
산출 이름은 raw 복제와 같은 `<col>_cat`이다.
`cols`는 빈 목록을 허용해 파생 열만 복제하는 선언을 가능하게 한다.

대안은 파생 열을 dataset-wide 단계에서 만드는 새 kind를 두는 것이다.
새 kind는 사다리의 세 단계에서 4열의 열 순서를 바꾸고 REGISTRY와 재사용 선언을 함께 늘리므로 첫 회차는 인자 추가를 택한다.

### RealMLP의 2단계

RealMLP adapter는 raw 범주 3열이 아닌 범주형 입력을 거부한다(`realmlp.py` 188~197행).
따라서 `<col>_cat` 복제 열을 RealMLP에 줄 수 없다.
RealMLP에서 "범주 복제"에 해당하는 것은 adapter가 raw 수치 9열에 하는 처리다.
중앙값 대체, 결측 지시자 `_miss_<col>`, 정확값 임베딩 `<col>_cat_`, 그리고 설정에 따른 분위-정규 좌표다(`realmlp.py` 211~233행, 285~300행).
2단계 RealMLP 설정은 모형 인자 `extra_raw_numeric_columns: [fake_daily, fake_social, fake_work, fake_game]`로 4열을 중앙값 대체, 결측 지시자, 정확값 임베딩 처리에 넣는다.
`reference_qnormal_columns`에는 4열을 더하지 않는다.
adapter가 그 인자를 원시 수치 9열 그대로의 순서로 고정하고 있고(`realmlp.py` 1135~1140행, #331 계약), 4열은 65~69%의 행에서만 정의돼 결측 행의 좌표가 0으로 채워지므로 첫 회차에서는 그 계약을 건드리지 않는다.
분위 구간 `BIN_CONFIG`는 daily와 social에 고정돼 있고 구간 제공자는 이 회차 범위 밖이므로 4열에 구간은 만들지 않는다.
정확값 TE 열은 수치 열이므로 RealMLP에 통과 입력으로 들어간다.

1단계 RealMLP 설정은 4열을 통과 수치 입력(`median_center`, `robust_scale`)으로만 준다.
RealMLP 기준 설정 `exp139`는 `target_encoding`을 쓰지 않으므로 2단계에서 처음으로 TE 제공자가 붙는다.

### 3단계의 비율과 자리수 표현

비율 7열은 `_guarded_ratio`(`features.py` 99~106행) 규약을 따른다.
분자와 분모가 유한하고 분모가 양수일 때만 정의하고 그 밖은 결측이다.

| 이름 | 분자 / 분모 |
| --- | --- |
| `fake_daily_share_screen` | `fake_daily / daily_screen_time_hours` |
| `fake_social_share_screen` | `fake_social / daily_screen_time_hours` |
| `fake_work_share_screen` | `fake_work / daily_screen_time_hours` |
| `fake_game_share_screen` | `fake_game / daily_screen_time_hours` |
| `social_share_fake_daily` | `social_media_hours / fake_daily` |
| `gaming_share_fake_daily` | `gaming_hours / fake_daily` |
| `work_share_fake_daily` | `work_study_hours / fake_daily` |

앞 네 열은 4열을 기존 비율 4열(`SCREEN_RELATION_RATIOS`)과 같은 분모로 다루고, 뒤 세 열은 `fake_daily`를 daily와 같은 분모 지위로 다룬다.

자리수 표현은 정체성·자리수 블록(`features.py` 213~283행)의 규약을 4열에 적용하되 0.01 격자에서 항등이 되는 `round2`와 `absdiff_round2`는 뺀다.
열마다 8개, 합계 32열이다.

| 접미 | 뜻 |
| --- | --- |
| `_round0`, `_absdiff_round0`, `_is_round0` | 정수 반올림값, 그 절대 편차, 정수 여부 |
| `_round1`, `_absdiff_round1`, `_is_round1` | 소수 첫째 자리 반올림값, 그 절대 편차, 0.1 격자 여부 |
| `_tenths`, `_hundredths` | 소수 첫째 자리와 둘째 자리 값 |

반올림 키 TE는 `target_encoding`에 `key_digits: 1`, `suffix: _te_r1`을 준 두 번째 제공자로 4열과 `placebo_noise`에 붙인다.
같은 설정에 정확값 TE(`_te`)와 반올림 키 TE(`_te_r1`)가 함께 있으므로 카나리아 열도 `placebo_noise_te`와 `placebo_noise_te_r1` 둘이다.

## 계열별 기준 설정

| 계열 | 기준 설정 | 3시드 OOF | 근거 |
| --- | --- | --- | --- |
| LightGBM | `exp117_ag25_gbm_r21` | 0.9687158 | 2026-08-21 풀 기준선 파일의 값(run `d107ea87`). 현재 풀의 LightGBM 최고 `exp208`은 이 설정의 결측 증강 판이다. |
| XGBoost | `exp135_xgb_hpo_trial30` | 0.9683307 | 현재 풀의 XGBoost 최고(run `119b9c4e`). |
| CatBoost | `exp070_cat_exact_cats` | 0.9685793 | 2026-08-21 풀 기준선 파일의 값(run `6238d8c5`). 현재 풀의 CatBoost 최고 `mpv1_exp070`은 이 설정의 결측 증강 판이다. |
| RealMLP | `exp139_realmlp_reference_qnormal_train_test` | 0.9685456 | 현재 풀의 RealMLP 최고(run `1af9442e`). |

지도의 확정 사항은 "현재 풀의 계열 최고 구성"인데 LightGBM과 CatBoost의 풀 최고는 결측 증강 판(`training_rows: missingness_augmented`)이다.
초안은 두 계열에서 증강 판의 원본 행 부모를 기준으로 택했고 그 이유는 다음이다.

- `mpv1_exp070`은 `paired_training_length`로 학습 길이를 원본 행 실행 `exp070`의 관측 길이에 고정한다.
  새 피처 집합의 후보는 그 길이를 물려받을 근거가 없고, 물려받지 않으면 기준과 후보의 학습 길이 규칙이 갈린다.
- 결측 증강은 관측 셀을 무작위로 지운 복제 행을 더하므로 4열의 정의 여부가 복제본마다 달라진다.
  4열의 잔차 정보가 계열마다 다른지 묻는 첫 회차에서 이 교란을 섞지 않는다.
- 두 계열의 원본 행 부모는 증강 판과 제공자 목록과 모형 설정이 같으므로 사다리가 재는 차이는 4열과 그 변환뿐이다.

기준은 새 설정 파일을 만들지 않고 기존 설정 파일을 같은 커밋에서 3시드 `confirm` 단계로 다시 돌린다.
과거 실행의 OOF를 재사용하지 않는 이유는 코드 상태와 실행 환경을 사다리와 맞추기 위해서다(#281의 깨끗한 짝비교 선례).

## 이름 규약과 배치

- 디렉터리는 `configs/constraint-derived/`다.
  `configs/missingness-propagation/`처럼 회차 하나의 설정 묶음을 하위 디렉터리에 두는 선례를 따른다.
- 파일 이름은 `<순번>_<계열>_<기준 설정 번호>_<단계 접미>.yaml`이다.
  순번은 계열 순서(lgb, xgb, cat, realmlp)와 단계 순서로 01부터 12까지다.
- 실험 이름은 `cdv2_<계열>_<단계 접미>`다.
  `cdv2`는 constraint-derived 판본 2다.
  판본 번호는 4열 정의나 사다리 구성이 바뀔 때 올린다.
  판본 1(`cdv1`)은 LightGBM·XGBoost 2·3단계에 범주 복제 4열을 둔 구성이었고 fold 0 스모크만 돌렸으며 3시드 실행 기록은 없다.
  #634 결정으로 두 계열의 범주 복제를 빼면서 사다리 구성이 바뀌어 12개 전부 `cdv2`로 올렸다.
  스모크 표의 `cdv1_` 이름은 그때의 구성을 가리키므로 그대로 둔다.
- 실행 기록의 `config` 값은 실험 이름이므로 재현 전용 풀 명세(#632)는 `cdv2_` 접두로 재현 구성원을 가려낼 수 있다.

| 순번 | 파일 | 실험 이름 |
| --- | --- | --- |
| 01 | `01_lgb_exp117_raw4.yaml` | `cdv2_lgb_raw4` |
| 02 | `02_lgb_exp117_cats_te.yaml` | `cdv2_lgb_cats_te` |
| 03 | `03_lgb_exp117_ratio_round.yaml` | `cdv2_lgb_ratio_round` |
| 04 | `04_xgb_exp135_raw4.yaml` | `cdv2_xgb_raw4` |
| 05 | `05_xgb_exp135_cats_te.yaml` | `cdv2_xgb_cats_te` |
| 06 | `06_xgb_exp135_ratio_round.yaml` | `cdv2_xgb_ratio_round` |
| 07 | `07_cat_exp070_raw4.yaml` | `cdv2_cat_raw4` |
| 08 | `08_cat_exp070_cats_te.yaml` | `cdv2_cat_cats_te` |
| 09 | `09_cat_exp070_ratio_round.yaml` | `cdv2_cat_ratio_round` |
| 10 | `10_realmlp_exp139_raw4.yaml` | `cdv2_realmlp_raw4` |
| 11 | `11_realmlp_exp139_cats_te.yaml` | `cdv2_realmlp_cats_te` |
| 12 | `12_realmlp_exp139_ratio_round.yaml` | `cdv2_realmlp_ratio_round` |

## #622가 구현할 것

- `src/pipeline/features.py` `DERIVED_REGISTRY`에 4열, 비율 7열, 자리수 32열을 등록한다.
  4열은 `SCREEN_TOTAL`과 `SCREEN_PARTS` 상수를 쓰고 `np.round(x, 2)`로 끝낸다.
  자리수 32열은 `_digit_identity_registry`의 함수(`_rounded`, `_absdiff_rounded`, `_is_rounded`, `_decimal_digit`)를 재사용한다.
- `CategoricalCopies`에 `derived: list[str] = []` 인자를 더한다.
  `DERIVED_REGISTRY`에 없는 이름은 적재 시점에 거부하고, `columns()`는 `cols`와 `derived`의 순서대로 `<col>_cat`을 낸다.
- RealMLP adapter에 `extra_raw_numeric_columns: list[str] = []` 인자를 더한다.
  `_FoldFeatureEngineer`가 `RAW_NUMERICAL + extra_raw_numeric_columns`에 중앙값 대체, `_miss_`, `_cat_` 처리를 하고 `BIN_CONFIG`는 건드리지 않는다.
  `reference_qnormal_columns`의 원시 9열 고정 검사는 그대로 둔다.
- 단위 시험은 제약 복원 열과 같은 수준으로 둔다.
  4열의 결측 규약과 격자 반올림, `categorical_copies`의 `derived` 복제, RealMLP의 추가 열 처리가 대상이다.
- 완료 조건은 12개 설정의 `pipeline.run` 설정 파싱과 fold 0 스모크 통과다.

## 확정 사항(2026-09-03)

사용자는 1번에 동의했고 2~5번은 제안대로 정하라고 했다.
확정과 근거는 다음이다.

1. LightGBM과 CatBoost의 기준은 원본 행 부모 `exp117`과 `exp070`이다.
   사용자가 동의했다.
2. 3단계 비율은 7열 전부다.
   뒤 세 열(`fake_daily` 분모)은 관측 성분 합 안의 구성비라서 daily 분모 비율과 다른 정규화이고 저장소에 같은 정의가 없다.
   첫 회차는 변환 목록의 폭을 재는 팔이므로 줄이지 않는다.
3. 3단계 자리수 표현은 8종 전부다.
   원시 열이 0.01 격자에 있고 0.1 격자와 정수가 각각 10%와 1%라서 `round0`과 `round1` 계열이 모두 정보를 갖는다.
   `tenths`와 `hundredths`는 자료 생성기의 자릿수 흔적을 보는 열이며 원시 9열에 붙였던 것과 같은 규약이다.
   항등이 되는 `round2` 계열만 뺀다.
4. RealMLP 2단계에서 4열을 `reference_qnormal_columns`에 더하지 않는다.
   adapter가 그 인자를 원시 9열 순서로 고정하고 있어 더하려면 #331 계약을 풀어야 하고, 4열은 결측 행이 많아 좌표 0 채움이 늘어난다.
   `extra_raw_numeric_columns`의 정확값 임베딩과 결측 지시자가 2단계의 본체다.
5. XGBoost 2단계는 `cdv1`에서 사다리 규칙대로 4열만 범주 복제를 가졌다.
   XGBoost adapter는 category dtype을 `enable_categorical`로 그대로 학습하므로(`model.py` 902행) 1,100~1,300개 정확값의 분할 기반 범주 처리가 그대로 붙는다.
   raw 9열의 범주 복제를 함께 넣으면 기준과 후보의 차이가 4열 밖으로 번져 사다리의 짝비교가 흐려진다.
   `cdv2`에서는 #634 결정으로 LightGBM과 함께 범주 복제를 빼고 정확값 TE 4열만 둔다.

## 구현 결과(#622, 2026-09-03)

코드 변경은 명세의 세 곳에 XGBoost adapter 한 곳이 더해져 네 곳이다.

- `src/pipeline/features.py`: `DERIVED_REGISTRY`에 `CONSTRAINT_DERIVED_NAMES` 43개(4열, 비율 7열, 자리수 32열)를 등록했다.
  비율과 자리수 함수는 raw 열 이름 대신 계산 함수도 원천으로 받도록 `_guarded_ratio`, `_rounded`, `_absdiff_rounded`, `_is_rounded`, `_decimal_digit`를 일반화했다.
- `src/pipeline/features.py`: `CategoricalCopies`에 `derived` 인자를 더했다.
  등록되지 않은 이름, 빈 선언, 중복은 적재 시점에 거부하고 산출 순서는 `cols` 뒤에 `derived`다.
  복제 학습 행 경로(`plan.recompute_training_row_dataset_wide`)는 제공자의 `source_values`로 원천을 읽어 파생 복제도 다시 만든다.
- `src/pipeline/realmlp.py`: `extra_raw_numeric_columns` 인자를 더했다.
  `_FoldFeatureEngineer`가 원시 9열과 추가 열에 같은 중앙값 대체, `_miss_`, `_cat_` 처리를 하고 `BIN_CONFIG`와 `reference_qnormal_columns`의 원시 9열 고정은 그대로다.
  학습 진단에 `extra_raw_numeric_columns`를 기록한다.
- `src/pipeline/model.py`: XGBoost adapter가 범주가 부동소수인 범주 열을 거부하던 문제를 고쳤다.
  XGBoost는 범주 색인이 문자열이나 정수여야 해서 정확값 복제 열(`<col>_cat`)을 그대로 넘기면 `Category index from DataFrame has floating point dtype`로 실패한다.
  코드 배정은 그대로 두고 범주 이름만 문자열로 바꿔 넘긴다.
  기존 XGBoost 설정은 범주 복제를 쓴 적이 없어 이 경로가 처음 열렸다.

단위 시험은 `tests/test_features_constraint_derived.py`(4열 결측·격자 규약, 비율, 자리수, `derived` 복제, 피처 계획 연결)와 `tests/test_model_realmlp.py`(추가 열 처리와 계약), `tests/test_model.py`(XGBoost 부동소수 범주)에 있다.
설정 12개는 전부 `pipeline.run --plan`의 설정 파싱을 통과한다.

### fold 0 스모크

스모크 도구는 `scripts/smoke_constraint_derived_fold0.py`다.
정식 경로와 같은 피처 계획과 adapter로 fold 0, seed 42를 한 번 돌리고, 단계가 선언한 새 열이 행렬에 있는지와 4열의 결측·격자 규약을 확인한다.
나무 3계열은 전체 자료로 돌렸고(로컬 CPU, 계열당 4 스레드), RealMLP는 로컬에 CUDA가 없어 행 표본과 `device: cpu`로 연결만 확인했다.
RealMLP의 fold 0 AUC는 표본이 작아 읽지 않으며, 전체 자료 fold 0은 #623의 Vast.ai 첫 실행이 겸한다.

아래 표의 `cdv1_` 이름은 LightGBM·XGBoost 2·3단계에 범주 복제 4열을 둔 판본 1의 구성이다.
#634 결정 뒤의 `cdv2`는 그 4열을 빼며 다른 계열과 1단계는 판본 1과 같다.

| 설정 | fold 0 AUC | 열 수 | 학습 시간 | 비고 |
| --- | --- | --- | --- | --- |
| `cdv1_lgb_raw4` | 0.9679376 | 42 | 595초 | |
| `cdv1_lgb_cats_te` | 0.9656973 | 50 | 194초 | |
| `cdv1_lgb_ratio_round` | 0.9658636 | 94 | 317초 | |
| `cdv1_xgb_raw4` | 0.9675503 | 32 | 427초 | |
| `cdv1_xgb_cats_te` | 0.9629219 | 40 | 149초 | |
| `cdv1_xgb_ratio_round` | 0.9635759 | 84 | 324초 | |
| `cdv1_cat_raw4` | 0.9677501 | 51 | 376초 | |
| `cdv1_cat_cats_te` | 0.9677148 | 59 | 486초 | |
| `cdv1_cat_ratio_round` | 0.9676290 | 103 | 594초 | |
| `cdv1_realmlp_raw4` | (연결만) | 20 | 1226초 | 2,500행, cpu 1 스레드 |
| `cdv1_realmlp_cats_te` | (연결만) | 25 | 1319초 | 2,500행, cpu 1 스레드, `fake_*_cat_` 임베딩 4개 확인 |
| `cdv1_realmlp_ratio_round` | (연결만) | 69 | 3076초 | 6,000행, cpu 1 스레드, `fake_*_cat_` 임베딩 4개 확인 |

4열의 정의 행 비율은 `fake_daily` 65.35%, `fake_social` 68.47%, `fake_work` 65.27%, `fake_game` 67.73%이고 정확값 수는 1,145~1,308개다.
관측값 전부가 0.01 격자에 있고 음수는 없다.

### 스모크에서 드러난 관찰(판정 아님)

LightGBM과 XGBoost는 2단계에서 fold 0 AUC가 크게 떨어진다.
LightGBM은 raw4 대비 -0.0022, XGBoost는 -0.0046이며 champion fold 0의 시드 폭 0.0000446보다 두 자릿수 크다.
CatBoost는 -0.00004로 잡음 안이다.

원인을 가르기 위해 2단계에서 `categorical_copies`만 뺀 진단 설정(정확값 TE 4열만 추가)을 fold 0에서 한 번 더 돌렸다.
XGBoost는 0.9675237로 raw4와 같은 수준이다.
LightGBM도 0.9679622로 raw4와 같은 수준이라 두 계열 모두 하락은 정확값 범주 복제 4열이 만든다.
두 계열의 기준 설정은 고카디널리티 범주가 없는 상태에서 튜닝됐고(`max_cat_to_onehot: 71`, `cat_smooth: 0.001`, XGBoost는 `max_cat_threshold` 기본값), 1,100~1,300개 정확값 범주를 분할 기반으로 다루면 과적합한다.
CatBoost는 ordered target statistics로 같은 열을 다루므로 영향이 없다.

지도의 확정 사항대로 단일 모형 결과는 기록만 하고 판정에 쓰지 않으며 세 단계 12개 설정은 그대로 #623으로 넘긴다.
다만 LightGBM과 XGBoost의 2·3단계는 범주 복제 대신 정확값 TE만 두는 변형이 사다리의 뜻(변환 표현의 폭)에 더 맞을 수 있어, 그 판단은 #623 발주 전에 사용자가 정할 항목으로 지도에 올렸다(#634).

### #634 결정(2026-09-03)

사용자가 다음을 확정했다.

- LightGBM과 XGBoost의 2·3단계(`02`, `03`, `05`, `06`)는 `categorical_copies`를 빼고 정확값 TE 4열만 둔다.
  3단계는 누적이라 함께 바뀌고 반올림 키 TE(`_te_r1`)는 그대로다.
  두 팔을 다 돌리는 선택(16개)은 예산은 들지만(14코어 로컬에서 나무 실행 12개 약 8시간 대 16개 약 11시간) 얻는 것이 fold 0에서 이미 본 하락의 3시드 확인뿐이라 택하지 않았다.
- 2단계 접미 `cats_te`는 유지한다.
  RealMLP 2단계도 범주 열 없이 같은 접미를 쓰므로 접미는 "계열 고유의 정확값 키 표현 + 정확값 TE"라는 사다리 단의 이름이다.
  계열별 표현은 위 "2단계의 계열별 정확값 키 표현" 표에 둔다.
- 사다리 구성이 바뀌므로 12개 전부 판본을 `cdv2`로 올린다.
  3시드 실행 기록이 없어 비용은 이름 12개뿐이고, 스모크 표의 `cdv1_` 이름과 실행 판이 이름으로 갈린다.
- 범주 복제 4열의 fold 0 하락은 이 절의 진단 기록으로만 남긴다.
  LightGBM·XGBoost의 고카디널리티 범주 규제 재튜닝(`min_data_per_group`, `cat_smooth`, `max_cat_threshold`)은 지도 #619의 기준 모형 설정 불변 확정과 어긋나므로 지도 범위 밖이다.
  초기 LightGBM `exp003_categorical_copies`(원시 9열 범주 복제, 기본 규제)가 champion 계보였던 점은 LightGBM이 고카디널리티 복제를 원래 못 다루는 것이 아니라 기준 파라미터의 문제임을 뒷받침한다.
  결과가 이득으로 확정되면 "규제 재튜닝 후 복제 표현 재시험"을 새 지도 후보로 적는다.

설정 12개는 `cdv2` 이름과 위 구성으로 `pipeline.run --plan` 설정 파싱을 통과했다.

## 3시드 학습 결과(#623, 2026-09-03)

설정 12개와 계열별 기준 4개를 실행 커밋 `01d6cf3`(cdv2 `862b12d` 위에 실행 스크립트 `scripts/issue623/`만 더한 커밋, 가지 `issue623-run`)에서 3시드(42, 43, 44)·고정 5분할 `confirm` 단계로 학습했다.
나무 3계열 12개는 로컬 워크트리(14코어, 계열별 차선 동시 실행, OpenMP 스레드 4, CatBoost는 자체 스레드 풀)에서, RealMLP 4개는 Vast.ai 대만 RTX 4090 x4(GPU 하나에 설정 하나, 시드 순차)에서 돌렸다.
16개 실행 기록은 묶음으로 내보내 main MLflow에 반입했고 `scripts/issue623/import_and_audit.py`가 재채점, 시드 평균 일치, 제출 파일 일치를 감사했다.
아래 표는 `scripts/issue623/report.py`가 만든 것이며 run 열은 main MLflow 반입 run이다.
"시드 AUC 평균"은 시드별 OOF AUC의 평균, "시드 평균 예측 OOF"는 세 시드 예측을 평균한 뒤 잰 `auc_oof`(판정 눈금)다.
분할 부호는 시드 평균 예측의 fold별 AUC 차이 5개, 시드x분할 부호는 시드별 fold AUC 차이 15개의 양수 개수다.

### LightGBM

| 실험 | run | seed 42 | seed 43 | seed 44 | 시드 AUC 평균 | 기준 대비 | 시드 평균 예측 OOF | 기준 대비 | 분할 부호(시드 평균, 5) | 시드x분할 부호(15) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exp117_ag25_gbm_r21` (기준) | `fa1b60f1` | 0.9686392 | 0.9686205 | 0.9686496 | 0.9686364 | - | 0.9687177 | - | - | - |
| `cdv2_lgb_raw4` | `7386da78` | 0.9686031 | 0.9686052 | 0.9686211 | 0.9686098 | -0.0000267 | 0.9686860 | -0.0000317 | 0/5 ----- | 3/15 --------+-+-+-- |
| `cdv2_lgb_cats_te` | `800c976f` | 0.9685620 | 0.9685992 | 0.9686073 | 0.9685895 | -0.0000469 | 0.9686697 | -0.0000480 | 0/5 ----- | 2/15 ------+-----+-- |
| `cdv2_lgb_ratio_round` | `bf0d6913` | 0.9684645 | 0.9684863 | 0.9684894 | 0.9684801 | -0.0001564 | 0.9685463 | -0.0001714 | 0/5 ----- | 0/15 --------------- |

시드별 기준 대비 차이: `cdv2_lgb_raw4` s42 -0.0000361, s43 -0.0000153, s44 -0.0000285; `cdv2_lgb_cats_te` s42 -0.0000772, s43 -0.0000213, s44 -0.0000423; `cdv2_lgb_ratio_round` s42 -0.0001747, s43 -0.0001342, s44 -0.0001602

### XGBoost

| 실험 | run | seed 42 | seed 43 | seed 44 | 시드 AUC 평균 | 기준 대비 | 시드 평균 예측 OOF | 기준 대비 | 분할 부호(시드 평균, 5) | 시드x분할 부호(15) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exp135_xgb_hpo_trial30` (기준) | `398c6fcf` | 0.9682158 | 0.9682228 | 0.9682165 | 0.9682184 | - | 0.9683234 | - | - | - |
| `cdv2_xgb_raw4` | `4d16bada` | 0.9682179 | 0.9682419 | 0.9682256 | 0.9682285 | +0.0000101 | 0.9683321 | +0.0000087 | 4/5 -++++ | 10/15 -+-++++++---+++ |
| `cdv2_xgb_cats_te` | `02d89648` | 0.9681600 | 0.9681806 | 0.9681874 | 0.9681760 | -0.0000424 | 0.9682821 | -0.0000412 | 0/5 ----- | 1/15 ----------+---- |
| `cdv2_xgb_ratio_round` | `e80fbeed` | 0.9682216 | 0.9682223 | 0.9682226 | 0.9682222 | +0.0000038 | 0.9683163 | -0.0000071 | 1/5 ---+- | 8/15 -+-++--+-++-++- |

시드별 기준 대비 차이: `cdv2_xgb_raw4` s42 +0.0000020, s43 +0.0000191, s44 +0.0000091; `cdv2_xgb_cats_te` s42 -0.0000559, s43 -0.0000422, s44 -0.0000292; `cdv2_xgb_ratio_round` s42 +0.0000058, s43 -0.0000005, s44 +0.0000060

### CatBoost

| 실험 | run | seed 42 | seed 43 | seed 44 | 시드 AUC 평균 | 기준 대비 | 시드 평균 예측 OOF | 기준 대비 | 분할 부호(시드 평균, 5) | 시드x분할 부호(15) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exp070_cat_exact_cats` (기준) | `cb326d8d` | 0.9683661 | 0.9684145 | 0.9683634 | 0.9683813 | - | 0.9685685 | - | - | - |
| `cdv2_cat_raw4` | `89086060` | 0.9683773 | 0.9683990 | 0.9684001 | 0.9683921 | +0.0000108 | 0.9685743 | +0.0000058 | 3/5 +++-- | 11/15 +++--+++--+++++ |
| `cdv2_cat_cats_te` | `ef690ee7` | 0.9683230 | 0.9683397 | 0.9683200 | 0.9683276 | -0.0000538 | 0.9685094 | -0.0000592 | 0/5 ----- | 4/15 ++----------+-+ |
| `cdv2_cat_ratio_round` | `33836767` | 0.9683130 | 0.9683210 | 0.9683004 | 0.9683115 | -0.0000699 | 0.9684890 | -0.0000796 | 0/5 ----- | 1/15 -+------------- |

시드별 기준 대비 차이: `cdv2_cat_raw4` s42 +0.0000111, s43 -0.0000155, s44 +0.0000367; `cdv2_cat_cats_te` s42 -0.0000431, s43 -0.0000748, s44 -0.0000434; `cdv2_cat_ratio_round` s42 -0.0000531, s43 -0.0000935, s44 -0.0000630

### RealMLP

| 실험 | run | seed 42 | seed 43 | seed 44 | 시드 AUC 평균 | 기준 대비 | 시드 평균 예측 OOF | 기준 대비 | 분할 부호(시드 평균, 5) | 시드x분할 부호(15) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exp139_realmlp_reference_qnormal_train_test` (기준) | `fe6de111` | 0.9684344 | 0.9684601 | 0.9684581 | 0.9684509 | - | 0.9685464 | - | - | - |
| `cdv2_realmlp_raw4` | `9f3af7ff` | 0.9687411 | 0.9687363 | 0.9687174 | 0.9687316 | +0.0002807 | 0.9688287 | +0.0002823 | 5/5 +++++ | 15/15 +++++++++++++++ |
| `cdv2_realmlp_cats_te` | `2f261e39` | 0.9684874 | 0.9685198 | 0.9685027 | 0.9685033 | +0.0000524 | 0.9686201 | +0.0000737 | 5/5 +++++ | 15/15 +++++++++++++++ |
| `cdv2_realmlp_ratio_round` | `c5dec94a` | 0.9687459 | 0.9687698 | 0.9687449 | 0.9687536 | +0.0003027 | 0.9688737 | +0.0003274 | 5/5 +++++ | 15/15 +++++++++++++++ |

시드별 기준 대비 차이: `cdv2_realmlp_raw4` s42 +0.0003067, s43 +0.0002762, s44 +0.0002593; `cdv2_realmlp_cats_te` s42 +0.0000530, s43 +0.0000597, s44 +0.0000446; `cdv2_realmlp_ratio_round` s42 +0.0003115, s43 +0.0003097, s44 +0.0002868

### 읽기

- 기준 4개의 재실행은 풀 기록과 1e-5 안에서 맞는다(exp117 0.9687177 대 0.9687158, exp135 0.9683234 대 0.9683307, exp070 0.9685685 대 0.9685793, exp139 0.9685464 대 0.9685456).
- RealMLP는 세 단계 모두 시드 3개, 분할 5개, 시드x분할 15쌍 전부 양수다.
  raw4 +0.00028, ratio_round +0.00033이고 cats_te(정확값 임베딩 + TE)는 +0.00007로 가장 작다.
  4열을 원시 수치로 더한 것만으로 대부분의 이득이 나오고, 비율·반올림 표현이 조금 더 보탠다.
- 나무 3계열은 어느 단계도 이득이 없다.
  raw4는 잡음 안(LightGBM -0.00003, XGBoost +0.00001, CatBoost +0.00001)이고 cats_te와 ratio_round는 세 계열 모두 음수(-0.00004~-0.00017)로 열이 늘수록 내려간다.
  LightGBM ratio_round(-0.00017, 15/15 음수)가 가장 크게 내려간다.
- 단일 짝비교는 기록만 하며 결합 판정은 #624에서 한다(지도 확정 사항: 단일 탈락 설정도 결합 구성원으로 넣는다).
  원칙 A1의 "계열마다 다른 잔차 정보"는 단일 모형 눈금에서는 RealMLP에서만 보였다.

### 실행 기록

| 실험 | 실행 위치 | 소요 | main run | 출처 run |
| --- | --- | --- | --- | --- |
| exp117_ag25_gbm_r21 | 로컬 | 163분 | `fa1b60f1` | `56d1408f` |
| cdv2_lgb_raw4 | 로컬 | 158분 | `7386da78` | `fe673dab` |
| cdv2_lgb_cats_te | 로컬 | 151분 | `800c976f` | `ea76bc31` |
| cdv2_lgb_ratio_round | 로컬 | 148분 | `bf0d6913` | `ba41f584` |
| exp135_xgb_hpo_trial30 | 로컬 | 98분 | `398c6fcf` | `27e5d91f` |
| cdv2_xgb_raw4 | 로컬 | 113분 | `4d16bada` | `d640eb7b` |
| cdv2_xgb_cats_te | 로컬 | 9분(복구 fold 14개 이어받음) | `02d89648` | `20abb19b` |
| cdv2_xgb_ratio_round | 로컬 | 162분 | `e80fbeed` | `24e3bbb9` |
| exp070_cat_exact_cats | 로컬 | 91분 | `cb326d8d` | `9696137c` |
| cdv2_cat_raw4 | 로컬 | 89분 | `89086060` | `77680e55` |
| cdv2_cat_cats_te | 로컬 | 95분 | `ef690ee7` | `ef28f9f1` |
| cdv2_cat_ratio_round | 로컬 | 116분 | `33836767` | `b468403d` |
| exp139_realmlp_reference_qnormal_train_test | Vast.ai 49726833 | 약 190분(4개 동시) | `fe6de111` | `329a7a88` |
| cdv2_realmlp_raw4 | Vast.ai 49726833 | 약 160분(4개 동시) | `9f3af7ff` | `bff03858` |
| cdv2_realmlp_cats_te | Vast.ai 49726833 | 약 175분(4개 동시) | `2f261e39` | `0cea3a06` |
| cdv2_realmlp_ratio_round | Vast.ai 49726833 | 약 190분(4개 동시) | `c5dec94a` | `8b3de478` |

- 로컬은 2026-09-03 06:35 UTC에 시작해 16:16 UTC에 끝났다(이동 중 잠자기 약 1.5시간 포함).
  CatBoost 차선이 끝난 뒤 LightGBM·XGBoost의 남은 설정은 스레드 4개짜리 독립 프로세스로 나란히 돌렸다(실행별 스레드 수는 4로 같다).
  결과 JSON·묶음·로그는 `run-logs/issue623/`(git 무시)에 있고 감사 파일은 `run-logs/issue623/import-audit-local.json`이다.
- Vast.ai는 3회차 만에 성공했다.
  1회차(대만 장비 139778)는 running 뒤 SSH 포트가 열리지 않았고, 2회차(대만 장비 142576)는 의존성 wheel 3.8GB를 초당 1~4MB로 받아 준비 관문 30분을 넘겼다.
  3회차(대만 호스트 150178, 유효 CPU 112, 시간당 $1.62)는 설치 6분, 학습 07:45~10:55 UTC, fold당 11~13분, GPU 사용률 22~26%(CPU 병목)였다.
  잔액 $20.32 -> $12.76(세 회차 합 약 $7.6).
  장부는 `run-logs/issue623/vast/ledger.md`, 감사 파일은 `run-logs/issue623/vast/results-3/import-audit.json`이다.
- 3회차 결과 회수는 손으로 했다.
  노트북이 잠든 사이 제어 스크립트가 SSH 조회 실패 경로로 빠졌고 그 정리 절차는 결과 회수 전에 인스턴스를 지우므로, 강제 종료한 뒤 완료 표식(종료 코드 0)을 확인하고 SHA-256 검증으로 회수해 인스턴스 삭제와 종료 예약 제거까지 수동으로 마쳤다.
  다음 원격 실행 스크립트는 실패 경로에서도 완료 표식이 있으면 결과부터 회수하고, 제어는 잠들지 않는 기계에서 돌린다.

## 결합 판정 준비(#632, 2026-09-04)

[#620](https://github.com/tmheo/predicting-smartphone-addiction/issues/620)의 조사대로 기존 판정 도구의 진입점은 바꾸지 않고, 재현 구성원을 `JudgmentRound`에 넣기 위한 최소 구현을 했다.
코드는 commit `c921ca0`, 동결 명세는 commit `b375b9d`에 있다.

### 재현 전용 풀 동결 명세

- 생성기 `scripts/freeze_reproduction_pool.py`가 `docs/research/reproduction-pool-freeze/rpf-v1-6fa08f3da327.json`(schema `reproduction-pool-freeze/1`, spec_sha256 `13302eb32e7bdcc553acf15df514d6f60e51de484efddb614f893ecaa647cf2a`)을 만들었다.
- 재현 구성원은 위 3시드 학습 결과의 설정 12개 전부이며, 순서는 사다리 단계(raw4, cats_te, ratio_round) 오름차순, 단계 안에서는 LightGBM, XGBoost, CatBoost, RealMLP 순이다.
  누적 사다리 3단계(4개, 8개, 12개)는 이 순서의 앞 부분집합이다.
- 구성원마다 main MLflow 반입 run, 출처 run, 실행 커밋 `01d6cf3`, 설정 산출물과 커밋 파일의 일치, 입력 해시, OOF·시험 예측의 배열 해시와 예측 쌍 해시, 시드별·분할별 AUC를 담는다.
  감사는 세 시드 평균 일치, 재채점 AUC 일치(1e-9), 식별자 순서와 분할 배정 일치, 유한성을 검사한다.
- 계열별 기준 재실행 4개는 재현 구성원이 아니라 `baseline_reruns` 근거 기록으로만 남긴다.
  exp117과 exp070은 현재 풀에 결측 증강판(`mpv1_*`)으로만 남아 있어 풀 구성원 대응은 있으면 적고 없으면 비운다.
- 기준 팔 값도 같은 파일에 동결했다.
  자체 36개(`reference_arms.own36`)는 풀 진입 순서의 36개와 OOF 해시(이슈 513 precommit의 자체 구성원과 일치 확인), 이슈 514 최종 확정 실행 MLflow `223055f4`의 근거 산출물 `pool36_full-oof-evidence.json`에서 가져온 nested `0.9698828758140019`와 분할별 AUC, 결합기 `shrunk_rank_logit_logistic`이다.
  314 확장(`reference_arms.ext314`)은 이슈 513 재조립 판정의 구성 해시, nested `0.9703843058098193`, 분할별 AUC와 예측 해시, 봉인 분할 5개 기록의 파일 해시, 결합기 `c_selected_shrunk_rank_logit_logistic`이다.
- `judgment_rules`에 평가 팔 구성, 결합기, 게이트, 통과 구성이 여럿일 때의 제안 규칙(nested AUC 최고, 동률이면 구성원이 적은 쪽)을 결과 확인 전에 적었다.

### member source adapter와 회차 스펙

- `pipeline.member_sources.reproduction_pool_members(path, stage=None)`가 명세의 구성원을 동결 순서대로 `run_id + RunStore` 출처, OOF 해시 대조의 hash-verified `MemberSpec`으로 읽는다.
  `stage`를 주면 누적 사다리 단계의 부분집합만 남긴다.
  시험 예측은 선언하지 않는다(판정 전용이며 조립은 #625에서 명세의 시험 해시로 한다).
- 회차 스펙 두 개는 파일럿 `scripts/round_issue553_pilot.py`를 본떴다.
  `scripts/round_issue624_own36.py`(회차 id `reproduction-pool-own36/issue624`)와 `scripts/round_issue624_ext314.py`(회차 id `reproduction-pool-ext314/issue624`)이며 계약판은 둘 다 `reproduction-pool-judgment`다.
  평가 팔은 기준 팔 구성원 뒤에 사다리 단계의 재현 구성원을 이은 `*-raw4`, `*-cats-te`, `*-ratio-round` 3개다.
- 자기 검사 등급은 자체 36개가 전 분할 재현(36열이라 분할당 50초 안팎, 예측 해시는 기록에 없어 AUC 동일성만 대조), 314 확장이 봉인 분할 0 재현(분할당 14분 안팎, AUC와 예측 해시 대조)이다.
- `scripts/round_members_smoke.py <스펙 스크립트> [--replay-fold k]`가 run-logs를 건드리지 않고 팔 전부를 hash-verified로 적재하고, 옵션으로 기준 팔 분할 하나를 재현해 자기 검사 기대값과 대조한다.

### 스모크 결과

스모크는 `run-logs/`를 건드리지 않고 팔 전부를 hash-verified로 적재한 뒤 자체 36개 기준 팔의 분할 0을 재현했다(2026-09-04, 커밋 `b375b9d`).

| 회차 | 팔 | 구성원 | 검증 수준 | 구성 해시(앞 16자리) | 적재 시간 |
| --- | --- | ---: | --- | --- | ---: |
| own36 | pool36-current | 36 | hash-verified | `aa9371454a86c97f` | 5초 |
| own36 | own36-raw4 | 40 | hash-verified | `e9f8d2756e9aa870` | 5초 |
| own36 | own36-cats-te | 44 | hash-verified | `2293308a31d6aed4` | 6초 |
| own36 | own36-ratio-round | 48 | hash-verified | `dcc70204b5ec40d7` | 6초 |
| ext314 | reassembled-314 | 314 | hash-verified | `e3208ed93ee29126` | 36초 |
| ext314 | ext314-raw4 | 318 | hash-verified | `27103823ae115455` | 36초 |
| ext314 | ext314-cats-te | 322 | hash-verified | `5413bec4a10b121c` | 34초 |
| ext314 | ext314-ratio-round | 326 | hash-verified | `cadc041574b96833` | 35초 |

- 314 기준 팔의 구성 해시 `e3208ed9…`는 이슈 513 precommit의 `composition_sha256`과 같다.
- 자체 36개 기준 팔 분할 0 재현은 AUC `0.9693299877411192`로 MLflow `223055f4`의 분할 0 metric과 비트 단위로 같았다(49초).
  precommit 전 별도 확인에서 분할 1도 `0.9699927220915519`로 같았으므로 전 분할 재현 등급이 이 코드 상태에서 성립한다.
- 어댑터 단위 시험 `tests/test_members.py::test_reproduction_pool_adapter_reads_order_and_ladder_stage`가 순서 검사, 사다리 단계 부분집합, 검증 수준 선언을 고정한다.

### #624가 시작 전에 확인할 것

- 자체 36개 기준의 결합기와 기준값 출처는 위 기본값(`shrunk_rank_logit_logistic`, MLflow `223055f4`)으로 명세에 넣었다.
  314 기준과 결합기를 맞추려면 36개 기준값을 `c_selected_shrunk_rank_logit_logistic`으로 새로 재야 하므로 명세를 다시 만들어야 한다.
- `precommit`은 커밋된 코드 상태에서만 시작하고 재개 검사가 git commit을 대조하므로, 두 회차가 도는 동안 main에 커밋하거나 pull하지 않는다.
- 314 확장 회차의 사다리 팔 3개는 분할당 14분 안팎이라 5분할 3팔에 자기 검사 1분할을 더해 동시 3개로 두 시간 안팎이다.
  자체 36개 회차는 전체가 20분 안이다.
- 실행 순서는 각 스펙 스크립트 docstring에 있다(smoke → precommit → run → compare → report → publish).

## 결합 판정 결과(#624, 2026-09-04)

두 회차 모두 main `73748fd`에서 precommit을 봉인하고 같은 커밋에서 끝냈다.
자기 검사는 자체 36개가 전 분할 재현(5분할 AUC 전부 일치), 314 확장이 봉인 분할 0 재현(AUC와 예측 해시 일치)으로 통과했다.
판정 기록은 `docs/research/reproduction-pool-own36/issue624/`와 `docs/research/reproduction-pool-ext314/issue624/`에 있다(precommit, 분할별 기록, comparison, report, manifest).

### 결과 확인 전에 고정한 것

- 자체 36개 기준의 결합기와 기준값은 동결 명세 기본값(`shrunk_rank_logit_logistic`, MLflow `223055f4`의 nested `0.9698828758140019`)으로 확정했다.
  이 값만 현재 코드로 전 분할 비트 재현이 성립하고, 풀 등록 판정([#337](https://github.com/tmheo/predicting-smartphone-addiction/issues/337))이 쓴 결합기와 같다.
  두 기준 팔은 "자체 풀 등록 문턱"과 "확장 스택 위 기여"라는 다른 질문이므로 결합기를 일치시키지 않았다.
- 게이트, 사다리 구성, 구성원 순서, 제안 규칙은 동결 명세 `judgment_rules` 그대로다.
- 두 기준이 갈릴 때의 해석 제안은 [#624 시작 전 댓글](https://github.com/tmheo/predicting-smartphone-addiction/issues/624)에 남겼으나 결과가 갈리지 않아 쓰지 않았다.

### 자체 36개 기준(shrunk_rank_logit_logistic, precommit `4ddf9145…`)

| 팔 | 구성원 | nested AUC | 기준 대비 | 직전 단계 대비 | 분할 양수 | 판정 |
| --- | ---: | ---: | ---: | ---: | :-: | :-: |
| own36-raw4 | 40 | `0.9698802244` | `-0.0000027` | `-0.0000027` | 2/5 | 미달 |
| own36-cats-te | 44 | `0.9698787632` | `-0.0000041` | `-0.0000015` | 0/5 | 미달 |
| own36-ratio-round | 48 | `0.9698976318` | `+0.0000148` | `+0.0000189` | 5/5 | 미달 |

ratio_round 구성만 분할 5개 전부 양수이나 증분이 문턱 `+0.00002`의 4분의 3에 그친다.
분할별 증분은 `+0.0000047`, `+0.0000219`, `+0.0000036`, `+0.0000206`, `+0.0000230`으로 분할 1·3·4만 문턱을 넘는다.
분할 작업 합계는 15분이었다.

### 314 확장 기준(c_selected_shrunk_rank_logit_logistic, precommit `6d1b99eb…`)

| 팔 | 구성원 | nested AUC | 기준 대비 | 직전 단계 대비 | 분할 양수 | 판정 |
| --- | ---: | ---: | ---: | ---: | :-: | :-: |
| ext314-raw4 | 318 | `0.9703884175` | `+0.0000041` | `+0.0000041` | 4/5 | 미달 |
| ext314-cats-te | 322 | `0.9703852091` | `+0.0000009` | `-0.0000032` | 2/5 | 미달 |
| ext314-ratio-round | 326 | `0.9703914332` | `+0.0000071` | `+0.0000062` | 4/5 | 미달 |

세 구성 모두 규제 강도 선택은 `C=0.03`, λ는 1이었다.
분할 2가 세 구성 전부에서 음수(`-0.0000067`, `-0.0000115`, `-0.0000004`)라 5/5 조건을 만족하는 구성이 없다.
분할 작업 합계는 200분(동시 3개, 벽시계 78분)이었다.

### 읽기

- 결론은 두 기준 모두 통과 구성 없음이며 현재 풀 36개와 확장 스택 314개를 유지한다.
  지도 619의 성공 기준(중첩 결합 판정에서 현행 등록 문턱을 넘음)은 충족되지 않았다.
- 단일 짝비교에서 RealMLP 세 단계가 시드x분할 15/15 양수였던 이득(`+0.00028`~`+0.00033`)이 결합에서는 최대 `+0.0000148`로 줄었다.
  결합기가 이미 RealMLP 계열 구성원(exp139 등)에서 같은 정보를 얻고 있고, 파생 4열의 표현이 잔차에서 새 정보를 거의 더하지 않는다고 읽는다.
- 나무 3계열은 단일에서도 이득이 없었고 결합에서도 raw4·cats_te 단이 자체 36개 기준에서 음수였다.
  cats_te 단은 두 기준 모두 직전 단계보다 낮아, 정확값 TE 4열은 결합에서 잡음을 더한다.
- ratio_round 단이 두 기준에서 가장 높다는 점은 단일 짝비교의 RealMLP ratio_round가 최고였던 것과 일치한다.
  그러나 그 값이 문턱 아래이고 314 기준에서는 4/5라, 재현 구성원 12개는 어느 풀에도 등록하지 않는다.
- 원칙 A1(제약 파생 열의 변환 표현은 계열마다 다른 잔차 정보를 준다)이 이 자료에서 이득을 내는지의 최종 판단은 [#626](https://github.com/tmheo/predicting-smartphone-addiction/issues/626) 보고서에서 한다.

### 보조 진단(같은 크기 대조군과 짝지은 부트스트랩)

이슈 624 본문과 회고 원칙 13(A5·A6)에 따라 판정에 쓰지 않는 보조값 두 가지를 `scripts/aux_diagnostics_issue624.py`로 기록했다.
결과는 `docs/research/reproduction-pool-aux-diagnostics/issue624/{own36,ext314}.json`이다.

- 대조군은 기준 팔에 동결 명세 `baseline_reruns` 4개(exp117·exp135·exp070·exp139 재실행, 3시드, 새 특성 없음)를 raw4 단과 같은 크기로 더한 구성이다.
  풀 밖 검증 구성원 모집단에서 20회 뽑는 원칙 13의 정식 대조군은 아니며, 같은 계열 기준의 재실행 하나로 "4열을 더한 일반 효과"의 눈금만 잰다.
- 부트스트랩은 기준 팔 nested 예측과 평가 팔 nested 예측을 같은 행 재표집(복원 추출, 전체 행, 1000회, 시드 20260904)으로 채점한 AUC 차이의 백분위 95% 구간이다.
  기준 팔 nested 예측은 회차가 저장하지 않아 같은 캐시와 결합기로 다시 만들었고, 두 기준 모두 nested AUC가 기록값과 일치했다(자체 36개는 분할 5개 전부 비트 동일, 314는 소수점 15자리).

| 기준 | 팔 | 구성원 | 차이 | 부트스트랩 95% 구간 | P(차이 ≤ 0) | 분할 양수 |
| --- | --- | ---: | ---: | --- | ---: | :-: |
| own36 | raw4 | 40 | `-0.0000027` | `[-0.0000079, +0.0000026]` | 0.842 | 2/5 |
| own36 | cats_te | 44 | `-0.0000041` | `[-0.0000096, +0.0000019]` | 0.919 | 0/5 |
| own36 | ratio_round | 48 | `+0.0000148` | `[+0.0000055, +0.0000239]` | 0.001 | 5/5 |
| own36 | 대조군(재실행 4) | 40 | `-0.0000046` | `[-0.0000093, +0.0000000]` | 0.975 | 2/5 |
| ext314 | raw4 | 318 | `+0.0000041` | `[+0.0000002, +0.0000081]` | 0.021 | 4/5 |
| ext314 | cats_te | 322 | `+0.0000009` | `[-0.0000033, +0.0000050]` | 0.344 | 2/5 |
| ext314 | ratio_round | 326 | `+0.0000071` | `[+0.0000012, +0.0000129]` | 0.007 | 4/5 |
| ext314 | 대조군(재실행 4) | 318 | `-0.0000005` | `[-0.0000057, +0.0000045]` | 0.567 | 2/5 |

읽기.

- raw4 단은 두 기준 모두 대조군과 구간이 겹친다.
  자체 36개에서는 대조군과 같은 2/5이고, 314 확장에서는 `+0.0000041`이 0을 겨우 벗어나지만 문턱의 5분의 1이다.
  파생 4열을 원시로 더한 구성원 4개는 같은 계열 기준을 다시 돌린 4개와 구별되지 않는다.
- cats_te 단은 두 기준 모두 구간이 0을 품고 대조군보다 낫지 않다.
- ratio_round 단만 두 기준에서 구간이 0을 벗어나며 대조군 구간과 겹치지 않는다.
  그러나 문턱 `+0.00002` 이상인 재표집 비율은 자체 36개에서 12.4%, 314 확장에서 0%다.
  이득은 실재하되 등록 문턱보다 작다고 읽는다.
- 대조군 자체는 두 기준 모두 0 근처(자체 `-0.0000046`, 확장 `-0.0000005`)라, 폭을 4 늘리는 일반 효과는 없거나 약간 음수다.
  ext314 대조군의 분할 0이 `+0.0000116`인데 세 평가 팔의 분할 0도 `+0.0000109`~`+0.0000140`으로 비슷해, 분할 0의 양수는 4열 추가의 일반 효과로 본다.
