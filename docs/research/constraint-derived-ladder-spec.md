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
  1단계는 4열 원시 추가, 2단계는 그 위에 범주 복제와 정확값 TE, 3단계는 그 위에 비율 7열과 자리수 표현 8종이다.
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
| 2 | `cats_te` | `categorical_copies`의 `derived` 인자로 4열의 `<col>_cat`, `target_encoding`의 cols에 4열 추가 | 8 |
| 3 | `ratio_round` | `derived`에 비율 7열과 자리수 32열, `key_digits: 1`인 `target_encoding`(`_te_r1`) 4열 | 43 |

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
- 실험 이름은 `cdv1_<계열>_<단계 접미>`다.
  `cdv1`은 constraint-derived 판본 1이다.
  판본 번호는 4열 정의나 사다리 구성이 바뀔 때 올린다.
- 실행 기록의 `config` 값은 실험 이름이므로 재현 전용 풀 명세(#632)는 `cdv1_` 접두로 재현 구성원을 가려낼 수 있다.

| 순번 | 파일 | 실험 이름 |
| --- | --- | --- |
| 01 | `01_lgb_exp117_raw4.yaml` | `cdv1_lgb_raw4` |
| 02 | `02_lgb_exp117_cats_te.yaml` | `cdv1_lgb_cats_te` |
| 03 | `03_lgb_exp117_ratio_round.yaml` | `cdv1_lgb_ratio_round` |
| 04 | `04_xgb_exp135_raw4.yaml` | `cdv1_xgb_raw4` |
| 05 | `05_xgb_exp135_cats_te.yaml` | `cdv1_xgb_cats_te` |
| 06 | `06_xgb_exp135_ratio_round.yaml` | `cdv1_xgb_ratio_round` |
| 07 | `07_cat_exp070_raw4.yaml` | `cdv1_cat_raw4` |
| 08 | `08_cat_exp070_cats_te.yaml` | `cdv1_cat_cats_te` |
| 09 | `09_cat_exp070_ratio_round.yaml` | `cdv1_cat_ratio_round` |
| 10 | `10_realmlp_exp139_raw4.yaml` | `cdv1_realmlp_raw4` |
| 11 | `11_realmlp_exp139_cats_te.yaml` | `cdv1_realmlp_cats_te` |
| 12 | `12_realmlp_exp139_ratio_round.yaml` | `cdv1_realmlp_ratio_round` |

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
5. XGBoost 2단계는 사다리 규칙대로 4열만 범주 복제를 갖는다.
   XGBoost adapter는 category dtype을 `enable_categorical`로 그대로 학습하므로(`model.py` 902행) 1,100~1,300개 정확값의 분할 기반 범주 처리가 그대로 붙는다.
   raw 9열의 범주 복제를 함께 넣으면 기준과 후보의 차이가 4열 밖으로 번져 사다리의 짝비교가 흐려진다.

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
다만 LightGBM과 XGBoost의 2·3단계는 범주 복제 대신 정확값 TE만 두는 변형이 사다리의 뜻(변환 표현의 폭)에 더 맞을 수 있어, 그 판단은 #623 발주 전에 사용자가 정할 항목으로 지도에 올린다.
