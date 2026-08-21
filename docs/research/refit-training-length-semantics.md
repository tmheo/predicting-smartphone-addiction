# 전체 자료 재학습의 학습 길이 의미

이 문서는 GitHub 이슈 [모델별 최적 위치를 표준 학습 길이로 변환하는 규칙 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/326)의 조사 결과다.
조사 범위는 현재 [`artifacts/full-refit-plan.yaml`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/artifacts/full-refit-plan.yaml)의 32개 구성원 가운데 `model.kind`로 식별되는 반복형 모델 계열이다.
저장소 용어로 학습 길이는 조기 종료 없이 전체 자료 재학습에서 실제로 실행할 반복 횟수 또는 epoch 수다.
따라서 0부터 세는 최적 위치와 학습 길이는 같은 값이 아니다.

## 결론

계열 연결부는 원시 필드의 의미를 알고 아래 표준 학습 길이 `L`로 먼저 바꿔야 한다.
공통 계산부는 변환이 끝난 `L`만 받아 시드별 중앙값과 전체 자료 비율을 적용해야 한다.

```text
L = model_family_converter(raw_value)
B_seed = floor(1.25 * median(L_seed) + 0.5)
```

`L`과 `B_seed`는 모두 1 이상의 실제 실행 횟수다.
두 번째 식은 값이 양수일 때 `ROUND_HALF_UP` 사사오입과 같다.
배수 `1.25`, 시드별 중앙값, 사사오입은 [`docs/adr/0002-full-data-refit-protocol.md`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/docs/adr/0002-full-data-refit-protocol.md#L8-L16)에 이미 정해져 있다.

현재 반복형 9계열에서 `+1`이 필요한 외부 또는 저장 원시 필드는 XGBoost의 `best_iteration`, CatBoost의 `get_best_iteration()`, Lookup-Transformer의 `best_epoch`다.
PyTabKit TabM의 내부 로그 `epoch`도 0부터 세지만 저장소가 이를 `selected_epoch_count`로 기록할 때 이미 `+1`을 적용하므로 공통 계산부에서 다시 더하면 안 된다.
LightGBM 조기 종료 콜백 내부의 위치는 0부터 세지만 공개 추정기의 `best_iteration_`은 이미 1부터 세는 실제 부스팅 횟수다.
나머지 자체 구현의 저장 필드는 실제 epoch 수이므로 그대로 `L`로 쓴다.

## 현재 계획의 계열별 변환표

잠금 파일은 CatBoost `1.2.10`, LightGBM `4.7.0`, XGBoost `3.4.0`, PyTabKit `1.7.3`을 고정한다([`uv.lock`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/uv.lock)).
아래 의미는 이 판본과 현재 저장소 구현에 한정한다.

| 모델 계열 | 현재 구성원 | CV 원시 필드와 저장 상태 | 원시 의미 | 표준 학습 길이 `L` | 전체 자료 재학습 인수 | 근거 |
| --- | --- | --- | --- | --- | --- | --- |
| `lightgbm` | exp006_te_drop_gaming, exp011_resid_pair, exp022_orig_knn, exp023_orig_proxy_residual, exp025_constrained_impute, exp027_recon_ce, exp032_recon_orig_mean_top3, exp033_recon_orig_mean_top3_raw, exp035_lattice_te, exp048_lgb_orig_cdf_diff, exp110_lgb_kitopl_no_te, exp117_ag25_gbm_r21 | 라이브러리 `best_iteration_`; 현재 연결부는 구조화 학습 진단에 저장하지 않음 | 1부터 세는 최적 부스팅 횟수 | `L = best_iteration_` | `n_estimators=L` | LightGBM 콜백의 내부 위치는 0부터 세지만 학습 엔진이 `+1`하여 `Booster.best_iteration`에 넣는다([콜백](https://github.com/microsoft/LightGBM/blob/v4.7.0/python-package/lightgbm/callback.py#L43-L51), [학습 엔진](https://github.com/microsoft/LightGBM/blob/v4.7.0/python-package/lightgbm/engine.py#L305-L343)). 저장소 연결부는 전체 자료 경로에서 `training_budget`을 `n_estimators`에 넣는다([`model.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/model.py#L183-L242)). |
| `xgboost` | exp111_xgb_depth8_no_te, exp135_xgb_hpo_trial30 | `model_training_diagnostics.json`의 `details.best_iteration` | 0부터 세는 최적 부스팅 위치 | `L = best_iteration + 1` | `n_estimators=L` | XGBoost `3.4.0`은 첫 라운드의 `best_iteration`을 0이라고 명시하고, 예측에도 반열린 범위 `(0, best_iteration + 1)`을 쓴다([원본 코드](https://github.com/dmlc/xgboost/blob/v3.4.0/python-package/xgboost/sklearn.py#L1450-L1458), [속성 정의](https://github.com/dmlc/xgboost/blob/v3.4.0/python-package/xgboost/sklearn.py#L1627-L1638)). 저장소는 이 값을 그대로 진단에 저장하고 전체 자료 경로에서 `n_estimators`에 넣는다([`model.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/model.py#L327-L397)). |
| `catboost` | exp070_cat_exact_cats, exp071_cat_exact_no_te | 실행 로그의 `get_best_iteration()`; 현재 연결부는 구조화 학습 진단에 저장하지 않음 | 0부터 세는 최적 부스팅 위치 | `L = get_best_iteration() + 1` | `iterations=L` | 공식 문서는 반환값을 최적 iteration의 식별자로 정의하며, CatBoost 원본은 최적 위치에 `+1`한 수만큼 모형을 보존한다([공식 문서](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_best_iteration), [CatBoost `1.2.10` 원본](https://github.com/catboost/catboost/blob/v1.2.10/catboost/libs/train_lib/train_model.cpp#L517-L528)). 저장소는 전체 자료 경로에서 `training_budget`을 `iterations`에 넣는다([`model.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/model.py#L428-L486)). |
| `lookup_transformer` | exp059_lookup_transformer, exp081_lookup_fold_initialization_avg3, exp106_lookup_fixed24_train_test_preprocessing, exp127_lookup_muon, exp131_lookup_bivariate_plr5 | `details.fold_initialization_members[*].best_epoch`와 `observed_best_epoch` | 0부터 세는 epoch 위치 | `L = best_epoch + 1`; 최적 검증 선택에서는 `observed_best_epoch + 1`과 같음 | `epochs=L` | 학습 반복은 `range(epochs)`이고 선택 위치 `ep`를 그대로 저장한다([`lookup_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/lookup_transformer.py#L553-L685)). 전체 자료 경로는 전달된 양의 횟수만큼 같은 반복을 실행한다([`lookup_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/lookup_transformer.py#L441-L451)). |
| `contextualized_spline_transformer` | exp085_contextual_spline_m0 | `details.best_epoch`와 `observed_best_epoch` | 1부터 세는 실제 epoch 수 | `L = best_epoch` | `epochs=L` | 검증과 전체 자료 반복이 모두 `range(1, epochs + 1)`을 사용하고 그 값을 저장한다([`contextualized_spline_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/contextualized_spline_transformer.py#L911-L997)). |
| `tab_cnn` | exp113_tab_cnn_m0 | `details.best_epoch` | 1부터 세는 실제 epoch 수 | `L = best_epoch` | `training_budget=L` | 검증 반복은 `range(1, configured_epochs + 1)`이고 선택된 횟수를 그대로 `best_epoch`에 저장한다([`tab_cnn.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/tab_cnn.py#L455-L520)). 전체 자료 반복도 `range(1, training_budget + 1)`이다([`tab_cnn.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/tab_cnn.py#L588-L698)). |
| `realmlp` | exp124_realmlp_dtype_fix, exp134_realmlp_muon, exp136_realmlp_muon_recon_widths | `details.fixed_epochs` | 검증 선택이 없는 고정 실제 epoch 수 | `L = fixed_epochs` | `training_budget=L`, 내부 `fixed_epochs=L` | 학습은 `range(fixed_epochs)`를 정확히 실행하고 기록에는 `epoch + 1`을 남긴다([`realmlp.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/realmlp.py#L734-L815)). 전체 자료 연결부는 고정 횟수를 `fixed_epochs`로 설정한다([`realmlp.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/realmlp.py#L1084-L1144)). |
| `tabm` | exp137_tabm_recon_widths | `details.members[*].selected_epoch_count`; PyTabKit 내부 로그는 `epoch` | 내부 `epoch`는 0부터 세는 위치이고 저장 필드는 이미 실제 epoch 수 | `L = selected_epoch_count`; 내부 로그를 직접 읽을 때만 `L = epoch + 1` | `epochs=L`, 내부 `n_epochs=L` | PyTabKit `1.7.3`은 `range(n_epochs)`의 `epoch`를 최적 위치로 로그에 남긴다([원본 코드](https://github.com/dholzmueller/pytabkit/blob/v1.7.3/pytabkit/models/alg_interfaces/tabm_interface.py#L329-L405)). 저장소 수집기는 로그 값에 `+1`해 `selected_epoch_count`로 저장하고, 전체 자료 경로는 그 횟수를 `n_epochs`로 실행한다([`tabm.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/tabm.py#L95-L128), [`tabm.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/tabm.py#L245-L275)). |
| `scalar_token_transformer` | exp133_scalar_token_transformer_oof_te | 진입 진단의 `observations.best_epoch`; 현재 일반 CV 학습 진단에는 저장하지 않음 | 1부터 세는 실제 epoch 수 | `L = best_epoch` | `epochs=L` | 검증과 전체 자료 반복이 모두 `range(1, epochs + 1)`을 쓰며 선택 횟수를 그대로 기록한다([`scalar_token_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/scalar_token_transformer.py#L450-L525), [`scalar_token_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/scalar_token_transformer.py#L536-L636)). |

`model.fit_full()`의 공통 계약은 양의 정수 학습 길이만 허용하고, 모델별 해석은 연결부가 소유한다([`model.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/model.py#L145-L167)).
재학습 실행기는 계획의 시드별 `training_budget`을 이 계약에 그대로 전달한다([`refit.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/refit.py#L180-L226)).

## 현재 숫자에 대한 조건부 검사점

이 문서는 기존 실행의 원시 산출물을 전수 복원하지 않으므로 현재 장부 값을 직접 교정하지 않는다.
다만 ADR에 적힌 중앙값이 아래 원시 필드에서 나온 값으로 확인되면 전수 근거 조사에서 다음 교정 후보를 우선 확인해야 한다.

| 구성원 | ADR의 시드별 중앙값 | 원시 필드라고 가정한 표준 학습 길이 | 규약에 따른 예산 | 현재 예산 | 판정 조건 |
| --- | --- | --- | --- | --- | --- |
| exp135_xgb_hpo_trial30 | `7806`, `8314`, `8294` | `7807`, `8315`, `8295` | `9759`, `10394`, `10369` | `9758`, `10393`, `10368` | 중앙값이 XGBoost `best_iteration`에서 나온 0부터 세는 위치일 때 |
| exp127_lookup_muon | `9`, `11`, `11` | `10`, `12`, `12` | `13`, `15`, `15` | `11`, `14`, `14` | 중앙값이 Lookup-Transformer `best_epoch`에서 나온 0부터 세는 위치일 때 |
| exp131_lookup_bivariate_plr5 | `11`, `11`, `11` | `12`, `12`, `12` | `15`, `15`, `15` | `14`, `14`, `14` | 중앙값이 Lookup-Transformer `best_epoch`에서 나온 0부터 세는 위치일 때 |

이 계산은 원시 필드의 출처가 확인되기 전에는 조건부다.
ADR의 숫자가 이미 `+1`된 실제 횟수라면 다시 더하지 않아야 하므로, 숫자만 보고 장부를 바꾸면 안 된다.
exp106_lookup_fixed24_train_test_preprocessing은 설정의 실제 횟수 24가 근거이므로 예산 30이 이 변환 규칙과 일치한다.

## 내부 구성원이 여러 개인 계열

중앙값의 입력 단위는 모델 계열이 실제로 선택한 독립 모형 상태다.
한 바깥쪽 분할에서 초기화 구성원 여러 개가 각자 모형 상태를 선택하면 각 구성원의 표준 학습 길이를 모두 포함한다.

| 계열과 설정 | 시드당 중앙값 입력 | 근거 |
| --- | --- | --- |
| Lookup-Transformer 기본 단일 초기화 | 바깥쪽 분할 5개에서 각 1개, 총 5개 `L` | 기본 `fold_seed_offsets`는 `[0]`이고 진단은 구성원별로 저장된다([`lookup_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/lookup_transformer.py#L751-L776)). |
| exp081, exp127, exp131 Lookup-Transformer | 바깥쪽 분할 5개와 초기화 3개, 총 15개 `L` | 각 설정의 `fold_seed_offsets`가 `[0, 1000, 2000]`이고 전체 자료에서도 세 구성원을 같은 고정 횟수로 학습한다([`lookup_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/lookup_transformer.py#L847-L865)). |
| exp137 TabM | 바깥쪽 분할 5개와 내부 시드 3개, 총 15개 `L` | 연결부가 PyTabKit 구성원별 `selected_epoch_count`를 저장한다([`tabm.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/tabm.py#L201-L241)). |
| exp124, exp134, exp136 RealMLP | 바깥쪽 분할마다 같은 고정 `fixed_epochs` 1개 | 내부 초기화 두 개는 같은 고정 일정을 공유하므로 같은 수를 중복해도 중앙값은 변하지 않지만, 근거 단위는 설정의 고정 횟수다([`realmlp.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/realmlp.py#L1049-L1081)). |

## 경계 사례와 거부 규칙

### 첫 반복이 최적인 경우

XGBoost, CatBoost, Lookup-Transformer에서 원시 위치 `0`은 결측이나 무효값이 아니라 첫 반복이 선택됐다는 뜻이다.
표준 학습 길이는 `1`이어야 하며, 원시 값이 0이라는 이유로 설정 상한으로 대체하면 안 된다.
반대로 실제 횟수 필드에서 `0`은 유효한 학습 길이가 아니므로 거부해야 한다.

### 선택 위치와 학습 종료 위치

조기 종료의 인내 구간 때문에 실제 학습은 최적 모형 상태 뒤까지 진행될 수 있다.
재학습 길이의 근거는 `end_epoch`, 수행한 전체 반복 수, 또는 마지막 로그 위치가 아니라 복원된 checkpoint의 최적 위치다.
현재 자체 구현은 최적 가중치를 복원하므로 `best_epoch` 계열을 쓰고 `end_epoch`를 쓰지 않는다.

### Lookup-Transformer의 `validation_selection=final`

exp106은 검증 최고점이 아니라 설정에 고정된 마지막 epoch를 선택한다.
이 경로의 `best_epoch`는 실제로는 0부터 세는 선택 위치이며 `observed_best_epoch`는 관찰된 검증 최고점이다([`lookup_transformer.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/lookup_transformer.py#L655-L688)).
따라서 exp106의 근거는 설정의 `epochs=24`, 또는 같은 뜻인 저장 `best_epoch=23`을 `+1`한 24다.
전체 자료 비율은 그 고정 학습 길이 24에 적용한다.

### 고정 일정 RealMLP

RealMLP에는 검증 최고 위치가 없고 `fixed_epochs` 자체가 관측 학습 길이다.
현재 전체 자료 구현은 계산된 길이가 `schedule_epochs`보다 크면 거부한다([`realmlp.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/realmlp.py#L1084-L1096)).
따라서 미래 설정에서 1.25배 결과가 일정 지평을 넘으면 값을 조용히 자르지 말고 일정 지평과 학습률 규약을 함께 다시 결정해야 한다.

### 값이 없거나 조기 종료가 적용되지 않은 경우

최적 위치가 `None`, 음수, 비유한값이거나 필요한 구조화 경로에 없으면 변환하지 않고 근거 미확정으로 처리한다.
LightGBM의 `best_iteration_=0`은 조기 종료로 고른 학습 길이 0이 아니라 유효한 최적 반복이 설정되지 않은 상태이므로 거부한다.
CatBoost에서 검증 집합이 없으면 `get_best_iteration()`은 `None`일 수 있다([공식 문서](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_best_iteration)).
고정 일정 계열은 실행 진단이 없더라도 실행에 연결된 설정과 커밋으로 설정값을 확인할 수 있을 때만 그 값을 근거로 쓴다.

### 사사오입과 중앙값

0부터 세는 위치에는 중앙값을 내기 전에 구성원별로 `+1`해야 한다.
일반적으로 `median(position) + 1`과 `median(position + 1)`은 같지만, 원시 의미가 섞이는 것을 막고 구성원별 계보를 남기기 위해 변환을 먼저 수행한다.
사사오입은 Python 내장 `round()`의 짝수 쪽 반올림이 아니라 양수에 대한 `floor(value + 0.5)` 또는 `Decimal(...).quantize(1, rounding=ROUND_HALF_UP)`이어야 한다.

## 반복형 범위에서 제외되는 구성원

| 모델 계열 | 현재 구성원 | 제외 이유 | 전체 자료 계약 근거 |
| --- | --- | --- | --- |
| `logistic_onehot` | exp058_logreg_onehot, exp107_logreg_onehot_nn10, exp108_logreg_onehot_nn10_l1 | `max_iter`는 수렴 최적화의 안전 상한이며 CV에서 선택해 옮기는 학습 길이가 아님 | 전체 자료 연결부는 `training_budget`이 `None`이 아니면 거부하고 같은 수렴 조건으로 학습한다([`model.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/model.py#L652-L703)). |
| `tabpfn3` | exp067_tabpfn3 | `n_estimators`는 사전 학습 모형의 추론 앙상블 크기이며 이 자료에서 CV로 선택한 반복 학습 길이가 아님 | 전체 자료 연결부는 `training_budget`이 `None`이 아니면 거부한다([`model.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/model.py#L1181-L1194)). |

모델 계열은 저장소 용어상 `model.kind` 하나로 식별된다.
따라서 exp023의 `initial_score.original_proxy_lightgbm`과 피처 제공자가 내부에서 학습하는 모형은 구성원 모델의 학습 길이 변환 대상이 아니다.
이 보조 학습들은 `training_budget`을 받지 않으며, 구성원 모델의 표준 학습 길이에 합치거나 대신 사용할 수 없다([`initial_score.py`](https://github.com/tmheo/predicting-smartphone-addiction/blob/8d4ba0a19127d2b7db5a793a8cf980d2c310768d/src/pipeline/initial_score.py#L110-L155)).

## 구현 규약으로 넘길 최소 자료형

공통 계산부가 필드 이름을 추측하지 않도록 계열 연결부는 최소한 다음 의미를 가진 값을 만들어야 한다.

| 필드 | 의미 |
| --- | --- |
| `raw_field` | 원본 필드 이름과 중첩 경로 |
| `raw_value` | 출처에서 읽은 변경 전 값 |
| `raw_semantics` | `zero_based_position`, `one_based_count`, `fixed_count` 중 하나 |
| `training_length` | 계열 변환을 마친 1 이상의 실제 실행 횟수 |
| `source_run_id`, `seed`, `fold`, `inner_member` | 값을 원래 모형 상태까지 되짚는 계보 |

공통 계산부는 `training_length`만으로 중앙값과 1.25배 사사오입을 수행하고, 원시 값과 변환 의미는 결과 장부에 함께 보존해야 한다.
이 규약이면 같은 `best_epoch` 이름을 가진 Lookup-Transformer와 자체 1부터 세는 계열을 잘못 합치는 오류를 구조적으로 막을 수 있다.
