# 트리 3종 튜닝 탐색 공간과 예산 근거 조사

이슈 [#275](https://github.com/tmheo/predicting-smartphone-addiction/issues/275)(지도 [#273](https://github.com/tmheo/predicting-smartphone-addiction/issues/273)의 research 티켓)의 조사 기록이다.
조사일: 2026-08-20.
질문은 셋이다.
첫째, AutoGluon zeroshot portfolio 2025가 쓰는 트리 3종(LightGBM, XGBoost, CatBoost)의 탐색 공간.
둘째, 이 저장소 디스커션 조사 문서에 실린 튜닝 실측("25~50 트라이얼 수확 체감", "GPU XGBoost Optuna 200 트라이얼 T4 약 1.5시간")의 원문 맥락.
셋째, 이 데이터 규모(train 691,369행 x 14열, 이진 분류, AUC)에서 라이브러리 공식 문서 기준의 권장 탐색 설계.

## 1. AutoGluon zeroshot portfolio 2025의 트리 3종 탐색 공간

### 1.1 포트폴리오 설정의 성격과 출처

이 저장소가 [#197](https://github.com/tmheo/predicting-smartphone-addiction/issues/197)에서 이식한 포트폴리오의 출처는 [autogluon/autogluon](https://github.com/autogluon/autogluon)의 `tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2025.py`, 커밋 `2d7e6056b8b64dc44114faf652d4c99ec3c3770f`이다(`docs/research/zeroshot-portfolio-2025-screen.md`).
원본 파일 머리말은 이 포트폴리오가 "optimized for <=10000 samples and <=500 features, with a GPU present"라고 명시한다(2026-08-20 원본 재확인).
설정 이름의 `_rXX` 접미사는 무작위 추첨 설정(random config)의 번호다.
즉 포트폴리오의 GBM `_r21` 같은 항목은 탐색 공간 그 자체가 아니라, 아래 1.2의 탐색 공간에서 무작위로 뽑아 대규모 벤치마크로 평가한 뒤 zeroshot 시뮬레이션으로 선별한 "당첨 지점"들이다.
따라서 "검증된 탐색 공간"의 1차 출처는 그 무작위 설정들을 생성한 공간 정의 코드다.

### 1.2 설정을 생성한 탐색 공간: TabArena(구 TabRepo)의 hpo.py

무작위 설정 생성기는 [autogluon/tabarena](https://github.com/autogluon/tabarena)(구 tabrepo, main 브랜치 커밋 `4327429`, 2026-08-19 기준)의 `packages/tabarena/src/tabarena/models/{lightgbm,xgboost,catboost}/hpo.py`에 있다.
세 파일 모두 ConfigSpace로 공간을 정의하고 기본 200개(`num_random_configs=200`)를 무작위 추첨한다(시드 1234).
아래 표는 소스 코드를 그대로 옮긴 것이다.

LightGBM (`generate_configs_lightgbm`):

| 파라미터 | 분포 | 범위 | 비고 |
| --- | --- | --- | --- |
| learning_rate | Float, log | 5e-3 ~ 1e-1 | |
| feature_fraction | Float | 0.4 ~ 1.0 | |
| bagging_fraction | Float | 0.7 ~ 1.0 | |
| bagging_freq | Categorical | {1} | 고정 |
| num_leaves | Int, log | 2 ~ 200 | |
| min_data_in_leaf | Int, log | 1 ~ 64 | |
| extra_trees | Categorical | {False, True} | |
| min_data_per_group | Int, log | 2 ~ 100 | 범주형용 |
| cat_l2 | Float, log | 5e-3 ~ 2 | 범주형용 |
| cat_smooth | Float, log | 1e-3 ~ 100 | 범주형용 |
| max_cat_to_onehot | Int, log | 8 ~ 100 | 범주형용 |
| lambda_l1 | Float | 1e-4 ~ 1.0 | 주석: "these seem to help a little bit but can also make things slower" |
| lambda_l2 | Float | 1e-4 ~ 2.0 | 같은 주석 |
| (max_bin) | 탐색 안 함 | - | 주석: "could search max_bin but this is expensive" |

XGBoost (`generate_configs_xgboost`):

| 파라미터 | 분포 | 범위 | 비고 |
| --- | --- | --- | --- |
| learning_rate | Float, log | 5e-3 ~ 1e-1 | |
| max_depth | Int, log | 4 ~ 10 | |
| min_child_weight | Float, log | 1e-3 ~ 5.0 | |
| subsample | Float | 0.6 ~ 1.0 | |
| colsample_bylevel | Float | 0.6 ~ 1.0 | |
| colsample_bynode | Float | 0.6 ~ 1.0 | |
| reg_alpha | Float | 1e-4 ~ 5.0 | |
| reg_lambda | Float | 1e-4 ~ 5.0 | |
| grow_policy | Categorical | {depthwise, lossguide} | |
| max_cat_to_onehot | Int, log | 8 ~ 100 | |
| max_leaves | Int, log | 8 ~ 1024 | |
| enable_categorical | 고정 True | - | 추첨 후 일괄 부여 |
| (max_bin, num_parallel_tree) | 탐색 안 함 | - | 주석: "could search max_bin and num_parallel_tree but this is expensive" |

CatBoost (`generate_configs_catboost`):

| 파라미터 | 분포 | 범위 | 비고 |
| --- | --- | --- | --- |
| learning_rate | Float, log | 5e-3 ~ 1e-1 | |
| bootstrap_type | Categorical | {Bernoulli} | 고정, 주석: "this is a bit faster than 'Bayesian'" |
| subsample | Float | 0.7 ~ 1.0 | |
| grow_policy | Categorical | {SymmetricTree, Depthwise} | |
| depth | Int | 4 ~ 8 | 주석: "not too large for compute/memory reasons" |
| colsample_bylevel | Float | 0.85 ~ 1.0 | |
| l2_leaf_reg | Float, log | 1e-4 ~ 5.0 | |
| leaf_estimation_iterations | Int, log | 1 ~ 20 | |
| one_hot_max_size | Int, log | 8 ~ 100 | 범주형용 |
| model_size_reg | Float, log | 0.1 ~ 1.5 | |
| max_ctr_complexity | Int | 2 ~ 5 | 범주형용 |
| boosting_type | Categorical | {Plain} | 고정, GPU/CPU 동일 설정 목적 주석 |
| max_bin | Categorical | {254} | 고정, 주석: "could be tuned, in principle" |
| (min_data_in_leaf) | 탐색 안 함 | - | 주석: Depthwise에서만 동작해서 보류 |
| (random_strength) | 탐색 안 함 | - | 주석: "could add random_strength here but leaving it out for now" |

교차 검증: 이 저장소가 이식한 설정들의 값이 전부 위 범위 안에 있음을 대조로 확인했다.
예를 들어 GBM `_r21`(`configs/exp117_ag25_gbm_r21.yaml`)의 learning_rate 0.00559, feature_fraction 0.456, bagging_fraction 0.722, num_leaves 30, min_data_in_leaf 50, cat_smooth 0.00103, max_cat_to_onehot 71이 모두 LightGBM 공간 안이고, XGB `_r40`의 max_depth 10, max_leaves 35, min_child_weight 0.140, reg_alpha 3.496, subsample 0.695도 XGBoost 공간 안이며, CAT `_r51`의 depth 7, colsample_bylevel 0.877, max_ctr_complexity 4, max_bin 254, boosting_type Plain, bootstrap_type Bernoulli도 CatBoost 공간 안이다.
포트폴리오 원본에서 각 계열이 실제로 변주한 파라미터 집합도 위 공간의 파라미터 집합과 정확히 일치한다(`scripts/screen_zeroshot_portfolio.py`에 이식된 dict 전체 참조).

### 1.3 대비: AutoGluon 본체의 legacy HPO 공간은 훨씬 좁다

AutoGluon 본체에도 `hyperparameter_tune_kwargs`용 기본 탐색 공간이 따로 있다(master 브랜치, `tabular/src/autogluon/tabular/models/*/hyperparameters/searchspaces.py`).
이진 분류 기준으로 LightGBM은 learning_rate(5e-3~0.2, log), feature_fraction(0.75~1.0), min_data_in_leaf(2~60), num_leaves(16~96)의 4개뿐이다.
XGBoost는 learning_rate(5e-3~0.2, log), max_depth(3~10), min_child_weight(1~5), colsample_bytree(0.5~1.0)의 4개이고, gamma·subsample·reg_alpha·reg_lambda는 "Below lines are commented out as they made search worse."라는 주석과 함께 꺼져 있다.
CatBoost는 learning_rate(5e-3~0.2, log), depth(5~8), l2_leaf_reg(1~5)의 3개다.
시사점이 둘이다.
첫째, 파라미터 수를 3~4개로 줄인 좁은 공간도 AutoGluon이 실전 기본값으로 채택할 만큼 유효하다.
둘째, XGBoost의 subsample·정칙화 항은 "공간에 넣으면 오히려 탐색을 해친다"는 기록(legacy)과 "포트폴리오 당첨 설정이 그 축에서 이득을 봤다"는 기록(2025, `_r40`의 subsample 0.695·reg_alpha 3.5)이 공존하므로, 데이터마다 갈리는 축이라 보고 넣되 결과를 의심하며 봐야 한다.

### 1.4 이 저장소 실측과의 연결

fold 0 약식 검증(`docs/research/zeroshot-portfolio-2025-screen.md`)에서 이득이 확인된 축은 "낮은 학습률 + 강한 정칙화·부분표본" 하나였다.
GBM `_r21`(+0.00094)과 XGB `_r40`(+0.00047)이 통과했고, CatBoost 5설정은 전부 기준 ±0.0002 안이라 포트폴리오 이득이 없었다.
당첨 설정 상위 세 개가 서로 스피어만 0.998 이상이라 설정 수 확장은 다양성 이득이 없다는 판정도 이미 내려져 있다.
따라서 이 저장소에서 튜닝을 연다면 목적은 "포트폴리오 지점 근방의 국소 개선"이지 공간 전체 재탐색이 아니다.

## 2. 디스커션 튜닝 실측의 원문 맥락

### 2.1 "GPU XGBoost Optuna 200 트라이얼이 T4 약 1.5시간"

출처는 S6E8 디스커션 732985 "XGBoost + Optuna on GPU | 0.96514 LB - sharing what worked"(작성자 Rugved Bane, 최종 502위)이다.
URL: <https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985> (2026-08-20 Jina Reader로 원문 재확인).
조건이 중요하다.
같은 S6E8 train 데이터(691,369행)에서 XGBoost를 `device='cuda'` + `tree_method='hist'`로 돌렸고, 트라이얼당 검증은 3-fold stratified shuffle split이었다.
그 조건에서 원문 표현으로 "Around 1.5 hours for 200 trials on Kaggle T4 GPU"다.
즉 트라이얼당 약 27초이고, 이는 GPU + 3-fold 약식 검증 기준이지 이 저장소의 정식 5-fold(로컬 CPU) 기준이 아니다.

### 2.2 "튜닝은 25~50 트라이얼에서 수확 체감"

같은 스레드의 코멘트 두 개가 근거다(`docs/research/discussion-batch-b.md` 7~33행).
Zih-Chen Hung(516위)은 Optuna 25 트라이얼만으로 LB 0.96572를 얻어 본문의 200 트라이얼 결과(0.96514)를 웃돌았고, "200 trials seems like a lot for the gain... Might be hitting diminishing returns pretty early on."이라고 지적했다.
작성자 본인도 100에서 200으로 늘린 것이 "0.003~0.004% 수준"의 이득뿐이라 다시는 안 하겠다고 인정했다.
저장소 문서의 "25~50 트라이얼" 권고(`discussion-batch-b.md` 33행, `discussion-insights.md` 239~241행)는 이 두 실측(25 트라이얼로 충분 + 100 초과는 무의미)을 보수적으로 묶은 저장소 자체의 결론이며, 원문에 "25~50"이라는 숫자 구간이 그대로 있는 것은 아니다.
같은 스레드에서 Tilii(30위)의 조언 두 가지도 원문 확인했다.
n_estimators는 튜닝 대상이 아니고 "put a large number there that will never be reached (say, 100_000) and use early stopping"이 맞으며, learning_rate는 탐색 단계에서 0.02~0.05로 고정하고 최종 런에서만 0.01 또는 0.005로 낮추라는 것이다.

### 2.3 Deotte의 GPU 루프는 하이퍼파라미터 튜닝이 아니다

`docs/research/playground-meta-insights.md` 49~57행의 "GPU로 수 일" 루프는 S4E12 1위 Chris Deotte의 피처 조합 탐색이다.
URL: <https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328>.
컬럼 2~3개를 무작위로 뽑아 TE/CE 후보 피처를 만들고 CV가 오르면 채택하는 시행을 수천 번 반복해 강한 조합 약 170개를 수집했고, 611피처 단일 XGBoost로 우승했다.
즉 이 예산은 피처 탐색 예산이지 하이퍼파라미터 예산이 아니므로, 튜닝 예산 산정의 전거로 쓰면 안 된다.
다만 그 절의 프록시 CV 규율(fold 하나만 holdout, 커밋된 `folds.parquet` fold 재사용, 상위 후보만 정식 5-fold 재검증)은 튜닝에도 그대로 이전할 수 있는 충실도 축소 절차다(같은 문서 57~63행).

## 3. 이 데이터 규모에서의 권장 탐색 설계

### 3.1 공통 원칙과 근거

부스팅 라운드 수는 탐색하지 않는다.
LightGBM 공식 튜닝 문서가 num_iterations와 learning_rate를 짝으로 묶고(작은 학습률 + 큰 반복 수가 정확도 축), XGBoost 공식 튜닝 노트도 "reduce stepsize eta... Remember to increase num_round"라고 짝으로 다루며, CatBoost 공식 문서도 "iterations를 크게 + overfitting detector(use_best_model)"를 권한다.
Tilii의 실전 조언(2.2)과 이 저장소 규약(n_estimators/iterations 10000 + early_stopping_rounds 200)도 같다.
learning_rate 취급은 두 근거가 갈린다.
Tilii는 탐색 중 0.02~0.05 고정을 권하지만, 이 저장소 fold 0 실측에서는 낮은 학습률(0.0056~0.011) 자체가 이득 축이었다(1.4).
절충안은 learning_rate를 TabArena와 같은 log 5e-3~1e-1로 공간에 넣되, 탐색 시간을 아끼려면 상한을 5e-2로 줄이는 것이다(포트폴리오 당첨 3설정이 전부 0.016 이하였다).
트라이얼 예산은 라이브러리당 30~50이면 충분하다는 것이 실측 근거다(2.2).
공식 자료 쪽 근거로는, Optuna TPESampler가 처음 10 트라이얼을 무작위로 쓰고(`n_startup_trials` 기본 10) 그 뒤부터 TPE가 작동하므로 10 이하 예산은 무의미하고, Optuna 공식 LightGBMTuner의 단계별 총예산이 68 트라이얼(feature_fraction 7 + num_leaves 20 + bagging 10 + feature_fraction 2단계 6 + lambda_l1/l2 20 + min_child_samples 5, optuna-integration `_lightgbm_tuner/optimize.py`)이라는 사실이 "수십 트라이얼 규모"의 상식선을 보여 준다.
충실도 축소는 이 저장소 관례를 그대로 쓴다.
커밋된 `artifacts/folds.parquet`의 fold 0 하나를 holdout으로, 시드 1개(42), early stopping 200으로 트라이얼을 돌리고(#48 규약, `scripts/screen_zeroshot_portfolio.py` 선례), 상위 1~2개만 seed 42 5-fold 스크리닝과 3시드 확정 재검증으로 올린다.
fold 0 약식은 GBDT 설정당 로컬 CPU(14코어)에서 이미 13설정을 소화한 실측 경로라 비용 추정도 그 기록을 그대로 쓸 수 있다.
Optuna를 쓴다면 TPESampler(multivariate=True 권장, 공식 문서가 독립 TPE보다 낫다고 보고)로 단일 공간을 돌리는 편이 단계별 그리드보다 저장소 구조에 맞다.

### 3.2 LightGBM

우선 탐색: num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction(+bagging_freq=1 고정), lambda_l1, lambda_l2, learning_rate.
근거: 공식 튜닝 문서가 num_leaves를 "This is the main parameter to control the complexity of the tree model"로, min_data_in_leaf를 "a very important parameter to prevent over-fitting in a leaf-wise tree"로 못박고, 대용량 데이터에서는 min_data_in_leaf를 수백~수천으로 두라고 권한다(<https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html>).
과적합 대응 축으로 bagging·feature 부분표본과 lambda_l1/l2도 같은 문서가 명시한다.
범위는 TabArena 공간(1.2)을 그대로 쓰되, 691k행 실측을 반영해 min_data_in_leaf 상한을 64에서 수백대로 늘리는 것이 문서 권고에 더 맞다(exp074의 min_child_samples 229가 이미 그 대역에서 검증된 지점이다).
고정: objective/metric, n_estimators 10000 + ES 200, max_bin(TabArena도 비용 사유로 탐색 제외, 단 exp074의 max_bin 1023이 검증된 값이므로 1023 고정 변형은 별도 한 점으로만 확인).
extra_trees는 포트폴리오에서 True인 당첨 설정(`_r11`)이 이 데이터에서 절대 하한 미달로 탈락했으므로 False 고정이 안전하다.

### 3.3 XGBoost

우선 탐색: grow_policy{depthwise, lossguide} + max_depth(4~10, log) + max_leaves(8~1024, log), min_child_weight(1e-3~5, log), subsample(0.6~1.0), colsample_bylevel/bynode(0.6~1.0), reg_alpha/reg_lambda(1e-4~5), learning_rate.
근거: 공식 튜닝 노트가 과적합 제어를 "모델 복잡도 직접 제어(max_depth, min_child_weight, gamma)"와 "무작위성 추가(subsample, colsample_bytree)"의 두 축으로 정리하고(<https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html>), 파라미터 문서가 subsample은 uniform 샘플링 기준 0.5 이상을 권한다(<https://xgboost.readthedocs.io/en/stable/parameter.html>).
불균형(양성 70.9%)은 공식 노트 기준 "Use AUC for evaluation"으로 충분하며 scale_pos_weight는 확률 자체가 필요할 때의 도구라 AUC 지표인 이 대회에서는 건드리지 않는다(저장소 디스커션 결론 731764와 일치).
고정: tree_method hist, eval_metric auc, n_estimators + ES(저장소 규약), max_bin 기본 256(TabArena도 비용 사유로 탐색 제외).
gamma는 AutoGluon legacy가 "made search worse"로 껐고 2025 공간에도 없으므로 넣지 않는다.
이 저장소의 XGBoost 재직 구성원(exp045/exp111)은 아직 lr 0.05 + depth 8 수동값이라 트리 3종 중 탐색 여지가 가장 크고, 실제로 `_r40`(lossguide, max_leaves 35, lr 0.011)이 fold 0에서 base_xgb를 +0.00047 웃돈 것이 그 증거다.

### 3.4 CatBoost

우선 탐색: depth(4~8), l2_leaf_reg(log), learning_rate, subsample(Bernoulli), colsample_bylevel(0.85~1.0), leaf_estimation_iterations, one_hot_max_size, grow_policy{SymmetricTree, Depthwise}.
근거: 공식 파라미터 튜닝 문서가 "In most cases, the optimal depth ranges from 4 to 10. Values in the range from 6 to 10 are recommended."라 하고, learning_rate는 데이터와 iterations에서 자동 산정되므로 과적합 여부를 보고 조정하라고 안내한다(<https://catboost.ai/docs/en/concepts/parameter-tuning>).
random_strength와 bagging_temperature도 공식 튜닝 항목이지만, TabArena는 bootstrap_type을 Bernoulli로 고정해 bagging_temperature(Bayesian 전용) 축을 아예 닫았고 random_strength도 보류했다.
고정: iterations 10000 + ES 200, eval_metric AUC, boosting_type Plain, max_bin 254(공식 문서도 CPU 최고 품질 기준 border_count 254), bootstrap_type Bernoulli.
max_ctr_complexity는 주의 대상이다.
TabArena 공간은 2~5를 탐색하지만, 이 저장소 CatBoost 재직 구성원(exp070/exp071)은 수치 9열의 정확값 범주 복제와 조합 CTR의 폭발을 막으려고 max_ctr_complexity=1을 의도적으로 고정했다(#107 설계).
범주 복제 피처 계획을 유지하는 한 이 고정은 탐색에서 풀면 안 되고, 풀려면 범주 복제가 없는 피처 계획에서만 푼다.
포트폴리오 실측에서 CatBoost 5설정이 전부 기준 ±0.0002 안이었다는 사실(1.4)도 이 계열의 튜닝 기대 이득이 트리 3종 중 가장 작음을 시사한다.

### 3.5 저장소 현재 파라미터의 위치 대비

| 계열 | 설정 | 핵심 파라미터 | 출처 | TabArena 공간과의 관계 |
| --- | --- | --- | --- | --- |
| LightGBM | exp001 기본선 | lr 0.05, num_leaves 255 | 저장소 장수 기본값 | num_leaves 255는 공간 상한(200) 밖, lr은 상한 근처 |
| LightGBM | exp074 (kitopl D2) | lr 0.0317, leaves 45, min_child_samples 229, max_bin 1023, colsample_bytree 0.465, subsample 0.923, reg_lambda 0.918 | 공개 노트북 kitopl 이식(#122) | min_child_samples 229와 max_bin 1023은 공간 밖, 나머지는 공간 안 대역 |
| LightGBM | exp117 (AG `_r21`) | lr 0.0056, leaves 30, ff 0.456, bf 0.722, l1 0.522, l2 0.107 | AutoGluon 포트폴리오 당첨점(#197) | 공간 안(정의상) |
| XGBoost | exp045/exp111 | hist, depthwise, depth 8, lr 0.05 | 수동 용량 비교(#59) | depth는 공간 안, lr은 상한 근처, 부분표본·정칙화 미사용 |
| XGBoost | exp119 (AG `_r40`, 스크리닝만) | lossguide, depth 10, max_leaves 35, lr 0.011, subsample 0.695, reg_alpha 3.5 | AutoGluon 포트폴리오 당첨점 | 공간 안(정의상), exp117과 상관 0.9989라 풀 미진입 |
| CatBoost | exp070/exp071 | 기본값 + depth 6, lr 0.05, max_ctr_complexity 1 | CatBoost 기본값 + #107 설계 | depth·lr은 공간 안, max_ctr_complexity 1은 공간(2~5) 밖의 의도적 고정 |

관찰: LightGBM은 이미 공간 안 당첨점(exp117)과 공간 밖 대용량 특화점(exp074, 큰 min_child_samples와 max_bin)을 모두 보유해 추가 튜닝의 한계 이득이 작다.
XGBoost는 재직 구성원이 수동값 그대로라 탐색 여지가 가장 크지만, 공간 안 당첨점(exp119)이 exp117과 0.9989 상관으로 풀에 못 들어간 전례가 있어, 튜닝의 목표를 "점수"가 아니라 "기존 풀과의 낮은 상관"으로 잡아야 실익이 있다.
CatBoost는 포트폴리오 전 설정이 기준과 동급이었으므로 튜닝 우선순위가 가장 낮다.

### 3.6 권장 예산 설계 요약

- 검증 충실도: fold 0 holdout(커밋된 `folds.parquet` 재사용), 시드 1개(42), n_estimators 큰 값 + early stopping 200. 근거: #48 규약과 `screen_zeroshot_portfolio.py` 선례, playground-meta-insights의 프록시 CV 규율.
- 트라이얼 예산: 라이브러리당 30~50(TPESampler multivariate). 근거: 25 트라이얼로 200 트라이얼급 LB 도달 실측(732985 코멘트), TPE 무작위 시동 10 트라이얼, LightGBMTuner 공식 총예산 68.
- 공간: TabArena hpo.py 범위를 기본으로 하되, LightGBM min_data_in_leaf 상한을 수백대로 확장(공식 문서의 대용량 권고 + exp074 실측)하고, CatBoost max_ctr_complexity는 피처 계획에 종속시킨다.
- 승격 절차: 프록시 상위 1~2개만 seed 42 5-fold 스크리닝, 통과 시 3시드 확정. 판정은 점수와 함께 기존 풀과의 스피어만 상관(중복 게이트 0.998)을 반드시 본다.

## 출처 목록

1차 출처(2026-08-20 확인):

- AutoGluon zeroshot portfolio 2025: <https://github.com/autogluon/autogluon/blob/2d7e6056b8b64dc44114faf652d4c99ec3c3770f/tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2025.py>
- TabArena 탐색 공간(LightGBM/XGBoost/CatBoost): <https://github.com/autogluon/tabarena/tree/main/packages/tabarena/src/tabarena/models> (main 커밋 `432742939067b75c1d16627c8cef010eeb9d68c6` 기준, lightgbm/hpo.py 최종 변경 커밋 `df2a0e30d09a86fbf21aec6739f873a12c36e19a`)
- AutoGluon legacy HPO 공간: `tabular/src/autogluon/tabular/models/{lgb,xgboost,catboost}/hyperparameters/searchspaces.py` (master)
- S6E8 디스커션 732985(원문 재확인): <https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732985>
- Deotte S4E12 피처 루프: <https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328>
- LightGBM 공식 튜닝 문서: <https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html>
- XGBoost 공식 튜닝 노트와 파라미터 문서: <https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html>, <https://xgboost.readthedocs.io/en/stable/parameter.html>
- CatBoost 공식 파라미터 튜닝 문서: <https://catboost.ai/docs/en/concepts/parameter-tuning>
- Optuna TPESampler: <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html>
- Optuna LightGBMTuner 단계별 예산: <https://github.com/optuna/optuna-integration/blob/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py>

저장소 내부 전거:

- `docs/research/zeroshot-portfolio-2025-screen.md`, `scripts/screen_zeroshot_portfolio.py`
- `docs/research/discussion-batch-b.md` 7~36행, `docs/research/discussion-insights.md` 236~242행, `docs/research/playground-meta-insights.md` 49~63행
- `configs/exp001_lgbm_baseline.yaml`, `configs/exp074_lgb_kitopl_d2_bundle.yaml`, `configs/exp117_ag25_gbm_r21.yaml`, `configs/exp045_xgb_depth8.yaml`, `configs/exp111_xgb_depth8_no_te.yaml`, `configs/exp119_ag25_xgb_r40.yaml`, `configs/exp070_cat_exact_cats.yaml`, `configs/exp071_cat_exact_no_te.yaml`, `configs/exp120_ag25_cat_default.yaml`
