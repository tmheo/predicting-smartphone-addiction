# 장부 밖의 검증 가능한 개별 외부 구성원 전수 조사

작성일은 2026-08-28이고 기준 시각대는 KST이다.

## 결론

외부 구성원 장부 판본 2에 없으면서 지도의 엄격한 자격 규칙을 모두 통과한 공개 OOF·시험 예측 후보는 4개 노트북 출처의 8개 구성원이다.
후보는 Beicicc의 TabNet과 RealMLP, BusyPrime의 LightGBM·HistGradientBoosting·XGBoost, Ravi의 XGBoost·LightGBM·CatBoost이다.
8개 모두 OOF 691,369행과 시험 예측 296,302행이 있고, 값이 모두 유한하며, 저장소의 `StratifiedKFold(5, shuffle=True, random_state=42)`에 맞는 원래 행 순서를 코드와 저장 산출물로 확인할 수 있다.
8개 모두 장부 판본 2의 400개 통과 구성원과 정확한 OOF·시험 쌍 해시가 다르다.
장부의 스피어만 0.998 규칙을 적용하면 Beicicc RealMLP와 BusyPrime LightGBM에는 근접 중복 주의 사항이 붙지만, 판본 2의 정책처럼 이를 자격 제외 사유로 쓰지 않는다.
2026-08-27 조사 뒤 갱신된 Kaggle 자료 6개에서는 새 OOF·시험 예측 쌍이 나오지 않았다.

## 판정 규칙

[상위 지도 477번](https://github.com/tmheo/predicting-smartphone-addiction/issues/477)의 정의와 결정을 그대로 적용했다.
후보는 공식 훈련 자료에서 직접 학습한 하나의 모델 계보가 만든 OOF·시험 예측 한 쌍이어야 한다.
공개된 고정 판본에서 바깥쪽 검증 행의 목표값이 모델 학습, 목표값 기반 전처리, 학습 시점 선택과 설정 선택에 닿지 않았음을 소스나 재현 가능한 저장 산출물로 확인할 수 있어야 한다.
검증 분할을 이용한 조기 종료나 최적 상태 선택, 여러 모델이나 설정의 결합, 외부 예측 재학습, 의사 목표값 학습과 결합 예측 증류는 제외했다.
공개 이전의 설정 탐색 이력은 알 수 없음으로 남기고 자동 제외하지 않았지만, 공개 소스가 특정 바깥쪽 검증 결과로 설정을 골랐다고 밝힌 경우에는 제외했다.
같은 고정 설정에서 여러 시드 예측을 평균하는 일과 시험 자료의 목표값을 쓰지 않는 무상태 전처리는 허용했다.
분할 벡터, 같은 작성자의 다른 코드나 작성자 설명만으로는 자격을 주지 않았다.

## 고정 조사 범위

공식 Kaggle CLI 2.2.4의 [노트북 명령](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)과 [자료 명령](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)을 사용했다.
2026-08-28 현재 `playground-series-s6e8`의 `dateRun` 정렬 공개 노트북은 615개이고, 그중 저장 출력 자료가 있는 노트북은 476개이다.
이전 452번 조사가 확인한 당시 상위 500개에 현재 새로 실행되거나 갱신된 18개와 현재 목록의 오래된 꼬리 115개를 대조했고, 저장 출력 자료가 있는 오래된 꼬리 76개의 파일 목록을 모두 확인했다.
그 결과 현재 615개 모집단에서 이전 조사가 보지 못한 구간과 갱신 구간을 모두 덮었다.
OOF처럼 보이는 파일이 있으면 현재 고정 판본의 소스, 출력 파일 목록, 저장 배열과 연결 입력을 순서대로 확인했다.
결합 노트북은 제외하지 않고 개별 기초 모델의 출처를 찾는 경로로 사용했지만, 결합 결과 자체에는 자격을 주지 않았다.

자료 검색은 아래 17개 검색어를 각각 `updated` 정렬 100개씩 3쪽까지 실행했다.

`s6e8`, `oof`, `smartphone addiction`, `playground s6e8`, `playground-series-s6e8`, `blend members`, `oof library`, `stack members`, `smartphone oof`, `addiction oof`, `s6e8 stack`, `s6e8 oof`, `s6e8 members`, `s6e8 predictions`, `s6e8 test preds`, `s6e8 artifacts`, `s6e8 submission`을 고정 검색어로 썼다.

51번의 목록 호출은 294개 결과와 148개 고유 자료 참조를 반환했다.
452번 결론 시각인 `2026-08-27T07:37:48Z` 뒤 갱신된 자료는 6개였다.
[Anthony의 vault](https://www.kaggle.com/datasets/anthonytherrien/predicting-smartphone-addiction-vault)는 제출 CSV 2개만 있었고, [Stephen의 artifacts](https://www.kaggle.com/datasets/stephentarter/ps-s06e08-artifacts)는 설정 JSON만 있었다.
[Naji의 S6E8 PSA](https://www.kaggle.com/datasets/najiama/s6e8-psa)와 [Qamrodz의 submission v2](https://www.kaggle.com/datasets/qamrodz/prediction-smartphone-addiction-submission-v2)는 제출 CSV만 있었다.
[Dariush의 leaderboard intelligence](https://www.kaggle.com/datasets/dariushafshar/kaggle-competition-leaderboard-intelligence)는 대회 목록 분석 자료였고, [Xainab의 ResNet18 OOF](https://www.kaggle.com/datasets/xainab123/resnet18-oof-prediction)는 이 대회와 무관한 작은 NPZ였다.
따라서 갱신 자료 6개에서는 새 후보가 나오지 않았다.

## 통과 후보 8개

아래 쌍 해시는 OOF와 시험의 양성 확률을 각각 float64 연속 배열로 바꾼 뒤 순서대로 이어 붙여 계산한 SHA-256이다.

| 출처와 구성원 | OOF AUC | 쌍 SHA-256 | 장부 판본 2 대조 |
| --- | ---: | --- | --- |
| `beicicc/s6e8-fold-safe-tabnet:tabnet` | 0.965656810 | `b339d0b025bc3989e2e87c0c092b1e11d3ceb7df9ca792bfd9e4b9b645535722` | 정확 중복 없음, 최대 스피어만 0.981541 |
| `beicicc/s6e8-fold-safe-realmlp:realmlp` | 0.968156387 | `e21c22c3b2416598bd2bdc198cbbbbb2e8cdedd14f3434daa751282b97784665` | 정확 중복 없음, 기존 `realmlp_seed01_fixed4`와 0.999097 |
| `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` | 0.962557588 | `ff58548f9868bdd4a5dd3fe330060b39ad21f18f232a9a776f7a7ecdf20e618f` | 정확 중복 없음, 기존 `raw12`와 0.998125 |
| `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:hgb` | 0.962048339 | `b7b0afba77e4c3352a3c03b555c5c68fdf5fd9d6c234e4b5b00a402a5f02564a` | 정확 중복 없음, 최대 스피어만 0.997343 |
| `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:xgb` | 0.962314187 | `3f683ae1e737a53a2c220103b6c31375685f705030998628ecf2090c7e3d8351` | 정확 중복 없음, 최대 스피어만 0.997243 |
| `ravi20076/playgrounds6e8-public-baseline-v1:XGB1C` | 0.964201482 | `d795573efce0daf7fa1f87e82bd0843f1e12960bc621e033e9d93c207be822ab` | 정확 중복 없음, 최대 스피어만 0.992776 |
| `ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` | 0.964173099 | `15ea60831189c09204e17cdefbaa8e262cee346fab45ce9f7f97e32870446b66` | 정확 중복 없음, 최대 스피어만 0.992242 |
| `ravi20076/playgrounds6e8-public-baseline-v1:CB1C` | 0.963944102 | `f3e04b96a6bb416cab11bf092570657e9bc6d74c7446ab8bf7f97815f17e80a0` | 정확 중복 없음, 최대 스피어만 0.993464 |

### Beicicc의 두 구성원

[Fold-Safe TabNet 실행 339872430](https://www.kaggle.com/code/beicicc/s6e8-fold-safe-tabnet?scriptVersionId=339872430)은 바깥쪽 5분할 시드 42와 각 바깥쪽 학습 부분 안의 내부 5분할 목표 부호화를 사용한다.
모델은 모든 바깥쪽 분할에서 고정 35회 학습하고 `eval_set=[]`로 실행되므로 바깥쪽 검증 목표값은 채점에만 쓰인다.
공개 실행의 manifest도 `model_selection=none_within_outer_fold`, `fixed_epochs=35`, `outer_valid_role=scoring_only`를 기록한다.
소스 SHA-256은 `3f97a7351a649a6a583edf3492fe4db190d926ef94778c74db3bba41c8abfeb7`이다.
OOF CSV SHA-256은 `e93038b7859d5da0f0410e62a177a6965c9c83933e840bf0b79ce3d5b12d7451`이고 시험 CSV SHA-256은 `84254c4ea4e0bf27a05b3fcabc4bec906c8be5fd71e6d8e751e458c5138b230d`이다.

[Fold-Safe RealMLP 실행 339864149](https://www.kaggle.com/code/beicicc/s6e8-fold-safe-realmlp?scriptVersionId=339864149)은 같은 바깥쪽 5분할과 바깥쪽 학습 부분 안의 교차 적합 목표 부호화를 사용한다.
모델은 고정 4회 학습하고 바깥쪽 검증 자료를 학습 함수, 중간 상태 선택이나 모형 저장에 전달하지 않는다.
공개 실행의 manifest도 `model_selection=none_within_outer_fold`, `fixed_epochs=4`, `outer_valid_role=scoring_only`를 기록한다.
소스 SHA-256은 `60a0bd05332e8932468d9cc796855013be3c3798344fd75c15c016764eba58ef`이다.
OOF CSV SHA-256은 `ba5440a6ebe836d57a0720f9c92847ffe55cb4cb049163c356ab94aeb7a03933`이고 시험 CSV SHA-256은 `2a8b8891a7a2fce02b96d72aad72f5a65fd9affaed7bcf369016e581dbc12076`이다.

### BusyPrime의 세 구성원

[BusyPrime 실행 339485089](https://www.kaggle.com/code/busyaprime/s6e8-tabular-baseline-that-autodetects-the-task?scriptVersionId=339485089)은 `SEED=42`, `N_FOLDS=5`, `FAST=False`를 고정한다.
LightGBM 600그루, HistGradientBoosting 400회, XGBoost 450그루를 고정해 학습하며 검증 자료를 이용한 조기 종료나 상태 선택이 없다.
범주 값 정수화는 목표값을 읽지 않고 훈련·시험 범주의 합집합만 사용한다.
소스 SHA-256은 `52c509d1b726d2ddeaddd0e07ada8c345a21483c09f0128e0070359552025235`이다.
OOF 파일 SHA-256은 HGB `fad2e34d2108497f5f4d91f0dc123a9d44e6e4e424fb29008f3045af8e0898fe`, LGB `025869d2ea30905a2599a69fcae6b4c7d67bf5eb74d0c96926cbb803d6021bd4`, XGB `8caca844702b44ea99fb29a447fa7a32c0f4ed3a4f436238b0a7c09b8d3638aa`이다.
시험 파일 SHA-256은 HGB `8c16745ba49754a18683e55a3571f0c35fa0b995d4343dec6fa631b4af0129ae`, LGB `4fe3bccea74736787e9d53aa8b357a97718c6d5b411041abce6c056d6ad2eec2`, XGB `46268e9e123bbdf907f20e655691e85564d435e36a3a3ab5d16de61b4ad9a65d`이다.

### Ravi의 세 구성원

[Ravi 기준선 실행 339444387](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1?scriptVersionId=339444387)은 `state=42`, `n_splits=5`, `mdlcv_mthd='SKF'`를 고정하고 원자료 추가 사용을 `nb_orig=0`으로 끈다.
XGBoost 3,000그루, LightGBM 2,500그루, CatBoost 3,000회를 고정한다.
목표 부호화는 각 바깥쪽 학습 부분에서 sklearn Pipeline 안의 `TargetEncoder`가 교차 적합으로 계산한다.
[공개 보조 코드 실행 339439580](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1?scriptVersionId=339439580)의 `ModelTrainer`는 sklearn Pipeline에 최상위 `eval_set`을 전달해 실패하면 예외 처리 뒤 `model.fit(Xtr, ytr)`을 다시 호출하므로 세 모델에는 바깥쪽 검증 기반 상태 선택이 없다.
기준선 소스 SHA-256은 `186d26a1aba7549fd182ed89322daff43f6083d8f9275175215c7c0207d31c30`이고 보조 `training.py` SHA-256은 `26504708be69444a8df97ac7b0ecc95e788340b88b5cfaef823dfe2c9d8a1405`이다.
OOF parquet SHA-256은 `474c9ae81249bfaf026dd8e716ea79a4e4514f8f1e1d0cbd4fece758867879d9`이고 시험 parquet SHA-256은 `3ded242bf1efefc5acbeeeb340a0fb62fe116633d36d1a5fb62b24c38befdd8c`이다.

## 대표 제외와 보류

[RepLeafGBM 실행 345401099](https://www.kaggle.com/code/masayakawamata/s6e8-repleafgbm-cv-0-968187?scriptVersionId=345401099)은 최종 실행 자체는 5분할 시드 42와 고정 704회를 사용하고 행과 출력도 모두 맞는다.
그러나 공개 소스가 접힌 0번 검증 결과로 잎 모형, 가림 증강과 다른 설정을 고른 비교 사다리를 명시하므로 바깥쪽 검증 목표값이 공개 설정 선택에 닿았고 엄격 후보에서 제외했다.

[MNK RealMLP 실행 345294879](https://www.kaggle.com/code/mnkaggler/predict-smartphone-addiction-ps6e8-v2?scriptVersionId=345294879)은 시드 42·789·1011·2026의 5분할 예측을 평균하지만 각 바깥쪽 검증 자료를 `model.fit`에 넘기고 `best_val_probs_`를 OOF로 저장한다.
같은 설정의 여러 시드 평균은 허용되지만 바깥쪽 검증 상태 선택은 허용되지 않으므로 제외했다.

[Paiky 실행 344896821](https://www.kaggle.com/code/paiky1995/s6e8-correlation-does-not-predict-contribution?scriptVersionId=344896821)은 정확한 5분할과 분할 안 전처리를 쓰고 5개 개별 OOF·시험 열을 저장한다.
그러나 XGBoost와 LightGBM은 바깥쪽 검증 조기 종료를 쓰고 신경망은 바깥쪽 검증 AUC가 가장 좋은 가중치를 복원하므로 5개 모두 제외했다.

[Zhenrui RealMLP 실행 339638191](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction?scriptVersionId=339638191)은 정확한 5분할과 분할 안 목표 부호화를 쓰지만 `best_val_probs_`를 OOF로 저장하므로 제외했다.
[Beicicc lattice 실행 339839496](https://www.kaggle.com/code/beicicc/s6e8-fold-safe-lattice-target-encoding?scriptVersionId=339839496)은 바깥쪽 검증 조기 종료를 사용하므로 제외했다.
[lattice residual 실행 339847478](https://www.kaggle.com/code/beicicc/s6e8-lattice-residual-blend-audit?scriptVersionId=339847478), [strict RealMLP residual 실행 339882789](https://www.kaggle.com/code/beicicc/s6e8-strict-realmlp-residual-audit?scriptVersionId=339882789), [seed diversity residual 실행 339863952](https://www.kaggle.com/code/beicicc/s6e8-seed-diversity-residual-audit?scriptVersionId=339863952)과 [lattice seed ensemble 실행 339854348](https://www.kaggle.com/code/beicicc/s6e8-lattice-seed-20260803-ensemble?scriptVersionId=339854348)의 출력은 외부 예측이나 여러 설정을 결합한 2단계 결과라 제외했다.

Abdullah Safwan의 [CatBoost 실행 339721032](https://www.kaggle.com/code/abdullahsafwan333/s6e8-catboost-sap?scriptVersionId=339721032), [LightGBM 실행 339724247](https://www.kaggle.com/code/abdullahsafwan333/s6e8-lightgbm-sap?scriptVersionId=339724247), [RealMLP 실행 339796643](https://www.kaggle.com/code/abdullahsafwan333/s6e8-realmlp-sap?scriptVersionId=339796643), [XGBoost 실행 339722491](https://www.kaggle.com/code/abdullahsafwan333/s6e8-xgboost-sap?scriptVersionId=339722491)은 7분할이라 제외했다.
[Dnyanesh 실행 339545109](https://www.kaggle.com/code/dnyaneshbharambe/addiction-lgb-xgb-cat-ensemble?scriptVersionId=339545109)의 LightGBM·XGBoost·CatBoost는 10분할이고 검증 조기 종료도 사용하므로 제외했다.
[Nawfeel RealMLP 실행 339698542](https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014?scriptVersionId=339698542)은 분할 시드 63이고 바깥쪽 분할 밖에서 만든 목표 부호화를 재사용하므로 제외했다.
[Yunsu 실행 339497263](https://www.kaggle.com/code/yunsuxiaozi/agent-in-pss6e8-cv-0-9623?scriptVersionId=339497263)은 3분할이라 제외했다.
Stephen Tarter의 [CatBoost 실행 343553572](https://www.kaggle.com/code/stephentarter/ps-s06e08-catboost?scriptVersionId=343553572), [HistGradientBoosting 실행 345471100](https://www.kaggle.com/code/stephentarter/ps-s06e08-histgradientboosting?scriptVersionId=345471100), [LightGBM 실행 345355040](https://www.kaggle.com/code/stephentarter/ps-s06e08-lightgbm?scriptVersionId=345355040), [XGBoost 실행 345379578](https://www.kaggle.com/code/stephentarter/ps-s06e08-xgboost?scriptVersionId=345379578)는 설정의 첫 시드 10301을 써 커뮤니티 분할과 다르므로 제외했다.
Omid Baghchehsaraei의 [FLAML 실행 339934291](https://www.kaggle.com/code/omidbaghchehsaraei/flaml-lgbm-for-predicting-smartphone-addiction?scriptVersionId=339934291), [RealMLP 실행 339560589](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction?scriptVersionId=339560589), [ResNet 실행 339598951](https://www.kaggle.com/code/omidbaghchehsaraei/resnet-for-predicting-smartphone-addiction?scriptVersionId=339598951), [TabM 실행 339582777](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction?scriptVersionId=339582777), [TabNet 실행 339671420](https://www.kaggle.com/code/omidbaghchehsaraei/tabnet-for-predicting-smartphone-addiction?scriptVersionId=339671420), [XGBoost 실행 339532612](https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-for-predicting-smartphone-addiction?scriptVersionId=339532612)는 전체 훈련 자료로 미리 만든 목표 부호화를 모든 바깥쪽 분할에서 재사용하므로 제외했고, 다섯 모델은 검증 조기 종료나 자동 모형 선택도 사용한다.
[Ravi 기준선 판본 2 실행 339444275](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2?scriptVersionId=339444275)의 RealMLP는 `use_early_stopping=True`라 제외했다.
[Naji 단일 LGBM 최신 실행 344072919](https://www.kaggle.com/code/najiama/single-lgbm-model-lb-0-96990-cv-0-96862?scriptVersionId=344072919)은 결합 제출에서 얻은 의사 목표값을 학습에 사용하므로 제외했다.
[Nikita 실행 343204793](https://www.kaggle.com/code/nikita7364777/rank-gauss-logit-rank-blending?scriptVersionId=343204793)은 기존 OOF를 입력으로 다시 학습한 메타 결합이라 제외했다.
나머지 갱신 노트북과 자료는 제출 파일만 있거나 OOF와 시험 예측 중 하나가 없어서 제외했다.

## 장부 대조와 후보 동결

정확 중복은 장부 판본 2의 400개 통과 구성원과 같은 방식의 float64 OOF·시험 쌍 해시로 확인했다.
근접 중복은 먼저 고정 난수 42로 뽑은 OOF 100,000행에서 400개 전부를 검사한 뒤 각 후보의 상위 3개를 691,369행 전체로 다시 계산했다.
Beicicc RealMLP와 BusyPrime LightGBM만 장부 기준 0.998을 넘었다.
판본 2는 0.998 초과를 `near_duplicate_cluster` 주의 사항으로 남기고 자격을 박탈하지 않으므로 이번 후보 8개도 모두 동결한다.
상위 지도 결정에 따라 전체 OOF의 단독 성능이나 근접 중복을 보고 후보를 미리 제거하지 않고, 후속 중첩 선별이 바깥쪽 채점 분할의 목표값을 보지 않는 학습 부분에서만 중복성을 계산하게 한다.

## 2026-08-30 증분 확인 명세

노트북 기준점은 `lastRunTime=2026-08-28T00:34:34.393Z`이고 자료 기준점은 `lastUpdated=2026-08-28T00:21:30.937Z`이다.
먼저 대회 노트북 전체 목록과 `--output-type data` 목록을 `dateRun` 정렬, 쪽당 100개로 읽고 기준점보다 새 항목이 더는 없는 쪽에서 멈춘다.
새 참조뿐 아니라 기존 참조의 새 실행 판본도 조사 대상에 넣는다.
새롭거나 갱신된 참조마다 `kaggle kernels files`로 저장 파일을 확인하고, OOF·시험 쌍이 있으면 정확한 실행 판본의 소스와 저장 출력을 내려받는다.
파일 목록 호출은 1.5초 이상 간격을 두고 429 응답이 나오면 45초 뒤 같은 참조부터 다시 실행한다.

자료 검색은 이번 조사의 17개 고정 검색어를 `updated` 정렬 100개씩 3쪽까지 그대로 다시 실행한다.
검색 결과에 없어도 새 노트북이 연결한 자료와 입력 노트북은 반드시 따라가서 파일, 사용 조건과 계보를 확인한다.
새 자료는 기준점보다 갱신 시각이 늦거나 같은 참조의 판본이 바뀐 경우에만 다시 내려받는다.

출처 우선순위 1은 같은 공개 실행 판본에 개별 OOF·시험 쌍, 소스와 manifest 또는 분할 벡터가 함께 있는 노트북이다.
출처 우선순위 2는 공개 소스가 연결되고 개별 OOF·시험 쌍과 계보가 있는 자료이다.
출처 우선순위 3은 결합 노트북이며, 결합 출력은 받지 않고 그 노트북이 가리키는 개별 기초 모델 출처만 추적한다.
작성자 설명, 같은 작성자의 다른 코드나 분할 벡터만 있는 출처는 발견 목록에는 남기되 후보 자격은 주지 않는다.

새 후보마다 실행 판본 식별자, 소스 SHA-256, 원본 출력 SHA-256, 정규화한 OOF·시험 쌍 SHA-256, 행 수, dtype, 유한값, 원래 id 순서, 재채점 AUC와 장부 판본 2 대조를 기록한다.
검증 자료의 목표값이 전처리, 학습, 조기 종료, 상태 선택과 공개 설정 선택에 닿는지 소스 전체에서 확인한다.
조건을 통과한 후보는 전체 OOF 성능이나 중복성을 이유로 미리 제거하지 않고 판본 3 동결 후보에 추가한다.

## 사용 조건과 한계

Kaggle은 공개 노트북 소스를 Apache License 2.0으로 배포한다고 [공식 Meta Kaggle Code 자료](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)에 명시한다.
노트북의 저장 예측 출력에는 별도 사용 조건 표시가 없으므로 8개 배열은 확장 스택 결합 입력으로만 사용하고 저장소 커밋, 재배포와 자체 산출물 첨부를 하지 않는다.
공개 이전의 설정 탐색 이력은 알 수 없으며, 이번 판정은 고정 공개 판본에서 확인할 수 있는 계보만 보증한다.
검색어와 대회 연결 정보에 모두 나타나지 않는 독립 자료는 검색으로 찾을 수 없지만, 새 노트북의 연결 입력을 별도로 추적해 이 빈틈을 줄인다.
