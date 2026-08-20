# CNN과 스칼라 token Transformer 점수 척도 정렬

이 문서는 [공개 점수와 OOF 척도를 정렬해 두 모델의 실제 격차를 확정](https://github.com/tmheo/predicting-smartphone-addiction/issues/269)의 조사 결과다.
조사 기준 시각은 2026-08-20 10:05 JST다.

## 결론

자료, fold, seed와 예측 평균 방식을 실제로 맞추면 공개 CNN과 `exp113_tab_cnn_m0` seed 42의 OOF AUC 격차는 `0.0079032656`이다.
같은 조건에서 공개 CNN이 높으며, 행 단위 paired DeLong 95% 구간은 `[0.0076908033, 0.0081157280]`이다.

공개 스칼라 token Transformer와 `exp115_scalar_token_transformer_m0`는 둘 다 seed 42의 fold 0만 맞춰 비교할 수 있다.
이 조건의 OOF AUC 격차는 `0.0118633932`이고, 행 단위 paired DeLong 95% 구간은 `[0.0113182109, 0.0124085755]`이다.

사용자가 본 `0.962`는 CNN의 3개 seed 예측 평균 OOF `0.9620748339`다.
스칼라 token Transformer는 5-fold 또는 3개 seed 전체 OOF를 만들지 않았고, 유일한 자체 결과는 fold 0 seed 42의 `0.9551717921`이다.
따라서 두 모델 모두 자체 OOF가 약 `0.962`라는 전제는 성립하지 않는다.

CNN의 Public `0.96947`과 3개 seed 평균 OOF `0.9620748339`를 바로 빼면 표면 격차는 `0.0073951661`이다.
공개 CNN 자체의 OOF `0.9677056335`를 사이에 두면 Public 이동 `+0.0017643665`가 표면 격차의 `23.9%`를 차지하고, OOF 대 OOF 격차 `0.0056307996`이 `76.1%` 남는다.
다만 이 OOF 비교는 공개판 1개 seed와 자체판 3개 seed 예측 평균을 비교하므로 공개판에 불리한 비교다.
엄격하게 seed 42 한 개씩 맞추면 남는 OOF 격차는 오히려 `0.0079032656`으로 커진다.

스칼라 token Transformer의 Public `0.96909`와 자체 fold 0 `0.9551717921`의 표면 격차 `0.0139182079`는 Public 이동 `0.0016210788`, 공개 OOF의 전체 5-fold와 fold 0 차이 `0.0004337360`, 같은 fold OOF 격차 `0.0118633932`로 나뉜다.
각 몫은 표면 격차의 `11.6%`, `3.1%`, `85.2%`다.

결론적으로 약 `+0.001`의 일률적인 Public 보정을 빼도 두 격차의 주된 부분은 사라지지 않는다.
실제로 관측된 Public 이동을 정확히 빼면 CNN은 같은 seed에서 `0.00790`, 스칼라 token Transformer는 같은 fold에서 `0.01186`이 남는다.
점수 척도 차이는 원인의 일부일 뿐 주된 원인이 아니다.

## 조사 시점의 공개 판본과 Public 점수

두 사용자 링크의 현재 페이지를 실제 브라우저로 열고 판본 단추와 점수 연결을 확인했다.
Kaggle CLI `2.2.4`로 같은 현재 slug의 소스, 메타데이터, 실행 기록, `oof.csv`와 `submission.csv`를 다시 내려받았다.
현재 페이지는 두 노트북 모두 `Version 1 of 1`을 표시했으며 이전 조사 뒤 새 공개 판본이 생기지 않았다.

| 모형 | 현재 페이지에서 확인한 점수 | 판본 | 실행 판본 식별자 | 노트북 식별자 | 최신 실행 시각 |
| --- | ---: | ---: | ---: | ---: | --- |
| [CNN for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction) | Public `0.96947` | 1 | [`342747549`](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction?scriptVersionId=342747549) | `130924695` | 2026-08-16 10:37:07.743 UTC |
| [TabTransformer: Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction) | Public `0.96909` | 1 | [`342815072`](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction?scriptVersionId=342815072) | `130940978` | 2026-08-16 17:05:34.067 UTC |

각 현재 페이지의 점수 연결은 해당 `scriptVersionId`를 가리켰다.
Public 점수는 페이지가 다섯 자리까지만 표시하므로 표시 자체의 반올림 범위는 각 값에서 `±0.000005`다.

다시 내려받은 내용 해시는 다음과 같다.

| 증거 | CNN | 스칼라 token Transformer |
| --- | --- | --- |
| 노트북 SHA-256 | `2310c4fa1b98230989f8e3bcf3f9661985a2c30df90597786e739cd34321f4dc` | `eeb3e1cccbaab29c71ef946876f7042509f6ef537df4a9b04ced36e3c424e46c` |
| Kaggle 메타데이터 SHA-256 | `64b436cc582602417df8753b716b5c9e380af34b7129018b0de76d3617c4fd33` | `3b9ab90b326ac4cf9cc6ce45f6c76ef04980fb2c44eb7d9f1e1e604bcb32470c` |
| 실행 기록 SHA-256 | `7712720b42e9f360da482877bb32025dcbd0005965fbac9c3bde4ca1b18ae3af` | `21e24aa8ad869aaeb87d2d72f40c3e59be3b37fd3f666fd3dab58b1a222bd657` |
| OOF SHA-256 | `3da3917f23b3c636cf3af11792ab483f89a09d6b38416897b0626d93b10c3c91` | `1594f8e7f72ee8c6bf5dacbdddc56fb29d8998c24c43ad9424b39abd65e80cb4` |
| 제출 예측 SHA-256 | `66b40c14c8c133e9228889e713183e007a8294cb300a807dab1c7fd0c2fab9d6` | `6228dfc18fe458c6f061f684daff8daef7e2a4aed39acf245be5ec0a190877a9` |

이 해시는 [이전 CNN 조사](s6e8-cnn-notebook-review.md)와 [이전 스칼라 token Transformer 조사](omid-tabtransformer-notebook.md)의 고정값과 모두 일치한다.
[CNN 공개 출력](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction/output?scriptVersionId=342747549)과 [스칼라 token Transformer 공개 출력](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/output?scriptVersionId=342815072)이 원시 OOF와 제출 예측의 1차 출처다.

## 비교 단위 정렬

두 공개 노트북은 `SEED = 42`와 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 사용한다.
OOF에서는 각 행에 그 행을 검증한 모형 하나의 예측만 저장한다.
시험 예측은 서로 다른 학습 fold에서 얻은 다섯 모형의 예측을 산술 평균한다.
따라서 Public과 OOF의 이동에는 다른 평가 행 표본뿐 아니라 시험 예측의 5개 fold 모형 평균 효과도 섞인다.
Public 이동을 단순한 리더보드 잡음이나 고정 절편으로 해석할 수 없는 이유다.

공개 OOF의 691,369개 ID와 라벨은 로컬 `data/train.csv`와 모두 일치했다.
학습 자료 SHA-256은 `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c`이고 시험 자료 SHA-256은 `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e`다.
공개 분할과 로컬 [`artifacts/folds.parquet`](../../artifacts/folds.parquet)의 fold 번호도 691,369개 행에서 모두 일치했으며 fold 파일 SHA-256은 `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4`다.

엄격한 비교 단위는 다음과 같다.

| 축 | 공개 CNN 대 exp113 | 공개 스칼라 token Transformer 대 exp115 |
| --- | --- | --- |
| 평가 자료 | 같은 학습 OOF 691,369행 | 같은 학습 OOF의 fold 0 138,274행 |
| fold | 같은 5개 fold 전체 | 같은 fold 0 |
| seed | 42 대 42 | 42 대 42 |
| 행별 예측 평균 | 각 행에 모형 하나 | 각 행에 모형 하나 |
| AUC | 전체 OOF ROC AUC | fold 0 ROC AUC |

[`exp113_tab_cnn_m0`](../../configs/exp113_tab_cnn_m0.yaml)의 확정 실행은 seed 42, 43, 44를 각각 5-fold로 실행했다.
최종 `oof.parquet`의 예측은 세 seed별 OOF 예측의 행별 산술 평균과 모든 행에서 정확히 일치한다.
이 평균은 단일 seed인 공개 CNN과 평균 방식이 다르므로 별도 참고 비교로만 둔다.

[`exp115_scalar_token_transformer_m0`](../../configs/exp115_scalar_token_transformer_m0.yaml)은 진입 진단에서 fold 0 seed 42만 실행한 뒤 중단됐다.
전체 5-fold OOF, 3개 seed 평균 OOF와 전체 시험 예측은 존재하지 않는다.

## CNN의 격차

공개 OOF와 exp113의 결과는 다음과 같다.

| 비교값 | OOF AUC | 공개 CNN과의 차이 |
| --- | ---: | ---: |
| 공개 CNN, seed 42 | `0.9677056335` | 기준 |
| exp113, seed 42 | `0.9598023679` | `-0.0079032656` |
| exp113, seed 43 | `0.9599774621` | 평균 방식 불일치 |
| exp113, seed 44 | `0.9600750741` | 평균 방식 불일치 |
| exp113, 3개 seed 예측 평균 | `0.9620748339` | `-0.0056307996` |

엄격히 맞춘 seed 42 격차는 `0.0079032656`이다.
공개 CNN이 다섯 fold 모두에서 exp113 seed 42보다 높았고 fold별 격차 범위는 `0.0070459112`부터 `0.0098707337`까지다.

exp113은 세 seed의 개별 AUC를 단순 평균한 것이 아니라 예측을 먼저 평균해 AUC를 다시 계산한다.
그 결과 3개 seed 예측 평균 AUC는 seed 42보다 `0.0022724661` 높다.
공개판에는 같은 3개 seed 예측 평균이 없으므로 `0.0056307996`은 자체판의 평균화 이득만 반영한 비대칭 하한 성격의 격차다.

Public과 자체 결과를 직접 비교한 표면 격차의 분해는 다음과 같다.

| 비교 | 값 | 표면 격차에서의 몫 |
| --- | ---: | ---: |
| Public `0.96947` 대 exp113 seed 42 | `0.0096676321` | `100%` |
| 공개판 Public 대 공개판 OOF 이동 | `0.0017643665` | `18.3%` |
| 같은 seed OOF 대 OOF 격차 | `0.0079032656` | `81.7%` |

사용자가 본 3개 seed 평균을 기준으로 하면 표면 격차 `0.0073951661` 가운데 Public 이동은 `23.9%`이고 OOF 격차는 `76.1%`다.
고정 `+0.001`만 빼면 `0.0063951661`이 남으므로 실제 공개 OOF 대 자체 3개 seed 평균 격차 `0.0056307996`을 `0.0007643665`만큼 크게 잡는다.

## 스칼라 token Transformer의 격차

공개판의 전체 OOF AUC는 `0.9674689212`지만 exp115에는 전체 OOF가 없다.
비교 가능한 fold 0에서 공개판은 `0.9670351852`, exp115는 `0.9551717921`이다.
엄격히 맞춘 격차는 `0.0118633932`다.

Public과 자체 fold 0를 바로 비교한 표면 격차는 다음처럼 나뉜다.

| 구성 요소 | 값 | 표면 격차에서의 몫 |
| --- | ---: | ---: |
| Public `0.96909` 대 exp115 fold 0 | `0.0139182079` | `100%` |
| 공개판 Public 대 전체 OOF 이동 | `0.0016210788` | `11.6%` |
| 공개판 전체 OOF 대 fold 0 차이 | `0.0004337360` | `3.1%` |
| 같은 fold OOF 대 OOF 격차 | `0.0118633932` | `85.2%` |

고정 `+0.001`만 빼면 `0.0129182079`가 남는다.
이는 Public 이동의 초과분 `0.0006210788`과 전체 OOF 대 fold 0 차이 `0.0004337360`을 제거하지 못하므로 같은 fold 격차를 `0.0010548148`만큼 크게 잡는다.
그 보정 오류를 모두 없애도 실제 같은 fold 격차는 `0.01186`이다.

## Public 이동과 점수 잡음

두 공개판에서 직접 관측한 Public 대 OOF 이동은 CNN `+0.0017643665`, 스칼라 token Transformer `+0.0016210788`이다.
둘 다 `+0.001`보다 크므로 이번 두 모델에 `+0.001`을 상수로 적용해서는 안 된다.

로컬 `mlflow.db`의 2026-08-19 00:00 KST 이전 3개 seed 단일 모형 제출 10건을 읽기 전용으로 다시 조회했다.
그 표본의 Public 대 OOF 이동은 중앙값 `+0.0012438749`, 평균 `+0.0012581415`, 범위 `[+0.0011342382, +0.0014323576]`였다.
이는 [`notebooks/oof-public-retrospective.ipynb`](../../notebooks/oof-public-retrospective.ipynb)의 동결 모집단과 같은 표본이다.

공개 CNN의 이동은 이 중앙값보다 `0.0005204916`, 스칼라 token Transformer는 `0.0003772038` 크다.
둘 다 기존 10건 범위 위에 있지만 아래에서 구한 Public 단일 점수 표본 잡음의 약 `0.91배`, `0.66배`에 해당한다.
따라서 두 노트북만 보고 모형별로 다른 고정 Public 절편이 존재한다고 결론 내릴 수 없다.

[`scripts/estimate_carryover.py`](../../scripts/estimate_carryover.py)를 현재 champion OOF 원시 예측으로 다시 실행했다.
Public 채점 행 수 59,260개로 500회 복원 재표집한 단일 점수의 표준편차는 `0.000573`이었고 Hanley-McNeil 근사는 `0.000634`였다.
59,260개는 시험 296,302행과 [Meta Kaggle의 S6E8 `LeaderboardPercentage=20`](https://www.kaggle.com/datasets/kaggle/meta-kaggle)을 사용한 값이다.

이 `0.000573`은 Public 점수 하나의 표본 잡음 규모이며 OOF 격차에서 기계적으로 빼는 값이 아니다.
한 표준편차를 엄격히 맞춘 OOF 격차와 비교하면 CNN 격차의 `7.3%`, 스칼라 token Transformer 격차의 `4.8%`다.
`1.96` 표준편차는 각각 격차의 약 `14.2%`, `9.5%`다.

행 단위 paired DeLong으로 직접 구한 OOF 격차의 표준오차와 95% 구간은 다음과 같다.

| 비교 | 격차 표준오차 | 95% 구간 |
| --- | ---: | --- |
| 공개 CNN 대 exp113 seed 42, 전체 OOF | `0.0001083991` | `[0.0076908033, 0.0081157280]` |
| 공개 CNN 대 exp113 3개 seed 평균, 전체 OOF | `0.0000955852` | `[0.0054434526, 0.0058181466]` |
| 공개 스칼라 token Transformer 대 exp115, fold 0 | `0.0002781542` | `[0.0113182109, 0.0124085755]` |

이 구간은 OOF 행을 서로 독립인 관측으로 보는 기술적 근사다.
합성 자료 생성 과정의 행 간 의존성, 공개 노트북 개발 중 반복 선택과 조기 종료 선택을 포함하지 않으므로 완전한 재현 불확실성보다 좁을 수 있다.
그 한계를 감안해도 관측 격차의 크기는 마지막 자릿수 잡음으로 설명할 수 있는 범위를 크게 넘는다.

Public 대 OOF 이동은 다음 요소가 합쳐진 관찰값이다.

1. OOF는 각 행에 단일 fold 모형 예측을 쓰지만 제출은 다섯 fold 모형 예측을 평균한다.
2. 학습 OOF 행과 시험 Public 행은 서로 다른 표본이다.
3. Public은 시험 자료의 20%만 채점하므로 표본 잡음이 있다.
4. 공개된 판본과 제출은 저자가 선택한 결과이므로 공개되지 않은 시행에 대한 선택 편향을 알 수 없다.

이 네 요소를 현재 산출물만으로 더 분해할 수 없다.
따라서 실제 관측 이동은 표면 격차를 정리하는 데 사용하되 구조 차이의 성능으로 귀속하지 않는다.

## 1차 근거와 로컬 산출물

공개판 근거는 조사 시점의 현재 페이지, 고정 `scriptVersionId` 페이지와 공개 출력에서 직접 확보했다.
공개 노트북 소스는 [Kaggle 공개 노트북 사용 조건](../agents/kaggle-public-notebook-licensing.md)에 따라 Apache License 2.0으로 다뤘다.
이번 조사에서는 소스나 외부 예측을 저장소에 복사하거나 후보 풀에 넣지 않았다.
대회 자료와 출력 파일의 사용 조건은 노트북 소스의 사용 조건과 별개다.

CNN 자체 실행의 1차 로컬 근거는 다음과 같다.

- 실행 식별자는 원격 원본 `a3c4f73cce764f8f85c3620631f16a71`, 로컬 반입본 `b47a4184249746cdb1d062aa98364437`이다.
- 실행 커밋은 `b684ce9b0d618cab8ce392b8cc5d57b3ef22bb0b`다.
- 로컬 경로 `mlruns/1/b47a4184249746cdb1d062aa98364437/artifacts/oof.parquet`의 SHA-256은 `c2ba050cd4a0b8c0590766cecb973ccb82d65a1ed7691bf451c3310608fa39ad`다.
- 같은 실행의 `oof_seed_42.parquet` SHA-256은 `605e7be0b9ed6eb35c11439cbc81c41ca71da12fb36c6d395d746e7c70f5123e`다.
- 실행 기록 묶음 SHA-256은 `172d535c4852551fce4a66443e41151fe4ff462a009b603e34df9f7092eb8ed9`다.
- 결과와 실행 무결성의 기존 기록은 [표 합성곱망 3시드 5-fold 확정 재검증](tab-cnn-confirmation.md)이다.

스칼라 token Transformer 자체 실행의 1차 로컬 근거는 다음과 같다.

- 실행 범위는 `m0-exp115-fold0-seed42`이고 실행 커밋은 `c54591dbfe552ec77324943556ae69d582d91e16`이다.
- 로컬 경로 `run-logs/vast-issue178/results/extracted/results/m0-exp115-fold0-seed42/validation_predictions.parquet`의 SHA-256은 `cb943a4db59aea3db9afd11e8c8568ae8d703dbee14c3f589bac68526facdc85`다.
- 같은 경로의 `entry_diagnostic.json` SHA-256은 `c0a5862a52cf216f57db5743408c6ff25cd690907be8ebf474c80433c7b2a5fa`다.
- 전체 결과 묶음 `run-logs/vast-issue178/results/issue178-scalar-token-entry-vast-a4000-20260818-result.tar.gz`의 SHA-256은 `562cadf6137813a9e985a574fe65db7935a6b43a89161134d29d0829d06cffe6`다.
- 결과와 실행 무결성의 기존 기록은 [스칼라 token Transformer 진입 진단](scalar-token-transformer-entry-diagnostic.md)이다.

## 판정

점수 척도를 맞춘 뒤에도 두 공개판의 우위는 크고 방향이 안정적이다.
CNN은 자체 3개 seed 평균과 비교하면 `0.00563`, 엄격한 같은 seed 비교에서는 `0.00790` 높다.
스칼라 token Transformer는 엄격한 같은 fold와 같은 seed 비교에서 `0.01186` 높다.

Public 이동은 사용자가 본 표면 격차의 CNN `18.3%` 또는 비대칭 3개 seed 비교의 `23.9%`, 스칼라 token Transformer `11.6%`만 설명한다.
스칼라 token Transformer에서 추가로 `3.1%`는 전체 5-fold OOF와 fold 0의 차이다.
나머지 `81.7%` 또는 `85.2%`는 같은 OOF 평가 단위에서도 남는다.

따라서 후속 원인 분석은 `+0.001` 보정이나 리더보드 잡음보다 공개판과 자체 이식판의 피처, 전처리, 모형, 학습 및 추론 차이를 우선 조사해야 한다.
이 문서는 그 원인을 미리 귀속하지 않고 비교 척도와 남는 격차만 확정한다.
