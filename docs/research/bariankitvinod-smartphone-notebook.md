# bariankitvinod 스마트폰 중독 노트북 신규 실험 단서 조사

## 결론

2026-08-16 JST 기준 [Predicting Smartphone Addiction (our ipynb file)](https://www.kaggle.com/code/bariankitvinod/predicting-smartphone-addiction-our-ipynb-file?scriptVersionId=342490082)의 최신 공개 판본 10을 코드 셀, 저장 출력, 입력, 실행 메타데이터와 출력 파일 목록까지 대조했다.
새 실험 항목을 열 만한 단서는 없다.
정확값 타깃 인코딩, 쌍 타깃 인코딩, 산술 잔차, 일반 비율, 결측 표시, LightGBM·XGBoost·CatBoost·HistGradientBoosting, 선형 다양성 모델, 작은 MLP, 결측 구간별 2단 결합과 의사 라벨링은 모두 기존 채택·기각 또는 열린 항목과 중복된다.
노트북의 저장 실행본이 보고한 최고 OOF AUC `0.96799`는 현재 champion `0.9690978395`보다 `0.00111` 낮다.
더 중요한 문제는 노트북이 타깃 인코딩 열을 전체 5-fold에서 한 번 미리 만든 뒤 같은 5-fold의 모델 입력으로 다시 사용한다는 점이다.
이 구조에서는 outer 검증 fold의 라벨이 다른 훈련 행의 타깃 인코딩 값에 들어가므로, 본문이 주장하는 누출 없는 OOF가 아니다.
Optuna 선택, permutation importance 특성 선택과 네 결합 전략 선택도 최종 점수를 보고하는 같은 OOF 라벨을 사용하므로 선택 편향이 추가된다.
현재 v4 소스와 저장 실행 결과도 일치하지 않는다.
v4에서 추가된 두 쌍 타깃 인코딩, MLP, 동적 앙상블과 의사 라벨링 셀은 실행 결과가 없고, 이전 v3 메모리 상태를 이어받은 제출 셀만 `stacked_regime_logit`의 `0.96799`를 출력했다.
따라서 `0.96799`는 최신 공개 소스를 처음부터 실행한 재현 결과로 사용할 수 없다.
결측 구간별 결합이라는 방향은 기존 [비선형·구간별 2단 결합의 추가 가치 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)에 이미 포함되어 있으므로 그 항목에서 완전한 nested OOF로 판단하면 된다.
의사 라벨링도 기존 [의사 라벨링의 마지막 단계 진입 여부 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/68)이 fold별 독립 의사 라벨을 요구하므로, 이 노트북의 전역 테스트 예측 재사용 구현은 채택하지 않는다.

## 조사 판본과 내용 해시

Kaggle 인증 API의 `GetKernel` 응답은 노트북 번호 `129892776`과 최신 공개 판본 번호 `10`을 반환했다.
공개 페이지의 oEmbed 주소는 이 판본을 `scriptVersionId=342490082`로 고정한다.
Kaggle CLI `2.2.4`로 2026-08-16에 내려받은 `.ipynb`의 SHA-256은 `4c32a2a406ebee7ef9650b661af986a35afc4b4d6b1034a47f99ddb31e0f7414`다.
같이 내려받은 `kernel-metadata.json`의 SHA-256은 `2e27a0a3a6253e2a3f3236b6467bb18c85fa23365154ebe3eba4a0c7cc1599bb`다.
셀 종류와 셀 원문만 순서대로 NUL 구분해 계산한 SHA-256은 `ed1707cec959792afe2743bd53ef21aee3d3bac21fc0479496e4920711b4a523`다.
코드 셀 원문만 순서대로 NUL 구분해 계산한 SHA-256은 `621a54bb914d305063d97d2b5622b24a4cab97f9de40f130e9807d82a3cdab86`다.
코드 셀의 `outputs` JSON만 순서대로 NUL 구분해 계산한 SHA-256은 `ae047fecd201acbdb4c43266ab80cfb378aa2d69f7986af2cec57eebdb749802`다.
원본 노트북은 코드 셀 35개와 Markdown 셀 19개로 구성된다.
노트북 메타데이터의 Python 판본은 `3.12.13`이고 Papermill 판본은 `2.7.0`이다.
Kaggle 이미지 식별자는 `gcr.io/kaggle-images/python@sha256@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`로 기록되어 있다.
CPU 실행이며 GPU와 TPU는 꺼져 있고 인터넷은 켜져 있다.
Papermill 실행 시작 시각은 `2026-08-15T05:51:05.314644+00:00`, 종료 시각은 `2026-08-15T16:10:57.388025+00:00`, 기록된 실행 시간은 `37,192.073381`초다.
조사 시점의 `kaggle kernels list`는 `2026-08-16T11:04:07.173000`에 시작된 후속 실행을 가리켰고 `kaggle kernels status`는 `RUNNING`을 반환했다.
따라서 이 문서는 완료되지 않은 후속 실행을 추정하지 않고 공개 판본 10에 포함된 실행 메타데이터와 저장 출력만 근거로 삼는다.

## 입력, 출력 산출물과 실행 기록

노트북 메타데이터에는 대회 입력 `playground-series-s6e8` 하나만 연결되어 있다.
별도 Kaggle 자료, 다른 노트북, 사전 학습 모델과 Kaggle Model 입력은 연결되어 있지 않다.
실행 첫 셀도 `/kaggle/input/competitions/playground-series-s6e8/` 아래의 `train.csv`, `test.csv`, `sample_submission.csv` 세 파일만 출력했다.
훈련 자료는 691,369행과 14열, 시험 자료는 296,302행과 13열이다.
저장소가 보유한 같은 대회 입력의 SHA-256은 `train.csv`가 `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c`, `test.csv`가 `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e`, `sample_submission.csv`가 `206763fe5786fb9c80d4e9289a3b812030d3dbb36450c6eb63348098154ce63e`다.
Kaggle `kernels files`는 공개 실행 출력으로 `submission.csv`와 CatBoost의 `catboost_training.json`, `learn_error.tsv`, `test_error.tsv`, `time_left.tsv`, TensorBoard event 파일 두 개를 열거했다.
OOF 예측, 모델별 시험 예측, fold 배정, Optuna study, 선택 전후 특성 목록의 기계 판독 산출물, 의존성 목록과 전체 실행 로그 파일은 공개 출력 목록에 없다.
노트북 본문이 언급한 이전 실행의 `predicting-smartphone-addiction.log`도 연결 입력이나 공개 출력에 없다.
조사 시점에는 더 최신 세션이 실행 중이어서 `kaggle kernels output`이 완료된 공개 판본 10의 파일을 내려주지 않았고, 따라서 열거된 출력 파일의 바이트와 SHA-256은 확인할 수 없었다.
이 제한 때문에 실행 증거는 `.ipynb` 안에 저장된 출력과 Kaggle API의 출력 파일 목록으로 한정한다.

## 최신 소스와 저장 실행 결과의 불일치

최신 Markdown은 스스로를 v4라고 부르며 MLP, 쌍 타깃 인코딩 두 개 추가, 의사 라벨링과 동적 모델 목록을 새 기능으로 설명한다.
그러나 현재 쌍 타깃 인코딩 셀의 `execution_count`는 비어 있고 출력도 없다.
이 셀은 기존 여섯 쌍에 `sleep_hours × work_study_hours`와 `age × daily_screen_time_hours`를 추가해 모두 여덟 쌍을 선언한다.
뒤의 실행 출력은 전체 특성이 81개라고 기록하는데, 여덟 쌍이 실제로 실행됐다면 83개가 되어야 한다.
81개는 이전 여섯 쌍만 메모리에 있던 v3 실행 상태와 정확히 맞는다.
MLP 셀도 `execution_count`와 출력이 없고, 저장 결과에서 MLP AUC나 fold 진행 기록을 찾을 수 없다.
모델 목록과 상관표 셀, 네 앙상블 계산 셀도 `execution_count`와 출력이 없다.
의사 라벨링 셀과 실험 장부 셀도 실행 결과가 없다.
반면 제출 셀은 실행 번호 32와 v3의 `stacked_regime_logit` OOF AUC `0.96799`를 출력한다.
노트북 JSON에는 일부 미실행 셀에도 과거 Papermill 완료 시각이 남아 있어, 실행 메타데이터와 현재 원문이 함께 갱신되지 않았음을 보여 준다.
이는 Kaggle에서 이전 실행 결과를 유지한 채 소스 셀을 수정하고 다시 완주하지 않은 판본으로 해석하는 것이 가장 자연스럽다.
따라서 최신 공개 소스의 v4 기능과 저장된 v3 수치를 결합해 하나의 실행 결과로 읽어서는 안 된다.

## 전체 파이프라인 재구성

### 자료 검사와 분할

노트북은 목표값 `addicted_label`을 분리하고 `id`를 모델 입력에서 뺀다.
`id`의 순위 단독 AUC는 `0.5007`로 저장됐다.
훈련과 시험의 열별 결측률을 비교하고, 원시 값과 범주 값을 사용한 3-fold LightGBM 적대적 검증을 수행한다.
적대적 검증 AUC는 `0.5639`다.
모델 검증은 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 사용한다.
Optuna 대리 평가는 이 다섯 fold 가운데 첫 세 fold를 사용하고 CatBoost는 첫 두 fold만 사용한다.

### 원시 특성과 행별 파생 특성

원시 수치 열 9개, 범주 열 3개와 모든 결측 열의 결측 표시를 만든다.
행별 비율은 분모 0을 결측으로 바꾸고 `[-10, 10]` 또는 일부 `[-5, 5]`로 자른다.
파생 특성에는 행동 시간 비율, 알림과 앱 실행 비율, 수면 부족, 평일·주말 차이, 합계, 곱, 제곱과 로그가 포함된다.
생성 제약 후보로 `passive_hours = daily - social - gaming - work`, `other_frac`, `constraint_violation`, `weekend_slack`과 두 slack의 차이를 만든다.
나이는 고정 구간으로 나눈다.

### 타깃 인코딩과 값 눈금 표현

범주 열 세 개에 평활 상수 20의 타깃 인코딩을 만든다.
`daily_screen_time_hours`, `weekend_screen_time`, `notifications_per_day`, `app_opens_per_day`, `sleep_hours`에는 소수 둘째 자리로 반올림한 값 키와 평활 상수 30을 사용한다.
현재 소스는 수치 쌍 여덟 개를 소수 첫째 자리 문자열 쌍으로 묶고 평활 상수 40을 사용하지만, 저장 실행은 이전 여섯 쌍만 사용했다.
`notifications_per_day`와 `app_opens_per_day`에는 반올림하지 않은 정확값 타깃 인코딩도 추가한다.
두 열의 정수 반올림값에서 마지막 자리, 첫 자리와 자리 합을 만들고 원시 값의 `sin(x)`, `cos(x)`, `sin(2x)`, `cos(2x)`를 만든다.
모든 타깃 인코딩은 같은 외부 5-fold를 순회해 각 행이 속한 fold를 제외한 값표로 그 행의 값을 만든다.
그러나 인코더 전체가 모델 outer CV 밖에서 한 번만 실행되므로 모델 훈련 행의 인코딩에는 현재 outer 검증 fold의 라벨이 들어갈 수 있다.

### 특성 선택

전체 자료 행을 다시 무작위 80:20으로 나누고 LightGBM 한 개로 permutation importance를 계산한다.
평균 중요도가 0 이하인 18개 열을 제거해 81개 가운데 63개를 남겼다.
제거 목록에는 결측 표시 다수, `academic_work_impact`, `academic_work_impact_te`, 나이 구간, 로그, 제곱과 `constraint_violation`이 포함된다.
정확값 타깃 인코딩, 자리 특성, 삼각함수 특성은 제거 목록에 없지만, 개별 중요도와 제거 전후 OOF 차이는 저장하지 않았다.

### 모델 탐색과 최종 적합

LightGBM, XGBoost와 CatBoost는 Optuna TPE sampler와 seed 42를 사용한다.
LightGBM과 XGBoost의 제한 시간은 각각 150분이고 CatBoost는 남은 시간 때문에 63분이 됐다.
시행 횟수는 LightGBM 20회, XGBoost 13회, CatBoost 3회다.
각 학습기는 선택된 매개변수 하나를 같은 다섯 fold에 다시 적합해 OOF와 fold 평균 시험 예측을 만든다.
HistGradientBoosting과 elastic-net 로지스틱 회귀는 탐색 없이 같은 fold에서 실행한다.
현재 소스의 MLP는 남은 시간이 1.5시간보다 많을 때만 실행하도록 되어 있지만 저장 실행 결과는 없다.
두 번째 LightGBM·XGBoost seed도 남은 시간이 두 시간보다 많을 때만 실행하며, 저장 실행에서는 남은 시간이 음수여서 건너뛰었다.

### 결합과 제출

현재 소스는 LightGBM, XGBoost, CatBoost, HistGradientBoosting과 elastic-net 로지스틱 회귀를 기본 구성원으로 사용한다.
MLP가 실행되면 여섯 번째 구성원으로 추가한다.
후보 결합은 확률 단순 평균, 구성원 전체 OOF AUC로 정한 가중 평균, fold 교차 예측 로지스틱 회귀와 결측 구간 상호작용 로짓 로지스틱 회귀 네 가지다.
결측 구간 결합은 각 구성원의 잘린 로짓, 결측 0개 여부, 결측 6개 이상 여부, 구성원 불일치와 집계 통계를 입력으로 쓴다.
네 결합의 같은 OOF AUC 가운데 최댓값을 제출 예측으로 고른다.
저장 제출은 이전 실행의 결측 구간 결합을 사용했고 296,302행을 기록했다.

## 저장 실행에서 확인된 수치

| 단계 | 저장 수치 | 증거 상태 |
| --- | ---: | --- |
| 원시 LightGBM baseline | `0.96298` | 실행 출력 있음 |
| permutation importance용 holdout | `0.96633` | 실행 출력 있음, 선택용 같은 자료 |
| LightGBM Optuna 대리 CV | `0.96766` | 20회 중 최고, 첫 3-fold 선택값 |
| LightGBM 최종 5-fold OOF | `0.96780` | 실행 출력 있음, 선택과 TE 누출 영향 포함 |
| XGBoost Optuna 대리 CV | `0.96755` | 13회 중 최고, 첫 3-fold 선택값 |
| XGBoost 최종 5-fold OOF | `0.96778` | 실행 출력 있음, 선택과 TE 누출 영향 포함 |
| CatBoost Optuna 대리 CV | `0.96703` | 3회 중 최고, 첫 2-fold 선택값 |
| CatBoost 최종 5-fold OOF | `0.96742` | 실행 출력 있음, 선택과 TE 누출 영향 포함 |
| HistGradientBoosting OOF | `0.96696` | 실행 출력 있음, TE 누출 영향 포함 |
| elastic-net 로지스틱 회귀 OOF | `0.95606` | 실행 출력 있음, TE 누출 영향 포함 |
| 두 번째 seed 평균 | 실행 안 함 | 남은 시간 `-0.31h`로 건너뜀 |
| MLP OOF | 없음 | 최신 소스 셀 미실행 |
| 결합 상관표 | 없음 | 최신 소스 셀 출력 없음 |
| 네 결합의 개별 OOF | 없음 | 최신 소스 셀 출력 없음 |
| 제출에 남은 결합 OOF | `0.96799` | 이전 메모리 상태의 제출 셀 출력만 있음 |

본문은 v2가 6.43시간, v3가 10.33시간 걸렸고 v3의 두 번째 seed 개선이 LightGBM `+0.0001`, XGBoost `+0.00006`이었다고 설명한다.
이 이전 판본의 실행 로그와 예측은 현재 입력이나 출력에 없으므로 저자 보고치로만 분류한다.
본문은 exact-value·자리·삼각함수 특성이 공개 제거 실험에서 유효했고 70개에서 95개 모형의 공동 OOF 묶음이 상위 점수를 만든다고 설명한다.
인용한 원 노트북이나 산출물은 연결 입력에 없고 이 노트북 자체도 특성별 제거 실험을 저장하지 않았으므로 저자 설명 이상의 독립 실행 증거가 아니다.
v4의 MLP와 추가 두 쌍은 코드 후보일 뿐 저장 실행 증거가 전혀 없다.
의사 라벨링은 기본값이 꺼져 있고 저장 실행 증거가 없다.

## 검증과 누출 검토

### 타깃 인코딩의 outer fold 역류

한 검증 fold를 `k`라고 하자.
노트북은 fold `l`에 속한 훈련 행의 타깃 인코딩을 `l`을 제외한 네 fold의 라벨로 미리 만든다.
`l`이 `k`와 다르면 이 값표에는 outer 검증 fold `k`의 라벨이 포함된다.
그 뒤 모델은 fold `k`를 검증할 때 이런 행들을 훈련 입력으로 사용한다.
따라서 검증 라벨이 훈련 입력 특성에 간접적으로 들어가며, 행마다 자기 라벨만 제외했다는 사실로는 outer CV 누출을 막을 수 없다.
범주 TE, 값 눈금 TE, 쌍 TE와 정확값 TE가 모두 같은 문제를 가진다.
또한 평활 prior인 `global_mean`도 전체 `y_full.mean()`으로 계산해 각 검증 fold의 라벨 비율을 사용한다.
현재 저장소의 fold-fit 컬럼 제공자는 각 outer 학습 fold 안에서 다시 내부 OOF를 만들고 outer 검증과 시험에는 outer 학습 fold로만 맞춘 표를 적용하므로 이 역류를 막는다.

### 탐색과 특성 선택의 선택 편향

Optuna는 최종 OOF에 다시 포함되는 첫 세 fold 또는 첫 두 fold의 라벨로 매개변수를 고른다.
선택된 한 점을 같은 다섯 fold에서 다시 평가하는 것은 nested CV가 아니므로, 본문의 `honest OOF` 표현은 맞지 않는다.
Permutation importance도 전체 OOF 타깃 인코딩을 가진 자료의 20% 라벨로 18개 특성을 제거한 뒤 그 20%를 포함한 전체 OOF를 다시 평가한다.
이 선택 단계에는 플라시보 열, 반복 자료 분할 또는 별도 outer 평가가 없고 중요도 0이라는 불안정한 단일 문턱만 사용한다.
결합 단계도 네 후보를 같은 OOF에서 모두 평가하고 최댓값을 다시 같은 OOF 점수로 보고하므로 결합 전략 선택 자유도를 반영하지 않는다.
AUC 가중 평균의 가중치는 전체 OOF 라벨로 계산한 구성원 AUC에서 직접 만들어져 행별 평가에도 같은 라벨 정보가 들어간다.
현재 저장소는 채택 판정과 결합 전략을 ADR 0001의 고정 fold, 3시드 확정과 nested OOF 규약으로 분리한다.

### 의사 라벨링 경로의 간접 누출

현재 소스는 모든 fold 모델의 시험 예측을 평균한 `pred_final`에서 한 세트의 의사 라벨을 만든다.
fold `k`를 검증할 때 이 `pred_final`을 만든 다른 fold 모델 네 개는 fold `k`의 실라벨을 학습에 사용했다.
그 의사 라벨을 fold `k`의 훈련 자료에 넣으면 `k`의 라벨 정보가 시험 예측을 거쳐 다시 훈련으로 돌아온다.
검증 행 자체를 증강 훈련 표에 직접 넣지 않았다는 사실만으로는 이 경로가 사라지지 않는다.
기존 이슈 68이 요구하듯 outer fold마다 그 fold의 라벨을 전혀 보지 않은 교사와 의사 라벨을 독립 생성해야 한다.

### 분포 이동 해석

적대적 검증 `0.5639`를 `0.5`에 가깝고 결측률 차이가 단순 표본 오차라는 본문 해석은 현재 근거와 맞지 않는다.
훈련과 시험의 최대 결측률 차이는 `social_media_hours`의 약 3.4%p이며, 기존 [디스커션 종합](discussion-insights.md)은 12개 열 모두의 차이가 z 값 13에서 44로 우연이 아니라고 기록한다.
관측값 분포를 결측 대체로 통제하면 적대적 AUC가 약 `0.503`으로 내려가므로, 허용되는 결론은 값 분포 이동이 아니라 결측 패턴 이동이 있다는 것이다.
결측 표시는 목표값 신호보다 자료 출처 신호를 담아 CV를 올리고 시험 일반화를 해칠 수 있으므로 기본 피처로 채택하지 않는다.

### 재현성

벽시계 제한으로 정한 Optuna 시행 수와 MLP·두 번째 seed 실행 여부는 하드웨어 속도와 Kaggle 부하에 따라 달라진다.
Optuna를 현장에서 설치할 때 판본을 고정하지 않고, 이미 설치된 NumPy, pandas, scikit-learn, LightGBM, XGBoost, CatBoost와 SciPy의 판본도 출력하지 않는다.
라이브러리별 스레드 수와 결정적 실행 설정을 고정하지 않는다.
고정 fold 파일, OOF, 시험 예측, study와 모델 파일을 저장하지 않아 결과를 다시 채점하거나 후보 풀 다양성을 측정할 수 없다.
소스와 저장 출력의 불일치까지 있어 같은 공개 판본을 처음부터 실행했을 때 제출 파일이 같다는 보장이 없다.

## 현재 저장소와 기법별 대조

| 노트북 기법 | 현재 저장소의 더 강한 근거 | 판정 |
| --- | --- | --- |
| `id` 순위와 적대적 검증 | [ID 구간 진단](https://github.com/tmheo/predicting-smartphone-addiction/issues/55)과 [디스커션 종합](discussion-insights.md)이 값과 결측 이동을 분리했다 | 기존 진단 재확인 |
| 결측 표시 12개 | 통제된 제거 실험과 현재 champion이 기본 결측 표시를 쓰지 않는다 | 기각 유지 |
| 일반 비율·곱·로그·구간 | [남은 실험 공간 전수 재점검](remaining-experiment-space-audit.md)과 복원 조성 실험이 소수 조성 열만 채택했다 | 넓은 묶음 기각 |
| `passive_hours`와 slack | [산술 잔차 표현 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/46)이 `other_screen + screen_slack`을 3시드 `+0.00081`로 채택했다 | 이미 채택 |
| 제약 결측 재구성 | 현재 champion은 제약 재구성, XGBoost 복원과 조성 다섯 열을 사용한다 | 더 강한 형태로 채택 |
| 범주·정확값 TE | [정확값 TE 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/34)과 Lookup 실험이 누출 없는 fold-fit으로 검증했다 | 이미 반영 |
| 추가 단일 값 표현 | [단일 컬럼 추가 표현 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/49)에서 빈도, 대체, 소수 자리, 반올림 TE와 잔차 TE가 전패했다 | 기각 유지 |
| 여섯 쌍 또는 여덟 쌍 TE | [쌍 TE·CE 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/51)의 66쌍과 [전 쌍 격자 TE](https://github.com/tmheo/predicting-smartphone-addiction/issues/75)의 36쌍 블록이 champion 교체에 실패했다 | 새 쌍 근거 없음 |
| 자리 분해와 삼각함수 | [남은 실험 공간 전수 재점검](remaining-experiment-space-audit.md)이 600개 특성 제거 실험의 삼각함수·두 열 TE 생존을 이미 검토했으나 최종 OOF 약 0.962와 정확값 표현 중복으로 기각했다 | 기각 유지 |
| LightGBM·XGBoost·CatBoost | 같은 피처와 고정 3시드로 비교해 CatBoost와 XGBoost를 후보 풀에 이미 등록했다 | 기존 구성원이 더 강함 |
| HistGradientBoosting | [트리 모델 계열 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/59)에서 스크리닝 `-0.00014`, 풀 기여 `-0.00001`로 종결했다 | 기각 유지 |
| elastic-net 로지스틱 회귀 | 후보 풀의 exact-value one-hot L2 로지스틱 회귀는 OOF `0.95966`과 양의 기여를 기록했고, 노트북 모델 `0.95606`은 현재 풀 하한보다 낮다 | 새 구성원 기각 |
| 작은 MLP | [외부 OOF 진단](https://github.com/tmheo/predicting-smartphone-addiction/issues/77)과 [남은 실험 공간 재점검](remaining-experiment-space-audit.md)이 약한 일반 MLP 추가를 기각했다 | 미실행 코드, 기각 유지 |
| 벽시계 Optuna | 저장소는 설정별 고정 실행과 ADR 0001의 스크리닝·3시드 확정을 사용한다 | 재현 불가 운영 방식 기각 |
| 확률 평균과 OOF 가중 평균 | 후보 풀과 [순위 평균·nested 선형 스태킹](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)이 더 엄격한 평가를 맡는다 | 기존 열린 항목에 흡수 |
| 결측 구간별 로짓 결합 | [비선형·구간별 2단 결합](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)이 완전한 nested OOF와 단순 결합 대비 문턱을 요구한다 | 기존 열린 항목에 흡수 |
| 의사 라벨링 | [의사 라벨링 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/68)이 fold별 독립 의사 라벨과 완전한 중첩 검증을 이미 요구한다 | 방향은 중복, 현재 구현 기각 |

현재 champion은 `exp067_lookup_xgb_impute_comps5`의 3시드 평균본이며 OOF AUC는 `0.9690978395195489`다.
후보 풀에는 정확값 TE LightGBM, 잔차, 원본 프록시, 제약 재구성, 전 쌍 격자 TE, XGBoost, 정확값 one-hot 로지스틱 회귀, Lookup-Transformer, TabM, CatBoost 정확값 범주, TabPFN-3, LightGBM 규제 묶음과 Lookup fold 내 초기화 평균 등 서로 다른 오차 계열이 이미 들어 있다.
노트북의 최고 저장 수치 `0.96799`는 champion보다 약 `-0.00111`이고 누출과 선택 편향까지 있으므로 단독 challenger 우선순위를 만들지 못한다.
노트북의 LightGBM·XGBoost·CatBoost·HGB와 선형 모델은 현재 풀의 같은 계열보다 약하거나 이미 기각된 계열이다.

## 사용 조건과 출처

Kaggle 공개 노트북 소스는 [Kaggle 공식 Meta Kaggle Code](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)의 설명과 공개 페이지의 사용 조건에 따라 Apache License 2.0으로 공개된다.
따라서 판본 10의 코드를 참고하거나 수정해 배포할 수 있지만 [Apache License 2.0 원문](https://www.apache.org/licenses/LICENSE-2.0.txt), 원 저작권·사용 조건·고지 유지와 수정 표시 의무를 따라야 한다.
이번 작업은 코드나 산출물을 복사하지 않고 작동 방식만 분석했으므로 저장소에 새 라이선스 파일이나 NOTICE를 추가하지 않는다.
노트북은 외부 코드 저장소를 복사하거나 Kaggle 모델을 불러오지 않는다.
대회 `train.csv`, `test.csv`와 `sample_submission.csv`의 사용은 공개 노트북 소스의 Apache License 2.0과 별개이며 [S6E8 공식 규칙](https://www.kaggle.com/competitions/playground-series-s6e8/rules)을 따른다.
입력 자료를 이 보고서나 커밋에 재배포하지 않는다.

직접 import하는 주요 패키지의 원 배포 사용 조건은 다음과 같다.

- NumPy는 [공식 저장소의 BSD 3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)를 따른다.
- pandas는 [공식 저장소의 BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/main/LICENSE)를 따른다.
- SciPy는 [공식 저장소의 BSD 3-Clause License](https://github.com/scipy/scipy/blob/main/LICENSE.txt)를 따른다.
- scikit-learn은 [공식 저장소의 BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)를 따른다.
- LightGBM은 [공식 저장소의 MIT License](https://github.com/microsoft/LightGBM/blob/master/LICENSE)를 따른다.
- XGBoost는 [공식 저장소의 Apache License 2.0](https://github.com/dmlc/xgboost/blob/master/LICENSE)을 따른다.
- CatBoost는 [공식 저장소의 Apache License 2.0](https://github.com/catboost/catboost/blob/master/LICENSE)을 따른다.
- Optuna는 [공식 저장소의 MIT License](https://github.com/optuna/optuna/blob/master/LICENSE)를 따른다.

의존성 사용 조건에는 재구현을 막는 요소가 없지만, 노트북이 정확한 패키지 판본을 고정하지 않아 실행 재현성은 별도 문제로 남는다.
Kaggle Docker 이미지에는 더 많은 전이 의존성이 포함되므로 이미지 전체를 재배포할 때는 각 구성요소의 사용 조건을 따로 확인해야 한다.

## 최종 판정

새 실험 항목은 만들지 않는다.
직접 채택할 코드는 없다.
산술 제약과 정확값 신호는 저장소가 이미 더 엄격한 fold-fit과 3시드 판정으로 채택했다.
결측 구간별 결합은 기존 이슈 67, 의사 라벨링은 기존 이슈 68의 질문과 구현 조건을 바꾸지 않고 참고 반례로만 남긴다.
특히 이 노트북의 전역 OOF 타깃 인코딩과 전역 의사 라벨은 해당 기존 항목에서 피해야 할 누출 사례다.
추가 두 쌍 TE와 MLP는 미실행이고, 자리·삼각함수 후보는 기존 더 강한 조사에서 이미 기각됐으므로 조건부 실험도 열지 않는다.
후속 공개 판본이 현재 v4 소스를 처음부터 완주하고 fold별 OOF·시험 예측·의존성·산출물 해시와 outer fold 안에서 다시 만든 타깃 인코딩을 공개한다면 수치 근거를 다시 볼 수 있다.
그 경우에도 새 후보는 현재 champion과 같은 `artifacts/folds.parquet`, seed 42 짝비교, 플라시보 게이트와 ADR 0001을 통과해야 한다.

## 한계

이 조사는 공개 소스와 저장 실행 결과를 정적으로 감사했으며 노트북 전체를 10시간 넘게 다시 실행하지 않았다.
조사 중 후속 Kaggle 세션이 실행 중이어서 공개 판본 10의 출력 파일 바이트를 내려받지 못했고, 출력 파일별 SHA-256을 확정하지 못했다.
저장 OOF와 모델별 시험 예측이 없어 `0.96799`를 독립 재채점하거나 현재 후보 풀과 상관·기여를 직접 측정할 수 없다.
Kaggle이 후속 실행을 새 공개 판본으로 승격하면 판본 번호, 소스, 출력과 결론은 달라질 수 있다.
