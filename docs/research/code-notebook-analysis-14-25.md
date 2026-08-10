# Playground Series S6E8 득표 노트북 코드 분석: 14위부터 25위

## 조사 범위와 근거

이 문서는 2026-08-10에 확정한 [득표순 전수 목록](code-notebook-inventory.md)의 14위부터 25위까지를 고정 대상으로 삼는다.
득표 수와 마지막 변경 시각은 고정 목록의 조사 시점 값을 그대로 사용했다.
각 고정 주소에서 공식 [Kaggle CLI 2.2.4](https://github.com/Kaggle/kaggle-cli)의 `kaggle kernels pull <owner>/<slug> -m` 명령으로 최신 공개 판본의 노트북 원문과 메타데이터를 내려받았다.
노트북 원문은 12개 모두 정상적인 JSON으로 읽혔고, 모든 코드 셀을 처음부터 끝까지 확인했다.
Kaggle CLI가 내려준 노트북에는 실행 출력이 포함되지 않았으므로, 검증 점수는 코드에 문자열로 남아 있거나 노트북 본문에 명시된 값만 기록했다.
공개 점수도 제목이나 본문이 공개 점수라고 명시한 경우에만 기록했다.
따라서 점수가 없다는 표시는 점수가 낮다는 뜻이 아니라 현재 공개 소스에서 확인할 수 없다는 뜻이다.

## 전체 결론

- 가장 바로 재사용할 만한 뼈대는 16위의 5겹 층화 교차 검증 LightGBM과 21위의 10겹 층화 교차 검증 LightGBM이다.
- 반복해서 등장하는 유망 특성은 결측 여부, 행별 결측 개수, 주말과 평일 화면 시간 차이, 활동별 화면 시간 비율, 설명되지 않은 화면 시간, 반올림값과 소수 자릿값, 화면 활동 조합이다.
- 이 특성들이 실제로 점수를 높인다는 주장은 현재 코드만으로는 대부분 분리 검증되지 않았으므로, 같은 분할을 고정한 제거 실험으로 다시 확인해야 한다.
- 15위의 2단계 결합은 여러 공개 OOF 예측을 재사용하는 구조가 가치 있지만, 행을 `id`로 맞추지 않고 순서만 같다고 가정하며 누수 가능성이 있는 22위와 24위의 OOF도 입력으로 사용한다.
- 20위의 가중 결합은 전체 OOF 목표값에 직접 가중치를 맞춘 뒤 같은 OOF에서 성능을 보고하므로 결합 성능을 낙관적으로 추정할 수 있다.
- 22위 RealMLP와 24위 TabM은 목표값 인코딩을 바깥쪽 교차 검증 전에 한 번만 만들기 때문에 바깥쪽 검증 목표값이 학습 행의 특성에 간접적으로 들어가는 교차 폴드 누수가 있다.
- 18위 ANN과 25위 모형 비교는 전체 학습 자료로 결측 대체나 표준화를 마친 뒤 검증 자료를 나누므로 검증 분포를 미리 본다.
- 23위는 사전 학습 LightGBM 묶음을 불러오는 추론 전용 노트북이라 학습 자료, 분할, 검증 점수, 개별 모형 구성을 현재 코드에서 확인할 수 없다.
- 12개 중 합성 자료 생성 규칙을 코드로 밝힌 노트북은 없다.
- 16위와 21위의 자릿값, 반올림값, 조합값 특성 및 19위, 22위, 24위의 수치형 값을 범주로 취급하는 방식은 양자화된 합성 자료 구조를 노리는 시도로 볼 수 있지만, 생성 규칙을 입증하는 근거로 사용해서는 안 된다.

## 핵심 비교표

| 순위 | 노트북 | 득표 | 주 접근법 | 검증 방식 | 명시된 공개 점수 | 핵심 판정 |
| ---: | --- | ---: | --- | --- | --- | --- |
| 14 | [Complete EDA: Predicting Smartphone Addition](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition) | 17 | EDA와 효과 크기 | 없음 | 없음 | 자료 점검용이며 모형 근거는 아니다. |
| 15 | [PlaygroundS6E8\|Public\|L2Stack\|V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1) | 17 | 공개 OOF 예측의 Ridge 2단계 결합 | 새 5겹 분할의 `PredefinedSplit` | 없음 | 구조는 재사용할 만하지만 행 순서 가정과 입력 OOF 누수를 제거해야 한다. |
| 16 | [S6E8: LGBM](https://www.kaggle.com/code/donmarch14/s6e8-lgbm) | 17 | 고급 특성의 LightGBM | 5겹 층화 OOF와 조기 종료 | 없음 | 가장 실용적인 단일 모형 기준선 후보다. |
| 17 | [S6E8 \| 13 FE Features + XGBoost + Optuna \| 0.96602](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602) | 16 | 13개 특성과 XGBoost | 단일 층화 80 대 20 보류 | 0.96602 | 공개 점수는 확인되지만 특성 향상 경로는 현재 코드로 재현되지 않는다. |
| 18 | [Smartphone Addiction Prediction \| ANN](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann) | 16 | 5층 완전 연결 신경망 | Keras `validation_split=0.2` | 없음 | 전체 자료 전처리 뒤 검증을 나눠 검증 분포를 미리 본다. |
| 19 | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction) | 16 | 직접 구현 RealMLP | 5겹 층화 OOF | 없음 | 목표값 인코딩 위치는 타당하지만 자료 기반 전처리를 폴드 안으로 옮겨야 한다. |
| 20 | [Feature-Engineered GBDT: Smartphone Addiction AUC](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc) | 16 | 네 트리 계열 모형의 가중 결합 | 5겹 층화 OOF | 없음 | 같은 OOF에 가중치를 맞추고 평가해 결합 성능이 낙관적일 수 있다. |
| 21 | [S6E8 Single LGB](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb) | 16 | 확장 특성의 LightGBM | 10겹 층화 OOF와 조기 종료 | 없음 | 풍부한 특성 후보가 있으나 폴드 밖 결측 대체를 고쳐야 한다. |
| 22 | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction) | 15 | 목표값 인코딩과 RealMLP | 5겹 층화 OOF | 없음 | 교차 폴드 목표값 누수로 현재 OOF를 신뢰할 수 없다. |
| 23 | [Predicting smartphone addiction](https://www.kaggle.com/code/jek1wantaufik/predicting-smartphone-addiction) | 14 | 사전 학습 LightGBM 평균 추론 | 현재 코드에 없음 | 없음 | 학습과 검증을 감사할 수 없는 추론 전용 코드다. |
| 24 | [TabM for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction) | 14 | 목표값 인코딩과 TabM | 5겹 층화 OOF | 없음 | 22위와 같은 교차 폴드 목표값 누수가 있다. |
| 25 | [predicting-smartphone-addiction](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction) | 13 | 기본 분류기와 무작위 매개변수 탐색 | 단일 80 대 20 보류와 내부 5겹 F1 | 없음 | 경쟁 지표와 검증 설계가 맞지 않고 제출 코드가 없다. |

## 항목별 분석

### 14위: [Complete EDA: Predicting Smartphone Addition](https://www.kaggle.com/code/sarveshchhetri/complete-eda-predicting-smartphone-addition)

- 득표 수는 17개이고 마지막 변경 시각은 2026-08-06T16:50:10.190000Z다.
- 공개 점수는 명시되지 않았다.
- 이 노트북은 모형을 학습하지 않고 결측률, 목표값 비율, 수치 특성 분포, 목표값별 커널 밀도, Cohen의 d, Welch t 검정, 상관관계, 학습 자료와 시험 자료 분포를 시각적으로 비교한다.
- 검증 분할, OOF 예측, 제출 파일 생성은 없다.
- 결측 대체나 특성 생성도 실행하지 않고 마지막 본문에서 비율, 합계, 결측 표시 특성을 후속 아이디어로만 제안한다.
- 합성 자료 생성 규칙을 추정하거나 검증하는 코드는 없다.

핵심 코드 근거는 목표값별 평균 차이를 표준화하고 Welch t 검정을 함께 계산하는 부분이다.

```python
pooled_std = np.sqrt((a.var() + b.var()) / 2)
cohens_d = (a.mean() - b.mean()) / pooled_std
t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
```

재사용 가치는 자료를 빠르게 점검하는 출발점에 있다.
특히 목표값별 효과 크기와 학습 및 시험 자료의 결측률을 함께 보는 코드는 모형을 만들기 전에 신호와 분포 차이를 점검하기 좋다.

주의할 점은 마지막 요약의 `None detected`라는 학습 및 시험 자료 이동 결론이 커널 밀도 그림을 눈으로 본 결과일 뿐, 적대적 검증이나 통계 검정으로 뒷받침되지 않는다는 것이다.
또한 t 검정의 매우 작은 p값은 약 69만 행이라는 큰 표본 크기의 영향을 강하게 받으므로 효과 크기와 함께 해석해야 한다.

### 15위: [PlaygroundS6E8|Public|L2Stack|V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1)

- 득표 수는 17개이고 마지막 변경 시각은 2026-08-08T05:36:30.160000Z다.
- 공개 점수는 명시되지 않았다.
- 본 노트북은 여러 공개 노트북에서 모은 OOF 및 시험 예측 열을 입력으로 읽고, `MinMaxScaler`와 `Ridge`로 구성한 2단계 모형을 학습한다.
- 공개 입력 예측을 모으는 과정은 연결된 [자료 취합 노트북](https://www.kaggle.com/code/ravi20076/playgrounds6e8-datacollation-v1)에 있고, 학습기 구현은 연결된 [공통 코드 노트북](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1)에 있다.
- 자료 취합 코드는 Don Mani의 CatBoost와 LightGBM, Ravi Ramakrishnan의 두 기준 모형, Omid Baghcheh Saraei의 XGBoost, RealMLP, TabM, ResNet, FLAML XGBoost, FT-Transformer, Hamza의 XGBoost, Tamerlan Omralinov의 세 예측을 합친다.
- 취합 단계가 새로 만든 5겹 층화 분할 번호를 `fold_nb`로 저장하고, 본 노트북의 `ModelTrainer`가 이를 `PredefinedSplit`에 넣어 Ridge의 OOF 예측과 시험 예측 평균을 만든다.
- 2단계 모형은 분류기가 아니라 Ridge 회귀이고, 평가는 `roc_auc_score`다.
- 합성 자료 규칙이나 원자료를 직접 다루는 코드는 없다.

핵심 코드 근거는 사전 계산 예측을 그대로 특성으로 쓰는 다음 부분이다.

```python
Xtrain = pd.read_parquet("OOF_Preds_PublicV1_1.parquet")
Xtest = pd.read_parquet("Mdl_Preds_PublicV1_1.parquet")
pipe = Pipeline([("PP", MinMaxScaler()), ("M", Ridge(max_iter=100_000, random_state=42))])
cv = PredefinedSplit(ygrp)
```

재사용 가치는 기본 모형의 OOF와 시험 예측을 같은 열 구조로 보존한 뒤 별도 2단계 모형을 학습하는 구성에 있다.
이 방식은 공개 예측 파일을 단순 평균하는 것보다 각 모형의 중복 신호를 조정할 여지가 있다.

주의할 점은 자료 취합 코드가 모든 입력의 `id`를 버리고 `range(len(df))`로 색인을 다시 붙인다는 것이다.
행 순서가 하나라도 다르면 예측과 목표값이 조용히 어긋나므로 반드시 `id`로 일대일 결합하고 중복 및 누락을 검사해야 한다.
또한 입력 OOF 중 22위와 24위에서 생성된 파일은 이 문서가 확인한 목표값 인코딩 누수 위험을 포함하므로, 이 2단계 모형의 OOF도 완전히 정직하다고 볼 수 없다.
Ridge 출력은 확률 범위로 제한되지 않으므로 제출값 범위도 따로 검사하는 편이 안전하다.

### 16위: [S6E8: LGBM](https://www.kaggle.com/code/donmarch14/s6e8-lgbm)

- 득표 수는 17개이고 마지막 변경 시각은 2026-08-04T03:09:55.667000Z다.
- 공개 점수는 명시되지 않았고, 내려받은 소스에는 실행 출력이 없어 OOF AUC 수치도 확인할 수 없다.
- 결측 표시와 행별 결측 개수, 여가 시간, 설명되지 않은 화면 시간, 주말과 평일 차이, 수면 부족, 알림당 앱 실행, 화면 활동 비율을 만든다.
- 연속값의 일의 자리, 소수 첫째 자리, 소수 둘째 자리, 내림값, 한 자리 반올림값, 나머지, 소수부 구간을 만들고 화면 시간 쌍을 정수와 문자열 범주로 결합한다.
- 5겹 층화 교차 검증에서 LightGBM을 학습하며, 각 폴드의 검증 AUC를 기준으로 최대 300회 조기 종료하고 시험 예측을 다섯 모형의 산술 평균으로 만든다.
- GPU 사용을 먼저 시도하고 실패하면 CPU로 다시 학습한다.
- 원시 범주와 파생 범주의 코드는 학습 자료와 시험 자료를 합쳐 공통 번호를 만들지만 목표값은 사용하지 않는다.
- 합성 자료 생성 규칙을 밝히지는 않으며, 자릿값과 반올림값 특성은 수치 양자화 구조를 탐색하는 구현일 뿐이다.

핵심 코드 근거는 결측 및 행동 특성과 자릿값 특성을 함께 만드는 부분이다.

```python
out['other_screen'] = out.daily_screen_time_hours - (
    out.social_media_hours + out.gaming_hours + out.work_study_hours
)
out['weekend_minus_daily_screen'] = out.weekend_screen_time - out.daily_screen_time_hours
out[f'{c}_digit_hundredths'] = (np.floor(np.abs(v) * 100).astype('Int32') % 10).astype('float32')
```

재사용 가치는 단일 LightGBM 기준 모형으로 OOF와 시험 예측을 모두 저장한다는 점, 그리고 특성 생성 함수가 학습 자료와 시험 자료에 동일하게 적용된다는 점에 있다.
특히 `other_screen`, 주말 차이, 행별 결측 개수, 반올림 및 자릿값 특성은 같은 분할에서 제거 실험을 해볼 가치가 있다.

주의할 점은 범주 번호를 학습 자료와 시험 자료 전체에서 만든다는 점이다.
목표값 누수는 아니지만 시험 자료 분포를 미리 사용하는 처리이므로, 새 자료에 대한 일반화 성능만 재려는 엄격한 검증과는 맞지 않는다.
또한 코드 중간에 많은 파생 범주를 `Categorical`로 만든 뒤 학습 직전에 `cat_cols`를 원래 세 범주로 다시 덮어쓰므로, LightGBM 버전에 따른 자동 범주 인식 동작에 의존한다.
재사용할 때는 모든 범주 열을 명시적으로 한 목록에서 관리하는 편이 안전하다.

### 17위: [S6E8 | 13 FE Features + XGBoost + Optuna | 0.96602](https://www.kaggle.com/code/rugvedbane/s6e8-13-fe-features-xgboost-optuna-0-96602)

- 득표 수는 16개이고 마지막 변경 시각은 2026-08-08T14:58:09.067000Z다.
- 제목과 본문이 공개 점수를 0.96602라고 명시한다.
- 주 접근법은 세 개의 임계값 표시, 주말 차이와 설명되지 않은 화면 시간, 다섯 개의 비율, 세 개의 순서형 구간으로 총 13개 파생 특성을 만든 뒤 XGBoost를 학습하는 것이다.
- 수치 특성은 중앙값으로 결측을 채우고, `stress_level`은 순서형 번호로 바꾸며, `gender`의 각 범주는 독립된 0/1 열로 펼친다.
- 공개된 현재 코드가 직접 실행하는 검증은 목표값 비율을 보존한 단일 80 대 20 분할이다.
- Optuna의 3겹 층화 교차 검증 코드는 주석 처리되어 있고, 200회 탐색에서 얻었다고 설명한 매개변수만 고정값으로 남아 있다.
- 최종 XGBoost는 전체 학습 자료로 다시 학습하고 시험 확률을 제출한다.
- 합성 자료 생성 규칙을 다루는 코드는 없다.

핵심 코드 근거는 화면 시간에서 임계값, 차이, 비율, 구간을 만드는 부분이다.

```python
df['high_screen_time'] = (df['daily_screen_time_hours'] > 7).astype(int)
df['weekend_gap'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
df['work_ratio'] = df['work_study_hours'] / (df['daily_screen_time_hours'] + 1)
df['screen_time_bin'] = pd.cut(df['daily_screen_time_hours'], bins=[0, 4, 7, 10, 15], labels=[0, 1, 2, 3])
```

재사용 가치는 작고 이해하기 쉬운 특성 묶음을 명시적으로 한 함수에 모아 두었다는 점이다.
0.96602라는 공개 점수가 확인되는 유일한 대상이므로 빠른 제출 기준선으로도 참고할 수 있다.

주의할 점은 특성별 점수 향상 경로가 현재 실행 코드로 재현되지 않고 본문 설명에만 남아 있다는 것이다.
현재 검증은 단일 보류 자료이므로 분할 운에 민감하고, 공개 점수로 특성을 고른 흔적이 있다면 공개 순위표 과적합도 배제할 수 없다.
본문은 `scale_pos_weight`를 적용했다고 설명하지만 실제 `best_params_200_trials`에는 그 매개변수가 없다.
따라서 설명보다 코드를 기준으로 재현해야 한다.

### 18위: [Smartphone Addiction Prediction | ANN](https://www.kaggle.com/code/hamidrana/smartphone-addiction-prediction-ann)

- 득표 수는 16개이고 마지막 변경 시각은 2026-08-02T16:10:31.970000Z다.
- 공개 점수는 명시되지 않았다.
- 수치 특성은 평균 또는 중앙값으로 결측을 채우고, 범주 특성은 최빈값으로 채운 뒤 각 범주를 독립된 0/1 열로 펼친다.
- 수치 특성은 `StandardScaler`로 표준화하고, 128개 노드를 가진 ReLU 은닉층 다섯 개와 시그모이드 출력층으로 구성한 신경망을 학습한다.
- `model.fit(..., validation_split=0.2)`의 검증 AUC가 세 차례 연속 개선되지 않으면 학습을 중단하고 가장 좋았던 가중치로 돌아간다.
- 검증 자료는 명시적인 층화 분할이 아니며, 별도의 OOF 예측도 없다.
- 시험 확률을 `submission1.csv`로 저장한다.
- 합성 자료 규칙이나 파생 특성은 없다.

핵심 코드 근거는 다음 학습부다.

```python
model = Sequential([
    Dense(128, activation='relu', input_dim=17),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid'),
])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[callback])
```

재사용 가치는 신경망 기준선을 매우 짧게 만드는 예시에 한정된다.
현재 자료 크기에서 비선형 신경망이 트리 모형과 다른 잔차를 만드는지 확인하는 비교 대상으로는 쓸 수 있다.

주의할 점은 결측 대체, 범주를 독립 열로 펼치는 변환, 표준화를 전체 학습 자료에 적용한 뒤 Keras가 마지막 20%를 검증 자료로 떼므로 검증 분포를 미리 본다는 것이다.
학습 자료와 시험 자료의 결측값을 각각의 통계로 따로 채워 처리 규칙도 일치하지 않는다.
무작위 시드를 고정하지 않아 재현성도 낮고, 경쟁 평가지표가 ROC AUC인데 모형 비교를 위한 OOF 설계가 없다.

### 19위: [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction)

- 득표 수는 16개이고 마지막 변경 시각은 2026-08-02T07:18:37.677000Z다.
- 공개 점수는 명시되지 않았고, 실행 출력이 없어 코드가 출력하도록 한 전체 OOF AUC 값은 확인할 수 없다.
- PyTorch로 여덟 개 내부 모형을 동시에 계산하는 RealMLP를 직접 구현한다.
- 수치 특성은 학습 가능한 주기 함수로 여러 값으로 확장하고, 범주 특성은 값 개수에 따라 독립된 0/1 열 또는 학습 벡터로 바꾼다.
- 결측 표시를 먼저 만든 뒤 학습 자료 중앙값으로 수치 결측을 채우고, 모든 수치 원값의 범주형 복사본과 화면 시간 두 열의 10분위 구간을 만든다.
- 바깥쪽 5겹 층화 교차 검증 안에서 `TargetEncoder.fit_transform`으로 학습 폴드의 내부 교차 검증 목표값 인코딩을 만들고, 바깥쪽 검증과 시험 자료에는 그 인코더를 변환만 적용한다.
- 각 바깥쪽 폴드에서 검증 AUC가 가장 높았던 학습 반복 시점의 지수 이동 평균 가중치를 저장하고, 시험 예측은 다섯 폴드와 여덟 내부 모형의 평균이다.
- 합성 자료 생성 규칙은 밝히지 않지만, 연속값을 원값 범주와 분위 구간으로 함께 넣는다.

핵심 코드 근거는 목표값 인코딩이 바깥쪽 폴드 안에서 만들어지는 부분이다.

```python
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    encoder = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED)
    tr_enc = encoder.fit_transform(X_tr[te_cols], y_tr)
    val_enc = encoder.transform(X_val[te_cols])
    tst_enc = encoder.transform(X_tst[te_cols])
```

재사용 가치는 신경망 다양성을 확보하는 모형 후보와 올바른 목표값 인코딩 위치를 보여주는 데 있다.
특히 여덟 내부 모형, 지수 이동 평균, 시기별 학습률과 결측 표시를 한 번의 학습 절차에 결합한 구현은 독립적인 시험 가치가 있다.

주의할 점은 중앙값, 범주 번호, `KBinsDiscretizer`를 바깥쪽 분할 전에 전체 학습 자료로 맞춘다는 것이다.
목표값을 쓰지는 않지만 OOF 검증 폴드의 분포를 미리 보므로 엄격한 검증에서는 폴드 안으로 옮겨야 한다.
시험 자료에서 학습에 없던 범주가 `-1`이 된 뒤 `np.clip`으로 0에 붙어 첫 번째 기존 범주와 같아지는 처리도 안전하지 않다.
별도의 미지 범주 번호를 확보하는 편이 낫다.

### 20위: [Feature-Engineered GBDT: Smartphone Addiction AUC](https://www.kaggle.com/code/avikdas567/feature-engineered-gbdt-smartphone-addiction-auc)

- 득표 수는 16개이고 마지막 변경 시각은 2026-08-01T12:44:48.820000Z다.
- 공개 점수는 명시되지 않았고, 실행 출력이 없어 개별 모형과 결합 모형의 OOF AUC 값은 확인할 수 없다.
- 여가 및 생산 시간, 활동 비율, 설명되지 않은 화면 시간, 주말 차이, 앱 실행당 시간, 알림 밀도, 깨어 있는 시간 대비 화면 시간, 결측 개수를 만든다.
- `HistGradientBoostingClassifier`는 항상 학습하고, 설치 여부에 따라 LightGBM, XGBoost, CatBoost도 같은 5겹 층화 분할에서 학습한다.
- 각 모형의 OOF와 시험 예측을 저장한 뒤 SLSQP가 음이 아닌 합계 1의 가중치를 전체 OOF ROC AUC에 직접 맞춘다.
- 합성 자료 규칙을 다루는 코드는 없다.

핵심 코드 근거는 전체 OOF에서 결합 가중치를 직접 찾는 부분이다.

```python
def loss_func(weights):
    weights = np.array(weights) / np.sum(weights)
    return -roc_auc_score(y, np.dot(OOF_matrix, weights))

res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
```

재사용 가치는 설치된 라이브러리에 따라 모형을 추가하면서 여러 트리 계열 모형을 한 노트북에서 비교하고, 같은 폴드로 OOF를 정렬하는 구조에 있다.
행동 비율 특성 묶음도 단일 함수라 제거 실험에 옮기기 쉽다.

주의할 점은 가중치를 찾는 데 쓴 전체 OOF 목표값으로 같은 결합의 성능을 다시 계산한다는 것이다.
가중치까지 검증하려면 바깥쪽 교차 검증을 한 겹 더 두거나, 가중치 선택 전용 보류 자료와 최종 평가 자료를 분리해야 한다.
범주 인코더도 전체 학습 자료에 맞춘 뒤 교차 검증하므로 목표값 누수는 아니지만 검증 분포를 미리 본다.
`unexplained_screen_hours`는 일부 구성 요소의 결측만 0으로 바꾸므로 결측이 많을 때 물리적으로 해석하기 어려운 값이 생긴다.

### 21위: [S6E8 Single LGB](https://www.kaggle.com/code/evgendvorkin/s6e8-single-lgb)

- 득표 수는 16개이고 마지막 변경 시각은 2026-08-04T18:17:46.023000Z다.
- 공개 점수는 명시되지 않았고, 실행 출력이 없어 코드가 출력하는 10겹 OOF AUC 값은 확인할 수 없다.
- 원시 결측 표시를 먼저 만들고, 범주 결측은 별도 문자열로, 수치 결측은 `stress_level`과 `gender` 조합별 중앙값으로 채운다.
- 빈도, 화면 시간 합계와 비율, 설명되지 않은 화면 시간, 깨어 있는 시간, 화면 특성의 최댓값과 최솟값을 만든다.
- 이어서 자릿값, 내림값, 반올림값, 소수부 구간, 활동 요약 통계, 화면 시간 쌍을 추가한다.
- 10겹 층화 교차 검증에서 CPU LightGBM을 학습하고 시험 예측을 10개 폴드 모형의 평균으로 만든다.
- 합성 자료 생성 규칙은 밝히지 않으며, 자릿값 및 조합값은 양자화 구조를 탐색하는 코드다.

핵심 코드 근거는 그룹별 결측 대체와 고급 특성 생성이다.

```python
train_clean[col] = train_clean.groupby(['stress_level', 'gender'])[col].transform(
    lambda x: x.fillna(x.median())
)
out[f'{col}_frac20'] = np.floor((np.abs(values) % 1) * 20).fillna(-999999).astype('int32')
out['pair_daily_weekend'] = out['daily_screen_time_hours_floor'] * 10000 + out['weekend_screen_time_floor']
```

재사용 가치는 16위보다 더 넓은 특성 집합을 10겹 OOF 기준으로 시험할 수 있다는 점이다.
그룹별 결측 대체는 범주 조합마다 자료가 충분한지 확인한 뒤 독립적으로 비교할 가치가 있다.

주의할 점은 그룹 중앙값과 빈도를 바깥쪽 교차 검증 전에 전체 학습 자료로 계산한다는 것이다.
목표값을 직접 쓰지는 않지만 검증 폴드의 분포를 미리 본다.
첫 결측 대체가 끝난 뒤 `make_advanced_features`가 다시 결측 표시를 만들기 때문에 그 함수 안의 새 결측 표시는 모두 0이고, 실제 결측 정보는 앞에서 만든 `_nan` 열에만 남는다.
시험 자료에 학습 자료에 없던 `stress_level`과 `gender` 조합이 있으면 그룹 중앙값 대체 뒤에도 결측이 남을 수 있으므로 전체 중앙값 예비 처리가 필요하다.

### 22위: [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)

- 득표 수는 15개이고 마지막 변경 시각은 2026-08-01T20:26:22.647000Z다.
- 공개 점수는 명시되지 않았고, 실행 출력이 없어 OOF AUC 값은 확인할 수 없다.
- 모든 원시 특성을 문자열 범주로 바꾼 뒤 결측 개수, 활동 시간 합계, 화면 시간 비율, 설명되지 않은 화면 시간, 수면 및 알림 비율을 수치 특성으로 다시 만든다.
- 원시 12개 특성 각각에 5겹 OOF 목표값 인코딩과 빈도 인코딩을 한 번 만들어 붙인다.
- 같은 `StratifiedKFold`를 다시 사용해 RealMLP를 다섯 번 학습하고 시험 예측을 평균한다.
- RealMLP는 `pytabkit.RealMLP_TD_Classifier`를 사용하며 여덟 내부 모형, 최대 100회 학습, 검증 조기 종료를 설정한다.
- 합성 자료 규칙을 밝히지는 않지만 고유값이 많은 수치 원값까지 범주 및 목표값 인코딩 대상으로 삼는다.

핵심 코드 근거는 목표값 인코딩을 바깥쪽 모형 반복 전에 전체 자료에 대해 한 번만 만드는 부분이다.

```python
train_encoded, test_encoded, skf = apply_encodings_cv(train_df, test_df, cat_cols=target_enc_cols, target_col=TARGET)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    model.fit(X_train, y_train, X_val, y_val)
```

재사용 가치는 RealMLP 설정과 화면 활동 비율 특성에 있다.
다만 목표값 인코딩 부분은 그대로 재사용하면 안 된다.

주의할 점은 교차 폴드 누수가 가장 중요하다는 것이다.
인코딩 단계에서 폴드 `j`의 행은 `j`를 뺀 모든 목표값으로 인코딩된다.
그 뒤 바깥쪽 모형의 검증 폴드가 `k`일 때 학습 자료에 들어가는 `j != k` 행의 인코딩 통계에는 `k` 폴드의 목표값이 포함된다.
즉 바깥쪽 검증 목표값이 바깥쪽 학습 특성에 간접적으로 들어가므로 보고된 OOF는 정직한 일반화 추정치가 아니다.
각 바깥쪽 폴드 안에서 바깥쪽 학습 자료만 사용해 목표값 인코더를 새로 만들고, 그 인코더로 바깥쪽 검증과 시험 자료를 변환해야 한다.

### 23위: [Predicting smartphone addiction](https://www.kaggle.com/code/jek1wantaufik/predicting-smartphone-addiction)

- 득표 수는 14개이고 마지막 변경 시각은 2026-08-03T08:43:25.227000Z다.
- 공개 점수와 검증 점수는 명시되지 않았다.
- 이 노트북은 시험 자료만 읽는 추론 전용 코드다.
- 범주 결측을 `Missing`으로 채우고 수치 결측 표시와 다섯 개 비율 또는 합계 특성을 만든 뒤, [Kaggle Model의 `buddy` 판본 1](https://www.kaggle.com/models/jek1wantaufik/buddy/ScikitLearn/addicted/1)에서 `lgb_models.pkl`과 `features.pkl`을 불러온다.
- 여러 사전 학습 모형의 양성 확률을 산술 평균해 제출한다.
- 현재 노트북에는 학습 코드, 학습 자료 처리, 분할 방식, 개별 모형 수와 매개변수, OOF 예측이 없다.
- 합성 자료 규칙을 다루는 코드도 없다.

핵심 코드 근거는 직렬화된 모형 묶음의 추론 평균이다.

```python
models = joblib.load(".../lgb_models.pkl")
features = joblib.load(".../features.pkl")
for model in models:
    preds += model.predict_proba(X_test)[:, 1]
preds /= len(models)
```

재사용 가치는 학습과 추론을 분리해 모형 묶음과 정확한 특성 목록을 함께 배포하는 방식에 있다.

주의할 점은 현재 공개 노트북만으로 모형이 어떤 자료와 검증으로 만들어졌는지 감사할 수 없다는 것이다.
사전 처리도 학습 때의 구현과 정확히 같다는 외부 보장이 필요하다.
특히 수치 결측은 표시 열만 추가하고 값 자체는 채우지 않으므로 직렬화된 LightGBM이 결측을 직접 처리하도록 학습되었다는 가정에 의존한다.
검증된 재현 가능한 기준선으로 채택하려면 학습 코드와 OOF 산출물을 함께 공개해야 한다.

### 24위: [TabM for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction)

- 득표 수는 14개이고 마지막 변경 시각은 2026-08-02T00:29:45.990000Z다.
- 공개 점수는 명시되지 않았고, 실행 출력이 없어 OOF AUC 값은 확인할 수 없다.
- 전처리, 파생 특성, 목표값 인코딩, 빈도 인코딩은 22위와 사실상 같다.
- 모형만 `pytabkit.TabM_D_Classifier`로 바뀌며 24개 내부 모형, 구간별 선형 수치 확장, 은닉 블록 여섯 개, 최대 100회 학습을 사용하고 검증 성능이 다섯 차례 연속 개선되지 않으면 중단한다.
- 같은 5겹 층화 분할에서 OOF와 시험 예측을 만들고 시험 예측을 평균한다.
- 합성 자료 규칙을 밝히지는 않지만 고유값이 많은 수치 원값을 범주와 목표값 인코딩 대상으로 삼는다.

핵심 코드 근거는 22위와 같은 사전 인코딩 뒤 TabM을 학습하는 부분이다.

```python
train_encoded, test_encoded, skf = apply_encodings_cv(train_df, test_df, cat_cols=target_enc_cols, target_col=TARGET)
model = TabM_D_Classifier(
    arch_type='tabm-mini-normal', tabm_k=24, num_emb_type='pwl', n_blocks=6
)
```

재사용 가치는 TabM이 LightGBM 및 RealMLP와 다른 잔차를 만드는지 비교할 수 있는 모형 설정에 있다.
목표값을 쓰지 않는 화면 활동 비율 특성도 독립적으로 옮길 수 있다.

주의할 점은 22위와 같은 교차 폴드 목표값 누수가 있다는 것이다.
바깥쪽 검증 폴드 `k`의 목표값이 다른 폴드 행의 목표값 인코딩 통계에 포함된 상태로 바깥쪽 모형이 학습된다.
따라서 현재 OOF AUC는 모형 비교나 2단계 결합 입력 선별에 사용하면 안 된다.
바깥쪽 폴드 안에서 인코딩 전체를 다시 계산한 뒤 TabM을 재평가해야 한다.

### 25위: [predicting-smartphone-addiction](https://www.kaggle.com/code/devashish001/predicting-smartphone-addiction)

- 득표 수는 13개이고 마지막 변경 시각은 2026-08-10T06:11:07.303000Z다.
- 공개 점수는 명시되지 않았다.
- 결측을 전체 학습 자료의 최빈값과 평균으로 채우고 세 범주를 `LabelEncoder`로 정수화한다.
- 단일 80 대 20 무작위 분할에서 기본 로지스틱 회귀, L1 및 L2 로지스틱 회귀, Gaussian Naive Bayes, Random Forest, XGBoost, LightGBM을 비교한다.
- XGBoost와 LightGBM은 학습 쪽 80% 안에서 각각 50개 무작위 매개변수 조합과 5겹 교차 검증을 사용해 F1을 최대화한다.
- 최종 20% 평가는 확률 ROC AUC가 아니라 임계값 0.5의 정확도와 F1이다.
- 시험 자료를 읽거나 제출 파일을 만드는 코드는 없고, 파생 특성도 실제로 만들지 않는다.
- 합성 자료 규칙을 다루는 코드도 없다.

핵심 코드 근거는 경쟁 지표와 다른 F1 중심 탐색이다.

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
rfc_grid = RandomizedSearchCV(xgbc, param_distributions=param_dist, n_iter=50, cv=5, scoring="f1")
lgbm_tuned_search = RandomizedSearchCV(lgbm, param_distributions=param_dist_lgbm, n_iter=50, cv=5, scoring="f1")
```

재사용 가치는 여러 기본 분류기를 한 자료에서 빠르게 비교하는 교육용 골격에 있다.
경쟁용 기준선으로는 평가 지표와 검증 설계를 먼저 바꿔야 한다.

주의할 점은 결측 대체와 범주 번호화를 80 대 20 분할 전에 전체 학습 자료로 맞춘다는 것이다.
검증 분포를 미리 보며, 분할도 목표값 비율을 보존하지 않는다.
매개변수 탐색 지표와 최종 평가 지표가 모두 경쟁 지표인 ROC AUC와 다르고, 확률이 아닌 이진 예측을 평가한다.
따라서 이 결과로 다른 노트북의 AUC 기준 모형과 우열을 비교할 수 없다.

## 재사용 우선순위

1. 먼저 16위의 5겹 층화 OOF LightGBM을 최소 기준선으로 옮기되, 범주 인코더와 모든 자료 기반 변환을 각 폴드 안에서 학습한다.
2. 14위의 효과 크기 및 결측 점검을 실행하고, 16위와 21위에서 반복되는 결측 표시, `other_screen`, 주말 차이, 활동 비율, 자릿값과 조합값을 하나씩 제거하며 같은 폴드에서 검증한다.
3. 19위의 RealMLP와 수정한 24위의 TabM을 같은 폴드에서 재학습해 LightGBM OOF와의 잔차 상관을 측정한다.
4. 기본 모형 예측은 반드시 `id`로 결합하고, OOF 행마다 그 행의 목표값을 보지 않은 모형이 예측했는지 출처별로 확인한다.
5. 2단계 결합 또는 가중치 선택은 별도의 바깥쪽 검증에서 평가하고, 가중치를 맞춘 OOF와 같은 OOF로 결합 성능을 보고하지 않는다.
6. 공개 순위표 점수는 최종 확인에만 쓰고 특성 제거와 가중치 선택은 OOF 또는 별도 보류 자료에서 끝낸다.

## 검증 및 한계

- 고정 목록의 12개 주소와 내려받은 메타데이터의 `id`가 모두 일치했다.
- 12개 노트북에서 총 241개 코드 셀과 3,860개 코드 줄을 읽었다.
- 코드 셀 수와 코드 줄 수는 순서대로 14위 20개 및 163줄, 15위 7개 및 165줄, 16위 5개 및 387줄, 17위 13개 및 286줄, 18위 78개 및 265줄, 19위 6개 및 790줄, 20위 15개 및 393줄, 21위 12개 및 362줄, 22위 12개 및 326줄, 23위 1개 및 97줄, 24위 12개 및 311줄, 25위 60개 및 315줄이다.
- 15위의 현재 노트북이 실행하는 외부 코드와 입력 예측의 구성을 확인하기 위해 연결된 공통 코드 노트북과 자료 취합 노트북도 별도로 내려받아 확인했다.
- 23위의 직렬화된 모형 파일은 현재 노트북 코드에서 학습 이력을 드러내지 않으므로 정적 추론 경로만 분석했다.
- 내려받은 노트북에 실행 출력이 없고 전체 학습을 다시 돌리지는 않았으므로, 코드가 출력하도록 작성된 OOF 값과 실행 성공 여부는 독립적으로 재현하지 않았다.
- 공개 점수는 17위의 0.96602만 제목과 본문에서 확인되며, 나머지 11개는 공개 소스에 명시되지 않았다.
- 생성기 코드, 원자료와 합성 자료의 행 단위 대응, 생성 분포 검정이 없으므로 합성 자료 규칙에 관한 확정적인 결론은 내리지 않았다.
