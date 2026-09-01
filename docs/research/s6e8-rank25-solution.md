# S6E8 최종 25위 해법 공개 근거 조사

## 조사 질문과 결론

이 문서는 S6E8 최종 25위 작성자의 해법 글과 그 글이 직접 연결한 공개 자료에서 특성 계열, 단일 모형, 검증, 후보 구성, 2단 결합, 계산 자원과 재현 절차를 어디까지 확정할 수 있는지 조사한다.

결론은 세 가지다.

첫째, 작성자가 주장한 해법의 큰 구조는 600개가 넘는 특성 저장소, 다양한 단일 모형의 OOF 예측, 15겹 층화 교차검증, 400개가 넘는 후보를 입력으로 받는 Logistic Regression 결합이다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

둘째, 공개 자료는 이 구조의 일부를 실제 코드와 예측 배열로 뒷받침하지만 최종 해법을 재현하지는 못한다.
공개 입문용 학습 코드는 5겹 검증이고, 공개 2단 결합 코드는 Logistic Regression이 아니라 14개 공개 예측을 입력으로 받는 Ridge이다.
[공개 Baseline V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1), [공개 Baseline V2](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2), [공개 L2Stack V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1), [공개 예측 수집 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-datacollation-v1)

셋째, 공개 OOF 데이터셋에는 381개 후보와 15개 fold 표식이 있지만, 최종 400개가 넘는 후보 전체와 Logistic Regression 설정·계수·선택 규칙은 없다.
더구나 공개 fold 표식은 글에 적힌 정확한 `StratifiedKFold(15, random_state=42, shuffle=True)` 설정을 현재 대회 학습 자료에 다시 적용한 결과와 일치하지 않는다.
[공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1), [공개 PrivateModels 점검 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-privatemodels-v1)

따라서 이 해법에서 확정할 수 있는 것은 방향과 구성 계열이다.
최종 제출을 그대로 재현하거나 각 아이디어의 기여도를 판정할 수 있는 수준의 공개는 아니다.

## 근거 수준

이 문서는 근거를 다음처럼 나눈다.

- **작성자 주장**은 해법 글이나 그림에 명시됐지만 공개 코드로 독립 확인하지 못한 내용이다.
- **공개 자료 사실**은 연결된 노트북 소스, 메타데이터 또는 예측 배열에서 직접 확인한 내용이다.
- **우리 확인**은 공개 자료를 내려받아 스키마, hash 또는 점수를 다시 계산한 결과다.
- **해석**은 위 사실이 재현성과 비교 분석에 갖는 의미다.

순위 차이를 어느 한 기법의 인과 효과로 해석하지 않는다.

## 조사한 판본과 출처 기록

2026-09-01에 Kaggle CLI 2.2.4로 글이 직접 연결한 현재 공개 판본을 내려받았다.
공개 노트북은 실행 출력이 제거된 소스 판본이었으므로, 출력으로만 알 수 있는 점수나 실행 시간은 확정하지 않았다.

| 자료 | Kaggle 식별자 | 내려받은 소스의 SHA-256 |
| --- | --- | --- |
| 비공개 실험 예측 점검 코드 | [`ravi20076/playgrounds6e8-privatemodels-v1`](https://www.kaggle.com/code/ravi20076/playgrounds6e8-privatemodels-v1) | `ce0408a25c4fa6823e28b5e6845709fa8836bb1d832905be102085d718f716af` |
| 공개 Baseline V1 | [`ravi20076/playgrounds6e8-public-baseline-v1`](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1) | `186d26a1aba7549fd182ed89322daff43f6083d8f9275175215c7c0207d31c30` |
| 공개 Baseline V2 | [`ravi20076/playgrounds6e8-public-baseline-v2`](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2) | `eeacd159a83fd188ecdf00e78862443bcd5fdb4a1f8849ff5d6a4cfd697c56ef` |
| 공개 L2Stack V1 | [`ravi20076/playgrounds6e8-public-l2stack-v1`](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1) | `c9a6f2f1e2fd622617d99060f19aab7faa7a41d079975e134ff8ca3cb2e7f901` |
| 공개 예측 수집 코드 | [`ravi20076/playgrounds6e8-datacollation-v1`](https://www.kaggle.com/code/ravi20076/playgrounds6e8-datacollation-v1) | `c45f0e86a2b6938dbaa1d00f0a63ce031e836a0da9715acf93f9043ad6589950` |
| 공통 가져오기·학습 코드 | [`ravi20076/playgrounds6e8-public-imports-v1`](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1) | `737900653b1433481548e0b25d066dd2a1de78ca0ddd2727ab2ff1412c933df3` |
| 공개 OOF 예측 배열 | [`OOF_PredsV1_1.parquet`](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1) | `5a160acda85a29e6de96c1ec06f49e466f7d081ef69432a64fded39b26d8e5ec` |

해법 글의 연결 목록에는 위 네 노트북과 PrivateModels 데이터셋이 들어 있다.
공개 예측 수집 코드와 공통 학습 코드는 연결된 노트북의 입력 의존성이라 함께 조사했다.
[작성자 해법 글의 연결 자료](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

## 특성 생성

### 작성자 주장

작성자는 다음 여덟 계열로 600개가 넘는 특성을 만들고, 단일 모형마다 그중 30개에서 500개를 사용했다고 썼다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

- 수치 열을 범주형으로 바꾼 쌍둥이 열.
- 수치 열 1개, 2개 또는 3개의 산술 조합.
- 범주형 열을 이용한 n-gram 표적 부호화 상호작용.
- 한 열로 묶고 다른 고유값이 많은 열을 집계하는 group-by 특성.
- 모든 범주형 열의 전역 빈도 부호화.
- 반올림 열.
- 구간화 열.
- 결측값을 예측하는 보조 모형.

해법 그림은 본문보다 조금 다른 표현을 쓴다.
그림에는 범주형 쌍둥이, n-gram 수치 상호작용, n-gram 문자열 상호작용, 사인·코사인과 수학 특성, 수치 반올림, 숫자·문자·구간화, 공개 노트북 특성이 표시돼 있다.
[학습 흐름 원본 그림](https://www.kaggle.com/writeup-links/192163/images/original)

### 공개 자료 사실

공개 Baseline V1과 V2는 최종 600개 특성 저장소가 아니라 공개 입문용 특성 함수 하나를 공개한다.
이 함수에는 결측 표식과 결측 개수, 여가·화면 사용 시간의 합·차·비율, 숫자의 일의 자리·소수 첫째·둘째 자리 분해, 바닥값·소수 첫째 자리 반올림·10 나머지·소수부 구간의 범주형 쌍둥이, 활동량의 평균·표준편차·최솟값·최댓값·범위·최댓값 열, 두 범주형 열의 문자열 조합이 들어 있다.
[공개 Baseline V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1), [공개 Baseline V2](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2)

공개 Baseline은 특성 아이디어를 Don March의 공개 LightGBM 노트북에서 빌렸다고 명시한다.
연결된 원본에는 결측 표식, 화면 시간 합·차·비율, 숫자 분해, 범주형 쌍둥이, 활동 집계와 열 조합이 구현돼 있다.
[Don March의 S6E8 LightGBM](https://www.kaggle.com/code/donmarch14/s6e8-lgbm)

공개 Baseline의 `TargetEncoder`는 모형과 같은 scikit-learn `Pipeline` 안에 있고, 공개 `ModelTrainer`는 각 fold마다 그 Pipeline을 복제해 맞춘다.
따라서 이 공개 입문용 경로의 표적 부호화는 적어도 바깥 검증 fold 전체를 미리 보고 맞추는 형태는 아니다.
[공개 Baseline V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1), [공통 학습 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1)

반면 범주값의 공통 숫자화는 학습 자료와 시험 자료를 합친 뒤 수행되고, Baseline V2의 수치 평균 대체도 합친 자료에서 계산된다.
두 연산은 정답을 사용하지 않지만, 엄격한 학습 자료 전용 전처리는 아니다.
[공개 Baseline V2](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2)

### 확인 한계

공개 자료에는 600개가 넘는 최종 열 이름, 생성식, 자료형, 결측 처리, 특성 묶음별 구성, 모형별 선택 목록이 없다.

공개 Baseline에서 확인되는 특성은 작성자 주장의 일부와 방향이 같지만, 최종 특성 저장소 자체라고 볼 근거는 없다.

결측값 보조 모형, n-gram 표적 부호화, group-by 집계, 전역 빈도 부호화와 사인·코사인 특성은 최종 해법에 들어갔다는 작성자 진술만 있고 연결된 코드로 재현되지 않는다.

## 단일 모형

### 작성자 주장

작성자는 다음 모형 계열을 사용했다고 썼다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

- CatBoost, XGBoost, LightGBM을 포함한 경사 상승 결정 트리.
- PyTabKit의 RealMLP, TabM, TabResNet과 FTT.
- cuML Random Forest.
- 선형 모형.
- GNN.
- Yggdrasil.
- HistGradientBoostingClassifier.
- RepleafGBM.

작성자는 RealMLP가 가장 강한 단일 모형이었고 TabM이 근소하게 뒤따랐다고 평가했다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

해법 그림에는 본문에 없는 CNN과 AutoInt가 있고, 본문의 HistGradientBoostingClassifier는 별도 상자로 표시되지 않는다.
[학습 흐름 원본 그림](https://www.kaggle.com/writeup-links/192163/images/original)

### 공개 자료 사실

공개 Baseline V1은 XGBoost, LightGBM, CatBoost 세 모형을 같은 특성 Pipeline으로 학습한 뒤 Ridge로 한 번 더 결합한다.
[공개 Baseline V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1)

공개 Baseline V2는 `RealMLP_TD_Classifier` 하나를 `n_ens=32`, `n_epochs=3`, 은닉층 `[512, 256, 128]`로 학습한다.
[공개 Baseline V2](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2)

공개 OOF 데이터셋의 이름이 드러난 열에는 CatBoost, LightGBM, XGBoost, RealMLP, RealMLP-PLR, TabM, TabResNet, FTT, TabTransformer, AutoInt, GANDALF, GNN, CNN, FM·FFM·FwFM 계열로 보이는 식별자가 있다.
다만 381개 후보 중 222개는 `PUBLICM0`부터 `PUBLICM221`, 21개는 `PRIVATEM0`부터 `PRIVATEM20`으로 익명화돼 있어 실제 모형 계열과 설정을 복원할 수 없다.
[공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1)

### 우리 확인

공개 OOF 배열에서 정답 열과 각 후보 열의 전체 자료 ROC AUC를 다시 계산했다.

381개 중 가장 높은 열은 익명 열 `PRIVATEM20`의 `0.969218967`이었다.
이름이 드러난 열 중에는 `PTABMV1_1`이 `0.969159768`, `XGBV1_1`이 `0.969093636`, 가장 높은 이름 공개 RealMLP 열 `RMLP1CREALMLPV1_80`이 `0.968969607`이었다.
[재계산에 사용한 공개 OOF 배열](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1)

이 결과는 이름이 공개된 열만 보면 TabM이 RealMLP보다 높다는 뜻이다.
그러나 가장 높은 후보들이 익명이고 최종 후보 전체도 공개되지 않았으므로, 작성자의 "RealMLP가 전체 최강"이라는 평가를 반증하지도 입증하지도 못한다.

PrivateModels 점검 노트북은 각 열의 ROC AUC를 계산해 상위 50개를 표시하도록 작성돼 있지만 공개 소스 판본에는 실행 출력이 없다.
따라서 위 수치는 작성자가 공개한 표가 아니라 이 조사에서 공개 배열을 다시 계산한 값이다.
[공개 PrivateModels 점검 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-privatemodels-v1)

## 검증과 OOF 정합성

### 작성자 주장

작성자는 모든 모형을 `StratifiedKFold(15, random_state=42, shuffle=True)`로 학습했다고 썼다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

### 공개 자료 사실

공개 Baseline V1, Baseline V2, 공개 예측 수집 코드와 공개 L2Stack V1은 모두 `n_splits=5`, `shuffle=True`, `random_state=42`를 사용한다.
따라서 이 코드들은 글이 설명한 최종 15겹 학습 코드가 아니다.
[공개 Baseline V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1), [공개 Baseline V2](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2), [공개 예측 수집 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-datacollation-v1), [공개 L2Stack V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1)

공개 OOF 데이터셋은 691,369행이며 정답은 0이 200,895개, 1이 490,474개다.
`fold_nb`는 0부터 14까지 있고 각 fold는 46,040행에서 46,107행이므로, 15개로 층화한 것으로 보이는 표식은 실제로 존재한다.
[공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1)

### 글과 공개 자료의 불일치

공개 OOF의 정답 열은 현재 대회 `train.csv`의 정답과 행별로 완전히 일치한다.

그러나 공통 가져오기 코드가 고정한 scikit-learn 1.7.2에서 현재 대회 정답 순서에 `StratifiedKFold(n_splits=15, shuffle=True, random_state=42)`를 적용해 만든 fold 표식은 공개 OOF의 `fold_nb`와 691,369행 중 645,291행이 달랐다.
[공통 가져오기 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1), [공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1), [대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)

이 불일치의 원인은 공개 자료만으로 알 수 없다.
다른 구현, 다른 난수 상태, 중간 행 순서 변경, 공개되지 않은 fold 생성 코드 가운데 무엇 때문인지 추정하지 않는다.

따라서 "15개 fold 표식이 공개 배열에 있다"는 사실은 확정할 수 있지만, "글에 적힌 정확한 분할을 공개 자료로 재현할 수 있다"고 할 수는 없다.

후보별 OOF가 정말 같은 15개 fold에서 생성됐는지 확인하는 실행 기록, fold별 학습 행 hash와 모형별 fold 표식도 공개되지 않았다.

## 후보 구성과 선택

### 작성자 주장

작성자는 실패한 단일 모형도 다양성을 더했기 때문에 모든 모형 후보를 결합 대상으로 골랐다고 썼다.
최종 결합에는 400개가 넘는 후보 모형을 사용했다고도 썼다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

### 공개 자료 사실

공개 OOF parquet에는 정답, fold, 출처와 저장 색인 외에 결측값 없는 `float32` 후보 열 381개가 있다.
이 가운데 익명 공개 열이 222개, 익명 비공개 열이 21개, 이름이 드러난 열이 138개다.
[공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1)

후보 381개 중 16개는 최솟값이 0보다 작거나 최댓값이 1보다 크다.
따라서 공개 후보 묶음에는 확률뿐 아니라 변환된 점수 또는 범위를 벗어난 예측도 섞여 있다.
[공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1)

공개 예측 수집 코드는 14개 공개 OOF·시험 예측 원천을 합쳐 `PUBLICM0`부터 번호를 붙인다.
이 코드가 만드는 공개 묶음은 14열이므로, 222개 익명 공개 열이 들어 있는 최종 공개 데이터셋의 전체 수집 과정은 별도로 공개되지 않았다.
[공개 예측 수집 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-datacollation-v1), [공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1)

### 확인 한계

381개는 작성자가 말한 400개가 넘는 최종 후보보다 적다.

공개 자료에는 빠진 후보, 후보별 특성 묶음, 초매개변수, 씨앗, fold별 점수, 후보 상관, 다양성 기준, 포함·제외 장부와 최종 입력 열 목록이 없다.

"모든 후보를 선택했다"는 문장은 후보 생성과 최종 선택의 경계를 정의하지 않는다.
학습에 실패한 모형, OOF가 완성되지 않은 모형, 중복 후보, 결측 후보와 성능 하한을 어떻게 처리했는지도 공개되지 않았다.

## 2단 결합

### 작성자 주장

작성자는 AutoGluon은 교차검증 점수가 좋았지만 순위표 점수가 낮아 제외했고, Ridge는 교차검증과 순위표 점수가 낮아 제외했으며, 빠르고 교차검증·순위표 안정성이 좋았던 Logistic Regression을 최종 결합기로 골랐다고 썼다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

해법 그림도 여러 특성 계열과 단일 모형 계열이 Logistic Regression으로 들어가는 2단 구조를 보여 준다.
[학습 흐름 원본 그림](https://www.kaggle.com/writeup-links/192163/images/original)

### 공개 자료 사실

직접 연결된 공개 L2Stack V1은 Logistic Regression이 아니다.
14개 공개 OOF 열을 `MinMaxScaler`로 바꾼 뒤 `Ridge(max_iter=100000, random_state=42)`를 5겹으로 학습한다.
[공개 L2Stack V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1), [공개 예측 수집 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-datacollation-v1)

공통 가져오기 코드에는 Logistic Regression 도구와 별도의 다중 분류용 결합 도우미가 있지만, 최종 이진 분류 400개 이상 후보 결합을 실행하는 공개 노트북은 없다.
[공통 가져오기 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1)

### 확인 한계

최종 Logistic Regression의 입력 열 순서, 전처리, 확률·logit·순위 변환 여부, 규제 종류와 세기, 절편, class weight, solver, 반복 상한, 난수 씨앗, fold별 계수와 최종 계수는 공개되지 않았다.

메타 모형의 성능을 같은 OOF로 후보 선택하고 평가했는지, 선택과 계수 추정을 중첩 검증으로 분리했는지, 서로 다른 fold에서 생성된 공개 OOF를 어떻게 정렬하고 검증했는지도 공개되지 않았다.

따라서 2단 Logistic Regression이라는 결합기 종류는 작성자 주장으로 확정할 수 있지만, 누출 방지 절차와 최종 계수를 사실로 복원할 수는 없다.

## 계산 자원

작성자는 XGBoost·LightGBM·RealMLP와 나머지 모형에 Colab L4, CatBoost에 Colab A100, TabM에 Runpod A100 80GB를 사용했다고 썼다.
[작성자 해법 글](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach)

연결된 공개 Baseline 메타데이터는 Kaggle T4 실행 설정을 담고 있어 최종 해법 글의 Colab·Runpod 자원 장부와 다르다.
[공개 Baseline V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1), [공개 Baseline V2](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2)

공개 자료에는 각 모형의 실제 실행 시간, GPU 시간, 메모리 최고점, 재시도 횟수, 전체 비용과 병렬 실행 구조가 없다.

따라서 GPU 종류는 작성자 진술로 기록할 수 있지만 "중간 정도로 빠른 학습 흐름"이라는 평가는 정량 확인할 수 없다.

## 재현 가능성 판정

| 구성 요소 | 판정 | 이유 |
| --- | --- | --- |
| 해법의 큰 구조 | 부분 재현 가능 | 특성 계열, 모형 계열, 15겹, Logistic Regression이라는 방향은 글과 그림에 있다. |
| 공개 입문용 특성·GBDT·RealMLP | 소스 수준 재현 가능 | Baseline V1·V2와 공통 학습 코드가 공개돼 있다. 입력 의존성 중 `playgrounds6e8-data-v1`은 접근할 수 없어 그대로 실행하는 데는 추가 자료가 필요하다. |
| 600개 이상 최종 특성 저장소 | 재현 불가 | 전체 열과 생성 코드, 모형별 부분집합이 없다. |
| 최종 15겹 분할 | 재현 불가 | 공개 OOF에는 15개 표식이 있지만 글의 정확한 splitter를 다시 적용한 결과와 다르다. |
| 400개 이상 최종 후보 | 재현 불가 | 공개 OOF에는 381개만 있고 243개가 익명이다. |
| 최종 Logistic Regression | 재현 불가 | 공개 결합 코드는 Ridge이며 최종 설정과 계수가 없다. |
| 누출 방지와 선택 판정 | 감사 불가 | 중첩 선택, fold 정합성, 행 hash와 후보 장부가 없다. |
| 최종 제출 | 재현 불가 | 최종 실행 명령, 고정 환경, 전체 입력, test 후보 배열과 제출 manifest가 한 묶음으로 공개되지 않았다. |

재현에 필요한 최소 추가 자료는 다음과 같다.

- 원자료 hash와 691,369개 행에 대한 최종 fold 표식 생성 코드.
- 600개 이상 특성의 명세와 생성 코드.
- 후보마다 모형 계열, 특성 묶음, 초매개변수, 씨앗, 학습 fold와 OOF·시험 예측 hash를 잇는 장부.
- 최종 400개 이상 입력 열의 순서와 후보 선택 판정.
- Logistic Regression 전처리, 설정, fold별 계수, 최종 계수와 평가 절차.
- 고정 패키지 환경, 실행 순서, 계산 자원과 최종 제출 hash.

## 라이선스와 출처 구분

Kaggle 공개 노트북 소스는 공개 시 Apache License 2.0 조건으로 제공된다.
이 조사는 노트북 코드를 저장소에 복사하지 않고 동작을 요약했으며, 내려받은 원본과 SHA-256만 조사 기록에 남겼다.
[Kaggle 공식 Meta Kaggle Code 설명](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code), [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

PrivateModels 데이터셋 메타데이터에는 MIT가 표시돼 있다.
다만 데이터셋 안의 익명 `PUBLICM*` 예측 배열은 원래 공개 노트북과 작성자를 역추적할 수 없으므로, 데이터셋의 표시만으로 각 제3자 배열의 재배포 조건과 출처 표기가 완전하다고 판정하지 않는다.
[공개 PrivateModels 데이터셋](https://www.kaggle.com/datasets/ravi20076/playgrounds6e8privatemodelsv1)

대회 `train.csv`, `test.csv`와 `sample_submission.csv`의 사용 조건은 공개 노트북의 Apache License 2.0이나 예측 데이터셋의 MIT에서 파생되지 않는다.
대회 규칙과 대회 자료 조건을 별도로 따라야 한다.
[S6E8 대회 규칙](https://www.kaggle.com/competitions/playground-series-s6e8/rules), [S6E8 대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)

공통 가져오기 코드는 scikit-learn 1.7.2, XGBoost 3.3.0, CatBoost 1.2.10, LightGBM 4.7.0, Polars 1.38.1과 판본 미고정 pytabkit을 설치한다.
이 패키지들의 조건은 노트북 소스 조건과 별개이며, 대표적으로 scikit-learn은 BSD-3-Clause, XGBoost와 CatBoost는 Apache-2.0, LightGBM은 MIT를 사용한다.
[공통 가져오기 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1), [scikit-learn](https://github.com/scikit-learn/scikit-learn), [XGBoost](https://github.com/dmlc/xgboost), [CatBoost](https://github.com/catboost/catboost/blob/master/LICENSE), [LightGBM](https://github.com/lightgbm-org/LightGBM)

연결 노트북 메타데이터에는 사전 학습 모형 입력이 없다.
RealMLP와 TabM은 설치 패키지에서 학습하는 모형이며 별도의 공개 가중치를 불러오는 근거가 없다.
[공개 Baseline V2](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v2), [공통 가져오기 코드](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-imports-v1)

이 조사에서 내려받은 대회 자료와 예측 배열은 임시 조사 경로에만 두었고 저장소에 커밋하지 않는다.

## 다음 비교 티켓에 넘길 판정 재료

25위 해법에서 우리 14위 해법과 비교할 가치가 있는 축은 "600개"나 "400개"라는 수량 자체가 아니다.

비교할 수 있는 실질적 후보는 다음이다.

- 명시적인 특성 저장소와 모형별 특성 부분집합 운용.
- RealMLP·TabM·TabResNet·FTT와 결정 트리 밖 모형 계열의 다양성.
- 약한 후보를 포함할 때 단일 점수가 아니라 OOF 다양성을 근거로 삼는 절차.
- 15겹 검증의 계산량과 안정성, 그리고 정확한 fold 정합성 기록.
- 수백 개 OOF 입력을 받는 규제 Logistic Regression과 엄격한 중첩 선별.

이 자료만으로는 어느 축도 순위 상승의 원인이라고 말할 수 없다.
특히 최종 후보 장부와 중첩 선별 근거가 없으므로 "약한 모형을 모두 넣는다"를 일반 규칙으로 채택해서는 안 된다.

새 조사 티켓은 만들지 않는다.
공개 자료가 남긴 미공개 항목은 질문으로는 명확하지만 작성자의 비공개 코드나 추가 설명 없이는 더 조사할 1차 자료가 없고, 지도 목적지는 공개되지 않은 설정의 추정을 명시적으로 제외한다.
특성 계열과 재사용 가치는 이미 후속 티켓인 [25위 해법과 우리 14위 해법의 차이와 재사용 가치를 판정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/559)가 다룬다.
