# Playground Series S4E7부터 S4E4까지 상위 10위 해법과 댓글 조사

이 문서는 GitHub 이슈 [리서치: S4E7-S4E4 상위권 해법 글과 댓글 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/473)의 조사 결과다.
조사 기준일은 2026-08-28이다.

## 결론

네 대회의 공식 해법 범주에 보이는 글 23개를 모두 확인했고, 제목이나 상세 본문에 Private 최종 1위부터 10위가 명시된 글 19개를 포함했다.
S4E7은 1, 2, 3, 4, 6, 8, 9위, S4E6은 6, 10위, S4E5는 1위부터 4위, S4E4는 1, 2, 3, 4, 5, 8위 글을 포함했다.
누락 순위는 다른 글로 채우지 않았다.

포함 글의 목록 화면 댓글 표시값 합계는 364개였다.
상세 화면 머리말의 일반 댓글 표시값 합계는 333개였고 감사 댓글 표시값 합계는 11개였다.
상위 10위 밖의 제외 글까지 포함한 공식 목록 전체의 댓글 표시값은 378개였다.
모든 포함 글에서 댓글 영역을 끝까지 확인하고 접힌 답글을 펼쳤으며 삭제 댓글 11개는 내용에 접근할 수 없어 삭제 사실만 기록했다.

이번 구간에서 현재 S6E8의 새 후보를 보강하는 방법은 잡음 제거 자동부호화(DAE)의 낮은 차원 잠재 표현 하나다.
S4E7 3위는 8차원 잠재 표현을 CatBoost와 embedding 신경망에 붙여 약 0.0002의 구성 요소 이득을 보고했다.
이는 기존 1년 조사의 S6E3 원본 전용 DAE와 S5E6 지도형 자동부호화에 독립 대회 근거를 더하고, 자체 35개와 외부 278개에는 같은 구현이 없다.
반면 S4E5 3위는 자동부호화 잠재 특성이 도움 되지 않았다고 보고했고 S4E7 글은 DAE를 fold마다 다시 맞췄는지 설명하지 않았다.
따라서 한 개의 누출 없는 고정 5-fold DAE만 새 후보로 사전 고정하고 단독 OOF, 기존 구성원과의 순위 상관 및 313개 결합 한계 기여를 판정하는 것이 맞다.

나머지 강한 반복은 현재 결정과 중복이다.
다양한 XGBoost, LightGBM, CatBoost, 신경망과 AutoML의 OOF 수집, 원본 자료 관점, 선형 및 탐욕 및 비선형 결합, 같은 fold와 순위 변환은 자체 35개 또는 외부 278개 확장 결합이 이미 직접 또는 인접하게 덮는다.
S4E6 댓글은 작은 결합과 큰 결합의 상반 사례를 모두 보여 모형 수가 아니라 현재 nested OOF의 포함 및 제외 기여가 기준이어야 함을 강화했다.
S4E5와 S4E4의 OpenFE, 행 집계, RMSLE 손실, 홍수 목표 변환, Abalone 도메인 특성은 회귀 자료 구조에 의존하거나 독립 재현이 반박해 현재 ROC AUC 과제에 옮기지 않는다.
원본 자료 목표 반전, OOF 없는 공개 제출의 Public 가중치, Public 기반 맹목 혼합도 현행 채택 계약에 맞지 않는다.

현재 비교 기준은 자체 후보 풀 35개와 외부 구성원 278개를 더한 313개 확장 결합이다.
자체 35개 풀의 최고 결합은 shrunk_rank_logit_logistic, nested OOF 0.9698105828이다.
최신 확장 결합 실행 443b3a71a2b045ba9052fbb3d821255d는 같은 전략으로 nested OOF 0.9703509469와 가중 OOF 0.9712170271을 기록했다.
Public 0.97135는 사후 참고값이며 후보 판정에는 사용하지 않는다.

## 조사 범위와 방법

Kaggle API, 웹 검색 결과 요약, 직접 HTTP 요청과 리더보드 조회는 사용하지 않았다.
agent-browser의 격리된 이름 세션으로 각 대회의 실제 Kaggle 공식 competitionWriteUps 화면과 상세 글을 직접 확인했다.
각 조사는 사용자가 지정한 [공식 해법 범주 화면](https://www.kaggle.com/competitions/playground-series-s4e7/discussion?category=competitionWriteUps&sort=recent-comments)에서 대회 slug만 바꾸어 시작했다.
최근 댓글순은 발견 순서로만 사용했고 표본 선정에는 쓰지 않았다.
순위는 글 제목 또는 상세 화면의 Solution Writeup · Nth place 표기만 사용했다.
상위 10위 글이 없는 순위는 다른 글로 채우지 않았다.

포함 글마다 본문, 작성자와 순위 표시, 검증 설명, 특성, 모형, 결합, 보고 점수, 계산 자원, 외부 자료, 일반 댓글, 접힌 답글과 감사 댓글을 확인했다.
댓글 영역을 끝까지 내려 지연 표시되는 댓글을 모두 불러왔고 more replies로 접힌 답글도 모두 펼쳤다.
화면에 없는 팀 구성, 점수, 설정과 제거 기여는 추정하지 않고 미보고로 남겼다.

Kaggle 목록의 댓글 수와 상세 화면 머리말의 일반 댓글 및 감사 댓글 수는 여러 글에서 서로 일치하지 않았다.
목록 수에는 접힌 답글, 삭제 댓글과 댓글 종류가 다르게 반영되는 것으로 보이지만 Kaggle 화면은 집계 규칙을 설명하지 않는다.
따라서 아래 장부는 두 화면의 원시 표시값을 따로 기록한다.

## 표본 장부

| 대회 | 과제와 평가지표 | 공식 해법 글 | 포함 순위 | 결측 순위 | 목록 댓글 합 | 상세 일반 댓글 | 감사 댓글 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| S4E7 | 보험 교차 판매 이진 분류, ROC AUC | 7 | 1, 2, 3, 4, 6, 8, 9 | 5, 7, 10 | 174 | 160 | 4 |
| S4E6 | 학생 학업 결과 다중 분류, Accuracy | 6 | 6, 10 | 1, 2, 3, 4, 5, 7, 8, 9 | 29 | 27 | 1 |
| S4E5 | 홍수 확률 회귀, R2 | 4 | 1, 2, 3, 4 | 5, 6, 7, 8, 9, 10 | 96 | 87 | 1 |
| S4E4 | 전복 고리 수 회귀, RMSLE | 6 | 1, 2, 3, 4, 5, 8 | 6, 7, 9, 10 | 65 | 59 | 5 |
| 합계 |  | 23 | 19개 글 |  | 364 | 333 | 11 |

## 제외와 결측 장부

S4E7의 공식 글 7개는 모두 상위 10위 조건을 충족했고 5, 7, 10위 글은 없었다.

S4E6에서는 131위, Private 61위, 118위와 52위 글을 제외했다.
이 네 제외 글의 목록 댓글 표시값은 각각 0, 10, 4, 0으로 합계 14개였다.
포함 순위는 6위와 10위뿐이고 1위부터 5위, 7위부터 9위는 결측이다.

S4E5의 공식 글 4개는 모두 포함 조건을 충족했고 5위부터 10위 글은 없었다.

S4E4의 공식 글 6개는 모두 포함 조건을 충족했고 6, 7, 9, 10위 글은 없었다.

팀원들이 같은 해법을 별도 글로 중복 게시한 사례와 접근할 수 없는 포함 글은 없었다.
S4E7에서 삭제 댓글 7개, S4E5 1위에서 2개, S4E4 1위에서 2개를 확인했지만 삭제된 내용은 복구하지 않았다.

## S4E7: Binary Classification of Insurance Cross Selling

과제는 자동차 보험 교차 판매에 응답할 고객을 예측하는 이진 분류이고 [공식 평가 지표](https://www.kaggle.com/competitions/playground-series-s4e7/overview/evaluation)는 ROC AUC다.
공식 해법 범주에는 글 7개가 있었고 1, 2, 3, 4, 6, 8, 9위 글이 모두 포함 조건을 충족했다.
5, 7, 10위는 결측이며 제외 글이나 같은 팀원의 중복 글은 없었다.
목록 댓글 합은 174개, 상세 일반 댓글 합은 160개, 감사 댓글 합은 4개였다.

### 1위: Ravi Ramakrishnan과 Minato Namikaze

[Winning approach - Team Cross Sellers](https://www.kaggle.com/competitions/playground-series-s4e7/writeups/cross-sellers-winning-approach-team-cross-sellers)는 Ravi Ramakrishnan과 Minato Namikaze 팀의 1위 글이다.
목록에는 댓글 97개가 표시됐고 상세 화면에는 일반 댓글 91개와 감사 댓글 3개가 표시됐다.
삭제 댓글 세 개는 내용에 접근할 수 없어 삭제 사실만 확인했다.

훈련 자료만 쓴 판, 훈련과 보조 원본 자료를 합친 판, 각 fold 학습 부분에 원본 전체를 넣은 판을 모두 만들었고 마지막 방식이 CV를 가장 많이 올렸다고 보고했다.
검증은 StratifiedKFold 5-fold, shuffle=True, random_state=42로 통일했다.
CatBoost 한 fold의 순열 중요도로 특성 묶음을 먼저 거르고 12개 판본의 중앙 특성 저장소를 만들어 실험마다 재사용했다.

1단과 2단에는 수동 LightGBM, LAMA LightGBM, CatBoost, DenseLight 및 MLP, TabResNet, TabTransformer와 AutoInt가 들어갔다.
전체 행 모형 외에 Previously_Insured 두 구간, Vehicle_Damage 두 구간과 두 열을 함께 나눈 네 구간별 전문 모형을 각각 5-fold로 만들었다.
나무 1단을 신경망 2단에, 신경망 1단을 나무 2단에 주는 교차 계열 스택을 사용했다.
최종 XGBoost 3단은 선택한 단일 모형과 2단 예측 78개를 읽었고 전체 실험은 125개가 넘었다.
CatBoost가 최고 단일 모형이었고 LightGBM과 여러 신경망이 최종 다양성을 더했다.
XGBoost 1단, CatBoost 2단, 선형 결합, Optuna, 탐욕 결합, 조화 평균과 기하 평균은 실패했다고 적었다.

보조 자료는 공개 health-insurance-cross-sell-prediction-data였고 공개 노트북 OOF 한 개도 다양성 구성원으로 썼다.
원본과 시험 자료의 중복 및 목표 반전 규칙을 찾아 모든 제출에 후처리했다.
본문은 CV, Public과 Private의 수치를 따로 보고하지 않았다.
계산에는 A6000 두 장, A6000 Ada 두 장, A100, A5000 두 장, RTX 4090과 RTX 3090을 사용했고 작성자는 한 달 동안 하루 3시간에서 4시간을 들였다고 댓글에서 답했다.

댓글에서 전문 모형은 해당 표시 열 하나만 쓰는 모형이 아니라 그 표시값으로 행을 나눈 뒤 나머지 특성을 모두 쓰는 별도 모형이라고 명확히 했다.
Previously_Insured와 Vehicle_Damage를 각각 또는 함께 나눈 구간마다 원래 5-fold 관계를 유지했다고 답했다.
불균형 처리는 1,100만 행의 큰 자료라 별도로 하지 않았다고 했다.
특성 중요도는 한 fold의 순열 중요도였고 공개 OOF는 다른 오차를 얻기 위한 한 구성원이었다.
전체 구조 선택은 이론적 최적화보다 대규모 시행착오와 운이 컸다고 답했다.

S6E8 판단은 대부분 기존 결정과 중복 또는 현재 과제에 부적합이다.
행 구간별 전문 모형과 비선형 2단은 현재 자료에서 이미 nested OOF 음성 결과가 있고, 외부 278개와 현재 결합은 여러 모형 계열을 더 넓게 덮는다.
원본 행 주입과 목표 반전은 S6E8 자체 대조에서 기각됐고 과거 보험 자료의 중복 규칙을 옮길 수 없다.
같은 fold, OOF 및 시험 예측 저장, 공급원별 구성원 선택과 실험 계보 정리는 현재 장부 규율을 유지할 운영 근거다.

### 2위: Ujjwal Pandey

[2nd Place Solution, One model is all you need](https://www.kaggle.com/competitions/playground-series-s4e7/writeups/ujjwal-pandey-2nd-place-solution-one-model-is-all-)는 Ujjwal Pandey의 2위 글이다.
목록과 상세 화면 모두 댓글 15개가 표시됐다.

초기 XGBoost는 CV 0.8833, Public 0.88448이었고 조정한 XGBoost는 0.89113과 0.89387이었다.
CPU LightGBM은 CV 0.89302, Public 0.89344였고 SnapML은 CV 0.890을 넘지 못했다.
TabNet과 GANDALF는 CV 약 0.8910이었고 계산량 대비 GBDT를 이기지 못했다.
원본 자료를 합치고 Previously_Insured와 Vehicle_Damage, Vehicle_Age, Driving_License 및 Gender의 범주 결합을 추가했다.

최종 중심 모형은 CatBoost 한 개였다.
약 50회 분산 Optuna 탐색 뒤 학습률을 0.085로 낮추고 반복 수를 10,000으로 늘렸으며 Gradient 계열보다 NewtonCosine 또는 NewtonL2 점수 함수와 leaf_estimation_iterations 12가 더 좋았다고 보고했다.
충돌 중복을 원본에서 먼저 제거하고 합친 자료에서도 제거했으며 Age와 Annual_Premium을 구간화했다.
주요 fold CV는 0.89584에서 0.89625였고 시험 점수는 0.89620에서 0.89632였다.
작은 CatBoost 여러 개를 다른 분할과 난수로 평균한 뒤 목표 반전 후처리를 적용해 최종 Public 0.89788과 Private 0.89753을 보고했다.

신경망 잠재 표현은 처음 CV 0.895를 보였지만 난수 불일치로 생긴 오류였고 메모리 약 300차원을 써 계산 자원만 소모했다고 반례를 남겼다.
목표 부호화는 XGBoost와 LightGBM을 개선하지 못했고 CatBoost의 max_ctr_complexity를 기본보다 늘리면 과적합했다고 적었다.
4-fold 주요 모형 하나를 학습하면 약 48GB 메모리가 필요했고 RTX 4070, Kaggle P100 두 장, Colab L4와 여러 TPU를 분산 탐색에 사용했다.

댓글은 대부분 축하였고 작성자는 공개 노트북을 볼 때 리더보드 점수가 아니라 CV를 자신의 결과와 비교해 채택 여부를 정한다고 답했다.
다른 해법의 결합 결과만 복사한 노트북은 학습 가치가 낮아 피한다고 설명했다.

S6E8 판단은 기존 결정과 중복 또는 현재 과제에 부적합이다.
CatBoost, 정확값 범주 결합, 구간화와 낮은 학습률 및 긴 학습은 자체와 외부 풀에 이미 있고 현재 반복 수는 고정 nested 판정으로 결정한다.
목표 반전과 과거 자료 중복 제거 규칙은 현재 자료에 적용하지 않는다.
잠재 표현의 가짜 개선이 난수 불일치에서 생겼다는 사례는 자동부호화 후보를 평가할 때 fold와 난수 계보를 고정해야 한다는 반례다.

### 3위: Tilii

[#3 solution, Many individual models and many ensembles](https://www.kaggle.com/competitions/playground-series-s4e7/writeups/tilii-3-solution-many-individual-models-and-many-e)는 Tilii의 3위 글이다.
목록과 상세 화면 모두 댓글 11개가 표시됐다.
삭제 댓글 한 개는 내용에 접근할 수 없었다.

모든 열을 범주형으로 보고 Annual_Premium의 55,068개 값도 줄이지 않은 채 OrdinalEncoder로 처리했다.
최고 단일 모형은 CatBoost였고 CV 0.896733, Public 0.89728, Private 0.89699를 보고했다.
Keras Factorization Machine은 0.894276, 0.89527, 0.89498, Keras embedding은 0.894192, 0.89469, 0.89445였다.
xLearn Field-aware Factorization Machine은 0.893223, 0.89447, 0.89414였고 LAMA ResNet은 0.893647, 0.89378, 0.89359였다.

Factorization Machine은 각 열의 embedding 벡터를 모든 쌍의 내적으로 교차하고 선형 표현 및 밀집층과 합쳤다.
Field-aware Factorization Machine은 CatBoost보다 0 예측에 강하고 CatBoost는 1 예측에 강해 두 모형의 분포가 서로 보완적이었다.
모든 범주를 one-hot으로 바꾸되 Age, Annual_Premium과 Vintage는 0부터 1로 눈금을 맞춘 수치로 둔 잡음 제거 자동부호화도 학습했다.
병목 크기 3과 8 가운데 8차원 잠재 표현이 더 좋았고 그 표현을 CatBoost와 Keras embedding에 붙이면 약 0.0002가 올랐다고 보고했다.

최종은 38개 모형을 LAMA DenseLight 신경망으로 결합했고 LightGBM 2단도 거의 같은 결과였다.
구성에는 CatBoost 최소 8개, xLearn FM 및 FFM 각 6개에서 8개, Keras FM, Keras embedding, LAMA 신경망 세 개와 AutoGluon 두 개가 포함됐다.
OOF를 결합하기 전에 예측을 순위로 바꿔 서로 다른 눈금을 맞췄다.

댓글에서 AUC는 순위 지표이므로 순위 변환 뒤 Ridge와 Lasso도 결합기로 사용할 수 있다고 설명했다.
libFFM 실행 파일은 xLearn보다 빨랐지만 AUC 지표와 잘 맞지 않았다고 답했다.
전체 파이프라인은 수백 개의 분리된 스크립트로 만들어 완전한 재현 코드는 제공하기 어렵다고 밝혔다.

Factorization Machine과 순위 결합은 기존 결정과 중복이다.
외부 278개에는 Factorization Machine 다섯 개와 deepfm_exact가 있고 최신 313개도 순위 로짓 결합을 사용한다.

잡음 제거 자동부호화 잠재 표현은 새 후보를 보강한다.
자체 35개와 외부 278개의 명시적 구성원에는 같은 구현이 없고, 이 글은 CatBoost와 Keras embedding에서 약 0.0002의 구성 요소 이득을 따로 보고했다.
S6E3 1위의 원본 전용 DAE 및 S5E6 2위의 지도형 자동부호화 잠재 표현과 다른 대회에서 반복된 독립 근거이므로 이슈 307의 근거 부족 판단보다 강해졌다.
다만 이 글은 자동부호화를 fold마다 다시 맞췄는지 설명하지 않았고 완전한 코드도 없으므로 누출 없는 5-fold 구현 하나로만 진입 대조를 해야 한다.

### 4위: Optimistix

[4th place solution: Competing Without Compute](https://www.kaggle.com/competitions/playground-series-s4e7/writeups/optimistix-4th-place-solution-competing-without-co)는 Optimistix의 4위 글이다.
목록에는 댓글 16개가 표시됐고 상세 화면에는 일반 댓글 13개가 표시됐다.

CatBoost 단일 CV 범위는 0.8940에서 0.8950, LightGBM은 0.8920에서 0.8930, 신경망은 0.8910에서 0.8930, XGBoost는 0.8910에서 0.8920이었다.
공개 노트북을 고쳐 OOF와 시험 예측을 저장하고 모형, 설정, fold 수와 난수를 달리해 약 80개 후보를 모았다.
RidgeClassifier가 탐욕 결합과 재귀 제거보다 훨씬 빨라 주 결합기로 사용했다.
약 70개에서 Ridge 가중치가 매우 작은 구성원을 빼도 CV가 거의 같아 60개 안팎으로 줄였다.

튜닝하지 않은 XGBoost, CatBoost와 LightGBM 2단을 섞으면 Ridge와 같은 약 0.89771이었고 두 결합의 평균은 0.89776이었다.
다른 공개 제출까지 다시 섞어 약 0.89780을 만들었지만 이 수치는 본문이 Public과 Private 중 어느 쪽인지 명시하지 않았다.
조정한 GBDT 2단의 단독 점수는 올랐지만 전체 결합은 개선되지 않았다.
목표 반전 후처리는 약 0.89694에서 0.89751로 0.00057을 올렸다고 보고했다.
계산은 Kaggle GPU만 사용했고 큰 OOF 행렬과 12시간 제한 때문에 여러 번 실패했다고 적었다.

댓글에서 이전 S4E6에서는 Public 1위 유지에 과적합해 최종 113위가 됐고 이번에는 CV를 따랐다고 설명했다.
핵심 방법을 바꾸는 독립 재현이나 제거 대조는 없었다.

S6E8 판단은 기존 결정과 중복이다.
대규모 외부 OOF, Ridge와 GBDT 결합, 작은 가중치 구성원 제거는 현재 313개 사다리와 nested 구성원 절제가 더 엄격하게 다룬다.
Public 제출을 직접 섞는 경로와 목표 반전은 현행 계보 및 채택 계약에 맞지 않는다.

### 6위: Mahdi Ravaghi

[6th place solution](https://www.kaggle.com/competitions/playground-series-s4e7/writeups/mahdi-ravaghi-6th-place-solution)은 Mahdi Ravaghi의 6위 글이다.
목록에는 댓글 12개가 표시됐고 상세 화면에는 일반 댓글 11개가 표시됐다.
삭제 댓글 한 개는 내용에 접근할 수 없었다.

원본 자료를 모든 모형 학습에 추가하고 자료형을 줄인 뒤 Previously_Insured와 Annual_Premium, Vehicle_Age, Vehicle_Damage 및 Vintage의 결합 범주를 만들었다.
작성자는 결합 범주의 누출 위험을 걱정했지만 CV와 Public이 함께 올라 사용했다고 적었다.
CatBoost, LightGBM, XGBoost, 신경망과 로지스틱 회귀의 OOF와 시험 예측을 저장했다.
CatBoost와 로지스틱 회귀는 모든 열을 범주형으로, 신경망은 일부 one-hot과 목표 부호화 하나로 처리했다.

기본 LogisticRegression을 최종 추정기로 쓰는 StackingClassifier를 사용했다.
OOF를 로그 변환해 입력했고 조정한 LogisticRegression, XGBoost와 LightGBM 2단은 더 좋지 않았다.
목표 반전 후처리는 약 0.0006을 더했다.
본문 표가 그림으로 렌더링돼 모형별 CV, Public과 Private 수치는 텍스트에서 확인할 수 없었다.

댓글에서 Tilii는 AUC라면 로그보다 순위 변환이 서로 다른 예측 눈금을 맞추는 데 적합하다고 조언했고 작성자는 다음에 쓰겠다고 답했다.
나머지 댓글은 코드 재사용과 축하가 중심이었다.

S6E8 판단은 기존 결정과 중복이다.
선형 2단, 범주 결합, 원본 자료와 여러 GBDT 및 신경망은 현재 자체 및 외부 풀이 더 넓게 덮는다.
로그 변환보다 순위 결합이 낫다는 보충은 현재 shrunk_rank_logit_logistic 선택과 같은 방향이다.

### 8위: Yosef Lachman, Moshe Grama와 Shmuel Asher

[8th Place Solution](https://www.kaggle.com/competitions/playground-series-s4e7/writeups/technology-management-biu-8th-place-solution)은 Yosef Lachman, Moshe Grama와 Shmuel Asher 팀의 8위 글이다.
목록에는 댓글 11개가 표시됐고 상세 화면에는 일반 댓글 10개와 감사 댓글 1개가 표시됐다.
삭제 댓글 두 개는 내용에 접근할 수 없었다.

깊이 8, 학습률 0.02, 최대 30,000회, Newton 잎 추정, Bernoulli 표본추출의 단일 CatBoost를 사용했다.
StratifiedKFold 10-fold의 fold별 최선 반복은 14,044회에서 19,431회였고 fold AUC는 0.895431에서 0.896413이었다.
평균 CV는 0.895693이고 전체 학습은 약 15시간이었다.
RTX 4060과 대학에서 제공한 A100을 사용했다.
목표 반전 후처리도 적용했지만 Public과 Private 수치는 보고하지 않았다.

댓글에서 무작위 탐색과 Optuna를 포함해 95개 넘는 설정을 제출했고 CatBoost 단일 모형이 충분히 강해 결합하지 않았다고 답했다.
원본 열 외에 Previously_Insured와 Annual_Premium, Vehicle_Age, Vehicle_Damage 및 Vintage의 결합 범주를 사용했다.
낮은 학습률과 많은 반복이 미세한 규칙을 과적합 없이 잡는 데 도움이 됐지만 50,000회는 계산 시간이 매우 길 것이라고 경고했다.

S6E8 판단은 기존 결정과 중복이다.
CatBoost 정확값 범주, 결합 범주와 고정 반복 수 대조는 자체 풀에 있고 외부 278개에도 여러 CatBoost가 있다.
한 강한 단일 모형의 성공은 현재 넓은 결합을 줄일 근거가 아니다.

### 9위: Oscar Aguilar

[#9 Solution, 24 Models and Hill Climbing](https://www.kaggle.com/competitions/playground-series-s4e7/writeups/oscar-aguilar-9-solution-24-models-hill-climbing)은 Oscar Aguilar의 9위 글이다.
목록에는 댓글 12개가 표시됐고 상세 화면에는 일반 댓글 9개가 표시됐다.

훈련과 원본 자료를 모두 사용하고 훈련, 시험과 원본을 목표 없이 합쳐 여덟 개 결합 범주를 factorize했다.
CatBoost, LightGBM, TensorFlow, XGBoost와 LightGBM Random Forest를 각각 10-fold로 학습했다.
원본 추가 전후 CV는 CatBoost 0.895311에서 0.895819, LightGBM 0.892605에서 0.892753, TensorFlow 0.892083에서 0.892213이었다.
XGBoost는 0.890959에서 0.89105, Random Forest는 0.873091에서 0.875128이었다.

원본 포함 및 제외 판을 모두 합친 24개 모형을 탐욕 결합했다.
구성은 CatBoost 여덟 개, LightGBM 여섯 개, TensorFlow 여섯 개, XGBoost 두 개와 Random Forest 두 개였다.
최종 예측에는 목표 반전 후처리를 적용했고 Public과 Private 수치는 본문에 없다.

댓글에서 같은 구성원을 0.01 가중치 간격으로 비교한 Tilii의 독립 반례가 제시됐다.
최고 단일은 CV 0.896692와 Private 0.89699, 탐욕 결합은 0.897212와 0.89735, 일반 스택은 0.897386과 0.89748이었다.
같은 8,149개 목표 반전을 세 판에 적용했으므로 이 비교에서는 스택이 탐욕 결합보다 CV 약 0.000174와 Private 0.00013 높았다.
61개 모형과 1,100만 행에서는 탐욕 결합 자체가 매우 느리다는 반례도 있었다.

S6E8 판단은 기존 결정과 중복이다.
현재 313개는 순위 공간의 학습형 로지스틱 결합을 nested OOF로 선택했고 단순 탐욕 결합을 이미 비교했다.
원본 포함 및 제외를 다양성 축으로 쓰는 방법도 자체 원본 관점과 외부 OOF에 인접하며 원본 행 주입 자체는 기각됐다.

## S4E6: Classification with an Academic Success Dataset

과제는 학생의 학업 결과를 세 클래스로 예측하는 다중 분류이고 [공식 평가 지표](https://www.kaggle.com/competitions/playground-series-s4e6/overview/evaluation)는 Accuracy다.
공식 해법 글 6개 가운데 6위와 10위 글만 포함 조건을 충족했다.
포함 글의 목록 댓글 합은 29개, 상세 일반 댓글 합은 27개, 감사 댓글 합은 1개였다.

### 6위: Matt OP

[6th Place Solution: Many model ensembles were detrimental?](https://www.kaggle.com/competitions/playground-series-s4e6/writeups/matt-op-6th-place-solution-many-model-ensembles-we)는 Matt OP의 6위 글이다.
목록에는 댓글 23개가 표시됐고 상세 화면에는 일반 댓글 21개와 감사 댓글 1개가 표시됐다.

원본 자료를 포함한 XGBoost와 LightGBM을 StratifiedKFold 5-fold로 학습했다.
Optuna로 설정을 탐색하고 CV 기반 재귀 특성 제거로 결합에 넣을 모형을 골랐다.
최종 시험 예측은 원본 자료를 포함한 전체 자료로 다시 학습했다고 댓글에서 명확히 했다.
본문은 CV, Public, Private, 최종 가중치, 하드웨어와 계산량을 보고하지 않았다.
공개 노트북의 재귀 선택과 결합 아이디어에 의존했다.

본문은 작은 결합이나 단일 모형이 상대적으로 강해 보였다고 회고했지만 댓글은 양방향 반례를 남겼다.
Sergei는 4개에서 6개 스택이 미조정 XGBoost를 이기지 못했고 Anupam은 단일 LightGBM이 8모형 결합보다 Private이 높았다고 했다.
CatBoost는 Public이 높아도 Private이 낮았고 Shivam과 Khadidja도 특성 생성이 약해진 사례를 보탰다.
Optimistix의 미제출 14모형은 CV 0.83568, Private 0.83959로 3위 수준이었지만 18모형 탐욕 결합은 CV 0.835359, Public 0.83552, Private 0.83849로 8위에서 9위 수준이었다.

반대로 John Doe는 최고 단일 CV 0.83471에서 30모형 결합 CV 0.83571, Public 0.8359, Private 0.83903으로 올랐다고 보고했다.
Tilii도 여섯 기초 모형이 0.832131에서 0.835529인데 여섯 모형 결합은 0.836134였다고 반박했다.
훈련 Accuracy가 84%에서 95%로 올랐다는 댓글에는 Tilii가 훈련 과적합 가능성을 지적하고 실제 Private 결과를 요구했다.

S6E8 판단은 기존 결정과 중복이다.
XGBoost, LightGBM, 원본 자료, Optuna, 재귀 선택과 다양한 OOF 결합은 자체 35개 또는 외부 278개가 이미 덮는다.
상반된 댓글은 적은 모형이나 많은 모형 자체가 답이 아니라 현재 nested OOF의 포함 및 제외 기여와 상보성이 기준이어야 함을 강화한다.
Public과 Private 역전 사례도 Public을 판정에서 제외하는 현행 계약과 같은 방향이다.

### 10위: lash_fire

[10th Place Solution](https://www.kaggle.com/competitions/playground-series-s4e6/writeups/lash-fire-10th-place-solution)은 lash_fire의 10위 글이다.
목록과 상세 화면 모두 댓글 6개가 표시됐다.

조정한 LightGBM과 XGBoost를 각각 BaggingClassifier의 기초 모형으로 넣고 100개 bag을 만든 뒤 soft voting으로 합쳤다.
시험 예측에는 전체 자료 적합을 사용했고 상세 CV 절차와 CV 수치는 보고하지 않았다.
Public은 0.83699, Private은 0.83905였다.
100개 bag 외에 하드웨어와 계산량은 미보고이며 이전 AutoML Grand Prix 해법과 같은 제출이라고 적었다.

댓글에서 bag 수를 늘리면 일정 지점까지 좋아지지만 그 뒤 이득이 거의 없거나 낮아지고 계산량만 커졌다고 답했다.
Tilii는 초반 해법을 끝까지 유지해 마지막 공개 결과에 끌리지 않은 점을 높이 평가했다.
we would have won이라는 표현은 AutoML Grand Prix의 Private 기준 맥락이므로 실제 대회 순위 주장으로 확장하지 않았다.

S6E8 판단은 기존 결정과 중복이다.
bagging, soft voting, XGBoost와 LightGBM은 외부 278개의 랜덤 포리스트 및 bag 계열과 자체 나무 모형에 인접한다.
100개라는 수의 독립 제거 대조가 없고 더 늘릴 때 악화와 비용 증가도 있어 새 후보가 아니다.

## S4E5: Regression with a Flood Prediction Dataset

과제는 동질적인 수치 열 20개로 홍수 발생 확률을 예측하는 회귀이고 [공식 평가 지표](https://www.kaggle.com/competitions/playground-series-s4e5/overview/evaluation)는 R2다.
공식 해법 글은 1위부터 4위 네 개뿐이고 모두 포함했다.
목록 댓글 합은 96개, 상세 일반 댓글 합은 87개, 감사 댓글 합은 1개였다.

### 1위: aldparis

[#1st place solution](https://www.kaggle.com/competitions/playground-series-s4e5/writeups/aldparis-1st-place-solution)은 aldparis의 1위 글이다.
목록에는 댓글 58개가 표시됐고 상세 화면에는 일반 댓글 56개와 감사 댓글 1개가 표시됐다.
삭제 댓글 두 개는 내용에 접근할 수 없었다.

모든 모형의 OOF는 three repeated KFold로 만들었지만 K 값은 보고하지 않았다.
행 합, 표준편차, 최댓값, 열값 정렬, 값이 6, 7, 8보다 큰 열 개수, 합 기준 목표 부호화와 groupby(sum)의 목표 표준편차를 만들었다.
원시 열, 왜도, 첨도와 중복 열은 버렸고 순열 중요도와 후진 제거를 사용했다.
목표에서 행 평균의 0.1배를 뺀 값 또는 비슷한 정수형 잔차를 새 목표로 만든 변형도 사용했다.

30개가 넘는 CatBoost, XGBoost와 LightGBM을 특성 묶음과 목표 변환별로 만들고 AutoGluon OOF 여섯 개와 공개 LightGBM 출력도 넣었다.
처음에는 LinearRegression의 positive=True 및 intercept=False를 썼고 마지막에는 음수 계수를 허용한 Ridge와 intercept=False로 바꿨다.
댓글에서 최종 CV 0.86956과 Public 0.86943을 보고했다.
본문의 진행 수치는 Public 0.86939, 0.86941, 최종 0.86943이며 이전 0.96934 표기는 문맥상 오타 가능성이 있지만 원문 수치로만 남긴다.
최종 Private은 미보고이고 댓글의 Public과 Private 차이 0.00062는 다른 수치와 일관되지 않아 역산하지 않았다.

Optuna는 Kaggle GPU 시간 안에서 실행당 최대 5,400초를 사용했고 AutoGluon 여섯 OOF는 병렬 학습했다.
AutoGluon 시작 노트북과 공개 Flood LightGBM에 의존했다.

댓글은 원본 자료의 중요한 검증 반례를 남겼다.
원본을 훈련과 검증에 먼저 섞으면 합성 훈련 및 시험과 분포가 달라 낙관적인 CV가 생긴다고 설명했다.
각 fold 학습 부분에만 원본을 넣은 판과 넣지 않은 판을 짝비교했고 이 대회에서는 원본 미추가가 더 좋았다.
원본 목표가 단순 sum/200 관계라 사용하지 말아야 했다고 보충했다.
GARFIELD가 원본을 전체 자료에 먼저 합쳐 CV 0.87089를 얻었지만 Public 0.86845와 Private 0.86805로 내려간 사례도 확인됐다.
Tilii는 다섯째 소수점 결합 차이를 Ridge 제약보다 기초 모형 차이로 볼 수 있다고 반론했고 작성자는 자신의 CV에서는 개선을 봤다고 답했다.

S6E8 판단은 기존 결정과 중복 또는 현재 과제에 부적합이다.
원본 자료를 fold 학습 부분에만 넣고 미추가와 짝비교하는 규율은 현행 fold 격리와 원본 관점 판정에 이미 있다.
행 합, 정렬, 임계 개수와 합 기반 목표 변환은 동질적인 20개 수치와 연속 목표에 특화돼 S6E8의 이질적인 설명변수 및 이진 목표와 대응하지 않는다.
낙관 CV 실패 사례는 원본 자료를 검증 부분에 섞지 않는 현재 계약을 강화한다.

### 2위: lash_fire와 mdoroch

[#2nd Place Solution, Team Peaky Blenders](https://www.kaggle.com/competitions/playground-series-s4e5/writeups/peaky-blenders-2nd-place-solution-team-peaky-blend)는 lash_fire와 mdoroch 팀의 2위 글이다.
목록에는 댓글 3개가 표시됐고 상세 화면에는 일반 댓글 2개가 표시됐다.
실제 화면에는 답글을 포함한 댓글 세 개가 보였으므로 원시 표시값을 따로 보존했다.

공개 특성과 설정을 포함해 AutoGluon을 제외한 XGBoost, CatBoost와 LightGBM 56개를 만들었다.
성장 정책, tree_method, objective와 sampling_method를 바꿨고 최고 단일 점수는 약 0.86933이었다.
OOF 부분집합을 LinearRegression으로 먼저 합치고 이 중간 결합을 다시 후보 풀에 넣어 전진 선택했다.
최종은 자신의 최선 0.6과 팀원 mdoroch 해법 0.4의 평균이었고 Public으로 보이는 0.86943과 명시적 Private 0.86902를 보고했다.
mdoroch 쪽은 XGBoost, LightGBM, CatBoost, PyBoost와 공개 AutoGluon 두 개의 가중 결합이었다.
공개한 최종 가중치에는 음수가 여러 개 있었다.

NGBoost, Linear Tree Regression, Linear Forest Regression과 Linear Boosting Regression도 미조정 상태로 넣었고 전체 기여가 작았다고만 했다.
OOF를 썼지만 fold 수와 CV, 하드웨어 및 계산량은 미보고다.
AutoGluon OOF는 시간과 파일 판본 오류로 얻지 못해 팀원 예측을 OOF 없이 0.4로 섞었다.
댓글 세 개는 LinkedIn 연결 이야기뿐이고 기술 보충은 없다.

대규모 GBDT OOF와 음수 가중 선형 결합은 기존 결정과 중복이다.
NGBoost 및 Linear Tree, Forest, Boosting은 자체 35개와 외부 278개의 명시적 이름에서 비어 있을 가능성이 있지만 단독 OOF와 제거 기여가 없어 근거 부족이다.
OOF 없는 팀원 예측의 맹목 혼합은 현행 계보 계약에 맞지 않는다.

### 3위: Tilii

[#3 solution, A blend of 57 models](https://www.kaggle.com/competitions/playground-series-s4e5/writeups/tilii-3-solution-a-blend-of-57-models)는 Tilii의 3위 글이다.
목록에는 댓글 28개가 표시됐고 상세 화면에는 일반 댓글 22개가 표시됐다.
3 more replies를 펼쳐 답글 세 개까지 확인했다.

57개 모형을 Lasso로 합쳤고 약 40% 구성원의 가중치가 음수였다.
최고 단일 LightGBM과 CatBoost의 Private은 0.86890, 최고 AutoGluon은 0.86893, 최종은 0.86902였다.
CV와 Public은 미보고이고 CV와 리더보드 상관이 좋았다고만 적었다.

자료 표현은 집계 특성만 사용하고 원시 열을 뺀 판, 모든 열을 포함한 판, 유전 프로그래밍 특성 세 개, 자동부호화 잠재 특성, 행 값 개수 특성으로 나눴다.
각 표현에 LightGBM, XGBoost, CatBoost, AutoGluon, GBDT random-forest mode, Keras, PyTorch, TabNet과 Lasso를 곱해 하나씩 결합에 추가했다.
유전 프로그래밍과 자동부호화는 도움 되지 않았고 값 개수 특성은 조금 도왔다.
LightGBM과 XGBoost random-forest mode는 빠르고 최고점에 가까워 특성 및 설정 조합 선별에 유용했다고 했다.

26개 모형 Private 0.86900에서 30개 넘게 더해 0.00002만 올랐고 작성자는 실무 가치가 없다고 평가했다.
신경망은 CV보다 리더보드가 이상하게 좋았지만 결합 기여는 작았다.
댓글에서 Mart는 세심하게 조정한 CatBoost가 빠지고 AutoGluon, LightGBM과 XGBoost 하나씩만 남았다고 보고했다.
1위 작성자는 자신의 Lasso 구현에서 R2 -66이 나 실패했다고 했으므로 결합 방식 자체를 일반화할 수 없다.

S6E8 판단은 대부분 기존 결정과 중복이다.
57개 OOF와 음수 가중 결합은 현재 313개 결합이 더 엄격하게 다룬다.
random-forest mode는 외부 rf, xgb_raw_bag 및 ExtraTrees 구성원에 인접한다.
자동부호화 실패는 S4E7 3위의 성공과 충돌하므로 DAE를 일반적으로 채택하지 않고 고정 한 설정의 nested 판정을 요구하는 근거다.

### 4위: Matt OP

[4th Place Solution: Hill climbing through the noise](https://www.kaggle.com/competitions/playground-series-s4e5/writeups/4th-place-solution-hill-climbing-through-the-noise)는 Matt OP의 4위 글이다.
목록과 상세 화면 모두 댓글 7개가 표시됐다.

행 평균, 중앙값, 최빈값, 최댓값, 최솟값, 표준편차, 왜도, 첨도, 분위수와 행 안 각 고유값의 개수를 세 가지 특성 묶음으로 만들었다.
여러 수동 및 Optuna 설정과 DecisionTreeRegressor 여러 설정을 먼저 합쳤다.
Ridge와 Lasso보다 자신의 hillclimbers 패키지로 찾은 탐욕 가중치가 가장 좋았다고 보고했다.
CV, Public, Private, 모형 수, 최종 가중치, 하드웨어와 계산량은 미보고다.

댓글에서 Tilii는 같은 모형 풀에서 탐욕 결합 0.86901, Lasso 0.86902였다고 반박했다.
1위 작성자는 탐욕 결합이 모수가 많아 과적합할 것으로 예상했지만 4위의 Public 및 Private 순위를 보고 자신의 예상이 틀렸다고 인정했다.
따라서 탐욕 결합의 보편적 우위나 열위를 주장할 수 없다.

S6E8 판단은 기존 결정과 중복이다.
선형과 탐욕 결합은 현재 313개에서 직접 비교됐고 행 집계는 이 대회의 동질 열 구조에 특화됐다.

## S4E4: Regression with an Abalone Dataset

과제는 전복의 물리 측정으로 Rings를 예측하는 회귀이고 [공식 평가 지표](https://www.kaggle.com/competitions/playground-series-s4e4/overview/evaluation)는 RMSLE다.
공식 해법 글은 1, 2, 3, 4, 5, 8위 여섯 개뿐이고 모두 포함했다.
목록 댓글 합은 65개, 상세 일반 댓글 합은 59개, 감사 댓글 합은 5개였다.

### 1위: Johannes Heller

[1st Place Solution for the Regression with an Abalone Dataset Competition](https://www.kaggle.com/competitions/playground-series-s4e4/writeups/johannes-heller-1st-place-solution-for-the-regress)는 Johannes Heller의 1위 글이다.
목록에는 댓글 51개가 표시됐고 상세 화면에는 일반 댓글 47개와 감사 댓글 3개가 표시됐다.
삭제 댓글 두 개는 내용에 접근할 수 없었다.

검증은 10-fold였고 원본 Abalone 전체를 각 fold 학습 부분에만 넣고 검증 부분에는 넣지 않았다.
OpenFE로 원래 8개 열에서 192개 파생을 만든 뒤 높은 상관 특성을 일부 버렸다.
LightGBM, XGBoost와 CatBoost마다 Sequential Feature Selection으로 약 20개씩 다른 특성을 골랐다.
Shell_weight 빈도, Length와 Shell_weight의 차, Whole_weight와 Shucked_weight의 비율, Whole_weight 빈도 및 log, Whole_weight와 Shell_weight의 합, Length와 Shell_weight의 비율과 잔차 등이 포함됐다.

모형은 LightGBM, XGBoost, CatBoost, HistGradientBoosting, RandomForest, AutoGluon과 XGBoost 다중 분류 plus softmax 회귀 head였다.
LightGBM에는 맞춤 MSLE 손실, XGBoost에는 reg:squaredlogerror를 쓰고 나머지는 log1p 목표와 expm1 역변환을 사용했다.
개별 CV RMSLE는 LightGBM 0.14611, CatBoost 0.14620, XGBoost 0.14616, XGBoost 분류 head 0.14680, HistGradientBoosting 0.14648, AutoGluon 0.14592였다.
49개 모형 결합은 0.14514였다.

49개 OOF의 가중치를 Nelder-Mead로 맞췄고 음수 계수를 허용한 계수 합은 0.997이었다.
OOF가 없는 공개 ANN은 Public 점수로 17% 가중치를 맞춰 최종에 더했다.
그 ANN을 뺀 두 번째 제출도 Public 0.14372, Private 0.14379로 1위 수준이었다고 보고했다.
49모형 10-fold와 모형별 특성 선택을 밤새 실행했고 W&B sweep을 같은 ID로 최대 다섯 노트북에서 병렬 실행했다고 댓글에서 답했다.
하드웨어와 총시간은 미보고다.

댓글에서 작성자는 OpenFE 기본 192개를 만든 뒤 상관 제거와 모형별 선택을 밤새 실행했다고 설명했다.
작고 평평하며 고품질인 자료에는 적합하지만 더 큰 실제 자료에는 다른 방법이 낫고 OpenFE도 유지보수가 멈춘 듯하다고 제한했다.
Prajwal은 XGBoost 분류 plus 회귀 head가 단독 CV는 약해도 결합에서 예상보다 큰 가중치를 받았다는 사실을 독립 확인했다.
작성자의 StratifiedKFold 실패 주장에는 George가 희소 Rings를 층화하는 타당성을 물었고 Prajwal은 자신의 층화 CV와 리더보드가 잘 맞았다고 반례를 제시했지만 작성자 답은 없었다.

S6E8 판단은 대부분 기존 결정과 중복 또는 현재 과제에 부적합이다.
AutoGluon, 여러 GBDT, RandomForest, 음수 가중 OOF 결합과 원본 fold 격리는 자체 및 외부 풀에 이미 있다.
RMSLE 전용 손실, Abalone 물리 특성과 다중 분류 plus 회귀 head는 이진 ROC AUC 과제에 대응하지 않는다.
OOF 없는 ANN을 Public으로 가중한 경로는 ADR 0001에 맞지 않고 그 ANN 없이도 1위였으므로 필수성도 없다.
OpenFE와 모형별 선택은 댓글의 범위 제한과 독립 재현 반례 때문에 근거 부족이다.

### 2위: Lennart Purucker

[2nd Place Solution for the Regression with an Abalone Dataset Competition](https://www.kaggle.com/competitions/playground-series-s4e4/writeups/lennart-purucker-2nd-place-solution-for-the-regres)는 Lennart Purucker의 2위 글이다.
목록과 상세 화면 모두 댓글 5개가 표시됐고 감사 댓글 1개가 따로 표시됐다.

원본, 대회 훈련과 대회 시험의 쌍별 분포 이동을 AutoGluon 자료 출처 분류와 Mann-Whitney U 검정으로 진단한 뒤 원본 자료를 포함했다.
OpenFE 후보 생성기를 수동 제한해 빈도, 반올림, 잔차, 나눗셈, 곱, 그룹 중앙값 및 표준편차 및 빈도 및 값 종류 수, Combine과 분위수 임계 표시를 만들었다.
약 200개 특성을 만든 뒤 LightGBM proxy를 읽는 AutoGluon FeatureSelector로 가지치기했다.
본문은 이 선택 단계의 시간 제한을 1시간이라고 보고했다.

최종은 맞춤 AutoGluon 모형 선택과 최대 6단, 실제 5단의 결합이었다.
작성자는 3단과 5단 제출 점수가 같았고 5단도 과적합하지 않았다고 했으므로 더 깊은 두 단계의 추가 이득은 확인되지 않았다.
최종 CV, Public, Private, 하드웨어와 전체 시간은 미보고다.

댓글의 가지치기 예시는 BaggedEnsembleModel과 LightGBM, 4-fold, 최대 300,000표본, prune_ratio 0.15와 stopping_round 4였지만 TODO와 오타가 남은 불완전 예시였다.
MaxUhl98의 독립 재현은 모든 OpenFE를 넣은 AutoGluon 0.14517, 글의 수정 특성 0.14555, 수정 특성과 vanilla LightGBM 선택 0.14632로 오히려 나빠졌다.
작성자는 OpenFE 뒤 특성 선택, 좋은 검증과 선택기 조정이 중요하다고 답했지만 최종 재현 코드는 없다고 명시했다.

자료 출처 분류와 Mann-Whitney U 진단은 근거 부족이다.
자체 35개에 같은 명시적 진단은 비어 있지만 예측 후보가 아니고 원본 포함 또는 제외를 자동으로 정하는 규칙과 제거 이득이 보고되지 않았다.
OpenFE와 fold 내부 모형별 선택도 여러 상위 글에서 반복됐지만 독립 재현이 악화했고 현재 S6E8의 자동 특성 생성 및 고차 결합 음성 판단을 뒤집지 못한다.
다층 AutoGluon은 기존 결정과 중복이고 3단과 5단의 제출이 같아 단계 추가 근거도 없다.

### 3위: LuminousC

[3rd Place Solution for the Regression with an Abalone Dataset Competition](https://www.kaggle.com/competitions/playground-series-s4e4/writeups/luminousc-3rd-place-solution-for-the-regression-wi)는 LuminousC의 3위 글이다.
목록과 상세 화면 모두 댓글 2개가 표시됐고 감사 댓글 1개가 따로 표시됐다.

최종은 원시 AutoGluon, OpenFE 30특성 AutoGluon, 같은 30특성과 의사 라벨 AutoGluon, Optuna 가중 XGBoost 및 LightGBM 및 CatBoost 결합, 세 GBDT voting과 공개 LightGBM 결합의 여섯 덩어리였다.
OpenFE를 RMSLE에 맞게 고쳤고 log1p 및 expm1도 같은 효과라고 적었다.
LogisticRegression RFE로 30개를 골랐고 단일 fold가 아니라 여러 CV fold에서 선택한 중요도를 평균해야 안정적이라고 강조했다.
정확한 fold 수, CV, Public, Private, 계산 시간과 하드웨어는 미보고다.
회귀 대신 다중 분류와 원본 자료 가중치 증대는 실패했다.
공개 신경망은 지역 CV에서 충분한 추가 이득이 없어 제외했다.
댓글은 감사뿐이고 기술 보충은 없다.

S6E8 판단은 기존 결정과 중복 또는 근거 부족이다.
AutoGluon, 여러 GBDT, 의사 라벨과 공개 OOF는 현재 자체 또는 외부 결정에 있고 OpenFE 및 fold별 선택은 구성 요소 제거 수치가 없어 다시 열지 않는다.

### 4위: Bertan Pank

[4th Place Solution for the Regression with an Abalone Dataset](https://www.kaggle.com/competitions/playground-series-s4e4/writeups/bertan-pank-4th-place-solution-for-the-regression-)은 Bertan Pank의 4위 글이다.
목록과 상세 화면 모두 댓글 2개가 표시됐다.

공개 AutoGluon 노트북을 OpenFE, log 목표, 맞춤 RMSLE와 나무 전용 AutoGluon으로 고쳤다.
최종은 자신의 AutoGluon 출력과 공개 LightGBM 전용 노트북을 같은 가중치로 평균했다.
CV, Public, Private, fold, 계산량과 원본 자료 사용 여부는 미보고다.
댓글은 공개 노트북 주소를 다시 알려 준 것 외 기술 보충이 없다.

S6E8 판단은 기존 결정과 중복이다.
AutoGluon, LightGBM, 같은 가중치 평균은 현재 313개 결합보다 좁고 OpenFE 이득도 분리되지 않았다.

### 5위: Minato Namikaze

[5th Place Solution, Learnings](https://www.kaggle.com/competitions/playground-series-s4e4/writeups/minato-namikaze-5th-place-solution-learnings)은 Minato Namikaze의 5위 글이다.
목록에는 댓글 5개가 표시됐고 상세 화면에는 일반 댓글 3개가 표시됐다.

Top Surface Area, Water Loss, Measurement Ratios, Abalone Density와 BMI, 이산 및 범주 부호화와 수치 변환을 사용했다.
XGBoost, CatBoost, LightGBM과 ANN을 사용했고 15-fold와 20-fold가 5-fold보다 좋았다고 보고했다.
최종 결합은 조화 평균만 사용했다.
공개 3모형 voting regressor를 자신의 특성으로 바꾼 출력과 공개 ensemble 노트북을 참고했다.
CV, Public, Private, 하드웨어와 계산 시간은 미보고다.
댓글에서 학습을 위해 그동안 AutoML을 일부러 피했다고 보충했지만 제거 대조는 없다.

Abalone 도메인 특성은 현재 과제에 부적합하다.
15-fold와 20-fold 및 조화 평균은 구성 요소 제거 수치가 없고 S4E7 1위는 조화와 기하 평균이 실패했다고 보고해 근거 부족이다.
현재 고정 5-fold와 313개 순위 결합을 바꿀 근거가 아니다.

### 8위: EISLab_hwlee

[8th Place Solution, So simple](https://www.kaggle.com/competitions/playground-series-s4e4/writeups/eislab-hwlee-8th-place-solution-so-simple)은 EISLab_hwlee의 8위 글이다.
목록과 상세 화면 모두 댓글 0개였다.

공개 특성을 사용하고 AutoGluon best_quality를 num_trials 5, dynamic_stacking False, num_stack_levels 2와 시간 제한 24시간으로 실행했다.
KNN, NN_TORCH와 FASTAI는 제외하고 keep_only_best를 사용했다.
글의 AutoGluon eval_metric은 대회 RMSLE가 아니라 root_mean_squared_error로 적혀 있고 목표 log 변환 여부는 설명하지 않았다.
최종은 공개 LightGBM 전용 노트북과 AutoGluon을 합쳤지만 가중치는 미보고다.
CV, Public, Private, fold와 하드웨어도 미보고다.

S6E8 판단은 기존 결정과 중복이다.
AutoGluon과 LightGBM은 이미 포함되고 지표 불일치와 미보고가 많은 이 글은 새 결정을 지지하지 못한다.

## 네 대회에서 반복된 패턴과 반례

### 결합의 폭보다 상보성과 선택 규율이 중요했다

S4E7 1, 3, 4, 6, 9위, S4E6 6위, S4E5 네 글과 S4E4 1위부터 4위는 여러 OOF를 선형, 탐욕 또는 비선형으로 합쳤다.
약한 FM, FFM, 신경망, 분류 head와 random-forest mode가 단독 점수보다 결합 다양성으로 선택된 사례도 반복됐다.

그러나 큰 결합의 보편적 우위는 없었다.
S4E6 6위 댓글에는 단일 모형이 8개 또는 18개 결합을 이긴 사례와 6개 또는 30개 결합이 단일 모형을 크게 이긴 사례가 함께 있었다.
S4E5 3위는 30개 넘게 추가해 Private 0.00002만 올랐고 실무 가치가 없다고 평가했다.
S4E7 4위와 최신 S6E8 외부 사다리도 작은 가중치 또는 해로운 구성원 묶음을 제거할 필요를 보여 준다.

따라서 모형 수를 목표로 삼지 않는다.
단독 점수, 현재 풀과의 오류 상관, 공급원 및 계열 제거와 nested 한계 기여를 함께 보는 현행 절차를 유지한다.

### 같은 fold와 원본 자료 격리가 검증의 전제였다

S4E7 1위는 78개 구성원에 같은 StratifiedKFold 5-fold를 썼고 S4E7 6, 9위도 OOF와 시험 예측을 짝으로 저장했다.
S4E4 1위는 원본 자료를 각 fold 학습 부분에만 넣었고 S4E5 1위는 원본을 검증에 먼저 섞어 생긴 낙관 CV와 실제 Private 하락을 구체적으로 보고했다.
S4E4 3위는 특성 선택도 한 fold가 아니라 여러 fold에서 평균해야 안정적이라고 강조했다.

이는 새 후보가 아니라 현행 검증 계약을 유지할 근거다.
원본이나 시험 자료를 목표 없이 쓰더라도 변환 적합 범위와 fold 계보를 장부에 남기고 검증 행의 정보 조건을 시험 행과 맞춰야 한다.

### Public과 OOF 없는 제출은 선택 근거가 될 수 없었다

S4E7 4위는 이전 대회에서 Public 1위를 유지하려다 최종 113위가 된 경험을 기록했다.
S4E6 댓글은 CatBoost의 Public이 높아도 Private이 낮은 사례를 보였고 S4E5 1위는 낙관 CV와 Public 및 Private 하락을 함께 보고했다.
S4E4 1위의 OOF 없는 공개 ANN은 Public으로 17% 가중됐지만 이를 뺀 제출도 1위 수준이었다.
S4E5 2위는 팀원 예측의 OOF를 얻지 못해 0.4를 맹목 혼합했다.

이 경로들은 과거 순위의 일부였다는 이유로 현재 채택 계약에 허용하지 않는다.
Public 0.97135는 사후 참고로만 남기고 OOF 및 시험 예측 짝이 없는 외부 제출은 313개 결합 입력으로 쓰지 않는다.

### 자동 특성 생성은 작은 회귀 자료에서조차 재현이 불안정했다

S4E4 1, 2, 3, 4위는 OpenFE와 모형별 또는 fold별 선택을 반복 사용했다.
그러나 2위 글의 독립 재현은 모든 OpenFE 0.14517보다 수정 특성 0.14555와 선택 후 0.14632가 나빠졌다.
1위 작성자도 OpenFE가 작고 평평한 고품질 자료에는 맞지만 큰 실제 자료에는 다른 방법이 낫다고 제한했다.
S4E5 3위의 유전 프로그래밍 특성도 도움 되지 않았다.

현재 S6E8의 고차 결합 및 자동 특성 생성 음성 결과를 다시 열 근거는 없다.
분포 이동 진단과 fold 내부 선택은 별도 진단으로 연구할 수 있지만 예측 구성원 후보로는 근거 부족이다.

### DAE는 성공과 실패를 함께 가진 유일한 빈 표현이었다

S4E7 3위는 8차원 DAE 잠재 표현을 CatBoost와 Keras embedding에 붙여 약 0.0002가 올랐다고 보고했다.
S4E5 3위는 자동부호화 잠재 특성이 도움 되지 않았다고 명시했다.
기존 1년 조사에는 S6E3 1위의 원본 전용 DAE와 S5E6 2위의 지도형 자동부호화가 있었지만 단독 제거 기여가 부족했다.

S4E7의 별도 0.0002 보고는 독립 근거를 강화하지만 일반 채택을 정당화하지는 않는다.
고정 한 구현을 누출 없이 만들고 현재 결합의 한계 기여까지 통과해야 한다.

## 현재 S6E8에 대한 적용 판단

현재 자체 후보 풀은 [후보 풀 장부](../../artifacts/pool.yaml)의 35개다.
자체 풀은 원본 프록시, 잔차, 격자 및 재구성 목표 부호화, 정확값 범주 복제, 결측 대치, 선형 one-hot, XGBoost, LightGBM, CatBoost, AutoGluon, TabPFN, RealMLP, TabM, Lookup-Transformer, 표 형태 합성곱 신경망과 문맥 스플라인을 포함한다.

최신 외부 결합은 [두 번째 넓힌 확장 결합 기록](extended-stack-submission-2.md)의 실행 443b3a71a2b045ba9052fbb3d821255d다.
외부 278개에는 여러 XGBoost, LightGBM, CatBoost, 랜덤 포리스트, MLP, RealMLP, TabM, FT-Transformer, TabTransformer, Trompt, 선형 및 다항 및 RBF SVM, 다중 목표 부호화, 격자, 결측 대치, 잔차, Factorization Machine 다섯 개와 DeepFM이 포함된다.
최종 313개는 shrunk_rank_logit_logistic으로 nested OOF 0.9703509469와 가중 OOF 0.9712170271을 얻었다.

[기존 1년 조사와 실험 발주 기준인 이슈 307](https://github.com/tmheo/predicting-smartphone-addiction/issues/307)의 네 범주는 새 후보, 기존 결정과 중복, 현재 과제에 부적합, 근거 부족이다.
아래 표는 자체 35개 안의 빈 관점인지와 외부 278개가 이미 덮는 관점인지를 분리한 예비 판정이다.

| 조사 관점 | 자체 35개 기준 | 외부 278개 기준 | 이슈 307 예비 분류 | 판단 |
| --- | --- | --- | --- | --- |
| 8차원 DAE 잠재 표현을 CatBoost와 embedding 신경망에 추가 | 명시적 구현 없음 | 명시적 자동부호화 구성원 없음 | 새 후보 | S4E7의 약 0.0002 구성 요소 이득과 이전 독립 대회 근거가 있어 누출 없는 한 설정으로 진입 대조 |
| Factorization Machine과 Field-aware Factorization Machine | 자체 최종 구성원 없음 | Factorization Machine 다섯 개와 deepfm_exact 포함 | 기존 결정과 중복 | 외부 구성원 제외 기여를 우선하고 자체 재학습 안 함 |
| 행 구간별 전문 모형과 비선형 2단 | 자체 nested OOF 음성 결정과 결측 구간별 선형 결합이 있음 | 여러 원시 모형은 있으나 2단 산출물은 반입하지 않음 | 기존 결정과 중복 | 새 행 체제 근거가 없으면 재개 안 함 |
| XGBoost, LightGBM, CatBoost, 신경망, AutoGluon과 대규모 OOF 결합 | 주요 학습기와 19개 결합 전략 포함 | 278개가 더 넓은 오류 폭 제공 | 기존 결정과 중복 | 구성원 수가 아니라 nested 포함 및 제외 기여로 판단 |
| 순위 변환 뒤 Ridge, Lasso 또는 로지스틱 결합 | shrunk_rank_logit_logistic이 현재 최선 | 같은 278개를 순위 공간에서 결합 | 기존 결정과 중복 | 현행 결합 유지 |
| 원본 자료를 fold 학습 부분에만 추가하고 미추가와 짝비교 | 원본 프록시 및 원본 분포 구성원과 행 주입 음성 결과가 있음 | 원본 활용 외부 구성원 다수 | 기존 결정과 중복 | 검증 격리 규율로 유지하고 무조건 행 추가 안 함 |
| CatBoost Newton 계열, 낮은 학습률과 긴 반복 | CatBoost 정확값 및 고정 반복 수 구성원 포함 | 여러 CatBoost 설정 포함 | 기존 결정과 중복 | 현재 고정 반복 nested 판정을 우선함 |
| LightGBM 및 XGBoost random-forest mode와 100-bag | 정확한 최종 구성원은 없으나 나무 다양성 있음 | rf, xgb_raw_bag과 ExtraTrees 인접 구성원 포함 | 기존 결정과 중복 | 독립 제거 기여가 없고 bag 증가 이득도 포화 |
| OpenFE 뒤 모형별 또는 fold 내부 특성 선택 | 정확한 구현 없음 | 정확한 구현 확인 안 됨 | 근거 부족 | 반복 사용됐지만 독립 재현 악화와 범위 제한이 있어 열지 않음 |
| 자료 출처 분류와 Mann-Whitney U 원본 이동 진단 | 같은 명시적 진단 없음 | 외부 원본 활용 폭은 있으나 진단 출력은 없음 | 근거 부족 | 예측 후보가 아니고 원본 포함 제거 기여 및 자동 결정 규칙 미보고 |
| NGBoost와 Linear Tree, Forest, Boosting | 명시적 구성원 없음 | 명시적 같은 계열 확인 안 됨 | 근거 부족 | 미조정 작은 전체 기여만 있고 단독 OOF와 제거 대조 없음 |
| 15-fold 또는 20-fold와 조화 평균 | 고정 5-fold 계약과 조화 평균 음성 근거가 있음 | 대응 결합을 별도 선택하지 않음 | 근거 부족 | 수치 제거 대조가 없고 다른 1위 글은 조화 평균 실패 보고 |
| 보험 목표 반전과 충돌 중복 후처리 | S6E8 라벨 반전 기각 | 대응 없음 | 현재 과제에 부적합 | 과거 보험 원본과 시험 중복 규칙에 의존함 |
| 홍수 행 합 및 정렬 및 임계 개수와 잔차 목표 변환 | 대응하는 동질 열 구조 없음 | 대응 없음 | 현재 과제에 부적합 | 동질 수치 20개와 연속 목표에 특화됨 |
| Abalone 면적, 무게 비율, 밀도, BMI와 RMSLE 전용 손실 | 대응 열과 목표 없음 | 대응 없음 | 현재 과제에 부적합 | 이진 ROC AUC로 옮길 수 없음 |
| OOF 없는 공개 ANN 또는 팀원 제출의 Public 가중 혼합 | ADR 0001이 금지함 | OOF와 시험 짝이 없는 제출은 장부에서 제외함 | 현재 과제에 부적합 | Public은 사후 참고만 유지 |

### 새 후보의 정확한 진입 조건

DAE 후보의 근거는 S4E7 3위가 보고한 약 0.0002 구성 요소 이득과 S6E3 및 S5E6의 독립 사용 사례다.
S4E5 3위 실패와 S4E7의 fold 적합 미보고가 있으므로 구조와 학습 범위를 결과 전에 하나로 고정해야 한다.

S4E7에서 확인된 핵심은 범주 one-hot과 일부 수치 눈금 입력, 병목 크기 3보다 나았던 8차원 표현, CatBoost 및 embedding 신경망에 잠재 열을 추가한 방식이다.
잡음 종류, 층 폭, 학습 회수와 규제는 글에서 완전히 보고되지 않았으므로 여러 구조를 동시에 탐색해 가장 좋은 것을 고르면 안 된다.
구현 전 사전 장부에 단일 구조, fold별 전처리와 DAE 적합 범위, 난수, 회수와 중단 규칙을 고정해야 한다.

각 바깥쪽 fold의 학습 부분만으로 전처리와 DAE를 맞추고 검증 및 시험에는 변환만 적용하는 것이 가장 보수적인 기본안이다.
기준 모형과 같은 fold에서 DAE 열 미포함 및 포함을 짝비교하고 잠재 열만 읽는 약한 모형도 별도 OOF로 저장해 표현 자체의 예측력을 확인해야 한다.
Ujjwal Pandey의 난수 불일치 가짜 CV 0.895 사례 때문에 fold와 난수 계보 및 행 정렬을 기계 검증해야 한다.

후보 풀 진입과 최종 결합 채택은 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)을 그대로 따른다.
자체 3시드 OOF, champion 하한, 0.998 순위 중복 검사와 현재 자체 35개 및 외부 278개 포함 전후의 nested 한계 기여를 모두 판정한다.
DAE가 단독 기준 모형을 못 이겨도 낮은 상관과 양의 결합 기여가 있으면 다양성 후보로 판단하고, S4E5처럼 어느 경로도 개선하지 못하면 기각한다.

### 실제로 적용할 부분

- 모든 구성원은 같은 행 순서의 OOF와 시험 예측을 짝으로 저장하고 fold 및 난수 계보를 확인한다.
- 원본 자료나 목표 없는 시험 자료를 변환에 쓸 때는 적합 범위와 검증 행의 정보 조건을 사전 고정한다.
- 단독 모형이 약해도 버리지 않되 현재 풀의 nested 포함 및 제외 기여와 오류 상관으로만 다양성을 인정한다.
- 외부 구성원 폭을 늘린 뒤에는 공급원 및 계열 단위 제거 대조로 해로운 묶음과 작은 가중치 묶음을 확인한다.
- Public 점수와 OOF 없는 제출은 구성원, 가중치, 학습 회수와 중단 선택에 사용하지 않는다.
- 자동 특성 생성과 깊은 다층 결합은 과거 순위만으로 열지 않고 현재 자료의 독립 제거 대조를 먼저 요구한다.
- 전체 자료 재학습과 제출 유한성 및 범위 검사는 [전체 자료 재학습 규약](../adr/0002-full-data-refit-protocol.md)을 유지한다.

## 사실과 추론의 경계

각 대회 절의 순위, 작성자, 팀, 특성, 모형, 검증, 점수, 계산 자원, 외부 자료, 댓글 보충과 실패 사례는 링크된 Kaggle 공식 해법 본문과 댓글에서 확인한 작성자 보고 사실이다.
댓글의 독립 재현 수치와 반례도 해당 댓글 작성자의 보고이며 이 저장소에서 다시 실행한 값은 아니다.
본문과 댓글이 충돌하거나 수치가 일관되지 않은 S4E5 1위의 0.96934 및 Public과 Private 차이는 고치거나 역산하지 않았다.
S4E4 8위의 root_mean_squared_error 표기도 대회 RMSLE와 다르지만 원문 사실로 보존했다.

재현성, 현재 과제와의 유사성, 자체 35개 및 외부 278개에 대한 겹침, 이슈 307의 네 범주와 실험 우선순위는 조사자의 추론이다.
외부 구성원의 이름이 유사하다는 사실만으로 과거 글과 정확히 같은 구현이라고 단정하지 않았다.
DAE를 새 후보로 보강한 판단은 S4E7의 구성 요소 이득, 이전 독립 대회와 현재 풀의 빈 관점을 함께 읽은 추론이다.
과거 대회의 Public과 Private은 후보 발굴의 참고 근거일 뿐 현재 S6E8의 채택 근거가 될 수 없다.

## 한계

조사는 Kaggle 공식 화면이 현재 렌더링한 본문과 댓글을 대상으로 했으므로 삭제 댓글 11개의 과거 내용은 복구하지 않았다.
Kaggle 목록, 상세 머리말과 실제 펼친 댓글 노드의 숫자가 달라 원시 표시값을 따로 보존했다.
작성자가 전체 코드, fold 벡터, 구성원별 OOF 또는 제거 대조를 공개하지 않은 경우 본문만으로 재현 가능하다고 판정하지 않았다.
S4E7 3위 DAE는 잡음 방식과 fold별 적합 여부가 미보고이고 완전한 실행 코드도 없다.
S4E4 OpenFE와 AutoGluon 글들은 공개 노트북과 라이브러리 판본에 크게 의존하며 최종 선택 설정이 불완전한 경우가 있었다.
순위가 없는 글과 10위 밖 글은 흥미로운 방법이 있어도 이번 표본의 반복 성공 근거로 사용하지 않았다.
