# Playground Series S3E26부터 S3E23까지 상위 10위 해법과 댓글 조사

이 문서는 GitHub 이슈 [리서치: S3E26-S3E23 상위권 해법 글과 댓글 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/472)의 조사 결과다.
조사 기준일은 2026-08-28이다.

## 결론

네 대회의 공식 해법 범주에 올라온 글 10개를 모두 확인했고, 제목이나 본문에서 Private 최종 1위부터 10위가 확인된 글은 7개였다.
포함 순위는 S3E26의 2위와 4위, S3E24의 3위, 4위, 7위, 8위, S3E23의 2위다.
S3E25 공식 해법 범주는 비어 있었고 없는 순위를 다른 범주의 글이나 10위 밖 글로 채우지 않았다.

현재 S6E8에 바로 추가할 만큼 근거가 갖춰진 새 후보는 발견하지 못했다.
수치 피처의 구간별 선형 표현, OOF와 원시 피처를 함께 받는 비선형 2단 모형, 탐욕 결합, 원본 자료, 산술 파생 피처, 의사 라벨과 여러 모형 계열은 자체 35개 후보나 최신 외부 278개 결합과 이미 중복되거나 자체 음성 실험이 있다.
Public 순위 기반 가중치, 제출 탐색과 난수값 선택은 현재 채택 계약과 맞지 않고 이 조사 안에서도 Private 하락 반례가 확인됐다.

정확한 구현만 놓고 보면 S3E26 2위의 구간별 선형 수치 표현과 S3E23 2위의 Nyström 로지스틱 회귀가 조건부 단서다.
전자는 자체 `exp085_contextual_spline_m0`, Lookup 계열과 `exp133_scalar_token_transformer_oof_te`가 같은 수치 구간 표현 관점을 이미 덮는다.
후자는 자체 35개에는 없지만 외부 278개에 선형, 다항식과 RBF SVM이 모두 들어 있어 커널 기반 저차원 관점을 이미 덮는다.
두 방법 모두 현재 과제의 깨끗한 OOF 대조와 한계 기여가 없으므로 새 실험으로 열지 않는다.

가장 재사용 가치가 높은 결과는 새 모형보다 검증 절차에 있다.
S3E24 4위는 탐욕 결합의 각 추가 모형이 모든 검증 fold를 개선하는지 확인하고 다른 난수값의 다섯 fold로 다시 검사했다.
반대로 S3E24 7위는 마지막 날 난수값을 바꿔 Public을 올렸지만 Private이 나빠졌고, 댓글에서 작성자도 과거 제출이 더 잘 일반화했다고 확인했다.
현재 nested OOF와 Public 사후 참고 원칙은 이 두 사례가 요구하는 안전장치를 이미 더 엄격하게 구현한다.

## 조사 범위와 방법

Kaggle API, 웹 검색 결과 요약, 직접 HTTP 요청과 리더보드 조회는 사용하지 않았다.
`agent-browser`의 `issue472` 전용 세션을 만들고 콘텐츠 경계를 켠 뒤 Kaggle 도메인만 허용했다.
사용자가 지정한 공식 `competitionWriteUps` 화면에서 대회 식별자만 S3E26, S3E25, S3E24, S3E23으로 바꾸어 조사했다.
목록 정렬은 `recent-comments`를 유지했고 표본 포함 여부는 글 제목이나 본문의 Private 최종 순위 표기로만 정했다.

각 포함 글은 본문 끝까지 읽고 댓글 내부 스크롤의 마지막까지 이동했다.
어느 포함 글에도 `more replies`로 접힌 답글은 남지 않았고 현재 화면에 나타난 답글은 모두 읽었다.
S3E23 2위 글은 Hotness 정렬이 한 번에 댓글 일부만 렌더링해 Newest와 Oldest 정렬을 모두 확인하고 합집합으로 일반 댓글 42개를 읽은 뒤 `#appreciation` 위치에서 감사 댓글 8개를 따로 확인했다.
별도의 다음 댓글 쪽이나 더 불러오기 단추는 어느 포함 글에도 남지 않았다.
삭제된 댓글은 S3E26 2위에 다섯 개, S3E24 3위에 두 개, S3E24 7위에 한 개가 있었고 삭제 표시만 확인할 수 있었다.

Kaggle 목록의 댓글 수와 상세 글 머리말의 일반 댓글 및 감사 댓글 수는 서로 일치하지 않았다.
숫자를 임의로 맞추지 않고 두 화면의 표시값을 따로 기록했다.
포함 글의 목록 댓글 수 합계는 151개였고 상세 글 머리말의 일반 댓글 수 합계는 143개였으며 별도 감사 댓글 표시는 12개였다.
집계 차이의 원인은 공식 화면에서 설명하지 않으므로 추정하지 않는다.

## 표본 장부

| 대회 | 과제와 평가지표 | 공식 해법 글 | 포함 순위 | 결측 순위 | 목록 댓글 합 | 상세 일반 댓글 | 감사 댓글 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| S3E26 | 간경변 결과 3종 분류, 다중분류 log loss | 3 | 2, 4 | 1, 3, 5, 6, 7, 8, 9, 10 | 48 | 45 | 2 |
| S3E25 | 광물 모스 굳기 회귀, 중앙절대오차 | 0 | 없음 | 1부터 10까지 전부 | 0 | 0 | 0 |
| S3E24 | 생체 신호 흡연 여부 이진 분류, ROC AUC | 6 | 3, 4, 7, 8 | 1, 2, 5, 6, 9, 10 | 61 | 56 | 2 |
| S3E23 | 소프트웨어 결함 이진 분류, ROC AUC | 1 | 2 | 1, 3, 4, 5, 6, 7, 8, 9, 10 | 42 | 42 | 8 |

과제와 평가지표는 Kaggle의 [S3E26 평가](https://www.kaggle.com/competitions/playground-series-s3e26/overview/evaluation), [S3E25 평가](https://www.kaggle.com/competitions/playground-series-s3e25/overview/evaluation), [S3E24 평가](https://www.kaggle.com/competitions/playground-series-s3e24/overview/evaluation), [S3E23 평가](https://www.kaggle.com/competitions/playground-series-s3e23/overview/evaluation) 화면에서 확인했다.

## 제외 장부

[S3E26 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s3e26/discussion?sort=recent-comments&category=competitionWriteUps)에서는 Private 39위 Luficer G의 글을 제외했다.
[S3E25 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s3e25/discussion?sort=recent-comments&category=competitionWriteUps)은 `No discussions found`를 표시했으므로 포함 및 제외 글이 모두 없다.
[S3E24 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s3e24/discussion?sort=recent-comments&category=competitionWriteUps)에서는 35위 Algorex의 글과 제목 및 본문에서 최종 순위를 확인할 수 없는 Thomas Meißner의 글을 제외했다.
[S3E23 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s3e23/discussion?sort=recent-comments&category=competitionWriteUps)의 공식 글 하나는 2위가 확인돼 포함했다.
팀원의 중복 글이나 접근할 수 없는 글은 없었다.

## S3E26: Multi-Class Prediction of Cirrhosis Outcomes

### 2위: Hardy Xu

[2nd Place: with help from NNs](https://www.kaggle.com/competitions/playground-series-s3e26/writeups/hardy-xu-2nd-place-with-help-from-nns)는 단독 참가자 Hardy Xu의 글이다.
목록에는 댓글 39개가 표시됐고 상세 화면에는 일반 댓글 36개와 감사 댓글 2개가 표시됐다.

기초 예측은 XGBoost와 LightGBM의 Optuna 설정 각 10개 평균 및 신경망이었다.
수치 피처에는 논문에서 제안한 piecewise linear encoding을 적용하고 `edema`와 `stage`에는 임베딩을, 이진 피처에는 0과 1 값을 그대로 사용했다.
논문은 수치 피처마다 같은 구간 수를 썼지만 작성자는 서로 다른 값이 많은 피처일수록 구간 수를 늘리는 편이 더 좋았다고 댓글에서 설명했다.
이 신경망 단독 Private은 약 0.401로 상위 10% 수준이었고 최종 결합에는 약 0.001을 더했다고 보고했다.

최종 2단 신경망은 단순한 전역 가중 평균이 아니었다.
각 모형의 세 클래스 확률에 서로 다른 가중치를 주고 각 가중치를 해당 예측 하나가 아니라 세 모형 예측 전체에서 계산했다.
이 구조는 단순 평균보다 log loss를 약 0.004 개선했다고 보고했다.
원시 피처, RandomForest와 로지스틱 회귀 예측을 2단 입력에 더하는 실험은 CV를 개선하지 않았다.

댓글에서 일부 피처 반올림은 나무 계열에 작은 이득이 있었고 그 밖의 피처 생성은 도움 되지 않았다고 했다.
모든 작업은 로컬에서 수행했고 공개 노트북은 없었다.
최종 CV, Public, Private 수치와 fold 수, 계산량은 제공하지 않았다.

수치 구간 표현은 현재 자체 spline, Lookup과 scalar-token 계열이 이미 덮는다.
클래스별 및 행별로 달라지는 2단 가중치는 다중분류 log loss에는 자연스럽지만 현재 이진 ROC AUC에서는 클래스별 자유도가 줄고, 행별 비선형 결합은 자체 nested OOF에서 이미 음성이었다.

### 4위: Kirderf

[4th Place Solution: Stacking Approach with XGB as Meta Model](https://www.kaggle.com/competitions/playground-series-s3e26/writeups/kirderf-4th-place-solution-stacking-approach-with-)은 단독 참가자 Kirderf의 글이다.
목록과 상세 화면 모두 일반 댓글 9개를 표시했다.

작성자는 AutoGluon, LightAutoML, 5-fold AutoXGB와 공개 XGBoost 및 LightGBM 노트북에서 OOF를 만들었다.
AutoGluon은 사전 배포 1.0.1b20231208의 무설정 모형 묶음을 썼고 가중 결합의 내부 검증 log loss는 0.436142였다.
LightAutoML은 LightGBM과 CatBoost 네 구성을 다섯 모형씩 평균해 합쳤고 같은 가중치 출발점 점수는 0.4164338이었다.
AutoGluon 증류와 의사 라벨은 더 좋지 않았다.

Age 구간, `log1p` Age와 최소최대 변환 Age만 공통 파생 피처로 만들었다.
최종 2단은 기초 OOF, 원시 피처와 파생 피처를 함께 받은 20-fold XGBoost였다.
여러 XGBoost 2단 결과를 다시 합친 2단계 결합도 두 번째 제출용으로 만들었다.
작성자는 모든 기초 모형을 세밀하게 맞추기보다 서로 다른 예측을 추가 입력으로 쓰는 편이 과적합 위험을 줄인다고 해석했다.

댓글은 축하와 코드 공개 요청뿐이었고 최종 코드, 최종 CV, Public과 Private 수치 및 하드웨어는 제공되지 않았다.
여러 공개 노트북의 OOF에 의존해 전체 계보의 독립 재현성은 낮다.
OOF와 원시 피처를 함께 받는 XGBoost 2단은 현재 S6E8의 자체 음성 결정과 중복이다.

## S3E25: Regression with a Mohs Hardness Dataset

공식 해법 범주에는 글이 하나도 없었다.
따라서 본문과 댓글을 분석할 포함 표본이 없으며 이 대회에서 방법론 결론을 만들지 않았다.

## S3E24: Binary Prediction of Smoker Status using Bio-Signals

### 3위: Ravi Ramakrishnan

[#3 Private, #8 Public Approach](https://www.kaggle.com/competitions/playground-series-s3e24/writeups/ravi-ramakrishnan-3-private-8-public-approach-simp)는 단독 참가자 Ravi Ramakrishnan의 글이다.
목록에는 댓글 37개가 표시됐고 상세 화면에는 일반 댓글 32개와 감사 댓글 1개가 표시됐다.

작성자는 10-fold 층화 분할 한 번을 사용했고 10-fold를 세 번 반복하는 구성은 추가 이득이 없어 버렸다.
합성 자료와 원본 자료를 함께 썼지만 별도 보조 원본 자료는 초기 결과가 나빠 사용하지 않았다.
공개 노트북과 비슷한 산술 피처를 대량 생성한 뒤 순열 중요도로 80개부터 120개만 남겼고 130개 이상은 CV와 리더보드 모두 개선하지 않았다.
피처 생성과 제거는 오래 걸려 로컬 PC에서 수행했다.

기초 모형은 CatBoost 세 개, LightGBM 다섯 개, XGBoost 세 개, RandomForest, 로지스틱 회귀, TabNet, 다층 신경망과 일반화 가법 모형이었다.
TabNet과 신경망의 최종 가중치는 매우 작았다.
탐욕 결합, Optuna 가중 결합과 2단 결합을 비교한 뒤 CV와 리더보드 움직임이 잘 맞았던 Optuna 결합을 골랐다.

최종 제출 하나는 fold 안의 일부 가중치를 Public 결과에 맞춰 수동 조정했고 다른 하나는 조정하지 않았다.
두 제출 모두 좋았지만 Public 탐색판이 Private에서 조금 더 좋아 3위가 됐다고 보고했다.
동시에 맹목적인 공개 예측 혼합은 대체로 일반화하지 않으므로 CV를 올리지 않는 탐색은 피해야 한다고 경고했다.

댓글에서 모형 설정을 Optuna로 찾지 않았고 기본 설정과 좋은 피처 선택에 의존했다고 명확히 했다.
Public 탐색은 진행 중 제출 결과를 보고 결합 가중치를 맞추는 것이라고 정의했다.
최종 CV, Public과 Private 수치는 제공하지 않았다.
Public을 직접 선택에 쓰는 과정은 현재 계약에서 채택 근거로 사용할 수 없다.

### 4위: aldparis

[#4 Place Solution: Robust Hill Climbing](https://www.kaggle.com/competitions/playground-series-s3e24/writeups/aldparis-4-th-place-solution-robust-hill-climbing)은 단독 참가자 aldparis의 글이다.
목록과 상세 화면 모두 일반 댓글 12개를 표시했다.

작성자는 자체 및 공개 노트북에서 25개 OOF를 모으고 탐욕 결합으로 7개를 골랐다.
가장 큰 계수를 받은 세 구성원은 paddykb 평균 OOF, arunklenin LightGBM과 신경망 순서였고 분할을 바꿔도 이 순서가 유지됐다.
자체 모형은 피처 생성 없는 LightGBM 하나와 XGBoost 두 개였으며 원본 자료를 훈련에 추가하고 `hearing(left)`, `hearing(right)`, `Urine protein`을 뺐다.
XGBoost 설정 탐색에는 GPU와 4-fold 무작위 격자 탐색을 썼고 상위 20개 모형은 반복 층화 분할로 다시 평가했다.

탐욕 결합 내부를 다섯 fold로 나누고 후보가 학습 부분 AUC뿐 아니라 검증 부분 AUC도 올리는지 확인했다.
선택된 일곱 모형이 모든 검증 fold를 개선해야 한다는 조건을 두었고 다른 난수값의 다섯 fold로 전 과정을 반복했다.
Public이 시험 자료의 20%이고 점수가 소수 다섯째 자리까지 갈리는 상황이라 우연과 Public 과적합을 줄이는 것이 목적이었다.

최종 CV, Public과 Private 수치는 제공하지 않았다.
작업은 로컬에서 수행했고 최종 노트북은 없었으며 댓글에서 이전 대회의 CV 탐욕 결합 노트북을 참고하라고 답했다.
현재 nested OOF는 같은 안정성 요구를 바깥쪽 분할에서 직접 검사하므로 새 결합기가 아니라 현행 검증 규율을 지지하는 근거다.

### 7위: Sarun P M

[#7 Private and #2 Public Solution](https://www.kaggle.com/competitions/playground-series-s3e24/writeups/sarun-p-m-7-private-lb-and-2-public-lb-solution)은 단독 참가자 Sarun P M의 글이다.
목록과 상세 화면 모두 일반 댓글 7개를 표시했다.

피처 생성 없는 Optuna XGBoost의 Public은 0.87392였다.
공개 의사 라벨 방법을 넣으면 0.87901이 됐고 작성자는 시험 자료의 최대 85%를 의사 라벨로 훈련에 넣어도 Public은 조금 더 오르는 정도였다고 했다.
공개 노트북의 대량 파생 피처와 Public 상위 제출 평균을 더하면 0.88116이 됐다.

헤모글로빈, 체중, 키, GTP, 혈청 크레아티닌과 충치가 중요하다고 보고 이 열들의 곱을 대량 추가했다.
Public은 0.88126이 됐고 마지막 날 난수값을 42에서 43으로 바꾸면 0.88136으로 올랐다.
최종 Private은 0.87926으로 7위였고 작성자는 Public이 낮던 과거 제출 중 Private이 더 높은 것이 있었다고 적었다.

댓글에서 Ravi Ramakrishnan은 난수값 변경의 Public 개선을 신뢰하지 않으며 Public에 맞은 우연 때문에 Private이 내려갔을 가능성이 높다고 지적했다.
작성자는 과거 제출이 실제로 Private에서 더 좋았고 자신의 선택이 경험 부족에서 나온 실수였다고 확인했다.
반복 횟수 질문의 답글 하나는 삭제돼 설정 탐색 계산량은 확인할 수 없었다.

의사 라벨, Public 상위 제출 평균과 난수값 탐색은 현재 채택 계약과 맞지 않는다.
이 글은 Public 개선을 독립 근거로 사용하면 안 된다는 직접 반례다.

### 8위: Minato Namikaze

[#8 Private, #7 Public Solution Approach](https://www.kaggle.com/competitions/playground-series-s3e24/writeups/master-jiraiya-8-private-lb-7-public-solution-appr)는 단독 참가자 Minato Namikaze의 글이다.
목록에는 댓글 5개가 표시됐고 상세 화면에는 일반 댓글 5개와 감사 댓글 1개가 표시됐다.

작성자는 빈도가 2보다 큰 피처를 이산형으로 다루고 여러 부호화를 적용했다고 설명했다.
기존 열의 산술 조합을 무차별 생성한 뒤 CatBoost, XGBoost와 LightGBM의 중요도 상위 50개 및 100개 합집합을 사용했다.
더 많은 피처는 실행 시간 제한을 넘었다.

XGBoost, CatBoost, LightGBM, 신경망, 로지스틱 회귀와 DecisionTree를 Optuna 가중치로 합쳤다.
신경망 추가는 쓸 만한 이득이 있었다고 했지만 수치는 제공하지 않았다.
자체 결합을 공개 예측과 다시 합칠 때 Public 순위를 가중치로 사용했다.

실행 시간이 길어 여러 제출이 시간 제한으로 중단됐고 외부 예측을 쓰지 않는 노트북을 구분하려면 입력 자료 수를 확인하라고 조언했다.
최종 CV, Public과 Private 수치는 제공하지 않았다.
댓글은 축하뿐이었고 방법을 보충하거나 반증하는 내용은 없었다.

산술 조합과 나무 계열별 중요도 합집합은 현재 자체 피처 관점 및 외부 278개와 중복된다.
Public 순위 가중치는 현재 판정에 사용할 수 없다.

## S3E23: Binary Classification with a Software Defects Dataset

### 2위: Oscar Aguilar

[#2 Solution: 8 Models Ensemble](https://www.kaggle.com/competitions/playground-series-s3e23/writeups/oscar-aguilar-2-solution-8-models-ensemble)은 단독 참가자 Oscar Aguilar의 글이다.
목록과 상세 화면 모두 일반 댓글 42개를 표시했고 상세 화면에는 감사 댓글 8개도 표시됐다.

원시 입력의 초기 CV는 약 0.793, 리더보드는 약 0.790이었다.
모든 입력에 로그 변환을 적용하면 RandomForest와 여러 나무 부스팅을 포함한 대부분 나무 모형이 조금 좋아졌다고 보고했지만 변환 전후의 완전한 짝 수치는 제공하지 않았다.
RandomForest, ExtraTrees, HistGradientBoosting, LightGBM, XGBoost와 CatBoost 여섯 개를 탐욕 결합하면 리더보드가 0.7907이었다.

여섯 나무 결합에 Nyström 로지스틱 회귀를 더하면 0.79099, 신경망까지 더하면 0.79101이 됐다.
작성자는 이 제출을 고르지 않고 Public이 조금 더 높은 다른 결합을 선택했으므로 두 추가 모형의 Private 기여는 확인할 수 없다.
최종 Private 수치도 제공하지 않았다.

댓글에서 최종 세 묶음은 3:2:1 가중 평균이었다고 설명했다.
여섯 나무의 탐욕 결합은 `RepeatedStratifiedKFold` 25-fold를 다섯 번 반복했고 Nyström 로지스틱 회귀와 신경망은 10-fold를 다섯 번 반복했다.
다른 참가자는 공개 탐욕 결합 구현이 다음 S3E24에서도 잘 작동했다고 했지만 수치는 제공하지 않았다.
탐욕 결합 반복문의 세부 질문에는 답이 없었다.

PCA는 10개 성분이 분산의 99% 넘게 설명해도 성능을 올리지 못했다.
t-SNE, 군집과 군집 목표 부호화도 실패했다.
전체 접근 노트북은 공개됐지만 25-fold 5회 반복은 계산량이 크고 하드웨어는 적지 않았다.

Nyström 로지스틱 회귀는 자체 35개에 정확히 같은 모형이 없다.
그러나 외부 278개에는 선형, 다항식 및 RBF SVM이 모두 포함돼 같은 커널 관점을 직접 덮고, 이 글은 CV와 Private 한계 기여를 분리하지 못했다.
따라서 정확한 구현은 `근거 부족`이며 새 실험으로 열지 않는다.

## 교차 분석

### Public 탐색은 성공담 안에서도 반례가 더 강하다

S3E24 3위는 CV가 함께 오르는 범위에서 Public으로 결합 가중치를 조금 조정해 Private 3위를 했다고 보고했다.
그러나 S3E24 7위는 난수값 변경으로 Public을 올린 최종 제출이 과거 제출보다 Private에서 나빴고 작성자도 이를 선택 실수로 인정했다.
S3E23 2위는 커널 선형 모형과 신경망을 더한 결합의 Public이 올랐지만 더 높은 Public의 다른 결합을 선택해 해당 추가분의 Private 기여를 잃었다.
이 표본에서는 Public 탐색의 성공을 일반화할 수 없고 현재의 사후 참고 원칙을 유지하는 편이 타당하다.

### 탐욕 결합의 안정성은 모형 수보다 중요하다

S3E24 4위는 25개에서 일곱 개를 고를 때 모든 검증 fold 개선과 다른 난수값 반복을 요구했다.
S3E23 2위는 나무 여섯 개 뒤 커널 선형 모형과 신경망을 각각 더해 다른 함수 계열의 작은 이득을 노렸다.
반면 S3E24 8위와 7위는 Public 상위 예측을 직접 평균하거나 순위 가중치를 사용해 Private 하락 위험을 키웠다.
현재 313개 결합도 물량 자체가 아니라 출처 계보, 누출 제거와 nested 제외 기여로 판정해야 한다.

### 비선형 2단의 성공은 지표와 클래스 구조에 의존한다

S3E26 2위의 신경망 2단은 세 클래스마다 다른 가중치를 모든 입력 예측에서 계산해 단순 평균보다 log loss 0.004를 개선했다.
S3E26 4위는 OOF와 원시 피처를 함께 받은 XGBoost 2단으로 CV와 리더보드가 모두 좋아졌다고 했지만 수치는 제공하지 않았다.
현재 S6E8은 이진 ROC AUC이고 얕은 XGBoost 2단 및 잔차 2단이 자체 nested OOF에서 이기지 못했다.
과거 다중분류 성공만으로 현재 음성 결정을 다시 열 수 없다.

### 수치 표현의 새 이름보다 현재 정보 관점의 포괄 범위를 본다

piecewise linear encoding, 로그 변환, 최소최대 변환과 산술 교차는 구현 이름은 다르지만 수치 순서, 구간, 정확값과 상호작용을 표현하려는 방법이다.
현재 자체 35개에는 spline, Lookup, scalar-token, TabM, RealMLP과 여러 산술 및 원본 프록시 관점이 있다.
외부 278개에는 다양한 신경망, 나무, 커널 SVM과 Factorization Machine까지 들어 있다.
따라서 새 이름을 그대로 복제하기보다 현행 구성원 제거 대조에서 비어 있는 정보 관점인지 먼저 확인해야 한다.

### 댓글이 본문의 선택 편향을 드러냈다

S3E24 7위 댓글은 Public을 올린 난수값이 Private을 낮췄다는 본문의 의문을 선택 편향으로 해석했고 작성자가 이를 인정했다.
S3E23 2위 댓글은 본문에 없던 3:2:1 가중치와 25-fold 5회 반복 계산량을 공개했다.
S3E26 2위 댓글은 고유값 수에 따라 구간 수를 바꾼 사실과 다른 모형 및 원시 피처의 무익을 보충했다.
따라서 본문의 순위와 Public 개선만 읽을 때보다 댓글을 포함한 판단이 더 보수적이고 재현 가능하다.

## 현재 S6E8에 대한 적용 판단

현재 자체 후보 풀은 [후보 풀 장부](../../artifacts/pool.yaml)의 35개이며 [champion 장부](../../artifacts/champion.yaml)의 champion은 `exp156_lookup_bivariate_plr5_initavg8`, 3시드 평균 OOF AUC 0.969367610562다.
최신 외부 결합은 [두 번째 넓힌 확장 결합 기록](extended-stack-submission-2.md)의 실행 `443b3a71a2b045ba9052fbb3d821255d`다.
이 실행은 자체 35개와 외부 278개, 총 313개를 `shrunk_rank_logit_logistic`으로 결합해 nested OOF 0.9703509469와 가중 OOF 0.9712170271을 얻었다.
Public 0.97135는 사후 참고값일 뿐 채택 근거로 쓰지 않는다.

[기존 1년 조사와 실험 발주 기준인 이슈 307](https://github.com/tmheo/predicting-smartphone-addiction/issues/307)의 네 범주는 `새 후보`, `기존 결정과 중복`, `현재 과제에 부적합`, `근거 부족`이다.
아래 표는 자체 35개 안의 빈 관점인지와 외부 278개가 이미 덮는 관점인지를 분리한 예비 판정이다.

| 조사 관점 | 자체 35개 기준 | 외부 278개 기준 | 이슈 307 예비 분류 | 판단 |
| --- | --- | --- | --- | --- |
| 고유값 수에 따라 구간 수를 바꾸는 piecewise linear encoding | `exp085_contextual_spline_m0`, Lookup과 scalar-token 계열이 같은 구간 및 정확값 관점을 덮음 | 여러 표 신경망과 Lookup 변형이 있음 | 기존 결정과 중복 | 정확한 층만 다를 뿐 빈 정보 관점이 아님 |
| 모든 기초 예측을 보고 행별 및 클래스별 가중치를 만드는 신경망 2단 | 비선형 XGBoost 및 잔차 2단 nested OOF 음성, 결측 구간별 선형 결합은 양성 | 외부 2단 산출물은 원칙상 들이지 않고 기초 OOF만 사용 | 기존 결정과 중복 | 다중분류 log loss 성공으로 이진 AUC 음성 결정을 뒤집지 않음 |
| OOF와 원시 피처를 함께 받는 XGBoost 2단 | 자체 비선형 2단과 원시 피처 재결합이 이미 음성 | 2단 외부 산출물은 현행 계보 범위 밖 | 기존 결정과 중복 | 새 행별 체제가 독립적으로 증명되기 전에는 재개 안 함 |
| 모든 검증 fold 개선과 다른 난수값 반복을 요구하는 탐욕 결합 | nested 바깥쪽 분할과 경계 구간 3/5 승리 조건이 더 엄격함 | 최신 결합도 외부 부류별 절제와 nested 사다리를 거침 | 기존 결정과 중복 | 새 결합기보다 현행 규율을 지지하는 근거 |
| Nyström 로지스틱 회귀 | one-hot 로지스틱 회귀는 있으나 정확한 커널 근사는 없음 | 선형, 다항식 및 RBF SVM이 모두 포함됨 | 근거 부족 | Public만 개선했고 선택되지 않아 CV 및 Private 한계 기여가 없음 |
| 전 피처 로그 변환 | 직접 같은 구성원은 없지만 spline, 분위 변환과 신경망 수치 표현이 인접 | 여러 전처리 및 나무 OOF가 있으나 정확한 로그 여부는 장부에서 불명 | 근거 부족 | 단조 변환이 나무를 개선한 이유와 짝 수치가 없어 재실험 근거 부족 |
| 산술 피처 무차별 생성과 모형별 중요도 합집합 | 산술 잔차, 비율, 원본 프록시와 선택된 재구성 피처가 다수 있음 | 화면 관계 및 여러 공개 피처 묶음이 있음 | 기존 결정과 중복 | 현재 제거 대조가 새 피처 수보다 강한 근거 |
| 원본 자료 행 추가 | 원본 최근접, 원본 프록시 잔차, 평균과 CDF 계열이 있음 | 원본을 쓴 공개 OOF가 다수 있음 | 기존 결정과 중복 | 현행 정보 관점과 중복 |
| 의사 라벨과 시험 행 최대 85% 추가 | 자체 음성 결정과 엄격한 중첩 비용 문제가 있음 | 외부 구성원의 시험 의사 목표 계보는 허용하지 않음 | 기존 결정과 중복 | Public 개선만 있고 Private 선택 반례가 있어 재개 안 함 |
| Public 순위 가중치, 제출 탐색과 난수값 선택 | Public은 사후 참고만 허용 | 외부 예측도 Public 점수로 구성원 채택하지 않음 | 현재 과제에 부적합 | 현행 채택 계약을 직접 위반함 |
| PCA, t-SNE, 군집과 군집 목표 부호화 | 자체 후보 풀의 빈 관점일 수 있으나 이번 글에서 모두 실패 | 직접 대응 여부와 무관하게 양성 근거 없음 | 근거 부족 | 실패 보고만 있어 열지 않음 |

### 새 실험을 열지 않는 이유

이번 조사에는 이슈 307의 `새 후보`로 분류할 방법이 없다.
piecewise linear encoding은 현재 수치 구간 표현과 중복되고 Nyström 로지스틱 회귀는 외부 커널 관점이 이미 있으며 정확한 방법의 CV 및 Private 기여가 없다.
강한 순위 근거가 있는 나머지 방법은 대규모 OOF 결합, 원본 자료, 산술 피처와 비선형 2단으로 현행 채택 또는 음성 결정에 포함된다.

추가 자료가 생긴다면 Nyström 로지스틱 회귀만 먼저 외부 선형, 다항식 및 RBF SVM 세 구성원의 부류 제외 기여와 비교한다.
외부 커널 부류가 양의 한계 기여를 이미 만들면 정확한 Nyström 구현을 자체 재현할 이유가 없다.
외부 커널 부류가 무익한데 현재 S6E8의 같은 분할 OOF와 독립 Private 없는 대조가 새로 공개되는 경우에만 한 설정의 진입 진단을 다시 논의한다.

후보 풀 진입과 최종 결합 채택은 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)을 그대로 따른다.
새 후보는 자체 3시드 OOF, champion 대비 0.01 하한, 0.998 스피어만 중복 검사와 현재 풀 포함 전후의 nested 결합 기여를 통과해야 한다.
최종 결합 교체는 champion 대비 nested OOF 0.00002 이상이어야 하고 0.00002 이상 0.0002 미만이면 바깥쪽 분할 다섯 개 중 셋 이상에서 이겨야 한다.

### 열지 않을 방법

- Public 상위 제출의 평균, Public 순위 가중치와 제출 결과에 맞춘 수동 가중치 탐색을 사용하지 않는다.
- 난수값 하나의 Public 상승을 성능 개선으로 판정하지 않는다.
- 의사 라벨 비율과 Public 점수만 보고 시험 행을 훈련에 넣지 않는다.
- 다중분류용 클래스별 신경망 가중치와 원시 피처 XGBoost 2단을 현재 이진 AUC에 다시 열지 않는다.
- 25-fold 5회 반복을 안정성의 기본값으로 삼지 않고 현행 3시드, 공통 5-fold와 nested 바깥쪽 검증을 유지한다.
- 로그 변환, PCA, t-SNE와 군집을 짝지은 OOF 근거 없이 다시 열지 않는다.
- 공개 OOF를 사용할 때 출처별 fold, 시험 예측 쌍, 누출 및 라이선스 장부 요구를 완화하지 않는다.
- 공개 노트북 코드를 재사용할 때는 Apache License 2.0 출처 절차를 따르고 입력 자료, 사전 학습 모형, 패키지와 외부 자산의 라이선스를 별도로 확인한다.

## 사실과 추론의 경계

각 대회 절의 순위, 작성자, 피처, 모형, 검증, 점수, 댓글 보충과 실패 사례는 링크된 Kaggle 공식 해법 본문과 댓글에서 확인한 작성자 보고 사실이다.
댓글의 재현 경험과 선택 편향 지적도 해당 댓글 작성자의 보고이며 이 저장소에서 다시 실행한 값은 아니다.
S3E23의 Hotness, Newest와 Oldest 정렬은 같은 댓글의 다른 화면 순서이므로 중복을 새 댓글로 세지 않았다.
삭제 댓글의 내용은 사실로 복원하지 않았다.

재현성, 현재 과제와의 유사성, 자체 35개 및 외부 278개에 대한 겹침, 이슈 307의 네 범주와 실험 우선순위는 조사자의 추론이다.
외부의 선형, 다항식 및 RBF SVM이 Nyström 로지스틱 회귀와 같은 커널 관점을 덮는다는 판단은 함수 계열의 유사성에 관한 구조상 추론이지 동일 구현이라는 주장이 아니다.
모든 피처 로그 변환이 나무 계열을 개선했다는 과거 보고는 짝 수치와 현재 자료 재현이 없어 현재 S6E8의 채택 근거로 사용하지 않았다.
과거 대회의 Public과 Private은 후보 발굴의 참고 근거일 뿐 현재 S6E8의 채택 근거가 될 수 없다.

## 한계

조사는 Kaggle 공식 화면이 현재 렌더링한 본문과 댓글을 대상으로 했으므로 삭제된 댓글의 과거 내용은 복구하지 않았다.
Kaggle 목록, 상세 머리말과 정렬별 렌더링 댓글 수가 달라 원시 표시값을 따로 보존했다.
글 안의 그림으로만 제시된 일부 점수는 본문이 직접 설명한 값만 기록했고 원자료로 다시 계산하지 않았다.
공개 OOF와 공개 노트북의 내부 모형 전체 계보는 작성자가 설명한 범위를 넘어 복원하지 않았다.
이 문서는 새 실험을 실행하거나 GitHub 이슈를 편집하지 않았다.
