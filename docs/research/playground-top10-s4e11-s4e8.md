# Playground Series S4E11부터 S4E8까지 상위 10위 해법과 댓글 조사

이 문서는 GitHub 이슈 [리서치: S4E11-S4E8 상위권 해법 글과 댓글 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/475)의 조사 결과다.
조사 기준일은 2026-08-28이다.

## 결론

네 대회의 공식 해법 범주에 올라온 글 25개를 모두 확인했고, 제목이나 본문에서 Private 최종 1위부터 10위가 확인된 글은 17개였다.
포함 순위는 S4E11의 1위와 4위, S4E10의 1위, 2위, 4위, 8위, 10위, S4E9의 1위부터 5위, S4E8의 1위, 3위, 6위, 8위, 10위다.
없는 순위를 다른 범주의 글이나 10위 밖 글로 채우지 않았다.

현재 S6E8에 바로 추가할 만큼 근거가 갖춰진 새 후보는 발견하지 못했다.
AutoGluon과 LightAutoML의 대규모 자동 결합, 여러 모형의 OOF 수집, 선형 또는 탐욕 결합, 범주형 전용 모형, 원본 자료 활용, 잔차 관점은 자체 35개 후보나 최신 외부 278개 결합이 이미 직접 또는 인접하게 덮는다.
회귀의 목표값 꼬리 확률, 대출 위험 교차 피처, 버섯 문자열 정제는 현재 이진 ROC AUC 과제에 그대로 옮길 수 없다.

가장 가까운 조건부 신규 관점은 S4E10 1위의 OOF 로짓을 CatBoost의 시작 기준값으로 주어 각 기초 모형을 잔차 보정한 방법이다.
자체 35개 안의 `exp023_orig_proxy_residual`은 원본 자료 프록시 하나를 시작 로짓으로 쓰므로 정확히 같은 방법은 아니다.
외부 278개에는 `view_resid_cat`, `view_resid_lgbm`, `view_resid_xgb`라는 인접 구성원이 있지만 장부만으로 CatBoost의 `baseline` 방식과 같다고 확인할 수 없다.
다만 독립 재현이 없고, 현재 과제의 비선형 2단 결합과 잔차 보정 선행 결과가 약하거나 음성이므로 이슈 307의 네 범주 가운데 `근거 부족`으로 남긴다.

댓글에서 가장 중요한 검증 반례도 새 모형보다 검증 규율에 관한 것이었다.
S4E11 1위 글의 다층 결합 누출 논쟁에서 AutoGluon 개발자는 같은 fold를 공유해도 상위 단계 학습 행의 기초 OOF를 만든 모형이 바깥 검증 행을 보았을 수 있어 미세한 오염이 생긴다고 확인했다.
현재 저장소의 nested OOF 계약은 이 위험을 이미 더 엄격하게 막으므로, 과거 글의 다층 결합 점수를 그대로 재현 대상으로 삼지 않는다.

## 조사 범위와 방법

Kaggle API, 웹 검색 결과 요약, 직접 HTTP 요청과 리더보드 조회는 사용하지 않았다.
`agent-browser`의 `issue475` 전용 세션을 만들고 콘텐츠 경계를 켠 뒤 Kaggle 도메인만 허용했다.
사용자가 지정한 공식 `competitionWriteUps` 화면에서 대회 식별자만 S4E11, S4E10, S4E9, S4E8로 바꾸어 조사했다.
목록 정렬은 `recent-comments`를 유지했고, 표본 포함 여부는 글 제목이나 본문의 Private 최종 순위 표기로만 정했다.

각 포함 글은 본문 끝까지 읽고 댓글 내부 스크롤의 마지막까지 이동했다.
`more replies`가 나타난 S4E11 1위의 답글 3개, S4E9 4위의 답글 5개, S4E8 3위의 답글 3개, S4E8 6위의 답글 4개를 모두 펼친 뒤 다시 끝까지 확인했다.
별도의 다음 댓글 쪽이나 더 불러오기 단추는 어느 포함 글에도 남지 않았다.
삭제된 댓글은 S4E11 1위에 세 개, S4E10 1위에 한 개, S4E8 10위에 한 개가 있었고 삭제 표시만 확인할 수 있었다.

Kaggle 목록의 댓글 수와 상세 글 머리말의 일반 댓글 및 감사 댓글 수는 서로 일치하지 않았다.
숫자를 임의로 맞추지 않고 두 화면의 표시값을 따로 기록했다.
포함 글의 목록 댓글 수 합계는 645개였고, 상세 글 머리말의 일반 댓글 수 합계는 540개였으며, 별도 감사 댓글 표시는 39개였다.
S4E11 4위 글처럼 머리말은 일반 댓글 6개라고 표시하지만 펼쳐진 화면에는 답글을 포함한 댓글 노드 8개가 보이는 사례도 있었다.
집계 차이의 원인은 공식 화면에서 설명하지 않으므로 추정하지 않는다.

## 표본 장부

| 대회 | 과제와 평가지표 | 공식 해법 글 | 포함 순위 | 결측 순위 | 목록 댓글 합 | 상세 일반 댓글 | 감사 댓글 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| S4E11 | 우울 여부 이진 분류, Accuracy | 6 | 1, 4 | 2, 3, 5, 6, 7, 8, 9, 10 | 99 | 82 | 8 |
| S4E10 | 대출 승인 여부 이진 분류, ROC AUC | 6 | 1, 2, 4, 8, 10 | 3, 5, 6, 7, 9 | 186 | 170 | 23 |
| S4E9 | 중고차 가격 회귀, RMSE | 7 | 1, 2, 3, 4, 5 | 6, 7, 8, 9, 10 | 190 | 156 | 6 |
| S4E8 | 독버섯 여부 이진 분류, MCC | 6 | 1, 3, 6, 8, 10 | 2, 4, 5, 7, 9 | 170 | 132 | 2 |

과제와 평가지표는 Kaggle의 [S4E11 평가](https://www.kaggle.com/competitions/playground-series-s4e11/overview/evaluation), [S4E10 평가](https://www.kaggle.com/competitions/playground-series-s4e10/overview/evaluation), [S4E9 평가](https://www.kaggle.com/competitions/playground-series-s4e9/overview/evaluation), [S4E8 평가](https://www.kaggle.com/competitions/playground-series-s4e8/overview/evaluation) 화면에서 확인했다.

## 제외 장부

[S4E11 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s4e11/discussion?sort=recent-comments&category=competitionWriteUps)에서는 35위 Baseer Shah, 25위 Chris Deotte, 13위 Optimistix, 438위 Vasco의 글을 제외했다.
[S4E10 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s4e10/discussion?sort=recent-comments&category=competitionWriteUps)에서는 299위 Optimistix의 글을 제외했다.
[S4E9 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s4e9/discussion?sort=recent-comments&category=competitionWriteUps)에서는 239위와 81위 글을 제외했다.
[S4E8 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s4e8/discussion?sort=recent-comments&category=competitionWriteUps)에서는 19위 Oscar의 글을 제외했다.
제목에 잠재 순위를 적은 글도 실제 Private 최종 순위가 10위 밖이면 포함하지 않았다.
접근할 수 없는 포함 글은 없었다.

## S4E11: Exploring Mental Health Data

### 1위: Mahdi Ravaghi

[1st Place Solution](https://www.kaggle.com/competitions/playground-series-s4e11/writeups/mahdi-ravaghi-1st-place-solution)은 단독 참가자 Mahdi Ravaghi의 글이다.
목록에는 댓글 91개가 표시됐고 상세 화면에는 일반 댓글 76개와 감사 댓글 8개가 표시됐다.

작성자는 XGBoost, 세 종류 LightGBM, 두 AutoGluon 구성을 여러 전처리와 원본 자료 포함 여부로 변형해 69개 모형을 만들고 24개를 최종에 썼다.
피처 생성이나 열 제거는 하지 않았고 `Name`도 유지했다.
공개 노트북에서 가져온 OOF와 자체 OOF를 AutoGluon으로 결합했으며, AutoGluon 두 구성이 가장 강하고 중요했다고 추정했다.
탐욕 결합, Ridge와 로지스틱 회귀는 자신의 CV에서 더 나빴다고 보고했다.
최종 선택 제출은 CV Accuracy 0.94173, Public 0.94284였고 다른 제출은 CV 0.94150, Public 0.94285였으며 Private 수치는 적지 않았다.

댓글에서 작성자는 AutoGluon OOF를 `predict_proba_oof`로 저장했다고 설명했다.
개별 AutoGluon 구성원의 OOF는 과적합하기 쉬워 가중 결합 OOF만 사용했다고도 했다.
합성 자료의 이상값을 과도하게 정리하거나 피처를 늘리면 생성 잡음을 키울 수 있으므로 기본적으로 원형을 유지했다는 설명은 같은 대회 4위의 정제 주장과 충돌한다.

가장 중요한 댓글은 다층 결합의 누출 논쟁이다.
Hugo는 같은 fold를 써도 상위 단계 학습 행의 OOF를 만든 기초 모형이 바깥 검증 행을 학습에 썼다면 엄밀한 바깥 검증이 아니라고 지적했다.
작성자는 같은 fold이므로 안전하다고 답했지만 AutoGluon 개발자 Nick Erickson은 Hugo의 지적이 맞고 자료에 따라 영향이 무해할 수도 치명적일 수도 있다고 확인했다.
Nick Erickson은 AutoGluon의 Dynamic Stacking이 이 문제를 감지해 해로운 다층 결합을 끌 수 있다고 설명했다.

이 글은 넓은 자동 모형 묶음의 효용을 보여 주지만 최종 24개 계보와 완전한 실행 코드는 제공하지 않았다.
무엇보다 댓글이 확인한 오염 위험 때문에 보고된 AutoGluon CV를 현재 nested OOF와 같은 등급의 근거로 볼 수 없다.

### 4위: Jack Lee

[4th Place Solution: Preprocess + AutoML](https://www.kaggle.com/competitions/playground-series-s4e11/writeups/jack-lee-4th-place-solution-preprocess-automl)은 단독 참가자 Jack Lee의 글이다.
목록에는 댓글 8개가 표시됐고 상세 화면 머리말에는 일반 댓글 6개가 표시됐다.

작성자는 열 사이 관계로 보아 불가능하거나 비정상인 값을 결측으로 바꾼 뒤 AutoGluon의 2024 설정 묶음 약 200개와 LightAutoML의 LightGBM, CatBoost, 다층 신경망, DenseLight, ResNet, SNN, NODE, AutoInt, FT-Transformer 계열을 학습했다.
최종 제출은 AutoGluon과 LightAutoML의 같은 가중치 평균이었다.
작성자는 값 정리가 CV, Public과 Private을 모두 개선했다고 보고했지만 전후 수치는 제공하지 않았다.

AutoGluon은 CV가 더 높아도 실험적 `experimental_quality` 설정의 이전 판보다 Private이 조금 낮았고, LightAutoML은 CV가 조금 낮아도 Public과 Private이 조금 높았다고 했다.
댓글에서는 LightAutoML 신경망이 AutoGluon 신경망보다 대체로 강했다고 답했다.
Public을 보고 가중치를 고르지 않고 CV를 신뢰해 같은 가중치를 썼다는 설명도 남겼다.

값 정리의 방향은 1위 글의 원형 유지 주장과 정반대이며 독립 대조가 없다.
현재 과제에 옮기려면 구체적인 규칙별 fold 밖 대조가 필요하다.

## S4E10: Loan Approval Prediction

### 1위: Hardy Xu

[1st Place Solution: CatBoost All the Way](https://www.kaggle.com/competitions/playground-series-s4e10/writeups/hardy-xu-1st-place-solution-catboost-all-the-way-d)은 단독 참가자 Hardy Xu의 글이다.
목록에는 댓글 105개가 표시됐고 상세 화면에는 일반 댓글 95개와 감사 댓글 17개가 표시됐다.

작성자는 수치 열과 범주형 복사본을 함께 두고 원본 자료를 추가했다.
LightGBM, XGBoost와 CatBoost는 Optuna로 서로 다른 설정 10개씩을 만들었고 신경망 하나도 사용했다.
기본 모형의 CV, Public, Private은 LightGBM 0.96811, 0.97005, 0.96637, XGBoost 0.96767, 0.96989, 0.96540, CatBoost 0.96972, 0.97299, 0.96865, 신경망 0.96678, 0.97088, 0.96577이었다.

핵심 방법은 각 기초 모형의 OOF 확률을 로짓으로 바꾸어 CatBoost의 시작 기준값으로 주고 남은 오차를 학습하는 것이었다.
보정 뒤 CV, Public, Private은 LightGBM 0.96856, 0.97048, 0.96713, XGBoost 0.96815, 0.97024, 0.96611, CatBoost 0.96997, 0.97334, 0.96903, 신경망 0.96732, 0.97117, 0.96667이었다.
네 보정 예측을 작은 신경망으로 합친 최종 결과는 CV 0.97059, Public 0.97344, Private 0.96938이었다.

댓글에서 시작 기준값은 새 피처나 라벨이 아니라 CatBoost가 첫 반복 전에 출발할 원시 점수라고 명확히 했다.
확률이 아니라 로짓을 줘야 하며 0 또는 1로 자르지 않았다고도 했다.
설정 탐색은 5-fold를 두 번 반복한 10개 분할마다 별도 Optuna 연구를 만들고, 그 설정들을 다시 다른 k-fold OOF 생성에 썼다고 설명했다.
조기 종료는 대체로 log loss를 썼고 AUC는 설정 평가에 썼다.
GBM에는 결측 대치를 하지 않았고 신경망만 중앙값으로 채웠으며 복잡한 대치는 이득이 없었다고 보고했다.

기초 모형 네 종류 모두에서 보정 전후 수치가 같은 방향이라 단일 우승 글 안의 대조는 강하다.
그러나 엄밀한 바깥 중첩 분할에서 기준 OOF와 보정 모형을 다시 만드는 절차는 보고되지 않았으므로 현재 계약에 맞는 독립 근거는 아니다.

### 2위: Omid Baghcheh Saraei

[2nd Place Solution](https://www.kaggle.com/competitions/playground-series-s4e10/writeups/omid-baghcheh-saraei-2nd-place-solution)은 단독 참가자 Omid Baghcheh Saraei의 글이다.
목록에는 댓글 22개가 표시됐고 상세 화면에는 일반 댓글 19개와 감사 댓글 1개가 표시됐다.

작성자는 CatBoost, XGBoost, LightGBM, HistGradientBoosting, GOSS, DART, GradientBoosting, ExtraTrees, RandomForest, 여러 신경망, TabNet, FastAI, k-NN과 PerpetualBooster를 포함한 넓은 모형 묶음을 만들었다.
공개 그림에서 최고 단일 모형은 CatBoost 구성의 CV 0.96944였고 최종 21개 결합은 CV 0.97107, Public 0.97217이었다.
다른 24개 결합은 CV 0.97026, Public 0.97335였으며 Private 수치는 적지 않았다.

탐욕 결합은 양수와 음수 가중치를 모두 허용했다.
댓글에서 `default_on_file`과 `loan_grade`를 교차한 위험 표시 피처를 두세 모형에만 넣어 다양성을 만들었다고 설명했다.
Mahdi Ravaghi는 같은 대회에서 CV와 조기 종료를 붙인 탐욕 결합도 자신에게는 작동하지 않았다고 반례를 남겼다.

넓은 모형 묶음과 부호가 있는 탐욕 결합은 현재 자체 및 외부 결합에서 이미 검토한 관점이다.
대출 도메인 피처는 현재 자료에 대응 열이 없다.

### 4위: Ravi Ramakrishnan

[Rank 4 Approach: Thoughtful Model Selection](https://www.kaggle.com/competitions/playground-series-s4e10/writeups/ravi-ramakrishnan-rank-4-approach-thoughtful-model)은 단독 참가자 Ravi Ramakrishnan의 글이다.
목록에는 댓글 41개가 표시됐고 상세 화면에는 일반 댓글 40개와 감사 댓글 4개가 표시됐다.

작성자는 난수값 42의 층화 10-fold를 공통으로 쓰고, 기본 열에 15개부터 25개의 단순 교차 피처를 더한 세 피처 묶음을 포함해 다섯 피처 묶음을 만들었다.
50개 넘는 피처는 더 나빴고 OpenFE도 과거 실험에서 실패했다고 했다.
XGBoost, AutoGluon, 로지스틱 회귀, RandomForest와 k-NN은 자신의 설정에서 도움 되지 않아 버리고 서로 다른 나무와 신경망을 골랐다.
기초 예측의 OOF를 작은 PyTorch 신경망으로 합친 최종 결과는 CV 0.97002, Public 0.97393, Private 0.96902였고 공개 예측을 뺀 독립 구성은 0.969954, 0.97353, 0.96899였다.

댓글에서 피처는 단순 교차를 많이 만든 뒤 CV로 골랐다고 설명했다.
작성자는 알고리즘, 설정과 CV 점수가 서로 다르면 다양하다고 보았지만 Tilii는 진짜 다양성은 서로 다른 행에서 잘 맞히는 상보성이라고 바로잡았다.
작성자는 최종 선택에 운이 작용했다고 인정했다.

현재 35개 후보는 정보 관점과 제외 기여를 함께 판정하므로 단순 모형 종류 차이보다 엄격하다.
대량 교차 피처의 반복 CV 선택은 선택 편향을 더할 수 있어 새 후보 근거가 아니다.

### 8위: Mahdi Ravaghi

[8th Place Solution](https://www.kaggle.com/competitions/playground-series-s4e10/writeups/mahdi-ravaghi-8th-place-solution)은 단독 참가자 Mahdi Ravaghi의 글이다.
목록에는 댓글 7개가 표시됐고 상세 화면에는 일반 댓글 6개와 감사 댓글 1개가 표시됐다.

작성자는 다섯 전처리 묶음을 만들고 그중 세 묶음에는 원본 자료를 넣었다.
CatBoost에는 모든 피처를 범주형으로 주었고 XGBoost, LightGBM의 GBDT, DART, GOSS, HistGradientBoosting, GradientBoosting, AutoGluon과 신경망을 조합했다.
52개 OOF를 AutoGluon으로 24시간 결합한 결과는 CV 0.970887, Public 0.97329, Private 0.96900이었다.

대안은 여러 단계 Ridge와 로지스틱 회귀를 만든 뒤 가중 평균하는 방식이었다.
두 결합기마다 19개 OOF를 무차별 대입으로 골랐고 RFECV는 더 나빴으며 Private은 같은 수준이었다.
현재 외부 278개와 순위 로짓 로지스틱 결합은 이 관점을 더 큰 현행 자료에서 이미 다룬다.

### 10위: aldparis

[10th Place Solution: No Blind Blend](https://www.kaggle.com/competitions/playground-series-s4e10/writeups/aldparis-10th-place-solution-no-blind-blend)은 단독 참가자 aldparis의 글이다.
목록에는 댓글 11개가 표시됐고 상세 화면에는 일반 댓글 10개가 표시됐다.

작성자는 30개 넘는 XGBoost, CatBoost와 LightGBM을 만들고 네 개의 2단 결합기로 묶은 뒤 로지스틱 회귀로 다시 합쳤다.
공개 그림의 2단 결합 CV는 CatBoost 0.96998, 두 LightGBM 0.96750과 0.96749, XGBoost 0.96678이었고 최종은 0.97041이었다.
같은 절차를 네 난수값으로 반복했다.
수치와 범주형 복사본을 함께 쓰고 특히 소득 열을 여러 형태로 표현했으며 원본 자료는 훈련에만 넣었다.

댓글에서 각 구현의 기본 성장 정책 차이가 결합 다양성을 만들었다고 설명했다.
공개 신경망을 더하는 것은 이득이 없었다.
여러 나무 계열 2단 결합과 최종 선형 결합은 현재 결합 및 외부 구성원이 이미 덮는 관점이다.

## S4E9: Regression of Used Car Prices

### 1위: Mart Preusse

[#1 Solution: Stacked NN](https://www.kaggle.com/competitions/playground-series-s4e9/writeups/mart-preusse-1-solution-stacked-nn)은 단독 참가자 Mart Preusse의 글이다.
목록에는 댓글 92개가 표시됐고 상세 화면에는 일반 댓글 83개와 감사 댓글 4개가 표시됐다.

작성자는 20-fold를 공통으로 쓰고 RBF SVR, LightGBM, CatBoost, XGBoost, AutoGluon FastAI와 신경망을 만들었다.
CatBoost로 가격 꼬리 이상 여부를 분류하고 그 확률을 LightGBM이나 신경망의 연속 입력 피처로 넣은 점이 독특하다.
최종 우승판은 SVR, 강한 LightGBM, 이상 확률과 XGBoost OOF를 받은 DeepTables 계열 신경망이었다.
Ridge 결합은 CV RMSE 72,300으로 2위 수준이었고 최종 신경망은 CV 72,468로 더 나빴지만 Private에서 앞섰다.

댓글에서 이상 분류 출력은 문턱으로 자르지 않은 확률이며, 추가하면 CV도 좋아졌다고 설명했다.
시험 자료는 훈련보다 가격 꼬리 행이 적었고 작성자는 예측 상자 그림에서 극단값이 덜한 신경망을 선택한 점도 밝혔다.
Tilii는 이 자료에서는 20-fold가 더 좋았다고 했고 다른 참가자들은 서로 다른 CV 체계를 직접 비교할 수 없다고 지적했다.

원본 자료를 AutoGluon의 `pseudo_data`로 넣었다는 초기 설명은 댓글 뒤 수정됐다.
해당 호출은 실제 의사 라벨 학습을 하지 않았고 `fit_pseudolabel`을 써야 한다고 작성자가 인정했다.
AutoGluon 개발자는 `pseudo_data`가 첫 단계에만 들어가고 검증이나 상위 단계에는 쓰이지 않아 제거될 수 있다고 설명했다.

목표값 꼬리 확률은 연속 회귀 목표를 하위 과제로 분해한 방법이다.
현재 이진 목표에는 같은 의미의 꼬리가 없으므로 직접 적용할 수 없다.

### 2위: Gerlando Re

[2nd Position: Just Feature Engineering and AutoML](https://www.kaggle.com/competitions/playground-series-s4e9/writeups/gerlando-re-2-position-just-fe-and-automl)은 단독 참가자 Gerlando Re의 글이다.
목록에는 댓글 10개가 표시됐고 상세 화면에는 일반 댓글 9개가 표시됐다.

작성자는 변속기 표기 통합, 고급 브랜드 표시, 엔진 마력과 실린더, 주행 거리와 연식, 희소 범주 및 결측 처리 같은 중고차 전용 피처를 만들었다.
CatBoost와 LightGBM을 Optuna로 맞추고 원본 자료도 넣었으며 AutoGluon 가중 결합이 가장 좋았다고 보고했다.

댓글에서 피처별 대조가 없다는 지적과 희소 범주 분위수 규칙 질문이 나왔지만 작성자는 원본 자료 출처만 답하고 기여 수치는 제시하지 않았다.
도메인 피처도 현재 자료에 대응하지 않으므로 적용 근거가 없다.

### 3위: Optimistix

[3rd Place Solution: Gather OOFs and Blend Them](https://www.kaggle.com/competitions/playground-series-s4e9/writeups/optimistix-3rd-place-solution-an-open-secret-gathe)은 단독 참가자 Optimistix의 글이다.
목록에는 댓글 44개가 표시됐고 상세 화면에는 일반 댓글 29개가 표시됐다.

작성자는 공개 외부 OOF까지 77개를 모았고 Ridge, AutoGluon과 탐욕 결합을 반복했다.
초기 30개 OOF Ridge는 CV 72,018, Public 71,958, Private 63,001이었고 AutoGluon과 탐욕 결합 변형은 CV가 약간 나빠도 Private이 비슷하거나 더 좋았다.
48개 OOF를 쓴 숨은 제출은 2위 가능성이 있었고 41개 CPU AutoGluon 숨은 제출은 1위 가능성이 있었다고 사후 보고했다.
KaggleX GAN 자료는 Public만 돕고 Private에는 도움 되지 않았다.

댓글에서 서로 다른 fold와 분할의 OOF를 섞으면 다양성이 생길 수도 있지만 CV를 흐릴 수도 있으므로 가능하면 섞지 않는 편이 낫다고 작성자가 답했다.
외부 예측은 OOF와 시험 예측 쌍이 함께 있어야 한다고 강조했다.
현재 외부 278개 장부는 바로 이 계보 검사를 별도로 수행하고 있으므로 방법론은 중복이다.

### 4위: Tilii

[#4 Solution: Beating a Dead Horse, Blending Works](https://www.kaggle.com/competitions/playground-series-s4e9/writeups/tilii-4-solution-beating-a-dead-horse-blending-wor)은 단독 참가자 Tilii의 글이다.
목록에는 댓글 41개가 표시됐고 상세 화면에는 일반 댓글 32개와 감사 댓글 2개가 표시됐다.

최종 32개에는 Keras Factorization Machine, xLearn Factorization Machine, Lasso, CatBoost, LightAutoML 신경망, AutoGluon, RandomForest, ExtraTrees, XGBoost, LightGBM과 FastAI가 들어갔다.
첫 네 모형은 모든 열을 범주형으로 취급했고 원본 자료를 약 20개 모형에 넣었다.
피처 생성은 CV를 조금 올렸지만 리더보드를 올리지 않아 버렸다.
탐욕 결합 32개 가운데 절반가량은 공개 예측이었고 작성자 자체 모형은 하나뿐이었다.

댓글과 펼친 답글 다섯 개에서 AutoGluon의 `predict_proba_multi`를 자료 없이 부르면 OOF를, 자료와 함께 부르면 일반 예측을 반환한다는 사용법을 설명했다.
원본 자료와 합쳐 학습한 경우에는 OOF에서 원본 행을 제거한 뒤 대회 행만 다시 평가했다고 했다.
범주형 Factorization Machine의 단독 강점과 외부 OOF 수집은 외부 278개에 Factorization Machine 다섯 개와 DeepFM이 포함된 현재 구성에서 이미 직접 덮인다.

### 5위: AutoML Grandmasters

[#5 Solution: AutoGluon Submission from First Day](https://www.kaggle.com/competitions/playground-series-s4e9/writeups/automl-grandmasters-5-solution-autogluon-submissio)은 팀 AutoML Grandmasters의 글이다.
목록과 상세 화면 모두 일반 댓글 3개를 표시했다.

대회 첫날 만든 사전 배포 AutoGluon 제출은 Public RMSE 72,221로 626위였지만 Private 5위가 됐다.
두 번째 단계 결합기는 원래 피처를 빼고 기초 예측만 받도록 `use_orig_features=False`로 설정했다.
목표값의 반복된 정확값으로 층화했지만 작성자는 연속 목표를 구간화했어야 한다고 사후 평가했다.
실험적 2024 무설정 모형 묶음을 썼고 n-gram과 텍스트 처리는 껐다.

시험 자료의 이상값 밀도가 달라 순위가 크게 뒤집혔다고 해석했다.
원시 피처를 빼고 OOF만 받는 상위 선형 결합은 현재 nested 결합의 기본 구조와 중복이다.

## S4E8: Binary Prediction of Poisonous Mushrooms

### 1위: Optimistix

[1st Place Solution: 72 OOFs and a Whole Lot of Ensembling](https://www.kaggle.com/competitions/playground-series-s4e8/writeups/optimistix-1st-place-solution-72-oofs-a-whole-lott)은 단독 참가자 Optimistix의 글이다.
목록에는 댓글 65개가 표시됐고 상세 화면에는 일반 댓글 54개와 감사 댓글 1개가 표시됐다.

작성자는 약 80개 OOF를 만들고 72개를 사용했다.
원본 버섯 자료의 독성 확률 프록시, RandomForest와 ExtraTrees, 여러 나무와 신경망, AutoGluon, Ridge와 탐욕 결합을 폭넓게 사용했다.
66개 OOF 결합은 CV MCC 0.985087, Public 0.98533, Private 0.98513이었고 탐욕 결합이 고른 여덟 OOF와 강한 AutoGluon을 다시 AutoGluon으로 합치면 CV 0.985124, Public 0.98532, Private 0.98516이었다.
72개 CPU AutoGluon은 Public 0.98535, Private 0.98512였고 외부 32개 CPU 판은 Public 0.98535, Private 0.98513이었다.

Public에서 자신 있게 불일치한 행만 이용한 혼합은 Public 0.98535였지만 Private 0.98506으로 실패했다.
두 견고한 제출을 같은 가중치로 합친 숨은 제출은 Private 0.98517이었다.
댓글에서 합성 자료에서는 피처 생성이 대체로 약했고, OOF를 AutoGluon 입력으로 직접 넣었으며, Public 기반 맹목 혼합이 실패했다고 재확인했다.

원본 목표 확률 프록시와 대규모 외부 OOF 결합은 현재 자체 `exp022`, `exp023`, 여러 Lookup 계열과 외부 278개가 이미 덮는다.
Public 불일치 선택은 현재 계약에서 허용되지 않는다.

### 3위: AutoML Grandmasters

[AutoML Grandmasters Solution](https://www.kaggle.com/competitions/playground-series-s4e8/writeups/automl-grandmasters-1st-place-solution-automl-gran)은 글 제목에 1위라고 적혀 있지만 공식 해법 화면 머리말은 `Solution Writeup · 3rd place`로 표시하므로 3위 표본으로 포함했다.
목록에는 댓글 65개가 표시됐고 상세 화면에는 일반 댓글 46개와 감사 댓글 1개가 표시됐다.

작성자는 시험 자료에 나타나지 않는 훈련 범주를 결측으로 바꾸는 시험 어휘 기반 전처리를 했다.
16-fold, 각 단계 약 200개 무설정 모형의 다층 결합, 상위 N개 결합 피처와 3단 탐욕 결합을 사용했다.
조기 종료는 log loss로 하고 최종 평가는 MCC로 했으며 8번째 소수점의 동률까지 가르는 수동 사후 조정을 했다.
댓글에는 MCC 0.9850930506, ROC AUC 0.9957310457, log loss 0.0735866, 균형 Accuracy 0.9926471, F1 0.9932400과 최적 문턱 0.5가 적혔다.

댓글과 펼친 답글 세 개에서 작성자는 시험 어휘를 쓰는 정제가 허용된 전이적 학습이라고 보았다.
가중 결합이 같은 검증 자료로 학습되고 조기 종료도 검증 자료를 보므로 CV가 조금 낙관적이라고 명시했다.
탐욕 결합은 과적합을 줄이려고 양의 계수만 썼다고 설명했다.

시험 어휘 기반 정제는 S4E11 4위와 S4E8 8위에서도 반복되지만 S4E11 1위는 합성 잡음을 건드리지 않는 편이 낫다고 반대했다.
현재 자체 `exp106_lookup_fixed24_train_test_preprocessing`과 `exp139_realmlp_reference_qnormal_train_test`는 목표값 없이 훈련 및 시험 설명변수의 분포를 함께 쓰는 같은 정보 관점을 이미 덮는다.
다만 시험에 없는 범주를 결측으로 바꾸는 정확한 연산까지 같다고 확인된 것은 아니다.

### 6위: Tilii

[#6 Place: A Quick Reflection](https://www.kaggle.com/competitions/playground-series-s4e8/writeups/tilii-6-place-a-quick-reflection)은 단독 참가자 Tilii의 글이다.
목록에는 댓글 30개가 표시됐고 상세 화면에는 일반 댓글 22개가 표시됐다.

최종 25개에는 LightAutoML TabularNN 여덟 개, AutoGluon 여섯 개, CatBoost 네 개, Keras Factorization Machine 세 개, xLearn Factorization Machine 두 개, LightGBM과 XGBoost 각 하나가 들어갔다.
단독 최강은 모든 열과 상호작용을 범주형으로 둔 Keras Factorization Machine이었고 Private은 약 0.98413부터 0.98433이었다.
탐욕 결합이 고른 13개 판은 CV가 낮고 Private은 같아 최종에 쓰지 않았다.

댓글과 펼친 답글 네 개에서 MCC 문턱은 0.5였다고 설명했다.
서로 다른 값 수가 10,000보다 적어 모든 열을 범주형으로 두었고 추가 구간화는 조금 더 나빴다고 했다.
잡음 범주와 결측 처리는 핵심 이득이 아니었다.

Factorization Machine은 자체 선행 진입 판단이 약했고 최신 외부 278개에는 다섯 Factorization Machine과 DeepFM이 이미 있다.
따라서 새 후보가 아니다.

### 8위: Jack Lee

[8th Place Solution with AutoGluon](https://www.kaggle.com/competitions/playground-series-s4e8/writeups/jack-lee-8th-place-solution-with-autogluon)은 단독 참가자 Jack Lee의 글이다.
목록과 상세 화면 모두 일반 댓글 1개를 표시했다.

작성자는 시험에 없는 잡음 수치와 문자열을 결측으로 바꾸고 단일 문자 범주만 유효하다고 보았다.
초기 AutoGluon은 기초 모형 17개와 상위 모형 13개로 CV 약 0.985, Public 0.98522, Private 0.98506이었다.
두 CPU에서 3일 학습한 판은 기초 19개와 상위 22개로 Public 0.98525, Private 0.98507이었고, 5.5일 판은 XGBoost를 빼고 기초 98개와 상위 64개로 Public 0.98528, Private 0.98507이었다.
본문에 뒤 두 판의 CV가 0.9581로 적혀 있는데 앞뒤 점수와 맞지 않아 오타 가능성이 있지만 원문 값 그대로 기록한다.

더 큰 모형 묶음이 Private을 더 올리지 않아 복잡성이 과적합했을 수 있다고 작성자가 해석했다.
댓글은 제목 오타 지적뿐이었다.

### 10위: Mahdi Ravaghi

[10th Place Solution and a Potential #5](https://www.kaggle.com/competitions/playground-series-s4e8/writeups/mahdi-ravaghi-10th-place-solution-and-a-potential-)은 단독 참가자 Mahdi Ravaghi의 글이다.
목록과 상세 화면 모두 일반 댓글 9개를 표시했다.

개별 CV와 Public은 AutoGluon 0.98492와 0.98523, XGBoost DART 0.98490과 0.98499, XGBoost 0.98488과 0.98503, LightGBM DART 0.98482와 0.98507, LightGBM 0.98480과 0.98501, HistGradientBoosting 0.98474와 0.98496, XGBoost RF 0.98465와 0.98473, CatBoost 0.98454와 0.98487, 신경망 0.98434와 0.98469였다.
이들을 합친 초기 결합은 CV 0.98501, Public 0.98521이었다.
실험 32개 전체 결합은 CV 0.985050, Public 0.98528이었고 중복을 제거하면 CV 0.985053, Public 0.98531이었다.
숨은 제출은 Private 5위 가능성이 있었지만 최종 선택하지 않았다.

최종 방법은 28개 확률을 로짓으로 바꾼 뒤 로지스틱 회귀로 합치는 것이었다.
Optuna로 OOF의 MCC 문턱을 조정하면 CV와 Public은 올랐지만 Private은 오르지 않았다.
댓글에서 자료가 균형이라 문턱 0.5가 자연스럽다고 답했다.

로짓 로지스틱 결합과 중복 제거는 최신 313개 결합의 핵심 전략과 중복이다.
MCC 문턱 조정은 ROC AUC를 쓰는 현재 과제에 적용되지 않는다.

## 교차 분석

### OOF 물량보다 계보와 선택 편향이 더 중요하다

S4E8 1위는 72개, S4E9 3위는 77개, S4E10 8위는 52개 OOF를 썼고 여러 AutoGluon 해법은 수백 기초 모형을 만들었다.
그러나 S4E8 1위의 Public 불일치 혼합은 Private에서 실패했고, S4E8 8위는 모형 수를 크게 늘려도 Private이 그대로였으며, S4E10 4위는 모형 다양성 판단에 운이 있었다고 인정했다.
현재 외부 278개를 더한 구성도 단순히 모두 넣은 판이 아니라 누출 구성원 두 개와 nhtquyn 고전 확률 모형 120개를 뺀 절제판이다.
따라서 과거의 모형 수를 목표로 삼지 않고 현행 nested 제외 기여와 계보 검사를 유지해야 한다.

### 다층 결합의 같은 fold는 엄밀한 바깥 검증을 보장하지 않는다

S4E11 1위 댓글에서 AutoGluon 개발자는 같은 fold를 공유하는 다층 결합에도 미세한 오염이 있을 수 있다고 확인했다.
S4E8 3위도 가중 결합과 조기 종료가 검증 자료를 보아 CV가 조금 낙관적이라고 적었다.
S4E9 3위는 서로 다른 분할의 외부 OOF를 섞으면 CV가 흐려진다고 경고했다.
이 세 독립 관찰은 현재 ADR의 바깥쪽 재학습, 같은 행 분할과 출처별 OOF 계보 요구를 강화한다.

### 합성 자료 정제에는 일관된 규칙이 없다

S4E11 4위와 S4E8 3위 및 8위는 시험 어휘나 열 관계로 이상값을 결측 처리해 이득을 보았다고 했다.
반면 S4E11 1위는 합성 잡음을 정리하거나 피처를 늘리면 생성 잡음을 증폭할 수 있어 원형을 유지했다고 했다.
S4E8 6위도 잡음 범주와 결측 처리가 핵심 이득은 아니라고 보았다.
따라서 시험 어휘 사용 자체를 일반 법칙으로 만들 수 없고 현재 S6E8의 구체적인 정보 관점별 대조만 유효하다.

### 지표가 다르면 마지막 단계의 비법도 달라진다

S4E11은 Accuracy, S4E8은 MCC라 확률 문턱과 동률 처리가 순위에 직접 영향을 주었다.
S4E9는 RMSE라 목표값 꼬리 확률과 예측 극단값 억제가 중요했다.
현재 S6E8은 ROC AUC이므로 순위만 중요하고 확률 문턱, 균형 자료의 0.5, 회귀 꼬리 보정은 직접 적용되지 않는다.

### 댓글이 본문만 읽었을 때의 결론을 바꿨다

S4E11 1위는 댓글에서 다층 결합 오염이 확인됐다.
S4E9 1위는 원본 자료가 실제로 의사 라벨 학습에 들어가지 않았다는 코드 사용 오류를 수정했다.
S4E9 3위는 서로 다른 분할 OOF 혼합을 피하라고 보충했고 S4E8 3위는 자신의 CV가 낙관적이라고 인정했다.
따라서 본문의 최종 순위만으로 재사용 가능성을 판정하지 않고 댓글의 실패 조건과 수정 사항을 함께 근거로 삼아야 한다.

## 현재 S6E8에 대한 적용 판단

현재 자체 후보 풀은 [후보 풀 장부](../../artifacts/pool.yaml)의 35개이며 [champion 장부](../../artifacts/champion.yaml)의 champion은 `exp156_lookup_bivariate_plr5_initavg8`, 3시드 평균 OOF AUC 0.969367610562다.
최신 외부 결합은 [두 번째 넓힌 확장 결합 기록](extended-stack-submission-2.md)의 실행 `443b3a71a2b045ba9052fbb3d821255d`다.
이 실행은 자체 35개와 외부 278개, 총 313개를 `shrunk_rank_logit_logistic`으로 결합해 nested OOF 0.9703509469와 가중 OOF 0.9712170271을 얻었다.
Public 0.97135는 사후 참고값일 뿐 채택 근거로 쓰지 않는다.

[기존 1년 조사와 실험 발주 기준인 이슈 307](https://github.com/tmheo/predicting-smartphone-addiction/issues/307)의 네 범주는 `새 후보`, `기존 결정과 중복`, `현재 과제에 부적합`, `근거 부족`이다.
아래 표는 자체 35개 안의 빈 관점인지와 외부 278개가 이미 덮는 관점인지를 분리한 예비 판정이다.

| 조사 관점 | 자체 35개 기준 | 외부 278개 기준 | 이슈 307 예비 분류 | 판단 |
| --- | --- | --- | --- | --- |
| 기초 OOF 로짓을 CatBoost 시작 기준값으로 주는 잔차 보정 | `exp023_orig_proxy_residual`은 인접하지만 기준값이 원본 프록시 하나라 정확히 같지 않음 | `view_resid_cat`, `view_resid_lgbm`, `view_resid_xgb`가 인접하지만 장부로 같은 `baseline` 구현인지 확인 안 됨 | 근거 부족 | 독립 재현과 엄밀한 nested 대조가 생길 때만 다시 판단 |
| Dynamic Stacking으로 다층 오염을 감지하고 해로운 단계를 끄기 | nested OOF 계약이 더 엄격하게 막음 | 외부 OOF도 fold 및 계보 장부로 제한함 | 기존 결정과 중복 | 모형 후보가 아니라 현재 검증 규율 유지 근거 |
| 시험에 없는 범주를 결측으로 바꾸는 시험 어휘 전처리 | `exp106_lookup_fixed24_train_test_preprocessing`과 `exp139_realmlp_reference_qnormal_train_test`가 같은 전이형 정보 관점을 사용하지만 정확한 연산은 다름 | 정확히 같은 범주 제거를 쓴 구성원은 장부에서 확인 안 됨 | 기존 결정과 중복 | 시험 분포를 목표 없이 쓰는 관점은 현행 풀이 이미 평가했고, 정확한 정제는 상반된 과거 보고 때문에 별도 근거가 부족함 |
| AutoGluon 및 LightAutoML 대규모 자동 결합 | `exp117_ag25_gbm_r21`과 현행 선형 결합이 인접 | 다양한 나무, 신경망, TabM, Lookup과 공개 노트북 OOF가 이미 결합됨 | 기존 결정과 중복 | 모형 수 확대만으로 새 정보 관점이 되지 않음 |
| 범주형 Factorization Machine과 모든 열 범주화 | 자체 풀에는 최종 구성원 없음이고 선행 진입 판단은 약함 | Factorization Machine 다섯 개와 `deepfm_exact` 포함 | 기존 결정과 중복 | 외부 제외 기여를 이기지 못하면 재학습 이유 없음 |
| 원본 자료 확률 프록시 및 원본 행 추가 | `exp022_orig_knn`, `exp023_orig_proxy_residual`, 원본 평균 및 CDF 계열 포함 | 원본과 정확값 파생을 쓰는 여러 공개 OOF 포함 | 기존 결정과 중복 | 현행 정보 관점과 중복 |
| 회귀 목표값 꼬리 분류 확률을 연속 피처로 넣기 | 빈 관점 | 대응 구성원 확인 안 됨 | 현재 과제에 부적합 | 이진 목표에는 같은 꼬리 하위 과제가 없음 |
| 대출 위험 표시, 중고차 엔진 및 브랜드 같은 도메인 피처 | 빈 관점 | 대응 구성원 확인 안 됨 | 현재 과제에 부적합 | 대응 열과 의미가 없음 |
| Accuracy 및 MCC용 확률 문턱 최적화 | ROC AUC에서는 문턱을 쓰지 않음 | 순위 로짓 결합이라 문턱 없음 | 현재 과제에 부적합 | 현행 지표와 목적이 다름 |
| 공개 OOF의 서로 다른 분할을 그대로 혼합 | 같은 행 분할과 자체 재현 요구에 어긋남 | 외부 장부가 fold 근거와 주의 사항을 따로 관리함 | 기존 결정과 중복 | 현행 계보 규율을 완화하지 않음 |
| 음수 포함 탐욕 결합, Ridge, 로지스틱 회귀와 다층 결합 | 현재 결합 결정과 비선형 2단 음성 결과가 있음 | 최신 313개가 순위 로짓 로지스틱을 사용함 | 기존 결정과 중복 | 단순 재실행하지 않음 |

### 조건부 후보의 정확한 진입 조건

CatBoost 시작 기준값 방법은 네 기초 모형에서 보정 전후 CV와 Private이 모두 개선됐다는 점에서 이번 조사에서 가장 구체적인 신규 단서다.
그러나 현재 자체 잔차 계열과 외부 잔차 구성원이 이미 가까이 있고, S4E11 댓글이 보여 준 다층 오염 위험도 있다.
따라서 현 시점에는 새 실험 이슈를 열지 않는다.

재검토하려면 먼저 외부 `view_resid_*` 세 구성원의 학습 코드를 확인해 CatBoost `baseline`과 같은 관점인지 판별해야 한다.
같은 관점이면 외부 278개 제외 기여가 이 방법의 현행 가치 판단이므로 자체 재현을 우선하지 않는다.
다른 관점이면 고정된 기초 OOF 하나만 로짓 기준값으로 주는 CatBoost 한 설정을 바깥 nested 학습 부분 안에서 완전히 다시 만들고, 기준 모형과 보정 모형을 짝비교해야 한다.
하이퍼파라미터 탐색, 여러 기준 OOF와 여러 CatBoost 설정을 동시에 열지 않는다.

후보 풀 진입과 최종 결합 채택은 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)을 그대로 따른다.
새 후보는 자체 3시드 OOF, champion 대비 0.01 하한, 0.998 스피어만 중복 검사와 현재 풀 포함 전후의 nested 결합 기여를 통과해야 한다.
최종 결합 교체는 champion 대비 nested OOF 0.00002 이상이어야 하고 0.00002 이상 0.0002 미만이면 바깥쪽 분할 다섯 개 중 셋 이상에서 이겨야 한다.

### 열지 않을 방법

- 과거 대회의 AutoGluon 모형 수를 따라 자체 후보나 외부 OOF를 무차별 추가하지 않는다.
- 시험 어휘 정제를 일반 규칙으로 확대하지 않고 현재 `exp106`과 `exp139`의 전이형 전처리 기여로 판단한다.
- Factorization Machine을 S4E8과 S4E9의 성공만으로 다시 학습하지 않는다.
- Accuracy 및 MCC 문턱 최적화와 회귀 목표값 꼬리 분해를 ROC AUC 과제에 옮기지 않는다.
- 서로 다른 fold의 공개 OOF, OOF가 없는 제출 파일과 Public 기반 맹목 혼합을 허용하지 않는다.
- 다층 AutoGluon의 자체 표시 CV를 현재 nested OOF와 직접 비교하지 않는다.
- 공개 노트북 코드를 재사용할 때는 Apache License 2.0 출처 절차를 따르고 입력 자료, 사전 학습 모형, 패키지와 외부 자산의 라이선스를 별도로 확인한다.

## 사실과 추론의 경계

각 대회 절의 순위, 작성자, 피처, 모형, 검증, 점수, 댓글 보충과 실패 사례는 링크된 Kaggle 공식 해법 본문과 댓글에서 확인한 작성자 보고 사실이다.
댓글의 재현 수치와 개발자 설명도 해당 댓글 작성자의 보고이며 이 저장소에서 다시 실행한 값은 아니다.
S4E8 AutoML Grandmasters 글의 순위는 제목이 아니라 공식 해법 화면 머리말의 3위 표시를 따랐다.
S4E8 8위 글의 CV 0.9581은 오타로 보이지만 원문 값이라는 사실과 오타 가능성이라는 추론을 분리했다.

재현성, 현재 과제와의 유사성, 자체 35개 및 외부 278개에 대한 겹침, 이슈 307의 네 범주와 실험 우선순위는 조사자의 추론이다.
외부 구성원의 이름이 `view_resid_*`라는 사실만으로 S4E10 1위의 CatBoost `baseline` 구현과 같다고 단정하지 않았다.
과거 대회의 Public과 Private은 후보 발굴의 참고 근거일 뿐 현재 S6E8의 채택 근거가 될 수 없다.

## 한계

조사는 Kaggle 공식 화면이 현재 렌더링한 본문과 댓글을 대상으로 했으므로 삭제된 댓글의 과거 내용은 복구하지 않았다.
Kaggle 목록, 상세 머리말과 실제 펼친 댓글 노드의 숫자가 달라 원시 표시값을 따로 보존했다.
일부 점수와 가중치는 글 안의 그림을 화면에서 직접 읽었지만 공개 원자료로 다시 계산하지 않았다.
AutoGluon과 외부 OOF의 내부 모형 전체 계보는 작성자가 공개한 범위를 넘어서 복원하지 않았다.
이 문서는 새 실험을 실행하거나 GitHub 이슈를 편집하지 않았다.
