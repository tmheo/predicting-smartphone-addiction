# Playground Series S5E11부터 S5E8까지 상위 10위 해법과 댓글 조사

이 문서는 GitHub 이슈 [리서치: S5E11-S5E8 상위권 해법 글과 댓글의 재사용 가능한 인사이트 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/306)의 조사 결과다.
조사 기준일은 2026-08-20이다.

## 결론

네 대회의 공식 해법 범주에서 제목이나 본문에 Private 최종 1위부터 10위가 명시된 글은 20개였다.
S5E11과 S5E8은 강한 단일 모형보다 피처 표현과 다양한 OOF 결합이 순위를 갈랐고, S5E10은 표시 점수 0.00001 안에 수백 명이 몰린 상황에서 잔차 학습과 여러 단계 결합이 마지막 미세 이득을 만들었다.
S5E9에는 조건을 만족하는 공식 해법 글이 하나도 없었다.

현재 S6E8에 그대로 옮길 새 1순위 피처나 모형 계열은 발견하지 못했다.
반복해서 등장한 정확값 부호화, 자릿수와 반올림, 범주형 전용 모형, 대규모 OOF 풀, 부분집합 선택, 의사 라벨, 전체 자료 재학습은 이미 자체 실험에서 채택 또는 기각이 끝났다.
남은 조건부 후보는 기존 결합의 OOF와 원시 피처를 함께 받는 작은 잔차 2단 모형 한 가지뿐이다.
이 후보도 현재의 얕은 XGBoost 2단 결합 실패를 뒤집어야 하므로, 하이퍼파라미터 탐색 없이 nested OOF 한 번으로 진입 여부만 가리는 것이 맞다.

Chris Deotte가 S6E3에서 사용한 4단 구조를 그대로 복제할 근거는 이 네 대회에서 강화되지 않았다.
S5E8에서는 여러 단계 결합이 반복해서 성공했지만 S5E11에서는 비선형 2단 모형의 과적합이 반복됐고, S5E10의 매우 깊은 결합 이득은 대부분 0.00001에서 0.00002 수준이었다.
현재 S6E8에서는 29개 자체 OOF와 결측 구간별 선형 결합이 이미 있고 얕은 XGBoost 2단 결합은 nested OOF에서 하락했으므로, 단계 수 자체가 아니라 새 잔차 신호의 존재를 먼저 증명해야 한다.

## 조사 범위와 방법

Kaggle API, 웹 검색, 직접 HTTP 요청과 리더보드 조회는 사용하지 않았다.
`agent-browser`의 전용 세션과 콘텐츠 경계를 켜고 Kaggle 도메인만 허용했다.
각 대회는 사용자가 지정한 [`competitionWriteUps`와 득표순 화면](https://www.kaggle.com/competitions/playground-series-s6e3/discussion?category=competitionWriteUps&sort=votes)에서 대회 slug만 바꾸어 시작했다.
득표순은 발견 순서로만 사용했고 표본 포함 여부는 글 제목이나 본문의 Private 최종 순위 표기로만 정했다.
순위표는 열지 않았고 누락 순위를 다른 글로 채우지 않았다.

각 포함 글은 본문 끝까지 읽고, `more replies`로 접힌 답글을 모두 펼친 뒤 댓글 화면 끝까지 다시 확인했다.
별도의 댓글 다음 쪽이나 `load more` 단추는 어느 포함 글에서도 남지 않았다.
Kaggle 목록의 댓글 수와 상세 화면의 일반 댓글 및 감사 댓글 수는 서로 일치하지 않았다.
따라서 숫자를 임의로 맞추지 않고 두 화면의 원시 표시값을 함께 기록했다.

포함 글 20개의 목록 댓글 수 합계는 310개였다.
상세 화면 머리말의 일반 댓글 수 합계는 275개였고 감사 댓글은 19개였다.
이 차이는 접힌 답글과 댓글 종류를 Kaggle의 두 화면이 다르게 집계하는 것으로 보이지만, 원인은 화면에서 설명되지 않았으므로 추론으로만 남긴다.

## 표본 장부

| 대회 | 과제와 평가지표 | 공식 해법 글 | 포함 순위 | 결측 순위 | 목록 댓글 합 | 상세 일반 댓글 | 감사 댓글 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| S5E11 | 대출 상환 확률, ROC AUC | 7 | 1, 2, 4, 5, 6, 8, 10 | 3, 7, 9 | 107 | 87 | 7 |
| S5E10 | 도로 사고 위험 회귀, RMSE | 9 | 1, 3, 4, 5, 7, 8 | 2, 6, 9, 10 | 129 | 121 | 5 |
| S5E9 | 노래 BPM 회귀, RMSE | 2 | 없음 | 1부터 10까지 전부 | 0 | 0 | 0 |
| S5E8 | 정기예금 가입 이진 분류, ROC AUC | 16 | 2, 3, 4, 5, 6, 8, 10 | 1, 7, 9 | 74 | 67 | 7 |

과제와 평가지표는 Kaggle의 [S5E11 평가](https://www.kaggle.com/competitions/playground-series-s5e11/overview/evaluation), [S5E10 평가](https://www.kaggle.com/competitions/playground-series-s5e10/overview/evaluation), [S5E9 평가](https://www.kaggle.com/competitions/playground-series-s5e9/overview/evaluation), [S5E8 평가](https://www.kaggle.com/competitions/playground-series-s5e8/overview/evaluation) 화면에서 확인했다.

## 제외 장부

S5E11에는 목록에 나타난 7개 글이 모두 포함 대상이었고 팀원의 중복 글은 없었다.

S5E10에서는 제목에 14위, 184위, 347위가 적힌 글 세 개를 제외했다.
184위 글 제목에 잠재적 5위라는 표현이 있지만 실제 Private 최종 순위 표기는 184위이므로 포함하지 않았다.
팀원의 중복 글은 없었고 4위 글 하나만 두 명이 함께 작성한 팀 해법이었다.

S5E9에서는 [26위 해법](https://www.kaggle.com/competitions/playground-series-s5e9/discussion?category=competitionWriteUps&sort=votes)과 제목에 Private 573위가 적힌 글만 있었으므로 둘 다 제외했다.
이 대회는 상위 10위 글이 없다는 사실 자체가 조사 결과이며, 다른 범주의 글이나 다른 순위 글로 보충하지 않았다.

S5E8에서는 제목에 11위, 12위, 15위, 17위, 19위, 21위, 25위권, 267위, 428위가 적힌 아홉 글을 제외했다.
팀원의 중복 글은 없었다.

삭제되어 접근할 수 없는 포함 글이나 제목과 본문 순위가 충돌한 포함 글은 없었다.
S5E11 6위 글의 댓글에는 삭제된 댓글 하나가 있었고, 본문은 볼 수 없다는 사실만 장부에 반영했다.

## S5E11: Predicting Loan Payback

### 1위: Mahog

[1st place - A lot of features, a lot of models, and a little bit of luck](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an)은 단독 참가자 Mahog의 글이다.
목록에는 댓글 54개가 표시됐고 상세 화면에는 일반 댓글 43개와 감사 댓글 6개가 표시됐다.

본문의 핵심은 기본 피처 쌍, 수치 자릿수, 같은 수치 피처 안의 자릿수 쌍부터 사중 결합, 서로 다른 피처의 자릿수 결합에 타깃 평균과 빈도 부호화를 붙인 대규모 피처 생성이다.
최고 단일 XGBoost는 CV 0.928175, Public 0.92831, Private 0.92923이었고 작성자는 이 단일 모형만으로 2위가 가능했다고 보고했다.
최종 후보에는 XGBoost, LightGBM, RealMLP, TabM, CatBoost, 여러 신경망과 고전 모형을 포함한 약 100개 모형이 있었고 Ridge와 탐욕 가중 결합이 가장 좋았으며 비선형 2단 모형은 더 나빴다고 했다.
본문은 별도의 폴드 수를 설명하지 않았지만 공개 노트북에 20개 fold 학습본이 있다는 사실이 댓글에서 확인됐다.

댓글은 재현성에 중요한 반례를 남겼다.
Tilii는 공개 노트북의 시험 예측이 20개 fold 합인데 5로 나뉘는 오류를 찾아 20으로 나누어야 한다고 지적했고 작성자는 이 오류 때문에 제출값이 1을 넘었다고 확인했다.
Tilii가 같은 분할에서 다시 계산한 단일 모형은 CV 0.928071, Private 0.92923이었고 범주형 중심 모형들을 더한 탐욕 결합은 CV 0.92853, Private 0.92947이었다.
추가 댓글에서는 범주형 모형 다섯 개가 개선 대부분을 만들고 그 뒤 20개 모형은 약 0.00003만 더했다고 설명했다.
Factorization Machine은 원래 11개 피처를 범주로 바꾸고 수치 피처를 이산화했으며, 이항 결합은 모형이 자체 학습하므로 삼항과 사항을 만들기 위해 단일 범주와 이항 범주를 함께 넣었다고 보충했다.
이 고차 결합의 추가 이득은 약 0.0005였다고 했다.
작성자는 피처를 작은 묶음으로 추가하고 CV가 오를 때만 유지했다고 설명했다.

공개 노트북 링크는 있지만 100개 전체 모형의 코드와 결합 계보는 모두 제공되지 않았다.
더구나 공개 단일 모형 노트북에는 시험 예측 평균 오류가 있었으므로, 본문 수치의 출처는 강하지만 공개 코드의 결합 재현성은 중간 이하로 판정한다.

### 2위: AngelosMar

[2nd Place Solution - 7 models, but 1 was also enough](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/2nd-place-solution-7-models-but-1-was-also-enou)은 단독 참가자 AngelosMar의 글이다.
목록에는 댓글 16개가 표시됐고 상세 화면에는 일반 댓글 10개와 감사 댓글 1개가 표시됐다.

최종 구성은 LightGBM 다섯 개, TabM 한 개, RealMLP 한 개의 Ridge 결합이었다.
최고 단일 LightGBM은 5-fold CV 0.92813이었고 이 단일 모형도 2위가 가능한 수준이었다.
핵심 피처는 값 종류가 많은 `annual_income`과 `loan_amount`를 분위수, 균등 구간, 반올림, 정수 나눗셈, 소수부 제거로 여러 해상도에 이산화한 뒤 타깃 평균 부호화한 것이었다.
원본 자료의 타깃으로 만든 부호화, 훈련과 원본 자료 사이의 값별 빈도 비율, 수치 자릿수도 사용했다.
일반 상호작용과 원본 행 추가는 CV를 올리지 못했다고 명시했다.
최고 단일 LightGBM은 낮은 깊이와 열 표본추출, 큰 L1 및 L2 규제를 사용했다.

댓글에서 작성자는 이산화 피처와 자릿수가 가장 중요한 피처였다고 확인했다.
Tilii는 10-fold 동일 분할로 이 모형을 재현해 CV 0.928266, Public 0.92819, Private 0.92927을 얻었고 자신의 범주형 모형 및 1위 공개 모형과 합치면 Private 0.92959가 됐다고 보고했다.
반면 Mahog가 자신의 강한 피처 집합에 이 이산화 피처를 추가한 실험은 CV가 0.928189로 조금 올랐지만 Public 0.92819와 Private 0.92908로 나빠졌다.
이는 좋은 피처라도 기존 표현과 겹치면 CV 미세 상승이 Private 개선으로 이어지지 않는다는 직접 반례다.

단일 LightGBM의 단순화 노트북은 공개됐지만 7개 전체 결합은 완전한 실행 형태로 제공되지 않았다.
단일 모형 재현성과 독립 댓글 검증은 높고 최종 결합 재현성은 중간으로 판정한다.

### 4위: Ali_Haider_Ahmad

[4th Place Solution](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/4th-place-solution)은 단독 참가자 Ali_Haider_Ahmad의 글이다.
목록과 상세 화면 모두 댓글 5개를 표시했다.

작성자는 약 100개의 새 피처를 쓴 LightGBM과 심층 신경망을 각각 층화 5-fold와 시드 5개로 학습했다.
앞서 만든 여덟 모형 결합의 시험 예측을 의사 라벨로 써서 약 0.0004의 CV 이득을 얻었다고 보고했다.
각 모형의 시드별 예측은 `differential_evolution`으로 양의 가중치를 찾았고 이 절차가 심층 신경망에 약 0.0004, LightGBM에 약 0.0002를 더했다고 했다.
최종 가중 평균의 Public은 0.928, Private은 0.92915였다.

댓글에서 작성자는 모든 fold의 학습 부분에 같은 의사 라벨 시험 부분집합을 추가하고 검증 부분은 원래 훈련 자료만 유지했다고 설명했다.
Tilii는 CatBoost가 범주 피처가 많을 때는 잘 작동했지만 수치 피처가 다수이면 XGBoost를 이기기 어려웠다고 덧붙였다.

전체 코드 링크는 없고 의사 라벨 선택 문턱과 피처 목록도 완전하지 않다.
따라서 방법 설명은 중간이지만 독립 재현성은 낮다.

### 5위: Masaya Kawamata

[5th Place Solution - (XGB+LGBM+TabM)*5SEEDs+AG](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/5th-place-solution-xgb-lgbm-tabm5seeds-ag)은 단독 참가자 Masaya Kawamata의 글이다.
목록에는 댓글 11개가 표시됐고 상세 화면에는 일반 댓글 10개가 표시됐다.

같은 피처 집합으로 XGBoost, LightGBM, TabM을 시드 5개씩 학습하고 Ridge로 합친 뒤 AutoGluon 예측을 더했다.
최종 CV는 0.92805, Public은 0.92800, Private은 0.92912였다.
자릿수, 반올림과 구간화, 같은 수치 열 내부의 자릿수 2항부터 4항 결합, 원본 자료 기준 평균 및 빈도 부호화를 사용했다.
피처는 기본 피처에 하나씩 더하는 단일 80:20 분할로 먼저 거른 뒤 XGBoost와 LightGBM은 5-fold, TabM 설정 탐색은 80:20 분할로 수행했다.
fold별 타깃 부호화 자료를 미리 생성해 설정 탐색과 다중 시드 학습 시간을 줄였고, 전체 자료 재학습은 Public이 fold 평균보다 계속 낮아 사용하지 않았다.

댓글에서 작성자는 피처 수가 50개에서 100개 늘 때마다 `colsample_bytree`와 `colsample_bynode`를 낮췄다고 설명했다.
후보 피처 묶음을 만들 때마다 CV를 보고 약한 묶음을 버렸기 때문에 최종 전진 선택의 입력은 이미 한 차례 정제됐다고 했다.
Tilii는 수백 피처라면 후진 제거보다 전진 선택이 현실적이지만 현대 모형은 자체 선택 능력이 있어 피처 선택의 주목적은 학습 시간 절감이라고 지적했다.

자료 생성, Ridge, LightGBM, XGBoost, TabM 노트북을 거의 모두 공개했지만 TabM은 Kaggle 환경에서 메모리 부족이나 12시간 제한을 넘길 수 있어 검증하지 못했다고 명시했다.
공개 범위는 높지만 실행 환경 의존성 때문에 재현성은 중간에서 높음으로 판정한다.

### 6위: Tilii

[#6 solution - Ensembling was the key](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/6-solution-ensembling-was-the-key)는 단독 참가자 Tilii의 글이다.
목록에는 댓글 15개가 표시됐고 상세 화면에는 일반 댓글 13개가 표시됐다.

수치 열을 반올림해 값 종류 수를 줄인 뒤 전부 범주형으로 취급한 Keras Factorization Machine이 단독 CV 약 0.926에서 이항 결합 후 약 0.9265가 됐다.
이 모형은 최고 단일 모형보다 약했지만 탐욕 결합 첫 단계에서 약 0.9275를 약 0.928로 끌어올렸다.
최종 구성은 XGBoost, CatBoost, LightGBM, LAMA 계열, RealMLP, Keras Factorization Machine과 여러 신경망을 포함했다.
작성자는 20개 모형을 최종 표에 보였고 한때 60개 넘게 썼다고 댓글에서 설명했다.

CatBoost와 Keras 2단 결합은 과적합했고 강한 RealMLP와 Trompt도 CV는 올렸지만 Public과 Private을 낮춰 제외됐다.
댓글에서 작성자는 규제된 CatBoost와 Keras조차 일부 결합에서 크게 과적합했고 데이터에 이상한 점이 있었다고 말했다.
또한 모형 선택은 L1 로지스틱 회귀, 나무 기반 2단 모형, 탐욕 결합이 약한 입력을 자동으로 무시할 수 있으므로 신경망 입력 대부분이 무익한 경우가 아니면 사전 제거가 필요하지 않다고 설명했다.

외부 공개 노트북과 여러 라이브러리에 크게 의존하지만 최종 코드는 공개하지 않았다.
상세한 수치와 댓글 반례는 유용하지만 재현성은 중간 이하로 판정한다.

### 8위: Ravi Ramakrishnan

[Rank8 approach - trust the CV score](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/rank8-approach-trust-the-cv-score)는 단독 참가자 Ravi Ramakrishnan의 글이다.
목록과 상세 화면 모두 댓글 6개를 표시했다.

작성자는 1,000개 넘는 피처 저장소에서 65개부터 800개 피처를 골라 XGBoost, LightGBM, CatBoost, TabM, RealMLP을 층화 5-fold로 학습했다.
작은 피처 집합의 XGBoost와 LightGBM이 가장 강했고, TabM과 RealMLP은 단독 성능보다 결합 다양성에 기여했다.
AutoGluon 결합은 CV 0.92781부터 0.92820이었지만 LB는 0.92776부터 0.92787이었고, 좋은 단일 모형보다 CV가 높고 LB가 낮은 결합이 많았다고 보고했다.
의사 라벨은 단일 모형에는 도움이 됐지만 결합에서는 실패했고 CatBoost에는 완전히 실패했다고 했다.

댓글에서 작성자는 AutoGluon으로 단일 예측을 먼저 결합한 뒤 그 결과와 단일 모형들을 규제 로지스틱 회귀로 다시 결합했다고 설명했다.
Tilii와 작성자 모두 높은 CV가 LB로 이어지지 않는 결합 과적합을 경험했다고 확인했다.

공개 기준 노트북과 OOF 자료 링크는 있지만 A100, A6000 Ada, L4 등 여러 GPU와 광범위한 피처 저장소가 필요하다.
전체 재현성은 중간으로 판정한다.

### 10위: Gerald Schwartz

[A 10th Place Experiment](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/a-10th-place-experiment)은 단독 참가자 Gerald Schwartz의 글이다.
목록과 상세 화면 모두 댓글 0개를 표시했다.

24개 후보의 최고 단일 CV는 0.92781이었고 탐욕 가중 결합은 21단계 뒤 0.92821에 도달했다.
최종 과정은 단독 CV가 0.87601과 0.82128인 열 제거 모형에도 각각 0.01 가중치를 주고 여러 후보에 음수 가중치도 주었다.
이는 단독 점수가 낮은 모형도 오차 방향이 다르면 결합에 들어갈 수 있음을 보여 주지만, 선택 편향 없는 재평가는 보고되지 않았다.

코드와 fold 규약, Public 및 Private 수치는 제공되지 않았다.
재현성은 낮다.

## S5E10: Predicting Road Accident Risk

### 1위: Tilii

[1st place - I think it was genetic programming](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/1st-place-i-think-it-was-genetic-programming)은 단독 참가자 Tilii의 글이다.
목록에는 댓글 88개가 표시됐고 상세 화면에는 일반 댓글 84개와 감사 댓글 5개가 표시됐다.

상위 네 팀이 표시 RMSE 0.05563으로 같았고 다음 약 200명이 0.05564였다고 작성자가 설명할 만큼 점수 분해능이 낮은 대회였다.
핵심 후보는 XGBoost, CatBoost, 여러 신경망, TabM, Factorization Machine, Autoencoder 잠재 표현과 유전식 탐색 피처였다.
Autoencoder 피처와 유전식 탐색 피처는 단독 모형이나 기본 피처 추가로는 경쟁력이 없었다.
그러나 유전식 탐색으로 만든 11개 예측형 피처를 2단 결합 입력으로 넣으면 약 0.00001이 개선됐고, Keras 결합을 시작값으로 둔 CatBoost 잔차 결합은 Public과 Private을 약 0.00002 개선했다고 보고했다.
마지막에는 네 개의 앞선 결합 결과를 탐욕 가중 결합으로 다시 합쳤다.

댓글에서 작성자는 Autoencoder 피처는 실제 개선이 없었고, 유전식 탐색 피처를 단일 모형 입력이 아니라 결합 구성원처럼 넣었을 때만 도움이 됐다고 재확인했다.
컴퓨터 자원은 로컬 GPU 최소 3개에서 4개를 동시에 쓸 수 있는 수준이라고 했다.
본문은 AutoGluon이 여러 그럴듯한 모형을 만들었지만 결합에는 도움이 되지 않았다고 명시했다.

TabM 설정과 유전식 탐색 피처 노트북은 공개됐지만 전체 결합 계보와 대규모 계산 환경은 제공되지 않았다.
재현성은 중간이다.

### 3위: steubk

[3rd Place: From Base to Stacking: A Multilevel Ensembling Solution](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/3rd-place-from-base-to-stacking-a-multilevel-ens)은 단독 참가자 steubk의 글이다.
목록과 상세 화면 모두 댓글 7개를 표시했다.

1단은 TabM, 잔차 TabM, XGBoost, LightGBM, TabM 틀의 다층 신경망이었고 모두 목표값 층화 7-fold와 여러 시드로 학습했다.
작성자는 7-fold가 5-fold보다, 층화가 비층화보다 OOF를 작지만 일관되게 개선했다고 했다.
2단 신경망은 기초 모형 예측을 입력으로 받았고 3단 YDF는 2단 예측과 원래 피처를 함께 받았다.
4단은 2단 신경망과 3단 YDF의 50:50 평균으로 CV 0.05585, LB 0.05564였으며 마지막에 최고 공개 제출과 다시 섞어 0.05563을 얻었다.

댓글에는 방법 질문에 대한 작성자 답변과 공개 코드가 없었다.
외부 노트북 의존성과 공개 제출 혼합 때문에 재현성은 낮다.

### 4위: benkerrouche abdelbasset와 Ravi Ramakrishnan

[4th Place - Residual XGBoost + Meta NN + Hill Climb Opt](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/4th-place-residual-xgboost-meta-nn-hill-clim)은 benkerrouche abdelbasset와 Ravi Ramakrishnan 팀의 글이다.
목록과 상세 화면 모두 댓글 6개를 표시했다.

잔차 XGBoost, 피처 생성과 설정 탐색으로 CV를 0.05598에서 0.05588까지 낮췄다.
같은 기본 피처에 외부 OOF를 더한 XGBoost 14개와 외부 OOF 7개를 시드 17개의 신경망 2단 결합에 넣어 0.0558347을 얻었다.
탐욕 가중 결합은 0.055821까지 낮췄고 마지막에는 신경망 결과와 탐욕 결합 결과를 다시 섞었다.

댓글에서 작성자는 시간이 부족해 여러 모형을 넓게 탐색하지 않고 XGBoost 개선을 단계별로 계속한 이유를 설명했다.
두 개의 공개 노트북은 있지만 외부 OOF 계보와 누출 검증은 본문만으로 완결되지 않는다.
재현성은 중간이다.

### 5위: Chris Deotte

[5th Place - One Hundred Folds!](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/5th-place-one-hundred-folds)은 단독 참가자 Chris Deotte의 글이다.
목록에는 댓글 25개가 표시됐고 상세 화면에는 일반 댓글 21개가 표시됐다.

기초 계열은 XGBoost와 TabM 두 종류뿐이었지만 각각 여러 변형을 만들었다.
TabM 세 개를 100-fold로 학습하고 그 OOF를 새 피처로 받은 XGBoost도 같은 100-fold로 학습한 뒤, XGBoost 두 개, TabM 세 개와 TabM 위에 쌓은 XGBoost 두 개를 탐욕 가중 결합했다.
그 결합으로 시험 자료에 의사 라벨을 만든 뒤 TabM을 다시 학습해 더 나은 7개 모형 결합을 만들었다.
최종 제출 하나는 이 결합과 최고 공개 노트북을 50:50으로 섞었다.

댓글에서 작성자는 100-fold가 전체 자료 재학습 대신 각 학습본이 99% 자료를 보면서 조기 종료도 쓰게 하려는 선택이었다고 설명했다.
XGBoost는 100개 fold도 빠르게 학습됐고 TabM은 여러 A100 GPU에서 하루 종일 돌렸다고 했다.
또한 TabM만 1단 입력으로 쓴 이유는 최적 설계가 아니라 다른 대회와 업무 때문에 시간이 없었기 때문이라고 답했다.

최종 점수와 분리 대조는 제공되지 않았고 전체 결합 노트북도 공개되지 않았다.
계산량이 매우 크므로 재현성은 낮음에서 중간으로 판정한다.

### 7위: Patryk

[7th Place - Ridge](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/7th-place-ridge)는 단독 참가자 Patryk의 글이다.
목록과 상세 화면 모두 댓글 0개를 표시했다.

9-fold XGBoost, 7-fold와 5-fold 층화 XGBoost, 신경망, LightGBM, YDF, TabM, XGBoost와 LightGBM 및 CatBoost 결합을 15-fold Ridge로 합쳤다.
9-fold XGBoost 안의 타깃 부호화는 별도 5-fold와 평활 10을 썼고 모든 피처 쌍을 조합했다.

공개 노트북 링크는 있지만 수치 결과와 피처 및 설정 전체가 본문에 없다.
재현성은 중간 이하로 판정한다.

### 8위: Matt graham

[8th Place Solution for S5E10: Predict Road Accident Risk](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/8th-place-solution-for-s5e10-predict-road-acciden)은 단독 참가자 Matt graham의 글이다.
목록과 상세 화면 모두 댓글 3개를 표시했다.

5-fold부터 55-fold까지 학습한 XGBoost, TabM, HistGradientBoosting, LightGBM, 신경망과 AutoGluon의 OOF 22개부터 48개를 모았다.
분산이 거의 없거나 중앙값보다 RMSE가 크게 나쁜 열을 먼저 버리고 상관 절댓값 0.9995를 넘는 쌍에서 더 강한 열을 남겼다.
그 뒤 탐욕 비음수 최소제곱과 LassoCV가 4개부터 6개 입력을 고르게 했다.
Ridge는 CV 0.05579와 LB 0.05564였고 Ridge와 잔차 신경망을 합친 2단 Ridge는 CV 표시값이 같지만 LB 0.05563이어서 제출했다면 더 좋았을 것이라고 보고했다.

댓글에서 Tilii는 상관이 거의 같은 두 모형 가운데 단독 점수가 높은 쪽만 남기면 결합에 더 좋은 약한 모형을 버릴 수 있으므로 상관 중복만 확인하고 실제 선택은 탐욕 결합에 맡기라고 지적했다.
작성자는 상관과 RMSE는 초기 청소이고 실제 입력 선택은 비음수 최소제곱과 LassoCV가 했다고 보충했다.

핵심 코드 조각은 공개됐지만 전체 실행 묶음과 외부 입력 계보는 제공되지 않았다.
재현성은 중간이다.

## S5E9: Predicting the Beats-per-Minute of Songs

공식 해법 범주에는 26위 글과 Private 573위 글만 있었고 상위 10위 글은 없었다.
따라서 본문과 댓글을 분석할 포함 표본이 없으며, 이 대회에서 방법론 결론을 만들지 않았다.

## S5E8: Binary Classification with a Bank Dataset

### 2위: Mahog

[2nd place - Yet another ensemble](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/2nd-place-yet-another-ensemble)은 단독 참가자 Mahog의 글이다.
목록에는 댓글 32개가 표시됐고 상세 화면에는 일반 댓글 29개와 감사 댓글 5개가 표시됐다.

최종 구성은 59개 모형을 CatBoost로 합친 것이었다.
최고 단일 TabM은 CV 0.976810, Public 0.97765, Private 0.97750이었고 최종 CatBoost 결합은 CV 0.977432, Public 0.97817, Private 0.97796이었다.
Ridge와 탐욕 가중 결합은 각각 Private 0.97786과 0.97789로 CatBoost보다 낮았다.
피처는 이항 범주의 대회 및 원본 목표 타깃 평균과 빈도 부호화, 수치 이항 곱과 주기 표현이었다.

댓글에서 xLearn Field-aware Factorization Machine은 최고 XGBoost와 점수는 비슷하고 분포 차이는 컸지만 다른 모형과 잘 합쳐지지 않았다는 반례가 제시됐다.
작성자는 Gandalf가 `pytorch-tabular`, GRN이 앞선 공개 신경망, Bartz가 GPU BART 구현이라고 설명했다.

TabM 노트북 하나는 공개됐지만 59개 전체 모형과 CatBoost 2단 결합은 공개되지 않았다.
재현성은 중간 이하로 판정한다.

### 3위: bestwater

[3rd Place Solution - OOF Stacking + AutoGluon](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon)은 단독 참가자 bestwater의 글이다.
목록과 상세 화면 모두 일반 댓글 10개를 표시했고 상세 화면에는 감사 댓글 1개도 표시됐다.

작성자는 자신의 모형과 공개 노트북에서 OOF 및 시험 예측을 모으고 누출이나 부풀려진 CV가 의심되는 입력을 제거한 뒤 AutoGluon을 학습했다.
새 OOF의 피처 중요도를 보고 입력을 다시 고르는 과정을 여러 번 반복했다.
약 50개 OOF를 입력으로 한 최종 V109 AutoGluon은 CV 0.977631, Public 0.97821, Private 0.97790이었다.
단일 XGBoost에서는 2항 결합과 타깃 및 빈도 부호화가 가장 꾸준했고 3항 결합은 미미하거나 잡음을 더했다.
AutoGluon의 수동 5-fold와 10-fold 지정은 기본 자동 결합보다 좋아지지 않았고 단순 또는 가중 평균은 CV만 올리고 LB를 올리지 못했다.

댓글에서 작성자는 최종 제출이 OOF를 수동으로 평균한 것이 아니라 선택한 OOF를 입력 피처로 넣어 AutoGluon을 다시 학습한 결과라고 명확히 했다.
AutoGluon의 상세 모형 표는 `predictor.leaderboard()`가 자동 생성한다고도 설명했다.

최종 노트북이 공개됐고 단계별 수치도 상세하지만 입력 공개 노트북 전부의 누출 규율은 각 출처에 의존한다.
재현성은 중간에서 높음으로 판정한다.

### 4위: Masaya Kawamata

[4th Place Solution](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/4th-place-solution)은 단독 참가자 Masaya Kawamata의 글이다.
목록에는 댓글 13개가 표시됐고 상세 화면에는 일반 댓글 11개가 표시됐다.

1단은 여러 피처 집합으로 만든 AutoGluon 약 100개와 자체 모형 약 100개의 OOF였다.
2단은 XGBoost와 GRN 신경망을 각각 의사 라벨 시험 행을 새 행으로 쓰는 판과 중첩 k-NN으로 새 열을 만드는 판으로 나눈 네 모형이었다.
의사 라벨은 1단 탐욕 가중 평균이 0.99 초과 또는 0.01 미만인 시험 행에서 만들었다.
3단은 네 모형을 탐욕 가중 평균했고 최종 가중은 XGBoost 0.3, 신경망 0.7이었다.
최종 CV는 0.977594, Public은 0.97828, Private은 0.97790이었다.

작성자는 별도 80:20 보류 집합에서 fold 학습본 평균과 전체 자료 재학습을 비교했고, fold 평균이 모든 경우 더 좋고 안정적이었다고 보고했다.
AutoGluon은 피처 집합마다 1단 기초 모형만 만들도록 `num_stack_levels=0`으로 제한했고 디스크 부족을 피하려고 모형 종류도 제한했다.

댓글에서 작성자는 의사 라벨이 CV와 Public을 모두 개선했기 때문에 최종 두 제출에 썼고 Public만 올랐다면 한 제출에는 쓰지 않았을 것이라고 설명했다.
Mahog는 Kaggle의 `/kaggle/tmp`에 약 60GB가 있다고 알려 줬지만 작성자는 자신의 디스크 오류 원인이 다른 데 있을 수 있다고 답했다.
작성자는 2단 신경망이 GRN이라고 공개했다.

설계와 코드 조각 및 참조 노트북이 매우 상세하지만 200개 OOF 전부의 계보와 계산 환경은 크다.
재현성은 높음에 가깝지만 완전한 독립 재현은 어렵다.

### 5위: Ravi Ramakrishnan

[Rank-3 Public Rank-5 Private Approach](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/rank-3-public-rank-5-private-approach)은 단독 참가자 Ravi Ramakrishnan의 글이다.
목록에는 댓글 9개가 표시됐고 상세 화면에는 일반 댓글 7개와 감사 댓글 1개가 표시됐다.

470개 넘는 단일 모형 후보를 같은 층화 5-fold로 만들고 AutoGluon 7개를 각각 16시간씩 학습해 2단 결합을 만들었다.
3단에서는 양수와 음수 가중치를 허용한 탐욕 결합 10개와 로지스틱 회귀 5개를 만들고 최종 제출을 50:50으로 합쳤다.
대안 제출은 320개 후보를 쓴 AutoGluon이었다.
AutoGluon, 탐욕 결합과 로지스틱 회귀의 최고 CV는 각각 0.97745, 0.97746, 0.9773이었다.
원본 자료를 행과 열로 쓰고 fold 안 목표 평균과 fold 밖 훈련 및 시험 합산 빈도, 2항부터 4항 결합, 의사 라벨을 사용했다.

댓글에서 작성자는 Polars와 50,000행 Parquet 분할의 속도 이득을 실제로 계측하지 않았다고 답했다.
의사 라벨은 확신 높은 시험 예측에 라벨을 붙여 훈련에 더하는 뜻이라고 설명했지만 선택 문턱은 공개하지 않았다.

여러 공개 기준 노트북과 OOF 자료는 제공됐지만 전체 후보 생성 및 결합은 다양한 외부 GPU와 100시간 넘는 AutoGluon 실행에 의존한다.
재현성은 중간이다.

### 6위: Kyr1ll

[6th Place Solution -> OOF Stacking with LGBM](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/6th-place-solution-oof-stacking-with-lgbm)은 단독 참가자 Kyr1ll의 글이다.
목록과 상세 화면 모두 댓글 3개를 표시했다.

CNN, TabR, LightGBM, XGBoost, CatBoost 두 개와 신경망 하나의 OOF 일곱 개를 설정 탐색한 LightGBM 2단 모형으로 합쳤다.
OOF만 넣으면 CV 0.97736, Private 0.97766이었고 원래 피처도 함께 넣으면 CV 0.97739, Private 0.97775로 올랐다.
작성자는 계산 자원이 부족해 기초 모형마다 fold 수와 난수값이 달랐고 같은 분할을 써야 한다는 사실을 늦게 알았다고 인정했다.

댓글에서 기초 모형 다수가 실제로는 fold 평균과 시드 평균이어서 엄밀한 단일 모형이 아니었다고 보충했다.
다른 참가자는 대규모 결합이 빨리 포화되지 않은 이유가 타깃 부호화가 없는 모형도 많이 들어갔기 때문일 수 있다고 추정했다.

코드는 제공되지 않았고 분할 불일치가 있어 재현성과 인과 해석은 낮다.
다만 원시 피처를 2단 입력에 더한 전후 수치가 있는 점은 조건부 후보의 직접 근거다.

### 8위: DanteTheAbstract

[8th place - hill climb selected meta-learners](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/8th-place-hill-climb-selected-meta-learners)은 단독 참가자 DanteTheAbstract의 글이다.
목록과 상세 화면 모두 댓글 4개를 표시했다.

모든 기초 모형을 같은 시드 208의 층화 5-fold로 학습하고 탐욕 결합에서 양의 가중치를 받은 19개를 선택했다.
선택 가중치는 버리고 선택된 OOF 자체를 AutoGluon, 신경망과 나무 계열의 여러 2단 모형에 넣은 뒤 그 결과를 다시 탐욕 가중 결합했다.
2단 모형은 10-fold로 학습했고 최종 CV는 0.97742733, Public은 0.97801, Private은 0.97768이었다.
원본 자료는 행 증강과 원본 목표 평균 열의 두 방식으로 사용했다.

댓글은 대부분 축하였고 방법을 수정하는 반례는 없었다.
피처 코드와 선택 가중치 및 참조 출처는 상세하지만 전체 실행 묶음은 제공되지 않았다.
재현성은 중간이다.

### 10위: Thiago Lima Santos

[10th place - NODE (Neural Oblivious Decision Ensembles)](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/10th-place-node-neural-oblivious-decision-ensemble)은 단독 참가자 Thiago Lima Santos의 글이다.
목록과 상세 화면 모두 댓글 3개를 표시했다.

XGBoost 10개, LightGBM 5개, CatBoost 2개, DeepTables 계열, 다층 신경망, 여러 배깅 및 RAPIDS 모형, AutoGluon을 K-fold OOF로 만들었다.
2단 NODE가 탐욕 결합, Ridge, Lasso, Optuna 가중 평균과 다른 2단 가중 조합보다 좋았다고 보고했다.
구체적인 fold 수, CV와 Public 및 Private 수치, 코드가 제공되지 않았다.

댓글은 NODE가 독특한 2단 모형이라는 평가와 축하뿐이었다.
재현성은 낮다.

## 교차 분석

### 강한 단일 모형과 다양한 결합은 서로 대체 관계가 아니다

S5E11 1위와 2위는 단일 XGBoost 또는 LightGBM만으로 2위가 가능한 수준이었지만 서로 다른 피처 표현을 합치면 그 위로 더 올라갔다.
댓글 재실험에서는 1위와 2위 단일 모형에 범주형 중심 모형 다섯 개를 더했을 때 대부분의 추가 이득이 생겼고 이후 20개는 약 0.00003만 더했다.
S5E8 6위는 일곱 OOF만으로 6위를 했고 S5E11 6위는 60개 넘게 만들고도 일부 강한 신경망을 결합에서 버렸다.
따라서 모형 수가 아니라 기존 풀과 다른 오류를 만드는 표현이 핵심이다.

### 비선형 2단 모형의 성패는 대회와 입력 표현에 달렸다

S5E8에서는 CatBoost, AutoGluon, LightGBM, GRN과 NODE 2단 모형이 모두 상위 10위 해법에 등장했다.
반면 S5E11 1위는 LightGBM, CatBoost와 신경망 2단 모형이 선형 결합보다 훨씬 나빴다고 했고 6위도 Keras와 CatBoost가 크게 과적합했다고 보고했다.
S5E10 8위는 잔차 신경망과 2단 Ridge의 LB 이득을 보고했지만 CV 표시값은 같았고 실제 최종 제출도 아니었다.
비선형 2단 모형은 일반 법칙이 아니라 선형 결합 뒤 남은 잔차가 원시 피처나 구간 맥락과 구조적으로 연결될 때만 열 후보라는 결론이 맞다.

### 부분집합 선택은 단독 성능 필터보다 결합 안 선택이 안전하다

S5E10 8위 댓글은 상관이 거의 같은 두 모형에서 강한 쪽만 남기면 실제 결합에 더 좋은 약한 쪽을 버릴 수 있다는 위험을 구체적으로 지적했다.
S5E11 10위는 단독 CV가 매우 낮은 열 제거 모형에도 작은 가중치를 줬고, S5E11 6위의 범주형 Factorization Machine도 단독 점수보다 결합 기여가 훨씬 컸다.
따라서 정확 중복과 거의 완전한 상관만 사전 제거하고 나머지는 선택 편향 없는 결합 평가가 고르게 해야 한다.

### 의사 라벨과 전체 자료 재학습은 일관된 승리 공식이 아니다

의사 라벨은 S5E11 4위, S5E10 5위, S5E8 4위와 5위에서 사용됐지만 S5E11 8위는 단일 모형에는 도움이 되고 결합에서는 실패했다고 했으며 CatBoost에서는 완전히 실패했다.
S5E8 4위는 fold 평균이 전체 자료 재학습보다 안정적이라고 실험했고 S5E10 5위는 100-fold로 99% 자료와 조기 종료를 함께 확보하려 했다.
이 결과는 의사 라벨이나 전체 자료 재학습을 기본값으로 삼지 말고 대회별 누출 없는 대조가 필요하다는 뜻이다.

### 댓글 검토가 본문만 읽을 때의 잘못된 결론을 막았다

S5E11 1위 댓글은 공개 코드의 시험 예측 분모 오류를 밝혔고, 2위 댓글은 이산화 피처가 다른 강한 피처 집합에서는 CV만 올리고 Private을 낮춘 반례를 남겼다.
S5E10 8위 댓글은 상관 기반 약한 모형 제거의 위험을 바로잡았고, S5E11 6위 댓글은 범주형 전용 모형의 실제 결합 기여와 신경망 결합 과적합을 수치로 보충했다.
따라서 본문의 우승 구성만 복제하는 것보다 댓글의 독립 재실험과 실패 조건을 함께 옮기는 것이 더 재사용 가능하다.

## 현재 S6E8에 대한 적용 판단

현재 champion은 `exp127_lookup_muon`의 3시드 평균 OOF AUC 0.9692840450이고 후보 풀에는 자체 재현한 OOF 29개가 있다.
후보 풀에는 LightGBM, XGBoost, CatBoost, Lookup-Transformer, TabM, RealMLP, TabPFN-3, 표 합성곱망, 단변량 spline Transformer와 로지스틱 회귀 계열이 이미 들어 있다.
이 사실은 [champion 장부](../../artifacts/champion.yaml)와 [후보 풀 장부](../../artifacts/pool.yaml)에 기록돼 있다.

29개 전부를 쓰는 선택 편향 없는 결합이 탐욕, 배깅 탐욕과 Optuna가 고른 세 개만 쓰는 결합보다 나았으므로, 과거 우승 해법의 OOF 물량을 근거로 후보를 더 무차별 생성할 이유는 없다.
기존 결과는 [부분집합 선택과 가중치 탐색 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/62)에 있다.

현재 결측 구간별 선형 결합의 nested OOF는 0.9695103693이고 일반 선형 결합보다 0.0000268776 높으며 다섯 outer fold에서 모두 이겼다.
같은 평가에서 얕은 XGBoost 2단 결합은 0.0000600361 하락했다.
기존 결과는 [비선형 및 구간별 2단 결합 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)에 있다.

### 우선순위 1: 기존 결합과 검증 규율을 유지한다

모든 새 기초 모형은 현재 5-fold와 같은 행 분할로 OOF를 만들고 후보 풀 진입 때 성능 하한, 0.998 상관 중복 검사와 nested OOF 기여를 거쳐야 한다.
공개 OOF, 공개 제출 파일과 다른 fold의 OOF를 직접 섞지 않는다.
이는 S5E8 6위가 분할 불일치를 인정한 약점과 S5E10 및 S5E11의 결합 과적합을 현재 규약이 이미 막고 있기 때문이다.

### 우선순위 2: 조건부 진입 진단으로 원시 피처를 받은 잔차 2단 모형 한 개만 검토한다

새로운 실험 가치가 남은 유일한 조합은 현재 29개 구성원의 순위 및 잘린 logit, 원시 12열과 결측 구간을 함께 받아 현재 결측 구간별 선형 결합의 잔차를 학습하는 작은 2단 모형이다.
직접 근거는 S5E8 6위에서 LightGBM 2단 모형에 원시 피처를 더했을 때 CV가 0.97736에서 0.97739, Private이 0.97766에서 0.97775로 오른 대조다.
보조 근거는 S5E10 3위의 YDF가 앞 단계 예측과 원래 피처를 함께 썼고 S5E10 8위가 Ridge 잔차를 작은 신경망으로 학습했다는 사실이다.

진입 진단은 하이퍼파라미터 탐색 없이 작은 LightGBM 한 설정이나 작은 완전연결 신경망 한 설정으로 제한한다.
같은 nested outer 5-fold에서 첫 대조는 OOF 표현만, 둘째 대조는 같은 OOF 표현과 원시 피처를 함께 넣어 원시 피처의 한계 기여를 분리한다.
목표는 직접 분류와 현재 선형 결합 잔차 학습 중 하나를 사전에 고정하고, 결과를 본 뒤 유리한 쪽을 고르지 않는다.

현재 기본 결합 대비 평균 OOF AUC가 0.00002 이상 오르고 outer fold 다섯 개 중 셋 이상에서 이길 때만 확장한다.
OOF 전용 판도 기본 결합보다 낮거나 원시 피처 추가분이 0.00002 미만이면 즉시 닫는다.
현재 얕은 XGBoost 2단 결합이 이미 음성이므로 여러 깊이, 여러 학습률, 여러 신경망 구조로 탐색하지 않는다.

### 우선순위 3: 여러 단계 결합은 우선순위 2가 통과할 때만 연다

우선순위 2가 통과하면 현재 결측 구간별 선형 결합과 새 잔차 2단 모형 두 결과만 nested outer 학습 부분에서 가중치를 맞춰 한 번 합친다.
두 결과의 순위 상관이 0.998 이상이면서 한계 이득이 0.00002 미만이면 추가 단계를 만들지 않는다.
이 조건을 통과해야만 S6E3식 4단 구조의 일부를 옮길 근거가 생긴다.

### 열지 않을 방법

- 자릿수, 반올림과 다중 해상도 타깃 부호화는 `exp015_te_r1`이 OOF -0.00005282, `exp106_lgb_kitopl_digit_identity`가 -0.00006475였고 66개 쌍 TE 및 CE의 정식 통과자도 없었으므로 다시 열지 않는다.
- 의사 라벨은 고확신 시험 행이 쉬운 양성 꼬리에 치우쳤고 누출 없는 판정에는 전체 앙상블의 중첩 재학습이 필요하다는 [의사 라벨 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/68)을 유지한다.
- 100-fold는 S5E10 5위가 분리 이득을 보고하지 않았고 현재 규약이 3시드 5-fold와 1.25배 전체 자료 재학습 및 5:1 혼합을 이미 사용하므로 열지 않는다.
- 전체 자료 예측만 쓰지 않고 fold 평균과 전체 자료 예측을 5:1로 합치는 현재 [전체 자료 재학습 규약](../adr/0002-full-data-refit-protocol.md)을 유지한다.
- Factorization Machine은 S6E8 자체 진입 진단에서 기존 74개 구성원에 대한 기여가 약 0.000006이었고 Lookup 계열이 더 강하고 더 비상관이었으므로 S5E11의 성공만으로 다시 열지 않는다.
- 유전식 피처 탐색은 S5E10 1위에서도 단일 모형에 도움이 되지 않았고 결합 단계 이득이 약 0.00001뿐이었으므로 현재 문턱을 넘을 기대가 없다.
- 공개 OOF와 최고 공개 제출의 50:50 혼합은 출처별 fold, 누출, 입력 판본과 라이선스를 하나의 자체 OOF 계보로 증명할 수 없으므로 사용하지 않는다.
- 공식 공개 노트북 코드를 재사용할 때는 Apache License 2.0 출처 절차를 따르고 입력 자료, 사전 학습 모형, 패키지와 외부 자산의 라이선스를 별도로 확인한다.

## 사실과 추론의 경계

각 대회 절의 순위, 작성자, 피처, 모형, 검증, 점수, 댓글 보충과 실패 사례는 링크된 Kaggle 본문과 댓글에서 확인한 작성자 보고 사실이다.
독립 재현이라고 표시한 수치도 댓글 작성자가 자신의 분할과 코드로 보고한 값이며 이 저장소에서 다시 실행한 값은 아니다.
재현성 평가는 공개 코드 범위, fold 및 시드 명세, 외부 OOF 계보와 계산 자원 정보를 근거로 한 조사자의 판단이다.
S6E8 적용 우선순위와 중단 조건은 과거 대회 사실을 현재 저장소의 자체 OOF 결과 및 판정 계약에 대조한 추론이다.
과거 대회의 Public과 Private 결과는 방법 후보를 고르는 참고 근거일 뿐 S6E8 채택 근거가 될 수 없고, 실제 채택은 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)을 따라야 한다.

## 한계

조사는 로그아웃 상태의 Kaggle 화면이 렌더링한 본문과 댓글을 대상으로 했으므로 삭제 댓글의 과거 내용은 복구하지 않았다.
Kaggle 목록과 상세 화면의 댓글 숫자가 달라 두 숫자를 모두 보존했으며, 화면 밖 내부 집계 규칙은 추정하지 않았다.
글 안의 그림은 화면의 대체 텍스트와 본문 설명으로 확인했으며 그림에만 있는 수치를 별도로 전사하지 않았다.
이 문서는 새 실험을 실행하거나 새 실험 발주 이슈를 만들지 않았다.
