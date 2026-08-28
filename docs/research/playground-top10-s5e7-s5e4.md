# Playground Series S5E7부터 S5E4까지 상위 10위 해법과 댓글 조사

이 문서는 GitHub 이슈 [리서치: S5E7-S5E4 상위권 해법 글과 댓글 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/469)의 조사 결과다.
조사 기준일은 2026-08-28이다.

## 결론

네 대회의 공식 해법 범주에 보이는 글 35개를 모두 확인했고, 제목이나 상세 본문에 Private 최종 1위부터 10위가 명시된 글 23개를 포함했다.
S5E7은 공식 해법 글 한 개의 제목에 3위 점수라는 표현이 있었지만 상세 화면은 최종 42위를 표시했으므로 상위 10위 글이 하나도 없다.
S5E6은 1, 2, 3, 4, 5, 6, 7, 10위, S5E5는 1위부터 9위, S5E4는 1위부터 6위 글을 포함했다.
누락 순위는 다른 글로 채우지 않았다.

포함 글의 목록 화면 댓글 표시값 합계는 658개였다.
상세 화면 머리말의 일반 댓글 표시값 합계는 531개였고 감사 댓글 표시값 합계는 48개였다.
모든 포함 글에서 댓글 영역을 끝까지 내려 지연 표시되는 댓글을 불러왔고 more replies로 접힌 답글도 모두 펼쳐 확인했다.

현재 S6E8에 바로 새 실험을 열 만큼 강한 새 정보 관점은 발견하지 못했다.
고차 결합 목표 부호화, 원본 자료 활용, 잔차 학습, 다양한 OOF의 Ridge 또는 탐욕 결합, 결측 구간별 비선형 2단 결합은 자체 35개 후보 풀의 채택 또는 음성 결정과 외부 278개 확장 결합이 이미 직접 또는 인접하게 덮는다.
S5E6 10위의 순위 오류 기반 표본 가중은 MAP@3 전용이고, S5E4의 목표 비율, 목표 대리 열 예측과 생성 흔적 수정은 회귀 과제의 강한 구조에 의존하므로 ROC AUC 이진 분류에 그대로 옮길 수 없다.

이번 조사에서 상대적으로 새로운 단서는 세 가지다.
첫째는 S5E4 1위의 범주 one-hot 열과 지배적인 연속 열의 곱을 대량 생성해 선형 모형이 집단별 기울기를 배우게 한 방법이다.
둘째는 S5E6 2위의 지도형 자동부호화 잠재 표현과 S5E5 9위의 비지도 랜덤 포리스트 잎 표현이다.
셋째는 S5E4 2위의 여러 차수 목표 부호화 값을 행별 평균, 표준편차, 최솟값과 최댓값으로 다시 요약하는 방법이다.
세 방법 모두 자체 35개에 정확히 같은 구현은 없지만 단독 제거 기여나 서로 독립된 성공 사례가 부족하고, 외부 278개에는 인접한 선형 SVM, 랜덤 포리스트, 격자 및 다중 목표 부호화 구성원이 있다.
따라서 세 방법은 새 후보가 아니라 근거 부족으로 남긴다.

현재 비교 기준은 자체 후보 풀 35개와 외부 구성원 278개를 더한 313개 확장 결합이다.
자체 35개 풀의 최고 결합은 shrunk_rank_logit_logistic, nested OOF 0.9698105828이다.
최신 확장 결합 실행 443b3a71a2b045ba9052fbb3d821255d는 자체 35개와 외부 278개를 같은 전략으로 결합해 nested OOF 0.9703509469와 가중 OOF 0.9712170271을 기록했다.
Public 0.97135는 사후 참고값이며 후보 판정에는 사용하지 않는다.

## 조사 범위와 방법

Kaggle API, 웹 검색 결과 요약, 직접 HTTP 요청과 리더보드 조회는 사용하지 않았다.
agent-browser의 격리된 이름 세션으로 각 대회의 실제 Kaggle 공식 competitionWriteUps 화면과 상세 글을 직접 확인했다.
각 조사는 사용자가 지정한 [S5E7 해법 화면](https://www.kaggle.com/competitions/playground-series-s5e7/discussion?category=competitionWriteUps&sort=recent-comments)에서 대회 slug만 바꾸어 시작했다.
최근 댓글순은 발견 순서로만 사용했고 표본 선정에는 쓰지 않았다.
순위는 글 제목 또는 상세 화면의 Solution Writeup · Nth place 표기만 사용했다.
제목의 점수 순위와 상세 화면의 최종 순위가 충돌하면 상세 화면의 최종 순위를 따랐다.

과제와 평가지표는 Kaggle의 [S5E7 평가](https://www.kaggle.com/competitions/playground-series-s5e7/overview/evaluation), [S5E6 평가](https://www.kaggle.com/competitions/playground-series-s5e6/overview/evaluation), [S5E5 평가](https://www.kaggle.com/competitions/playground-series-s5e5/overview/evaluation), [S5E4 평가](https://www.kaggle.com/competitions/playground-series-s5e4/overview/evaluation) 화면에서 확인했다.
포함 글마다 본문, 작성자와 순위 표시, 검증 설명, 특성, 모형, 결합, 보고 점수, 계산 자원, 외부 자료, 일반 댓글, 접힌 답글과 감사 댓글을 확인했다.
화면에 없는 팀 구성, 점수, 설정과 제거 기여는 추정하지 않고 미보고로 남겼다.

Kaggle 목록의 댓글 수와 상세 화면 머리말의 일반 댓글 및 감사 댓글 수는 여러 글에서 서로 일치하지 않았다.
목록 수에는 접힌 답글, 삭제 댓글과 댓글 종류가 다르게 반영되는 것으로 보이지만 Kaggle 화면은 집계 규칙을 설명하지 않는다.
따라서 아래 장부는 두 화면의 원시 표시값을 따로 기록한다.

## 표본 장부

| 대회 | 과제와 평가지표 | 공식 해법 글 | 포함 순위 | 결측 순위 | 목록 댓글 합 | 상세 일반 댓글 | 감사 댓글 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| S5E7 | 성격 이진 분류, Accuracy | 1 | 없음 | 1부터 10까지 전부 | 0 | 0 | 0 |
| S5E6 | 비료 다중 분류, MAP@3 | 12 | 1, 2, 3, 4, 5, 6, 7, 10 | 8, 9 | 239 | 188 | 17 |
| S5E5 | 운동 칼로리 회귀, RMSLE | 15 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 10 | 142 | 122 | 16 |
| S5E4 | 팟캐스트 청취 시간 회귀, RMSE | 7 | 1, 2, 3, 4, 5, 6 | 7, 8, 9, 10 | 277 | 221 | 15 |
| 합계 |  | 35 | 23개 글 |  | 658 | 531 | 48 |

## 제외와 결측 장부

S5E7의 유일한 글 [Top #3 score solution writeup](https://www.kaggle.com/competitions/playground-series-s5e7/writeups/3rd-place-solution-predict-the-introverts-from-the)은 제목만 보면 3위처럼 보이지만 상세 화면은 Solution Writeup · 42nd place를 표시했다.
이 글은 나무 모형 50%, MLP 45%, 로지스틱 회귀 5%와 40% 분류 문턱을 설명했지만 최종 42위이므로 제외했다.
목록과 상세 화면에는 댓글 4개가 표시됐고 상위 10위 댓글 합계에는 넣지 않았다.

S5E6에서는 순위가 없는 Logistic Regression for Fertilizer Prediction, 28위, Public 3위에서 Private 22위, 21위 글을 제외했다.
포함 순위는 1위부터 7위와 10위이며 8위와 9위는 결측이다.

S5E5에서는 22위, 38위, Public 3위에서 Private 40위, 43위, 17위와 68위 글을 제외했다.
포함 순위는 1위부터 9위이며 10위는 결측이다.

S5E4에서는 139위 글을 제외했다.
포함 순위는 1위부터 6위이며 7위부터 10위는 결측이다.

삭제되어 접근할 수 없는 포함 글이나 같은 순위를 중복 주장한 포함 글은 없었다.
S5E4 1위 댓글에는 삭제된 댓글 하나가 있었고 내용에 접근할 수 없으므로 삭제 사실만 확인했다.

## S5E7: Predict the Introverts from the Extroverts

상위 10위 조건을 만족하는 공식 해법 글이 없으므로 분석할 순위별 글도 없다.
유일한 42위 글의 방법과 댓글은 표본 제외 판단까지만 확인했고, 상위권 반복 근거로 사용하지 않았다.
Accuracy용 40% 문턱은 현재 S6E8의 ROC AUC가 요구하지 않는 결정이므로 과제에도 직접 적용되지 않는다.

## S5E6: Predicting Optimal Fertilizers

### 1위: Chris Deotte

[1st Place - Fast GPU Experimentation with RAPIDS cuDF cuML](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi)는 Chris Deotte의 1위 글이다.
목록에는 댓글 137개가 표시됐고 상세 화면에는 일반 댓글 99개와 감사 댓글 12개가 표시됐다.

여덟 입력 열을 전부 범주형으로 보고 28개 쌍, 56개 삼중 조합과 70개 사중 조합을 만들었다.
일대다 목표 평균 부호화를 일곱 비료별로 만들고 합성 훈련 자료와 원본 자료에 각각 적용해 XGBoost 입력을 약 2,268열로 늘렸다.
원본 자료는 별도 행과 파생 열로 모두 활용했다.
XGBoost와 신경망을 여러 단계로 쌓고, 반복 KFold, 선형 로짓을 base margin으로 주는 보정, 의사 라벨, 전체 자료 재학습과 탐욕 결합을 사용했다.
최종 제출은 약 300개 예측을 아홉 모형 묶음으로 결합했으며 CV MAP@3 0.386, Public 0.38450, Private 0.38652를 보고했다.

댓글에서 작성자는 고차 조합과 훈련 및 원본 목표 부호화까지 합치면 약 3,000열, 750,000행, A100 80GB에서 약 50GB를 썼다고 설명했다.
단일 5-fold 대형 XGBoost의 CV는 0.37970으로 아주 높지 않았지만 기존 결합 0.384를 0.386으로 올렸다고 답해 약한 단독 모형의 다양성 효과를 강조했다.
XGBoost는 깊이 4, 열 표본 비율 0.2, 행 표본 비율 0.5를 썼고 조건부 z-score 묶음 통계는 작은 이득만 냈다.
목표 부호화 통계를 평균 외로 늘리면 약 15,000열이 되고 이항 목표의 최솟값과 최댓값은 상수에 가까워 평균만 남겼다고 답했다.
결합과 2단 학습에서는 모든 OOF가 같은 fold를 써야 하며, 2단 모형을 전체 OOF에 다시 맞춘 판은 누출 때문에 개선되지 않았다고 경고했다.
원본 자료가 강한 대회에서는 원본 생성 구조 복원의 효용이 줄고 원본 신호가 약하거나 무작위에 가까울 때 더 유용하다고 보충했다.

S6E8 판단은 기존 결정과 중복이다.
자체 풀의 exp035_lattice_te, exp027_recon_ce, exp197_issue419_lgb_recon_ce_fixed20과 원본 프록시 계열이 고차 결합 및 원본 목표 통계의 핵심 관점을 덮는다.
외부 278개에도 foldsafe_te_multi, cat_nested_te, 여러 lattice와 원본 파생 모형이 있어 같은 정보 관점을 더 넓게 포함한다.
고차 조합 하나의 제거 기여가 없고 현재 과제의 쌍 및 삼중 조합 음성 결과를 뒤집는 독립 근거도 아니므로 새 후보로 열지 않는다.

### 2위: Masaya Kawamata

[2nd Place Solution - L3 Ensemble of 100+ OOFs](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/masaya-kawamata-2nd-place-solution-l3-ensemble-of-)는 Masaya Kawamata의 2위 글이다.
목록에는 댓글 36개가 표시됐고 상세 화면에는 일반 댓글 30개와 감사 댓글 1개가 표시됐다.

다섯 층화 fold로 약 100개 OOF를 만들고 2단에서 XGBoost, LightGBM, 로지스틱 회귀와 신경망, 3단에서 탐욕 결합을 사용했다.
1단에는 여러 XGBoost와 LightGBM, CatBoost, 원본 자료 선학습 뒤 합성 자료 이어 학습, 구간화 및 군집화, 목표 부호화, 지도형 자동부호화 잠재 표현, ExtraTrees와 랜덤 포리스트, TabTransformer가 포함됐다.
약한 특성 공학과 잠재 표현 모형도 최종 다양성에는 도움이 됐다고 보고했다.
2단 CV는 신경망 0.38321, LightGBM 0.38343, XGBoost 0.38348, 로지스틱 회귀 0.38362였고 3단은 0.38418이었다.
로지스틱 회귀와 신경망의 입력 OOF 부분집합을 Optuna로 고르면 로지스틱 회귀 CV가 약 0.3828에서 0.3836, 신경망은 약 0.3825에서 0.3832로 올랐다고 했다.
이 선택으로 Private도 약 0.38472에서 0.38518 또는 0.38527로 올랐다고 보고했다.

댓글에서 작성자는 2단 OOF도 1단과 같은 fold 관계를 지키는 것이 중요하다고 답했다.
fold를 다시 나누지 않은 탐욕 결합도 거의 같은 결과였지만 이를 일반적인 누출 안전성 증거로 제시하지는 않았다.
원본 선학습은 XGBoost의 xgb_model 인수로 원본 자료 모형을 이어 학습했다고 설명했다.

S6E8의 대규모 OOF와 선형 또는 비선형 결합은 기존 결정과 중복이다.
자체 35개에는 XGBoost, LightGBM, CatBoost, AutoGluon, RealMLP, TabPFN, Lookup-Transformer와 여러 신경망이 있고 외부 278개는 더 넓은 모형군을 이미 제공한다.
지도형 자동부호화 잠재 표현은 자체 풀과 외부 장부의 명시적 방법에서 비어 있으나 단독 점수나 제거 기여가 없어 근거 부족이다.

### 3위: Mahog

[3rd place - Ridge and CV are all you need](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/mahog-3rd-place-ridge-and-cv-are-all-you-need)는 Mahog의 3위 글이다.
목록에는 댓글 10개가 표시됐고 상세 화면에는 일반 댓글 8개가 표시됐다.

60개가 넘는 모형의 일곱 클래스 확률을 이어 붙이고 목표를 one-hot 다중 회귀로 바꾼 Ridge를 최종 결합기로 사용했다.
구성원 수가 늘자 탐욕 결합과 유전 알고리즘의 개선이 멈췄지만 Ridge는 약 1분 안에 안정적으로 동작했다고 보고했다.
모형군은 XGBoost 약 20개, LightGBM과 CatBoost 각 5개, RealMLP, TabM, SAINT, TabTransformer, GRN, Gandalf, LNN, 랜덤 포리스트와 ExtraTrees를 포함했다.
곱 특성을 범주로 부호화한 변형은 약 0.0001을 더했다고 보고했다.
댓글에는 방법을 바꾸는 보충이나 실패 반례가 없었다.

S6E8 판단은 기존 결정과 중복이다.
현재 313개 결합이 더 큰 OOF 폭에서 Ridge를 포함한 19개 전략을 비교했고 shrunk_rank_logit_logistic을 선택했으므로 과거 대회의 Ridge 우승만으로 결합기를 바꾸지 않는다.

### 4위: hahahaj

[4th Place - Stacking Ensemble using XGB only](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/hahahaj-4th-place-stacking-ensemble-using-xgb-only)는 hahahaj의 4위 글이다.
목록에는 댓글 18개가 표시됐고 상세 화면에는 일반 댓글 17개와 감사 댓글 2개가 표시됐다.

제목의 XGB only는 전체 구성원이 아니라 주로 2단 결합기를 가리킨다.
약 50개 구성원에는 TabTransformer, YDF, CatBoost, LightGBM, XGBoost, 신경망, AutoGluon, 랜덤 포리스트와 선형 모형이 포함됐다.
최종 CV는 0.3842, Public은 0.38384, Private은 0.38454였다.
신경망 단독 CV는 약 0.350에서 0.365로 약했지만 결합을 개선했고, XGBoost 2단은 복잡도를 조절하기 쉬워 선택했다고 설명했다.
댓글은 축하와 일반 질문이 중심이었고 독립 제거 대조는 제시되지 않았다.

S6E8 판단은 기존 결정과 중복이다.
약한 신경망도 다른 오차를 제공할 수 있다는 관찰은 현재 외부 278개의 폭과 구성원 절제 판정이 이미 더 직접적으로 다룬다.
비선형 2단 XGBoost는 자체 nested OOF에서 음성이었으므로 다시 열지 않는다.

### 5위: Optimistix

[5th Place Solution - An ensemble of 53 OOFs](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/optimistix-5th-place-solution-an-ensemble-of-53-oo)는 Optimistix의 5위 글이다.
목록에는 댓글 13개가 표시됐고 상세 화면에는 일반 댓글 11개가 표시됐다.

53개 OOF는 GBDT, 신경망, 랜덤 포리스트와 로지스틱 회귀를 포함했고 원본 자료 반복 횟수, 수치 열의 범주 취급과 fold 수를 달리했다.
Private 0.385를 넘는 해법 세 개를 만들었으며 Public이 가장 좋은 판은 Private 0.38502였고 CV가 가장 좋은 미선택 판은 Private 0.38509였다고 보고했다.
이 결과를 Public 기준 선택의 작은 역전 사례로 제시했다.
댓글에는 핵심 방법을 바꾸는 보충이 없었다.

S6E8 판단은 기존 결정과 중복이다.
원본 반복과 다양한 학습기 결합은 현재 풀에 이미 있고 Public으로 후보를 고르지 말라는 교훈은 ADR 0001의 현행 계약과 같다.

### 6위: paperxd

[6th place - 1 week rush](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/paperxd-6th-place-1-week-rush)는 paperxd의 6위 글이다.
목록과 상세 화면 모두 댓글 6개가 표시됐고 감사 댓글 1개가 따로 표시됐다.

일주일 동안 직접 만든 135개 OOF 가운데 약 70%는 XGBoost, 약 30%는 LightGBM이었고 최종은 로지스틱 회귀와 탐욕 결합의 평균이었다.
모든 입력을 범주로 취급하고 원본 자료를 추가했으며 XGBoost의 gradient_based 표본추출과 refresh_leaf 0 설정을 사용했다.
CatBoost와 여러 특성 공학은 실패했다고 적었고 로지스틱 회귀 결합에는 약 3시간이 걸렸다.
댓글은 실행 시간과 OOF 규모 확인이 중심이었고 별도 제거 기여는 없었다.

S6E8 판단은 기존 결정과 중복이다.
XGBoost와 LightGBM 폭, 범주 복제, 원본 자료와 로지스틱 결합은 자체 35개와 외부 278개가 모두 덮는다.

### 7위: Haruki Kakinuma

[7th place solution - HC + Ridge](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/haruki-kakinuma-7th-place-solution-hc-ridge)는 Haruki Kakinuma의 7위 글이다.
목록에는 댓글 10개가 표시됐고 상세 화면에는 일반 댓글 9개와 감사 댓글 1개가 표시됐다.

1단에서 원본 자료 반복 횟수, 수치 열의 범주 또는 정수 처리와 규제를 달리한 XGBoost 일곱 개와 LightGBM 여섯 개를 주로 남겼고 신경망도 포함했다.
2단과 3단에서 Ridge와 탐욕 결합을 계층적으로 사용했으며 비선형 신경망과 XGBoost 2단은 실패했다고 보고했다.
3단 탐욕 결합은 OOF 0.38396, Private 0.38460이었고 Ridge는 0.38368과 0.38449였다.
4단 결합은 OOF 0.38412, Private 0.38486이었다.
댓글에서 계층 구조는 미리 정한 설계가 아니라 실험을 반복하며 생겼고 LightGBM 전용 Ridge와 XGBoost 및 신경망 Ridge를 먼저 따로 만들었다고 설명했다.

S6E8 판단은 기존 결정과 중복이다.
계열별 선형 결합 뒤 다시 합치는 방법은 자체 계보 묶음과 현재 순위 로짓 결합이 더 엄격한 nested OOF로 다룬다.
비선형 2단 실패는 자체 결과와 같은 방향의 반례다.

### 10위: Ole-Jakob

[10th Place Solution - 350 oofs to 9 hillclimbing versions to Final Autogluon ensemble](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/ole-jakob-10th-place-solution-350-oofs-9-hillclimb)는 Ole-Jakob의 10위 글이다.
목록에는 댓글 9개가 표시됐고 상세 화면에는 일반 댓글 8개가 표시됐다.

약 350개 OOF에 20가지 탐욕 결합 변형을 적용했고 최고 탐욕 결합은 14개 OOF로 CV 0.383830을 얻었다.
좋은 탐욕 결합 아홉 개를 AutoGluon으로 다시 합쳐 CV 0.383918, Private 0.38472를 보고했다.
최고 단일 XGBoost는 고정 6-fold, 원본 자료 표본 가중치 4에서 CV 0.37799였다.
예측 확률 여백이 작고 정답이 두 번째, 세 번째 또는 네 번째 순위에 놓인 OOF 행의 가중치를 올리는 방법이 약 0.001 CV를 더했다고 보고했다.
약 200개 특성 공학은 단일 모형을 개선하지 못했지만 일부는 다양성 때문에 최종 선택됐다.
DAP와 Urea 전용 모형 및 별도 가중치는 실패했다.

댓글에서 작성자는 OOF와 시험 예측을 같은 구조로 저장해 재사용했고 첫 열 개 검증 인덱스로 구성원 fold 일치를 검사했다고 설명했다.
탐욕 결합은 MAP@3의 국소 최적점 때문에 구성원을 더할수록 나빠질 수 있다고 경고했다.
최고 XGBoost의 깊이는 17이었다고 보충했다.

순위 오류 기반 표본 가중은 현재 과제에 부적합하다.
이는 한 행의 정답 클래스가 세 후보 중 몇 번째인지에 직접 반응하는 MAP@3 전용 장치이고, 이진 ROC AUC에는 같은 두 번째 또는 세 번째 오답 구조가 없다.
OOF 구조화와 fold 일치 검사는 현재 장부와 감사 절차에 이미 반영돼 있다.

## S5E5: Predict Calorie Expenditure

### 1위: Chris Deotte

[1st Place - GPU Hill Climbing](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/chris-deotte-1st-place-gpu-hill-climbing)은 Chris Deotte의 1위 글이다.
목록에는 댓글 77개가 표시됐고 상세 화면에는 일반 댓글 63개와 감사 댓글 10개가 표시됐다.

수백 개 후보를 만든 뒤 GPU 탐욕 결합으로 일곱 개를 선택했다.
목표 부호화 XGBoost 세 개는 각각 CV 0.060대의 약한 모형이지만 최종 가중치 합이 25%였다.
곱, 로그, 합, 차와 비율 특성의 XGBoost는 CV 0.05951, 아홉 개 같은 폭 구간과 그룹 z-score의 CatBoost는 0.05937이었다.
선형 모형 잔차를 학습한 신경망은 CV 0.05999, 신경망 잔차를 학습한 XGBoost는 0.05989였다.
최종 결합은 CV 0.05880, Public 0.05677, Private 0.05841이었다.
전체 자료 재학습에서는 GBDT 반복 수를 fold 학습 반복 수보다 1/(K-1)만큼 늘리고 여러 시드를 평균했다.

댓글에서 같은 폭 구간화는 합성 자료에서 특히 잘 작동하지만 일반 자료에도 쓸 수 있다고 설명했다.
잔차 모형은 앞 모형과 다른 학습기 계열일 때 가장 유용했고 목표 비율 학습은 이 대회에서 실패했다고 답했다.
모든 구성원에 같은 fold를 쓴 이유는 다른 fold가 만드는 다양성이 시험 예측에서는 사라지기 때문이라고 설명했다.
RBF SVR은 A100에서 fold마다 무작위 100,000행 다섯 묶음을 학습해 평균했다고 보충했다.

S6E8 판단은 기존 결정과 중복이다.
자체 exp011_resid_pair, exp023_orig_proxy_residual과 외부 view_resid_cat, view_resid_lgbm, view_resid_xgb가 교차 학습기 잔차 관점을 이미 제공한다.
같은 fold, 전체 자료 재학습 반복 수와 시드 평균은 현행 재학습 계약이 따로 고정한다.
약한 목표 부호화 구성원의 다양성도 외부 278개 구성원 절제로 현재 자료에서 직접 판정됐다.

### 2위: Mahog

[2nd place - Trust CV and diversity](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/mahog-2nd-place-trust-cv-and-diversity)는 Mahog의 2위 글이다.
목록과 상세 화면 모두 댓글 6개가 표시됐다.

74개 OOF를 Ridge로 합친 판을 최종 선택했지만 사후 최고 Private은 11개 모형의 양수 탐욕 결합이었다.
모형군에는 AutoGluon, CatBoost, 목표 부호화 LightGBM, GOSS와 Huber 손실, 선형 모형, ResMLP와 LNN이 포함됐다.
연속 목표를 구간 분류한 확률 위에 CatBoost 잔차 모형을 얹는 변형도 포함했다.
댓글은 축하와 코드 공개 요청이 중심이었고 각 구성원의 제거 기여는 보고하지 않았다.

S6E8 판단은 대부분 기존 결정과 중복이다.
다양한 OOF와 Ridge는 현재 확장 결합보다 좁고, 목표 구간 분류 뒤 잔차 회귀는 이진 목표에 대응하는 별도 하위 과제가 없어 현재 과제에 부적합하다.

### 3위: nice kazusan

[3rd Place - Diversity and Hill Climbing](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/nice-kazusan-rd-place-diversity-and-hill-climbing)은 nice kazusan의 3위 글이다.
목록과 상세 화면 모두 댓글 2개가 표시됐고 감사 댓글 1개가 따로 표시됐다.

CatBoost, LightGBM, XGBoost, 신경망과 AutoGluon을 합쳐 18개 모형을 만들었다.
AutoFeat와 Optuna로 특성과 설정을 찾았고 원본 자료는 CV가 약해 사용하지 않았다.
선형 2단 결합의 CV 0.05893에서 여섯 모형 탐욕 결합으로 0.05885까지 개선했다.
댓글에는 방법을 바꾸는 보충이 없었다.

S6E8 판단은 기존 결정과 중복이다.
학습기 폭, 선형 결합과 탐욕 결합은 현재 313개 결합이 더 넓고 엄격한 기준으로 비교한다.

### 4위: AngelosMar

[4th Place Solution - Ridge Ensemble of 12 Models](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/angelosmar-4th-place-solution-ridge-ensemble-of-12)는 AngelosMar의 4위 글이다.
목록에는 댓글 9개가 표시됐고 상세 화면에는 일반 댓글 7개와 감사 댓글 2개가 표시됐다.

12개 Ridge 결합은 CV 0.05868, Public 0.05698, Private 0.05846이었고 11개 판은 0.05870, 0.05688, 0.05847이었다.
1단과 2단의 ExtraTrees, 신경망과 LightGBM 예측 약 30개 중 순차 특성 선택으로 12개를 골랐다.
AutoGluon 단일 실행은 약 15시간이 걸렸고 CV 0.058800이었다.
안쪽 fold 선형 모형의 잔차를 XGBoost로 학습하고 선형 예측을 원시 특성과 함께 넣었다.
약 400개 특성의 선형 모형은 0.05976, 신경망은 0.05954였고 277개 목표값을 다중표지로 바꾼 모형도 만들었다.
댓글에서 2단 입력을 만드는 안쪽 fold 구조와 순차 선택 순서를 보충했지만 별도 제거 기여는 없었다.

S6E8 판단은 기존 결정과 중복이다.
OOF 예측과 원시 특성을 함께 받는 잔차 XGBoost는 이슈 307의 조건부 후보였으나 이후 현재 자료의 비선형 2단 nested OOF가 음성이었다.
과거 회귀 한 건의 성공만으로 그 결정을 되돌리지 않는다.

### 5위: Alan1305

[5th Place Solution: Ensemble of 68 models](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/alan1305-5th-place-solution-ensemble-of-68-models)는 Alan1305의 5위 글이다.
목록과 상세 화면 모두 댓글 4개가 표시됐다.

나무 모형 2,000개 이상과 MLP 열 개를 약 열흘 동안 학습했다.
입력 순서를 무작위로 바꾼 순차 탐욕 선택을 수천 번 반복해 79개를 고른 뒤 cvxpy 이차 계획으로 68개를 남겼다.
이차 계획은 순서에 의존하지 않는 결정적 가중치를 주며 CV 0.0588017, Public 0.05671, Private 0.05846을 보고했다.
열 개 MLP 가운데 일곱 개가 선택돼 단독 성능보다 다양성이 중요했다고 해석했다.
댓글에는 이차 계획 설정 문의가 있었지만 별도 일반화 대조는 없었다.

S6E8 판단은 기존 결정과 중복이다.
구성원 수와 최적화 기법을 늘리는 것 자체는 새 정보 관점이 아니며 현재 313개에서 로지스틱 축소 결합이 직접 선택됐다.

### 6위: Omid Baghcheh Saraei

[6th Place Solution](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/omid-baghcheh-saraei-6th-place-solution)은 Omid Baghcheh Saraei의 6위 글이다.
목록에는 댓글 17개가 표시됐고 상세 화면에는 일반 댓글 14개가 표시됐다.

30개 모형 Ridge는 CV 0.05884, Public 0.05669, Private 0.05846이었고 탐욕 결합은 0.05879, 0.05670, 0.05848이었다.
외부 FLAML과 AutoGluon 출력을 별도 노트북에서 만들고 같은 fold로 결합했다.
댓글에서 관련 수치 열을 AutoGluon에서 범주형으로 취급한 변형이 다양성을 더했다고 답했다.
나머지 댓글은 코드 위치와 축하가 중심이었다.

S6E8 판단은 기존 결정과 중복이다.
AutoML, 같은 fold와 수치 열 범주 복제는 자체 exp117_ag25_gbm_r21, CatBoost 정확값 계열과 외부 자동 및 범주형 모형이 이미 덮는다.

### 7위: Mahdi Ravaghi

[7th place solution](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/mahdi-ravaghi-7th-place-solution)은 Mahdi Ravaghi의 7위 글이다.
목록과 상세 화면 모두 댓글 12개가 표시됐고 감사 댓글 1개가 따로 표시됐다.

대부분 CatBoost를 사용했고 특성 공학과 원본 자료는 약했으며 AutoGluon도 좋지 않았다고 보고했다.
CV는 탐욕 결합이 가장 좋았지만 최종 순위는 Ridge가 더 안정적이었다.
댓글에서 Public 점수로 혼합을 조정하는 위험과 단순 결합의 안정성을 강조했다.

S6E8 판단은 기존 결정과 중복이다.
CatBoost와 선형 결합은 이미 포함되고 Public 비사용 규칙도 현행 계약과 같다.

### 8위: pinoystat

[8th Place Solution for the Predict Calorie Expenditure Competition](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/pinoystat-8th-place-solution-for-the-predict-calor)는 pinoystat의 8위 글이다.
목록과 상세 화면 모두 댓글 4개가 표시됐다.

TensorFlow 모형 세 개, CatBoost와 XGBoost를 탐욕 결합해 CV 0.05846814를 보고했다.
별도 허용 오차나 넓은 후보 풀 없이 다섯 모형만 사용했다.
댓글은 실행 문의와 축하가 중심이었고 특성 또는 구성원 제거 대조는 없다.

S6E8 판단은 기존 결정과 중복이다.
신경망, CatBoost, XGBoost와 탐욕 결합은 자체 및 외부 풀에 모두 있다.

### 9위: Iqbal Syah Akbar

[9th Place Solution, 9 Models in the Ensemble](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/iqbal-syah-akbar-9th-place-solution-9-models-in-th)는 Iqbal Syah Akbar의 9위 글이다.
목록에는 댓글 11개가 표시됐고 상세 화면에는 일반 댓글 10개와 감사 댓글 2개가 표시됐다.

성별에 M-estimate 부호화를 적용하고 역수 특성을 신경망과 선형 모형에, 다항 특성을 선형 모형에 사용했다.
표준화 뒤 Nystroem 다항 근사와 Ridge를 연결한 모형도 만들었다.
훈련, 시험과 원본 자료를 목표 없이 합친 랜덤 포리스트의 잎 표현을 CatBoost 입력으로 써 새로운 다양성을 만들었다.
최종 모형군은 신경망, 선형 및 Ridge, CatBoost, 랜덤 포리스트, XGBoost와 LightGBM이었고 Ridge로 먼저 거른 뒤 Optuna로 가중치를 찾았다.
원본 자료는 fold를 나눈 뒤 학습 부분에만 붙여 검증 부분을 시험 자료와 같은 상태로 유지했다.
댓글에는 자료 추가 순서와 특성 생성 설명이 있었지만 랜덤 포리스트 표현의 단독 이득은 보고하지 않았다.

비지도 랜덤 포리스트 잎 표현은 근거 부족이다.
자체 35개에 정확히 같은 표현은 없고 외부 278개에는 랜덤 포리스트 예측 구성원과 비선형 SVM이 있지만 잎 표현을 다른 모형에 주는 방식과 같다고 확인할 수 없다.
한 9위 글에서 다양성 구성원으로만 보고됐고 제거 기여가 없으므로 새 실험을 열지는 않는다.

## S5E4: Predict Podcast Listening Time

### 1위: Chris Deotte

[1st Place - RAPIDS cuML Stack - 3 Levels](https://www.kaggle.com/competitions/playground-series-s5e4/writeups/chris-deotte-1st-place-rapids-cuml-stack-3-levels)는 Chris Deotte의 1위 글이다.
목록에는 댓글 152개가 표시됐고 상세 화면에는 일반 댓글 122개와 감사 댓글 12개가 표시됐다.

목표 청취 시간은 Episode_Length_minutes의 약 0.72배였고 이 열 하나가 신호의 90% 이상을 담았지만 행의 11.6%에서 결측이었다.
작성자는 중요한 열이 있는 행과 없는 행에서 서로 다른 구성원이 잘 작동하므로 선형 탐욕 결합이나 Ridge가 아닌 비선형 2단이 필요하다고 판단했다.
같은 5-fold로 75개 1단 모형을 만들고 XGBoost와 MLP 2단을 같은 비율로 합친 3단 구조를 사용했다.
1단에는 Lasso, 선형 SVR, KNN, 랜덤 포리스트, MLP, TabPFN, XGBoost, LightGBM, 선형 및 신경망 잔차 GBDT와 AutoGluon이 포함됐다.
특성 집합과 GBDT 깊이를 달리하고 Episode_Length_minutes를 전부 제거한 모형, 목표를 이 열로 나눈 비율 모형, 이 열 자체를 훈련과 시험 자료로 예측한 모형, 의사 라벨 모형을 만들었다.
이 열의 예측은 결측 대치, 전체 열 교체, 목표 비율 예측과의 곱에 각각 사용했다.
1단 단독 CV는 약 11.8부터 13.2였고 73개 OOF를 받은 XGBoost와 MLP 2단은 각각 11.56, 같은 비율 3단은 11.54였다.
최종은 CV 11.54, Public 약 11.50에서 11.51, Private 11.44였다.
사후 같은 73개 구성원의 탐욕 결합은 CV 11.64, Public 11.57, Private 11.503이었고 비선형 스택은 11.54, 11.51, 11.448이었다.

댓글에서 2단에는 1단 OOF뿐 아니라 Episode_Length_minutes 결측 표시를 포함한 원시 특성도 함께 넣었다고 설명했다.
전체 CV가 24에서 26인 매우 약한 결측 구간 전문 모형도 일부 행에서는 RMSE 8처럼 강하거나 강한 두 모형을 고르는 신호가 돼 결합을 크게 개선할 수 있다고 답했다.
범주 one-hot 110열과 Episode_Length_minutes의 곱, 모든 one-hot 쌍과 이 연속 열의 곱을 만들어 총 6,103열의 float32 선형 SVR 입력을 구성했다.
이 선형 모형은 750,000행에서 약 18GB였고 RAPIDS LinearSVR 학습은 약 10분이었다.
범주 삼중 곱은 약 200,000열이 더 생겨 만들지 않았다.
1단 GBDT는 500에서 1,500개 특성을 썼고 그룹별 평균, 최솟값, 최댓값, 중앙값과 표준편차 부호화를 포함했다.
2단 XGBoost는 깊이 10이었고 원시 특성을 다시 넣는 재결합이 결측 구간별 구성원 선택에 도움이 됐다고 답했다.
매일 약 12개 새 모형을 한 개씩 추가해 1개에서 3개만 남기는 전진 선택을 했고 모든 실험의 OOF와 시험 예측을 저장했다.

의사 라벨 댓글은 중요한 누출 반례를 남겼다.
단일 모형이나 단순 혼합은 fold 학습 부분에 시험 의사 라벨을 붙이는 일반 절차를 쓸 수 있지만, 그 OOF를 2단 학습에 쓰면 검증 행과 시험 행의 정보 조건이 달라진다고 설명했다.
스택용 OOF는 각 fold의 검증 행을 먼저 예측해 그 검증 행을 의사 라벨 자료로 넣은 별도 학습으로 만들고, 시험 예측은 시험 의사 라벨을 넣은 별도 학습으로 만들어 조건을 맞췄다.
의사 라벨을 한 번에 모두 넣어 외우는 것을 줄이기 위해 의사 라벨 부분의 서로 다른 80%를 다섯 번 학습해 평균했다고 보충했다.

S6E8의 비선형 2단과 원시 특성 재결합은 기존 결정과 중복이다.
현재 자료에서는 결측 구간별 선형 결합이 이미 있고 얕은 XGBoost 2단과 잔차 2단이 nested OOF에서 이기지 못했다.
S5E4는 한 열이 신호 90% 이상을 담고 동시에 11.6% 결측이라는 특수한 조건에서 2단이 탐욕 결합보다 CV 0.10과 Private 약 0.055를 개선한 사례다.
S6E8에는 같은 지배 열 및 결측 체제가 확인되지 않았으므로 이 성공이 현재 음성 결정을 뒤집지 않는다.

one-hot 범주와 지배 연속 열의 곱을 선형 모형에 주는 방법은 근거 부족이다.
자체 exp058_logreg_onehot에는 정확히 같은 집단별 기울기 상호작용이 없고 외부 278개에는 poly_svm과 여러 격자 모형이 있지만 같은 표현인지 확인할 수 없다.
그러나 S5E4에서도 단독 모형은 CV 13.2의 다양성 구성원이었고 제거 기여가 없으며 S6E8에서 대응하는 지배 연속 열도 정해지지 않았다.

### 2위: Farukcan Saglam

[2nd Place - Single LightGBM and Target Encoding](https://www.kaggle.com/competitions/playground-series-s5e4/writeups/farukcan-saglam-2nd-place-single-lightgbm-and-targ)는 Farukcan Saglam의 2위 글이다.
목록에는 댓글 50개가 표시됐고 상세 화면에는 일반 댓글 38개와 감사 댓글 1개가 표시됐다.

794,868행과 1,552개 특성을 쓴 단일 LightGBM을 CPU로 약 4시간 학습하고 전체 자료 다섯 시드를 평균했다.
원본 열과 반올림 열의 1개부터 6개 조합에 목표 부호화를 만들었다.
그 목표 부호화 값을 행별 평균, 표준편차, 최솟값과 최댓값으로 다시 요약했고 전체, 조합 차수별, 원본 및 반올림 출처별 요약을 따로 만들었다.
본문은 12,000회, 깊이 15, 학습률 0.008을 적었다.
댓글에는 실험 3,000회 학습률 0.1과 최종 12,000회 학습률 0.08이라는 답변도 있어 학습률 표기는 서로 충돌한다.

댓글에서 각 집계 차수의 행별 요약이 약 0.02씩, 전체가 약 0.05 RMSE 개선을 만들었다고 답했다.
폭 2 반올림 구간이 가장 강했고 마지막 주에 큰 도약을 만들었다고 설명했다.
다섯 Kaggle 노트북을 병렬로 돌렸고 float32와 CPU 메모리 약 29GB에서 30GB를 사용했다.

여러 목표 부호화의 행별 분포 요약은 근거 부족이다.
자체 풀은 고차 lattice와 recon CE를, 외부 278개는 foldsafe_te_multi, cat_nested_te와 여러 lattice를 포함해 원천 정보는 이미 덮는다.
다만 행별 평균, 표준편차, 최솟값과 최댓값으로 다시 압축하는 정확한 구현은 장부 이름만으로 확인되지 않는다.
한 2위 단일 모형의 저자 보고 제거 수치는 구체적이지만 회귀 목표와 최대 6차 조합에 의존하고 현재 자료에서 독립 재현이 없어 별도 실험으로 열지 않는다.

### 3위: Johannes Heller

[3rd Place - Target Encoding and 3 Levels](https://www.kaggle.com/competitions/playground-series-s5e4/writeups/johannes-heller-3rd-place-target-encoding-and-3-le)는 Johannes Heller의 3위 글이다.
목록에는 댓글 18개가 표시됐고 상세 화면에는 일반 댓글 14개와 감사 댓글 1개가 표시됐다.

1단은 LightGBM 열 개, XGBoost 다섯 개, CatBoost 네 개, 랜덤 포리스트 두 개, ExtraTrees 한 개와 HistGradientBoosting 네 개였다.
2단은 탐욕 결합과 LightGBM, 3단은 이 둘을 80%와 20%로 합쳤다.
첫 스택은 CV를 11.66에서 11.62로 개선했지만 작성자는 핵심 특성 하나를 놓쳤을 가능성을 언급했다.
문자열 2-gram부터 7-gram의 목표 평균 부호화를 사용했고 중앙값, 최솟값, 최댓값과 값 종류 수는 시도했지만 평균이 가장 좋았다.
카디널리티가 극단적으로 큰 조합은 버렸고 최고 모형은 목표 부호화 약 270개를 사용했다.
공통 5-fold 자료를 미리 parquet로 저장해 모든 구성원에 재사용했다.
소수 자릿수, 이상값 수정과 원본 자료 행 추가는 사용했고 비율, 선형 모형, 신경망, 결측 대치, 구간화, 군집화와 label encoding은 실패했다.

댓글에서 2단 LightGBM은 원시 특성과 OOF를 함께 받아 결측 또는 특성 구간에 따라 구성원을 조건부 선택할 수 있다고 설명했다.
깊이 19처럼 매우 깊은 나무가 잘 작동한 점은 이 자료의 높은 상호작용을 보여 준다고 해석했다.

S6E8 판단은 기존 결정과 중복이다.
고차 목표 부호화와 OOF 및 원시 특성 비선형 결합은 자체 음성 결과와 외부 격자 구성원이 이미 다룬다.
중앙값, 최솟값과 최댓값이 평균보다 약했다는 댓글은 이진 목표에서 평균 외 통계가 퇴화한다는 S5E6 1위 댓글과도 같은 방향이다.

### 4위: Ravi Ramakrishnan

[Rank 4 approach - lots of features, lots of simple models and a ridge blend](https://www.kaggle.com/competitions/playground-series-s5e4/writeups/ravi-ramakrishnan-rank-4-approach-lots-of-features)는 Ravi Ramakrishnan의 4위 글이다.
목록에는 댓글 32개가 표시됐고 상세 화면에는 일반 댓글 25개와 감사 댓글 1개가 표시됐다.

10-fold로 382개 모형을 만들고 2-gram부터 7-gram까지 상호작용과 결측 개수를 대량 생성했다.
XGBoost, LightGBM, CatBoost와 AutoGluon 예측을 모두 눈금 조정한 뒤 Ridge로 합쳤다.
최종 Ridge는 CV 11.614142, Public 11.64459, Private 11.54182였다.
A6000, 로컬 RTX 3090, Colab과 Runpod를 사용했고 메모리는 128GB에서 256GB, 비용은 150달러 이상이었다.
댓글에서 계산량과 저장 방식이 주로 논의됐고 개별 특성의 제거 기여는 없었다.

S6E8 판단은 기존 결정과 중복이다.
대규모 조합과 OOF Ridge는 현재 자체 35개와 외부 278개 결합이 더 넓고 현재 지표에서 직접 검증됐다.

### 5위: Optimistix

[5th place: 100 OOFs, laziness, and a blunder or two](https://www.kaggle.com/competitions/playground-series-s5e4/writeups/optimistix-5th-place-100-oofs-laziness-and-a-blund)는 Optimistix의 5위 글이다.
목록에는 댓글 8개가 표시됐고 상세 화면에는 일반 댓글 6개가 표시됐다.

100개 OOF에 Ridge 등을 적용했고 목표 부호화, 범주 쌍과 중앙값 목표 부호화를 사용했다.
30-fold XGBoost가 가장 강한 단일 모형이었고 Public 11.75387, Private 11.67004를 보고했다.
AutoGluon의 2단 OOF도 1단 구성원처럼 섞어 95개 입력에서 만든 AutoGluon과 나머지를 합쳐 99개 OOF를 사용했다.
최종 제출 하나는 극단값을 자르지 않아 Private 177.25가 됐고 자르면 4위 수준이었다고 사후 확인했다.
댓글에서 잔차 학습은 대부분 구성원이 GBDT라 겹치는 오차가 많아 큰 효과가 없었다고 설명했다.

S6E8 판단은 기존 결정과 중복이다.
OOF 폭, 중앙값 목표 부호화, AutoML과 Ridge는 이미 덮이며 시험 예측 범위 검사는 현재 제출 조립 검증의 운영 교훈으로만 남긴다.

### 6위: masaishi

[6th Place: Select Feature Combinations based on RMSE Scores](https://www.kaggle.com/competitions/playground-series-s5e4/writeups/masaishi-6th-place-select-feature-combinations-bas)는 masaishi의 6위 글이다.
목록에는 댓글 17개가 표시됐고 상세 화면에는 일반 댓글 16개가 표시됐다.

단일 5-fold LightGBM은 Public 11.70, Private 11.63이었다.
Episode_Length_minutes의 소수 자릿수가 둘보다 많으면 0.9554를 곱하고 Number_of_Ads가 소수 자릿수 셋보다 많은 일곱 행에는 1.0588을 곱하는 생성 흔적 수정을 사용했다.
같은 특성 조합의 그룹 평균과 네 개의 특정 그룹 조합을 추가했다.
주기형, 비율, 정수부 및 소수부와 다항 특성을 만들었고 Episode_Length_minutes의 내림 정수부가 가장 영향이 컸다.
원본 자료를 직접 행으로 붙이고 일치하는 원본 행도 활용했다.

1개부터 4개 열 조합의 그룹 평균 목표 부호화를 80:20 분할에서 체계적으로 검사했다.
그룹 개수, 표준편차, RMSE와 적용 행 비율을 기록했고 한 주에 50개가 넘는 실험을 돌렸다고 했다.
최종 결합에는 HistGradientBoosting, LightGBM, SVR, TabNet과 XGBoost가 포함됐고 전체 실험은 1,000개가 넘었다.
댓글에서 정수부와 소수 자릿수는 일반적인 관계라기보다 합성 생성기의 흐릿한 흔적으로 보인다고 답했다.

생성 흔적 상수 수정과 회귀 그룹 평균은 현재 과제에 부적합하다.
S6E8에도 자릿수 및 정확값 관점은 있지만 현재 자체 및 외부 구성원이 현 자료에서 직접 판정했고, 과거 팟캐스트 열의 두 상수를 옮길 근거는 없다.
그룹 조합 선택 절차 자체는 exp035_lattice_te와 recon CE 계열의 기존 결정과 중복이다.

## 네 대회에서 반복된 패턴과 반례

### 다양한 OOF는 반복됐지만 무조건 많이 넣는 전략은 아니었다

S5E6의 여덟 포함 글 모두와 S5E5의 아홉 포함 글 대부분은 서로 다른 모형, 특성, 원본 자료 처리와 fold 수에서 나온 OOF를 선형 또는 탐욕 결합했다.
S5E4도 1, 3, 4, 5위가 3단 결합, 2단 결합 또는 대규모 Ridge를 사용했다.
약한 단독 모형이 최종 결합을 개선했다는 보고는 S5E6 1, 2, 4, 10위와 S5E4 1위에서 반복됐다.

반례도 분명하다.
S5E6 10위는 탐욕 결합이 MAP@3 국소 최적점 때문에 구성원을 더할수록 나빠질 수 있다고 했고, S5E5 5위는 2,000개가 넘는 후보를 68개로 줄였다.
최신 S6E8 외부 사다리에서도 약한 nhtquyn 고전 확률 모형 120개를 뺀 구성이 전체 추가판보다 좋아졌다.
따라서 과거 상위권의 공통점은 모형 수 자체가 아니라 OOF와 시험 예측을 짝으로 보존하고 현재 검증에서 한계 기여를 판정했다는 점이다.

### 같은 fold는 성능 기법이면서 누출 통제였다

S5E6 1, 2, 7, 10위와 S5E5 1, 6위는 구성원과 2단 모형의 fold 관계를 맞추는 것을 명시했다.
S5E4 1위는 의사 라벨을 포함한 스택에서 OOF와 시험 예측의 정보 조건이 달라지는 문제를 별도 학습 경로로 해결했다.
S5E4 3위는 fold별 목표 부호화 자료를 미리 저장해 모든 모형이 같은 누출 없는 입력을 쓰게 했다.

이는 새 모형 후보가 아니라 현재 계보 규율을 유지할 근거다.
외부 278개 장부도 OOF와 시험 예측 짝, 행 순서, fold 근거와 상류 주의 사항을 구성원별로 기록하고 있다.

### 비선형 2단은 행별 체제가 실제로 다를 때만 강했다

S5E4 1위에서는 지배적 열이 있는 행과 결측인 행이라는 두 체제가 명확했고 비선형 스택이 같은 73개 구성원의 탐욕 결합보다 CV 0.10과 Private 약 0.055를 개선했다.
S5E4 3위도 원시 특성과 OOF를 함께 받은 2단 LightGBM으로 CV를 약 0.04 개선했다.
반면 S5E6 7위는 신경망 및 XGBoost 2단이 실패했고 S5E5 4위와 1위의 잔차 결합은 서로 다른 학습기 계열일 때만 유용했다.
S5E4 5위는 대부분 같은 GBDT 계열이라 잔차 학습이 중복됐다고 설명했다.

따라서 단계 수만 늘리는 일반 규칙은 없다.
현재 자료에서 행별 체제를 정의하는 사전 근거가 있고 그 체제 표시가 바깥쪽 학습 부분 안에서 만들어질 때만 다시 검토할 수 있다.

### 목표 부호화의 효용은 과제와 목표 형태에 강하게 의존했다

S5E6 1위는 최대 4차 범주 조합의 클래스별 평균을, S5E4 2위는 최대 6차 조합을, S5E4 3위는 문자열 2-gram부터 7-gram을 사용했다.
S5E4 2위는 여러 목표 부호화의 행별 분포 요약에 구체적 개선을 보고했다.
그러나 S5E6 1위는 평균 외 통계가 이항 및 다중 분류에서 열 폭만 키운다고 했고 S5E4 3위도 중앙값, 최솟값과 최댓값이 평균보다 약했다고 보고했다.
S5E5 1위의 약한 목표 부호화 모형은 단독 점수보다 다양성으로 선택됐다.

현재 S6E8은 이진 목표이므로 회귀 목표 분포의 중앙값, 최솟값, 최댓값과 여러 봉우리 자체를 옮길 수 없다.
결합값별 양성률과 교차 적합 부호화는 이미 자체 및 외부 풀에서 판정됐다.

### Public은 방법 발견의 참고일 뿐 선택 기준이 아니었다

S5E6 5위는 Public이 가장 좋은 판보다 CV가 가장 좋은 미선택 판의 Private이 높았다.
S5E5 7위는 CV가 좋은 탐욕 결합보다 Ridge를 선택했고, S5E5 제외 글에는 Public 3위에서 Private 40위와 68위 과적합 사례가 있었다.
S5E4 5위의 극단값 미절단은 Public 선택과 무관하게 제출 안전 검사가 필요하다는 사례다.

현재 ADR 0001이 Public을 채택 판단에서 제외하고 전체 OOF, 바깥쪽 분할과 중복 및 한계 기여를 요구하는 이유를 강화한다.

## 현재 S6E8에 대한 적용 판단

현재 자체 후보 풀은 [후보 풀 장부](../../artifacts/pool.yaml)의 35개다.
자체 풀은 원본 프록시, 잔차, 격자 및 재구성 목표 부호화, 정확값 범주 복제, 결측 대치, 선형 one-hot, XGBoost, LightGBM, CatBoost, AutoGluon, TabPFN, RealMLP, TabM, Lookup-Transformer, 표 형태 합성곱 신경망과 문맥 스플라인을 포함한다.

최신 외부 결합은 [두 번째 넓힌 확장 결합 기록](extended-stack-submission-2.md)의 실행 443b3a71a2b045ba9052fbb3d821255d다.
외부 278개에는 여러 XGBoost, LightGBM, CatBoost, 랜덤 포리스트, MLP, RealMLP, TabM, FT-Transformer, TabTransformer, Trompt, 선형 및 다항 및 RBF SVM, 다중 목표 부호화, 격자, 결측 대치, 잔차 관점과 공개 노트북 OOF가 포함된다.
최종 313개는 shrunk_rank_logit_logistic으로 nested OOF 0.9703509469와 가중 OOF 0.9712170271을 얻었다.

[기존 1년 조사와 실험 발주 기준인 이슈 307](https://github.com/tmheo/predicting-smartphone-addiction/issues/307)의 네 범주는 새 후보, 기존 결정과 중복, 현재 과제에 부적합, 근거 부족이다.
아래 표는 자체 35개 안의 빈 관점인지와 외부 278개가 이미 덮는 관점인지를 분리한 예비 판정이다.

| 조사 관점 | 자체 35개 기준 | 외부 278개 기준 | 이슈 307 예비 분류 | 판단 |
| --- | --- | --- | --- | --- |
| 새 후보 | 해당 없음 | 해당 없음 | 새 후보 | 현재 바로 열 실험 없음 |
| 고차 범주 조합 목표 평균과 원본 자료 목표 통계 | exp035_lattice_te, exp027_recon_ce, exp197_issue419_lgb_recon_ce_fixed20과 원본 프록시 계열이 포함됨 | foldsafe_te_multi, cat_nested_te, lattice 및 exact 계열이 포함됨 | 기존 결정과 중복 | 같은 정보 관점 재실행 안 함 |
| 교차 학습기 잔차 보정 | exp011_resid_pair과 exp023_orig_proxy_residual이 포함됨 | view_resid_cat, view_resid_lgbm, view_resid_xgb가 포함됨 | 기존 결정과 중복 | 현재 결합의 제외 기여를 우선함 |
| OOF와 원시 특성을 함께 받는 비선형 2단 | 자체 nested OOF 음성 결정과 결측 구간별 선형 결합이 있음 | 2단 산출물은 원칙상 반입하지 않고 1단 구성원만 결합함 | 기존 결정과 중복 | 지배적 결측 체제가 새로 확인되지 않는 한 재개 안 함 |
| Ridge, 탐욕 결합, AutoGluon과 다층 선형 결합 | 19개 결합 전략 비교와 자체 계보 묶음이 있음 | 최신 313개가 더 넓은 OOF 폭을 제공함 | 기존 결정과 중복 | 과거 순위만으로 결합기 교체 안 함 |
| 중요한 열을 목표 없이 예측해 결측 대치 또는 전체 교체 | exp025_constrained_impute와 전이형 전처리 구성원이 인접함 | imp_cat, imp_lgbm, imp_xgb, lgb_missing_global과 tabm_missing이 인접함 | 기존 결정과 중복 | 현재 자료의 기존 대치 한계 기여로 판단 |
| 원본 자료 선학습 뒤 XGBoost 이어 학습 | 원본 프록시와 원본 분포 관점은 있으나 정확한 이어 학습은 없음 | 원본 및 합성 자료 변형 구성원은 있으나 정확한 구현은 장부로 확인 안 됨 | 근거 부족 | S5E6 2위 한 글에 있고 단독 기여가 없어 열지 않음 |
| 범주 one-hot 쌍과 지배 연속 열의 곱을 쓰는 선형 SVR 또는 Lasso | exp058_logreg_onehot은 인접하지만 집단별 기울기 곱은 없음 | poly_svm과 rbf_svm은 인접하지만 같은 표현은 아님 | 근거 부족 | 대응 지배 열과 제거 기여가 생길 때만 재검토 |
| 지도형 자동부호화 잠재 표현 | 명시적 구현 없음 | 장부 이름에서 같은 구현 확인 안 됨 | 근거 부족 | S5E6 2위의 약한 다양성 구성원이고 단독 수치 없음 |
| 비지도 랜덤 포리스트 잎 표현을 CatBoost 입력으로 사용 | 명시적 구현 없음 | rf 예측 구성원은 있으나 잎 표현 재사용은 확인 안 됨 | 근거 부족 | S5E5 9위 한 글이고 제거 기여 없음 |
| 여러 목표 부호화의 행별 평균, 표준편차, 최솟값과 최댓값 요약 | 고차 목표 부호화는 있으나 정확한 행 요약은 확인 안 됨 | 다중 목표 부호화와 격자 구성원은 있으나 정확한 행 요약은 확인 안 됨 | 근거 부족 | 회귀 한 대회의 저자 보고 제거 수치는 있으나 이진 목표 독립 근거 없음 |
| MAP@3 순위 오류 기반 표본 가중 | 대응 없음 | 대응 없음 | 현재 과제에 부적합 | 이진 ROC AUC에는 두 번째 및 세 번째 순위 오류가 없음 |
| 연속 목표의 구간 분류 확률 뒤 잔차 회귀 | 대응 없음 | 대응 없음 | 현재 과제에 부적합 | 이진 목표에는 별도 연속 목표 분해가 없음 |
| 지배 연속 열로 목표를 나눈 비율과 두 예측의 곱 | 대응 지배 열이 없음 | 대응 없음 | 현재 과제에 부적합 | 팟캐스트 회귀의 약 0.72배 구조에 의존함 |
| 소수 자릿수별 상수 보정과 특정 열 조합 하드코딩 | 자릿수 관점은 있으나 과거 상수는 없음 | identity_digit 계열이 현 자료 자릿수를 직접 판정함 | 현재 과제에 부적합 | 과거 합성 생성 흔적 상수를 옮기지 않음 |
| Public 점수로 구성원, 가중치, 반복 수 또는 제출을 선택 | ADR 0001이 금지함 | 외부 장부도 OOF와 fold 근거로만 구성원을 고름 | 현재 과제에 부적합 | Public 0.97135는 사후 참고만 유지 |

### 실제로 적용할 부분

새 모형 이슈는 열지 않지만 현재 절차에 유지할 근거는 강화됐다.

- 모든 구성원은 같은 행 순서의 OOF와 시험 예측을 짝으로 저장하고 fold 계보를 확인한다.
- 단독 OOF가 약해도 버리지 않되 현재 풀 포함 전후의 nested 한계 기여로만 다양성을 인정한다.
- 외부 구성원을 많이 모은 뒤에도 공급원 및 계열 단위 제거 대조로 해로운 묶음을 뺀다.
- 비선형 2단을 다시 검토하려면 먼저 원시 특성으로 정의되는 행별 체제와 전문 구성원의 구간별 이득을 바깥쪽 학습 부분 안에서 증명한다.
- 의사 라벨 OOF와 시험 예측은 정보 조건이 같아야 하며 일반 단일 모형용 의사 라벨 절차를 2단 입력에 그대로 쓰지 않는다.
- 전체 자료 재학습 반복 수, 시드 평균, 시험 예측 범위와 유한성은 [전체 자료 재학습 규약](../adr/0002-full-data-refit-protocol.md)과 제출 조립 검사를 유지한다.
- Public 점수는 후보 발굴의 사후 참고로만 기록하고 선택, 가중치와 중단 결정에는 쓰지 않는다.

### 조건부 아이디어의 재검토 조건

범주 one-hot과 연속 열 곱은 S6E8의 한 연속 열이 목표 신호를 지배하고 범주별 기울기가 안정적으로 다르다는 교차 적합 탐색 근거가 먼저 필요하다.
그 조건이 생기면 단일 선형 모형 한 개만 사전 고정하고 exp058_logreg_onehot 대비 정보 관점 및 nested 한계 기여를 판정해야 한다.

지도형 자동부호화와 비지도 랜덤 포리스트 잎 표현은 공개된 전체 5-fold OOF와 같은 fold 시험 예측이 먼저 필요하다.
그 예측이 자체 35개와 외부 278개에서 낮은 순위 상관과 충분한 단독 AUC를 보일 때만 현재 결합 포함 전후를 읽기 전용으로 대조한다.

목표 부호화 행 요약은 이진 목표에서 평균 외 통계가 실제로 상수가 아닌지 먼저 수학적으로 확인해야 한다.
정확한 구현이 외부 foldsafe_te_multi나 lattice 구성원에 이미 있으면 외부 구성원의 제외 기여로 판단하고 자체 재현을 중복 수행하지 않는다.

모든 후보 풀 진입과 최종 결합 교체는 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)을 그대로 따른다.
새 후보라는 이름만으로 실험을 열지 않고 자체 3시드 OOF, 중복 검사, 현재 풀 포함 전후 nested 기여와 바깥쪽 분할 일관성을 요구한다.

## 사실과 추론의 경계

각 대회 절의 순위, 작성자, 특성, 모형, 검증, 점수, 계산 자원, 댓글 보충과 실패 사례는 링크된 Kaggle 공식 해법 본문과 댓글에서 확인한 작성자 보고 사실이다.
댓글의 재현 수치와 작성자 설명도 해당 댓글 작성자의 보고이며 이 저장소에서 다시 실행한 값은 아니다.
S5E7 글을 42위로 제외한 판단은 상세 화면의 Solution Writeup · 42nd place 표시라는 직접 사실에 따른다.
S5E4 2위의 학습률 0.008과 0.08은 본문과 댓글이 충돌하므로 어느 값을 맞다고 고치지 않고 충돌 사실을 그대로 기록했다.

재현성, 현재 과제와의 유사성, 자체 35개 및 외부 278개에 대한 겹침, 이슈 307의 네 범주와 실험 우선순위는 조사자의 추론이다.
외부 구성원의 이름이 유사하다는 사실만으로 과거 글과 정확히 같은 구현이라고 단정하지 않았다.
과거 대회의 Public과 Private은 후보 발굴의 참고 근거일 뿐 현재 S6E8의 채택 근거가 될 수 없다.

## 한계

조사는 Kaggle 공식 화면이 현재 렌더링한 본문과 댓글을 대상으로 했으므로 삭제된 댓글의 과거 내용은 복구하지 않았다.
Kaggle 목록, 상세 머리말과 실제 펼친 댓글 노드의 숫자가 달라 원시 표시값을 따로 보존했다.
작성자가 전체 코드, fold 벡터, 구성원별 OOF 또는 제거 대조를 공개하지 않은 경우 본문만으로 재현 가능하다고 판정하지 않았다.
S5E7에는 조건을 만족하는 글이 없어 한 대회의 상위권 방법을 비교할 수 없었다.
순위가 없는 글과 10위 밖 글은 흥미로운 방법이 있어도 이번 표본의 반복 성공 근거로 사용하지 않았다.
