# S6E3부터 S5E12까지 상위 10위 해법과 댓글 조사

## 결론

S6E3, S6E2, S6E1, S5E12에서 공식 해법 글 47개를 전수 조사한 결과, 제목이나 본문이 private 최종 1위부터 10위라고 밝힌 글은 20개였다.
가장 일관되게 재사용할 수 있는 원칙은 많은 모델 자체가 아니라 동일한 fold의 누출 없는 OOF, 예측 오류가 다른 모델과 특성 관점, 그리고 선택과 가중치 학습까지 바깥 fold 안에서 다시 수행하는 결합 검증이다.
네 대회의 상위권 글에서는 단순 Ridge, 로지스틱 회귀 또는 선형 회귀가 hill climbing과 비선형 2단 모델을 자주 이겼지만, S6E2의 한 사례에서는 신경망 결합이 선형 결합을 이겼으므로 결합기 종류는 nested OOF로 결정해야 한다.
Chris Deotte의 S6E3 1위 해법은 4단계라는 깊이보다 850개 후보에서 오류 계열이 다른 약 150개를 고르고, 5겹 바깥 분할마다 5겹 안쪽 계산을 다시 수행하며, 마지막에는 정규화한 선형 결합기로 자유도를 낮춘 점이 핵심이다.
현재 S6E8 저장소는 이 원칙의 대부분을 이미 구현해 2026-08-20 현재 29개 후보 풀, fold가 정렬된 OOF, nested OOF 결합, TabM, RealMLP, 여러 트리 계열, 정확값 및 원본 프록시 계열을 보유한다.
따라서 Chris의 4단계를 그대로 복제하거나 850개 모델 규모로 늘리는 일은 우선순위가 낮고, 남은 실험은 현재 풀에서 모델 관점의 결손을 찾고 누출 없는 한계 기여를 재는 작은 대조로 제한하는 편이 타당하다.

## 조사 범위와 판정 규칙

조사 기준일은 2026-08-20이다.
각 대회는 사용자가 지정한 공식 `competitionWriteUps` 화면에서 시작했고, 대회 식별자만 바꿔 votes 정렬의 끝까지 확인했다.

- [S6E3 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s6e3/discussion?category=competitionWriteUps&sort=votes)은 고객 이탈 예측 대회이며 평가지표는 [ROC-AUC](https://www.kaggle.com/competitions/playground-series-s6e3/overview/evaluation)다.
- [S6E2 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s6e2/discussion?category=competitionWriteUps&sort=votes)은 심장병 예측 대회이며 평가지표는 [ROC-AUC](https://www.kaggle.com/competitions/playground-series-s6e2/overview/evaluation)다.
- [S6E1 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s6e1/discussion?category=competitionWriteUps&sort=votes)은 학생 시험 점수 예측 대회이며 평가지표는 [RMSE](https://www.kaggle.com/competitions/playground-series-s6e1/overview/evaluation)다.
- [S5E12 공식 해법 목록](https://www.kaggle.com/competitions/playground-series-s5e12/discussion?category=competitionWriteUps&sort=votes)은 당뇨 예측 대회이며 평가지표는 [ROC-AUC](https://www.kaggle.com/competitions/playground-series-s5e12/overview/evaluation)다.

순위는 해법 글의 제목 또는 본문에 적힌 최종 순위만 사용했고, 별도 순위표나 외부 자료로 빈 순위를 채우지 않았다.
제목에 순위가 없던 다섯 글은 본문을 열어 순위를 확인했으며, S6E3의 151위, 311위, 118위와 S6E2의 1144위 및 S5E12의 1365위로 확인되어 제외했다.
포함된 20개 글은 본문 끝까지 읽고, 댓글 영역의 지연 로딩을 끝까지 진행하고, 숨은 답글과 별도 appreciation 댓글을 모두 열어 접근 가능한 마지막 항목까지 확인했다.
댓글 제목에 표시된 수의 합은 244개였고, 실제로 접근해 읽은 일반 댓글과 답글은 269개이며, 별도 appreciation 댓글은 14개로 접근 가능한 댓글 항목은 모두 283개였다.
표시 수와 실제 접근 수의 차이는 접힌 답글, 별도 appreciation 영역, 삭제되었거나 DOM에 답글 항목으로 노출되지 않은 항목에서 생겼다.
Kaggle API, 웹 검색, 직접 HTTP 요청과 순위표 조회는 사용하지 않았다.
아래에서 `사실`은 글이나 댓글에 직접 적힌 내용이고, `추론`은 그 내용을 현재 S6E8 검증 계약과 후보 풀에 옮긴 판단이다.

## 모집단과 포함 현황

| 대회 | 공식 해법 글 | 포함된 1위부터 10위 글 | 제외 글 | 댓글 제목 합 | 접근한 일반 댓글과 답글 | appreciation 댓글 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S6E3 | 15개 | 4개로 1위, 3위, 5위, 9위였다 | 11개였다 | 98개였다 | 118개였다 | 4개였다 |
| S6E2 | 17개 | 6개로 1위, 2위, 3위, 4위, 8위, 10위였다 | 11개였다 | 57개였다 | 53개였다 | 3개였다 |
| S6E1 | 11개 | 7개로 1위부터 7위까지였다 | 4개였다 | 44개였다 | 50개였다 | 2개였다 |
| S5E12 | 4개 | 3개로 1위, 2위, 4위였다 | 1개였다 | 45개였다 | 48개였다 | 5개였다 |
| 합계 | 47개 | 20개였다 | 27개였다 | 244개였다 | 269개였다 | 14개였다 |

팀 이름이나 팀 구성원을 직접 밝힌 글은 없었으므로, 아래의 저자는 공식 글에 표시된 작성자이고 팀 정보는 모두 공개되지 않은 것으로 기록한다.
S6E3에서 공식 상위 10위 글이 없던 순위는 2위, 4위, 6위, 7위, 8위와 10위였다.
S6E2에서 공식 상위 10위 글이 없던 순위는 5위, 6위, 7위와 9위였다.
S6E1에서 공식 상위 10위 글이 없던 순위는 8위, 9위와 10위였다.
S5E12에서 공식 상위 10위 글이 없던 순위는 3위와 5위부터 10위까지였다.
빈 순위에는 다른 순위의 글을 채우지 않았고, 같은 팀원의 중복 글이나 삭제 또는 접근 불가 상태인 공식 해법 글은 발견하지 못했다.

## 제외 글 장부

### S6E3 제외 11개

- [Ravi Ramakrishnan의 Rank 38 approach](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/rank-38-approach)는 제목이 38위를 밝혀 제외했다.
- [Vladimir Demidov의 22nd Place Solution](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/22nd-place-solution)은 제목이 22위를 밝혀 제외했다.
- [hamzah의 16th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/16th-place-solution-ridge-ensembling)은 제목이 16위를 밝혀 제외했다.
- [Mizushima Toshihiko의 21st Place Solution](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/21st-place-solution-final-blend-selection-with-ri)은 제목이 21위를 밝혀 제외했다.
- [Evan Arlen Handy의 17th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/17th-place-solution)은 제목이 17위를 밝혀 제외했다.
- [Shiv Satyam의 18th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/18th-place-many-oofs-neural-networks-over-gbdts)은 제목이 18위를 밝혀 제외했다.
- [Baseer Shah의 34th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/34th-place-solution-ridge-multi-view-ensemble)은 제목이 34위를 밝혀 제외했다.
- [r0tor의 Stacked Boosting Ensemble](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/stacked-boosting-ensemble-with-ridge-blending)은 본문이 151위를 밝혀 제외했다.
- [yunsuxiaozi의 Trust CV can beat blind blending](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/trust-cv-can-beat-blind-blending)은 본문이 311위를 밝혀 제외했다.
- [Michael Y. Qiu의 Topological Depth vs. Pipeline Breadth](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/topological-depth-vs-pipeline-breadth-3694143)는 제목이 369위를 밝혀 제외했다.
- [Jeki Wan Taufik의 Adaptive Rank-Based Ensemble](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/simple-yet-effective-weighted-submission-blending)은 본문이 118위를 밝혀 제외했다.

### S6E2 제외 11개

- [Tilii의 22nd place](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/22nd-place-nns-again-better-than-gbms)는 제목이 22위를 밝혀 제외했다.
- [Oscar Aguilar의 43 Solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/43-solution-catboost-realmlp)은 제목이 43위를 밝혀 제외했다.
- [Rattan Singh의 35 Place](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/35-place-time-for-a-write-up)은 제목이 35위를 밝혀 제외했다.
- [yunsuxiaozi의 Magic noise](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/magic-noise)는 제목이 96위를 밝혀 제외했다.
- [Mizushima Toshihiko의 20th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/20th-place-solution-private-0-95533-ridge-stac)은 제목이 20위를 밝혀 제외했다.
- [Evan Arlen Handy의 25th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/25th-place-solution)은 제목이 25위를 밝혀 제외했다.
- [AshishSinghRawat의 44th Place](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/44th-place-simple-as-slime)은 제목이 44위를 밝혀 제외했다.
- [Baseer Shah의 16th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/16th-place-solution-trust-your-cv)은 제목이 16위를 밝혀 제외했다.
- [Emre Duman의 15th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/15th-place-solution)은 제목이 15위를 밝혀 제외했다.
- [eric15342335의 140 place](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/140-place-autogluon-feature-engineering)은 제목이 140위를 밝혀 제외했다.
- [Kimoly의 Less is More](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/less-is-more-lessons-from-the-complexity-trap)는 본문이 1144위를 밝혀 제외했다.

### S6E1 제외 4개

- [Ravi Ramakrishnan의 Rank14 approach](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/rank14-approach-grand-blend-of-diverse-models)는 제목이 14위를 밝혀 제외했다.
- [Jaswinder Singh의 13th place](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/13th-place-diversityslop)는 제목이 13위를 밝혀 제외했다.
- [1st Contest 15th Place](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/1st-contest-15th-place)는 제목이 15위를 밝혀 제외했다.
- [Дворкин Евгений Владимирович의 1263 place](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/1263-place)는 제목이 1263위를 밝혀 제외했다.

### S5E12 제외 1개

- [AxW의 Diabetes Prediction Challenge](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/diabetes-prediction-challenge-s5e12)는 본문이 1365위를 밝혀 제외했다.

## 계산량과 외부 의존성 장부

S6E3 1위는 IBM 원본 자료, 세 종류의 대형 언어 모델, 4장의 A100, 96개 CPU와 1TB 메모리에 의존했고, 3위는 자체 OOF와 공개 OOF를 함께 사용했다.
S6E3 5위는 pytabkit, YDF, H2O와 BARTZ를 포함한 많은 공개 구현에 의존했지만 전체 계산 시간과 장비는 밝히지 않았고, 9위는 GPU hill climbing을 밝혔지만 전체 계산량은 밝히지 않았다.
S6E2 1위는 원본 자료, AutoGluon, RealMLP, RGF와 TabICL에 의존하고 트리 및 RGF를 20개 seed로 재학습했으며, 댓글에서 TabICL의 전체 자료 적합은 메모리와 시간 한계로 포기했다고 밝혔다.
S6E2 2위는 원본 자료와 RealMLP을 사용하고 한 RealMLP에서 내부 평균 수 20을 썼으며, 나머지 전체 계산량은 밝히지 않았다.
S6E2 3위와 4위 및 10위는 원본 자료를 사용했고 3위와 10위는 GPU hill climbing을 사용했으며, 8위는 여러 공개 OOF와 OOF 없는 외부 결합에 의존했다.
S6E1 1위부터 7위까지는 31개부터 330개 사이의 OOF 후보를 만들었지만 장비와 전체 계산 시간은 밝히지 않았고, 5위는 유전 프로그래밍을 10회부터 20회 반복한 계산량만 보충했다.
S5E12 1위와 2위는 원본 자료를 사용했고, 2위는 50묶음에 각 2000회 hill climbing을 수행했으며, 4위는 공개 기본 결합과 GPU 학습에 의존했다.
외부 OOF나 공개 제출 파일을 사용하지 않았다고 명시하지 않은 글은 의존성이 없다고 추정하지 않고 미보고로 남겼다.

## S6E3 상위권 해법

### [1위 Chris Deotte](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)

- 저자와 범위에 관한 사실은 작성자가 Chris Deotte이고 팀 정보는 공개하지 않았으며, 한 달 동안 약 60만 줄의 코드, 모델 약 850개, 탐색 스크립트 약 50개를 만들고 GPT-5.4, Gemini 3.1, Claude Opus 4.6과 A100 80GB 네 장을 사용했다는 것이다.
- 1단계의 사실은 cuML 최근접 이웃, PyTorch 잡음 제거 자동부호기, PCA 군집, cuML 목표값 부호화 등으로 다른 행과 원본 자료에서 정보를 집약해 각 행의 표현을 늘렸다는 것이다.
- 2단계의 사실은 1단계 특성을 여러 GBDT와 신경망이 읽고 5겹 바깥 분할 안에서 5겹 안쪽 계산을 다시 수행해 OOF를 만들었다는 것이다.
- 3단계의 사실은 2단계 OOF를 다시 여러 GBDT와 신경망이 입력으로 읽었으며, 파일 이름에서 `_stk` 접미사가 이 단계 출력을 표시했다는 것이다.
- 4단계의 사실은 최종 cuML L2 로지스틱 회귀가 2단계와 3단계에서 고른 OOF 및 시험 예측 배열 154개를 입력으로 사용했다는 것이다.
- 최종 후보의 사실은 약 90개 트리와 60개 신경망으로 이루어졌고, 신경망은 RealMLP, TabM, TabICL, TabPFN, GraphSAGE, FT-Transformer, TabTransformer, DAE, FFM, DeepFM, ResNet, TabNet, Trompt, DANet 등 25개 구조 계열을 포함했다는 것이다.
- 모든 기본 모델은 StratifiedKFold 5겹과 seed 42를 사용했고, 목표값 부호화는 바깥 학습 부분 안에서 다시 5겹으로 계산했으며, OOF와 시험 예측은 `oof_<설명>_v<판본>.npy`와 `pred_<설명>_v<판본>.npy`로 맞춰 저장했다.
- 주요 특성의 사실은 MonthlyCharges와 TotalCharges를 원본 IBM 값에 붙인 값과 잔차, 소수 및 자릿수, 범주 조합과 수치 구간의 중첩 목표값 통계, `TotalCharges - tenure * MonthlyCharges`, 다양한 해상도의 구간, 빈도와 합성 대 원본 빈도비, 서비스 수, 원본 최근접 고객의 라벨, PCA와 무작위 투영, 숫자 문자열 TF-IDF, Benford 편차 등이다.
- 모델 선택의 사실은 약 850개 후보 중 대략 150개를 고를 때 탐욕적 전진 hill climbing을 사용했고, 최종 로지스틱 결합의 OOF AUC가 0.91985였다는 것이다.
- 계산 의존성의 사실은 대부분의 모델이 1시간 이내였지만 전체 탐색에는 4장의 A100, 96개 CPU, 1TB 메모리가 필요했고 일부 실행은 수 시간이 걸렸다는 것이다.
- 재현성 평가는 fold와 파일 규칙, 모델 계열, 특성 아이디어는 매우 상세하지만 850개 전체 설정, 최종 154개 파일 목록의 모든 생성 경로와 댓글에서 질문받은 3단계 fold별 시험 예측 평균 절차의 정확한 답이 없어 완전 재현에는 부족하다는 것이다.
- 댓글 범위는 제목에 82개가 표시됐고, 숨은 답글 두 묶음과 지연 로딩을 펼쳐 일반 댓글과 답글 95개 및 appreciation 댓글 4개를 읽었다.
- 저자는 댓글에서 hill climbing에 음수 가중치를 허용했고, Ridge와 로지스틱 회귀가 hill climbing보다 OOF에서 나았으며, 실제 제출한 로지스틱 회귀가 private에서도 가장 좋았다고 명확히 했다.
- 저자는 댓글에서 hill climbing은 일부 OOF를 골랐지만 Ridge와 로지스틱 회귀는 전체 OOF를 사용했고, 25개 심층학습 접근 중 탐욕 선택은 앞의 13개를 골랐으나 모든 트리 접근은 선택됐다고 밝혔다.
- 저자는 자신의 결과에 S6E3 3위와 7위 예측을 더해도 private 이득이 약 0.00001뿐이었다고 밝혀 상단 앙상블의 포화와 중복을 직접 보여 줬다.
- 저자는 모델 아이디어의 약 절반을 대형 언어 모델에 현재 앙상블에 없는 다른 모델을 제안하라고 물어 얻었고, 초매개변수는 대규모 자동 탐색보다 소수의 번호 붙인 변형을 수동 비교했다고 설명했다.
- 저자는 숫자 자릿수 추출은 실제 자료에서는 대개 해롭지만 합성 자료 생성 흔적을 잡을 때만 유용할 수 있다고 경고했다.
- 추론으로는 4단계의 성과를 층 수로만 해석하면 안 되고, 각 단계가 새 오류 계열을 만들며 모든 지도 특성과 결합 선택을 바깥 fold 안에 가둔 점이 핵심이다.
- 추론으로는 글이 로지스틱 회귀가 자료의 영역별로 다른 모델을 신뢰한다고 표현하지만 전역 선형 계수 자체는 영역별 가중치를 만들지 못하므로, 그런 동작은 앞 단계 특성과 3단계 모델이 영역 정보를 이미 부호화했을 때만 가능하다.

### [3위 Traiko Dinev](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/3rd-place-solution-an-ensemble-of-100-oofs)

- 사실로는 작성자가 Traiko Dinev이고 팀 정보는 공개하지 않았으며, 자체 및 공개 OOF 약 100개를 XGBoost 변형, LightAutoML, xLearn FFM, RealMLP, ResNet, DCN-V, DNET, TabTransformer, AutoGluon 등에서 모았다.
- 사실로는 hill climbing, 원시값 및 순위값 선형 회귀와 Ridge를 비교했고, 단순 LinearRegression이 public과 private에서 가장 좋았다.
- 사실로는 서로 다른 fold와 seed의 OOF가 섞였고 일부 특성 생성이 fold 안에서 수행되지 않았다고 저자가 인정했으며, 이 때문에 CV와 순위표 차이가 커졌을 가능성을 적었다.
- 댓글 범위는 제목에 8개가 표시됐고, 숨은 답글 3개를 펼쳐 일반 댓글과 답글 12개를 읽었다.
- 댓글에서 Chris Deotte도 Ridge와 로지스틱 회귀가 hill climbing보다 좋았다고 답했고, Traiko는 OOF를 하나씩 CV와 공개 점수로 확인해 수동 선별했다고 밝혔다.
- 댓글에서 두 사람의 OOF를 합친 실험은 private 0.91860에서 0.91862 정도의 작은 이득만 보였고, Chris는 완전한 3단계 OOF에 5×5×5로 125번 적합해야 누출을 막을 수 있다고 지적했다.
- 재현성은 모델 가족과 결합 비교는 공개했지만 fold 불일치, 외부 OOF의 생성 계보와 수동 선택 규칙이 고정되지 않아 낮다.
- 반증으로는 많은 OOF가 있어도 fold 불일치와 2단계 누출이 있으면 선형 결합의 OOF가 낙관적으로 보일 수 있다는 저자들의 합의가 남는다.

### [5위 kobby_](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/5th-place-solution-149-models-6-meta-models)

- 사실로는 작성자가 kobby_이고 팀 정보는 공개하지 않았으며, LightGBM, XGBoost, CatBoost, xLearn, TabNet, pytabkit 신경망과 트리, RealMLP, Keras, 로지스틱 회귀, YDF, H2O, HistGradientBoosting과 BARTZ를 사용했다.
- 사실로는 가장 좋은 단일 RealMLP의 OOF AUC가 0.91941이었고, BayesianRidge는 CV 0.91983 및 private 0.91846, Ridge는 CV 0.91986 및 private 0.91845, hill climbing은 CV 0.91989 및 private 0.91840이었다.
- 사실로는 최종 제출에서 BayesianRidge를 고르지 않아 private에서 0.00001을 잃었다고 저자가 적었다.
- 댓글 범위는 제목과 접근 가능한 일반 댓글이 모두 4개였고, 추가 기술 설명은 없었다.
- 재현성 경고는 제목이 149개 모델, 본문 표의 머리말이 199개 기본 모델, 표의 수량 합이 169개로 서로 맞지 않고 fold, seed, 선택 규칙도 충분히 공개되지 않았다는 것이다.
- 반증으로는 가장 높은 OOF의 hill climbing이 private에서는 가장 낮았고, 세 결합기의 private 차이는 0.00006 안쪽이라 OOF에서 결합기 하나를 미세 선택한 근거가 약하다.

### [9위 Mert Bayraktar](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/9th-place-solution)

- 사실로는 작성자가 Mert Bayraktar이고 팀 정보는 공개하지 않았으며, 두 특성 묶음에 CatBoost, XGBoost, LightGBM, HistGradientBoosting, RealMLP, TabM, TabTransformer, FT-Transformer, ResNet, TorchFrame, GNN, GateNet, DANet, DCNv2, BARTZ와 xLearn을 적용했다.
- 사실로는 모든 OOF를 순위값으로 바꾼 뒤 GPU hill climbing으로 결합했고 최종 CV AUC는 0.919911이었다.
- 사실로는 공개한 32개 모델 표에서 RealMLP가 0.919389로 가장 높고 DCNv2가 0.912211로 가장 낮았다.
- 댓글 범위는 제목에 4개가 표시됐고, 답글을 포함해 일반 댓글 7개를 읽었으나 실험 절차를 보충하는 설명은 없었다.
- 재현성은 StratifiedKFold를 썼다는 사실은 분명하지만 fold 파일, 모든 seed, 두 특성 묶음과 hill climbing의 제약을 완전하게 고정하지 않아 중간 수준이다.
- 추론으로는 단독 성능 폭이 큰 32개 모델을 결합한 사례가 약한 모델도 오류가 다르면 쓸 수 있음을 보이지만, 개별 제외 대조가 없어 어느 약한 모델이 실제로 기여했는지는 알 수 없다.

## S6E2 상위권 해법

### [1위 Masaya Kawamata](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)

- 사실로는 작성자가 Masaya Kawamata이고 팀 정보는 공개하지 않았으며, 약 150개 OOF를 만들고 Optuna로 2500개 부분집합을 비교한 뒤 반복해서 선택되는 약 10분의 1과 Ridge를 사용했다.
- 사실로는 모든 모델이 StratifiedShuffleSplit 5겹과 seed 42를 공유했고, 구간화와 반올림, 자릿수, 전 열 범주화, 빈도, 유전 프로그래밍, 원본 자료의 목표 평균과 평활값, WoE, 엔트로피 및 DVAE 특성을 사용했다.
- 사실로는 XGBoost, LightGBM, CatBoost, RealMLP, RGF, TabICL과 AutoGluon을 사용했고, 단독 점수가 낮은 RGF와 TabICL도 다양성 때문에 반복 선택됐다.
- 사실로는 최종 선택 제출이 CV 0.9557801, public 0.95396, private 0.95535였고, CV가 가장 높은 0.955865 제출의 private은 0.95534로 더 낮았다.
- 사실로는 완전 자료 재학습 때 트리와 RGF를 20개 seed로 평균하고 반복 수를 fold별 최적 반복의 평균보다 1.25배로 정했다.
- 사실로는 의사 라벨, 지식 증류, 깊은 GBDT, 고차 상호작용, 다른 자동부호기, 비선형 결합, 전 모델 단순 평균과 공개 점수 오르기가 효과가 없었다.
- 댓글 범위는 제목에 49개가 표시됐고, 접근 가능한 일반 댓글과 답글 46개 및 appreciation 댓글 2개를 읽었으며 나머지 한 항목은 접근 가능한 답글로 노출되지 않았다.
- Chris Deotte는 댓글에서 과적합으로 의심한 모델을 빼자 CV는 0.95581로 낮아졌지만 private은 0.95534로 좋아졌다고 독립적으로 보고했다.
- 저자는 댓글에서 Ridge는 분류기가 아니라 제곱오차 Ridge 회귀였고, 10회 선택 패턴에서 안정적으로 남는 핵심 OOF는 10개부터 13개이며 5개부터 7개는 바뀌었다고 설명했다.
- 저자는 TabICL v2를 GPU 메모리 때문에 10만 행 부분표본에 다섯 번 적합했으며 전체 자료나 40만 행은 느리거나 메모리 부족이었다고 밝혔다.
- 재현성은 fold, seed, 특성 계열, 선택 반복과 실패 목록이 상세해 높지만 150개 OOF 전체 목록과 2500개 선택의 정확한 탐색 공간은 완전 공개되지 않았다.
- 반증으로는 CV와 순위표 관계를 보고 멈춘 규칙이 사후 순위표 정보를 사용했으므로, S6E8에서는 채택 근거가 아니라 과탐색 위험을 보여 주는 회고 증거로만 사용할 수 있다.

### [2위 Akiyoshi Kinoshita](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/2nd-place-solution-avoid-leaks-and-overfitting)

- 사실로는 작성자가 Akiyoshi Kinoshita이고 팀 정보는 공개하지 않았으며, train과 test를 구분하는 적대적 검증 AUC가 0.5017 ± 0.0013이었다.
- 사실로는 목표값 통계를 fold 안에서 계산한 파이프라인과 전체 학습 자료에서 계산한 파이프라인을 분리해 결합하지 않았다.
- 사실로는 fold 안 파이프라인의 CV가 0.955759이고 public은 0.95394, private은 0.95535였으며, fold 밖 파이프라인은 CV 0.955774와 private 0.95534로 CV만 더 높았다.
- 사실로는 105개 모델을 50개로 줄이고 15개 다중 seed 평균본을 만든 뒤 상관 0.9999 문턱과 전진 및 후진 선택으로 네 개를 고르고 두 개의 순위 변환을 더해 신경망으로 결합했다.
- 사실로는 최종 원천 출력은 CatBoost 두 개와 RealMLP 네 개였고, 모든 비교는 StratifiedKFold 5겹이었다.
- 댓글 범위는 제목에 3개가 표시됐고, 일반 댓글 2개와 appreciation 댓글 1개를 모두 읽었으나 새 기술 설명은 없었다.
- 재현성은 단계별 개수와 누출 경계가 선명하지만 상관 및 선택의 세부 구현과 신경망 결합 설정이 부족해 중간 이상이다.
- 반증으로는 fold 밖 목표 통계가 CV를 높이고 private을 낮춘 직접 사례가 있어, 아주 작은 CV 상승을 누출 없는 개선으로 오인하지 말아야 한다.

### [3위 Mert Bayraktar](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/3rd-place-solution)

- 사실로는 작성자가 Mert Bayraktar이고 팀 정보는 공개하지 않았으며, 모든 열을 범주형으로 보는 묶음과 고유값 수가 10보다 작은 열만 범주형으로 보는 묶음을 사용했다.
- 사실로는 RealMLP가 CV 0.95576으로 가장 강했고 CatBoost ordered가 0.95575였으며, XGBoost, HistGradientBoosting, LightGBM과 로지스틱 회귀를 다양성 원천으로 썼다.
- 사실로는 원본 자료에서 열별 목표 평균을 만드는 것이 유일하게 효과가 있었던 특성 생성이라고 적었고, 모든 OOF를 순위값으로 바꿔 GPU hill climbing으로 결합해 CV 0.955803을 얻었다.
- 댓글 범위는 제목과 접근 가능한 일반 댓글이 모두 3개였고, 노트북 요청에 저자가 글을 갱신했다고 답한 것 외에 새 설명은 없었다.
- 재현성은 모델별 점수와 특성 관점은 있으나 fold, seed, 원본 통계의 fold 경계와 최종 가중치가 부족해 낮은 편이다.
- 추론으로는 RealMLP와 CatBoost가 거의 같은 강도여도 다른 학습 편향을 결합할 가치가 있다는 근거지만, hill climbing의 nested 검증이 없어 가중치 성능은 보수적으로 읽어야 한다.

### [4위 BlamerX](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution)

- 사실로는 작성자가 BlamerX이고 팀 정보는 공개하지 않았으며, one-hot 449차원 로지스틱 회귀가 CV 0.95550 및 public 0.95371을 내 선형 신호가 강하다고 판단했다.
- 사실로는 train과 test 적대적 검증 AUC가 0.501이었고, 트리에는 fold 안 목표값 및 빈도 통계를 주고 원본 행을 추가했지만 신경망에는 원본 집단 통계만 주고 원본 행 추가는 사용하지 않았다.
- 사실로는 깊이 2의 얕은 GBDT와 주기 임베딩을 넣은 RealMLP형 신경망을 만들고, OOF와 public 차이가 약 0.00185보다 나빠지는 모델을 버렸다.
- 사실로는 제약 없는 최적화가 CatBoost에 65%를 줘 public을 악화시켜 트리 비중을 35%로 제한했고, 교사 RealMLP 60%와 CatBoost 40% 결합은 CV 0.95580, public 0.95396, private 0.95535였다.
- 사실로는 교사가 확신한 약 4만 8천 개 시험 행을 단단한 라벨로 바꾸는 증류를 수행했으며, 네 개 결합을 순위 평균한 최종 private은 0.95534였다.
- 댓글 범위는 제목과 접근 가능한 일반 댓글이 모두 2개였고, 노트북 요청 외의 기술 보충은 없었다.
- 재현성은 구조와 의사결정은 설명하지만 public 차이를 선별에 사용하고 완전한 OOF 계보와 가중치 탐색 범위가 없어 낮다.
- 반증으로는 증류 학생과 최종 순위 결합이 교사 결합보다 private에서 나았다는 증거가 없고, 공개 점수에 맞춘 비중 제한도 S6E8 채택 절차로 옮길 수 없다.

### [8위 Arko Bera](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/8th-place-ensemble-and-trustcv)

- 사실로는 작성자가 Arko Bera이고 팀 정보는 공개하지 않았으며, 목표값 부호화 XGBoost 변형은 약 0.95522부터 0.95535였고, 원본 자료를 쓴 TabM은 0.95532였지만 기본 TabM은 약 0.9547이었다.
- 사실로는 공개 RealMLP, XGBoost, ResNet과 TabM 출력으로 만든 신경망 결합이 CV 0.95569, public 0.95394, private 0.95533이었고 선형 결합의 private은 0.95530이었다.
- 사실로는 마지막에 OOF가 없는 Mikhail의 공개 결합과 50 대 50으로 섞었으며, 이를 직관과 시간 부족에 따른 결정이라고 밝혔다.
- 댓글은 0개였다.
- 재현성은 외부 OOF의 계보와 마지막 결합의 검증이 없어 낮고, 외부 제출 파일 의존성이 크다.
- 반증으로는 이 사례의 신경망 결합 우위가 선형 결합 보편 우위를 반박하지만 차이는 작고, OOF 없는 마지막 혼합은 개선 근거가 아니다.

### [10위 Bala Baskar](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/10th-rank-solution-playground-series-s6e2)

- 사실로는 작성자가 Bala Baskar이고 팀 정보는 공개하지 않았으며, StratifiedKFold 10겹을 사용했다.
- 사실로는 수치 열의 범주 복제, 로지스틱 회귀 계수로 고른 이중 및 삼중 범주 조합, 원본 집단 통계, 주기 12, 14, 20의 사인과 코사인, 자릿수, 136개 열 쌍 빈도 통계와 영역 특성을 사용했다.
- 사실로는 XGBoost, CatBoost, XGBoost 잎을 입력으로 받는 로지스틱 회귀, RealMLP 세 개와 사용자 정의 임베딩 MLP 등 아홉 모델 유형을 사용했다.
- 사실로는 OOF 로짓 공간에서 음수 가중치를 허용한 GPU hill climbing을 수행했고, 최종 OOF가 0.95578 이상이며 public 0.95393과 private 0.95534를 보고했다.
- 댓글은 0개였다.
- 재현성은 10겹과 특성 수는 구체적이지만 모델별 설정, 조합 목록 전체, 탐색 순서와 최종 가중치가 없어 중간 이하이다.
- 추론으로는 자릿수와 주기 변환은 합성 값 눈금이 있을 때만 후보가 되며, 실제 S6E8에서는 이미 지문과 정확값 계열의 진단 결과를 우선해야 한다.

## S6E1 상위권 해법

### [1위 Mahog](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/1st-place-ive-ran-out-of-catchy-phrases-v)

- 사실로는 작성자가 Mahog이고 팀 정보는 공개하지 않았으며, 주기 변환, 산술식, 수치의 범주 복제, 자릿수, 범주 조합의 목표 평균과 표준편차 및 왜도 등 두 특성 묶음을 사용했다.
- 사실로는 RealMLP가 CV RMSE 8.58742, public 8.54280, private 8.58005로 가장 강한 단일 모델이었고, 약 190개 모델을 Ridge로 결합했다.
- 사실로는 최종 후처리 포함 결합이 CV 8.56634, public 8.53096, private 8.57273이었지만 isotonic을 빼면 private이 8.57152로 더 좋아 후처리가 private을 악화시켰다.
- 사실로는 초매개변수 탐색의 모든 OOF를 버리지 않고 후보로 남겨 같은 구조 안의 설정 다양성도 활용했다.
- 댓글 범위는 제목에 22개가 표시됐고, 숨은 답글 6개를 펼쳐 일반 댓글과 답글 29개 및 appreciation 댓글 1개를 읽었다.
- 댓글에서 Mirko의 특성 관점이 다른 OOF를 더하자 Mahog의 CV가 8.56959에서 8.56454로, public이 8.53099에서 8.52632로, private이 8.57152에서 8.56782로 좋아졌다.
- 댓글에서 Tilii는 두 사람의 474개 모델을 합쳐 private 8.56740을 얻어 추가 이득이 다시 작아졌다고 보고했다.
- 댓글에서 RealMLP의 무작위 탐색은 약 30회였고 pytabkit 작성자는 가장 좋은 설정만 남기지 말고 무작위 설정 전체를 결합 후보로 유지할 수 있다고 설명했다.
- 재현성은 특성 계열과 모델 점수가 있지만 190개 전체 구성과 Ridge 설정이 충분하지 않아 중간 수준이다.
- 반증으로는 isotonic 후처리가 OOF를 개선하면서 private을 악화했고, 선형 회귀용 수동 특성은 효과가 없었다는 저자 답변이 남는다.

### [2위 Tilii](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/2nd-place-nns-sometimes-work-better-than-gbms)

- 사실로는 작성자가 Tilii이고 팀 정보는 공개하지 않았으며, 75개 모델 중 60개가 TabM이고 신경망이 68개로 대부분을 차지했다.
- 사실로는 170개부터 700개까지 다른 특성 조합, 약 여섯 개 TabM 설정, 선형식 잔차와 유전 프로그래밍 특성 14개를 사용했다.
- 사실로는 가장 좋은 TabM의 CV가 8.590414이고 private이 8.59254였으며, XGBoost 단일 모델은 약 8.6051이었다.
- 댓글 범위는 제목에 11개가 표시됐고, 일반 댓글과 답글 10개 및 appreciation 댓글 1개를 모두 읽었다.
- 댓글에서 Keras FM의 잠재 차원 16 또는 32와 정규화 범위를 공개했으며, 분포 꼬리를 분류기로 나누는 방식은 일반화가 의심되어 사용하지 않았다고 밝혔다.
- 최종 결합 점수와 정확한 fold, seed, 75개 구성 및 가중치는 공개하지 않아 재현성은 낮다.
- 추론으로는 회귀에서도 TabM 다수 변형이 트리를 앞설 수 있음을 보이지만, 같은 가족 60개의 실제 한계 기여를 분리하지 않아 모델 수를 그대로 늘릴 근거는 아니다.

### [3위 Funguscakehead](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/3rd-place-135-oofs)

- 사실로는 작성자가 Funguscakehead이고 팀 정보는 공개하지 않았으며, XGBoost, LightGBM, CatBoost, TabM, RealMLP과 LNN에서 135개 OOF를 만들었다.
- 사실로는 XGBoost private RMSE가 8.59171, TabM이 8.58800, RealMLP이 8.58555였고 최종 Ridge는 CV 8.57299와 private 8.57775였다.
- 댓글은 0개였다.
- 재현성은 모델별 결과 외에 fold, seed, 특성, 135개 구성과 Ridge 설정이 거의 없어 낮다.
- 추론으로는 최종 Ridge의 private RMSE가 가장 좋은 단일 RealMLP보다 약 0.0078 낮아, 여러 강한 신경망과 트리의 선형 결합이 실제 이득을 만든 사례다.

### [4위 Yew Jin Lim](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/first-loser-aka-4th-place)

- 사실로는 작성자가 Yew Jin Lim이고 팀 정보는 공개하지 않았으며, 상호작용, 비율, 다항식, 5겹 목표값 부호화, 집단 통계와 산술식을 사용했다.
- 사실로는 TabM, XGBoost, LightGBM, CatBoost, MLP, ResNet, FT-Transformer, TabPFN과 최근접 이웃 모델 약 330개를 Ridge alpha 1로 결합해 CV 8.5659와 public 8.53642를 얻었다.
- 사실로는 음수 가중치가 중요했고, BayesianRidge는 RMSE가 약 0.002, NNLS는 약 0.041 나빴으며 CatBoost 2단 모델과 단순 평균도 더 나빴다.
- 사실로는 상관 0.99가 넘는 모델 추가가 CV는 개선하고 순위표는 악화시켰고, 다른 seed는 상관 0.998보다 높았으며 후처리는 해로워 예측을 19.6부터 100 범위로 자르기만 했다.
- 댓글 범위는 제목과 접근 가능한 일반 댓글이 모두 4개였고, 새 기술 설명은 없었다.
- 재현성은 모델 가족과 결합 비교는 좋지만 330개 구성과 선택 규칙 및 private 점수가 없어 중간 이하이다.
- 반증으로는 상관이 매우 높은 모델이 OOF만 높이고 외부 일반화를 악화시킨 사례가 있어, 후보 풀의 중복 문턱과 nested 선택이 필요하다.

### [5위 Mirko](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/5th-place-feature-driven-diversity-and-iterative-e)

- 사실로는 작성자가 Mirko이고 팀 정보는 공개하지 않았으며, 자동부호기, 유전 프로그래밍, PCA, 서로 다른 열 부분집합과 중요도가 정확히 0인 열만 제거하는 방식으로 특성 관점을 나눴다.
- 사실로는 자체 OOF를 Ridge로 결합했고, Mahog의 RealMLP OOF를 실험적으로 더했을 때 CV가 8.58053에서 8.57613으로 좋아졌으며 public 8.53309와 private 8.57353을 보고했다.
- 사실로는 글의 최종 제출 설명은 CV 규율을 지키기 위해 자체 OOF만 사용했다고 적어, 외부 OOF 통합 결과와 실제 제출 구성은 구분해야 한다.
- 댓글 범위는 제목과 접근 가능한 일반 댓글이 모두 4개였다.
- 댓글에서 유전 프로그래밍은 10회부터 20회 실행해 10개를 고르고 자동부호기 병목은 8 또는 32였으며, 최근 대회에서는 유전 프로그래밍이 자동부호기보다 다양성을 더 안정적으로 만들었다고 밝혔다.
- 재현성은 특성 관점과 일부 설정은 있으나 모델 목록, fold, seed와 최종 구성 설명의 모호함 때문에 중간 이하이다.
- 추론으로는 같은 원시 열을 다른 표현 공간으로 보내는 특성 관점 다양성이 모델 가족을 무작정 늘리는 것보다 재사용 가치가 높다.

### [6위 Traiko Dinev](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/6th-place-a-lot-of-features-a-lot-of-ensembling)

- 사실로는 작성자가 Traiko Dinev이고 팀 정보는 공개하지 않았으며, 약 200개 XGBoost 변형을 만들었고 가장 좋은 단일 모델은 CV 약 8.5950, public 8.56195, private 8.59951이었다.
- 사실로는 모든 OOF에 같은 분할과 seed를 사용했고, 로지스틱 회귀 예측 특성과 목표값 통계에는 세 겹의 중첩 검증을 적용했다.
- 사실로는 목표값 꼬리를 분류하는 gating은 약했지만 구간별 후처리가 단일 모델을 약 0.01 개선하고 다양성을 만들었다고 주장하면서도, 서로 다른 fold 결과가 섞여 CV가 지나치게 낙관적이라고 인정했다.
- 사실로는 글이 최종 결합을 231개 모델이라고 시작한 뒤 구성과 최종 점수를 제시하지 않고 끝난다.
- 댓글 범위는 제목과 접근 가능한 일반 댓글이 모두 3개였다.
- 댓글에서 Tilii가 gating은 과적합했고 임베딩 신경망의 특성 생성이 더 잘 됐다고 지적하자 저자도 과적합에 동의했다.
- 재현성은 동일 fold 규칙 일부는 좋지만 최종 231개 구성과 점수가 없고 저자가 누출 가능성을 인정해 낮다.
- 반증으로는 gating과 구간 후처리를 현재 S6E8에 옮기기 전에 선택 과정 전체를 nested OOF로 재현해야 하며, 그렇지 않으면 중단해야 한다.

### [7위 W-Bruno](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/private-7-diversity-ensemble-and-cv-trust)

- 사실로는 작성자가 W-Bruno이고 팀 정보는 공개하지 않았으며, 31개 1단계 모델을 Ridge로 결합했다.
- 사실로는 모든 기본 모델이 고정 seed 42의 5겹 OOF를 사용했고, 비슷한 특성에 서로 다른 특성 처리, `log1p`, 제곱근 변환과 잔차 학습을 적용해 상관을 낮췄다.
- 사실로는 후반에 public 하락을 지나치게 믿어 CV를 조금씩 개선하던 다양한 모델을 버리고, 비슷한 모델의 seed와 설정만 바꾸면서 CV가 거의 나아지지 않았다고 회고했다.
- 사실로는 최종 제출에서 5일 또는 6일 전의 최고 CV 모델과 최고 public 모델을 비교한 뒤 가장 강한 CV 제출을 선택해 private 7위가 됐다.
- 댓글은 0개였다.
- 재현성은 fold와 기본 구조는 분명하지만 31개 모델 목록, 변환 열, Ridge 설정과 점수가 없어 낮다.
- 추론으로는 공개 점수 하락 때문에 OOF 다양성 후보를 버리는 행동이 가장 직접적인 실패이며, S6E8의 public 점수 비채택 원칙을 지지한다.

## S5E12 상위권 해법

### [1위 wind1234it](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)

- 사실로는 작성자가 wind1234it이고 팀 정보는 공개하지 않았으며, 자료 뒤쪽을 검증으로 두고 다양한 GBDT와 신경망, 초매개변수, 특성 묶음, seed와 작은 결합을 만들었다.
- 사실로는 hill climbing이 CV 약 0.7088 및 public 0.70722에서 정체됐고, 상위 36개 순위 예측의 Ridge alpha 10은 CV 0.70860으로 낮았지만 public과 private이 더 좋았다.
- 사실로는 public과 CV 차이 약 0.00121을 중단 판단에 참고했고, 상위 34개 Ridge alpha 5는 private 0.70514가 가능했지만 public 0.70734가 낮아 최종 선택하지 않았다.
- 댓글 범위는 제목에 24개가 표시됐고, 숨은 답글 5개를 펼쳐 일반 댓글과 답글 24개 및 appreciation 댓글 5개를 읽었으며 삭제된 댓글 자리도 확인했다.
- 저자는 댓글에서 비율, 상위 10개부터 15개 수치 열의 다항식, 원본 자료는 학습 부분에만 추가, 모든 전처리와 목표값 통계는 fold 안에서 수행했다고 설명했다.
- 댓글 작성자는 상위 10개 다항식 특성이 private을 약 0.0005 개선했지만 비율은 악화시켰다고 재현 결과를 남겼다.
- Tilii가 RidgeCV를 제안하자 저자는 CV가 고른 alpha 0.1의 private이 0.70504이고 수동 alpha 6부터 8은 0.70514여서 CV가 private 최적 alpha를 고르지 못했다고 답했다.
- 재현성은 fold 경계와 특성 보충 설명은 좋지만 전체 후보 목록, Ridge 입력과 최종 가중치는 부족해 중간 수준이다.
- 반증으로는 결합 초매개변수의 OOF 미세 최적값이 private 최적값과 달랐으므로, alpha를 촘촘히 탐색해 OOF에서 고르는 행동 자체도 바깥 fold로 감싸야 한다.

### [2위 DaylightH](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an)

- 사실로는 작성자가 DaylightH이고 팀 정보는 공개하지 않았으며, 식별자 순서에 따른 train과 test 분포 이동을 찾아 train 마지막 약 3%인 `id >= 678260`을 검증으로 썼다.
- 사실로는 뒤쪽 약 2만 3천 행과 앞쪽 약 67만 7천 행을 구분하고, 뒤쪽과 닮은 정도로 앞쪽을 10구간으로 나누자 목표 평균이 약 0.64에서 0.521까지 낮아지는 개념 이동을 관찰했다.
- 사실로는 LightGBM, XGBoost, CatBoost, DAE 신경망, 결합 신경망, 뒤쪽 전용 로지스틱 회귀와 GAM을 사용하고 뒤쪽 16배, 원본 자료 8배, 앞쪽 1배의 표본 가중치를 줬다.
- 사실로는 뒤쪽 전용 목표값 통계, 의사 라벨, 잔차 학습, 구간화와 상호작용을 사용했고, 가장 좋은 단일 결합 신경망의 절단점 AUC는 0.70616이었다.
- 사실로는 양수 Ridge가 0.70751이고 50묶음에 각 2000회 탐색을 적용한 bagged hill climbing이 0.70771이었다.
- 댓글 범위는 제목에 21개가 표시됐고, 숨은 답글까지 일반 댓글과 답글 24개를 읽었다.
- 저자는 선택하지 않은 제출이 선택 제출보다 public이 0.00001 높았지만 CV가 구분할 수 없는 크기였다고 밝혔다.
- 저자는 뒤쪽 검증의 CV와 순위표 일관성이 XGBoost와 로지스틱 회귀에는 비교적 있었지만 다른 모델에는 없었고, 이 전략은 해당 대회에 특화돼 일반화하기 어렵다고 명확히 했다.
- 저자는 특성 생성은 모델이 보는 관계를 이해하고 식별자가 보이면 이동부터 조사하며, 대형 언어 모델 제안은 실험과 사람의 판단을 거쳐야 한다고 설명했다.
- 재현성은 분할 위치, 가중치와 모델 계열이 상세하지만 전체 특성 및 후보 목록과 결합 파일이 없어 중간 이상이다.
- 반증으로는 ID 뒤쪽 검증과 표본 가중은 실제 이동 진단이 있을 때만 유효하며, S6E8의 기존 적대적 및 ID 진단이 이동을 지지하지 않으면 적용하지 말아야 한다.

### [4위 Don Mani](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/add-tabm-for-diversity)

- 사실로는 작성자가 Don Mani이고 팀 정보는 공개하지 않았으며, 공개 기본 결합에 표준 TabM, 가중 표본추출을 흉내 낸 TabM과 가중 XGBoost를 추가했다.
- 사실로는 XGBoost 네 개와 TabM 두 개를 결합하고 XGBoost에 더 높은 비중을 줬으며, CatBoost는 XGBoost와 예측 누적분포가 비슷하다는 KS 및 CDF 진단 때문에 제외했다.
- 사실로는 모든 모델을 GPU로 학습했고, 공개 기본 결합의 OOF가 없어 최종 비중 일부는 직관으로 정했다.
- 사실로는 XGBoost에 열별 목표 평균과 빈도, `log1p(y)` 목표 변환을 썼다고 적었는데 이 대회 목표값은 이진이라 `log1p(y)`의 추가 의미가 불분명하다.
- 댓글은 0개였다.
- 재현성은 외부 기본 결합, OOF 부재, 모호한 목표 변환과 직관적 비중 때문에 낮다.
- 추론으로는 TabM이 단독 최강이 아니어도 트리와 다른 오류 분포를 주는 점은 재사용 가능하지만, KS나 CDF 유사성만으로 후보를 제거하지 말고 같은 fold의 OOF 상관과 nested 기여를 사용해야 한다.

## 대회 사이에서 반복된 패턴

### 1. 같은 fold와 누출 없는 OOF가 모델 종류보다 먼저다

사실로는 S6E3 1위가 목표값 통계와 3단계 OOF를 중첩했고, S6E3 3위는 fold 불일치가 결합을 막았다고 인정했으며, S6E2 2위는 fold 밖 목표 통계가 CV만 높이고 private을 낮춘 결과를 보였다.
사실로는 S6E1 6위도 서로 다른 fold가 섞인 gating의 CV가 지나치게 낙관적이라고 인정했다.
추론으로는 새 특성, 모델과 결합기보다 먼저 OOF 행 순서, fold ID, 전처리 적합 범위, 외부 OOF 계보와 2단계 선택의 바깥 fold 격리를 검사해야 한다.

### 2. 다양성은 모델 수가 아니라 오류 계열의 차이다

사실로는 S6E2 1위의 RGF와 TabICL, S5E12 4위의 TabM, S6E1 5위의 특성 관점처럼 단독 점수가 약한 모델도 결합에 남았다.
사실로는 S6E3 1위와 3위의 대형 OOF를 합쳐도 private 이득이 약 0.00001뿐이었고, S6E1 1위와 다른 상위권 OOF를 합친 이득도 구성원이 늘수록 빠르게 포화됐다.
추론으로는 새 후보를 평가할 때 단독 점수와 가장 가까운 OOF 순위 상관, 잔차 상관, 포함 및 제외 nested OOF를 함께 보며, 같은 가족의 seed와 작은 설정 변경만 늘리는 일은 피해야 한다.

### 3. 정규화한 선형 결합이 강하지만 보편 법칙은 아니다

사실로는 S6E3 1위, 3위, 5위, S6E2 1위, S6E1 1위, 3위, 4위, 5위, 7위와 S5E12 1위가 Ridge, 로지스틱 회귀 또는 선형 회귀를 최종 또는 유력 결합기로 사용했다.
사실로는 여러 글에서 hill climbing은 너무 적은 모델을 고르거나 OOF가 가장 높아도 private이 낮았고, 음수 가중치가 유용하다는 보고도 반복됐다.
반대 사실로는 S6E2 2위와 8위에서 신경망 결합이 쓰였고, S6E2 8위의 신경망 결합 private은 선형 결합보다 0.00003 높았다.
추론으로는 순위, 잘린 로짓과 원시 확률의 선형 결합을 기본으로 삼되 Ridge, 로지스틱 회귀, 제약 있는 hill climbing과 작은 신경망 결합을 동일한 nested OOF에서 비교해야 한다.

### 4. 원본 자료와 합성 지문은 강하지만 대회별이다

사실로는 네 대회 모두 원본 자료의 목표 통계, 최근접 원본 행, 값 눈금, 자릿수, 빈도, 제약 잔차 또는 분포 이동을 활용한 상위권 해법이 있었다.
사실로는 Chris Deotte가 숫자 자릿수는 실제 자료에서는 대개 해롭다고 경고했고, DaylightH는 ID 이동 전략이 해당 대회 전용이라고 밝혔다.
추론으로는 S6E8에서 값 눈금, 정확값, 원본 프록시와 화면 시간 예산 제약을 쓰는 것은 자체 진단이 있을 때만 정당하며, 과거 대회의 자릿수와 구간 수를 그대로 복제하면 안 된다.

### 5. 후처리, 의사 라벨과 공개 점수 적합에는 반증이 더 많다

사실로는 S6E1 1위의 isotonic이 CV를 높이고 private을 낮췄고, S6E2 1위의 의사 라벨과 증류는 효과가 없었으며, S6E2 4위의 증류 학생도 교사보다 좋아졌다는 증거가 없었다.
사실로는 S6E1 7위가 public 하락 때문에 유용한 CV 후보를 버린 일을 실패로 꼽았고, 여러 글의 OOF와 public 차이 선별은 private 최적과 어긋났다.
추론으로는 보정과 의사 라벨은 강한 사전 진단과 바깥 fold 모의 절차가 없으면 열지 않고, 공개 점수는 채택이 아니라 분포 방향의 건전성 확인에만 사용해야 한다.

## 현재 S6E8에 대한 적용 판정

### 이미 적용되어 추가 작업이 필요 없는 항목

현재 저장소의 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)은 공통 5겹 OOF, 3개 seed 평균본, 후보 풀 중복 문턱, 구성원 선택과 가중치 학습을 바깥 fold 안에서 수행하는 nested OOF를 이미 요구한다.
현재 [champion 장부](../../artifacts/champion.yaml)는 `exp127_lookup_muon`의 3개 seed 평균본 OOF AUC 0.9692840450을 기록하고, [후보 풀 장부](../../artifacts/pool.yaml)는 29개 구성원을 기록한다.
후보 풀에는 LightGBM, XGBoost, CatBoost, 정확값 one-hot 로지스틱 회귀, Lookup-Transformer, TabM, TabPFN, RealMLP, 표 합성곱망과 contextual spline이 이미 있어 과거 글의 주요 모델 가족을 단순 추가하는 일은 중복이다.
순위와 로짓의 이중 표현, 결측 구간 및 결측 상호작용 선형 결합도 이미 구현되고 nested OOF로 평가됐으므로 과거 글의 Ridge 또는 로지스틱 결합기를 새로 재현할 필요가 없다.
원본 프록시 최근접 라벨, 원본 목표 평균, 정확값 목표값 부호화, 빈도, 값 눈금과 화면 시간 제약 잔차 및 복원 특성도 이미 후보 풀이나 확정 실험에 존재한다.
따라서 과거 대회의 원본 자료, 자릿수, 빈도와 산술 잔차를 한 묶음으로 다시 넣는 실험은 원인 분리가 안 되고 기존 결과와 중복된다.

### 1순위 권고는 4단계 복제가 아니라 현재 결합의 깊이 제거 대조다

가설은 현재 29개 OOF의 선형 결합 위에 3단계 모델을 하나 더 학습하면 비선형 잔차를 얻을 수 있지만, 상위권 글 전체로는 깊은 결합보다 선형 결합의 근거가 강하다는 것이다.
최소 진단은 현재 최선 nested 선형 결합과 동일한 입력을 사용해 자유도가 매우 작은 3단계 후보 하나만 만들고, 각 바깥 fold의 학습 부분 안에서 다시 안쪽 fold OOF를 생성하는 것이다.
후보는 깊이 1 또는 2의 GBDT 하나와 작은 MLP 하나 중 하나만 사전 선택해야 하며, 둘을 모두 탐색해 더 좋은 것을 같은 OOF에서 고르면 선택 과정까지 한 겹 더 감싸야 한다.
채택 판정은 현재 ADR 0001의 앙상블 문턱과 바깥 fold 승리 조건을 그대로 사용한다.
중단 조건은 선형 결합 대비 nested OOF 개선이 `+0.00002` 미만이거나 바깥 fold 3개 미만에서 이기거나, 새 3단계 출력이 기존 결합과 스피어만 0.998 이상이면서 점수도 낮은 경우다.
이 검사는 Chris 해법의 깊이가 아니라 3단계가 실제로 새 정보를 만드는지 한 번만 확인하는 낮은 우선순위의 종결 실험이다.

### 2순위 권고는 모델 수 확대가 아니라 특성 관점의 중복 지도다

가설은 현재 풀의 서로 다른 모델 이름 중 일부가 같은 특성 관점을 공유하고, 반대로 같은 모델 가족 안에서도 원본 프록시, 결측 복원, 정확값과 제약 잔차 관점이 다른 후보가 결합에 기여한다는 것이다.
최소 진단은 29개 구성원을 모델 가족과 특성 관점의 두 축으로 분류하고, 각 후보의 최근접 OOF 상관과 nested 제외 변화를 같은 표에 놓는 것이다.
그 결과 비어 있는 관점이 있을 때만 기존 강한 Lookup-Transformer, RealMLP 또는 트리 하나에 그 관점만 이식한 짝비교를 연다.
중단 조건은 새 관점이 기존 후보와 스피어만 0.998 이상이거나, 진입 하한을 넘지 못하거나, 포함 및 제외 nested OOF에서 바깥 fold 3개 미만만 개선하는 경우다.
이 권고는 S6E1 5위와 S6E2 1위가 보여 준 특성 관점 다양성을 현재 풀에 맞춰 확인하는 것이며 새 모델 쇼핑 목록이 아니다.

### 3순위 권고는 원본 값 눈금 지문의 제한된 추가 대조다

가설은 Chris의 붙이기 잔차, 자릿수와 합성 대 원본 빈도비 중 S6E8에 아직 남은 신호가 있다면, 현재 원본 프록시와 정확값 계열의 제거 대조에서만 드러난다는 것이다.
현재 [후보 생성 과정 지문 결과](generator-fingerprint-results.md)는 생성기 제품 식별을 종결하고 정확값, 예산 복원과 잔차 표현이 이미 예측 가능한 지문을 감당한다고 결론 내렸다.
따라서 새 실험을 열려면 기존 특성에 없는 단일 표현이어야 하고, 예를 들면 각 수치의 원본 프록시 최근접 눈금까지의 부호 있는 거리처럼 계산 과정이 타깃을 보지 않는 열로 제한해야 한다.
최소 진단은 빠른 대리 모델에서 해당 열 하나 또는 사전 고정한 작은 묶음의 제거 대조를 수행하고, 통과할 때만 champion 모델의 seed 42 및 3개 seed 확정으로 넘기는 것이다.
중단 조건은 플라시보 중요도와 짝지은 OOF를 함께 통과하지 못하거나, 기존 원본 프록시 후보와 사실상 중복이거나, 효과가 특정 fold 하나에만 집중되는 경우다.
자릿수, Benford와 TF-IDF를 한꺼번에 대량 추가하는 실험은 Chris가 실제 자료에서 해로울 수 있다고 경고했고 S6E8 자체 지문 결과도 새 트랙을 지지하지 않으므로 열지 않는다.

### 보류하거나 기각할 항목

850개 모델과 4장의 A100을 전제로 한 Chris의 전체 규모는 3위와 7위 예측 추가 이득이 약 0.00001이었다는 포화 증거와 현재 29개 풀의 존재 때문에 기각한다.
fold가 다른 공개 OOF, OOF가 없는 제출 파일, 공개 점수로 고른 가중치와 순위표 상승을 멈춤 규칙으로 쓰는 방법은 현재 검증 계약과 충돌하므로 기각한다.
S5E12의 ID 뒤쪽 검증과 큰 표본 가중치는 S6E8에서 독립적인 ID 또는 적대적 이동 진단이 다시 양성으로 바뀔 때만 재검토하고 지금은 보류한다.
의사 라벨, 증류, isotonic과 gating은 이번 표본에서 반증 또는 과적합 인정이 반복됐고 현재 S6E8에도 더 강한 자체 검증 근거가 없으므로 보류한다.
대규모 hill climbing이나 미세한 Ridge alpha 탐색은 같은 OOF에서 선택 편향을 키우므로 결합 후보를 사전 고정하고 nested OOF가 선택하게 한다.

## Chris Deotte 해법을 S6E8 용어로 재구성한 도식

```text
원시 행과 원본 자료
  -> 1단계: 최근접 이웃, 잡음 제거 표현, 원본 눈금, 목표값 통계, 군집과 분포 지문
  -> 2단계: 동일한 바깥 fold 안에서 다양한 GBDT와 신경망의 OOF 생성
  -> 3단계: 2단계 OOF를 입력으로 받는 GBDT와 신경망을 다시 중첩 OOF로 생성
  -> 4단계: 2단계와 3단계 출력 154개를 L2 로지스틱 회귀로 결합
```

사실로 확인되는 누출 통제는 목표값 부호화와 2단계 및 3단계의 5×5 중첩 OOF다.
추론이 필요한 부분은 비지도 최근접 이웃, PCA, 군집과 자동부호기의 모든 변형이 각 바깥 fold 안에서만 적합됐는지 글이 개별적으로 명시하지 않는다는 점이다.
S6E8에서 이 구조를 시험한다면 1단계 제공자도 행 전체 분포를 읽는 순간 fold-fit 계약에 넣고, 2단계 및 3단계의 후보 선택과 결합기 설정까지 바깥 fold의 학습 부분 안에서만 정해야 한다.
현재 저장소는 이미 1단계 특성 제공자, 2단계 OOF 풀과 4단계에 해당하는 nested 선형 결합을 갖고 있으므로, 실제 미확인 부분은 3단계의 추가 한계 가치뿐이다.

## 최종 판단

가장 높은 신뢰도로 바로 유지할 방법은 동일 fold OOF, fold 안 목표값 통계, seed 평균본, OOF 파일 계보, 다양성 중복 검사와 nested OOF 결합이다.
현재 S6E8이 추가로 확인할 가치는 3단계 비선형 출력 하나의 엄격한 제거 대조와 모델 가족 및 특성 관점의 중복 지도에 제한된다.
원본 눈금 거리처럼 새롭고 타깃을 보지 않는 작은 지문 특성은 기존 지문 결과와 겹치지 않을 때만 낮은 순위로 검사할 수 있다.
나머지 대규모 모델 확대, 순위표 기반 선택, OOF 없는 혼합, 의사 라벨, 후처리와 ID 가중치는 이번 상위권 글과 댓글의 반증까지 고려하면 적용하지 않는 편이 낫다.
