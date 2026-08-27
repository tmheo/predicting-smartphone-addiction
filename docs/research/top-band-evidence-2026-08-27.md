# 상위권 공개 근거 조사 (2026-08-27)

이 문서는 [이슈 #453](https://github.com/tmheo/predicting-smartphone-addiction/issues/453)의 답이다.
2026-08-27 공개 리더보드에서 우리 위에 있는 0.9714~0.9719 밴드가 어떻게 도달됐는지에 대한 공개 근거, 2026-08-26 이후의 디스커션 증분, 최종 두 장 선택에 관한 정량 근거, 그리고 마감 전 4일 안에 재현 가능한 새 아이디어의 유무를 읽기 전용으로 조사했다.
확인 시점은 리더보드 CSV 기준 2026-08-27 06:56 UTC이고, 노트북과 데이터셋 목록은 같은 날 06:40~07:10 UTC에 Kaggle 공식 CLI로 조회했다.
Kaggle 제출, 유료 자원 생성, 후보 풀과 안전판 제출의 변경은 하지 않았다.
저장 실행 산출물 없이 저자의 서술만 있는 수치는 **저자 보고치**로 표시한다.
디스커션 증분은 [디스커션 증분 절차](../agents/discussion-update.md)의 커밋 단계를 수행하지 않고 보고만 한다.

## 결론

- 0.9714~0.9719 밴드 4팀 가운데 누구도 방법, OOF, 코드를 공개하지 않았다.
  Chris Deotte(0.97186)는 스타터 노트북 4개와 공지 스레드 2개뿐이고 S6E6 때처럼 OOF 데이터셋을 내지 않았으며, Changye Li(0.97154)와 MILANFX(0.97149)는 대회 관련 공개물이 0건이다.
- 유일한 실측 단서는 cstdy(0.97140)의 08-24 공개 노트북이다.
  공개 OOF 라이브러리 230개를 로지스틱 회귀로 결합해 저장 OOF 0.970175를 남겼고, 상위권 점수의 뼈대가 우리 확장 스택과 같은 공개 라이브러리 스택임을 보여 준다.
- dariushafshar의 "0.97184 Leader XGB"와 "0.97113 Residual NN Repro"는 제목의 점수를 재현하지 않는다.
  전자는 cdeotte 스타터의 복제(엔지니어링 특성 OOF 0.9648)이고 대조군이 목표값을 포함해 AUC 1.0을 찍는 결함이 있으며, 후자는 잔차 신경망 재현(분할 AUC 0.937~0.940)이다.
- 디스커션 증분(새 스레드 14개, 재방문 6개)에 0.9714 이상 방법을 설명하는 글은 없다.
  실측이 붙은 것은 잔차 기하 재현(44구성원 스택에 +0.000003), 결측 가리기 증강 재현(스택에 +0.00003), YKuma의 시간 예산 특성(+0.0008, 저자 보고치)뿐이며 모두 우리 풀이 이미 가진 축이다.
- 최종 선택 근거는 안전판 + 확장 스택 구성을 바꾸라고 하지 않는다.
  Kaggle은 선택한 두 장 가운데 private 점수가 높은 쪽을 최종 순위에 쓰므로 안전판 자리는 비용이 없고, 완료 7개 보드 중 3개가 공개 top 10을 전멸시킨 기저율과 판별 자의 불안정성은 "가장 다른 구성 한 장"을 유지할 이유다.
- 오늘 리더보드에서 우리는 0.97134 타이 블록(5팀) 안의 7위이고 위에 4팀뿐이다.
  공개 노트북들의 일치된 읽기는 분할이 옮기는 기대 순위가 수 위 단위이고 타이 블록 안 동전 던지기가 분산의 대부분이라는 것이며, 남은 위험은 분할보다 마감까지 추월당하는 쪽이 크다.
- 새 아이디어는 없다.
  실측 산출물이 있는 새 공개물은 Trompt OOF(0.96667), RealMLP 단일 OOF(0.96899), paiky1995 신경망 5분할 구성원 6개(0.9680~0.9687)이고, 자체 35개 풀에 더한 읽기 전용 nested 진단은 Trompt -0.0000097(분할 1/5), RealMLP 단일 +0.0000349(5/5), paiky 6개 +0.0000095(4/5)다.
  유일한 양의 신호도 우리 RealMLP 계열의 변형을 외부 구성원으로 하나 더하는 일이라, 질문이 제외한 "검증된 공개 구성원 추가"의 범주다.

## 상위권 4팀 공개 근거 표

리더보드는 2026-08-27 06:56 UTC CSV([리더보드](https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard)) 기준이다.

| 순위 | 팀 (프로필) | Public | 제출 수 | 최고 제출 시각 (UTC) | 대회 노트북 | 대회 데이터셋 | 디스커션 (08-24 이후) | 판정 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| 1 | Chris Deotte ([cdeotte](https://www.kaggle.com/cdeotte)) | 0.97186 | 64 | 08-27 02:25 | 스타터 4개 (08-20~21) | 없음 | 공지 2건, 방법 글 없음 | 공개 근거 없음 |
| 2 | Changye Li ([antoinegg1](https://www.kaggle.com/antoinegg1)) | 0.97154 | 39 | 08-24 22:48 | 없음 | 없음 | 게시 0건 | 공개 근거 없음 |
| 3 | MILANFX ([milanfx](https://www.kaggle.com/milanfx)) | 0.97149 | 37 | 08-27 06:09 | 없음 | 원본 자료 재게시만 | 게시 0건 | 공개 근거 없음 |
| 4 | cstdy ([kirill0212](https://www.kaggle.com/kirill0212)) | 0.97140 | 123 | 08-25 20:54 | [S6e8 Public Ensemble](https://www.kaggle.com/code/kirill0212/s6e8-public-ensemble) (08-24) | 없음 | 게시 0건 | 공개 라이브러리 230개 로지스틱 스택, 저장 OOF 0.970175 |

### Chris Deotte

- 대회 노트북은 [Basic EDA](https://www.kaggle.com/code/cdeotte/basic-eda-smartphone-addiction)(08-20), [Simple XGB Starter](https://www.kaggle.com/code/cdeotte/simple-xgb-starter)(08-20), [Simple CAT Starter](https://www.kaggle.com/code/cdeotte/simple-cat-starter)(08-21), [Simple NN Starter](https://www.kaggle.com/code/cdeotte/simple-nn-starter)(08-21) 네 개이고 모두 최근 실행이 08-21 이전이다.
- XGB 스타터의 저장 `fold_metrics.csv`는 분할 AUC 0.964247, 0.964754, 0.964714, 0.965790, 0.964567이다.
  스타터가 `oof_predictions.csv`를 남기지만 이는 CV 0.96대 기준선이지 0.97186의 재료가 아니다.
- 데이터셋 목록에 S6E8 항목이 없다.
  S6E6에서는 대회 중반인 2026-06-08에 [S6E6 OOF and Test PREDS](https://www.kaggle.com/datasets/cdeotte/s6e6-oof-and-test-preds)와 [GPU Logistic Regression Stacker](https://www.kaggle.com/code/cdeotte/gpu-logistic-regression-stacker)를 냈지만, S6E8에서는 08-27 07:00 UTC까지 같은 유형의 공개가 없다.
- 디스커션은 [736409](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736409)(XGB·EDA 스타터 공지, 20표, 코멘트 2)와 [736585](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736585)(CAT·NN 스타터 공지, 12표)뿐이고 본문은 노트북 링크와 "NN은 순열 중요도를 표시한다"는 한 줄이다.
  736585는 목록에 코멘트 2개로 표시되지만 페이지에는 casati8의 FastAI 홍보 코멘트 하나만 보인다.
- 프로필의 최근 게시글은 다른 대회(웰보어 지질, NeuroGolf, ARC)의 코딩 에이전트 실험 이야기이며 S6E8 방법 언급이 없다([프로필 디스커션](https://www.kaggle.com/cdeotte/discussion)).
- 따라서 0.97186이 자체 모델인지 공개 라이브러리 스택인지 판별할 공개 근거가 없다.

### Changye Li

- 프로필은 가입 2개월, 대회 13개, 코드 1개(ARC Prize 마운트 점검), 데이터셋 2개(RSNA Knee), 디스커션 0건이다([프로필](https://www.kaggle.com/antoinegg1)).
- 최고 제출은 08-24 22:48 UTC이고 그 뒤 제출이 없다.

### MILANFX

- 프로필(Xin Feng)은 대회 17개, 데이터셋 43개, 디스커션 0건이며 공개 코드가 없다([프로필 디스커션](https://www.kaggle.com/milanfx/discussion)).
- 데이터셋은 전부 "OriginalData" 재게시이고 S6E8분은 [S6E08OriginalData](https://www.kaggle.com/datasets/milanfx/s6e08originaldata)(08-01)다.
- 최고 제출은 리더보드 확인 47분 전인 08-27 06:09 UTC로, 아직 제출을 올리고 있다.

### cstdy

- [S6e8 Public Ensemble](https://www.kaggle.com/code/kirill0212/s6e8-public-ensemble)(최근 실행 08-24 05:47 UTC, 7표)은 szymonkapiski 74+50, adarsh1077 22, dariushafshar golem, boltuzamaki(deepfm_exact·ebm_exact 제외), mohankrishnathalla 3, beicicc 산출물, omidbaghchehsaraei·redamountassir·ravi20076·donmarch14·zhenruiweng·nawfeelrahman 노트북 산출물을 모아 230개 구성원을 만든다.
- 각 구성원을 logit으로 바꾼 뒤 `LogisticRegression(C=0.00599484)`를 5분할로 학습한다.
  저장 로그의 분할 AUC는 0.969571, 0.970281, 0.970234, 0.970847, 0.969970이고 합산 OOF는 0.970175다.
- 출력에 230개 구성원의 OOF·시험 배열을 logit 단위 npy로 다시 내보냈지만 기존 라이브러리의 복사본이라 새 구성원이 아니다.
- 저자는 다른 대회 코멘트에서 여러 노트북의 OOF·시험 예측을 모아 데이터셋에 올린 뒤 결합 노트북을 돌린다고 밝혔다([프로필 디스커션](https://www.kaggle.com/kirill0212/discussion)).
  0.970175에 우리와 상위권이 공통으로 관찰한 nested-Public 오프셋 약 +0.0011을 더하면 0.9713 부근이므로, 0.97140에는 공개 노트북 밖의 자체 구성원이나 제출 선택이 더 들어갔다고 추정한다(추정).

### dariushafshar의 두 노트북

- [0.97184 Leader XGB: Feature Ablation](https://www.kaggle.com/code/dariushafshar/0-97184-leader-xgb-feature-ablation)(08-26 13:58 UTC)은 본문에서 "0.97184는 상류 맥락이지 이 사본이 재현했다는 주장이 아니다"라고 명시한 cdeotte 스타터의 짝지은 절제 사본이다.
  저장 `fold_metrics.csv`의 엔지니어링 특성 AUC는 0.964239, 0.964788, 0.964737, 0.965890, 0.964735로 스타터와 같고, 원시 특성 대조군은 다섯 분할 모두 정확히 1.0이다.
  원인은 `raw_train_matrix = train.drop(columns=[ID_COL])`이 `addicted_label`을 남겨 대조군 학습 행렬에 목표값이 들어간 것이며, 따라서 "엔지니어링 - 원시" 차이 -0.0357은 무의미하다.
  선두의 방법에 대한 근거는 없다.
- [0.97113 Residual NN Repro | Golem Blend](https://www.kaggle.com/code/dariushafshar/0-97113-residual-nn-repro-golem-blend)(08-26 22:43 UTC)은 AnthonyTherrien의 잔차 신경망 구조(폭 128, 블록 2, 내부 시드 선별)를 재현해 golem 구성원 `a`와 0.02 가중으로 섞는다.
  저장 `result.json`의 분할 결합 AUC는 0.93784, 0.93782, 0.93980, 0.93965, 0.93893이고 시드 선별 관문은 분할 1이 음수라 실패했다.
  제목의 0.97113은 본문이 "Anthony의 맥락"이라고 밝힌 값이며 이 실행과 무관하다.
- AnthonyTherrien의 원본 [NN Residual Network](https://www.kaggle.com/code/anthonytherrien/predicting-smartphone-addict-nn-residual-network)(54표)은 90/10 홀드아웃으로 신경망을 학습한 뒤 [vault 데이터셋](https://www.kaggle.com/datasets/anthonytherrien/predicting-smartphone-addiction-vault)의 제출 CSV 두 장과 가중 2.9 : 0.1 : 0.0001(sub1 : sub2 : 신경망)로 평균한다.
  즉 0.97113은 사실상 vault의 제출 CSV 한 장이고 신경망은 점수에 기여하지 않는다.

### 상위 밴드 팀의 OOF·시험 예측 공개 여부

사용자가 라이선스 `unknown` 배열도 결합기 입력으로 쓸 수 있다고 정했으므로, 상위 4팀이 S6E8 OOF·시험 예측 배열을 공개하거나 예고했는지 따로 확인했다.

| 팀 | 공개 배열 | 구성원 수 | 라이선스 | 비고 |
| --- | --- | ---: | --- | --- |
| Chris Deotte | 데이터셋 없음. 스타터 3개의 노트북 출력 `oof_predictions.csv`(XGB, CAT, NN 각 1개) | 3 | 노트북 소스 Apache 2.0, 출력 자료 라이선스 표시 없음 | CV 0.96대 기준선이라 결합기 입력 가치가 없고, 08-27 07:00 UTC까지 OOF 데이터셋 예고 글도 없다 |
| Changye Li | 없음 | 0 | - | S6E8 공개물 0건 |
| MILANFX | 없음 | 0 | - | 원본 자료 재게시만 |
| cstdy | [S6e8 Public Ensemble 출력](https://www.kaggle.com/code/kirill0212/s6e8-public-ensemble)의 `oof_*.npy`·`test_*.npy` 230쌍(logit 단위)과 스택 `submission.csv` | 230 | 노트북 소스 Apache 2.0, 배열은 상류 라이브러리의 라이선스를 그대로 따름(대부분 CC0, beicicc 일부 other) | 전부 기존 공개 라이브러리의 복사본이라 #442 장부 밖의 새 구성원은 없고, 스택 OOF 자체는 저장하지 않는다 |

즉 상위 밴드 팀이 새로 공개한 OOF 배열은 없다.
08-24 이후 새로 공개된 배열은 상위 밴드 밖의 paiky1995(CC0, 11개), kodaifukuda0311(노트북 출력 1개), yekenot(노트북 출력 1개)이며 다음 절에서 다룬다.

## 디스커션 증분

기준 장부는 [discussion-insights.md 부록](discussion-insights.md)(마지막 표적 반영 2026-08-21)이고, 2026-08-26의 [마감 직전 공개 후보 재점검](s6e8-last-minute-public-candidate-scan-2026-08-26.md)이 737422와 737590을 장부 밖에서 검토했다.
목록은 "Recent Comments" 정렬 첫 페이지(최근 8일 활동)를 Jina Reader의 링크 요약으로 읽었고 코멘트 수는 목록 표시값이다.

### 새 스레드 (장부에 없음)

| id | 제목 (작성자) | 코멘트 | 읽은 내용 |
| --- | --- | ---: | --- |
| [735689](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735689) | How to think? (Marwan_Mostafa) | 4 | 초보 질문. 한 답글이 "검증 없이 앙상블만으로 0.97을 찍었고 private에서 떨어질 것"이라고 자평 |
| [736409](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736409) | Simple XGB and EDA Starter - CV 0.96 (Chris Deotte) | 2 | 스타터 공지, 방법 언급 없음 |
| [736513](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736513) | Residual Geometry Boosting (Ern711) | 4 | 재현자가 fold 0을 1,750라운드까지 돌린 결과 ADD 0.967274 > LOGP2 0.966845 > MIX 0.966295 > POWER15 0.965650. 초반 +0.006 우위는 700라운드 안에 사라짐 |
| [736585](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736585) | Simple CAT and NN Starter - CV 0.96 and 0.94 (Chris Deotte) | 2 (페이지 1) | 스타터 공지. casati8의 FastAI 노트북 LB 0.96703 홍보 |
| [736595](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736595) | Seeking for help, xgboost at a limit, 0.96844 (fryhat) | 6 | najiama의 특성 조언 뒤 저자가 0.96997 도달(저자 보고치) |
| [737015](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737015) | Residual Geometry Spline Transformer (Ern711) | 2 | 44구성원 스택 보유자의 5분할 재현. 분할 평균 BASE 0.96761 > POWER150 0.96740, 합산 OOF는 POWER150 0.967013 > BASE 0.966282. 여섯 변형 전부 더해 +0.000003, IDENTITY와 목표 학습 스플라인의 상관 0.88 |
| [737023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737023) | what's your best CV? (hamzah) | 5 | 저자 0.969694(30모델), Will(134위) 0.9703251, Jaideep 0.96967(3모델), najiama 순수 LightGBM 0.96348. 전부 저자 보고치 |
| [737068](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737068) | I am not sure what to do with missing data (SamarthxUmrao) | 7 | najiama가 "결측 개수 특성은 잡음"으로 정정하고 단일 LightGBM LB 0.96990·CV 0.96862 노트북 홍보(저자 보고치) |
| [737108](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737108) | Just a Silly Question (Pruthviii1) | 0 | 본문 미렌더링, 초보 질문 |
| [737231](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737231) | highh log loss (Pruthviii1) | 9 | 초보가 확률 대신 하드 라벨로 log loss를 계산한 문제 |
| [737369](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737369) | Predicting Smartphone Addiction with LightGBM 0.96357 (VENGATESH A) | 0 | 노트북 홍보 |
| [737422](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737422) | max_bin is worth +0.0024 or +0.0005 here (kito_pl) | 0 | 08-26 재점검이 반영, 변화 없음 |
| [737590](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737590) | Exploring Alternative Ensemble Blends (Ern711) | 5 | 08-26 재점검 뒤 코멘트 5개 추가. broccoli beef: 정규화 기하 평균은 logit 산술 평균과 동치. Tilii: 거듭제곱 결합은 CV 소폭·LB 무이득이었고 신경망 결합기 뒤 CatBoost 초기화가 hill climbing·Ridge보다 나았다(저자 보고치) |
| [737682](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737682) | Is it possible to make over 0.97 with one model? (hypecoef) | 3 | Tilii(180위): 단일 모델 2개가 0.97 초과, 5개가 0.9699x, 열쇠는 특성 공학과 RealMLP(저자 보고치) |

### 재방문 스레드 (코멘트 수 증가)

| id | 제목 | 코멘트 (장부 → 지금) | 읽은 내용 |
| --- | --- | --- | --- |
| [733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) | As a Beginner, What's the First Thing You Check | 9 → 19 | 초보 토론, 읽지 않음 |
| [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005) | Changing the random seed moves you 60 places | 9 → 11 | Georgy Mamarin의 시즌 6 완료 7개 보드 표: 공개 top 10 유지 수 8, 0, 6, 5, 8, 0, 0, top 300 스피어만 S6E2 -0.51·S6E3 0.63·S6E5 0.35, private top 10 자리 70개 중 31개가 공개 100위 밖 팀. 제출 수와 순위 변동의 상관은 -0.08~+0.19로 약함 |
| [734628](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734628) | How are you designing Private-LB-robust validation | 1 → 2 | Georgy: 74구성원 라이브러리에서 상위 10 균등 0.96931, 상위 10 hill climb 0.96942, 전체 균등 0.96738, 전체 hill climb 0.96948, nested 로지스틱 0.96963. 분해능 바닥은 두 후보의 유사도에 따라 0.000005~0.00043 |
| [735861](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735861) | What actually creates a decorrelated model when you work alone? | 2 → 4 | YKuma: 시간 예산 제약 특성이 네 계열 모두 약 +0.0008, 재구축한 lookup 신경망 0.96607 → 0.96719, 결합 OOF 0.96821 → 0.96905, LB 0.96922 → 0.97012, 소수 격자 특성은 +0.00001~+0.00004(저자 보고치) |
| [736062](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736062) | Achieving 0.971+ LB: Residual NN + LightGBM Stacking | 0 → 4 | 코멘트는 "왜 반대표냐" 세 개와 삭제 하나. 글 자체는 -12표, 여전히 정량 근거 없음 |
| [736522](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736522) | Neural nets vs boosting: a missing-data blind spot? | 0 → 1 | tomasa2 5분할 재현: 스플라인·주의 신경망 0.96771 → 0.96779, 분할별 +0.00009~+0.00040 전부 양수, 스택 기여 +0.00003, LB 0.97066 → 0.97071. 값 토큰·결측 표시·원시값·목표 부호화·빈도 다섯 표현을 실제 결측 행에서 복사해야 효과가 남 |

장부 재방문 목록의 나머지(735421, 734063, 733983, 733708, 732434, 732428)는 첫 페이지에 없어 최근 8일 활동이 없다.

### 증분에서 읽히는 것

- (a) 0.9714 이상 방법: 없다.
  0.9713 군집의 실체는 08-26 이후에도 공개 라이브러리 스택과 공개 제출 CSV 혼합이다.
  r0tor(19위, 0.97130)의 [Rank-Gauss + logit-rank blending](https://www.kaggle.com/code/nikita7364777/rank-gauss-logit-rank-blending)은 cstdy 노트북을 포크해 najiama·szymonkapiski·hboyang 라이브러리를 rank-gauss 로지스틱으로 결합한 뒤 anthony·amanatar 제출 CSV를 25%씩 logit 순위 공간에서 섞는다.
  abhirajhiwale의 [Mapping the Public Plateau 0.97128](https://www.kaggle.com/code/abhirajhiwale/s6e8-mapping-the-public-plateau-0-97128)(08-26)은 atakan·hboyang·azzam·dari·souvik·anthony·najiama 제출 CSV의 가족 평균이며, 저장 출력은 atakan 앵커와의 스피어만(hboyang nested 0.99977, dari 0.99974, naji13 0.99823, souvik 0.999, anthony 0.99691, azzam 0.99669)뿐이다.
  같은 노트북의 "앵커와 스피어만 0.9998 이상이면 점수가 같고, 벗어나면 단위 거리당 약 1e-5를 잃으며, 의사 라벨·원본 증강·신경망·집단 집계·자릿수 특성은 각각 OOF +0.00002 이하"는 저자 보고치다.
- (b) 새 공개 OOF 라이브러리와 구성원: paiky1995의 [S6E8 OOF Library: 11 Neural Members](https://www.kaggle.com/datasets/paiky1995/s6e8-oof-library-11-members)(08-25 15:54 UTC, CC0)가 #442 장부 밖의 유일한 새 라이브러리다.
  설명문 기준 5분할·seed 42·원본 행 순서이며 5분할 구성원 6개(v14_lookup_bag 0.968727, v13_lookup 0.968293, v17_realmlp 0.968282, v16_lookup_aug 0.968154, v15_lookup_wide 0.968147, v10_tabm 0.968006)와 10분할 구성원 5개로 이뤄진다.
  10분할 5개는 우리 규칙상 제외 대상이다.
  그 밖에 [kodaifukuda0311 RealMLP 단일 노트북](https://www.kaggle.com/code/kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only)(08-27 03:09 UTC)이 `oof_realmlp.npy`(저장 로그 OOF 0.96899291, 분할 0.96833~0.96959, Public 0.97016은 저자 보고치)를, [yekenot Trompt 노트북](https://www.kaggle.com/code/yekenot/ps-s6-e8-trompt-pytorch-frame)(08-25)이 `oof_preds.csv`(저장 로그 OOF 0.96667)를 남겼다.
  [atakanaldemir V13 Diversity Anchor](https://www.kaggle.com/datasets/atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124)(08-24)는 결합 산출물의 OOF라 2단계 산출물 제외 규칙에 걸리고, [anthonytherrien vault](https://www.kaggle.com/datasets/anthonytherrien/predicting-smartphone-addiction-vault)(08-25)는 제출 CSV 두 장뿐이다.
  hboyang의 [catstrall 구성원](https://www.kaggle.com/datasets/hboyang/s6e8-catstrall-member)(08-24)은 이미 장부에 있다.
- (c) 최종 선택과 private 분할: 734005의 Georgy 표, 734628의 분해능 바닥, 736062의 반대표 소동이 새 정보이고 정량 근거는 다음 절에서 다룬다.

## 최종 선택 위험 근거

### 공개 노트북 네 편의 정량 주장

[georgymamarin/s6e8-will-your-0-971-survive-the-private-split](https://www.kaggle.com/code/georgymamarin/s6e8-will-your-0-971-survive-the-private-split)(최근 실행 08-25 19:48 UTC, 48표)은 저장 실행 로그와 코드 셀이 있고, 12절의 라이브 보드 판독값만 손으로 입력한 값이다.

- 시즌 6 완료 7개 보드의 공개 top 10 유지 수는 [0, 0, 0, 5, 6, 8, 8]로 1~4 사이가 한 번도 없다.
  ROC AUC 보드만 보면 S6E2 0, S6E3 6, S6E5 8로 3개 중 1개가 전멸했다.
- 전멸 3개 보드는 top 10이 추격자(공개 101~300위)에 대해 자기 산포의 1.95~3.34배 가라앉았고 유지 4개는 -0.24~+1.69다.
  이 자는 private 반쪽이 있어야 계산되므로 설명이지 예보가 아니다.
- 사전 판별 자 두 개(top 1% 대 top 10% 폭 비율, top 10 대 top 300 폭 비율)는 완료 보드 7개를 가르지만 라이브 보드에서는 열이틀 사이에 세 번 답이 바뀌었다.
  08-21 판독은 폭 비율 0.462(2,455팀 절단 시 전멸 0.327~0.643, 유지 0.709~0.971 → 전멸 범위), 팀 수 비율 0.283(라이브 깊이에서 전멸 0.127~0.242, 유지 0.292~0.689 → 두 집단 사이)로 서로 다른 답을 냈다.
  저자 결론은 "행동에 옮길 자가 없다"이며, 최종 보드에서만 유효한 기준은 top 10/top 300이 전멸 0.137~0.400 대 유지 0.550~0.745다.
- 노이즈: 공개 59,260행의 단일 AUC 표준오차 0.00061, private 237,042행 0.00031, 한 제출의 보드 간 이동 sd 약 0.0007.
  두 제출의 격차 sd는 sd(이동) × sqrt(2(1 - rho))이고, 같은 라이브러리의 근친 단일 모델 둘은 0.000005, 포함 관계의 블렌드 둘은 0.000022, 구성원을 공유하지 않는 블렌드 둘은 0.000106, 최고 단일 대 20위 단일은 0.000428을 구분한다.
  08-21 공개 top 10의 인접 격차 9개(0.00000~0.00009, 4개는 정확히 0)는 가장 안정된 짝으로도 5개, 나머지 짝으로는 0개만 구분된다.
- 분할 검증 풀(CV 0.969816 → LB 0.97096)과 boltuzamaki 포함 풀(CV 0.970064 → LB 0.97109)의 두 제출 실측은 CV가 약속한 0.00025 중 0.00013만 보드가 지불했음을 보였고, 자기 오프셋은 +0.00109~+0.00114로 안정적이었다.
- 12b의 사전 예측: 공개 top 10 유지 수는 2 이하 아니면 5 이상, 방향은 무콜, 필드가 약 3,000팀이면 private top 10에 공개 100위 밖 팀이 최소 1팀.

[dariushafshar/p-you-keep-top-5-run-it-on-your-own-rank](https://www.kaggle.com/code/dariushafshar/p-you-keep-top-5-run-it-on-your-own-rank)(08-25 22:03 UTC, 3표)는 코드 셀만 있고 저장 출력이 없어 수치는 저자 보고치다.

- 자기 25구성원 블렌드 6개를 private 크기(237,042행)로 재표본해 잰 짝 격차의 sd는 최소 1.51e-05, 중앙값 2.09e-05, 최대 3.07e-05다.
- 132위·상위 5% 선까지 13위 여유에서 "경계 팀이 나를 추월할 확률"은 31.6%지만 "상위 5% 밖으로 떨어질 확률"은 0.24%이며, 기대 순위 이동은 +4.1(sd 3.33)이다.
- 같은 표시 점수를 공유하는 15팀 타이 블록이 전체 분산의 32%를 차지하고, 보드 전체의 생존 곡선은 타이 블록 단위의 계단이라 "안전"에서 "탈락"까지 약 10위 안에서 바뀐다.
  sigma를 1.51e-05에서 3.07e-05로 두 배 키워도 자기 확률은 0.9997에서 0.9857로만 움직인다.
- 한계로 짝 격차의 독립 가정, 5자리 반올림(격차당 최대 5e-06), 보드 고정 가정을 스스로 적었다.

[dariushafshar/the-split-costs-4-ranks-the-week-costs-127](https://www.kaggle.com/code/dariushafshar/the-split-costs-4-ranks-the-week-costs-127)(08-25 22:03 UTC, 3표)도 저장 출력이 없어 저자 보고치다.

- 08-24 06:06 UTC(2,730팀)와 08-25 19:58 UTC(2,911팀) 두 스냅샷에서 0.97121 위의 팀 수가 62에서 127로 늘어 시간당 1.72팀이 추월했고 뒤로 밀린 팀은 0이다.
- 추월 66팀 중 52팀은 기존 팀의 개선이고 중앙값 개선 폭은 +0.00011이며, 신규 진입은 시간당 4.78팀이다.
- 분할이 옮기는 기대 순위는 약 4위이고 반속으로 잡아도 남은 한 주의 추월은 127위다.
- 자기가 잰 최선의 미발사 후보는 공개 프런티어 혼합 +0.000049(짝지은 5분할, 전 분할 양수)로 추월 팀 중앙값의 44%라서 쏘지 않았다.

[abhirajhiwale/s6e8-mapping-the-public-plateau-0-97128](https://www.kaggle.com/code/abhirajhiwale/s6e8-mapping-the-public-plateau-0-97128)(08-26 05:12 UTC)의 저장 출력은 앞 절의 스피어만 표와 `submission.csv`뿐이고, 응답 곡면 주장은 저자 보고치다.
"모든 사람이 같은 고원을 공유하며 탈상관은 벌을 받는다"는 저자 해석은 public 점수 기준이므로 private에 대한 진술이 아니다.

### 오늘 리더보드에서의 우리 위치

리더보드 CSV(2026-08-27 06:56 UTC, 3,049팀)에서 계산했다.

- 0.97134 타이 블록은 5팀(5~9위)이고 우리는 그 안의 7위다.
  위에는 4팀이 있고 1위와 5위 블록의 격차는 0.00052, 1위와 2위는 0.00032다.
- 0.97130 이상 19팀, 0.97128 이상 109팀(0.97128에 72팀, 0.97129에 18팀), 0.97120 이상 191팀, 0.97100 이상 369팀이다.
- 상위 5% 선은 152위 0.97124, 상위 10% 선은 304위 0.97113이다.
- 안전판 제출(0.97099)은 오늘 보드에서 약 370위(상위 12.1%), CV 전용판(0.97096)은 약 406위에 해당한다.
- 상위 10팀의 제출 수는 64, 39, 37, 123, 185, 100, 23, 96, 147, 16이고 상위 20팀 중앙값은 56, 상위 100팀 중앙값은 19.5다.
  우리 23회는 상위 20 안에서 두 번째로 적다.
  상위 20팀 중 9팀이 08-27 당일에 제출했다.
- Georgy의 자를 오늘 보드에 적용하면 top 1%/top 10% 폭 비율 0.781, top 10/top 300 폭 비율 0.740으로 둘 다 유지 범위(0.459~0.953, 0.550~0.745)다.
  그러나 1위 한 팀을 빼면 0.610과 0.537로 팀 수 비율은 두 집단 사이로 내려가므로, 오늘 판독은 Chris Deotte의 0.97186이 top 10 폭을 늘린 결과이지 보드 전체의 성질이 아니다.

### 저장소 기존 결론과의 대조

- [carry-over 사전 추정](carryover-preclose-estimate.md)은 상위 5%(당시 110팀)가 0.00022 폭 안에 몰려 단일 점수 노이즈의 0.4 시그마라 상단 내부 public 순위가 신호가 아니라고 했고, 보정 뒤에는 25% 컷에서 유지 체제 쪽(0.94~0.98)으로 좁혔다.
  오늘 자 계산은 그 구조가 유지되고(0.97128~0.97134에 105팀) 1위만 튀어나왔음을 보여 준다.
- [tomasa2·georgymamarin 갱신 재검토](gaming-hours-whatmoved-notebook-recheck.md)는 CV → LB 오프셋 +0.00109~+0.00114, 두 제출 누수 가격 측정, "8월 31일 남은 지렛대는 탈상관뿐"을 기록했고 마감 직전 packing 판독을 권고했다.
  이번 조사가 그 판독을 계산했으며 결과는 위와 같이 1위 이상치에 좌우되는 값이다.
- 지도 [#441](https://github.com/tmheo/predicting-smartphone-addiction/issues/441)의 "nested + 약 0.0011 = Public" 관계는 우리 확장 스택(0.9702876 → 0.97134, +0.00105)에서 다시 성립했다.

### 안전판 자리에 대한 찬반

- Kaggle은 팀이 고른 최대 두 장을 각각 private로 채점하고 그중 높은 점수로 최종 순위를 정한다(플랫폼 규칙, [대회 개요](https://www.kaggle.com/competitions/playground-series-s6e8/overview)와 [Kaggle 대회 문서](https://www.kaggle.com/docs/competitions)에는 선택 장수 외의 세부 표기가 없다).
  따라서 안전판을 두는 비용은 "확장 스택의 다른 변형 한 장"을 포기하는 것뿐인데, 같은 라이브러리 변형끼리는 격차 sd가 0.00002 수준이라 두 번째 변형의 기대 이득이 잡음 바닥 안이다.
- 안전판이 확장 스택보다 private에서 높으려면 확장 스택 쪽이 0.00035 이상 더 떨어져야 한다.
  공개 근거 가운데 그런 방향을 지지하는 것은 Georgy의 기저율(7개 중 3개 전멸, top 10이 추격자 대비 집단으로 침몰)과 두 제출 누수 가격(혼합 분할 풀의 CV 이득 절반만 실현)뿐이고, 반대 근거는 우리 확장 스택이 public이 아니라 nested OOF로 골라졌고 외부 구성원 209개가 분할·행 순서·재채점을 통과했다는 점이다.
- 안전판이 방어하는 위험은 "공개 라이브러리 공통 요인의 private 붕괴"이며, 그 위험이 실현되면 공개 0.9712x 군집 전체가 함께 내려간다.
  이때 안전판(자체 35개 풀, public 370위권)이 상대적으로 얼마나 오르는지는 공개 근거로 정할 수 없다.
- 결론: 두 장을 안전판 + 확장 스택으로 유지한다.
  추월 위험(시간당 1.72팀, 상위 5% 선 0.97124)은 우리 0.97134와 무관하게 이미 3,000팀 규모에서 상위 0.3%이고, 남은 4일에 public 순위를 지키려는 제출은 근거가 없다.

## 새 아이디어 판정

기준은 저장 산출물이나 코드가 있고, 우리 35개 풀이나 242구성원 확장 스택에 새 정보를 더하며, 2026-08-29 23:59 UTC 전에 자체 5분할 재현과 판정을 끝낼 수 있는가다.

| 항목 | 근거 종류 | 판정 |
| --- | --- | --- |
| 잔차 기하 부스팅·스플라인 변환기(736513, 737015) | 재현자 저장 없음, 코멘트 수치 | 기각. 44구성원 스택에 +0.000003, ADD 기준선이 수렴 뒤 우세 |
| 결측 가리기 증강의 다섯 표현 복사(736522) | tomasa2 코멘트 수치 | 기각. 스택 기여 +0.00003, champion은 이미 value_dropout 0.10 |
| 다중 연산 결합(737590) | S6E8 실측 없음 | 기각. Tilii도 LB 무이득 보고 |
| 신경망 결합기 뒤 CatBoost 초기화(737590 Tilii) | 저자 보고치 | 기각. 코드·수치 없음, 우리 결합기 비교(#337)는 학습형 로지스틱이 최선 |
| 시간 예산 제약 특성(735861 YKuma) | 저자 보고치 +0.0008 | 기각. 우리 풀의 재구성·잔차 특성이 같은 제약을 이미 사용([hidden-constraint-diagnosis](hidden-constraint-diagnosis.md)) |
| 단일 RealMLP 0.97 초과(737682 Tilii) | 저자 보고치 | 기각. 산출물 없음 |
| Trompt OOF(yekenot, 08-25) | 저장 OOF 0.96667, 코드, 5분할 seed 42 일치 | 아래 진단 |
| RealMLP 단일 OOF(kodaifukuda0311, 08-27) | 저장 OOF 0.96899, 코드, 5분할 seed 42 일치 | 아래 진단 |
| paiky1995 신경망 5분할 6개(08-25) | 저장 OOF 0.9680~0.9687, CC0, 코드 비공개 | 아래 진단 |

### 읽기 전용 진단: 자체 35개 풀에 더했을 때

진단은 `scripts/judge_extended_stack.py`의 own35 행렬 적재와 `pipeline.ensemble.evaluate_nested`를 그대로 써서 scratch 경로에서만 실행했고 후보 풀, 장부, MLflow, 제출은 건드리지 않았다.
후보 OOF는 우리 `artifacts/folds.parquet` 분할로 다시 채점해 저자 로그의 분할 AUC와 일치함을 확인했다.

후보의 단독 성능과 우리 풀과의 거리는 다음과 같다.

| 후보 | OOF AUC | 우리 분할 재채점 | 가장 가까운 풀 구성원 (스피어만) | 중앙값 스피어만 |
| --- | ---: | --- | --- | ---: |
| Trompt (yekenot) | 0.966671 | 저자 로그의 분할 0.96717, 0.96725, 0.96622와 일치 | exp070_cat_exact_cats (0.98998) | 0.98477 |
| RealMLP 단일 (kodaifukuda0311) | 0.968993 | 저자 로그의 분할 0.96833~0.96959와 일치 | exp140_realmlp_orig_cdf_diff (0.98406) | 0.97677 |
| paiky1995 v14_lookup_bag | 0.968726 | 선언값과 1e-06 차이 | exp059_lookup_transformer (0.97324) | 0.96776 |
| paiky1995 v13_lookup | 0.968293 | 선언값과 일치 | exp059_lookup_transformer (0.96870) | 0.96376 |
| paiky1995 v17_realmlp | 0.968282 | 선언값과 일치 | exp117_ag25_gbm_r21 (0.98167) | 0.97383 |
| paiky1995 v16_lookup_aug | 0.968154 | 선언값과 일치 | exp059_lookup_transformer (0.96544) | 0.96012 |
| paiky1995 v15_lookup_wide | 0.968147 | 선언값과 일치 | exp059_lookup_transformer (0.96700) | 0.96226 |
| paiky1995 v10_tabm | 0.968006 | 선언값과 일치 | exp117_ag25_gbm_r21 (0.99261) | 0.98502 |

nested OOF 증분은 다음과 같다.
기준 own35의 `shrunk_rank_logit_logistic` 0.9698106은 [#443 사다리](extended-stack-ladder.md)의 값과 같다.

| 구성 | 구성원 수 | shrunk nested | own35 대비 | 분할 양수 | rank_logit nested | own35 대비 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| own35 (기준) | 35 | 0.9698106 | - | - | 0.9697948 | - |
| own35 + Trompt | 36 | 0.9698009 | -0.0000097 | 1/5 | 0.9697853 | -0.0000095 |
| own35 + RealMLP 단일 | 36 | 0.9698455 | +0.0000349 | 5/5 | 0.9698355 | +0.0000406 |
| own35 + Trompt + RealMLP 단일 | 37 | 0.9698489 | +0.0000383 | 5/5 | 0.9698384 | +0.0000435 |
| own35 + paiky1995 5분할 6개 | 41 | 0.9698200 | +0.0000095 | 4/5 | 0.9698065 | +0.0000117 |

읽히는 것은 다음과 같다.

- Trompt는 단독 0.96667로 [#145](https://github.com/tmheo/predicting-smartphone-addiction/issues/145)에서 우리가 중단한 판(fold 0 0.9401)보다 훨씬 강하지만, 풀에 더하면 nested가 내려가고 분할 5개 중 1개만 양수라 기각이다.
  PyTorch Frame 구현(채널 64, prompt 16, 층 4, 8 epoch)이 TALENT 판보다 이 자료에 맞는다는 정보는 남지만, 정보 관점이 기존 정확값 CatBoost와 겹친다.
- RealMLP 단일은 +0.0000349로 [08-26 재점검](s6e8-last-minute-public-candidate-scan-2026-08-26.md)이 참고선으로 쓴 +0.00002를 넘고 분할 5개 전부 양수다.
  그러나 이는 우리 RealMLP 계열(exp140과 0.984)의 변형 하나를 외부 구성원으로 더한 결과이고, 242구성원 확장 스택 대비 증분은 재지 않았다.
  확장 스택은 같은 저자의 이전 정확값 목표 부호화 구성원(`hboyang6:koda_exact_te`, 0.968404)과 RealMLP 계열 여러 개를 이미 담고 있어 남는 몫은 더 작을 것이다.
  자체 풀 진입은 외부 구성원 규칙상 불가하고, 자체 재현은 조기 종료에 바깥쪽 검증 분할을 쓰는 학습 상태 선택과 원본 7,500행 자료의 경험 누적분포 특성을 우리 계약대로 다시 만들어야 하므로 GPU 없이 08-29까지 끝낼 근거가 없다.
- paiky1995 6개는 lookup 변형들이 우리 풀과 스피어만 0.965~0.973으로 멀리 있음에도 +0.0000095(4/5)로 참고선에 못 미친다.
  낮은 상관이 곧 기여가 아니라는 [#386](external94-width-diagnostic.md)의 관찰과 같다.
- 따라서 질적으로 새로운 아이디어는 없다.
  유일한 양의 신호도 "검증된 공개 구성원 하나 추가"이며, 실행한다면 [#443](https://github.com/tmheo/predicting-smartphone-addiction/issues/443) 사다리 도구로 own35 + ext207 + RealMLP 단일(+ paiky 6개)을 nested로 다시 재는 일이고 244구성원 shrunk 한 번에 약 44분이 든다.
  이 경우에도 안전판 제출은 바뀌지 않는다.

## 한계

- Kaggle 디스커션 목록은 Jina Reader가 첫 페이지 20개만 렌더링했고 코멘트 수는 목록 표시값이라 페이지 본문과 다를 수 있다(736585는 목록 2·본문 1, 736062는 목록 4·본문 3 + 삭제 1).
- 상위 3팀의 방법은 공개물이 없어 판정 불가이며, cstdy의 0.97140이 공개 노트북 판본으로 얻어졌는지도 확인할 수 없다.
- dariushafshar의 두 노트북과 abhirajhiwale 노트북의 응답 곡면 주장, paiky1995 라이브러리의 분할 설명은 저자 서술이며 코드가 공개되지 않은 부분은 재현하지 않았다.
- Kaggle API가 조사 중 429 제한을 걸어 일부 데이터셋 메타데이터는 웹 페이지로 대체 확인했다.
- 읽기 전용 진단은 자체 35개 풀 기준이며 242구성원 확장 스택에 더했을 때의 증분은 재지 않았다.
  `shrunk_rank_logit_logistic`의 λ 선택은 재실행 사이 약 7e-06 흔들릴 수 있다(#386).
  진단 산출물은 scratch 경로에만 있고 저장소에는 이 문서의 표만 남긴다.
- RealMLP 단일 노트북은 목표 부호화를 바깥쪽 학습 분할 안에서만 맞추지만, 조기 종료에 바깥쪽 검증 분할의 목표값을 쓰고 [jayjoshi37 원본 자료](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction)를 특성 참조로 쓴다.
  외부 구성원으로 쓸 때는 이 두 가지가 주의 사항 부류에 들어가야 한다.
