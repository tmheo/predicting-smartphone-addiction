# S6E8 공개 증분에서 지도 172 범위 밖의 도약 후보 (이슈 #191)

## 결론

2026-08-18 KST에 2026-08-14 전수 재점검과 지도 172 리서치(#173, #174) 이후의 S6E8 공개 노트북, 디스커션, 데이터셋 증분을 전수 확인했다.
증분의 절대량은 얇고, 대부분은 외부 제출 파일 재활용이거나 이미 열린 티켓의 근거 보강이다.
기대 이득 +0.0002급 또는 기존 풀과 상관 0.98 미만의 새 정보 축 요건을 충족할 가능성이 있는 완전히 새로운 후보는 retrieval 계열 TabR 하나뿐이며, 그것도 S6E8 직접 실증이 없는 조건부 후보다.
나머지 증분은 열린 이슈 #63, #64와 마일스톤 이슈 #188에 흡수할 근거들이고, 새 티켓을 정당화하지 않는다.

## 우선순위 표

| 순위 | 후보 | 기대 이득 | 근거 품질 | 10일 내 이식 | 누수 위험 | 처리 제안 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | TabR(retrieval 증강 tabular NN) 스모크 게이트 진입 | 불명, 상관 0.98 미만 가능성 중간 | 낮음(S6E8 실증 없음, 구조 정합 추론) | 가능하나 빠듯함, GPU 필수 | 중간, retrieval 후보 집합의 fold 격리 필요 | 조건부 새 티켓(#102 TabPFN 선례의 스모크 게이트 방식) |
| 2 | 재분할 평균 OOF의 스택 기여 과대 검사 | 성능이 아니라 판정 신뢰도 | 높음(공개 노트북이 홀드아웃 대조 코드 공개) | 즉시 | 없음 | 이슈 #63에 흡수 |
| 3 | 스태커 증거 보강(로짓 스택, 음수 계수, 전량 nested 스택 우위) | #64 판정의 방향 근거 | 중간(공개 노트북 실행 수치, 우리 fold 아님) | 즉시 | 낮음 | 이슈 #64에 흡수 |
| 4 | 최종 제출 전략 증거(지난 S6 7개 보드 전례, 정정된 분해능, blind-blend 파동) | 순위 방어 | 높음(종료 보드 원자료 데이터셋 공개) | 즉시 | 없음 | 이슈 #188 판단 자료로 반영 |
| 5 | 삼각함수(주기) 특성, multi-level spline 변형, NN residual, 빈도 인코딩·비율 재부상 | 문턱 미달 또는 반증 | 아래 상세 | - | - | 기각 근거만 기록 |

## 조사 범위와 방법

- Kaggle 공식 CLI로 2026-08-18 09시 KST 기준 S6E8 코드 탭의 최신 실행 100개와 득표 상위 60개를 열거하고, 2026-08-14 이후 실행된 노트북 전부를 제목·저자 수준에서 걸렀다.
- 신규성 가능성이 있는 노트북 15개를 `kaggle kernels pull`로 내려받아 코드 셀과 마크다운 셀을 직접 읽었다.
- `s6e8` 검색으로 최근 갱신 데이터셋 60개를 열거하고, 2026-08-14 이후 갱신분의 파일 목록과 메타데이터를 확인했다.
- 디스커션 목록을 최신순과 인기순으로 확인해 신규 스레드 2건(735421, 735404)과 코멘트가 늘어난 기존 스레드 5건(734005, 733495, 732358, 734990, 732223)을 전부 읽었다.
- 기존 조사와의 중복은 `docs/research/remaining-experiment-space-audit.md`, `docs/research/public-stack-provenance.md`, `docs/research/discarded-candidate-reaudit.md`, `docs/research/spline-comment-reassessment.md`, `docs/research/s6e8-167-spline-notebooks.md`와 이슈 #173, #174 코멘트로 대조했다.
- 저장 출력이나 제거 실험이 없는 주장은 아래에서 근거 등급을 명시해 낮춰 표기했다.

지도 190의 제외 규칙에 따라 과거 탈락 재검토, 공개 스택 구성원 재현, 쌍 TE, CE 재판정, CDF·KDE·orig_knn·FM 묶음, 외부 예측 직접 편입은 후보에서 제외했다.

## 1순위 조건부 후보: TabR retrieval 계열

### 무엇이 새로운가

TabR는 각 행의 예측에 학습 자료에서 검색한 최근접 이웃 표현을 함께 쓰는 retrieval 증강 tabular NN이다(yandex-research, arXiv 2307.14338).
현재 후보 풀(트리 3계열, Lookup-Transformer, TabM 등)과 공개 스택 94~178구성원 계보(#174) 어디에도 retrieval 계열이 없고, 2026-08-14 전수 재점검의 모델 계열 판정에도 등장하지 않는다.
즉 이 대회에서 한 번도 검토된 적 없는 마지막 주요 tabular 계열이다.

### 근거

- [tomasa2의 2026-08-16 갱신판](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 신규 4개 아키텍처 실험에서 TabR만 "GPU 최근접 이웃 의존성 설치 실패로 아예 실행하지 못했다, 미검증이지 음성이 아니다"라고 명시한다.
- 같은 노트북의 짝비교에서 낮은 상관의 신규 계열 중 단독 OOF 0.966을 넘은 spline transformer만 스택 기여 양수(+0.00004)였고, 단독 0.963 이하인 ResNet, MLP-PLR, FM은 상관이 0.85 수준으로 낮아도 기여가 0 또는 부호 불안정이었다.
- 이 자료의 목표값은 정확값을 키로 쓰는 조회 표 구조이고(같은 노트북 8.1절: 인접 정수값 사이 중독률 차이 평균 0.22, 표본 노이즈의 22배), 우리 최고 모델도 정확값 임베딩 기반 Lookup-Transformer다.
- retrieval 메커니즘은 같은 정확값 조합의 이웃 행을 직접 참조하므로, 이 구조와 정합적이라는 추론이 성립한다.
- 반대 근거로, 우리 풀의 Lookup 계열과 정확값 TE가 이미 값별 신호를 직접 표현하므로 TabR의 신규 정보가 그 위에 남아 있을지는 실측 전에 알 수 없다.

근거 등급: 낮음.
S6E8에서 TabR를 실제로 학습한 공개 결과는 확인되지 않았고, 위 판단은 구조 정합 추론과 인접 실험의 간접 증거다.

### 기대 이득, 이식 가능성, 누수 위험

- 기대 이득: 단독 OOF가 0.966대 후반을 넘지 못하면 tomasa2의 단독 성능 하한 관측에 따라 기여가 0으로 수렴할 위험이 크다.
  성공 시나리오에서는 풀 최근접 상관 0.98 미만의 새 축이 될 수 있다.
- 이식 가능성: 공식 구현(yandex-research/tabular-dl-tabr, MIT)과 faiss GPU가 필요하고 Vast.ai 실행 규약과 호환된다.
  691,369행에서 후보 검색이 비싸므로 축소 표본 스모크가 선행돼야 하며, 10일 창에서 빠듯하다.
- 누수 위험: 중간.
  retrieval 후보 집합이 outer fold의 평가 부분을 포함하면 그 자체가 누수이므로, fold별로 검색 집합을 outer 학습 부분으로 제한하는 구현 규율이 필수다.

### 처리 제안

TabPFN-3 티켓(#102)과 같은 스모크 게이트 방식의 조건부 티켓 한 장으로 연다.
게이트는 사전 고정한다: 축소 표본 스모크에서 학습이 정상 수렴하고, fold 0 단독 AUC가 풀 하위 구성원 수준(0.966대) 이상이며, 풀 최근접 스피어만 상관이 0.98 미만일 때만 전체 5-fold로 확장한다.
게이트 미달 시 즉시 닫고 매몰 비용을 늘리지 않는다.

## 2순위 흡수: 재분할 평균 OOF의 스택 기여 과대 검사 (이슈 #63)

- [The OOF That Passes Every Leakage Test](https://www.kaggle.com/code/maximolorenzoylosada/the-oof-that-passes-every-leakage-test) (2026-08-15)는 같은 모델, 같은 자료에서 OOF를 세 가지 방식으로 만들었을 때, 여러 재분할에 걸쳐 평균한 OOF가 행 단위 누수 검사를 전부 통과하면서도 홀드아웃 실측 대비 스택 기여를 과대 보고함을 보였다(보고 +0.000189 대 실측 +0.000088).
- 원인은 누수가 아니라 학습·서빙 불일치이며, 정확값을 암기하는 구성원(TE 키, 최근접 이웃 계열)에서만 문제가 되고 트리 구성원은 거의 영향이 없다고 명시한다.
- 우리 풀의 시드 평균은 같은 고정 fold 안의 모델 시드 평균이므로 직접 해당되지 않지만, 이슈 #63의 진입 검사에 "구성원 OOF가 단일 고정 분할에서 나왔는지, 재분할 평균인지"를 계보 확인 항목으로 추가할 가치가 있다.

근거 등급: 높음.
실행 코드와 이중 홀드아웃 재현이 노트북에 포함되어 있다.
기대 이득은 점수가 아니라 후보 판정의 신뢰도이고, 누수 위험은 없다.

## 3순위 흡수: 스태커 방향 증거 보강 (이슈 #64)

2026-08-14 이후 실행된 두 공개 노트북이 이슈 #64(순위 평균 대 nested 선형 스태킹)의 설계 선택에 같은 방향의 근거를 더했다.

- [tomasa2 갱신판](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t): 로짓 스택이 확률 스택보다 다양한 12구성원 라이브러리에서 +0.00047, 전부 GBM인 라이브러리에서 +0.00008이었다.
  약한 구성원에 음수 계수를 줄 수 있는 선형 스태커가 hill climbing보다 낫고, hill climbing이 0 가중을 준 구성원도 스태커에서는 보정 항으로 기여했다.
- [Georgy Mamarin의 gaming_hours 노트북](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new) (2026-08-17, 46표) 10절: 74구성원 전량 균등 평균은 상위 10개 균등 평균보다 나쁘고, 전량 nested 로지스틱 스택은 상위 10개 스택과 균등 10개 모두를 이겼다.
  결합을 잘못 하는 비용(전량 균등 평균)이 가중치 미세 조정의 이득보다 10배 이상 컸다.
- 같은 절은 가중치 최적화의 "public 과적합 프리미엄"이 라이브러리 규모와 무관하게 측정 불가능한 수준임을 재표본 대조로 보였다.

근거 등급: 중간.
실행 수치는 공개 라이브러리와 그들의 fold에서 나온 것이라 우리 고정 fold의 채택 근거는 아니고, #64의 자체 nested 비교가 정식 판정이다.
새 티켓은 필요 없고 #64 실행 시 참고 자료로 쓴다.

## 4순위 반영: 최종 제출 전략 증거 (이슈 #188)

- [playground-series-s6-leaderboards 데이터셋](https://www.kaggle.com/datasets/georgymamarin/playground-series-s6-leaderboards) (2026-08-17)은 종료된 S6 7개 에피소드의 public·private 보드 원자료를 제공한다.
- [동반 노트북](https://www.kaggle.com/code/georgymamarin/three-of-seven-s6-boards-erased-the-public-top-ten)과 [734005 스레드의 코멘트](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)에서 계산된 결과: 7개 중 3개 보드에서 public 상위 10팀이 private 상위 10에서 전멸했고, 유지된 자리 수는 {0, 0, 0, 5, 6, 8, 8}로 1~4 사이가 없었다.
  AUC 지표 3개 에피소드 중 전멸은 1개(S6E2)였다.
- 같은 스레드에서 broccoli beef의 수식 상수 오류가 정정됐다: public 격차의 95% 단측 판별 한계는 약 0.000141이 아니라 약 0.000216이다.
  이 값은 우리가 마일스톤 격차 해석에 쓰는 노이즈 대역을 넓히는 쪽의 정정이다.
- Tilii(현재 25위)는 2026-08-15 무렵부터 blind blending 파동이 시작되어 public 상단이 외부 제출 재활용으로 재편되고 있으며, S6E3과 비슷한 결말을 예상한다고 관측했다.
  시드 상위권에 노련한 참가자가 많은 달은 top 10이 덜 흔들린다는 경험칙도 덧붙였다.

근거 등급: 보드 원자료 기반 계산은 높음, Tilii의 관측은 일화 수준.
시사점은 기존 규약(최종 선택은 CV 기준, public은 파손 감지용)의 재확인이며, #188의 격차 추정 노이즈 대역에 정정된 0.000216을 반영할 가치가 있다.
디스커션 증분 자체는 다음 증분 회차(`docs/agents/discussion-update.md`)에서 종합 문서에 정식 반영한다.

## 기각 기록

### 삼각함수(주기) 특성

- tomasa2 갱신판은 `notifications_per_day`, `app_opens_per_day` 두 열의 sin/cos 특성이 단일 모델 CV를 +0.00017 올렸음을 두 분할 시드에서 재현했다(저장 출력 있는 통제 실험).
- 그러나 메커니즘은 새 정보가 아니라 분할 비용 절약이다: 조회 표 신호를 이미 가진 모델에서 하나의 분할로 흩어진 값 구간의 합집합을 잡게 해 줄 뿐이다.
- 같은 실험에서 스택 기여는 +0.00002에 그쳤고, 정확값을 범주 수준으로 직접 분할할 수 있는 모델(우리의 CatBoost native 범주, Lookup-Transformer에 해당)에서는 효과가 0이었다.
- 우리 풀 기준 기대 이득이 문턱 미달이므로 열지 않는다.

### multi-level spline 변형과 hypernetwork

- [ern711의 multi-level 판](https://www.kaggle.com/code/ern711/multi-level-deep-univariate-spline-transformer) (2026-08-17)은 contextualized spline 위에 깊이별 예측 헤드 4개, 행별 학습 혼합, 동결 후 hypernetwork 보정 2단계를 얹었다.
- 저자 스스로 [디스커션 735421](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)에서 CV 이득이 매우 작고 복잡도 대비 가치가 불확실하다고 밝혔으며, 수치와 저장 출력이 없다(근거 등급 낮음).
- contextualized spline 자체의 재개는 #166이 이미 제한적으로 결정했고 hypernetwork 등 구조 탐색은 그 결정이 명시적으로 닫았으므로, 이 변형은 지도 172의 spline 트랙 범위이지 새 축이 아니다.
- 같은 스레드의 Tilii 코멘트(KS 0.118790, 100+ 앙상블에서 약 +0.00005)는 `docs/research/spline-comment-reassessment.md`가 이미 평가했다.

### NN Residual Network 0.97101

- [anthonytherrien의 노트북](https://www.kaggle.com/code/anthonytherrien/predicting-smartphone-addict-nn-residual-network) (34표)은 평범한 ResNet MLP에 OOF LightGBM 확률 열을 더한 구조인데, public 0.97101은 NN이 아니라 첨부 vault 데이터셋의 외부 제출 파일 두 개에서 나온다.
- [najiama의 감식 스레드 735404](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735404)가 NN 가중치 1e-6과 vault 파일이 공개 제출물의 바이트 단위 복사임을 스피어만 1.0, 최대 절대 차 0.0으로 증명했다.
- 범용 ResNet 계열은 08-14 재점검에서 이미 제외됐고, 이 건은 그 판정을 바꿀 정보가 없다.

### 빈도 인코딩과 비율 특성 재부상

- [kodaifukuda0311의 XGB 노트북](https://www.kaggle.com/code/kodaifukuda0311/s6e8-xgb-the-power-of-exact-value-te-fe)은 정확값 TE 위에 정확값 빈도 인코딩과 비율 특성이 LB를 더 올렸다고 보고한다.
- 근거가 public LB 이동뿐이고 OOF 제거 실험 수치가 없으며(근거 등급 낮음), 빈도(count) 인코딩 재판정은 지도 172가 닫은 후보이고 비율 특성 무익은 다수의 통제 실험이 확인한 기존 결론이라 열지 않는다.

### 그 밖의 저신호 증분

- 외부 제출 파일 블렌딩 계열: amanatar 0.97092, direcm top-30, yadoy666 94·177구성원 GPU 스택, daniilkrasnovvv memes 3 등은 전부 외부 예측 직접 편입이라 규칙상 제외다.
- 자체 학습 계열 중 koushikkumardinda(44표), kava1(31표), stephentarter 앙상블(자체 CatBoost 0.9645), nurasylserikov bagged 3 boosters(고정 200 iter Optuna 설정), mikhailnaumov XGB, devashish001 EDA는 모두 우리 풀 하한(0.969대)에 못 미치는 기존 레시피 조합이다.
- 개인 산출물 데이터셋 masayakawamata/s6e8-catstr-aug16(OOF 배열만, 설명·코드 없음)과 kenchanhodgkin/pg-s6e8-exp 시리즈(LGBM+TE 계열, OOF 0.96696)는 재현 계약이 없거나 성능이 낮아 진입 진단 가치가 없다.
- omidbaghchehsaraei hill climbing 앙상블은 공개 OOF 라이브러리 대상 탐욕 결합이라 #62·#64 범위와 외부 예측 제외 규칙에 걸린다.

## 한계

- 이 조사는 후보를 실제로 학습하지 않았고, 모든 수치는 공개 자료의 보고치이거나 그 노트북의 자체 실행 결과다.
- 대회 마감(2026-08-31)까지 새 공개 자료가 계속 올라오므로, 실험 동결선(2026-08-28) 전 마지막 디스커션 증분 회차에서 이 문서의 결론을 한 번 재확인해야 한다.
- www.kaggle.com이 현재 망에서 차단되어 디스커션 본문은 원격 렌더러 경유로 읽었으며, 목록에 노출되지 않는 저활동 스레드가 누락됐을 가능성은 배제할 수 없다.
