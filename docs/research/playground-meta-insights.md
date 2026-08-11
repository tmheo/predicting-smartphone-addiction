# Playground Series 메타 인사이트 종합

과거 Playground Series 대회 리서치 세 갈래를 주제별로 재구성하고, S6E8 디스커션 종합 문서(`docs/research/discussion-insights.md`)의 확립 결론과 교차 검증한 문서다.
이슈 [#28](https://github.com/tmheo/predicting-smartphone-addiction/issues/28)의 산출물이며(맵 [#24](https://github.com/tmheo/predicting-smartphone-addiction/issues/24)의 자식), 작성 기준일은 2026-08-10이다.

원자료는 세 개의 리서치 노트다.

- 유사 에피소드 11개 선별표: [이슈 #25](https://github.com/tmheo/predicting-smartphone-addiction/issues/25), `research/ps-episode-inventory` 브랜치의 `docs/research/ps-episode-inventory.md`
- Chris Deotte 글 14건 읽기 노트: [이슈 #26](https://github.com/tmheo/predicting-smartphone-addiction/issues/26), `research/deotte-corpus` 브랜치의 `docs/research/deotte-corpus.md`
- 상위 솔루션 12건 정독 노트: [이슈 #27](https://github.com/tmheo/predicting-smartphone-addiction/issues/27), `research/ps-top-solutions` 브랜치의 `docs/research/ps-top-solutions.md`

교차 검증 기준은 `docs/research/discussion-insights.md`, 특히 "실행 요약: 파이프라인 기본값" 6항이다.
과거 대회 경험이 기존 결론과 일치하는지, 보강하는지, 상충하는지를 각 장에서 표시하고, [기본값 6항 대조 판정표](#기본값-6항-대조-판정표)에 모아 두었다.

## 1. 총론: 합성 데이터 대회의 승리 공식

세 노트 전체를 관통하는 골격은 Deotte의 두 신호 원칙과 상위 솔루션의 3층 구조로 요약된다.

- 두 신호 원칙: 모든 플레이그라운드에는 원본 데이터셋의 신호와 합성 생성기의 신호가 있고, 원본 타깃이 무작위면 생성기 역공학에 집중한다 ([S5E9 디스커션](https://www.kaggle.com/competitions/playground-series-s5e9/discussion/604028), [S5E6 1위 writeup](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi)).
  S6E8은 원본(임계값 규칙, AUC 0.9888)과 생성기(완만한 확률 구조)를 이미 같은 틀로 분리해 놓았으므로 ([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428), [732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)), 이 원칙과 정확히 일치한다.
- 3층 구조 ([ps-top-solutions 총론](https://github.com/tmheo/predicting-smartphone-addiction/issues/27)): 신호 층(생성기를 노이즈가 아니라 구조화된 신호로 취급), 물량 층(다양한 계열의 OOF 수십~수백 개), 규율 층(선택과 결합은 OOF 기준, 최종 제출은 CV-LB 관계가 건강한 구간에서).
- 행동 범위는 데이터 크기가 정한다: 행이 많으면 공격적 피처 엔지니어링과 스태킹, 행이 적으면 피처 엔지니어링 없이 균등 평균이다 ([S5E3 2위 writeup](https://www.kaggle.com/competitions/playground-series-s5e3/writeups/chris-deotte-2nd-place-gbdt-nn-svr-original-data)).
  S6E8은 train 91만 행 대용량이므로 "공격적 피처 엔지니어링 + 스태킹" 구간이다.
- 상위권과 중위권을 가른 요소는 대회마다 달랐다: 피처가 가른 대회(S6E3, S5E11, S4E1), 검증 설계가 가른 대회(S5E12), 물량과 완성도가 가른 대회(S5E8, S6E5), 제출 선택 규율이 가른 대회(S6E2).
  S6E8은 눈금값 아티팩트가 확인된 대용량 대회라 S6E3·S5E11형(피처)에 가깝고, 후반부는 S5E8형(물량전)이 겹칠 전망이다.
  여기서 눈금값 아티팩트란 수치 컬럼 값이 연속적으로 고르게 퍼져 있지 않고 자의 눈금처럼 정해진 이산값들 위에만 몰려 있는 현상을 말한다.
  스마트폰 도메인의 성질이 아니라 원본 데이터가 특정 값 위에서 생성·반올림되고 합성 생성기가 그 값 패턴을 그대로 복제하면서 남은 데이터 생성 과정의 흔적이며, 덕분에 수치 컬럼이 사실상 유한한 범주형처럼 동작해 정확값 타깃 인코딩 같은 피처 엔지니어링이 순위를 가르게 된다 (`docs/research/discussion-insights.md`의 "값이 정해진 눈금 위에만 있어, 정확한 값 자체가 강력한 단서다" 절).

기존 결론과의 관계: 일치하며, "데이터 크기에 따른 행동 범위"라는 상위 프레임을 보강한다.

## 2. 정확값·결합 컬럼 타깃 인코딩

S6E8의 확립 결론(정확값 문자열화 TE +0.0032, [733495](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495) 코멘트)과 같은 계열의 기법이 과거 대회에서 가장 반복적으로 확인된 승리 기법이다.

### 수치의 범주형 취급은 검증된 정석이다

- Deotte의 피처 엔지니어링 중심이 "수치 컬럼을 범주형으로 취급 + 타깃 인코딩"이다 ([S5E2 1위](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/565539), [S4E12 1위](https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328)).
- S4E10 1위는 수치 피처를 그대로 두고 범주형 복사본을 추가하는 것이 가장 효과적이었다고 밝혔고 ([S4E10 1위](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/543725)), S6E2 1위도 전 피처 문자열화 표현(ALL_CATS)을 상위 구성 대부분에 포함시켰다 ([S6E2 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)).
- TE가 통하는 이론적 근거도 Deotte가 제시했다: 인코딩은 범주 집합을 바꾸지 않고, 트리가 못 하는 일(고카디널리티 값 묶음의 효율적 분할)을 대신해 줄 뿐이다 ([S5E2 디스커션](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/564056)).
  이는 S6E8의 "트리가 스스로 만들 수 있는 경계를 다시 쓰는 피처는 실패한다"는 결론의 이론적 보강이다.

기존 결론과의 관계: 강한 일치.

### 결합 컬럼(쌍/삼중) TE/CE가 다음 확장이다

여기서 TE는 타깃 인코딩(그 값을 가진 행들의 타깃 평균을 피처로 주는 것), CE는 빈도 인코딩(그 값을 가진 행이 데이터에 몇 번 등장하는지를 피처로 주는 것)이다.
결합 컬럼 TE/CE란 컬럼 하나가 아니라 두세 컬럼의 값 조합(예: sleep_hours가 6.5이면서 app_opens_per_day가 42인 경우)을 하나의 키로 묶고, 그 조합 키에 대해 TE와 CE를 적용하는 확장을 말한다.

- Deotte는 S5E6에서 쌍(28) + 삼중(56) + 사중(70) 결합 컬럼에 TE를 적용한 2,268 컬럼 XGBoost로 우승했다 ([S5E6 1위 writeup](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi)).
- 독립 표본이 셋 더 있다: S6E3 1위의 범주 쌍/삼중 결합 중첩 TE(약 37개 모델, TE 통계량은 std·min·max·분위수까지 다양화) ([S6E3 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)), S5E8 2위의 bigram TE/CE ([S5E8 2위 writeup](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/2nd-place-yet-another-ensemble)), S5E11 1위의 기본 피처 쌍 결합 + 자릿수 결합 TE/CE ([S5E11 1위 writeup](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an)).
- 확장의 경험칙: 쌍 결합은 꾸준히 이득, 삼중 이상은 미미하거나 노이즈다 ([S5E8 3위 writeup](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon)).
- 탐색 절차는 S4E12 1위의 "CV가 오르는 조합만 자동 수집" 루프가 모범이다 ([S4E12 1위](https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328)).
  루프의 구조는 이렇다: 컬럼 2~3개를 무작위로 뽑아 조합 키를 만들고, 거기에 TE/CE를 적용한 후보 피처를 베이스라인 피처 세트에 추가해 CV를 재고, CV가 오르면 그 조합을 저장하고 아니면 버리는 시행을 수천 번 반복한다.
  그는 이 루프를 수 일 동안 GPU로 돌려 살아남은 강한 조합 약 170개를 얻었고, 그렇게 수집한 조합 피처로 구성한 611피처 단일 XGBoost로 앙상블 없이 우승했다.
  어떤 조합이 의미 있을지를 사람이 도메인 가설로 고르는 대신, 조합 생성과 채택 판정을 "무작위 생성 + CV 상승 여부"라는 기계적 절차로 바꾼 것이 핵심이다.
  수천 번 시도 중 우연히 CV가 오른 조합이 섞이지 않느냐는 의문에 대한 그의 방어 논리가 "대용량이면 OOF 전체가 우연히 좋아지기 어렵다"이며(데이터가 적으면 이 루프 대신 2~3개 조합만 손으로 시도해야 한다), 이 논리는 91만 행 S6E8에서도 성립한다.
- 단 그는 이 루프를 GPU로 돌렸고, 이 프로젝트의 로컬 CPU 환경에서 같은 절차를 쓰려면 탐색 단계의 검증 비용을 낮추는 프록시 CV가 필요하다.
  프록시 CV란 정식 판정 기준(커밋된 `artifacts/folds.parquet`의 5-fold CV) 대신 탐색 단계에서만 쓰는 약식 검증을 말한다.
  예를 들어 `folds.parquet`의 특정 fold 하나만 holdout으로 쓰면 조합당 학습이 5번에서 1번으로 줄고, 행 샘플링이나 가벼운 학습 설정(높은 학습률, 공격적 early stopping)을 더하면 더 줄어든다.
  탐색 단계에 필요한 것은 조합의 정확한 점수가 아니라 조합 간 순위뿐이고, 약식 검증에서도 순위는 대체로 보존되므로 이것으로 충분하다.
  규율은 두 가지다.
  첫째, 프록시 점수는 후보 걸러내기에만 쓰고, 살아남은 상위 후보는 반드시 정식 5-fold로 재검증해 채택을 판정한다.
  둘째, holdout fold는 새로 만들지 않고 커밋된 `folds.parquet`의 fold를 그대로 골라 써서, 정식 검증과의 비교 가능성과 fold 분할 읽기 전용 원칙을 유지한다.
- S6E8 쪽에도 개연성 근거가 있다: gaming_hours 등이 단독으론 무익하지만 강한 컬럼과의 조합으로 +0.00380을 기여한다는 관측이다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트).

기존 결론과의 관계: 보강이자 확장이다.
S6E8은 단일 컬럼 정확값 TE까지만 검증됐고, 결합 컬럼 TE는 S6E8 디스커션에서 아무도 보고하지 않은 미개척 영역이다.

### 같은 키의 추가 표현: CE(빈도)와 native categorical

- 상위 솔루션들은 TE와 CE(값 빈도)를 항상 짝으로 적용했다 ([S6E3 1위](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm), [S5E8 2위](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/2nd-place-yet-another-ensemble), [S5E11 1위](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an)).
  CE는 "생성기가 각 원본 값 주위에 합성 행을 얼마나 몰아 만들었는가"라는 오버샘플링 정보를 인코딩하며, `합성 빈도 / 원본 빈도` 드리프트 비율까지 확장된다.
- 한 범주형 컬럼에 라벨 인코딩, TE mean/median/min/max/nunique, CE의 7가지 표현을 동시에 주면 GBDT의 이해 경로가 늘어난다 ([S4E12 1위](https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328)).
- TE를 거치지 않고 native categorical(CatBoost, LightGBM categorical_feature)로 직접 먹이는 변형도 미실험 트랙이다 ([S4E10 1위](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/543725) 계열).
  S6E8은 전 피처가 눈금값이라 카디널리티가 유한하므로 구현 비용이 낮다.

기존 결론과의 관계: 보강(미실험 확장 제시).

### 자릿수 추출은 단독이 아니라 결합으로만

- 자릿수·소수점 추출은 S6E3(약 60개 모델), S6E2, S5E11에서 널리 쓰였고, S5E11 1위는 "서로 다른 피처의 자릿수끼리" 결합 후 TE/CE를 적용한 것이 단독 2위감 XGBoost의 핵심이었다 ([S5E11 1위 writeup](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an)).
- 그러나 S6E8의 정밀 재측정은 other_screen 잔차가 있으면 _decimals 계열의 한계 기여가 거의 0이라고 판정했다 ([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트).

기존 결론과의 관계: 부분 상충이나 양립 가능하다.
단순 자릿수 피처 단독 추가는 기존 판정(후순위)을 유지하고, 과거 대회가 실증한 것은 "자릿수 결합 + TE/CE"라는 별개 기법이므로 결합 TE 실험 트랙 안에서만 소화한다.
근거 강도는 S6E8 자체 ablation이 우선이다: 과거 대회의 유효성은 그 대회 생성기의 성질에 의존하므로, S6E8에서 겹침 측정 없이 이식하지 않는다.

## 3. 원본 데이터 활용의 네 경로

S6E8의 기존 결론은 "원본 데이터 훈련 미사용(검증 참고 용도만)"이다.
과거 대회 경험은 이 결론을 경로별로 쪼개서 보라고 말한다: 실패가 확인된 것은 "행 주입" 한 경로뿐이고, 나머지 세 경로는 미실험이다.

### 3.1 행 이어붙이기(주입): 기각 유지

- 과거 대회에서는 유효했던 사례가 있다: S5E12 1위는 대부분 모델에서 원본을 train에 이어붙였고 ([S5E12 1위 writeup](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)), S4E1 3위는 원본 이중 이어붙이기가 private 최고였다 ([S4E1 3위](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472413)).
- 그러나 S5E8 3위는 원본 추가 주입이 LB 개선 없음을 보고했고 ([S5E8 3위 writeup](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon)), S3E24 3위도 원본 병합이 초반엔 오히려 손해였다고 밝혔다 ([S3E24 3위](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/455248)).
- S6E8은 50배 가중치 주입 실험에서 10개 폴드 전부 하락이 확인됐다 ([733552](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733552)).

판정: 상충처럼 보이지만 S6E8 자체 실험 근거가 더 강하므로 기각을 유지한다.
대회마다 원본과 합성의 분포 거리가 다르고, S6E8은 원본 프록시가 7,500행뿐이며 그 원본조차 합성이다.
다만 S5E8 1위처럼 "원본 포함/제외를 앙상블 다양성 축으로 쓰는" 용법 ([S5E8 1위](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/603210))은 주입 실패와 별개이므로, 스태킹 재료 단계에서 소규모 재검 여지만 남긴다.

### 3.2 원본 기준 타깃 통계 prior(새 컬럼): 이식 후보

- S6E3 1위는 원본 7,032행에서 피처 값별 churn 확률을 계산해 피처로 썼고, 원본에는 합성 train 라벨이 없으므로 누수가 0이라는 점을 강조했다 ([S6E3 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)).
- S6E2 1위도 원본에서 타깃 평균, WoE, 엔트로피를 추출한 "외부 타깃 인코딩" 세트(ORIG)를 반복 사용했다 ([S6E2 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)).
- Deotte의 S5E6 방식도 train 기준과 원본 기준 인코딩을 병행했다 ([S5E6 1위 writeup](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi)).

판정: 기존 결론과 조건부 상충이며, 이 지점은 과거 대회 쪽 근거가 더 강하다.
"원본 훈련 미사용" 결론의 실험 근거는 행 주입 실패뿐이고, 새 컬럼 경로는 폴드 누수가 원리적으로 0이며 같은 시즌 1위 두 명이 공통 사용했다.
S6E8의 상한 0.835 함정(임계값 규칙을 예측값으로 쓰는 경우, [732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434))에도 걸리지 않는다: 규칙이 아니라 피처 하나로 주는 것이기 때문이다.
따라서 기본값 문구를 "원본 데이터 행 주입 미사용"으로 좁히고, 새 컬럼 prior를 도전자 실험으로 개방하는 것이 맞다.

### 3.3 원본 규칙 잔차 부스팅(base_margin): 도전자 후보

- 1단계 모델을 원본으로만 학습하고 그 로짓을 XGBoost의 base_margin으로 넣으면 2단계는 생성기 신호만 학습하면 된다는 제안이며, AUC 이진 분류에서 LB +0.00068 실증 코멘트가 있다 ([S5E11 디스커션](https://www.kaggle.com/competitions/playground-series-s5e11/discussion/614986)).
- 모델 예측판 변형도 있다: S4E10 1위는 4개 모델의 OOF 로짓을 CatBoost baseline으로 넣어 재학습해 자기 자신의 예측조차 개선했다 ([S4E10 1위](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/543725)).
  다만 S5E8 1위가 같은 기법을 시도했을 때는 대부분 개선이 없었으므로 ([S5E8 1위](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/603210)), 항상 통하는 기법은 아니다.

판정: 조건부 상충이며 미실험 경로다.
S6E8의 임계값 규칙(원본 AUC 0.9888)을 1단계로 삼으면 상한 0.835 논리가 적용되지 않는다: 2단계가 그 위에 쌓기 때문이다.
개선이 없어도 스태킹 다양성 재료로 남는다는 기대치로 실험할 가치가 있다.

### 3.4 스냅 피처·부분집합 매칭·최근접 이웃: 조건부 후보

- S6E3 1위의 최고 효과 기법은 합성 값을 원본 최근접 값으로 되돌린 snap과 이탈 크기 snap_diff였다 ([S6E3 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)).
  S6E8은 생성기가 눈금값 패턴을 그대로 복제했으므로 snap 자체는 항등에 가까울 가능성이 높고, 눈금 이탈 행 비율 측정을 선행 조건으로 하는 조건부 트랙이다.
- S4E1 2위는 1~10개 피처 부분집합의 값 조합이 원본에 그대로 존재하는지를 0/1 피처로 넣어 극적인 타깃 분리(0.203 대 0.708)를 얻었다 ([S4E1 2위](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472496)).
  S6E8은 원본 프록시가 7,500행뿐이라 2~3개 컬럼 조합까지만 유효할 것이고, 정확값 TE와의 기여 겹침 확인이 필수다.
- 원본 cKDTree 최근접 이웃 라벨 조회와 원본 전용 PCA/DAE 피처는 구현이 저렴한 보조 다양성 재료다 ([S6E3 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)).

판정: 기존 결론에 없던 신규 후보(보강)이며, 전부 측정 선행 조건부다.

## 4. 결측치와 분포 이동: 기존 결론이 이긴다

### 결측 피처

- 과거 대회에는 결측이 강력한 신호였던 반례가 있다: S5E2의 NaN 패턴 base-2 피처다 ([S5E2 1위](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/565539)).
  그러나 그 대회는 결측이 생성기 복제 구조의 일부였던 경우다.
- S6E8은 결측이 합성 후 무작위 삭제이고 타깃과 독립임이 복수 ablation으로 확인됐다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214), [731764](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/731764) 코멘트).

판정: 조건부 상충이나 S6E8 근거가 압도적으로 강하다.
결측 피처 배제 결론을 유지한다.
교훈은 "결측의 신호 여부는 대회마다 재검증한다"는 메타 규칙 쪽이다.

### 행 위치(ID) 분포 이동

- S5E12 2위는 train 꼬리로 갈수록 test 분포에 가깝다는 것을 적대적 검증 미세 스캔으로 확정하고, 표본 가중치(tail 16배)와 cutoff 검증으로 우승권에 들었다 ([S5E12 2위 writeup](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an)).
- S6E8은 결측을 채우면 적대적 AUC 0.503으로 값 분포 차이가 없음이 확인돼 있다 ([733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214)).

판정: 일치(기존 결론 유지)하되 진단 절차만 보강한다.
기존 확인은 전역 검사였으므로, id 구간별 미세 스캔(rolling 타깃 평균, 구간별 적대적 AUC) 1회는 저렴한 보험이다.
음성이 확인되면 표준 Stratified K-Fold의 근거가 더 단단해진다.

### 라벨 뒤집기 트릭

- S4E1에서 +0.01, S5E8에서 +0.00001 수준으로 통했던 후처리 트릭이다 ([S4E1 2위](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472496), [S5E8 1위](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/603210)).
- train/test 중복 행 존재가 전제인데 S6E8은 중복 0이다 ([732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434)).

판정: 기각(적용 불가). 기존 결론과 일치.

## 5. 앙상블·스태킹 체계

### OOF 물량과 다양성이 기본 체력이다

- 1위 솔루션들의 OOF 규모: S6E3 150개, S6E2 150개, S5E8 136개, S6E5 186개, S5E11 100개.
- 다양성 축은 모델 계열(GBDT 3종 + RealMLP, TabM, FT-Transformer 등 NN 다계열 + RF, FM, 로지스틱), 피처 세트, 깊이 극단화, 시드, 원본 포함 여부다.
- 단독 CV가 낮은 모델도 다양성 기여로 반복 선택된다 ([S5E5 1위](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611), [S6E2 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)).

기존 결론과의 관계: 강한 일치 + 보강.
S6E8 결론(모든 런에서 OOF/테스트 예측 저장, 스태킹 없이는 0.970+ 어려움, [733023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023))과 같고, 남은 것은 NN 계열 확장이다.
기존 기본값의 "여력이 되면 스태킹"이라는 표현은 과거 대회 경험에 비추면 너무 약하다: 상위권에서 스태킹은 선택이 아니라 필수 골격이다.

### 부분집합 선택은 필수 단계다

- 전부 평균하면 오히려 나빠진다는 것이 공통 관측이다 ([S6E2 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)).
- 구현 후보: 탐욕적 힐클라이밍 ([S5E5 1위](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611)), 배깅 힐클라이밍(50 bags, [S5E12 2위 writeup](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an)), Optuna 부분집합 선택 + Ridge 재결합 ([S6E2 1위](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t), [S5E12 1위](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)), 시드 평균본만 후보로 쓰는 규칙 ([S6E2 2위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/2nd-place-solution-avoid-leaks-and-overfitting)).

기존 결론과의 관계: 보강(기존 문서에 없던 단계).

### 메타러너: 선형 기본값, 비선형은 도전자

- 선형 우세 표본: S6E3 1위(L2 로지스틱), S6E2 1위(Ridge, 비선형 메타는 과적합), S5E11 1위(Ridge/힐클라이밍, GBDT/NN 메타는 크게 나쁨), S5E12 1위(Ridge).
- 비선형 우세 표본: S5E8 1~3위(AutoGluon, CatBoost 메타), S4E10 1위(NN 스택).
- 비선형이 이기는 조건에 대한 Deotte의 해석: 가장 강한 피처가 일부 행에서 결측이라 "결측일 때 잘 맞히는 모델"과 "값이 있을 때 잘 맞히는 모델"을 행마다 갈아타야 할 때다 ([S5E4 1위](https://www.kaggle.com/competitions/playground-series-s5e4/discussion/575784)).
- S6E8은 12개 전 컬럼에 16~19% 결측이 있어 이 조건이 성립할 수 있는 구조다.

기존 결론과의 관계: 보강.
기본값은 선형(순위 평균/Ridge)으로 두되, 비선형 메타(XGB/NN)를 도전자로 비교할 구조적 근거가 있다.
전제 조건(모든 모델이 같은 K-fold 분할 공유, TE 누수 제거)은 기존 검증 위생 결론과 일치한다.

### 블렌딩 방식: 순위 평균 대 OOF 가중치 탐색

- S6E8 기본값은 순위 평균이다(확률 눈금 차이 제거, [734063](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)).
- Deotte의 기본값은 OOF 기반 가중치 탐색(힐클라이밍)이며 ([S5E5 1위](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611)), 눈금 문제를 OOF 점수 최적화로 우회한다.

판정: 상충(긴장)이며 실험으로 재확정할 사안이다.
91만 행 OOF면 가중치 과적합 위험이 낮아 양쪽 다 성립하므로, 어느 쪽 근거가 강하다고 단정하지 않고 둘 다 재서 비교한다.
다만 가중 블렌드 미세 최적화가 CV만 올리고 LB가 정체된 실패 사례 ([S5E8 3위 writeup](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon))가 있으므로, paired sigma(0.00015) 이하의 이득은 노이즈로 취급한다.

### 마무리 상수 이득 레퍼토리

- 시드 다중 학습 확률 평균: AUC 같은 순위 기반 지표에서 확률 흔들림을 줄여 실질 이득이 있다 ([S5E6 1위 writeup](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi)).
  S6E8의 시드 앙상블 결론(순위 ±60계단 노이즈 축소, [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005))과 일치한다.
- 100% 데이터 재학습 + 반복 수 1/(K-1) 증가(5-fold면 1.25배): Deotte의 레퍼토리 ([S5E5 1위](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611))와 S6E2 1위의 독립 보고 ([S6E2 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t))가 같은 수치로 재확인됐다.
- 의사 라벨링: Deotte 레퍼토리에는 있으나 S6E2 1위는 CV 개선 없음으로 기각했다.
  후순위로 강등하고 시도한다면 마지막 주간 소규모로만 한다.

기존 결론과의 관계: 시드 앙상블은 일치, full-data 재학습은 보강(기존 문서에 없던 기법).

## 6. 검증·제출 규율

- CV 절대값이 아니라 CV-LB 관계를 신뢰한다: S6E2 1위는 CV 0.95578 위부터 CV 개선이 LB로 이어지지 않는 split 과적합을 진단하고, 최고 CV 제출을 버리고 관계가 건강한 구간의 제출을 골라 우승했다 ([S6E2 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)).
  S6E8의 paired sigma 결론(비슷한 블렌드 구분 한계 0.00015, [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005))과 결합하면 운영 규칙이 완성된다: 마일스톤 제출로 CV-LB 짝을 기록하고, CV만 오르는 구간에 들어서면 그 계열의 CV 이득을 할인한다.
- 제출 물량전의 함정: S5E8 1위조차 마지막 주간 제출 대부분이 public 최고점을 못 넘었고, 최종 선택하지 않은 제출이 private에서 더 좋았던 사례를 공개했다 ([S5E8 1위](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/603210)).
  S6E8의 best-of-N 함정 결론 ([733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618))의 실전 표본이다.
- 최종 2제출: CV 최고 + 구성이 가장 다른 차선으로 위험을 분산한다 ([S5E1 2위](https://www.kaggle.com/competitions/playground-series-s5e1/discussion/560549)의 변형).
- 실험 속도가 곧 탐색량이고 탐색량이 곧 점수다 ([NVIDIA cuDF FE 블로그](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)).
  champion/challenger 루프의 회전 속도 자체가 경쟁력이라는 뜻이다.

기존 결론과의 관계: 강한 일치 + 보강(CV-LB 관계 관측이라는 조건 추가).

## 7. 실패 사례 종합

상위권 writeup에 명시된 실패 목록이다.
같은 시도를 반복하기 전에 먼저 확인한다.
상세 표는 `research/ps-top-solutions` 브랜치의 `docs/research/ps-top-solutions.md` 5장에 있다.

- 의사 라벨링, 지식 증류, 아주 깊은 GBDT, 고차 상호작용 대량 전개, 비선형 스태킹(그 대회에선), 무선별 OOF 전체 평균, public LB 등반 ([S6E2 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)).
- 3-way 이상 조합 피처, 고상관 피처의 곱 상호작용, 가중 블렌드 미세 최적화, 원본 데이터 추가 주입 ([S5E8 3위 writeup](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon)).
- AutoFE 라이브러리(FeatureTools, AutoFeat) ([S6E5 1위 writeup](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth)).
- 손수 만든 도메인 피처는 합성 데이터에서 체계적 인코딩 대비 열세다 ([S4E10 1위](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/543725)).
  S6E8 결론(도메인 피처 +0.0002 이하)과 일치한다.

## 기본값 6항 대조 판정표

`docs/research/discussion-insights.md`의 "실행 요약: 파이프라인 기본값" 6항 각각에 대해, 과거 대회 경험(세 노트)의 판정을 정리한다.

| # | 기존 기본값 | 판정 | 근거와 조치 |
| --- | --- | --- | --- |
| 1 | 데이터: NaN 그대로, 대치 없음, 결측 피처 없음, 원본 데이터 훈련 미사용 | 부분 수정 | NaN 처리와 결측 피처 배제는 일치(S5E2의 반례는 결측이 생성 구조였던 특수 사례, S6E8 ablation이 우선). 단 "원본 훈련 미사용"은 행 주입 실패만 근거이므로 "행 주입 미사용"으로 좁히고, 원본 기준 타깃 통계 prior(새 컬럼)와 잔차 부스팅(base_margin)을 도전자로 개방한다. 이 지점은 과거 대회 근거(S6E3·S6E2 1위 공통 사용, 누수 0 원리)가 더 강하다 |
| 2 | 검증: Stratified K-Fold 10-fold, 플라시보 피처, OOF/테스트 예측 저장 | 유지 + 보강 | 전 항목 일치. 보강: 모든 모델의 폴드 분할 공유를 스태킹 전제로 명문화, id 구간별 미세 스캔 1회(S5E12형 이동의 음성 확인), CV-LB 짝 기록 |
| 3 | 피처: 정확값 TE 최우선, other_screen 잔차/slack 추가, 선형 결합/비율/결측 피처 금지 | 유지 + 확장 | 정확값 TE는 과거 대회 4개 독립 표본이 재확인한 정석. 확장: 결합 컬럼(쌍부터) TE/CE, TE 통계량 다양화, native categorical 표현. 금지 목록도 일치(곱/비율 피처 실패는 S5E8 3위도 확인) |
| 4 | 모델: LightGBM/XGBoost 고용량, early stopping, monotone 금지, 불균형 대응 없음 | 유지 + 보강 | 낮은 학습률 + 큰 반복 수 + early stopping은 Deotte 정석과 일치. 아주 깊은 GBDT 실패(S6E2)는 "깊이가 아니라 잎 수 + 낮은 학습률" 경로를 지지. 보강: 스태킹 재료로 NN 계열(RealMLP, TabM 등)과 고전 모델 확장 |
| 5 | 앙상블: 시드 앙상블 + 순위 평균 블렌딩, 여력이 되면 OOF 스태킹 | 수정 | 시드 앙상블은 강한 일치. 그러나 "여력이 되면 스태킹"은 격상 필요: 과거 대회 전 표본에서 스태킹/선택 앙상블은 상위권 필수 골격. 순위 평균 대 OOF 가중치 탐색은 상충이며 어느 쪽도 근거 우위를 단정할 수 없어 비교 실험으로 재확정. 부분집합 선택 단계 신설 |
| 6 | 제출: 확률 제출, 최종 선택 CV 기준, public 0.0001 미만 노이즈 취급 | 유지 + 보강 | "Trust Your CV"와 일치. 보강: 선택 규칙을 "최고 CV"가 아니라 "CV-LB 관계가 유지되는 최고 CV"로 조건 강화(S6E2 1위의 split 과적합 교훈), 최종 2제출은 CV 최고 + 가장 다른 구성 |

종합하면 6항 중 4항은 유지(보강 포함), 2항(1번의 원본 활용 문구, 5번의 앙상블 체계)은 수정 판정이다.
상충 지점 중 과거 대회 근거가 이긴 곳은 원본 새 컬럼 경로(3.2절)뿐이고, 나머지 상충(결측 피처, 자릿수 단독, 행 주입)은 전부 S6E8 자체 ablation이 이겼다.
블렌딩 방식(순위 평균 대 가중치 탐색)만 무승부로 실험 재확정 대상이다.

## 우리 파이프라인에 시사하는 조정 후보

세 노트의 이식 우선순위를 병합한 최종 순위다.
실제 실험 티켓화는 이 맵([#24](https://github.com/tmheo/predicting-smartphone-addiction/issues/24)) 범위 밖이며, 여기서는 후보와 우선순위만 기록한다.

1. 결합 컬럼 정확값 TE/CE: 독립 표본 4개(S5E6, S6E3, S5E8, S5E11)가 지목한 최우선 트랙.
   쌍부터 시작하고 삼중은 CV가 오를 때만, TE 통계량 다양화와 CE 병행, 자릿수 결합도 이 트랙에서 소화, 탐색은 "CV가 오르는 조합만 수집" 루프(S4E12)로.
2. 원본 프록시 기준 타깃 통계 prior: 누수 0, 구현 저렴, S6E3·S6E2 1위 공통 사용.
   기본값 1번 문구 수정(행 주입 미사용으로 축소)과 함께 간다.
3. 앙상블 체계 격상: NN 계열(RealMLP, TabM) 추가로 OOF 풀 확장, 시드 평균본만 후보로 한 부분집합 선택(배깅 힐클라이밍 또는 Optuna+Ridge) 단계 신설, 순위 평균 대 OOF 가중치 탐색 비교 실험.
4. 비선형 level 2 스태킹 도전: 결측 구간별 전문가 갈아타기 가설(S5E4)의 검증을 겸한다.
   기본값은 선형 유지.
5. 원본 규칙 base_margin 잔차 부스팅과 baseline 재부스팅: 개선 없어도 스태킹 다양성 재료로 남긴다는 기대치로.
6. 스냅 잔차와 원본 부분집합 매칭: 눈금 이탈 행 비율 측정과 정확값 TE 겹침 확인을 선행 조건으로 하는 조건부 트랙.
7. full-data 재학습(반복 수 1/(K-1) 증가) + 시드 평균: 마일스톤 제출 직전의 상수 이득 레퍼토리.
8. id 구간별 미세 스캔: 1회성 진단, 음성 확인이 목적.
9. 후순위: native categorical 표현, 정확값 컬럼의 TE nunique 등 추가 표현, 의사 라벨링(마지막 주간 소규모).

기각 유지: 원본 행 주입, 라벨 뒤집기 트릭, 결측 피처, 단순 자릿수 피처 단독 추가, AutoFE 라이브러리, 도메인 직관 피처.
