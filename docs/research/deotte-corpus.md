# Chris Deotte 코퍼스 읽기 노트

Kaggle 그랜드마스터 Chris Deotte([cdeotte](https://www.kaggle.com/cdeotte))의 Playground Series 관련 글을 직접 읽고 정리한 노트다.
이슈 [#26](https://github.com/tmheo/predicting-smartphone-addiction/issues/26)의 산출물이며, 읽기 기준일은 2026-08-10이다.
모든 항목은 실제로 본문을 열어 확인한 글만 담았고, 각 항목마다 링크, 핵심 주장, S6E8 적용 판단을 기록했다.
S6E8 적용 판단은 `docs/research/discussion-insights.md`에 확립된 결론과의 일치 여부를 함께 표시한다.

수집 경로 참고: Kaggle 페이지는 Jina Reader로 읽었고, 검색엔진으로 글을 식별했다.
Deotte의 노트북 본문(코드 셀)은 로그인 없는 렌더링으로는 열리지 않아, 디스커션/솔루션 writeup과 본인이 쓴 NVIDIA 기술 블로그를 1차 자료로 삼았다.
S6E8 대회 자체에는 Deotte가 남긴 글이 검색되지 않았다(2026-08-10 기준).

## 총론: Deotte의 플레이그라운드 전략 골격

읽은 글 전체에서 반복되는 그의 원칙을 먼저 요약한다.

1. 모든 플레이그라운드 대회에는 두 가지 신호가 있다: 원본 데이터셋의 신호와, Kaggle 합성 데이터 생성기의 신호다.
   원본 타깃이 무작위면 생성기 신호에만 집중하고, 원본에 신호가 있으면 두 신호를 모두 학습한다.
2. 첫 작업은 항상 로컬 검증 체계 구축이고, 그다음이 "이 데이터가 과적합되기 쉬운가"의 평가다.
   행이 많으면 공격적 피처 엔지니어링과 스태킹, 행이 적으면 피처 엔지니어링 없이 다양한 모델의 균등 평균이다.
3. 피처 엔지니어링의 중심은 범주형 취급 + 타깃 인코딩(TE)과 count 인코딩(CE)이며, 수치 컬럼도 범주형으로 취급해 TE한다.
4. 앙상블은 다양성이 전부다.
   GBDT + NN + 고전 ML(SVR, KNN, 선형 모델)을 섞고, OOF 기반으로 가중치를 탐색하거나(level 2가 선형) 스태킹한다(level 2가 비선형).
5. 마지막 주간의 상수 이득 레퍼토리: 시드 여러 개 학습 후 확률 평균, 100% 데이터 재학습(반복 수는 K-fold 평균의 1/(K-1)만큼 증가), 테스트 데이터 의사 라벨링.
6. 최종 제출은 CV 기준으로 고른다.

## 글별 노트

### 1. S5E9 디스커션: 플레이그라운드 원본 타깃의 절반은 무작위다

- 링크: [Are 50% of Kaggle Playground Regression Competitions created from Random Targets?](https://www.kaggle.com/competitions/playground-series-s5e9/discussion/604028)
- 핵심 주장:
  - 원본 데이터셋에 신호가 있는지 판별하는 절차를 제시한다: 원본에 XGB를 학습시킨 CV와, 타깃을 무작위로 섞어 학습시킨 XGB 100개의 CV 분포를 비교해, z-score -2~2 안이면 원본 타깃은 무작위라고 판정한다.
  - 최근 회귀 에피소드 6개 중 3개의 원본 타깃이 무작위였다.
  - 원본이 무작위라도 합성 생성 과정 자체가 신호를 추가하므로, 대회가 복권이 되는 것은 아니다.
  - 전체 에피소드를 일괄 분석하는 노트북 [Analyze Original Dataset from Kaggle Playgrounds](https://www.kaggle.com/code/cdeotte/analyze-original-dataset-from-kaggle-playgrounds)을 공개했다(본문은 미열람, 존재만 확인).
- S6E8 적용:
  - 일치: S6E8 분석([732428](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732428), [732434](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732434))이 이미 같은 틀로 원본(임계값 규칙, AUC 0.9888)과 생성기(완만한 확률 구조)를 분리했다.
  - 일치: "생성 과정이 신호를 추가한다"는 주장은 원본에서 동전 던지기였던 중간 구간이 합성에서 AUC 0.896 신호 영역이 됐다는 S6E8 관측과 정확히 같은 현상이다.
  - 참고: 타깃 셔플 널 분포로 신호를 판정하는 절차는 우리 널 임포턴스/플라시보 검증과 같은 사상이며, 이미 파이프라인에 반영되어 있다.

### 2. S5E2 디스커션: 무작위 타깃 데이터에서 신호를 찾는 법

- 링크: [Backpack Data Explained - How To Find Signal](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/564056)
- 핵심 주장:
  - 원본 타깃이 무작위인 대회에서도 CV가 꾸준히 오른다면 그것은 노이즈 적합이 아니라 실제 신호다.
    무작위 타깃은 원리적으로 적합이 불가능하기 때문이다.
  - 이 대회의 신호는 생성기가 원본 행을 사실상 80벌 복제한 구조였고, groupby 집계(또는 KNN)가 그 복제 행들을 찾아내는 도구였다.
  - 인코딩(라벨, TE 등)은 범주 집합 자체를 바꾸지 않는다.
    인코딩의 유일한 효과는 트리가 고카디널리티 컬럼을 효율적으로 분할할 수 있게 해 주는 것이다.
  - 반복 관측이 충분한 패턴(수십 회 이상)만 통계적으로 신뢰하고, 몇 번 본 패턴은 결론에 쓰지 않는다.
- S6E8 적용:
  - 일치: "정확한 값이 같은 행들을 묶는 것"이 점수의 원천이라는 S6E8 결론(정확값 타깃 인코딩 +0.0032)과 같은 메커니즘이다.
    S6E8은 중복 행이 0개지만, 값이 정해진 눈금 위에만 있으므로 같은 값의 행들이 사실상의 복제 집단 역할을 한다.
  - 일치: 인코딩의 효과에 대한 설명은 "트리가 스스로 만들 수 있는 경계를 다시 쓰는 피처는 실패한다"는 S6E8 결론의 이론적 근거를 보강한다.
    TE가 통하는 이유는 트리가 못 하는 일(고카디널리티 값 묶음의 효율적 분할)을 대신해 주기 때문이다.

### 3. S5E2 솔루션: 1위, 단일 모델과 인코딩 레퍼토리

- 링크: [1st Place - Single Model - Feature Engineering](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/565539)
- 핵심 주장:
  - 무작위 타깃 원본에서 나온 400만 행 대회를 500피처 단일 XGBoost로 우승했다.
  - 인코딩 레퍼토리: groupby(COL1)[COL2].agg(STAT)의 다양한 통계량, 그룹별 타깃 히스토그램 구간 빈도, 그룹별 분위수, 여러 자릿수로 반올림한 컬럼(round7~9), float 자릿수 추출, 범주형 컬럼 쌍 결합, 파생 컬럼끼리의 나눗셈.
  - 전 컬럼의 NaN 여부를 2진수 한 컬럼으로 합친 피처("All NaNs as Single Base-2 Column")가 강력했고, 이를 다시 groupby 재료로 썼다.
  - 원본 데이터셋을 "제조사 권장가"로 보고, 키 컬럼으로 원본의 타깃 평균을 병합하는 피처를 만들었다(원본 데이터를 새 컬럼으로 쓰는 방식).
  - 타깃이 개입하는 인코딩은 폴드 중첩으로 누수를 막는다.
- S6E8 적용:
  - 일치: "수치 컬럼을 범주형으로 취급해 TE"는 S6E8의 정확값(문자열화) 타깃 인코딩과 같은 기법이다.
  - 적용 후보(미실험): S6E8은 단일 컬럼 정확값 TE까지만 검증했다.
    Deotte의 다음 단계는 컬럼 쌍/삼중 결합의 정확값 TE다(항목 8 참조).
  - 상충(조건부): NaN 패턴 base-2 피처는 S6E8 금지 목록(결측 피처 배제)과 정면으로 부딪힌다.
    차이는 데이터 성질이다: S5E2는 결측이 생성기 복제 구조의 일부라 신호였고, S6E8은 결측이 합성 후 무작위 삭제라 타깃과 독립임이 ablation으로 확인됐다.
    S6E8에서는 기존 결론(배제)을 유지한다.
  - 겹침 주의: 반올림 구간/자릿수 추출 피처는 S6E8의 _decimals 피처 논쟁과 같은 계열이다.
    S6E8 정밀 재측정에서 잔차 피처가 있으면 한계 기여가 거의 0이었으므로, 넣더라도 겹침 측정이 선행돼야 한다.

### 4. S4E12 솔루션: 1위, TE/CE 조합 무차별 탐색

- 링크: [1st Place - Single Model - Feature Engineering](https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328)
- 핵심 주장:
  - 611피처 단일 XGBoost로 우승했으며, 비결은 컬럼 조합의 무차별 탐색이다.
    수 일 동안 GPU로 수천 개의 무작위 컬럼 조합을 만들어 TE/CE를 적용하고 CV가 오르는 조합만 저장해 170개의 강한 조합을 찾았다.
  - 한 범주형 컬럼에 7가지 표현(라벨 인코딩, TE mean/median/min/max/nunique, CE)을 동시에 주면 GBDT가 컬럼을 이해하는 경로가 늘어난다.
  - 과적합 방어 논리: 120만 행이면 OOF 전체가 우연히 좋아지기 어렵다.
    데이터가 적으면 2~3개 조합까지만 시도해야 한다.
  - 낮은 학습률(0.01 이하) + 큰 n_estimators + 10-fold TE가 최종 런 설정이다.
  - 코멘트에서: TE의 결측 대치에 폴드별 평균 대신 train 전체 평균을 쓰는 것은 약간 누수지만, NaN이 폴드마다 다른 값으로 갈라지지 않아 CV/LB가 오히려 올랐다고 인정했다.
- S6E8 적용:
  - 일치: 낮은 학습률 + early stopping + 10-fold TE는 S6E8 파이프라인 기본값(고용량 + 최종만 학습률 하향, 10-fold 평균표)과 같다.
  - 적용 후보(미실험): "CV가 오르는 조합만 자동 수집"하는 조합 탐색 루프는 S6E8 실험 체계(champion/challenger)에 그대로 이식 가능한 절차다.
    S6E8은 91만 행이라 그의 과적합 방어 논리(대용량 OOF는 우연히 안 오른다)도 성립한다.
  - 적용 후보(미실험): 정확값 컬럼에 TE mean 외 CE(빈도), TE nunique 같은 추가 표현을 주는 것은 아직 S6E8에서 검증되지 않았다.

### 5. S5E3 솔루션 writeup: 2위, 작은 데이터의 규율

- 링크: [2nd Place - GBDT + NN + SVR + Original Data](https://www.kaggle.com/competitions/playground-series-s5e3/writeups/chris-deotte-2nd-place-gbdt-nn-svr-original-data)
- 핵심 주장:
  - 데이터가 작으면(원본 366행을 증강한 2,190행) 피처 엔지니어링을 하지 않고, 다양한 모델의 균등 평균만 쓴다.
    증강으로 행이 늘어도 신호량은 원본 행 수만큼이며, 컬럼/행 비율이 포화되면 차원의 저주로 가짜 CV 개선이 생긴다.
  - 원본 데이터는 새 행(concat)으로도, 새 컬럼(키 컬럼으로 원본 타깃 평균 병합)으로도 쓸 수 있고, 이 대회에선 새 컬럼 방식의 RAPIDS SVC 단일 모델이 2위 점수였다.
  - 흔들림(shakeup)의 크기는 train/test 행 수가 결정한다.
    행이 적으면 public LB를 무시하고 균등 평균으로 보수적으로 가고, 행이 많으면 과적합 걱정 없이 복잡한 스태킹까지 간다.
  - 실험 진행의 첫 원칙: 로컬 검증 체계를 만들고, 과적합 난이도를 평가하고, 그에 맞는 행동 범위를 정한다.
- S6E8 적용:
  - 일치: S6E8은 91만 행 대용량이므로 그의 분류상 "공격적 피처 엔지니어링 + 스태킹" 구간이고, 이는 S6E8 상위권이 스태킹으로 0.970+를 낸 관측과 일치한다.
  - 일치: 로컬 검증 우선 원칙은 S6E8 실행 요약(모든 판단은 OOF 기준)과 같다.
  - 참고: 원본 데이터를 새 컬럼으로 쓰는 방식은 S6E8에서 미실험이다(항목 9에서 종합).

### 6. S5E4 솔루션: 1위, 스태킹이 가중 평균을 이기는 조건

- 링크: [1st Place - RAPIDS cuML Stack - 3 Levels!](https://www.kaggle.com/competitions/playground-series-s5e4/discussion/575784)
- 핵심 주장:
  - 75개 level 1 모델(GBDT, NN, Lasso, SVR, KNN, RF, TabPFN, AutoML) 위에 XGB와 NN의 level 2를 얹고 level 3에서 50/50 평균한 3층 스태킹으로 우승했다.
  - 스태킹(비선형 level 2)이 가중 평균(선형 level 2)을 이긴 이유: 가장 강한 피처(전체 신호의 90%)가 11.6% 행에서 결측이라, "결측일 때 잘 맞히는 모델"과 "값이 있을 때 잘 맞히는 모델"을 행마다 갈아타야 했다.
    선형 결합은 이 갈아타기를 못 한다.
    사후 비교: 같은 73개 모델로 가중 평균은 private 11.503, 스태킹은 11.448이었다.
  - 스태킹의 전제 조건: 모든 모델이 같은 K-fold 분할을 쓰고, TE/의사 라벨의 누수를 전부 제거해야 한다.
  - 다양성 생성법: 피처 세트를 모델마다 다르게, 트리 깊이를 극단적으로 다르게(depth 5~20, 또는 leaves 1024), 타깃을 재정의(비율 예측, 결측 컬럼 예측 후 대입, 잔차 예측)한다.
- S6E8 적용:
  - 일치: OOF 저장 + 스태킹이 상위권 필수라는 S6E8 결론([733023](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733023))과 정확히 같다.
  - 적용 후보(미실험): S6E8도 12개 전 컬럼에 16~19% 결측이 있다.
    "결측 구간별 전문가 모델을 비선형 level 2가 갈아타게 한다"는 그의 논리가 성립할 수 있는 구조이므로, 스태킹 단계에서 level 2를 선형(순위 평균)으로 한정하지 말고 비선형(XGB/NN) level 2를 도전자로 시험할 근거가 된다.
  - 일치: 같은 K-fold 분할 공유와 누수 제거는 S6E8 검증 위생 결론과 같다.

### 7. S5E5 솔루션: 1위, 가중치 탐색 앙상블과 마지막 주간 레퍼토리

- 링크: [1st Place - GPU Hill Climbing!](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611)
- 핵심 주장:
  - 수백 개 후보 모델에서 시작해, 가장 강한 모델에 다른 모델을 하나씩 얹으며 OOF 점수가 오르는 조합만 남기는 탐욕적 가중치 탐색(hill climbing)으로 7개 모델을 자동 선택했다.
  - 단독 CV가 나쁜 모델(TE 피처 XGB, CV 0.0605~0.0608)이 최종 앙상블 가중치의 25%를 차지했다.
    다양성이 단일 모델 CV보다 중요하다는 실증이다.
  - 다양성 재료: log1p 파생 + 전 컬럼 쌍의 곱/나눗셈/합/차 피처 XGB, 구간화 + groupby z-score 피처 CatBoost, 선형 회귀 잔차 위 NN, NN 잔차 위 XGB.
  - 상수 이득 마무리: 5-fold OOF로 가중치를 정한 뒤 전 모델을 100% 데이터로 재학습하고(반복 수는 early stopping 평균의 1/(K-1)만큼 증가), 시드를 바꿔 K벌 학습해 평균한다.
- S6E8 적용:
  - 일치: 시드 앙상블의 실질 효과는 S6E8 결론(순위 ±60계단 노이즈 축소)과 같다.
  - 적용 후보(미실험): 100% 데이터 재학습 + 반복 수 1.25배 규칙은 S6E8 문서에 없는 마무리 기법이다.
    AUC 대회에서도 통용되는 그의 상수 레퍼토리이므로 마일스톤 제출 직전 도전자로 시험할 가치가 있다.
  - 긴장(판정 유보): S6E8 기본값은 순위 평균 블렌딩인데, Deotte 기본값은 OOF 기반 가중치 탐색이다.
    S6E8 순위 평균 근거는 모델 간 확률 눈금 차이 제거이고, Deotte 방식은 눈금 문제를 OOF 점수 최적화로 우회한다.
    91만 행 OOF면 가중치 과적합 위험이 낮으므로, 순위 평균과 가중치 탐색을 둘 다 재서 비교하는 것이 맞다.
  - 주의: 곱/합/차 파생 피처는 S6E8에서 비율/선형 결합 피처 무익 판정과 충돌하므로 가져오지 않는다.
    S5E5는 연속값 회귀 데이터라 성립했던 기법이다.

### 8. S5E6 솔루션 writeup: 1위, 결합 컬럼 타깃 인코딩과 두 신호 원칙

- 링크: [1st Place - Fast GPU Experimentation with RAPIDS cuDF cuML](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi)
- 핵심 주장:
  - 8개 피처 전부를 범주형으로 취급하고, 쌍(28) + 삼중(56) + 사중(70) 결합 컬럼을 만들어 7개 이진 타깃 각각에 TE를 적용, train 기준과 원본 데이터 기준으로 각각 인코딩해 2,268 컬럼의 XGBoost를 학습했다.
    이 "괴물 모델"이 공개 노트북 앙상블을 CV 0.384에서 0.386으로 끌어올린 비결이었다.
  - 코멘트에서 두 신호 원칙을 명시했다: 모든 플레이그라운드에는 원본 신호와 생성기 신호가 있고, 원본 타깃이 무작위면 결합 컬럼 + 원본 병합으로 "생성기를 역공학"하는 데 집중한다.
  - 원본 데이터를 쓰는 두 방법(새 행/새 컬럼) 중, 최종 앙상블에는 둘 다로 학습한 모델이 다 필요하다.
  - 순위 기반 지표(MAP@3)는 확률의 미세한 흔들림에 민감하므로, 같은 모델을 시드만 바꿔 수십~100번 학습해 확률을 평균하는 것만으로 점수가 올랐다(fold당 0.376이 100회 평균으로 0.380).
  - 최종 제출은 "best CV 앙상블"이었고 그것이 best LB이기도 했다.
- S6E8 적용:
  - 적용 후보(최우선, 미실험): S6E8의 정확값 TE는 단일 컬럼까지만 검증됐다.
    Deotte의 핵심 확장인 "정확값 컬럼 쌍/삼중 결합 후 TE"는 S6E8에서 아무도 보고하지 않은 기법이다.
    S6E8에서 gaming_hours 등이 단독으론 무익하지만 강한 컬럼과의 조합으로 +0.0038을 기여한다는 관측([732223](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732223) 코멘트)이 있어, 결합 TE가 컬럼 간 산술 관계를 잡아낼 개연성이 있다.
    91만 행이므로 그의 대용량 전제도 충족한다.
  - 일치: AUC도 순위 기반 지표이므로 시드 확률 평균의 논리가 그대로 적용되며, 이는 S6E8 시드 앙상블 결론과 일치한다.
  - 일치: 최종 선택 CV 기준("Trust Your CV")은 S6E8 실행 요약과 같다.

### 9. S5E11 디스커션: 원본 데이터 위 잔차 부스팅 제안 (AUC 이진 분류)

- 링크: [Here's an Idea - Boosting Over Residuals](https://www.kaggle.com/competitions/playground-series-s5e11/discussion/614986)
- 핵심 주장:
  - 1단계 모델을 원본 데이터로만 학습하고, 그 예측을 로짓으로 바꿔 XGBoost의 base_margin(시작 예측값)으로 넣으면, 2단계 모델은 생성기 신호만 학습하면 된다.
    분류에서도 로짓 변환으로 성립한다.
  - 원본이 코드로 생성된 데이터면 원본에는 최적해(생성 공식)가 존재하므로, 1단계는 그 최적해에 최대한 접근시키는 것이 목표다.
  - 단일 XGB를 그냥 학습하는 것과 다른 최소값에 도달하므로, 개선이 없어도 다양성 재료로 가치가 있다.
  - 실증 코멘트: 한 참가자가 이 방식으로 LB +0.00068을 보고했고, Deotte는 원본에 강한 1단계(TabPFN, 원본 AUC 0.8985)일수록 좋다고 답했다.
- S6E8 적용:
  - 적용 후보(미실험): S6E8에서 실패로 확인된 것은 원본을 훈련 행으로 주입(50배 가중치)하는 방식뿐이다.
    원본의 임계값 규칙(원본 AUC 0.9888, 합성 위 0.835)을 1단계로 삼아 base_margin으로 넣고 LightGBM/XGBoost가 잔차를 학습하게 하는 방식은 아무도 보고하지 않았다.
    규칙을 그대로 쓰면 상한 0.835짜리 함정이지만, 잔차 부스팅은 2단계가 그 위에 쌓으므로 상한 논리가 적용되지 않는다.
  - 긴장(조건부): S6E8 결론 "원본은 검증 참고 용도만"과 부딪히는 것처럼 보이나, 그 결론의 실험 근거는 행 주입 실패였다.
    잔차 부스팅은 별개 경로이므로 도전자 실험으로 가치가 있고, 실패해도 다양성 재료(스태킹 level 1)로 남는다.
  - 주의: S6E8 원본 프록시는 7,500행뿐이고 원본 자체도 합성이므로, 1단계는 복잡한 모델보다 이미 알려진 임계값 규칙이 안전한 출발점이다.

### 10. S5E1 솔루션: 2위, 이종 모델 잔차 스태킹

- 링크: [2nd Place - Stacking Transformer and Linear Regression](https://www.kaggle.com/competitions/playground-series-s5e1/discussion/560549)
- 핵심 주장:
  - 선형 회귀가 잡는 패턴 위에, 트랜스포머를 잔차(정답 - 선형 예측)에 학습시켜 합산하는 이종 모델 잔차 스태킹으로 2위를 했다.
  - 시드 5개 모델의 중앙값 사용, 자기 예측을 다음 라운드 학습에 쓰는 의사 라벨링 2회 등 반복 기법을 썼다.
  - 예측 불가능한 미래 승수(외삽)가 순위를 결정하는 대회에서는 가정이 다른 제출 두 개로 위험을 분산했다.
- S6E8 적용:
  - 참고: 잔차 스태킹의 원형 사례로, 항목 9의 방법론적 배경이다.
  - 참고: S6E8은 외삽 문제가 없으므로(train/test 분포 동일 확인) 최종 2제출 위험 분산 논리는 "CV 최고 + 가장 다른 구성의 차선" 정도로만 가져온다.

### 11. NVIDIA 블로그: 그랜드마스터의 7가지 정석 (Deotte 공저)

- 링크: [The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- 핵심 주장:
  - 7가지 정석: train/test 분포 비교 중심의 EDA, 다양한 계열의 베이스라인 동시 구축, 대규모 피처 엔지니어링(컬럼 결합), 탐욕적 가중치 탐색 앙상블, 스태킹(잔차 학습 또는 예측을 피처로), 의사 라벨링, 마무리 학습(시드 평균 + 100% 재학습).
  - CV 설계는 데이터 구조에 맞춘다: 표준 데이터는 K-fold, 시계열은 TimeSeriesSplit, 그룹 구조는 GroupKFold.
- S6E8 적용:
  - 일치: train/test 분포 비교를 EDA의 첫 단계로 두는 것은 S6E8의 adversarial validation 결론(값 분포 동일, 결측률만 차이)과 같은 문제의식이다.
  - 일치: S6E8은 그룹/시계열 구조가 없으므로 Stratified K-Fold 선택이 이 지침과 부합한다.
  - 참고: 다양한 계열의 베이스라인(선형, GBDT, NN)을 초기에 함께 세우라는 지침은 S6E8의 one-hot 로지스틱 회귀 0.96 관측과 연결되며, 스태킹 재료 확보 차원에서 NN 베이스라인 추가의 근거가 된다.

### 12. NVIDIA 블로그 2편: 본인 솔루션 해설 (S5E2, S5E4)

- 링크: [Feature Engineering Using cuDF pandas](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/), [Stacking Using cuML](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)
- 핵심 주장:
  - S5E2 해설: 10,000개 이상의 피처를 생성/시험해 최고 500개를 남겼고, 타깃이 개입하는 인코딩은 중첩 교차 검증으로 누수를 막았다.
    실험 속도가 곧 탐색량이고, 탐색량이 곧 점수라는 것이 핵심 교훈이다.
  - S5E4 해설: 500개 실험 후보 중 75개 level 1 모델 선별, level 2 입력에 예측 신뢰도/합의도 같은 메타 피처 추가, OOF 예측의 전진 선택(forward selection)으로 level 2 입력을 골랐다.
- S6E8 적용:
  - 일치: "실험 속도가 점수"라는 교훈은 S6E8 운영 체계(빠른 개선 판정 루프)의 방향과 같다.
  - 적용 후보(미실험): 스태킹 level 2 입력에 OOF 예측 외에 예측 합의도(모델 간 표준편차 등) 메타 피처를 추가하는 것은 S6E8 스태킹 설계에 가져올 수 있는 세부 기법이다.

### 13. 보조 관측: S6E3 원본 데이터셋 실종 스레드

- 링크: [Original dataset gone missing again](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679654) (작성자는 Optimistix, Deotte는 원본 사본 공유로 등장)
- 핵심 주장: 시즌 6 들어 원본 데이터셋이 반복적으로 실종되고, Kaggle은 사본을 만들어 주지 않으므로 참가자가 월초에 원본을 직접 확보해 둬야 한다.
- S6E8 적용: S6E8도 원본 실종 상태에서 커뮤니티 사본(jayjoshi37 등)으로 프록시를 확보한 상황이라, 같은 패턴의 재연이다.
  사본의 df.describe() 일치 확인은 이미 돼 있다.

## S6E8 적용 후보 우선순위

discussion-insights.md의 기존 결론과 겹치지 않는, Deotte 코퍼스 고유의 실험 후보다.

1. 정확값 컬럼 결합(쌍/삼중) 타깃 인코딩 (항목 8).
   현재 champion의 단일 컬럼 정확값 TE를 결합 컬럼으로 확장한다.
   S4E12의 "CV가 오르는 조합만 수집" 루프(항목 4)를 절차로 쓴다.
2. 원본 규칙 기반 잔차 부스팅 (항목 9).
   임계값 규칙의 로짓을 base_margin으로 넣은 LightGBM/XGBoost 도전자를 만든다.
   개선이 없어도 스태킹 다양성 재료로 남긴다.
3. 비선형 level 2 스태킹 (항목 6).
   결측 구간별 전문가 갈아타기 논리가 S6E8 결측 구조에 성립하는지 확인한다.
4. 100% 재학습 + 반복 수 1/(K-1) 증가 규칙 (항목 7).
   마일스톤 제출 직전의 상수 이득 후보다.
5. 정확값 컬럼의 추가 표현: CE(빈도), TE nunique (항목 4).
6. 순위 평균 대 OOF 가중치 탐색 비교 (항목 7).
   블렌딩 기본값을 데이터로 재확정한다.
7. 테스트 데이터 의사 라벨링 (항목 7, 11).
   마지막 주간 후보이며, AUC 지표에서의 이득 여부는 자체 측정이 필요하다.

## discussion-insights.md 결론과의 대조표

| S6E8 확립 결론 | Deotte 코퍼스 판정 | 비고 |
| --- | --- | --- |
| NaN 대치 없이 트리 모델에 투입 | 일치 (기본 관행) | 단 S4E12 코멘트는 TE 결측 대치에서 의도적 소량 누수가 이득이었던 사례도 언급 |
| 결측 플래그 금지 | 조건부 상충 | S5E2에선 NaN base-2 피처가 강력했으나 결측이 생성 구조의 일부였던 경우. S6E8은 ablation으로 무익 확인이 이미 끝났으므로 기존 결론 유지 |
| 정확값(문자열화) 타깃 인코딩 최우선 | 강한 일치 + 확장 제시 | 수치의 범주형 취급 + TE는 그의 핵심 기법. 결합 컬럼 TE로 확장 여지 |
| 예산 제약 잔차 피처 유효 | 일치 (원리 차원) | 생성기 산술의 역공학이 점수가 된다는 그의 원칙과 동일 |
| 비율/구간화 피처 무익 | 대체로 일치 | 그는 데이터에 따라 곱/비율 피처를 쓰지만 항상 CV 검증 후 채택. S6E8 ablation 결과가 우선 |
| monotone 제약 금지 | 무접점 (상충 없음) | 읽은 범위에서 monotone 제약을 권한 글 없음 |
| 고용량 GBM + early stopping | 일치 | 낮은 학습률 + 큰 반복 수 + early stopping. 깊이를 달리한 복수 모델로 다양성 확보까지 권장 |
| 시드 앙상블 + 순위 평균 | 시드 앙상블 강한 일치, 순위 평균은 긴장 | 그의 기본값은 OOF 가중치 탐색. 양쪽 비교 실험 권장 |
| OOF 저장 후 스태킹 | 강한 일치 | 같은 K-fold 공유, 누수 제거, 비선형 level 2까지 구체화 |
| 최종 선택은 CV 기준 | 일치 | "Trust Your CV" 명시 |
| 원본 데이터 훈련 미사용 | 조건부 상충 | 행 주입 실패는 인정하되, 새 컬럼 방식과 잔차 부스팅이라는 미실험 경로 둘을 제시 |

## 읽은 자료 장부

| 자료 | 유형 | 대회/출처 |
| --- | --- | --- |
| [604028](https://www.kaggle.com/competitions/playground-series-s5e9/discussion/604028) | 디스커션 | S5E9 |
| [564056](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/564056) | 디스커션 | S5E2 |
| [565539](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/565539) | 솔루션 | S5E2 1위 |
| [554328](https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328) | 솔루션 | S4E12 1위 |
| [S5E3 writeup](https://www.kaggle.com/competitions/playground-series-s5e3/writeups/chris-deotte-2nd-place-gbdt-nn-svr-original-data) | 솔루션 | S5E3 2위 |
| [575784](https://www.kaggle.com/competitions/playground-series-s5e4/discussion/575784) | 솔루션 | S5E4 1위 |
| [582611](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611) | 솔루션 | S5E5 1위 |
| [S5E6 writeup](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi) | 솔루션 | S5E6 1위 |
| [614986](https://www.kaggle.com/competitions/playground-series-s5e11/discussion/614986) | 디스커션 | S5E11 |
| [560549](https://www.kaggle.com/competitions/playground-series-s5e1/discussion/560549) | 솔루션 | S5E1 2위 |
| [Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) | 블로그 | NVIDIA (공저) |
| [cuDF FE 블로그](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/) | 블로그 | NVIDIA |
| [cuML Stacking 블로그](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/) | 블로그 | NVIDIA |
| [679654](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679654) | 디스커션 (보조) | S6E3 |
