# adarsh1077 EDA부터 스태킹까지 노트북 검토

이 문서는 [리서치: adarsh1077 EDA부터 스태킹까지 노트북의 신규 실험 단서 확인](https://github.com/tmheo/predicting-smartphone-addiction/issues/157)의 근거다.
조사 시점은 2026-08-16이며, Kaggle이 공개한 최신 판본의 소스, 실행 기록, 출력 파일, 입력 자료 메타데이터와 이 저장소의 현재 판정 근거를 사용했다.

## 결론

[S6E8 Complete Walkthrough EDA to Stacking 판본 3](https://www.kaggle.com/code/adarsh1077/s6e8-complete-walkthrough-eda-to-stacking/versions/3)은 교육용 설명과 완주한 실행 기록은 갖췄지만, 새 실험 티켓을 열 근거는 제공하지 않는다.
결정은 새 실험에 대한 기각과 기존 P4 티켓으로의 조건부 흡수다.
외부 OOF와 제출 예측 177개, 이들이 만든 최종 제출 파일과 보고 Public 점수는 채택하지 않는다.
배열 계보와 정확 중복 검사, fold 일치 확인은 이미 열린 [OOF 후보 풀의 품질과 다양성 진입 기준 점검](https://github.com/tmheo/predicting-smartphone-addiction/issues/63)에 흡수한다.
outer 학습 부분 안에서만 맞춘 rank-gauss 표현, 표준화와 수렴 확인은 이미 열린 [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)에 들어 있다.
두 티켓의 질문은 이번 조사 전에 이미 이 검사를 명시했으므로 본문을 바꾸거나 새 티켓을 만들 필요가 없다.

노트북의 최종 `0.970094`는 이 저장소의 ADR 0001이 정의한 nested OOF로 인정할 수 없다.
전체 OOF 라벨로 구성원 AUC를 계산해 2단 모델 평가 전에 구성원을 고르고, rank-gauss를 outer 분할 전에 전체 OOF에서 맞추며, 외부 배열의 fold와 생성 이력을 검증하지 않기 때문이다.
입력 중에는 자체 문서가 정직한 end-to-end nested 평가가 아니며 낙관적일 수 있다고 명시한 2단 예측도 포함된다.
노트북은 이 2단 예측과 기존 결합 예측을 일반 구성원처럼 다시 넣으므로 선택 편향과 단계 중첩을 해소하지 못한다.

EDA와 단일 모델 절에서 재사용 가능한 내용도 모두 기존 결정과 겹친다.
원시 NaN 유지, 화면 시간 잔차와 조성, 소수 첫째 자리, exact-value 목표 인코딩, LightGBM, XGBoost, CatBoost, 정확값 선형 모델과 다양성 기여는 이미 자체 공통 fold와 플라시보 규약으로 실험했다.
결측 표시와 일반 결측 개수는 이 노트북 실행에서도 직전 단계 대비 `-0.000001`이었고 지도에서 배제한 항목이다.
새로 보이는 일반 비율과 차이 묶음은 개별 제거 실험이 없으며, 저장소에서는 같은 계열의 여러 열을 플라시보 중요도와 한계 OOF로 이미 걸렀다.

## 확인한 판본과 내용 해시

Kaggle `GetKernel` API가 돌려준 현재 판본 번호는 3이고, 공개 판본의 `scriptVersionId`는 `342715768`이다.
Kaggle 공개 페이지, 목록 API와 출력 파일 시각은 최신 공개 실행을 2026-08-16 07:36:57.860 UTC로 표시한다.
판본 1, 2, 3의 `scriptVersionId`는 각각 `342704689`, `342710877`, `342715768`이며 판본 4 페이지는 존재하지 않았다.
`GetKernel` 응답의 `lastRunTime`은 조회할 때마다 현재 시각에 가깝게 달라져 판본 시각 근거로 쓰지 않았고, 고정된 공개 페이지 시각과 출력 파일 생성 시각을 사용했다.

내려받은 최신 소스는 52개 셀로 이뤄졌고, 그중 코드 셀은 28개다.
모든 코드 셀의 `execution_count`는 `null`이고 저장된 셀 출력은 비어 있다.
별도 [공개 출력](https://www.kaggle.com/code/adarsh1077/s6e8-complete-walkthrough-eda-to-stacking/output)에는 실행 기록과 296,302행 `submission.csv`가 남아 있다.
실행 기록은 마지막 예측 작성까지 약 1,531초, 변환 작업까지 약 1,540초를 기록하므로 본문의 CPU 20분 주장은 실제 공개 실행보다 약 5분 짧다.

- 소스 `.ipynb`는 44,672바이트이며 SHA-256은 `c2c00b5b230a3abfe2c99ff9417a776504e59f0aaa2d4ff6986410bcec6b1d8e`다.
- `kernel-metadata.json`은 1,629바이트이며 SHA-256은 `57d06d169fed7fa01122278c916f9cfd9b3991b98c5b50f43d2eda42c193f4e4`다.
- 공개 실행 기록은 27,517바이트이며 SHA-256은 `08f473ed0881e04ce882f338634585096a97ec6d45cd02598aad2c415707a523`다.
- 공개 `submission.csv`는 7,783,805바이트이며 SHA-256은 `e934690a46c42b2bca21c4fd693bf8475082496b18a9821a103a4d1f9cf8ee54`다.

실행 기록은 코드가 출력하도록 한 표와 점수를 같은 순서로 남기고, 제출 파일 첫 행의 예측도 기록의 `0.793267`과 원정밀도에서 일치한다.
따라서 최신 소스와 실행 산출물이 전혀 다른 프로그램에서 나온 정황은 없다.
다만 비어 있는 셀 출력, 고정하지 않은 입력 판본과 패키지 버전 때문에 내려받은 `.ipynb` 하나만으로 같은 결과를 재현할 수는 없다.

## 실행 환경, 입력과 사용 조건

노트북 메타데이터는 CPU 실행, GPU와 TPU 비활성, 인터넷 비활성, Python 노트북과 `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9` 이미지를 선언한다.
노트북 안의 언어 메타데이터는 Python 3.11이지만 실행 기록 경로는 Python 3.12 패키지 경로를 보여 주므로 두 메타데이터가 일치하지 않는다.
소스는 `numpy`, `pandas`, `matplotlib`, `lightgbm`, `scikit-learn`, `xgboost`, `catboost`와 `scipy`를 가져온다.
별도 설치 셀, 패키지 버전 출력, 잠금 파일과 모델 직렬화 파일은 없다.
따라서 Docker 이미지 내용 해시는 고정됐지만, 문서만 보고 개별 패키지 버전과 실행 환경을 재구성할 수는 없다.

노트북은 대회 자료와 외부 Kaggle 자료 18개를 입력으로 선언한다.
자료 참조에는 판본 번호나 파일 SHA-256이 없으므로 같은 slug가 갱신되면 같은 소스가 다른 배열을 읽을 수 있다.
실제로 `adarsh1077/s6e8-adarsh-oof-library`의 현재 파일 생성 시각은 2026-08-15이고 제목은 11개 모델이라고 쓰지만 파일 목록에는 OOF와 시험 예측 쌍 22개가 있어 자료 제목과 내용도 함께 고정해야 한다.

Kaggle 자료 메타데이터에 표시된 사용 조건은 다음과 같다.

- CC0-1.0은 [adarsh1077의 OOF 자료](https://www.kaggle.com/datasets/adarsh1077/s6e8-adarsh-oof-library), [XGBoost identity-digit](https://www.kaggle.com/datasets/beicicc/s6e8-fixed1500-xgb-identity-digit-artifacts), [XGBoost screen-relation](https://www.kaggle.com/datasets/beicicc/s6e8-fixed1500-xgb-screen-relation-artifacts), [szymonkapiski OOF 자료](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models), [dariushafshar OOF 자료](https://www.kaggle.com/datasets/dariushafshar/s6e8-golem-oof-library), [boltuzamaki OOF 자료](https://www.kaggle.com/datasets/boltuzamaki/s6e8-oof-prediction-library), [CatBoost MLP OOF](https://www.kaggle.com/datasets/mohankrishnathalla/s6e8-cat-mlp-oof), [LightGBM DART OOF](https://www.kaggle.com/datasets/mohankrishnathalla/s6e8-lgb-dart-oof)와 [XGBoost OOF](https://www.kaggle.com/datasets/mohankrishnathalla/s6e8-xgb-oof)에 표시돼 있다.
- CC BY 4.0은 [RealMLP 자료](https://www.kaggle.com/datasets/beicicc/s6e8-fixed4-realmlp-two-seed-artifacts), [LightGBM identity-digit 자료](https://www.kaggle.com/datasets/beicicc/s6e8-fixed900-identity-digit-lightgbm-artifacts), [exact-value CatBoost 자료](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts), [Lookup Transformer 자료](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts)와 [두 번째 시드 Lookup 자료](https://www.kaggle.com/datasets/beicicc/s6e8-second-seed-fixed-schedule-lookup-artifacts)에 표시돼 있다.
- Apache License 2.0은 [factorization-machine lattice 자료](https://www.kaggle.com/datasets/raykkretzschmar/s6e8-fm-lattice-blend-members)에 표시돼 있다.
- `other`는 [structural LightGBM 자료](https://www.kaggle.com/datasets/beicicc/s6e8-fixed900-structural-lgbm-artifacts)와 [여섯 구성원 2단 예측 자료](https://www.kaggle.com/datasets/beicicc/s6e8-sixmember-crossfit-logitlr-artifacts)에 표시돼 있다.
- `unknown`은 [najiama 결합 예측 자료](https://www.kaggle.com/datasets/najiama/predicting-smartphone-addiction-oof-submission-csv)에 표시돼 있고, 파일 목록에는 별도 README나 사용 조건 파일이 없다.

`other`로 표시된 두 자료의 README는 산출물이 대회 규칙과 자료 조건을 따른다고만 쓰고 독립적인 재배포 허가를 명시하지 않는다.
`unknown` 자료도 별도 허가를 확인할 수 없으므로 세 자료의 배열을 저장소 산출물에 포함하거나 재배포해서는 안 된다.
CC BY 4.0 자료는 출처 표시 의무를 지켜야 하고, 대회 자료와 대회 자료에서 파생한 예측은 [대회 규칙](https://www.kaggle.com/competitions/playground-series-s6e8/rules)의 사용 범위를 별도로 따른다.

Kaggle 공개 노트북 소스는 공개 시 Apache License 2.0이 적용되므로 조건과 고지를 지키면 코드를 참고하거나 수정할 수 있다.
코드를 복사해 배포한다면 Apache License 2.0 원문, 원래 고지, 변경 사실과 NOTICE가 있을 때의 관련 고지를 보존해야 한다.
이 허가는 노트북이 읽는 외부 OOF, 대회 자료와 출력 파일의 사용 조건을 대신하지 않는다.
이번 결정은 코드를 복사하지 않고 방법을 자체 구현한 기존 저장소 코드와 티켓으로만 흡수한다.

## EDA와 전처리 경로 재구성

노트북은 `train.csv`, `test.csv`와 `sample_submission.csv`를 읽고 식별자와 목표값을 제외한 12개 설명변수를 수치 9개와 범주 3개로 나눈다.
훈련 자료는 691,369행, 시험 자료는 296,302행이며 양성 비율은 `0.709424`다.
수치 요약, 결측률, 결측 여부별 양성률, 행별 결측 개수, 3 IQR 바깥 값, 완전 동일 설명변수 행, 단일 수치 열 AUC, 소수 첫째 자리별 양성률과 범주별 양성률을 차례로 출력한다.

탐색 서술에는 실행 출력과 맞지 않는 문장이 여럿 있다.
본문은 나이가 15세에서 60세 정도라고 하지만 실행 출력은 최솟값 18, 최댓값 35다.
본문은 훈련과 시험의 열별 결측률이 같다고 해 의도적인 마지막 결측 주입을 추론하지만, `social_media_hours`는 `19.38%` 대 `16.00%`, `daily_screen_time_hours`는 `13.86%` 대 `11.07%`, `gaming_hours`는 `18.34%` 대 `20.05%`로 차이가 보인다.
본문은 결측 개수가 늘수록 양성률이 단조롭게 변한다고 하지만 실행 출력은 6개 결측 `0.7192` 뒤 7개 결측 `0.6963`으로 내려가고 희소 구간도 오르내린다.
본문은 훈련 중복과 시험-훈련 완전키 겹침이 모두 0이라고 하지만 실행 출력은 훈련 중복 0과 시험 겹침 2행을 기록한다.
2행은 전체 시험의 `0.0007%`라 실용적 누출 통로는 아니지만, 출력과 반대인 본문 결론은 그대로 인용할 수 없다.

결측 표시의 목표값 차이도 크지 않다.
가장 큰 단일 열 차이는 `sleep_hours`와 `age`의 약 `0.0042`이고, 다수 열은 `0.001` 안팎이거나 0에 가깝다.
행별 결측 개수의 양성률도 큰 구간에서는 약한 차이만 보이며 희소 구간은 표본 오차가 크다.
이 출력은 저장소의 결측 표시와 일반 결측 개수 배제 결정을 뒤집지 않는다.

`daily_screen_time_hours`의 관측 행 단일 AUC는 `0.8896`이고, `weekend_screen_time`은 `0.8810`, `social_media_hours`는 `0.8578`이다.
첫째 소수 자릿값별 양성률은 `0.6296`부터 `0.7402`까지 달라 생성 지문 후보가 있음을 다시 보여 준다.
그러나 이 표는 강한 잔차와 정확값 표현 위의 한계 기여를 재지 않으므로 소수 자리 특성의 채택 근거는 아니다.

## 단일 모델과 특성 생성 경로 재구성

첫 LightGBM은 원시 수치의 NaN을 유지하고 세 범주 열을 pandas category로 바꾼 뒤 80:20 층화 단일 분할에서 AUC `0.961580`을 기록한다.
정식 비교 절은 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 한 번 만들어 모든 실험이 공유한다.
기준 LightGBM은 400개 나무, 학습률 `0.08`, 잎 63개, 최소 자식 표본 60개, 행 표본 비율 `0.9`, 열 표본 비율 `0.8`과 seed 0을 사용해 OOF `0.962644`를 기록한다.

비율 묶음은 화면 시간 잔차, social/gaming/work 비율, 주말 차이, 열기당 화면 시간, 열기당 알림, 깨어 있는 시간과 깨어 있는 시간 중 화면 비율의 9개 열을 한꺼번에 더한다.
이 묶음은 OOF를 `0.963503`으로 높여 기준 대비 `+0.000859`을 기록한다.
한꺼번에 9개 열을 바꿨기 때문에 어느 열이 효과를 냈는지와 이미 채택된 화면 시간 예산 잔차가 효과를 얼마나 설명하는지는 분리할 수 없다.

다음 단계는 수치 9개 열의 결측 표시와 전체 12개 열의 결측 개수를 더한다.
OOF는 `0.963502`라 기준 대비로는 `+0.000858`이지만 직전 비율 묶음 대비 `-0.000001`이다.
노트북 본문도 LightGBM이 원시 NaN을 이미 쓰므로 명시 결측 열은 아무 가치가 없었다고 해석한다.

다음 단계는 여섯 수치 열의 소수 첫째 자리를 더한다.
OOF는 `0.963573`이라 기준 대비 `+0.000930`, 직전 단계 대비 `+0.000072`다.
이 값은 단일 seed 누적 비교이고, 새 피처 플라시보 게이트와 3시드 확인을 거치지 않았으며 ADR 0001의 최종 개선 문턱 `+0.0001`보다 작다.

exact-value 목표 인코딩은 나이, 일일 화면 시간, 앱 열기, 알림, 주말 화면 시간과 수면 시간 6개 열에 적용한다.
각 outer 학습 부분을 다시 5개로 나눠 학습 행용 인코딩을 만들고, outer 검증에는 전체 outer 학습 부분의 표를 적용한다.
평활 계수는 50이며 인코딩을 더한 LightGBM OOF는 `0.966813`으로 기준 대비 `+0.004169`이다.

이 구현은 outer 검증 라벨을 직접 읽지 않아 전체 훈련으로 한 번 만든 누출판보다 훨씬 낫다.
그러나 inner 인코딩의 사전 평균 `gm`은 각 inner 학습 부분이 아니라 outer 학습 부분 전체의 라벨로 한 번 계산한다.
따라서 inner 검증 행의 라벨과 해당 행 자신의 라벨도 평활 사전 평균에 아주 작게 들어가며, 소스 주석의 어떤 행도 자기 라벨을 보지 않는다는 보장은 엄밀히 성립하지 않는다.
비교용 누출판 `0.967342`와 안전판 `0.966813`의 `0.000529` 차이는 전체 라벨 평균표 누출의 낙관성을 보여 주지만, 안전판의 미세한 사전 평균 누출까지 제거한 수치는 아니다.

현재 저장소는 fold-fit 목표 인코딩, outer 학습 부분 전용 평균표, 플라시보 카나리아와 고정 fold를 이미 구현한다.
따라서 이 코드를 복사하지 않고 저장소의 더 엄격한 구현을 유지한다.

## 네 모델 비교와 작은 2단 결합 재구성

노트북은 전체 훈련 행에서 seed 0으로 200,000행을 고르고 공통 5-fold와 교차시켜 모델 계열 비교용 OOF를 만든다.
LightGBM, XGBoost, CatBoost는 소수 자리까지 누적한 같은 특성을 쓰고, 로지스틱 회귀는 범주 열을 버린 뒤 각 fold 안에서 중앙값 대체와 표준화를 맞춘다.
기록된 OOF는 LightGBM `0.959259`, XGBoost `0.959483`, CatBoost `0.956135`, 로지스틱 회귀 `0.920058`이다.
부분 표본의 최선 단일 모델은 `0.959483`, 산술 평균은 `0.956170`, 순위 평균은 `0.955277`이다.
단독 성능 차이가 큰 로지스틱 회귀에 같은 가중치를 준 탓에 두 평균이 최선 단일 모델보다 낮다.

각 구성원의 200,000행 OOF를 전체 행에서 한 번 순위화하고 표준정규 분위수로 옮긴 뒤, 같은 공통 fold의 outer 학습 부분에서 StandardScaler와 L2 로지스틱 회귀를 맞춘다.
작은 2단 결합 OOF는 `0.960241`이고 전체 OOF로 다시 맞춘 계수는 LightGBM `2.038`, XGBoost `2.247`, CatBoost `0.456`, 로지스틱 회귀 `-0.381`이다.
이 결과는 약한 구성원도 다른 오차 방향을 제공하면 음의 보정 계수로 쓰일 수 있다는 예시다.

다만 2단 모델의 outer 학습 행에 들어가는 base OOF 중 일부는 현재 2단 평가 fold 라벨을 학습한 base 모델에서 나왔다.
각 행 자신의 라벨은 해당 행의 base OOF에서 빠지지만, end-to-end로 base 모델까지 outer fold 바깥에서 다시 만든 엄격한 nested 평가는 아니다.
저장소 ADR 0001은 자체 후보 풀의 고정 OOF를 결합 전략 비교에 쓰는 별도 계약을 채택했으므로, 이 교육용 수치를 그 계약의 채택 증거로 가져오지 않는다.

## 외부 OOF 2단 결합과 최종 예측 경로 재구성

외부 로더는 모든 입력 폴더의 `.npy`, `oof` 이름이 든 parquet와 `*_blend_oof_predictions.csv`를 탐색한다.
`.npy`는 파일 이름으로 OOF와 시험 예측 쌍을 찾고 길이만 검사하며, parquet와 CSV는 id가 있으면 대회 id 순서로 다시 맞춘다.
구성원 이름이 겹치면 Python 사전의 같은 키를 조용히 덮어쓰므로 어떤 자료가 남았는지에 대한 충돌 기록은 없다.
실행 기록은 최종 사전에서 외부 구성원 182개를 모았다고 출력한다.

OOF float64 바이트의 MD5가 같은 배열 두 개를 제거한다.
실행에서 확인된 중복은 `xgb_d7_alt2 == xgb_d7_alt1`과 `xgb_screen_relations_baseline103 == xgb_identity_digit_enhanced103`이다.
그 뒤 비유한 값, 단독 OOF AUC `0.90` 미만과 OOF-시험 KS 통계 `0.05` 초과를 제외하되 이름에 `perp`가 있으면 AUC와 KS 문턱을 면제한다.
실행은 `knn`, `rf`, `extratrees_support` 세 개를 KS로 제외하고 177개를 유지한다.

KS 검사는 각 OOF와 시험 예측을 각각 자기 배열 안의 백분위 순위로 먼저 바꾼 뒤 40,000개씩 뽑아 비교한다.
연속 예측은 이 변환 뒤 두 주변분포가 거의 균등해지므로 일반적인 OOF-시험 분포 이동 검사가 되지 않는다.
이 수치는 동점 구조와 지지집합 차이에 더 민감하므로 원시 예측이나 잘린 logit의 이동, 고유값 개수와 동점률을 함께 보지 않고 품질 문턱으로 채택해서는 안 된다.

177개 구성원 각각을 전체 OOF와 전체 시험 안에서 따로 rank-gauss로 바꾼다.
각 공통 fold를 한 번씩 제외하고 나머지 네 fold OOF로 StandardScaler와 `C=0.03`인 L2 로지스틱 회귀를 맞춰 제외 fold를 예측한다.
최대 반복 횟수는 5,000이고 실제 최대 반복은 202라 수렴 assertion을 통과했다.
기록된 전체 2단 결합 OOF는 `0.970094`다.

최종 시험 예측은 177개 전체 OOF와 전체 라벨로 표준화와 로지스틱 회귀를 다시 맞추고 시험 rank-gauss에 적용한 결정함수를 다시 백분위 순위로 바꾼 값이다.
이 값으로 296,302행 `submission.csv`를 쓴다.
가장 큰 양의 계수는 `foldsafe_te_wide 0.986`, `naji_18 0.794`, `catnative 0.761`이며, 가장 큰 음의 계수는 `pub_tabm -0.518`, `lat_lgbm_s5 -0.410`, `naji04 -0.332`다.

## outer-fold 정직성과 선택 편향 감사

첫째, 구성원 선별이 nested 평가 바깥에 있다.
코드는 182개 전체 OOF와 전체 라벨로 각 구성원의 단독 AUC를 계산해 `0.90` 문턱을 적용한 뒤 같은 OOF에서 2단 결합을 평가한다.
outer 평가 fold 라벨이 구성원 생존 여부를 정하므로 구성원 선택의 자유도가 최종 `0.970094`에 반영되지 않는다.

둘째, rank-gauss가 outer 분할 바깥에서 맞춰진다.
순위 변환은 라벨을 보지 않지만 outer 평가 fold의 예측 주변분포를 이용해 outer 학습 값의 좌표까지 정하는 전이식 변환이다.
이슈 64는 outer 학습 부분의 경험적 누적분포만 맞추고 outer 평가 부분에는 적용하도록 이미 규정한다.

셋째, 외부 `.npy`의 행과 fold 일치를 검증하지 않는다.
길이가 691,369와 296,302면 받아들이며, 일부 입력이 제공하는 `fold_id.npy`, manifest, 원본 id와 파일 해시를 읽지 않는다.
공통 seed 42라고 설명한 문장만으로는 각 배열이 같은 원본 행 순서, 같은 fold, fold 안 전처리와 학습 행 제외를 지켰는지 입증되지 않는다.

넷째, 외부 구성원 중에는 이미 2단 결합으로 만든 OOF가 있다.
[여섯 구성원 2단 예측 자료](https://www.kaggle.com/datasets/beicicc/s6e8-sixmember-crossfit-logitlr-artifacts)의 README는 각 행 자신의 base OOF는 그 행을 제외하지만 다른 2단 학습 행의 base 모델이 현재 2단 평가 fold 라벨을 쓸 수 있다고 명시한다.
그 README는 결과가 진단용이며 엄격한 2단 검증보다 낙관적일 수 있다고 직접 경고한다.
노트북은 `sixmember_meta_oof.npy`, `sixmember_equal_rank_oof.npy`와 `sixmember_meta_perp_oof.npy`를 일반 구성원처럼 다시 결합한다.

다섯째, najiama 자료에는 이미 `*_blend_oof_predictions.csv`라는 결합 예측이 여러 개 있다.
같은 작성자의 앞선 167-model 노트북은 이 기존 결합 예측을 포함한 수치가 낙관적이라고 경고하고 제외 수치도 냈지만, 이번 walkthrough는 해당 경고와 제외 계산을 삭제한 채 최종 177개에 포함한다.
따라서 새 노트북의 `nested CV of the full stack`이라는 문구는 앞선 노트북이 인정한 한계보다 더 강한 주장을 한다.

여섯째, 구성원 이름 충돌과 자료 판본이 고정되지 않는다.
같은 이름을 여러 자료가 제공하면 mount와 glob 순서에 따라 나중 배열이 조용히 남고, 입력 자료가 갱신되면 소스 변경 없이 풀 내용이 달라질 수 있다.
최종 submission 해시는 이번 실행을 고정하지만 그 예측의 모든 외부 입력 계보를 재구성하지는 못한다.

일곱째, 같은 OOF에서 방법을 반복 비교한 선택 이력이 실행 코드에 없다.
본문의 `C` 탐색, 79개에서 132개로 늘린 효과, greedy 방식, 35개 추가 생성 지문, 원본 자료 추가, 여섯 추가 모델 계열과 AUC 상한 계산은 최신 코드 셀이 실행하지 않는다.
따라서 이 값들은 최신 판본의 저장 출력으로 재현된 제거 실험이 아니라 작성자 보고치다.

## 같은 작성자의 167-model 노트북과 소스 수준 비교

이번 walkthrough의 외부 OOF 절은 이미 검토한 [S6E8 167 Models Diversity Beats Strength 판본 7](https://www.kaggle.com/code/adarsh1077/s6e8-167-models-diversity-beats-strength/versions/7)의 핵심 경로를 다시 싣는다.
기존 조사 문서 [167-model 앙상블과 단변량 spline Transformer 노트북 검토](s6e8-167-spline-notebooks.md)는 판본 7에서 182개 수집, 정확 중복 2개 제거, 3개 제외, 177개 유지와 nested rank-gauss 결합 `0.970093`을 확인했다.
새 실행도 같은 입력 18개에서 182개를 모으고 같은 두 중복과 같은 세 KS 제외를 거쳐 177개를 유지하며 `0.970094`를 출력한다.

작성자가 이후 제목을 바꿔 공개한 [S6E8 Diversity Beats Strength 판본 8](https://www.kaggle.com/code/adarsh1077/s6e8-diversity-beats-strength/versions/8)의 최신 소스와도 직접 줄 단위로 대조했다.
walkthrough의 46번 셀은 판본 8의 풀 로더에서 출처 기록만 제거하고 변수 이름을 바꾼 형태다.
walkthrough의 48번 셀은 판본 8의 정확 중복, AUC와 KS 격리 코드를 변수 이름과 출력 형식만 바꿔 유지한다.
walkthrough의 49번 셀은 판본 8의 전체 rank-gauss, fold별 StandardScaler, L2 로지스틱 회귀와 수렴 assertion을 함수에서 본문으로 펼친 형태다.
walkthrough의 50번 셀은 판본 8의 전체 OOF 재학습과 시험 순위 제출 경로를 같은 순서로 수행한다.

새로운 부분은 EDA, 누적 LightGBM 특성 비교, 목표 인코딩 누출 예시와 200,000행 네 모델 교육용 2단 결합이다.
최종 177개 결합 방법, 중복 검사, KS 문턱, rank-gauss, 수렴 검사와 제출 경로는 새로운 실험 단서가 아니다.
오히려 앞선 167-model 판본의 기존 결합 예측 제외 수치와 낙관성 경고를 걷어내 신뢰 범위 설명이 약해졌다.

## 현재 저장소와의 중복 및 적용 판단

현재 champion은 `exp067_lookup_xgb_impute_comps5`, run `2bd55026ae63430aa774bce20a359b4a`이고 3시드 평균본 OOF AUC는 `0.9690978395`다.
현재 후보 풀에는 고정 3시드 평균본 16개가 있으며, Lookup-Transformer 초기화 평균 `0.9691957618`, TabM `0.9683261182`, TabPFN-3 `0.9672432267`, 정확값 one-hot 로지스틱 회귀 `0.9596583964`처럼 강도와 오류 계열이 다른 자체 실행이 이미 들어 있다.
따라서 외부 노트북의 약한 선형 모델이 음의 계수로 기여할 수 있다는 관찰은 현재 후보 풀의 다양성 규약과 겹친다.

- 원시 NaN 유지와 결측 표시 무익은 기존 지도 결정과 이번 실행의 `-0.000001`이 일치하므로 현행 결정을 유지한다.
- 화면 시간 잔차와 조성 표현은 현재 champion의 `screen_slack`, `sgw_sum`, `slack_frac`, `sgw_frac`, 복원 조성 5열과 겹치므로 새 실험을 열지 않는다.
- 노트북의 `notif_per_open`을 포함한 일반 조성 묶음은 저장소 `exp054`에서 플라시보 미달 열을 분리했고 채택된 5열만 현재 champion에 남겼으므로 9열 묶음을 다시 열지 않는다.
- 소수 첫째 자리 6열은 기존 구현과 조사에 있고 강한 잔차 위 한계 기여가 거의 없다는 근거가 더 강하며, 이번 단일 seed `+0.000072`는 판정 문턱보다 낮으므로 새 실험을 열지 않는다.
- exact-value 목표 인코딩은 이미 여러 모델 계열과 누출 카나리아로 판정했으며, 이번 구현은 inner 사전 평균 누출까지 있어 대체 구현으로 채택하지 않는다.
- LightGBM, XGBoost, CatBoost와 정확값 선형 모델은 현재 후보 풀에서 같은 공통 fold와 3시드로 이미 측정했으므로 다시 실행하지 않는다.
- 배열 exact 중복, 계보, 정렬과 fold 일치는 이슈 63이 이미 직접 다루므로 새 티켓을 열지 않는다.
- rank-gauss, 순위와 logit 비교, outer 학습 부분 안의 표준화, 로지스틱 회귀 수렴 assertion은 이슈 64가 이미 직접 다루므로 새 티켓을 열지 않는다.
- 외부 OOF와 제출 파일은 지도에서 후보 풀과 최종 제출 사용이 명시적으로 범위 밖이므로 가져오지 않는다.
- Public `0.97113`은 공개 실행에 Kaggle 제출 채점 증거가 없고 Public 점수는 ADR 0001의 판정 근거가 아니므로 채택 판단에 쓰지 않는다.

## 최종 결정

새 진입 진단이나 실험 티켓은 만들지 않는다.
이 노트북의 최종 예측, 외부 OOF, 목표 인코딩 구현, KS 문턱과 EDA의 인과 해석은 기각한다.
배열 계보와 중복 검사는 이슈 63, outer 학습 부분 전용 rank-gauss와 수렴 검사는 이슈 64의 기존 질문으로 조건부 흡수한다.
두 티켓이 이미 필요한 조건을 갖췄으므로 추가 범위 변경도 하지 않는다.

적용할 때 지킬 조건은 자체 후보 풀의 3시드 평균본만 사용하고, 모든 구성원과 fold 파일의 커밋 및 SHA-256을 확인하며, 구성원 선택과 변환 학습을 각 outer 학습 부분 안에서만 수행하는 것이다.
결합 전략의 nested OOF가 champion보다 `+0.0001` 이상이어야 하며, 차이가 그보다 작으면 ADR 0001에 따라 자유도가 낮은 단순 전략을 선택한다.

## 1차 출처

- [S6E8 Complete Walkthrough EDA to Stacking 판본 3](https://www.kaggle.com/code/adarsh1077/s6e8-complete-walkthrough-eda-to-stacking/versions/3)은 최신 소스, 입력 선언과 본문 주장의 1차 출처다.
- [판본 3 공개 출력](https://www.kaggle.com/code/adarsh1077/s6e8-complete-walkthrough-eda-to-stacking/output)은 실행 기록과 제출 파일의 1차 출처다.
- [S6E8 167 Models Diversity Beats Strength 판본 7](https://www.kaggle.com/code/adarsh1077/s6e8-167-models-diversity-beats-strength/versions/7)은 이미 검토한 방법과 수치의 비교 출처다.
- [S6E8 Diversity Beats Strength 판본 8](https://www.kaggle.com/code/adarsh1077/s6e8-diversity-beats-strength/versions/8)은 현재 공개된 같은 코드 계열과 줄 단위 중복의 확인 출처다.
- [Kaggle Meta Kaggle Code](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)는 공개 노트북 소스의 Apache License 2.0 적용 근거다.
- [Apache License 2.0 원문](https://www.apache.org/licenses/LICENSE-2.0)은 코드 재사용 조건의 우선 근거다.
- [대회 규칙](https://www.kaggle.com/competitions/playground-series-s6e8/rules)은 대회 자료와 파생 산출물의 사용 범위를 정하는 근거다.
- [`artifacts/champion.yaml`](../../artifacts/champion.yaml)과 [`artifacts/pool.yaml`](../../artifacts/pool.yaml)은 현재 champion과 후보 풀의 기록 원본이다.
- [ADR 0001](../adr/0001-experiment-adoption-contract.md)은 특성, 다양성 구성원과 nested OOF 앙상블의 채택 판정 원본이다.
