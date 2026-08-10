# Playground Series S6E8 분석 대상 노트북 코드 분석: 26위부터 37위

## 조사 범위와 방법

이 문서는 [`code-notebook-inventory.md`](code-notebook-inventory.md)의 고정 목록 가운데 26위부터 37위까지 12개 분석 대상 노트북을 분석한다.
득표 수와 순서는 고정 목록의 조사 시점인 2026-08-10 JST를 따른다.
각 고정 주소에서 Kaggle CLI 2.2.4로 2026-08-10에 내려받은 최신 공개 `.ipynb`와 `kernel-metadata.json`을 끝까지 읽었다.
12개 고정 주소는 모두 비로그인 HTTP 요청에서 상태 200을 반환했고, 내려받은 메타데이터의 식별자와 제목도 고정 목록과 일치했다.
공식 Kaggle CLI로 [대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)도 따로 내려받아 결측값, 중복 행, 시간 구성 제약처럼 코드가 전제하는 사실을 다시 계산했다.
공개 점수는 제목이나 본문이 `LB` 또는 `Public Score`라고 직접 밝힌 수치만 기록했다.
현재 12개 `.ipynb`는 모두 실행 출력이 제거된 상태이므로, 실행 결과처럼 적힌 본문 수치와 현재 코드에서 직접 확인되는 절차를 구분했다.
셀 번호는 내려받은 최신 공개 `.ipynb`에서 위에서부터 센 번호다.

## 한눈에 보는 결론

| 순위 | 분석 대상 노트북 | 득표 | 중심 접근 | 검증 설계 | 명시된 공개 점수 | 코드 근거 강도 |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 26 | [S6E8: HistGradientBoosting \| LB 0.96945](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945) | 12 | 정확한 값 목표 부호화와 파생 변수를 넣은 HistGradientBoosting | 5겹 바깥 검증과 5겹 안쪽 목표 부호화 | 0.96945 | 높음, 실행 출력 없음 |
| 27 | [Smartphone Addiction](https://www.kaggle.com/code/cv13j0/smartphone-addiction) | 12 | 결측 대체 뒤 로지스틱 회귀, SGD, XGBoost를 차례로 실행 | 중복시킨 훈련 행에 기본 5겹 검증, AUC 미측정 | 없음 | 낮음, 중복 누출 있음 |
| 28 | [Smartphone addiction GBM rank blend nb01](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01) | 12 | 세 부스팅 모형의 순위 결합, 공개 OOF 적층, 외부 제출 결합 | 자체 모형은 공통 5겹, 공개 적층은 공개 `fold_nb` 사용 | 없음 | 높음, 최종 제출 경로 모호 |
| 29 | [S6E8: LGBM \| LB 0.96965](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965) | 11 | 정확한 값 목표 부호화와 파생 변수를 넣은 LightGBM | 5겹 바깥 검증과 5겹 안쪽 목표 부호화 | 0.96965 | 높음, 실행 출력 없음 |
| 30 | [S6E8 \| Continuous Blender](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender) | 11 | 파일명 점수로 고른 제출 파일의 순위 결합 5종 | OOF 검증 없음 | 없음 | 낮음 |
| 31 | [PS:S6E8 EDA+ XGB LGBM Ensemble](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble) | 11 | 전체 자료에서 반복 결측 대체 후 XGBoost와 LightGBM 평균 | 계층 5겹, 전처리는 겹 밖에서 한 번만 학습 | 없음 | 보통 이하 |
| 32 | [📱 Predicting Smartphone Addiction - EDA](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda) | 11 | 분포, 상관, 행동 비율을 살펴보는 탐색 | 예측 검증 없음 | 없음 | 낮음 |
| 33 | [S6E8 XGBoost \| Public Score 0.96983](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983) | 10 | 정확한 값 목표 부호화와 구성 제약 파생 변수를 넣은 XGBoost | 5겹 바깥 검증과 5겹 안쪽 목표 부호화 | 0.96983 | 높음, 실행 출력 없음 |
| 34 | [🧠⚡ SmartAddict - OOF Signal Forge](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge) | 10 | XGBoost와 두 LightGBM의 중첩 OOF 결합 | 공통 5겹과 별도 5겹 결합 평가 | 없음 | 높음, 실행 출력 없음 |
| 35 | [S6:E8\|EDA](https://www.kaggle.com/code/santosh1974/s6-e8-eda) | 10 | KS 검정, 단일 변수 AUC, 범주별 목표율 탐색 | 예측 검증 없음 | 없음 | 보통 이하 |
| 36 | [Smartphone Addiction - EDA](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda) | 10 | 신호, 결측, 중복, 분포 차이를 단계적으로 진단 | 분포 구분 모형만 3겹 OOF | 없음 | 높음, 목표 예측 없음 |
| 37 | [🚀 Baseline](https://www.kaggle.com/code/pavloivanin/baseline) | 10 | LightGBM, XGBoost, CatBoost의 고정 확률 평균 | 공통 계층 5겹 | 없음 | 보통, 실행 출력 없음 |

## 노트북별 분석

### 26위: S6E8: HistGradientBoosting | LB 0.96945

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), 고정 목록 득표 12개, 마지막 변경 2026-08-06T08:48:17.080000Z다.
- 접근: 원시 12개 변수에 결측 개수, 시간 구성 합계와 비율, 화면 시간 잔여분, 시간 구성 제약에서 얻은 상한과 하한을 더하고 HistGradientBoosting을 학습한다.
- 접근: 모든 원시 값을 문자열로 바꾼 정확한 값 목표 부호화와 훈련 및 시험 자료를 합쳐 센 값 빈도를 추가한다.
- 검증 설계: [셀 11](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945)은 계층 5겹의 각 바깥 훈련 부분에서 `TargetEncoder(cv=5)`를 다시 맞춘다.
- 검증 설계: 바깥 훈련 행의 목표 부호화도 `fit_transform`의 안쪽 교차 적합으로 만들고, 바깥 검증과 시험 행은 바깥 훈련 부분의 매핑으로만 변환하므로 행 자기 목표 누출을 막는다.
- 공개 점수: 제목이 공개 순위표 점수 0.96945를 직접 밝힌다.
- 핵심 코드: [셀 7](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945)은 `daily_screen_time_hours`에서 세 구성 시간의 합을 뺀 값과 구성 시간이 하나만 빠진 행의 가능한 범위를 변수로 만든다.
- 핵심 코드: 같은 셀은 빈도를 셀 때 훈련과 시험 문자열을 합치므로 목표는 쓰지 않지만 시험 분포를 사용하는 전이형 전처리다.
- 재사용 가능한 인사이트: 정확한 값이 반복되는 합성 표에서는 히스토그램 구간만 늘리기보다 겹 안쪽 정확값 목표 부호화를 별도 변수로 주는 실험을 우선할 가치가 있다.
- 재사용 가능한 인사이트: [공식 대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)를 다시 계산하면 네 구성 시간이 모두 관측된 421,427행에서 `daily >= social + gaming + work` 위반이 0개이므로 제약 기반 파생 변수의 자료 근거는 확인된다.
- 주의점: 노트북 본문은 `daily_screen_time_hours`의 정확한 값 수준이 4,062개라고 쓰지만, 현재 공식 훈련 자료의 결측 제외 값은 1,389개이고 훈련과 시험 합집합도 1,397개여서 서술과 자료가 충돌한다.
- 주의점: 저장된 실행 출력이 하나도 없어 겹별 AUC와 전체 OOF AUC를 현재 판본에서 확인할 수 없고, 제시된 초매개변수의 선택 과정도 기록되지 않았다.

### 27위: Smartphone Addiction

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/cv13j0/smartphone-addiction), 고정 목록 득표 12개, 마지막 변경 2026-08-04T01:13:34.227000Z다.
- 접근: 숫자 결측값을 훈련 중앙값으로 채우고 범주는 훈련 자료의 최빈값으로 채운 뒤 로지스틱 회귀, SGD 분류기, XGBoost를 차례로 실행한다.
- 접근: 마지막 XGBoost의 시험 확률로 `submission.csv`를 만들므로 앞의 두 선형 모형은 비교 출력일 뿐 최종 제출에는 들어가지 않는다.
- 검증 설계: [셀 15](https://www.kaggle.com/code/cv13j0/smartphone-addiction)은 `cross_validate(..., cv=5)`로 정확도, 정밀도, 재현율, F1을 계산하지만 대회 지표 ROC AUC는 교차 검증 항목에 넣지 않는다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 3, 셀 4, 셀 7](https://www.kaggle.com/code/cv13j0/smartphone-addiction)은 `trn_path`와 원본 자료라고 이름 붙인 `ogt_path`를 모두 같은 대회 `train.csv`로 지정한 뒤 두 자료를 이어 붙인다.
- 핵심 코드: 따라서 691,369개 훈련 행이 정확히 두 번 들어가고, [공식 목표값](https://www.kaggle.com/competitions/playground-series-s6e8/data)으로 같은 기본 5겹을 다시 만들면 모든 검증 행의 복제본이 학습 부분에 들어간다.
- 재사용 가능한 인사이트: 결측 대체 통계를 훈련 자료에서만 구하고 새 범주를 별도 값으로 처리하는 보조 함수의 의도 자체는 올바르다.
- 주의점: 훈련 자료 복제 때문에 교차 검증 결과는 일반화 성능 근거로 쓸 수 없다.
- 주의점: [셀 12와 셀 13](https://www.kaggle.com/code/cv13j0/smartphone-addiction)의 범주 부호화 결과 `train_enc`, `test_enc`는 이후 학습 코드에서 사용되지 않는다.
- 주의점: 저장된 실행 출력이 없어 누출된 검증 수치조차 확인할 수 없고, 대회 AUC와 다른 네 지표만 선택 근거로 남는다.

### 28위: Smartphone addiction GBM rank blend nb01

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01), 고정 목록 득표 12개, 마지막 변경 2026-08-04T10:42:21.810000Z다.
- 접근: 원시 변수와 18개 비율 후보를 한 겹에서 비교하고, 개선 폭이 0.0005를 넘지 않으면 원시 변수만으로 LightGBM, XGBoost, CatBoost를 같은 5겹에서 학습한다.
- 접근: 세 자체 모형은 OOF 예측을 순위로 바꿔 같은 OOF 전체에서 가중치를 찾고, 별도로 공개된 10개 기초 OOF는 공개 `fold_nb`에 맞춰 로지스틱 회귀로 적층한다.
- 검증 설계: [셀 35, 셀 39, 셀 41](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 같은 계층 5겹을 세 부스팅 모형에 재사용하므로 자체 OOF 사이의 행 정렬이 맞는다.
- 검증 설계: [셀 52](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 공개 기초 예측의 `fold_nb`를 그대로 따라 각 검증 겹을 보지 않은 2단 모형만 평가한다.
- 공개 점수: 이 노트북의 최종 공개 점수는 명시되지 않으며, 본문 표의 0.97068, 0.97069, 0.97024는 세 구성 계보의 수치다.
- 핵심 코드: [셀 31](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 원시 변수와 18개 파생 변수의 비교를 첫 번째 겹 하나에서만 실행하고 그 결과로 이후 전체 파이프라인의 변수 집합을 고른다.
- 핵심 코드: [셀 44](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 0.02 간격의 세 모형 가중치를 같은 OOF 전체에서 찾고 그 OOF AUC를 그대로 보고하므로 자체 결합 수치에는 선택 편향이 남는다.
- 핵심 코드: [셀 56과 셀 58](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)은 외부 제출이 하나라도 발견되면 자체 적층 예측을 최종 제출에서 빼고 외부 제출끼리만 평균한다.
- 재사용 가능한 인사이트: 공개 OOF를 2단 결합할 때 기초 예측의 겹 식별자를 함께 요구하고 그 겹을 그대로 따라야 한다는 구현이 이 구간에서 가장 분명하다.
- 재사용 가능한 인사이트: 강해 보이는 비율 변수를 단일 변수 상관만으로 채택하지 않고 같은 분할의 직접 비교로 거르는 절차도 재사용할 가치가 있다.
- 주의점: 저장 출력이 제거되어 본문이 말하는 원시 변수 OOF 약 0.9646, 공개 적층 약 0.9689, 비율 변수 손실 약 0.0002를 현재 코드 실행 결과로 확인할 수 없다.
- 주의점: 메타데이터에 연결된 Rayk 노트북 식별자는 `s6e8-mix-the-meta-models-then-fix-the-weak-bands`인데 셀 56의 파일 찾기 패턴은 `s6e8-missingness-aware*`라서 해당 제출을 실제로 찾는지 저장 출력 없이 확정할 수 없다.
- 주의점: 본문은 세 계보를 결합한다고 설명하지만 현재 코드는 외부 두 계보만 `blend_parts`에 넣고 자체 적층은 넣지 않아 서술과 최종 코드가 일치하지 않는다.

### 29위: S6E8: LGBM | LB 0.96965

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), 고정 목록 득표 11개, 마지막 변경 2026-08-06T08:50:09.903000Z다.
- 접근: 26위와 같은 정확값 목표 부호화, 값 빈도, 비율, 시간 구성 제약 변수를 LightGBM에 넣고 최대 구간 수를 1,023으로 높인다.
- 검증 설계: [셀 11](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965)은 계층 5겹의 바깥 훈련 부분마다 `TargetEncoder(cv=5)`를 다시 맞춰 안쪽 OOF 부호화를 만든다.
- 공개 점수: 제목이 공개 순위표 점수 0.96965를 직접 밝힌다.
- 핵심 코드: [셀 7과 셀 11](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965)은 원시 12개 값 전체를 정확값 부호화하고 범주 세 변수는 LightGBM의 범주 변수로도 함께 전달한다.
- 핵심 코드: 훈련과 시험 자료를 합쳐 값 빈도를 세므로 목표 누출은 아니지만 시험 분포를 사용하는 전이형 전처리다.
- 재사용 가능한 인사이트: 같은 변수, 분할, 주요 초매개변수로 HistGradientBoosting과 LightGBM을 비교하도록 짝을 맞춘 구조는 함수 계열 차이를 살피는 출발점으로 좋다.
- 주의점: 본문은 LightGBM OOF 0.968259와 HistGradientBoosting OOF 0.968020을 서술하지만 저장된 실행 출력이 없어 현재 판본에서 다시 확인되지 않는다.
- 주의점: 26위와 마찬가지로 정확한 값 수준 4,062개라는 본문 설명은 현재 공식 자료의 결측 제외 1,389개와 충돌하고, 시간 제약 위반 0개도 코드 안에서 재검사하지 않고 주석으로만 둔다.
- 주의점: 두 모형에 거의 같은 초매개변수를 옮겼으므로 비교는 공정한 제거 실험이라기보다 하나의 설정을 두 구현에 적용한 결과로 읽어야 한다.

### 30위: S6E8 | Continuous Blender

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender), 고정 목록 득표 11개, 마지막 변경 2026-08-06T11:02:03.447000Z다.
- 접근: `/kaggle/input` 아래에서 파일명에 `0.xxxx` 형식 수치가 있는 CSV를 찾고 그 수치를 공개 점수처럼 사용해 가장 높은 제출을 기준으로 다섯 결합 파일을 만든다.
- 검증 설계: OOF, 목표값, 교차 검증 코드는 전혀 없고 파일명의 점수 순서가 구성원 선택과 가중치 구조를 결정한다.
- 공개 점수: 이 노트북 자체의 공개 점수는 명시되지 않는다.
- 핵심 코드: [셀 3](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 같은 점수 수치를 사전의 키로 쓰므로 파일명이 다른 두 제출이 같은 점수를 가지면 먼저 읽은 하나만 남긴다.
- 핵심 코드: [셀 5](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)은 최상위 제출 95%와 나머지 순위 평균 5%, 0.5의 거듭제곱 가중 순위, 상위 3개 순위 평균, 상위 5개 기하 평균, 제곱 순위 결합을 각각 저장한다.
- 재사용 가능한 인사이트: 예측 확률의 척도가 다른 제출을 결합할 때 정규화 순위를 사용하면 극단 확률 하나가 결합을 지배하는 문제를 피할 수 있다.
- 주의점: 공개 점수가 든 파일명으로 구성원을 고르고 최상위 제출에 95%를 주므로 독립적인 OOF 근거가 없고 공개 순위표 과적합 위험이 크다.
- 주의점: 행 수, `id` 집합, 목표 열 존재를 입력마다 검증하지 않고 단순히 `id`로 정렬하므로 잘못된 CSV가 검색되면 조용히 행이 어긋나거나 오류가 난다.
- 주의점: 저자는 `1_linear_anchor.csv`를 먼저 제출하라고 설명하지만 일반 이름 `submission.csv`에는 가공하지 않은 최상위 제출을 저장해 산출물 이름만으로 의도를 알기 어렵다.

### 31위: PS:S6E8 EDA+ XGB LGBM Ensemble

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble), 고정 목록 득표 11개, 마지막 변경 2026-08-03T09:06:23.450000Z다.
- 접근: 세 범주를 수동 숫자로 바꾸고 숫자 결측값을 반복 결측 대체로 채운 뒤 XGBoost와 LightGBM 확률을 같은 비중으로 평균한다.
- 검증 설계: [셀 15](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)은 두 모형에 같은 계층 5겹을 사용하고 겹별 평균 확률과 전체 OOF AUC를 계산한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 12](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)의 `IterativeImputer`는 교차 검증을 만들기 전에 전체 훈련 자료에서 한 번 맞춰진다.
- 핵심 코드: 셀 15의 `xgb.train`은 조기 종료를 켜지만 `predict`에 `best_iteration` 범위를 지정하지 않아 반환된 마지막 부스터 전체로 검증과 시험을 예측할 수 있다.
- 재사용 가능한 인사이트: 두 부스팅 모형에 같은 계층 분할을 쓰고 OOF와 시험 평균을 동시에 만드는 뼈대는 간단한 기준선으로 옮길 수 있다.
- 주의점: 전체 훈련 자료에서 결측 대체기를 먼저 맞추므로 각 검증 겹의 변수값이 학습 부분 전처리에 간접적으로 들어가며, 결측 대체기도 겹 안에서 다시 맞춰야 한다.
- 주의점: [셀 3](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)은 결측값이 없다고 주석을 달았지만 공식 자료의 12개 변수 모두에 결측값이 있어 코드와 자료가 충돌한다.
- 주의점: 저장된 실행 출력이 없고, 수동으로 정한 범주 순서와 50:50 결합에 대한 제거 실험이나 가중치 근거도 없다.

### 32위: 📱 Predicting Smartphone Addiction - EDA

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda), 고정 목록 득표 11개, 마지막 변경 2026-08-01T00:41:37.757000Z다.
- 접근: 목표 비율, 숫자 변수의 훈련 및 시험 밀도, 목표별 상자 그림, 상관, 나이 구간, 행동 비율을 시각화하는 탐색 전용 노트북이다.
- 검증 설계: 예측 모형, 교차 검증, OOF, 제출 생성 코드가 없다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 17](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)은 여가 시간 합계, 여가 비율, 화면 시간 대비 수면, 알림 대비 앱 열기, 주말 대비 평일 비율을 만들고 목표와의 피어슨 상관만 계산한다.
- 핵심 코드: [셀 13](https://www.kaggle.com/code/pavloivanin/predicting-smartphone-addiction-eda)은 숫자 변수별 훈련 및 시험 밀도를 겹쳐 보지만 분포 차이 검정이나 분포 구분 모형은 사용하지 않는다.
- 재사용 가능한 인사이트: 행동량을 시간 구성 비율과 사용 강도로 바꾸는 후보 목록은 후속 겹 안쪽 제거 실험의 입력으로 쓸 수 있다.
- 주의점: 본문은 훈련과 시험 분포가 가깝고 파생 비율이 강한 보조 신호라고 결론내지만 저장된 그림과 수치 출력이 없고 예측 성능 비교도 없다.
- 주의점: 변수 생성은 훈련 자료 복사본에만 적용되어 시험 자료 변환이나 재사용 가능한 학습 파이프라인으로 이어지지 않는다.
- 주의점: 상관은 결측을 쌍별로 제외한 선형 관계만 보여 주므로 나무 모형에 추가 가치가 있는지 판단할 근거가 아니다.

### 33위: S6E8 XGBoost | Public Score 0.96983

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983), 고정 목록 득표 10개, 마지막 변경 2026-08-07T04:03:16.557000Z다.
- 접근: 모든 원시 변수의 정확한 값 목표 부호화, 값 빈도, 화면 시간 구성 비율과 제약 변수를 XGBoost에 넣고 시험 예측의 순위를 제출한다.
- 검증 설계: [셀 4](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)는 계층 5겹의 각 바깥 훈련 부분에서 `TargetEncoder(cv=5)`를 다시 맞추고 조기 종료한다.
- 검증 설계: 바깥 훈련 행은 안쪽 OOF 목표 부호화, 바깥 검증과 시험 행은 바깥 훈련 전체 매핑을 사용하므로 목표 부호화 누출을 막는다.
- 공개 점수: 제목이 공개 순위표 점수 0.96983을 직접 밝힌다.
- 핵심 코드: [셀 2와 셀 4](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)는 시간 구성 제약 변수와 12개 정확값 목표 부호화를 결합하고 최대 12,000회 학습에 조기 종료 100회를 둔다.
- 핵심 코드: [셀 3](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 범주 정수 매핑은 훈련 자료로만 만들지만 값 빈도는 훈련과 시험 문자열을 합쳐 센다.
- 재사용 가능한 인사이트: 26위와 29위의 같은 변수 설계를 XGBoost에 옮기면서 바깥 겹과 안쪽 목표 부호화를 명확히 분리한 단일 모형 기준선이다.
- 재사용 가능한 인사이트: [셀 5](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)은 제출 전에 시험 확률을 순위로 바꾸므로 확률 보정 없이 AUC 순서만 보존한다.
- 주의점: 공식 자료에서 시간 구성 제약은 확인되지만 코드 자체는 위반 수를 검사하지 않아 다른 자료나 판본에 그대로 쓰면 잘못된 파생 변수를 만들 수 있다.
- 주의점: 저장된 실행 출력이 없어 제목의 공개 점수와 내부 OOF AUC 사이의 차이, 겹별 변동, 최적 반복 수를 확인할 수 없다.
- 주의점: 빈도 부호화가 시험 분포를 사용하므로 다른 대회에 옮길 때 전이형 전처리 허용 여부를 확인해야 한다.

### 34위: 🧠⚡ SmartAddict - OOF Signal Forge

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge), 고정 목록 득표 10개, 마지막 변경 2026-08-07T14:04:59.853000Z다.
- 접근: 목표와 시험 통계를 쓰지 않은 결측 형태, 시간 구성, 행동 비율, 고정 위험 개수, 범주 교차를 만들고 XGBoost, 전체 변수 LightGBM, 문맥 변수 LightGBM을 학습한다.
- 접근: 선택적으로 원본 스마트폰 자료가 연결되면 원본의 중독 심각도만 학습한 CatBoost 교사와 원본 행을 네 번째 LightGBM 구성원에 사용한다.
- 검증 설계: [셀 13](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)은 모든 기초 모형에 같은 계층 5겹을 사용하고 각 모형의 OOF와 겹별 시험 예측을 저장한다.
- 검증 설계: [셀 16](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)은 각 결합 검증 겹을 제외한 나머지 OOF에서 탐욕 가중치를 찾은 뒤 보지 않은 결합 겹에 적용한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: 결합은 별도 OOF AUC가 최고 단일 모형보다 0.00002 넘게 좋고 다섯 겹 중 적어도 세 겹에서 이길 때만 채택하며, 통과하지 못하면 최고 단일 모형으로 되돌아간다.
- 핵심 코드: [셀 19](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)은 제출과 함께 행별 겹, 구성원 OOF, 최종 OOF, 파일 해시를 저장해 이후 결합 감사에 필요한 계보를 남긴다.
- 재사용 가능한 인사이트: 기초 OOF와 결합 선택을 한 단계 더 분리하고, 평균 개선뿐 아니라 겹별 승수 조건을 함께 두는 방식은 작은 결합 이득을 거르는 강한 절차다.
- 재사용 가능한 인사이트: 시험 자료를 변수 선택과 가중치 선택에 쓰지 않고 결측을 나무가 직접 처리하게 둔 점도 이 구간의 재현 가능한 기준선 가운데 가장 엄격하다.
- 주의점: 현재 메타데이터에는 `najiama/predicting-smartphone-addiction-oof-submission-csv`만 자료 원천으로 선언되어 있고 셀 9이 요구하는 `guriya79/smart-phone`은 없으므로 원본 자료 분기는 현재 공개 판본에서 비활성화된다.
- 주의점: 저장된 실행 출력이 없어 어떤 단일 모형이나 결합이 실제로 선택되었는지, OOF AUC가 얼마였는지, 고정 위험 기준이 도움됐는지 확인할 수 없다.
- 주의점: 많은 고정 임계값과 변수군에 대한 제거 실험이 없어 코드의 누출 방지 품질과 각 파생 변수의 예측 가치는 구분해서 읽어야 한다.

### 35위: S6:E8|EDA

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/santosh1974/s6-e8-eda), 고정 목록 득표 10개, 마지막 변경 2026-08-01T14:54:34.567000Z다.
- 접근: 자료형과 결측률, 목표 비율, 숫자 변수 분포, 단일 변수 AUC, 범주별 목표율, 훈련 및 시험의 숫자 분포 차이를 살펴보는 탐색 전용 노트북이다.
- 검증 설계: 목표 예측 모형, 교차 검증, OOF, 제출 생성 코드가 없다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 4](https://www.kaggle.com/code/santosh1974/s6-e8-eda)는 숫자 변수마다 두 표본 KS 통계량과 p값을 계산하고 p값 0.01 미만을 분포 차이로 표시한다.
- 핵심 코드: [셀 5](https://www.kaggle.com/code/santosh1974/s6-e8-eda)는 결측을 변수별로 제외하고 원시 값의 단일 변수 AUC가 0.5보다 작으면 방향을 뒤집어 신호 크기를 비교한다.
- 재사용 가능한 인사이트: 단순 상관과 별도로 단일 변수 AUC를 계산하고 KS 통계량 자체로 훈련 및 시험 차이를 정렬하는 짧은 진단 뼈대는 재사용할 수 있다.
- 주의점: 표본 수가 약 99만 행이라 작은 차이도 매우 작은 p값을 만들 수 있으므로 `p < 0.01`만으로 실질적인 분포 차이를 판정하면 안 되고 KS 통계량 크기도 함께 봐야 한다.
- 주의점: 범주 변수의 훈련 및 시험 분포 차이, 다변수 조합의 분포 차이, 결측 형태 차이는 검사하지 않는다.
- 주의점: 저장된 실행 출력과 해석 문단이 없어 탐색 결과가 후속 변수 선택이나 검증 설계로 이어지지 않는다.

### 36위: Smartphone Addiction - EDA

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda), 고정 목록 득표 10개, 마지막 변경 2026-08-01T14:01:19.273000Z다.
- 접근: 표준화 평균 차이, 단일 변수 AUC, 상호 정보량, 결측 목표율과 결측 동시 발생, 정확한 중복 행, 단변수 분포 차이, 다변수 분포 구분 검증을 단계적으로 수행한다.
- 검증 설계: 목표 예측 모형은 없고, 훈련 행과 시험 행을 구분하는 HistGradientBoosting만 계층 3겹 OOF로 평가한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: [셀 39](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)은 훈련 내부, 시험 내부, 두 자료 사이의 정확한 변수 행 중복을 각각 다른 계산으로 센다.
- 핵심 코드: [셀 51](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)은 원시 변수만, 결측 표시만, 둘을 합친 경우의 분포 구분 OOF AUC를 따로 계산해 결측 형태와 관측값의 역할을 분리한다.
- 핵심 코드: [셀 35](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)은 15만 행의 계층 표본에서 정수형 변수와 범주형 변수를 이산 변수로 표시하고, 진단 전용 중앙값과 최빈값 대체 뒤 상호 정보량을 계산한다.
- 재사용 가능한 인사이트: 결측 표시의 단일 목표 신호와 결측 때문에 훈련 및 시험이 구분되는 현상을 다른 질문으로 나누고 각각 직접 측정하는 설계가 좋다.
- 재사용 가능한 인사이트: 정확한 중복 행만으로 누출이 없다고 단정하지 않고 단변수 검정 뒤 다변수 분포 구분 검증까지 이어가는 범위도 후속 자료 감사를 위한 좋은 기준이다.
- 주의점: 본문의 양성 비율 0.7094와 중복 수는 공식 자료 재계산 결과인 훈련 내부 0개, 시험 내부 0개, 두 자료 사이 2개와 맞지만 저장 출력이 없어 나머지 표와 분포 구분 AUC는 현재 판본에서 확인되지 않는다.
- 주의점: 본문은 원시 변수만 쓴 모형과 결측 표시만 쓴 모형의 분포 구분 AUC가 비슷하다고 요약하지만 정확한 수치를 남기지 않아 차이의 실질 크기를 판단할 수 없다.
- 주의점: 목표 예측 기준선과 파생 변수 제거 실험이 없으므로 탐색 결과가 실제 대회 AUC를 높이는지는 별도 겹 안쪽 실험이 필요하다.

### 37위: 🚀 Baseline

- 링크와 판본: [최신 공개 노트북](https://www.kaggle.com/code/pavloivanin/baseline), 고정 목록 득표 10개, 마지막 변경 2026-08-02T00:11:31.987000Z다.
- 접근: 범주 결측을 별도 문자열로 바꾸고 빈도, 숫자 변수 전체의 행별 평균과 표준편차와 최솟값과 최댓값을 추가한 뒤 LightGBM, XGBoost, CatBoost를 결합한다.
- 검증 설계: [유일한 셀](https://www.kaggle.com/code/pavloivanin/baseline)은 같은 계층 5겹에서 세 모형을 학습하고 검증 확률을 0.4, 0.3, 0.3으로 고정 평균해 겹별 및 전체 OOF AUC를 계산한다.
- 공개 점수: 명시된 공개 점수가 없다.
- 핵심 코드: 변수 준비 전에 훈련과 시험 자료를 합치므로 범주 빈도와 범주 수준은 시험 자료까지 본 전이형 전처리로 만들어진다.
- 핵심 코드: 세 모형 모두 검증 자료를 `eval_set`으로 넘기지만 조기 종료 조건은 지정하지 않아 고정 1,200회 전체를 학습한다.
- 재사용 가능한 인사이트: 같은 겹의 세 나무 계열을 고정 평균하고 OOF와 시험 예측을 동시에 쌓는 한 셀짜리 기준선으로는 간결하다.
- 주의점: 행별 숫자 요약은 나이, 시간, 알림 수처럼 단위가 전혀 다른 변수를 그대로 평균하고 극값을 구해 의미와 척도가 불분명하다.
- 주의점: 훈련과 시험 결합 전처리는 목표 누출은 아니지만 시험 분포를 사용하며, 고정 가중치와 파생 요약의 제거 실험이 없다.
- 주의점: 저장된 실행 출력이 없어 겹별 AUC, 전체 OOF AUC, 세 단일 모형 대비 결합 이득을 확인할 수 없다.

## 주제별 종합

### 누출과 검증

이 구간에서 가장 엄격한 단일 모형 검증은 [26위](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29위](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33위](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)의 5겹 바깥 검증과 5겹 안쪽 목표 부호화다.
세 노트북은 검증 행의 목표가 부호화 통계에 들어가지 않도록 바깥 훈련 부분마다 부호화기를 다시 맞춘다.

[34위](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)는 기초 OOF와 결합 가중치 평가를 다시 나눠 결합 이득을 별도 겹에서 재며, 평균 개선과 겹별 승수 조건을 모두 통과하지 못하면 단일 모형으로 돌아간다.
[28위](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)의 공개 OOF 적층도 공개된 `fold_nb`를 그대로 따른다는 점은 좋지만, 자체 세 모형 가중치는 같은 OOF에서 선택하고 평가한다.

[27위](https://www.kaggle.com/code/cv13j0/smartphone-addiction)는 같은 공식 훈련 자료를 두 번 붙여 중복 쌍을 서로 다른 학습 및 검증 겹에 보내므로 검증 근거로 사용할 수 없다.
[31위](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)는 반복 결측 대체기를 겹 밖의 전체 훈련 자료에서 맞춰 전처리 경계가 느슨하고, [30위](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)는 OOF가 전혀 없다.

### 정확한 값과 합성 자료의 구조

[26위](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29위](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33위](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983)는 모두 원시 수치를 연속량으로만 보지 않고 정확한 문자열 값의 목표율을 안쪽 OOF 변수로 사용한다.
세 함수 계열에서 같은 설계를 비교할 수 있으므로 후속 실험에서는 같은 분할로 세 구현을 다시 실행하고 OOF 상관과 결합 기여를 비교하는 것이 가장 직접적이다.

시간 구성 제약 `daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours`는 [공식 대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data)의 네 값 완전 관측 421,427행에서 위반 0개로 재확인된다.
다만 세 노트북은 코드 실행 중 이 불변식을 검사하지 않으므로 입력 판본이 바뀌었을 때 조용히 잘못된 상한과 하한을 만들 수 있다.
또한 26위와 29위의 정확한 값 수준 4,062개라는 설명은 현재 공식 자료와 맞지 않으므로 생성기 해석은 코드보다 강하게 단정하면 안 된다.

### 결측값과 분포 차이

[36위](https://www.kaggle.com/code/tuannm3812/smartphone-addiction-eda)는 결측 표시의 목표 신호, 결측 동시 발생, 훈련 및 시험 분포 구분을 분리해 측정하고, [34위](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)는 이를 고정 변수군과 별도 문맥 모형으로 실제 예측 파이프라인에 연결한다.
반면 [31위](https://www.kaggle.com/code/bhaskarmishra44796/ps-s6e8-eda-xgb-lgbm-ensemble)는 결측값이 없다는 잘못된 주석을 남긴 채 전체 자료에서 결측 대체기를 먼저 맞추고, [37위](https://www.kaggle.com/code/pavloivanin/baseline)는 나무의 자연 결측 처리를 유지한다.

[26위](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945), [29위](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965), [33위](https://www.kaggle.com/code/byerscrip/s6e8-xgboost-public-score-0-96983), [37위](https://www.kaggle.com/code/pavloivanin/baseline)는 목표를 쓰지 않지만 훈련과 시험을 합쳐 빈도를 세는 전이형 전처리를 사용한다.
이는 목표 누출과는 다르지만 다른 대회나 배포 환경에서는 미래 자료를 미리 볼 수 있다는 전제가 성립하는지 따로 확인해야 한다.

### 모형과 결합

공개 순위표 수치가 직접 붙은 단일 모형은 26위 HistGradientBoosting 0.96945, 29위 LightGBM 0.96965, 33위 XGBoost 0.96983이다.
세 노트북은 변수 설계가 거의 같아 XGBoost가 가장 높은 제목 수치를 기록했다는 사실만으로 함수 계열의 우열을 확정할 수 없고, 저장 OOF 출력도 없어 같은 내부 기준으로 비교되지 않는다.

[28위](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01)는 공개 OOF와 겹 식별자를 활용한 2단 결합 절차가 강하지만 최종 제출은 외부 제출 파일 탐색 결과에 따라 달라진다.
[30위](https://www.kaggle.com/code/anhadmahajan06/s6e8-continuous-blender)는 공개 점수 파일명에 직접 의존하므로 연구용 OOF 결합과 분리해야 하고, [34위](https://www.kaggle.com/code/lucifer19/smartaddict-oof-signal-forge)의 중첩 결합 통과 조건이 후속 자체 실험에 더 적합하다.

## 후속 실험 우선순위

1. 26위, 29위, 33위의 정확값 목표 부호화 구조를 동일한 5겹 분할로 다시 실행해 단일 OOF, 겹별 변동, 세 모형 사이 순위 상관을 확보한다.
2. 시간 구성 제약 변수는 매 실행마다 완전 관측 행의 위반 수를 단언으로 확인하고, 제약 변수 전체를 뺀 제거 실험과 함께 평가한다.
3. 34위의 별도 결합 겹과 겹별 승수 조건을 공통 결합 절차로 채택하고, 같은 OOF에서 가중치를 찾고 같은 OOF를 보고하는 28위 방식과 직접 비교한다.
4. 36위의 원시 변수, 결측 표시, 두 집합 결합의 분포 구분 검증을 재현한 뒤 결측률 차이가 목표 OOF의 겹별 성능에도 영향을 주는지 별도로 확인한다.
5. 27위의 중복 자료 경로와 31위의 겹 밖 결측 대체는 후속 기준선에서 제외하고, 모든 전처리를 겹 안쪽 파이프라인으로 옮긴다.
6. 28위와 30위의 외부 제출 결합은 자체 모형 연구와 분리해 출처, 겹 식별자, OOF 존재 여부를 장부로 남긴 경우에만 후보로 사용한다.
