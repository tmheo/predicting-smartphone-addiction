# kodaifukuda 원본 분포 좌표 피처 검증

## 조사 목적과 판정

이 문서는 GitHub 이슈 [#84](https://github.com/tmheo/predicting-smartphone-addiction/issues/84)의 조사 결과를 기록한다.
조사 대상은 `Feature Engineering: What Worked and What Didn't`의 최신 공개 소스이며, Kaggle 메타데이터가 가리키는 `jayjoshi37/smartphone-usage-and-addiction-prediction` 원본 자료를 참조한다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)
[Kaggle 원본 자료](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction)

정확한 구현을 확인한 결과, 최종 원본 기반 묶음은 전체 경험적 누적분포 3개, 클래스별 누적분포 차 5개, 중앙값 거리 14개, 20분위 구간 원본 목표 평균 4개, 커널 밀도 로그우도비 3개로 모두 29개다.
노트북의 `ORIG quantile IDs`는 실패한 시도로 설명될 뿐 최종 구현에는 들어가지 않는다.
따라서 이 이슈에서 말한 `quantile statistics`는 최종 코드의 분위수 구간별 원본 목표 평균으로 좁혀 이해해야 한다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

구현은 재현할 수 있을 만큼 구체적이지만, 공개 노트북 소스에는 실행 출력과 피처 계열별 제거 실험 수치가 없다.
저자가 전체 CDF와 클래스별 CDF 차를 강한 개선, 중앙값 거리와 구간 목표 평균을 중간 개선, 밀도 로그우도비를 작지만 일관된 개선이라고 요약한 주장은 독립적으로 검증되지 않았다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

후속 실험 가치는 있다.
다만 전체 29개를 한꺼번에 더하는 표보다, 기존 정확값 원본 사전확률 및 다변량 이웃 거리와 다른 핵심인 클래스별 누적분포 차와 1차원 밀도 로그우도비를 각각 분리해 검증하는 한 개의 실험 표가 타당하다.

## 원본 자료 정리와 중복 제거

노트북은 목표값이 없는 원본 행을 제거하고 목표값을 정수로 바꾼 뒤, 9개 수치 열과 3개 범주 열 전체로 행 해시를 만든다.
수치값은 소수 여덟 자리로 반올림하고 범주 결측값은 `__MISSING__` 문자열로 바꾸며, 나머지 결측값도 같은 문자열로 통일한다.
대회 학습 자료의 해시와 일치하는 원본 행을 제거한 다음 12개 피처가 같은 원본 중복 행을 하나만 남긴다.
이 과정은 대회 학습 자료의 목표값을 사용하지 않는다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

저장소의 로컬 원본 파일 SHA-256은 `2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074`이며, 고정 대리 자료 문서에 기록된 값과 같다.
노트북과 같은 해시 규칙을 로컬 대회 학습 자료와 원본 자료에 적용한 결과, 목표값이 있는 원본 7,500행 가운데 정확 중복은 0행이고 원본 내부 중복도 0행이어서 7,500행이 모두 남았다.
남은 원본의 클래스 0은 2,192행, 클래스 1은 5,308행이고 클래스 1 비율은 0.7077333333이다.
[저장소의 고정 대리 원본 자료 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/original-proxy-data.md)

## 최종 29개 피처의 정확한 식

### 전체 경험적 누적분포 3개

대상 열은 `daily_screen_time_hours`, `weekend_screen_time`, `social_media_hours`다.
각 원본 열의 결측값을 버리고 오름차순으로 정렬한 참조 배열을 만든다.
입력값 `x`의 피처는 `x` 이하인 원본 관측값의 수를 전체 참조 관측값 수로 나눈 우측 포함 경험적 누적분포 `F_orig(x)`다.
입력값이 결측이면 결과도 결측으로 유지한다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

이 피처는 원본 목표값을 쓰지 않는 단변량 순위 좌표다.
원시 열의 순서를 보존하는 단조 변환이므로 충분히 유연한 나무 모형에서는 원시 열과 상당 부분 중복될 수 있으며, 이것은 효과에 대한 분석적 추론이지 노트북이 제공한 제거 실험 결과가 아니다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

### 클래스별 누적분포 차 5개

대상 열은 `daily_screen_time_hours`, `weekend_screen_time`, `social_media_hours`, `notifications_per_day`, `app_opens_per_day`다.
원본의 클래스 0과 클래스 1에서 각각 경험적 누적분포 `F0(x)`와 `F1(x)`를 만들고 최종 피처는 `F0(x) - F1(x)`다.
입력값이 결측이면 결과도 결측으로 유지하며, 어느 한 클래스에 참조값이 없으면 코드는 오류를 낸다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

이 피처는 한 열의 값이 원본 두 클래스 분포를 얼마나 가르는지 모든 임곗값에서 누적해 표현한다.
정확히 같은 값의 원본 목표 평균을 찾는 기존 정확값 사전확률과 달리 인접한 수치값의 순서 정보를 모으므로, 원본 값이 성긴 구간에서도 정의된다.
[정확값 원본 사전확률 이슈 #53](https://github.com/tmheo/predicting-smartphone-addiction/issues/53)
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

### 중앙값 거리 14개

`daily_screen_time_hours`, `weekend_screen_time`, `social_media_hours`, `notifications_per_day`, `app_opens_per_day`에는 전체 원본 중앙값과 클래스 0 중앙값까지의 절댓값 거리 두 개씩을 만든다.
앞의 다섯 열 가운데 `notifications_per_day`를 뺀 네 열에는 클래스 1 중앙값까지의 절댓값 거리도 만든다.
따라서 중앙값 거리 피처는 `5 x 2 + 4 = 14`개다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

이 피처는 단변량의 고정된 꺾임점을 제공하며, 여러 열을 함께 써서 가까운 원본 행을 찾는 기존 kNN과 다르다.
원시 수치 열을 쓰는 나무 모형이 비슷한 분할을 학습할 수 있으므로, 효과의 크기는 반드시 분리 실험으로 판단해야 한다.
[원본 이웃 거리 이슈 #54](https://github.com/tmheo/predicting-smartphone-addiction/issues/54)
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

### 20분위 구간 원본 목표 평균 4개

대상 열은 `daily_screen_time_hours`, `weekend_screen_time`, `notifications_per_day`, `app_opens_per_day`다.
각 원본 열에서 0부터 1까지 21개 분위수 지점을 계산하고 중복 경계를 제거한 뒤 첫 경계를 음의 무한대, 마지막 경계를 양의 무한대로 바꾼다.
각 원본 행을 해당 구간에 넣어 원본 목표 평균과 행 수를 계산하고, 대회 행에는 그 값이 들어가는 구간의 원본 목표 평균을 붙인다.
입력 결측 또는 대응하지 않는 구간에는 원본 전체의 클래스 1 비율을 넣는다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

이는 연속 구간으로 값을 모은 원본 목표 평균이므로 정확값 원본 사전확률의 평활화된 변형에 가장 가깝다.
새로운 독립 원리라기보다 같은 가설의 구간 집계 대안이므로, 핵심 분포 피처 실험 안의 비교군으로만 두는 편이 타당하다.
[정확값 원본 사전확률 이슈 #53](https://github.com/tmheo/predicting-smartphone-addiction/issues/53)

### 커널 밀도 로그우도비 3개

대상 열은 `weekend_screen_time`, `notifications_per_day`, `app_opens_per_day`다.
각 열을 전체 원본의 평균과 모집단 표준편차로 표준화한 다음, 클래스 0과 클래스 1에 각각 1차원 가우시안 `KernelDensity`를 맞춘다.
최종 피처는 클래스 1 로그밀도에서 클래스 0 로그밀도를 뺀 `log p(x|y=1) - log p(x|y=0)`이고, 결과는 `[-20, 20]`으로 자르며 입력 결측은 유지한다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

대역폭은 클래스마다 따로 계산한다.
표준편차와 `IQR / 1.34` 가운데 유효한 작은 값을 `scale`로 골라 `0.9 x scale x n^(-1/5)`를 계산하고 `[0.10, 1.00]` 범위로 자르며, 관측값이 둘보다 적으면 0.30을 쓴다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

로컬 고정 원본에 같은 식을 적용한 결과 `weekend_screen_time`의 클래스 0과 1 대역폭은 각각 0.10964520과 0.14876804였다.
`notifications_per_day`의 두 대역폭은 0.19233400과 0.16224284였고, `app_opens_per_day`는 0.19298097과 0.16200613이었다.
이 계산은 저장소가 고정한 원본 자료의 검산 결과이며 공개 노트북의 출력으로 확인된 값은 아니다.
[저장소의 고정 대리 원본 자료 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/original-proxy-data.md)

밀도 로그우도비는 정확값 일치가 아니라 한 열의 주변 원본 밀도를 클래스별로 평활화해 비교한다.
여러 열의 공동 거리로 이웃을 고르는 kNN과도 다르므로, 기존 정확값 사전확률과 kNN이 실패했다는 사실만으로 이 피처를 기각할 수 없다.
[정확값 원본 사전확률 이슈 #53](https://github.com/tmheo/predicting-smartphone-addiction/issues/53)
[원본 이웃 거리 이슈 #54](https://github.com/tmheo/predicting-smartphone-addiction/issues/54)

## 학습과 검증 절차

노트북은 정리된 원본 전체에서 위 참조 통계를 한 번 맞춘 뒤 대회 학습 자료와 시험 자료에 적용한다.
대회 학습 자료의 목표값은 원본 기반 피처 계산에 쓰이지 않으며, 원본 목표값만 클래스별 통계와 구간 평균에 쓰인다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

최종 모형 입력은 원래 9개 수치 열과 3개 범주 열, 비율 8개, 차이 1개, 깨어 있는 시간 대비 화면 비율 1개, 원본 기반 29개다.
LightGBM은 `[42, 202, 2026, 777, 4946]` 다섯 시드마다 5겹 층화 교차 검증을 수행하도록 작성돼 있다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

하지만 공개 소스는 최종 29개 묶음만 학습하며 전체 CDF, 클래스별 CDF 차, 중앙값 거리, 구간 목표 평균, 밀도 로그우도비를 하나씩 제거한 비교 코드를 포함하지 않는다.
내려받은 노트북의 출력 배열과 실행 번호도 비어 있어 각 계열의 효과나 전체 교차 검증 점수를 독립 확인할 수 없다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

## 기존 연구와의 경계

정확값 원본 사전확률은 한 열의 정확히 같은 값 또는 키에 속한 원본 행의 목표 평균을 붙인다.
클래스별 누적분포 차와 밀도 로그우도비는 단변량 원본 분포 전체에서 주변 값의 정보를 모으므로 정확값 일치가 없어도 계산되며, 기존 사전확률과 가설 및 평활화 방식이 다르다.
[정확값 원본 사전확률 이슈 #53](https://github.com/tmheo/predicting-smartphone-addiction/issues/53)
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

기존 kNN은 여러 수치 열로 정의한 거리에서 가까운 원본 행의 목표값 또는 거리를 집계한다.
이번 피처들은 열마다 별도의 1차원 누적분포, 중앙값 또는 밀도를 쓰므로 다변량 이웃 탐색과 다르다.
[원본 이웃 거리 이슈 #54](https://github.com/tmheo/predicting-smartphone-addiction/issues/54)
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

20분위 구간 원본 목표 평균은 기존 정확값 사전확률과 가장 가깝고, 전체 경험적 누적분포와 중앙값 거리는 원시 수치 열의 결정적 변환이라 나무 모형에서 중복될 가능성이 크다.
이 둘은 별도 연구 표보다 클래스별 분포 피처 실험의 대조군으로 포함하는 것이 적절하다는 것이 이번 조사의 판단이다.

## 정당화되는 후속 실험 표

새로 만들 가치가 있는 구체적인 표는 `원본 클래스별 누적분포 차와 밀도 로그우도비의 한계 기여 결정` 하나다.
현재 우승 구성을 고정한 뒤 아래 구성을 같은 접기와 같은 고정 대리 순위표에서 비교해야 한다.

1. 전체 경험적 누적분포 3개와 중앙값 거리 14개만 더한 음성 대조군을 평가한다.
2. 클래스별 누적분포 차 5개만 더한 핵심 가설을 평가한다.
3. 커널 밀도 로그우도비 3개만 더한 핵심 가설을 평가한다.
4. 20분위 구간 원본 목표 평균 4개만 더한 기존 사전확률의 평활화 비교군을 평가한다.
5. 이긴 최소 피처 묶음을 합쳐 각 계열의 한계 기여가 남는지 평가한다.

첫 단계에서는 노트북의 전체 원본 참조 방식과 중복 제거 규칙을 그대로 재현해야 한다.
그 다음 원본 참조 통계를 대회 교차 검증 접기마다 다시 계산해도 값이 변하지 않는지 검사하고, 대회 목표값을 무작위로 바꿔도 원본 피처가 변하지 않는 누출 불변성 검사를 추가해야 한다.
채택 여부는 저장소의 고정 대리 순위표 규약과 기존 효과 크기 문턱에 따라 판단해야 한다.
[대리 순위표 규약 이슈 #47](https://github.com/tmheo/predicting-smartphone-addiction/issues/47)
[Kaggle 노트북 원문](https://www.kaggle.com/code/kodaifukuda0311/feature-engineering-what-worked-and-what-didn-t)

전체 CDF, 중앙값 거리, 구간 목표 평균을 각각 독립 표로 만들 근거는 충분하지 않다.
공개 소스가 개별 수치 증거를 제공하지 않고, 각각 원시 열 또는 기존 정확값 사전확률과 중복될 가능성이 크기 때문이다.

## 접근 제한

Kaggle 명령줄 도구로 최신 공개 노트북 소스를 직접 내려받아 식과 피처 목록을 확인했다.
공개 다운로드에는 이전 버전 전체, 비공개 실행 기록, 실행 출력이 없으므로 저자의 정성적 효과 표와 개별 계열의 개선 폭은 검증하지 못했다.

로컬 원본 파일은 공개 노트북이 지정한 자료와 저장소의 고정 대리 자료가 같은 SHA-256을 가진다는 저장소 기록을 기준으로 사용했다.
로컬에서 중복 제거 결과와 KDE 대역폭은 검산했지만, 이번 조사에서는 전체 모형 교차 검증을 다시 실행하지 않았으므로 예측 성능 향상을 주장하지 않는다.
