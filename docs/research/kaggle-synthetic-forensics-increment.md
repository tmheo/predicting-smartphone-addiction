# Kaggle 합성 데이터 생성 지문 추가 조사

## 조사 범위

이 문서는 GitHub 이슈 [#81](https://github.com/tmheo/predicting-smartphone-addiction/issues/81)의 조사 결과를 기록한다.
조사 기준 시각은 2026-08-12이며, 2026-08-10 이후 공개된 Kaggle 디스커션과 새 댓글을 확인했다.
Kaggle Code는 공식 명령줄 도구의 `dateRun` 정렬 결과 첫 100개를 기준으로 2026-08-11 이후 실행된 33개를 모두 내려받았고, 같은 결과에 포함된 나머지 노트북 가운데 추천 수가 10 미만인 62개도 함께 내려받아 원본 자료, 생성기, 결측값 복원, 분포 변환, 제약식, 숫자 격자와 관련된 코드 무늬를 검색했다.
따라서 이번 Code 조사는 대회의 모든 공개 노트북을 전수 조사한 것이 아니라, 최신 실행 결과와 그 결과 안의 저추천 후보를 넓게 훑은 조사다.

## 결론

이번 증분 조사에서 기존 범주와 다른 새로운 원본 데이터에서 합성 데이터로의 생성 규칙은 발견하지 못했다.
새 디스커션의 시간 합계 제약은 이미 제약 복원 실험에서 다룬 내용이고, 저추천 노트북의 원본 목표 평균, 거리, 숫자 격자와 비율 피처도 기존 연구 범주와 겹쳤다.
다만 2026-08-12에 갱신된 `tomasa2` 노트북은 XGBoost 조건부 예측으로 각 결측 수치 열을 복원한 뒤, 복원 열과 원래 결측값 보존 열을 나란히 제공하는 정확한 방식을 공개했다.
이 아이디어 자체는 저장소의 이전 노트북 조사에 이미 언급됐지만, 이번 갱신본은 구현 매개변수와 저자 보고 효과를 충분히 구체화했으므로 독립된 한계 기여 실험을 만들 근거가 생겼다.

## 디스커션에서 확인한 내용

### 시간 합계 제약은 새 생성 지문이 아니다

`The Generator Didn't Just Smooth the Labels - It Fixed the Data` 글은 원본 7,500행에서 `Social_Media_Hours + Gaming_Hours > Daily_Screen_Time_Hours`가 26.3%였지만 생성된 학습 데이터와 시험 데이터에서는 0%라고 보고했다.
같은 글은 오락 시간 비율 평균이 원본 0.815, 생성 데이터 0.524이며 생성 학습 데이터와 시험 데이터의 요약값이 소수 셋째 자리까지 같다고 보고했다.
이 관찰은 생성 과정이 시간 열의 결합 제약을 학습하거나 후처리했을 가능성을 뒷받침하지만, 글 자체는 생성기 종류나 제약 적용 순서를 식별하지 않는다.
[Kaggle 디스커션 원문](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734501)

`Dilligence will prevail` 글의 새 댓글도 `Daily_Screen_Time_Hours`가 `Work_Hours + Social_Media_Hours + Gaming_Hours`보다 작지 않다는 제약을 지적한다.
이 제약은 더 약한 오락 시간 합계 제약을 포함하므로, 새 글의 발견을 별도 생성 원리로 볼 수 없다.
[Kaggle 디스커션 원문과 댓글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733983)

저장소에서는 이미 제약식 위반량과 여유값을 다뤘고, 결측값 복원 뒤 제약을 적용하는 실험도 수행했으므로 이 관찰만으로 새 후속 실험 표를 만들 이유는 없다.
[제약 기반 실험 이슈 #46](https://github.com/tmheo/predicting-smartphone-addiction/issues/46)
[결측값 복원과 제약 적용 이슈 #74](https://github.com/tmheo/predicting-smartphone-addiction/issues/74)

### 새 댓글의 교차 검증 누출 지적은 생성 규칙이 아니다

`Exact Target Encoding: A Different Way to Use the Original Data` 글의 새 댓글은 내부 교차 적합으로 만든 목표 인코딩을 바깥 교차 검증 전에 고정하면, 바깥 검증 행의 목표값이 다른 학습 행의 인코딩에 들어갈 수 있다고 지적했다.
이는 점수 추정 절차의 누출 문제이며, 원본 데이터에서 합성 데이터가 만들어진 방식을 설명하는 생성 지문은 아니다.
[Kaggle 디스커션 원문과 댓글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734063)

`Exact Target Encoding`은 이번 조사에서 제외하도록 지정된 기존 범주이고, 저장소의 실험도 바깥 접기마다 인코딩을 다시 맞추는 방식으로 평가했다.
[정확값 목표 인코딩 이슈 #53](https://github.com/tmheo/predicting-smartphone-addiction/issues/53)

### 나머지 새 글과 댓글은 생성 원리를 제시하지 않는다

`Are synthetic-data competitions usually like this?`는 원본 자료의 효과가 지나치게 크다는 질문이지만 생성 방법에 대한 관찰이나 코드를 제시하지 않는다.
[Kaggle 디스커션 원문](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734355)

`I want to know how to improve this model`은 모델 선택과 점수 개선 조언을 묻는 글이며 원본에서 합성 자료가 만들어지는 과정을 다루지 않는다.
[Kaggle 디스커션 원문](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734354)

`the beauty of a big validation set`의 새 댓글은 공개 순위표 변동성을 논의하며 생성 지문을 제시하지 않는다.
[Kaggle 디스커션 원문과 댓글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)

`An excellent starting point`의 새 댓글은 기존 수치와 주장에 대한 정정 또는 일반적인 모델링 의견이며, 새로운 생성 과정을 제시하지 않는다.
[Kaggle 디스커션 원문과 댓글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733495)

## Code에서 확인한 내용

### 조건부 결측값 복원 보조 열은 구체적인 후속 실험 가치가 있다

`S6E8: What moved the score and what didn't`의 최신 공개 소스는 학습 데이터와 시험 데이터의 피처를 합친 뒤, 결측값이 있는 각 수치 열을 나머지 수치 열과 범주 열로 예측하는 `XGBRegressor`를 맞춘다.
목표 변수인 `addicted_label`은 이 결측값 모형의 입력에 포함하지 않는다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)

노트북은 관측 행 전체로 각 열의 모형을 맞추고, `n_estimators=400`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=20`, `tree_method="hist"`, `enable_categorical=True`를 사용한다.
복원한 수치 행렬에서 비율과 차이 피처를 만들며, 원래 수치 열의 결측값은 `rawnan_` 접두사 열에 그대로 남기고 결측 여부 열도 별도로 추가한다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)

저자는 원래 결측 열을 유지하면서 복원 열을 나란히 추가하면 점수가 약 0.0012 좋아졌고, 원래 열을 복원값으로 대체하면 나빠졌다고 적었다.
그러나 내려받은 공개 소스에는 실행 출력과 셀 실행 번호가 비어 있으므로 이 수치는 코드 실행 결과로 독립 확인하지 못했다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)

이 방식은 단순 중앙값 보조 열을 평가한 이슈 #49와 다르고, 반복 회귀식 결측값 복원 뒤 시간 제약을 적용한 이슈 #74와도 다르다.
특히 학습 데이터와 시험 데이터의 결합 분포에서 비선형 조건부 복원을 맞추되 원래 결측 상태를 보존한다는 점이 구별된다.
[단순 결측값 보조 열 이슈 #49](https://github.com/tmheo/predicting-smartphone-addiction/issues/49)
[반복 회귀식 결측값 복원 이슈 #74](https://github.com/tmheo/predicting-smartphone-addiction/issues/74)

다만 이 아이디어는 2026-08-10의 저장소 노트북 조사에 이미 기록돼 있었으므로, 이번 조사에서 처음 발견한 생성 지문이라고 표현해서는 안 된다.
이번에 새로 확보한 가치는 2026-08-12 갱신본의 정확한 모형 설정과 저자 보고 효과다.
[기존 Code 조사 문서](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/code-notebook-insights.md)

### 저추천 후보에서 새 생성기는 나오지 않았다

`phone-addiction-prediction`은 CTGAN과 CopulaGAN을 가능한 선택지로 서술하지만, 공개 코드에는 해당 생성기를 맞추거나 대회 합성 데이터를 재현하는 구현이 없다.
같은 노트북의 실제 구현은 원본 목표 인코딩과 일반 모형 조합이므로 새 생성 지문으로 채택할 수 없다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/echloeprice/phone-addiction-prediction)

`predicting-smartphone-addiction`은 원본 행을 붙이고 일부 피처의 집단별 백분위 순위를 만들지만, 학습 쪽 순위와 시험 쪽 순위를 서로 다른 모집단에서 계산한다.
생성기 재현이나 일관된 원본 기준 좌표가 아니며, 제거 실험이나 검증 증거도 없어 후속 실험 근거로 삼지 않았다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/zqx960151285/predicting-smartphone-addiction)

`smartphone-addiction-xgb`은 원본의 정확값 목표 평균, 시간 제약식, 반올림 피처를 사용하고 예측값 행렬의 주성분 분석은 시각화에만 쓴다.
따라서 원본 잠재 공간이나 생성기 구조를 복원한 것으로 볼 수 없다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/mikhailnaumov/smartphone-addiction-xgb)

나머지 내려받은 후보의 관련 코드도 정확값 목표 평균, 원본 사전확률, 거리, 시간 합계와 비율, 숫자 반올림과 격자, 일반 결측값 대체의 반복이었다.
이 범주는 이슈 #81에서 이미 다뤘거나 제외하도록 정한 범주이므로 새 메커니즘으로 세지 않았다.
[GitHub 이슈 #81](https://github.com/tmheo/predicting-smartphone-addiction/issues/81)

## 후속 실험 판단

새로 만들 가치가 있는 구체적인 실험 표는 `XGBoost 조건부 결측값 복원 보조 열의 한계 기여 결정` 하나다.
이 실험은 현재 우승 구성에 원래 수치 열과 결측 여부 열을 유지한 상태에서, 정확히 재현한 조건부 복원 열과 그 복원 열로 만든 비율 및 차이 피처를 순서대로 더해 고정된 대리 순위표에서 한계 기여를 측정해야 한다.
학습 데이터만으로 복원 모형을 맞춘 경우와 학습 및 시험 피처를 함께 쓴 경우를 나눠야 시험 분포를 활용한 효과를 분리할 수 있다.
복원 열로 원래 결측 열을 대체하는 구성은 저자의 음성 대조군으로 포함하되 별도 실험 표로 만들 필요는 없다.
[Kaggle 노트북 원문](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)
[대리 순위표 규약 이슈 #47](https://github.com/tmheo/predicting-smartphone-addiction/issues/47)

시간 제약은 이슈 #46, #74와 겹치고 생성 순서 탐색은 이슈 #85가 이미 담당하므로 새 표가 필요하지 않다.
후보 생성기 계열 자체의 조사는 이슈 #83이 담당하므로 이번 결의에서 중복 표를 만들지 않는다.
[제약 기반 실험 이슈 #46](https://github.com/tmheo/predicting-smartphone-addiction/issues/46)
[결측값 복원과 제약 적용 이슈 #74](https://github.com/tmheo/predicting-smartphone-addiction/issues/74)
[생성기 후보 조사 이슈 #83](https://github.com/tmheo/predicting-smartphone-addiction/issues/83)
[생성 순서 조사 이슈 #85](https://github.com/tmheo/predicting-smartphone-addiction/issues/85)

## 접근 제한

Kaggle 디스커션은 공개 웹 화면에서 본문과 댓글을 직접 확인했지만, 보조 텍스트 추출 서비스는 일부 글에서 빈 문서나 오래된 내용을 반환했다.
따라서 조사 근거는 공개 Kaggle 화면을 우선했고, 로그인 사용자에게만 보이는 내용이나 삭제된 댓글은 포함하지 못했다.

Kaggle Code는 공식 명령줄 도구가 반환한 최신 공개 소스를 기준으로 조사했다.
버전별 변경 내역 전체와 비공개 버전은 확인하지 못했고, 여러 노트북의 내려받은 소스에서 실행 출력이 제거돼 있어 저자가 적은 점수 차이를 독립 재현하지 못했다.

Code 검색의 기준은 `dateRun` 첫 100개였으므로 대회의 모든 공개 Code를 전수 조사했다고 주장하지 않는다.
대신 이 결과 안의 최신 33개와 저추천 후보 62개를 모두 내려받아 코드 수준으로 확인했다.
