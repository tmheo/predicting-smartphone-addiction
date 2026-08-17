# nishantdhingra S6E8 성패 회고 노트북 감사

이 문서는 [리서치: nishantdhingra S6E8 성패 회고 노트북의 신규 실험 단서 확인](https://github.com/tmheo/predicting-smartphone-addiction/issues/165)의 근거다.
조사 시점은 2026-08-17이며, [최신 공개 판본 1](https://www.kaggle.com/code/nishantdhingra/s6e8-what-worked-and-what-didnt/versions/1)의 원문, Kaggle 공개 실행 기록과 출력 파일, 이 저장소의 완료 실험을 대조했다.
전체 노트북은 다시 실행하지 않았다.

## 결론

새 실험 티켓을 열지 않는다.
바로 적용할 새 방법도 없고, 제한된 진입 진단으로 승격할 후보도 없다.
앞부분 회고는 알려진 정확값 신호, 쌍 격자 목표 인코딩, FM, TabM, 순위 결합과 결측 구간 결합을 폭넓게 요약하지만, 현재 공개 판본은 37개 실험의 코드, OOF 예측, fold별 짝지은 차이와 실행 이력을 제공하지 않는다.
표의 설명도 여러 행에서 문장 중간에 잘려 있어 실험 구성을 정확히 복원할 수 없다.

마지막 코드와 Kaggle 실행 기록이 확인해 주는 범위는 훨씬 좁다.
단일 seed 42의 5-fold에서 LightGBM, HistGradientBoosting, 로지스틱 회귀, 사용자 구현 FM을 학습하고, 전체 OOF에서 고른 순위 가중치로 `0.968066 ± 0.000413`을 얻어 296,302행 제출 파일을 만들었다는 사실은 확인된다.
그러나 이 코드는 표의 최고 실험 설명과 일치하지 않고, 학습행 목표 인코딩의 자기 목표값 포함, outer 검증 입력을 본 전역 어휘, 결합 가중치의 비-nested 선택이라는 세 가지 판정 결함이 있다.
따라서 `0.96806`은 실행 재현 수치이지만 이 저장소의 개선 판정에 사용할 수 있는 누출 없는 nested OOF가 아니다.

노트북의 공개 LB `0.96939`도 작성자 보고치다.
공개 출력에는 제출 파일만 있고 Kaggle 채점 기록은 없어서 해당 파일과 점수의 대응을 독립 확인할 수 없다.

핵심 주장 대부분은 이미 더 엄격하게 판정됐다.
정확값과 쌍 격자 표현은 [단일 컬럼 빈도와 추가 정확값 표현의 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/49)과 [전 쌍 격자 TE 블록의 기여 재검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/75), FM 후속 여부는 [선형 정확값 모델의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/56), TabM은 [RealMLP·TabM의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/61), 결합은 [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)와 [비선형·구간별 2단 결합의 추가 가치 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)에서 자체 공통 fold OOF로 끝까지 판정했다.
현재 champion `exp081_lookup_fold_initialization_avg3`의 3시드 OOF는 `0.969195761811`이고, 채택된 결측 구간 독립 선형 결합의 nested OOF는 `0.969510369267`이다.
둘 다 이 노트북의 단일 시드 전체 OOF 선택 수치보다 높고 계보도 더 강하다.
노트북의 `StratifiedKFold(5, shuffle=True, random_state=42)`를 대회 학습 자료에서 다시 만들어 `artifacts/folds.parquet`과 행별로 비교한 결과 691,369행의 fold가 모두 일치했으므로, 이 절대 OOF 차이는 분할 차이에서 생긴 것이 아니다.

## 확인한 판본과 출처

Kaggle 공개 페이지와 API가 가리킨 최신 판본은 판본 1이며 `scriptVersionId`는 `342855500`이다.
작성자는 Kaggle 계정 `nishantdhingra`의 Nishant Dhingra다.
공개 및 마지막 실행 시각은 `2026-08-16 21:57:38.140 UTC`이고, Kaggle kernel 식별자는 `130973329`이다.
판본 2부터 5까지의 공개 주소는 모두 404를 반환했으므로 조사 시점의 공개 판본은 하나뿐이다.

Kaggle API로 받은 메타데이터는 다음 실행 조건을 선언한다.

- CPU 실행이며 GPU, TPU와 인터넷은 비활성이다.
- 입력은 `playground-series-s6e8` 대회 자료 하나뿐이다.
- 별도 dataset, kernel, model 입력은 없다.
- Kaggle Python 이미지 digest는 `sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`다.

감사에 사용한 파일의 내용 해시는 다음과 같다.

| 자료 | SHA-256 |
| --- | --- |
| 내려받은 `.ipynb` | `24169ee7aacc59119cc7db2847351933dfcbd6082be60517dff2ff586c6f957a` |
| `kernel-metadata.json` | `601adace4bb6b8b3ee6ca5d681d38c5f4baaa57ddec2da7d179d7c5a48b1f61a` |
| 공개 실행 기록 | `ae7f1291b8d2ad16d9424d9cfa2f097e1fc6e242a5ababa126b8acff21dbfdff` |
| 공개 `submission.csv` | `cca53bf72aea69c4c57ebf7b38803b10b7f1e2895ef2eb3424614abbd5e33b2a` |

노트북은 마크다운 셀 4개와 코드 셀 1개뿐이다.
코드 셀의 `execution_count`는 `null`이고 `.ipynb` 안의 저장 출력은 비어 있다.
따라서 [판본 원문](https://www.kaggle.com/code/nishantdhingra/s6e8-what-worked-and-what-didnt/versions/1)과 별도로 [공개 출력](https://www.kaggle.com/code/nishantdhingra/s6e8-what-worked-and-what-didnt/output)의 실행 기록을 내려받아 대조했다.

## 앞부분 설명의 신뢰 범위

### 좋은 방향이지만 새롭지 않은 부분

첫 문단은 모든 실험을 같은 5-fold에서 비교하고 실패 실험도 남긴다고 밝힌다.
같은 fold를 유지하고 음성 결과를 기록하는 방향은 옳지만, 이 저장소는 이미 커밋된 `artifacts/folds.parquet`, 3시드 확정 재검증, OOF와 시험 예측의 `float64` 보존, 실행 저장소와 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)을 더 강한 형태로 적용한다.
따라서 이 방향은 바로 적용할 새 발견이 아니라 기존 규율의 재확인이다.

### fold 표준편차를 개선 문턱으로 쓰는 설명의 문제

첫 문단은 개선 폭이 fold spread보다 작으면 개선이 아니라고 말한다.
하지만 표의 `±`는 최신 코드에서 다섯 검증 fold AUC의 단순 표준편차이고, 같은 행에 대한 짝지은 challenger-baseline 차이의 불확실성이 아니다.
검증 fold마다 표본 난이도가 달라 생기는 절대 AUC 변동과, 같은 검증행에서 두 예측의 차이가 안정적인지는 다른 질문이다.
같은 fold를 썼다는 사실만으로 원시 fold 표준편차가 개선의 통계 문턱이 되지는 않는다.

회고의 자체 판정도 이 원칙과 맞지 않는다.
최고 행 `0.96806 ± 0.00041`은 바로 아래 결합 행 `0.96781 ± 0.00041`보다 `0.00025` 높을 뿐인데 `better`로 표시된다.
어휘 변경 행 `0.96735 ± 0.00042`도 정확값 행 `0.96729 ± 0.00041`보다 `0.00006` 높은데 `better`로 표시된다.
두 차이 모두 서두가 제시한 spread보다 작다.

이 저장소는 같은 OOF 행의 예측을 직접 짝지어 비교하고, 확정 재검증의 3시드 방향과 경계 구간의 fold 승리를 함께 본다.
따라서 회고의 spread 규칙을 가져오지 않는다.

### 회고 표를 제거 실험표로 읽을 수 없는 이유

표는 CV 내림차순으로 정렬돼 있고 각 행이 어느 실험을 기준으로 한 변화인지 보여 주지 않는다.
예를 들어 쌍 격자 목표 인코딩 행은 `0.96582`, 단일 정확값 목표 인코딩 행은 `0.96729`, 둘을 쌓은 행은 `0.96684`다.
마지막 요약은 쌍 격자 자체를 `+0.00147`이라고 적지만, 최초 기준 `0.96376`과의 차이는 `+0.00206`이고 바로 앞 결합 기준 `0.96396`과의 차이는 `+0.00186`이다.
숨은 부모 실험이 있을 수 있지만 공개 판본에는 그 계보가 없다.

여러 technique 설명은 `needs`, `in r`, `trea`, `blen`처럼 문장 중간에서 끝난다.
37개 실험 중 공개 표에 있는 33개 행의 설정, seed, 부모 구성, fold별 예측과 실행 파일이 제공되지 않는다.
마지막 코드도 최고 행 하나를 재현한다고만 주장하므로 나머지 32개 수치는 작성자 보고치다.

## 마지막 코드와 저장 실행 기록의 대조

### 저장 기록으로 확인되는 수치

공개 실행 기록은 약 17분에 완주했다.
실제 출력은 다음과 같다.

| 항목 | 저장 출력 |
| --- | ---: |
| FM 어휘 크기와 필드 수 | `4996`, `12` |
| LightGBM 전체 OOF | `0.96780` |
| HistGradientBoosting 전체 OOF | `0.96700` |
| 로지스틱 회귀 전체 OOF | `0.95371` |
| FM 전체 OOF | `0.96386` |
| 고른 가중치 `(LGB, HGB, LR, FM)` | `(0.7, 0.1, 0.0, 0.2)` |
| 전체 OOF에서 고른 결합 | `0.96806` |
| fold AUC 평균 | `0.968066` |
| fold AUC 표준편차 | `0.000413` |

이 값은 앞부분 headline의 CV와 반올림 범위에서 정확히 일치한다.
공개 제출 파일도 id `691369`부터 `987670`까지 중복과 결측 없이 296,302행을 담고 있으며 예측 범위는 약 `0.00000641`부터 `0.99871314`다.

### 최고 실험 설명과 코드가 일치하지 않는다

표의 최고 행은 FM이 격자 코드와 단일 컬럼 OOF 정확값 목표 인코딩을 함께 학습했다고 말한다.
하지만 `FM.fit()`에 전달되는 입력은 `fm_tr[tri]`뿐이고, 이는 원시 9개 수치와 3개 범주 열의 정수 코드 12개다.
목표 인코딩 `TEtr`, `TEva`, `TEte`는 LightGBM, HistGradientBoosting과 로지스틱 회귀 입력에만 이어 붙는다.
FM에는 목표 인코딩 값이나 개수 열이 전달되지 않는다.

따라서 공개 코드가 실행한 최종 결합을 표의 `exp 15`라고 볼 수 없다.
저장 기록이 `0.968066`을 확인해도, 그 값을 표가 주장한 FM과 목표 인코딩 결합의 효과로 귀속할 수 없다.

### 목표 인코딩은 학습행에 대해 OOF가 아니다

각 outer fold에서 코드는 `tri` 전체의 키와 목표값으로 평균표를 만든 뒤 같은 `tri` 행에 다시 매핑한다.
즉 `TEtr`의 각 학습행 인코딩에는 자기 목표값이 포함된다.
주석과 최고 행은 이를 out-of-fold 단일 컬럼 목표 인코딩이라고 부르지만 내부 분할이나 leave-one-out 계산은 없다.

outer 검증과 시험 인코딩은 `tri`의 목표값만 사용하므로 검증 목표값이 직접 들어가지는 않는다.
그러나 학습 입력에 자기 목표값이 들어가고 검증 입력에는 들어가지 않아 학습행 내부의 자기 목표값 포함과 학습-검증 분포 불일치가 생긴다.
이 결함은 검증 점수를 반드시 낙관적으로 만든다고 단정할 수는 없지만, 정확값 목표 인코딩의 효과량을 판정 불가능하게 만든다.

노트북은 올바르게 학습 부분 안에서 다시 맞춘 목표 인코딩이 `0.00272` 나빴다고 요약한다.
비교 대상 코드가 자기 목표값을 포함하므로 이 음성 결과를 올바른 fold-fit 목표 인코딩의 기각 근거로 쓰면 안 된다.
이 저장소는 지도 규칙대로 모든 목표 인코딩을 outer 학습 부분 안의 내부 OOF로 만들며, 이 규율을 유지한다.

### 어휘 설명과 코드가 일치하지 않는다

회고의 `exp 20`은 train과 test를 합쳐 격자 어휘를 만들었다고 말한다.
실제 코드는 test 열 `s2`를 읽지만 빈도와 `MIN_COUNT >= 15` 유지 목록은 `s1.value_counts()`, 즉 전체 train만으로 만든다.
test는 만들어진 mapping을 적용받을 뿐 어휘나 빈도 문턱을 바꾸지 않는다.

반면 어휘는 outer fold 루프 밖에서 전체 train으로 한 번 만들어진다.
따라서 각 fold의 검증 입력값이 목표값 없이 어휘 항목과 희귀값 문턱을 결정한다.
이는 직접 목표값 누출은 아니지만, 검증 입력을 미리 본 전이식 전처리이며 이 저장소의 학습 fold 전용 어휘 규칙과 맞지 않는다.

현재 [Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58)은 어휘를 각 outer 학습 fold에서만 만들었고, [조회 어휘 미등록값 진단](https://github.com/tmheo/predicting-smartphone-addiction/issues/128)은 그 결과 검증 미등록행이 `0.01736%`에 불과하다고 확인했다.
따라서 train 또는 train과 test 전체 어휘를 미리 보는 변형을 열 이유가 없다.

### 결합 가중치 선택은 nested 평가가 아니다

코드는 네 구성원의 전체 OOF 순위를 만든 뒤 목표값 전체로 129개 가중치 조합을 채점한다.
가장 높은 조합을 고른 뒤 같은 OOF를 `best_s`와 `CV_SCORE`로 다시 보고한다.
따라서 구성원 선택과 가중치 선택의 자유도가 평가 OOF에 노출된다.

최종 제출을 위해 전체 OOF로 가중치를 정하는 것은 가능하지만, 그 가중치의 일반화 성능을 같은 OOF 점수로 주장할 수는 없다.
이 저장소의 [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)는 순위 변환, logit 자르기, 표준화와 계수 학습을 outer 학습 부분에서만 맞춘다.
그 결과 16개 구성원의 순위와 logit 이중 표현 로지스틱 회귀가 nested OOF `0.969483491650`을 냈고, [비선형·구간별 2단 결합의 추가 가치 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)은 이를 `0.969510369267`까지 높였다.
노트북의 가중치 탐색은 이 완료 결정을 바꾸지 않는다.

### 재현성과 저장 산출물

최종 코드는 `SEED=42`를 고정하고 Kaggle 이미지 digest가 메타데이터에 남아 있어 해당 공개 실행의 환경을 대략 식별할 수 있다.
그러나 패키지 판본을 코드나 별도 잠금 파일로 고정하지 않고, fold id, OOF 예측, 모델 파일, 선택한 129개 조합의 전 점수와 37개 실험 기록을 저장하지 않는다.
LightGBM에 이 저장소가 bit 재현 실행에서 쓰는 `force_row_wise`도 고정하지 않는다.

공개 실행 기록과 제출 파일은 마지막 스크립트의 완주를 입증한다.
그것들은 회고 전체의 실험 계보와 제거 실험을 입증하지 않는다.

## 주장별 분류

`바로 적용`과 `제한된 진입 진단 또는 실험 후보`에 해당하는 항목은 없다.
아래 표는 공개 ablation 표의 33개 행을 모두 분류한다.

| exp | 주장 | 분류 | 판정 근거와 기존 대응 |
| ---: | --- | --- | --- |
| 1 | 원시 열, 결측 표시 LightGBM 기준 | 중복 | 저장소 기준 실행과 채택 계약이 이미 더 재현 가능한 형태로 존재한다. |
| 2 | 약 12개 비율, 합, 차이 피처는 효과 없음 | 중복 | 일반 파생식 대량 추가는 지도 범위 밖이다. 복원값 기반 조성 5열은 [복원 행렬 기반 비율·차이 피처의 한계 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/90)에서 3시드 `+0.00019`로 별도 채택돼 더 구체적인 자체 근거가 우선한다. |
| 3 | 세 범주 열의 LightGBM native categorical 처리는 효과 없음 | 중복 | 범주 표현은 여러 자체 트리 실험에서 이미 다뤘다. 공개 행의 코드와 출력은 없다. |
| 4 | LightGBM, HistGradientBoosting, 로지스틱 회귀 순위 평균은 효과 없음 | 중복 | 현재 16개 후보의 nested 결합과 부분집합 선택이 더 넓고 엄격하다. |
| 5 | 수치 36쌍 전체 정확값 격자 목표 인코딩은 개선 | 중복 | [전 쌍 격자 TE 블록의 기여 재검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/75)에서 champion 교체는 `-0.00037`로 기각하고, 다른 오차 계열인 exp035만 후보 풀에 유지했다. |
| 6 | 단일 컬럼 정확값 목표 인코딩은 개선 | 중복 | 정확값 신호는 실험 프로그램의 완료 이력이다. 공개 최종 코드의 학습행 인코딩은 내부 OOF가 아니어서 효과량 근거로는 쓰지 않는다. |
| 7 | 격자 embedding FM은 개선 | 중복 | 공개 FM 단독 OOF는 `0.96386`이고 코드상 목표 인코딩을 받지 않는다. [선형 정확값 모델의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/56)은 자체 후보 풀에서 FM 후속을 열 근거가 없다고 이미 결정했다. |
| 8 | 생성 규칙 잔차 피처는 효과 없음 | 근거 부족 또는 기각 | 설정 코드가 없고, 자체 [산술 잔차 표현의 최적 구성 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/46)은 `other_screen + screen_slack`을 3시드 `+0.00081`, 5/5 fold 개선으로 채택했다. 자체 짝비교가 우선한다. |
| 9 | TabM 또는 RealMLP는 결합 다양성을 개선 | 중복 | [RealMLP·TabM의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/61)은 TabM을 OOF `0.96833`, 기여 `+0.00008`로 풀에 채택하고 약한 RealMLP는 열지 않았다. |
| 10 | 구성원 logit 로지스틱 결합은 순위 평균보다 효과 없음 | 중복 | [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)가 선택 편향 없이 logit, 순위, 표준정규 분위수와 이중 표현을 모두 비교했다. |
| 11 | 결측을 별도 학습 범주로 두면 악화 | 중복 | 공개 설정과 출력이 없다. 현재 모델별 학습 fold 전용 미등록값과 원시 NaN 처리가 이미 정해져 있다. |
| 12 | 쌍 격자를 단일 정확값 인코딩 위에 더하면 악화 | 중복 | 이 방향은 [전 쌍 격자 TE 블록의 기여 재검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/75)의 champion 기각과 일치한다. |
| 13 | 정확값, FM, 신경망 결합은 개선 | 중복 | 구성원 OOF와 결합 코드가 없다. 현재 16개 감사 후보의 nested 결합이 더 높은 OOF와 완전한 계보를 가진다. |
| 14 | 반복 2x5 stratified CV와 OOF bagging은 효과 없음 | 중복 | 저장소는 고정 5-fold의 비교 가능성을 유지하고 3시드 평균본으로 확정한다. 추가 분할은 민감도 검사일 뿐 정식 fold를 대체하지 않는다는 기존 결정과 맞는다. |
| 15 | 목표 인코딩을 함께 넣은 FM이 최고 | 근거 부족 또는 기각 | 공개 FM 입력에는 목표 인코딩이 없다. 표의 설명과 실행 코드가 직접 불일치한다. |
| 16 | 목표 인코딩 MLP 구성원은 효과 없음 | 중복 | 코드와 OOF가 없고, 약한 일반 MLP는 외부 풀 진단에 따라 범위 밖이다. TabM은 별도로 자체 판정됐다. |
| 17 | native categorical CatBoost는 `-0.00357` | 근거 부족 또는 기각 | 코드와 설정이 없고, 자체 CatBoost 실험은 한때 champion OOF `0.96820`을 만들었다. [CatBoost 수치 정확값 범주 복제의 성능·다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/107)의 정밀 비교도 공개 음성 주장보다 강하다. |
| 18 | 0.1 격자와 3중 화면 격자는 악화 | 중복 | 0.1 격자는 자체 실험에서 악화했고, 3중 격자는 쌍의 정식 채택이 없어서 지도에서 닫혔다. |
| 19 | 알림 비율 격자와 결측 모양 문자열은 악화 | 중복 | 일반 비율과 결측 조합 대량 전개는 지도 범위 밖이며 공개 코드도 없다. |
| 20 | train과 test 결합 어휘가 개선 | 근거 부족 또는 기각 | 실제 코드는 train 빈도만 사용하며 설명과 불일치한다. 어휘는 full train으로 미리 만들어 outer 검증 입력을 본다. |
| 21 | 엄격한 fold 내부 목표 인코딩은 악화 | 근거 부족 또는 기각 | 비교 대상 코드의 학습행 인코딩에 자기 목표값이 포함된다. 이 음성 결과로 누출 없는 인코딩을 기각할 수 없다. |
| 22 | 정확값 조회 전용 구성원은 약함 | 중복 | 공개 행 `0.95392`는 현재 Lookup-Transformer champion 계열과 구조가 다르다. 현재 champion은 3시드 OOF `0.969195761811`이다. |
| 23 | 첫 소수 자릿수 범주는 효과 없음 | 중복 | [단일 컬럼 빈도와 추가 정확값 표현의 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/49)의 소수 첫째 자리 6열이 `-0.00031`과 전 열 중요도 미달로 이미 기각됐다. |
| 24 | 목표 인코딩 없는 원시 LightGBM 구성원은 악화 | 중복 | 현재 후보 풀은 구성원 무결성, 중복과 nested 결합을 모두 판정했고, 약한 원시 트리 하나를 추가할 이유가 없다. |
| 25 | FM embedding의 더 깊은 MLP head는 잡음 범위 | 근거 부족 또는 기각 | 해당 제거 실험 코드는 없다. 최종 FM에도 이미 hidden 384의 ReLU deep 경로가 있어 표가 구분한 head와의 차이도 복원할 수 없다. |
| 26 | rank-gauss 로지스틱 결합은 악화 | 중복 | [순위 평균과 nested 선형 스태킹 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)이 변환을 outer 학습 부분 안에서 맞춰 직접 비교했다. |
| 27 | 정확값 빈도와 개수 열은 악화 | 중복 | 단일 개수 열은 [단일 컬럼 빈도와 추가 정확값 표현의 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/49), 쌍 셀 개수 36열은 [전 쌍 격자 TE 블록의 기여 재검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/75)에서 모두 기각됐다. |
| 28 | 오류를 고친 원시 LightGBM 재시도도 악화 | 근거 부족 또는 기각 | 설명이 잘렸고 코드와 오류 수정 내용이 없어 `0.95462`의 원인을 감사할 수 없다. |
| 32 | stack OOF 잔차에 대한 피처 선별은 악화 | 근거 부족 또는 기각 | `0.90989`가 무엇의 AUC인지조차 공개 판본에서 복원할 수 없다. 현재 특성은 짝지은 OOF 변화와 플라시보 중요도로 직접 판정한다. |
| 33 | 결측 구간 meta의 2:1 혼합은 잡음 범위 | 중복 | 자체 [비선형·구간별 2단 결합의 추가 가치 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)은 세 결측 구간의 독립 선형 결합을 nested OOF `+0.000026877617`, 5/5 fold 개선으로 채택했다. 고정 2:1 혼합보다 이 증거가 강하다. |
| 34 | signed rank-gauss LR와 원시 구성원 재투입은 악화 | 중복 | signed 선형 결합과 표현 변환은 현재 nested 결합 비교 범위에서 더 엄격하게 다뤘다. |
| 35 | 화면 시간 특정 구간 FM 보정은 악화 | 중복 | 현재 구간 결합 실험은 결측 4개 이상 전용 보정과 얕은 비선형 결합까지 직접 기각했다. |
| 37 | Google TabFM zero-shot은 악화 | 근거 부족 또는 기각 | 노트북에 코드, 실행 출력, model source와 설정이 없다. 가중치 사용 조건도 별도 비상업 조건이어서 단순 재사용할 수 없다. |

## 자료, 코드와 모형 사용 조건

Kaggle 공개 노트북 소스는 저장소의 [Kaggle 공개 노트북 사용 조건](../agents/kaggle-public-notebook-licensing.md)이 정리한 대로 공개 시 Apache License 2.0이 적용된다.
이번 조사는 코드를 복사하거나 파생 구현을 만들지 않았고, 원문 주소, 판본과 내용 해시만 기록했다.
향후 코드를 복사하거나 수정해 배포한다면 Apache License 2.0 원문, 원래 고지와 NOTICE를 보존하고 변경 사실을 표시해야 한다.

최종 실행이 읽는 외부 자료는 대회 train과 test뿐이며 별도 공개 dataset이나 외부 OOF는 없다.
대회 자료는 노트북 소스 사용 조건과 별개로 [대회 규칙](https://www.kaggle.com/competitions/playground-series-s6e8/rules)을 따라야 한다.
최종 코드가 import하는 NumPy, pandas, LightGBM, scikit-learn과 SciPy는 Kaggle 실행 이미지에 들어 있지만 판본과 개별 사용 조건은 노트북이 고정하거나 열거하지 않는다.
따라서 마지막 실행의 이미지 digest는 확인 가능해도 패키지별 재배포 조건을 이 노트북만으로 완결해서 증명할 수 없다.

회고의 역사적 `exp 37`은 `google/tabfm-1.0.0-pytorch`를 언급하지만 현재 메타데이터의 model source는 비어 있고 마지막 코드도 TabFM을 불러오지 않는다.
[Google Research TabFM 소스](https://github.com/google-research/tabfm)는 Apache License 2.0이지만, [TabFM 1.0.0 PyTorch 가중치](https://huggingface.co/google/tabfm-1.0.0-pytorch)는 별도 `tabfm-non-commercial-v1.0`이며 상업 및 production 사용을 허용하지 않는다고 명시한다.
노트북은 당시 사용한 가중치의 고정 revision, 조건 수락 기록과 실행 코드를 제공하지 않는다.
성능 근거와 사용 조건이 모두 부족하므로 TabFM을 새 후보로 열지 않는다.

## 권고

이 노트북에서 새 티켓을 만들지 않는다.
정확값, 격자, FM, TabM, 결측 구간과 선형 결합은 모두 기존 완료 티켓의 근거가 더 강하므로 그 결정에 흡수한다.
특히 `strict in-fold target encoding이 나빴다`는 문장은 현재 코드의 자기 목표값 포함과 대비된 결과이므로 기존 내부 OOF 규율을 바꾸는 근거로 사용하지 않는다.
`train+test 어휘가 좋았다`는 문장도 코드가 실제로 test를 어휘 빈도에 넣지 않아 실험 후보로 사용하지 않는다.

향후 작성자가 37개 실행의 완전한 소스, 동일 fold id, 각 실험의 OOF와 시험 예측, 부모 실험 계보와 fold별 짝지은 차이를 공개하더라도 먼저 현재 16개 후보의 nested 결합에 대한 한계 기여를 읽기 전용으로 진단해야 한다.
그 자료가 없고 현재 자체 결과가 더 높고 엄격한 상태에서는 전체 재실행이나 FM, TabFM 재구현에 계산 자원을 쓰지 않는다.
