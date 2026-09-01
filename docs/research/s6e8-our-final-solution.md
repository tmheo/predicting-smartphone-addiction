# S6E8 우리 최종 해법과 제출 계보 복원

## 조사 질문과 기준 시점

이 문서는 저장소의 고정 실행 기록, 기계 판독 산출물, ADR, 연구 문서와 GitHub Issues를 기준으로 S6E8의 우리 최종 해법을 복원한다.
저장소 근거의 기준 커밋은 [`60f0795692efc433d2de003573502f435b9fe052`](https://github.com/tmheo/predicting-smartphone-addiction/commit/60f0795692efc433d2de003573502f435b9fe052)다.
최종 순위와 비공개 점수는 저장소에 아직 고정 기록이 없으므로 2026-09-01에 공식 Kaggle 명령줄 도구로 별도 확인했다.

## 한 문장 결론

최종 선택된 주 제출은 자체 모형 36개의 전체 자료 재학습 예측과 검증한 공개 OOF 구성원 278개의 시험 예측을 합친 314열 행렬에, 순위와 로짓을 함께 쓰는 L2 로지스틱 결합기를 전체 OOF에 맞춘 제출이다.
이 제출은 중첩 OOF AUC `0.9703843058098193`, 공개 점수 `0.97135`, 비공개 점수 `0.97109`를 기록했고 최종 순위 14위의 점수를 냈다.
최종 선택의 다른 한 장은 자체 35개 안전판이었지만 비공개 점수는 `0.97063`으로 더 낮았다.

## 최종 상태

| 항목 | 확정값 | 근거 |
| --- | --- | --- |
| 최종 주 제출 | 자체 36 전체 자료 재학습 + 외부 278, 총 314개 | [`submission-record.json`의 `candidates.extended314_own_full`](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/submission-record.json) |
| 결합 방식 | `c_selected_shrunk_rank_logit_logistic`, `C=0.03`, `lambda=1.0` | [최종 조립 기록 55-60행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L55-L60) |
| 중첩 OOF AUC | `0.9703843058098193` | [최종 조립 기록 9-12행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L9-L12) |
| Kaggle 제출 번호 | `55907610` | [최종 조립 기록 9-12행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L9-L12) |
| 공개 점수 | `0.97135` | [최종 조립 기록 9-12행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L9-L12) |
| 비공개 점수 | `0.97109` | 2026-09-01 공식 Kaggle 제출 목록 조회 |
| 최종 순위 | 14위 | 2026-09-01 공식 Kaggle 최종 순위표 조회 |
| 최종 안전판 | `e88f706e`, Kaggle `55795055`, 자체 35의 5:1 교차 검증-전체 자료 혼합판 | [최종 두 장 고정 확인](https://github.com/tmheo/predicting-smartphone-addiction/issues/488#issuecomment-5486018877) |
| 최종 안전판 비공개 점수 | `0.97063` | 2026-09-01 공식 Kaggle 제출 목록 조회 |

공식 조회에 사용한 명령은 다음 두 개다.

```bash
kaggle competitions leaderboard -c playground-series-s6e8 --show
kaggle competitions submissions -c playground-series-s6e8 --csv
```

공식 최종 순위표의 상위 14행에는 `Taemyung Heo`가 점수 `0.97109`로 14번째에 있었고, 제출 목록에서 `55907610`의 공개 점수와 비공개 점수는 각각 `0.97135`, `0.97109`였다.
이 조회 결과는 [대회 최종 순위표](https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard)에서 다시 확인할 수 있다.

## 데이터 활용

### 확인된 사실

자체 36개 구성은 대회 `train.csv`, `test.csv`와 커밋된 5분할을 공통 입력으로 삼았다.
판정 규약은 `artifacts/folds.parquet`의 5분할 OOF AUC와 시드 42, 43, 44의 평균 예측을 정본으로 둔다.
근거는 [ADR-0001 8-18행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/adr/0001-experiment-adoption-contract.md#L8-L18)이다.

자체 구성 가운데 5개는 공개 원본 프록시 자료 `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`에서 최근접 이웃, 사전 평균, 클래스별 경험적 누적분포 차이 또는 1단계 초기 예측을 만들었다.
대표 근거는 [최근접 이웃 구성](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/exp022_orig_knn.yaml), [원본 프록시 초기 예측 49-79행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/exp023_orig_proxy_residual.yaml#L49-L79), [원본 사전 평균 20-31행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/exp032_recon_orig_mean_top3.yaml#L20-L31), [누적분포 차이 13-28행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/exp048_lgb_orig_cdf_diff.yaml#L13-L28)이다.

외부 278개는 우리가 다시 학습한 모형이 아니라 공개 OOF와 시험 예측 쌍이다.
최종 선택으로 이어진 판본은 판본 1의 207개에 새 71개를 더했고, 해로운 nhtquyn 고전 확률 모형 120개와 코드 수준 목표 평균 누출 2개는 제외했다.
새 71개는 공개 노트북 출력물 45개, hboyang 11개, paiky1995 6개, najiama 재게시 5개, beicicc 3개, masayakawamata 1개다.
근거는 [확장 제출 기록 20-30행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-submission-2.md#L20-L30)이다.

외부 장부의 기본 무결성 조건은 OOF 691,369행과 시험 296,302행, 유한값, 우리 라벨 재채점 AUC와 선언 AUC의 `1e-5` 이내 일치, 예측 쌍 해시에 의한 중복 제거, 동봉 분할 벡터와 우리 분할의 일치다.
근거는 [외부 구성원 장부 39-44행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/external-member-ledger.md#L39-L44)이다.

외부 278개 가운데 분할 근거는 저자 서술 152개, 공개 코드 98개, 같은 저자의 형제 코드 13개, 분할 벡터 12개, 근거 없음 3개였다.
라이선스는 CC0 203개, CC BY 4.0 6개, Apache 2.0 5개, unknown 61개, other 3개였다.
unknown과 other를 합친 64개는 결합 입력으로만 사용하고 재배포하지 않는 사용 한정 구성원이었다.
근거는 [확장 사다리 기록 164-171행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-ladder-2.md#L164-L171)이다.

### 저장소 기반 해석

우리 해법은 단일 학습 자료와 단일 모형을 깊게 밀기보다, 같은 대회 자료에서 생성된 매우 넓은 예측 도서관과 소수의 원본 프록시 신호를 결합한 해법이다.
외부 278개 가운데 상당수는 학습 코드보다 예측 배열의 무결성과 공통 분할 정합성을 근거로 받아들였기 때문에, 데이터 활용의 핵심 자산은 원자료 자체보다 OOF 예측 생태계였다.

## 특성

### 확인된 사실

최종 36개 설정의 `features.providers[*].kind`를 집계하면 다음과 같다.
이는 [`artifacts/full-refit-plan.yaml`의 `members[*].config_path`](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/artifacts/full-refit-plan.yaml)에서 각 설정 파일을 읽어 계산한 저장소 기반 집계다.

| 특성 공급 방식 | 사용하는 자체 구성 수 |
| --- | ---: |
| 기본 파생 특성 | 30 |
| 제약식 기반 결측 보조 특성 | 27 |
| 목표 평균 인코딩 | 18 |
| XGBoost 기반 결측 보조 특성 | 14 |
| 범주 복제 | 3 |
| 빈도 인코딩 | 2 |
| 원본 프록시 클래스별 누적분포 차이 | 2 |
| 원본 프록시 최근접 이웃 | 1 |
| 원본 프록시 사전 평균 | 1 |
| 격자 쌍 목표 평균 | 1 |

반복되는 핵심 파생 특성은 `other_screen`, `screen_slack`, 화면 사용량 합과 비율, 주중과 주말 차이 계열이었다.
제약식 기반 결측 복원과 XGBoost 결측 보조 특성은 원시 수치 9개를 보완했고, 여러 신경망과 나무 모형에 공통으로 공급됐다.
대표 설정은 [결측 증강 격자 목표 평균 구성 7-56행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/missingness-propagation/07_exp035_lattice_te_missingness_augmented.yaml#L7-L56), [Lookup-Transformer 구성](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/exp059_lookup_transformer.yaml), [표 합성곱망 구성](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/exp113_tab_cnn_m0.yaml)이다.

최종 36개 가운데 6개는 관측 셀을 확률 `0.25`로 추가 결측 처리한 복제 행 2개를 더하는 독립 결측 증강을 사용했다.
설정 계약은 [결측 증강 구성 68-75행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/configs/missingness-propagation/07_exp035_lattice_te_missingness_augmented.yaml#L68-L75)에 명시돼 있다.
결측 증강 전파 일괄 판정은 1,658개 상태를 정확 채점해 5개 원자 교체를 골랐고, 풀 중첩 OOF를 `+0.000046886614` 높이며 바깥 분할 5/5를 이겼다.
근거는 [이슈 512 최종 기록 36-47행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/missingness-propagation-batch/issue512/report.md#L36-L47)이다.

### 저장소 기반 해석

특성 설계의 중심은 새로운 한두 열의 대박보다 같은 원시 12개 열을 여러 표현으로 반복 관측하게 하는 데 있었다.
목표 평균, 결측 복원, 원본 프록시, 정확값 조회, 비율과 잔차, 결측 증강을 서로 다른 모형 계열에 배분해 예측 상관을 낮추는 방식이었다.

## 모형

### 확인된 사실

최종 36개 자체 구성의 `model.kind`를 집계하면 다음과 같다.
이 표도 전체 자료 재학습 계획의 `members[*].config_path`를 따라 각 설정의 `model.kind`를 센 값이다.

| 모형 계열 | 구성 수 |
| --- | ---: |
| LightGBM | 14 |
| Lookup-Transformer | 5 |
| RealMLP | 4 |
| CatBoost | 3 |
| 표 합성곱망 | 3 |
| XGBoost | 2 |
| 정확값 원-핫 로지스틱 회귀 | 1 |
| Contextualized Spline Transformer | 1 |
| Scalar Token Transformer | 1 |
| TabM | 1 |
| TabPFN-3 | 1 |

구성원 하나는 시드 42, 43, 44 예측의 평균본이며, 후보 풀 장부의 최종 구성 이름과 실행 식별자는 [`artifacts/pool.yaml`의 `members[*]`](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/artifacts/pool.yaml)에 있다.
가장 강한 자체 단일 구성은 최종 풀 기록상 결측 증강 Lookup-Transformer `mpv1_exp131_lookup_bivariate_plr5_missingness_augmented`의 OOF AUC `0.9694062694182052`였다.
근거는 [후보 풀 장부 361-380행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/artifacts/pool.yaml#L361-L380)이다.

외부 278개는 LightGBM, XGBoost, CatBoost, RealMLP, TabM, 각종 신경망과 선형 모형을 포함하지만 모든 구성의 정확한 학습 설정을 저장소가 보유하지는 않는다.
저장소가 확정하는 것은 구성원 이름, 출처, OOF와 시험 예측, 재채점 AUC, 분할 근거 종류, 해시와 주의 사항이다.
기계 기록은 [`external-member-ledger.json`의 구성원 항목](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/external-member-ledger.json)에 있다.

### 저장소 기반 해석

자체 풀은 나무 모형 19개와 신경망 및 선형 모형 17개로 균형을 잡았다.
다만 최종 성능의 가장 큰 단일 상승은 자체 모형을 35개에서 36개로 바꾼 일이 아니라 외부 예측 폭을 278개까지 넓힌 일이었다.
35개 자체 풀 대비 외부 209개의 초기 기여가 약 `+0.00051`이었고, 이후 좋은 71개를 더한 313 구성은 직전 242 구성보다 `+0.0000633` 높았다.
근거는 [확장 사다리 결론 3-10행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-ladder-2.md#L3-L10)이다.

## 검증

### 확인된 사실

기초 구성은 고정 `StratifiedKFold(5, shuffle=True, random_state=42)` 분할에서 OOF 예측을 만들고, 확정 구성은 시드 42, 43, 44 예측을 평균했다.
분할 생성의 정본은 [`scripts/make_folds.py` 1-27행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/scripts/make_folds.py#L1-L27)이다.
목표값을 쓰는 변환은 바깥 학습 부분 안에서 다시 교차 적합했고 플라시보 카나리아를 상시 포함했다.
공통 규약은 [ADR-0001 8-24행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/adr/0001-experiment-adoption-contract.md#L8-L24)에 있다.

결합기 판정은 바깥 분할 하나를 봉인하고 나머지 4개 분할 OOF로 결합기와 내부 선택을 맞춘 뒤, 봉인 분할을 예측하는 중첩 OOF였다.
다섯 봉인 예측을 원래 행 순서로 이어붙여 전체 AUC를 계산했다.
계약 설명은 [ADR-0001 89-99행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/adr/0001-experiment-adoption-contract.md#L89-L99), 구현은 [`evaluate_outer_fold` 1421-1467행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/src/pipeline/ensemble.py#L1421-L1467)과 [`evaluate_nested` 1470-1504행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/src/pipeline/ensemble.py#L1470-L1504)이다.

최종 314 구성은 직전 313 구성보다 중첩 OOF `+0.0000234117693961311` 높았고 바깥 분할 5/5가 모두 양수라 사전 고정 교체 문턱을 통과했다.
근거는 [이슈 513 해결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/513#issuecomment-5473015364)이다.

공개 점수는 판정이나 결합기 선택에 사용하지 않았다.
최종 조립 기록도 공개 점수가 모든 판정과 조립 고정 뒤의 사후 기록임을 명시한다.
근거는 [최종 조립 기록 73-84행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L73-L84)이다.

### 저장소 기반 해석

우리 검증의 강점은 예측 행렬과 결합기 선택을 같은 바깥 검증 분할에 맞추지 않도록 중첩 구조를 끝까지 유지한 점이다.
반면 외부 구성원 상류의 OOF 생성 과정은 공개 코드가 없는 경우 저자 서술이나 위치 정렬에 의존했으므로, 중첩 검증이 외부 예측 생성 단계의 모든 선택 편향까지 제거하지는 못했다.
이 한계는 저장소 자체도 [확장 사다리 기록 164-174행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-ladder-2.md#L164-L174)에 적었다.

## 결합 방식

### 확인된 사실

각 구성원의 원시 확률에서 경험적 순위와 로짓을 함께 만들어 표준화하고, L2 로지스틱 회귀로 314개 구성의 메타 결합을 학습했다.
순위와 로짓 이중 표현의 구현은 [`_linear_features` 508-529행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/src/pipeline/ensemble.py#L508-L529)에 있다.

메타 결합 예측과 단순 구성원 순위 평균 사이의 수축 계수 `lambda`와 로지스틱 규제 강도 `C`는 바깥 학습 부분 안의 분할 하나 제외 방식으로 함께 골랐다.
격자는 `C=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)`, `lambda=(0.25, 0.5, 0.75, 1.0)`이었고, 최종 전체 OOF에서는 `C=0.03`, `lambda=1.0`이 선택됐다.
선택 절차 구현은 [`CSelectedShrunkRankLogitCombiner` 827-943행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/src/pipeline/ensemble.py#L827-L943), 최종 값은 [최종 조립 기록 55-60행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L55-L60)에 있다.

`lambda=1.0`이므로 최종 예측에는 단순 순위 평균이 섞이지 않고 규제한 메타 로지스틱 예측의 순위가 그대로 사용됐다.
가중치는 음수도 허용됐으며 비음수 평균 결합이 아니었다.

결합기를 시험 예측에 적용할 때는 전체 314열 OOF와 전체 라벨에 한 번 다시 적합했고, 같은 열 순서의 시험 예측 행렬에 적용했다.
이 계약은 [`full_fit_predictions` 1572-1587행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/src/pipeline/ensemble.py#L1572-L1587)에 있다.

### 저장소 기반 해석

이 해법은 전형적인 넓은 2단 결합이다.
결합기의 복잡성보다 열의 폭과 다양성이 더 큰 성능 요인이었고, 약한 모형 120개를 무작정 추가했을 때는 오히려 중첩 OOF가 `-0.000057` 악화됐다.
근거는 [확장 사다리 기록 135-153행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-ladder-2.md#L135-L153)이다.

## 전체 자료 재학습과 추론

### 확인된 사실

최종 자체 풀은 36개였다.
구성원 항목이 바뀌지 않은 29개는 이미 검증한 전체 자료 예측을 항목 해시로 다시 확인해 재사용했고, 변경된 7개는 학습 자료 전체로 다시 적합했다.
7개는 결측 증강 교체 6개와 정확값 원-핫 로지스틱 초기 점수를 쓰는 LightGBM 1개였다.
근거는 [최종 조립 기록 26-42행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L26-L42)이다.

전체 자료 재학습은 교차 검증 분할별 시험 예측 평균과 구분된다.
자체 36개는 학습 자료 전체에 적합한 시험 예측을 사용했고, 외부 278개는 공개 구성원이 제공한 교차 검증 분할 평균 시험 예측을 사용했다.
근거는 [최종 조립 기록 3-7행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L3-L7)과 [55-60행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L55-L60)이다.

36개 자체 전용 제출과 314개 확장 제출을 독립된 새 과정과 새 출력 폴더에서 두 번 조립했고, CSV와 명세가 바이트 단위로 같았다.
외부 278개의 OOF와 시험 예측 의미 해시도 전부 다시 확인했다.
근거는 [최종 조립 기록 62-71행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L62-L71)이다.

최종 314 제출은 296,302행, `id`와 `addicted_label` 두 열, 전부 유한한 `[0, 1]` 값, 동률 없음 조건을 통과했고 SHA-256은 `cbb0419a8b34b54ed11ece481d5927da3d98f2aa574839756eb8e965d3ecceaf`였다.
근거는 [최종 조립 기록 73-80행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L73-L80)이다.

## 계산 자원

### 확인된 사실

대부분의 나무 모형 전체 자료 재학습과 최종 결합은 로컬에서 실행했다.
변경된 Lookup-Transformer 하나만 Vast.ai의 RTX A4000 4장 장비에서 GPU 3장을 사용해 시드 42, 43, 44를 병렬 학습했다.
실패한 첫 전송을 이식 가능한 형식으로 고친 뒤 성공 결과의 원격과 로컬 SHA-256이 일치했고, 사용 뒤 인스턴스와 별도 저장 공간이 모두 0개임을 확인했다.
실제 잔액 차이는 `$0.393844836990070`였다.
근거는 [최종 조립 기록 40-53행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L40-L53)이다.

313열 규제 강도 선택 판정은 로컬 Apple Silicon 14코어, 메모리 48GB에서 분할 작업 합계 111분, 작업 최대 메모리 14.1GB가 들었다.
근거는 [규제 강도 판정 보고 79-99행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/logistic-c-selection/issue489/report.md#L79-L99)이다.

폭이 큰 결합 판정은 메모리가 병목이었다.
400열대 작업은 하나당 10GB에서 16GB를 사용했고 동시 5개 실행에서 기계 재시작이 발생해 이후 동시 3개를 상한으로 고정했다.
근거는 [확장 사다리 기록 155-162행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-ladder-2.md#L155-L162)이다.

## 재현 가능성

### 확인된 사실

최종 자체 풀과 전체 자료 재학습 계획의 SHA-256은 각각 `40947563a00cab8212498c7e339517e387979b14c6477c6ce8e196036e02044c`, `89edf321b1821f1de645799f2353705c461065a275263375d5479e3edd6b006c`였다.
최종 조립 명세 SHA-256은 `7f3179e577748dda1ea5b36a498d07a7a01a9b120f8e00e64a10099502e51495`였다.
근거는 [최종 조립 기록 26-30행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L26-L30)과 [62-65행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L62-L65)이다.

`artifacts/full-refit-plan.yaml`은 각 자체 구성의 원 설정 커밋, 설정 해시, 관측 학습 길이, 시드별 전체 자료 예산과 근거 산출물 해시를 담는다.
표준 `pipeline.refit --assemble`은 설정, 입력, 계보, 예산, 시드와 예측 해시를 확인한 뒤 제출을 만들었다.
근거는 [`full-refit-plan.yaml`의 `members[*].lineage`와 `training_length_evidence`](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/artifacts/full-refit-plan.yaml)와 [최종 조립 기록 40-42행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/extended-stack-final-assembly/issue514/report.md#L40-L42)이다.

외부 예측 배열은 저장소에 커밋하지 않았지만 장부에 원본 경로, 예측 의미 해시, 파일 자료형, 분할 근거, 선언 및 재채점 AUC와 출처를 기록했다.
재현하려면 Kaggle의 원 데이터셋과 노트북 출력물을 같은 경로에 다시 내려받아야 한다.
근거는 [외부 구성원 장부 21-31행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/external-member-ledger.md#L21-L31)이다.

### 저장소 기반 해석

자체 36개와 최종 조립은 해시와 두 번의 독립 조립으로 강하게 고정돼 있다.
반면 외부 278개의 완전 재현은 공개 자료의 장기 보존, 노트북 판본과 저자 서술에 의존하므로 자체 구성보다 약하다.

## 제출 계보

| 시점과 역할 | 구성 | 중첩 OOF AUC | 공개 | 비공개 | 최종 선택 |
| --- | --- | ---: | ---: | ---: | --- |
| 안전판 `e88f706e`, Kaggle `55795055` | 자체 35, 5:1 교차 검증-전체 자료 혼합 | `0.9698106` | `0.97099` | `0.97063` | 예 |
| `4f2466f8`, Kaggle `55810100` | 자체 35 + 외부 207, 242개 | `0.9702876` | `0.97134` | `0.97106` | 아니요 |
| `443b3a71`, Kaggle `55823369` | 자체 35 + 외부 278, 313개 | `0.9703509` | `0.97135` | `0.97108` | 아니요 |
| `30b6f97c`, Kaggle `55844886` | 같은 313개, C 선택 결합 | `0.9703609` | `0.97135` | `0.97109` | 아니요 |
| `3279e114`, Kaggle `55907610` | 자체 36 전체 자료 재학습 + 외부 278, 314개 | `0.9703843` | `0.97135` | `0.97109` | 예 |
| `0e423c9a`, Kaggle `55920131` | 314개 + 엄격 외부 후보 13개, 327개 | `0.9703890` | `0.97133` | `0.97108` | 아니요, 기록용 제출 |

안전판과 242개 제출의 저장소 근거는 [이슈 445 해결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/445#issuecomment-5434536694)이다.
313개 고정 결합 제출은 [이슈 457 해결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/457#issuecomment-5442676809), C 선택 제출은 [이슈 489 결정 기록 32-40행](https://github.com/tmheo/predicting-smartphone-addiction/blob/60f0795692efc433d2de003573502f435b9fe052/docs/research/logistic-c-selection/issue489/override.md#L32-L40), 314개와 327개 제출은 [이슈 526 해결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/526#issuecomment-5481420865)에 연결돼 있다.
비공개 점수는 모두 2026-09-01 공식 Kaggle 제출 목록 조회값이다.

계보에서 중요한 사실은 313개 C 선택판과 최종 314개 판이 비공개 점수 `0.97109`로 같았다는 점이다.
최종 풀 갱신은 중첩 OOF를 `+0.0000234` 높였지만 공개 점수도 `0.97135`로 같았고 비공개 점수도 표시 단위에서 같았다.
327개 기록용 제출은 중첩 OOF 점추정이 `+0.0000047` 높았지만 교체 문턱과 분할 부호를 통과하지 못했고 실제 비공개 점수도 `0.97108`로 낮았다.

## 확인된 사실, 저장소 기반 해석, 불명확점

### 확인된 사실

- 실제 최종 선택은 안전판 `55795055`와 314개 확장판 `55907610`이었다.
- 최종 점수 제출은 자체 36개와 외부 278개를 합친 314개 2단 결합이었다.
- 자체 풀은 11개 모형 계열의 36개 시드 평균본으로 구성됐다.
- 외부 278개는 공통 5분할과 배열 무결성 기준으로 장부화한 공개 OOF와 시험 예측이었다.
- 결합은 순위와 로짓 이중 표현 L2 로지스틱이며 최종 `C=0.03`, `lambda=1.0`이었다.
- 자체 36개는 전체 자료에 다시 적합했고 외부 278개의 시험 예측은 각 공개 구성의 분할 평균 예측이었다.
- 최종 주 제출의 공식 비공개 점수는 `0.97109`, 최종 순위는 14위였다.

### 저장소 기반 해석

- 최종 성능의 주된 동력은 특정 자체 단일 모형보다 외부 예측 도서관의 폭과 다양성이었다.
- 결측 증강과 전체 자료 재학습 갱신은 검증상 정당했지만 최종 비공개 점수를 표시 단위에서 더 올리지는 못했다.
- 매우 넓은 결합에서도 약한 열을 무차별적으로 늘리는 것은 해로웠고, 출처 묶음 절제가 실제 성능에 중요했다.
- 검증과 기록 규율은 강했지만 외부 구성원의 상류 학습 과정까지 모두 재현한 해법은 아니었다.

### 불명확점

- 저장소에는 최종 14위와 비공개 점수 `0.97109`가 아직 커밋된 기계 기록으로 들어 있지 않다.
  이 문서는 공식 Kaggle API 조회로 그 공백을 메웠다.
- 외부 278개 전부의 정확한 학습 설정, 원자료 사용 여부와 학습 비용은 알 수 없다.
  일부는 공개 코드, 일부는 형제 코드나 저자 서술, 3개는 분할 근거 없음에 의존한다.
- 라이선스 unknown 또는 other인 64개는 당시 사용 한정 결정으로 포함됐으나, 독립 재배포 가능한 완전한 해법 묶음으로는 만들 수 없다.
- 공개 및 비공개 점수가 소수점 다섯 자리이므로 `30b6f97c`와 `3279e114` 사이의 실제 미세 차이 방향은 알 수 없다.
- 최종 314개 결합의 전체 계수 표는 조립 명세에 들어 있지만, 이 문서는 314개 가중치를 개별 해석하지 않았다.
