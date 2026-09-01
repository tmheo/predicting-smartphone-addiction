# S6E8 사내 회고 발표용 성적과 실험 계보 근거

## 이 문서의 역할

이 문서는 비전문가 대상 사내 발표에서 사용할 숫자를 최소한으로 고정한다.
발표 본문에는 이 문서의 `화면 표시` 값만 쓴다.
더 긴 소수점은 실제로 읽을 필요가 있을 때만 발표자 메모에 두고, 실행 식별자와 해시는 단일 기술 근거 부록에서 원본 기록으로 연결한다.
근거는 저장소의 소스, 설정, 실행 산출물, Kaggle 공식 조회 결과와 해당 결정을 처음 종결한 GitHub 이슈로 제한했다.

## 발표에서 먼저 고정할 결과

| 질문 | 화면 표시 | 정확한 기록 | 근거 |
| --- | ---: | ---: | --- |
| 최종 순위 | 14위 | 14위 | 2026-09-01 Kaggle 공식 최종 순위표 조회, [최종 해법 복원 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/547#issuecomment-5487179345) |
| 14위 성적을 만든 비공개 점수 | 0.97109 | `0.97109` | 2026-09-01 Kaggle 공식 제출 목록 조회, [최종 해법 복원 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/547#issuecomment-5487179345) |
| 같은 제출의 공개 점수 | 0.97135 | `0.97135` | [최종 조립 실행 기록](extended-stack-final-assembly/issue514/report.md), Kaggle 제출 `55907610` |
| 최종 결합의 내부 평가 | 0.97038 | `0.9703843058098193` | [최종 조립 실행 기록](extended-stack-final-assembly/issue514/report.md), [재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/513#issuecomment-5473015364) |
| 최종 결합 입력 | 자체 36 + 공개 278 | 314개 예측 열 | [최종 조립 실행 기록](extended-stack-final-assembly/issue514/report.md) |

Kaggle 공식 최종 순위표에서 `Taemyung Heo`는 비공개 점수 `0.97109`로 14번째였다.
제출 `55907610`은 공개 점수 `0.97135`, 비공개 점수 `0.97109`였다.
마지막으로 올린 327열 제출 `55920131`은 비공개 점수 `0.97108`이었으므로, 14위 성적을 만든 제출은 마지막 업로드가 아니라 314열 제출이다.

공식 재조회 명령은 다음과 같다.

```bash
kaggle competitions leaderboard -c playground-series-s6e8 --show
kaggle competitions submissions -c playground-series-s6e8 --csv
```

## 대회 소개에 쓸 최소 숫자

| 항목 | 화면 표시 | 발표자 메모 | 근거 |
| --- | --- | --- | --- |
| 입력 | 12개 생활 습관 변수 | 식별자와 목표값을 뺀 원시 열 12개 | [첫 기준 실행 설정](../../configs/exp001_lgbm_baseline.yaml), [정확값 범주 실험 설정](../../configs/exp002_all_categorical.yaml) |
| 학습 행 | 약 69만 건 | 691,369행 | [첫 기준 실행 종결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/18#issuecomment-5239693077), [외부 구성원 무결성 기록](external-member-ledger.md) |
| 예측 행 | 약 30만 건 | 296,302행 | [최종 제출 파일 검사](extended-stack-final-assembly/issue514/report.md) |
| 예측 대상 | 스마트폰 중독 여부 | 이진 목표값 `addicted_label` | [첫 기준 실행 설정](../../configs/exp001_lgbm_baseline.yaml) |
| 평가 | 위험한 사람을 더 위에 놓는 능력 | ROC AUC, 무작위 순서는 0.5, 완벽한 순서는 1.0 | [첫 기준 실행 종결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/18#issuecomment-5239693077) |

화면에서는 행 수를 `약 69만`, `약 30만`으로 단순화한다.
정확한 행 수는 자료 규모나 재현성을 묻는 질문에만 보여 준다.

## 서로 섞지 말아야 할 네 가지 점수

| 점수 | 무엇을 재는가 | 발표에서 부를 이름 | 제한 |
| --- | --- | --- | --- |
| 일반 OOF AUC | 한 구성의 고정 5분할 검증 예측을 이어 붙여 채점 | 내부 단일 구성 점수 | 실험 선택 이력 전체의 편향을 제거한 점수는 아니다 |
| 중첩 OOF AUC | 바깥쪽 분할을 보지 않고 결합 방식과 가중치를 고른 뒤 봉인 분할을 채점 | 내부 결합 점수 | 기초 구성 생성 이전의 모든 선택까지 다시 수행한 완전 중첩 평가는 아니다 |
| Public AUC | 시험 자료 가운데 공개 부분의 Kaggle 점수 | 공개 점수 | 실험 채택 근거로 사용하지 않았다 |
| Private AUC | 대회 종료 뒤 공개된 최종 채점 부분의 Kaggle 점수 | 최종 점수 | 한 번의 최종 표본 결과이며 개별 기법의 인과 효과를 증명하지 않는다 |

고정 5분할은 [`StratifiedKFold(5, shuffle=True, random_state=42)`](../../scripts/make_folds.py)로 한 번 만들었다.
중첩 OOF는 바깥쪽 분할 하나를 봉인하고 나머지 네 분할에서 결합을 고르는 절차다.
이 정의와 구현은 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)과 [`evaluate_nested`](../../src/pipeline/ensemble.py)에 있다.

## 발표용 점수 계보

한 개의 계단 차트로 모든 점수를 이어 붙이지 않는다.
아래 세 구간을 서로 다른 패널로 나눈다.

### 패널 A: 한 구성의 내부 점수

| 장면 | 화면 표시 | 정확한 값과 직접 비교 | 발표에서 말할 한 문장 | 근거 |
| --- | ---: | --- | --- | --- |
| 첫 기준 실행 | 0.96270 | OOF `0.96270` | 원시 12개 열만 넣은 첫 출발점이다 | [첫 기준 실행](https://github.com/tmheo/predicting-smartphone-addiction/issues/18#issuecomment-5239693077) |
| 수치와 정확값 범주를 함께 표현 | 0.96605 | 직전 기준 `0.96276` 대비 `+0.00329` | 같은 숫자를 숫자와 이름표 두 관점으로 함께 보여 주자 큰 폭으로 올랐다 | [정확값 범주 실험](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228) |
| 첫 Lookup-Transformer | 0.96892 | 직전 champion `0.96854` 대비 `+0.00038` | 정확한 값 조회와 연속 추세를 함께 학습한 신경망이 새 최고점이 됐다 | [Lookup-Transformer 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965) |
| 최종 자체 풀의 최고 단일 구성 | 0.96941 | OOF `0.9694062694182052` | 마지막에는 결측 증강 Lookup-Transformer가 자체 구성 가운데 가장 높았다 | [후보 풀 장부](../../artifacts/pool.yaml), [결측 증강 일괄 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/512#issuecomment-5472767484) |

### 패널 B: 결합의 내부 점수

| 장면 | 화면 표시 | 정확한 값과 직접 비교 | 발표에서 말할 한 문장 | 근거 |
| --- | ---: | --- | --- | --- |
| 자체 35개 결합 | 0.96981 | 중첩 OOF `0.9698106` | 강한 하나만 고르지 않고 서로 다르게 틀리는 자체 예측을 조립했다 | [엄격 외부 후보 사다리 계약](../adr/0006-strict-external-candidate-ladder.md) |
| 기존 외부 207개까지 포함한 242열 결합 | 0.97029 | 중첩 OOF `0.9702876097776773` | 검증한 공개 예측을 더하면서 처음 0.970대를 넘었다 | [확장 사다리 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/455#issuecomment-5442629886) |
| 해로운 120개를 뺀 313열 결합 | 0.97035 | 242열 대비 `+0.0000633`, 바깥 분할 `5/5` 양수 | 다양성은 개수가 아니라 검증을 통과할 때만 자산이었다 | [확장 사다리 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/455#issuecomment-5442629886) |
| 결합 규제를 내부에서 고른 313열 | 0.97036 | 중첩 OOF `0.9703608940404231` | 같은 입력에서도 결합 설정을 봉인 분할 밖에서 고르도록 바꿨다 | [규제 강도 선택 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/489) |
| 최종 자체 36 + 공개 278의 314열 결합 | 0.97038 | 직전 313열 대비 `+0.0000234117693961311`, 바깥 분할 `5/5` 양수 | 마지막 작은 교체도 미리 정한 문턱을 넘었을 때만 받아들였다 | [재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/513#issuecomment-5473015364) |

### 패널 C: 공식 결과

| 장면 | 화면 표시 | 의미 |
| --- | ---: | --- |
| 314열 제출 공개 점수 | 0.97135 | 대회 진행 중 보이던 시험 자료 일부의 점수 |
| 314열 제출 비공개 점수 | 0.97109 | 대회 종료 뒤 순위를 정한 점수 |
| 최종 순위 | 14위 | 팀 최고 비공개 점수 기준 최종 순위 |

내부 점수와 Kaggle 점수의 차이를 성능 향상량으로 계산하지 않는다.
두 점수는 서로 다른 행과 다른 평가 경계를 사용한다.

## 효과가 있었던 대표 사례

### 1. 같은 값을 두 관점으로 보여 주기

수치 열 아홉 개를 전부 범주로 바꾸면 OOF AUC가 `0.95859`로 내려가 직전 기준보다 `-0.00417` 낮았다.
같은 수치 열을 그대로 두고 정확값 범주 복제 아홉 개를 추가하면 OOF AUC가 `0.96605`로 올라 직전 기준보다 `+0.00329` 높았다.
발표에서는 `숫자의 크기와 정확한 눈금을 둘 다 보존해야 했다`고 설명한다.
두 결과는 같은 [정확값 범주 실험](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)의 사전 고정 변형이다.

권장 그림은 한가운데의 직전 기준에서 왼쪽으로 `전부 범주화 -0.00417`, 오른쪽으로 `수치 + 범주 복제 +0.00329`가 갈라지는 대칭 막대다.

### 2. 다르게 틀리는 구성의 가치

첫 Lookup-Transformer의 OOF AUC는 `0.96892`였고 당시 champion보다 `+0.00038` 높았다.
더 중요한 점은 후보 풀의 가장 가까운 구성과 스피어만 순위 상관이 `0.98149`로 중복 기준 `0.998`보다 낮았고, 표준 평가 앙상블을 `+0.00025` 높였다는 사실이다.
발표에서는 `혼자 잘하는가`와 `팀에 들어왔을 때 보탬이 되는가`를 별도 질문으로 나눈 사례로 쓴다.
근거는 [Lookup-Transformer 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965)이다.

권장 그림은 두 예측기의 전체 오답 수가 아니라 겹치는 오답과 서로 다른 오답을 보여 주는 두 원이다.

### 3. 새 아이디어보다 구현 정확성이 더 컸던 순간

RealMLP 이식판은 어휘 매핑 전에 `float32`로 바뀌면서 검증 미등록값 수가 `800,896`까지 늘어나는 결함이 있었다.
변환 순서를 고친 뒤 같은 짝비교에서 미등록값은 `23`으로 줄었고, 3시드 평균 OOF AUC는 `0.9637131967`에서 `0.9683223458`로 `+0.0046091491` 회복됐다.
발표에서는 `새 모형을 하나 더 찾은 것이 아니라, 이미 가진 모형이 본래 정보를 읽게 만든 개선`으로 설명한다.
근거는 [RealMLP 자료형 정합 복원 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/243#issuecomment-5343200265)과 [수정 설정](../../configs/exp124_realmlp_dtype_fix.yaml)이다.

권장 그림은 `800,896 -> 23`의 결함 지표와 `0.96371 -> 0.96832`의 점수 회복을 나란히 놓는 전후 비교다.

### 4. 작은 개선도 같은 규칙으로 판단하기

결측 증강 일괄 판정은 1,658개 가능한 상태를 정확히 비교해 원본 다섯 자리를 결측 증강판으로 바꿨다.
중첩 OOF는 `0.9698359892003905`에서 `0.9698828758140019`로 `+0.00004688661361140767` 높아졌고 바깥 분할 다섯 곳이 모두 양수였다.
발표에서는 개선 폭보다 `결과를 보기 전에 후보와 문턱을 고정했다`는 판단 절차를 중심에 둔다.
근거는 [결측 증강 일괄 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/512#issuecomment-5472767484)과 [기계 판독 판정 기록](missingness-propagation-batch/issue512/judgment.json)이다.

## 효과가 없었던 대표 사례

### 1. 모든 숫자를 범주로만 바꾸기

전부 범주화한 구성은 OOF AUC `0.95859`로 직전 기준보다 `-0.00417` 낮았다.
정확값 신호를 얻는 대신 숫자의 순서 정보를 버린 손실이 더 컸다.
근거는 [정확값 범주 실험](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)이다.

### 2. Lookup-Transformer 설정 열일곱 개 바꾸기

학습률 네 개, 학습률 일정 다섯 개, 최적화 알고리즘 조합 여덟 개가 모두 같은 fold 0 기준보다 낮았다.
차이는 `-0.0000074`부터 `-0.0018451`까지였고 전체 5분할로 승격한 후보는 없었다.
발표에서는 `많이 시도했다`보다 `싸게 걸러 멈췄다`를 강조한다.
근거는 [Lookup-Transformer 제한 탐색 종결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/160#issuecomment-5308772959)이다.

### 3. 새로운 신경망 네 종류

TabR-S, TabICLv2, AMFormer, Trompt의 fold 0 AUC는 각각 `0.941996`, `0.941745`, `0.725021`, `0.940145`였다.
같은 fold champion 대비 격차는 각각 `-0.02653`, `-0.02678`, `-0.24351`, `-0.02838`로 중단 하한 `-0.01`보다 컸다.
따라서 전체 5분할과 3시드 실행으로 확장하지 않았다.
근거는 [TabR-S](https://github.com/tmheo/predicting-smartphone-addiction/issues/142#issuecomment-5305354067), [TabICLv2](https://github.com/tmheo/predicting-smartphone-addiction/issues/143#issuecomment-5306245219), [AMFormer](https://github.com/tmheo/predicting-smartphone-addiction/issues/144#issuecomment-5306592704), [Trompt](https://github.com/tmheo/predicting-smartphone-addiction/issues/145#issuecomment-5306813049)의 원 판정이다.

권장 그림은 네 구성의 정확한 막대 높이보다 모두 `전체 검증으로 가지 않음`에 모이는 중단 깔때기다.

### 4. 약한 공개 예측 120개를 한꺼번에 더하기

단독 OOF AUC가 `0.853`부터 `0.930`인 고전 확률 모형 120개를 더하면 433열 결합은 기존 242열보다 `+0.0000063`, 바깥 분할 `3/5`에 그쳐 교체 문턱을 넘지 못했다.
120개 계열을 빼고 좋은 신규 예측 71개만 더한 313열 결합은 기존 242열보다 `+0.0000633`, 바깥 분할 `5/5`로 통과했다.
사다리 안에서 120개 계열의 한계 기여는 `-0.000057`이었다.
발표에서는 `다양한 재료`와 `검증된 다양한 재료`가 다르다는 사례로 쓴다.
근거는 [외부 구성원 장부](external-member-ledger.md)와 [확장 사다리 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/455#issuecomment-5442629886)이다.

### 5. 가장 높은 점추정이 곧 채택은 아니었던 마지막 사례

314열에 새 엄격 후보 13개를 더한 327열 결합의 중첩 OOF는 `0.9703889677646016`으로 `+0.0000047` 높았다.
이 값은 잡음 바닥 `0.0000057`보다 작고 바깥 분할도 `3/5`만 양수여서 교체 문턱에 미달했다.
사용자 지시로 기록용 제출은 했지만 공개 점수 `0.97133`, 비공개 점수 `0.97108`로 314열 제출보다 낮았고 최종 두 장 선택에는 남지 않았다.
근거는 [327열 판정과 제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/526#issuecomment-5481420865)과 [재현 관례 ADR](../adr/0008-common-module-record-conventions.md)이다.

## 계산 자원 숫자 가운데 본문에 남길 것

최종 자체 36개 가운데 변경되지 않은 29개는 이미 검증한 전체 자료 예측을 다시 사용했고, 변경된 일곱 개만 다시 학습했다.
이 가운데 Lookup-Transformer 하나의 시드 42, 43, 44만 Vast.ai GPU 세 장에서 병렬로 실행했고 나머지 변경 구성은 로컬에서 실행했다.
Vast.ai 잔액 차이로 확인한 최종 실행 비용은 `$0.393844836990070`이었다.
화면에는 `필요한 신경망 한 개, GPU 3장, 약 0.39달러`만 표시한다.
근거는 [최종 조립 실행 기록](extended-stack-final-assembly/issue514/report.md)과 [최종 조립 원 이슈](https://github.com/tmheo/predicting-smartphone-addiction/issues/514#issuecomment-5473985714)다.

이 값은 전체 대회 비용이 아니다.
앞선 탐색, 실패한 자원, 로컬 전력과 사람 시간은 포함하지 않으므로 `대회 전체를 0.39달러에 끝냈다`고 말하지 않는다.

## 차트와 숫자의 과장 방지 규칙

1. 일반 OOF, 중첩 OOF, Public, Private을 한 선으로 연결하지 않는다.
2. 패널마다 점수 종류를 제목에 적고 색도 고정한다.
3. ROC AUC 전체 맥락은 `0.5`에서 `1.0` 눈금의 작은 기준 막대로 한 번 보여 준다.
4. 세부 차트가 잘린 축을 쓰면 `확대 눈금` 표시와 실제 최솟값 및 최댓값을 함께 적는다.
5. 델타는 `개선율 %`가 아니라 `AUC 차이 +0.000xx`로 쓴다.
6. 큰 글씨에는 다섯째 소수 자리까지, 정확한 값은 발표자 메모에 둔다.
7. `0.0000057` 이하의 중첩 OOF 차이는 이 판정 체계의 잡음 바닥 안으로 표시한다.
8. 미리 고정한 문턱, 바깥 분할 승수와 함께 보여 주고 점추정 하나만으로 성공 색을 칠하지 않는다.
9. Public 점수는 후보 선택 뒤의 건전성 확인으로만 표시하고 채택 화살표의 원인으로 그리지 않는다.
10. Private 점수와 순위는 대회 결과로만 표시하고 개별 실험의 효과를 역산하지 않는다.
11. `314개 모형`이라고 부르지 않고 `314개 예측 열` 또는 `자체 36개와 공개 278개의 결합`이라고 부른다.
12. 공개 278개는 모두 우리가 다시 학습한 모형이 아니므로 `직접 학습한 314개 모형`이라고 말하지 않는다.
13. `0.39달러`는 최종 Vast.ai 재학습 작업의 비용이라고 범위를 붙인다.
14. 314열과 313열의 Public 점수는 둘 다 `0.97135`였으므로 마지막 변경이 공개 성능을 높였다고 말하지 않는다.

## 발표 자료에 바로 옮길 최소 숫자 묶음

### 첫 화면

- 약 69만 행으로 학습해 약 30만 행의 위험 순서를 예측했다.
- 비공개 점수 `0.97109`, 최종 14위였다.

### 문제와 접근법

- 입력 변수는 12개였다.
- 자체 예측 36개와 무결성을 확인한 공개 예측 278개를 314열로 조립했다.
- 최종 내부 결합 점수는 중첩 OOF `0.97038`이었다.

### 성공과 실패

- 수치 유지와 정확값 범주 복제: `+0.00329`.
- RealMLP 자료형 정합 복원: `+0.00461`.
- Lookup-Transformer 제한 탐색: 17개 모두 1차 기준 미달.
- 새 신경망 네 종류: 모두 중단 하한 미달.
- 약한 공개 예측 120개 계열의 한계 기여: `-0.000057`.

### 마지막 운영 장면

- 마지막 작은 결측 증강 교체: `+0.0000469`, 바깥 분할 `5/5`.
- 최종 Vast.ai 작업: 신경망 한 개의 세 시드, GPU 세 장, 약 `$0.39`.

## 근거를 읽는 순서

1. 최종 결과와 파일 계보는 [최종 조립 실행 기록](extended-stack-final-assembly/issue514/report.md)을 본다.
2. 단일 구성의 현재 값은 [champion 장부](../../artifacts/champion.yaml)와 [후보 풀 장부](../../artifacts/pool.yaml)를 본다.
3. 결합 선택의 비교값은 [확장 사다리 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/455#issuecomment-5442629886), [재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/513#issuecomment-5473015364), [327열 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/526#issuecomment-5481420865)을 본다.
4. 검증 경계는 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)과 [`evaluate_nested`](../../src/pipeline/ensemble.py)을 본다.
5. 개별 성공과 실패는 각 절에서 연결한 원 실험 이슈를 본다.

## 결론

발표의 숫자 중심 줄거리는 `0.96270에서 출발한 단일 구성 탐색`, `0.97038까지 올라간 검증된 결합`, `0.97109와 14위라는 공식 결과`의 세 층으로 나눈다.
가장 큰 성공 수치는 자료형 결함 복원의 `+0.00461`이고, 가장 설명하기 좋은 아이디어 성공은 수치와 범주 복제 병행의 `+0.00329`다.
가장 좋은 실패 교훈은 약한 예측 120개 계열의 `-0.000057`과 문턱 미달 327열 결합이다.
이 구분을 지키면 작은 차이를 부풀리지 않으면서도, 결과를 보기 전에 판단 규칙을 고정하고 다른 오차를 내는 예측만 조립했다는 핵심 이야기를 정확하게 전달할 수 있다.
