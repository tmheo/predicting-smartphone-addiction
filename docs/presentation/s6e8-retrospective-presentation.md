이 문서는 35개 화면 전체 제작에 앞서 문장 밀도, 정보 위계, 발표자 메모, Confluence 보충 설명과 시각 자료 표현을 검토하기 위한 대표 화면 시제품이다.
화면 08, 12, 21, 23, 24, 31, 35를 실제 원본 경로와 최종 자산 경로에 먼저 작성했다.

[기술 정의와 근거 찾아보기](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 08. 두 관점을 함께 쓰자 OOF AUC가 +0.00329 올랐습니다

수치를 그대로 두고 정확값 범주를 함께 보여 주자 같은 비교 기준보다 OOF AUC가 `+0.00329` 높아졌습니다.

![확대 눈금 0.958부터 0.968에서 일반 OOF 파랑 원 두 개가 0.96276과 0.96605에 놓이고, 두 점 사이에 AUC 차이 +0.00329가 표시된 차트](assets/screen-08-oof-score.png)

시각 자료 대체 설명: 일반 OOF AUC가 비교 기준 `0.96276`에서 수치와 정확값 범주를 함께 쓴 구성 `0.96605`로 올랐으며, 두 값의 차이는 `+0.00329`다.

### 발표자 메모

- OOF AUC는 각 행이 자기 정답으로 학습하지 않은 모델에서 받은 OOF 예측을 ROC AUC로 채점한 값입니다.
- 화면의 차이는 약 `0.0033`으로 읽고, Public 점수나 Private 점수로 이어지는 상승이라고 말하지 않습니다.

### Confluence 보충 설명

이 비교는 같은 자료 분할과 같은 LightGBM 설정에서 수치 열 아홉 개를 유지한 채 정확값 범주 복제 아홉 개를 추가한 단일 시드 직접 비교다.
비교 기준 `0.96276`은 이 짝비교의 기준 실행이며 첫 기준 실행을 소개할 때 쓰는 `0.96270`과 같은 값으로 줄이지 않는다.

근거: [전 피처 범주형 challenger 실험: 실행과 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)

## 화면 12. nested OOF는 바깥 fold를 봉인합니다

나머지 fold에서 구성원과 결합 방식을 고른 뒤, 보지 않은 바깥 fold에서 한 번 평가합니다.

![다섯 바깥 fold 가운데 하나를 봉인하고, 나머지 네 fold에서 구성원과 결합 방식을 고른 뒤 봉인한 fold에서 평가해 다섯 결과를 nested OOF로 잇는 흐름](assets/screen-12-nested-oof.png)

시각 자료 대체 설명: 다섯 바깥 fold를 차례로 하나씩 봉인하고, 남은 네 fold 안에서 선택한 결합을 봉인한 fold에 적용한 뒤 다섯 예측을 이어 nested OOF를 만든다.

### 발표자 메모

- 다섯 검사실 비유는 이 화면에서 한 번만 사용한 뒤 곧바로 바깥 fold와 안쪽 선택이라는 정식 표현으로 돌아옵니다.
- nested OOF는 구성원과 결합 방식 선택의 낙관을 줄이지만 기초 구성 생성 전의 모든 과거 선택까지 되풀이한 완전한 중첩 평가는 아닙니다.

### Confluence 보충 설명

한 바깥 fold의 목표값과 예측은 구성원 및 가중치 선택에 들어가지 않는다.
선택은 나머지 바깥 fold의 OOF만 사용하고, 선택 결과를 봉인한 fold에 적용해 얻은 예측을 다섯 번 이어 붙인다.

근거: [실험 채택 판정 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md), [결합 평가 구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)

## 화면 21. RealMLP 자료형 결함 수정이 +0.00461을 만들었습니다

가장 큰 단일 구성 상승은 새 복잡성을 더한 것이 아니라 입력 값의 의미와 구현을 맞춘 결과였습니다.

![잘못된 float32 변환 뒤 어휘 매핑과 올바른 어휘 매핑 뒤 float32 변환을 나란히 비교해 미등록값이 800896개에서 23개로 줄고 일반 OOF AUC가 0.96371에서 0.96832로 오른 흐름](assets/screen-21-realmlp-fix.png)

시각 자료 대체 설명: 어휘 매핑 전에 값을 `float32`로 바꾸던 순서를 고치자 검증 미등록값이 `800,896`개에서 `23`개로 줄었고, 3시드 평균 OOF AUC가 `0.9637131967`에서 `0.9683223458`로 회복됐다.

### 발표자 메모

- 화면에서는 `+0.00461`만 읽고 미등록값 수는 왜 영향이 컸는지 질문이 나올 때 설명합니다.
- 새로운 모델을 추가한 결과가 아니라 같은 RealMLP 이식판의 자료형 변환 순서 하나를 고친 짝비교입니다.

### Confluence 보충 설명

수정판은 어휘 매핑과 분위 구간 변환을 끝낸 뒤 `float32` 변환을 적용한다.
결함판과 수정판은 이 차이 외의 설정을 고정했고 같은 실행 환경 등급에서 난수 42, 43, 44를 짝지어 비교했다.

근거: [exp124 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp124_realmlp_dtype_fix.yaml), [자료형 정합 복원 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/243#issuecomment-5343200265)

## 화면 23. 근거가 약한 탐색은 일찍 멈췄습니다

후보 수나 새로움이 아니라 미리 정한 1차 기준, 반복 근거와 전체 결합 기여로 탐색을 중단했습니다.

![Lookup 설정 17개, 새 신경망 네 종류, 외부 예측 120개 계열과 327열 결합을 점추정, 반복 근거, 사전 관문과 결론 다섯 열에서 비교하고 중단 또는 미채택으로 표시한 표](assets/screen-23-stop-table.png)

시각 자료 대체 설명: Lookup 설정 17개와 새 신경망 네 종류는 첫 fold의 진입 기준을 넘지 못해 중단했고, 외부 예측 120개 계열과 327열 결합은 전체 결합 기여 또는 반복 근거가 부족해 미채택했다.

### 발표자 메모

- Lookup 설정과 새 신경망은 fold 0 진입 진단에서 멈췄으므로 전체 5분할 결과처럼 말하지 않습니다.
- `중단`은 더 큰 실행을 열지 않았다는 뜻이고 `미채택`은 완성된 비교를 최종 구성에 넣지 않았다는 뜻입니다.

### Confluence 보충 설명

Lookup-Transformer 설정 17개는 학습률, 학습률 일정과 최적화 알고리즘을 바꿨지만 fold 0 기준보다 모두 낮았다.
약한 외부 예측 120개 계열의 한계 기여는 `-0.000057`이었고, 327열 결합은 점추정이 `+0.0000047` 높았지만 바깥 fold 다섯 곳 중 세 곳에서만 같은 방향이라 사전 관문을 넘지 못했다.

근거: [Lookup-Transformer 제한 탐색 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/160#issuecomment-5308772959), [확장 사다리 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ladder-2.md), [327열 판정 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ext327/issue526/comparison.json)

## 화면 24. 실행 장소가 달라도 판정은 한 검수대로 모였습니다

실행 장소마다 역할은 달랐지만 정식 판정은 모두 로컬의 같은 반입과 재채점 절차를 통과했습니다.

![로컬 CPU, Kaggle CPU, Kaggle GPU, Vast.ai와 Runpod에서 나온 실행 기록 묶음이 중앙의 로컬 검수대로 모이고 해시 대조, 재채점과 판정을 거치는 흐름](assets/screen-24-execution-system.png)

시각 자료 대체 설명: 다섯 실행 장소에서 만든 실행 기록 묶음이 로컬 검수대로 모이며, 로컬은 해시 대조, 입력 경계 확인, 재채점과 최종 판정을 맡는다.

### 발표자 메모

- 로컬은 개발과 소규모 실행뿐 아니라 결과 반입, 재채점, 판정과 최종 조립을 맡았습니다.
- 장소 자체를 성공이나 실패로 칠하지 않고 모든 관문을 통과한 결론에만 채택 표식을 붙입니다.

### Confluence 보충 설명

한 비교 짝은 같은 공급자와 같은 실행 환경 등급에 묶었다.
서로 다른 공급자에서 완결된 비교 짝은 각각 같은 계약을 통과한 뒤에만 한 판정 입력에 함께 넣었으며, 한쪽 실행끼리 이어 붙이지 않았다.

근거: [발표용 실행 환경과 전환 사건 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md)

## 화면 30. 자체 35개에서 최종 314개 예측 열까지 갔습니다

자체 35개 결합의 nested OOF AUC `0.96981`에서 최종 314개 예측 열의 `0.97038`까지 내부 결합 점수가 높아졌습니다.

![확대 눈금 0.9695부터 0.9705에서 nested OOF 청록 네모 두 개가 자체 35개 결합 0.96981과 최종 314개 예측 열 0.97038에 놓이고, 시간순 차이 +0.00057과 직접 효과가 아니라는 제한이 표시된 차트](assets/screen-30-internal-assembly-score.png)

시각 자료 대체 설명: 자체 35개 결합의 nested OOF AUC `0.96981`과 최종 314개 예측 열의 `0.97038`을 같은 결합 내부 점수 눈금에 놓되, 두 값 사이에 여러 변경이 있었음을 함께 표시한다.

### 발표자 메모

- 두 점은 시간순 출발점과 도착점이며 외부 예측 추가 하나만의 직접 효과가 아닙니다.
- 중간에는 자체 풀이가 35개에서 36개로 바뀌었고 외부 예측 검수와 결합 설정 선택도 함께 있었습니다.

### Confluence 보충 설명

결합 내부 점수 계보는 자체 35개, 외부 207개를 더한 242열, 해로운 외부 계열을 뺀 313열, 결합 규제를 내부에서 고른 313열과 최종 314열로 이어진다.
각 단계의 구성과 선택 절차가 함께 달라졌으므로 출발점과 도착점의 차이를 한 기법의 인과 효과로 읽지 않는다.

근거: [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md), [314열 재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md)

## 화면 31. 314개 예측 열은 검수해 남긴 조립 재료입니다

자체 36개와 외부 278개의 예측 열을 무결성과 nested OOF 판정으로 검수해 최종 314열 조립 입력으로 남겼습니다.

![자체 예측 36열과 외부 예측 278열이 무결성 검수와 nested OOF 판정을 통과해 314열 조립판으로 들어가는 깔때기](assets/screen-31-final-assembly.png)

시각 자료 대체 설명: 자체 예측 36열과 외부 예측 278열이 열 순서, 행 정렬, 해시와 라이선스 장부 검수 및 nested OOF 판정을 지나 최종 314열 결합 입력이 된다.

### 발표자 메모

- 314개는 모델 수가 아니라 출처별 예측값 열 수입니다.
- 외부 278개를 모두 우리가 다시 학습했다는 뜻이 아니며, 자체 36개만 전체 자료로 다시 학습했습니다.

### Confluence 보충 설명

외부 구성원의 분할 근거와 라이선스 수준은 완전히 같지 않다.
외부 278개 가운데 64개는 사용 한정으로 분류했고, 이 한계는 최종 구성원 장부와 기술 근거 부록에 남겼다.

근거: [우리 최종 해법과 제출 계보 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md), [314열 재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md)

## 화면 32. 공식 결과는 Public 0.97135, Private 0.97109, 최종 14위였습니다

내부 검증으로 고른 314개 예측 열 제출은 공식 대회에서 Public 점수 `0.97135`, Private 점수 `0.97109`와 최종 14위를 기록했습니다.

![공식 대회 결과 패널에 Public 점수 0.97135를 호박 마름모로, Private 점수 0.97109를 보라 별로, 최종 14위를 별도 칸으로 표시한 그림](assets/screen-01-official-result.png)

시각 자료 대체 설명: 같은 제출 `55907610`의 Public 점수 `0.97135`, Private 점수 `0.97109`와 최종 순위 14위를 서로 구분된 세 칸에 표시한다.

### 발표자 메모

- Public 점수와 Private 점수는 같은 제출을 서로 다른 공식 표본에서 채점한 결과입니다.
- 결합 규제를 내부에서 고른 313열 판과 최종 314열 판의 두 공식 표시값은 같았으므로 마지막 변경이 공식 점수를 높였다고 말하지 않습니다.

### Confluence 보충 설명

내부 nested OOF와 공식 점수는 평가한 행과 선택 경계가 달라 그 차이를 개선량으로 계산할 수 없다.
마지막으로 올린 327열 제출은 Private 점수 `0.97108`이었으므로 최종 14위 성적을 만든 제출은 마지막 업로드가 아니라 314열 제출이다.

근거: [우리 최종 해법과 제출 계보 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md), [최종 조립 제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json)

## 화면 33. 남은 차이는 더 강한 단일 모델 탐색이었습니다

이 회고에서는 우승권과의 차이를 더 강한 단일 모델을 더 빨리 찾는 탐색 역량의 차이로 해석합니다.

![고정 fold와 사전 중단 관문, 서로 다른 오차와 nested OOF 조립, 공통 검수 절차라는 기반 위에 더 넓고 이른 단일 모델 탐색을 다음 보강 축으로 올리고, 우리와 1등의 단일 구성 점수는 직접 비교가 아님을 표시한 그림](assets/screen-33-search-gap.png)

시각 자료 대체 설명: 우리가 지킬 검증과 조립 기반 옆에 더 넓고 이른 단일 모델 탐색을 보강 축으로 세우며, 우리 최고 자체 단일 구성과 1등 글의 RealMLP 점수는 검증 명세가 달라 직접 비교하지 않는다고 표시한다.

### 발표자 메모

- 공개 기록에서 1등 RealMLP의 바깥쪽 예측 AUC는 `0.970706453`이고 우리 최고 자체 단일 구성의 OOF AUC는 `0.9694062694`지만 검증 명세가 달라 점수 차이를 직접 계산하지 않습니다.
- 1등 RealMLP의 최종 Private 점수와 전체 재현 명세는 공개되지 않았으므로 이 해석을 인과적으로 증명한 결론처럼 말하지 않습니다.

### Confluence 보충 설명

1등 글에서 확인되는 것은 높은 단일 RealMLP 바깥쪽 예측 AUC, 449개까지 늘어난 결합과 여러 에이전트의 병렬 탐색이다.
분할표, 전처리 경계, 설정, 구성원 장부와 결합식은 공개되지 않았으므로 이 화면은 확인된 사실에서 도출한 회고 해석이지 같은 조건의 대조 결과가 아니다.

근거: [1등 해법 원문 조사](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-first-place-writeup.md), [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md)

## 화면 34. 넓은 단일 모델 탐색을 일찍 엽니다

다음 대회에서는 넓은 단일 모델 탐색을 일찍 시작하되 고정 fold와 사전 중단 관문은 유지합니다.

![다음 대회 초반에는 넓은 단일 모델 탐색, 중반에는 고정 fold와 중단 관문, 후반에는 서로 다른 오차 조립을 배치하고 더 많은 실험이 아니라 더 이른 탐색과 더 빠른 중단이라고 표시한 시간선](assets/screen-34-next-competition-timeline.png)

시각 자료 대체 설명: 다음 대회 초반에 서로 다른 작동 원리의 단일 모델을 넓게 탐색하고, 고정 fold와 중단 관문을 거쳐 통과한 서로 다른 오차만 후반에 조립하는 순서를 보여 준다.

### 발표자 메모

- 더 많은 실험을 무조건 수행한다는 뜻이 아니라 강한 후보를 일찍 넓게 찾고 작은 근거로 빨리 멈추는 구조입니다.
- 결과를 보기 전에 분할과 중단 기준을 고정하고, 혼자 잘하는가와 함께할 때 돕는가를 나누는 원칙은 바꾸지 않습니다.

### Confluence 보충 설명

이 시간선은 이번 회고에서 도출한 다음 대회 권고이며 이미 확정한 실험 프로그램이나 우승 해법을 그대로 복제하겠다는 계획이 아니다.
새 후보의 구체적인 범위와 자원 배분은 다음 대회의 자료와 제약을 확인한 뒤 별도 결정한다.

근거: [1등 해법 원문 조사](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-first-place-writeup.md), [발표 제작 장부](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/presentation/s6e8-retrospective-production-ledger.md)

## 화면 35. 작게 검증하고, 다르게 틀리는 예측을 모으고, 같은 검수대로 조립합니다

작은 검증, 서로 다른 오차와 공통 검수대라는 세 원칙이 이번 결과를 설명합니다.

![작게 검증하기, 다르게 틀리는 예측 모으기, 같은 검수대로 조립하기라는 세 조각이 최종 14위 결과로 모이고 다음에는 넓은 단일 모델 탐색을 더 일찍 연다는 방향으로 이어지는 닫힌 고리](assets/screen-35-conclusion.png)

시각 자료 대체 설명: 작은 검증, 서로 다른 오차, 공통 검수대라는 세 조각이 최종 14위 결과로 모이며, 다음 대회에서는 같은 원칙 위에서 넓은 단일 모델 탐색을 더 일찍 시작한다.

### 발표자 메모

- 새로운 수치는 덧붙이지 않고 앞에서 본 RealMLP 수정, Lookup-Transformer, 실행 기록 묶음과 314열 조립을 한 문장씩 회수합니다.
- 마지막 문장은 질의응답을 여는 질문으로 바꿔 읽습니다.

### Confluence 보충 설명

작은 검증은 결과를 보기 전에 비교와 중단 관문을 고정하는 습관을 뜻한다.
서로 다른 오차는 단독 점수만 높이는 것이 아니라 기존 예측과 결합했을 때 보완하는 예측을 남긴다는 뜻이며, 공통 검수대는 실행 장소와 상관없이 같은 무결성 및 재채점 절차를 적용한다는 뜻이다.

근거: [발표 제작 장부](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/presentation/s6e8-retrospective-production-ledger.md)
