이 문서는 `A 발표 우선` 형식으로 구성한 35개 화면의 Markdown 발표 원본입니다.
각 화면은 청중용 제목, 핵심 문장 하나, 시각 자료 하나, 발표자 메모, Confluence 보충 설명과 근거 연결로 구성하며 모든 화면에서 같은 정보 위계를 유지합니다.

[기술 정의와 근거 찾아보기](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 01. Private 0.97109, 최종 14위에서 시작합니다

공식 Private 점수 `0.97109`로 최종 14위에 올랐습니다.
오늘은 성적 자체보다 어떤 방식으로 이 결과를 냈는지 되짚습니다.

![공식 최종 결과 14위와 같은 제출의 Public 점수 0.97135, Private 점수 0.97109를 호박 마름모와 보라 별로 분리한 결과 패널](assets/screen-01-official-result.png)

시각 자료 대체 설명: 공식 최종 순위는 14위입니다.
같은 제출의 Public 점수 `0.97135`와 최종 순위를 정한 Private 점수 `0.97109`는 서로 다른 표식과 설명으로 구분되어 있습니다.

### 발표자 메모

- `0.97109`는 대회 종료 뒤 최종 순위를 정한 Private 점수이고, `0.97135`는 같은 제출의 Public 점수입니다.
- 두 점수는 내부 검증 점수와 연결하지 않습니다.
  다음 화면에서는 같은 12개 변수로 이 결과를 낸 세 축을 소개합니다.

### Confluence 보충 설명

Public 점수와 Private 점수는 같은 제출을 서로 다른 시험 표본에서 채점한 결과입니다.
두 값의 차이로 개별 실험의 개선량이나 일반화 효과를 역산하지 않습니다.

근거: [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md), [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md)

부록: [A. 공식 결과와 자료 범위](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 02. 검증 가능한 작은 실험을 조립한 결과였습니다

14위라는 성적은 작은 실험과 서로 다르게 틀리는 예측을 공통 검수 절차로 조립해 얻은 결과였습니다.
비밀 모델 하나에서 나온 결과는 아니었습니다.

![검증 가능한 작은 실험, 서로 다르게 틀리는 예측, 실험 실행 체계라는 세 길이 Private 0.97109와 14위 결과로 모이는 지도](assets/screen-02-three-part-route.png)

시각 자료 대체 설명: 작은 변화를 비교하고 함께 쓸 때 서로 보완하는 예측을 남기며 여러 실행 장소에서 나온 결과를 한 검수대로 모았습니다.
이 세 축이 최종 14위 결과로 합쳐집니다.

### 발표자 메모

- 실험 실행 체계는 같은 실험 명세를 여러 장소에서 실행하고 그 결과를 로컬의 같은 검수 절차로 모으는 전체 구조라고 처음 설명합니다.
- 이 세 축을 전반과 후반의 길 안내판으로 사용합니다.
  하나의 모델이나 한 번의 큰 실험이 결과를 냈다고 말하지 않습니다.

### Confluence 보충 설명

작은 실험은 화면 07부터 23까지, 서로 다른 오차는 화면 13부터 31까지, 공통 검수 절차는 화면 17부터 31까지 실제 사례로 이어집니다.
세 축은 최종 결과를 사후에 설명하기 위한 회고 구조이며 하나의 비밀 모델을 가리키지 않습니다.

근거: [우리 최종 해법과 제출 계보 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md), [발표용 실행 환경과 전환 사건 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md)

부록: [E. 실험 실행 체계, G. 서로 다른 오차와 Lookup-Transformer, H. 최종 314개 예측 열](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 03. 12개 생활 습관 변수로 중독 위험의 순서를 예측했습니다

식별자와 목표값을 제외한 12개 생활 습관 변수로 사람들의 스마트폰 중독 위험 순서를 예측했습니다.

![한 사람의 12개 생활 습관 변수가 0부터 1 사이의 중독 위험도 하나로 바뀌고 여러 사람이 위험도가 높은 순서로 놓이는 흐름](assets/screen-03-risk-order.png)

시각 자료 대체 설명: 한 사람의 나이, 화면 사용, 수면, 알림 등 12개 입력이 위험도 하나로 바뀝니다.
모든 사람의 위험도를 정렬해 높은 사람부터 낮은 사람까지 순서를 매깁니다.

### 발표자 메모

- 자료 규모를 묻는 질문이 나오면 학습 자료는 약 69만 건, 예측 자료는 약 30만 건이라고 말합니다.
- 이 값은 대회의 목표값을 기준으로 사람들의 순서를 예측한 결과입니다.
  개인을 진단하거나 실제 임상 위험을 판정한 결과라고 말하지 않습니다.

### Confluence 보충 설명

학습 자료는 `691,369`행, 시험 자료는 `296,302`행이며 이진 목표값은 `addicted_label`입니다.
12개 입력 열의 정확한 이름과 자료 범위는 기술 근거 부록 A에서 확인할 수 있습니다.

근거: [첫 기준 실행 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp001_lgbm_baseline.yaml), [첫 기준 실행 종결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/18#issuecomment-5239693077)

부록: [A. 공식 결과와 자료 범위](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 04. ROC AUC는 순서 전체를 봅니다

ROC AUC는 임의로 고른 중독 한 명에게 비중독 한 명보다 높은 위험도를 준 비율입니다.

![0.5 무작위 순서부터 1.0 완전한 순서까지의 기준 막대와 중독 한 명 및 비중독 한 명의 위험도를 비교해 올바른 순서 비율을 구하는 흐름](assets/screen-04-roc-auc.png)

시각 자료 대체 설명: 중독 한 명과 비중독 한 명을 짝지어 누가 더 높은 위험도를 받았는지 가능한 모든 쌍에서 확인합니다.
이 가운데 순서가 올바른 쌍의 비율이 ROC AUC입니다.

### 발표자 메모

- `0.5`는 무작위 순서, `1.0`은 완전한 순서라는 전체 맥락을 먼저 보여 줍니다.
- ROC AUC는 정답 문턱 하나의 정확도가 아니라 위험 순서 전체를 평가하는 값이라고 설명합니다.

### Confluence 보충 설명

여기서는 이진 목표값의 쌍 순서로 점수를 해석합니다.
이 해석으로 구한 값은 ROC 곡선 아래 넓이와 같습니다.
동점 처리 방식과 실제 점수 계산 위치는 기술 근거 부록 B의 원본으로 연결합니다.

근거: [비전문가 발표의 기술 용어와 표기 원칙](https://github.com/tmheo/predicting-smartphone-addiction/issues/579#issuecomment-5489318781), [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md)

부록: [B. 점수와 검증 경계](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 05. 청중 참여 1에서 더 좋은 위험 순서를 고릅니다

실제 중독 여부를 본 뒤 A와 B 가운데 더 좋은 위험 순서를 직접 골라 봅니다.

![중독인 사람 가와 다, 비중독인 사람 나와 라를 위험도가 높은 순서로 배열한 A와 B를 답이나 점수 없이 나란히 보여 주는 투표 화면](assets/screen-05-risk-order-vote.png)

시각 자료 대체 설명: A는 가, 나, 다, 라 순서이고 B는 가, 다, 나, 라 순서입니다.
청중에게 어느 위험 순서가 더 좋은지 손들어 선택하게 합니다.

### 발표자 메모

- 손들기나 짧은 구두 투표로 선택만 받고 점수 계산은 요구하지 않습니다.
- 답을 먼저 말하지 않습니다.
  한두 명에게 선택한 이유를 물은 뒤 다음 화면에서 같은 위치를 유지한 채 결과를 공개합니다.

### Confluence 보충 설명

네 사람과 두 순서는 ROC AUC의 작동 원리를 설명하기 위한 교육용 예시이며 실제 대회 행에서 뽑은 표본은 아닙니다.
중독 두 명과 비중독 두 명이 있으므로 비교할 수 있는 쌍은 모두 네 개입니다.

근거: [비전문가용 핵심 개념 설명 장면](https://github.com/tmheo/predicting-smartphone-addiction/issues/570#issuecomment-5488821634)

부록: [B. 점수와 검증 경계의 교육용 네 사람 예시](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 06. B는 네 쌍 모두에서 중독 한 명을 더 위에 놓았습니다

교육용 예시에서 A는 네 쌍 중 세 쌍, B는 네 쌍 모두를 올바른 순서로 놓았습니다.

![앞 화면과 같은 위치의 A와 B 위험 순서 아래에 A는 동그라미 세 개와 가위표 한 개 및 네 쌍 중 3쌍, B는 동그라미 네 개 및 네 쌍 중 4쌍이라고 공개한 결과](assets/screen-06-risk-order-answer.png)

시각 자료 대체 설명: A 순서는 중독과 비중독을 짝지은 네 쌍 가운데 세 쌍을 올바르게 놓았습니다.
B 순서는 네 쌍 모두를 올바르게 놓았으므로 B가 더 좋은 위험 순서입니다.

### 발표자 메모

- 앞 화면에서 받은 선택을 확인한 뒤, B가 더 좋은 이유를 네 쌍의 순서만으로 설명합니다.
- 교육용 네 사람의 계산을 실제 대회 점수라고 말하지 않습니다.
  ROC AUC가 순서 전체를 본다는 뜻만 다시 짚습니다.

### Confluence 보충 설명

동점이 없는 이 교육용 예시에서 A의 쌍 순서 비율은 `3/4`, B는 `4/4`입니다.
실제 대회의 ROC AUC는 같은 원리를 학습 자료 전체의 양성 행과 음성 행으로 만들 수 있는 모든 쌍에 적용해 계산합니다.

근거: [비전문가용 핵심 개념 설명 장면](https://github.com/tmheo/predicting-smartphone-addiction/issues/570#issuecomment-5488821634)

부록: [B. 점수와 검증 경계의 교육용 쌍 계산표](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 07. 같은 숫자에도 두 관점이 있습니다

같은 수치 열을 연속적인 크기로도 보고 정확히 같은 값끼리 묶은 범주로도 볼 수 있습니다.

![예시 값 7.5시간이 순서와 거리를 보는 연속적인 크기 렌즈와 같은 값끼리 묶는 정확한 값 범주 렌즈로 갈라지는 그림](assets/screen-07-two-views.png)

시각 자료 대체 설명: 같은 `7.5시간`은 `7.4 < 7.5 < 7.6`처럼 순서와 거리로 볼 수 있습니다.
동시에 `[7.5시간]`이라는 이름표를 붙여 정확히 같은 값끼리 묶을 수도 있습니다.

### 발표자 메모

- 새 입력 변수를 수집한 것이 아니라, 같은 숫자를 두 가지 방식으로 모델에 보여 준 실험이라고 설명합니다.
- 범주 표현은 같은 값끼리 묶는 신호를 주고, 수치 표현은 순서와 거리 정보를 보존한다고 구분합니다.

### Confluence 보충 설명

전체 12개 입력 가운데 원래 수치 열 아홉 개는 그대로 유지했습니다.
여기에 각 열에서 값이 같은 항목끼리 묶는 범주 복제 열 아홉 개를 추가했습니다.
대상 열과 두 설정의 정확한 차이는 기술 근거 부록 C에서 확인할 수 있습니다.

근거: [같은 값의 범주 복제 실험 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp003_categorical_copies.yaml), [수치와 범주 복제 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)

부록: [C. 같은 값을 표현한 실험](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 08. 두 관점을 함께 쓰자 OOF AUC가 +0.00329 올랐습니다

수치를 그대로 두면서 같은 값끼리 묶는 범주 표현을 함께 쓰자 OOF AUC가 같은 비교 기준보다 `+0.00329` 높아졌습니다.

![확대 눈금 0.958부터 0.968에서 일반 OOF 파랑 원 두 개가 0.96276과 0.96605에 놓이고, 두 점 사이에 AUC 차이 +0.00329가 표시된 차트](assets/screen-08-oof-score.png)

시각 자료 대체 설명: 일반 OOF AUC는 비교 기준 `0.96276`에서 수치와 같은 값끼리 묶는 범주 표현을 함께 쓴 구성 `0.96605`로 올랐습니다.
두 값의 차이는 `+0.00329`입니다.

### 발표자 메모

- OOF AUC는 각 행이 자기 정답으로 학습하지 않은 모델에서 받은 OOF 예측을 ROC AUC로 채점한 결과입니다.
- 화면의 차이는 약 `0.0033`으로 읽습니다.
  Public 점수나 Private 점수도 그만큼 올랐다고 말하지 않습니다.

### Confluence 보충 설명

이 비교에서는 같은 자료 분할과 같은 LightGBM 설정을 사용했습니다.
수치 열 아홉 개를 유지한 채 같은 값을 묶는 범주 복제 열 아홉 개를 추가한 단일 시드 직접 비교입니다.
비교 기준 `0.96276`은 이 직접 비교의 기준 실행입니다.
첫 기준 실행을 소개할 때 쓰는 `0.96270`과 같은 값으로 줄이지 않습니다.

근거: [전 피처 범주형 challenger 실험: 실행과 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)

부록: [C. 같은 값을 표현한 실험](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 09. 전부 범주로만 쓰자 AUC 차이 -0.00417이었습니다

숫자의 순서 정보를 없애고 전부 범주로만 쓰자 OOF AUC가 같은 비교 기준보다 `-0.00417` 낮아졌습니다.

![같은 자료 분할과 LightGBM에서 수치만 0.96276, 전부 범주로만 0.95859, 수치와 같은 값끼리 묶은 범주 병행 0.96605를 세 행으로 비교하고 전부 범주 행에 AUC 차이 -0.00417과 중단을 표시한 표](assets/screen-09-categorical-only.png)

시각 자료 대체 설명: 수치만 쓴 비교 기준의 일반 OOF AUC는 `0.96276`, 전부 범주로만 쓴 값은 `0.95859`, 수치와 같은 값끼리 묶는 범주 표현을 함께 쓴 값은 `0.96605`입니다.
전부 범주로만 쓴 변형은 차이가 `-0.00417`이어서 중단했습니다.

### 발표자 메모

- 범주 표현 자체가 나쁜 것은 아닙니다.
  수치의 순서 정보를 모두 없앤 데서 손실이 생겼다고 범위를 분명히 합니다.
- 어느 한 표현만 고르기보다 서로 보완하는 두 관점을 함께 보존한 것이 답이었다고 설명합니다.

### Confluence 보충 설명

전부 범주화하는 변형과 수치를 유지하면서 범주를 복제하는 변형은 결과를 보기 전에 같은 이슈에서 고정한 두 직접 비교입니다.
두 변형은 같은 자료 분할, 같은 LightGBM 설정, 같은 난수 42를 사용했습니다.

근거: [전부 범주화 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp002_all_categorical.yaml), [수치와 범주 복제 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228)

부록: [C. 같은 값을 표현한 실험](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 10. 모든 실험은 같은 다섯 fold에서 비교했습니다

목표값 비율을 유지해 나눈 다섯 fold를 결과 확인 전에 고정했고 모든 실험 비교에서 같은 분할을 사용했습니다.

![학습 자료를 목표값 비율을 유지한 fold 다섯 개로 나누고 결과 확인 전에 고정한 뒤 모든 실험이 같은 다섯 묶음을 사용하는 흐름](assets/screen-10-fixed-folds.png)

시각 자료 대체 설명: 학습 자료는 fold 1부터 fold 5까지 다섯 묶음으로 나뉩니다.
`결과 확인 전 고정` 단계를 거친 뒤, 이후의 모든 실험 비교가 같은 분할을 사용합니다.

### 발표자 메모

- 결과를 보기 전에 자료를 같은 다섯 묶음으로 나눴습니다.
  이때 자료 묶음 하나를 fold라고 부릅니다.
- fold는 자료 묶음 하나이며 모델 다섯 개를 뜻하지 않습니다.

### Confluence 보충 설명

목표값 비율을 유지하는 5분할을 `shuffle=True`, 난수 42로 한 번 만들었습니다.
커밋한 분할 파일은 이후 실행에서 읽기만 하며 실험 결과에 맞춰 다시 나누지 않습니다.

근거: [분할 생성 코드](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/scripts/make_folds.py), [실험 채택 판정 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)

부록: [B. 점수와 검증 경계의 고정 분할](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 11. OOF는 자기 정답을 보지 않은 모델의 예측입니다

각 학습 행은 자신의 정답으로 학습하지 않은 모델에서 OOF 예측 하나를 받았습니다.

![fold 하나를 검증용으로 비우고 나머지 네 fold로 학습해 비운 fold의 OOF 예측을 만든 뒤, 이 과정을 다섯 번 반복해 원래 행 순서로 잇는 흐름](assets/screen-11-oof-predictions.png)

시각 자료 대체 설명: fold 3을 검증용으로 비운 예에서는 fold 1, 2, 4, 5로 학습하고 fold 3의 OOF 예측을 만듭니다.
같은 과정을 다섯 fold에 반복하고 OOF 조각을 원래 행 순서로 이어 전체 OOF 예측을 만듭니다.

### 발표자 메모

- OOF는 예측값이며, 이 예측을 ROC AUC로 채점한 값이 OOF AUC입니다.
- 각 행의 목표값은 그 행을 예측한 모델의 학습에 들어가지 않습니다.

### Confluence 보충 설명

각 차례에는 한 fold의 행을 검증용으로 두고 나머지 네 fold의 행만 모델 학습에 사용합니다.
검증 행의 예측을 원래 행 위치에 저장합니다.
이렇게 다섯 조각을 모두 채운 배열이 OOF 예측입니다.

근거: [교차검증 실행 구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/run.py), [OOF 채점 구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/cv.py), [실험 채택 판정 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md)

부록: [B. 점수와 검증 경계의 OOF 생성과 채점](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 12. nested OOF는 바깥 fold를 봉인합니다

나머지 fold에서 구성원과 결합 방식을 선택하고 따로 봉인해 둔 바깥 fold에서 한 번 평가합니다.

![다섯 바깥 fold 가운데 하나를 봉인하고, 나머지 네 fold에서 구성원과 결합 방식을 고른 뒤 봉인한 fold에서 평가해 다섯 결과를 nested OOF로 잇는 흐름](assets/screen-12-nested-oof.png)

시각 자료 대체 설명: 다섯 바깥 fold를 차례로 하나씩 봉인합니다.
남은 네 fold 안에서 선택한 결합을 봉인한 fold에 적용하고 다섯 예측을 이어 nested OOF를 만듭니다.

### 발표자 메모

- 다섯 검사실 비유는 이 화면에서 한 번만 사용하고 곧바로 바깥 fold와 안쪽 선택이라는 정식 표현으로 돌아옵니다.
- nested OOF는 구성원과 결합 방식을 선택할 때 생기는 낙관을 줄입니다.
  다만 기초 구성 생성 전에 이뤄진 과거의 모든 선택까지 반복하는 완전한 중첩 평가는 아닙니다.

### Confluence 보충 설명

한 바깥 fold의 목표값과 예측은 구성원이나 가중치를 선택할 때 사용하지 않습니다.
선택에는 나머지 바깥 fold의 OOF만 사용합니다.
선택 결과를 봉인한 fold에 적용해 얻은 예측을 다섯 번 이어 붙입니다.

근거: [실험 채택 판정 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md), [결합 평가 구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)

부록: [B. 점수와 검증 경계의 nested OOF 절차](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 13. 개인전 점수와 ensemble 기여는 다른 질문입니다

혼자 거둔 점수와 기존 예측에 더했을 때 전체 결과가 얼마나 좋아지는지는 서로 다른 기준으로 판단합니다.

![파랑 원으로 표시한 일반 OOF AUC의 개인전 검수와 청록 네모로 표시한 nested OOF AUC 차이의 팀 기여 검수를 나란히 비교하는 두 검수대](assets/screen-13-individual-team.png)

시각 자료 대체 설명: 왼쪽 검수대는 일반 OOF AUC로 `혼자 잘하는가`를 묻습니다. 오른쪽 검수대는 기존 예측에 더한 뒤의 nested OOF AUC 차이로 `함께할 때 돕는가`를 묻습니다.

### 발표자 메모

- ensemble은 여러 예측을 합치면서 개인전 점수와 팀에 들어왔을 때의 기여를 따로 살피는 과정입니다.
- 단독 점수가 최고가 아니어도 기존 예측과 다르게 틀리면 함께할 때 도움이 될 수 있습니다.

### Confluence 보충 설명

단일 구성 점수에서는 후보 하나의 일반 OOF AUC를 비교합니다.
결합 기여에서는 기존 구성원만 쓴 결합과 후보를 더한 결합을 같은 nested OOF 절차로 비교합니다.

근거: [Lookup-Transformer의 다양성 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965), [결합 평가 구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/ensemble.py)

부록: [B. 점수와 검증 경계, G. 서로 다른 오차와 Lookup-Transformer](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 14. 청중 참여 2에서 +0.0000469만 보고 결정합니다

전체 nested OOF AUC 차이 `+0.0000469` 하나만 보고는 채택 여부를 정할 수 없습니다.

![청록 네모 안의 nested OOF AUC 차이 +0.0000469만 공개하고 채택, 중단, 근거 더 보기 가운데 하나를 고르게 하는 참여 화면](assets/screen-14-evidence-choice.png)

시각 자료 대체 설명: 현재 공개된 근거는 전체 nested OOF AUC 차이 `+0.0000469` 하나뿐입니다. 청중은 `채택`, `중단`, `근거 더 보기` 중 하나를 고릅니다.

### 발표자 메모

- 손들기나 짧은 구두 투표로 세 선택 가운데 하나를 받습니다.
- 첫 답은 수치가 작아서가 아닙니다. 후보와 문턱, 평가 절차를 아직 보지 않았으므로 `근거 더 보기`입니다.

### Confluence 보충 설명

현재 풀과 제안 풀의 직접 nested OOF AUC를 비교하면 정확한 차이는 `+0.00004688661361140767`입니다.
이 값은 Public 점수나 Private 점수의 차이가 아닙니다. 전체 점추정의 부호만으로 채택 여부를 결정하지도 않았습니다.

근거: [결측 증강 전파 일괄 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/report.md), [기계 판독 판정 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/judgment.json)

부록: [D. 결측 증강 판정의 전체 점추정](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 15. 바깥 fold 다섯 곳에서 모두 같은 방향이었습니다

제안 풀은 전체 점수와 바깥 fold 다섯 곳에서 모두 양의 차이를 보였습니다.

![전체 nested OOF AUC 차이 +0.0000469 옆에 바깥 fold 1부터 5까지의 청록 네모가 모두 양수 방향으로 표시된 반복 근거](assets/screen-15-five-fold-wins.png)

시각 자료 대체 설명: 전체 nested OOF AUC 차이 `+0.0000469` 옆에서 바깥 fold 다섯 곳의 차이가 모두 양수로 표시됩니다. `5/5 양수`라는 반복 근거를 보여 줍니다.

### 발표자 메모

- 다섯 값은 고정한 바깥 fold에서 방향이 반복됐다는 뜻입니다. 독립된 새 대회를 다섯 번 치렀다는 뜻은 아닙니다.
- 반복 방향을 확인한 뒤에도 결과를 보기 전에 채택 조건을 정했는지 한 번 더 묻습니다.

### Confluence 보충 설명

발표에 표기한 바깥 fold 1부터 5는 기계 판독 기록의 0부터 4에 대응합니다. 직접 nested OOF AUC 차이는 각각 `+0.0000566500`, `+0.0000477462`, `+0.0000399849`, `+0.0000281048`, `+0.0000619473`이었습니다.
다섯 곳 모두 양수여서 분할 승수는 `5/5`였습니다.

근거: [직접 nested OOF 관문 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/direct-nested-gate.json), [결측 증강 전파 일괄 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/report.md)

부록: [D. 결측 증강 판정의 바깥 fold별 차이](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 16. 사전에 고정한 관문을 통과해 채택했습니다

결측 증강은 후보와 판정 절차를 결과 전에 고정했고 두 관문을 통과해 채택됐습니다.

![nested OOF AUC 차이 +0.0000469, 바깥 fold 5/5 양수, 결과 전에 고정한 두 관문 통과를 차례로 확인한 뒤 초록 원 채택에 도달하는 네 열 판정표](assets/screen-16-precommitted-gates.png)

시각 자료 대체 설명: 점추정과 반복 근거, 사전에 고정한 관문을 차례로 확인합니다. 마지막 열에만 초록 원으로 `채택` 결론을 표시합니다.

### 발표자 메모

- 결측 증강은 학습 자료의 관측값 일부를 추가로 가린 복제본을 보여 주어 값이 비어도 견디게 한 방법입니다.
- 채택 근거는 Public 점수가 아니라 선택과 평가를 분리한 두 OOF 관문의 통과 여부였습니다.

### Confluence 보충 설명

결과를 확인하기 전에 완결된 비교 짝과 후보 교체 단위, 중복 제한, 검색 절차와 두 채택 관문을 고정했습니다.
완결된 24개 비교 짝으로 1,658개 상태를 정확히 채점한 뒤 원본 구성 다섯 자리를 결측 증강판으로 바꿨습니다.
동결 OOF 조건부 절차 차이 `+0.000044152982`와 직접 nested OOF 차이 `+0.000046886614`가 모두 관문을 통과했습니다.

근거: [결측 증강 전파 일괄 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/report.md), [교정 종결 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/512#issuecomment-5472767484)

부록: [D. 결측 증강 판정의 두 관문과 최종 판정](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 17. 하나의 레시피, 여러 주방, 하나의 검수대

같은 실험 명세를 여러 장소에서 실행하되 결과는 로컬의 한 검수 절차로 모았습니다.

![하나의 고정된 실험 명세가 로컬, Kaggle CPU와 GPU, Vast.ai, Runpod으로 나뉘어 실행된 뒤 각 장소의 실행 기록 묶음이 로컬 검수대로 모이는 구조](assets/screen-17-one-recipe.png)

시각 자료 대체 설명: 하나로 고정한 실험 명세를 역할이 다른 다섯 실행 장소로 보냅니다. 각 장소에서 나온 실행 기록 묶음은 중앙의 로컬 검수대로 모입니다.

### 발표자 메모

- `하나의 레시피, 여러 주방, 하나의 검수대`라는 비유는 이 화면에서만 사용합니다.
- 역할이 다른 장소들이 같은 실행 명세와 중앙 검수 경계를 공유했다는 뜻입니다. 모든 장소에서 같은 실험을 복제했다는 의미는 아닙니다.

### Confluence 보충 설명

로컬은 개발과 소규모 실행, 중앙 반입, 재채점, 판정, 최종 조립을 맡았습니다.
Kaggle CPU는 고정 CPU 비교 짝의 병렬 실행을 맡았습니다. Kaggle GPU는 초반 정식 실행과 후반 호환성 확인 및 진단을, Vast.ai는 주 GPU 실행을, Runpod은 예비 GPU 실행을 맡았습니다.
한 비교 짝의 대조군과 후보군은 같은 공급자와 같은 실행 환경 등급에서 완결했습니다.

근거: [발표용 실행 환경과 전환 사건 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md)

부록: [E. 실험 실행 체계의 환경별 역할](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 18. 모든 결과는 실행 기록 묶음으로 다시 검수됐습니다

원격 결과는 실행 기록 묶음으로 회수했습니다. 해시와 OOF를 로컬에서 다시 검사한 결과만 판정에 사용했습니다.

![설정과 입력 고정, 실행 장소 배정, 공통 명령 실행, 실행 기록 묶음, SHA-256 대조, 입력 경계 확인, OOF 재채점, 판정과 보관 및 자원 정리의 여덟 단계 흐름](assets/screen-18-record-verification.png)

시각 자료 대체 설명: 결과를 확인하기 전에 설정과 입력을 고정하고 실행 장소에서 공통 명령을 실행합니다. 실행 기록 묶음은 SHA-256으로 대조하며 로컬에서 입력 경계와 OOF를 다시 검사한 결과만 판정에 사용합니다.

### 발표자 메모

- 실행 장소에서 로컬로 돌아온 것은 모델 자체가 아니라 설정과 예측, 지표, 진단을 함께 담은 실행 기록 묶음입니다.
- 원격에서 보고한 점수를 그대로 믿지 않고 로컬의 입력과 OOF로 다시 계산했습니다.

### Confluence 보충 설명

묶음을 반입할 때는 입력 해시, 출처 커밋, 커밋 시점 설정과 묶음 설정의 일치 여부, 깨끗한 코드 상태, 주장 지표의 재채점 결과를 검사합니다.
검사를 통과한 실행만 로컬 실행 저장소에서 정상 실행으로 재생합니다. 실패한 묶음은 격리하고 필요한 실행을 다시 수행합니다.
결과를 회수한 뒤에는 원격 계산 자원과 저장 공간을 삭제하고 과금이 중지됐는지 확인합니다.

근거: [실행 기록 묶음 반입 구현](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/src/pipeline/bundle.py), [발표용 실행 환경과 전환 사건 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md)

부록: [E. 실험 실행 체계의 실행 기록 묶음과 반입 검사](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 19. 휴식 동안 세 질문을 남깁니다

전반은 무엇을 예측했고 어떻게 평가했으며 실험 결과를 어떻게 믿었는지, 세 질문으로 정리합니다.

![무엇을 예측했는가, 어떻게 평가했는가, 실험 결과를 어떻게 믿었는가라는 세 질문과 휴식 뒤 다룰 성공과 중단, 실행 장소 변화, 최종 조립 예고](assets/screen-19-break-questions.png)

시각 자료 대체 설명: 전반에 다룬 문제와 평가, 신뢰를 세 질문으로 되짚습니다. 10분 휴식 뒤에는 성공과 중단, 실행 장소의 변화와 최종 조립을 다룬다고 예고합니다.

### 발표자 메모

- 새 사실은 설명하지 않고 10분 휴식 뒤 다시 시작할 시각만 안내합니다.
- 후반에는 잘된 실험과 일찍 멈춘 실험, 실행 장소의 변화와 최종 314개 예측 열 조립을 다룬다고 예고합니다.

### Confluence 보충 설명

첫 질문은 화면 03의 문제와 자료 범위로 돌아갑니다. 두 번째는 화면 04부터 16까지의 ROC AUC와 검증 경계, 세 번째는 화면 17과 18의 실행 체계로 돌아갑니다.
이 화면에서는 새 근거를 추가하지 않고 앞선 화면 묶음을 되짚습니다.

근거: [2시간 사내 대회 회고 이야기 흐름 설계서](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/presentation/s6e8-retrospective-story-flow.md)

부록: [A. 공식 결과와 자료 범위, B. 점수와 검증 경계, E. 실험 실행 체계](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 20. 성공과 실패를 같은 판정표에서 시작합니다

성공과 실패는 같은 판단 절차를 거쳐 내린 결론입니다.

![실험, 점추정 또는 진입 결과, 반복 근거, 사전 관문과 결론이라는 같은 다섯 칸으로 성공과 실패를 함께 읽는 빈 판정표](assets/screen-20-common-decision-table.png)

시각 자료 대체 설명: 모든 실험을 다섯 칸에 놓습니다. 무엇을 바꿨고 얼마나 달랐는지, 다시 확인했는지, 결과를 보기 전에 관문을 정했는지, 최종 결론은 무엇인지 보여 줍니다.

### 발표자 메모

- 성공은 자랑 목록으로, 실패는 비용 낭비 목록으로 따로 떼지 않습니다. 둘 다 다음 운영 원칙을 세운 근거로 다룹니다.
- `채택`, `중단`, `미채택`은 실제 판정입니다. 뒤의 세 화면에서 같은 다섯 칸을 채워 갑니다.

### Confluence 보충 설명

점추정 하나만으로 결론을 내리지 않습니다. 반복 근거와 결과를 확인하기 전에 정한 관문을 함께 읽습니다.
`근거 더 보기`는 앞선 참여 장면에서 판단을 미룬 선택입니다. 실제 실험 판정인 `미채택`과는 구분합니다.

근거: [차트와 다이어그램의 시각 문법](https://github.com/tmheo/predicting-smartphone-addiction/issues/569#issuecomment-5489136191), [발표 제작 장부의 공통 판정표](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/presentation/s6e8-retrospective-production-ledger.md#화면-20-성공과-실패를-같은-판정표에서-시작합니다)

부록: [F. 성공과 중단 사례](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 21. RealMLP 자료형 결함 수정이 +0.00461을 만들었습니다

새로운 복잡성을 더한 실험보다 입력 값의 의미와 구현을 맞춘 수정에서 가장 큰 단일 구성 상승이 나왔습니다.

![잘못된 float32 변환 뒤 어휘 매핑과 올바른 어휘 매핑 뒤 float32 변환을 나란히 비교해 미등록값이 800896개에서 23개로 줄고 일반 OOF AUC가 0.96371에서 0.96832로 오른 흐름](assets/screen-21-realmlp-fix.png)

시각 자료 대체 설명: 어휘 매핑 전에 값을 `float32`로 바꾸던 순서를 고치자 검증 미등록값이 `800,896`개에서 `23`개로 줄었습니다. 3시드 평균 OOF AUC도 `0.9637131967`에서 `0.9683223458`로 회복됐습니다.

### 발표자 메모

- 화면에서는 `+0.00461`만 읽습니다. 미등록값 수는 왜 영향이 컸는지 질문이 나올 때 설명합니다.
- 같은 RealMLP 이식판에서 자료형 변환 순서 하나를 고친 짝비교입니다. 새로운 모델을 추가한 결과는 아닙니다.

### Confluence 보충 설명

수정판은 어휘 매핑과 분위 구간 변환을 끝낸 뒤 `float32` 변환을 적용합니다.
결함판과 수정판은 이 차이 외의 설정을 고정했으며 같은 실행 환경 등급에서 난수 42, 43, 44를 짝지어 비교했습니다.

근거: [exp124 설정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/configs/exp124_realmlp_dtype_fix.yaml), [자료형 정합 복원 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/243#issuecomment-5343200265)

부록: [F. 성공과 중단 사례](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 22. 남긴 실험은 서로 다른 관점과 오차를 보탰습니다

남긴 실험은 같은 값의 다른 표현, 다른 오차, 결측에 견디는 학습 중 하나로 뚜렷한 기여를 했습니다.

![수치와 같은 값끼리 묶은 범주 표현, Lookup-Transformer와 결측 증강을 같은 다섯 칸 판정표에 놓고 각각 다른 정보 관점, 다른 오차와 결측 견고성이라는 이유로 채택한 표](assets/screen-22-adopted-experiments.png)

시각 자료 대체 설명: 수치와 같은 값끼리 묶은 범주 표현은 일반 OOF `+0.00329`, Lookup-Transformer는 일반 OOF `+0.00038`과 결합 기여 `+0.00025`를 근거로 채택됐습니다. 결측 증강은 nested OOF 약 `+0.0000469`와 바깥 fold `5/5` 양수를 근거로 채택됐습니다.

### 발표자 메모

- 세 방법은 각자 다른 기준과 평가 관문을 통과했습니다. 같은 설정이나 같은 효과 크기로 채택됐다는 뜻은 아닙니다.
- 상세 설정은 읽지 않고 `다른 정보 관점`, `다른 오차`, `결측 견고성`이라는 세 기여만 남깁니다.

### Confluence 보충 설명

수치와 같은 값끼리 묶은 범주 표현은 같은 분할과 설정에서 직접 비교했습니다. Lookup-Transformer는 단일 점수와 결합 기여를 따로 확인했습니다.
결측 증강은 후보 풀 전체를 대상으로 동결 OOF 조건부 절차와 직접 nested OOF의 두 관문을 통과했으며 바깥 fold 다섯 곳에서도 모두 같은 방향이었습니다.

근거: [수치와 같은 값끼리 묶은 범주 표현 직접 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/31#issuecomment-5242350228), [Lookup-Transformer 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965), [결측 증강 일괄 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/missingness-propagation-batch/issue512/report.md)

부록: [C. 같은 값을 표현한 실험, D. 결측 증강 판정, G. 서로 다른 오차와 Lookup-Transformer](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 23. 근거가 약한 탐색은 일찍 멈췄습니다

미리 정한 1차 기준과 반복 근거, 전체 결합 기여를 보고 탐색을 중단했습니다. 후보 수나 새로움은 중단 기준이 아니었습니다.

![Lookup 설정 17개, 새 신경망 네 종류, 외부 예측 120개 계열과 327열 결합을 점추정, 반복 근거, 사전 관문과 결론 다섯 열에서 비교하고 중단 또는 미채택으로 표시한 표](assets/screen-23-stop-table.png)

시각 자료 대체 설명: Lookup 설정 17개와 새 신경망 네 종류는 첫 fold의 진입 기준을 넘지 못해 중단했습니다. 외부 예측 120개 계열과 327열 결합은 전체 결합 기여 또는 반복 근거가 부족해 미채택했습니다.

### 발표자 메모

- Lookup 설정과 새 신경망은 fold 0 진입 진단에서 멈췄습니다. 전체 5분할 결과처럼 말하지 않습니다.
- `중단`은 더 큰 실행을 열지 않았다는 뜻입니다. `미채택`은 완성된 비교를 최종 구성에 넣지 않았다는 뜻입니다.

### Confluence 보충 설명

Lookup-Transformer 설정 17개는 학습률과 학습률 일정, 최적화 알고리즘을 바꿨지만 fold 0 기준보다 모두 낮았습니다.
약한 외부 예측 120개 계열의 한계 기여는 `-0.000057`이었습니다. 327열 결합은 점추정이 `+0.0000047` 높았지만 바깥 fold 다섯 곳 중 세 곳에서만 같은 방향이어서 사전 관문을 넘지 못했습니다.

근거: [Lookup-Transformer 제한 탐색 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/160#issuecomment-5308772959), [확장 사다리 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ladder-2.md), [327열 판정 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-ext327/issue526/comparison.json)

부록: [F. 성공과 중단 사례](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 24. 실행 장소가 달라도 판정은 한 검수대로 모였습니다

실행 장소마다 역할은 달랐지만 정식 판정은 모두 로컬의 같은 반입과 재채점 절차를 거쳤습니다.

![로컬 CPU, Kaggle CPU, Kaggle GPU, Vast.ai와 Runpod에서 나온 실행 기록 묶음이 중앙의 로컬 검수대로 모이고 해시 대조, 재채점과 판정을 거치는 흐름](assets/screen-24-execution-system.png)

시각 자료 대체 설명: 다섯 실행 장소에서 만든 실행 기록 묶음이 로컬 검수대로 모입니다. 로컬은 해시 대조와 입력 경계 확인, 재채점, 최종 판정을 맡습니다.

### 발표자 메모

- 로컬은 개발과 소규모 실행뿐 아니라 결과 반입, 재채점, 판정과 최종 조립을 맡았습니다.
- 실행 장소를 성공이나 실패로 나누지 않습니다. 모든 관문을 통과한 결론에만 채택 표식을 붙입니다.

### Confluence 보충 설명

한 비교 짝은 같은 공급자와 같은 실행 환경 등급에 묶었습니다.
서로 다른 공급자에서 완결된 비교 짝은 각각 같은 계약을 통과한 뒤에만 한 판정 입력에 함께 넣었습니다. 한쪽 실행끼리 이어 붙이지 않았습니다.

근거: [발표용 실행 환경과 전환 사건 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md)

부록: [E. 실험 실행 체계](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 25. 비용과 운영 경험에 따라 실행 장소를 바꿨습니다

실제 비용과 재고 접속 경험을 새 근거로 삼아 주 실행 장소와 예비 장소를 바꿨습니다.

![Kaggle GPU가 초반 정식 실행에서 후반 호환성 확인과 진단으로 역할이 바뀐 지점, Runpod과 Vast.ai 실제 비교, Runpod 우선 결정, Vast.ai 우선 전환과 Vast.ai 접속 준비 실패 뒤 Runpod 복구를 다섯 지점으로 잇는 시간선](assets/screen-25-provider-timeline.png)

시각 자료 대체 설명: Kaggle GPU는 초반에 정식 실행을 맡았지만 후반에는 호환성 확인과 진단에만 사용했습니다. Runpod `$0.24`와 Vast.ai `$0.12`를 실제로 비교한 뒤에는 준비 속도와 메모리 여유를 고려해 Runpod을 우선했습니다. 이후 재고와 운영 경험을 반영해 Vast.ai 우선으로 바꿨고 실제 접속 준비가 실패했을 때는 Runpod으로 전환했습니다.

### 발표자 메모

- Kaggle GPU는 초반에 정식 신경망 실행을 맡았습니다. 후반 운영 정책에서는 사람이 지켜보는 호환성 확인과 진단으로 역할을 좁혔습니다.
- 처음에는 준비 속도와 메모리 여유를 중시해 Runpod을 우선했습니다. Vast.ai가 처음부터 주 실행 장소였던 것은 아닙니다.
- Runpod은 실패한 서비스가 아니라 예비 장소였습니다. Vast.ai의 서로 다른 두 호스트에서 접속 준비가 실패하자 미리 정한 전환 규칙에 따라 그곳에서 실행을 마쳤습니다.

### Confluence 보충 설명

동일한 선별 실행에서 Runpod은 모델 실행에 26분 24초가 걸렸고 표시된 차감액은 `$0.24`였습니다. Vast.ai는 31분 45초가 걸렸고 `$0.12`가 차감됐습니다.
후속 운영에서는 적합한 Vast.ai 자원을 제때 확보하지 못하거나 서로 다른 두 호스트에서 접속과 사전 검사가 실패하면 비교 짝 전체를 Runpod으로 옮기도록 정했습니다. 두 공급자의 일부 결과를 이어 붙이지 않았습니다.

근거: [발표용 실행 환경과 전환 사건](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md), [Vast.ai 실패 뒤 Runpod 복구 기록](https://github.com/tmheo/predicting-smartphone-addiction/issues/108#issuecomment-5303015536)

부록: [E. 실험 실행 체계](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 26. $0.39는 최종 Vast.ai 재학습 작업 비용입니다

약 `$0.39`는 마지막에 바뀐 신경망 하나를 전체 자료로 재학습하는 데 든 비용이며 대회 전체 비용은 아닙니다.

![최종 Vast.ai 재학습 작업의 비용 0.39달러와 Lookup-Transformer 하나의 난수 42, 43, 44를 GPU 세 장에 배정한 범위, 학습 완료부터 결과 회수, 로컬 재검증, 자원 삭제와 과금 중지까지의 흐름](assets/screen-26-final-refit-cost.png)

시각 자료 대체 설명: Vast.ai에서 Lookup-Transformer 하나를 난수 42, 43, 44로 각각 설정해 GPU 세 장에서 전체 자료로 재학습했습니다. 잔액 차이는 정확히 `$0.393844836990070`입니다. 결과를 회수하고 로컬에서 다시 검증한 뒤에는 계산 자원과 별도 저장 공간을 모두 삭제했습니다.

### 발표자 메모

- 화면에서는 `$0.39`로 읽습니다. 이 금액을 대회 전체 원격 비용이나 전체 제출 비용, 모든 최종 모델의 재학습 비용으로 넓혀 말하지 않습니다.
- 인스턴스에는 RTX A4000 네 장이 있었지만 실제 모델 실행에는 같은 설정에서 난수 42, 43, 44를 맡은 세 장만 사용했습니다.

### Confluence 보충 설명

정확한 잔액 차이는 `$0.393844836990070`입니다.
앞선 검증과 실패, CPU 실행, 로컬 전력과 사람 시간은 이 비용에 포함되지 않습니다. 결과를 회수하고 로컬에서 다시 검증한 뒤에는 활성 계산 자원과 별도 저장 공간이 각각 0개임을 확인했습니다.

근거: [발표용 실행 환경과 비용 범위](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-environment-evidence.md#6-마지막-제출에서는-필요한-신경망-하나만-vastai에서-다시-학습했다), [최종 조립 실행 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/report.md)

부록: [E. 실험 실행 체계](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 27. 혼자 잘하는가와 함께할 때 돕는가를 두 번 시험했습니다

개인전 성능과 결합 기여는 서로 대신할 수 없으므로 두 가지 검수로 나누어 판단했습니다.

![개인전 검수에서 파란 원 일반 OOF로 혼자 얼마나 잘하는지 보고, 팀 기여 검수에서 청록 네모 nested OOF로 함께할 때 얼마나 돕는지 따로 평가하는 두 패널](assets/screen-13-27-dual-evaluation.png)

시각 자료 대체 설명: 왼쪽의 개인전 검수에서는 단일 구성의 예측을 고정 fold에서 일반 OOF로 평가합니다. 오른쪽의 팀 기여 검수에서는 봉인한 바깥 fold에서 결합 전후를 nested OOF로 평가합니다. 두 결과는 따로 기록합니다.

### 발표자 메모

- 일반 OOF와 nested OOF는 서로 다른 질문에 답합니다. 두 점수를 한 점수선으로 잇거나 서로 대신하지 않습니다.
- 단독 점수가 최고가 아니어도 기존 예측과 다른 오차를 보여 전체 결합을 높인다면 후보 풀에 남을 수 있습니다.

### Confluence 보충 설명

개인전 검수에서는 단일 구성의 재현성과 점수를 봅니다. 팀 기여 검수에서는 구성원과 결합 방식을 봉인한 바깥 fold에서 결합 전후를 비교합니다.
Lookup-Transformer의 실제 값은 다음 두 화면과 원 판정에서 이어서 확인합니다.

근거: [실험 채택 판정 계약](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md), [Lookup-Transformer 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965)

부록: [G. 서로 다른 오차와 Lookup-Transformer](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 28. 틀린 행이 다르면 함께할 이유가 생깁니다

단독 점수가 같더라도 서로 다른 행에서 순서를 틀리면 두 예측을 합칠 이유가 생길 수 있습니다.

![같은 여덟 행 위치에서 예측 A와 B가 각각 두 곳에 오차 표식을 가지며 2번 행의 오차는 겹치고 5번과 7번 행의 오차는 서로 다른 교육용 개념 예시](assets/screen-28-error-diversity-example.png)

시각 자료 대체 설명: 예측 A와 B는 오차 표식이 각각 두 개로 같지만 2번 위치만 겹치고 A의 5번과 B의 7번 위치는 달라, 두 예측이 서로 보완할 가능성을 보여 줍니다.

### 발표자 메모

- 이 그림은 실제 두 구성의 행별 오차율을 측정한 결과가 아닙니다. 같은 단독 점수와 서로 다른 오차를 설명하기 위한 교육용 개념 예시입니다.
- 다양성은 모델 이름이 몇 개인지가 아니라 같은 행의 예측 순서가 얼마나 다르게 어긋나는지로 설명합니다.

### Confluence 보충 설명

ROC AUC의 순서 오류는 단순한 행별 정오표와 같지 않습니다. 그림의 두 표식을 실제 비율로 읽지 않습니다.
Lookup-Transformer의 실제 다양성은 행 그림이 아니라 후보 풀의 최근접 구성과 비교한 순위 상관과 결합 기여로 확인했습니다.

근거: [오차 겹침 시각 문법](https://github.com/tmheo/predicting-smartphone-addiction/issues/569#issuecomment-5489136191), [Lookup-Transformer 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965)

부록: [G. 서로 다른 오차와 Lookup-Transformer](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 29. Lookup-Transformer는 결합 기여로 자리를 얻었습니다

Lookup-Transformer는 낮은 중복과 양의 결합 기여를 모두 확인한 뒤 후보 풀에 들어갔습니다.

![Lookup-Transformer가 개인전 검수에서 일반 OOF 0.96892와 당시 기준 대비 0.00038 상승을, 팀 기여 검수에서 결합 기여 0.00025와 최근접 순위 상관 0.98149를 보여 두 검수를 통과해 후보 풀에 들어가는 흐름](assets/screen-29-lookup-contribution.png)

시각 자료 대체 설명: Lookup-Transformer의 일반 OOF AUC는 `0.96892`로 당시 기준보다 `+0.00038` 높았습니다. 최근접 구성과의 순위 상관은 `0.98149`로 중복 기준 `0.998`보다 낮았고 표준 평가 결합은 `+0.00025` 높아졌습니다.

### 발표자 메모

- 같은 값끼리 묶어 조회하고 연속 수치의 부드러운 추세를 반영한 방식이 tree-based model과 다른 오차를 만들었다고만 설명합니다. 신경망 구조의 자세한 설명은 부록으로 보냅니다.
- 당시 단일 점수도 기준을 넘었습니다. 다만 후보 풀에 남긴 결정은 낮은 중복과 양의 결합 기여를 별도로 확인한 결과였습니다.

### Confluence 보충 설명

최근접 구성은 `exp045_xgb_depth8`이었고 스피어만 순위 상관은 `0.98149`로 중복 기준 `0.998`보다 낮았습니다.
표준 평가 결합은 표시값 기준 `0.96813`에서 `0.96839`로 높아졌고 정확한 한계 기여는 `+0.00025`였습니다.

근거: [Lookup-Transformer 판정](https://github.com/tmheo/predicting-smartphone-addiction/issues/58#issuecomment-5287565965), [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md#2-다르게-틀리는-구성의-가치)

부록: [G. 서로 다른 오차와 Lookup-Transformer](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 30. 자체 35개에서 최종 314개 예측 열까지 갔습니다

자체 35개 결합의 nested OOF AUC `0.96981`에서 최종 314개 예측 열의 `0.97038`까지 내부 결합 점수가 높아졌습니다.

![확대 눈금 0.9695부터 0.9705에서 nested OOF 청록 네모 두 개가 자체 35개 결합 0.96981과 최종 314개 예측 열 0.97038에 놓이고, 시간순 차이 +0.00057과 직접 효과가 아니라는 제한이 표시된 차트](assets/screen-30-internal-assembly-score.png)

시각 자료 대체 설명: 자체 35개 결합의 nested OOF AUC `0.96981`과 최종 314개 예측 열의 `0.97038`을 같은 내부 결합 점수 눈금에 놓았습니다. 두 값 사이에는 여러 변경이 있었다는 점도 함께 표시합니다.

### 발표자 메모

- 두 점은 시간순 출발점과 도착점입니다. 외부 예측 추가 하나만의 직접 효과를 나타내지는 않습니다.
- 중간에는 자체 풀이가 35개에서 36개로 바뀌었고 외부 예측 검수와 결합 설정 선택도 함께 진행됐습니다.

### Confluence 보충 설명

결합 내부 점수의 계보는 자체 35개에서 시작해 외부 207개를 더한 242열, 해로운 외부 계열을 뺀 313열, 결합 규제를 내부에서 고른 313열, 최종 314열로 이어집니다.
각 단계에서 구성과 선택 절차가 함께 달라졌으므로 출발점과 도착점의 차이를 한 기법의 인과 효과로 읽지 않습니다.

근거: [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md), [314열 재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md)

부록: [H. 최종 314개 예측 열의 점수 계보](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 31. 314개 예측 열은 검수해 남긴 조립 재료입니다

자체 36개와 외부 278개의 예측 열을 무결성과 nested OOF 판정으로 검수해 최종 314열 조립 입력으로 남겼습니다.

![자체 예측 36열과 외부 예측 278열이 무결성 검수와 nested OOF 판정을 통과해 314열 조립판으로 들어가는 깔때기](assets/screen-31-final-assembly.png)

시각 자료 대체 설명: 자체 예측 36열과 외부 예측 278열의 열 순서와 행 정렬, 해시, 라이선스 장부를 검수합니다. 이어 nested OOF 판정을 거쳐 최종 314열 결합 입력으로 남깁니다.

### 발표자 메모

- 314개는 모델 수가 아니라 출처별 예측값의 열 수입니다.
- 외부 278개를 모두 우리가 다시 학습했다는 뜻은 아닙니다. 전체 자료로 다시 학습한 대상은 자체 36개뿐입니다.

### Confluence 보충 설명

외부 구성원의 분할 근거와 라이선스 수준이 모두 같지는 않습니다.
외부 278개 가운데 64개는 사용 한정으로 분류했습니다. 이 한계는 최종 구성원 장부와 기술 근거 부록에 남겼습니다.

근거: [우리 최종 해법과 제출 계보 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md), [314열 재조립 판정](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-pool-reassembly/issue513/report.md)

부록: [H. 최종 314개 예측 열의 구성과 검수](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 32. 공식 결과는 Public 0.97135, Private 0.97109, 최종 14위였습니다

내부 검증으로 고른 314개 예측 열 제출은 공식 대회에서 Public 점수 `0.97135`, Private 점수 `0.97109`와 최종 14위를 기록했습니다.

![공식 최종 결과 14위와 참가자 이름을 왼쪽에 두고, 같은 제출의 Public 점수 0.97135를 호박 마름모로, Private 점수 0.97109를 보라 별로 오른쪽에 구분한 결과 패널](assets/screen-01-official-result.png)

시각 자료 대체 설명: 공식 최종 순위 14위와 참가자 이름은 왼쪽에 표시합니다. 같은 제출 `55907610`의 Public 점수 `0.97135`와 Private 점수 `0.97109`는 서로 다른 표식으로 오른쪽에 표시하고 각각의 채점 범위도 설명합니다.

### 발표자 메모

- Public 점수와 Private 점수는 같은 제출을 서로 다른 공식 표본에서 채점한 결과입니다.
- 결합 규제를 내부에서 고른 313열 판과 최종 314열 판의 공식 표시값은 같았습니다. 마지막 변경이 공식 점수를 높였다고 말하지 않습니다.

### Confluence 보충 설명

내부 nested OOF와 공식 점수는 평가한 행과 선택 경계가 다릅니다. 두 값의 차이를 개선량으로 계산할 수 없습니다.
마지막으로 올린 327열 제출의 Private 점수는 `0.97108`이었습니다. 최종 14위 성적을 낸 제출은 마지막 업로드가 아니라 314열 제출입니다.

근거: [우리 최종 해법과 제출 계보 복원](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md), [최종 조립 제출 기록](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/extended-stack-final-assembly/issue514/submission-record.json)

부록: [A. 공식 결과와 자료 범위, H. 최종 314개 예측 열](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 33. 남은 차이는 더 강한 단일 모델 탐색이었습니다

이 회고에서는 우승권과의 차이가 더 강한 단일 모델을 일찍 찾는 탐색 역량에서 비롯됐다고 해석합니다.

![고정 fold와 사전 중단 관문, 서로 다른 오차와 nested OOF 조립, 공통 검수 절차라는 기반 위에 더 넓고 이른 단일 모델 탐색을 다음 보강 축으로 올리고, 우리와 1등의 단일 구성 점수는 직접 비교가 아님을 표시한 그림](assets/screen-33-search-gap.png)

시각 자료 대체 설명: 우리가 지킬 검증과 조립 기반 옆에 더 넓고 이른 단일 모델 탐색을 다음 보강 축으로 세웠습니다. 우리 최고 자체 단일 구성과 1등 글의 RealMLP 점수는 검증 명세가 달라 직접 비교하지 않는다고 표시합니다.

### 발표자 메모

- 공개 기록에서 1등 RealMLP의 바깥쪽 예측 AUC는 `0.970706453`이고 우리 최고 자체 단일 구성의 OOF AUC는 `0.9694062694`입니다. 다만 검증 명세가 달라 두 점수의 차이를 직접 계산하지 않습니다.
- 1등 RealMLP의 최종 Private 점수와 전체 재현 명세는 공개되지 않았습니다. 이 해석을 인과적으로 증명된 결론처럼 말하지 않습니다.

### Confluence 보충 설명

1등 글에서 확인할 수 있는 내용은 높은 단일 RealMLP 바깥쪽 예측 AUC, 449개까지 늘어난 결합, 여러 에이전트가 병렬로 탐색했다는 사실입니다.
분할표, 전처리 경계, 설정, 구성원 장부와 결합식은 공개되지 않았습니다. 이 화면은 확인된 사실에서 도출한 회고 해석이며 같은 조건에서 얻은 대조 결과는 아닙니다.

근거: [1등 해법 원문 조사](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-first-place-writeup.md), [발표용 성적과 실험 계보 근거](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/presentation-score-evidence.md)

부록: [I. 1등과의 비교 및 다음 원칙](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 34. 넓은 단일 모델 탐색을 일찍 엽니다

다음 대회에서는 넓은 단일 모델 탐색을 일찍 시작하되 고정 fold와 사전 중단 관문은 유지합니다.

![다음 대회 초반에는 넓은 단일 모델 탐색, 중반에는 고정 fold와 중단 관문, 후반에는 서로 다른 오차 조립을 배치하고 더 많은 실험이 아니라 더 이른 탐색과 더 빠른 중단이라고 표시한 시간선](assets/screen-34-next-competition-timeline.png)

시각 자료 대체 설명: 다음 대회 초반에는 작동 원리가 서로 다른 단일 모델을 넓게 탐색합니다. 고정 fold와 중단 관문을 거친 뒤, 이를 통과한 서로 다른 오차만 후반에 조립하는 순서입니다.

### 발표자 메모

- 실험 수를 무조건 늘린다는 뜻은 아닙니다. 강한 후보를 일찍 넓게 찾고 작은 근거로 빨리 멈추는 구조입니다.
- 결과를 보기 전에 분할과 중단 기준을 고정합니다. 혼자 잘하는가와 함께할 때 돕는가를 나누는 원칙도 바꾸지 않습니다.

### Confluence 보충 설명

이 시간선은 이번 회고에서 도출한 다음 대회 권고입니다. 이미 확정한 실험 계획이나 우승 해법을 그대로 복제하겠다는 뜻은 아닙니다.
새 후보의 구체적인 범위와 자원 배분은 다음 대회의 자료와 제약을 확인한 뒤 따로 결정합니다.

근거: [1등 해법 원문 조사](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-first-place-writeup.md), [발표 제작 장부](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/presentation/s6e8-retrospective-production-ledger.md)

부록: [I. 1등과의 비교 및 다음 원칙의 다음 대회 권고](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)

## 화면 35. 작게 검증하고, 다르게 틀리는 예측을 모으고, 같은 검수대로 조립합니다

작은 검증, 서로 다른 오차, 공통 검수대라는 세 원칙이 이번 결과를 설명합니다.

![작게 검증하기, 다르게 틀리는 예측 모으기, 같은 검수대로 조립하기라는 세 조각이 최종 14위 결과로 모이고 다음에는 넓은 단일 모델 탐색을 더 일찍 연다는 방향으로 이어지는 닫힌 고리](assets/screen-35-conclusion.png)

시각 자료 대체 설명: 작은 검증, 서로 다른 오차, 공통 검수대라는 세 조각이 최종 14위 결과로 모입니다. 다음 대회에서는 같은 원칙을 지키면서 넓은 단일 모델 탐색을 더 일찍 시작합니다.

### 발표자 메모

- 새로운 수치는 덧붙이지 않습니다. 앞에서 본 RealMLP 수정, Lookup-Transformer, 실행 기록 묶음, 314열 조립을 한 문장씩 되짚습니다.
- 마지막 문장은 질의응답을 여는 질문으로 바꿔 읽습니다.

### Confluence 보충 설명

작은 검증은 결과를 보기 전에 비교와 중단 관문을 고정하는 습관을 뜻합니다.
서로 다른 오차는 기존 예측과 결합했을 때 서로 보완하는 예측을 남긴다는 뜻입니다. 단독 점수만 높이는 데 그치지 않습니다. 공통 검수대는 실행 장소와 관계없이 같은 무결성 검수와 재채점 절차를 적용한다는 뜻입니다.

근거: [발표 제작 장부](https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/presentation/s6e8-retrospective-production-ledger.md)

부록: [C. 같은 값을 표현한 실험, D. 결측 증강 판정, E. 실험 실행 체계, G. 서로 다른 오차와 Lookup-Transformer, H. 최종 314개 예측 열](https://lgucorp.atlassian.net/wiki/spaces/~7120202a66323266d44ee697a3e30c7a270829/pages/1501596455)
