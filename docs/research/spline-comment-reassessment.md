# Spline Transformer 댓글 근거에 따른 중단 결정 재검토

이 문서는 [조사: 새 spline Transformer 댓글 근거로 기존 중단 결정 재검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/166)의 근거다.
조사 시점은 2026-08-17이며, Kaggle의 공개 댓글, 공개 노트북 판본과 출력, 공개 프로필, 현재 공개 순위표와 이 저장소의 자체 OOF 결과를 대조했다.

## 결론

제한적인 후속 실험을 다시 여는 것이 타당하다.
다만 Optimistix가 현재 2위라는 사실이나 Tilii의 경력만으로 spline Transformer의 성능 향상을 확정해서는 안 된다.
Optimistix의 댓글은 노트북 공유에 감사한다는 한 문장일 뿐이며, 이 모델을 사용했다거나 이 모델 때문에 순위가 올랐다는 말은 없다.
Tilii는 자신의 변형이 Public LB `0.96982`였고 대규모 앙상블에서 약 `+0.00005`를 더할 것이라고 추정했지만, 목표값이 붙은 OOF 제거 대조나 해당 앙상블의 구성원 포함 전후 결과는 공개하지 않았다.

재실험을 여는 결정적 근거는 댓글이 아니라 판정 계약의 사후 변경이다.
[기존 진입 진단](contextualized-spline-entry-diagnostic.md)은 M0의 균등 순위 평균 기여 `+0.0001244004`가 기존 구성원 복제 대조 상단 `+0.0001279202`보다 `0.0000035199` 낮다는 이유로 다양성 목표를 기각했다.
그 결과를 기록한 커밋은 2026-08-16 19:37 JST의 [`00ff69d`](https://github.com/tmheo/predicting-smartphone-addiction/commit/00ff69d99ed0894f529e59d03a38faadc7fd4e11)다.
이후 2026-08-17 09:02 JST의 [`5289dcc`](https://github.com/tmheo/predicting-smartphone-addiction/commit/5289dcc2e36c5e709def2f691ab7907274f67f07)에서 균등 순위 평균 기여를 후보 진입과 제거의 게이트에서 참고값으로 낮추고, 무결성과 중복 검사를 통과한 후보를 모두 nested OOF 평가에 넘기도록 [판정 계약](../adr/0001-experiment-adoption-contract.md)을 바꿨다.
M0는 이미 성능 하한과 최근접 스피어만 상관 `< 0.998`을 통과했으므로, 현재 계약을 적용하면 과거의 유일한 다양성 탈락 사유가 사라진다.

따라서 전체 구조나 하이퍼파라미터 탐색을 다시 시작하지 않고, 이미 구현한 M0와 짝지은 A0의 공통 5-fold 단일 시드 OOF를 완성해 현재 계약으로 다시 거르는 한 건만 열 가치가 있다.

## 댓글이 실제로 말하는 것

[토론 원문](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)에는 현재 댓글 네 개가 있다.
Optimistix의 댓글 원문은 "Thanks for sharing your cool experiments with Neural Networks!"이며, 모델 사용, OOF, 제출 점수, 앙상블 기여나 순위 변화의 원인을 언급하지 않는다.
따라서 이 댓글은 상위 참가자의 관심 신호이지만 사용 증거나 효능 증거는 아니다.

Tilii는 원 저자의 모형에 몇 가지 생성 특성을 더한 자신의 변형이 Public LB `0.96982`였다고 썼다.
Tilii는 이 모형이 100개가 넘는 다른 모형과 함께 쓰일 때 최상위 Public LB를 약 `0.00005` 올릴 가능성이 거의 확실하다고 평가했다.
Tilii는 자신의 TabM 예측과 이 모형 예측의 주변 누적분포 사이 KS 통계량이 `0.118790`이라고 제시하고, 낮은 예측 확률 구간과 높은 예측 확률 구간에서 두 모형의 강점이 다르다고 해석했다.
이는 실제 대규모 앙상블을 반복해 온 참가자의 유용한 정성 판단이지만, 주변 예측분포가 다르다는 사실만으로 목표값에 대한 오차가 상보적이라고 결론 낼 수는 없다.
같은 예측의 단조 변환이나 보정 차이도 큰 KS 통계량을 만들 수 있으므로, 목표값이 붙은 같은-fold OOF의 상관, 포함 전후 nested OOF와 제거 대조가 필요하다.

원 저자는 답글에서 hypernetwork를 더한 실험이 작은 CV 개선을 보였지만 아직 공유할 만큼 크지 않다고 밝혔다.
공개 코드나 OOF가 없는 이 답글은 hypernetwork 후보를 새로 열 근거가 아니다.

## 작성자 신뢰도와 주장 신뢰도를 분리한 결과

[Tilii의 공개 프로필](https://www.kaggle.com/tilii7)은 그를 `Discussions Legacy Grandmaster`로 표시하지만, 현재 경진대회 등급은 `Competitions Master`이고 노트북 등급도 `Notebooks Master`다.
프로필에는 Playground S5E10 1위와 S6E1 2위 해설이 공개되어 있어, 신경망과 대규모 앙상블 다양성에 관한 경험은 강한 전문성 신호다.
그러나 토론 분야의 예전 최고 등급과 경진대회 분야의 현재 Master 등급을 `Competitions Grandmaster`로 합쳐 부르면 안 된다.
전문성은 실험 우선순위를 올릴 수 있지만, 이 저장소의 공통 fold OOF와 선택 편향 없는 결합 평가를 대신하지 않는다.

[Optimistix의 공개 프로필](https://www.kaggle.com/optimistix)에는 대규모 OOF 앙상블을 사용해 Playground S4E8에서 1위를 한 해설이 공개되어 있어, 짧은 관심 표현에도 경험자의 신호라는 가치는 있다.
그러나 이번 댓글에는 자신의 사용 여부나 결과가 전혀 없으므로 그 경력을 spline 구조의 효능 근거로 옮길 수 없다.

## Optimistix 순위 변화와 인과관계

2026-08-17 06:22 UTC에 Kaggle 공식 CLI로 내려받은 [현재 공개 순위표](https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard)에서 Optimistix는 2위, 점수 `0.97125`, 제출 수 109개, 마지막 제출 시각 `2026-08-16 23:55:08`이다.
같은 스냅샷에서 Tilii는 25위 `0.97114`, Ern711은 494위 `0.96940`이다.
순위표 압축 파일의 SHA-256은 `216a2a91547144df12c2d90a2aca3f6006417e49c59ad8958e3dd6234a85746d`이고 그 안의 CSV SHA-256은 `42f1623b88705e9a5351810fe1af0a399ebd10525c2c2d3dc184d5581624c4de`다.

이 스냅샷은 현재 2위를 확인하지만, 댓글 전의 5위 또는 6위 상태를 담지 않는다.
토론 화면의 `2nd in this Competition`과 `25th in this Competition` 표시는 현재 순위를 보여 주며 댓글 작성 당시 순위의 고정 기록이 아니다.
공개 순위표 내려받기도 각 팀의 현재 최고 점수와 마지막 제출 시각만 제공하고 제출별 점수 이력이나 과거 시점 순위를 제공하지 않는다.
확인 가능한 과거 순위표 스냅샷도 찾지 못했으므로, 사용자가 기억한 5위 또는 6위에서 2위로의 이동은 검증도 반증도 할 수 없다.

설령 순위 이동 시점을 확인하더라도 댓글보다 뒤에 일어났다는 시간 순서만으로 모델 사용을 증명할 수 없다.
Optimistix는 109번 제출했고 댓글에는 사용 주장이 없으므로, 다른 모형, 생성 특성, 결합 방식이나 Public LB에 맞춘 선택이 원인일 가능성을 분리할 수 없다.
검증되지 않은 순위 이동과 모델 효능의 인과관계는 별개의 주장으로 남겨야 한다.

## 공개 노트북과 출력의 변화 여부

[Contextualized Deep Univariate Spline Transformer 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)은 2026-08-17 재확인 시에도 최신 공개 판본이다.
Kaggle 목록 API의 최신 실행 시각은 `2026-08-15T17:47:16.697000`이고, 공개 토론 댓글 뒤에 실행된 새 판본은 없다.
현재 소스 SHA-256 `c308b69cfeabad223a1e147fa174f78d1ddaccc09991b2075eecaf757f4781a2`는 [기존 노트북 조사](s6e8-167-spline-notebooks.md)에 기록한 값과 같다.
[공개 출력](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/output)의 실행 기록 SHA-256 `c683bef73188b3cf5f8362b4394518faa2ede9c4f3a3102603a1354f05b681d6`과 OOF SHA-256 `8547bf8397dcca85aef00a31e35faef1a12ef433c0555b40e17ab1eb49fa91ef`도 기존 조사 때와 같다.
따라서 새 댓글은 새 코드, 새 OOF, 새 제거 대조나 판본 변경을 동반하지 않았다.

공개 OOF의 전체 AUC `0.9665204982`, seed 21 fold, 학습·검증·시험을 합쳐 만든 범주 어휘와 여러 구성 요소가 섞인 최종 경로 대 가산 경로 비교라는 기존 한계도 그대로다.
공개 OOF나 제출 예측을 후보 풀에 직접 넣지 않는 기존 결정은 유지해야 한다.

## 기존 자체 실험을 현재 기준으로 다시 읽기

[이슈 149](https://github.com/tmheo/predicting-smartphone-addiction/issues/149)의 M0는 공통 seed 42 fold 0에서 AUC `0.9667574340`으로 당시 같은-fold Lookup 기준 `0.9682949114`보다 `0.0015374774` 낮았다.
따라서 champion 교체 가능성이 낮다는 판단은 새 댓글로 바뀌지 않는다.
짝지은 주기 표현 대조 A0는 `0.9667160779`였고 M0의 이득은 `+0.0000413560`에 불과해, 한 fold만으로 조각선형 경로의 일반화 기여를 확정할 수 없다.

반면 M0의 최근접 후보 스피어만 상관 `0.9781382739`는 중복 문턱 `0.998`보다 충분히 낮았고, 균등 순위 평균 기여 `+0.0001244004`도 양수였다.
당시 탈락은 이 기여가 복제 대조 상단보다 단지 `0.0000035199` 낮았기 때문이었다.
현재 [판정 계약](../adr/0001-experiment-adoption-contract.md)은 상관을 중복 제거에만 쓰고 균등 순위 평균 기여는 참고값으로만 기록하므로, M0는 현행 다양성 스크리닝 관점에서 살아 있다.

[순위 평균과 nested 선형 결합 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)는 균등 순위 평균에서 기여가 작거나 음수였던 후보를 포함한 16개 풀이 학습형 결합에서 더 좋을 수 있음을 실제 OOF로 확인했다.
그 결정 뒤 현재 champion은 `exp081_lookup_fold_initialization_avg3`, OOF AUC `0.9691957618114269`로 바뀌었다.
[결측 구간 결합 비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/67)는 16개 후보의 결측 구간별 독립 선형 결합이 nested OOF AUC `0.969510369267`을 기록해 이전 이중 표현 선형 결합보다 `+0.000026877617` 높았음을 확인했다.
따라서 M0의 가치는 고정 균등 평균이 아니라 현재 16개 후보와 결측 구간 결합 안에서 선택 편향 없이 다시 평가해야 한다.

## 열어야 할 최소 후속 질문

후속 질문은 "현행 판정 계약에서 M0의 비중복 OOF가 현재 16개 후보의 선택 편향 없는 결합 성능을 높이며, 그 효과에 조각선형 수치 경로가 실제로 기여하는가"로 한정한다.

첫 단계의 후보는 이미 구현한 `M0` 하나이고 짝지은 대조는 이미 구현한 `A0` 하나다.
두 모형은 동일한 현재 피처 계획, 공통 `artifacts/folds.parquet`, seed 42, 초기화 규칙, 학습 예산과 후처리를 사용하고 수치 표현만 다르게 유지한다.
기존 fold 0 산출물은 커밋, 설정, 입력 해시와 실행 환경이 완전히 일치할 때만 재사용하고, 하나라도 다르면 두 모형의 fold 0도 함께 다시 실행한다.
M0와 A0의 seed 42 전체 5-fold OOF를 완성하기 전에는 hypernetwork, knot 수, 해상도, 동적 결합, 목표·빈도 인코딩, 최종 MLP 크기나 생성 특성을 탐색하지 않는다.

다음 증거 게이트와 중단 조건을 적용한다.

1. M0가 현재 champion의 같은 seed OOF보다 `0.01` 넘게 낮거나, 현재 16개 후보 중 최근접 스피어만 상관이 `0.998` 이상이면서 더 약하면 중단한다.
2. M0와 A0의 seed 42 전체 OOF 차이와 fold별 차이를 보고하며, M0가 A0보다 `+0.00002` 이상이고 5개 중 3개 이상 fold에서 높을 때만 조각선형 경로가 기여했다는 가설을 다음 단계에 유지한다.
3. 조각선형 경로 게이트가 실패해도 M0 전체 구조가 성능 하한과 중복 검사를 통과하면, 구조 전체의 다양성 질문을 위해 M0만 seed 43과 44로 확정 재검증한다.
4. M0 3시드 평균본이 현행 후보 풀의 성능 하한, 무결성과 중복 조건을 통과하지 못하면 후보 풀에 넣지 않고 중단한다.
5. M0가 후보 자격을 얻으면 기존 16개 후보만 쓴 현재 결측 구간 결합과 M0를 더한 17개 후보 결합을 같은 outer fold에서 짝지어 비교한다.
6. M0를 더한 nested OOF가 현재 선택 결과 `0.969510369267`보다 높지 않으면 기존 16개 후보와 결합 전략을 유지하고 중단한다.
7. 새 결과가 더 높을 때만 현행 [판정 계약](../adr/0001-experiment-adoption-contract.md)에 따라 최종 전략과 후보 풀 변경을 검토한다.

이 범위는 기존 중단 결정을 뒤집어 전체 모델 탐색을 재개하는 것이 아니다.
중단 뒤 바뀐 판정 계약을 기존 M0 결과에 일관되게 적용하고, Tilii의 정성적 다양성 신호를 이미 구현된 가장 작은 공통-fold 검증으로 확인하는 절차다.

## 앞으로 결정을 다시 바꿀 수 있는 외부 근거

Optimistix가 이 모델 또는 그 OOF를 실제로 사용했다고 명시하고, 포함 전후 OOF나 제출별 점수와 시각을 공개하면 현재의 사용 증거 공백을 줄일 수 있다.
Tilii가 같은-fold OOF, 목표값 정렬, 구성원 목록과 spline 구성원 포함 전후의 nested OOF를 공개하면 `+0.00005` 추정을 재현 가능한 근거로 올릴 수 있다.
원 저자가 hypernetwork 판본, 공통 fold OOF와 사전 고정 제거 대조를 공개하면 현재 제외한 확장 구조를 별도 후보로 다시 판단할 수 있다.
그 전에는 순위, 득표 수, Public LB와 예측 주변분포만으로 새 구조 탐색 범위를 넓히지 않는다.

## 출처

- [Spline Transformer 토론](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)은 Optimistix, Tilii와 원 저자 댓글의 1차 출처다.
- [Contextualized Deep Univariate Spline Transformer 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)과 [공개 출력](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/output)은 공개 코드, OOF와 실행 기록의 1차 출처다.
- [S6E8 공개 순위표](https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard)는 현재 순위, 점수와 마지막 제출 시각의 1차 출처다.
- [Tilii 프로필](https://www.kaggle.com/tilii7)과 [Optimistix 프로필](https://www.kaggle.com/optimistix)은 분야별 등급과 공개 경진대회 이력의 1차 출처다.
- [기존 노트북 조사](s6e8-167-spline-notebooks.md), [진입 진단 결과](contextualized-spline-entry-diagnostic.md), [판정 계약](../adr/0001-experiment-adoption-contract.md), [champion 장부](../../artifacts/champion.yaml)와 [후보 풀 장부](../../artifacts/pool.yaml)는 자체 판정의 기록 원본이다.
