# spline Transformer 새 댓글 근거 재검토

이 문서는 GitHub 이슈 [새 spline Transformer 댓글 근거로 기존 중단 결정 재검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/166)의 조사 결과다.
조사 시점은 2026-08-17이며, Kaggle 토론 원문과 공개 읽기 API, 공개 노트북 산출물, 저장소의 자체 OOF 기록만 사용했다.

## 결론

새 댓글만으로 spline Transformer를 채택하거나 Tilii의 수정판을 재현했다고 볼 수는 없다.
`0.96982`, 100개 초과 앙상블에서 약 `+0.00005`, KS `0.118790`은 모두 구성원 목록, OOF, 분할, 결합 절차와 실행 산출물이 공개되지 않은 한 참가자의 보고다.
특히 `0.96982`는 공개 노트북 원본이 아니라 Tilii가 파생 변수를 추가한 변형의 public LB 점수다.

그러나 추가 실험을 전부 중단한 결정은 제한적으로 뒤집을 가치가 있다.
기존 M0는 자체 공통 fold 0에서 후보 풀 최근접 스피어만 상관 `0.9781382739`와 순위 평균 기여 `+0.0001244004`를 보였고, 당시 사전 고정한 복제 대조 상한 `+0.0001279202`에 불과 `0.0000035199` 못 미쳐 중단됐다.
이 결과는 2026-08-16 19:37 JST의 [`00ff69d`](https://github.com/tmheo/predicting-smartphone-addiction/commit/00ff69d99ed0894f529e59d03a38faadc7fd4e11)에 기록됐다.
그 뒤 2026-08-17 09:02 JST의 [`5289dcc`](https://github.com/tmheo/predicting-smartphone-addiction/commit/5289dcc2e36c5e709def2f691ab7907274f67f07)에서 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)은 균등 순위 평균 기여를 후보 진입과 제거의 게이트가 아닌 참고값으로 바꿨고, 무결성과 중복 검사를 통과한 후보를 nested OOF 평가에 넘기도록 개정됐다.
실제로 현재 16개 후보의 학습형 결합은 균등 순위 평균에서 음수 기여를 보인 구성원까지 유지했을 때 3개 후보 부분집합보다 nested OOF AUC가 `+0.000033467042` 높고 5개 outer fold 모두에서 이겼다.
따라서 이슈 149의 마지막 중단 근거는 현재 계약에서 더 이상 하드 게이트가 아니며, 새 댓글은 이미 두 자체 진단이 보인 다양성 방향을 독립 참가자의 실제 앙상블 경험으로 보강한다.

다시 여는 범위는 기존 M0 설정 한 개의 공통 5-fold와 조건부 3시드 확정 및 nested OOF 평가뿐이다.
해상도 수, 동적 결합, 목표·빈도 인코딩, 최종 MLP 크기, Tilii의 미공개 파생 변수와 hypernetwork 탐색은 계속 닫는다.

## Kaggle 원문에서 확인한 사실

[토론 원문](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)은 Ern711이 2026-08-15 19:59:38 UTC에 공개 노트북을 소개한 글이다.
본문은 깊은 단변량 수치 인코더, 작은 학습형 비선형 문맥 보정과 얕은 self-attention을 결합했다고 설명하지만 수치 성능은 적지 않았다.

[Tilii의 댓글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421#3513480)은 2026-08-16 16:35:20 UTC에 게시됐다.
Kaggle 공개 읽기 API는 작성자를 Tilii, 댓글 작성자 등급을 Grandmaster, 현재 대회 순위를 89로 표시하므로 앙상블 경험에 관한 전문성 신호는 강하다.
다만 공개 프로필의 세부 등급은 Discussions Legacy Grandmaster와 Competitions Master이므로 이를 Competitions Grandmaster 경력으로 확대해 읽지 않는다.
다만 전문성은 재현 가능한 OOF와 결합 실험을 대신하지 않는다.

Tilii는 Ern711의 모델에 여러 파생 변수를 더한 자신의 변형이 public LB `0.96982`였다고 썼다.
또한 이 모델을 보통 100개가 넘는 모델과 함께 쓸 때 상위권 public LB에서 약 `0.00005` 상승을 거의 확실하게 만든다고 평가했다.
Lookup-Transformer를 다양성의 현재 선두로 부르면서 spline 모델도 그에 근접한다고 덧붙였다.

Tilii는 자신의 TabM 예측과 spline 예측의 누적분포를 비교해 KS `0.118790`을 보고했다.
그의 설명에 따르면 spline은 예측 확률 `0.2` 이하에서 목표 분포를 더 잘 맞추고 TabM은 `0.4` 초과에서 더 잘 맞추며, 중간 구간에서는 우위가 번갈아 나타난다.
같은 경향이 자신의 거의 모든 다른 모델과도 보였다고 썼다.
댓글의 [CDF 그림 원본](https://i.postimg.cc/cHgq8CNy/distributions-spline-tabm.png)은 이 정성 설명을 시각화하지만 비교 예측 배열과 계산 코드는 공개하지 않는다.

[Ern711의 후속 답글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421#3513485)은 2026-08-16 17:15:15 UTC에 게시됐다.
Ern711은 모델 위에 hypernetwork를 얹는 등의 추가 실험에서 교차검증이 소폭 개선됐지만 아직 공개할 만큼 크지 않다고 밝혔다.
정확한 구조, 입력, 분할, 수치, 시드, 실행 산출물과 public LB 결과는 제시하지 않았다.
2026-08-17 확인 시점의 [공개 노트북 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)은 여전히 최신 공개 판본이며 Kaggle 목록 API의 최신 실행 시각은 2026-08-15 17:47:16 UTC다.
따라서 hypernetwork 답글 뒤에 공개된 새 실행이나 산출물은 없고 hypernetwork는 재현 가능한 후보가 아니다.
최신 공개 소스, 실행 기록과 OOF의 SHA-256은 각각 `c308b69cfeabad223a1e147fa174f78d1ddaccc09991b2075eecaf757f4781a2`, `c683bef73188b3cf5f8362b4394518faa2ede9c4f3a3102603a1354f05b681d6`, `8547bf8397dcca85aef00a31e35faef1a12ef433c0555b40e17ab1eb49fa91ef`로 기존 조사 때와 같다.

2026-08-17 06:22 UTC에 Kaggle 공식 CLI로 내려받은 현재 공개 순위표는 Optimistix를 `0.97125`의 2위, 제출 수 109개, 최상 제출 시각 2026-08-16 23:55:08 UTC로 기록한다.
같은 스냅샷에서 Tilii는 `0.97114`의 25위, Ern711은 `0.96940`의 494위다.
순위표 압축 파일과 내부 CSV의 SHA-256은 각각 `216a2a91547144df12c2d90a2aca3f6006417e49c59ad8958e3dd6234a85746d`, `42f1623b88705e9a5351810fe1af0a399ebd10525c2c2d3dc184d5581624c4de`다.
이는 Optimistix가 토론에 단순 감사 댓글을 남긴 뒤 순위표 최상 제출을 갱신했다는 시간 순서만 확인한다.
공개 자료에는 직전 5위 또는 6위였다는 이력, spline 예측 사용 여부, 제출 구성과 점수 상승 원인이 없으므로 순위 변화를 spline 모델의 효과로 귀속할 수 없다.

## 네 가지 주장별 대조

### 1. 단독 점수

[기존 공개 노트북 검토](s6e8-167-spline-notebooks.md)가 저장 OOF에서 다시 계산한 원본 모델의 단일 시드 5-fold OOF AUC는 `0.9665204982`다.
현재 champion `exp081_lookup_fold_initialization_avg3`의 3시드 평균본 OOF AUC `0.9691957618`보다 `0.0026752636` 낮지만 후보 풀 진입 하한인 champion 대비 `-0.01` 안에는 충분히 든다.
분할은 원본이 seed 21, 저장소가 커밋된 seed 42 fold이므로 이 절대 차이는 정식 짝비교가 아니다.

Tilii의 `0.96982`는 원본보다 훨씬 높지만 파생 변수의 목록과 검증 OOF가 없어 모델 구조의 개선량으로 분해할 수 없다.
Public 점수는 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)에 따라 모델 채택 근거가 아니며, 해당 점수는 모델 원본의 단독 재현 점수도 아니다.
따라서 이 수치는 champion 교체 가능성을 높이는 증거로 사용하지 않는다.

### 2. 대규모 앙상블 기여

Tilii가 말한 약 `+0.00005`는 100개가 넘는 비공개 구성원 위에서의 public LB 추정이라 구성원 하나의 OOF 한계 기여나 nested OOF 개선으로 환산할 수 없다.
구성원 수가 많을수록 한 구성원의 직접 가중치가 작아지고 결합 전략과 기존 구성원 중복에 따라 변화량이 달라지므로 앙상블 크기만으로 신뢰도를 높일 수도 없다.

그 크기는 자체 참고 진단과 일관된다.
[기존 공개 OOF 참고 진단](s6e8-167-spline-notebooks.md)은 공개 spline OOF의 현재 16개 후보 대비 최근접 스피어만 상관을 `0.975128`, 고정 순위 평균 추가 변화를 `+0.00008088`로 측정했다.
[자체 M0 진입 진단](contextualized-spline-entry-diagnostic.md)은 공통 fold 0에서 최근접 상관 `0.9781382739`, 16개 후보 순위 평균 추가 변화를 `+0.0001244004`로 측정했다.
서로 다른 세 관찰이 모두 낮은 중복과 `10^-4` 안팎의 작은 양수 변화를 가리키므로 다양성 가설은 우연한 인상보다 강하다.
그러나 전체 OOF, 3시드와 nested OOF가 없는 상태에서는 채택 증거가 아니라 다음 검증의 우선순위 근거다.

[외부 OOF 라이브러리 진입 진단](https://github.com/tmheo/predicting-smartphone-addiction/issues/77#issuecomment-5253573982)은 낮은 상관만으로 기여가 생기지 않으며 단독 성능과 탈상관이 함께 필요하다고 확인했다.
M0의 fold 0 AUC `0.9667574340`은 해당 진단에서 실익이 거의 없던 OOF `0.955` 이하 비-GBDT 계열보다 높고 현재 진입 하한 안에 있으므로, 낮은 상관만 있는 약한 모델과는 구분된다.

### 3. 예측 분포 차이

KS `0.118790`은 spline과 TabM의 예측 주변분포가 다르다는 증거다.
그러나 목표값과 짝지은 오차, 행별 순위 상관, 잔차 상관이나 한계 AUC를 측정하지 않으므로 앙상블 보완성을 직접 증명하지 않는다.
같은 행 순서를 보존하는 단조 보정만으로도 주변분포의 KS는 커질 수 있고 AUC 다양성은 생기지 않을 수 있다.

구간별 설명도 목표 분포를 어떻게 추정했는지, bin과 표본 수가 무엇인지 공개하지 않아 재계산할 수 없다.
따라서 KS는 자체 공통 fold의 스피어만 상관과 nested OOF를 대체하지 않는다.
다만 공개 OOF와 자체 M0에서 이미 확인한 `0.975128`과 `0.9781382739`의 낮은 최근접 상관을 다른 예측 분포 관점에서 보강한다.

### 4. hypernetwork 후속 언급

Ern711 스스로 개선이 작고 공개할 만큼 크지 않다고 평가했다.
효과량, 대조, 재현 코드와 출력이 없으므로 새 실행 후보나 M0의 변형 후보로 특정할 수 없다.
공개 판본이 실제로 갱신되어 공통 fold에 옮길 수 있는 구조와 제거 대조가 생기기 전까지는 근거 갱신 관찰 후보로만 본다.

## 현재 후보 풀과 nested OOF에 비춘 의미

[OOF 후보 풀 감사](oof-pool-audit.md)는 정확·순위 중복 제거 뒤 16개 시드 평균본을 모두 유지한다.
단독 OOF 범위는 `0.9596583964`부터 `0.9691957618`이고, 최근접 스피어만 상관 범위에는 M0보다 높은 중복을 보이는 후보가 여럿 있다.
따라서 M0의 단독 수준과 상관은 현재 풀의 허용 범위에서 벗어나지 않는다.

[순위와 logit 이중 표현 결합 결과](https://github.com/tmheo/predicting-smartphone-addiction/issues/64#issuecomment-5310505173)는 16개 후보에서 nested OOF AUC `0.969483491650`을 기록해 champion보다 `+0.000287729839` 높았다.
[부분집합 선택 결과](https://github.com/tmheo/predicting-smartphone-addiction/issues/62#issuecomment-5310705986)는 세 선택법이 모두 같은 3개 후보만 골랐지만 16개 결합보다 `0.000033467042` 낮고 5개 fold 모두에서 졌다.
[결측 구간 결합 결과](https://github.com/tmheo/predicting-smartphone-addiction/issues/67#issuecomment-5310828188)는 현재 채택 전략을 nested OOF AUC `0.969510369267`로 다시 높였다.

이 결과는 Tilii가 말한 `+0.00005` 규모가 대회 말기의 포화된 앙상블에서도 의사결정상 의미가 있을 수 있음을 보여 준다.
동시에 실제 가치는 전체 후보와 결합 전략을 outer fold 안에서 다시 맞춘 뒤에야 드러나므로 public LB 경험담이나 균등 평균 한 번으로 판정할 수 없음을 보여 준다.

## 제한적으로 다시 열 검증

### 검증 질문

기존 누출 경계를 바로잡은 M0의 낮은 중복이 seed 42 전체 5-fold와 3시드 평균본에서도 유지되어, 현재 16개 후보의 채택된 결측 구간 결합을 nested OOF에서 개선하는가?

### 최소 후보

후보는 기존 `configs/exp085_contextual_spline_m0.yaml` 한 개로 고정한다.
모델 구조, 33개 피처 계획, 학습 설정, 전처리, knot, 어휘, 보조 손실과 시드는 이슈 149 구현에서 바꾸지 않는다.
A0는 fold 0에서 M0보다 `0.0000413560` 낮았고 조각선형 경로의 짝지은 제거 대조 역할을 이미 마쳤으므로 전체 fold로 확장하지 않는다.
Tilii의 파생 변수와 Ern711의 hypernetwork는 구체적으로 재현할 수 없으므로 후보에 넣지 않는다.

### 기준 대조와 단계

1단계는 M0 seed 42의 커밋된 5-fold OOF다.
champion 축은 같은 seed의 `exp081_lookup_fold_initialization_avg3` OOF AUC `0.9690874005`와 짝지어 비교한다.
다양성 축은 현재 16개 후보에 대한 최근접 스피어만 상관과 단독 OOF를 측정하고, 균등 순위 평균 기여는 참고값으로만 기록한다.

1단계가 champion 스크리닝 `delta >= 0`을 통과하거나 다양성 스크리닝에서 현재 champion 대비 `-0.01` 진입 하한과 최근접 스피어만 `< 0.998`을 함께 통과하면 seed 43과 44를 실행해 3시드 평균본을 만든다.
두 축을 모두 놓치면 다시 닫고 구조나 학습 설정을 탐색하지 않는다.

2단계의 champion 교체 판정은 ADR 0001의 `+0.00002`, 2/3 시드 개선과 경계 구간 fold 승리 규칙을 그대로 적용한다.
후보 풀 진입은 3시드 평균본 OOF가 진입 시점 champion 대비 `-0.01` 이상이고 최근접 스피어만 `< 0.998`이며 채택 자격을 갖춘 경우로 한정한다.
균등 순위 평균 기여와 Tilii의 public LB 추정은 진입 게이트로 쓰지 않는다.

후보 풀 게이트를 통과하면 M0를 더한 17개 후보로 기존에 구현된 결합 전략만 같은 nested OOF 절차에서 다시 평가한다.
새 결합 전략이나 hypernetwork를 함께 탐색하지 않는다.
최종 승격은 최고 17개 후보 전략의 nested OOF AUC가 현재 채택된 16개 후보 결측 구간 결합 `0.969510369267`보다 높고, 동시에 ADR 0001의 champion 대비 `+0.00002` 채택 문턱을 통과할 때로 고정한다.
M0가 계열 2 진입 조건을 통과하면 최종 결합 개선 여부와 무관하게 후보 풀 진입 자체는 성립하고, nested OOF 결과를 이유로 다시 제거하지 않는다.
17개 결합이 현재 최고와 같거나 낮으면 기존 16개 결합을 유지하고 spline 계열의 추가 탐색만 닫는다.

## 판정

새 댓글은 신뢰할 만한 전문가의 정성 증언이지만 재현 가능한 채택 증거는 아니다.
다만 공개 OOF와 자체 fold 0이 이미 보인 낮은 중복 및 작은 양수 기여를 독립적으로 보강하고, 이슈 149의 유일한 다양성 탈락 폭이 `0.0000035199`였으며 그 하드 게이트가 현재 계약에서 폐기됐다는 점을 함께 고려하면 M0 한 개의 제한적 검증을 다시 열 근거는 충분하다.
이 판정은 spline 구조가 좋다는 결론이 아니라 현재 nested OOF 계약으로 미해결 다양성 질문을 한 번 끝까지 측정하자는 결정이다.

## 출처

- [Kaggle 토론 원문](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)과 [공개 읽기 API](https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById?forumTopicId=735421&includeComments=true)는 본문, 댓글, 답글, 작성자와 게시 시각의 1차 출처다.
- [Tilii 댓글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421#3513480)과 [Ern711 답글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421#3513485)은 새 주장 각각의 직접 출처다.
- [공개 노트북 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)과 [공개 출력](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/output)은 원본 구조, OOF와 현재 공개 산출물의 1차 출처다.
- [Kaggle 공개 순위표](https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard)는 Optimistix의 현재 순위, 점수와 최상 제출 시각의 1차 출처다.
- [이슈 149 해결 결과](https://github.com/tmheo/predicting-smartphone-addiction/issues/149#issuecomment-5307002982), [자체 진입 진단](contextualized-spline-entry-diagnostic.md), [공개 OOF 참고 진단](s6e8-167-spline-notebooks.md), [후보 풀 감사](oof-pool-audit.md)와 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)은 저장소 판정의 근거다.
