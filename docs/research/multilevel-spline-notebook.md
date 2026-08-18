# 다층 헤드 spline Transformer 공개 확장판 분석

이 문서는 GitHub 이슈 [P3 재개: contextualized spline M0의 전체 OOF·nested 결합 기여 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/171)의 실행 전에 도착한 새 공개 증거의 조사 결과다.
조사 시점은 2026-08-18이며, Kaggle 공개 읽기 API, Kaggle 공식 CLI로 내려받은 공개 노트북 소스와 실행 산출물, 저장소의 자체 OOF 기록만 사용했다.

## 결론

Ern711이 2026-08-17에 공개한 확장판은 기존 spline 백본을 그대로 두고 다층 예측 헤드와 행별 혼합기, 그리고 고정 기반 2단계 hypernetwork 보정만 더한 판이다.
소스 대조 결과 공유된 최상위 정의 86개가 완전히 일치하므로, 저자의 "core backbone is mostly unchanged" 서술은 코드 수준에서 사실로 확인된다.
따라서 이슈 171의 1단계 계획인 기존 `configs/exp085_contextual_spline_m0.yaml` M0 고정 검증은 바꿀 이유가 없고, M0는 여전히 공개 구조의 대표 후보다.

공개 전체 OOF는 `0.9675984885`로 이전 판 `0.9665204982`보다 `+0.0010779903` 높다.
그 이득의 대부분은 1단계의 다층 헤드와 혼합기에서 나오고, 2단계 hypernetwork의 fold별 이득은 `+0.000000`에서 `+0.000214`에 그친다.
저자 스스로도 이득이 매우 작아 추가 구조의 가치를 확신하지 못한다고 썼다.

[기존 재검토](spline-comment-reassessment.md)가 hypernetwork를 후보에서 제외한 근거였던 "구체적으로 재현할 수 없다"는 이제 성립하지 않는다.
공개 코드, 전체 OOF와 fold별 실행 기록이 모두 존재하기 때문이다.
다만 이 사실은 이슈 171의 범위를 여는 근거가 아니라, 171 해결 뒤 확장 변형을 별도로 판단할 때 쓸 새 증거다.

## 출처와 판본

[토론 735421의 새 댓글](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)은 Ern711(Expert)이 2026-08-17 15:37:06 UTC에 게시했고, 공개 읽기 API의 댓글 id는 `3513783`이다.
댓글이 가리키는 [공개 노트북](https://www.kaggle.com/code/ern711/multi-level-deep-univariate-spline-transformer?scriptVersionId=342998269)의 판본은 `scriptVersionId=342998269`이며, Kaggle 목록 API의 최신 실행 시각은 2026-08-17 12:50:43 UTC다.
같은 시점의 토론에는 이 밖의 새 댓글이 없다.

Kaggle 공식 CLI로 내려받은 산출물의 SHA-256은 다음과 같다.

- 노트북 소스(`.ipynb`): `3ae3e15d975892f90875d57e3877ba299ea073c5e019295b58870a09a35793fd`
- 실행 기록(log): `fb5a06d16a42dd60f94f1dfd4c880d5ce79a49c6b1c6a0f2c27c9327d044dd67`
- 전체 OOF(`multilevel_output_heads_oof.csv`): `6560892cefe522da8c276eba1eafd7742976260faf285516f431a47c7780a55f`
- fold 지표(`multilevel_output_heads_fold_metrics.csv`): `18707788360c20e9e8b4261f251ab105e4b0c0b2172c70d31eab385e95705050`
- 제출 파일(`submission.csv`): `4decee60220b10b8922fdb7146125607e43b3873684bd1f06fb7863206249a30`

공개 노트북 소스는 [Kaggle 공개 노트북 사용 조건](../agents/kaggle-public-notebook-licensing.md)에 따라 Apache License 2.0으로 참고했다.
이 조사에서는 코드를 저장소로 복사하지 않았고, 구조 분석과 산출물 대조에만 사용했다.

## 마크다운 설명 요약

노트북 마크다운은 설계 원리를 "피처 간 상호작용을 배우기 전에 피처별 표현을 먼저 강하게 학습한다"로 유지하면서 두 가지 확장을 설명한다.

첫째, 서로 다른 표현 깊이에 감독된 예측 경로 4개를 둔다.

- `ADD`: 수치 피처 token별 학습 점수의 합으로 예측한다.
- `UNI`: self-attention 이전의 단변량 token 묶음을 pooling해 예측한다.
- `ATTN`: self-attention 이후의 token 묶음을 pooling해 예측한다.
- `FINAL`: 표준화 수치, 단변량 token, attention token, 범주 embedding, 피처별 가산 점수와 가산 확률을 모두 받는 기존 최종 MLP다.

작은 학습형 혼합기가 네 logit을 행마다 다른 가중치로 결합하고, FINAL을 선호하도록 초기화되어 기존 직접-최종-헤드 기준선 근처에서 학습을 시작한다.

둘째, 학습을 2단계로 나눈다.
1단계는 혼합 예측을 주 목적 함수로 전체 모델을 끝까지 학습하고 ADD, UNI, ATTN에 보조 이진 분류 손실을 준다.
2단계는 1단계 최적 checkpoint를 고정하고, 작은 hypernetwork가 단변량 token의 scale·shift, attention 이후 token의 scale, 최종 MLP 입력 직전 표현의 scale이라는 세 지점에 행별 소규모 보정을 학습한다.
보정은 경계가 있고 0으로 초기화되어 2단계가 정확히 1단계 예측에서 시작한다.

교차검증 서술은 이전 판과 같다.
5-fold 층화 분할에서 목표·빈도 인코딩, 스케일러와 spline knot을 outer 학습 부분에서만 맞추고, 학습 행의 목표 인코딩은 내부 fold OOF 매핑으로 만든다.

## 코드 구조 대조

이전 판 [contextualized-deep-univariate-spline-transformer](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)와 새 판의 소스를 최상위 정의 단위로 정규화해 대조했다.

### 그대로인 부분

공유된 최상위 정의 86개가 문자 그대로 일치한다.
여기에는 백본 전체가 포함된다.

- 전처리와 피처: `engineer_features`, 중첩 목표·빈도 인코딩 `apply_nested_encodings`, 결측 표시 범주, `prepare_fold`의 fold별 스케일러·knot·어휘 구성.
- 수치 인코더: `DynamicGate`, `UnivariateSpline`, `MultiResolutionSplineExpert`, `CoarseSplineExpert`, `TinyMLPExpert`, `RawPathExpert`, `SwiGLU`, `LocalSwiGLUBlock`, `DeepUnivariateFeatureEncoder`, `DeepUnivariateEncoder`, `NonlinearPreSplineContext`.
- 상호작용과 가산 경로: `SmallAttentionBlock`, `InteractionEncoder`, `AdditiveHead`.
- 학습 설정: seed 21 분할, batch 4096, 35 epoch, patience 7, LR 7e-4, weight decay 5e-4, label smoothing 0.005와 피처별 spline 해상도 명세 전부.

### 바뀌거나 추가된 부분

바뀐 정의는 `SplineTransformerModel`, `compute_loss`, `train_one_fold` 셋뿐이고, 나머지는 전부 새 정의다.

- `PooledTokenHead`(신규): token 묶음의 원소별 mean과 max를 이어 붙여 한 logit을 내는 작은 분류기로, UNI(단변량 token 64차원)와 ATTN(attention token 64차원) 두 경로에 같은 구조를 쓴다.
- `MultiLogitMixer`(신규): 네 logit과 여섯 쌍별 절대 차이를 입력(10차원)으로 받아 hidden 16의 작은 망이 softmax 가중치 4개를 내고, 출력층은 0으로 초기화하되 FINAL bias만 `2.0`으로 두어 초기 혼합이 FINAL을 강하게 선호한다.
- `compute_loss`(변경): 혼합 logit의 주 손실 1.0에 ADD 0.3, UNI 0.1, ATTN 0.1의 보조 손실과 `1e-5` 계수의 혼합 가중치 entropy 벌점을 더한다.
- `SplineTransformerModel.forward`(변경): 기존 최종 logit을 `base_final_logit`으로 두고 혼합기가 최종 logit을 만든다.
  early stopping과 checkpoint 선택 지표도 혼합 logit의 검증 AUC다.
- `HyperAdaptableSplineTransformerModel`(신규): 1단계 모델을 상속해 세 보정 지점을 노출한 forward를 추가한다.
- `SmallMultiSiteHyperNetwork`(신규): 고정된 단변량 token 묶음을 피처별 bottleneck 32로 요약하고 hidden 192의 전역 망을 거쳐 세 지점의 보정을 낸다.
  token scale과 shift는 `±0.15`, attention 이후 scale과 최종 입력 scale은 `±0.05`의 tanh 경계를 갖고, 모든 보정 헤드는 0으로 초기화된다.
  shift는 피처별 학습 단위 방향 벡터에 행별 계수를 곱하는 방식이다.
- 2단계 학습(변경): 1단계 매개변수를 전부 고정하고 hypernetwork만 20 epoch, patience 5, LR 5e-4로 학습하며, 시작 상태의 검증 AUC가 1단계 최적과 정확히 같음을 실행 기록이 확인한다.

매개변수 규모는 기본 모델 약 714만, hypernetwork 약 147만이다.

## 공개 실행 결과

공개 실행 기록의 fold별 수치는 다음과 같다.
BASE는 1단계 최적 epoch의 헤드별 검증 AUC이고, MIXED가 1단계 선택 지표다.

| fold | ADD | UNI | ATTN | BASEFINAL | MIXED | hyper 최적 | hyper 이득 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.965540 | 0.966138 | 0.966793 | 0.966646 | 0.967685 | 0.967868 | +0.000183 |
| 1 | 0.965548 | 0.966396 | 0.966575 | 0.966274 | 0.967919 | 0.968071 | +0.000153 |
| 2 | 0.965664 | 0.966085 | 0.966354 | 0.966230 | 0.967495 | 0.967495 | +0.000000 |
| 3 | 0.965453 | 0.966115 | 0.966505 | 0.967090 | 0.967800 | 0.968014 | +0.000214 |
| 4 | 0.964346 | 0.965051 | 0.965837 | 0.966049 | 0.966824 | 0.966931 | +0.000106 |

전체 OOF는 FINAL `0.967598`, ADD `0.962729`이며 내려받은 OOF 파일 재계산으로 `0.9675984885`, `0.9627289521`을 확인했다.
혼합기의 평균 가중치는 fold에 걸쳐 ADD 약 0.10, UNI 약 0.14에서 0.18, ATTN 약 0.12에서 0.17, FINAL 약 0.57에서 0.63이다.
실행은 CUDA에서 fold당 30.5분에서 32.6분, 전체 약 158분이다.

이득의 분해는 다음과 같다.

- 이전 판 전체 OOF `0.9665204982` 대비 새 판 `0.9675984885`로 `+0.0010779903`이다.
- 2단계 hypernetwork의 fold별 이득은 평균 `+0.000131`이고, fold 2에서는 초기 상태를 한 번도 이기지 못했다.
  hypernetwork의 최적 epoch은 다섯 fold 모두 2 이하로 매우 이르다.
- 따라서 이득의 대부분은 1단계의 다층 헤드와 혼합기 몫이다.
  같은 epoch에서 MIXED는 BASEFINAL보다 `+0.000710`에서 `+0.001645` 높지만, checkpoint 선택이 MIXED 기준이므로 이 차이는 혼합기 몫을 다소 과대평가할 수 있다.

저자의 자평은 "CV 이득이 매우 작고, 더해진 복잡도를 고려하면 추가 구조가 실제로 가치가 있는지 확신하지 못한다"이며, 명확한 개선이 아니라 추가 실험으로 공유한다고 썼다.

## 자체 후보 풀 대비 참고 진단

공개 OOF는 seed 21 분할이고 저장소는 커밋된 seed 42 fold를 쓰므로, 아래 수치는 [기존 공개 OOF 참고 진단](s6e8-167-spline-notebooks.md)과 같은 참고값이며 정식 판정 자료가 아니다.
id와 목표값을 맞춘 뒤 현재 `artifacts/pool.yaml`의 22개 시드 평균본과 비교했다.

- 새 공개 OOF의 최근접 스피어만 상관은 `0.9829016977`(exp070_cat_exact_cats)로 중복 제거 기준 `0.998`보다 훨씬 낮다.
- 현재 champion `exp081_lookup_fold_initialization_avg3`와의 상관은 `0.9709779990`이다.
- 22개 균등 순위 평균 `0.968844710994`에 새 공개 OOF를 더하면 `0.968902422617`로 `+0.000057711623` 변한다.
- 같은 측정에서 이전 판 공개 OOF를 더하면 `+0.000048535667`이므로, 확장판이 참고 기여를 조금 더 키운다.
- 새 판과 이전 판 공개 OOF끼리의 스피어만 상관은 `0.9774248210`으로, 같은 백본에서도 헤드 구성 변경이 예측 순위를 상당히 바꿨음을 보인다.

## 이슈 171에 미치는 영향

이슈 171의 계획은 바꾸지 않는다.

- 1단계 M0 고정 검증은 그대로 유효하다.
  백본이 코드 수준에서 동일하므로, M0의 seed 42 전체 5-fold와 champion 짝비교라는 최소 검증 순서는 새 판 등장으로 흔들리지 않는다.
- 티켓이 닫아 둔 hypernetwork 범위도 그대로 둔다.
  다만 제외 근거는 [기존 재검토](spline-comment-reassessment.md)의 "재현 불가"에서 "공개 증거상 이득이 작고 M0 판정보다 우선하지 않음"으로 갱신된다.
  공개 fold별 이득 `+0.000000`에서 `+0.000214`는 이 저장소 채택 문턱 `+0.00002`와 규모가 겹치지만, 분할이 다르고 자체 대조가 없어 지금 열 이유가 되지 않는다.
- 새로 열 가치가 있을 수 있는 부분은 hypernetwork보다 1단계 다층 헤드와 행별 혼합기다.
  공개 증거에서 이득의 대부분이 여기서 났고, 구조 변경이 M0 위에 국소적으로 얹히는 형태라 짝지은 제거 대조가 쉽다.
  이 확장 변형을 별도 후보로 열지는 이슈 171의 M0 판정 결과가 나온 뒤 이 문서의 증거로 다시 판단한다.

## 출처

- [토론 735421](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)과 [공개 읽기 API](https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById?forumTopicId=735421&includeComments=true)는 새 댓글 본문, 작성자와 게시 시각의 1차 출처다.
- [Multi-Level Deep Univariate Spline Transformer 공개 노트북](https://www.kaggle.com/code/ern711/multi-level-deep-univariate-spline-transformer?scriptVersionId=342998269)과 그 공개 출력은 새 구조, 실행 기록과 OOF의 1차 출처다.
- [Contextualized Deep Univariate Spline Transformer 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)과 그 공개 출력은 이전 판 대조의 1차 출처다.
- [spline Transformer 새 댓글 근거 재검토](spline-comment-reassessment.md), [기존 공개 노트북 검토](s6e8-167-spline-notebooks.md), [자체 M0 진입 진단](contextualized-spline-entry-diagnostic.md)은 기존 판정과 참고 진단 방법의 저장소 근거다.
- [`artifacts/pool.yaml`](../../artifacts/pool.yaml)과 [`artifacts/champion.yaml`](../../artifacts/champion.yaml)은 참고 진단에 쓴 후보 풀과 champion 수치의 기록 원본이다.
- [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)은 채택 문턱과 중복 기준의 근거다.
