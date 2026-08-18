# 모델 검증과 target encoding 자료

## Knowledge

- [scikit-learn: Cross-validation - evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
  cross-validation의 목적과 stratified K-fold split의 동작을 설명하는 공식 문서다.
  outer 5-fold의 역할을 확인할 때 사용한다.
- [scikit-learn: Target Encoder's Internal Cross fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html)
  학습 행의 target encoding을 cross-fitting으로 만들어 과적합을 막는 이유를 실험으로 보여 주는 공식 예제다.
  inner 10-fold의 역할을 확인할 때 사용한다.
- [Kaggle 노트북: S6E8 Lookup-Transformer + Insights lb 0.97041 - tamerlanomralinov](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)
  champion 모델(exp059)의 원문이다.
  정확값 lookup embedding과 PLR 수치 embedding의 원래 설계, 그리고 저자의 실험 서사를 확인할 때 사용한다.
  단, fold 규율과 어휘 적합 범위는 이 프로젝트와 다르므로 `configs/exp059_lookup_transformer.yaml`의 주석과 함께 읽는다.
- [Gorishniy, Rubachev, Babenko: On Embeddings for Numerical Features in Tabular Deep Learning (NeurIPS 2022)](https://arxiv.org/abs/2203.05556)
  PLR(periodic-linear) 수치 embedding의 출처 논문이다.
  수치 feature를 신경망에 넣는 embedding 방식들의 비교 근거가 필요할 때 사용한다.
- [MLWave: Kaggle Ensembling Guide](https://usermanual.wiki/Document/Kaggle20ensembling20guide.685545747.pdf)
  순위 평균, 가중 평균, 스태킹의 실전 비교를 다루는 고전 가이드다.
  결합 전략(수업 0003)의 원리와 스태킹 과적합 위험을 확인할 때 사용한다.
  원 블로그(mlwave.com)는 내려가 있어 보존본 링크를 쓴다.
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  `rank_logit_logistic` 결합기의 학습기 문서다.
  C, lbfgs, max_iter 같은 결합기 상수의 의미를 확인할 때 사용한다.

## Gaps

- 이 프로젝트의 inner fold 수를 10으로 정한 실험 근거는 아직 별도 문서로 정리되어 있지 않다.
- "합성 데이터에서 정확값이 반복되는 이유"(대회 데이터 생성 방식)를 다룬 고신뢰 자료는 아직 찾지 못했다. 대회 디스커션 정리(`docs/research/discussion-insights.md`)가 현재 최선이다.

## Wisdom (Communities)

- [Kaggle Learn: Data Leakage discussions](https://www.kaggle.com/learn/data-leakage)
  실제 대회에서 발생하는 누수 사례를 다른 참가자와 비교해 볼 때 사용한다.
- [Playground S6E8 디스커션](https://www.kaggle.com/competitions/playground-series-s6e8/discussion)
  Lookup-Transformer를 포함한 상위 기법이 실제로 논쟁되고 검증되는 곳이다.
  수업에서 배운 내용을 다른 참가자의 재현 결과와 대조할 때 사용한다.
