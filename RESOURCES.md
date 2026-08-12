# 모델 검증과 target encoding 자료

## Knowledge

- [scikit-learn: Cross-validation - evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
  cross-validation의 목적과 stratified K-fold split의 동작을 설명하는 공식 문서다.
  outer 5-fold의 역할을 확인할 때 사용한다.
- [scikit-learn: Target Encoder's Internal Cross fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html)
  학습 행의 target encoding을 cross-fitting으로 만들어 과적합을 막는 이유를 실험으로 보여 주는 공식 예제다.
  inner 10-fold의 역할을 확인할 때 사용한다.

## Gaps

- 이 프로젝트의 inner fold 수를 10으로 정한 실험 근거는 아직 별도 문서로 정리되어 있지 않다.

## Wisdom (Communities)

- [Kaggle Learn: Data Leakage discussions](https://www.kaggle.com/learn/data-leakage)
  실제 대회에서 발생하는 누수 사례를 다른 참가자와 비교해 볼 때 사용한다.
