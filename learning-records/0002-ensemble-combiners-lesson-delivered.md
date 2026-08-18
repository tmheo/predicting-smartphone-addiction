# 0002: 상위 세 앙상블 제출의 구조와 결합 기법 수업 전달

- 날짜: 2026-08-18
- 상태: 전달됨 (`lessons/0003-three-ensembles-one-pool.html`)

## 배운 것

Public 0.97055-0.97057의 세 제출이 같은 후보 풀 위의 결합 방식 변주임을 다뤘다.

- 재료: 후보 풀 16 구성원(LightGBM 9, XGBoost 1, CatBoost 1, 로지스틱 one-hot 1, Lookup-Transformer 2, TabM 1, TabPFN 1)의 3시드 평균본.
  진입 기준은 단독 성능이 아니라 중복 아님(스피어만)과 앙상블 기여.
- #63 `rank_mean` pool3: 균등 순위 평균은 학습이 없으므로 구성원 선택이 곧 최적화. 감사 고정점이 exp059·exp070·exp081 세 개.
- #64 `rank_logit_logistic` pool16: 가중치를 학습하는 순간 nested OOF가 필요해지는 이유(결합기의 자기 채점 방지)를 중심에 둠.
  순위(눈금 강건)와 logit(확신 보존) 이중 표현.
- #65 `missing_segmented_rank_logit`: 결측 0-1/2-3/4+ 구간별 독립 로지스틱. 구간은 목표값 없이 사전 고정, 제출 전 유일 후보 선언, 전이 비율 0.76.

## 다음 세션 후보 (zone of proximal development)

- nested OOF(결합기 층)와 inner 10-fold(TE 층)의 구분을 아직 명시적으로 확인하지 않았다. 퀴즈 반응을 보고 복습 여부 결정.
- 왜 복잡한 결합기(탐욕 선택, XGBoost 2단)가 선형에 졌는지의 원리(결합기 층의 분산-편향)는 얕게만 다뤘다.
- 전체 데이터 재학습(full-data refit) 제출과의 관계는 범위 밖이었다. 사용자가 "전체 데이터 학습 제외"라고 언급했으므로 다음 주제로 자연스럽다.

## 수정 이력

- 없음
