# Lookup-Transformer의 핵심 구조를 수업 0002로 다뤘다

champion 모델(exp059)의 원리를 수업 0002에서 다뤘다: 합성 데이터의 정확값 반복이 lookup embedding의 근거이고, 컬럼 토큰은 lookup + PLR 두 채널의 합이며, NA와 UNK를 구분한다.
아직 노출 단계이며 퀴즈 응답 등 이해의 증거는 확인하지 못했다.

## Implications

- 다음 자연스러운 주제 후보: self-attention이 실제로 무엇을 계산하는지, EMA·value dropout 같은 학습 안정화 장치, 앙상블에서 다양성이 점수로 이어지는 원리.
- 사용자는 이 저장소의 fold 규율(수업 0001)과 exact-value 성질(이슈 #107 계열)을 이미 프로젝트 경험으로 알고 있으므로, 그 위에 쌓으면 된다.
