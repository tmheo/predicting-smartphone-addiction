# 디스커션 736062 분석: 'CV 이득 노이즈' 주장의 소재 확인과 실제 스레드 판정

이 문서는 GitHub 이슈 [디스커션 736062 'Your CV gain is probably noise' 본문과 코멘트 심층 분석](https://github.com/tmheo/predicting-smartphone-addiction/issues/240)의 산출물이며, 지도 [디스커션 736062와 CV 이득 노이즈 노트북 심층 분석](https://github.com/tmheo/predicting-smartphone-addiction/issues/239)의 일부다.
조사 기준일은 2026년 8월 19일이다.

## 결론 요약

- 티켓 전제와 달리, 디스커션 [736062](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736062)는 'Your CV gain is probably noise'가 아니다.
  실제 제목은 "🚀 Achieving 0.971+ LB: Residual NN + LightGBM Stacking Strategy"이고, 작성자는 dynamo14324(Yogesh Jadhav), 게시 시각은 2026-08-19 10:31 UTC, 코멘트는 0건이다.
- 'Your CV gain is probably noise'는 디스커션이 아니라 [alexchilton의 노트북](https://www.kaggle.com/code/alexchilton/your-cv-gain-is-probably-noise)(2026-08-19 09:11 UTC 마지막 실행)이다.
  S6E8 대회 포럼의 전체 스레드 37개를 전수 조회했고, 그 제목을 가진 스레드는 존재하지 않는다.
  따라서 "CV 개선분이 노이즈"라는 주장의 정량 근거 분석은 노트북을 다루는 [이슈 241](https://github.com/tmheo/predicting-smartphone-addiction/issues/241)의 몫이다.
- 실제 스레드 736062 자체는 정량 근거가 전혀 없는 홍보성 전략 나열 글로 판정한다.
  "리더보드 최상단 도달" 주장과 작성자의 실제 대회 순위 114위가 상충하고, 본문이 문장 중간에서 잘려 있으며, 약속한 노트북 링크도 없다.
  이 저장소의 판정 체계에 반입할 인사이트는 없다.
- "CV 이득이 노이즈일 수 있다"는 명제 자체는 이 저장소가 이미 스레드 [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005) 계열의 분석으로 정량 흡수했고, ADR 0001의 판정 규칙과 상충하지 않는다.
  상세한 대조는 아래 6절에 있다.

## 1. 티켓 전제 검증: 제목과 id의 불일치

티켓과 지도 이슈 239는 디스커션 736062를 'Your CV gain is probably noise'로 지칭했다.
세 가지 독립 경로로 실측한 결과, 이 연결은 성립하지 않는다.

| 검증 경로 | 실측 결과 |
| --- | --- |
| Kaggle 내부 API `GetForumTopicById` (topic 736062) | 제목 "🚀 Achieving 0.971+ LB: Residual NN + LightGBM Stacking Strategy", 작성자 dynamo14324, 게시 2026-08-19T10:31:31Z |
| Kaggle 내부 API `GetTopicListByForumId` (forum 9538188, 전 페이지) | 전체 스레드 37개 중 제목에 'Your CV gain'을 포함하는 스레드 없음 |
| `kaggle kernels list --user alexchilton` | 노트북 "Your CV gain is probably noise" 존재, 마지막 실행 2026-08-19 09:11:11 UTC, 상태 COMPLETE |

노트북(09:11)과 무관한 스레드 736062(10:31)가 같은 날 한 시간 간격으로 만들어졌다.
지도 작성 시점에 노트북 제목과 그 시각 최신 디스커션 id가 하나의 항목으로 합쳐진 것으로 판단한다.
Jina Reader 1차 조회에서도 같은 본문(스태킹 전략 글)이 렌더링됐으므로, id 736062의 내용은 조회 수단과 무관하게 일치한다.

## 2. 스레드 736062의 실제 본문 정리

본문 전문은 Kaggle 내부 API의 rawMarkdown으로 확보했다 ([736062](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736062)).
주장을 단계별로 정리하면 다음과 같다.

- 자칭 성과: "리더보드 최상단(top of the leaderboard)"에 도달한 핵심 전략을 공유한다며 제목에 "0.971+ LB"를 내걸었다.
- 특성 공학: 고카디널리티 범주형에 타깃 인코딩, 기준 XGBoost의 중요도 상위 특성 간 다항 상호작용, 오토인코더 기반 잡음 제거 표현.
- 모델 A: skip connection, batch normalization, dropout, Mish 활성 함수를 쓴 잔차 신경망.
- 모델 B: Optuna로 `num_leaves`, `feature_fraction`, `lambda_l1/l2`를 튜닝한 LightGBM.
- 결합: 두 모델의 OOF 예측을 Ridge 회귀 메타 러너로 스태킹.
- 교훈: 모든 모델에 같은 K-Fold 분할을 쓰는 것이 중요하고, 최종 제출은 확률 블렌딩보다 순위 기반 블렌딩이 public LB에서 더 안정적이었다.
- 마지막 문장은 "I've published a full starter notebook implementing this approach her"에서 잘려 있고, 노트북 링크는 없다.

## 3. 코멘트 전수 정리

코멘트는 0건이다.
세 경로가 일치한다: 페이지 렌더링의 "0 Comments" 표기, 스레드 목록 API의 `commentCount: 0`, 스레드 상세 API 응답에 코멘트 배열 부재.
따라서 "코멘트의 반론·보강·실측 사례"는 존재하지 않으며, 판정할 대상이 없다.

## 4. 신뢰도 판정: 정량 근거 없는 홍보성 글

본문의 어떤 주장에도 수치 근거가 없다.
CV 점수, fold 구성, 표본 수, ablation, 개선 폭 어느 것도 제시되지 않았다.
그 위에 신뢰를 깎는 정황이 겹친다.

- 성과 주장과 실측의 상충: API 응답의 작성자 대회 순위는 114위다.
  "리더보드 최상단"이라는 자기 서술과 맞지 않는다.
- 본문 미완성: 마지막 문장이 단어 중간("her")에서 잘렸고, 공유하겠다던 노트북 링크가 없다.
- 데이터셋과의 불일치: S6E8 학습 데이터에 타깃 인코딩이 필요한 고카디널리티 범주형이 사실상 없다는 것은 이 저장소의 EDA와 기존 디스커션 분석([discussion-insights.md](discussion-insights.md))에서 확립된 사실이다.
  본문은 대회 데이터를 특정하지 않은 범용 전략 서술에 가깝다.
- 작성자의 전력: 같은 작성자의 스레드 [732955](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732955)("결측 자체가 시그널")는 ablation 없는 정성 주장으로, 통제 실험을 제시한 [733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214)에 의해 반박된 바 있다([discussion-insights.md](discussion-insights.md) 2장).

판정: 반입할 인사이트가 없는 저신뢰 글이다.
개별 기법(잔차 신경망, 스태킹, 순위 블렌딩)은 일반론으로는 유효하지만, 이 글은 그 기법들이 이 대회에서 얼마를 벌었는지 아무것도 측정하지 않았다.
"public 점수 변화가 0.0001 미만이면 아무것도 측정한 게 아니다"라는 이 대회 포럼의 실용 규칙([734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005))에 비추면, 측정 자체가 없는 이 글은 판정 이전 단계다.

## 5. 'CV 이득이 노이즈' 주장의 실제 소재

지도가 검증하려는 명제("CV 개선분이 노이즈일 수 있다")의 정량 논의는 이 포럼에서 다음 소재들에 있다.

- 노트북 [alexchilton/your-cv-gain-is-probably-noise](https://www.kaggle.com/code/alexchilton/your-cv-gain-is-probably-noise): 명제의 직접 출처이며 [이슈 241](https://github.com/tmheo/predicting-smartphone-addiction/issues/241)이 코드와 마크다운을 심층 분석한다.
- 스레드 [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005): 시드 변경만으로 순위가 약 60계단 움직인다는 정량 분석과, 두 제출 점수 차이의 노이즈 폭(paired sigma 0.00009~0.00011), public 우위가 private로 전이될 확률 틀(95% 신뢰에 public 차이 2.16e-4 필요)까지 코멘트에서 정리·정정됐다.
- 스레드 [733618](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618): best-of-N 효과로 겉보기 순위 상승의 절반은 실력이 아니라는 분석.
- 스레드 [733214](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214) 코멘트: rho 자리에 예측 벡터 상관을 넣으면 리더보드 분해능을 과대평가한다는 교정.

이들은 모두 [discussion-insights.md](discussion-insights.md) 3장에 이미 정량 수치와 함께 반입돼 있다.
즉 "CV 이득 노이즈" 명제에 대한 이 포럼의 실질 근거는 새로 발견할 것이 아니라 이미 저장소가 흡수한 상태이고, 남은 신규 자료는 이슈 241의 노트북뿐이다.

## 6. 우리 판정 체계와의 대조

스레드 736062 자체는 판정 체계와 대조할 정량 주장이 없으므로, 지도의 질문에 맞춰 "CV 이득 노이즈" 명제를 우리 체계와 대조한다.

- 명제가 겨냥하는 실패 양식은 "노이즈 규모보다 작은 개선을 실재로 오인"하는 것이다.
  우리 체계는 이를 세 겹으로 막는다([ADR 0001](../adr/0001-experiment-adoption-contract.md)).
  첫째, 판정 지표를 public LB가 아니라 691,369행 전체의 5-fold OOF AUC로 고정했고 public 점수는 어떤 계열에서도 판정 근거가 아니다.
  둘째, 같은 시드끼리 짝지은 비교(스크리닝은 champion의 `seed_aucs[42]` 대비)로 시드 노이즈를 비교에서 상쇄한다.
  이는 [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005)가 보인 "짝지은 차이는 공통 오차가 상쇄되어 절대 오차보다 6~7배 작다"는 원리를 판정 규칙으로 먼저 구현한 것과 같다.
  셋째, 확정은 3시드 평균본으로 하되 3시드 중 2시드 이상 개선을 요구하고, 경계 구간(+0.00002~+0.0002)에서는 fold 5개 중 3개 이상 승리를 추가로 요구한다.
- 채택 문턱 +0.00002가 LB 노이즈 플로어(비슷한 블렌드 짝 기준 약 0.00015, [734005](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005) 코멘트)보다 훨씬 작다는 점은 상충이 아니다.
  문턱이 적용되는 척도가 다르기 때문이다.
  LB는 59,260행(test의 20%) 채점이지만 우리 OOF는 691,369행 채점이고, 결정적 파이프라인에서 같은 분할·같은 시드로 짝지어 비교하므로 비교 노이즈가 LB 짝 비교보다 한 자릿수 이상 작다.
  같은 구성의 재실행이 소수점 아래 9자리까지 같은 OOF를 낸다는 실측([mlflow-3seed-population-audit.md](mlflow-3seed-population-audit.md))이 이 척도의 재현성을 뒷받침한다.
- 시드 평균의 이득도 노이즈가 아니라 실측으로 관리된다.
  자체 3시드 평균 이득 +0.0000581, beicicc 2초기화 평균 이득 +0.0000944, LB 노이즈 플로어 0.00015 등 [realmlp-gap-attribution.md](realmlp-gap-attribution.md)의 수치들은 개선 몫과 노이즈 몫을 분리해 귀속하는 관행이 이미 작동함을 보여 준다.
  그 문서의 격차 분해는 표면 격차 0.0064를 척도 차이, 재현 격차, 구성 차이로 나눠 LB 노이즈 0.00015 이내로 정합시켰다.
- 남는 노출면은 두 가지다.
  하나는 경계 구간 밖의 +0.00002~ 근처 개선이 시드 2/3 승리만으로 확정될 때의 위양성 확률로, 이는 [이슈 242](https://github.com/tmheo/predicting-smartphone-addiction/issues/242)가 노트북 분석 결과와 함께 판단할 몫이다.
  다른 하나는 앙상블 계열(계열 3)의 nested OOF 문턱 +0.00002로, nested OOF는 outer fold별 선택·가중 학습의 분산이 더해지므로 단일 모델 짝 비교보다 노이즈가 클 수 있다.
  다만 채택된 앙상블들이 5/5 outer fold 개선을 동반했다는 기록(ADR 0001 개정 이력)이 보조 증거로 남아 있다.
- 결론: "CV 이득이 노이즈일 수 있다"는 명제는 우리 체계에 대한 반박이 아니라 우리 체계가 설계된 이유다.
  스레드 736062은 이 명제에 아무 근거도 더하지 않았고, 판정 규칙을 고칠 사유도 만들지 않았다.

## 7. 후속 처리

- 명제의 직접 출처인 노트북 분석은 [이슈 241](https://github.com/tmheo/predicting-smartphone-addiction/issues/241)에서 진행한다.
- 판정 체계 시사점 결정은 [이슈 242](https://github.com/tmheo/predicting-smartphone-addiction/issues/242)가 이 문서와 241 산출물을 입력으로 내린다.
- 스레드 736062는 반입 가치가 없으므로 `discussion-insights.md` 종합 문서에는 읽은 스레드 장부 갱신 외의 반영을 제안하지 않는다.
  장부 반영 여부는 지도의 미결 사항("종합 문서 반입 형태")과 함께 정한다.

## 출처

- 스레드 736062 본문: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/736062 (코멘트 0건, 2026-08-19 실측)
- 노트북: https://www.kaggle.com/code/alexchilton/your-cv-gain-is-probably-noise
- 시드·노이즈 정량 분석 스레드: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734005
- 리더보드 분해능·best-of-N 스레드: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733618
- 결측 플래그 ablation과 rho 교정 스레드: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733214
- 같은 작성자의 이전 스레드: https://www.kaggle.com/competitions/playground-series-s6e8/discussion/732955
