# 표 데이터 최신 기법의 10일 내 이식 가능성 조사

이 문서는 GitHub 이슈 [#192](https://github.com/tmheo/predicting-smartphone-addiction/issues/192)의 질문에 답한다.
질문은 학계와 업계의 표 데이터 최신 기법 가운데, 이 저장소의 파이프라인(고정 5-fold, fold-fit 누수 규율, Vast.ai GPU)에 10일 안에 이식 가능하면서, 현재 후보 풀(nested OOF 0.96951, champion exp081 Lookup-Transformer OOF 0.96920)에 기대 이득 +0.0002급 이상 또는 최근접 스피어만 상관 0.98 미만의 새 정보 축을 줄 수 있는 방법이 무엇인가다.
조사 시점은 2026-08-18이고, 1차 출처는 논문 원문(arXiv), 공식 저장소, 공식 문서다.

## 결론

이식 우선순위 상위 후보는 다섯 개다.

| 순위 | 후보 | 기대 이득 | 근거 품질 | 이식 비용 | 누수 위험 | 처리 제안 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | AutoGluon zeroshot portfolio 설정을 우리 fold에 이식해 다양성 구성원을 대량 선별 | 새 구성원 여러 개, 상관 0.98 미만 가능 | 높음(공개 설정 파일, TabRepo 실측) | 낮음(기존 adapter 재사용) | 없음 | 새 task 티켓 |
| 2 | Muon optimizer를 TabM과 Lookup-Transformer에 적용 | 구성원 단독 +0.0002급 | 높음(2026 표 데이터 전용 실측 벤치마크) | 낮음(학습기 교체) | 없음 | 새 task 티켓 |
| 3 | TabPFN-3 공식 fine-tuning으로 기존 풀 구성원을 강화 | 단독 상승과 비상관화 동시 기대 | 중간(공식 API와 changelog 실측, 대용량 재현은 부족) | 중간(24 GPU 시간 내 추정) | 낮음 | 라이선스 판단 후 조건부 티켓 |
| 4 | 결합기 상향: 복원추출 탐욕 선택과 가중치 정규화 CMA-ES | +0.0001급 | 중간(논문과 MIT 구현, 이득은 소폭이라고 자체 보고) | 낮음(하루 규모) | 낮음(nested 규율 유지) | 이슈 64 계열에 흡수 |
| 5 | xRFM(재귀 특성 커널 머신) 진입 진단 | 새 작동 원리 계열, 상관 0.98 미만 기대 | 중간(ICLR 2026 논문, 분류는 "동급" 수준 보고) | 중간(adapter 구현과 fold 0 진단) | 낮음 | 진입 진단 티켓 |

TabICLv2 fine-tuning, ModernNCA, TabDPT, Mitra, LimiX, TabFlex, KAN 계열, 같은 자료 자기 지도 사전학습, mixup류 증강, 앙상블 증류는 아래 근거로 이번 회차에서 제외한다.

## 전제와 기존 진단과의 겹침 정리

- 학습 자료는 약 691,369행, 원시 수치 12열이고 champion 특성 계획은 33열이다.
- 후보 풀은 16구성원이며 LightGBM, XGBoost, CatBoost 변형, 로지스틱 회귀, TabM, Lookup-Transformer 두 판, TabPFN-3 zero-shot(exp067_tabpfn3)을 이미 포함한다.
- zero-shot 문맥 학습 계열은 이 자료에서 이미 결판이 났다.
  TabICL 2.1.1(기본 체크포인트가 TabICLv2)은 fold 0 AUC 0.9417, TabR-S는 0.9420, Trompt는 0.9401, AMFormer는 0.7250으로 모두 진입 진단에서 탈락했다([tabiclv2-entry-diagnostic.md](tabiclv2-entry-diagnostic.md), [tabr-s-entry-diagnostic.md](tabr-s-entry-diagnostic.md), [trompt-entry-diagnostic.md](trompt-entry-diagnostic.md), [amformer-entry-diagnostic.md](amformer-entry-diagnostic.md)).
- 예외는 TabPFN-3이다.
  학습 fold 55만 행 전부를 문맥으로 넣는 구조라 소규모 문맥 ICL과 달리 이 자료에서 진입에 성공했고, 이슈 [#102](https://github.com/tmheo/predicting-smartphone-addiction/issues/102)로 이미 풀에 반입됐다(최근접 스피어만 0.9950, 표준 순위 평균 기여 +0.0000284).
- 조각선형 표현 계열은 contextualized spline 진단이 다양성 대조 상한을 근소하게 넘지 못하고 닫혔다([contextualized-spline-entry-diagnostic.md](contextualized-spline-entry-diagnostic.md)).
- 따라서 이번 조사는 "zero-shot 새 모델 계열"이 아니라, 설정 포트폴리오, 학습 기법, fine-tuning, 결합기, 그리고 신경망도 GBDT도 아닌 새 작동 원리에 초점을 맞췄다.

## 1순위: AutoGluon zeroshot portfolio 설정 이식

### 내용

AutoGluon이 244개 자료에서 학습해 고른 모델 설정 포트폴리오를 가져와, 우리 파이프라인과 고정 fold로 직접 학습해 다양성 구성원 후보를 대량으로 만든다.

### 근거

- 포트폴리오는 AutoGluon 저장소(Apache-2.0)에 하이퍼파라미터 dict가 그대로 적힌 Python 파일로 공개돼 있다.
  `zeroshot_portfolio_2023.py`는 약 100개 설정으로 `best_quality` 프리셋을 구동하고, `zeroshot_portfolio_2025.py`는 LightGBM 3개, CatBoost 5개, XGBoost 2개, TabM 6개 등 19개 설정을 우선순위 순으로 담는다 ([2023 파일](https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2023.py), [2025 파일](https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2025.py)).
- 이 포트폴리오를 학습한 TabRepo 연구는 zeroshot 포트폴리오가 AutoGluon 0.8 대비 1.0에서 75% 승률을 만든 핵심이라고 보고한다 ([TabRepo 논문](https://arxiv.org/abs/2311.02971), [AutoGluon 1.0 릴리스](https://github.com/autogluon/autogluon/releases/tag/v1.0.0)).
- TabArena(NeurIPS 2025)는 튜닝 단독보다 "튜닝 후 설정 간 사후 앙상블"이 순위를 바꾸는 핵심 단계라고 측정했다.
  TabM, LightGBM, RealMLP는 사후 앙상블이 있을 때만 CatBoost를 이긴다 ([TabArena 논문](https://arxiv.org/abs/2506.16791)).
- ML Contests의 2025년 보고서와 AutoGluon 공식 AWESOME 목록은 2024년 Kaggle 표 데이터 대회 18개 중 15개 메달(금 7개)을 보고한다 ([보고서](https://mlcontests.com/state-of-machine-learning-competitions-2025/), [AWESOME.md](https://github.com/autogluon/autogluon/blob/master/AWESOME.md)).
  이 수치는 대조 실험이 아닌 집계 보고라 방향 근거로만 쓴다.

### 이식 방식과 판단

- AutoGluon 자체를 학습 루프로 쓰지 않는다.
  `TabularPredictor`의 `groups` 매개변수가 사용자 fold를 받지만 공식 문서가 experimental이라고 경고하므로, 설정 dict만 가져와 이미 있는 LightGBM, XGBoost, CatBoost, TabM adapter로 우리 fold 규율 안에서 학습하는 편이 안전하다 ([TabularPredictor 문서](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)).
- 설정당 학습은 기존 계열 재사용이라 구현 비용이 거의 없고, 약식 검증으로 대량 선별한 뒤 통과분만 스크리닝으로 올리면 된다.
- 누수 위험은 자체 파이프라인 실행이므로 새로 생기지 않는다.
- 기대 이득: 벤치마크들이 일관되게 "결합기 개선보다 풀 다양성이 이득의 원천"이라고 보고하므로, 상관 0.98 미만 구성원을 추가할 가장 확률 높은 경로다.

## 2순위: Muon optimizer로 신경망 구성원 강화

### 내용

TabM과 Lookup-Transformer의 AdamW를 Muon으로 바꿔 같은 설정에서 단독 OOF를 올린다.

### 근거

- 2026년 표 데이터 전용 optimizer 벤치마크가 15개 optimizer를 17개 자료의 MLP와 TabM에서 비교했다.
  Muon은 vanilla MLP에서 AdamW 대비 상대 점수 +0.32%(12승 3무 2패), TabM 변형에서 +0.21%에서 +0.40%를 측정했고, TabM 기준 학습 시간은 1.18배에서 1.57배다 ([논문](https://arxiv.org/abs/2604.15297)).
- AdamW+EMA도 vanilla MLP에서 +0.66%를 측정했지만 강한 구조일수록 이득이 줄어든다고 보고한다.
  우리 champion은 이미 EMA 0.999를 쓰므로 EMA 축은 소진됐고 Muon이 남은 축이다.
- 우리 TabM(exp065)은 이미 PWL 수치 embedding을 쓰므로, 수치 embedding 추가라는 잘 알려진 상향 축도 소진 상태다.
  따라서 optimizer가 TabM 계열에 남은 가장 근거 강한 학습 기법이다.

### 이식 방식과 판단

- 학습기 내부의 optimizer 생성만 바뀌므로 이식 비용이 낮고 fold 규율에 영향이 없다.
- 벤치마크의 +0.2%대 상대 이득이 우리 자료에서 크게 줄어도 +0.0002(약 +0.02% 상대) 문턱은 넘을 여지가 있다.
- 실패해도 seed 42 짝비교 한 번으로 닫을 수 있어 중단 비용이 작다.

## 3순위: TabPFN-3 공식 fine-tuning

### 내용

풀에 이미 있는 TabPFN-3 zero-shot 구성원을 공식 fine-tuning API로 각 fold 학습 부분에 맞춰 이어 학습한다.

### 근거

- `tabpfn` 패키지는 v8.3.0(2026-08-13)이 최신이고 TabPFN-3(2026-05-12 공개, [기술 보고서](https://arxiv.org/abs/2605.13986))이 기본 모델이다 ([릴리스](https://github.com/PriorLabs/TabPFN/releases), [changelog](https://github.com/PriorLabs/TabPFN/blob/main/CHANGELOG.md)).
- 공식 fine-tuning 예제와 API가 있고, changelog는 Higgs 이진 분류 fine-tuning 예제의 test AUC가 v8.2.0에서 v8.3.0 개선으로 0.8247에서 0.8322로 오른 실측을 기재한다 ([finetune 예제](https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_classifier.py)).
- 반대 근거도 있다.
  fine-tuning을 체계 비교한 WWW 2026 논문은 zero-shot이 이미 강한 조건에서 전체 fine-tuning이 오히려 정확도와 보정을 해치는 경우가 잦다고 보고한다 ([논문](https://arxiv.org/abs/2601.09654)).
- 메모리는 v8.0.4의 fp8 KV 캐시와 v8.0.8의 KV 캐시 CPU 이송으로 24GB급에서 55만 행 문맥이 현실적이라고 추산되지만, 공식 수치는 H100 기준이라 우리 GPU 실측이 필요하다.

### 이식 방식과 판단

- 각 outer fold의 학습 부분만으로 fine-tuning하고 검증 부분은 평가에만 쓰면 기존 fold-fit 규율과 같다.
- 현재 zero-shot 구성원의 최근접 상관이 0.9950이므로, fine-tuning이 단독 AUC를 올리면서 예측 표면을 바꾸면 단독 상승과 비상관화를 동시에 노릴 수 있다.
- 가장 큰 장벽은 성능이 아니라 라이선스다.
  TabPFN-3 가중치는 비상업 라이선스(테스트, 평가, 내부 벤치마킹 허용)이고 Kaggle 대회 사용은 명시돼 있지 않다 ([가중치 저장소](https://huggingface.co/Prior-Labs/tabpfn_3)).
  zero-shot 반입 때의 판단을 fine-tuning 산출물에도 적용할 수 있는지 사람이 먼저 확정해야 하므로 조건부 티켓으로 제안한다.

## 4순위: 결합기 상향(복원추출 탐욕과 정규화 CMA-ES)

### 내용

nested 평가 절차 안에서 현재 탐욕 선택을 Caruana 원형(복원추출과 부트스트랩 선택 평균)으로 올리고, 가중치 정규화 CMA-ES를 결합 전략 한 개로 추가해 비교한다.

### 근거

- CMA-ES 사후 앙상블 논문은 ROC AUC에서 무제약 가중치 탐색이 심하게 과적합하며, 가중치 정규화를 넣으면 탐욕 선택과 같거나 더 낫다고 71개 자료에서 측정했다 ([논문](https://arxiv.org/abs/2307.00286)).
- 품질 다양성 선택(QDO-ES)은 검증에서만 유의하고 시험에서는 유의하지 않았다 ([논문](https://arxiv.org/abs/2307.08364)).
- 2026년 foundation model 앙상블 연구는 최강 단일 모델 대비 최선 앙상블 이득이 정확도 +0.18%에 그쳤고, 실무 기본값으로 탐욕 선택을 권고한다 ([논문](https://arxiv.org/abs/2605.18696)).
- 위 방법들은 MIT 라이선스 `phem` 라이브러리에 구현돼 있으나 저장소가 작업 중 상태라고 표시한다 ([phem](https://github.com/LennartPurucker/phem)).

### 이식 방식과 판단

- 이 저장소의 결합 전략 계약(fit/predict/summary adapter)에 전략 하나를 추가하는 하루 규모 작업이다.
- 문헌이 말하는 기대 이득은 1e-4급으로 정확히 우리 문턱 언저리라, 큰 도약이 아니라 마무리 손질이다.
- 결합 전략 비교는 이미 이슈 64 계열(nested OOF 평가)이 소유하므로 새 축이 아니라 그 범위의 확장으로 처리한다.

## 5순위: xRFM 진입 진단

### 내용

재귀 특성 커널 머신(xRFM)을 새 모델 계열로 fold 0 진입 진단에 올린다.

### 근거

- xRFM(ICLR 2026)은 AGOP 기반 특성 학습 커널과 나무형 분할을 결합해, 회귀 100개 자료 최상위, 분류 200개 자료에서 GBDT 이상이라고 보고한다 ([논문](https://arxiv.org/abs/2508.10053)).
- MIT 라이선스 공식 구현이 pip으로 설치되고 GPU 가속, `predict_proba`, AUC 튜닝 지표를 지원한다 ([저장소](https://github.com/dmbeaglehole/xRFM)).
- 신경망도, GBDT도, 검색 기반도 아닌 작동 원리라 현재 풀에 없는 유일한 계열이고, 상관 0.98 미만이 나올 확률이 구조적으로 가장 높다.

### 이식 방식과 판단

- 분류 성능 주장이 "동급 이상" 수준이고 69만 행 규모의 독립 재현이 없어, 기존 진입 진단 절차(fold 0, champion 대비 -0.01 허용 폭, 시간과 메모리 관문)를 그대로 적용해 싸게 판정한다.
- 기존 zero-shot 실패 계열과 달리 자료 전체를 학습하는 방식이라 TabICL류 실패가 이 후보의 반증이 되지 않는다.
- adapter 구현과 원격 진단 실행을 합쳐 2일에서 3일 규모다.

## 제외한 후보와 근거

- TabICLv2 fine-tuning: zero-shot이 champion 대비 -0.027이었고, fine-tuning의 문헌상 기대 이득(대개 AUC 0.005에서 0.01)이 이 격차에 못 미친다 ([TabICL](https://github.com/soda-inria/tabicl), [fine-tuning 비교](https://arxiv.org/abs/2601.09654)).
  라이선스(BSD-3)가 가장 깨끗하다는 전략적 가치만 있어, TabPFN-3 경로가 라이선스로 막힐 때의 대안으로만 남긴다.
- ModernNCA: TALENT 벤치마크 분류 1위지만 TabR-S와 같은 검색 기반 계열이고, 그 계열이 이 자료에서 -0.0265로 탈락했다 ([논문](https://arxiv.org/abs/2407.03257)).
  xRFM이 먼저 실패하고 새 근거가 생길 때만 재고한다.
- TabDPT v1.2, Mitra, LimiX, TabFlex, EquiTabPFN: 각각 TabArena-Lite에서 TabICLv2 아래, 공식 문서의 5천 행 미만 권장, 가중치 상업 허가 필요와 대용량 근거 부재, 정확도 열세, 소규모 지향으로 탈락한다 ([TabDPT](https://arxiv.org/abs/2608.01400), [Mitra 문서](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html), [LimiX](https://github.com/limix-ldm-ai/LimiX), [TabFlex](https://arxiv.org/abs/2506.05584)).
- 같은 자료 자기 지도 사전학습(SCARF, VIME 계열): 라벨이 풍부하면 지도 학습이 일관되게 우세하다는 측정이 있고, 우리는 라벨 69만 개를 다 가진다 ([전이 연구](https://arxiv.org/abs/2206.15306)).
- mixup류 증강: 강한 기준 모델에서 +0.0002급 이득을 보인 공개 측정이 없고, 정확값 어휘를 쓰는 champion에서는 값 정체성 신호를 파괴한다.
- 앙상블 증류와 시험 자료 의사 라벨: S6E3 우승 해법이 쓴 패턴이지만, 이 저장소는 증류를 누수 위험과 nested 비용으로 이미 기각했고 의사 라벨은 이슈 68이 소유한다 ([기존 판정](remaining-experiment-space-audit.md)).
- AutoGluon을 결합기 전체로 도입: `groups` 매개변수가 experimental이고, 결합 규율(판정 계약, nested 평가)이 이미 저장소에 있으므로 설정 포트폴리오만 가져오는 1순위로 대체한다.

## 한계

- TabArena는 학습 행 10만에서 25만 규모까지만 다루므로, fold당 55만 행 영역의 foundation model 우세 주장은 Prior Labs 내부 벤치마크뿐이다 ([TabArena](https://arxiv.org/abs/2506.16791), [TabPFN-3 보고서](https://arxiv.org/abs/2605.13986)).
- Muon과 xRFM의 이득 추정은 다른 자료의 측정을 옮긴 것이라, 채택 여부는 ADR 0001의 자체 실행 판정만 근거가 될 수 있다.
- Kaggle 공개 해법의 수치는 저장 출력이 없는 한 작성자 보고치다.
