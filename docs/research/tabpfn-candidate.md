# TabPFN 최신판의 다양성 후보 적합성 리서치

이슈 [#101](https://github.com/tmheo/predicting-smartphone-addiction/issues/101)의 실행 기록이다.
[#61](https://github.com/tmheo/predicting-smartphone-addiction/issues/61)의 신경망 다양성 트랙에 TabPFN 최신판이 champion급 단독 성능 후보로 들어갈 수 있는지, [#77](https://github.com/tmheo/predicting-smartphone-addiction/issues/77)의 약한 모델 배제 기준(단독 AUC 0.955 이하는 앙상블 기여 ~0)을 전제로 조사했다.
조사 시점은 2026-08-14이고, 모든 주장에 1차 출처 URL을 달았다.

## 결론 요약

- 사용자가 언급한 "TabPFN-3"은 실재하며, 2026-05-12에 공개된 현행 최신 공개 가중치 모델이다.
  파이썬 패키지 `tabpfn`의 최신판은 8.3.0(2026-08-13)이고 TabPFN-3이 기본 모델이다.
- 가중치 라이선스(TABPFN-3 License v1.0)는 비상업 전용이지만, Kaggle 등 플랫폼의 데이터 사이언스 대회 참가를 비상업 목적으로 명시해 허용한다.
  라이선스는 통과다.
- 규모 한도(1M 행 x 200 피처)도 fold당 학습 문맥 ~553k 행을 덮으므로 통과다.
  다만 공식 벤치마크 하드웨어는 H100이고, T4에서의 실행 시간은 공개 실측이 없어 미지수다.
- 대규모 playground 실전에서 champion급 단독 성능을 낸 증거는 찾지 못했다.
  찾은 유일한 유사 규모 사례(S6E5 8위)에서 TabPFN은 200k 부분표본으로 CV 0.9488, 같은 팀의 CatBoost 0.9532 대비 -0.0044의 약체-다양성 역할이었다.
- 권고: 3-seed 확정까지 가는 정식 실험 티켓을 바로 열지 말고, fold 0 단일 스모크 게이트가 앞에 붙은 조건부 실험 티켓 하나를 연다.
  게이트 미달이면 #77과 같은 근거로 즉시 닫는다.
  상세 구성은 마지막 절에 있다.

## 1. 버전 확인: "TabPFN-3"은 실재한다

- TabPFN-3은 Prior Labs가 2026-05-12에 공개한 표 형식 자료용 기반 모델이다.
  기술 보고서는 [priorlabs.ai/technical-reports/tabpfn-3](https://priorlabs.ai/technical-reports/tabpfn-3)와 [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)(Grinsztajn 외 41인, "TabPFN-3: Technical Report", 2026-05-13 제출)이다.
- 버전 연대표(교차 확인: [Wikipedia TabPFN](https://en.wikipedia.org/wiki/TabPFN), [PyPI 릴리스 이력](https://pypi.org/project/tabpfn/#history), [GitHub 릴리스](https://github.com/PriorLabs/TabPFN/releases)):
  - TabPFNv2: 2025-01 공개(Nature 게재), 10k 행 x 500 피처.
  - TabPFN-2.5: 2025-11-06 공개, 50k 행 x 2k 피처, 보고서는 [arXiv 2511.08667](https://arxiv.org/abs/2511.08667).
  - Scaling Mode: 2025-11 말 도입, 학습 표본 상한을 제거하는 실행 방식으로 최대 10M 행 실험을 보고.
  - TabPFN-2.6: 2.5와 3 사이의 중간 공개판, 100k 행 x 2k 피처([공식 모델 문서](https://docs.priorlabs.ai/models)).
  - TabPFN-3: 2026-05-12 공개, 1M 행 x 200 피처(셀 예산 트레이드오프로 100k x 2k 또는 1k x 20k).
  - TabPFN-3-Plus: 2026-06 공개, thinking 모드와 텍스트 피처를 더한 상위판이나 API·기업 라이선스 전용이라 공개 가중치가 없다([공식 모델 문서](https://docs.priorlabs.ai/models)).
- 파이썬 패키지 `tabpfn`은 2.2.1(2025-09-17) 이후 6.0.0(2025-11-06, TabPFN-2.5 도입), 7.0.0(2026-03-24), 8.0.0(2026-05-12, TabPFN-3 기본화)으로 올라갔고, 최신판은 8.3.0(2026-08-13)이다([PyPI](https://pypi.org/project/tabpfn/#history)).
  버전은 `ModelVersion.V3`, `ModelVersion.V2_6` 식으로 골라 쓴다([GitHub README](https://github.com/PriorLabs/TabPFN)).
- 공개 가중치는 [Hugging Face Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3)에 있고, 2.5/2.6/3 가중치는 접근 승인(gated)이라 PriorLabs 계정 인증 또는 `TABPFN_TOKEN` 환경 변수가 필요하다([GitHub README](https://github.com/PriorLabs/TabPFN)).
  Kaggle Models에도 공식 배포([prior-labsai/tabpfn-2-5](https://www.kaggle.com/models/prior-labsai/tabpfn-2-5))가 있어 Kaggle 커널 사용을 전제로 한 배포가 이미 이뤄지고 있다.

## 2. 라이선스: Kaggle 대회 사용은 명시적으로 허용된다

- 코드와 TabPFNv2 가중치는 Prior Labs License(Apache 2.0에 표기 의무를 더한 것)이고, TabPFN-2.5/2.6/3 가중치는 각 판 전용 비상업 라이선스다([GitHub README](https://github.com/PriorLabs/TabPFN)).
- TabPFN-3 가중치의 라이선스는 TABPFN-3 License v1.0(비상업)이다([전문](https://huggingface.co/Prior-Labs/tabpfn_3/raw/main/LICENSE)).
  핵심 조항을 그대로 옮긴다.
  - 허용 범위: "access, use, create Derivatives of, and Distribute the TABPFN-3 Model and Derivatives solely for your Non-Commercial Purposes".
  - 비상업 목적 정의에 대회가 명시된다: "This includes internal benchmarking, academic research, and experimentation on private or public datasets as well as Data Science Competitions as defined below".
  - 대회 정의: "a publicly accessible contest hosted on established platforms (such as Kaggle, DrivenData, or ChallengeData) ... where participants compete to develop predictive models for specified datasets".
  - 출력물: "We claim no ownership rights in and to the Outputs"이되, 출력물도 비상업 목적에만 쓸 수 있고 경쟁 모델의 학습·증류에는 쓸 수 없다.
  - 표기 의무: 배포 시 "The TABPFN-3 Model is licensed by Prior Labs GmbH under the TABPFN-3 Non-Commercial License" 문구를 남긴다.
- 판정: S6E8은 상금 없는 playground 대회이고 결과를 상업 의사결정이나 유료 산출물에 쓰지 않으므로, 위 정의의 비상업 대회 사용에 정확히 들어맞는다.
  공개 노트북이나 문서로 예측을 배포할 때 위 표기 문구만 남기면 된다.

## 3. 691k 행 적용 경로와 T4 실행 시간

- 우리 자료는 691,370행 x 13열이라 fold당 학습 문맥이 ~553k행, 검증 fold가 ~138k행이다.
  TabPFN-3의 1M 행 x 200 피처 한도 안이므로, 부분표본 없이 fold 학습분 전체를 문맥으로 넣는 것이 공식 경로다([기술 보고서](https://priorlabs.ai/technical-reports/tabpfn-3)).
- 메모리: 공식 OOM 문서가 T4 기준 추정치를 준다([OOM 문서](https://docs.priorlabs.ai/troubleshooting/OOM-errors)).
  1 estimator, 200 피처, 테스트 1,000행 조건에서 학습 100k행 ~3GB, 500k행 ~10GB VRAM이다.
  13열짜리 우리 자료의 553k행 문맥은 T4 16GB에 실릴 가능성이 높고, 테스트 예측은 1,000행 안팎 청크로 나누고 `fit_mode="fit_with_cache"`로 문맥 재인코딩을 건너뛰는 것이 공식 권장이다.
  부족하면 `memory_saving_mode=True`와 `SUBSAMPLE_SAMPLES`(예: 50k 균형 부분표본 + `n_estimators` 증가) 조합이 공식 대안이다.
- 하드웨어: FAQ가 최소 T4, 권장 A100/H100을 명시한다([FAQ](https://docs.priorlabs.ai/faq)).
  단 기술 보고서의 "1M 행에서 빠른 추론" 수치는 H100 단일 GPU 기준이고, T4는 fp16 연산력이 H100의 대략 1/10 이하인 데다 세대 차이로 최적화(예: FlashAttention-3의 Hopper 전용 경로, [v8.3.0 릴리스 노트](https://github.com/PriorLabs/TabPFN/releases))를 못 받는다.
- 실행 시간 추정(공개 실측 없음, 자체 추정임을 명시한다).
  fold당 예측 대상은 OOF ~138k행 + 테스트 ~296k행 = ~434k행이고, 1,000행 청크로 ~434회 예측 호출이다.
  H100에서 청크당 1초 미만이라는 보고를 T4 감속비 10~30배로 환산하면 estimator 하나에 fold당 대략 1~4시간대이고, 기본값 `n_estimators="auto"`가 복수 추정기를 쓰면 그 배수가 된다.
  seed 42 스크리닝(5 fold)만으로 한 Kaggle 세션(T4 x2, 12시간)을 넘길 위험이 있고, 3-seed 확정(15 fold-실행)은 여러 세션이 필요할 수 있다.
  이 수치는 스모크 실행으로 반드시 실측해야 한다.
- fine-tuning 경로: 공식 문서에 경사 기반 fine-tuning API가 있고([fine-tuning 문서](https://docs.priorlabs.ai/capabilities/fine-tuning)), 실자료로 fine-tune한 Real-TabPFN-2.5 공개본이 보고돼 있다([arXiv 2511.08667](https://arxiv.org/abs/2511.08667)).
  tabpfn-extensions의 AutoTabPFN(튜닝+앙상블)은 50k행 규모까지만 검증돼 있어([arXiv 2511.08667](https://arxiv.org/abs/2511.08667)) 우리 규모의 1차 경로로는 부적합하다.

## 4. 대규모 Kaggle 실전 성능 증거

- 공식 주장: TabPFN-3은 "1M 학습 행까지 8시간 튜닝한 GBDT 기준선을 이긴다"고 보고하고, TabArena 최대 서브셋에서 forward pass만으로 전 모델을 앞선다([arXiv 2605.13986](https://arxiv.org/abs/2605.13986)).
  대형 자료 벤치마크는 100k/250k/500k/1M 행 부분표본으로 구성했다.
  단 이는 실자료 벤치마크 기준이고 자사 보고서라는 한계가 있다.
- Kaggle 실전 증거(유사 규모): Playground S6E5(2026-05, F1 피트스톱 예측, 자료 약 78MB 33열) 8위 팀의 앙상블 명세에 TabPFN이 "200k 1seed"(CV 0.9488)와 "150k 2seed"(CV 0.9482) 구성으로 들어 있다([8th Place - L5 Ensemble](https://www.kaggle.com/c/playground-series-s6e5/writeups/l5-ensemble)).
  같은 팀의 CatBoost는 0.9532, RealMLP는 0.9545였고, 저자는 TabPFN을 단독으로는 약하지만 오류 패턴이 달라 기여하는 약체-다양성 모델로 분류했다.
  즉 전체 자료가 아닌 150k~200k 부분표본으로 돌렸고, 단독 성능은 GBDT 대비 -0.004~-0.006이었다.
- 소표본 대회에서는 단독 상위권 사례가 있다: S5E3(소규모 자료)에서 TabPFN + 기본 피처 공학만으로 37위를 한 기록이 있다([37th place writeup](https://www.kaggle.com/competitions/playground-series-s5e3/writeups/kirderf-37th-place-solution-tabpfn-with-only-comp-)).
- TabPFN-3 공개(2026-05-12) 이후의 대형 playground(S6E6~S6E8)에서 TabPFN-3이 champion급 단독 성능을 냈다는 writeup은 검색으로 찾지 못했다.
  부재의 증명은 아니지만, "대형 playground에서 champion급"을 지지하는 실전 증거는 현재 없다.
- 해석: 이 대회의 champion(exp059, OOF 0.96892)은 fold 안에서 학습한 TE 피처가 지배하는 구성이다.
  보고서 벤치마크는 원시 피처 기준의 실자료라서, 합성 playground + TE 피처 공학 영역으로 성능이 그대로 이전된다는 보장이 없다.
  반대로 TabPFN에 champion과 같은 fold 내 TE 피처를 먹이는 것은 가능하므로, 원시 피처 성능만으로 미리 닫는 것도 근거가 약하다.

## 5. 판정 계약(ADR 0001) 적합성

- 시드 재현성: FAQ가 "고정 시드와 동일 환경에서 TabPFN 추론은 결정적이다"라고 명시하고, 하드웨어(CPU/GPU/MPS)가 바뀌면 미세 변동이 있을 수 있다고 적는다([FAQ](https://docs.priorlabs.ai/faq)).
  같은 Kaggle T4 환경에서 seed 42 스크리닝 후 3-seed 확정으로 가는 [ADR 0001](../adr/0001-experiment-adoption-contract.md) 경로와 충돌하지 않고, `n_estimators` 앙상블의 시드가 곧 모델 시드 축이 된다.
- 예측 출력: sklearn 호환 `predict_proba`가 numpy 배열을 반환하므로, 파이프라인에서 float64로 받아 OOF·테스트 예측을 보존하는 계약은 문제없다.
- fold 계약: TabPFN의 "학습"은 문맥 인코딩이라 fold당 재학습 비용이 없고, 문맥을 fold 학습분으로 제한하면 committed 5-fold 계약과 정확히 맞는다.
  비용의 전부가 예측 쪽이라, 5 fold x 3 seed = 15회의 예측 비용이 그대로 곱해지는 구조인 점만 3절의 시간 추정과 함께 계획하면 된다.
- 가중치 접근: gated 가중치라서 커널에 `TABPFN_TOKEN`(Kaggle Secrets) 또는 Kaggle Models 배포본 연결이 필요하다.
  S6E8은 CSV 제출 대회라 인터넷 켠 커널로 받는 데 제약이 없다.

## 권고

정식 실험 티켓을 무조건 열기에는 근거가 부족하고, #77 논리로 바로 닫기에도 근거가 부족하다.
불확실성의 정체는 두 가지다: (1) 대형 playground에서 champion급 단독 성능의 실전 증거가 없다, (2) T4에서 553k 문맥의 실행 시간이 실측된 적이 없다.
둘 다 fold 하나짜리 저비용 실행으로 해소되므로, 스모크 게이트가 앞에 붙은 조건부 실험 티켓 하나를 열 것을 권고한다.

- 스모크 구성(게이트, ADR 0001 스크리닝 전 단계):
  - 패키지 `tabpfn` 8.3.0, `ModelVersion.V3`, fold 0 하나, seed 42.
  - 문맥은 fold 0 제외 4개 fold 전체(~553k행), 예측은 fold 0(~138k행)만.
  - 피처는 두 판으로 잰다: 원시 13열 판과 champion 피처 세트(fold 안 TE 포함) 판.
  - `fit_mode="fit_with_cache"`, 테스트 청크 1,000행, OOM 시 `memory_saving_mode=True` 순서로 후퇴.
  - 벽시계 시간과 fold-0 AUC를 기록한다.
- 게이트 판정:
  - 성능: fold-0 AUC가 0.9650 이상(OOF 환산으로 0.966+ 궤도)이어야 정식 seed 42 스크리닝으로 진행한다.
    S6E5 사례처럼 GBDT 대비 -0.004급이면 #77의 약한 모델 배제와 같은 근거로 닫는다.
  - 시간: 5-fold 스크리닝이 한 Kaggle 세션(12시간) 안에 끝나는 속도가 아니면, `SUBSAMPLE_SAMPLES` 앙상블로 재측정하고 그래도 안 되면 닫는다.
- 게이트 통과 시 정식 구성: committed 5-fold + seed 42 스크리닝 + 3-seed 확정의 기존 계약 그대로, float64 OOF·테스트 예측을 풀에 등록하고 앙상블 기여를 측정한다.
- 라이선스 준수 사항: 공개 산출물에 "The TABPFN-3 Model is licensed by Prior Labs GmbH under the TABPFN-3 Non-Commercial License" 표기를 남기고, TabPFN 출력을 다른 모델의 학습 입력(증류)으로 쓰지 않는다.
  앙상블 가중 평균은 출력의 이용이지 경쟁 모델 학습이 아니므로 저촉되지 않는다고 판단하나, 스태킹 메타 모델의 입력으로 쓰는 경우도 "우리 대회 앙상블" 용도라 비상업 조항 안이라고 본다.
