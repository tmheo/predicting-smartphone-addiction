# Omid TabTransformer 공개 노트북의 실행 후보 적합성 조사

## 결론

[TabTransformer : Predicting Smartphone Addiction 공개 판본 1](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/versions/1)은 원 논문의 TabTransformer를 구현한 것이 아니다.
실제 구조는 59개 수치 스칼라 각각에 열별 ReLU 기저와 학습 주기 기저를 만들고, 모든 열을 self-attention으로 섞은 뒤 token 전체와 원래 스칼라를 펼쳐 큰 MLP에 넣는 수치 token Transformer다.
원 논문은 범주 값 lookup embedding만 Transformer에 통과시키고 연속값은 별도 경로로 최종 MLP에 합치므로 입력 표현과 attention 대상이 근본적으로 다르다 ([TabTransformer 원 논문 2절](https://ar5iv.labs.arxiv.org/html/2012.06678#S2)).

판정 권고는 **다양성 목적의 제한적 실행 후보**다.
champion 교체의 우선 후보로는 권하지 않는다.
공개 OOF AUC `0.967468921215`는 같은 seed 42 공통 fold의 현재 champion `0.969087400510`보다 `0.001618479295` 낮다.
반면 공개 OOF를 현재 후보 풀 16개와 읽기 전용으로 대조하면 최근접 Spearman 상관이 TabPFN-3의 `0.966985606000`으로 중복 문턱 `0.998`보다 충분히 낮고, 표준 순위 평균 AUC가 `0.968677978702`에서 `0.968709071056`으로 `+0.000031092354` 올랐다.
ADR 0001에서 이 기여값은 진입 게이트가 아니라 참고값이지만, 낮은 중복도와 양의 기여가 함께 있어 고친 구조의 저비용 진입 진단을 한 번 수행할 근거는 된다 ([실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)).

공개 실행을 그대로 후보 풀에 넣을 수는 없다.
외부 단일 시드 실행이고, 학습 행의 정확값 목표값 인코딩이 자기 라벨을 포함하며, 카나리아와 계열 무관 중요도가 없고, 정확한 패키지 잠금과 완전한 결정론 설정도 없다.
이 결함은 공개 결과의 채택 자격을 막지만, 저장소에서 내부 OOF 목표값 인코딩, 공통 피처 계획, 결정론 설정과 실행 기록 규약으로 고쳐 구현할 수 있다.

## 조사 대상과 고정 판본

조사일은 2026-08-17 JST다.
Kaggle API가 반환한 `currentVersionNumber`는 `1`이고, 이 판본을 Kaggle CLI `2.2.4`로 내려받았다.
공개 판본 주소는 [versions/1](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/versions/1)이고 현재 페이지는 [최신 공개 페이지](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction)다.

| 증거 | 고정값 |
| --- | --- |
| 노트북 SHA-256 | `eeb3e1cccbaab29c71ef946876f7042509f6ef537df4a9b04ced36e3c424e46c` |
| Kaggle 메타데이터 SHA-256 | `3b9ab90b326ac4cf9cc6ce45f6c76ef04980fb2c44eb7d9f1e1e604bcb32470c` |
| 실행 로그 SHA-256 | `21e24aa8ad869aaeb87d2d72f40c3e59be3b37fd3f666fd3dab58b1a222bd657` |
| OOF SHA-256 | `1594f8e7f72ee8c6bf5dacbdddc56fb29d8998c24c43ad9424b39abd65e80cb4` |
| 제출 예측 SHA-256 | `6228dfc18fe458c6f061f684daff8daef7e2a4aed39acf245be5ec0a190877a9` |
| Kaggle 실행 환경 | NVIDIA Tesla T4, Python `3.12.13` |
| Kaggle 컨테이너 | `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` |
| 학습 자료 SHA-256 | `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c` |
| 시험 자료 SHA-256 | `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e` |

Kaggle 실행 파일 목록에는 `oof.csv`와 `submission.csv`가 2026-08-16 17:05 UTC에 생성된 것으로 기록돼 있다 ([공개 출력 페이지](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/output)).
노트북 본문의 셀 출력은 비어 있지만 Kaggle 실행 로그와 출력 파일은 내려받을 수 있으므로, 본문에 표시된 수치가 아니라 원시 OOF를 다시 채점했다.

## 공개 OOF와 실행 결과 검증

공개 `oof.csv`는 `id`, `oof_pred`, `addicted_label` 세 열과 691,369개 행을 가진다.
ID는 중복 없이 공식 학습 자료의 원래 순서와 완전히 같고, 저장 라벨도 공식 학습 라벨과 모두 일치한다.
예측은 전부 유한하고 최솟값 `0.0025405883799062`, 최댓값 `1.0`이다.
커밋된 [`artifacts/folds.parquet`](../../artifacts/folds.parquet)의 ID를 결합했을 때 모든 행이 정확히 한 fold에 들어갔다.
이 fold는 노트북이 선언한 `StratifiedKFold(5, shuffle=True, random_state=42)`와 저장소 공통 분할의 생성 규칙이 같다 ([fold 생성 코드](../../scripts/make_folds.py)).

| fold | 행 수 | 재채점 AUC |
| ---: | ---: | ---: |
| 0 | 138,274 | `0.967035185241` |
| 1 | 138,274 | `0.967915793426` |
| 2 | 138,274 | `0.967747784244` |
| 3 | 138,274 | `0.968295231076` |
| 4 | 138,273 | `0.967447460187` |
| 전체 | 691,369 | `0.967468921215` |

로그의 소수 다섯 자리 fold 값과 전체 `0.96747`은 원시 OOF 재채점과 일치한다.
따라서 공개 결과는 Public 점수가 아니라 실제 공통 분할 OOF라는 점에서 검토 가치가 있다.
다만 매 fold의 최고 checkpoint를 그 fold의 AUC로 고른 뒤 같은 fold에서 점수를 보고하므로 epoch 선택 편향은 남는다.
저장소 모델도 검증 fold 조기 종료를 사용하므로 이 사실 하나가 탈락 사유는 아니지만, 공개 점수의 마지막 자릿수를 독립 시험 결과처럼 해석하면 안 된다.

## 정확한 입력 표현

공식 자료는 수치형 9개와 범주형 3개로 이루어진다 ([저장소 자료 설명](../../README.md)).
노트북은 행 단위 파생값 26개를 추가한다.
26개는 결측 개수 1개와 합계, 비율, 잔차, 주간 환산, 로그 및 제곱근 변환 25개다.
원래 수치 열 9개와 이 26개가 `num_base_cols` 35개를 이룬다.

노트북은 `cat_cols`라는 이름에 원래 입력 열 12개 전부를 넣는다.
각 열의 정확값 문자열별 학습 fold 빈도 1개와 평활 목표값 평균 1개를 만들어 24개 스칼라를 더한다.
따라서 신경망 입력은 `35 + 24 = 59`개 수치 스칼라다.
범주형 원래 값 자체, 범주 ID와 범주 lookup embedding은 입력에 없다.
`gender`, `stress_level`, `academic_work_impact`는 빈도와 목표값 평균 두 수치로만 표현된다.

각 outer fold에서 `QuantileTransformer(output_distribution="normal", n_quantiles=1000)`를 학습 부분에만 맞추고 검증과 시험 부분에 적용한다.
이 경계는 검증 자료 전체를 먼저 정규화하는 누출이 아니며 저장소에서 그대로 고칠 필요가 없는 부분이다.

## 정확한 신경망 구조

각 입력 스칼라에는 두 경로가 적용된다.
첫 경로는 열마다 16개의 `ReLU(xw+b)`를 만드는 학습 가능한 ReLU 기저다.
공개 클래스 이름은 `PiecewiseLinearEmbedding`이지만, quantile 또는 목표값 기반 bin 경계와 누적 조각선형 좌표를 만드는 공식 PLE가 아니다.
[수치 특성 embedding 논문](https://ar5iv.labs.arxiv.org/html/2203.05556#S3.SS2)은 PLE를 학습 부분에서 정한 bin 경계 사이의 누적 조각선형 좌표로 정의한다.
따라서 공개 첫 경로는 그 논문의 PLE보다 같은 논문이 구분한 단순 선형-ReLU embedding에 가깝다.

둘째 경로는 열마다 학습 주파수 8개와 학습 위상 8개를 두고 sine과 cosine 16개를 만든다.
이는 수치 특성 embedding 논문의 학습 주파수 periodic 표현과 가깝지만, 원 논문 식에는 없는 학습 위상 항을 추가한다 ([Periodic 정의](https://ar5iv.labs.arxiv.org/html/2203.05556#S3.SS3)).
두 16차원 경로를 이어 붙인 32차원을 공유 선형층으로 64차원에 투영하고, 59개 열별 학습 식별자를 더한다.

59개 token은 폭 64, 머리 4개, feed-forward 폭 256인 post-normalization Transformer block 3개를 통과한다.
마지막 token 59개를 모두 펼친 3,776차원 벡터와 정규화된 원래 스칼라 59개를 이어 붙여 3,835차원으로 만든다.
그 뒤 `3835 -> 256 -> 128 -> 64 -> 1` MLP가 최종 logit을 낸다.
각 은닉층에는 BatchNorm, GELU와 dropout이 있고 마지막 선형층 앞에는 같은 가중치를 공유하는 dropout 표본 8개의 평균이 적용된다.
학습 가능 매개변수는 이 59열 구성에서 `1,182,801`개로 계산된다.

학습은 batch 256, 최대 120 epoch, AdamW, 초기 학습률 `0.001`, 가중치 감쇠 `0.03`, 20 epoch 주기 cosine restart를 사용한다.
라벨 평활 `0.005`, mixup alpha `0.2`, gradient norm 제한 `1.0`, EMA decay `0.999`가 함께 적용된다.
조기 종료 대기는 18 epoch이고 fold별 최고 EMA checkpoint를 예측에 쓴다.

## 원 논문 TabTransformer와의 차이

[원 논문](https://ar5iv.labs.arxiv.org/html/2012.06678#S2)은 범주형 각 열의 값별 lookup embedding과 열 식별자 부분을 합쳐 범주 token을 만든다.
Transformer는 범주 token만 문맥화하며, 연속값은 Transformer 밖에서 마지막 MLP에 바로 합친다.
원 논문 기본 설정은 embedding 폭 32, Transformer 6층과 머리 8개이고 최종 MLP 폭은 입력 크기 `l`에 대해 `4l`, `2l`이다 ([원 논문 실험 설정](https://ar5iv.labs.arxiv.org/html/2012.06678#S3)).

| 축 | 원 논문 TabTransformer | 공개 노트북 |
| --- | --- | --- |
| 범주 값 표현 | 열별 값 lookup embedding과 결측 embedding | 정확값 빈도와 목표값 평균 두 스칼라 |
| attention 대상 | 범주형 열만 | 원시·파생·목표값·빈도 수치 59개 전부 |
| 연속값 경로 | attention 밖에서 최종 MLP에 합침 | ReLU·주기 token과 원시 skip 양쪽에 사용 |
| 열 식별 | 범주 embedding 안의 전용 식별자 | 모든 수치 token에 더하는 열별 벡터 |
| readout | 문맥화 범주 embedding과 연속값을 이어 붙인 MLP | 모든 token을 펼치고 원시 스칼라를 이어 붙인 MLP |
| 기본 크기 | 폭 32, 6층, 8머리 | 폭 64, 3층, 4머리 |
| 사전학습 | 선택적 MLM 또는 RTD | 없음 |

원 논문도 전체 라벨을 쓰는 감독학습에서는 사전학습의 이득을 거의 찾지 못했다고 보고하므로, 사전학습 부재 자체는 이번 대회에서 중요한 결함이 아니다 ([원 논문 3절](https://ar5iv.labs.arxiv.org/html/2012.06678#S3)).
중요한 차이는 범주형 문맥 embedding을 전혀 구현하지 않았다는 점이다.
따라서 후속 기록에서 이 구조를 `TabTransformer`라고 부르면 원 논문 재현과 혼동된다.
정확한 작업 이름은 `quantile-ReLU-periodic token Transformer` 정도가 적합하다.

원 저자 측 [Amazon Science 설명](https://www.amazon.science/blog/bringing-the-power-of-deep-learning-to-data-in-tables)도 범주형 변수를 Transformer로 처리하고 연속형 변수를 병렬 경로로 처리한다고 설명한다.
arXiv 기록과 Amazon Science 페이지에서 공개 원저자 소스 저장소는 확인되지 않았고, Amazon은 SageMaker JumpStart 접근 경로를 안내한다.
[lucidrains/tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch)는 널리 쓰이는 MIT 제3자 구현이지만 원저자 공식 구현이 아니며 이 노트북도 이를 가져오지 않는다.

## 전처리와 검증 경계

행 단위 파생값은 라벨을 읽지 않고 학습과 시험 자료에 각각 같은 함수로 적용되므로 자료 전체 통계 누출이 없다.
빈도와 목표값 평균은 outer 학습 부분에서만 계산해 검증과 시험 부분에 적용하므로 검증 라벨이 변환에 들어가지는 않는다.
따라서 공개 OOF를 검증 라벨 직접 누출로 무효화할 근거는 없다.

그러나 학습 행의 목표값 평균은 내부 OOF가 아니다.
같은 outer 학습 부분 전체의 `groupby(...)[target].mean()`을 만든 뒤 그 통계를 같은 학습 행에 다시 매핑하므로 각 학습 행의 입력이 자기 라벨을 포함한다.
빈도가 1인 값은 smoothing 10에서 자기 라벨이 목표값 평균의 약 `1/11`을 직접 담당한다.
검증 행은 자기 라벨이 없는 통계를 받으므로 학습과 검증 표현의 생성 규칙도 달라진다.

이 문제는 공개 검증 결과의 채택 자격과 저장소에서 고쳐 쓸 수 있는 구현 문제를 구분해야 한다.
공개 OOF는 outer 검증 라벨을 보지 않았으므로 관찰값으로는 유효하지만, 저장소의 실행 규약은 타깃을 쓰는 변환의 학습 행에도 내부 OOF를 요구한다 ([실험 프로그램 지도](https://github.com/tmheo/predicting-smartphone-addiction/issues/44)).
저장소 구현에서는 원래 12열의 목표값 평균을 outer 학습 부분 안의 내부 fold로 만들어야 하고, 검증·시험 통계는 outer 학습 부분 전체로 만들어야 한다.
플라시보 목표값 카나리아가 원래 플라시보보다 낮은 중요도를 보이지 않으면 실행 전체를 무효로 처리해야 한다.

빈도 표현도 outer 학습 부분에서만 계산돼 직접 누출은 없지만, 이 저장소에서는 단일 열 빈도 표현이 이미 성능과 중요도 게이트에서 기각됐다 ([단일 열 빈도와 추가 정확값 표현 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/49)).
따라서 공개 OOF의 낮은 상관을 신경망 구조 하나에 귀속할 수 없다.
빈도, 자기 포함 목표값 평균, 26개 파생값, quantile 변환, mixup, EMA, 펼친 token readout이 한꺼번에 바뀌었기 때문이다.

`Tuned`라는 클래스 이름 외에 하이퍼파라미터를 고른 탐색 기록, 대조군과 별도 선택 fold는 공개돼 있지 않다.
판본 1의 한 실행만 공개됐으므로 같은 공통 fold를 반복 관찰해 설정이나 노트북을 고른 선택 편향의 크기는 알 수 없다.
이 불확실성은 저장소 재구현으로 과거 공개 증거에서 제거할 수 없고, 자체 공통 진입 진단과 구조 제거 대조로 새 근거를 만들어야 한다.

## 재현 가능성과 사용 조건

노트북은 NumPy와 PyTorch CPU·CUDA seed를 42로 고정하고 CuDNN 결정론을 요청한다.
그러나 `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, fold별 독립 seed와 패키지 잠금은 없다.
seed를 노트북 시작에서 한 번만 설정하므로 fold 하나를 따로 재시작하면 전체 순서 실행에서 그 fold가 받았던 난수 상태를 재현하지 못한다.
PyTorch, NumPy, pandas와 scikit-learn의 정확한 판본 목록도 실행 출력에 없다.
컨테이너 내용 해시는 고정됐지만 장기 재현 기록으로는 부족하다.

저장소 구현에서는 fold별 seed를 독립적으로 파생하고, CuBLAS 결정론 작업 공간, 입력·fold·환경 해시, 복구 경계와 실행 기록 묶음을 기존 공통 경로로 보존하면 된다.
외부 출력에는 피처 중요도와 플라시보 측정이 없으므로 계열 무관 permutation importance도 추가해야 한다.

Kaggle 공개 노트북 소스는 저장소의 [Kaggle 공개 노트북 사용 조건](../agents/kaggle-public-notebook-licensing.md)에 따라 Apache License 2.0으로 사용할 수 있다.
소스를 복사하거나 수정해 배포하면 Apache License 2.0 원문, 원래 고지와 변경 표시를 보존해야 한다 ([Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)).

노트북은 외부 자료, 다른 Kaggle kernel, 사전학습 가중치나 외부 모형을 참조하지 않는다.
메타데이터의 유일한 자료 원천은 `playground-series-s6e8` 대회다.
직접 import하는 외부 패키지는 NumPy, pandas, scikit-learn과 PyTorch이며 코드를 노트북 안에 포함하지 않는다.
각 패키지의 상류 사용 조건은 [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [pandas](https://github.com/pandas-dev/pandas/blob/main/LICENSE), [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING), [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE)에서 따로 확인할 수 있다.
따라서 공개 노트북 코드의 Apache 2.0과 대회 자료 및 설치 패키지의 사용 조건을 섞어 해석하면 안 된다.

## 현재 후보 풀과의 차별성

현재 champion과 후보 풀 수치는 [`artifacts/champion.yaml`](../../artifacts/champion.yaml)과 [`artifacts/pool.yaml`](../../artifacts/pool.yaml)을 2026-08-17 main 판본에서 읽었다.
공개 OOF는 외부 예측이므로 후보 풀이나 제출에 넣지 않고, 이 절의 측정은 [외부 OOF 읽기 전용 진입 진단 결정](https://github.com/tmheo/predicting-smartphone-addiction/issues/76)과 같은 방식의 우선순위 근거로만 사용한다.

| 비교 대상 | 3시드 OOF AUC | 공개 후보와 Spearman |
| --- | ---: | ---: |
| 현재 champion `exp081_lookup_fold_initialization_avg3` | `0.969195761811` | `0.964172586225` |
| 기본 Lookup `exp059_lookup_transformer` | `0.968922178533` | `0.955127777627` |
| TabM `exp065_tabm` | `0.968326118177` | `0.966353085938` |
| TabPFN-3 `exp067_tabpfn3` | `0.967243226668` | `0.966985606000` |

현재 champion의 같은 seed 42 OOF AUC는 `0.969087400510`이고 공개 후보는 `0.967468921215`이므로 champion 교체 스크리닝의 개선 `>= 0` 조건과 거리가 크다.
공개 후보는 TabM보다 `0.000857196962` 낮고 TabPFN-3보다 `0.000225694547` 높다.
단독 성능만 보면 후보 풀 진입 하한 `champion - 0.01 = 0.959195761811`은 넉넉히 넘는다.

현재 16개 후보의 OOF를 ID 순서로 맞춘 뒤 각 예측을 백분위 순위로 바꿔 평균하고 공개 후보를 넣기 전후를 재채점했다.
후보를 넣지 않은 AUC는 `0.968677978702`, 넣은 AUC는 `0.968709071056`, 변화는 `+0.000031092354`다.
최근접 상관 `0.966985606000`도 중복 문턱보다 낮다.
이 결과는 공개 후보가 현재 풀과 다른 오류를 낸다는 직접 근거지만, 그 원인을 Transformer 구조로 좁히지는 못한다.

Lookup-Transformer는 정확값 lookup embedding과 학습 주기 수치 표현을 더한 token을 attention으로 섞고 CLS token으로 읽는다 ([현재 구현](../../src/pipeline/lookup_transformer.py)).
공개 후보는 정확값 lookup이 없고 ReLU·주기 표현을 이어 붙이며, 모든 token을 펼친 큰 MLP가 읽는다는 차이가 있다.
attention과 학습 주기 표현 자체는 이미 Lookup과 겹친다.

TabM은 공식 조각선형 수치 embedding과 깊은 MLP 계열을 사용하므로 수치 표현과 MLP라는 축이 겹친다 ([현재 구현](../../src/pipeline/tabm.py)).
공개 후보의 ReLU 기저는 TabM의 PWL과 같지 않지만, 수치 스칼라를 열별 고차원 표현으로 바꾼 뒤 비선형 backbone에 넣는 큰 틀은 새롭지 않다.

TabPFN-3은 사전학습한 표 자료 기반 모형이라 학습 원리와 계산 경로가 다르다.
공개 후보와 가장 가까운 상관이 TabPFN-3인데도 `0.967`에 불과하다는 사실은 공개 전처리와 학습 recipe가 풀의 새 오류 축을 만든다는 뜻이지, 두 구조가 비슷하다는 뜻이 아니다.

최근 [contextualized spline Transformer 진입 진단](contextualized-spline-entry-diagnostic.md)은 열별 수치 표현, attention과 펼친 최종 MLP의 유사한 조합을 이미 시험했다.
그 후보는 fold 0 AUC `0.9667574340`, 최근접 상관 `0.9781382739`와 순위 평균 기여 `+0.0001244004`를 보였지만 복제 대조 상한을 넘지 못해 중단됐다.
공개 후보는 그 결과보다 직접 공통 OOF가 강하지만, ReLU·주기 표현과 readout의 기여를 떼어 보지 않았다는 같은 증거 한계를 가진다.

## 69만 행 적합성과 예상 비용

공개 실행은 691,369행 전체에서 T4 16GB로 5-fold를 완주했으므로 자료 규모와 단일 16GB GPU 적합성은 추정이 아니라 관찰된 사실이다.
59개 token의 attention 행렬은 batch 256, 머리 4개에서 층마다 `256 * 4 * 59 * 59 = 3,564,544`개 원소라 이 자료의 열 수에서는 제곱 비용이 제한적이다.
매개변수도 약 118만 개라 모형 가중치보다 batch별 attention과 중간 activation이 메모리를 지배한다.
공개 로그에는 최고 CUDA 메모리가 없으므로 16GB에서 완주했다는 상한 증거 이상으로 정밀한 메모리 수치를 주장하지 않는다.

첫 fold는 kernel 시작 뒤 약 `1,721`초에 끝났고, fold 준비 시작부터는 약 `1,678`초, `0.47` GPU 시간이었다.
5-fold 전체 OOF 출력은 약 `8,294`초, 컨테이너 결과 변환까지는 약 `8,307`초로 `2.31` GPU 시간이었다.
각 epoch은 대체로 약 49.5초이고 fold는 29에서 39 epoch 사이에 조기 종료됐다.

고친 단일 구조의 fold 0 진입 진단은 T4급에서 후보 실행 약 `0.5` GPU 시간으로 잡을 수 있다.
내부 OOF 목표값 인코딩, 중요도와 실행 기록 비용을 포함한 후보 단독 상한은 `0.75` GPU 시간이 합리적이다.
attention 제거 대조까지 같은 fold에서 실행하면 후보 두 벌 합계 상한을 `1.5` GPU 시간으로 둘 수 있다.
동등 단계 champion 재실행 시간은 별도로 더해야 하며 같은 공급자와 같은 GPU 등급에서 짝지어야 한다.
공개 속도가 그대로 유지되면 후보의 seed 42 5-fold는 약 `2.3` GPU 시간, 3시드 확정은 후보만 약 `6.9` GPU 시간이지만 자체 첫 fold 측정으로 다시 예측해야 한다.

## 권장 진입 진단 경계

공개 recipe를 그대로 복사하는 실행은 권하지 않는다.
후속 HITL 결정이 실행을 택한다면 목적은 champion 교체가 아니라 공개 OOF의 새 오류 축이 고친 구현에서도 남는지 확인하는 것이다.

주 구조 `M0`는 공개 59개 스칼라 계약을 보존하되 목표값 인코딩 학습 행을 내부 OOF로 만들고, outer 학습 부분 전용 빈도·quantile 변환, 독립 fold seed, 결정론과 카나리아를 적용해야 한다.
공개 `PiecewiseLinearEmbedding`이라는 이름은 쓰지 말고 ReLU 기저라는 실제 동작을 명시해야 한다.
주기 표현, 3층 attention, token 펼침과 원시 스칼라 skip, 최종 MLP, mixup, EMA는 공개 구조의 정체성이므로 첫 진단에서는 유지해야 한다.

대조 `A0`는 같은 입력, 같은 매개변수 예산과 같은 학습 recipe를 사용하되 self-attention 세 층만 token별 잔차 MLP 또는 매개변수 규모를 맞춘 비-attention 결합으로 바꿔야 한다.
이 대조가 있어야 양의 후보 풀 기여를 attention 구조와 전처리 효과로 나눌 수 있다.
원 논문도 Transformer를 제거하고 나머지를 고정한 MLP와 비교해 attention 효과를 주장했다 ([원 논문 3.1절](https://ar5iv.labs.arxiv.org/html/2012.06678#S3.SS1)).

공통 fold 0에서 `M0`가 동등 단계 champion보다 `0.01` 넘게 낮거나 목표값 카나리아가 실패하면 즉시 중단해야 한다.
`M0`가 `A0`보다 높지 않으면 이 후보를 새 Transformer 구조로 승격할 근거가 없으므로 5-fold를 열지 않아야 한다.
두 조건을 통과하면 seed 42 5-fold에서 최근접 Spearman, 표준 순위 평균 참고 기여와 5개 fold 성능을 측정하고, 그 뒤에만 ADR 0001의 3시드 후보 풀 진입 판정을 적용할 수 있다.

이 보고서는 실행 후보 근거와 경계까지만 제안한다.
구현 티켓이나 유료 GPU 실행은 만들지 않는다.

## 최종 판정표

| 판정 축 | 근거 | 결과 |
| --- | --- | --- |
| 공개 판본·출처 고정 | version 1, 소스·로그·OOF·자료 해시 확보 | 통과 |
| 69만 행 실행 가능성 | T4 16GB에서 5-fold 2.31시간 완주 | 통과 |
| 공통 OOF 무결성 | ID·라벨·fold 일치, AUC `0.967468921215` 재현 | 통과 |
| champion 교체 근거 | 같은 seed champion보다 `-0.00161848` | 약함 |
| 후보 풀 차별성 | 최근접 Spearman `0.96699`, 참고 기여 `+0.00003109` | 있음 |
| 원 논문 TabTransformer 재현 | 범주 lookup과 연속 병렬 경로 부재 | 실패 |
| 목표값 변환 규율 | 학습 행 자기 라벨 포함, 카나리아 없음 | 공개 실행 채택 불가 |
| 재현성 | 컨테이너 해시는 있으나 패키지 잠금·fold 독립 seed·완전 결정론 없음 | 보완 필요 |
| 사용 조건 | 노트북 Apache 2.0, 외부 가중치·kernel 없음 | 통과 |
| 최종 권고 | champion 목표가 아닌 1.5 GPU 시간 이하의 구조 대조형 다양성 진입 진단 | 제한적 실행 후보 |

## 한계

공개 OOF와 현재 후보 풀의 직접 상관·기여 측정은 외부 예측을 채택한 것이 아니라 조사 우선순위에만 사용했다.
공개 실행은 단일 seed이고 목표값 인코딩 규율이 다르므로, 공개 OOF의 양의 기여가 고친 자체 구현에서도 유지된다고 단정할 수 없다.
attention 제거 대조가 없어 낮은 상관의 원인을 구조, 전처리와 학습 recipe 사이에서 분해할 수 없다.
정확한 패키지 판본과 최고 CUDA 메모리가 공개되지 않아 내용 해시 컨테이너 밖의 세부 재현성과 메모리 여유는 확인하지 못했다.
