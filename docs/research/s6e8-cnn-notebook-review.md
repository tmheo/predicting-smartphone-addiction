# S6E8 표 자료 CNN 공개 구조 검토

이 문서는 [리서치: CNN 공개 구조의 실행 후보 적합성 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/168)의 근거다.
조사 시점은 2026-08-17 JST이며, Kaggle에서 고정한 최신 공개 판본의 소스, 실행 기록, 출력 파일, 1차 자료와 이 저장소의 현재 판정 근거를 사용했다.

## 결론

[CNN for Predicting Smartphone Addiction 고정 판본](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction?scriptVersionId=342747549)은 champion 교체보다 후보 풀 다양성 확인을 목표로 한 저비용 실행 후보로 분류한다.
공개 구현은 실제 691,369행을 단일 T4에서 5-fold 완주했고, 공개 분할은 저장소의 고정 fold와 691,369행 모두 일치한다.
저장 OOF를 다시 채점한 AUC는 `0.9677056335`로 현재 champion의 같은 seed 42 AUC `0.9690874005`보다 `0.0013817670` 낮고 5개 fold에서 모두 졌다.
따라서 단일 모델 성능으로 Lookup-Transformer를 바꿀 가능성은 낮다.

반면 공개 OOF와 현재 16개 후보의 최근접 스피어만 상관은 TabM과의 `0.9559346433`이고, 현재 Lookup 초기화 평균본과는 `0.9522251740`이다.
현재 16개 후보의 균등 순위 평균 AUC `0.9686779787`에 공개 CNN을 더하면 `0.9687461646`으로 `+0.0000681859` 올랐다.
공개 예측을 후보 풀에 넣을 수는 없지만, 공통 fold에서 관측한 낮은 중복과 양의 참고 기여는 문제를 고친 자체 구조의 fold 0 진입 진단을 정당화한다.

추천 범위는 공개 구현 전체를 복제하는 일이 아니다.
이미 자체 실험에서 기각된 정확값 목표 인코딩과 빈도 인코딩을 빼고, 현재 champion과 같은 피처 계획에서 수치 임베딩 뒤 열 순서를 따라 합성곱하는 경로와 매개변수 규모를 맞춘 합성곱 제거 대조를 함께 재야 한다.
이 대조에서 합성곱 경로가 champion 목표나 통제된 다양성 목표를 하나도 통과하지 못하거나 합성곱 제거 대조보다 나은 근거를 만들지 못하면 5-fold로 넓히지 않는다.

## 고정한 공개 판본과 산출물

Kaggle 공개 페이지의 `scriptVersionId`는 `342747549`이고 게시 및 최종 수정 시각은 모두 `2026-08-16T10:37:07.7433333Z`다.
Kaggle 목록 API의 최신 실행 시각도 `2026-08-16 10:37:07.743 UTC`이며 조사 당시 득표 수는 1개였다.
Kaggle CLI가 받은 메타데이터의 노트북 식별자는 `130924695`이고 실행 장치는 `NvidiaTeslaT4`다.
컨테이너는 `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`로 기록됐다.

내려받은 소스의 SHA-256은 `2310c4fa1b98230989f8e3bcf3f9661985a2c30df90597786e739cd34321f4dc`다.
Kaggle CLI 메타데이터의 SHA-256은 `64b436cc582602417df8753b716b5c9e380af34b7129018b0de76d3617c4fd33`다.
공개 실행 기록의 SHA-256은 `7712720b42e9f360da482877bb32025dcbd0005965fbac9c3bde4ca1b18ae3af`다.
공개 `oof.csv`의 SHA-256은 `3da3917f23b3c636cf3af11792ab483f89a09d6b38416897b0626d93b10c3c91`이고 `submission.csv`는 `66b40c14c8c133e9228889e713183e007a8294cb300a807dab1c7fd0c2fab9d6`다.

소스 노트북의 셀에는 실행 횟수와 출력이 없지만 Kaggle의 별도 실행 기록과 출력 파일은 완주를 입증한다.
소스 안쪽 Kaggle 메타데이터는 GPU와 인터넷을 비활성으로 적었으나 내려받은 외부 메타데이터는 GPU와 인터넷을 활성으로 적는다.
실제 실행 기록은 CUDA 자동 혼합 정밀도 학습을 보여 주므로 실행 사실은 외부 메타데이터와 기록을 기준으로 판단했다.

노트북은 대회 학습·시험 자료만 읽고 외부 자료, 다른 노트북, 사전 학습 모형이나 원격 코드를 받지 않는다.
Kaggle 메타데이터의 `dataset_sources`, `kernel_sources`와 `model_sources`도 모두 비어 있다.

## 실제 입력과 합성곱 축

원래 설명변수는 수치 9개와 범주 3개를 합친 12개다.
행별 파생값은 결측 수 1개와 시간 합계, 비율, 로그 및 제곱근 변환 25개다.
수치 원본 9개와 행별 파생값 26개가 첫 35개 수치 입력을 이룬다.
원래 12개 열 각각에 정확값 빈도와 평활 목표 평균을 붙여 24개 입력을 더하므로 모형이 받는 최종 폭은 59개다.

각 outer fold에서 `QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)`를 학습 부분에만 맞춘 뒤 검증과 시험 자료를 변환한다.
[scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)는 이 변환이 각 열의 누적분포를 독립적으로 추정하는 비선형 변환임을 명시한다.
이 fold 경계는 올바르지만 `subsample`과 scikit-learn 판본을 코드에서 고정하지 않아 다른 실행 환경에서 분위수 기준점이 달라질 수 있다.

모형의 입력 처리 흐름은 다음과 같다.

1. 59개 스칼라 각각을 12차원 학습 가능한 ReLU 경첩 표현과 12차원 주기 표현으로 바꾼다.
2. 두 표현을 붙인 24차원을 공유 선형층과 Mish로 32차원으로 줄인다.
3. 텐서 축을 `[행, 32채널, 59열]`로 바꾼다.
4. 커널 크기 3인 `Conv1d` 두 개가 `32 -> 64 -> 128`채널로 바꾸며 두 층의 열 방향 수용 범위는 5개 위치다.
5. 128채널 squeeze-and-excitation 블록 뒤 열 방향 전체 평균과 최댓값을 구해 256차원으로 줄인다.
6. 이 256차원과 변환된 원시 입력 59개를 붙여 `315 -> 128 -> 64 -> 1` 완전 연결 경로로 예측한다.

학습 가능 매개변수는 코드에서 계산하면 91,565개다.
59열 가운데 이웃이라는 관계는 CSV 열 순서, 파생 열 추가 순서와 인코딩 추가 순서로만 정해지며 자료의 의미나 그래프로 정의되지 않는다.
[PyTorch `Conv1d` 문서](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv1d.html)는 마지막 길이 축에 교차상관을 적용한다고 명시하므로, 이 모형의 합성곱은 의미가 고정되지 않은 인접 열 다섯 개에 공유 가중치를 적용한다.
열 순서가 바뀌면 다른 함수가 되며, 공개 노트북에는 열 순서 변경이나 합성곱 제거 대조가 없다.
다만 최종 완전 연결 경로가 59개 변환 입력을 직접 다시 받으므로 합성곱 경로가 약해도 일반 MLP 경로가 예측을 지탱할 수 있다.

## 이름과 원래 방법의 차이

노트북은 출처나 논문을 인용하지 않는다.
구성요소 이름으로 보아 수치 임베딩, squeeze-and-excitation과 다중 표본 탈락 규제를 조합했지만, 공개 코드는 원래 방법을 그대로 구현하지 않는다.

[수치 임베딩 논문](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)과 [공식 구현](https://github.com/yandex-research/rtdl-num-embeddings)은 구간 경계에 따른 조각선형 인코딩과 주기 활성화 뒤 선형층 및 ReLU를 별도 방법으로 정의한다.
노트북의 `PiecewiseLinearEmbedding`은 양자 구간 경계나 구간별 선형 보간을 만들지 않고 열마다 `ReLU(w x + b)` 12개만 학습한다.
따라서 `num_bins=12`는 실제 구간 수가 아니며 이 부분은 공식 조각선형 인코딩보다 학습 가능한 경첩 기저에 가깝다.
노트북의 주기 표현도 원 방법과 관련은 있지만, 학습 가능한 위상과 경첩 표현을 붙인 뒤 Mish를 쓰므로 공식 PLR 설정과 같지 않다.

[squeeze-and-excitation 원 논문](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper)은 전역 정보를 이용해 합성곱 채널을 다시 가중하는 블록을 제안한다.
노트북의 1차원 변형은 이 구조를 대체로 따르되 중간 활성화로 Mish를 쓴다.

[다중 표본 탈락 규제 원 논문](https://arxiv.org/abs/1905.09788)은 여러 탈락 표본의 손실을 각각 계산해 평균하도록 정의한다.
노트북은 다섯 탈락 표본의 logit을 먼저 평균한 뒤 손실을 한 번 계산하므로 같은 목적 함수가 아니다.
평가 모드에서는 다섯 `Dropout`이 모두 비활성화되고 같은 공유 선형층을 다섯 번 통과한 동일한 값이 평균되므로 추론 시 앙상블 효과도 없다.

## 공개 검증 근거

노트북은 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 쓴다.
공개 OOF 목표값으로 이 분할을 다시 만들고 [`artifacts/folds.parquet`](../../artifacts/folds.parquet)과 대조한 결과 691,369행 모두 fold 번호가 일치했다.
따라서 공개 OOF와 현재 후보 풀의 행별 순위 상관 및 참고 기여를 같은 검증 행에서 직접 계산할 수 있다.
모든 순위 상관과 균등 순위 평균은 `id`로 행을 맞추고 동률에 평균 순위를 부여해 계산했다.

공개 OOF의 fold별 AUC와 현재 champion seed 42의 같은 fold AUC는 다음과 같다.

| fold | 공개 CNN | 현재 champion seed 42 | 차이 |
| ---: | ---: | ---: | ---: |
| 0 | 0.9670378195 | 0.9684993008 | -0.0014614813 |
| 1 | 0.9678791183 | 0.9691818147 | -0.0013026965 |
| 2 | 0.9678513144 | 0.9692466411 | -0.0013953267 |
| 3 | 0.9683972219 | 0.9696230157 | -0.0012257938 |
| 4 | 0.9674815504 | 0.9689057429 | -0.0014241924 |

공개 CNN은 같은 seed의 모든 fold에서 현재 champion보다 낮다.
전체 AUC 차이도 `-0.0013817670`이므로 champion 교체 근거는 약하다.
현재 3시드 champion 평균본 `0.9691957618`과의 단순 차이는 `-0.0014901283`이다.

현재 후보 풀과의 스피어만 상관 상위 값은 TabM `0.9559346433`, 저용량 LightGBM `0.9550554721`, TabPFN-3 `0.9545042979`, XGBoost `0.9540114935` 순이다.
Lookup 초기화 평균본과는 `0.9522251740`, 원래 Lookup과는 `0.9520866764`다.
현재 중복 문턱 `0.998`과 충분히 떨어져 있으므로 공개 예측은 기존 후보와 다른 순위 오차를 보인다.

현재 16개 후보의 균등 순위 평균 AUC는 `0.9686779787`이다.
공개 CNN을 17번째로 더한 균등 순위 평균은 `0.9687461646`이고 변화량은 `+0.0000681859`다.
이 값은 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)에 따라 진입의 단독 문턱이 아니라 참고값이며, 외부 OOF 자체는 후보 풀에 편입하지 않는다.

## 누출, 선택 편향과 고칠 수 있는 경계

행별 파생값은 목표값을 읽지 않으며 학습·시험 자료 각각에서 독립적으로 만들어진다.
분위수 변환, 빈도 및 목표 평균은 outer 학습 부분에만 맞추므로 검증 목표값이 전처리에 직접 들어가지는 않는다.
따라서 공개 OOF가 검증 목표값의 직접 누출로 무효라고 볼 근거는 없다.

그러나 outer 학습 행의 목표 평균 입력은 각 행 자신의 목표값을 포함한다.
정확값이 드문 수치 열에서는 학습 행의 입력에 자신의 목표값이 섞이고, 검증과 시험 행은 학습 부분의 통계만 받으므로 학습과 추론의 입력 분포가 달라진다.
이는 outer 검증 목표값의 직접 누출은 아니지만 공개 구조를 자체 구현으로 옮길 때 그대로 보존해서는 안 되는 학습 부분 내부 누출과 분포 불일치다.
교정하려면 학습 행의 목표 평균을 inner cross-fitting이나 leave-one-out으로 만들고 검증·시험은 outer 학습 전체 통계를 써야 한다.

더 중요한 비교 경계는 이 저장소가 정확값 목표 인코딩과 빈도 입력을 이미 별도 실험에서 판정했다는 점이다.
[S6E8 남은 실험 공간 전수 재점검](remaining-experiment-space-audit.md)과 [단변량 spline Transformer 검토](s6e8-167-spline-notebooks.md)는 정확값 목표 인코딩의 카나리아 실패와 빈도 인코딩 기각 때문에 새 신경망의 첫 진입 진단에서 두 입력을 제외하도록 정했다.
공개 CNN의 낮은 상관이 합성곱에서 나온 것인지 24개 정확값 인코딩에서 나온 것인지 제거 실험이 없어 구분할 수 없다.
따라서 자체 진입 진단은 현재 champion 피처 계획을 공유해 학습기 차이만 재야 한다.

각 fold는 같은 검증 fold AUC로 최적 epoch를 고르고 그 fold 점수를 OOF에 넣는다.
이는 조기 종료 선택에 따른 약한 낙관 편향을 남기지만 현재 저장소의 신경망 기준 실행도 같은 조기 종료 규약을 쓰므로 이 구조만의 제외 사유는 아니다.
공개 노트북은 한 seed만 실행하고, 구성요소 제거 실험, 열 순서 대조, 독립 반복과 현재 후보 대비 같은-fold 비교를 스스로 제시하지 않는다.
노트북 개발 과정에서 같은 fold를 반복 관찰한 정도도 알 수 없으므로 공개 OOF의 선택 편향은 정량화할 수 없다.

예측 함수는 CUDA 자동 혼합 정밀도 구간 안에서 sigmoid를 계산한 값을 그대로 저장한다.
공개 OOF 691,369개에는 서로 다른 예측값이 4,708개뿐이어서 많은 순위 동률이 생긴다.
자체 구현은 logit 또는 확률을 float32로 바꾼 뒤 저장하고 행 수, 순서, 유한성과 정밀도를 검사해야 한다.

## 재현 가능성과 사용 조건

소스는 NumPy와 PyTorch seed를 42로 고정하고 cuDNN 결정론 설정을 켠다.
Python `random`은 고정하지 않지만 현재 코드에서 사용하지 않는다.
fold별 모형 초기화 seed를 독립적으로 다시 고정하지 않아 fold를 별도 프로세스로 복구하면 공개 순차 실행과 같은 초기화가 되지 않는다.
저장소 구현은 실행 seed와 fold에서 초기화 seed를 명시적으로 파생해야 한다.

[PyTorch 결정론 문서](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)는 CUDA `Conv1d`를 결정론적으로 강제할 수 있다고 명시하며 이 설정만으로 모든 재현성이 보장되지는 않는다고 경고한다.
공개 코드는 `torch.use_deterministic_algorithms(True)`와 CUDA 작업 공간 설정을 쓰지 않고 라이브러리 판본도 출력하지 않는다.
컨테이너 digest, Python 3.12.13과 Kaggle 이미지 판본 28755는 남아 있지만 의존성 판본을 별도 잠금 파일로 고정하지 않았으므로 장기 재현 근거는 불완전하다.

Kaggle 공개 노트북 소스는 [Meta Kaggle Code](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)와 [저장소 사용 조건 절차](../agents/kaggle-public-notebook-licensing.md)에 따라 Apache License 2.0으로 참고하거나 수정해 배포할 수 있다.
코드를 재사용하면 원문 사용 조건, 저작권과 고지를 보존하고 수정 사실을 표시해야 한다.
공개 노트북에는 별도 `NOTICE`나 외부 소스 고지가 없다.

직접 의존성은 PyTorch, NumPy, pandas와 scikit-learn이다.
공식 저장소는 각각 [PyTorch 사용 조건](https://github.com/pytorch/pytorch/blob/main/pyproject.toml), [NumPy 사용 조건](https://github.com/numpy/numpy/blob/main/pyproject.toml), [pandas 사용 조건](https://github.com/pandas-dev/pandas/blob/main/LICENSE), [scikit-learn 사용 조건](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)을 공개하며 주 코드 사용 조건은 허용적이다.
설치 묶음에는 별도 사용 조건을 가진 하위 구성요소가 포함될 수 있으므로 저장소 잠금 파일이 고정하는 실제 wheel의 사용 조건 파일을 실행 명세에 보존해야 한다.
대회 자료와 출력은 노트북 코드의 Apache License 2.0이 대신하지 않으며 대회 규칙과 저장소의 비공개 입력 관리 규약을 그대로 따른다.

## 69만 행 적합성과 계산량

공개 실행은 691,369개 학습 행과 296,302개 시험 행을 batch 128로 처리했다.
fold별 종료 epoch는 45, 42, 29, 43, 27이고 최고 epoch는 대체로 첫 30 epoch 안에 있었다.
전체 실행 기록은 시작부터 결과 HTML 생성까지 7,786.98초, 약 2.163시간이다.
fold 0은 전처리부터 검증 및 시험 예측까지 약 1,876.5초, 31.3분이었다.

59개 float32 입력 전체는 학습 부분 약 130.5MB, 시험 부분 약 69.9MB이고 모형 가중치는 약 0.37MB다.
batch 128의 가장 큰 주 합성곱 활성화도 대략 `[128, 128, 59]`이므로 16GB 단일 GPU 메모리는 충분하다.
공개 기록은 최고 GPU 메모리를 출력하지 않았으므로 실제 진입 검사에서 할당 및 예약 최고값을 별도로 측정해야 한다.

최소 진입 진단은 M0와 합성곱 제거 대조 A0를 각각 fold 0에서 최대 30 epoch로 실행하는 구성이다.
공개 T4 처리량을 단순 적용하면 두 실행의 학습·예측 합계는 약 0.75 GPU 시간이고, 환경 검사와 순열 중요도를 포함한 비용 예약 상한은 1.5 GPU 시간으로 잡을 수 있다.

2026-08-17 JST에 공식 `vastai` 1.5.4의 읽기 전용 매물 검색으로 검증 호스트, 신뢰도 0.98 이상, 직접 포트, 단일 GPU와 5GiB 저장 공간을 요구했을 때 RTX A4000 16GB의 최저 `dph_total`은 시간당 약 `$0.075`, RTX 3090 24GB는 약 `$0.099`, T4 15GB는 약 `$0.148`이었다.
[Vast.ai 공식 가격 문서](https://docs.vast.ai/guides/instances/pricing)는 가격이 실시간 공급과 수요에 따라 변하고 계산·저장·전송 비용을 합쳐야 한다고 명시하며, [공식 매물 검색 문서](https://docs.vast.ai/api-reference/search/search-offers)는 `dph_total` 정렬 경로를 제공한다.
따라서 A4000의 1.5 GPU 시간 계산·5GiB 저장 공간 상한은 당시 매물 기준 약 `$0.11`이고 운영 여유를 포함한 진입 진단 비용 예약은 `$0.15`가 타당하다.
실제 유료 실행은 [Vast.ai 원격 자원 제어 절차](../agents/vast-resource-control.md)의 공급자, 비용 예약, 독립 종료 예약과 정리 관문을 별도로 통과해야 한다.

## 현재 모델과 구별되는 정도

현재 [Lookup-Transformer](../../src/pipeline/lookup_transformer.py)는 열별 정확값 lookup과 주기 수치 표현을 합친 뒤 모든 열을 self-attention으로 섞는다.
공개 CNN은 주기 표현을 공유하므로 수치 임베딩 자체는 새 축이 아니지만, 고정 열 순서의 국소 공유 합성곱, 채널 재가중과 전역 평균·최댓값 요약은 Lookup과 다른 상호작용 경로다.
다만 원시 입력 우회 경로와 24개 정확값 통계가 함께 있어 공개 다양성을 합성곱의 효과라고 단정할 수 없다.

현재 [TabM 설정](../../configs/exp065_tabm.yaml)은 공식 조각선형 수치 임베딩과 매개변수 효율적인 여러 예측기 MLP를 쓴다.
공개 CNN의 `PiecewiseLinearEmbedding`은 공식 조각선형 표현이 아니고 평가 시 다중 표본 탈락 경로도 단일 선형층과 같아 TabM의 수치 표현이나 여러 예측기 구조를 새롭게 대체하지 않는다.
공개 CNN과 TabM의 상관이 후보 중 가장 높지만 `0.95593`으로 낮다는 사실은 입력과 학습 경로를 포함한 전체 예측이 다르다는 근거일 뿐 합성곱 단독의 근거는 아니다.

TabPFN-3은 외부 자료로 사전 학습한 문맥 내 예측 모형이라 작동 원리가 근본적으로 다르다.
현재 TabPFN-3의 3시드 OOF `0.9672432267`보다 공개 CNN 단일 seed OOF가 약 `0.0004624` 높지만, 서로 다른 학습 절차와 시드 수이므로 champion 우선순위 근거로 사용하지 않는다.

[Contextualized spline Transformer 진입 진단](contextualized-spline-entry-diagnostic.md)은 더 충실한 다중 해상도 조각선형 표현과 주기 대조가 낮은 중복을 보였어도 fold 0 champion 및 통제된 다양성 관문을 모두 놓쳐 중단됐다.
이 선행 결과는 수치 표현과 공개 OOF 다양성만으로 전체 실행을 열 수 없음을 보여 준다.
CNN은 모형이 훨씬 작고 진입 비용이 낮으며 국소 합성곱이라는 별도 가설이 있으므로 같은 방식의 한 fold 대조까지만 열 가치가 있다.

## 권장 진입 진단과 중단 조건

M0는 현재 champion의 33개 피처 계획, seed 42와 fold 0을 그대로 사용한다.
목표·빈도 인코딩은 빼고 연속 입력의 fold-fit 분위수 변환, 학습 가능한 경첩 및 주기 표현, 두 합성곱, 채널 재가중, 평균·최댓값 요약과 원시 입력 우회 경로를 유지한다.
예측은 float32로 저장하고 fold별 초기화 seed, 의존성 판본과 결정론 설정을 고정한다.

A0는 같은 입력, 임베딩, 원시 입력 우회 경로, 매개변수 규모와 학습 설정을 유지하고 합성곱·채널 재가중·전역 요약을 매개변수 규모를 맞춘 완전 연결 상호작용으로 바꾼다.
M0와 A0의 차이는 공개 구조에서 근거가 없는 열 인접 합성곱이 실제로 기여하는지를 가른다.

- champion 목표는 M0 fold 0 AUC가 같은 seed의 저장 champion fold 0 AUC보다 낮지 않은 경우다.
- 다양성 목표는 M0가 champion 대비 `-0.01` 하한 안에 있고 최근접 스피어만 상관이 `0.998` 미만이며, 순위 평균 기여가 무정보 및 기존 구성원 복제 대조의 상단보다 큰 경우다.
- M0가 두 목표를 모두 놓치면 즉시 닫는다.
- M0가 목표 하나를 통과해도 A0가 AUC와 다양성 근거에서 같거나 더 좋으면 CNN이라는 구조 가설은 닫고 M0 5-fold로 확장하지 않는다.
- M0만 목표 하나와 합성곱 기여를 함께 통과하면 seed 42 전체 5-fold를 실행하고, 이후 3시드 확정과 후보 풀 진입은 [ADR 0001](../adr/0001-experiment-adoption-contract.md)을 그대로 따른다.

이 판정은 새 실험 티켓을 만들지 않는다.
후속 HITL 결정에서 CNN을 실행 후보로 확정할 때만 기존 실험 프로그램에 구현과 유료 진입 진단을 넘긴다.

## 출처

- [CNN for Predicting Smartphone Addiction 고정 판본](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction?scriptVersionId=342747549)은 소스, 입력, 모형, 학습 및 검증 절차의 1차 출처다.
- [CNN 공개 출력](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction/output?scriptVersionId=342747549)은 OOF, 시험 예측과 실행 기록의 1차 출처다.
- [On Embeddings for Numerical Features in Tabular Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)과 [공식 구현](https://github.com/yandex-research/rtdl-num-embeddings)은 조각선형 및 주기 수치 표현과 공개 코드의 차이를 판단하는 1차 출처다.
- [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper)과 [Multi-Sample Dropout](https://arxiv.org/abs/1905.09788)은 공개 코드가 이름을 가져온 구성요소의 원 정의다.
- [PyTorch `Conv1d`](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv1d.html), [PyTorch 결정론](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)과 [scikit-learn 분위수 변환](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)은 연산 축, 결정론과 전처리 동작의 공식 근거다.
- [`artifacts/champion.yaml`](../../artifacts/champion.yaml), [`artifacts/pool.yaml`](../../artifacts/pool.yaml)과 [실험 채택 판정 계약](../adr/0001-experiment-adoption-contract.md)은 현재 champion, 후보 풀과 판정 기준의 기록 원본이다.
- [S6E8 남은 실험 공간 전수 재점검](remaining-experiment-space-audit.md), [단변량 spline Transformer 검토](s6e8-167-spline-notebooks.md)와 [Contextualized spline Transformer 진입 진단](contextualized-spline-entry-diagnostic.md)은 중복 입력 판정과 최근 딥러닝 진입 진단의 저장소 근거다.
