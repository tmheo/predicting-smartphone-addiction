# 스칼라 token Transformer 진입 진단

이 문서는 GitHub 이슈 [P3 재개: 스칼라 token Transformer 구현과 fold 0 진입 진단](https://github.com/tmheo/predicting-smartphone-addiction/issues/178)의 구현과 Vast.ai fold 0 진입 진단 결과를 기록한다.

## 결론

스칼라 token Transformer M0는 fold 0 seed 42에서 AUC `0.9551717921`을 기록했다.
같은 RTX A4000에서 재현한 champion AUC `0.9685566249`보다 `0.0133848329` 낮아 허용 폭 `0.01`을 넘었다.
attention을 매개변수 규모가 같은 열별 잔차 MLP로 바꾼 A0는 AUC `0.9564251321`을 기록해 M0보다 `0.0012533400` 높았다.
따라서 M0는 성능 하한과 `M0 > A0` 조건을 모두 통과하지 못했다.
플라시보, 자료 경계, 예측 무결성, 시간과 메모리 검사는 통과했다.
이슈 규칙에 따라 3시드 5-fold 확정 재검증을 열지 않고 스칼라 token Transformer 트랙을 닫는다.
champion과 후보 풀은 바꾸지 않는다.

## 구현과 출처

`src/pipeline/scalar_token_transformer.py`에 스칼라별 ReLU 기저와 학습 주기 기저, 3층 attention, 펼친 token과 원래 스칼라를 함께 읽는 MLP를 구현했다.
`ScalarTokenTransformerAdapter`와 모델 등록부의 `scalar_token_transformer` 항목으로 기존 fold 실행 계약에 연결했다.
M0 설정은 `configs/exp115_scalar_token_transformer_m0.yaml`이고 A0 설정은 `configs/exp116_scalar_token_mlp_a0.yaml`이다.
두 설정은 champion `exp081_lookup_fold_initialization_avg3`과 같은 33개 피처 계획을 사용한다.

기준 구조는 Kaggle의 [TabTransformer : Predicting Smartphone Addiction 공개 판본 1](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction/versions/1)이다.
고정한 노트북 SHA-256은 `eeb3e1cccbaab29c71ef946876f7042509f6ef537df4a9b04ced36e3c424e46c`다.
이 공개 구조는 원 논문의 TabTransformer와 다르므로 저장소에서는 `scalar_token_transformer`로 명명했다.
노트북 소스의 Apache License 2.0 원문은 `src/pipeline/scalar_token_transformer.LICENSE`에 보존했고 파생 구현의 변경 사항은 모듈 머리말에 기록했다.

공개 노트북의 정확값 목표값 인코딩, 빈도 인코딩과 자체 파생 피처는 제거했다.
범주값 스칼라화와 분위 변환은 outer 학습 부분에만 맞추고 결측 범주와 학습 중 보지 못한 범주를 다른 ID로 표현한다.
학습 방식은 batch 256, AdamW, 라벨 평활, mixup, EMA, cosine restart, 다중 dropout head와 patience 18을 보존했다.
fold별 독립 시드, 결정론 설정, 플라시보와 permutation importance를 추가했다.

A0는 attention block만 열 사이 정보를 섞지 않는 열별 잔차 MLP로 바꾼 제거 대조다.
attention block은 `49,984`개, A0 block은 `49,986`개 매개변수를 가져 상대 차이가 `0.0000400128`이다.
전체 학습 가능 매개변수는 M0 `747,249`개, A0 `747,255`개다.

## 고정 원격 환경

| 항목 | 값 |
| --- | --- |
| 공급자 | Vast.ai |
| GPU | NVIDIA RTX A4000 1장, 16,376MiB |
| 컨테이너 | `nvcr.io/nvidia/pytorch:26.01-py3` |
| Python | `3.12.3` |
| NumPy | `2.5.2` |
| pandas | `3.0.5` |
| scikit-learn | `1.9.0` |
| PyTorch | `2.13.0+cu130` |
| CUDA | `13.0` |
| 실행 커밋 | `c54591dbfe552ec77324943556ae69d582d91e16` |

학습 자료 SHA-256은 `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c`다.
시험 자료 SHA-256은 `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e`다.
fold 자료 SHA-256은 `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4`다.
입력 묶음 SHA-256은 `398d3e6bd102ae8a71e865919e14ea738722767efee0265488f8758b506e60e3`다.

champion 기준선은 표 합성곱망 진입 진단과 공유해 한 번만 실행했다.
기준 진단 JSON과 검증 예측 SHA-256은 각각 `869dfdeab4f0512b1fa02c7bbd0d72a1289613c9a402e66b0e39a9d3f746bc12`, `a916c98377b19e08b46b7c9ab1e090aad9c448ae9d96e60ec4843bbe4675f4c8`다.
기준선, M0와 A0는 같은 공급자와 GPU 등급, Python 의존성, 입력 자료, fold와 seed를 사용했다.

## fold 0 결과

| 측정값 | M0 attention | A0 열별 잔차 MLP |
| --- | ---: | ---: |
| 학습 행 | 553,095 | 553,095 |
| 검증 행 | 138,274 | 138,274 |
| 검증 AUC | 0.9551717921 | 0.9564251321 |
| champion 대비 | -0.0133848329 | -0.0121314928 |
| 최고 epoch | 38 | 38 |
| 전체 진단 시간 | 955.4356초 | 607.0677초 |
| 5-fold 예상 시간 | 4,765.0575초 | 3,023.2766초 |
| PyTorch 최고 할당 메모리 | 296,756,224바이트 | 256,878,080바이트 |
| PyTorch 최고 예약 메모리 | 335,544,320바이트 | 276,824,064바이트 |
| 장치 메모리 대비 예약 비율 | 0.0200298883 | 0.0165246579 |

후보 두 개의 전체 진단 시간 합계는 `1,562.5033`초, 약 `0.4340` GPU 시간으로 예약 상한 1.5 GPU 시간 안이다.
공급자 장치 조회에서 관측한 최고 메모리 사용량은 589MiB였다.
두 실행의 검증과 시험 예측은 모두 행 수, 순서와 유한성 검사를 통과했다.

## 자료 경계와 플라시보

두 실행 모두 `preprocessing_training_rows_only`, `validation_labels_excluded_from_preprocessing`, `missing_and_unknown_categories_distinct`, `attention_ablation_parameter_matched` 단언을 통과했다.
목표값 인코딩 수는 두 실행 모두 0이다.
분위 변환은 outer 학습 부분에서 최대 10,000개의 분위 표본만 사용했다.
M0의 플라시보 중요도는 `0.0000303478`로 33개 피처 중 31위였다.
A0의 플라시보 중요도는 `0.0000138989`로 33개 피처 중 32위였다.
따라서 플라시보 게이트는 유효했지만 구조 승격 실패를 뒤집지는 않는다.

M0 중요도 상위 세 피처는 `weekend_screen_time_xgb_recon`, `daily_screen_time_hours_xgb_recon`, `screen_slack`이었다.
A0 중요도 상위 세 피처는 `weekend_screen_time_xgb_recon`, `screen_slack`, `imp_social_frac`이었다.

## 산출물과 자원 정리

M0 진단 JSON, 검증 예측과 중요도 SHA-256은 각각 `c0a5862a52cf216f57db5743408c6ff25cd690907be8ebf474c80433c7b2a5fa`, `cb943a4db59aea3db9afd11e8c8568ae8d703dbee14c3f589bac68526facdc85`, `7e8f2cc8b7ed3e6908ad0510bd579da0a526539d76b426576a3d3ea70bc8810d`다.
A0 진단 JSON, 검증 예측과 중요도 SHA-256은 각각 `61e580ac6d42b6220f75a731fc68131193bfbe28cdaa7055eec32fac721df931`, `75046f58dba2e47b6b95e2d7cb3a48dc2f1c36fff41c79a62a2bf0c85e2bada1`, `c6c12579d960281a51853d009906408ef47be65eaaa954bf0f87251ca6b7c3db`다.
전체 결과 묶음 SHA-256은 `562cadf6137813a9e985a574fe65db7935a6b43a89161134d29d0829d06cffe6`다.
결과 묶음의 내부 해시 목록을 로컬에서 다시 검증했다.

Vast.ai 인스턴스 `48027890`은 삭제됐고 이 작업의 독립 종료 예약도 제거됐다.
계정에 별도 저장 공간이 없음을 확인했다.
이 작업 라벨의 청구 합계는 `$0.091`이다.

## 최종 판정

| 판정 조건 | 결과 | 판정 |
| --- | ---: | --- |
| M0 AUC가 champion보다 0.01 이내 | -0.0133848329 | 실패 |
| M0 AUC가 A0보다 높음 | -0.0012533400 | 실패 |
| 플라시보 게이트 유효 | 두 실행 모두 유효 | 통과 |
| 후보 합산 시간 1.5 GPU 시간 이내 | 약 0.4340시간 | 통과 |
| 예측과 자료 경계 무결성 | 모든 검사 통과 | 통과 |

성능 하한과 attention 제거 대조가 모두 실패했으므로 3시드 5-fold 확정 재검증을 열지 않는다.
