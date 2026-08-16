# Contextualized spline Transformer 진입 진단

이 문서는 GitHub 이슈 [#149](https://github.com/tmheo/predicting-smartphone-addiction/issues/149)의 구현과 Runpod fold 0 진입 진단 결과를 기록한다.

## 결론

M0 조각선형 모델은 fold 0 seed 42에서 AUC `0.9667574340`을 기록했다.
같은 seed의 exp067 fold 0 AUC `0.9682949114`보다 `0.0015374774` 낮아 champion 목표를 통과하지 못했다.
현재 후보 풀과의 최근접 Spearman 상관은 `0.9781382739`이고 순위 평균 기여는 `0.0001244004`였지만, 기존 구성원 복제 대조 상한 `0.0001279202`를 넘지 못해 다양성 목표도 통과하지 못했다.
이슈의 중단 조건에 따라 M0 seed 42 전체 5-fold와 추가 구조 탐색은 실행하지 않는다.
champion과 후보 풀은 바꾸지 않는다.

A0 주기 모델은 AUC `0.9667160779`로 M0보다 `0.0000413560` 낮았다.
최종 예측 성능은 사실상 같았지만 열별 가산 경로의 최적 AUC는 M0 `0.9629711618`, A0 `0.9558064166`으로 조각선형 표현이 가산 경로에는 더 잘 맞았다.
이 차이는 champion 교체나 후보 풀 편입을 정당화할 만큼 최종 예측에 이어지지 않았다.

## 구현과 출처

`src/pipeline/contextualized_spline_transformer.py`에 전체 행 사전 문맥 보정, 열별 다중 해상도 전문가, 동적 결합, 열별 잔차 블록, 가산 보조 손실, 1층 self-attention, 정확값 embedding과 최종 MLP를 구현했다.
M0는 다중 해상도 조각선형 전문가를 쓰고 A0는 나머지 구조를 유지한 채 해당 전문가만 주기 수치 표현으로 바꾼다.
두 설정은 exp067의 같은 33개 특성 계획을 사용하며 목표·빈도 인코딩은 넣지 않는다.
표준화, knot와 정확값 어휘는 outer 학습 부분에서만 맞추고 검증 라벨은 전처리에서 제외한다.
결측 식별자와 미등록값 식별자는 서로 다르며 검증·시험 예측의 행 수, 순서와 유한성을 공통 진단에서 확인한다.

기준 구조는 Kaggle 공개 노트북 [Contextualized Deep Univariate Spline Transformer 판본 3](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer/versions/3)을 참고했다.
원문 소스 SHA-256은 `c308b69cfeabad223a1e147fa174f78d1ddaccc09991b2075eecaf757f4781a2`이고 Apache License 2.0 고지 원문을 `src/pipeline/contextualized_spline_transformer.LICENSE`에 포함했다.
누출 경계, 재현성, 저장소 실행 계약과 짝지은 대조를 위해 원문 구현을 크게 수정했다.

## 고정 원격 환경

| 항목 | 값 |
| --- | --- |
| 공급자 | Runpod |
| GPU | NVIDIA RTX A4000 1장, 16,376MiB |
| 컨테이너 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Python | `3.12.3` |
| NumPy | `2.5.2` |
| pandas | `3.0.5` |
| scikit-learn | `1.9.0` |
| XGBoost | `3.4.0` |
| PyTorch | `2.8.0+cu128` |
| CUDA | `12.8` |
| 실행 커밋 | `d066dcc1752556500f11a439ee2836cc622f2dde` |

Vast.ai의 서로 다른 RTX A4000 호스트 두 곳은 `running` 전환 뒤에도 SSH 호스트 키를 제공하지 않아 입력 전송 전에 정리했다.
공급자 전환 조건에 따라 Runpod의 같은 RTX A4000 등급으로 M0와 A0를 함께 실행했다.
첫 Runpod 실행은 모델 진입 전에 PEP 668 시스템 환경 설치 금지로 실패했으며, 실패 결과를 보존한 뒤 새 작업 식별자와 경로에서 가상환경을 사용하도록 고쳐 재실행했다.
두 번째 입력 묶음 SHA-256은 `6b042b0a3edc29b25b977d2582393a32bba44c4ac56400fda8ae8631f0272693`이다.
학습 자료 SHA-256은 `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c`이고 fold 자료 SHA-256은 `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4`다.

## fold 0 결과

| 측정값 | M0 조각선형 | A0 주기 |
| --- | ---: | ---: |
| 학습 가능 매개변수 | 5,944,123 | 5,954,164 |
| 검증 AUC | 0.9667574340 | 0.9667160779 |
| 최적 가산 경로 AUC | 0.9629711618 | 0.9558064166 |
| 학습·검증 예측 시간 | 1,080.6570초 | 922.0889초 |
| 중요도 시간 | 64.6949초 | 60.0767초 |
| 전체 진단 시간 | 1,330.5874초 | 1,142.5578초 |
| 최고 할당 CUDA 메모리 | 3,391,676,416바이트 | 3,559,453,184바이트 |
| 최고 예약 CUDA 메모리 | 5,582,618,624바이트 | 5,796,528,128바이트 |
| 장치 메모리 대비 예약 비율 | 0.3332876868 | 0.3460582893 |
| seed 42 5-fold 예상 시간 | 6,599.4295초 | 5,663.7719초 |

M0와 A0의 진단 시간 합계는 `2,473.1452`초, 약 `0.6870`시간으로 1.5시간 상한을 통과했다.
두 모델 모두 30개 수치 token과 원래 12개 열의 정확값 embedding을 사용했다.
학습 부분 전용 전처리, 검증 라벨 제외, 결측·미등록 식별자 분리 assertion과 예측 무결성 검사를 모두 통과했다.

M0 중요도 상위 특성은 `notifications_per_day`, `imp_social_frac`, `app_opens_per_day`, `weekend_screen_time`, `daily_screen_time_hours` 순이었다.
A0도 같은 상위 구조를 보였지만 `weekend_screen_time_xgb_recon`의 중요도가 M0보다 컸다.

## 다양성 대조와 판정

| 측정값 | 결과 |
| --- | ---: |
| 기존 16개 후보 순위 평균 AUC | 0.9680148066 |
| M0 추가 순위 평균 AUC | 0.9681392070 |
| M0 기여 | 0.0001244004 |
| M0 최근접 후보 Spearman | 0.9781382739 |
| A0 기여 | 0.0001162246 |
| A0 최근접 후보 Spearman | 0.9793171317 |
| M0-A0 Spearman | 0.9773680939 |
| 무작위 대조 100개 최대 기여 | -0.0006693635 |
| 기존 구성원 복제 16개 최대 기여 | 0.0001279202 |
| 최종 대조 상한 | 0.0001279202 |

M0는 exp067 대비 `-0.01` 이내이고 최근접 상관 `< 0.998` 조건을 통과했다.
그러나 M0 기여가 최종 대조 상한보다 `0.0000035199` 낮아 다양성 목표의 마지막 조건을 놓쳤다.
champion 목표와 다양성 목표가 모두 거짓이므로 5-fold 승격은 하지 않는다.

## 결과 무결성과 자원 정리

M0 진단 JSON, 검증 예측과 중요도 SHA-256은 각각 `5320c8565b0d1d6016a5dcfe5cc8f89c6af3b8761338bce8760207a8cb6bcd2e`, `33b97265f90449561c3772a95eb9cea754aa18082b50f3c3ca28b49a0dc4fce0`, `6bacda18e6f296c5786984ae9048db85b75b5b74619e34c2632dd18209bd0632`다.
A0 진단 JSON, 검증 예측과 중요도 SHA-256은 각각 `82c821ffbbadc645bda6172672c8261f45c35f90ef71ed908be5b5cc6caeefa1`, `ddda46c4db6bd860c1300e6de0bc609f7cb8b665ef01f77a3b41d93c4b4f8fd5`, `64b27512b58c4f418c8d2f039f6e6d8bd896834727699881324a7bded4919b35`다.
비교 JSON SHA-256은 `7689723d27ba0180d3d8660edf109d827dbbd0b2f9ba2b1f249f69f2be48cdda`이고 전체 결과 묶음 SHA-256은 `872b25319b5b8d5586f3ef294f5716efcb87cc72f835481fad1a7bbb8d0c235b`다.
내부 결과 manifest의 모든 파일이 로컬 회수 뒤 다시 일치함을 확인했다.

Runpod Pod를 중지한 뒤 영구 종료해 30GB 컨테이너 디스크와 20GB Pod 볼륨을 삭제했다.
Pods와 네트워크 저장 공간은 비어 있고 Serverless endpoint와 cluster도 없으며 최종 지출 속도는 `$0.000`/시간이다.
Runpod 실제 사용액은 잔액 감소 기준 약 `$0.27`이고 Vast.ai의 두 실패 시도 비용은 합계 `$0.052`다.
Pod 생성 시점에 Stripe를 통한 `$10` 크레딧 충전이 발생했으며, 이 금액은 사용액이 아니라 Runpod 계정 잔액으로 남아 있다.
