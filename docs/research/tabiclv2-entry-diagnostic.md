# TabICLv2 진입 진단

이 문서는 GitHub 이슈 [#143](https://github.com/tmheo/predicting-smartphone-addiction/issues/143)의 구현과 Vast.ai fold 0 진입 진단 결과를 기록한다.

## 결론

TabICLv2 추정기 1개 진입 진단은 fold 0 seed 42에서 AUC `0.9417453063`을 기록했다.
같은 fold의 champion AUC `0.9685285491`보다 `0.0267832428` 낮아 허용 폭 `0.01`을 넘었다.
시간, 예측 무결성, 학습 자료 경계와 CPU 이동은 모두 통과했지만 성능 문턱에서 탈락했다.
이슈 규칙에 따라 공식 기본값인 추정기 8개 진입 진단, seed 42 5-fold, 중복·표준 순위 평균 기여 측정과 3시드 확정 재검증은 실행하지 않는다.
추정기 1개 결과를 정식 구성으로 채택하지 않으며 champion과 후보 풀은 바꾸지 않는다.

결측이 없는 행의 AUC도 `0.9529766844`로 전체 fold 승격 하한 `0.9585285491`보다 낮았다.
결측 행 AUC는 `0.9339012788`이었지만 약점이 결측 행에만 국한되지 않았으므로 champion의 복원·재구성 특성을 더한 두 번째 진입 진단은 열지 않는다.

## 구현

`src/pipeline/tabiclv2.py`에 공식 전처리와 TabICLv2 실행 경계를 별도 구현했다.
기준 코드는 `jingang01/TabICL`의 BSD 3-Clause 판본 커밋 `59a957cd644be4e1f2e1582757203ecbd630afa2`다.
Python 묶음은 `tabicl==2.1.1`이고 wheel SHA-256은 `cb4405cc93335c688bc9bcb703c7944032fcf542b43ebb66820f1a5acb5651b1`이다.
가중치는 `jingang/TabICL` 판본 `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`의 `tabicl-classifier-v2-20260212.ckpt`다.
가중치 SHA-256은 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`이다.

`TabICLv2Adapter`와 `MODEL_REGISTRY`의 `tabiclv2` 항목을 통해 기존 실행 계약에 연결했다.
`configs/exp083_tabiclv2_probe.yaml`은 추정기 1개 진입 진단이고 `configs/exp084_tabiclv2.yaml`은 실행하지 않은 공식 기본값 추정기 8개 설정이다.
공식 전처리와 원래 12개 열을 사용하고 카나리아 `placebo_noise`를 함께 측정했다.
permutation importance는 seed로 고정한 검증 행 1,000개와 반복 1회를 사용했다.
반복 예측은 표현 캐시를 재사용하고 메모리 이동은 `auto`로 요청했으며 디스크 재시도는 발생하지 않았다.

## 고정 원격 환경

| 항목 | 값 |
| --- | --- |
| 공급자 | Vast.ai |
| GPU | NVIDIA RTX A4000 1장, 16,376MiB |
| 컨테이너 | `nvcr.io/nvidia/pytorch:25.01-py3` |
| Python | `3.12.3` |
| NumPy | `1.26.4` |
| pandas | `2.2.3` |
| scikit-learn | `1.6.1` |
| PyTorch | `2.6.0a0+ecf3bae40a.nv25.01` |
| CUDA | `12.8` |
| TabICL | `2.1.1` |
| 실행 커밋 | `acd350fc9be041091af8629b1f2eb5d0ecb0634f` |

입력 묶음 SHA-256은 `8f5358b760bc07190ee464cb86aba70832c035ee489032347357be62abde3c6a`다.
학습 자료 SHA-256은 `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c`이고 fold 자료 SHA-256은 `5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4`다.

## fold 0 결과

| 측정값 | 결과 |
| --- | ---: |
| 학습 행 | 553,095 |
| 검증 행 | 138,274 |
| 검증 AUC | 0.9417453063 |
| 결측 없는 행 AUC | 0.9529766844 |
| 결측 있는 행 AUC | 0.9339012788 |
| 학습과 검증 예측 시간 | 230.7559초 |
| 테스트 예측 시간 | 292.2827초 |
| importance 시간 | 2,634.3589초 |
| 전체 진단 시간 | 3,159.2087초, 약 0.8776시간 |
| seed 42 5-fold 예상 시간 | 15,788.0954초, 약 4.3856시간 |
| PyTorch 최고 할당 메모리 | 14,352,397,312바이트 |
| PyTorch 최고 예약 메모리 | 16,313,745,408바이트 |
| 장치 메모리 대비 예약 비율 | 0.9738281342 |
| 공급자 장치 조회에서 관측한 최고 사용량 | 15,769MiB |

검증 예측과 테스트 예측은 모두 행 수, 순서와 유한성 검사를 통과했다.
학습 자료에 검증 라벨이 들어가지 않았고 후보 저장소는 학습 행 553,095개만 사용했다.
`candidate_store_training_only`와 `validation_labels_excluded_from_context` assertion은 모두 통과했다.
중요도 상위 열은 `daily_screen_time_hours`, `social_media_hours`, `weekend_screen_time` 순이었다.
카나리아보다 중요도가 낮은 원래 열도 있어 중요도만으로 후속 실행을 열 근거는 없었다.

진단 JSON, 검증 예측과 importance의 SHA-256은 각각 `d8292850d85c1409c662b605b95e285ee7a31069577009af6b5b5c929828d8f6`, `fc162b8516d0869cd21ddc8401b1f83fdc180f34d089eec175457d11c2e37600`, `007e46df822a33d2d9d174b78c04daee834ab9dc31ef31adf316cf6e08f18fd9`다.
전체 결과 묶음 SHA-256은 `8640ab509b12d082bf463991fb77dd4bf8b2d0c4a3be2d2de54be2f1c6d0434a`다.

## 판정과 자원 정리

5-fold 예상 시간은 24시간 한도를 통과했고 CPU 이동으로 RTX A4000 16GB에서 메모리 부족 없이 끝났다.
fold AUC가 champion보다 `0.01` 넘게 낮아 성능 중단 조건이 발생했다.
Vast.ai 인스턴스 `47844037`과 별도 저장 공간이 계정 목록에서 사라졌고 GitHub 보조 종료 예약도 빈 배열로 되돌렸다.
작업 중 만든 Vast.ai SSH 공개 키와 로컬 작업용 키 쌍도 삭제했으며 계정의 SSH 공개 키 목록이 빈 것을 재확인했다.
잔액 감소 기준 비용은 약 `$0.111691`이다.
