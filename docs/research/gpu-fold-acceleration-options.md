# 외부 GPU 실행의 fold-fit 피처와 폴드 결과 확정 가속 경로

## 조사 질문

고정한 원격 Python 및 CUDA 환경과 CPU 기본 경로를 함께 지원하면서, 측정된 `fold-fit 피처` 준비와 `폴드 결과 확정` 병목을 어떤 공식 지원 경로로 줄일 수 있는지 조사했다.
검토 범위는 cuDF와 `cudf.pandas`, CuPy, XGBoost CUDA 실행, scikit-learn 계열의 대체 범위, PyTorch 텐서 기반 순열과 묶음 추론이다.
판정 기준은 seed 42와 고정 5폴드에서 복구 가능한 결과가 나올 때까지의 유료 원격 자원 실제 시간이며, 피처 값, 예측, 중요도 정밀도와 재현 계약을 유지해야 한다.

## 결론

전역 `pandas`를 `cuDF`로 바꾸는 방식은 이 저장소의 첫 선택으로 적합하지 않다.
현재 확인된 긴 지연은 주로 DataFrame 산술이 아니라 fold마다 반복되는 `IterativeImputer`와 다섯 개 XGBoost 복원기 학습, 그리고 복구 확인보다 먼저 이 계산을 수행하는 순서에서 생긴다.
따라서 우선순위는 완료 폴드의 계산 생략, 동일한 fold-fit 결과의 안전한 재사용, 제공자와 모델 어댑터 내부의 제한된 GPU 실행 순이다.

권장하는 경계는 CV와 피처 계획에는 계속 pandas DataFrame을 사용하고, 실제 병목을 소유한 제공자나 모델 어댑터만 선택적으로 GPU 자료 구조로 변환한 뒤 기존 pandas 및 NumPy 결과 계약으로 되돌리는 방식이다.
CPU 실행 경로는 항상 남기되, 요청한 실행 방식과 실제 실행 방식 및 CPU로 돌아간 이유를 결과에 기록해야 한다.
성능 판정 실행에서는 GPU 요청이 조용히 CPU 실행으로 바뀌면 안 된다.

현재 이슈 303에서 TabCNN 순열 중요도는 폴드당 약 2.21초에서 3.13초였고, 목표 평균 팔도 약 3.90초에서 3.96초였다.
따라서 이 실행의 30분 이상 지연을 순열 중요도 탓으로 볼 근거는 없다.
반면 이전 [이슈 274의 RTX 3090 실측](https://github.com/tmheo/predicting-smartphone-addiction/issues/274#issuecomment-5351040466)은 `training` 단계 대부분이 CPU의 목표 평균, 제약 복원과 XGBoost 복원 피처 생성에 쓰였다고 기록했다.
이슈 303 수치는 실행 묶음의 `model_training_diagnostics.json`에서 확인했지만, 제공자별 영구 시간 기록은 아직 없으므로 세부 귀속은 지도 318의 별도 계측 티켓에서 확정해야 한다.

## 현재 코드에서 확인한 계산 경로

[`src/pipeline/cv.py`](../../src/pipeline/cv.py)는 각 폴드에서 모든 fold-fit 제공자의 `fit`과 train 및 test `transform`을 끝낸 다음 복구 지점을 읽는다.
완료된 폴드 13개를 가진 이슈 303 재개 실행도 이 순서 때문에 완료 폴드의 비싼 피처 준비를 다시 수행한다.
[`FeaturePlan.all_columns()`](../../src/pipeline/plan.py)은 fold-fit 출력 열까지 선언으로 돌려주므로, 실제 변환 행렬 없이도 복구 자료의 열 계약을 먼저 검사할 여지가 있다.

[`ConstrainedImputeAux`](../../src/pipeline/features.py)는 scikit-learn의 `IterativeImputer(max_iter=20)`을 폴드마다 맞춘다.
scikit-learn 문서는 기본 `IterativeImputer` 계산량이 대략 `O(k n p^3 min(n,p))`이며, `n_nearest_features`, `skip_complete` 또는 `tol`로 계산량을 줄일 수 있다고 설명한다.
그러나 이 선택지는 출력값이나 현재 복원 명세를 바꾸므로 이번 성능 작업의 투명한 가속 수단으로 쓰면 안 된다.
[scikit-learn `IterativeImputer` 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)

[`XgbImputeAux`](../../src/pipeline/features.py)는 필요한 복원 열마다 별도 `XGBRegressor`를 만들고 현재 이슈 303에서는 다섯 모델을 각각 400개 트리로 맞춘다.
설정에는 `tree_method="hist"`가 있지만 `device="cuda"`가 없으므로 이 제공자는 GPU 모델과 함께 실행해도 CPU에서 학습한다.
제공자 내부의 이 다섯 모델은 측정 가치가 가장 높은 GPU 계산 후보지만, CPU와 GPU의 복원값이 같다는 가정은 할 수 없다.

[`TabCNNFold.importance()`](../../src/pipeline/tab_cnn.py)는 열마다 pandas DataFrame 전체를 복사하고 열 하나를 섞은 다음, 매번 전처리를 거쳐 GPU 예측을 수행한다.
이 구조는 실제로 순열 중요도가 느린 신경망 어댑터에서는 불필요한 호스트 복사와 반복 인코딩을 만들 수 있다.
다만 이슈 303은 8,192행, 반복 1회로 제한돼 있으므로 이 최적화의 대표 병목 사례가 아니다.

LightGBM 어댑터는 모델의 내장 gain 중요도를 바로 읽으므로 순열 예측 반복이 없다.
CPU LightGBM 대표 실행에서는 pandas 입력과 내장 중요도 경로를 그대로 유지하는 것이 기준이다.

## 공식 지원 경로별 판정

### `cudf.pandas`

`cudf.pandas`는 지원되는 pandas 호출을 GPU에서 실행하고, 지원되지 않는 호출은 자료를 옮겨 pandas로 실행한다.
함수 및 줄 단위 분석 도구로 GPU 실행과 CPU 전환을 확인할 수 있다.
[cuDF pandas 가속 실행과 분석 도구](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/usage/)

가속 기능은 pandas를 가져오기 전에 켜야 하며, `multiprocessing`이나 `ProcessPoolExecutor` 작업 프로세스에서도 같은 초기화가 필요하다.
현재 파이프라인은 여러 모듈에서 pandas를 최상위에서 가져오고 시드별 프로세스를 만들기 때문에, 전역 활성화는 작은 내부 최적화가 아니라 전체 실행 초기화 계약 변경이다.
[cuDF pandas 가속의 다중 프로세스 초기화 조건](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/usage/#working-with-multiprocessing-or-concurrent-futures)

CPU 전환은 호스트와 장치 사이의 복사를 만들 수 있고, GPU와 CPU 호출이 자주 교차하면 이득이 사라질 수 있다.
또한 공식 호환성 안내에는 cuDF 병합 결과의 행 순서가 결정적이지 않다고 명시돼 있다.
[cuDF pandas 작동 방식](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/how-it-works/)
[cuDF pandas 호환성 차이](https://docs.rapids.ai/api/cudf/stable/cudf/pandascompat/)

`cudf.pandas`는 지원 범위를 찾는 진단 도구로는 유용하다.
대표 seed 42 한 폴드에서 분석기를 먼저 돌리고, 실제 DataFrame 호출이 전체 지연의 큰 비중을 차지할 때만 제한된 경로의 구현 후보로 올리는 것이 맞다.
전역 기본 실행 방식으로 채택하는 것은 제외한다.

### 직접 cuDF 사용

직접 cuDF를 쓰면 제공자 하나의 자료 이동 시점과 지원 연산을 통제할 수 있다.
cuDF는 수치, 문자열, 범주와 pandas의 nullable dtype 다수를 지원하지만 모든 dtype이 nullable이고 임의의 Python `object` dtype은 지원하지 않는다.
[cuDF 지원 dtype](https://docs.rapids.ai/api/cudf/stable/cudf/data-types/)
[cuDF 자료형 설계](https://docs.rapids.ai/api/cudf/legacy/developer_guide/library_design/#data-types)

cuDF의 결측 표현은 pandas 및 NumPy와 완전히 같지 않다.
기본 생성에서 부동소수점 `NaN`이 null로 해석될 수 있으므로, 변환 전후 결측 마스크와 dtype을 명시적으로 검사해야 한다.
[cuDF 결측값 공식 안내](https://docs.rapids.ai/api/cudf/legacy/user_guide/missing-data/)

현재 `XgbImputeAux`는 수치 열, pandas 범주 열, 행 순서와 열 순서를 명시적으로 검사한다.
이 제공자 내부에서 fold 입력을 한 번 cuDF로 옮겨 다섯 XGBoost 모델이 공유하게 하는 방식은 전역 교체보다 경계가 좁다.
그러나 먼저 `device="cuda"`만 추가한 pandas 입력 경로를 측정해야 한다.
XGBoost가 pandas 입력을 스스로 GPU용 행렬로 만들 수 있으므로, cuDF가 추가로 필요한지는 반복 자료 전송이 실제 병목인지 확인한 뒤 판단할 수 있다.

### XGBoost CUDA

XGBoost 공식 Python 인터페이스는 `XGBRegressor(tree_method="hist", device="cuda")`를 지원한다.
scikit-learn 인터페이스에서는 `QuantileDMatrix`와 제자리 예측 경로가 자동으로 사용된다.
[XGBoost GPU 공식 안내](https://xgboost.readthedocs.io/en/stable/gpu/)
[XGBoost Python 인터페이스 예시](https://xgboost.readthedocs.io/en/stable/python/python_intro.html#scikit-learn-interface)

XGBoost는 pandas DataFrame과 cuDF DataFrame을 모두 scikit-learn 입력으로 지원하고, cuDF 입력은 GPU 자료를 그대로 받을 수 있다.
범주 열은 pandas와 cuDF 모두 `category` dtype으로 전달할 수 있으며 `enable_categorical=True`와 `hist`를 함께 쓸 수 있다.
[XGBoost 지원 Python 자료 구조](https://xgboost.readthedocs.io/en/stable/python/data_input.html)
[XGBoost 범주형 자료 공식 안내](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)

XGBoost 3.1부터 Python DataFrame 입력의 범주 값을 저장하고 예측 때 다시 부호화하는 기능이 있지만, 입력 범주 집합과 순서를 기존 pandas 경로와 맞추는 검사는 여전히 필요하다.
현재 제공자가 이미 수행하는 train 및 test 범주 합의 검사와 출력 열 순서 검사를 GPU 경로에서도 유지해야 한다.
[XGBoost 범주 자동 부호화](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html#auto-recoding-data-consistency)

GPU와 CPU의 병렬 부동소수점 계산이 같은 트리와 복원값을 비트 단위로 만든다는 공식 보장은 찾지 못했다.
따라서 `XgbImputeAux`의 CUDA 실행은 투명한 속도 개선으로 바로 채택하지 말고, 같은 seed 42와 5폴드에서 복원 열, 최종 예측, 중요도와 복구 자료를 CPU 기준과 비교해야 한다.
복원 열이 정확히 같지 않으면 현재 출력 보존 조건 아래에서는 별도 모델 변형으로 분류하거나 첫 구현 범위에서 제외해야 한다.

### CuPy

CuPy는 NumPy와 비슷한 배열 연산을 GPU에서 제공하고, PyTorch와 CUDA Array Interface 또는 DLPack으로 GPU 메모리를 복사하지 않고 공유할 수 있다.
[CuPy와 PyTorch 상호 운용](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch)

CuPy는 DataFrame 범주와 인덱스 의미를 소유하지 않으므로, 전체 피처 계획의 pandas 대체물로는 맞지 않는다.
연속형 배열만 다루는 제공자 내부 계산이나 PyTorch와 다른 GPU 배열 라이브러리 사이의 변환 경계에서만 후보가 된다.

CuPy와 NumPy는 범위를 벗어난 인덱스, 중복 위치 대입, 일부 형 변환, 축약과 난수에서 차이가 있다.
CuPy는 같은 seed라도 주 판본이 바뀌면 같은 난수열을 보장하지 않는다.
[CuPy와 NumPy의 동작 차이](https://docs.cupy.dev/en/stable/user_guide/difference.html)
[CuPy 난수 재현 범위](https://docs.cupy.dev/en/stable/reference/random.html)

기존 순열 중요도의 정확한 순열을 보존하려면 CuPy 난수로 바꾸지 말고 현재 NumPy가 만든 순열 인덱스를 GPU로 한 번 옮겨 사용해야 한다.

### scikit-learn 및 cuML

`cuml.accel`은 지원하는 scikit-learn 추정기를 GPU로 보내고 나머지는 CPU 원본 구현으로 돌린다.
공식 지원 목록에는 `IterativeImputer`가 없으며, cuML이 직접 제공하는 대치기는 단변량 `SimpleImputer`다.
따라서 현재 `ConstrainedImputeAux`의 다변량 반복 복원을 의미와 결과를 보존한 채 자동으로 GPU에서 돌리는 공식 경로는 없다.
[cuML 가속 지원 추정기와 CPU 전환 조건](https://docs.rapids.ai/api/cuml/stable/cuml-accel/compatibility/)
[cuML 전처리 API 목록](https://docs.rapids.ai/api/cuml/stable/api/#other-preprocessing-methods-single-gpu)

cuML 공식 문서는 GPU와 CPU 구현의 품질은 비교 가능해야 하지만 수치가 같다고 보장하지 않고, 작은 자료에서는 초기화와 자료 이동 비용 때문에 더 느릴 수 있다고 밝힌다.
현재 복원기 교체가 피처 값 보존 범위 밖이므로 `SimpleImputer`, cuML 선형 모델 또는 다른 반복 복원 구현으로 바꾸는 일은 이번 가속 작업에서 제외한다.
[cuML 가속의 결과와 성능 조건](https://docs.rapids.ai/api/cuml/stable/cuml-accel/compatibility/#general-behavior)

CPU에서 실제로 느린 순열 중요도는 scikit-learn의 `n_jobs`로 열 단위 병렬화할 수 있다.
기존 표본 수, 반복 수, seed와 점수 계산을 바꾸지 않는다는 장점이 있지만, 모델 자체 스레드와 시드 병렬 실행이 겹치면 과도한 병렬화가 생길 수 있으므로 해당 어댑터에서만 측정해야 한다.
[scikit-learn 순열 중요도 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)

### PyTorch 텐서 순열과 묶음 추론

PyTorch는 지정한 장치와 난수 생성기를 받는 `torch.randperm`을 제공한다.
그러나 기존 NumPy 순열과 같은 seed가 같은 순열을 뜻하지는 않으므로, 출력 보존이 필요할 때는 기존 NumPy 순열을 그대로 사용해야 한다.
[PyTorch `torch.randperm`](https://docs.pytorch.org/docs/stable/generated/torch.randperm.html)

신경망 어댑터가 한 열씩 독립적으로 변환하고 변환 결과가 원래 피처 열과 일대일로 대응한다면, 검증 자료를 한 번만 인코딩한 뒤 해당 텐서 열에 기존 순열 인덱스를 적용하는 것은 매 순열마다 pandas 복사와 인코딩을 반복할 필요를 없앤다.
여러 피처의 변형 배치를 합쳐 예측하면 Python 호출 횟수도 줄일 수 있다.
교차 피처 변환, 자료 전체 통계, 행 사이 문맥 또는 검색 자료 구조가 예측 전처리에 들어가는 어댑터에는 이 등가성이 성립하지 않으므로 적용하면 안 된다.

현재 코드는 이미 `torch.no_grad()`를 사용한다.
`torch.inference_mode()`는 추가로 view 추적과 판본 계수 비용을 줄일 수 있지만, 출력과 메모리 사용을 먼저 비교해야 하는 작은 후보다.
[PyTorch 추론 모드 안내](https://docs.pytorch.org/docs/stable/torch.html#locally-disabling-gradient-computation)

PyTorch의 CUDA 실행은 비동기이므로, 폴드 세부 시간 계측에는 CUDA 동기화 또는 CUDA 이벤트가 필요하다.
단순한 호스트 시계만 재면 커널 실행 시간이 다음 단계로 밀려 잘못 귀속될 수 있다.
[PyTorch CUDA 동기화](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize)
[PyTorch 벤치마크 도구의 가속기 동기화](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

결정적 알고리즘 설정은 같은 소프트웨어와 하드웨어에서 결정적 구현을 선택하거나 지원되지 않는 연산에서 오류를 내지만, 이것만으로 전체 실행 재현을 보장하지는 않는다.
현재 어댑터의 seed, 결정적 알고리즘과 비결정적 연산 금지 설정을 GPU 중요도 경로에서도 유지해야 한다.
[PyTorch 결정적 알고리즘 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
[PyTorch 재현 공식 안내](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## pandas 의미 보존 조건

GPU 경로는 다음 조건을 모두 통과해야 한다.

- 입력과 출력의 열 이름 및 열 순서가 CPU 기준과 정확히 같아야 한다.
- 행 식별자, 인덱스와 행 순서가 CPU 기준과 정확히 같아야 한다.
- 각 열의 dtype, 범주 목록, 범주 순서와 ordered 표시가 같아야 한다.
- `NaN`, pandas `NA`와 null의 위치를 별도 마스크로 비교해야 한다.
- 양의 무한대와 음의 무한대, 부호 있는 0과 유한값 여부를 확인해야 한다.
- 반환되는 복원 열은 현재 계약대로 `float64`여야 한다.
- 순열 중요도는 같은 검증 행, 같은 표본 수, 같은 반복 수, 같은 NumPy 순열 인덱스와 같은 AUC 계산을 사용해야 한다.
- GPU 커널 실행이 끝난 뒤 시간을 기록해야 한다.
- 요청한 실행 방식, 실제 실행 방식, CPU 전환 사유, GPU 종류, Python, CUDA, 드라이버와 관련 패키지 판본을 복구 실행 정체성과 진단에 포함해야 한다.

GPU XGBoost처럼 계산 구현 자체가 달라지는 후보는 두 단계로 판정해야 한다.
먼저 복원 열과 최종 산출물의 정확한 동일성을 검사한다.
정확히 같지 않으면 허용 오차를 임의로 만들지 않고 투명한 가속 후보에서 탈락시킨다.
향후 별도 모델 변형으로 검토하려면 그때 품질 동등성, 판본 고정과 재현 기준을 새로 승인받아야 한다.

## 권장 인터페이스 경계

CV 루프와 `FeaturePlan`의 공용 자료형은 pandas로 고정한다.
GPU 자료형 선택을 모델 종류 하나로 결정하지 않고, 병목을 소유한 제공자 또는 모델 어댑터가 자신의 실행 방식을 선택하게 한다.

권장하는 최소 계약은 다음 정보를 가진다.

```text
requested_engine: cpu | cuda | auto
effective_engine: cpu | cuda
fallback_reason: null | 문자열
device_identity: null | GPU 및 CUDA 정보
implementation_versions: 관련 패키지 판본
```

CPU 기본값은 선택 GPU 의존성을 가져오지 않고 기존 pandas 경로를 그대로 실행해야 한다.
`auto`는 일반 로컬 실행에서만 CPU 전환을 허용할 수 있다.
외부 GPU 성능 판정은 `cuda`를 명시하고 지원하지 않는 환경이나 연산이면 즉시 실패해야 한다.

`XgbImputeAux`의 후보 경계는 pandas 입력을 받고 내부에서 `cpu` 또는 `cuda` XGBoost 실행을 선택하며, 필요할 때만 cuDF로 한 번 변환하고 마지막에는 기존 pandas `float64` 출력으로 돌아오는 것이다.
신경망 중요도의 후보 경계는 모델 어댑터가 현재 중요도 표본과 seed 규칙을 소유한 채 내부의 인코딩 재사용 및 묶음 추론만 선택하는 것이다.
LightGBM CPU와 내장 중요도 경로는 GPU 의존성을 전혀 읽지 않는 pandas 기준 구현으로 남긴다.

전역 DataFrame 별칭 또는 `DataFrameLike` 공용 추상화는 만들지 않는다.
pandas와 cuDF의 세부 차이를 모든 제공자와 CV 호출자에게 퍼뜨리며, 실제 병목이 아닌 코드까지 두 실행 방식으로 시험하게 만들기 때문이다.

## 환경과 선택 의존성 조건

현재 프로젝트는 Python 3.12 이상, pandas 3.0.5 이상, NumPy 2.5.2 이상, scikit-learn 1.9.0 이상, SciPy 1.16.1 이상, PyTorch 2.5 이상과 XGBoost 3.4.0 이상을 선언한다.
RAPIDS 26.08은 Python 3.11부터 3.14, CUDA 12.2부터 12.9 또는 CUDA 13.0부터 13.3을 지원하고 CUDA 12에서는 드라이버 535 이상을 요구한다.
GPU는 Volta 세대, 계산 능력 7.0 이상이어야 한다.
[RAPIDS 26.08 지원 환경](https://docs.rapids.ai/platform-support/#rapids-26-08)

RAPIDS 26.08부터 cuDF는 pandas 3.0 이상을 요구하므로 현재 pandas 하한과 방향은 맞는다.
다만 실제 `uv.lock`과 원격 이미지에서 함께 풀리는 정확한 판본은 구현 전에 별도 설치 검증이 필요하다.
[RAPIDS 26.08 pandas 3 전환 공지](https://docs.rapids.ai/notices/rsn0059/)

cuDF의 pip 패키지는 설치된 CUDA 주 판본과 맞는 `-cu12` 또는 `-cu13` 변형을 골라야 한다.
RAPIDS는 Linux를 기본 지원하며 macOS CPU 개발 환경의 필수 의존성으로 넣을 수 없다.
[RAPIDS 설치 공식 안내](https://docs.rapids.ai/install/)

CuPy 14는 Python 3.10 이상, CUDA 12 이상, NumPy 2.0 이상과 SciPy 1.14 이상을 지원한다.
CUDA 12와 13의 pip 패키지 이름이 다르고 여러 CuPy 패키지를 함께 설치하면 충돌한다.
[CuPy 호환 표](https://docs.cupy.dev/en/stable/upgrade.html#compatibility-matrix)
[CuPy 설치 공식 안내](https://docs.cupy.dev/en/stable/install.html)

XGBoost 3.4의 기본 Linux wheel은 CUDA 13으로 빌드되며, CUDA 12 드라이버 환경에는 `xgboost-cu12` 변형이 필요하다.
CPU 전용 `xgboost-cpu` 패키지는 GPU 알고리즘을 포함하지 않는다.
[XGBoost 설치 공식 안내](https://xgboost.readthedocs.io/en/stable/install.html#cuda-toolkit-variants-linux)

따라서 RAPIDS, CuPy와 GPU XGBoost는 기본 프로젝트 의존성이 아니라 Linux 외부 GPU 실행용 선택 의존성으로 두는 것이 맞다.
원격 실행 전 확인은 실제 Python, 드라이버, CUDA, GPU 계산 능력, 패키지 변형과 간단한 GPU 계산을 모두 검사해야 한다.

## seed 42 고정 5폴드 대표 실험 후보

### 이슈 303 정상 실행과 복구 실행

기준 설정은 이슈 303의 `exp113_tab_cnn_m0` 또는 같은 fold-fit 제공자를 가진 대표 설정 하나를 seed 42와 고정 5폴드로 실행한다.
정상 실행에서는 제공자별 fit, train 변환, test 변환, 모델 fit, 검증 예측, test 예측, 중요도 준비, 순열 및 추론, AUC, 복구 저장 시간을 분리한다.
복구 실행에서는 5개 완료 지점을 다시 읽고 비싼 제공자 fit과 transform 호출 횟수가 0인지 확인한다.

비교 순서는 현재 CPU 기준, 복구 선확인, 안전한 fold-fit 결과 재사용, `XgbImputeAux`의 pandas 입력 CUDA XGBoost, 마지막으로 필요할 때만 제공자 내부 cuDF 입력이다.
`cudf.pandas` 전역 실행은 채택 후보가 아니라 한 폴드 분석 자료로만 둔다.

### 실제로 순열 중요도가 느린 GPU 모델

세부 계측에서 중요도 시간이 큰 모델 하나를 선택한다.
같은 NumPy 순열과 같은 중요도 표본을 사용해 현재 pandas 복사 및 재전처리 경로와, 어댑터 내부 인코딩 재사용 및 GPU 묶음 추론 경로를 비교한다.
TabCNN은 이슈 303에서 중요도가 수 초에 그쳤으므로 다른 모델이 실제 병목으로 확인되지 않으면 이 후보를 구현하지 않는다.

### LightGBM CPU 및 내장 중요도

LightGBM 대표 설정을 seed 42와 고정 5폴드로 기존 pandas 입력에서 실행한다.
GPU 선택 의존성이 설치되지 않은 환경에서도 결과가 나오고, 내장 중요도 값과 복구 자료가 기존 기준과 같으며, 공용 인터페이스 때문에 시간이 늘지 않는지 확인한다.

## 후보 우선순위와 제외 조건

| 우선순위 | 후보 | 예상 이득 | 의미 위험 | 판정 |
| --- | --- | --- | --- | --- |
| 1 | 복구 확인을 fold-fit 계산 앞으로 이동 | 완료 폴드의 비싼 계산 전체 생략 | 낮음 | 즉시 설계 후보 |
| 2 | 동일 fold-fit 결과의 내용 기반 재사용 | 반복 실험의 제공자 fit 및 transform 생략 | 정체성 오류 위험 | 엄격한 키와 내용 검증 뒤 후보 |
| 3 | `XgbImputeAux` 내부 `device="cuda"` | 다섯 개 400트리 복원기 가속 가능 | 복원값 변경 가능 | seed 42 5폴드 정확 동일성 관문 |
| 4 | 같은 제공자 내부 cuDF 자료 유지 | 반복 호스트-장치 이동 감소 가능 | dtype, 결측, 범주와 행 순서 위험 | 3번에서 자료 이동이 병목일 때만 후보 |
| 5 | 느린 신경망 어댑터의 텐서 순열 및 묶음 추론 | pandas 복사, 반복 인코딩과 호출 감소 | 전처리 등가성 위험 | 실제 중요도 병목 어댑터에 한정 |
| 6 | CPU 순열 중요도의 열 단위 병렬화 | CPU 모델 중요도 단축 가능 | 과도한 병렬화 위험 | 모델 스레드 수와 함께 측정 |
| 제외 | 전역 `cudf.pandas` 기본 실행 | 넓은 pandas 연산 가속 가능 | 전체 초기화, CPU 전환, 의미와 시험 범위 확산 | 분석 전용 |
| 제외 | cuML `SimpleImputer`로 교체 | GPU 대치 가능 | 현재 다변량 복원 알고리즘 변경 | 범위 밖 |
| 제외 | CuPy를 전체 DataFrame 공용 실행 방식으로 사용 | 배열 산술 가속 가능 | 범주, 인덱스와 결측 계약 부재 | 범위 밖 |

다음 조건 중 하나라도 만족하면 GPU 후보를 제외한다.

- 전체 유료 원격 자원 시간이 CPU 기준보다 줄지 않는다.
- 피처, 예측, 중요도 또는 복구 산출물이 요구한 정확 동일성 관문을 통과하지 못한다.
- 현재 importance 표본 수, 반복 수, 순열 또는 AUC 계산을 바꿔야만 빨라진다.
- CPU LightGBM 경로가 GPU 패키지 설치나 초기화를 요구한다.
- 지원하지 않는 연산이 CPU로 조용히 돌아가 성능 판정이 왜곡된다.
- 원격 이미지의 Python, CUDA, 드라이버 또는 GPU 계산 능력이 공식 지원 범위 밖이다.
- GPU 메모리 부족을 피하려고 피처 정의, 모델 설정이나 복구 단위를 바꿔야 한다.

## 구현 명세로 넘길 결정

첫 구현 명세는 GPU DataFrame 공용 계층이 아니라 다음 세 경계를 중심으로 작성하는 것이 적절하다.

1. 선언된 전체 피처 열과 기존 복구 정체성으로 완료 폴드를 제공자 fit 전에 안전하게 읽는 경계.
2. 제공자 정체성, 입력 해시, seed, fold, 코드와 실제 실행 방식을 포함해 fold-fit 결과를 재사용하는 경계.
3. 제공자와 모델 어댑터가 CPU 구현을 보존하면서 선택 GPU 실행을 내부에 가두고, 실제 실행 방식과 CPU 전환 사유를 보고하는 경계.

cuDF는 3번 경계 안에서 측정 결과가 요구할 때 쓰는 구현 수단이다.
그 자체가 파이프라인 전체 인터페이스가 되어서는 안 된다.
