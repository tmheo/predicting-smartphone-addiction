# 후보 원격 GPU 환경의 재현성과 결과 보존 방식

## 결론

현재 실험 프로그램을 가장 적게 바꾸면서 재현성과 결과 보존을 함께 확보할 수 있는 후보는 Runpod Pod와 Vast.ai 일반 인스턴스다.
Runpod는 Pod 생성 API에서 `Tesla T4`, `allowedCudaVersions: ["13.0"]`, 사용자 지정 Docker 이미지, 중단형 여부를 명시할 수 있고, Pod와 독립된 네트워크 저장 공간도 제공하므로 요구 조건을 가장 직접적으로 표현한다.
Vast.ai는 제안 검색에서 GPU 종류, `cuda_max_good`, NVIDIA 드라이버 버전, 검증된 공급자와 신뢰도를 걸러낸 뒤 사용자 지정 Docker 이미지와 SSH 실행을 쓸 수 있으므로 기술적으로 충족한다.
다만 Vast.ai의 영구 볼륨은 특정 물리 장비에 묶이므로 최종 실행 기록 묶음은 반드시 로컬이나 별도 객체 저장소로 한 번 더 복사해야 한다.
Lambda On-Demand Cloud와 Paperspace Machines도 SSH, Docker, 영구 저장 공간을 갖춰 실행할 수 있지만, 생성 요청에서 CUDA 13 호환 드라이버를 고르는 공식 조건을 찾지 못했으므로 실제 장비를 받은 직후 드라이버 580 이상인지 검사해야 한다.
Colab Pro는 현재 Python 3.12 환경과 약 6시간짜리 단일 실행에는 맞을 수 있지만, GPU 종류와 사용 한도를 보장하지 않고 가상 장비와 로컬 파일을 지우므로 수동 예비 경로로만 적합하다.
브라우저를 닫은 뒤에도 실행을 이어가는 기능은 Pro+에만 명시되어 있지만, Pro+도 GPU 종류와 가상 장비 수명을 보장하지 않는다 ([Colab 유료 구독 안내](https://colab.research.google.com/notebooks/pro.ipynb)).

기술 적합도 순서는 `Runpod > Vast.ai > Paperspace Machines > Lambda On-Demand Cloud > Colab Pro 또는 Pro+`다.
가격과 실제 재고를 반영한 최종 선택은 별도의 비용 및 가용성 조사와 함께 내려야 한다.

## 저장소가 요구하는 실행 계약

프로젝트는 Python 3.12 이상을 요구하고 `uv sync --frozen`으로 잠긴 환경을 설치하는 실행 경로를 이미 사용한다 ([pyproject.toml](../../pyproject.toml), [Kaggle GPU 실행 절차](../kaggle-gpu-run.md)).
현재 `uv.lock`은 Linux x86-64에서 PyTorch 2.13.0과 CUDA 13 계열의 실행 라이브러리를 설치한다 ([uv.lock](../../uv.lock)).
CUDA 실행 라이브러리가 Python 환경에 들어와도 호스트 NVIDIA 드라이버는 별도 조건이며, NVIDIA는 CUDA 13.x의 최소 드라이버를 580으로 규정한다 ([NVIDIA CUDA 호환성 표](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)).
따라서 모든 후보에서 `uv sync --frozen` 전에 `nvidia-smi`로 GPU 이름과 드라이버 버전을 확인하고, 드라이버가 580 미만이면 그 장비를 사용하지 않아야 한다.

현재 GPU 실행 절차는 커밋을 고정해 저장소를 복제하고 같은 `pipeline.run` 명령을 호출한 뒤, 완료된 MLflow 실행을 검증 가능한 ZIP 묶음으로 내보낸다 ([Kaggle GPU 실행 절차](../kaggle-gpu-run.md)).
묶음 반입은 자료와 fold의 해시, 실행 커밋과 설정 파일, `git_dirty=False`, 시드별 OOF 재채점을 검증하므로 원격 서비스가 바뀌어도 결과의 출처와 내용은 같은 규약으로 확인된다 ([묶음 구현](../../src/pipeline/bundle.py)).
2026-08-14 로컬 작업 공간에서 대회 자료는 167MiB였고, 실제 `exp059`와 `exp060` 실행 기록 묶음은 각각 약 15MiB였다.
이 크기는 모든 후보의 기본 디스크와 파일 전송 기능으로 처리할 수 있으므로 저장 용량보다 중단 시 데이터가 남는 위치가 더 중요한 조건이다.

현재 파이프라인은 모든 시드와 fold 학습이 끝난 뒤에만 OOF, 테스트 예측, 중요도와 제출 파일을 최종 산출물로 기록한다 ([실행 진입점](../../src/pipeline/run.py), [실험 기록 구현](../../src/pipeline/tracking.py)).
fold 완료 지표와 실행 로그는 진행 상황을 알려주지만 학습 모델 상태를 저장하지 않으며, 저장소에는 체크포인트를 읽어 이어가는 경로가 없다 ([실행 관찰 구현](../../src/pipeline/observe.py)).
따라서 영구 저장 공간은 중단 직전 로그와 MLflow 상태를 보존할 수는 있어도 중단된 학습을 이어주지 못한다.
실행 도중 장비가 회수되면 현재 프로그램으로는 해당 실험을 처음부터 다시 실행해야 한다.

동일 시드만으로 서로 다른 GPU에서 비트 단위로 같은 결과가 보장되지는 않는다.
저장소가 검증한 시드 병렬 동등성도 같은 GPU 모델을 전제로 한다 ([시드 병렬 실행 설명](../kaggle-gpu-run.md)).
개선 기준이 작으므로 확정 재검증의 기준 실행과 후보 실행은 같은 서비스의 같은 GPU 모델에서 함께 실행하는 것이 안전하다.
기존 Kaggle T4 결과와 장비 차이를 최소화하려면 Runpod 또는 Vast.ai에서 T4를 명시하는 구성이 가장 알맞다.

## 후보별 판정

| 후보 | 환경 고정 | GPU와 CUDA 13 | SSH 또는 무인 실행 | 결과가 남는 위치 | 판정 |
| --- | --- | --- | --- | --- | --- |
| Runpod Pod | 사용자 지정 Docker 이미지와 API 설정 가능 | T4 선택 및 CUDA 13.0 허용 조건 명시 가능 | SSH, 명령줄 도구, API 지원 | 네트워크 볼륨은 Pod 삭제 뒤에도 존속 | 가장 적합 |
| Vast.ai | 사용자 지정 Docker 이미지와 템플릿 가능 | T4, CUDA 상한, 드라이버 버전 검색 가능 | SSH, 시작 명령, API 및 명령줄 도구 지원 | 볼륨은 인스턴스 삭제 뒤에도 남지만 같은 물리 장비에 묶임 | 적합한 보조 경로 |
| Paperspace Machines | Linux 장비, ML-in-a-Box, Docker 및 사용자 지정 템플릿 가능 | A4000 이상 선택 가능, 드라이버는 생성 뒤 검사 필요 | SSH, API 및 명령줄 도구 지원 | 장비 디스크가 정지 뒤에도 존속 | 조건부 적합 |
| Lambda On-Demand Cloud | Linux 가상 장비, Docker와 NVIDIA Container Toolkit 기본 제공 | RTX 6000 이상 선택 가능, 드라이버는 생성 뒤 검사 필요 | SSH, JupyterLab 및 Cloud API 지원 | 별도 파일 시스템은 장비 종료 뒤에도 존속 | 조건부 적합 |
| Colab Pro 또는 Pro+ | 노트북 셀에서 매번 잠긴 환경 재설치 | GPU와 드라이버를 지정하거나 보장할 수 없음 | 소비자용 관리형 실행 환경은 공식 무인 실행 수단이 약함 | 가상 장비 로컬 파일은 삭제되며 Drive로 옮긴 파일만 존속 | 수동 예비 경로 |

### Runpod Pod

Runpod의 Pod 생성 API는 `gpuTypeIds`에 `Tesla T4`, `allowedCudaVersions`에 `13.0`, `imageName`에 Docker 이미지 태그를 지정할 수 있다 ([Pod 생성 API](https://docs.runpod.io/api-reference/pods/POST/pods)).
같은 API에서 `interruptible: false`를 지정할 수 있으므로 가격이 낮은 대신 언제든 회수될 수 있는 중단형 Pod를 피할 수 있다.
Runpod는 모든 Pod에 기본 SSH를 제공하고, 공식 PyTorch 실행 이미지나 포트 22를 연 사용자 지정 이미지에는 SCP와 SFTP가 가능한 직접 SSH도 제공한다 ([Pod SSH 문서](https://docs.runpod.io/pods/configuration/use-ssh)).
명령줄 도구로 Pod 생성, 시작, 정지와 삭제를 다룰 수 있으므로 커밋 고정 복제부터 실행 기록 묶음 회수까지 무인화할 수 있다 ([Pod 관리 문서](https://docs.runpod.io/pods/manage-pods)).

Runpod의 컨테이너 디스크는 정지나 재시작 때 사라진다.
기본 볼륨 디스크의 `/workspace`는 Pod 정지와 재시작 뒤에도 남지만 Pod 삭제 때 사라진다.
네트워크 볼륨은 Pod와 독립적으로 남고 다른 Pod에 붙일 수 있다 ([Runpod 저장 방식](https://docs.runpod.io/pods/storage/types)).
따라서 저장소, 대회 자료, `mlflow.db`, MLflow 산출물과 `run-logs`를 네트워크 볼륨의 `/workspace` 아래에 두는 구성이 가장 안전하다.
정상 완료 뒤에는 `pipeline.bundle export`로 만든 약 15MiB ZIP을 SCP나 Cloud Sync로 로컬 또는 별도 저장소에 복사한 다음 Pod를 삭제해야 한다.

권장 생성 조건은 Secure Cloud, 일반 요금, `interruptible=false`, GPU `Tesla T4`, GPU 수 1, `allowedCudaVersions=["13.0"]`, 네트워크 볼륨 연결이다.
Pod가 준비되면 `nvidia-smi`에서 드라이버 580 이상을 다시 확인해야 한다.

### Vast.ai

Vast.ai의 제안 검색 응답은 GPU 이름, 최대 지원 CUDA 버전인 `cuda_max_good`, NVIDIA `driver_version`, 장비 신뢰도와 데이터센터 검증 여부를 제공한다 ([제안 검색 API](https://docs.vast.ai/api-reference/search/search-offers)).
그러므로 T4, `cuda_max_good >= 13.0`, `driver_version >= 580`, 검증된 데이터센터와 높은 신뢰도를 동시에 요구할 수 있다.
Vast.ai 인스턴스는 Docker 컨테이너이며 생성 요청에 Docker 이미지, SSH 또는 Jupyter 실행 방식, 시작 명령과 볼륨을 지정할 수 있다 ([인스턴스 생성 문서](https://docs.vast.ai/api-reference/creating-instances-with-api)).
중단형 인스턴스는 다른 사용자가 더 높은 가격을 제시하면 정지될 수 있으므로 확정 재검증에는 일반 인스턴스를 사용해야 한다.

인스턴스를 정지하면 컨테이너 자료가 남고 저장 비용은 계속 부과되지만, 다시 시작할 때 같은 GPU를 확보하지 못할 수 있다 ([인스턴스 관리 문서](https://docs.vast.ai/guides/instances/manage-instances)).
인스턴스를 삭제하면 컨테이너 저장 공간은 함께 삭제된다.
Vast.ai 볼륨은 인스턴스 삭제 뒤에도 남고 새 인스턴스에 다시 붙일 수 있지만, 생성된 물리 장비에서 다른 장비로 옮길 수 없다 ([Vast.ai 저장 방식](https://docs.vast.ai/guides/instances/storage/types)).
공급 장비가 오프라인이 되면 같은 장비의 볼륨에도 접근하지 못할 수 있으므로 완료된 실행 기록 묶음은 즉시 SCP나 Cloud Sync로 장비 밖에 복사해야 한다.

권장 검색 조건은 일반 요금, `verified=true`, 높은 `reliability`, `gpu_name=Tesla T4`, `cuda_max_good>=13.0`, `driver_version>=580`이다.
최종 확정 실행보다 짧은 사전 실행으로 실제 설치와 결과 회수를 먼저 검증하는 편이 안전하다.

### Paperspace Machines

Paperspace Machines는 영구 디스크가 붙은 Linux 가상 장비이며 SSH, API와 명령줄 도구로 생성하고 접근할 수 있다 ([Machines 개요](https://docs.digitalocean.com/products/paperspace/machines/), [장비 생성 문서](https://docs.digitalocean.com/products/paperspace/machines/how-to/create/), [SSH 문서](https://docs.digitalocean.com/products/paperspace/machines/how-to/connect/)).
공식 GPU 목록에는 Turing RTX 4000 및 RTX 5000과 Ampere A4000 이상의 장비가 있고, 새 실행에는 VRAM과 성능 여유가 있는 A4000 이상이 알맞다 ([장비 종류](https://docs.digitalocean.com/products/paperspace/machines/details/machine-types/)).
ML-in-a-Box 실행 이미지는 PyTorch, CUDA, cuDNN과 NVIDIA Docker를 포함하며 추가 도구를 설치할 수 있다 ([ML-in-a-Box 문서](https://docs.digitalocean.com/products/paperspace/machines/getting-started/run-ml-in-a-box/)).
장비를 끄면 연산 요금이 멈추고 장비와 디스크는 계정에 남는다 ([Paperspace 가격 문서](https://docs.digitalocean.com/products/paperspace/pricing/)).

공식 장비 생성 조건에는 CUDA 13 호환 드라이버 선택 항목이 없으므로 장비를 받은 뒤 드라이버 580 이상인지 검사해야 한다.
SSH 연결은 Paperspace 콘솔이 활동으로 감지하지 않아 자동 종료가 예기치 않게 일어날 수 있으므로, 약 6시간 실행 전에 콘솔에서 자동 종료를 꺼야 한다 ([Machines 제한 사항](https://docs.digitalocean.com/products/paperspace/machines/details/limits/)).
장비의 영구 루트 디스크에 저장소와 실행 기록을 두고, 정상 완료 뒤 실행 기록 묶음을 SCP로 로컬에 내려받으면 된다.

### Lambda On-Demand Cloud

Lambda On-Demand Cloud는 Linux 기반 GPU 가상 장비를 제공하며 RTX 6000, A10, A6000, A100 이상의 GPU를 선택할 수 있다 ([On-Demand Cloud 개요](https://docs.lambda.ai/public-cloud/on-demand/)).
`lambda-stack-24-04`와 `gpu-base-24-04` 실행 이미지는 Python 3.12를 포함하므로 프로젝트의 Python 하한을 만족한다.
장비에는 Docker와 NVIDIA Container Toolkit이 기본 설치되어 사용자 지정 Docker 이미지에 GPU를 연결할 수 있다 ([시스템 환경 관리](https://docs.lambda.ai/public-cloud/on-demand/managing-system-environment/)).
SSH와 JupyterLab을 지원하고 Cloud API로 장비 수명주기를 다룰 수 있으므로 무인 실행 경로를 만들 수 있다 ([장비 연결 문서](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/), [Cloud API](https://docs.lambda.ai/public-cloud/cloud-api/)).

Lambda 장비는 정지 또는 일시 중지가 불가능하고 실행, 재시작, 종료만 가능하다 ([장비 관리 문서](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/)).
장비 종료 화면은 인스턴스 자료를 지우는 작업임을 확인하도록 요구하며, 종료 뒤에도 남겨야 하는 자료는 장비 시작 전에 별도 영구 파일 시스템을 연결해야 한다 ([Lambda Cloud 콘솔 문서](https://docs.lambda.ai/public-cloud/console/), [On-Demand Cloud 저장 방식](https://docs.lambda.ai/public-cloud/on-demand/#storage)).
따라서 저장소와 대회 자료, 실행 기록을 `/lambda/nfs/<파일시스템 이름>` 아래에 두고, 완료된 묶음을 로컬에도 복사해야 한다.
공식 생성 조건에는 CUDA 13 드라이버 선택 항목이 없으므로 장비를 받은 직후 드라이버 580 이상을 검사해야 한다.

### Google Colab Pro와 Pro+

Colab의 2026.04 실행 환경은 Python 3.12.13이므로 프로젝트의 Python 하한을 만족하고, 필요한 라이브러리는 노트북 셀에서 다시 설치할 수 있다 ([Colab 실행 환경 버전](https://research.google.com/colaboratory/runtime-version-faq.html)).
그러나 Colab은 GPU 종류, 사용 한도, 유휴 종료 시간과 최대 가상 장비 수명이 변하며 이를 보장하거나 공개하지 않는다고 명시한다 ([Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html#resource-limits)).
따라서 T4 이상과 드라이버 580 이상을 실행 전에 검사해 조건이 맞지 않으면 연결을 끊고 다른 실행 환경을 받아야 한다.
일반적으로 노트북은 최대 12시간까지 실행되며 Pro, Pro+와 사용량 결제도 연산 단위를 모두 쓰면 실행 장비가 종료될 수 있다.
Pro+만 충분한 연산 단위가 있을 때 최대 24시간 연속 실행을 명시한다 ([Colab FAQ의 실행 시간 설명](https://research.google.com/colaboratory/intl/en-GB/faq.html#resource-limits)).

Colab 노트북은 Google Drive에 남지만 실행 가상 장비와 그 안의 사용자 파일 및 설치한 라이브러리는 공유되거나 보존되지 않으며, 가상 장비는 유휴 상태와 최대 수명에 따라 삭제된다 ([Colab FAQ의 실행 상태 설명](https://research.google.com/colaboratory/intl/en-GB/faq.html)).
Google Drive를 파일 시스템에 연결할 수 있지만 Google은 많은 작은 입출력을 피하고 압축 파일을 로컬 가상 장비에 풀어 쓰라고 권고하며, 중단된 Drive 파일 이동은 자료를 잃을 수 있다고 경고한다.
따라서 대회 자료와 학습은 `/content`에서 수행하고, 정상 완료 직후 하나의 실행 기록 묶음을 Drive로 복사하는 방식이 알맞다.
가상 장비가 묶음 생성 전에 종료되면 현재 파이프라인에는 이어서 학습할 체크포인트가 없으므로 실행 결과를 회수할 수 없다.

Colab Pro는 짧은 스크리닝을 사람이 지켜보며 실행하는 예비 경로로는 쓸 수 있다.
약 6시간의 확정 재검증과 브라우저를 닫은 무인 실행이 목적이면 Pro+가 더 맞지만, 동일 GPU 재현성과 실행 보장은 Runpod나 Vast.ai보다 약하다.

## 공통 사전 검사

원격 장비에 결제 시간을 쓰기 전에 다음 검사를 모두 통과시켜야 한다.

```bash
python3 --version
git --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv --version
uv sync --frozen
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
uv run python -m pipeline.run configs/exp059_lookup_transformer.yaml --stage screen --plan
```

Python은 3.12 이상이어야 하고 NVIDIA 드라이버는 580 이상이어야 한다.
`torch.version.cuda`는 13 계열이어야 하며 `torch.cuda.is_available()`이 참이어야 한다.
계획 출력의 `git_dirty`는 `False`여야 하고 자료 및 fold 해시가 로컬 판정 환경과 같아야 한다.
이 검사를 한 번 통과한 Docker 이미지 태그 또는 가상 장비 템플릿을 고정해 이후 실행에서 재사용해야 한다.

## 결과 보존 규칙

1. 저장소를 특정 커밋으로 복제하고 해당 커밋을 checkout한 뒤 `uv sync --frozen`을 실행한다.
2. 대회 자료, 저장소, `mlflow.db`, MLflow 산출물과 `run-logs`를 서비스의 영구 저장 경로에 둔다.
3. Runpod와 Vast.ai에서는 중단형 장비를 쓰지 않고, Paperspace에서는 자동 종료를 끈다.
4. 기준 실행과 후보 실행은 같은 GPU 모델에서 실행한다.
5. 정상 완료 즉시 `pipeline.bundle export`로 ZIP 묶음을 만든다.
6. ZIP 묶음을 로컬 또는 Pod와 독립된 별도 저장소로 복사한 뒤 원격 장비를 삭제한다.
7. 로컬에서 `pipeline.bundle import`를 실행해 해시, 출처와 재채점 검증을 통과한 뒤에만 결과를 채택 판단에 사용한다.

서비스의 영구 저장 공간은 결과 유실을 줄이지만 현재 프로그램의 중단 복구 수단은 아니다.
긴 실행의 비용 손실까지 막으려면 시드 또는 fold 단위 체크포인트와 재개 기능을 별도 결정 항목으로 다뤄야 한다.
