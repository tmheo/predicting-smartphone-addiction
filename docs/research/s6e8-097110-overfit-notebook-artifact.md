# 지정 노트북의 고정 판본과 실행 재료

이 문서는 GitHub 이슈 [지정 노트북의 고정 판본과 실행 재료를 확정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/247)의 조사 결과다.
조사 시작 시점인 2026-08-19에 [지정 노트북](https://www.kaggle.com/code/raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits)의 최신 공개 판본, 실행 입력, 저장 출력과 실행 환경을 Kaggle 원문 및 공식 API 응답으로 고정했다.

## 결론

지정 노트북은 Kaggle 노트북 ID `129388002`, 공개 판본 번호 `22`, 실행 판본 ID `343453035`로 고정한다.
판본을 다시 열 때는 [실행 판본 고정 보기 주소](https://www.kaggle.com/code/raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits?scriptVersionId=343453035)를 기준으로 삼고, 아래의 입력 판본 ID와 SHA-256을 함께 확인해야 한다.
최신 공개 원본, 저장 실행 기록, `submission.csv`, 실행 결과 HTML과 두 차트까지 확보해 서로 대조했으므로 후속 이슈 248과 249가 코드, 글, 차트 및 실행 수치를 같은 판본에서 검토할 수 있다.

다만 저장소에는 이 조사 문서만 넣는다.
공개 노트북 원본은 Apache License 2.0이지만 Naji 예측 자료의 이용 조건이 `Unknown`이고, 저장 출력과 렌더링 결과의 별도 재배포 조건도 확인되지 않았으므로 입력 원자료와 파생 출력은 재배포하지 않는다.
아래의 고정 보기 주소, Kaggle 판본 ID, 파일 크기와 SHA-256이 재취득 자료를 검증하는 기준이다.

## 노트북 판본

| 항목 | 고정값 |
| --- | --- |
| 소유자와 슬러그 | `raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits` |
| Kaggle 노트북 ID | `129388002` |
| 공개 판본 번호 | `22` |
| 실행 판본 ID와 현재 실행 ID | `343453035` |
| 생성 시각 | `2026-08-19T11:36:54.693Z` |
| 평가 완료 시각 | `2026-08-19T11:36:55.897Z` |
| 실행 상태 | `COMPLETE`, `BATCH`, Python 노트북 |
| 실행 시간과 저장 출력 크기 | `66.9157353`초, `7,783,805`바이트 |
| 가속기와 인터넷 | GPU 없음, TPU 없음, 인터넷 비활성화 |
| 고정 보기 주소 | `?scriptVersionId=343453035`를 붙인 위 주소 |

Kaggle `LegacyKernelsService/GetKernelViewModel` 응답은 현재 실행 ID와 최근 실행 ID를 모두 `343453035`로, 실행 판본 번호를 `22`로 반환했다.
이 응답의 SHA-256은 `bee2edc5f8e9ca9bd17019b66fafa1b927ae3915ae422cf7cd4ffbfbcf961944`다.
예전 `GetKernel` 메타데이터가 돌려준 `lastRunTime`은 `2026-08-16T13:11:09.766Z`로 고정 실행의 생성 시각과 맞지 않았으므로 판본 식별에는 실행 판본을 지정한 보기 모델 응답을 우선한다.

현재 실행에는 Kaggle 제출 ID `55622435`, 원본 실행 판본 ID `343453035`, 공개 점수 `0.97115`인 제출이 연결돼 있다.
반면 노트북 본문과 마지막 실행 기록은 가중치 `-0.08`의 공개 점수 사례를 제출 ID `55584395`로 부른다.
따라서 현재 실행에 연결된 제출 객체와 글에서 분석 대상으로 든 이전 제출을 같은 제출로 취급하면 안 된다.

## 확보한 원본과 저장 출력

| 자료 | 크기 또는 형태 | SHA-256 | 판정 |
| --- | ---: | --- | --- |
| Kaggle CLI가 내려받은 최신 공개 `.ipynb` | 12개 셀, 글 7개와 코드 5개 | `1d584865641a31a101dd6fe24e89ad4c2b67394c2b978791f78f6df26dfd53ac` | 실행 출력이 지워진 공개 원본 |
| CLI `kernel-metadata.json` | JSON | `c8e43ef377822fc0eab11baf4e27c3511d9f0eb6b1aa839ec116c664d31f166c` | 소유자, 슬러그와 입력 선언 확인용 |
| 실행 판본 보기 모델 응답 | JSON | `bee2edc5f8e9ca9bd17019b66fafa1b927ae3915ae422cf7cd4ffbfbcf961944` | 판본, 실행 환경과 입력 판본의 기준 |
| 저장 실행 기록 | 텍스트 | `f780b709ed9c530c5ab351446013cd0c79f4a78c52a93a4165b93c1c4d76efb7` | 실행 수치와 오류 없음 확인 |
| 저장 `submission.csv` | `7,783,805`바이트, 296,302행 | `66781f9298c0b695f68669e18403fc230cfea2a7d9f0d93a3d91bacdd2c6d06d` | 현재 실행이 만든 제출 파일 |
| 저장 결과 HTML | `339,532`바이트 | `361aa9d3b58440b9b8439d7407d4dd920c827498279a8fdc30002afa813387ec` | 글, 표와 그림이 합쳐진 실행 결과 |
| 첫째 차트 PNG | `776 x 439` | `5aece2ecd2b886fe2b48e60d49fef553cccd627da755e4f995e54cf5f2a256e3` | OOF와 공개 순위표 가중치 곡선 |
| 둘째 차트 PNG | `764 x 404` | `2d1b32ad3c8fdeed1535b61c958b507f88c14bbe714543324620456761268725` | Season 6 공개 상위 10팀의 비공개 상위 10 잔존 수 |

공개 원본의 모든 코드 셀은 `execution_count: null`이고 저장 출력이 비어 있으므로 차트와 실행 표를 검토할 때 이 원본만 보면 안 된다.
Kaggle의 실행 판본 포함 보기 API `https://www.kaggle.com/k/embed/raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits?session_id=343453035`에서 새로 발급되는 결과 주소로 저장 결과 HTML을 다시 받을 수 있다.
그 결과 주소는 짧은 유효기간을 가진 서명 주소이므로 문서에는 남기지 않았다.
실행 ID `343453035`가 여전히 현재 실행일 때는 `kaggle kernels output raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits`로 `submission.csv`와 실행 기록을 다시 받을 수 있다.
더 새 판본이 생긴 뒤에는 이 명령이 새 실행을 가리킬 수 있으므로 고정 보기 주소와 위 SHA-256을 우선한다.

최신 판본을 지정하지 않은 `kaggle kernels pull`은 조사 시작 시점에 위 원본을 반환했지만, 익명 상태에서 슬러그 뒤에 `/22`를 붙인 과거 판본 직접 받기는 HTTP 403이었다.
따라서 원본 파일의 동일성은 실행 판본 ID, 공개 판본 번호와 원본 SHA-256을 함께 사용해 판정한다.

## 실행 입력과 이용 조건

실행 판본의 보기 모델 응답은 다음 네 입력 묶음과 정확한 판본 ID를 기록한다.
코드는 이 가운데 대회 `train.csv`와 세 외부 자료의 네 파일만 읽으며, 대회 `test.csv`와 `sample_submission.csv`는 직접 읽지 않는다.

| 입력 묶음 | 실행 판본 식별자 | 실제로 읽는 파일 | 파일 SHA-256 | 별도 이용 조건과 출처 |
| --- | --- | --- | --- | --- |
| [Playground Series S6E8 대회 자료](https://www.kaggle.com/competitions/playground-series-s6e8/data) | 대회 ID `125218`, 자료 묶음 판본 ID `18257486` | `train.csv`, 44,855,546바이트, 691,369행 | `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c` | CC BY 4.0 |
| [Naji OOF 및 제출 자료 v17](https://www.kaggle.com/datasets/najiama/predicting-smartphone-addiction-oof-submission-csv/versions/17) | 자료 ID `11446597`, 자료 판본 ID `18989881`, 판본 `17` | `19_blend_oof_predictions.csv`, 18,051,243바이트, 691,369행 | `8d4caad066e599b0afbb1a84a48af0af063a1c6ccafbdba92a91278f57ed0429` | Kaggle 표시 `Unknown` |
| 같은 Naji 자료 v17 | 같은 식별자 | `19_blend_submission.csv.csv`, 7,783,425바이트, 296,302행 | `476db09857e1e7b452a0b0987ab3ed4d16ac9aed1efec6188c82739892d907be` | Kaggle 표시 `Unknown` |
| [Season 6 공개 및 비공개 순위 자료 v9](https://www.kaggle.com/datasets/georgymamarin/playground-series-s6-leaderboards/versions/9) | 자료 ID `11549766`, 자료 판본 ID `18991764`, 판본 `9` | `s6_leaderboards.csv`, 2,276,213바이트, 26,345행 | `02937c8f5d0bf8266874bb9b42c03de1cda75ef364f018ca457afbecc26835a6` | Apache License 2.0 |
| [전이 교사 및 학생 신호 v2](https://www.kaggle.com/datasets/raykkretzschmar/s6e8-transductive-anti-student-signals/versions/2) | 자료 ID `11649540`, 자료 판본 ID `19045877`, 판본 `2` | `transductive_signals.npz`, 20,397,827바이트 | `16016e8a2d0941a7107f3dc12ecbbc03f53213bc3824043f48a44d7f408e8bbd` | CC0 1.0 |

행 수는 머리글을 제외한 값이다.
프로젝트의 `data/train.csv`도 위 대회 자료의 SHA-256과 691,369행이 일치하므로 후속 재실행의 로컬 기준 자료로 쓸 수 있다.
같은 위치의 `test.csv`는 296,302행이고 SHA-256이 `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e`지만 지정 노트북의 코드가 직접 읽는 입력은 아니다.

Naji 자료 설명은 이 파일이 여러 공개 OOF와 제출을 가중 결합하며 Szymon Kłapiński의 OOF 모음, Omid의 XGBoost, RealMLP과 TabM, Ravi의 L2Stack v1, Adarsh V8 및 Rayk의 결합 노트북을 재료로 삼았다고 밝힌다.
그러나 Kaggle 자료 자체의 이용 조건이 `Unknown`이고 각 예측 파일까지 이어지는 허가 계보가 완결되지 않았으므로 파일을 저장소에 복사하거나 파생 제출을 배포하지 않는다.
노트북이 부르는 `Naji v19`는 `19_blend_*` 파일의 결합 계보 이름이고, 실제 실행에 연결된 Kaggle 자료 판본은 `17`이다.

Season 6 순위 자료는 Kaggle 공개 및 비공개 순위표와 Meta Kaggle을 결합한 파생 자료이며 게시자가 Apache License 2.0을 표시했다.
전이 신호 자료의 `README.md` SHA-256은 `825eeebc585bab5036e40a546fdf70862e53dbb3f4e3600238078300106b5e36`이고, 교사, 학생, 재구성 및 검색 신호를 담되 목표값과 완성 제출은 담지 않는다고 설명한다.
압축 자료에는 배열 10개가 있지만 지정 노트북은 `oof_soft_student`와 `test_soft_student`만 사용한다.

대회 자료 페이지는 [Smartphone Addiction Prediction Dataset](https://www.kaggle.com/datasets/algozee/smartphone-addiction-prediction-data)을 생성 영감의 출처로 가리킨다.
조사 시점에는 그 원자료의 메타데이터 API가 403 또는 404를 반환해 별도 판본과 이용 조건을 확인하지 못했다.
이 원자료는 지정 노트북의 직접 실행 입력이 아니며, 대회가 배포한 고정 자료 묶음 자체의 표시는 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)이다.

## 코드, 패키지와 외부 자산

원본은 `glob`과 `pathlib`을 제외하면 Matplotlib, NumPy, pandas, SciPy와 scikit-learn만 불러온다.
실행 중 패키지를 설치하거나 원격 주소에서 코드를 받는 셀은 없고, 다른 Kaggle 노트북이나 모형을 입력으로 연결하지 않았다.
재귀 파일 검색 함수가 파일 이름과 맞는 첫 경로를 고르므로 같은 이름의 파일이 여러 입력 디렉터리에 있으면 다른 파일을 읽을 수 있다.
후속 재실행에서는 각 파일을 격리한 입력 디렉터리에 두거나 해시를 읽기 직전에 검증해야 한다.

| 구성 요소 | 실행에서의 역할 | 확인한 상위 이용 조건 |
| --- | --- | --- |
| Python 표준 라이브러리 `glob`, `pathlib` | 입력 경로 검색 | [Python Software Foundation License](https://github.com/python/cpython/blob/main/LICENSE) |
| Matplotlib | 두 차트 작성 | [Matplotlib License Agreement](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| NumPy | 배열, 압축 배열과 수치 계산 | [BSD 3-Clause 계열](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pandas | CSV 읽기, 표와 제출 작성 | [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| SciPy | 순위와 스피어만 상관 계산 | [BSD 3-Clause 계열](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| scikit-learn | AUC와 층화 무작위 분할 | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |

원본 메타데이터는 Python `3.11`을 적지만 저장 실행 기록의 라이브러리 경로는 Python `3.12`를 가리킨다.
실제 실행 환경은 `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`와 이미지 ID `sha256:baf69fed26af6330ecb1c477f3aa0470f5d83a390ab86b63ac3239d5c153b624`로 고정한다.
Kaggle Docker 이미지 판본 ID는 `31429`이고 고정 방식은 원래 이미지 유지 방식이다.
[Kaggle Docker Python 저장소](https://github.com/Kaggle/docker-python)는 빌드 정의를 Apache License 2.0으로 공개하지만 이미지에 들어간 각 패키지의 이용 조건까지 대신하지는 않는다.
노트북과 실행 기록은 패키지별 정확한 판본을 출력하지 않으므로, 바이트 단위로 같은 환경이 필요하면 위 이미지 다이제스트를 사용해야 한다.

두 PNG는 외부에서 가져온 그림이 아니라 Matplotlib 코드가 실행 입력에서 만든 결과다.
그림에 별도의 이미지 이용 조건이 표시되지 않았고 렌더링 결과 전체의 재배포 허가도 확인되지 않았으므로 이 조사에서는 보존용 해시만 기록했다.

## 실행 증거 대조

저장 실행 기록은 학습 691,369행, 시험 296,302행, 양성 비율 `0.709`를 출력한다.
OOF 최적 가중치는 `+0.12`이고 변화량은 `+0.00005267`이지만, 제출해 본 가중치 가운데 공개 순위표가 고른 최적값은 `-0.08`이고 해당 OOF 변화량은 `-0.00009056`이다.
전체 OOF 신호를 사용한 모의 최적 가중치는 `+0.10`이다.
공개 부분으로 선택했을 때의 평균 겉보기 이득은 `+0.00000472`이고 선택에 쓰지 않은 부분의 평균 변화는 `-0.00000633`이며, 10회 중 8회에서 선택 가중치가 달라졌다.
마지막 셀은 296,302행의 `submission.csv`를 쓰고 이전 제출 ID `55584395`의 알려진 공개 점수를 `0.97115`로 출력한다.
이 값들은 후속 코드 및 글 검토가 같은 저장 실행을 읽었는지 확인하는 빠른 지문이다.

## 후속 이슈의 재현 절차

1. 고정 보기 주소에서 실행 판본 ID `343453035`와 공개 판본 `22`를 확인한다.
2. [Kaggle 공식 CLI](https://github.com/Kaggle/kaggle-cli)로 소유자와 슬러그의 원본과 현재 저장 출력을 내려받은 뒤 위 SHA-256과 대조한다.
3. 대회 자료 묶음 판본 `18257486`과 외부 자료 판본 `18989881`, `18991764`, `19045877`을 각각 받는다.
4. 코드가 실제로 읽는 다섯 파일의 크기와 SHA-256을 대조하고 같은 이름의 파일이 검색 경로에 둘 이상 없는지 확인한다.
5. 실행 결과의 글, 표와 차트는 출력이 지워진 원본이 아니라 실행 판본의 저장 결과 HTML에서 검토한다.
6. 로컬에서 코드를 다시 돌릴 때는 가능한 한 고정 Docker 이미지 다이제스트를 사용하고, 다른 환경이면 Python 및 다섯 패키지의 실제 판본을 별도로 기록한다.

이 묶음은 이슈 248의 코드 및 차트 대조와 이슈 249의 글 및 주장 검토에 충분하다.
패키지별 판본이 원본에 적혀 있지 않은 점은 정확한 이미지 다이제스트로 보완할 수 있고, Naji 자료의 이용 조건이 불명확한 점은 검토는 막지 않지만 저장소 재배포는 막는다.

## 이용 조건 판정

[Meta Kaggle Code 설명](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)과 저장소의 공개 노트북 절차에 따라 공개된 지정 노트북 원본은 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)으로 다룬다.
원본을 수정하거나 재사용한다면 저작권과 이용 조건 고지, 변경 사실 및 출처 주소를 함께 남겨야 한다.
대회 자료는 CC BY 4.0, Season 6 순위 자료는 Apache License 2.0, 전이 신호 자료는 CC0 1.0으로 각각 별도 관리한다.
Naji 자료와 저장 출력 및 렌더링 자산은 허가가 명확해질 때까지 재배포하지 않는다.

## 남은 한계

익명 API로 공개 판본 `/22`의 원본을 직접 지정해 받지 못했지만, 조사 시작 시점의 최신 공개 원본, 실행 판본 ID, 공개 판본 번호와 SHA-256을 함께 고정했으므로 후속 검토의 대상 판본은 모호하지 않다.
패키지별 정확한 판본은 출력에 없으나 실행 컨테이너 다이제스트가 남아 있다.
생성 영감 원자료의 판본과 이용 조건은 확인하지 못했지만 직접 실행 입력이 아니므로 코드, 차트와 글을 검토하는 데 차단 요인은 아니다.
