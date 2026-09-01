# Kaggle 원격 실행 절차 (#98)

이 문서는 Kaggle CPU의 정식 개선 판정 실행과 과거 Kaggle GPU 실행의 재현, 사용자가 명시적으로 요청한 GPU 호환성 진단에 사용하는 절차다.
[Vast.ai 중심 원격 실험 운영 전환](https://github.com/tmheo/predicting-smartphone-addiction/issues/123) 이후 S6E8 외부 GPU 실행의 주 실행 환경은 Vast.ai이고 예비 실행 환경은 Runpod이다.
GPU가 필요하다는 이유만으로 이 경로를 자동 선택하지 않는다.
GPU를 쓰지 않는 실행은 kagglekit 형제 체크아웃의 `docs/agents/vast-resource-control.md`의 Kaggle CPU 무결성 관문을 모두 통과하면 정식 개선 판정에 포함할 수 있다.
현재 공급자 선택, 짝비교 배치와 전환 규칙은 그 문서를 따른다.

Kaggle에서 실행한 결과를 실행 기록 묶음으로 로컬에 반입하는 기술 절차는 아래에 보존한다.
용어는 CONTEXT.md의 "실행 기록 묶음"과 "묶음 반입"을 따른다.

## 원칙

- Kaggle에서도 `pipeline.run <config> --stage screen|confirm`을 그대로 실행한다.
  노트북 자체 학습 루프는 만들지 않는다(두 번째 CV 루프 divergence 금지).
- config는 실행 전에 main에 커밋돼 있어야 한다.
  반입이 "실행의 git_commit에 그 config가 그대로 존재하는가"를 검증한다.
- 단계(스크리닝·확정 재검증)는 config가 아니라 `--stage`가 정한다(#103).
  같은 커밋의 같은 config로 두 단계를 모두 실행할 수 있으므로,
  스크리닝 통과 후 승격에 config 편집·재커밋·재클론이 필요 없다.
- 커널은 인터넷을 켠 상태로 실행한다(GPU 실험 커널은 허용, 제출 노트북 아님).
- 자동으로 돌아온 결과도 반입 게이트(입력 해시·출처·재채점)가 검증하므로,
  실행 경로가 자동화됐다고 신뢰를 더 얹지 않는다.
- 정식 CPU 짝비교는 대조군과 후보군을 모두 Kaggle CPU에 배정한다.
  한 팔만 완주했거나 두 팔의 실행 이미지와 주요 의존성 판본이 다르면 그 짝을 판정에 사용하지 않는다.
- CPU 실행은 `REMOTE_RUN_PROVIDER=kaggle`을 설정하고 실행 이미지, Python과 주요 의존성 판본을 실행 기록 묶음에 남긴다.
- Kaggle Public 점수와 제출 결과는 정식 개선 판정에 사용하지 않는다.

## 자동 실행 (kaggle CLI, 권장)

kaggle CLI의 kernels 명령으로 사람 개입 없이 전체 루프를 돌린다.
push(업로드 + 즉시 배치 실행), status(폴링), logs(실행 로그), output(산출물 다운로드)을 쓴다.

전제: API 토큰 설정, 계정 전화번호 인증, 대회 규칙 동의.
실측 CPU 동시 실행 한도는 5개였고 배치 실행 상한은 12시간이다.
GPU 주간 할당은 30시간, 배치 실행 상한은 GPU 약 9시간이다.

1. 커널 폴더를 만든다: `kaggle/exp0NN/`(템플릿)을 `kaggle/expNNN/`으로 복사하고
   `run.py`의 `COMMIT`·`CONFIG`·`STAGE`·`BUNDLE`, `kernel-metadata.json`의 `id`·`title`만 바꾼다.
   내용물은 `kernel-metadata.json`과 `run.py`(노트북일 필요 없이 plain 스크립트) 두 파일이다.

   템플릿 스크립트는 커밋 고정 클론 → `uv sync --frozen` → 대회 자료 심볼릭 링크 →
   `pipeline.run` → `pipeline.bundle export` 순서로 돌며, 다음 세 가지를 지킨다(#99).

   - repo는 `/kaggle/working` 밖(`/tmp/repo`)에 클론한다.
     working 아래에 두면 `.venv`까지 커널 산출물 다운로드에 딸려 온다(#58에서 겪은 문제).
     bundle zip만 `/kaggle/working`에 남긴다.
   - `pipeline.run` 출력은 capture하지 않고 tee 방식으로 실시간 커널 로그에 흘린다.
     실행 중에도 `kaggle kernels logs`로 epoch 진행을 볼 수 있고, `run_id=` 줄 파싱은 유지된다.
   - `nvidia-smi -L`로 GPU 수를 세어 여럿이면 `PIPELINE_SEED_GPUS`(예: `0,1`)를 설정한다.
     그러면 `pipeline.run`이 시드 단위 워커 프로세스로 GPU를 나눠 쓴다(아래 참조).

   저장소가 비공개면 커널의 익명 git clone이 실패한다. 실행 전 저장소가 public인지
   확인한다(#58에서 public 전환). 추가 자료가 필요한 실험은 해당 파일을 담은 Kaggle
   데이터셋을 `dataset_sources`로 연결하고 저장소 경로에 심볼릭 링크한다
   (예: `kaggle/exp060`의 원본 프록시).

2. CPU 실행은 가속기 인수 없이 밀어 넣는다.

   ```bash
   uv run kaggle kernels push -p kaggle/expNNN
   ```

   GPU 진단 실행에서 `--accelerator`의 유효값은 CLI enum 문자열 `NvidiaTeslaT4`·
   `NvidiaTeslaP100`이다(#61). 잘못된 문자열은 오류 없이 무시되고 기본 P100이 배정되는데,
   torch 2.13 wheel은 sm_75 미만을 지원하지 않아 P100에서는 CUDA 커널 이미지 오류로
   즉시 죽는다. 항상 T4를 쓴다. T4는 2개가 붙으므로 시드가 여럿이면 시드 병렬이 켜진다.

   ```bash
   uv run kaggle kernels push -p kaggle/expNNN --accelerator NvidiaTeslaT4
   ```

3. 완료를 폴링하고 로그를 확인한다.

   ```bash
   uv run kaggle kernels status tmheo/exp0NN-gpu-run   # complete/error까지 반복
   uv run kaggle kernels logs tmheo/exp0NN-gpu-run
   ```

4. 묶음을 내려받아 반입한다.

   ```bash
   uv run kaggle kernels output tmheo/exp0NN-gpu-run -p run-logs/kaggle-out
   uv run python -m pipeline.bundle import run-logs/kaggle-out/exp0NN.bundle.zip
   ```

push → status 폴링 → output → import 네 단계를 감싸는 드라이버 스크립트를 만들면
"config 커밋 후 명령 한 번"이 된다.

### 시드 병렬 실행 (T4 x2, #99)

`PIPELINE_SEED_GPUS`(쉼표로 구분한 GPU 번호, 예: `0,1`)가 설정돼 있고 시드가 여럿이면
`pipeline.run`이 시드 단위로 워커 프로세스를 띄워 GPU를 나눠 쓴다(`pipeline/seed_parallel.py`).
환경 변수가 없으면(로컬, P100) 기존 순차 실행 그대로다.

- 재현성: 모든 adapter가 fold 학습 시작 때 자기 시드로 전역 RNG를 다시 심으므로
  시드 간 실행 순서로 상태가 흐르지 않고, 같은 GPU 모델(T4 x2)이면 병렬 결과가
  순차 실행과 동일하다. 순차·병렬 동등성은 `tests/test_seed_parallel.py`가 검증한다.
- 워커는 시작 직후 `CUDA_VISIBLE_DEVICES`로 GPU 하나를 배정받으므로
  adapter의 `device="cuda"`가 그대로 그 GPU로 해석된다. adapter 수정은 없다.
- 진행 기록: 병렬 경로는 시드별 단계가 겹치므로 `training` 단계 하나로 묶이고
  (`time.feature_build/training_seconds`의 시드별 step 기록 없음),
  fold 완료 통지(`progress.*`)는 워커 큐로 받아 그대로 기록된다.
- 3시드 x T4 2개면 시드 2개 분량의 벽시계 시간이 들어, 실험당 약 1/3이 준다
  (#58 기준 5.5~6시간 → 약 4시간). GPU 사용량(주 30시간 할당) 절약은 아니다.

## 수동 실행 (노트북, 대안)

웹 UI에서 노트북으로 실행할 때의 셀 구성이다. 원칙은 자동 실행과 같다.

1. 저장소를 커밋 고정으로 받는다.

   ```bash
   !git clone https://github.com/tmheo/predicting-smartphone-addiction.git repo
   %cd repo
   !git checkout <commit>   # 실행할 config가 포함된 커밋
   !pip install -e . -q
   ```

2. 대회 자료를 저장소가 기대하는 상대 경로에 연결한다.
   folds는 저장소에 커밋돼 있으므로 그대로 쓴다.

   ```bash
   !ln -sf /kaggle/input/playground-series-s6e8/train.csv data/train.csv
   !ln -sf /kaggle/input/playground-series-s6e8/test.csv data/test.csv
   !ln -sf /kaggle/input/playground-series-s6e8/sample_submission.csv data/sample_submission.csv
   ```

3. 실험을 실행한다. MLflow sqlite는 저장소 루트(mlflow.db)에 정상 생성된다.
   stdout의 `run_id=`를 확보한다.

   ```bash
   !python -m pipeline.run configs/expNNN_*.yaml --stage <screen|confirm>
   ```

4. 마지막 셀에서 묶음을 export하고 노트북 산출물에서 zip을 내려받는다.

   ```bash
   !python -m pipeline.bundle export <run_id> --out /kaggle/working/expNNN.bundle.zip
   ```

5. 로컬에서 반입한다: `uv run python -m pipeline.bundle import <zip>`.

## 반입 이후

반입이 통과하면 로컬 MLflow에 정상 run으로 재생되고 run_id를 출력한다.
이후 스크리닝·확정 재검증·풀 진입은 로컬 실행과 똑같이 그 run_id로 수행한다:

```bash
uv run python -m pipeline.compare <반입 run_id>
uv run python -m pipeline.pool <반입 run_id>
```

정식 CPU 일괄 판정은 각 짝비교의 두 팔이 같은 Kaggle CPU 실행 환경에서 완주했고 `remote.provider=kaggle` 태그와 모든 반입 관문이 일치하는지 먼저 확인한다.
서로 다른 짝비교의 검증된 OOF는 로컬 CPU 또는 Vast.ai CPU 결과와 같은 일괄 판정에 넣을 수 있다.

## 주의

- 반입 거부는 전부 이유가 출력된다. 해시 불일치는 자료·fold가 다른 것이고,
  재채점 불일치는 실행 환경이 어긋난 것이므로 Kaggle에서 재실행으로 다룬다.
- 기록 규약 확장(#98) 이전 실행(시드별 OOF 산출물 없음)은 export할 수 없다.
- 같은 묶음의 중복 반입은 거부된다. 다시 반입하려면 재실행해 새 묶음을 만든다.
- 일부 지역 사무실 사내망에서는 Kaggle API에 CA 번들 조치가 필요하다
  (memory: kernels 명령은 submit과 같은 API 도메인이라 같은 조치로 동작).
