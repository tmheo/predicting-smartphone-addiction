# Kaggle GPU 실행 절차 (#98)

GPU가 필요한 실험(#58 Lookup-Transformer 등)을 Kaggle에서 실행하고,
그 결과를 실행 기록 묶음으로 로컬 판정 경로에 반입하는 절차다.
용어는 CONTEXT.md의 "실행 기록 묶음"과 "묶음 반입"을 따른다.

## 원칙

- Kaggle에서도 `pipeline.run <config>`를 그대로 실행한다.
  노트북 자체 학습 루프는 만들지 않는다(두 번째 CV 루프 divergence 금지).
- config는 실행 전에 main에 커밋돼 있어야 한다.
  반입이 "실행의 git_commit에 그 config가 그대로 존재하는가"를 검증한다.
- 커널은 인터넷을 켠 상태로 실행한다(GPU 실험 커널은 허용, 제출 노트북 아님).
- 자동으로 돌아온 결과도 반입 게이트(입력 해시·출처·재채점)가 검증하므로,
  실행 경로가 자동화됐다고 신뢰를 더 얹지 않는다.

## 자동 실행 (kaggle CLI, 권장)

kaggle CLI의 kernels 명령으로 사람 개입 없이 전체 루프를 돌린다.
push(업로드 + 즉시 배치 실행), status(폴링), logs(실행 로그), output(산출물 다운로드)을 쓴다.

전제: API 토큰 설정, 계정 전화번호 인증(GPU·인터넷 커널 요건), 대회 규칙 동의.
GPU 주간 할당은 30시간, 배치 실행 상한은 GPU 약 9시간이다.

1. 커널 폴더를 저장소에 둔다(예: `kaggle/exp0NN/`). 내용물은 두 파일이다.

   `kernel-metadata.json`:

   ```json
   {
     "id": "tmheo/exp0NN-gpu-run",
     "title": "exp0NN gpu run",
     "code_file": "run.py",
     "language": "python",
     "kernel_type": "script",
     "is_private": true,
     "enable_gpu": true,
     "enable_internet": true,
     "competition_sources": ["playground-series-s6e8"]
   }
   ```

   `run.py` (노트북일 필요 없이 plain 스크립트면 된다):

   ```python
   import subprocess

   COMMIT = "<실행할 config가 포함된 커밋>"
   CONFIG = "configs/exp0NN_*.yaml"

   run = lambda cmd: subprocess.run(cmd, shell=True, check=True)
   run("git clone https://github.com/tmheo/predicting-smartphone-addiction.git repo")
   run(f"cd repo && git checkout {COMMIT}")
   # pip install -e .는 안 된다: mlflow의 pandas<3 선언과 이 프로젝트의 pandas 3
   # 강제(override-dependencies)를 pip은 해석하지 못해 ResolutionImpossible이 난다.
   # uv sync가 uv.lock 그대로 로컬과 동일한 환경(.python-version의 3.13 포함)을 만든다.
   run("pip install uv -q")
   run("cd repo && uv sync --frozen --no-dev -q")
   run("mkdir -p repo/data")
   for name in ("train.csv", "test.csv", "sample_submission.csv"):
       run(f"ln -sf /kaggle/input/playground-series-s6e8/{name} repo/data/{name}")
   out = subprocess.run(
       f"cd repo && uv run --no-sync python -m pipeline.run {CONFIG}",
       shell=True, check=True, capture_output=True, text=True,
   )
   print(out.stdout)
   run_id = next(
       line.split("=", 1)[1].split()[0]
       for line in out.stdout.splitlines() if line.startswith("run_id=")
   )
   run(f"cd repo && uv run --no-sync python -m pipeline.bundle export {run_id} --out /kaggle/working/exp0NN.bundle.zip")
   ```

   저장소가 비공개면 커널의 익명 git clone이 실패한다. 실행 전 저장소가 public인지
   확인한다(#58에서 public 전환). 추가 자료가 필요한 실험은 해당 파일을 담은 Kaggle
   데이터셋을 `dataset_sources`로 연결하고 저장소 경로에 심볼릭 링크한다.

2. 실행을 밀어 넣는다. `--accelerator`로 GPU 종류를 고른다.

   ```bash
   uv run kaggle kernels push -p kaggle/exp0NN --accelerator <gpu>
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
   !python -m pipeline.run configs/expNNN_*.yaml
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

## 주의

- 반입 거부는 전부 이유가 출력된다. 해시 불일치는 자료·fold가 다른 것이고,
  재채점 불일치는 실행 환경이 어긋난 것이므로 Kaggle에서 재실행으로 다룬다.
- 기록 규약 확장(#98) 이전 실행(시드별 OOF 산출물 없음)은 export할 수 없다.
- 같은 묶음의 중복 반입은 거부된다. 다시 반입하려면 재실행해 새 묶음을 만든다.
- 일부 지역 사무실 사내망에서는 Kaggle API에 CA 번들 조치가 필요하다
  (memory: kernels 명령은 submit과 같은 API 도메인이라 같은 조치로 동작).
