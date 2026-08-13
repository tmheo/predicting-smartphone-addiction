"""Kaggle GPU 실행 커널 템플릿. 새 실험은 이 폴더를 kaggle/expNNN으로 복사해
COMMIT·CONFIG·BUNDLE과 kernel-metadata.json의 id·title만 바꾼다.

절차와 원칙은 docs/kaggle-gpu-run.md를 따른다. #99의 세 가지 개선 반영:
- repo를 /kaggle/working 밖(/tmp)에 클론해 .venv가 산출물 다운로드에 딸려오지 않는다.
- pipeline.run 출력을 실시간으로 흘리면서(tee) run_id= 줄 파싱도 유지한다.
- GPU가 여럿이면(T4 x2) PIPELINE_SEED_GPUS로 시드 단위 프로세스 병렬을 켠다.
"""

import glob
import os
import subprocess

COMMIT = "<실행할 config가 포함된 main 커밋>"
CONFIG = "configs/exp0NN_*.yaml"
BUNDLE = "/kaggle/working/exp0NN.bundle.zip"
REPO = "/tmp/repo"  # /kaggle/working에 두면 repo(.venv 포함)가 산출물에 딸려온다. (#99)

run = lambda cmd: subprocess.run(cmd, shell=True, check=True)


def find_input(name: str) -> str:
    """/kaggle/input 아래에서 파일을 찾는다. 이미지에 따라 마운트 경로가 다르다."""
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name}을 찾지 못했다. /kaggle/input = {os.listdir('/kaggle/input')}")
    return hits[0]


run(f"git clone https://github.com/tmheo/predicting-smartphone-addiction.git {REPO}")
run(f"cd {REPO} && git checkout {COMMIT}")
# pip은 override-dependencies(pandas 3 강제)를 해석하지 못하므로 uv.lock 그대로
# 로컬과 동일한 환경을 재현한다(.python-version의 3.13 포함).
run("pip install uv -q")
run(f"cd {REPO} && uv sync --frozen --no-dev -q")
run(f"mkdir -p {REPO}/data")
for name in ("train.csv", "test.csv", "sample_submission.csv"):
    run(f"ln -sf {find_input(name)} {REPO}/data/{name}")
# 추가 자료가 필요한 실험(예: 원본 프록시)은 dataset_sources로 연결한 뒤 여기서
# find_input으로 찾아 repo 경로에 심볼릭 링크한다.

env = dict(os.environ)
gpus = subprocess.run(
    "nvidia-smi -L", shell=True, capture_output=True, text=True
).stdout.strip().splitlines()
if len(gpus) > 1:
    # 시드 단위 프로세스 병렬(#99). 같은 GPU 모델 x N이므로 순차 실행과 결과가 같다.
    env["PIPELINE_SEED_GPUS"] = ",".join(str(i) for i in range(len(gpus)))

# capture 대신 tee: 학습 로그를 실시간으로 커널 로그에 흘리면서 run_id 파싱용으로 보관.
proc = subprocess.Popen(
    f"cd {REPO} && uv run --no-sync python -m pipeline.run {CONFIG}",
    shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
)
lines = []
for line in proc.stdout:
    print(line, end="", flush=True)
    lines.append(line)
if proc.wait() != 0:
    raise SystemExit(proc.returncode)
run_id = next(
    line.split("=", 1)[1].split()[0] for line in lines if line.startswith("run_id=")
)
run(f"cd {REPO} && uv run --no-sync python -m pipeline.bundle export {run_id} --out {BUNDLE}")
