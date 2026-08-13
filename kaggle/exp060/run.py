import glob
import os
import subprocess
import sys

COMMIT = "495cc49df87e1de625da99f440b7d7fb80ac4f33"
CONFIG = "configs/exp060_lookup_transformer_nn10.yaml"
BUNDLE = "/kaggle/working/exp060.bundle.zip"
# 원본 프록시(jayjoshi37 판본 1과 동일 파일). features.OriginalKnnColumns의
# sha256 게이트가 실행 시점에 2194ce19… 일치를 다시 검증한다.
PROXY_NAME = "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"

run = lambda cmd: subprocess.run(cmd, shell=True, check=True)


def find_input(name: str) -> str:
    """/kaggle/input 아래에서 파일을 찾는다. 이미지에 따라 마운트 경로가 다르다."""
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name}을 찾지 못했다. /kaggle/input = {os.listdir('/kaggle/input')}")
    return hits[0]


run("git clone https://github.com/tmheo/predicting-smartphone-addiction.git repo")
run(f"cd repo && git checkout {COMMIT}")
# pip은 override-dependencies(pandas 3 강제)를 해석하지 못하므로 uv.lock 그대로
# 로컬과 동일한 환경을 재현한다(.python-version의 3.13 포함).
run("pip install uv -q")
run("cd repo && uv sync --frozen --no-dev -q")
run("mkdir -p repo/data/external")
for name in ("train.csv", "test.csv", "sample_submission.csv"):
    run(f"ln -sf {find_input(name)} repo/data/{name}")
run(f"ln -sf {find_input(PROXY_NAME)} repo/data/external/{PROXY_NAME}")
out = subprocess.run(
    f"cd repo && uv run --no-sync python -m pipeline.run {CONFIG}",
    shell=True, check=False, capture_output=True, text=True,
)
print(out.stdout)
print(out.stderr, file=sys.stderr)
if out.returncode != 0:
    raise SystemExit(out.returncode)
run_id = next(
    line.split("=", 1)[1].split()[0]
    for line in out.stdout.splitlines() if line.startswith("run_id=")
)
run(f"cd repo && uv run --no-sync python -m pipeline.bundle export {run_id} --out {BUNDLE}")
