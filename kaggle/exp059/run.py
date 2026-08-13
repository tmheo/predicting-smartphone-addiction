import subprocess

COMMIT = "495cc49df87e1de625da99f440b7d7fb80ac4f33"
CONFIG = "configs/exp059_lookup_transformer.yaml"
BUNDLE = "/kaggle/working/exp059.bundle.zip"

run = lambda cmd: subprocess.run(cmd, shell=True, check=True)
run("git clone https://github.com/tmheo/predicting-smartphone-addiction.git repo")
run(f"cd repo && git checkout {COMMIT}")
# pip은 override-dependencies(pandas 3 강제)를 해석하지 못하므로 uv.lock 그대로
# 로컬과 동일한 환경을 재현한다(.python-version의 3.13 포함).
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
run(f"cd repo && uv run --no-sync python -m pipeline.bundle export {run_id} --out {BUNDLE}")
