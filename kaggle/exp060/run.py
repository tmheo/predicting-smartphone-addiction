import subprocess

COMMIT = "495cc49df87e1de625da99f440b7d7fb80ac4f33"
CONFIG = "configs/exp060_lookup_transformer_nn10.yaml"
BUNDLE = "/kaggle/working/exp060.bundle.zip"
# 원본 프록시(jayjoshi37 판본 1과 동일 파일). features.OriginalKnnColumns의
# sha256 게이트가 실행 시점에 2194ce19… 일치를 다시 검증한다.
PROXY = (
    "/kaggle/input/smartphone-usage-and-addiction-prediction/"
    "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
)

run = lambda cmd: subprocess.run(cmd, shell=True, check=True)
run("git clone https://github.com/tmheo/predicting-smartphone-addiction.git repo")
run(f"cd repo && git checkout {COMMIT} && pip install -e . -q")
run("mkdir -p repo/data/external")
for name in ("train.csv", "test.csv", "sample_submission.csv"):
    run(f"ln -sf /kaggle/input/playground-series-s6e8/{name} repo/data/{name}")
run(f"ln -sf {PROXY} repo/data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
out = subprocess.run(
    f"cd repo && python -m pipeline.run {CONFIG}",
    shell=True, check=True, capture_output=True, text=True,
)
print(out.stdout)
run_id = next(
    line.split("=", 1)[1].split()[0]
    for line in out.stdout.splitlines() if line.startswith("run_id=")
)
run(f"cd repo && python -m pipeline.bundle export {run_id} --out {BUNDLE}")
