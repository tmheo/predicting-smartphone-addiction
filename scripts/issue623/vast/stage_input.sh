#!/bin/sh
# 이슈 #623 원격 실행 입력 묶음을 만든다.
# 저장소는 실행 커밋(main 끝)의 얕은 독립 복제본(.git 포함)이고, 파일별 해시 목록은 .git/을 제외한다.
# 사용법: sh scripts/issue623/vast/stage_input.sh <main 체크아웃> <출력 디렉터리> [가지 이름]
# 가지 이름을 주면 그 가지 끝 커밋을 실행 커밋으로 쓴다(기본은 현재 가지). 실행 커밋을 고정한 채 제어 스크립트만 고칠 때 쓴다.
set -eu

SOURCE=$(CDPATH= cd -- "$1" && pwd -P)
RUN_ROOT=$2
JOB_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
STAGING="$RUN_ROOT/input"
ARCHIVE="$RUN_ROOT/issue623-vast-input-v1.tar.gz"
BRANCH=${3:-$(git -C "$SOURCE" rev-parse --abbrev-ref HEAD)}
EXPECTED_COMMIT=$(git -C "$SOURCE" rev-parse "refs/heads/$BRANCH")
test "$BRANCH" != HEAD

mkdir -p "$RUN_ROOT"
[ ! -e "$STAGING" ] || { printf '준비 경로가 이미 있다: %s\n' "$STAGING" >&2; exit 1; }
[ ! -e "$ARCHIVE" ] || { printf '묶음이 이미 있다: %s\n' "$ARCHIVE" >&2; exit 1; }
[ -z "$(git -C "$SOURCE" --no-optional-locks status --porcelain=v1 --untracked-files=normal)" ] || {
    printf '작업 폴더가 깨끗하지 않다.\n' >&2; exit 1; }

mkdir -p "$STAGING"
# 얕은 단일 가지 복제로 .git 크기를 줄인다. 실행 커밋의 체크아웃만 필요하다.
git clone --quiet --no-hardlinks --depth 1 --single-branch --branch "$BRANCH" "file://$SOURCE" "$STAGING/repo"
git -C "$STAGING/repo" checkout --quiet --detach "$EXPECTED_COMMIT"
git -C "$STAGING/repo" remote remove origin
# 단일 가지 복제는 origin 제거 뒤 refs/remotes/origin/HEAD 심볼릭 참조가 남아 fsck --strict가 실패한다(#226 v1 묶음의 실패 원인).
git -C "$STAGING/repo" symbolic-ref --delete refs/remotes/origin/HEAD 2>/dev/null || true
git -C "$STAGING/repo" fsck --strict --no-progress --no-dangling
test "$(git -C "$STAGING/repo" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$STAGING/repo" --no-optional-locks status --porcelain=v1 --untracked-files=normal)"

mkdir -p "$STAGING/repo/data"
for f in train.csv test.csv sample_submission.csv; do
    cp "$SOURCE/data/$f" "$STAGING/repo/data/$f"
    chmod u+w "$STAGING/repo/data/$f"
done
test -z "$(git -C "$STAGING/repo" --no-optional-locks status --porcelain=v1 --untracked-files=normal)"

cp "$JOB_DIR/remote-job.sh" "$JOB_DIR/remote-start.sh" "$JOB_DIR/remote-finalize.py" "$STAGING/"
chmod +x "$STAGING/remote-job.sh" "$STAGING/remote-start.sh"
bash -n "$STAGING/remote-job.sh"
bash -n "$STAGING/remote-start.sh"
printf '%s\n' "$EXPECTED_COMMIT" > "$STAGING/expected-commit.txt"
cat > "$STAGING/execution-spec.json" <<EOF
{
  "schema_version": 1,
  "issue": 623,
  "provider": "vast",
  "job_id": "issue623-realmlp-ladder-v1",
  "image": "nvcr.io/nvidia/pytorch@sha256:a411b86de9ac003ce5db43894ea7920718512bc02c51a521157c0899aac75631",
  "git_commit": "$EXPECTED_COMMIT",
  "stage": "confirm",
  "seeds": [42, 43, 44],
  "folds": [0, 1, 2, 3, 4],
  "layout": "one process per GPU, seeds sequential inside each process",
  "experiments": [
    {"gpu": 0, "experiment": "exp139_realmlp_reference_qnormal_train_test", "config": "configs/exp139_realmlp_reference_qnormal_train_test.yaml"},
    {"gpu": 1, "experiment": "cdv2_realmlp_raw4", "config": "configs/constraint-derived/10_realmlp_exp139_raw4.yaml"},
    {"gpu": 2, "experiment": "cdv2_realmlp_cats_te", "config": "configs/constraint-derived/11_realmlp_exp139_cats_te.yaml"},
    {"gpu": 3, "experiment": "cdv2_realmlp_ratio_round", "config": "configs/constraint-derived/12_realmlp_exp139_ratio_round.yaml"}
  ],
  "manual_running_limit_hours": 8,
  "storage_gb": 60
}
EOF

(
    cd "$STAGING"
    find . -type f ! -path "./repo/.git/*" ! -name input-manifest.sha256 -print0 \
        | LC_ALL=C sort -z | xargs -0 shasum -a 256 > input-manifest.sha256
)
COPYFILE_DISABLE=1 tar -czf "$ARCHIVE" -C "$STAGING" .
shasum -a 256 "$ARCHIVE" | awk '{print $1}' > "$ARCHIVE.sha256"

printf '실행 커밋: %s\n' "$EXPECTED_COMMIT"
printf 'manifest 항목: %s개\n' "$(wc -l < "$STAGING/input-manifest.sha256" | tr -d ' ')"
printf '묶음 크기: %s 바이트\n' "$(wc -c < "$ARCHIVE" | tr -d ' ')"
printf '묶음 SHA-256: %s\n' "$(cat "$ARCHIVE.sha256")"
