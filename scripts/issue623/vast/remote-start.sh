#!/usr/bin/env bash
# 이슈 #623 원격 입력 관문. 입력 manifest, 저장소 커밋, 깨끗함을 확인한 뒤 remote-job.sh로 넘어간다.
set -Eeuo pipefail

readonly JOB_ROOT=/workspace/issue623-realmlp-v1
readonly INPUT_ROOT="$JOB_ROOT/input"
readonly STATE_ROOT="$JOB_ROOT/state"
readonly LOG_ROOT="$JOB_ROOT/logs"
readonly DELIVERABLE_ROOT="$JOB_ROOT/deliverable"
readonly RESULT_ARCHIVE="$JOB_ROOT/issue623-vast-result-v1.tar.gz"
PROJECT=

atomic_line() {
  local target=$1
  local value=$2
  printf '%s\n' "$value" >"$target.part"
  mv "$target.part" "$target"
}

archive_gate_failure() {
  local caught_rc=$?
  local archive_rc=0
  trap - EXIT
  set +e
  if [[ $caught_rc -eq 0 ]]; then
    caught_rc=1
  fi
  atomic_line "$STATE_ROOT/exit-code.txt" "$caught_rc"
  mkdir -p "$DELIVERABLE_ROOT/logs" "$DELIVERABLE_ROOT/state"
  cp -R "$LOG_ROOT/." "$DELIVERABLE_ROOT/logs/" 2>/dev/null || true
  cp -R "$STATE_ROOT/." "$DELIVERABLE_ROOT/state/" 2>/dev/null || true
  for source in "$INPUT_ROOT/execution-spec.json" "$INPUT_ROOT/input-manifest.sha256"; do
    if [[ -f $source ]]; then
      cp "$source" "$DELIVERABLE_ROOT/"
    fi
  done
  printf '{"phase":"input_gate","exit_code":%s,"finished_at":"%s"}\n' \
    "$caught_rc" "$(date -u +%FT%TZ)" >"$DELIVERABLE_ROOT/status.json"
  (
    cd "$DELIVERABLE_ROOT" || exit 1
    find . -type f ! -name result-manifest.sha256 -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum >result-manifest.sha256
  )
  if tar -czf "$RESULT_ARCHIVE.part" -C "$JOB_ROOT" deliverable \
    && mv "$RESULT_ARCHIVE.part" "$RESULT_ARCHIVE" \
    && sha256sum "$RESULT_ARCHIVE" >"$RESULT_ARCHIVE.sha256.part" \
    && mv "$RESULT_ARCHIVE.sha256.part" "$RESULT_ARCHIVE.sha256"; then
    :
  else
    archive_rc=$?
  fi
  if [[ $archive_rc -ne 0 ]]; then
    caught_rc=$archive_rc
    atomic_line "$STATE_ROOT/exit-code.txt" "$caught_rc"
  fi
  atomic_line "$STATE_ROOT/failed" "$(date -u +%FT%TZ)"
  exit "$caught_rc"
}

mkdir -p "$STATE_ROOT" "$LOG_ROOT"
atomic_line "$STATE_ROOT/started" "$(date -u +%FT%TZ)"
atomic_line "$STATE_ROOT/supervisor.pid" "$$"
exec > >(tee -a "$LOG_ROOT/input-gate.log") 2>&1
trap archive_gate_failure EXIT

PROJECT=$(realpath -e "$INPUT_ROOT/repo")
readonly PROJECT
case "$PROJECT" in
  "$INPUT_ROOT"/*) ;;
  *)
    printf '원격 저장소가 검증된 입력 경로 밖에 있다.\n' >&2
    exit 1
    ;;
esac

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$PROJECT"
export GIT_OPTIONAL_LOCKS=0

EXPECTED_COMMIT=$(tr -d '[:space:]' <"$INPUT_ROOT/expected-commit.txt")
[[ $EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]]
(
  cd "$INPUT_ROOT"
  sha256sum --check --quiet input-manifest.sha256
)
git -C "$PROJECT" fsck --strict --no-progress --no-dangling
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$PROJECT" status --porcelain=v1 --untracked-files=normal)"

trap - EXIT
exec bash "$INPUT_ROOT/remote-job.sh"
