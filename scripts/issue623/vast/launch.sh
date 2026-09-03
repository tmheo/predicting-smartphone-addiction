#!/usr/bin/env bash
# 외부 GPU 부하 관문(종료 코드 75)에 걸리면 장비를 제외하고 최대 3번까지 다른 호스트로 재시도한다.
set -u
HERE=$(cd -- "$(dirname -- "$0")" && pwd -P)
MAIN_ROOT=/Users/taemyungheo/workspace/github/kaggle/predicting-smartphone-addiction
LOG_DIR="$MAIN_ROOT/run-logs/issue623/vast"
mkdir -p "$LOG_DIR"
for attempt in 1 2 3; do
  ATTEMPT=$attempt bash "$HERE/run-vast.sh" >>"$LOG_DIR/control-$attempt.log" 2>&1
  rc=$?
  printf 'attempt=%s rc=%s at=%s\n' "$attempt" "$rc" "$(date -u +%FT%TZ)" >>"$LOG_DIR/launch-attempts.log"
  if [[ $rc -ne 75 ]]; then
    exit "$rc"
  fi
  sleep 30
done
exit 75
