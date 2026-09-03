#!/bin/sh
# 이슈 #623 로컬 나무 3계열 실행을 (다시) 시작한다. 여러 번 실행해도 안전하다.
# 살아 있는 차선은 lock으로 건너뛰고, 끝난 설정은 결과 JSON으로 건너뛰며, 중단된 설정은 fold 복구로 이어 달린다.
# 사용법: sh scripts/issue623/resume_local.sh [워크트리 경로]
set -eu
MAIN_ROOT=/Users/taemyungheo/workspace/github/kaggle/predicting-smartphone-addiction
WORKTREE=${1:-$MAIN_ROOT/.claude/worktrees/issue-623-ladder}
OUT_ROOT="$MAIN_ROOT/run-logs/issue623"
mkdir -p "$OUT_ROOT/local"
cd "$WORKTREE"
.venv/bin/python scripts/issue623/local_trees.py status --out-root "$OUT_ROOT"
if pgrep -f "local_trees.py drive --out-root $OUT_ROOT" >/dev/null 2>&1; then
    printf '드라이버가 이미 돌고 있다. 진행률만 표시했다.\n'
    exit 0
fi
nohup .venv/bin/python scripts/issue623/local_trees.py drive --out-root "$OUT_ROOT" \
    >>"$OUT_ROOT/local/drive.log" 2>&1 &
printf '드라이버 시작: pid %s, 로그 %s/local/drive.log\n' "$!" "$OUT_ROOT"
printf 'caffeinate -i는 유휴 잠자기만 막고 덮개 닫기는 못 막는다. 이동 뒤에는 이 스크립트를 다시 실행한다.\n'
