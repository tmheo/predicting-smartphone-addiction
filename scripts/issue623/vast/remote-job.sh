#!/usr/bin/env bash
# 이슈 #623 원격 본 실행. RealMLP 기준 1개 + 사다리 3개를 GPU 4장에 하나씩 배정해 동시에 3시드 확정 단계로 학습한다.
# 설정마다 프로세스 하나가 시드 42, 43, 44를 순서대로 돌리고 fold 복구 자료를 recovery/<실험>-confirm에 남긴다.
set -Eeuo pipefail

readonly JOB_ID=issue623-realmlp-ladder-v1
readonly JOB_ROOT=/workspace/issue623-realmlp-v1
readonly INPUT_ROOT="$JOB_ROOT/input"
readonly PROJECT="$INPUT_ROOT/repo"
readonly VENV="$JOB_ROOT/python-env"
readonly RESULT_ROOT="$JOB_ROOT/results"
readonly LOG_ROOT="$JOB_ROOT/logs"
readonly STATE_ROOT="$JOB_ROOT/state"
readonly RECOVERY_ROOT="$JOB_ROOT/recovery"
readonly DELIVERABLE_ROOT="$JOB_ROOT/deliverable"
readonly RESULT_ARCHIVE="$JOB_ROOT/issue623-vast-result-v1.tar.gz"
readonly SCRIPT_PID=$$
# 실험 이름과 설정 경로. 순서 = GPU 번호. 기준이 0번이다.
readonly -a EXPERIMENTS=(
  exp139_realmlp_reference_qnormal_train_test
  cdv2_realmlp_raw4
  cdv2_realmlp_cats_te
  cdv2_realmlp_ratio_round
)
readonly -a CONFIGS=(
  configs/exp139_realmlp_reference_qnormal_train_test.yaml
  configs/constraint-derived/10_realmlp_exp139_raw4.yaml
  configs/constraint-derived/11_realmlp_exp139_cats_te.yaml
  configs/constraint-derived/12_realmlp_exp139_ratio_round.yaml
)
# 수동 실행 상한 8시간보다 짧은 원격 자체 정지: 7시간 40분. 학습 하나의 timeout은 7시간 20분.
readonly HARD_STOP_SECONDS=27600
readonly RUN_TIMEOUT_SECONDS=26400
EXPECTED_COMMIT=
declare -a GROUP_PIDS=()
TELEMETRY_GROUP_PID=
HARD_STOP_PID=
OVERALL_RC=1
PREFLIGHT_RC=1
RUN_RC=1
FINALIZE_RC=1
FINALIZED=0

atomic_line() {
  local target=$1
  local value=$2
  printf '%s\n' "$value" >"$target.part"
  mv "$target.part" "$target"
}

stop_process_group() {
  local leader=${1:-}
  local attempt
  if [[ ! $leader =~ ^[0-9]+$ ]] || ! kill -0 -- "-$leader" 2>/dev/null; then
    return 0
  fi
  kill -TERM -- "-$leader" 2>/dev/null || true
  for attempt in {1..10}; do
    if ! kill -0 -- "-$leader" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  kill -KILL -- "-$leader" 2>/dev/null || true
}

stop_all_groups() {
  local pid
  for pid in "${GROUP_PIDS[@]}"; do
    stop_process_group "$pid"
  done
  GROUP_PIDS=()
}

write_deliverable() {
  local run_status=failed
  local archive_rc=0
  if [[ $OVERALL_RC -eq 0 ]]; then
    run_status=finished
  fi
  atomic_line "$STATE_ROOT/exit-code.txt" "$OVERALL_RC"
  printf '%s\n' "$OVERALL_RC" >"$JOB_ROOT/exit-status.txt"
  mkdir -p \
    "$DELIVERABLE_ROOT/results" \
    "$DELIVERABLE_ROOT/logs" \
    "$DELIVERABLE_ROOT/recovery" \
    "$DELIVERABLE_ROOT/state"
  cp -R "$RESULT_ROOT/." "$DELIVERABLE_ROOT/results/" 2>/dev/null || true
  cp -R "$LOG_ROOT/." "$DELIVERABLE_ROOT/logs/" 2>/dev/null || true
  cp -R "$RECOVERY_ROOT/." "$DELIVERABLE_ROOT/recovery/" 2>/dev/null || true
  cp -R "$STATE_ROOT/." "$DELIVERABLE_ROOT/state/" 2>/dev/null || true
  cp "$JOB_ROOT/exit-status.txt" "$INPUT_ROOT/execution-spec.json" \
    "$INPUT_ROOT/input-manifest.sha256" "$DELIVERABLE_ROOT/"
  printf '{"job_id":"%s","status":"%s","exit_code":%s,"preflight_exit_code":%s,"pipeline_exit_code":%s,"finalize_exit_code":%s,"finished_at":"%s"}\n' \
    "$JOB_ID" "$run_status" "$OVERALL_RC" "$PREFLIGHT_RC" "$RUN_RC" \
    "$FINALIZE_RC" "$(date -u +%FT%TZ)" >"$DELIVERABLE_ROOT/status.json"
  (
    cd "$DELIVERABLE_ROOT"
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
    OVERALL_RC=$archive_rc
    atomic_line "$STATE_ROOT/exit-code.txt" "$OVERALL_RC"
    atomic_line "$STATE_ROOT/failed" "$(date -u +%FT%TZ)"
    return "$archive_rc"
  fi
  if [[ $OVERALL_RC -eq 0 ]]; then
    atomic_line "$STATE_ROOT/finished" "$(date -u +%FT%TZ)"
  else
    atomic_line "$STATE_ROOT/failed" "$(date -u +%FT%TZ)"
  fi
}

finish() {
  local caught_rc=$?
  trap - EXIT TERM INT
  set +e
  if [[ $FINALIZED -eq 1 ]]; then
    exit "$caught_rc"
  fi
  FINALIZED=1
  stop_all_groups
  stop_process_group "$TELEMETRY_GROUP_PID"
  stop_process_group "$HARD_STOP_PID"
  if [[ $caught_rc -ne 0 ]]; then
    OVERALL_RC=$caught_rc
  fi
  write_deliverable
  local deliverable_rc=$?
  if [[ $deliverable_rc -ne 0 ]]; then
    OVERALL_RC=$deliverable_rc
  fi
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'overall_rc=%s\n' "$OVERALL_RC"
  exit "$OVERALL_RC"
}

handle_signal() {
  local signal_rc=$1
  stop_all_groups
  exit "$signal_rc"
}

mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$STATE_ROOT" "$RECOVERY_ROOT"
if [[ ! -f $STATE_ROOT/started ]]; then
  atomic_line "$STATE_ROOT/started" "$(date -u +%FT%TZ)"
fi
atomic_line "$STATE_ROOT/supervisor.pid" "$SCRIPT_PID"
exec > >(tee -a "$LOG_ROOT/supervisor.log") 2>&1
trap finish EXIT
trap 'handle_signal 143' TERM
trap 'handle_signal 130' INT

EXPECTED_COMMIT=$(tr -d '[:space:]' <"$INPUT_ROOT/expected-commit.txt")
[[ $EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]]
printf 'started_at=%s\n' "$(date -u +%FT%TZ)"
printf 'job_id=%s\n' "$JOB_ID"
printf 'instance_id=%s\n' "${VAST_RUN_INSTANCE_ID:?}"
printf 'commit=%s\n' "$EXPECTED_COMMIT"

setsid bash -c '
  sleep "$3"
  printf "hard_stop_at=%s\n" "$(date -u +%FT%TZ)" >>"$1"
  kill -TERM "$2"
' _ "$LOG_ROOT/hard-stop.log" "$SCRIPT_PID" "$HARD_STOP_SECONDS" \
  >"$LOG_ROOT/hard-stop-supervisor.log" 2>&1 &
HARD_STOP_PID=$!

cd "$INPUT_ROOT"
sha256sum --check --quiet input-manifest.sha256
cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=normal)"
test "$(sha256sum artifacts/folds.parquet | awk '{print $1}')" = 5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4
test "$(sha256sum data/train.csv | awk '{print $1}')" = f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c
test "$(sha256sum data/test.csv | awk '{print $1}')" = 8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e
test "$(sha256sum data/sample_submission.csv | awk '{print $1}')" = 206763fe5786fb9c80d4e9289a3b812030d3dbb36450c6eb63348098154ce63e
for config in "${CONFIGS[@]}"; do
  test -f "$config"
done

{
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,utilization.gpu,memory.used --format=csv,noheader
  printf 'compute_apps='; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'; printf '\n'
  nproc
  cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
  git status --porcelain=v1 --untracked-files=normal
} | tee "$LOG_ROOT/environment-before-python.log"

# 외부 GPU 부하 관문: 우리 프로세스가 시작되기 전에 GPU 프로세스가 보이면 공급 환경 실패다(#483 v1 사례).
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

# 이전 인스턴스의 fold 복구 자료가 입력에 있으면 같은 실행 정체성으로 반입해 이어 달린다.
if [[ -d "$INPUT_ROOT/recovery" ]]; then
  cp -R "$INPUT_ROOT/recovery/." "$RECOVERY_ROOT/"
  find "$RECOVERY_ROOT" -type f | wc -l | tee "$LOG_ROOT/recovery-seed-count.txt"
fi

run_in_own_group() {
  local log_path=$1
  shift
  setsid "$@" > >(tee -a "$log_path") 2>&1 &
  local pid=$!
  GROUP_PIDS+=("$pid")
  local group_rc
  if wait "$pid"; then
    group_rc=0
  else
    group_rc=$?
  fi
  stop_process_group "$pid"
  GROUP_PIDS=()
  return "$group_rc"
}

if run_in_own_group "$LOG_ROOT/python-preflight.log" \
  timeout --foreground --signal=TERM --kill-after=30s 1800s \
  "$PROJECT/scripts/run_remote_python.sh" \
    --system-python python3 \
    --project "$PROJECT" \
    --venv "$VENV" \
    --evidence "$RESULT_ROOT/python-environment.json" \
    -- \
    -c 'import json, torch; assert torch.cuda.is_available(); n = torch.cuda.device_count(); assert n >= 4, n; names = []
for i in range(n):
    layer = torch.nn.Linear(8, 4).cuda(i); x = torch.ones(2, 8, device=f"cuda:{i}", requires_grad=True); y = layer(x).square().mean(); y.backward(); assert x.grad is not None; names.append(torch.cuda.get_device_name(i))
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "devices": names, "device_count": n}))'; then
  PREFLIGHT_RC=0
else
  PREFLIGHT_RC=$?
fi

RUN_RC=$PREFLIGHT_RC
FINALIZE_RC=0
if [[ $PREFLIGHT_RC -eq 0 ]]; then
  export PYTHONUNBUFFERED=1
  export REMOTE_RUN_PROVIDER=vast
  export REMOTE_RUN_JOB_ID="$JOB_ID"
  setsid bash -c '
    log_path=$1
    while true; do
      printf "%s," "$(date -u +%FT%TZ)"
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits | tr "\n" ";"
      printf "\n"
      sleep 60
    done >>"$log_path" 2>&1
  ' _ "$LOG_ROOT/gpu-telemetry.csv" &
  TELEMETRY_GROUP_PID=$!

  # 프로세스 4개가 CPU 할당량을 넘지 않게 OpenMP/BLAS 스레드를 같은 몫으로 나눈다(cgroup 할당량 기준, #419 교훈).
  cpu_quota=$(nproc)
  if read -r quota period </sys/fs/cgroup/cpu.max 2>/dev/null && [[ $quota != max ]]; then
    cpu_quota=$(( quota / period ))
  fi
  threads=$(( cpu_quota / ${#EXPERIMENTS[@]} ))
  if (( threads < 1 )); then threads=1; fi
  printf 'cpu_quota=%s threads_per_process=%s\n' "$cpu_quota" "$threads" | tee "$LOG_ROOT/thread-budget.txt"

  declare -a RUN_PIDS=()
  for index in "${!EXPERIMENTS[@]}"; do
    experiment=${EXPERIMENTS[$index]}
    config=${CONFIGS[$index]}
    setsid env \
      CUDA_VISIBLE_DEVICES="$index" \
      OMP_NUM_THREADS="$threads" \
      OPENBLAS_NUM_THREADS="$threads" \
      MKL_NUM_THREADS="$threads" \
      PIPELINE_XGB_N_JOBS="$threads" \
      timeout --foreground --signal=TERM --kill-after=120s "$RUN_TIMEOUT_SECONDS" \
      "$VENV/bin/python" -m pipeline.run "$config" \
        --stage confirm \
        --recovery-dir "$RECOVERY_ROOT/$experiment-confirm" \
      >"$LOG_ROOT/pipeline-$experiment.log" 2>&1 &
    pid=$!
    RUN_PIDS+=("$pid")
    GROUP_PIDS+=("$pid")
    printf '%s gpu=%s pid=%s started_at=%s\n' "$experiment" "$index" "$pid" "$(date -u +%FT%TZ)" \
      | tee -a "$LOG_ROOT/run-launch.log"
  done
  RUN_RC=0
  for index in "${!RUN_PIDS[@]}"; do
    experiment=${EXPERIMENTS[$index]}
    if wait "${RUN_PIDS[$index]}"; then
      rc=0
    else
      rc=$?
    fi
    printf '%s rc=%s finished_at=%s\n' "$experiment" "$rc" "$(date -u +%FT%TZ)" \
      | tee -a "$LOG_ROOT/run-launch.log"
    printf '%s\n' "$rc" >"$LOG_ROOT/pipeline-$experiment-exit-code.txt"
    if [[ $rc -ne 0 ]]; then
      RUN_RC=$rc
    fi
  done
  stop_all_groups

  if [[ $RUN_RC -eq 0 ]]; then
    finalize_args=()
    for index in "${!EXPERIMENTS[@]}"; do
      finalize_args+=(--experiment "${EXPERIMENTS[$index]}=$LOG_ROOT/pipeline-${EXPERIMENTS[$index]}.log")
    done
    if run_in_own_group "$LOG_ROOT/bundle-export.log" \
      timeout --foreground --signal=TERM --kill-after=30s 1800s \
      "$VENV/bin/python" "$INPUT_ROOT/remote-finalize.py" \
        --results "$RESULT_ROOT" \
        --job-id "$JOB_ID" \
        --instance-id "$VAST_RUN_INSTANCE_ID" \
        --expected-commit "$EXPECTED_COMMIT" \
        "${finalize_args[@]}"; then
      FINALIZE_RC=0
    else
      FINALIZE_RC=$?
    fi
  fi
fi

printf '%s\n' "$PREFLIGHT_RC" >"$LOG_ROOT/preflight-exit-code.txt"
printf '%s\n' "$RUN_RC" >"$LOG_ROOT/pipeline-exit-code.txt"
printf '%s\n' "$FINALIZE_RC" >"$LOG_ROOT/finalize-exit-code.txt"
OVERALL_RC=$RUN_RC
if [[ $OVERALL_RC -eq 0 && $FINALIZE_RC -ne 0 ]]; then
  OVERALL_RC=$FINALIZE_RC
fi
exit "$OVERALL_RC"
