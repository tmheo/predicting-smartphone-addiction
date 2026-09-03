#!/usr/bin/env bash
# 이슈 #623의 Vast.ai 실행(RealMLP 기준 1개 + 사다리 3개, 3시드 확정)을 생성부터 결과 반입, 정리, 청구 재조회까지 소유한다.
# #483 v3 제어 스크립트를 바탕으로 했다. 외부 GPU 부하 관문에 걸리면 종료 코드 75로 나가고 launch.sh가 다른 장비로 재시도한다.
set -Eeuo pipefail

readonly MAIN_ROOT=/Users/taemyungheo/workspace/github/kaggle/predicting-smartphone-addiction
readonly RUN_ROOT="$MAIN_ROOT/run-logs/issue623/vast"
readonly INPUT_ROOT="$RUN_ROOT/input"
readonly INPUT_ARCHIVE="$RUN_ROOT/issue623-vast-input-v1.tar.gz"
readonly INPUT_ARCHIVE_SHA="$RUN_ROOT/issue623-vast-input-v1.tar.gz.sha256"
readonly RESULT_NAME=issue623-vast-result-v1.tar.gz
readonly ATTEMPT=${ATTEMPT:-1}
readonly RESULT_ROOT="$RUN_ROOT/results-$ATTEMPT"
readonly EXTRACT_ROOT="$RESULT_ROOT/extracted"
readonly STATE_ROOT="$RUN_ROOT/vast-state-$ATTEMPT"
readonly RECOVERY_LOCAL_ROOT="$RUN_ROOT/recovery-live-$ATTEMPT"
readonly LEDGER="$RUN_ROOT/ledger.md"
readonly SSH_KEY=/Users/taemyungheo/.ssh/predicting-smartphone-addiction-vast-ed25519
readonly SSH_PUBLIC_KEY=/Users/taemyungheo/.ssh/predicting-smartphone-addiction-vast-ed25519.pub
readonly SSH_FINGERPRINT=SHA256:8AgBh2VtIo6dOWddsToBBf/GRkHVWgOPiq+EoZz19Os
readonly VAST_WRAPPER="$MAIN_ROOT/run-logs/vast-issue419/vast.sh"
readonly VAST_CA_BUNDLE="$MAIN_ROOT/run-logs/vast-issue419/ca_bundle.pem"
readonly IMAGE=nvcr.io/nvidia/pytorch@sha256:a411b86de9ac003ce5db43894ea7920718512bc02c51a521157c0899aac75631
readonly REMOTE_ROOT=/workspace/issue623-realmlp-v1
readonly REMOTE_STATE="$REMOTE_ROOT/state"
readonly REMOTE_RESULT="$REMOTE_ROOT/$RESULT_NAME"
readonly REMOTE_FAILURE_RESULT="$REMOTE_ROOT/issue623-vast-failure-v1.tar.gz"
readonly JOB_ID=issue623-realmlp-ladder-v1
readonly GH_REPO=tmheo/predicting-smartphone-addiction
readonly REMOTE_KILL_AFTER_SECONDS=300
# 확정 재검증은 running 확인 뒤 8시간이 수동 종료 상한이고 보조 안전장치는 생성 뒤 9시간 30분이다(vast-termination.md).
readonly MANUAL_RUNTIME_SECONDS=28800
readonly GUARD_TERMINATE_SECONDS=34200
readonly BILLING_START_DATE=$(date -u +%Y-%m-%d)
readonly EXCLUDED_MACHINES_FILE="$RUN_ROOT/excluded-machines.txt"
readonly -a EXPERIMENTS=(
  exp139_realmlp_reference_qnormal_train_test
  cdv2_realmlp_raw4
  cdv2_realmlp_cats_te
  cdv2_realmlp_ratio_round
)

EXPECTED_COMMIT=
INSTANCE_ID=
COMPLETED_INSTANCE_ID=
CLEANUP_STARTED=0
SSH_READY=0
REMOTE_START_ATTEMPTED=0
REMOTE_PID=
RESULT_RETRIEVED=0
START_CREDIT=
MANUAL_DEADLINE=0
SCHEDULE_REGISTERED=0

vast() {
  "$VAST_WRAPPER" "$@"
}

record() {
  printf -- '- %s [attempt %s] %s\n' "$(date -u +%FT%TZ)" "$ATTEMPT" "$*" | tee -a "$LEDGER"
}

schedule_value() {
  gh api "repos/$GH_REPO/actions/variables/VAST_TERMINATION_SCHEDULES" --jq .value
}

schedule_add() {
  local instance_id=$1 terminate_at=$2 current current_again preserved updated observed
  current=$(schedule_value) || return 1
  jq -e --arg job "$JOB_ID" 'all(.[]; .job_id != $job)' <<<"$current" >/dev/null \
    || return 1
  preserved=$(jq -c '.' <<<"$current") || return 1
  updated=$(jq -c \
    --argjson id "$instance_id" \
    --arg job "$JOB_ID" \
    --arg at "$terminate_at" \
    '. + [{instance_id: $id, job_id: $job, terminate_at: $at, volume_id: null}]' \
    <<<"$current") || return 1
  current_again=$(schedule_value) || return 1
  test "$(jq -cS '.' <<<"$current_again")" = "$(jq -cS '.' <<<"$current")" \
    || return 1
  gh variable set VAST_TERMINATION_SCHEDULES --repo "$GH_REPO" --body "$updated" \
    || return 1
  observed=$(schedule_value) || return 1
  jq -e --argjson id "$instance_id" --arg job "$JOB_ID" --arg at "$terminate_at" '
    any(.[]; .instance_id == $id and .job_id == $job and .terminate_at == $at)
  ' <<<"$observed" >/dev/null || return 1
  jq -en --argjson preserved "$preserved" --argjson observed "$observed" '
    all($preserved[]; . as $entry | any($observed[]; . == $entry))
  ' >/dev/null || return 1
}

schedule_remove() {
  local current current_again preserved observed
  current=$(schedule_value) || return 1
  preserved=$(jq -c --arg job "$JOB_ID" '[.[] | select(.job_id != $job)]' <<<"$current") \
    || return 1
  current_again=$(schedule_value) || return 1
  test "$(jq -cS '.' <<<"$current_again")" = "$(jq -cS '.' <<<"$current")" \
    || return 1
  gh variable set VAST_TERMINATION_SCHEDULES --repo "$GH_REPO" --body "$preserved" \
    || return 1
  observed=$(schedule_value) || return 1
  jq -e --arg job "$JOB_ID" 'all(.[]; .job_id != $job)' <<<"$observed" >/dev/null \
    || return 1
  jq -en --argjson preserved "$preserved" --argjson observed "$observed" '
    ($preserved | length) == ($observed | length)
    and all($preserved[]; . as $entry | any($observed[]; . == $entry))
  ' >/dev/null || return 1
}

schedule_remove_if_present() {
  local current
  current=$(schedule_value) || return 1
  if jq -e --arg job "$JOB_ID" 'any(.[]; .job_id == $job)' <<<"$current" >/dev/null; then
    schedule_remove
  else
    jq -e --arg job "$JOB_ID" 'all(.[]; .job_id != $job)' <<<"$current" >/dev/null
  fi
}

destroy_and_confirm() {
  local instance_id=$1 attempt
  vast destroy instance "$instance_id" -y --raw >/dev/null 2>&1 || true
  for attempt in {1..30}; do
    if vast show instances --raw \
      | jq -e --argjson id "$instance_id" 'all(.[]; .id != $id)' >/dev/null
    then
      return 0
    fi
    sleep 10
  done
  record "명령줄 삭제 뒤 5분 동안 인스턴스가 남아 공식 REST DELETE와 부재 조회로 전환: \`$instance_id\`"
  vast_rest_destroy_and_confirm "$instance_id"
}

vast_rest_destroy_and_confirm() {
  local instance_id=$1
  VAST_API_KEY="$(security find-generic-password \
    -s predicting-smartphone-addiction/vast-daily \
    -a taemyungheo \
    -w)" \
  SSL_CERT_FILE="$VAST_CA_BUNDLE" \
  /usr/bin/python3 - "$instance_id" <<'PY'
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


instance_id = int(sys.argv[1])
api_key = os.environ["VAST_API_KEY"]
ca_file = os.environ.get("SSL_CERT_FILE") or None
context = ssl.create_default_context(cafile=ca_file)
base_url = "https://console.vast.ai"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "User-Agent": "issue623-vast-resource-control/1",
}


def request(method, url, data=None):
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                payload = response.read()
            return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as exc:
            if method == "DELETE" and exc.code == 404:
                return {}
            if exc.code not in {429, 502, 503, 504} or attempt == 2:
                raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("REST 요청 재시도 흐름이 예상하지 못한 상태로 끝났다.")


request("DELETE", f"{base_url}/api/v0/instances/{instance_id}/", data=b"{}")
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    params = {
        "select_filters": json.dumps({}),
        "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
        "limit": "25",
    }
    found = False
    while True:
        url = f"{base_url}/api/v1/instances/?{urllib.parse.urlencode(params)}"
        page = request("GET", url)
        if any(int(row["id"]) == instance_id for row in page.get("instances") or []):
            found = True
            break
        next_token = page.get("next_token")
        if not next_token:
            break
        params["after_token"] = next_token
    if not found:
        print(f"rest_absence_confirmed={instance_id}")
        raise SystemExit(0)
    time.sleep(10)
raise RuntimeError(f"REST 삭제 뒤에도 인스턴스가 남아 있다: {instance_id}")
PY
}

job_instance_ids() {
  vast show instances --raw \
    | jq -r --arg label "$JOB_ID" '.[] | select(.label == $label) | .id'
}

destroy_job_resources() {
  local cleanup_failed=0 target_id
  local -a target_ids=()
  if [[ "$INSTANCE_ID" =~ ^[0-9]+$ ]]; then
    target_ids+=("$INSTANCE_ID")
  fi
  while IFS= read -r target_id; do
    if [[ "$target_id" =~ ^[0-9]+$ ]]; then
      target_ids+=("$target_id")
    fi
  done < <(job_instance_ids)

  if [[ ${#target_ids[@]} -gt 0 ]]; then
    while IFS= read -r target_id; do
      destroy_and_confirm "$target_id" || cleanup_failed=1
    done < <(printf '%s\n' "${target_ids[@]}" | LC_ALL=C sort -u)
  fi

  vast show instances --raw \
    | jq -e --arg label "$JOB_ID" 'all(.[]; .label != $label)' >/dev/null \
    || cleanup_failed=1
  if [[ "$INSTANCE_ID" =~ ^[0-9]+$ ]]; then
    vast show instances --raw \
      | jq -e --argjson id "$INSTANCE_ID" 'all(.[]; .id != $id)' >/dev/null \
      || cleanup_failed=1
  fi
  vast show volumes --raw \
    | jq -e --arg job "$JOB_ID" --arg id "${INSTANCE_ID:-}" '
        all(.[];
          (((.name // .label // "") | tostring | contains($job)) | not)
          and ($id == "" or (((.instance_id // .contract_id // "") | tostring) != $id)))
      ' >/dev/null \
    || cleanup_failed=1
  return "$cleanup_failed"
}

ssh_read_retry() {
  local attempt
  for attempt in {1..3}; do
    if ssh "${SSH_ARGS[@]}" "$@"; then
      return 0
    fi
    sleep 10
  done
  return 1
}

probe_remote_result_once() {
  ssh "${SSH_ARGS[@]}" "test -s '$REMOTE_RESULT'"
}

remote_snapshot() {
  ssh_read_retry "
    printf 'started='; if test -e '$REMOTE_STATE/started'; then printf 1; else printf 0; fi; printf '\n'
    printf 'finished='; if test -e '$REMOTE_STATE/finished'; then printf 1; else printf 0; fi; printf '\n'
    printf 'failed='; if test -e '$REMOTE_STATE/failed'; then printf 1; else printf 0; fi; printf '\n'
    printf 'pid='; if test -s '$REMOTE_STATE/supervisor.pid'; then cat '$REMOTE_STATE/supervisor.pid'; fi; printf '\n'
    printf 'alive='; if test -s '$REMOTE_STATE/supervisor.pid' && kill -0 -- \"-\$(cat '$REMOTE_STATE/supervisor.pid')\" 2>/dev/null; then printf 1; else printf 0; fi; printf '\n'
    printf 'exit_code='; if test -s '$REMOTE_STATE/exit-code.txt'; then cat '$REMOTE_STATE/exit-code.txt'; fi; printf '\n'
    printf 'result_bytes='; if test -e '$REMOTE_RESULT'; then wc -c < '$REMOTE_RESULT'; else printf 0; fi; printf '\n'
  "
}

signal_remote_group() {
  local signal=$1
  if [[ "$SSH_READY" -eq 1 && "$REMOTE_PID" =~ ^[0-9]+$ ]]; then
    ssh "${SSH_ARGS[@]}" \
      "kill -'$signal' -- '-$REMOTE_PID' 2>/dev/null || true" >/dev/null 2>&1 || true
  fi
}

stop_remote_group() {
  local attempt
  if [[ "$SSH_READY" -ne 1 || ! "$REMOTE_PID" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  signal_remote_group TERM
  local kill_deadline=$(( $(date +%s) + REMOTE_KILL_AFTER_SECONDS ))
  while (( $(date +%s) < kill_deadline )); do
    if ! ssh "${SSH_ARGS[@]}" "kill -0 -- '-$REMOTE_PID' 2>/dev/null" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  signal_remote_group KILL
  for attempt in {1..6}; do
    if ! ssh "${SSH_ARGS[@]}" "kill -0 -- '-$REMOTE_PID' 2>/dev/null" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

retrieve_stream_with_sha() {
  local remote_path=$1 local_path=$2 expected_sha actual_sha
  expected_sha=$(ssh_read_retry "sha256sum '$remote_path'" | awk '{print $1}')
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]]
  test ! -e "$local_path"
  test ! -e "$local_path.part"
  ssh "${SSH_ARGS[@]}" "cat '$remote_path'" >"$local_path.part"
  actual_sha=$(shasum -a 256 "$local_path.part" | awk '{print $1}')
  test "$actual_sha" = "$expected_sha"
  mv "$local_path.part" "$local_path"
  printf '%s\n' "$expected_sha"
}

retrieve_primary_result() {
  local local_result="$RESULT_ROOT/$RESULT_NAME" result_sha
  if [[ "$RESULT_RETRIEVED" -eq 1 ]]; then
    return 0
  fi
  result_sha=$(retrieve_stream_with_sha "$REMOTE_RESULT" "$local_result")
  mkdir "$EXTRACT_ROOT"
  tar -xzf "$local_result" -C "$EXTRACT_ROOT"
  test -d "$EXTRACT_ROOT/deliverable"
  (
    cd "$EXTRACT_ROOT/deliverable"
    shasum -a 256 --check result-manifest.sha256 >/dev/null
  )
  printf '%s\n' "$result_sha" >"$STATE_ROOT/result.sha256"
  RESULT_RETRIEVED=1
  record "결과 묶음 회수와 내부 manifest 검증: SHA-256 \`$result_sha\`"
}

retrieve_fallback_failure() {
  local local_failure="$RESULT_ROOT/issue623-vast-failure-v1.tar.gz" failure_sha
  if [[ "$SSH_READY" -ne 1 || -e "$local_failure" ]]; then
    return 0
  fi
  ssh_read_retry "
    set -Eeuo pipefail
    cd '$REMOTE_ROOT'
    entries=()
    for entry in state logs results recovery bootstrap.log input/execution-spec.json input/input-manifest.sha256; do
      if test -e \"\$entry\"; then entries+=(\"\$entry\"); fi
    done
    test \"\${#entries[@]}\" -gt 0
    tar -czf '$REMOTE_FAILURE_RESULT.part' \"\${entries[@]}\"
    mv '$REMOTE_FAILURE_RESULT.part' '$REMOTE_FAILURE_RESULT'
  "
  failure_sha=$(retrieve_stream_with_sha "$REMOTE_FAILURE_RESULT" "$local_failure")
  printf '%s\n' "$failure_sha" >"$STATE_ROOT/failure-result.sha256"
  record "대체 실패 자료 회수: SHA-256 \`$failure_sha\`"
}

recover_failure_evidence() {
  local result_probe_rc
  if [[ "$SSH_READY" -ne 1 || "$REMOTE_START_ATTEMPTED" -ne 1 || "$RESULT_RETRIEVED" -eq 1 ]]; then
    return 0
  fi
  set +e
  probe_remote_result_once >/dev/null 2>&1
  result_probe_rc=$?
  set -e
  case "$result_probe_rc" in
    0)
      retrieve_primary_result || retrieve_fallback_failure || true
      ;;
    1)
      retrieve_fallback_failure || true
      ;;
    *)
      record "실패 자료 확인 중 SSH 조회 실패: 종료 코드 \`$result_probe_rc\`"
      ;;
  esac
}

query_cost_summary() {
  local query_attempt charges filtered current_count status
  local current_complete current_signature last_current_signature stable_count
  local billing_end_date
  last_current_signature=
  stable_count=0
  billing_end_date=$(date -u -r $(( $(date +%s) + 86400 )) +%Y-%m-%d)
  for query_attempt in {1..10}; do
    charges=$(vast show invoices-v1 \
      --charges \
      --start-date "$BILLING_START_DATE" \
      --end-date "$billing_end_date" \
      --limit 100 \
      --latest-first \
      --raw) || return 1
    filtered=$(jq -c \
      --arg current_job "$JOB_ID" \
      --arg current_id "$COMPLETED_INSTANCE_ID" '
        def belongs($job; $id):
          ((.metadata.label // .metadata.instance_label // .label // "") == $job)
          or ((.source // "") == ("instance-" + $id))
          or (((.metadata.instance_id // .metadata.source_instance_id
                // .metadata.contract_id // .instance_id // .source_instance_id // "")
               | tostring) == $id);
        def summary($job; $id):
          [(.results // [])[] | select(belongs($job; $id))]
          | {
              job_id: $job,
              instance_id: ($id | tonumber),
              charged_item_count: length,
              total: (map(.amount | tonumber) | add // 0),
              charge_types: ([.[] | (.items // [])[] | .type // "unknown"] | unique),
              line_items: ([.[] | (.items // [])[] | {
                type: (.type // "unknown"),
                amount: (.amount | tonumber),
                description: (.description // "")
              }])
            };
        {
          schema_version: 1,
          queried_at: (now | todateiso8601),
          query: "show invoices-v1 --charges",
          current_attempt: summary($current_job; $current_id)
        }
      ' <<<"$charges") || return 1
    unset charges
    current_count=$(jq -r '.current_attempt.charged_item_count' <<<"$filtered") \
      || return 1
    current_complete=$(jq -r '
      (.current_attempt.charged_item_count > 0)
      and (.current_attempt.total > 0)
      and ((.current_attempt.charge_types | sort) == ["bwd", "bwu", "disk", "gpu"])
    ' <<<"$filtered") || return 1
    current_signature=$(jq -cS '
      .current_attempt
      | {charged_item_count, total, charge_types, line_items}
    ' <<<"$filtered") || return 1
    if [[ "$current_complete" == true && "$current_signature" == "$last_current_signature" ]]; then
      stable_count=$((stable_count + 1))
    elif [[ "$current_complete" == true ]]; then
      stable_count=1
    else
      stable_count=0
    fi
    last_current_signature=$current_signature
    status=pending
    if [[ "$current_count" -gt 0 && "$stable_count" -ge 5 ]]; then
      status=final
    fi
    jq --arg status "$status" \
      --argjson query_attempt "$query_attempt" \
      --argjson stability_observations "$stable_count" '
        . + {
          status: $status,
          query_attempt: $query_attempt,
          current_stability_observations: $stability_observations
        }
      ' <<<"$filtered" \
      >"$STATE_ROOT/cost-summary.json.part" || return 1
    mv "$STATE_ROOT/cost-summary.json.part" "$STATE_ROOT/cost-summary.json" \
      || return 1
    if [[ "$status" == final ]]; then
      return 0
    fi
    if [[ "$query_attempt" -lt 10 ]]; then
      sleep 30
    fi
  done
  return 1
}

cleanup() {
  local original_rc=$? cleanup_failed=0
  if [[ "$CLEANUP_STARTED" -eq 1 ]]; then
    return "$original_rc"
  fi
  CLEANUP_STARTED=1
  trap - EXIT
  set +e

  if [[ "$original_rc" -ne 0 && "$REMOTE_START_ATTEMPTED" -eq 1 ]]; then
    stop_remote_group || cleanup_failed=1
    recover_failure_evidence
  fi
  if [[ "$INSTANCE_ID" =~ ^[0-9]+$ ]]; then
    COMPLETED_INSTANCE_ID=$INSTANCE_ID
  fi
  destroy_job_resources || cleanup_failed=1
  if [[ "$COMPLETED_INSTANCE_ID" =~ ^[0-9]+$ || "$SCHEDULE_REGISTERED" -eq 1 ]]; then
    schedule_remove_if_present >/dev/null 2>&1 || cleanup_failed=1
  fi
  if [[ "$COMPLETED_INSTANCE_ID" =~ ^[0-9]+$ ]]; then
    query_cost_summary || record "청구 반영 지연: 청구 자료를 pending 상태로 보존"
  fi

  if [[ "$cleanup_failed" -ne 0 ]]; then
    record "정리 실패: 이 작업 자원 또는 종료 예약을 수동으로 확인해야 함"
    return 1
  fi
  record "정리 상태: 이 작업 계산 자원, 저장 공간과 종료 예약 부재"
  return "$original_rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

test "$(vastai --version)" = 1.5.4
test -x "$VAST_WRAPPER"
test -f "$VAST_CA_BUNDLE"
test -s "$INPUT_ARCHIVE"
test -s "$INPUT_ARCHIVE_SHA"
test -s "$INPUT_ROOT/execution-spec.json"
test -s "$INPUT_ROOT/expected-commit.txt"
test -s "$MAIN_ROOT/scripts/issue623/import_and_audit.py"
test -f "$SSH_KEY"
test -f "$SSH_PUBLIC_KEY"
test "$(stat -f '%Lp' "$SSH_KEY")" = 600
test "$(stat -f '%Lp' "$SSH_PUBLIC_KEY")" = 644
test "$(ssh-keygen -lf "$SSH_PUBLIC_KEY" | awk '{print $2}')" = "$SSH_FINGERPRINT"
EXPECTED_COMMIT=$(tr -d '[:space:]' <"$INPUT_ROOT/expected-commit.txt")
[[ $EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]]
test "$(git -C "$INPUT_ROOT/repo" rev-parse HEAD)" = "$EXPECTED_COMMIT"
jq -e \
  --arg job "$JOB_ID" \
  --arg image "$IMAGE" \
  --arg commit "$EXPECTED_COMMIT" '
    .issue == 623
    and .provider == "vast"
    and .job_id == $job
    and .image == $image
    and .git_commit == $commit
    and .stage == "confirm"
    and .seeds == [42, 43, 44]
    and (.experiments | length) == 4
  ' "$INPUT_ROOT/execution-spec.json" >/dev/null
touch "$EXCLUDED_MACHINES_FILE"

mkdir -p "$RESULT_ROOT" "$STATE_ROOT" "$RECOVERY_LOCAL_ROOT"
test ! -e "$RESULT_ROOT/$RESULT_NAME"
test ! -e "$RESULT_ROOT/$RESULT_NAME.part"
test ! -e "$EXTRACT_ROOT"
test ! -e "$RESULT_ROOT/import-audit.json"

input_sha=$(shasum -a 256 "$INPUT_ARCHIVE" | awk '{print $1}')
sidecar_sha=$(awk 'NR == 1 {print $1}' "$INPUT_ARCHIVE_SHA")
test "$input_sha" = "$sidecar_sha"

instances=$(vast show instances --raw)
jq -e --arg label "$JOB_ID" 'all(.[]; .label != $label)' <<<"$instances" >/dev/null
volumes=$(vast show volumes --raw)
jq -e --arg label "$JOB_ID" \
  'all(.[]; (((.name // .label // "") | tostring | contains($label)) | not))' \
  <<<"$volumes" >/dev/null
local_public_key=$(<"$SSH_PUBLIC_KEY")
remote_keys=$(vast show ssh-keys --raw)
jq -e --arg key "$local_public_key" '
  length == 1
  and ([.[] | select((.public_key // .ssh_key // .key // "") == $key)] | length == 1)
  and all(.[]; ((.public_key // .ssh_key // .key // "") | startswith("ssh-ed25519 ")))
  ' <<<"$remote_keys" >/dev/null
unset remote_keys
foreign_count=$(jq 'length' <<<"$instances")
record "계정 사전 점검: 이 작업 중복 자원과 저장 공간 없음, 대회 SSH 공개 키 정확 일치 1개, 다른 세션 인스턴스 \`$foreign_count\`개"

START_CREDIT=$(vast show user --raw | jq -er '.credit')
awk -v credit="$START_CREDIT" 'BEGIN { exit !(credit > 0) }'
record "시작 잔액: \`\$$START_CREDIT\`, 입력 묶음 SHA-256 \`$input_sha\`, 실행 커밋 \`$EXPECTED_COMMIT\`"
record "매물 규칙: 4 GPU RTX 4090 또는 RTX 3090, 한국·일본 우선(대만 차순위, 중국 제외), Runpod 크레딧 없음"

excluded_offers='[]'
excluded_machines=$(jq -c --argjson extra "$(jq -Rn '[inputs | select(length > 0) | tonumber]' <"$EXCLUDED_MACHINES_FILE")" '[.[].machine_id] + $extra' <<<"$instances")
for create_attempt in {1..5}; do
  offer_json=$(
    vast search offers \
      'num_gpus=4 verified=true rentable=true rented=false reliability>=0.99 gpu_ram>=20 cuda_max_good>=13.0 direct_port_count>=2 duration>1 inet_down>=200 disk_space>=80 cpu_cores_effective>=16' \
      --type on-demand \
      --storage 60 \
      --order dph_total \
      --limit 200 \
      --raw \
      | jq -cer --argjson excluded "$excluded_offers" --argjson machines "$excluded_machines" '
          [.[]
           | select(.gpu_name == "RTX 4090" or .gpu_name == "RTX 3090")
           | select((.geolocation // "" | tostring | test("CN$")) | not)
           | .id as $id | select(($excluded | index($id)) == null)
           | .machine_id as $machine | select(($machines | index($machine)) == null)
           | . + {
               geography_rank:
                 (if ((.geolocation // "" | tostring) | test("KR$|JP$")) then 0
                  elif ((.geolocation // "" | tostring) | test("TW$")) then 1
                  else 2 end)
             }]
          | sort_by(.geography_rank, .dph_total, -(.reliability))
          | .[0]
        '
  )
  offer_id=$(jq -er '.id' <<<"$offer_json")
  offer_gpu=$(jq -er '.gpu_name' <<<"$offer_json")
  offer_gpu_ram=$(jq -er '.gpu_ram' <<<"$offer_json")
  offer_price=$(jq -er '.dph_total' <<<"$offer_json")
  offer_host=$(jq -er '.host_id' <<<"$offer_json")
  offer_machine=$(jq -er '.machine_id' <<<"$offer_json")
  offer_reliability=$(jq -er '.reliability' <<<"$offer_json")
  offer_cuda=$(jq -er '.cuda_max_good' <<<"$offer_json")
  offer_cpu=$(jq -r '.cpu_cores_effective' <<<"$offer_json")
  offer_geo=$(jq -r '.geolocation // "unknown"' <<<"$offer_json")

  before_create_instances=$(vast show instances --raw)
  before_create_ids=$(jq -c '[.[].id]' <<<"$before_create_instances")

  set +e
  create_result=$(
    vast create instance "$offer_id" \
      --image "$IMAGE" \
      --disk 60 \
      --ssh \
      --direct \
      --cancel-unavail \
      --label "$JOB_ID" \
      --raw 2>"$STATE_ROOT/create-attempt-$create_attempt.log"
  )
  create_rc=$?
  set -e
  if [[ "$create_rc" -eq 0 ]] && INSTANCE_ID=$(jq -er '.new_contract' <<<"$create_result"); then
    unset create_result
    break
  fi
  INSTANCE_ID=
  unset create_result

  uncertainty_query_count=0
  candidate_instances='[]'
  for lookup_attempt in {1..30}; do
    set +e
    observed_candidate_instances=$(vast show instances --raw \
      | jq -c \
          --arg label "$JOB_ID" \
          --argjson machine "$offer_machine" \
          --argjson before "$before_create_ids" '
          [.[ ]
             | .id as $id
             | select(
                 .label == $label
                 or (($before | index($id)) == null and .machine_id == $machine)
               )]
          ')
    candidate_lookup_rc=$?
    set -e
    if [[ "$candidate_lookup_rc" -eq 0 ]]; then
      uncertainty_query_count=$((uncertainty_query_count + 1))
      candidate_instances=$observed_candidate_instances
      if [[ "$(jq 'length' <<<"$candidate_instances")" -gt 0 ]]; then
        break
      fi
    fi
    if [[ "$lookup_attempt" -lt 30 ]]; then
      sleep 10
    fi
  done
  candidate_count=$(jq 'length' <<<"$candidate_instances")
  if [[ "$candidate_count" -eq 0 && "$uncertainty_query_count" -lt 20 ]]; then
    record "생성 응답 불확실 뒤 5분 재조회가 충분히 성공하지 않아 추가 생성 없이 중단"
    exit 1
  fi
  if [[ "$candidate_count" -eq 1 ]]; then
    INSTANCE_ID=$(jq -er '.[0].id' <<<"$candidate_instances")
    record "생성 응답 불확실성을 생성 전후 ID와 장비 재조회로 복구해 인스턴스 \`$INSTANCE_ID\` 채택"
    break
  fi
  if [[ "$candidate_count" -gt 1 ]]; then
    while IFS= read -r duplicate_id; do
      destroy_and_confirm "$duplicate_id" || true
    done < <(jq -r '.[].id' <<<"$candidate_instances")
    record "생성 응답 불확실 뒤 나타난 중복 인스턴스를 모두 정리하고 중단"
    exit 1
  fi
  excluded_offers=$(jq -c --argjson id "$offer_id" '. + [$id]' <<<"$excluded_offers")
  excluded_machines=$(jq -c --argjson id "$offer_machine" '. + [$id]' <<<"$excluded_machines")
  record "생성 전후 ID 차집합을 5분간 확인해 계약 부재를 확정: 매물 \`$offer_id\`, 다른 장비를 새로 조회"
  sleep 2
done
test "$INSTANCE_ID" -gt 0
COMPLETED_INSTANCE_ID=$INSTANCE_ID
printf '%s\n' "$INSTANCE_ID" >"$STATE_ROOT/instance-id.txt"
record "인스턴스 생성: \`$INSTANCE_ID\`, 매물 \`$offer_id\`, 호스트 \`$offer_host\`, 장비 \`$offer_machine\`, 위치 \`$offer_geo\`"
record "매물 조건: \`$offer_gpu\` x4 \`$offer_gpu_ram\`MB, 유효 CPU \`$offer_cpu\`, CUDA 상한 \`$offer_cuda\`, 신뢰도 \`$offer_reliability\`, 시간당 총액 \`\$$offer_price\`"

guard_terminate_at=$(date -u -r $(( $(date +%s) + GUARD_TERMINATE_SECONDS )) +%FT%TZ)
if schedule_add "$INSTANCE_ID" "$guard_terminate_at"; then
  SCHEDULE_REGISTERED=1
  record "GitHub Actions 종료 예약과 기존 항목 보존 재조회: \`$guard_terminate_at\`"
else
  record "GitHub Actions 종료 예약 등록 또는 기존 항목 보존 재조회 실패, 실행 전 정리"
  exit 1
fi

created_epoch=$(date +%s)
running_at=
previous_status=
previous_message=
last_status_change_epoch=$created_epoch
for attempt in {1..180}; do
  instance_json=$(vast show instance "$INSTANCE_ID" --raw)
  status=$(jq -r '.actual_status // .status // ""' <<<"$instance_json")
  status_message=$(jq -r '.status_msg // ""' <<<"$instance_json" | tail -c 180 | tr '\n' ' ')
  if [[ "$status" != "$previous_status" || "$status_message" != "$previous_message" ]]; then
    record "상태: \`$status\`, \`$status_message\`"
    previous_status=$status
    previous_message=$status_message
    last_status_change_epoch=$(date +%s)
  fi
  if [[ "$status" == running ]]; then
    running_at=$(date -u +%FT%TZ)
    break
  fi
  if (( $(date +%s) - last_status_change_epoch >= 1800 )); then
    record "시작 실패: 상태와 준비 문구가 30분 동안 바뀌지 않음"
    exit 1
  fi
  if (( $(date +%s) - created_epoch >= 5400 )); then
    record "시작 실패: 준비 단계 90분 상한 도달"
    exit 1
  fi
  sleep 30
done
test -n "$running_at"
MANUAL_DEADLINE=$(( $(date +%s) + MANUAL_RUNTIME_SECONDS ))
manual_terminate_at=$(date -u -r "$MANUAL_DEADLINE" +%FT%TZ)
printf '%s\n' "$manual_terminate_at" >"$STATE_ROOT/manual-terminate-at.txt"
record "실행 상태 확인: \`$running_at\`, 수동 종료 시각 \`$manual_terminate_at\`"

instance_json=$(vast show instance "$INSTANCE_ID" --raw)
test "$(jq -r '.gpu_name' <<<"$instance_json")" = "$offer_gpu"
jq -e '.num_gpus == 4' <<<"$instance_json" >/dev/null
current_cuda=$(jq -er '.cuda_max_good' <<<"$instance_json")
awk -v cuda="$current_cuda" 'BEGIN { exit !(cuda >= 13.0) }'

ssh_url=$(vast ssh-url "$INSTANCE_ID")
[[ "$ssh_url" == ssh://root@*:* ]]
ssh_endpoint=${ssh_url#ssh://root@}
ssh_host=${ssh_endpoint%:*}
ssh_port=${ssh_endpoint##*:}
[[ "$ssh_port" =~ ^[0-9]+$ ]]
known_hosts="$STATE_ROOT/known-hosts"
host_key_deadline=$(( $(date +%s) + 300 ))
while (( $(date +%s) < host_key_deadline )); do
  if ssh-keyscan -T 10 -p "$ssh_port" "$ssh_host" >"$known_hosts" 2>/dev/null; then
    break
  fi
  sleep 10
done
test -s "$known_hosts"

readonly -a SSH_ARGS=(
  -i "$SSH_KEY"
  -p "$ssh_port"
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=yes
  -o UserKnownHostsFile="$known_hosts"
  -o ConnectTimeout=12
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
  "root@$ssh_host"
)

ssh_authenticated=0
ssh_key_attached=0
ssh_tmux_recovered=0
for recovery_round in {1..3}; do
  ssh_auth_deadline=$(( $(date +%s) + 300 ))
  while (( $(date +%s) < ssh_auth_deadline )); do
    if ssh "${SSH_ARGS[@]}" true >"$STATE_ROOT/ssh-last-output.txt" 2>&1; then
      ssh_authenticated=1
      break
    fi
    sleep 10
  done
  if [[ "$ssh_authenticated" -eq 1 ]]; then
    break
  fi
  if grep -Eqi 'no sessions|open terminal failed: not a terminal|duplicate session: ssh_tmux' \
    "$STATE_ROOT/ssh-last-output.txt" && [[ "$ssh_tmux_recovered" -eq 0 ]]
  then
    ssh_tmux_recovered=1
    set +e
    printf 'touch ~/.no_auto_tmux\nexit\n' \
      | TERM=xterm-256color ssh -tt "${SSH_ARGS[@]}" \
        >"$STATE_ROOT/ssh-tmux-recovery.txt" 2>&1
    tmux_recovery_rc=$?
    set -e
    record "SSH 자동 tmux 초기화 복구 호출: 종료 코드 \`$tmux_recovery_rc\`"
    continue
  fi
  if grep -Eqi 'Permission denied \(publickey\)' "$STATE_ROOT/ssh-last-output.txt" \
    && [[ "$ssh_key_attached" -eq 0 ]]
  then
    ssh_key_attached=1
    vast attach ssh "$INSTANCE_ID" "$SSH_PUBLIC_KEY" --raw >/dev/null
    record "인스턴스에 기존 대회 SSH 공개 키를 다시 적용"
    continue
  fi
  break
done
test "$ssh_authenticated" -eq 1
SSH_READY=1
record "SSH 인증: 독립 5분 관문 통과, 작업별 호스트 키 고정"

# 외부 GPU 부하 관문: 입력 전송 전에 다른 GPU 프로세스나 부하가 보이면 장비를 제외하고 종료 코드 75로 나간다(#483 v1 사례).
foreign_apps=$(ssh_read_retry "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'")
gpu_util=$(ssh_read_retry "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ','")
printf 'compute_apps=%s\nutilization=%s\n' "$foreign_apps" "$gpu_util" >"$STATE_ROOT/gpu-foreign-load.txt"
if [[ -n "$foreign_apps" ]] || awk -v s="$gpu_util" 'BEGIN { n = split(s, a, ","); for (i = 1; i <= n; i++) if (a[i] != "" && a[i] + 0 > 10) exit 1; exit 0 }'; then
  :
else
  foreign_apps=busy
fi
if [[ -n "$foreign_apps" ]]; then
  printf '%s\n' "$offer_machine" >>"$EXCLUDED_MACHINES_FILE"
  record "공급 환경 실패: 입력 전송 전 외부 GPU 프로세스 또는 부하 감지(\`$foreign_apps\` / 사용률 \`$gpu_util\`), 장비 \`$offer_machine\` 제외 후 정리"
  exit 75
fi
record "외부 GPU 부하 관문: GPU 프로세스 없음, 사용률 \`$gpu_util\`"

ssh_read_retry \
  "test ! -e '/tmp/issue623-python-gate-v1'; \
   python3 -m venv '/tmp/issue623-python-gate-v1'; \
   '/tmp/issue623-python-gate-v1/bin/python' -m pip --version >/dev/null; \
   rm -rf '/tmp/issue623-python-gate-v1'"
ssh_read_retry "nvidia-smi -L" >"$STATE_ROOT/nvidia-smi-list.txt"
ssh_read_retry \
  "python3 -c 'import json, torch; assert torch.cuda.is_available(); n = torch.cuda.device_count(); assert n >= 4; layer = torch.nn.Linear(8, 4).cuda(); x = torch.ones(2, 8, device=\"cuda\", requires_grad=True); loss = layer(x).square().mean(); loss.backward(); assert x.grad is not None; print(json.dumps({\"torch\": torch.__version__, \"cuda\": torch.version.cuda, \"available\": torch.cuda.is_available(), \"devices\": n, \"device\": torch.cuda.get_device_name(0)}, sort_keys=True))'" \
  >"$STATE_ROOT/gpu-python.json"
jq -e '.available == true and .devices >= 4 and (.cuda | startswith("13."))' \
  "$STATE_ROOT/gpu-python.json" >/dev/null
record "원격 Python 가상환경과 CUDA 순전파 및 역전파 관문(GPU 4장): 통과"

transfer_started=$(date +%s)
ssh "${SSH_ARGS[@]}" \
  "umask 077; test ! -e '$REMOTE_ROOT'; mkdir -p '$REMOTE_ROOT/state'; cat > '$REMOTE_ROOT/input.tar.gz.part'" \
  <"$INPUT_ARCHIVE"
remote_input_sha=$(ssh_read_retry "sha256sum '$REMOTE_ROOT/input.tar.gz.part'" | awk '{print $1}')
test "$remote_input_sha" = "$input_sha"
ssh "${SSH_ARGS[@]}" \
  "set -Eeuo pipefail; \
   mv '$REMOTE_ROOT/input.tar.gz.part' '$REMOTE_ROOT/input.tar.gz'; \
   test ! -e '$REMOTE_ROOT/input'; \
   mkdir '$REMOTE_ROOT/input'; \
   tar --no-same-owner --no-same-permissions --warning=no-unknown-keyword \
     --exclude='._*' --exclude='*/._*' \
     -xzf '$REMOTE_ROOT/input.tar.gz' -C '$REMOTE_ROOT/input'; \
   cd '$REMOTE_ROOT/input'; \
   sha256sum --check --quiet input-manifest.sha256; \
   bash -n remote-start.sh remote-job.sh"
record "입력 전송: SHA-256 \`$input_sha\` 일치, 파일별 해시와 원격 셸 문법 검증 통과, 전송 \`$(( $(date +%s) - transfer_started ))\`초"

REMOTE_START_ATTEMPTED=1
set +e
ssh "${SSH_ARGS[@]}" "
  set -Eeuo pipefail
  umask 077
  mkdir '$REMOTE_STATE/start.lock'
  nohup setsid bash -c '
    set -Eeuo pipefail
    printf \"%s\\n\" \"\$\$\" > \"$REMOTE_STATE/supervisor.pid.part\"
    mv \"$REMOTE_STATE/supervisor.pid.part\" \"$REMOTE_STATE/supervisor.pid\"
    exec env VAST_RUN_INSTANCE_ID=$INSTANCE_ID bash \"$REMOTE_ROOT/input/remote-start.sh\"
  ' >'$REMOTE_ROOT/bootstrap.log' 2>&1 < /dev/null &
" >"$STATE_ROOT/remote-start-output.txt" 2>&1
remote_start_rc=$?
set -e
printf '%s\n' "$remote_start_rc" >"$STATE_ROOT/remote-start-ssh-rc.txt"
record "원격 시작 단일 호출 완료: SSH 종료 코드 \`$remote_start_rc\`, 재시도하지 않음"

start_marker_deadline=$(( $(date +%s) + 180 ))
while (( $(date +%s) < start_marker_deadline )); do
  snapshot=$(remote_snapshot)
  REMOTE_PID=$(awk -F= '$1 == "pid" {print $2}' <<<"$snapshot")
  remote_started=$(awk -F= '$1 == "started" {print $2}' <<<"$snapshot")
  remote_failed=$(awk -F= '$1 == "failed" {print $2}' <<<"$snapshot")
  remote_finished=$(awk -F= '$1 == "finished" {print $2}' <<<"$snapshot")
  remote_alive=$(awk -F= '$1 == "alive" {print $2}' <<<"$snapshot")
  if [[ "$REMOTE_PID" =~ ^[0-9]+$ ]] \
    && [[ "$remote_started" -eq 1 || "$remote_failed" -eq 1 || "$remote_finished" -eq 1 ]]
  then
    break
  fi
  sleep 5
done
[[ "$REMOTE_PID" =~ ^[0-9]+$ ]]
test "$remote_started" -eq 1 || test "$remote_failed" -eq 1 || test "$remote_finished" -eq 1
printf '%s\n' "$REMOTE_PID" >"$STATE_ROOT/remote-supervisor-pid.txt"
if [[ "$remote_alive" -eq 1 ]]; then
  remote_pgid=$(ssh_read_retry "ps -o pgid= -p '$REMOTE_PID' | tr -d ' '")
  test "$remote_pgid" = "$REMOTE_PID"
  record "원격 감독 프로세스 시작: PID와 프로세스 그룹 \`$REMOTE_PID\`, 원자적 표식 확인"
else
  record "원격 감독 프로세스가 시작 표식 확인 전에 끝나 최종 표식과 결과 묶음을 확인"
fi

poll_count=0
consecutive_probe_errors=0
final_marker_seen_at=0
dead_process_seen_at=0
while true; do
  if (( $(date +%s) >= MANUAL_DEADLINE )); then
    record "수동 실행 8시간 상한 도달, 원격 프로세스 그룹에 TERM 후 5분 뒤 KILL 적용"
    stop_remote_group || true
    recover_failure_evidence
    exit 124
  fi

  set +e
  probe_remote_result_once >/dev/null 2>&1
  result_probe_rc=$?
  set -e
  case "$result_probe_rc" in
    0)
      break
      ;;
    1)
      consecutive_probe_errors=0
      ;;
    *)
      consecutive_probe_errors=$((consecutive_probe_errors + 1))
      record "결과 단일 확인에서 SSH 오류: 종료 코드 \`$result_probe_rc\`, 연속 \`$consecutive_probe_errors\`회"
      if [[ "$consecutive_probe_errors" -ge 3 ]]; then
        exit 1
      fi
      sleep 10
      continue
      ;;
  esac

  if ! vast show instances --raw \
    | jq -e --argjson id "$INSTANCE_ID" 'any(.[]; .id == $id)' >/dev/null
  then
    record "실행 중 대상 인스턴스가 사라짐"
    exit 1
  fi

  snapshot=$(remote_snapshot)
  remote_started=$(awk -F= '$1 == "started" {print $2}' <<<"$snapshot")
  remote_finished=$(awk -F= '$1 == "finished" {print $2}' <<<"$snapshot")
  remote_failed=$(awk -F= '$1 == "failed" {print $2}' <<<"$snapshot")
  remote_alive=$(awk -F= '$1 == "alive" {print $2}' <<<"$snapshot")
  remote_exit_code=$(awk -F= '$1 == "exit_code" {print $2}' <<<"$snapshot")

  if [[ "$remote_failed" -eq 1 || "$remote_finished" -eq 1 ]]; then
    if [[ "$final_marker_seen_at" -eq 0 ]]; then
      final_marker_seen_at=$(date +%s)
    elif (( $(date +%s) - final_marker_seen_at >= 120 )); then
      record "완료 또는 실패 표식 뒤 2분 동안 결과 묶음이 생성되지 않음"
      exit 1
    fi
  fi
  if [[ "$remote_alive" -eq 0 ]]; then
    if [[ "$dead_process_seen_at" -eq 0 ]]; then
      dead_process_seen_at=$(date +%s)
    elif (( $(date +%s) - dead_process_seen_at >= 60 )); then
      record "원격 감독 프로세스가 결과 묶음 없이 종료됨"
      exit 1
    fi
  else
    dead_process_seen_at=0
  fi

  if (( poll_count % 10 == 0 )); then
    progress=$(ssh_read_retry \
      "printf 'state_started=$remote_started state_finished=$remote_finished state_failed=$remote_failed exit_code=$remote_exit_code '; \
       printf 'recovery_manifests='; find '$REMOTE_ROOT/recovery' -name manifest.json 2>/dev/null | wc -l; \
       for exp in ${EXPERIMENTS[*]}; do \
         printf ' %s=' \"\$exp\"; \
         printf 'folds:'; find '$REMOTE_ROOT/recovery/'\"\$exp\"'-confirm' -name manifest.json 2>/dev/null | wc -l | tr -d '\n'; \
         printf '/bytes:'; wc -c < '$REMOTE_ROOT/logs/pipeline-'\"\$exp\"'.log' 2>/dev/null | tr -d '\n' || printf 0; \
       done" \
      | tr '\n' ' ' | tail -c 600)
    record "진행: \`$progress\`"
  fi
  if (( poll_count > 0 && poll_count % 20 == 0 )); then
    if ssh_read_retry \
      "if test -d '$REMOTE_ROOT/recovery'; then \
         tar -czf '$REMOTE_ROOT/recovery-live.tar.gz.part' -C '$REMOTE_ROOT' recovery \
         && mv '$REMOTE_ROOT/recovery-live.tar.gz.part' '$REMOTE_ROOT/recovery-live.tar.gz'; \
       fi"
    then
      if recovery_sha=$(ssh_read_retry "sha256sum '$REMOTE_ROOT/recovery-live.tar.gz'" 2>/dev/null | awk '{print $1}'); then
        recovery_local="$RECOVERY_LOCAL_ROOT/recovery-$(date -u +%Y%m%dT%H%M%SZ)-${poll_count}.tar.gz"
        test ! -e "$recovery_local"
        ssh "${SSH_ARGS[@]}" "cat '$REMOTE_ROOT/recovery-live.tar.gz'" >"$recovery_local.part"
        test "$(shasum -a 256 "$recovery_local.part" | awk '{print $1}')" = "$recovery_sha"
        mv "$recovery_local.part" "$recovery_local"
        record "중간 복구 자료 회수: SHA-256 \`$recovery_sha\`"
      fi
    fi
  fi
  poll_count=$((poll_count + 1))
  sleep 30
done

final_state_deadline=$(( $(date +%s) + 120 ))
remote_finished=0
remote_failed=0
remote_exit_code=
while (( $(date +%s) < final_state_deadline )); do
  snapshot=$(remote_snapshot)
  remote_finished=$(awk -F= '$1 == "finished" {print $2}' <<<"$snapshot")
  remote_failed=$(awk -F= '$1 == "failed" {print $2}' <<<"$snapshot")
  remote_exit_code=$(awk -F= '$1 == "exit_code" {print $2}' <<<"$snapshot")
  if [[ "$remote_exit_code" =~ ^[0-9]+$ ]] \
    && [[ "$remote_finished" -eq 1 || "$remote_failed" -eq 1 ]]
  then
    break
  fi
  sleep 2
done
[[ "$remote_exit_code" =~ ^[0-9]+$ ]]
printf '%s\n' "$remote_exit_code" >"$STATE_ROOT/remote-exit-code.txt"
retrieve_primary_result
record "원격 종료 상태: 완료 표식 \`$remote_finished\`, 실패 표식 \`$remote_failed\`, 종료 코드 \`$remote_exit_code\`"
if [[ "$remote_exit_code" -ne 0 || "$remote_failed" -eq 1 ]]; then
  record "실패 결과 묶음을 보존했으며 main MLflow 반입은 수행하지 않음"
  exit 1
fi
test "$remote_finished" -eq 1
test "$remote_failed" -eq 0

deliverable="$EXTRACT_ROOT/deliverable"
bundle_args=()
for experiment in "${EXPERIMENTS[@]}"; do
  test -s "$deliverable/results/$experiment.bundle.zip"
  bundle_args+=(--bundle "$deliverable/results/$experiment.bundle.zip")
done
(
  cd "$MAIN_ROOT"
  env -u VAST_API_KEY uv run --frozen python scripts/issue623/import_and_audit.py \
    "${bundle_args[@]}" \
    --out "$RESULT_ROOT/import-audit.json" \
    --expected-commit "$EXPECTED_COMMIT" \
    --remote-job-id "$JOB_ID" \
    --remote-provider vast \
    >"$RESULT_ROOT/import.log" 2>&1
)
jq -e \
  --arg commit "$EXPECTED_COMMIT" '
    (.runs | length) == 4
    and all(.runs[]; .git_commit == $commit and .prediction_integrity_pass == true and (.main_run_id | test("^[0-9a-f]{32}$")))
  ' "$RESULT_ROOT/import-audit.json" >/dev/null
record "main MLflow 반입, 재조회와 예측 무결성 감사 관문: 통과, $(jq -r '[.runs[] | "\(.experiment)=\(.main_run_id[0:8]) OOF \(.auc_oof)"] | join(", ")' "$RESULT_ROOT/import-audit.json")"

destroy_job_resources
schedule_remove
SCHEDULE_REGISTERED=0
vast show instances --raw \
  | jq -e --arg label "$JOB_ID" 'all(.[]; .label != $label)' >/dev/null
vast show volumes --raw \
  | jq -e --arg job "$JOB_ID" --arg id "$INSTANCE_ID" '
      all(.[ ];
        (((.name // .label // "") | tostring | contains($job)) | not)
        and (((.instance_id // .contract_id // "") | tostring) != $id))
    ' >/dev/null
observed_schedule=$(schedule_value)
jq -e --arg job "$JOB_ID" 'all(.[]; .job_id != $job)' <<<"$observed_schedule" >/dev/null

final_credit=$(vast show user --raw | jq -er '.credit')
balance_delta=$(awk -v start="$START_CREDIT" -v final="$final_credit" \
  'BEGIN { printf "%.9f", start-final }')
if query_cost_summary; then
  current_cost=$(jq -r '.current_attempt.total' "$STATE_ROOT/cost-summary.json")
  record "invoices-v1 지연 청구 재조회 확정: \`\$$current_cost\`"
else
  record "청구 반영 지연: 청구 자료를 pending 상태로 보존"
fi
remaining_instances=$(vast show instances --raw | jq 'length')
remaining_volumes=$(vast show volumes --raw | jq 'length')
record "인스턴스 \`$INSTANCE_ID\`와 작업 전용 저장 공간 부재, 종료 예약 제거 및 기존 항목 보존 재조회"
record "종료 잔액: \`\$$final_credit\`, 동시 작업 포함 잔액 감소 \`\$$balance_delta\`"
record "계정의 다른 작업 자원: 계산 자원 \`$remaining_instances\`개, 저장 공간 \`$remaining_volumes\`개"
record "원격 운영 종료: 이 작업 계산 자원과 저장 공간 부재"

INSTANCE_ID=
CLEANUP_STARTED=1
