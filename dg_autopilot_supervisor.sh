#!/usr/bin/env bash
# Invoked by start_dg_autopilot_tmux.sh; do not run manually.
set -euo pipefail
STOP_METHOD="${1:?}"; POLL="${2:?}"; STATE="${3:?}"; LOG_DIR="${4:?}"
shift 4
MULTI_SESSIONS=("$@")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Informational reports only. Default 3600 seconds (one hour).
PROGRESS_INTERVAL="${DG_PROGRESS_INTERVAL_SECONDS:-3600}"
[[ "$PROGRESS_INTERVAL" =~ ^[0-9]+$ ]] && (( PROGRESS_INTERVAL >= 1 )) || { echo 'Invalid DG_PROGRESS_INTERVAL_SECONDS' >&2; exit 2; }
LAST_PROGRESS_AT="$(date +%s)"
progress_report() {
  python "$ROOT/dg_autopilot_progress.py" --project-root "$ROOT" --stage "$1" || say 'Progress report unavailable; training continues normally.'
}
maybe_progress_report() {
  local phase="$1" now
  now="$(date +%s)"
  if (( now - LAST_PROGRESS_AT >= PROGRESS_INTERVAL )); then
    progress_report "$phase"
    LAST_PROGRESS_AT="$now"
  fi
}
SELF='dg_autopilot'
TUNING='dg_siteval_all'

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
set_state() { printf '%s\n' "$1" > "$STATE"; say "State: $1"; }
die() { set_state "FAILED: $*"; say "ERROR: $*"; exit 1; }

# Use no unpredictable checks of process status: retain tmux pane after exit.
pane_result() {
  local s="$1" result
  if ! tmux has-session -t "=$s" 2>/dev/null; then echo MISSING; return; fi
  result="$(tmux list-panes -t "=$s:0" -F '#{pane_dead} #{pane_dead_status}' 2>/dev/null | head -n 1)" || { echo MISSING; return; }
  case "$result" in
    '0 '*) echo RUNNING ;;
    '1 0') echo SUCCESS ;;
    '1 '*) echo "ERROR:${result#1 }" ;;
    *) echo "ERROR:$result" ;;
  esac
}

# Check for new running tmux sessions besides this supervisor and handled jobs.
other_sessions() {
  local s
  while IFS= read -r s; do
    [[ -z "$s" || "$s" == "$SELF" ]] && continue
    local known=0 x
    for x in "${MULTI_SESSIONS[@]}"; do [[ "$x" == "$s" ]] && known=1; done
    (( known )) || printf '%s\n' "$s"
  done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)
}

wait_for_no_others() {
  local extras
  while :; do
    extras="$(other_sessions)"
    [[ -z "$extras" ]] && return 0
    say "Waiting for unrelated tmux session(s) to finish before using GPU/shutdown: $(echo "$extras" | tr '\n' ' ')"
    sleep "$POLL"
  done
}

# Conservatively check for GPU compute processes before shutting down.
no_gpu_jobs() {
  if ! command -v nvidia-smi >/dev/null; then say 'Cannot find nvidia-smi; refusing auto-stop for safety.'; return 1; fi
  local out
  out="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)" || {
    say "nvidia-smi query failed: $out"; return 1;
  }
  [[ -z "$(printf '%s' "$out" | tr -d '[:space:]')" ]]
}

verify_final_csv() {
  local csv="$ROOT/artifacts/results/siteholdout_final/all_final_runs.csv"
  [[ -s "$csv" ]] || return 1
  python - "$csv" <<'PY'
import csv, sys, itertools, math
p = sys.argv[1]
with open(p, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
sites={18,1,4,13,20}
seeds={100,101,102,103,104}
methods={'erm','mixstyle','csu','mixstyle_kl','mixstyle_kl_mmd','mixstyle_multi'}
expected=set(itertools.product(sites,methods,seeds))
actual=[]
for row in rows:
    try:
        key=(int(row['site']), row['method'], int(row['train_seed']))
        v=float(row['ood_bal_acc'])
    except (ValueError,KeyError):
        print('Invalid final CSV row',row,file=sys.stderr);sys.exit(1)
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        print('Invalid OOD bal acc:',row,file=sys.stderr);sys.exit(1)
    actual.append(key)
if set(actual) != expected or len(actual) != len(expected):
    print(f'Final CSV incomplete/duplicate: {len(actual)} rows, require {len(expected)} unique site/method/seed rows',file=sys.stderr)
    sys.exit(1)
print(f'Validated all {len(expected)} final OOD results in {p}')
PY
}

# Stop policy is applied BOTH after success and after failure.  An unrelated
# running tmux/GPU process blocks stop to avoid interrupting someone else's job.
ERROR_DIR="$LOG_DIR/errors"
mkdir -p "$ERROR_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"

record_failure() {
  local reason="$1" stage="$2" session="$3" terminal_log="${4:-}"
  local file="$ERROR_DIR/autopilot_error_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
  {
    echo "UTC: $(date -u '+%FT%TZ')"
    echo "Failed stage: $stage"
    echo "Failed tmux session: $session"
    echo "Reason: $reason"
    echo "Terminal log: ${terminal_log:-none}"
    echo; echo '=== LAST 200 LINES OF TERMINAL OUTPUT ==='
    if [[ -n "$terminal_log" && -f "$terminal_log" ]]; then tail -n 200 "$terminal_log"; else echo '(no terminal log available)'; fi
    echo; echo '=== LAST 80 LINES OF PIPELINE OUTPUT ==='
    if [[ -f "$PIPELINE_LOG" ]]; then tail -n 80 "$PIPELINE_LOG"; fi
  } > "$file"
  say "Failure recorded: $file"
  set_state "FAILED_LOGGED: $file"
}

stop_instance() {
  # Explicitly stop, do not terminate. Permission/shutdown behavior checked
  # by start_dg_autopilot_tmux.sh BEFORE the automation begins.
  sync
  if [[ "$STOP_METHOD" == os ]]; then
    say 'Requesting verified OS shutdown (EC2 shutdown behavior must be Stop).'
    sudo -n shutdown -h now
  else
    local token id az
    token="$(curl -fsS --connect-timeout 2 --max-time 5 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 'http://169.254.169.254/latest/api/token')" || return 1
    id="$(curl -fsS --connect-timeout 2 --max-time 5 -H "X-aws-ec2-metadata-token: $token" 'http://169.254.169.254/latest/meta-data/instance-id')" || return 1
    az="$(curl -fsS --connect-timeout 2 --max-time 5 -H "X-aws-ec2-metadata-token: $token" 'http://169.254.169.254/latest/meta-data/placement/availability-zone')" || return 1
    [[ "$id" =~ ^i-[0-9a-f]+$ && "$az" =~ ^[a-z0-9-]+[a-z]$ ]] || return 1
    say "Requesting ec2 stop-instances for $id in ${az::-1}"
    aws ec2 stop-instances --instance-ids "$id" --region "${az::-1}" --no-cli-pager
  fi
}

safe_shutdown() {
  local why="$1" extras
  set_state "WAITING_FOR_IDLE_${why}"
  # It is unsafe to stop while user-started GPU training is still active.
  # This loop deliberately waits if another session/job exists. It logs the
  # blocker, rather than silently shutting down independent work.
  while :; do
    extras="$(other_sessions)"
    if [[ -n "$extras" ]]; then
      say "EC2 stop blocked by unrelated tmux sessions: $(echo "$extras" | tr '\n' ' ')"
      sleep "$POLL"
      continue
    fi
    if ! no_gpu_jobs; then
      say 'EC2 stop blocked by GPU compute jobs (or nvidia-smi failure).'
      sleep "$POLL"
      continue
    fi
    break
  done
  # Explicit grace period catches a new tmux/GPU job started moments ago.
  say "No other tmux/GPU jobs. 30-second shutdown grace period ($why)."
  sleep 30
  if [[ -n "$(other_sessions)" ]] || ! no_gpu_jobs; then
    say 'A new tmux/GPU job started during grace period; restarting idle checks.'
    safe_shutdown "$why"
    return
  fi
  set_state "STOP_REQUESTED_${why}"
  if ! stop_instance; then
    set_state 'STOP_COMMAND_FAILED'
    say 'ERROR: EC2 stop request failed. Inspect AWS IAM or shutdown settings immediately.'
    exit 1
  fi
  say 'EC2 stop requested; verify instance status in the AWS console.'
}

fail_and_stop() {
  local reason="$1" stage="$2" session="$3" termlog="${4:-}"
  say "FAILURE: $reason"
  record_failure "$reason" "$stage" "$session" "$termlog"
  # For an already failed/dead tmux pane, remove its retained session so
  # it cannot accidentally block a legitimate safe shutdown.
  if [[ -n "$session" && "$session" != none ]] && tmux has-session -t "=$session" 2>/dev/null; then
    local state
    state="$(pane_result "$session")"
    if [[ "$state" != RUNNING ]]; then tmux kill-session -t "=$session" || true; fi
  fi
  safe_shutdown AFTER_FAILURE
  exit 1
}

set_state WAITING_FOR_MULTI
progress_report waiting_multi
for session in "${MULTI_SESSIONS[@]}"; do
  say "Waiting for existing MixStyle + Multi session $session"
  while :; do
    status="$(pane_result "$session")"
    case "$status" in
      RUNNING) sleep "$POLL"; maybe_progress_report waiting_multi ;;
      SUCCESS) say "Finished successfully: $session"; break ;;
      *)
        # Multi launcher already writes a terminal log in multiscale_runs.
        multi_log="$(find "$ROOT/artifacts/logs/multiscale_runs" -maxdepth 1 -name "${session}*.log" -type f 2>/dev/null | sort | tail -1)"
        fail_and_stop "Multi session exited unexpectedly ($status)" multi "$session" "$multi_log" ;;
    esac
  done
done
for session in "${MULTI_SESSIONS[@]}"; do
  [[ "$(pane_result "$session")" == SUCCESS ]] || fail_and_stop 'Success status lost during cleanup' multi "$session" ''
  tmux kill-session -t "=$session" || fail_and_stop 'Could not remove completed Multi tmux session' multi "$session" ''
done

set_state READY_FOR_TUNING
# Do not launch new compute if an unrelated tmux session is present.
while [[ -n "$(other_sessions)" ]]; do
  say "Waiting for unrelated tmux before starting stages: $(other_sessions | tr '\n' ' ')"
  sleep "$POLL"
  maybe_progress_report other
done

say 'Beginning site-held-out tuning: each stage runs in its own tmux session.'
set_state RUNNING_TUNING
# The per-stage manager does NOT run in a second tmux; it is a child of this
# supervisor and launches dg_siteval_{style,kl,joint,multi,final}.
set +e
bash "$ROOT/dg_siteval_pipeline.sh" 2>&1 | tee -a "$PIPELINE_LOG"
rc=${PIPESTATUS[0]}
set -e
if (( rc != 0 )); then
  error_path="$(sed -n 's/^ERROR_FILE: //p' "$PIPELINE_LOG" | tail -1)"
  if [[ -f "$error_path" ]]; then
    stage="$(sed -n 's/^Failed stage: //p' "$error_path" | head -1)"
    session="$(sed -n 's/^Failed tmux session: //p' "$error_path" | head -1)"
    termlog="$(sed -n 's/^Stage terminal log: //p' "$error_path" | head -1)"
  else
    stage='pipeline'; session='none'; termlog="$PIPELINE_LOG"
  fi
  fail_and_stop "Tuning pipeline returned exit code $rc; see ${error_path:-$PIPELINE_LOG}" "$stage" "$session" "$termlog"
fi
if ! grep -q 'PIPELINE_COMPLETE: all stages and results verified' "$PIPELINE_LOG"; then
  fail_and_stop 'Pipeline exit 0 but final integrity confirmation was absent' final_integrity none "$PIPELINE_LOG"
fi
set_state TUNING_COMPLETE
say 'All five stages complete and 150 final OOD results verified.'
safe_shutdown AFTER_SUCCESS
