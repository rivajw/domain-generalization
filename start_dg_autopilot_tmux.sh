#!/usr/bin/env bash
# Start a safety-checked, persistent supervisor for a running MixStyle+Multi job.
# Usage: bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop
#    or: bash start_dg_autopilot_tmux.sh --stop-method aws
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STOP_METHOD=''
VERIFIED=0
POLL=60
DRY_RUN=0
MULTI_ALREADY_COMPLETE=0
while (( $# > 0 )); do
  case "$1" in
    --stop-method) STOP_METHOD="${2:?Supply aws or os}"; shift 2 ;;
    --verified-shutdown-stop) VERIFIED=1; shift ;;
    --poll-seconds) POLL="${2:?Supply seconds}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --multi-already-complete) MULTI_ALREADY_COMPLETE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$STOP_METHOD" == 'aws' || "$STOP_METHOD" == 'os' ]] || { echo 'Provide --stop-method aws or --stop-method os' >&2; exit 2; }
[[ "$POLL" =~ ^[0-9]+$ ]] && (( POLL >= 5 )) || { echo 'Poll seconds must be >= 5' >&2; exit 2; }
command -v tmux >/dev/null || { echo 'tmux not found' >&2; exit 1; }
[[ -f "$ROOT/start_dg_source_site_tmux.sh" && -f "$ROOT/run_dg_tuning_source_site.py" && -f "$ROOT/dg_siteval_pipeline.sh" && -f "$ROOT/dg_autopilot_supervisor.sh" ]] || { echo 'Install DG_source_site_validation_AWS.zip and DG_autopilot_AWS.zip in the project root first.' >&2; exit 1; }
[[ -n "${CONDA_PREFIX:-}" ]] || { echo 'Run conda activate bratsdg first' >&2; exit 1; }
python -c 'import torch; import dg.experiment; import dg.multiscale' || { echo 'Active Python is missing the DG training dependencies' >&2; exit 1; }
if tmux has-session -t '=dg_autopilot' 2>/dev/null; then echo 'dg_autopilot already running; inspect tmux attach -t dg_autopilot' >&2; exit 1; fi
if tmux has-session -t '=dg_siteval_all' 2>/dev/null; then echo 'dg_siteval_all is already present. Refusing to launch a second full tuning study.' >&2; exit 1; fi
for stage in style kl joint multi final; do
  if tmux has-session -t "=dg_siteval_${stage}" 2>/dev/null; then
    echo "Existing dg_siteval_${stage} tmux session; refusing duplicate automation." >&2; exit 1
  fi
done

mapfile -t MULTI_SESSIONS < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^multi_' || true)
if (( MULTI_ALREADY_COMPLETE )); then
  if (( ${#MULTI_SESSIONS[@]} != 0 )); then
    echo 'Cannot use --multi-already-complete while multi_* tmux sessions still exist.' >&2
    exit 1
  fi
  # Opt-in recovery after an instance stop: require all 250 unique, valid
  # original Multi trials; never assume a disappeared tmux means success.
  python - "$ROOT/artifacts/results/multiscale_runs" <<'PYVERIFYMULTI'
import itertools, json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = set(itertools.product((18, 1, 4, 13, 20), (0, 1, 2, 3, 4), ('strong', 'weak'), (0.25, 0.5, 1.0, 2.0, 4.0)))
seen = set()
for path in root.glob('DG-mixstyle_multi-*_metrics.json'):
    try:
        row = json.loads(path.read_text(encoding='utf-8'))
        if int(row['split_seed']) != 18662:
            continue
        key = (int(row['site']), int(row['train_seed']), row['regime'], float(row['multi_weight']))
        value = float(row['ood_bal_acc'])
        if key in expected and math.isfinite(value) and 0 <= value <= 1:
            seen.add(key)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        continue
if seen != expected:
    print(f'Refusing resume: {len(seen)}/250 valid original Multi runs; {len(expected-seen)} missing.', file=sys.stderr)
    sys.exit(1)
print('Verified: all 250 original MixStyle+Multi trials completed; bypassing obsolete tmux wait.')
PYVERIFYMULTI
else
  if (( ${#MULTI_SESSIONS[@]} == 0 )); then
    echo 'No active multi_* session found. If the 250 original Multi runs completed before instance restart, pass --multi-already-complete (verifies all 250 metrics files).' >&2
    exit 1
  fi
fi

if [[ "$STOP_METHOD" == os ]]; then
  (( VERIFIED )) || { echo 'OS shutdown may TERMINATE an EC2 instance if its setting is Terminate. Check EC2 Console > Actions > Instance settings > Change shutdown behavior = Stop, then pass --verified-shutdown-stop.' >&2; exit 1; }
  sudo -n true || { echo 'Passwordless sudo is needed for unattended shutdown; or use --stop-method aws.' >&2; exit 1; }
else
  command -v aws >/dev/null || { echo 'AWS CLI missing: install it, or select verified OS shutdown instead.' >&2; exit 1; }
  command -v curl >/dev/null || { echo 'curl needed for IMDSv2 instance ID discovery' >&2; exit 1; }
  ID_TOKEN="$(curl -fsS --connect-timeout 2 --max-time 5 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 'http://169.254.169.254/latest/api/token')" || { echo 'EC2 IMDSv2 unavailable' >&2; exit 1; }
  INSTANCE_ID="$(curl -fsS --connect-timeout 2 --max-time 5 -H "X-aws-ec2-metadata-token: $ID_TOKEN" 'http://169.254.169.254/latest/meta-data/instance-id')"
  AZ="$(curl -fsS --connect-timeout 2 --max-time 5 -H "X-aws-ec2-metadata-token: $ID_TOKEN" 'http://169.254.169.254/latest/meta-data/placement/availability-zone')"
  REGION="${AZ::-1}"
  [[ "$INSTANCE_ID" =~ ^i-[0-9a-f]+$ && -n "$REGION" ]] || { echo 'Unable to discover a trustworthy EC2 instance ID / AWS region' >&2; exit 1; }
  # Dry-run returns nonzero even on success; check the specific authorized response.
  ERR="$(aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --dry-run --no-cli-pager 2>&1)" || true
  [[ "$ERR" == *DryRunOperation* ]] || { printf 'No EC2 StopInstances permission for %s (%s):\n%s\n' "$INSTANCE_ID" "$REGION" "$ERR" >&2; exit 1; }
  echo "Verified StopInstances permission for this instance: $INSTANCE_ID ($REGION)"
fi

if (( MULTI_ALREADY_COMPLETE )); then
  echo 'Original Multi already complete: verified 250/250 metrics.'
else
  echo 'Multi sessions to watch:'
  printf '  %s\n' "${MULTI_SESSIONS[@]}"
fi
echo 'Stop method:' "$STOP_METHOD"
if (( DRY_RUN )); then
  echo 'DRY RUN: no tmux options modified, no supervisor launched, no instance stop.'
  exit 0
fi

# Preserve status and exit codes for sessions that normally disappear on completion.
# Refuse to silently continue if a session is already lost.
for session in "${MULTI_SESSIONS[@]}"; do
  tmux set-window-option -t "=$session:0" remain-on-exit on || { echo "Cannot retain status for $session" >&2; exit 1; }
  tmux list-panes -t "=$session:0" -F '#{pane_dead} #{pane_dead_status}' >/dev/null || { echo "Missing pane for $session" >&2; exit 1; }
done

LOG_DIR="$ROOT/artifacts/logs/dg_autopilot"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/autopilot_${STAMP}.log"
STATE="$LOG_DIR/autopilot_${STAMP}.state"
# Record explicit paths and conda environment so tmux-server startup env cannot change them.
# Build command without trusting shell-splitting of session names.
printf -v RUN_Q '%q ' "bash" "$ROOT/dg_autopilot_supervisor.sh" "$STOP_METHOD" "$POLL" "$STATE" "$LOG_DIR" "${MULTI_SESSIONS[@]}"
printf -v ENV_Q 'export PATH=%q CONDA_PREFIX=%q; ' "$PATH" "$CONDA_PREFIX"
printf -v COMMAND '%scd %q; set -o pipefail; %s2>&1 | tee %q' "$ENV_Q" "$ROOT" "$RUN_Q" "$LOG"
tmux new-session -d -s dg_autopilot "bash -c $(printf '%q' "$COMMAND")"
echo "Started autonomous supervisor in tmux: dg_autopilot"
echo "Log: $LOG"
echo "State: $STATE"
echo 'It waits for Multi, runs each tuning stage in its own tmux, and requests EC2 stop after success OR failure once idle.'
echo 'Monitor: tmux attach -t dg_autopilot (detach with Ctrl+B, D)'
