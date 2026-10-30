#!/usr/bin/env bash
# Sequential stage manager. Run via the autotopilot supervisor or with the "all"
# entrypoint. Each stage has an independent tmux session, stdout log and marker.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
POLL="${DG_STAGE_POLL_SECONDS:-30}"
[[ "$POLL" =~ ^[0-9]+$ && "$POLL" -ge 1 ]] || { echo 'Invalid DG_STAGE_POLL_SECONDS' >&2; exit 2; }
LOG_DIR="$ROOT/artifacts/logs/siteholdout_launcher"
ERR_DIR="$LOG_DIR/errors"
mkdir -p "$ERR_DIR"
STAGES=(style kl joint multi final)
CURRENT_STAGE='preflight'
CURRENT_SESSION='none'
CURRENT_LOG=''
ERROR_FILE=''
# Report current stage and completed trial percentages once per hour.
PROGRESS_INTERVAL="${DG_PROGRESS_INTERVAL_SECONDS:-3600}"
[[ "$PROGRESS_INTERVAL" =~ ^[0-9]+$ ]] && (( PROGRESS_INTERVAL >= 1 )) || { echo 'Invalid DG_PROGRESS_INTERVAL_SECONDS' >&2; exit 2; }
LAST_PROGRESS_AT="$(date +%s)"
progress_report() {
  # The read-only counter accepts stage names but not internal pipeline phases.
  local display_stage="$CURRENT_STAGE"
  case "$display_stage" in
    waiting_multi|style|kl|joint|multi|final) ;;
    *) display_stage='other' ;;
  esac
  python "$ROOT/dg_autopilot_progress.py" --project-root "$ROOT" --stage "$display_stage" || echo 'WARNING: could not read progress metrics (training unaffected).'
}
maybe_progress_report() {
  local now
  now="$(date +%s)"
  if (( now - LAST_PROGRESS_AT >= PROGRESS_INTERVAL )); then
    progress_report
    LAST_PROGRESS_AT="$now"
  fi
}

record_error() {
  local reason="$1" path="$ERR_DIR/failure_$(date -u +%Y%m%dT%H%M%SZ)_${CURRENT_STAGE}_$$.txt"
  {
    printf 'Timestamp UTC: %s\n' "$(date -u '+%FT%TZ')"
    printf 'Failed stage: %s\nFailed tmux session: %s\n' "$CURRENT_STAGE" "$CURRENT_SESSION"
    printf 'Reason: %s\n' "$reason"
    printf 'Stage terminal log: %s\n' "${CURRENT_LOG:-not available}"
    printf '\n=== LAST 200 LINES OF TERMINAL OUTPUT ===\n'
    if [[ -n "$CURRENT_LOG" && -e "$CURRENT_LOG" ]]; then tail -n 200 "$CURRENT_LOG"; else echo '(No stage log created)'; fi
  } > "$path"
  ERROR_FILE="$path"
  echo "ERROR_FILE: $path" >&2
}

# If a stage exits nonzero or disappears, never advance to later stages.
progress_report
for CURRENT_STAGE in "${STAGES[@]}"; do
  CURRENT_SESSION="dg_siteval_${CURRENT_STAGE}"
  CURRENT_LOG=''
  echo "[$(date '+%F %T')] LAUNCHING $CURRENT_STAGE as $CURRENT_SESSION"
  if ! OUT="$(bash "$ROOT/start_dg_source_site_tmux.sh" "$CURRENT_STAGE" 2>&1)"; then
    record_error "Launcher failed: $OUT"
    exit 1
  fi
  echo "$OUT"
  CURRENT_LOG="$(sed -n 's/^Log: //p' <<< "$OUT" | tail -1)"
  MARKER="$(sed -n 's/^Exit: //p' <<< "$OUT" | tail -1)"
  [[ -n "$MARKER" && -n "$CURRENT_LOG" ]] || { record_error "Launcher did not provide Log/Exit paths: $OUT"; exit 1; }
  # If tmux is killed or the command OOM-kills before writing the exit marker,
  # a missing session is unambiguously a failed stage, never a success.
  while [[ ! -e "$MARKER" ]]; do
    if ! tmux has-session -t "=$CURRENT_SESSION" 2>/dev/null; then
      # Allow file-system scheduling a short grace period for marker creation.
      sleep 3
      if [[ ! -e "$MARKER" ]]; then record_error 'tmux stage session disappeared with no exit marker (killed, crash, OOM or host event)'; exit 1; fi
      break
    fi
    sleep "$POLL"
    maybe_progress_report
  done
  RC="$(tr -d '[:space:]' < "$MARKER")"
  if ! [[ "$RC" =~ ^[0-9]+$ ]] || (( RC != 0 )); then
    record_error "Stage process exit code: ${RC:-missing}"
    [[ ! -e "$MARKER" ]] || echo "Stage marker: $MARKER" >&2
    if tmux has-session -t "=$CURRENT_SESSION" 2>/dev/null; then tmux kill-session -t "=$CURRENT_SESSION" || true; fi
    exit 1
  fi
  # A reported success must also have the expected stage-completion line.
  if ! grep -Fq "Stage complete: ${CURRENT_STAGE}" "$CURRENT_LOG"; then
    record_error 'Exit code 0 but expected stage-completion message was not found in log'
    exit 1
  fi
  if tmux has-session -t "=$CURRENT_SESSION" 2>/dev/null; then
    tmux kill-session -t "=$CURRENT_SESSION" || true
  fi
  echo "[$(date '+%F %T')] STAGE_DONE: $CURRENT_STAGE ($CURRENT_SESSION)"
  progress_report
  LAST_PROGRESS_AT="$(date +%s)"
done

# Verify the complete expected matrix, not merely that the Python script exited.
CURRENT_STAGE='final_integrity'
CURRENT_SESSION='none'
CURRENT_LOG="$ROOT/artifacts/results/siteholdout_final/all_final_runs.csv"
if ! python - "$CURRENT_LOG" <<'PY'
import csv, itertools, math, sys
p = sys.argv[1]
try:
    with open(p, newline='', encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    required=set(itertools.product((18,1,4,13,20),
      ('erm','mixstyle','csu','mixstyle_kl','mixstyle_kl_mmd','mixstyle_multi'),
      (100,101,102,103,104)))
    actual=[(int(x['site']),x['method'],int(x['train_seed'])) for x in rows]
    assert len(actual)==len(required) and set(actual)==required, f'Expected 150 unique results, got {len(actual)}'
    assert all(0<=float(x['ood_bal_acc'])<=1 and math.isfinite(float(x['ood_bal_acc'])) for x in rows)
except Exception as e:
    print(f'Final result integrity error: {e}', file=sys.stderr)
    sys.exit(1)
print('Verified 150 unique final OOD results.')
PY
then
  record_error 'Final CSV missing, incomplete or invalid (expected 150 unique site/method/seed rows)'
  exit 1
fi
echo "[$(date '+%F %T')] PIPELINE_COMPLETE: all stages and results verified"
