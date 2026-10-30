#!/usr/bin/env bash
# Compatibility entrypoint: same filename and public CLI as the prior launcher.
# Every individual stage uses its OWN detached tmux session. "all" launches a
# dedicated orchestration session, which starts each stage sequentially.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-all}"
if (( $# )); then shift; fi
case "$MODE" in all|style|kl|joint|multi|final) ;; *) echo 'Usage: bash start_dg_source_site_tmux.sh [all|style|kl|joint|multi|final] [runner options]' >&2; exit 2 ;; esac
command -v tmux >/dev/null || { echo 'tmux is required' >&2; exit 1; }
[[ -n "${CONDA_PREFIX:-}" ]] || { echo 'Activate bratsdg with conda activate bratsdg first' >&2; exit 1; }
PYTHON="$(command -v python)"
"$PYTHON" -c 'import torch, dg.experiment; print("Training Python:", torch.__version__)' || { echo 'PyTorch/DG unavailable in the selected Conda environment' >&2; exit 1; }
LOG_DIR="$ROOT/artifacts/logs/siteholdout_launcher"
mkdir -p "$LOG_DIR"
if [[ "$MODE" == all ]]; then
  [[ $# -eq 0 ]] || { echo 'The all launcher does not take extra runner options; launch one stage with options instead.' >&2; exit 2; }
  SESSION=dg_siteval_all
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "Already running: $SESSION" >&2; exit 1; fi
  # Avoid a second manager fighting the autopilot.
  if tmux has-session -t '=dg_autopilot' 2>/dev/null; then echo 'dg_autopilot is already managing this study' >&2; exit 1; fi
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  LOG="$LOG_DIR/${SESSION}_${STAMP}.log"
  printf -v CMD 'export PATH=%q CONDA_PREFIX=%q; cd %q; set -o pipefail; bash %q 2>&1 | tee -a %q' "$PATH" "$CONDA_PREFIX" "$ROOT" "$ROOT/dg_siteval_pipeline.sh" "$LOG"
  tmux new-session -d -s "$SESSION" "bash -c $(printf '%q' "$CMD")"
  echo "Started manager: $SESSION (runs one tmux session per stage)"
  echo "Log: $LOG"
  exit 0
fi

SESSION="dg_siteval_${MODE}"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists; inspect or remove it before restarting." >&2
  exit 1
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG="$LOG_DIR/${SESSION}_${STAMP}.log"
EXIT_FILE="$LOG_DIR/${SESSION}_${STAMP}.exit"
WRAPPER="$LOG_DIR/${SESSION}_${STAMP}.sh"
# Bash printf emits a quoted empty string ('' ) when %q has zero args.
# That was passed to argparse as an unwanted positional argument (exit code 2).
PARAM_Q=''
if (( $# > 0 )); then
  printf -v PARAM_Q '%q ' "$@"
fi
# Keep stage stdout in a real session pane and persist all output on disk.
# Exit marker is written even when the Python command errors, and is unique
# to the launch (so stale successful runs cannot be misinterpreted).
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  printf 'export PATH=%q CONDA_PREFIX=%q\n' "$PATH" "$CONDA_PREFIX"
  printf 'cd %q\n' "$ROOT"
  printf 'echo "==== START stage: %s ====" | tee -a %q\n' "$MODE" "$LOG"
  printf '%q -u %q --stage %q %s 2>&1 | tee -a %q\n' "$PYTHON" "$ROOT/run_dg_tuning_source_site.py" "$MODE" "$PARAM_Q" "$LOG"
  echo 'result=${PIPESTATUS[0]}'
  printf 'echo "==== EXIT stage: %s code=${result} ====" | tee -a %q\n' "$MODE" "$LOG"
  printf 'printf "%%d\\n" "$result" > %q.tmp\n' "$EXIT_FILE"
  printf 'mv -f %q.tmp %q\n' "$EXIT_FILE" "$EXIT_FILE"
  echo 'exit "$result"'
} > "$WRAPPER"
chmod 700 "$WRAPPER"
tmux new-session -d -s "$SESSION" "bash $(printf '%q' "$WRAPPER")"
echo "Started detached stage session: $SESSION"
echo "Log: $LOG"
echo "Exit: $EXIT_FILE"
echo "Session: $SESSION"
