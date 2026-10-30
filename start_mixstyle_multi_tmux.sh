#!/usr/bin/env bash
# Launch a parameter-named MixStyle + Multi batch in detached tmux.
# Requires `conda activate bratsdg` in the calling shell.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo 'First run: conda activate bratsdg' >&2; exit 1
fi
command -v tmux >/dev/null || { echo 'Install tmux: sudo apt install tmux -y' >&2; exit 1; }
PYTHON="$(command -v python)"
# Keep readable parameter information; append digest so different args never clash.
LABEL=$(printf '%s_' "$@" | tr -cs '[:alnum:].-' '_' | cut -c1-64)
HASH=$(printf '%s\0' "$@" | sha256sum | cut -c1-8)
SESSION="multi_${LABEL:-default}_${HASH}"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Already active: $SESSION" >&2; exit 1
fi
LOG_DIR="$ROOT/artifacts/logs/multiscale_runs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
printf -v CMD '%q ' "$PYTHON" -u "$ROOT/run_mixstyle_multi.py" "$@"
printf -v ROOT_Q '%q' "$ROOT"
printf -v LOG_Q '%q' "$LOG_FILE"
tmux new-session -d -s "$SESSION" "cd $ROOT_Q && set -o pipefail && $CMD 2>&1 | tee $LOG_Q"
echo "Started: $SESSION"
echo "Log: $LOG_FILE"
echo "Monitor: tmux attach -t '$SESSION'"
echo 'Detach: Ctrl+B, D'
