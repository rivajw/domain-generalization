#!/usr/bin/env bash
# Run a serial study in a detached tmux session with Conda already activated.
# Pass ALL CLI args after the script name, e.g.:
# bash start_dg_multisite_tmux.sh --sites 18 --methods csu mixstyle --seeds 0 1 2 3 4
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CONDA_PREFIX:-}" ]; then echo 'Run conda activate bratsdg first' >&2; exit 1; fi
command -v tmux >/dev/null || { echo 'Install tmux first' >&2; exit 1; }
SESSION="dg_5site_5seed"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session exists: $SESSION. Attach: tmux attach -t $SESSION"; exit 1
fi
mkdir -p "$PROJECT_ROOT/artifacts/logs"
LOG_FILE="$PROJECT_ROOT/artifacts/logs/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
python_bin="$(command -v python)"
printf -v cmd '%q ' "$python_bin" -u "$PROJECT_ROOT/run_dg_multisite.py" "$@"
printf -v dir_q '%q' "$PROJECT_ROOT"
printf -v log_q '%q' "$LOG_FILE"
tmux new-session -d -s "$SESSION" "cd $dir_q && set -o pipefail && $cmd 2>&1 | tee $log_q"
echo "Started detached tmux: $SESSION"
echo "Log: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
