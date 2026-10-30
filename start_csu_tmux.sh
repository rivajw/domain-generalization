#!/usr/bin/env bash
# Start one parameter-named CSU experiment in a detached tmux session.
# Usage: conda activate bratsdg
#        bash start_csu_tmux.sh [p] [alpha] [layer] [seed] [site]
set -euo pipefail

P="${1:-0.75}"
ALPHA="${2:-0.1}"
LAYER="${3:-2}"
SEED="${4:-18662}"
SITE="${5:-18}"

if ! [[ "$P" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
  echo "Invalid p: expected a decimal number between 0 and 1" >&2; exit 2
fi
if ! [[ "$ALPHA" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$ALPHA" == "0" || "$ALPHA" == "0.0" ]]; then
  echo "Invalid alpha: expected a positive number" >&2; exit 2
fi
if ! [[ "$LAYER" =~ ^[123]$ ]] || ! [[ "$SEED" =~ ^[0-9]+$ ]] || ! [[ "$SITE" =~ ^[0-9]+$ ]]; then
  echo "Invalid layer, seed, or site" >&2; exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python)"
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "Activate your Conda environment before using this launcher:" >&2
  echo "  conda activate bratsdg" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed: sudo apt install tmux" >&2; exit 1
fi

TAG="CSU-site${SITE}-L${LAYER}-p${P}-a${ALPHA}-seed${SEED}"
SESSION="${TAG//./_}"
LOG_DIR="$PROJECT_ROOT/artifacts/logs"
mkdir -p "$LOG_DIR"
# Timestamp means re-launching a configuration won't erase a previous log.
LOG_FILE="$LOG_DIR/${TAG}_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "A tmux session for this configuration is already running: $SESSION" >&2
  echo "Attach with: tmux attach -t '$SESSION'" >&2
  exit 1
fi

printf -v PARAM_COMMAND '%q ' "$PYTHON_BIN" -u "$PROJECT_ROOT/run_csu_aws.py" \
    --p "$P" --alpha "$ALPHA" --layer "$LAYER" --seed "$SEED" --site "$SITE"
printf -v WORKDIR '%q' "$PROJECT_ROOT"
printf -v LOG_Q '%q' "$LOG_FILE"
tmux new-session -d -s "$SESSION" "cd $WORKDIR && $PARAM_COMMAND 2>&1 | tee $LOG_Q"

echo "Started CSU training in detached tmux."
echo "Experiment: $TAG"
echo "Session:    $SESSION"
echo "Log file:   $LOG_FILE"
echo "Python:     $PYTHON_BIN"
echo "Attach:     tmux attach -t '$SESSION'"
echo "Follow log: tail -f '$LOG_FILE'"
