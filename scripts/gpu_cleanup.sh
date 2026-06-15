#!/bin/bash
# GPU memory cleanup helper for H100 runs.
#
# Usage:
#   bash scripts/gpu_cleanup.sh           # show GPU status only
#   bash scripts/gpu_cleanup.sh --kill    # kill other Python GPU processes
#
# Before full-dataset SpatiO eval, stale jobs often hold VRAM:
#   bash scripts/gpu_cleanup.sh --kill
#   bash experiments/spatio/run_h100.sh full-cvbench
#
set -euo pipefail

echo "=== GPU status ==="
nvidia-smi

echo ""
echo "=== Compute processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true

if [[ "${1:-}" != "--kill" ]]; then
  echo ""
  echo "To free VRAM from stale Python jobs: bash scripts/gpu_cleanup.sh --kill"
  exit 0
fi

echo ""
echo "=== Killing other Python GPU processes (except this shell) ==="
MY_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)

for pid in $PIDS; do
  [[ -z "$pid" ]] && continue
  if ! ps -p "$pid" >/dev/null 2>&1; then
    continue
  fi
  comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  if [[ "$pgid" == "$MY_PGID" ]]; then
    echo "  skip PID $pid ($comm) — same process group"
    continue
  fi
  if echo "$comm" | grep -qi python; then
    echo "  kill PID $pid ($comm)"
    kill -9 "$pid" 2>/dev/null || true
  fi
done

sleep 2
echo ""
echo "=== GPU status after cleanup ==="
nvidia-smi
