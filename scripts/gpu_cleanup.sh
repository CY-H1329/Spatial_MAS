#!/bin/bash
# GPU memory cleanup helper for H100 runs.
#
# Usage:
#   bash scripts/gpu_cleanup.sh              # show GPU status only
#   bash scripts/gpu_cleanup.sh --kill         # kill GPU compute PIDs (except this shell tree)
#   bash scripts/gpu_cleanup.sh --kill-all     # kill ALL GPU compute PIDs (aggressive)
#
# Before full-dataset SpatiO eval:
#   bash scripts/gpu_cleanup.sh --kill
#   nvidia-smi   # should show < 5 GiB used
#
set -euo pipefail

echo "=== GPU status ==="
nvidia-smi

echo ""
echo "=== Compute processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true

MODE="${1:-}"
if [[ "$MODE" != "--kill" && "$MODE" != "--kill-all" ]]; then
  echo ""
  echo "To free VRAM:"
  echo "  bash scripts/gpu_cleanup.sh --kill       # safe (skip current shell tree)"
  echo "  bash scripts/gpu_cleanup.sh --kill-all     # kill every GPU PID (use if --kill fails)"
  exit 0
fi

_is_descendant_of() {
  local pid="$1"
  local root="$2"
  while [[ -n "$pid" && "$pid" -gt 1 ]]; do
    if [[ "$pid" == "$root" ]]; then
      return 0
    fi
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || echo "")
  done
  return 1
}

_in_our_shell_tree() {
  local pid="$1"
  local shell_pid=$$
  local shell_ppid
  shell_ppid=$(ps -o ppid= -p "$shell_pid" 2>/dev/null | tr -d ' ' || echo "")
  if [[ "$pid" == "$shell_pid" ]]; then return 0; fi
  if _is_descendant_of "$pid" "$shell_pid"; then return 0; fi
  if [[ -n "$shell_ppid" ]] && [[ "$pid" == "$shell_ppid" ]]; then return 0; fi
  return 1
}

echo ""
if [[ "$MODE" == "--kill-all" ]]; then
  echo "=== Killing ALL GPU compute PIDs ==="
else
  echo "=== Killing GPU compute PIDs (except current shell tree) ==="
fi

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
KILLED=0

for pid in $PIDS; do
  [[ -z "$pid" ]] && continue
  if [[ "$MODE" != "--kill-all" ]] && _in_our_shell_tree "$pid"; then
    comm=$(ps -p "$pid" -o comm= 2>/dev/null || echo "[unknown]")
    echo "  skip PID $pid ($comm) — current shell tree"
    continue
  fi
  comm=$(ps -p "$pid" -o comm= 2>/dev/null || echo "[not in ps — stale GPU context]")
  echo "  kill -9 $pid ($comm)"
  if kill -9 "$pid" 2>/dev/null; then
    KILLED=$((KILLED + 1))
  else
    echo "    → failed (try: sudo kill -9 $pid)"
  fi
done

sleep 3
echo ""
echo "Killed $KILLED process(es). Clearing PyTorch cache in current shell..."
python - <<'PY' 2>/dev/null || true
import gc
try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU free: {free / 1024**3:.1f} GiB / {total / 1024**3:.1f} GiB")
except Exception as e:
    print(f"  (cache clear skipped: {e})")
PY

echo ""
echo "=== GPU status after cleanup ==="
nvidia-smi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [[ -n "$USED" && "$USED" -gt 5000 ]]; then
  echo ""
  echo "WARNING: GPU still uses ${USED} MiB. Stale context may remain."
  echo "  sudo kill -9 \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
  echo "  Or restart the Jupyter/container session."
fi
