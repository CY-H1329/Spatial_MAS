#!/bin/bash
# Diagnose who holds GPU memory / compute (H100, Jupyter container).
#
# Usage: bash scripts/gpu_diagnose.sh
#
set -euo pipefail

echo "========== nvidia-smi (summary) =========="
nvidia-smi

echo ""
echo "========== Compute apps (may show PIDs container cannot kill) =========="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true

echo ""
echo "========== GPU util snapshot =========="
nvidia-smi pmon -c 1 2>/dev/null || true

echo ""
echo "========== Processes opening /dev/nvidia* (this namespace) =========="
if command -v fuser >/dev/null 2>&1; then
  fuser -v /dev/nvidia* 2>&1 || echo "(none in this namespace)"
else
  echo "fuser not installed"
fi

echo ""
echo "========== lsof /dev/nvidia0 (this namespace) =========="
if command -v lsof >/dev/null 2>&1; then
  lsof /dev/nvidia0 2>/dev/null | head -20 || echo "(none)"
else
  echo "lsof not installed"
fi

echo ""
echo "========== Python jobs in this container =========="
ps aux 2>/dev/null | grep -E '[p]ython|[t]orch|[j]upyter' | head -20 || true

echo ""
echo "========== PyTorch view of free memory =========="
python - <<'PY' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  allocatable free: {free/2**30:.2f} GiB / {total/2**30:.2f} GiB")
else:
    print("  CUDA not available")
PY

echo ""
echo "========== Interpretation =========="
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')

if [[ -n "$USED" && "$USED" -gt 50000 && "$NPROC" -eq 0 ]]; then
  echo "  HIGH memory ($USED MiB) but empty process table → stale VRAM or job outside this container."
  echo "  Fix: restart Jupyter pod / container (not kill from inside)."
elif [[ -n "$UTIL" && "$UTIL" -gt 10 && "$NPROC" -eq 0 ]]; then
  echo "  GPU util ${UTIL}% with no visible PIDs → compute likely on HOST or another pod sharing GPU."
  echo "  Fix: stop other users' jobs or request dedicated GPU / pod restart."
elif [[ "$NPROC" -gt 0 ]]; then
  echo "  $NPROC compute process(es) listed — try: bash scripts/gpu_cleanup.sh --kill-all"
else
  echo "  GPU looks mostly free."
fi
echo "  Doc: docs/GPU_STALE_CONTEXT.md"
