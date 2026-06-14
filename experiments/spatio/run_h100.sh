#!/bin/bash
# SpatiO — reproduction H100 (MAS v2 + benchmarks)
#
# Pipeline: Head (Qwen3-VL) → ScoreMap → 3 Specialists → SharedMemory → Final Reasoning (DeepSeek-R1)
#
# Usage:
#   cd ~/CY/Spatial_MAS
#   git fetch origin && git reset --hard origin/main
#   bash experiments/spatio/run_h100.sh quick          # test rapide (10 samples, cvbench)
#   bash experiments/spatio/run_h100.sh baseline         # CV-Bench + 3DSRBench (10, 50, 100)
#   bash experiments/spatio/run_h100.sh tto-cvbench      # SpatialTTO train + eval (CV-Bench)
#   bash experiments/spatio/run_h100.sh tto-3dsrbench    # SpatialTTO train + eval (3DSRBench)
#   bash experiments/spatio/run_h100.sh stvqa           # STVQA-7K (5 modèles, single-agent)
#
# Variables utiles:
#   LOW_MEMORY=1              → 3 agents (qwen3_4b, llava4d, spatial_reasoner)
#   TEMPERATURE=0.7           → sampling (défaut 0 = greedy)
#   REASONING_MODEL=...       → modèle reasoning local (défaut DeepSeek-R1-Distill-Qwen-7B)
#   SPATIALRGPT_PATH=...      → requis pour spatial_rgpt / STVQA spatialrgpt
#
set -euo pipefail

export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::UserWarning"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-baseline}"
shift || true

SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
OUTPUT_BASE="${OUTPUT_BASE:-results/spatio}"
REASONING_MODEL="${REASONING_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

EXTRA_ARGS=()
if [[ "${LOW_MEMORY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--low_memory)
fi
if [[ -n "${SPECIALIST_WHITELIST:-}" ]]; then
  EXTRA_ARGS+=(--specialist_whitelist "$SPECIALIST_WHITELIST")
fi
EXTRA_ARGS+=(--specialist_offload_after_use)
EXTRA_ARGS+=(--temperature "$TEMPERATURE" --top_p "$TOP_P")

run_mas_v2_test() {
  local benchmark="$1"
  local n="$2"
  echo ""
  echo ">>> SpatiO MAS v2 | $benchmark | $n samples"
  echo "----------------------------------------------"
  python run_eval_mas_v2.py \
    --benchmark "$benchmark" \
    --max_samples "$n" \
    --test_only \
    --seed "$SEED" \
    --output_dir "$OUTPUT_BASE/mas_v2_baseline" \
    --use_local_reasoning \
    --reasoning_local_model "$REASONING_MODEL" \
    --device cuda \
    "${EXTRA_ARGS[@]}" \
    "$@"
}

echo "=============================================="
echo "SpatiO — H100 reproduction"
echo "Mode: $MODE"
echo "Project: $PROJECT_ROOT"
echo "Output: $OUTPUT_BASE"
echo "Temperature: $TEMPERATURE | Seed: $SEED"
echo "=============================================="

case "$MODE" in
  quick)
    run_mas_v2_test cvbench 10 "$@"
    ;;
  baseline)
    for BENCHMARK in cvbench 3dsrbench; do
      for N in 10 50 100; do
        run_mas_v2_test "$BENCHMARK" "$N" "$@"
      done
    done
    ;;
  cvbench)
    for N in 10 50 100; do
      run_mas_v2_test cvbench "$N" "$@"
    done
    ;;
  3dsrbench|3dsr)
    for N in 10 50 100; do
      run_mas_v2_test 3dsrbench "$N" "$@"
    done
    ;;
  tto-cvbench)
    echo ">>> SpatialTTO — CV-Bench (train TTO + eval frozen)"
    python run_confidence_mas_step4_train_then_eval_frozen.py \
      --benchmark cvbench \
      --eval \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --seed "$SEED" \
      ${LOW_MEMORY:+--low_memory} \
      ${SPECIALIST_OFFLOAD:+--specialist_offload} \
      "$@"
    ;;
  tto-3dsrbench)
    echo ">>> SpatialTTO — 3DSRBench (train TTO + eval frozen)"
    bash scripts/run_spatialtto_3dsrbench.sh \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --seed "$SEED" \
      ${LOW_MEMORY:+--low_memory} \
      "$@"
    ;;
  tto-stvqa)
    echo ">>> SpatialTTO — STVQA (train TTO + eval frozen)"
    python run_confidence_mas_step4_train_then_eval_frozen.py \
      --benchmark stvqa \
      --eval \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --seed "$SEED" \
      ${LOW_MEMORY:+--low_memory} \
      "$@"
    ;;
  stvqa)
    echo ">>> STVQA-7K — single-agent baselines (5 modèles)"
    if [[ -n "${SPATIALRGPT_PATH:-}" ]]; then
      python scripts/stvqa7k/patch_spatialrgpt_py39.py || true
    fi
    bash experiments/stvqa7k/run_h100.sh
    ;;
  mindcube)
    echo ">>> MindCube — SpatiO MAS v2"
    bash scripts/evals/mindcube/run_mindcube_mas_v2_h100.sh
    ;;
  *)
    echo "Usage: $0 {quick|baseline|cvbench|3dsrbench|tto-cvbench|tto-3dsrbench|tto-stvqa|stvqa|mindcube} [extra args...]"
    echo ""
    echo "  quick         — 10 samples CV-Bench (sanity check)"
    echo "  baseline      — CV-Bench + 3DSRBench × 10/50/100 (MAS v2, test-only)"
    echo "  tto-*         — SpatialTTO (train score map + frozen eval)"
    echo "  stvqa         — STVQA-7K single-model eval"
    exit 1
    ;;
esac

echo ""
echo "=============================================="
echo "Done. Results under $OUTPUT_BASE/ or results/"
echo "=============================================="
