#!/usr/bin/env bash
# SpatiO ablation — CV-Bench 500 (seed), pool sans spatial_reasoner / spatial_rgpt :
# qwen3_4b, sa2va, llava4d, internvl2, qwen2_vl (tous GPU).
#
# Usage (depuis la racine du dépôt Spatial_MAS) :
#   bash scripts/evals/mas_pipeline/run_h100_ablation_internvl2_qwen2vl_cvbench500.sh
#
# Variables optionnelles :
#   CUDA_VISIBLE_DEVICES=0
#   HF_HOME=/chemin/cache

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export MAS_CANDIDATE_AGENTS="${MAS_CANDIDATE_AGENTS:-qwen3_4b,sa2va,llava4d,internvl2,qwen2_vl}"

python scripts/evals/mas_pipeline/run_eval_mas.py \
  --config scripts/evals/mas_pipeline/config_mas_ablation_internvl2_qwen2vl_cvbench500_h100.yaml \
  --benchmark cvbench \
  --max_samples 500 \
  --seed "${SEED:-42}"
