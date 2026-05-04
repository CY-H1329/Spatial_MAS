#!/usr/bin/env bash
# Exemple : entraînement + inférence pour les 5 backends (adapter les chemins / SPATIALRGPT_PATH).
set -euo pipefail
cd "$(dirname "$0")"

BF="--bf16"
OUT="./outputs"

python train_lora_qwen3vl.py --output_dir "$OUT/qwen3vl" $BF
python infer_mindcube_qwen3vl.py --adapter_dir "$OUT/qwen3vl" --output_dir "$OUT/mc_qwen3vl" $BF

python train_lora_llava.py --output_dir "$OUT/llava" $BF
python infer_mindcube_llava.py --adapter_dir "$OUT/llava" --output_dir "$OUT/mc_llava" $BF

python train_lora_spatial_reasoner.py --output_dir "$OUT/spatial_reasoner" $BF
python infer_mindcube_spatial_reasoner.py --adapter_dir "$OUT/spatial_reasoner" --output_dir "$OUT/mc_spatial_reasoner" $BF

python train_lora_sa2va.py --output_dir "$OUT/sa2va" || true
python infer_mindcube_sa2va.py --adapter_dir "$OUT/sa2va" --output_dir "$OUT/mc_sa2va" || python infer_mindcube_sa2va.py --output_dir "$OUT/mc_sa2va_base"

if [[ -n "${SPATIALRGPT_PATH:-}" ]]; then
  python train_lora_spatial_rgpt.py --output_dir "$OUT/spatial_rgpt" || true
  python infer_mindcube_spatial_rgpt.py --adapter_dir "$OUT/spatial_rgpt" --output_dir "$OUT/mc_spatial_rgpt" || python infer_mindcube_spatial_rgpt.py --output_dir "$OUT/mc_spatial_rgpt_base"
else
  echo "[skip] SpatialRGPT : exportez SPATIALRGPT_PATH"
fi
