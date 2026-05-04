#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Optionnel : réutiliser un cache MindCube déjà extrait (ex. dossier Spatial_MAS)
# export MINDCUBE_DIR="/chemin/vers/mindcube"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
# Qwen3-VL : souvent nécessite transformers récent (voir carte modèle HF)
python -m pip install "git+https://github.com/huggingface/transformers.git"

ADAPTER="./outputs/lora_3dsr_qwen3vl"
python train_lora_qwen3vl.py \
  --model_id "Qwen/Qwen3-VL-4B-Instruct" \
  --output_dir "$ADAPTER" \
  --max_train_samples 150 \
  --epochs 1 \
  --bf16 \
  --image_cache "./data/3dsr_image_cache"

python infer_mindcube_qwen3vl.py \
  --base_model_id "Qwen/Qwen3-VL-4B-Instruct" \
  --adapter_dir "$ADAPTER" \
  --mindcube_split tinybench \
  --max_samples 50 \
  --output_dir "./outputs/mindcube_infer_tiny" \
  --bf16

echo "Temps entraînement : voir ${ADAPTER}/timing_train.json"
echo "Temps inférence : voir ./outputs/mindcube_infer_tiny/timing_infer.jsonl et timing_infer_summary.json"
