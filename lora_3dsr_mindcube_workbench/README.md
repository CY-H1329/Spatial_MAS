# LoRA 3DSRBench + inférence MindCube (Qwen3-VL)

Projet **autonome** (hors Spatial_MAS) : fine-tuning LoRA sur **3DSRBench**, puis inférence **MindCube** avec les adaptateurs, avec mesure des durées.

## Nombre d’exemples MindCube (fichiers `data/raw/*.jsonl` dans `data.zip`)

| Fichier | Lignes |
|--------|--------|
| `MindCube.jsonl` (éval / test principal) | 21 154 |
| `MindCube_train.jsonl` | 10 000 |
| `MindCube_tinybench.jsonl` | 1 050 |

## Prérequis

- GPU NVIDIA (CUDA), idéalement H100.
- `transformers` récent : la carte modèle [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) recommande souvent une install depuis les sources :

```bash
pip install -r requirements.txt
pip install "git+https://github.com/huggingface/transformers.git"
```

## Où sont enregistrés les temps

- **Entraînement** : `timing_train.json` dans `--output_dir` (durée chargement données, modèle, LoRA, chaque epoch, sauvegarde, total).
- **Inférence** : `timing_infer.jsonl` (une ligne JSON par échantillon : chargement images, prétraitement, génération, total) + `timing_infer_summary.json` (dont `load_model_s`, moyenne `mean_total_sample_s`, précision MCQ sur l’échantillon).

## Commandes

```bash
python train_lora_qwen3vl.py --output_dir ./outputs/lora_3dsr --max_train_samples 150 --epochs 1 --bf16
python infer_mindcube.py --adapter_dir ./outputs/lora_3dsr --mindcube_split tinybench --max_samples 50 --bf16
```

Cache MindCube : par défaut `./data/mindcube` (téléchargement `data.zip` depuis le dataset HF `MLL-Lab/MindCube`). Variable optionnelle : `MINDCUBE_DIR`.

## Script H100

```bash
chmod +x run_h100.sh
./run_h100.sh
```

Les durées réelles dépendent du GPU, du réseau (téléchargement / images 3DSRBench URL) et de la charge machine.
