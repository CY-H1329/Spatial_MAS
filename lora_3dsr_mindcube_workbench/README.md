# LoRA 3DSRBench + inférence MindCube — 5 modèles

Ce dossier vit sous la racine **Spatial_MAS** (`lora_3dsr_mindcube_workbench/`) pour que les scripts **Sa2VA** et **SpatialRGPT** puissent importer `src2/models` (sinon exportez `SPATIAL_MAS_ROOT`).

## Modèles et scripts

| Modèle | Entraînement LoRA (3DSRBench) | Inférence MindCube |
|--------|-------------------------------|---------------------|
| **Qwen3-VL-4B** | `train_lora_qwen3vl.py` | `infer_mindcube_qwen3vl.py` |
| **LLaVA-NeXT 7B** | `train_lora_llava.py` | `infer_mindcube_llava.py` |
| **SpatialReasoner** | `train_lora_spatial_reasoner.py` | `infer_mindcube_spatial_reasoner.py` |
| **Sa2VA-4B** | `train_lora_sa2va.py` | `infer_mindcube_sa2va.py` |
| **SpatialRGPT** | `train_lora_spatial_rgpt.py` | `infer_mindcube_spatial_rgpt.py` |

- **LLaVA / Sa2VA / SpatialRGPT** (inférence) : vues MindCube fusionnées en **une grille** (`mc_common.tile_images_for_single_image_backend`), car ces backends sont mono-image.
- **Sa2VA** : l’entraînement repose sur une **loss** éventuelle renvoyée par le modèle ; si aucune, le script se termine avec **code 2** et `timing_train.json` l’explique — utilisez le repo ByteDance pour un fine-tuning complet.
- **SpatialRGPT** : nécessite `export SPATIALRGPT_PATH=...` (clone [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT)). L’entraînement tente `forward(..., labels=...)` ; **code 2** si non supporté.

## Nombre d’exemples MindCube (`data.zip`)

| Fichier | Lignes |
|--------|--------|
| `MindCube.jsonl` | 21 154 |
| `MindCube_train.jsonl` | 10 000 |
| `MindCube_tinybench.jsonl` | 1 050 |

## Prérequis

- GPU NVIDIA (CUDA).
- `pip install -r requirements.txt` puis souvent `pip install "git+https://github.com/huggingface/transformers.git"` (Qwen3-VL / versions récentes).

## Commandes détaillées

Voir **`COMMANDS.md`** (copier-coller par modèle).

Résumé :

```bash
python train_lora_qwen3vl.py --output_dir ./out/qwen3vl --bf16
python infer_mindcube_qwen3vl.py --adapter_dir ./out/qwen3vl --bf16
```

(`infer_mindcube.py` redirige vers **Qwen3-VL** pour compatibilité.)

## Temps (logs)

- Entraînement : `timing_train.json` dans chaque `--output_dir`.
- Inférence : `timing_infer.jsonl` + `timing_infer_summary.json` dans chaque `--output_dir` d’inférence.

## Scripts shell

- `run_h100.sh` — exemple minimal (Qwen3 + infer).
- `run_models_5.sh` — enchaîne les 5 (avec `|| true` là où l’environnement peut échouer).
