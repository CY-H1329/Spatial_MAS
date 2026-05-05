# Commandes — 5 modèles (train 3DSRBench + infer MindCube)

Placez-vous dans `lora_3dsr_mindcube_workbench/` (sous la racine **Spatial_MAS** pour Sa2VA / SpatialRGPT afin que `src2` soit trouvé).

```bash
cd /chemin/vers/Spatial_MAS/lora_3dsr_mindcube_workbench
source .venv/bin/activate   # si venv déjà créé
```

SpatialRGPT : `export SPATIALRGPT_PATH=/chemin/vers/SpatialRGPT` (clone officiel).

**Dataset complet** : ajoutez `--full_dataset` aux scripts **train** (tout le split test 3DSRBench) et **infer** (toutes les lignes du `--mindcube_split` choisi). Sinon `--max_train_samples` / `--max_samples` s’appliquent.

---

## 1) Qwen3-VL-4B

```bash
python train_lora_qwen3vl.py --output_dir ./out/qwen3 --bf16
python infer_mindcube_qwen3vl.py --adapter_dir ./out/qwen3 --bf16
```

Full 3DSRBench + MindCube tinybench entier :

```bash
python train_lora_qwen3vl.py --output_dir ./out/qwen3 --bf16 --full_dataset
python infer_mindcube_qwen3vl.py --adapter_dir ./out/qwen3 --bf16 --mindcube_split tinybench --full_dataset
```

## 2) LLaVA-NeXT 7B

```bash
python train_lora_llava.py --output_dir ./out/llava --bf16
python infer_mindcube_llava.py --adapter_dir ./out/llava --bf16
```

## 3) SpatialReasoner (`ccvl/SpatialReasoner`)

```bash
python train_lora_spatial_reasoner.py --output_dir ./out/spatial_reasoner --bf16
python infer_mindcube_spatial_reasoner.py --adapter_dir ./out/spatial_reasoner --bf16
```

## 4) Sa2VA-4B

```bash
python train_lora_sa2va.py --output_dir ./out/sa2va
python infer_mindcube_sa2va.py --adapter_dir ./out/sa2va
```

*(Si l’entraînement sort avec **code 2**, aucune loss exploitable : inférence base ou adaptateurs produits ailleurs ; `--adapter_dir` peut être omis pour base seule.)*

## 5) SpatialRGPT

```bash
export SPATIALRGPT_PATH=/chemin/SpatialRGPT
python train_lora_spatial_rgpt.py --output_dir ./out/spatial_rgpt
python infer_mindcube_spatial_rgpt.py --adapter_dir ./out/spatial_rgpt
```

*(Code 2 si `forward(..., labels=)` échoue sur ton build VILA.)*

---

Fichiers de temps : `timing_train.json` dans chaque `--output_dir` ; `timing_infer.jsonl` + `timing_infer_summary.json` dans chaque `--output_dir` d’inférence.
