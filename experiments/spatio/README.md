# SpatiO — Reproduction H100

Architecture MAS v2 :

```
Image + Query → Head Agent (Qwen3-VL) → ScoreMap → 3 Specialists → SharedMemory → Final Reasoning (DeepSeek-R1)
```

Benchmarks supportés : **CV-Bench**, **3DSRBench**, **STVQA-7K**, **MindCube**.

## 1. Push (local → GitHub)

```bash
cd /Users/flaxinger/Desktop/Spatial_MAS_github

git status
git add experiments/spatio/
git commit -m "SpatiO: unified H100 reproduction script for CV-Bench, 3DSRBench, STVQA"
git push origin main
```

## 2. Pull (H100 ← GitHub)

Si la branche locale diverge de `origin/main` :

```bash
cd ~/CY/Spatial_MAS
git fetch origin
git reset --hard origin/main
```

Sinon :

```bash
cd ~/CY/Spatial_MAS
git pull origin main
```

## 3. Exécution H100

```bash
cd ~/CY/Spatial_MAS
conda activate spatial_reasoning   # ou votre env GPU

# Test rapide (10 samples CV-Bench)
bash experiments/spatio/run_h100.sh quick

# Reproduction complète MAS v2 (CV-Bench + 3DSRBench, 10/50/100 samples)
bash experiments/spatio/run_h100.sh baseline

# SpatiO avec TTO (train score map + eval frozen)
bash experiments/spatio/run_h100.sh tto-cvbench
bash experiments/spatio/run_h100.sh tto-3dsrbench

# STVQA-7K (single-agent, 5 modèles)
export SPATIALRGPT_PATH=/path/to/SpatialRGPT   # optionnel pour spatialrgpt
bash experiments/spatio/run_h100.sh stvqa
```

## 4. Options utiles

| Variable | Description |
|----------|-------------|
| `LOW_MEMORY=1` | 3 agents seulement (évite OOM) |
| `TEMPERATURE=0.7` | Sampling (défaut 0 = greedy) |
| `REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Modèle reasoning plus petit |
| `MAX_SAMPLES=10` | Limite STVQA (via `experiments/stvqa7k/run_h100.sh`) |
| `SPATIALRGPT_PATH` | Chemin SpatialRGPT pour spatial_rgpt |

Exemple OOM :

```bash
LOW_MEMORY=1 REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  bash experiments/spatio/run_h100.sh quick
```

## 5. Résultats

| Mode | Dossier |
|------|---------|
| `baseline` / `quick` | `results/spatio/mas_v2_baseline/{benchmark}/{n}samples/{timestamp}/` |
| `tto-*` | `results/spatialtto_*` |
| `stvqa` | `results/stvqa7k/` |

Fichiers par run MAS v2 :
- `summary.json` — accuracy globale + par catégorie
- `details.jsonl` — résultats par sample
