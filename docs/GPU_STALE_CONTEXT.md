# GPU stale context (zombie VRAM)

## Symptômes

```text
nvidia-smi  →  Memory-Usage: 76639 MiB / 95830 MiB
Processes table → empty (or PIDs with [Not Found])
kill -9 832062 → No such process
```

Les PID (`832062`, etc.) appartiennent à un **autre namespace** (host, ancien conteneur) ou à un processus déjà mort dont le driver NVIDIA n’a pas libéré la VRAM.

## Ce qui ne marche pas

- `kill -9` / `sudo kill -9` → `No such process`
- `gpu_cleanup.sh --kill-all` → Killed 0 process(es)
- Relancer SpatiO sans redémarrer → OOM (seulement ~17 GiB réellement libres)

## Solutions (par ordre)

### 1. Redémarrer la session (recommandé)

Sur JupyterHub / Kubernetes :

- **Kernel → Restart** puis fermer tous les notebooks qui utilisent le GPU
- Ou **Stop/Start le pod** / redémarrer le conteneur `8467fe80c227`
- Ou se déconnecter et rouvrir une nouvelle session GPU

Après redémarrage, `nvidia-smi` doit afficher **< 1 GiB** utilisé.

### 2. GPU reset (admin / sudo)

```bash
sudo nvidia-smi --gpu-reset -i 0
# ou
bash scripts/gpu_cleanup.sh --reset
```

Peut échouer si le GPU est considéré comme occupé.

### 3. Continuer avec VRAM partielle (~17 GiB libres)

Si un redémarrage n’est pas possible tout de suite, PyTorch voit parfois encore **~17 GiB** libres (`torch.cuda.mem_get_info()`). Test rapide uniquement :

```bash
LOW_MEMORY=1 \
REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
bash experiments/spatio/run_h100.sh quick
```

Le pipeline charge **un seul modèle sur GPU à la fois** (`--specialist_offload_after_use`).  
Full dataset (2638 samples) **n’est pas recommandé** tant que ~74 GiB restent bloqués.

## Vérification

```bash
nvidia-smi
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'free {f/2**30:.1f} GiB / {t/2**30:.1f} GiB')"
```

GPU sain : `nvidia-smi` **< 5 GiB** used et **free > 80 GiB**.
