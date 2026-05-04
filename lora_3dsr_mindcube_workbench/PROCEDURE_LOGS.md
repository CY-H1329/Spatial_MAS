# Procédure d’entraînement LoRA (3DSRBench) et fichiers de temps

## Ordre d’exécution (`train_lora_qwen3vl.py` et équivalents)

1. **Chargement des exemples 3DSRBench** — `datasets.load_dataset("ccvl/3DSRBench", name="benchmark", split="test")`, puis sous-échantillon (`--max_train_samples`), téléchargement/cache des images URL (`--image_cache`).
2. **Chargement du modèle + processeur** — HF `from_pretrained`, passage sur GPU (`cuda`), dtype `--bf16` si demandé.
3. **Injection LoRA (PEFT)** — `get_peft_model`, affichage des paramètres entraînables.
4. **Boucle d’epochs** — batch size 1, une étape `forward` + `backward` + optimiseur par exemple.
5. **Sauvegarde** — adaptateurs + processeur dans `--output_dir`.
6. **Écriture des temps** — fichier unique **`timing_train.json`** à la fin (et en sortie console).

Les messages **tqdm** pendant l’epoch indiquent la progression ; ce n’est **pas** un second fichier de log ligne par ligne pendant le train.

## Contenu typique de `timing_train.json`

| Clé | Signification |
|-----|-----------------|
| `data_3dsrbench.hf_load_dataset_s` | Temps pour charger le split HF en mémoire. |
| `data_3dsrbench.image_fetch_total_s` | Temps cumulé pour télécharger/charger les images URL (avec cache après le 1er run). |
| `data_3dsrbench.rows_used` | Nombre d’exemples utilisés pour l’entraînement. |
| `steps[]` | Liste `{ "name": "...", "s": ... }` : durées des grosses étapes (chargement données, modèle, LoRA, sauvegarde). |
| `epochs_s` | Liste (une entrée par epoch) : durée totale de chaque epoch. |
| `train_total_s` | Durée totale script (données + entraînement + sauvegarde). |
| `per_epoch_mean_s` | Moyenne des durées d’epoch. |

## Inférence MindCube (après le train)

Fichiers dans `--output_dir` de l’infer :

- **`timing_infer.jsonl`** — une ligne JSON par échantillon (`preprocess_s`, `generate_s`, `total_sample_s`, etc.).
- **`timing_infer_summary.json`** — agrégats (`load_model_s`, `mean_total_sample_s`, `accuracy`, …).

---

## Avertissements fréquents (non bloquants pour le train)

### Torchvision / `image.so` / symbole manquant

Conflit de versions **PyTorch** vs **torchvision** (ou build conda incohérent). Le script 3DSRBench utilise **PIL** + **requests** pour les images, pas `torchvision.io` : en général **tu peux ignorer** l’avertissement tant que l’entraînement tourne. Pour le faire disparaître : réinstaller `torch` et `torchvision` **compatibles** depuis le même canal (ex. `conda install pytorch torchvision -c pytorch` aligné sur ta version de CUDA).

### `trust_remote_code` sur le **dataset** 3DSRBench

Avec **datasets** récents, `trust_remote_code` pour `load_dataset` n’est plus supporté pour ce hub. Le dépôt a été corrigé pour **ne plus** passer ce flag au dataset (seuls les **modèles** HF gardent `trust_remote_code=True` si nécessaire).
