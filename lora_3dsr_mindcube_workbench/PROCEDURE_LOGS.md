# Procédure d’entraînement LoRA (3DSRBench) et fichiers de temps

## Ordre d’exécution (`train_lora_qwen3vl.py` et équivalents)

1. **Chargement des exemples 3DSRBench** — `datasets.load_dataset("ccvl/3DSRBench", name="benchmark", split="test")`, puis sous-échantillon (`--max_train_samples`), téléchargement/cache des images URL (`--image_cache`).
2. **Chargement du modèle + processeur** — HF `from_pretrained`, passage sur GPU (`cuda`), dtype `--bf16` si demandé.
3. **Injection LoRA (PEFT)** — `get_peft_model`, affichage des paramètres entraînables.
4. **Boucle d’epochs** — batch size 1, une étape `forward` + `backward` + optimiseur par exemple.
5. **Sauvegarde** — adaptateurs + processeur dans `--output_dir`.
6. **Écriture des temps** — fichier unique **`timing_train.json`** à la fin (et en sortie console), plus en parallèle **`timing_train_steps.jsonl`** (une ligne JSON par step optimiseur, équivalent « par requête » avec batch 1), sauf si tu passes **`--no_train_step_log`**.

Les messages **tqdm** et, selon le backend, des logs console (`log_train` / `log_infer`) indiquent la progression.

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
| `timing_train_steps_jsonl` | Nom du fichier `timing_train_steps.jsonl` (ou `null` si `--no_train_step_log`). |

### `timing_train_steps.jsonl` (entraînement)

Chaque ligne est un objet JSON, typiquement avec `event: "train_step"`, `backend`, `step`, `epoch`, `index_in_epoch`, `loss`, `step_s` (secondes du step). Certains backends peuvent écrire des entrées `skipped` ou `error` au lieu d’un step avec loss.

## Inférence MindCube (après le train)

Fichiers dans `--output_dir` de l’infer :

- **`timing_infer.jsonl`** — une ligne JSON par échantillon, schéma unifié : `event: "infer_step"`, `backend`, `i`, `load_images_s`, `preprocess_s`, `generate_s`, `total_sample_s`, prédictions et `correct` si scoring disponible, plus **`sample_id`**, **`category`** (liste issue du JSONL MindCube) et **`mindcube_type`**. Pour les runners « monolithiques » (ex. Sa2VA / SpatialRGPT), `preprocess_s` peut être `0.0`.
- **`timing_infer_summary.json`** — agrégats (`load_model_s`, `mean_total_sample_s`, `sum_total_sample_s`, `timing_infer_jsonl`, `timing_infer_by_category_json`, `accuracy`, …).
- **`timing_infer_by_category.json`** — résumé **par catégorie** : `by_category_first` (1er tag), `by_mindcube_type`, `by_category_tag_any` (chaque tag de la liste), plus `overall` recalculé.
- Ancien JSONL sans métadonnées : `python rebuild_timing_infer_by_category.py --timing_infer_jsonl ... --mindcube_jsonl ...` (même ordre d’exemples que le JSONL MindCube utilisé pour l’éval).

---

## Avertissements fréquents (non bloquants pour le train)

### Torchvision / `image.so` / symbole manquant

Conflit de versions **PyTorch** vs **torchvision** (ou build conda incohérent). Le script 3DSRBench utilise **PIL** + **requests** pour les images, pas `torchvision.io` : en général **tu peux ignorer** l’avertissement tant que l’entraînement tourne. Pour le faire disparaître : réinstaller `torch` et `torchvision` **compatibles** depuis le même canal (ex. `conda install pytorch torchvision -c pytorch` aligné sur ta version de CUDA).

### `trust_remote_code` sur le **dataset** 3DSRBench

Avec **datasets** récents, `trust_remote_code` pour `load_dataset` n’est plus supporté pour ce hub. Le dépôt a été corrigé pour **ne plus** passer ce flag au dataset (seuls les **modèles** HF gardent `trust_remote_code=True` si nécessaire).
