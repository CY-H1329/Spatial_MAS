#!/usr/bin/env python3
"""Régénère timing_infer_by_category.json à partir d'un timing_infer.jsonl existant.

Utile si le JSONL a été produit avant l'ajout des champs sample_id / category sur chaque ligne :
on réinjecte les métadonnées depuis un fichier MindCube JSONL en supposant le même ordre que
l'inférence (même split, pas de sous-échantillonnage aléatoire, ou bien mêmes paramètres que le run).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from mindcube_io import aggregate_timing_infer_by_category, mindcube_row_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timing_infer_jsonl", type=str, required=True)
    p.add_argument("--mindcube_jsonl", type=str, required=True, help="Ex. .../data/raw/MindCube.jsonl")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Fichier JSON de sortie (défaut : même répertoire que le jsonl, timing_infer_by_category.json)",
    )
    args = p.parse_args()

    infer_path = Path(args.timing_infer_jsonl)
    mc_path = Path(args.mindcube_jsonl)
    out_path = Path(args.output) if args.output.strip() else infer_path.parent / "timing_infer_by_category.json"

    rows: List[Dict[str, Any]] = []
    with open(mc_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    step_records: List[Dict[str, Any]] = []
    with open(infer_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") == "model_load":
                continue
            if "i" not in rec:
                continue
            i = int(rec["i"])
            if i < 0 or i >= len(rows):
                raise SystemExit(f"Indice i={i} hors plage (mindcube lignes={len(rows)})")
            meta = mindcube_row_meta(rows[i])
            step_records.append({**rec, **meta})

    report = aggregate_timing_infer_by_category(step_records)
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
    print(json.dumps({"written": str(out_path), "steps": report.get("steps")}, indent=2))


if __name__ == "__main__":
    main()
