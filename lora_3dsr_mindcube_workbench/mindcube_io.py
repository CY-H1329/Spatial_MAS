"""Téléchargement / extraction MindCube (data.zip) et lecture JSONL."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

Split = Literal["test", "train", "tinybench"]


def mindcube_root() -> Path:
    root = os.environ.get("MINDCUBE_DIR", "").strip()
    if root:
        return Path(root).expanduser()
    return Path(__file__).resolve().parent / "data" / "mindcube"


def ensure_mindcube_extracted() -> Path:
    out_root = mindcube_root()
    raw_jsonl = out_root / "data" / "raw" / "MindCube.jsonl"
    if raw_jsonl.exists():
        return out_root
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Installez huggingface_hub : pip install huggingface_hub") from e
    zip_path = hf_hub_download(
        repo_id="MLL-Lab/MindCube",
        repo_type="dataset",
        filename="data.zip",
    )
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(str(out_root))
    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Extraction MindCube invalide : {raw_jsonl} manquant")
    return out_root


def jsonl_path_for_split(root: Path, split: Split) -> Path:
    if split in ("test", "eval"):
        return root / "data" / "raw" / "MindCube.jsonl"
    if split == "train":
        return root / "data" / "raw" / "MindCube_train.jsonl"
    if split in ("tiny", "tinybench"):
        return root / "data" / "raw" / "MindCube_tinybench.jsonl"
    raise ValueError(f"split inconnu: {split}")


def load_mindcube_rows(split: Split = "test", max_samples: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    root = ensure_mindcube_extracted()
    path = jsonl_path_for_split(root, split)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if max_samples is None or max_samples >= len(rows):
        return rows
    import random

    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), min(max_samples, len(rows)))
    idx.sort()
    return [rows[i] for i in idx]


def resolve_mindcube_image_path(root: Path, rel: str) -> Path:
    """Chemins relatifs dans le JSONL → fichier sous la racine extraite."""
    s = str(rel)
    if s.startswith("data/"):
        p = root / s
    else:
        p = root / "data" / s
    if p.exists():
        return p
    alt = root / s
    if alt.exists():
        return alt
    return p


def load_mindcube_images(root: Path, rel_paths: List[str]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for rel in rel_paths:
        p = resolve_mindcube_image_path(root, rel)
        if not p.exists():
            raise FileNotFoundError(f"Image MindCube introuvable : {rel} → {p}")
        out.append(Image.open(p).convert("RGB"))
    return out


def mindcube_row_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    """Champs dataset pour logs d'inférence (alignement sans dépendre du seul index)."""
    raw = row.get("category")
    if raw is None:
        category: List[Any] = []
    elif isinstance(raw, list):
        category = list(raw)
    else:
        category = [raw]
    return {
        "sample_id": row.get("id"),
        "category": category,
        "mindcube_type": row.get("type"),
    }


def aggregate_timing_infer_by_category(step_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrège précision et temps par catégorie / type à partir des lignes infer_step."""
    by_cat0: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "sum_total": 0.0})
    by_type: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "sum_total": 0.0})
    by_tag: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "sum_total": 0.0})
    n_scored = 0
    sum_correct = 0
    sum_time = 0.0
    n_all = 0
    for rec in step_records:
        n_all += 1
        ok = bool(rec.get("correct"))
        tot = float(rec.get("total_sample_s", 0.0))
        sum_time += tot
        if rec.get("gt") in ("A", "B", "C", "D"):
            n_scored += 1
            sum_correct += int(ok)
        cat = rec.get("category")
        if not isinstance(cat, list):
            cat = [] if cat is None else [cat]
        c0 = str(cat[0]) if cat else "(none)"
        by_cat0[c0]["n"] += 1
        by_cat0[c0]["correct"] += int(ok)
        by_cat0[c0]["sum_total"] += tot
        mt = rec.get("mindcube_type")
        tkey = str(mt) if mt is not None else "(none)"
        by_type[tkey]["n"] += 1
        by_type[tkey]["correct"] += int(ok)
        by_type[tkey]["sum_total"] += tot
        for tag in cat:
            s = str(tag)
            by_tag[s]["n"] += 1
            by_tag[s]["correct"] += int(ok)
            by_tag[s]["sum_total"] += tot

    def _finalize(m: Dict[str, dict]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in sorted(m.items(), key=lambda x: (-x[1]["n"], x[0])):
            n = v["n"]
            out[k] = {
                "count": n,
                "accuracy": (v["correct"] / n) if n else 0.0,
                "correct": v["correct"],
                "mean_total_sample_s": (v["sum_total"] / n) if n else 0.0,
            }
        return out

    return {
        "steps": n_all,
        "scored": n_scored,
        "overall": {
            "accuracy": (sum_correct / n_scored) if n_scored else 0.0,
            "correct": sum_correct,
            "mean_total_sample_s": (sum_time / n_all) if n_all else 0.0,
        },
        "by_category_first": _finalize(dict(by_cat0)),
        "by_mindcube_type": _finalize(dict(by_type)),
        "by_category_tag_any": _finalize(dict(by_tag)),
    }
