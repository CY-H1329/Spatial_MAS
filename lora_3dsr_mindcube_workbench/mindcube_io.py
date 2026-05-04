"""Téléchargement / extraction MindCube (data.zip) et lecture JSONL."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
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
