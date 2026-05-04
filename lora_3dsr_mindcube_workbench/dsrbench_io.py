"""Chargement 3DSRBench (HF) + cache des images URL."""
from __future__ import annotations

import hashlib
import io
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import load_dataset
from PIL import Image


def _url_cache_name(url: str) -> str:
    return hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()


def fetch_image(url: str, cache_dir: Optional[Path] = None, timeout: int = 60) -> Image.Image:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / f"{_url_cache_name(url)}.jpg"
        if cpath.exists():
            return Image.open(cpath).convert("RGB")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    if cache_dir is not None:
        cpath = cache_dir / f"{_url_cache_name(url)}.jpg"
        img.save(cpath, quality=95)
    return img


def load_3dsrbench_rows(
    max_samples: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Retourne des dicts normalisés: image (PIL), question, options (str), answer (A-D), category."""
    t0 = time.perf_counter()
    # `trust_remote_code` n'est plus pris en charge pour ce dataset (datasets récents) — ne pas le passer.
    ds = load_dataset("ccvl/3DSRBench", name="benchmark", split="test")
    rows = list(ds)
    timing: Dict[str, float] = {"hf_load_dataset_s": time.perf_counter() - t0}
    if max_samples is not None and max_samples < len(rows):
        rng = random.Random(seed)
        idx = rng.sample(range(len(rows)), max_samples)
        idx.sort()
        rows = [rows[i] for i in idx]

    out: List[Dict[str, Any]] = []
    t_img = time.perf_counter()
    for ex in rows:
        url = ex.get("image_url") or ex.get("image")
        if isinstance(url, dict) and "url" in url:
            url = url["url"]
        if not url:
            continue
        image = fetch_image(str(url), cache_dir=cache_dir)
        opts = []
        for k in ("A", "B", "C", "D"):
            if k in ex and ex[k] is not None:
                opts.append(f"{k}. {ex[k]}")
        options_block = "\n".join(opts)
        letter = str(ex.get("answer", "")).strip().upper()[:1]
        if letter not in ("A", "B", "C", "D"):
            continue
        out.append(
            {
                "image": image,
                "question": str(ex.get("question", "")),
                "options_block": options_block,
                "answer": letter,
                "category": ex.get("category"),
            }
        )
    timing["image_fetch_total_s"] = time.perf_counter() - t_img
    return out, timing


def build_user_text(question: str, options_block: str) -> str:
    return (
        f"{question}\n\n{options_block}\n\n"
        "Réponds uniquement par une seule lettre : A, B, C ou D."
    )
