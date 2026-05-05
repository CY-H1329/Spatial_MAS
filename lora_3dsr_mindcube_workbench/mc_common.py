"""Utilitaires partagés (MCQ, tuilage multi-vues pour backends mono-image)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from PIL import Image


def log_infer(tag: str, message: str) -> None:
    """Affiche une étape (stdout, flush) pendant les longs chargements HF."""
    print(f"[{tag}] {message}", flush=True)


def log_train(tag: str, message: str) -> None:
    """Affiche une étape pendant l’entraînement LoRA."""
    print(f"[{tag}] {message}", flush=True)


def collate_list_of_dicts(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pour `DataLoader` : ne pas fusionner les dicts (champs PIL, str, etc.)."""
    return batch


def parse_letter(text: str) -> Optional[str]:
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m2 = re.search(r"[ABCD]", text.upper())
    return m2.group(0) if m2 else None


def mcq_user_suffix(question: str) -> str:
    return question + "\n\nRéponds uniquement par une seule lettre : A, B, C ou D."


def tile_images_for_single_image_backend(images: List[Image.Image], max_side: int = 896) -> Image.Image:
    """Grille 2×2 (noir si <4 images), redimensionnement global — pour LLaVA / Sa2VA / SpatialRGPT mono-image."""
    n = len(images)
    if n == 0:
        raise ValueError("tile_images: liste vide")
    if n == 1:
        im = images[0].convert("RGB")
        im.thumbnail((max_side, max_side))
        return im
    slots = 4
    while len(images) < slots:
        images = list(images) + [Image.new("RGB", (64, 64), (0, 0, 0))]
    images = [im.convert("RGB") for im in images[:slots]]
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    grid = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
    for idx, im in enumerate(images):
        im = im.resize((w, h))
        x = (idx % 2) * w
        y = (idx // 2) * h
        grid.paste(im, (x, y))
    grid.thumbnail((max_side, max_side))
    return grid
