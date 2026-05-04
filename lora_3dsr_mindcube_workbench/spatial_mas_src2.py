"""Ajoute `src2` au path pour importer les runners du dépôt Spatial_MAS parent."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def spatial_mas_root() -> Path:
    env = os.environ.get("SPATIAL_MAS_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "src2" / "models").is_dir():
            return p
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here.parent.parent):
        if (cand / "src2" / "models").is_dir():
            return cand
    raise RuntimeError(
        "Impossible de trouver Spatial_MAS (dossier contenant src2/models). "
        "Placez `lora_3dsr_mindcube_workbench` sous Spatial_MAS/ ou exportez SPATIAL_MAS_ROOT."
    )


def insert_src2() -> Path:
    root = spatial_mas_root()
    s2 = root / "src2"
    if str(s2) not in sys.path:
        sys.path.insert(0, str(s2))
    return root
