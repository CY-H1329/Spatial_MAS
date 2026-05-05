"""
Unified loaders for vision benchmarks (CV-Bench, 3DSRBench, MMSI-Bench, …).
Returns normalized format: image(s), question, options (list or None), answer, category (optional).

Supports frozen benchmarks: when use_frozen=True (default), loads from data/frozen_benchmarks/
for reproducible paper experiments. Set use_frozen=False to load from HuggingFace with sampling.
"""
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import zipfile
from datasets import load_dataset
from PIL import Image
import io
import requests

# Frozen benchmark paths (DO NOT MODIFY - used for all paper experiments)
FROZEN_BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "frozen_benchmarks"
FROZEN_PATHS = {
    "cvbench": "cvbench_400",
    "3dsrbench": "3dsrbench_500",
}

# Task categories for Head-Agent classification (do NOT give to Head - it must infer)
SPATIAL_TASK_CATEGORIES = [
    "depth",
    "distance",
    "relation",
    "existence",
    "count",
    "instance_location",
    "orientation",
    "size",
    "reach",
]

BENCHMARK_CONFIGS = {
    "cvbench": {
        "name": "nyu-visionx/CV-Bench",
        "split": "test",
        "image_key": "image",
        "question_key": "question",
        "options_key": "choices",
        "answer_key": "answer",
        "category_key": "task",
    },
    "3dsrbench": {
        "name": "ccvl/3DSRBench",
        "split": "test",
        "subset": "benchmark",
        "image_key": "image_url",
        "question_key": "question",
        "options_keys": ["A", "B", "C", "D"],
        "answer_key": "answer",
        "category_key": "category",
    },
    # Multi-image spatial MCQ (HF decodes JPEGs). No frozen snapshot in repo.
    "mmsibench": {
        "name": "RunsenXu/MMSI-Bench",
        "split": "test",
        "images_key": "images",
        "question_key": "question",
        "answer_key": "answer",
        "category_key": "question_type",
        "answer_is_mcq_letter": True,
    },
    # MindCube: multi-image MCQ via a HF-hosted data.zip (jsonl + images paths).
    # We use the jsonl `gt_answer` as the MCQ letter.
    "mindcube": {
        "name": "MLL-Lab/MindCube",
        "split": "test",
        "images_key": "images",
        "question_key": "question",
        "answer_key": "gt_answer",
        "category_key": "category",
        "answer_is_mcq_letter": True,
    },
}

# Exact `question_type` strings on HuggingFace (RunsenXu/MMSI-Bench, test split).
MMSI_BENCH_QUESTION_TYPES = (
    "Attribute (Appr.)",
    "Attribute (Meas.)",
    "MSR",
    "Motion (Cam.)",
    "Motion (Obj.)",
    "Positional Relationship (Cam.–Cam.)",
    "Positional Relationship (Cam.–Obj.)",
    "Positional Relationship (Cam.–Reg.)",
    "Positional Relationship (Obj.–Obj.)",
    "Positional Relationship (Obj.–Reg.)",
    "Positional Relationship (Reg.–Reg.)",
)


def _fetch_image_from_url(url: str) -> Optional[Image.Image]:
    """Fetch image from URL. Returns PIL Image or None."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def load_benchmark(
    benchmark: str,
    max_samples: Optional[int] = None,
    max_per_category: Optional[int] = None,
    category_filter: Optional[List[str]] = None,
    seed: int = 42,
    use_frozen: bool = True,
):
    """
    Load a benchmark dataset.

    When use_frozen=True (default): loads from data/frozen_benchmarks/ for reproducible
    paper experiments. Ignores max_samples/max_per_category - returns exact frozen set.

    When use_frozen=False: loads from HuggingFace cache and applies sampling.

    Returns dataset with normalized access via get_benchmark_* helpers.

    Args:
        category_filter: If set (and use_frozen=False), keep only samples whose category is in this list.
    """
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(BENCHMARK_CONFIGS.keys())}")

    if benchmark == "mindcube":
        # MindCube isn't provided as a standard HF datasets table here (it is packaged as data.zip).
        # We download/extract it once and then load jsonl into a datasets.Dataset for compatibility.
        from datasets import Dataset

        ds = _load_mindcube_dataset(
            split="test",
            max_samples=max_samples,
            seed=seed,
        )
        return Dataset.from_list(ds)

    # Try frozen benchmark first
    if use_frozen and benchmark in FROZEN_PATHS:
        frozen_name = FROZEN_PATHS[benchmark]
        frozen_path = FROZEN_BENCHMARK_DIR / frozen_name
        if frozen_path.exists() and (frozen_path / "dataset_info.json").exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(frozen_path))
            return ds

    # Fallback: load from HuggingFace
    cfg = BENCHMARK_CONFIGS[benchmark]
    name = cfg["name"]
    split = cfg["split"]
    subset = cfg.get("subset")

    if subset:
        ds = load_dataset(name, subset, split=split)
    else:
        ds = load_dataset(name, split=split)

    rng = random.Random(seed)
    cat_key = cfg.get("category_key")

    if category_filter is not None and cat_key and cat_key in ds.features:
        cats_set = set(category_filter)
        indices = [i for i in range(len(ds)) if (ds[i].get(cat_key) or "").strip() in cats_set]
        ds = ds.select(indices)

    if max_per_category is not None and cat_key and cat_key in ds.features:
        by_cat = {}
        for i in range(len(ds)):
            c = ds[i].get(cat_key) or "unknown"
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(i)
        indices = []
        for c in sorted(by_cat.keys()):
            idx_list = by_cat[c]
            k = min(max_per_category, len(idx_list))
            indices.extend(rng.sample(idx_list, k))
        indices.sort()
        ds = ds.select(indices)
    elif max_samples is not None:
        n = min(max_samples, len(ds))
        indices = rng.sample(range(len(ds)), n)
        indices.sort()
        ds = ds.select(indices)

    return ds


def _mindcube_root() -> Path:
    # Default cache under repo data/. Override via env for shared storage.
    root = os.environ.get("MINDCUBE_DIR", "")
    if root.strip():
        return Path(root).expanduser()
    return Path(__file__).resolve().parent.parent.parent / "data" / "mindcube"


def _ensure_mindcube_extracted() -> Path:
    """
    Ensure MindCube `data.zip` is downloaded and extracted.
    Returns extraction root (contains `data/raw/*.jsonl` and `data/other_all_image/...`).
    """
    out_root = _mindcube_root()
    raw_jsonl = out_root / "data" / "raw" / "MindCube.jsonl"
    if raw_jsonl.exists():
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise ImportError("MindCube requires huggingface_hub. pip install huggingface_hub") from e

    zip_path = hf_hub_download(
        repo_id="MLL-Lab/MindCube",
        repo_type="dataset",
        filename="data.zip",
    )
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(str(out_root))
    if not raw_jsonl.exists():
        raise FileNotFoundError(f"MindCube extraction failed: missing {raw_jsonl}")
    return out_root


def _load_mindcube_dataset(split: str = "test", max_samples: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """
    Load MindCube jsonl into a list[dict]. Each dict includes:
    - question: str (already contains options A-D)
    - images: list[str] (relative paths under extracted root)
    - gt_answer: 'A'|'B'|'C'|'D'
    - category: list[str] (4-d taxonomy)
    """
    root = _ensure_mindcube_extracted()
    if split in ("test", "eval"):
        jsonl_path = root / "data" / "raw" / "MindCube.jsonl"
    elif split in ("train",):
        jsonl_path = root / "data" / "raw" / "MindCube_train.jsonl"
    elif split in ("tiny", "tinybench"):
        jsonl_path = root / "data" / "raw" / "MindCube_tinybench.jsonl"
    else:
        raise ValueError("MindCube split must be one of: test, train, tinybench")

    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Normalize schema for pyarrow/Datasets:
            # some fields can be scalar or list depending on the example.
            imgs = obj.get("images")
            if imgs is None:
                obj["images"] = []
            elif not isinstance(imgs, list):
                obj["images"] = [imgs]
            cat = obj.get("category")
            if cat is None:
                obj["category"] = []
            elif not isinstance(cat, list):
                obj["category"] = [cat]
            rows.append(obj)

    if max_samples is None or max_samples >= len(rows):
        return rows

    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), min(max_samples, len(rows)))
    idx.sort()
    return [rows[i] for i in idx]


def get_benchmark_image(example: Dict, benchmark: str) -> Optional[Image.Image]:
    """Extract a single PIL Image (None for multi-image benchmarks — use get_benchmark_images)."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    if cfg.get("images_key"):
        return None
    img_key = cfg.get("image_key")
    if not img_key:
        return None

    if img_key == "image_url":
        url = example.get(img_key)
        if url:
            return _fetch_image_from_url(url)
        return None

    img = example.get("images") or example.get("image")
    if img is None:
        return None
    if hasattr(img, "convert"):
        return img.convert("RGB")
    return img


def get_benchmark_images(example: Dict, benchmark: str) -> List[Image.Image]:
    """All images for the example (MMSI-Bench: 2–10 frames). Single-image benchmarks return a list of length 1."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    key = cfg.get("images_key")
    if key:
        raw = example.get(key) or []
        out: List[Image.Image] = []
        for im in raw:
            if im is None:
                continue
            if benchmark == "mindcube":
                # MindCube stores relative file paths inside extracted dataset.
                try:
                    root = _mindcube_root()
                    p = (root / "data" / im) if not str(im).startswith("data/") else (root / im)
                    if not p.exists():
                        # fallback: allow paths without `data/` prefix
                        p = root / str(im)
                    if p.exists():
                        out.append(Image.open(p).convert("RGB"))
                        continue
                except Exception:
                    # fall through to other decoding paths
                    pass
            if hasattr(im, "convert"):
                out.append(im.convert("RGB"))
            elif isinstance(im, (bytes, bytearray)):
                try:
                    out.append(Image.open(io.BytesIO(im)).convert("RGB"))
                except Exception:
                    continue
        return out
    one = get_benchmark_image(example, benchmark)
    return [one] if one is not None else []


def get_benchmark_prompt(example: Dict, benchmark: str, include_options: bool = True) -> str:
    """Build prompt (question + options if any)."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    q_key = cfg["question_key"]
    question = example.get(q_key) or ""

    if not include_options:
        return question

    opts_key = cfg.get("options_key")
    opts_keys = cfg.get("options_keys")

    if opts_key and opts_key in example:
        opts = example[opts_key]
        if opts:
            lines = [question, "Options:"]
            for i, o in enumerate(opts):
                label = chr(65 + i)
                lines.append(f"({label}) {o}")
            return "\n".join(lines)
    elif opts_keys:
        opts = [example.get(k) for k in opts_keys if example.get(k)]
        if opts:
            lines = [question, "Options:"]
            for i, o in enumerate(opts):
                label = chr(65 + i)
                lines.append(f"({label}) {o}")
            return "\n".join(lines)

    return question


def get_benchmark_answer(example: Dict, benchmark: str) -> str:
    """Ground-truth answer (letter A/B/C/D or raw string)."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    ans_key = cfg["answer_key"]
    ans = example.get(ans_key) or ""
    s = str(ans).strip()
    if cfg.get("answer_is_mcq_letter"):
        u = s.upper()
        for c in "ABCDEF":
            if u == c or u.startswith(f"{c}:") or u.startswith(f"{c})"):
                return c
        return u[:1] if u and u[0] in "ABCDEF" else ""
    # Benchmarks with options: extract letter (CV-Bench uses "(C)")
    if cfg.get("options_key") or cfg.get("options_keys"):
        for c in "ABCDEF":
            if f"({c})" in s.upper() or s.upper() == c:
                return c
    return s


def get_benchmark_category(example: Dict, benchmark: str) -> Optional[str]:
    """Category if available (for evaluation only - never pass to Head-Agent)."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    cat_key = cfg.get("category_key")
    if cat_key and cat_key in example:
        v = example[cat_key]
        if benchmark == "mindcube" and isinstance(v, list):
            # 4-axis taxonomy → a stable string key
            return "/".join(str(x) for x in v)
        return str(v)
    return None
