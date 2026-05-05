#!/usr/bin/env python3
"""Inférence MindCube — Sa2VA (ByteDance) + LoRA optionnel (PEFT sur le même `AutoModel`)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from mc_common import mcq_user_suffix, parse_letter, tile_images_for_single_image_backend
from mindcube_io import Split, ensure_mindcube_extracted, load_mindcube_images, load_mindcube_rows, mindcube_root
from spatial_mas_src2 import insert_src2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="ByteDance/Sa2VA-4B")
    p.add_argument("--adapter_dir", type=str, default="", help="Vide = base seule ; sinon dossier PEFT")
    p.add_argument("--mindcube_split", type=str, default="tinybench", choices=["test", "train", "tinybench"])
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--full_dataset", action="store_true", help="Tout le split MindCube.")
    p.add_argument("--output_dir", type=str, default="./outputs/mindcube_infer_sa2va")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    insert_src2()
    from models.sa2va import Sa2VARunner
    from peft import PeftModel

    split: Split = args.mindcube_split  # type: ignore[assignment]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "timing_infer.jsonl"

    ensure_mindcube_extracted()
    cap_rows = None if args.full_dataset else args.max_samples
    rows = load_mindcube_rows(split=split, max_samples=cap_rows, seed=42)

    t0 = time.perf_counter()
    runner = Sa2VARunner(model_id=args.model_id, device="cuda")
    if args.adapter_dir.strip():
        runner.model = PeftModel.from_pretrained(runner.model, args.adapter_dir)
    runner.model.eval()
    load_model_s = time.perf_counter() - t0

    correct = 0
    total = 0
    latencies: List[float] = []

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        fj.write(json.dumps({"event": "model_load", "backend": "sa2va", "load_model_s": load_model_s}, ensure_ascii=False) + "\n")
        for i, row in enumerate(tqdm(rows, desc="MindCube Sa2VA")):
            t_sample = time.perf_counter()
            gt = str(row.get("gt_answer", "")).strip().upper()[:1]
            rels = row.get("images") or []
            if not isinstance(rels, list):
                rels = [rels]
            images = load_mindcube_images(mindcube_root(), [str(x) for x in rels])
            tiled = tile_images_for_single_image_backend(images)
            prompt = mcq_user_suffix(str(row.get("question", "")))

            t_gen = time.perf_counter()
            text = runner.generate(tiled, prompt, max_new_tokens=8, temperature=0.0)
            generate_s = time.perf_counter() - t_gen

            pred = parse_letter(text) or ""
            ok = pred == gt and gt in ("A", "B", "C", "D")
            if gt in ("A", "B", "C", "D"):
                total += 1
                correct += int(ok)
            latencies.append(time.perf_counter() - t_sample)
            fj.write(
                json.dumps(
                    {
                        "i": i,
                        "generate_s": generate_s,
                        "total_sample_s": latencies[-1],
                        "pred_raw": text,
                        "pred": pred,
                        "gt": gt,
                        "correct": ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "backend": "sa2va",
        "mindcube_split": split,
        "full_dataset": bool(args.full_dataset),
        "rows_evaluated": len(rows),
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total_scored": total,
        "load_model_s": load_model_s,
        "mean_total_sample_s": sum(latencies) / max(len(latencies), 1),
    }
    with open(out_dir / "timing_infer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
