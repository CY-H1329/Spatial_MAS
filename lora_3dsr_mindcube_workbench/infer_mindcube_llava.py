#!/usr/bin/env python3
"""Inférence MindCube — LLaVA-NeXT + LoRA (vues fusionnées en une grille)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from tqdm import tqdm

from mc_common import log_infer, mcq_user_suffix, parse_letter, tile_images_for_single_image_backend
from mindcube_io import (
    Split,
    aggregate_timing_infer_by_category,
    ensure_mindcube_extracted,
    load_mindcube_images,
    load_mindcube_rows,
    mindcube_row_meta,
    mindcube_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--mindcube_split", type=str, default="tinybench", choices=["test", "train", "tinybench"])
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--full_dataset", action="store_true", help="Tout le split MindCube.")
    p.add_argument("--output_dir", type=str, default="./outputs/mindcube_infer_llava")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main() -> None:
    TAG = "infer_llava"
    args = parse_args()
    log_infer(TAG, "Démarrage (warnings torchvision souvent inoffensifs).")
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    split: Split = args.mindcube_split  # type: ignore[assignment]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "timing_infer.jsonl"

    log_infer(TAG, "MindCube (data.zip)…")
    ensure_mindcube_extracted()
    cap_rows = None if args.full_dataset else args.max_samples
    log_infer(TAG, f"Lecture JSONL split={split}…")
    rows = load_mindcube_rows(split=split, max_samples=cap_rows, seed=42)
    log_infer(TAG, f"{len(rows)} exemples à évaluer.")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    t0 = time.perf_counter()
    log_infer(TAG, "Chargement processeur + LLaVA-NeXT (plusieurs minutes possibles)…")
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)
    base = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model_id, dtype=dtype, trust_remote_code=True
    )
    log_infer(TAG, "Transfert GPU + adaptateurs PEFT…")
    base = base.to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    dev = next(model.parameters()).device
    load_model_s = time.perf_counter() - t0
    log_infer(TAG, f"Prêt en {load_model_s:.1f}s — génération (tqdm).")

    correct = 0
    total = 0
    latencies: List[float] = []
    step_records: List[Dict[str, Any]] = []

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        fj.write(json.dumps({"event": "model_load", "load_model_s": load_model_s, "backend": "llava_next"}, ensure_ascii=False) + "\n")
        for i, row in enumerate(tqdm(rows, desc="MindCube LLaVA")):
            t_sample = time.perf_counter()
            gt = str(row.get("gt_answer", "")).strip().upper()[:1]
            rels = row.get("images") or []
            if not isinstance(rels, list):
                rels = [rels]
            t_img0 = time.perf_counter()
            images = load_mindcube_images(mindcube_root(), [str(x) for x in rels])
            tiled = tile_images_for_single_image_backend(images)
            load_images_s = time.perf_counter() - t_img0
            prompt = mcq_user_suffix(str(row.get("question", "")))

            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            ]
            prompt_str = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            t_pre = time.perf_counter()
            try:
                inputs = processor(tiled, prompt_str, return_tensors="pt").to(dev)
            except TypeError:
                inputs = processor(images=[tiled], text=[prompt_str], padding=True, return_tensors="pt").to(dev)
            preprocess_s = time.perf_counter() - t_pre

            t_gen = time.perf_counter()
            with torch.inference_mode():
                gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            generate_s = time.perf_counter() - t_gen

            start = inputs["input_ids"].shape[1]
            text = processor.decode(gen_ids[0][start:], skip_special_tokens=True)
            pred = parse_letter(text) or ""
            ok = pred == gt and gt in ("A", "B", "C", "D")
            if gt in ("A", "B", "C", "D"):
                total += 1
                correct += int(ok)
            latencies.append(time.perf_counter() - t_sample)
            rec: Dict[str, Any] = {
                "event": "infer_step",
                "backend": "llava_next",
                "i": i,
                "load_images_s": load_images_s,
                "preprocess_s": preprocess_s,
                "generate_s": generate_s,
                "total_sample_s": latencies[-1],
                "pred_raw": text,
                "pred": pred,
                "gt": gt,
                "correct": ok,
                **mindcube_row_meta(row),
            }
            step_records.append(rec)
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_cat_path = out_dir / "timing_infer_by_category.json"
    with open(by_cat_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_timing_infer_by_category(step_records), f, indent=2, ensure_ascii=False)

    summary = {
        "backend": "llava_next",
        "mindcube_split": split,
        "full_dataset": bool(args.full_dataset),
        "rows_evaluated": len(rows),
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total_scored": total,
        "load_model_s": load_model_s,
        "mean_total_sample_s": sum(latencies) / max(len(latencies), 1),
        "sum_total_sample_s": sum(latencies),
        "timing_infer_jsonl": str(jsonl_path.name),
        "timing_infer_by_category_json": str(by_cat_path.name),
    }
    with open(out_dir / "timing_infer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
