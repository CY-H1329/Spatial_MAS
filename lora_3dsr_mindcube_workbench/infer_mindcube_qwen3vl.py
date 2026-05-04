#!/usr/bin/env python3
"""Inférence MindCube — Qwen3-VL + adaptateurs LoRA (PEFT)."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm

from mindcube_io import Split, ensure_mindcube_extracted, load_mindcube_images, load_mindcube_rows, mindcube_root


def _import_qwen():
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise SystemExit(f"Import Qwen3VL impossible. Détail: {e}") from e
    return AutoProcessor, Qwen3VLForConditionalGeneration


def parse_letter(text: str) -> Optional[str]:
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m2 = re.search(r"[ABCD]", text.upper())
    return m2.group(0) if m2 else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--mindcube_split", type=str, default="tinybench", choices=["test", "train", "tinybench"])
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./outputs/mindcube_infer_qwen3vl")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    AutoProcessor, Qwen3VLForConditionalGeneration = _import_qwen()
    split: Split = args.mindcube_split  # type: ignore[assignment]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "timing_infer.jsonl"

    root = ensure_mindcube_extracted()
    rows = load_mindcube_rows(split=split, max_samples=args.max_samples, seed=42)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    dev = next(model.parameters()).device
    load_model_s = time.perf_counter() - t0

    correct = 0
    total = 0
    latencies: List[float] = []

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        fj.write(json.dumps({"event": "model_load", "load_model_s": load_model_s, "backend": "qwen3vl"}, ensure_ascii=False) + "\n")
        for i, row in enumerate(tqdm(rows, desc="MindCube Qwen3-VL")):
            t_sample = time.perf_counter()
            gt = str(row.get("gt_answer", "")).strip().upper()[:1]
            question = str(row.get("question", ""))
            rels = row.get("images") or []
            if not isinstance(rels, list):
                rels = [rels]
            t_img0 = time.perf_counter()
            images = load_mindcube_images(mindcube_root(), [str(x) for x in rels])
            load_images_s = time.perf_counter() - t_img0

            user_tail = question + "\n\nRéponds uniquement par une seule lettre : A, B, C ou D."
            content: List[Dict[str, Any]] = []
            for im in images:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": user_tail})
            messages = [{"role": "user", "content": content}]

            t_pre = time.perf_counter()
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(dev)
            preprocess_s = time.perf_counter() - t_pre

            t_gen = time.perf_counter()
            with torch.inference_mode():
                gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            generate_s = time.perf_counter() - t_gen

            trimmed = gen_ids[:, inputs["input_ids"].shape[1] :]
            text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            pred = parse_letter(text) or ""
            ok = pred == gt and gt in ("A", "B", "C", "D")
            if gt in ("A", "B", "C", "D"):
                total += 1
                correct += int(ok)
            total_s = time.perf_counter() - t_sample
            latencies.append(total_s)

            fj.write(
                json.dumps(
                    {
                        "i": i,
                        "load_images_s": load_images_s,
                        "preprocess_s": preprocess_s,
                        "generate_s": generate_s,
                        "total_sample_s": total_s,
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
        "backend": "qwen3vl",
        "mindcube_split": split,
        "max_samples": args.max_samples,
        "load_model_s": load_model_s,
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total_scored": total,
        "mean_total_sample_s": sum(latencies) / max(len(latencies), 1),
        "sum_total_sample_s": sum(latencies),
    }
    with open(out_dir / "timing_infer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
