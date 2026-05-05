#!/usr/bin/env python3
"""Inférence MindCube — SpatialReasoner (Qwen2.5-VL) + LoRA."""
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


def parse_letter(text: str) -> Optional[str]:
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m2 = re.search(r"[ABCD]", text.upper())
    return m2.group(0) if m2 else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", type=str, default="ccvl/SpatialReasoner")
    p.add_argument("--processor_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--mindcube_split", type=str, default="tinybench", choices=["test", "train", "tinybench"])
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--full_dataset", action="store_true", help="Tout le split MindCube.")
    p.add_argument("--output_dir", type=str, default="./outputs/mindcube_infer_spatial_reasoner")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    split: Split = args.mindcube_split  # type: ignore[assignment]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "timing_infer.jsonl"

    ensure_mindcube_extracted()
    cap_rows = None if args.full_dataset else args.max_samples
    rows = load_mindcube_rows(split=split, max_samples=cap_rows, seed=42)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.processor_id, trust_remote_code=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_id, dtype=dtype, trust_remote_code=True
    )
    base = base.to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    dev = next(model.parameters()).device
    load_model_s = time.perf_counter() - t0

    correct = 0
    total = 0
    latencies: List[float] = []

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        fj.write(json.dumps({"event": "model_load", "backend": "spatial_reasoner", "load_model_s": load_model_s}, ensure_ascii=False) + "\n")
        for i, row in enumerate(tqdm(rows, desc="MindCube SpatialReasoner")):
            t0s = time.perf_counter()
            gt = str(row.get("gt_answer", "")).strip().upper()[:1]
            question = str(row.get("question", ""))
            rels = row.get("images") or []
            if not isinstance(rels, list):
                rels = [rels]
            images = load_mindcube_images(mindcube_root(), [str(x) for x in rels])
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
            inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
            inputs.pop("token_type_ids", None)
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
            latencies.append(time.perf_counter() - t0s)
            fj.write(
                json.dumps(
                    {
                        "i": i,
                        "preprocess_s": preprocess_s,
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
        "backend": "spatial_reasoner",
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
