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

from mc_common import log_infer
from mindcube_io import (
    Split,
    aggregate_timing_infer_by_category,
    ensure_mindcube_extracted,
    load_mindcube_images,
    load_mindcube_rows,
    mindcube_row_meta,
    mindcube_root,
)


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
    p.add_argument(
        "--adapter_dir",
        type=str,
        default="",
        help="Dossier PEFT (adapter_config.json). Vide = modèle de base seul.",
    )
    p.add_argument("--mindcube_split", type=str, default="tinybench", choices=["test", "train", "tinybench"])
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--full_dataset", action="store_true", help="Tout le split MindCube.")
    p.add_argument("--output_dir", type=str, default="./outputs/mindcube_infer_spatial_reasoner")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main() -> None:
    TAG = "infer_spatial_reasoner"
    args = parse_args()
    log_infer(TAG, "Démarrage.")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    split: Split = args.mindcube_split  # type: ignore[assignment]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "timing_infer.jsonl"

    log_infer(TAG, "MindCube…")
    ensure_mindcube_extracted()
    cap_rows = None if args.full_dataset else args.max_samples
    rows = load_mindcube_rows(split=split, max_samples=cap_rows, seed=42)
    log_infer(TAG, f"{len(rows)} exemples.")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    t0 = time.perf_counter()
    log_infer(TAG, "Chargement processeur + SpatialReasoner…")
    processor = AutoProcessor.from_pretrained(args.processor_id, trust_remote_code=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_id, dtype=dtype, trust_remote_code=True
    )
    base = base.to("cuda")
    adapter_dir = (args.adapter_dir or "").strip()
    if adapter_dir:
        cfg = Path(adapter_dir).expanduser().resolve() / "adapter_config.json"
        if not cfg.is_file():
            raise SystemExit(
                f"Adaptateur LoRA introuvable : {cfg} absent. "
                "Exécute train_lora_spatial_reasoner.py avec le même --output_dir, "
                "ou passe --adapter_dir \"\" pour l’inférence sans LoRA (base seule)."
            )
        log_infer(TAG, f"Chargement PEFT depuis {adapter_dir}…")
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        log_infer(TAG, "Pas d’adaptateur — inférence sur le modèle de base seul.")
        model = base
    model.eval()
    dev = next(model.parameters()).device
    load_model_s = time.perf_counter() - t0
    log_infer(TAG, f"Prêt en {load_model_s:.1f}s — tqdm.")

    correct = 0
    total = 0
    latencies: List[float] = []
    step_records: List[Dict[str, Any]] = []

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        fj.write(json.dumps({"event": "model_load", "backend": "spatial_reasoner", "load_model_s": load_model_s}, ensure_ascii=False) + "\n")
        for i, row in enumerate(tqdm(rows, desc="MindCube SpatialReasoner")):
            t0s = time.perf_counter()
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
            rec: Dict[str, Any] = {
                "event": "infer_step",
                "backend": "spatial_reasoner",
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
        "backend": "spatial_reasoner",
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
