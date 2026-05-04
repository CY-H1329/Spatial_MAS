#!/usr/bin/env python3
"""LoRA sur 3DSRBench — LLaVA-NeXT (HF). Écrit timing_train.json dans --output_dir."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dsrbench_io import build_user_text, load_3dsrbench_rows
from mc_common import collate_list_of_dicts


def _import_llava():
    try:
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
    except ImportError as e:
        raise SystemExit(f"LLaVA-NeXT requiert transformers récent. Détail: {e}") from e
    return AutoProcessor, LlavaNextForConditionalGeneration


class BenchDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.rows[i]


def collate_llava_next(
    batch: List[Dict[str, Any]],
    processor: Any,
    device: torch.device,
) -> Dict[str, Any]:
    ex = batch[0]
    image = ex["image"]
    user_text = build_user_text(ex["question"], ex["options_block"])
    letter = ex["answer"]
    messages_user = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    messages_full = messages_user + [
        {"role": "assistant", "content": [{"type": "text", "text": letter}]},
    ]
    def _tok(msgs, add_gen: bool):
        kw = dict(
            tokenize=True,
            add_generation_prompt=add_gen,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            return processor.apply_chat_template(msgs, images=[image], **kw)
        except TypeError:
            return processor.apply_chat_template(msgs, **kw)

    tok_p = _tok(messages_user, True)
    tok_f = _tok(messages_full, False)
    if "pixel_values" not in tok_f and "pixel_values" not in tok_p:
        pv = processor.image_processor(image, return_tensors="pt")["pixel_values"]
        tok_f["pixel_values"] = pv.to(device)
        tok_p["pixel_values"] = pv.to(device)
    prompt_len = int(tok_p["input_ids"].shape[1])
    input_ids = tok_f["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    out: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
    for k, v in tok_f.items():
        if k == "input_ids":
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif k != "labels":
            out[k] = v
    allowed = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_sizes",
        "labels",
    }
    return {k: v for k, v in out.items() if k in allowed}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--output_dir", type=str, default="./outputs/lora_3dsr_llava")
    p.add_argument("--max_train_samples", type=int, default=150)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_cache", type=str, default="./data/3dsr_image_cache")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    AutoProcessor, LlavaNext = _import_llava()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache)
    timing: Dict[str, Any] = {"steps": [], "backend": "llava_next"}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    rows, load_timing = load_3dsrbench_rows(max_samples=args.max_train_samples, seed=args.seed, cache_dir=cache_dir)
    timing["data_3dsrbench"] = load_timing
    timing["data_3dsrbench"]["rows_used"] = len(rows)
    timing["steps"].append({"name": "load_3dsrbench_rows", "s": time.perf_counter() - t0})

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = LlavaNext.from_pretrained(args.model_id, dtype=dtype, trust_remote_code=True)
    model = model.to(device)
    timing["steps"].append({"name": "load_model_processor", "s": time.perf_counter() - t0})

    t0 = time.perf_counter()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    timing["steps"].append({"name": "inject_lora", "s": time.perf_counter() - t0})

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    dl = DataLoader(BenchDataset(rows), batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_list_of_dicts)
    model.train()
    epoch_times: List[float] = []
    for ep in range(args.epochs):
        t_ep = time.perf_counter()
        for batch in tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}"):
            opt.zero_grad(set_to_none=True)
            inputs = collate_llava_next(batch, processor, device)
            if args.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**inputs)
            else:
                out = model(**inputs)
            out.loss.backward()
            opt.step()
        epoch_times.append(time.perf_counter() - t_ep)

    timing["epochs_s"] = epoch_times
    timing["train_total_s"] = time.perf_counter() - t_all
    timing["per_epoch_mean_s"] = sum(epoch_times) / max(len(epoch_times), 1)

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    with open(out_dir / "timing_train.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
