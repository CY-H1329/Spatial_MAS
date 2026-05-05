#!/usr/bin/env python3
"""
LoRA fine-tuning sur 3DSRBench (ccvl/3DSRBench, subset benchmark, split test)
avec Qwen3-VL (ex. Qwen/Qwen3-VL-4B-Instruct).

Écrit timing_train.json dans --output_dir.
"""
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
from mc_common import collate_list_of_dicts, log_train


def _import_qwen():
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise SystemExit(
            "Import Qwen3VL impossible. Installez une version récente de transformers "
            "(souvent : pip install git+https://github.com/huggingface/transformers).\n"
            f"Détail: {e}"
        ) from e
    return AutoProcessor, Qwen3VLForConditionalGeneration


class BenchDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.rows[i]


_FORWARD_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "video_grid_thw",
    "pixel_values_videos",
    "second_per_grid_ts",
    "image_sizes",
    "labels",
}


def collate_qwen3(
    batch: List[Dict[str, Any]],
    processor: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Un exemple à la fois (évite le padding multi-image complexe)."""
    ex = batch[0]
    image = ex["image"]
    user_text = build_user_text(ex["question"], ex["options_block"])
    letter = ex["answer"]
    messages_user = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    messages_full = messages_user + [
        {"role": "assistant", "content": [{"type": "text", "text": letter}]},
    ]
    tok_p = processor.apply_chat_template(
        messages_user,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    tok_f = processor.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = int(tok_p["input_ids"].shape[1])
    dev = device
    input_ids = tok_f["input_ids"].to(dev)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    out: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
    for k, v in tok_f.items():
        if k == "input_ids":
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.to(dev)
        else:
            out[k] = v
    return {k: v for k, v in out.items() if k in _FORWARD_KEYS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--output_dir", type=str, default="./outputs/lora_3dsr_qwen3vl")
    p.add_argument("--max_train_samples", type=int, default=150)
    p.add_argument(
        "--full_dataset",
        action="store_true",
        help="Utiliser tout le split test 3DSRBench (ignore --max_train_samples).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_cache", type=str, default="./data/3dsr_image_cache")
    p.add_argument("--bf16", action="store_true", help="Utiliser bf16 (recommandé sur H100)")
    p.add_argument(
        "--no_train_step_log",
        action="store_true",
        help="Ne pas écrire timing_train_steps.jsonl (un JSON par step optimiseur).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_train("train_qwen3vl", "Démarrage entraînement LoRA (3DSRBench).")
    AutoProcessor, Qwen3VLForConditionalGeneration = _import_qwen()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache)

    timing: Dict[str, Any] = {"steps": []}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    cap_train = None if args.full_dataset else args.max_train_samples
    rows, load_timing = load_3dsrbench_rows(max_samples=cap_train, seed=args.seed, cache_dir=cache_dir)
    timing["data_3dsrbench"] = load_timing
    timing["data_3dsrbench"]["rows_used"] = len(rows)
    timing["full_dataset_3dsrbench"] = bool(args.full_dataset)
    timing["steps"].append({"name": "load_3dsrbench_rows", "s": time.perf_counter() - t0})
    log_train("train_qwen3vl", f"Exemples 3DSRBench chargés : {len(rows)}")

    t0 = time.perf_counter()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis pour cet entraînement (GPU).")
    device = torch.device("cuda")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    timing["steps"].append({"name": "load_model_processor", "s": time.perf_counter() - t0})

    t0 = time.perf_counter()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    timing["steps"].append({"name": "inject_lora", "s": time.perf_counter() - t0})

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    ds = BenchDataset(rows)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_list_of_dicts)

    steps_jsonl = out_dir / "timing_train_steps.jsonl"
    step_fp = None
    if not args.no_train_step_log:
        step_fp = open(steps_jsonl, "w", encoding="utf-8")
        step_fp.write(
            json.dumps(
                {"event": "header", "backend": "qwen3vl", "note": "one line per optimizer step (batch_size=1)"},
                ensure_ascii=False,
            )
            + "\n"
        )
        log_train("train_qwen3vl", f"Journal par step : {steps_jsonl}")
    timing["timing_train_steps_jsonl"] = None if args.no_train_step_log else str(steps_jsonl.name)

    model.train()
    epoch_times: List[float] = []
    global_step = 0
    try:
        for ep in range(args.epochs):
            t_ep = time.perf_counter()
            pbar = tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}")
            for bi, batch in enumerate(pbar):
                t_step = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                inputs = collate_qwen3(batch, processor, device)
                if args.bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(**inputs)
                else:
                    out = model(**inputs)
                loss = out.loss
                loss.backward()
                opt.step()
                dt = time.perf_counter() - t_step
                pbar.set_postfix(loss=float(loss.item()))
                if step_fp:
                    step_fp.write(
                        json.dumps(
                            {
                                "event": "train_step",
                                "backend": "qwen3vl",
                                "step": global_step,
                                "epoch": ep,
                                "index_in_epoch": bi,
                                "loss": float(loss.item()),
                                "step_s": dt,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                global_step += 1
            epoch_times.append(time.perf_counter() - t_ep)
    finally:
        if step_fp:
            step_fp.close()

    timing["epochs_s"] = epoch_times
    timing["train_total_s"] = time.perf_counter() - t_all
    timing["per_epoch_mean_s"] = sum(epoch_times) / max(len(epoch_times), 1)

    t0 = time.perf_counter()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    timing["steps"].append({"name": "save_adapter", "s": time.perf_counter() - t0})

    with open(out_dir / "timing_train.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
