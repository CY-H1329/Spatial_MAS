#!/usr/bin/env python3
"""
LoRA / adaptateurs Sa2VA (ByteDance) sur 3DSRBench.

Sa2VA expose surtout `predict_forward` (inférence). Un entraînement supervisé complet
relève du repo ByteDance. Ce script :
  - charge le modèle comme `Sa2VARunner` (src2) ;
  - tente d'attacher LoRA (PEFT) ;
  - si une `loss` tensorielle est retournée par le forward interne, fait quelques steps ;
  - sinon enregistre `timing_train.json` avec `status: needs_official_repo` et quitte avec code 2.

Code 0 = entraînement effectif ; code 2 = pas de loss utilisable (comportement attendu sur beaucoup d'installations).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dsrbench_io import build_user_text, load_3dsrbench_rows
from mc_common import collate_list_of_dicts
from spatial_mas_src2 import insert_src2


class BenchDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.rows[i]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="ByteDance/Sa2VA-4B")
    p.add_argument("--output_dir", type=str, default="./outputs/lora_3dsr_sa2va")
    p.add_argument("--max_train_samples", type=int, default=32)
    p.add_argument("--full_dataset", action="store_true", help="Tout le split test 3DSRBench.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_cache", type=str, default="./data/3dsr_image_cache")
    return p.parse_args()


def _maybe_loss_from_predict(model: Any, image, tokenizer, user_text: str, letter: str) -> Any:
    """Tente une passe différentiable ; retourne un scalaire ou None."""
    text = f"<image>{user_text}\nRéponse attendue (une lettre) : {letter}"
    input_dict = {
        "image": image.convert("RGB"),
        "text": text,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }
    rd = model.predict_forward(**input_dict)
    if rd is None:
        return None
    if isinstance(rd, dict):
        for k in ("loss", "ce_loss", "lm_loss"):
            v = rd.get(k)
            if isinstance(v, torch.Tensor) and v.requires_grad:
                return v.mean()
    return None


def main() -> None:
    args = parse_args()
    insert_src2()
    from models.sa2va import Sa2VARunner

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing: Dict[str, Any] = {"backend": "sa2va", "steps": [], "status": "unknown"}

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis.")

    t_all = time.perf_counter()
    cap_train = None if args.full_dataset else args.max_train_samples
    rows, load_timing = load_3dsrbench_rows(max_samples=cap_train, seed=args.seed, cache_dir=Path(args.image_cache))
    timing["data_3dsrbench"] = load_timing
    timing["data_3dsrbench"]["rows_used"] = len(rows)
    timing["full_dataset_3dsrbench"] = bool(args.full_dataset)

    t0 = time.perf_counter()
    runner = Sa2VARunner(model_id=args.model_id, device="cuda")
    model = runner.model
    tokenizer = runner.tokenizer
    model.train()
    timing["steps"].append({"name": "load_sa2va", "s": time.perf_counter() - t0})

    t0 = time.perf_counter()
    lora_cfgs = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ["to_q", "to_k", "to_v", "to_out.0"],
    ]
    last_err: Optional[Exception] = None
    for tm in lora_cfgs:
        try:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=tm,
                ),
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
    if last_err is not None:
        timing["status"] = "peft_attach_failed"
        timing["error"] = repr(last_err)
        with open(out_dir / "timing_train.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        print(json.dumps(timing, indent=2))
        raise SystemExit(2)

    model.print_trainable_parameters()
    timing["steps"].append({"name": "inject_lora", "s": time.perf_counter() - t0})

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    dl = DataLoader(BenchDataset(rows), batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_list_of_dicts)

    trained = 0
    for ep in range(args.epochs):
        for batch in tqdm(dl, desc=f"sa2va epoch {ep+1}"):
            ex = batch[0]
            image = ex["image"]
            ut = build_user_text(ex["question"], ex["options_block"])
            letter = ex["answer"]
            loss = _maybe_loss_from_predict(model, image, tokenizer, ut, letter)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            trained += 1

    timing["train_total_s"] = time.perf_counter() - t_all
    timing["optimizer_steps_with_loss"] = trained

    if trained == 0:
        timing["status"] = "needs_official_repo"
        timing["hint"] = (
            "Sa2VA n'a pas exposé de loss tensorielle via predict_forward sur cet environnement. "
            "Utilisez le repo / la recette d'entraînement ByteDance pour un fine-tuning complet."
        )
        with open(out_dir / "timing_train.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        print(json.dumps(timing, indent=2))
        sys.exit(2)

    timing["status"] = "ok"
    model.save_pretrained(out_dir)
    with open(out_dir / "timing_train.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
