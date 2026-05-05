#!/usr/bin/env python3
"""
LoRA sur 3DSRBench — SpatialRGPT (VILA / repo SpatialRGPT via src2).

Requiert : clone https://github.com/AnjieCheng/SpatialRGPT et `export SPATIALRGPT_PATH=...`
(cf. src2/models/spatial_rgpt.py).

Tente `model.forward(..., labels=...)` comme Llava/VILA. Si non supporté sur ta version,
`timing_train.json` contient `status: forward_failed` et le script sort avec code 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dsrbench_io import build_user_text, load_3dsrbench_rows
from mc_common import collate_list_of_dicts, log_train
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
    p.add_argument("--model_id", type=str, default="a8cheng/SpatialRGPT-VILA1.5-8B")
    p.add_argument("--output_dir", type=str, default="./outputs/lora_3dsr_spatial_rgpt")
    p.add_argument("--max_train_samples", type=int, default=32)
    p.add_argument("--full_dataset", action="store_true", help="Tout le split test 3DSRBench.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_cache", type=str, default="./data/3dsr_image_cache")
    p.add_argument("--no_train_step_log", action="store_true", help="Sans timing_train_steps.jsonl.")
    return p.parse_args()


def build_batch_spatial_rgpt(
    runner: Any, image, user_text: str, letter: str, device: torch.device
) -> Optional[Dict[str, Any]]:
    """Construit input_ids + images + labels (masque prompt). Retourne None si échec."""
    try:
        query = runner._DEFAULT_IMAGE_TOKEN + "\n" + user_text + "\nRéponds par une seule lettre : A, B, C ou D."
        conv = runner._conv_templates[runner.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], letter)
        full_prompt = conv.get_prompt()

        image_rgb = image.convert("RGB")
        images_tensor = runner._process_images(
            [image_rgb], runner.image_processor, runner.model.config
        ).to(device, dtype=torch.float16)

        input_ids = runner._tokenizer_image_token(
            full_prompt, runner.tokenizer, runner._IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        # Masquer le prompt : tout sauf les ~4 derniers tokens (réponse courte)
        labels = input_ids.clone()
        if labels.shape[1] > 8:
            labels[:, :-4] = -100
        else:
            labels[:, :-1] = -100

        depths_tensor = images_tensor
        if getattr(runner.model.config, "enable_depth", False):
            depth_img = _make_placeholder_depth(image_rgb)
            depths_tensor = runner._process_images(
                [depth_img], runner.image_processor, runner.model.config
            ).to(device, dtype=torch.float16)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": [images_tensor],
            "depths": [depths_tensor],
            "masks": [None],
        }
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    log_train("train_spatial_rgpt", "Démarrage entraînement (3DSRBench).")
    insert_src2()
    from models.spatial_rgpt import SpatialRGPTRunner, _make_placeholder_depth

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing: Dict[str, Any] = {"backend": "spatial_rgpt", "steps": []}

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis.")

    t_all = time.perf_counter()
    cap_train = None if args.full_dataset else args.max_train_samples
    rows, load_timing = load_3dsrbench_rows(max_samples=cap_train, seed=args.seed, cache_dir=Path(args.image_cache))
    timing["data_3dsrbench"] = load_timing
    timing["data_3dsrbench"]["rows_used"] = len(rows)
    timing["full_dataset_3dsrbench"] = bool(args.full_dataset)
    log_train("train_spatial_rgpt", f"Exemples 3DSRBench : {len(rows)}")

    t0 = time.perf_counter()
    runner = SpatialRGPTRunner(model_id=args.model_id, device="cuda")
    model = runner.model
    model.train()
    timing["steps"].append({"name": "load_spatial_rgpt", "s": time.perf_counter() - t0})

    t0 = time.perf_counter()
    try:
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
    except Exception as e:
        timing["status"] = "peft_attach_failed"
        timing["error"] = repr(e)
        with open(out_dir / "timing_train.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        print(json.dumps(timing, indent=2))
        raise SystemExit(2)
    timing["steps"].append({"name": "inject_lora", "s": time.perf_counter() - t0})
    model.print_trainable_parameters()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    device = torch.device("cuda")
    steps = 0
    forward_ok = False

    steps_jsonl = out_dir / "timing_train_steps.jsonl"
    step_fp = None
    if not args.no_train_step_log:
        step_fp = open(steps_jsonl, "w", encoding="utf-8")
        step_fp.write(json.dumps({"event": "header", "backend": "spatial_rgpt"}, ensure_ascii=False) + "\n")
        log_train("train_spatial_rgpt", f"Journal par step : {steps_jsonl}")
    timing["timing_train_steps_jsonl"] = None if args.no_train_step_log else str(steps_jsonl.name)

    global_step = 0
    try:
        for ep in range(args.epochs):
            for bi, batch in enumerate(
                tqdm(
                    DataLoader(BenchDataset(rows), batch_size=1, shuffle=True, collate_fn=collate_list_of_dicts),
                    desc=f"srgpt ep{ep+1}",
                )
            ):
                t_step = time.perf_counter()
                ex = batch[0]
                ut = build_user_text(ex["question"], ex["options_block"])
                b = build_batch_spatial_rgpt(runner, ex["image"], ut, ex["answer"], device)
                if b is None:
                    if step_fp:
                        step_fp.write(
                            json.dumps(
                                {
                                    "event": "train_step",
                                    "backend": "spatial_rgpt",
                                    "step": global_step,
                                    "epoch": ep,
                                    "index_in_epoch": bi,
                                    "skipped": True,
                                    "reason": "batch_build_failed",
                                    "step_s": time.perf_counter() - t_step,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    global_step += 1
                    continue
                opt.zero_grad(set_to_none=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out = model(
                            input_ids=b["input_ids"],
                            images=b["images"],
                            depths=b["depths"],
                            masks=b["masks"],
                            labels=b["labels"],
                        )
                    loss = getattr(out, "loss", None)
                    if loss is None:
                        if step_fp:
                            step_fp.write(
                                json.dumps(
                                    {
                                        "event": "train_step",
                                        "backend": "spatial_rgpt",
                                        "step": global_step,
                                        "epoch": ep,
                                        "index_in_epoch": bi,
                                        "skipped": True,
                                        "reason": "no_loss_in_output",
                                        "step_s": time.perf_counter() - t_step,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        global_step += 1
                        continue
                    loss.backward()
                    opt.step()
                    dt = time.perf_counter() - t_step
                    steps += 1
                    forward_ok = True
                    if step_fp:
                        step_fp.write(
                            json.dumps(
                                {
                                    "event": "train_step",
                                    "backend": "spatial_rgpt",
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
                except Exception as e:
                    timing.setdefault("forward_errors", []).append(repr(e))
                    if step_fp:
                        step_fp.write(
                            json.dumps(
                                {
                                    "event": "train_step",
                                    "backend": "spatial_rgpt",
                                    "step": global_step,
                                    "epoch": ep,
                                    "error": repr(e),
                                    "step_s": time.perf_counter() - t_step,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    break
            if timing.get("forward_errors"):
                break
    finally:
        if step_fp:
            step_fp.close()

    timing["train_total_s"] = time.perf_counter() - t_all
    timing["optimizer_steps"] = steps

    if not forward_ok or steps == 0:
        timing["status"] = "forward_failed_or_no_steps"
        timing["hint"] = (
            "Le forward SpatialRGPT avec labels n'a pas fonctionné sur cet environnement. "
            "Voir le repo SpatialRGPT pour un fine-tuning VILA complet."
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
