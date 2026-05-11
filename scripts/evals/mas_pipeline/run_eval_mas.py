#!/usr/bin/env python3
"""
Spatial MAS Pipeline Evaluation — Head → 3 Specialists → Reasoning → Score Update.

Runs the full pipeline on CV-Bench (or 3DSRBench):
1. Head-Agent (GPT-5.2) infers category, selects 3 agents, creates coordination policy
2. 3 Specialist agents solve the task (CoT + answer)
3. Reasoning Agent (DeepSeek-VL) synthesizes final answer
4. Per-agent scores updated based on correctness

Usage:
  python scripts/evals/mas_pipeline/run_eval_mas.py --test
  python scripts/evals/mas_pipeline/run_eval_mas.py --max_samples 50
  python scripts/evals/mas_pipeline/run_eval_mas.py --full_dataset
  # Head = LLaVA-NeXT-7B (GPU), sans clé OpenAI pour le Head :
  python scripts/evals/mas_pipeline/run_eval_mas.py --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml --test
  # CV-Bench split HF complet (~2638) :
  python scripts/evals/mas_pipeline/run_eval_mas.py --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml --benchmark cvbench --full_dataset

Env (API head): OPENAI_API_KEY. Head GPU: pas de clé. Reasoning GPU: CUDA. API specialists: clés habituelles.

Ablation SpatiO (remplacer spatial_reasoner / spatial_rgpt par InternVL2 + Qwen2-VL) :
  export MAS_CANDIDATE_AGENTS="qwen3_4b,sa2va,llava4d,internvl2,qwen2_vl"
  Voir config_mas_ablation_internvl2_qwen2vl_cvbench500_h100.yaml et run_h100_ablation_internvl2_qwen2vl_cvbench500.sh
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm
from PIL import Image

from src.benchmarks import (
    load_benchmark,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_images,
    get_benchmark_category,
)
from src.agents.mas import run_spatial_mas_pipeline, ScoreManager
from src.agents.mas.config import CVBENCH_TO_UNIFIED, TASK_CATEGORIES

# API Runners
_runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
_runners_spec = __import__("importlib").util.spec_from_file_location("runners", _runners_path)
_runners = __import__("importlib").util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
GPT4oRunner = _runners.GPT4oRunner
ClaudeRunner = _runners.ClaudeRunner
GeminiRunner = _runners.GeminiRunner
DeepSeekVLRunner = _runners.DeepSeekVLRunner
OpenRouterRunner = _runners.OpenRouterRunner

# GPU Runners (H100)
try:
    from src.models.qwen3 import Qwen3Runner
    from src.models.sa2va import Sa2VARunner
    from src.models.llava import LLaVARunner
    from src.models.spatial_reasoner import SpatialReasonerRunner
    from src.models.qwen import QwenRunner
    from src.models.deepseek_vl import DeepSeekVLRunner as DeepSeekVLGPURunner
    GPU_AVAILABLE = True
except ImportError:
    Qwen3Runner = Sa2VARunner = LLaVARunner = SpatialReasonerRunner = QwenRunner = DeepSeekVLGPURunner = None
    GPU_AVAILABLE = False

InternVL2Runner = None
try:
    from src.models.internvl2 import InternVL2Runner
except ImportError:
    pass
Qwen2VLRunner = None
try:
    from src.models.qwen2_vl import Qwen2VLRunner
except ImportError:
    pass


def _norm_answer(s: str) -> str:
    """Normalize answer to (A)/(B)/(C)/(D)."""
    s = (s or "").strip().upper()
    for c in "ABCD":
        if c in s or f"({c})" in s:
            return f"({c})"
    return s


def _tile_views(views: list[Image.Image]) -> Image.Image:
    """
    MindCube (and some multi-view tasks) provide 2–4 images. Most GPU runners here are single-image.
    We tile them into a 2x2 grid (pad with black if needed) so we can run the MAS pipeline unchanged.
    """
    if not views:
        raise ValueError("No views to tile")
    ims = [v.convert("RGB") for v in views if v is not None]
    if len(ims) == 1:
        return ims[0]
    # make a 2x2 grid
    while len(ims) < 4:
        ims.append(Image.new("RGB", ims[0].size, (0, 0, 0)))
    w = max(im.width for im in ims)
    h = max(im.height for im in ims)
    ims = [im.resize((w, h), Image.Resampling.LANCZOS) if im.size != (w, h) else im for im in ims[:4]]
    grid = Image.new("RGB", (2 * w, 2 * h), (0, 0, 0))
    grid.paste(ims[0], (0, 0))
    grid.paste(ims[1], (w, 0))
    grid.paste(ims[2], (0, h))
    grid.paste(ims[3], (w, h))
    return grid


class _LazySpecialistRunner:
    """Load a heavy GPU specialist only on first use (reduces peak VRAM vs loading all at once)."""

    __slots__ = ("_builder", "_inst")

    def __init__(self, builder):
        self._builder = builder
        self._inst = None

    def generate(self, image, prompt, **kwargs):
        if self._inst is None:
            self._inst = self._builder()
        return self._inst.generate(image, prompt, **kwargs)


def build_runners(config: dict):
    """Build Head, Specialist, and Reasoning runners from config.
    Head: API (OpenAI) or GPU (LLaVA-NeXT / Qwen3-VL via backend).
    Specialists: GPU (qwen3_4b, sa2va, llava4d, internvl2, qwen2_vl, …) on H100, API for claude/gpt4o/gemini.
    GPU specialists are wrapped lazy-loaded to avoid loading every checkpoint at startup.
    """
    head_cfg = config.get("head_agent", {})
    head_runner = None
    head_runner_type = head_cfg.get("runner", "api")

    if head_runner_type == "gpu" and GPU_AVAILABLE:
        device = head_cfg.get("device", "cuda")
        model_id = head_cfg.get("model_id", "llava-hf/llava-v1.6-mistral-7b-hf")
        backend = head_cfg.get("backend", "llava").lower()
        try:
            if backend == "llava" and LLaVARunner:
                head_runner = LLaVARunner(model_id=model_id, device=device)
            elif backend == "qwen3" and Qwen3Runner:
                head_runner = Qwen3Runner(model_id=model_id, device=device)
            else:
                print(f"[skip] Head GPU: unknown backend {backend!r} (use llava or qwen3)")
        except Exception as e:
            print(f"[skip] Head GPU ({backend}): {e}")
    else:
        head_key = os.environ.get(head_cfg.get("api_key_env", ""), "").strip()
        if head_key:
            head_runner = GPT4oRunner(
                model_id=head_cfg.get("model_id", "gpt-4o"),
                api_key=head_key,
            )

    reason_cfg = config.get("reasoning_agent", {})
    reason_runner = None
    if reason_cfg.get("runner") == "gpu" and GPU_AVAILABLE and DeepSeekVLGPURunner:
        try:
            reason_runner = DeepSeekVLGPURunner(
                model_id=reason_cfg.get("model_id", "deepseek-community/deepseek-vl-7b-chat"),
                device=reason_cfg.get("device", "cuda"),
            )
        except Exception as e:
            print(f"[skip] Reasoning GPU (DeepSeek-VL): {e}")
    else:
        reason_key = os.environ.get(reason_cfg.get("api_key_env", ""), "").strip()
        if reason_key:
            reason_runner = DeepSeekVLRunner(
                model_id=reason_cfg.get("model_id", "deepseek-vl"),
                api_key=reason_key,
                base_url=reason_cfg.get("base_url", "https://api.deepseek.com"),
            )

    specialists_cfg = config.get("specialists", {})
    specialist_runners = {}
    for name, cfg in specialists_cfg.items():
        runner_type = cfg.get("runner", "api")
        model_id = cfg.get("model_id", "")

        if runner_type == "gpu" and GPU_AVAILABLE:
            device = cfg.get("device", "cuda")
            backend = (cfg.get("backend") or name).lower()
            try:
                if backend in ("qwen3", "qwen3_4b") and Qwen3Runner:
                    specialist_runners[name] = _LazySpecialistRunner(
                        lambda mid=model_id, dev=device: Qwen3Runner(model_id=mid, device=dev)
                    )
                elif backend == "sa2va" and Sa2VARunner:
                    specialist_runners[name] = _LazySpecialistRunner(
                        lambda mid=model_id, dev=device: Sa2VARunner(model_id=mid, device=dev)
                    )
                elif backend in ("llava", "llava4d") and LLaVARunner:
                    specialist_runners[name] = _LazySpecialistRunner(
                        lambda mid=model_id, dev=device: LLaVARunner(model_id=mid, device=dev)
                    )
                elif backend in ("spatial_reasoner", "spatialreasoner") and SpatialReasonerRunner:
                    processor_id = cfg.get("processor_id", "Qwen/Qwen2.5-VL-7B-Instruct")
                    bf16 = bool(cfg.get("bf16", True))
                    mid = model_id or "ccvl/SpatialReasoner"
                    specialist_runners[name] = _LazySpecialistRunner(
                        lambda mid=mid, dev=device, pid=processor_id, b=bf16: SpatialReasonerRunner(
                            model_id=mid,
                            processor_id=pid,
                            device=dev,
                            bf16=b,
                        )
                    )
                elif backend in ("internvl2", "intern_vl2") and InternVL2Runner:
                    mid = model_id or "OpenGVLab/InternVL2-8B"
                    inp = int(cfg.get("input_size", 448))
                    mx = int(cfg.get("max_num_tiles", 12))
                    ufa = bool(cfg.get("use_flash_attn", False))
                    specialist_runners[name] = _LazySpecialistRunner(
                        lambda mid=mid, dev=device, inp=inp, mx=mx, ufa=ufa: InternVL2Runner(
                            model_id=mid,
                            device=dev,
                            input_size=inp,
                            max_num_tiles=mx,
                            use_flash_attn=ufa,
                        )
                    )
                elif backend in ("qwen2_vl", "qwen2vl") and Qwen2VLRunner:
                    mid = model_id or "Qwen/Qwen2-VL-7B-Instruct"
                    specialist_runners[name] = _LazySpecialistRunner(
                        lambda mid=mid, dev=device: Qwen2VLRunner(model_id=mid, device=dev)
                    )
                elif backend in ("qwen25_vl", "qwen2_5_vl", "qwen2.5_vl") and QwenRunner:
                    # Stable HF Qwen2.5-VL runner (no InternLM remote-code)
                    mid = model_id or "Qwen/Qwen2.5-VL-7B-Instruct"
                    specialist_runners[name] = _LazySpecialistRunner(
                        lambda mid=mid, dev=device: QwenRunner(model_id=mid, device=dev)
                    )
                else:
                    specialist_runners[name] = None
            except Exception as e:
                print(f"[skip] {name} GPU: {e}")
                specialist_runners[name] = None
        elif runner_type == "api":
            api_runner = cfg.get("api_runner", "")
            key = os.environ.get(cfg.get("api_key_env", ""), "").strip()
            if not key:
                specialist_runners[name] = None
                continue
            if api_runner == "claude":
                specialist_runners[name] = ClaudeRunner(model_id=model_id, api_key=key)
            elif api_runner == "openai":
                specialist_runners[name] = GPT4oRunner(model_id=model_id, api_key=key)
            elif api_runner == "gemini":
                specialist_runners[name] = GeminiRunner(model_id=model_id, api_key=key)
            else:
                specialist_runners[name] = None
        else:
            specialist_runners[name] = None

    return head_runner, specialist_runners, reason_runner


def main():
    parser = argparse.ArgumentParser(description="Spatial MAS Pipeline Eval")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_mas.yaml"))
    parser.add_argument("--test", action="store_true", help="5 samples only")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--full_dataset", action="store_true")
    parser.add_argument("--benchmark", choices=["cvbench", "3dsrbench", "mindcube"], default="cvbench")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    head_runner, specialist_runners, reason_runner = build_runners(config)
    if not head_runner:
        print(
            "ERROR: Head runner missing. Either set head_agent.runner: gpu + CUDA + LLaVA/Qwen3, "
            "or set OPENAI_API_KEY for API head (see config_mas.yaml / config_mas_head_llava_next.yaml)."
        )
        sys.exit(1)
    if not reason_runner:
        print("ERROR: Reasoning runner required. Set runner: gpu in config or DEEPSEEK_API_KEY for API.")
        sys.exit(1)

    ds_cfg = config.get("dataset", {})
    max_samples = ds_cfg.get("test_samples", 5) if args.test else (
        None if args.full_dataset else (args.max_samples or ds_cfg.get("max_samples", 50))
    )
    benchmark = args.benchmark or ds_cfg.get("benchmark", "cvbench")
    seed = args.seed or ds_cfg.get("seed", 42)
    # Full benchmark split from HuggingFace (e.g. CV-Bench ~2638), not frozen local snapshot
    use_frozen = not args.full_dataset

    out_dir = Path(config.get("output", {}).get("dir", "results/runs/mas_pipeline"))
    subdir = "test" if args.test else (
        f"full_{benchmark}_hf" if args.full_dataset else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = Path(out_dir) / subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = load_benchmark(benchmark, max_samples=max_samples, seed=seed, use_frozen=use_frozen)
    print(
        f"[dataset] benchmark={benchmark}  n={len(ds)}  use_frozen={use_frozen}  "
        f"max_samples={max_samples}  out={run_dir}"
    )

    timing_path = run_dir / "timing.jsonl"
    timing_csv_path = run_dir / "timing.csv"
    with open(timing_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "idx,head_sec,specialist1_name,specialist1_sec,specialist2_name,specialist2_sec,"
            "specialist3_name,specialist3_sec,reasoning_sec,total_sec\n"
        )

    # Per-sample timing context populated by generator wrappers.
    _timing_ctx = {"head_sec": 0.0, "reasoning_sec": 0.0, "specialist_sec": {}}

    def _reset_timing_ctx():
        _timing_ctx["head_sec"] = 0.0
        _timing_ctx["reasoning_sec"] = 0.0
        _timing_ctx["specialist_sec"] = {}

    def head_gen(img: Image.Image, prompt: str) -> str:
        t = time.perf_counter()
        mod = type(head_runner).__module__ or ""
        if "src.models" in mod:
            out = head_runner.generate(img, prompt, max_new_tokens=2048)
        else:
            out = head_runner.generate(img, prompt, max_tokens=2048)
        _timing_ctx["head_sec"] += time.perf_counter() - t
        return out

    def spec_gen(agent_name: str, img: Image.Image, prompt: str) -> str:
        r = specialist_runners.get(agent_name)
        if not r:
            return ""
        t = time.perf_counter()
        # GPU runners use max_new_tokens, API use max_tokens
        mod = type(r).__module__ or ""
        if isinstance(r, _LazySpecialistRunner) or "src.models" in mod:
            out = r.generate(img, prompt, max_new_tokens=2048)
        else:
            out = r.generate(img, prompt, max_tokens=2048)
        _timing_ctx["specialist_sec"][agent_name] = _timing_ctx["specialist_sec"].get(agent_name, 0.0) + (
            time.perf_counter() - t
        )
        return out

    def reason_gen(img: Image.Image, prompt: str) -> str:
        t = time.perf_counter()
        mod = type(reason_runner).__module__ or ""
        if "src.models" in mod:
            out = reason_runner.generate(img, prompt, max_new_tokens=1024)
        else:
            out = reason_runner.generate(img, prompt, max_tokens=1024)
        _timing_ctx["reasoning_sec"] += time.perf_counter() - t
        return out

    score_manager = ScoreManager()
    category_seen = {c: False for c in TASK_CATEGORIES}
    score_history = [score_manager.to_dict()]

    results = []
    correct = 0
    total = 0

    for i in tqdm(range(len(ds)), desc="MAS Pipeline"):
        ex = ds[i]
        # Single-image benchmarks use get_benchmark_image; multi-view (MindCube) tiles views into one image.
        img = get_benchmark_image(ex, benchmark)
        if img is None:
            views = get_benchmark_images(ex, benchmark)
            if not views:
                continue
            img = _tile_views(views)
        query = get_benchmark_prompt(ex, benchmark)
        gt = get_benchmark_answer(ex, benchmark)
        gt_norm = _norm_answer(gt)

        _reset_timing_ctx()
        t0 = time.perf_counter()
        out = run_spatial_mas_pipeline(
            image=img,
            query=query,
            gt_answer=gt,
            head_generate=head_gen,
            specialist_generate=spec_gen,
            reasoning_generate=reason_gen,
            score_manager=score_manager,
            category_seen=category_seen,
        )
        t_total = time.perf_counter() - t0
        head_sec = float(_timing_ctx["head_sec"] or 0.0)
        reasoning_sec = float(_timing_ctx["reasoning_sec"] or 0.0)

        # If pipeline failed, still log total time.
        if "error" in out:
            results.append({"idx": i, "error": out["error"], "gt": gt})
            with open(timing_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "idx": i,
                            "error": out["error"],
                            "head_sec": head_sec,
                            "reasoning_sec": reasoning_sec,
                            "specialist_sec": _timing_ctx["specialist_sec"],
                            "total_sec": t_total,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue

        if "error" in out:
            results.append({"idx": i, "error": out["error"], "gt": gt})
            continue

        pred = out.get("final_answer", "")
        pred_norm = _norm_answer(pred)
        is_correct = pred_norm == gt_norm
        if is_correct:
            correct += 1
        total += 1

        cat = out.get("predicted_category", "")
        category_seen[cat] = True

        agent_results = [
            {k: v for k, v in r.items() if k != "raw"}
            for r in out.get("agent_results", [])
        ]
        score_history.append(score_manager.to_dict())
        results.append({
            "idx": i,
            "predicted_category": cat,
            "selected_agents": out.get("selected_agents", []),
            "final_answer": pred,
            "gt": gt,
            "correct": is_correct,
            "agent_results": agent_results,
            "reasoning_justification": out.get("reasoning_justification", ""),
            "score_table_after_turn": score_manager.to_dict(),
        })

        # Log per-role timings (Head + each selected specialist + Reasoning + Total)
        sel = out.get("selected_agents", []) or []
        s1 = sel[0] if len(sel) > 0 else ""
        s2 = sel[1] if len(sel) > 1 else ""
        s3 = sel[2] if len(sel) > 2 else ""
        s1_sec = float(_timing_ctx["specialist_sec"].get(s1, 0.0)) if s1 else 0.0
        s2_sec = float(_timing_ctx["specialist_sec"].get(s2, 0.0)) if s2 else 0.0
        s3_sec = float(_timing_ctx["specialist_sec"].get(s3, 0.0)) if s3 else 0.0
        timing_row = {
            "idx": i,
            "head_sec": head_sec,
            "specialist_sec": {s1: s1_sec, s2: s2_sec, s3: s3_sec},
            "reasoning_sec": reasoning_sec,
            "total_sec": t_total,
        }
        with open(timing_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(timing_row, ensure_ascii=False) + "\n")
        with open(timing_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{i},{head_sec:.4f},{s1},{s1_sec:.4f},{s2},{s2_sec:.4f},{s3},{s3_sec:.4f},{reasoning_sec:.4f},{t_total:.4f}\n"
            )

        if (i + 1) % 10 == 0:
            with open(run_dir / "progress.json", "w") as f:
                json.dump({
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total else 0,
                    "score_table": score_manager.to_dict(),
                }, f, indent=2)

    acc = correct / total if total else 0
    print(f"\nAccuracy: {correct}/{total} = {acc:.2%}")

    summary = {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "benchmark": benchmark,
        "max_samples": max_samples,
        "score_table": score_manager.to_dict(),
        "score_history": score_history,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
