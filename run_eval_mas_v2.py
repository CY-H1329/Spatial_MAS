#!/usr/bin/env python3
"""
MAS v2 Evaluation -- Train / Test split.

Models:
  HEAD            = Qwen3-VL-4B       (VLM, image+text -> category)
  4 SPECIALISTS (default) = Qwen3/Sa2VA/LLaVA4D/SpatialReasoner (SpatialRGPT: --specialist_whitelist …, spatial_rgpt)
  FINAL REASONING = DeepSeek-R1       (text-only, SharedMemory + query -> answer)

Usage:
    python run_eval_mas_v2.py \
        --benchmark 3dsrbench \
        --train_ratio 0.5 \
        --seed 42

Full HF: TTO on 500 samples, inference on the rest:
    python run_eval_mas_v2.py --benchmark 3dsrbench --hf_full_dataset \
        --train_samples 500 --use_tto --trust_step 4 --use_vlm_reasoning ...

Or import from Jupyter:
    from run_eval_mas_v2 import build_runners, run_experiment, run_test_only

Test-only (no train split):
    python run_eval_mas_v2.py --benchmark cvbench --max_samples 10 --test_only --use_local_reasoning
"""
import argparse
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Suppress noisy deprecation/generation warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")
warnings.filterwarnings("ignore", message=".*offload.*buffer.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2 import (
    ALL_CATEGORIES, SPECIALIST_LLMS_5, ROLES,
    ScoreMap, ScoreMapUpdater,
    run_train, run_test, compute_accuracy,
)
from src2.benchmarks.loaders import load_benchmark, load_benchmark_from_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def cuda_cleanup():
    """Free cached GPU memory in the current process."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def gpu_memory_free_gib() -> Optional[float]:
    """Return free GPU memory in GiB, or None if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return None


def warn_if_gpu_memory_low(min_free_gib: float = 20.0) -> None:
    free = gpu_memory_free_gib()
    if free is None:
        return
    if free < min_free_gib:
        logger.warning(
            "Low GPU memory: %.1f GiB free (recommend >= %.0f GiB). "
            "Run: bash scripts/gpu_cleanup.sh --kill-all",
            free, min_free_gib,
        )


def _offload_runner_to_cpu(runner) -> None:
    if runner is None:
        return
    if hasattr(runner, "model") and runner.model is not None:
        runner.model = runner.model.to("cpu")
    if hasattr(runner, "device"):
        runner.device = "cpu"
    cuda_cleanup()


def _ensure_runner_on_gpu(runner, device: str) -> None:
    if runner is None:
        return
    if hasattr(runner, "model") and runner.model is not None:
        runner.model = runner.model.to(device)
    if hasattr(runner, "device"):
        runner.device = device


# ======================================================================
# Model loading helpers
# ======================================================================
def build_runners(
    reasoning_api_base: str = "http://localhost:8000/v1",
    reasoning_api_key: str = "EMPTY",
    reasoning_model_name: str = "deepseek-r1",
    specialist_device: str = "cuda",
    specialist_only_device: Optional[str] = None,
    specialist_offload_after_use: bool = False,
    specialist_whitelist: Optional[List[str]] = None,
    use_local_reasoning: bool = False,
    reasoning_local_model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    use_vlm_reasoning: bool = False,
    reasoning_vlm_model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    temperature: float = 0.0,
    top_p: float = 0.9,
    **kwargs,
):
    """Instantiate all model runners.

    Returns (head_generate, specialist_generate, reasoning_generate).

    Signatures:
      head_generate(image, prompt) -> str          Qwen3-VL-4B
      specialist_generate(llm_name, image, prompt) -> str
      reasoning_generate(prompt, image=None) -> str  DeepSeek-R1 (text) or Qwen3-VL-8B (image+text)

    specialist_only_device: If set (e.g. "cpu"), specialists load on this device to avoid OOM
      when head+reasoning already fill GPU. Head and reasoning stay on specialist_device.
    specialist_offload_after_use: If True, specialists are loaded to CPU and moved to GPU only
      during inference, then offloaded back to CPU. Saves GPU memory (올렸다 내렸다).
    specialist_whitelist: If set, only load these specialists (e.g. ["qwen3_4b","llava4d","spatial_reasoner"]).
      Step2 호환용. None이면 전체 로드.
    use_local_reasoning: If True, load DeepSeek-R1-Distill locally (no API).
    use_vlm_reasoning: If True, use Qwen3-VL-8B for Final Reasoning (image+SharedMemory). Overrides use_local_reasoning for reasoning.
    """
    _spec_device = specialist_only_device if specialist_only_device is not None else specialist_device
    if specialist_offload_after_use:
        _spec_device = "cpu"  # load to CPU, move to GPU only during inference
    _load_device = "cpu" if specialist_offload_after_use else specialist_device
    from src2.models.qwen3 import Qwen3Runner
    from src2.models.llava import LLaVARunner
    from src2.models.sa2va import Sa2VARunner
    from src2.models.deepseek_r1 import DeepSeekR1Runner, DeepSeekR1LocalRunner

    # --- Head Agent (Qwen3-VL-4B, VLM) ---
    # Reused from specialist cache; loaded once, shared.
    _head_runner = None

    def _get_head():
        nonlocal _head_runner
        if _head_runner is None:
            _head_runner = Qwen3Runner(device=_load_device)
        return _head_runner

    def _offload_head_to_cpu():
        if _head_runner is not None:
            _offload_runner_to_cpu(_head_runner)

    def _ensure_head_on_gpu():
        _ensure_runner_on_gpu(_get_head(), specialist_device)

    def head_generate(image, prompt: str) -> str:
        if specialist_offload_after_use:
            _offload_reasoning_to_cpu()
        _ensure_head_on_gpu()
        out = _get_head().generate(
            image, prompt,
            temperature=temperature,
            max_new_tokens=64,
            top_p=top_p if temperature > 0 else 0.0,
        )
        if specialist_offload_after_use:
            _offload_head_to_cpu()
            cuda_cleanup()
        return out

    # --- 5 Specialist VLMs (lazy-loaded, cached) ---
    _specialist_cache = {}
    _last_specialist_on_gpu: Optional[str] = None

    def _offload_specialist_to_cpu(name: str):
        if name not in _specialist_cache:
            return
        if name == "qwen3_4b":
            _offload_head_to_cpu()
            return
        runner = _specialist_cache[name]
        _offload_runner_to_cpu(runner)

    def _ensure_specialist_on_gpu(name: str):
        if name not in _specialist_cache:
            return
        if name == "qwen3_4b":
            _ensure_head_on_gpu()
            return
        _ensure_runner_on_gpu(_specialist_cache[name], specialist_device)

    def _get_specialist(name: str):
        if specialist_whitelist is not None and name not in specialist_whitelist:
            raise ValueError(f"Specialist {name} not in whitelist {specialist_whitelist}")
        if name not in _specialist_cache:
            if name == "qwen3_4b":
                _specialist_cache[name] = _get_head()
            elif name == "sa2va":
                _specialist_cache[name] = Sa2VARunner(device=_spec_device)
            elif name == "llava4d":
                _specialist_cache[name] = LLaVARunner(
                    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
                    device=_spec_device,
                )
            elif name == "spatial_rgpt":
                from src2.models.spatial_rgpt import SpatialRGPTRunner
                _specialist_cache[name] = SpatialRGPTRunner(device=_spec_device)
            elif name == "spaceom":
                from src2.models.spaceom import SpaceOmRunner
                _specialist_cache[name] = SpaceOmRunner(device=_spec_device)
            elif name == "spatial_reasoner":
                from src2.models.spatial_reasoner import SpatialReasonerRunner
                _specialist_cache[name] = SpatialReasonerRunner(
                    model_id="ccvl/SpatialReasoner",
                    device=_spec_device,
                )
            else:
                raise ValueError(f"Unknown specialist: {name}")
        return _specialist_cache[name]

    def specialist_generate(llm_name: str, image, prompt: str) -> str:
        if specialist_offload_after_use:
            _offload_reasoning_to_cpu()
        runner = _get_specialist(llm_name)
        if specialist_offload_after_use:
            nonlocal _last_specialist_on_gpu
            if _last_specialist_on_gpu and _last_specialist_on_gpu != llm_name:
                _offload_specialist_to_cpu(_last_specialist_on_gpu)
                _last_specialist_on_gpu = None
            if llm_name != "qwen3_4b":
                _offload_head_to_cpu()
                cuda_cleanup()
            _ensure_specialist_on_gpu(llm_name)
            _last_specialist_on_gpu = llm_name
        out = runner.generate(
            image, prompt,
            temperature=temperature,
            max_new_tokens=1024,
            top_p=top_p if temperature > 0 else 0.0,
        )
        if specialist_offload_after_use:
            _offload_specialist_to_cpu(llm_name)
            _last_specialist_on_gpu = None
            cuda_cleanup()
        return out

    # --- Final Reasoning Agent ---
    _reasoning_local = None

    def _get_reasoning_local():
        nonlocal _reasoning_local
        if _reasoning_local is None:
            _reasoning_local = DeepSeekR1LocalRunner(
                model_id=reasoning_local_model_id,
                device=_load_device,
            )
        return _reasoning_local

    def _offload_reasoning_to_cpu():
        if _reasoning_local is not None:
            _offload_runner_to_cpu(_reasoning_local)

    def _ensure_reasoning_on_gpu():
        _ensure_runner_on_gpu(_get_reasoning_local(), specialist_device)

    if use_vlm_reasoning:
        # Qwen3-VL-8B: image + SharedMemory + query (can verify specialist claims)
        _reasoning_vlm = Qwen3Runner(
            model_id=reasoning_vlm_model_id,
            device=specialist_device,
        )

        def reasoning_generate(prompt: str, image=None):
            if image is None:
                raise ValueError("use_vlm_reasoning=True requires image for Final Reasoning")
            return _reasoning_vlm.generate(
                image, prompt,
                temperature=temperature,
                max_new_tokens=1024,
                top_p=top_p if temperature > 0 else 0.0,
            )
    elif use_local_reasoning:
        def reasoning_generate(prompt: str, image=None):
            if specialist_offload_after_use:
                _offload_head_to_cpu()
                if _last_specialist_on_gpu:
                    _offload_specialist_to_cpu(_last_specialist_on_gpu)
                cuda_cleanup()
            _ensure_reasoning_on_gpu()
            out = _get_reasoning_local().generate(
                prompt,
                temperature=temperature,
                max_tokens=1024,
                top_p=top_p if temperature > 0 else 0.0,
            )
            if specialist_offload_after_use:
                _offload_reasoning_to_cpu()
                cuda_cleanup()
            return out
    else:
        reasoning = DeepSeekR1Runner(
            api_base=reasoning_api_base,
            api_key=reasoning_api_key,
            model_name=reasoning_model_name,
        )

        def reasoning_generate(prompt: str, image=None):
            return reasoning.generate(
                prompt,
                temperature=temperature,
                max_tokens=1024,
                top_p=top_p if temperature > 0 else 0.0,
            )

    return head_generate, specialist_generate, reasoning_generate


# ======================================================================
# Dataset splitting
# ======================================================================
def split_dataset(dataset, train_ratio: float = 0.5, seed: int = 42):
    """Randomly split a HF dataset into train and test subsets."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    train_idx = sorted(indices[:split_point])
    test_idx = sorted(indices[split_point:])
    return dataset.select(train_idx), dataset.select(test_idx)


def split_dataset_fixed_train_size(dataset, train_size: int, seed: int = 42):
    """Shuffle then take exactly train_size samples for TTO; remaining for inference (no overlap)."""
    n = len(dataset)
    if train_size <= 0:
        raise ValueError(f"train_size must be positive, got {train_size}")
    if train_size >= n:
        raise ValueError(
            f"train_size ({train_size}) must be < dataset length ({n}) to leave a test set."
        )
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    train_idx = sorted(indices[:train_size])
    test_idx = sorted(indices[train_size:])
    return dataset.select(train_idx), dataset.select(test_idx)


# ======================================================================
# Test-only runner (no train/test split, no ScoreMap update)
# ======================================================================
def run_test_only(
    benchmark: str,
    head_generate,
    specialist_generate,
    reasoning_generate,
    max_samples: Optional[int],
    seed: int = 42,
    output_dir: str = None,
    random_agents: bool = True,
    use_vlm_reasoning: bool = False,
    specialist_llms: list = None,
    dataset_subdir: str = None,
    verbose: bool = True,
    updater=None,
    use_frozen: bool = True,
    timing_log_path: Optional[str] = None,
):
    """Run MAS v2 pipeline on N samples.

    Pipeline: Head → ScoreMap → 3 Specialists → SharedMemory → Final Reasoning.
    When updater is provided (--use_tto), scores are updated each step (TTO).
    Otherwise random_agents=True uses random selection, no score updates.
    """
    if dataset_subdir:
        dataset = load_benchmark_from_dataset(
            benchmark, dataset_subdir,
            project_root=Path(__file__).resolve().parent,
            max_samples=max_samples,
            seed=seed,
        )
        logger.info("Loaded from data/dataset/%s (%d samples)", dataset_subdir, len(dataset))
    else:
        dataset = load_benchmark(
            benchmark, max_samples=max_samples, seed=seed, use_frozen=use_frozen
        )
    logger.info("Loaded %d samples (test only, no train)", len(dataset))

    specialist_llms = specialist_llms or SPECIALIST_LLMS_5
    score_map = ScoreMap(categories=ALL_CATEGORIES, llms=specialist_llms, seed=seed)

    use_random = random_agents and updater is None
    logger.info("=" * 60)
    logger.info("TESTING (%d samples, random_agents=%s, tto_updates=%s)", len(dataset), use_random, updater is not None)
    logger.info("=" * 60)

    ts = None
    out_path = None
    if output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir) / ts
        out_path.mkdir(parents=True, exist_ok=True)

    tl_path = timing_log_path
    if tl_path is None and out_path is not None:
        tl_path = str(out_path / "mas_timing.jsonl")

    results = run_test(
        dataset=dataset,
        benchmark=benchmark,
        score_map=score_map,
        head_generate=head_generate,
        specialist_generate=specialist_generate,
        reasoning_generate=reasoning_generate,
        random_agents=use_random,
        use_vlm_reasoning=use_vlm_reasoning,
        verbose=verbose,
        updater=updater,
        update_scores=updater is not None,
        timing_log_path=tl_path,
    )
    metrics = compute_accuracy(results)
    logger.info(
        "Accuracy: %.2f%% (%d/%d)",
        100 * metrics["accuracy"],
        metrics["correct"], metrics["total"],
    )

    if output_dir and out_path is not None:
        summary = {
            "benchmark": benchmark,
            "samples": len(dataset),
            "seed": seed,
            "random_agents": use_random,
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            "per_category": metrics["per_category"],
            "specialist_llms": specialist_llms,
            "roles": ROLES,
            "timestamp": ts,
            "timing_log_jsonl": tl_path,
        }
        (out_path / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        with open(out_path / "details.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        logger.info("Results saved to %s", out_path)
        if tl_path:
            logger.info("Timing log (JSONL + .log): %s", tl_path)

    return {"results": results, "metrics": metrics}


# ======================================================================
# Main experiment runner (train + test split)
# ======================================================================
def run_experiment(
    benchmark: str,
    head_generate,
    specialist_generate,
    reasoning_generate,
    train_ratio: float = 0.5,
    train_samples: Optional[int] = None,
    seed: int = 42,
    output_dir: str = None,
    updater: ScoreMapUpdater = None,
    max_samples: int = None,
    use_vlm_reasoning: bool = False,
    specialist_llms: list = None,
    dataset_subdir: str = None,
    use_frozen: bool = True,
    timing_log_path: Optional[str] = None,
):
    """Run full MAS v2 experiment: load data -> split -> train -> test -> report.

    If train_samples is set, use exactly that many shuffled examples for TTO (train)
    and the rest for inference (test). Overrides train_ratio.
    """

    specialist_llms = specialist_llms or SPECIALIST_LLMS_5
    logger.info("Benchmark: %s | Categories: %d (fixed) | Specialists: %s | Seed: %d",
                benchmark, len(ALL_CATEGORIES), specialist_llms, seed)

    if dataset_subdir:
        dataset = load_benchmark_from_dataset(
            benchmark, dataset_subdir,
            project_root=Path(__file__).resolve().parent,
            max_samples=max_samples,
            seed=seed,
        )
        logger.info("Loaded from data/dataset/%s (%d samples)", dataset_subdir, len(dataset))
    else:
        dataset = load_benchmark(
            benchmark, max_samples=max_samples, seed=seed, use_frozen=use_frozen
        )
    logger.info("Loaded %d samples", len(dataset))

    if train_samples is not None:
        train_ds, test_ds = split_dataset_fixed_train_size(dataset, train_samples, seed=seed)
        logger.info(
            "Split: TTO optimization on %d samples | inference on %d (train_samples=%d, overrides train_ratio)",
            len(train_ds), len(test_ds), train_samples,
        )
    else:
        train_ds, test_ds = split_dataset(dataset, train_ratio=train_ratio, seed=seed)
        logger.info("Train: %d | Test: %d (train_ratio=%s)", len(train_ds), len(test_ds), train_ratio)

    score_map = ScoreMap(categories=ALL_CATEGORIES, llms=specialist_llms, seed=seed)
    updater = updater or ScoreMapUpdater()

    out_path = None
    ts = None
    tl_path = timing_log_path
    if output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir) / ts
        out_path.mkdir(parents=True, exist_ok=True)
        if tl_path is None:
            tl_path = str(out_path / "mas_timing.jsonl")

    # --- Train phase ---
    logger.info("=" * 60)
    logger.info("TRAIN PHASE")
    logger.info("=" * 60)
    train_results = run_train(
        dataset=train_ds,
        benchmark=benchmark,
        score_map=score_map,
        head_generate=head_generate,
        specialist_generate=specialist_generate,
        reasoning_generate=reasoning_generate,
        updater=updater,
        seed=seed,
        use_vlm_reasoning=use_vlm_reasoning,
        timing_log_path=tl_path,
    )
    train_metrics = compute_accuracy(train_results)
    logger.info(
        "Train accuracy: %.2f%% (%d/%d)",
        100 * train_metrics["accuracy"],
        train_metrics["correct"], train_metrics["total"],
    )

    # --- Test phase (frozen score map) ---
    logger.info("=" * 60)
    logger.info("TEST PHASE (score map frozen)")
    logger.info("=" * 60)
    test_results = run_test(
        dataset=test_ds,
        benchmark=benchmark,
        score_map=score_map,
        head_generate=head_generate,
        specialist_generate=specialist_generate,
        reasoning_generate=reasoning_generate,
        use_vlm_reasoning=use_vlm_reasoning,
        timing_log_path=tl_path,
    )
    test_metrics = compute_accuracy(test_results)
    logger.info(
        "Test accuracy: %.2f%% (%d/%d)",
        100 * test_metrics["accuracy"],
        test_metrics["correct"], test_metrics["total"],
    )

    # --- Save results ---
    if output_dir and out_path is not None:

        score_map.save(str(out_path / "score_map_final.json"))

        summary = {
            "benchmark": benchmark,
            "seed": seed,
            "train_ratio": train_ratio,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "train_accuracy": train_metrics["accuracy"],
            "train_per_category": train_metrics["per_category"],
            "test_accuracy": test_metrics["accuracy"],
            "test_per_category": test_metrics["per_category"],
            "specialist_llms": specialist_llms,
            "roles": ROLES,
            "timestamp": ts,
            "timing_log_jsonl": tl_path,
        }
        (out_path / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

        with open(out_path / "train_details.jsonl", "w") as f:
            for r in train_results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        with open(out_path / "test_details.jsonl", "w") as f:
            for r in test_results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        logger.info("Results saved to %s", out_path)
        if tl_path:
            logger.info("Timing log (JSONL + .log): %s", tl_path)

    return {
        "score_map": score_map,
        "train_results": train_results,
        "test_results": test_results,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


# ======================================================================
# CLI
# ======================================================================
def main():
    cuda_cleanup()
    warn_if_gpu_memory_low(min_free_gib=20.0)
    parser = argparse.ArgumentParser(description="MAS v2 evaluation")
    parser.add_argument("--benchmark", choices=["3dsrbench", "cvbench", "mindcube"], required=True)
    parser.add_argument("--train_ratio", type=float, default=0.5,
                        help="Random fraction for train split (ignored if --train_samples is set)")
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Exact train size for TTO (e.g. 500 on full HF); remaining rows = inference test. "
             "Requires full dataset (e.g. --hf_full_dataset). Overrides --train_ratio.",
    )
    parser.add_argument(
        "--timing_log",
        type=str,
        default=None,
        help="Chemin mas_timing.jsonl (append JSONL + .log). Par défaut: sous-dossier horodaté avec --output_dir.",
    )
    parser.add_argument(
        "--mindcube_split",
        type=str,
        default="train",
        help="Split Hugging Face MindCube (ex. train, test). Équivaut à MINDCUBE_SPLIT.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Testing only: run pipeline on N samples, no train/test split, no ScoreMap update",
    )
    parser.add_argument("--output_dir", type=str, default="results/mas_v2")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--reasoning_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--reasoning_api_key", type=str, default="EMPTY")
    parser.add_argument("--reasoning_model_name", type=str, default="deepseek-r1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--use_local_reasoning",
        action="store_true",
        help="Use DeepSeek-R1-Distill locally (H100, no API server)",
    )
    parser.add_argument(
        "--reasoning_local_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Local reasoning model when --use_local_reasoning",
    )
    parser.add_argument(
        "--specialist_offload_after_use",
        action="store_true",
        help="Load specialists to CPU, move to GPU only during inference, then offload (올렸다 내렸다)",
    )
    parser.add_argument(
        "--use_vlm_reasoning",
        action="store_true",
        help="Use Qwen3-VL-8B for Final Reasoning (image+SharedMemory)",
    )
    parser.add_argument(
        "--specialist_whitelist",
        type=str,
        default=None,
        help="Comma-separated specialist subset, e.g. qwen3_4b,llava4d,sa2va,spatial_rgpt. "
             "Use to avoid Sa2VA/SpatialRGPT when env incompatible (H100).",
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default=None,
        help="Load from data/dataset/<subdir> instead of HuggingFace. e.g. 3dsrbench_train_300",
    )
    parser.add_argument(
        "--hf_full_dataset",
        action="store_true",
        help="Load full benchmark from HuggingFace (MindCube, ccvl/3DSRBench, …) instead of frozen subset "
        "(data/frozen_benchmarks/…). Ignored if --dataset_subdir is set.",
    )
    parser.add_argument(
        "--use_tto",
        action="store_true",
        help="Use TTO (Trust Score) updater: trust_score.run_step4 (Beta+EMA).",
    )
    parser.add_argument(
        "--trust_step",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="TTO step when --use_tto: 1=s+=R, 2=s+=R̃, 3=s+=γ·R̃, 4=Beta+EMA.",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="3-agent mode (qwen3_4b,llava4d,spatial_reasoner) for GPU sharing / OOM.",
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        help="Disable verbose step logging (step/acc/cat/assign, scores every 5 steps).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature. 0=greedy, >0=sampling (e.g. 0.7).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p when temperature>0 (default 0.9).",
    )
    args = parser.parse_args()

    if args.benchmark == "mindcube":
        os.environ["MINDCUBE_SPLIT"] = args.mindcube_split

    specialist_whitelist = None
    if args.low_memory:
        specialist_whitelist = ["qwen3_4b", "llava4d", "spatial_reasoner"]
        logger.info("Low-memory mode: specialist whitelist %s", specialist_whitelist)
    elif args.specialist_whitelist:
        specialist_whitelist = [s.strip() for s in args.specialist_whitelist.split(",") if s.strip()]
        if specialist_whitelist:
            logger.info("Specialist whitelist: %s (excludes sa2va/spatial_rgpt when 3-agent)", specialist_whitelist)
        else:
            specialist_whitelist = None

    use_frozen = not args.hf_full_dataset
    if args.benchmark == "mindcube":
        use_frozen = False

    if args.test_only and not args.max_samples and not args.hf_full_dataset:
        parser.error(
            "--max_samples required when --test_only (unless --hf_full_dataset for all HF samples)"
        )
    if args.hf_full_dataset and args.dataset_subdir:
        parser.error("--hf_full_dataset cannot be combined with --dataset_subdir")
    if args.train_samples is not None and args.test_only:
        parser.error("--train_samples applies to train+test experiment only (do not use with --test_only)")
    if args.train_samples is not None and args.train_samples < 1:
        parser.error("--train_samples must be >= 1")

    head_gen, spec_gen, reason_gen = build_runners(
        reasoning_api_base=args.reasoning_api_base,
        reasoning_api_key=args.reasoning_api_key,
        reasoning_model_name=args.reasoning_model_name,
        specialist_device=args.device,
        specialist_offload_after_use=args.specialist_offload_after_use,
        specialist_whitelist=specialist_whitelist,
        use_local_reasoning=args.use_local_reasoning,
        reasoning_local_model_id=args.reasoning_local_model,
        use_vlm_reasoning=args.use_vlm_reasoning,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    specialist_llms = specialist_whitelist or SPECIALIST_LLMS_5

    updater = None
    if args.use_tto:
        try:
            if args.trust_step == 1:
                from test_confidence_mas_v3_step1 import TrustScoreMapUpdaterStep1
                updater = TrustScoreMapUpdaterStep1(kappa=1.0)
            elif args.trust_step == 2:
                from test_confidence_mas_v2_step2 import TrustScoreMapUpdaterStep2
                updater = TrustScoreMapUpdaterStep2(T=10.0, kappa=1.0)
            elif args.trust_step == 3:
                from test_confidence_mas_v3_step3 import TrustScoreMapUpdaterStep3
                updater = TrustScoreMapUpdaterStep3(T=10.0, kappa=1.0, gamma=0.1)
            else:
                from test_confidence_mas_v3_step4 import TrustScoreMapUpdaterStep4
                updater = TrustScoreMapUpdaterStep4(T=10.0, kappa=1.0, gamma=0.1)
            logger.info("Using TTO trust updater (step %d)", args.trust_step)
        except ImportError as e:
            logger.warning("TTO updater not available: %s. Using default ScoreMapUpdater.", e)

    if args.test_only:
        if args.max_samples:
            out_dir = f"{args.output_dir}/{args.benchmark}/{args.max_samples}samples"
        else:
            out_dir = f"{args.output_dir}/{args.benchmark}/full_hf"
        run_test_only(
            benchmark=args.benchmark,
            head_generate=head_gen,
            specialist_generate=spec_gen,
            reasoning_generate=reason_gen,
            max_samples=args.max_samples,
            seed=args.seed,
            output_dir=out_dir,
            specialist_llms=specialist_llms,
            dataset_subdir=args.dataset_subdir,
            verbose=not args.no_verbose,
            updater=updater if args.use_tto else None,
            use_frozen=use_frozen,
            timing_log_path=args.timing_log,
        )
    else:
        if args.max_samples:
            out_dir = f"{args.output_dir}/{args.benchmark}/{args.max_samples}samples"
        elif args.hf_full_dataset:
            out_dir = f"{args.output_dir}/{args.benchmark}/full_hf"
        else:
            out_dir = f"{args.output_dir}/{args.benchmark}"
        run_experiment(
            benchmark=args.benchmark,
            head_generate=head_gen,
            specialist_generate=spec_gen,
            reasoning_generate=reason_gen,
            train_ratio=args.train_ratio,
            train_samples=args.train_samples,
            seed=args.seed,
            output_dir=out_dir,
            max_samples=args.max_samples,
            specialist_llms=specialist_llms,
            updater=updater,
            use_vlm_reasoning=args.use_vlm_reasoning,
            dataset_subdir=args.dataset_subdir,
            use_frozen=use_frozen,
            timing_log_path=args.timing_log,
        )


if __name__ == "__main__":
    main()
