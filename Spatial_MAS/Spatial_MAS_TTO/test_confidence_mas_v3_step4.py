#!/usr/bin/env python3
"""
Confidence MAS v3 — run_step4 (Beta + EMA 신뢰도 업데이트).

step3: gamma=0.1, 단순 누적
step4: Beta 분포 + EMA (단기 f, 장기 g) → 최종 점수 s

Usage (Jupyter):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/home/jovyan/CY/Spatial_MAS")))

    from test_confidence_mas_v3_step4 import run_confidence_mas_test_step4, build_runners_for_confidence

    head_gen, spec_gen, reason_gen = build_runners_for_confidence(
        specialist_device="cuda",
        use_vlm_reasoning=True,
    )
    results = run_confidence_mas_test_step4(
        head_gen, spec_gen, reason_gen,
        benchmark="cvbench",
        max_samples=50,
        T=10.0,
        kappa=1.0,
        gamma=0.1,
    )
    print(f"Accuracy: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
"""
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2 import (
    ALL_CATEGORIES,
    ROLES,
    ScoreMap,
    run_step,
)
from src2.benchmarks.loaders import (
    load_benchmark,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
)

# Trust score (spatial_aomas or Spatial_AOMAS)
def _load_trust_step4():
    try:
        from spatial_aomas.trust_score import run_step4, TrustState
        return run_step4, TrustState
    except ImportError:
        pass
    for base in [
        Path(__file__).resolve().parent / "spatial_aomas",
        Path(__file__).resolve().parent.parent / "Spatial_AOMAS",
    ]:
        if base.exists() and (base / "trust_score.py").exists():
            sys.path.insert(0, str(base))
            try:
                from trust_score import run_step4, TrustState
                return run_step4, TrustState
            except ImportError:
                pass
    return None, None

trust_run_step4, TrustState = _load_trust_step4()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# TrustScoreMapUpdater — run_step4 (Beta + EMA)
# ======================================================================
class TrustScoreMapUpdaterStep4:
    """ScoreMap updater using trust_score run_step4 (Beta + EMA)."""

    def __init__(
        self,
        T: float = 10.0,
        kappa: float = 1.0,
        gamma: float = 0.1,
        lambda_f: float = 0.3,
        lambda_g: float = 0.1,
        mu: float = 0.5,
    ):
        self.T = T
        self.kappa = kappa
        self.gamma = gamma
        self.lambda_f = lambda_f
        self.lambda_g = lambda_g
        self.mu = mu
        self._N_c: dict = defaultdict(int)
        self._state: dict = {}  # {agent: {category: {role: TrustState}}}

    def _score_map_to_scores(self, score_map: ScoreMap) -> dict:
        """{agent: {category: {role: score}}}"""
        scores = {}
        for cat in score_map.categories:
            for role in score_map.roles:
                for llm in score_map.llms:
                    if llm not in scores:
                        scores[llm] = {}
                    if cat not in scores[llm]:
                        scores[llm][cat] = {}
                    scores[llm][cat][role] = score_map.get_score(cat, role, llm)
        return scores

    def _ensure_state(self, score_map: ScoreMap):
        """Initialize state from score_map if empty."""
        if self._state:
            return
        if TrustState is None:
            return
        for cat in score_map.categories:
            for role in score_map.roles:
                for llm in score_map.llms:
                    if llm not in self._state:
                        self._state[llm] = {}
                    if cat not in self._state[llm]:
                        self._state[llm][cat] = {}
                    s = score_map.get_score(cat, role, llm)
                    self._state[llm][cat][role] = TrustState(s=s)

    def _scores_to_score_map(self, scores: dict, score_map: ScoreMap):
        """Write scores dict back to ScoreMap."""
        for llm, cats in scores.items():
            for cat, roles in cats.items():
                for role, val in roles.items():
                    score_map.set_score(cat, role, llm, val)

    def update(
        self,
        score_map: ScoreMap,
        category: str,
        assignments: list,
        agent_results: list,
        final_answer: str,
        gt: str,
        step: int,
        total_steps: int,
    ) -> None:
        if trust_run_step4 is None:
            logger.warning("trust_score not available, skipping update")
            return

        self._N_c[category] += 1
        N_c = self._N_c[category]

        agent_answers = {r["llm_name"]: r.get("answer", "") for r in agent_results}
        agent_roles = {r["llm_name"]: r["role"] for r in agent_results}

        self._ensure_state(score_map)
        scores = self._score_map_to_scores(score_map)

        updated_state = trust_run_step4(
            self._state,
            scores,
            agent_answers,
            final_answer,
            gt,
            category,
            agent_roles,
            N_c=N_c,
            kappa=self.kappa,
            T=self.T,
            gamma=self.gamma,
            lambda_f=self.lambda_f,
            lambda_g=self.lambda_g,
            mu=self.mu,
        )
        self._state = updated_state
        self._scores_to_score_map(scores, score_map)


# ======================================================================
# build_runners_for_confidence
# ======================================================================
def build_runners_for_confidence(
    specialist_device: str = "cuda",
    specialist_llms: list = None,
    use_vlm_reasoning: bool = True,
    specialist_offload_after_use: bool = False,
):
    """Build runners for Confidence MAS."""
    from run_eval_mas_v2 import build_runners

    kwargs = dict(
        specialist_device=specialist_device,
        use_vlm_reasoning=use_vlm_reasoning,
    )
    if specialist_offload_after_use:
        kwargs["specialist_offload_after_use"] = True
    try:
        return build_runners(**kwargs)
    except TypeError:
        kwargs.pop("specialist_offload_after_use", None)
        return build_runners(**kwargs)


# ======================================================================
# run_confidence_mas_test_step4
# ======================================================================
def run_confidence_mas_test_step4(
    head_generate,
    specialist_generate,
    reasoning_generate,
    benchmark: str = "cvbench",
    max_samples: int = 50,
    seed: int = 42,
    T: float = 10.0,
    kappa: float = 1.0,
    gamma: float = 0.1,
    lambda_f: float = 0.3,
    lambda_g: float = 0.1,
    mu: float = 0.5,
    specialist_llms: list = None,
    use_vlm_reasoning: bool = True,
):
    """
    Run Confidence MAS with run_step4 (Beta + EMA).

    Returns: {correct, total, accuracy, per_category, score_map, ...}
    """
    specialist_llms = specialist_llms or ["qwen3_4b", "llava4d", "spatial_reasoner"]
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)

    score_map = ScoreMap(categories=ALL_CATEGORIES, llms=specialist_llms, roles=ROLES, seed=seed)
    updater = TrustScoreMapUpdaterStep4(
        T=T, kappa=kappa, gamma=gamma,
        lambda_f=lambda_f, lambda_g=lambda_g, mu=mu,
    )

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    samples = [dataset[i] for i in indices]

    logger.info(
        "Confidence MAS v3 (run_step4) — %s (n=%d)\n  step=0: qwen3_4b 고정, step>0: run_step4 기반 선택 (T=%.1f, kappa=%.1f, gamma=%.1f, λ_f=%.1f, λ_g=%.1f, μ=%.1f)\n  specialists: %s",
        benchmark.upper(),
        len(samples),
        T, kappa, gamma, lambda_f, lambda_g, mu,
        specialist_llms,
    )

    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for step, ex in enumerate(samples):
        image = get_benchmark_image(ex, benchmark)
        if image is None:
            continue
        query = get_benchmark_prompt(ex, benchmark)
        gt_raw = get_benchmark_answer(ex, benchmark)
        gt = (gt_raw or "").strip().upper()
        if not any(c in gt for c in "ABCD"):
            continue

        result = run_step(
            image=image,
            query=query,
            gt=gt,
            step=step,
            total_steps=len(samples),
            score_map=score_map,
            head_generate=head_generate,
            specialist_generate=specialist_generate,
            reasoning_generate=reasoning_generate,
            updater=updater,
            update_scores=True,
            use_vlm_reasoning=use_vlm_reasoning,
        )

        hit = result.get("correct", False)
        total += 1
        if hit:
            correct += 1
        cat = result.get("category", "unknown")
        by_category[cat]["total"] += 1
        if hit:
            by_category[cat]["correct"] += 1

        acc_pct = 100 * correct / total if total else 0
        assign = result.get("assignments", [])
        logger.info(
            "  Step %d/%d | acc: %.1f%% | cat: %s | assign: %s",
            step + 1,
            len(samples),
            acc_pct,
            cat,
            assign,
        )
        if (step + 1) % 5 == 0 or step == 0:
            maps = score_map.get_all_maps()
            logger.info("    scores (step %d): %s", step + 1, maps)

    per_cat = {c: v["correct"] / v["total"] if v["total"] else 0 for c, v in by_category.items()}
    accuracy = correct / total if total else 0

    print("\n" + "=" * 70)
    print(f"CONFIDENCE MAS v3 (run_step4) — {benchmark.upper()} — 최종 결과")
    print("=" * 70)
    print(f"Overall: {correct}/{total} = {100*accuracy:.1f}%\n")
    for c in sorted(by_category.keys()):
        v = by_category[c]
        pct = 100 * v["correct"] / v["total"] if v["total"] else 0
        print(f"  {c:30} {pct:5.1f}%  ({v['correct']}/{v['total']})")
    print("=" * 70)

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "per_category": per_cat,
        "by_category": dict(by_category),
        "score_map": score_map,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="cvbench")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lambda_f", type=float, default=0.3)
    parser.add_argument("--lambda_g", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_vlm_reasoning", action="store_true", default=True)
    args = parser.parse_args()

    head_gen, spec_gen, reason_gen = build_runners_for_confidence(
        specialist_device=args.device,
        use_vlm_reasoning=args.use_vlm_reasoning,
    )
    results = run_confidence_mas_test_step4(
        head_gen, spec_gen, reason_gen,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        T=args.T,
        kappa=args.kappa,
        gamma=args.gamma,
        lambda_f=args.lambda_f,
        lambda_g=args.lambda_g,
        mu=args.mu,
        use_vlm_reasoning=args.use_vlm_reasoning,
    )
    print(f"\nAccuracy: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
