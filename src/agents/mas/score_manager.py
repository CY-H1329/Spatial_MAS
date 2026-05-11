"""
Score table: w[model, category] for agent selection.
Initial: 0.5. Correct: +0.05. Wrong: -0.02.
"""
from typing import Dict, List, Tuple

from .config import TASK_CATEGORIES, INITIAL_WEIGHT, SCORE_DELTA_CORRECT, SCORE_DELTA_WRONG, get_candidate_agents


class ScoreManager:
    """Manages per-model per-category weights for agent selection."""

    def __init__(self):
        self._scores: Dict[str, Dict[str, float]] = {}
        for m in get_candidate_agents():
            self._scores[m] = {c: INITIAL_WEIGHT for c in TASK_CATEGORIES}

    def get(self, model: str, category: str) -> float:
        return self._scores.get(model, {}).get(category, INITIAL_WEIGHT)

    def update(self, model: str, category: str, correct: bool):
        if model not in self._scores:
            self._scores[model] = {c: INITIAL_WEIGHT for c in TASK_CATEGORIES}
        delta = SCORE_DELTA_CORRECT if correct else SCORE_DELTA_WRONG
        self._scores[model][category] = max(0.0, min(1.0, self._scores[model].get(category, INITIAL_WEIGHT) + delta))

    def get_top_k(self, category: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k models by score for this category."""
        pairs = [(m, self.get(m, category)) for m in self._scores]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def to_dict(self) -> dict:
        return {m: dict(self._scores[m]) for m in self._scores}
