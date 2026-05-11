"""
MAS configuration: agent profiles (per-category performance), score deltas, categories.
Profiles from configs/mas/agent_profiles/*.json or fallback to hardcoded baselines.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Project root (assume src/agents/mas/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_MAS = _PROJECT_ROOT / "configs" / "mas"

# Default candidate specialist agents (includes InternVL2 + Qwen2-VL for SpatiO-style ablations).
# Override at runtime: MAS_CANDIDATE_AGENTS="qwen3_4b,sa2va,llava4d,internvl2,qwen2_vl"
CANDIDATE_AGENTS = [
    "llava4d",
    "qwen3_4b",
    "sa2va",
    "internvl2",
    "qwen2_vl",
    "claude_sonnet_4_5",
    "gpt4o",
    "gemini_robotics_er",
]


def get_candidate_agents() -> List[str]:
    """Subset of CANDIDATE_AGENTS from env MAS_CANDIDATE_AGENTS (comma-separated)."""
    raw = os.environ.get("MAS_CANDIDATE_AGENTS", "").strip()
    if not raw:
        return list(CANDIDATE_AGENTS)
    allowed = set(CANDIDATE_AGENTS)
    out = [x.strip() for x in raw.split(",") if x.strip() in allowed]
    return out if out else list(CANDIDATE_AGENTS)

# Unified task categories (Head infers one of these)
TASK_CATEGORIES = [
    "depth",
    "distance",
    "relation",
    "existence",
    "count",
    "instance_location",
    "orientation",
    "size",
    "reach",
]

# CV-Bench 4 categories -> map to unified
CVBENCH_TO_UNIFIED = {
    "Count": "count",
    "Relation": "relation",
    "Depth": "depth",
    "Distance": "distance",
}

# 3DSRBench 12 categories -> map to unified
DSR3_TO_UNIFIED = {
    "location_above": "orientation",
    "height_higher": "orientation",
    "location_closer_to_camera": "distance",
    "multi_object_closer_to": "distance",
    "orientation_on_the_left": "orientation",
    "multi_object_facing": "orientation",
    "multi_object_same_direction": "orientation",
    "orientation_in_front_of": "orientation",
    "multi_object_viewpoint_towards_object": "orientation",
    "orientation_viewpoint": "orientation",
    "location_next_to": "relation",
    "multi_object_parallel": "orientation",
}

# Agent profiles: per-category performance (0-1, higher = better)
# From baseline: 3DSRBench overall, CV-Bench per-category
# Rank within each category used for initial selection
AGENT_PROFILES: Dict[str, Dict[str, float]] = {
    "llava4d": {
        "description": "LLaVA-4D: open-source generalist, strong on relation tasks.",
        "3dsrbench_overall": 0.311,
        "depth": 0.056,
        "relation": 0.145,
        "distance": 0.247,
        "count": 0.601,
        "orientation": 0.20,
        "existence": 0.35,
        "instance_location": 0.30,
        "size": 0.25,
        "reach": 0.28,
    },
    "qwen3_4b": {
        "description": "Qwen-3.0-VL 4B: open-source, strong on depth and count.",
        "3dsrbench_overall": 0.605,
        "depth": 0.957,
        "relation": 0.94,
        "distance": 0.852,
        "count": 0.652,
        "orientation": 0.85,
        "existence": 0.70,
        "instance_location": 0.75,
        "size": 0.80,
        "reach": 0.72,
    },
    "sa2va": {
        "description": "Sa2VA: spatial specialist, strong on depth.",
        "3dsrbench_overall": 0.224,
        "depth": 0.80,
        "relation": 0.077,
        "distance": 0.472,
        "count": 0.401,
        "orientation": 0.25,
        "existence": 0.40,
        "instance_location": 0.35,
        "size": 0.45,
        "reach": 0.38,
    },
    "claude_sonnet_4_5": {
        "description": "Claude 4.5 V Sonnet: proprietary, strong overall.",
        "3dsrbench_overall": 0.67,
        "depth": 0.75,
        "relation": 0.85,
        "distance": 0.78,
        "count": 0.82,
        "orientation": 0.80,
        "existence": 0.78,
        "instance_location": 0.76,
        "size": 0.74,
        "reach": 0.72,
    },
    "gpt4o": {
        "description": "GPT-4o: proprietary, strong on relation and count.",
        "3dsrbench_overall": 0.588,
        "depth": 0.70,
        "relation": 0.88,
        "distance": 0.72,
        "count": 0.85,
        "orientation": 0.75,
        "existence": 0.72,
        "instance_location": 0.74,
        "size": 0.70,
        "reach": 0.68,
    },
    "gemini_robotics_er": {
        "description": "Gemini Robotics-ER: embodied reasoning, 3D-aware.",
        "3dsrbench_overall": 0.429,
        "depth": 0.65,
        "relation": 0.72,
        "distance": 0.68,
        "count": 0.75,
        "orientation": 0.70,
        "existence": 0.68,
        "instance_location": 0.72,
        "size": 0.66,
        "reach": 0.70,
    },
    "internvl2": {
        "description": "InternVL2-8B: open multimodal model (replaces SpatialRGPT slot in ablations).",
        "3dsrbench_overall": 0.55,
        "depth": 0.82,
        "relation": 0.78,
        "distance": 0.74,
        "count": 0.70,
        "orientation": 0.76,
        "existence": 0.72,
        "instance_location": 0.71,
        "size": 0.70,
        "reach": 0.68,
    },
    "qwen2_vl": {
        "description": "Qwen2-VL-7B-Instruct: general VLM (replaces SpatialReasoner slot in ablations).",
        "3dsrbench_overall": 0.58,
        "depth": 0.88,
        "relation": 0.80,
        "distance": 0.78,
        "count": 0.68,
        "orientation": 0.78,
        "existence": 0.74,
        "instance_location": 0.73,
        "size": 0.72,
        "reach": 0.70,
    },
}

# Score update deltas
SCORE_DELTA_CORRECT = 0.05
SCORE_DELTA_WRONG = -0.02
INITIAL_WEIGHT = 0.5


def load_agent_profiles(profiles_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Load agent profiles from configs/mas/agent_profiles/*.json.
    Returns format compatible with AGENT_PROFILES (description + per-category floats).
    Falls back to AGENT_PROFILES if configs not found.
    """
    base = profiles_dir or (_CONFIGS_MAS / "agent_profiles")
    if not base.is_dir():
        return AGENT_PROFILES

    out = {}
    for path in base.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name", path.stem)
            # Build profile: description + unified_per_category (and cvbench/3dsrbench for Head prompt)
            profile = {
                "description": data.get("description", ""),
                "3dsrbench_overall": data.get("3dsrbench_overall", 0.5),
            }
            unified = data.get("unified_per_category", {})
            for cat in TASK_CATEGORIES:
                profile[cat] = unified.get(cat, INITIAL_WEIGHT)
            # Add cvbench categories for Head prompt
            cvb = data.get("cvbench_per_category", {})
            for k, v in cvb.items():
                c = k.lower().replace(" ", "_")
                if c not in profile:
                    profile[c] = v
            out[name] = profile
        except Exception:
            continue
    if out:
        return out
    return AGENT_PROFILES
