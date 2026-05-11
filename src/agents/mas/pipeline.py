"""
Spatial MAS Pipeline: Head → 3 Specialists → Reasoning → Score Update.
"""
import json
import re
from typing import Callable, Dict, List, Optional

from PIL import Image

from .config import AGENT_PROFILES, TASK_CATEGORIES, get_candidate_agents, load_agent_profiles
from .prompts import (
    build_head_agent_prompt,
    build_specialist_agent_prompt,
    build_reasoning_agent_prompt,
    format_agent_profiles,
    format_score_table_full,
)
from .score_manager import ScoreManager


def _parse_head_output(text: str) -> Optional[Dict]:
    """Parse Head-Agent JSON output."""
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _parse_specialist_output(text: str) -> Dict:
    """Parse specialist agent output."""
    text = (text or "").strip()
    out = {"strategy": "", "cot": "", "answer": "", "confidence": "", "log": ""}
    for k in out:
        m = re.search(rf"{k}\s*[:\=]\s*(.+?)(?=\n\w|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m:
            out[k] = m.group(1).strip()[:1000]
    # Extract (A) or (B) etc.
    ans_m = re.search(r"\(([A-D])\)|Final Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if ans_m:
        out["answer"] = f"({ans_m.group(1) or ans_m.group(2)})"
    return out


def _parse_reasoning_output(text: str) -> Dict:
    """Parse Reasoning Agent output."""
    text = (text or "").strip()
    out = {"final_answer": "", "justification": ""}
    m = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    if m:
        out["final_answer"] = m.group(1).strip()
    m = re.search(r"Justification:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        out["justification"] = m.group(1).strip()[:500]
    return out


def run_spatial_mas_pipeline(
    image: Image.Image,
    query: str,
    gt_answer: Optional[str] = None,
    head_generate: Callable[[Image.Image, str], str] = None,
    specialist_generate: Callable[[str, Image.Image, str], str] = None,
    reasoning_generate: Callable[[Image.Image, str], str] = None,
    score_manager: Optional[ScoreManager] = None,
    category_seen: Optional[Dict[str, bool]] = None,
) -> Dict:
    """
    Run full Spatial MAS pipeline.

    Args:
        image: Input image
        query: Question
        gt_answer: Ground truth (for score update)
        head_generate: fn(image, prompt) -> str
        specialist_generate: fn(model_name, image, prompt) -> str
        reasoning_generate: fn(image, prompt) -> str
        score_manager: ScoreManager for weight updates
        category_seen: {category: bool} — whether each category was seen before

    Returns:
        {
            "predicted_category": str,
            "selected_agents": [str],
            "agent_results": [...],
            "final_answer": str,
            "scores_updated": bool,
        }
    """
    score_manager = score_manager or ScoreManager()
    category_seen = category_seen or {c: False for c in TASK_CATEGORIES}
    for c in TASK_CATEGORIES:
        if c not in category_seen:
            category_seen[c] = False

    # 1. Head-Agent
    candidates = get_candidate_agents()
    agent_profiles = load_agent_profiles()
    profiles_filtered = {k: v for k, v in agent_profiles.items() if k in candidates}
    for k in candidates:
        if k not in profiles_filtered:
            profiles_filtered[k] = agent_profiles.get(k) or AGENT_PROFILES.get(k, {"description": k})
    profiles_text = format_agent_profiles(profiles_filtered)
    score_text = format_score_table_full(score_manager.to_dict(), TASK_CATEGORIES)

    head_prompt = build_head_agent_prompt(
        query=query,
        agent_profiles_text=profiles_text,
        score_table_text=score_text,
        category_seen=category_seen,
        candidate_agents=candidates,
    )
    head_output = head_generate(image, head_prompt) if head_generate else ""
    head_data = _parse_head_output(head_output)

    if not head_data:
        return {"error": "Head-Agent parse failed", "head_output": head_output}

    pred_cat = head_data.get("predicted_category", "relation")

    sel_list = head_data.get("selected_agents", [])
    if sel_list and isinstance(sel_list[0], dict):
        selected = [s.get("name", "") for s in sel_list if s.get("name")]
    else:
        selected = list(sel_list) if isinstance(sel_list, list) else []
    selected = [s for s in selected if s in candidates][:3]
    if len(selected) < 3:
        selected = (selected + [a for a in candidates if a not in selected])[:3]

    policy = head_data.get("perception_coordination_policy", {})
    policy_str = json.dumps(policy, indent=2) if isinstance(policy, dict) else str(policy)
    pkg = head_data.get("agent_instruction_package") or {}
    shared_rules = pkg.get("shared_rules", [
        "Accuracy is more important than speed.",
        "You may use tools if they meaningfully reduce uncertainty.",
    ])

    # 2. Specialist Agents
    specialist_prompt = build_specialist_agent_prompt(
        query=query,
        predicted_category=pred_cat,
        difficulty_estimate=head_data.get("difficulty_estimate", "medium"),
        coordination_policy=policy_str,
        shared_rules=shared_rules,
    )

    agent_results = []
    for agent_name in selected:
        out = specialist_generate(agent_name, image, specialist_prompt) if specialist_generate else ""
        parsed = _parse_specialist_output(out)
        parsed["agent_name"] = agent_name
        parsed["raw"] = out[:500]
        agent_results.append(parsed)

    # 3. Reasoning Agent
    reasoning_prompt = build_reasoning_agent_prompt(
        query=query,
        predicted_category=pred_cat,
        agent_results=agent_results,
    )
    reasoning_output = reasoning_generate(image, reasoning_prompt) if reasoning_generate else ""
    reasoning_parsed = _parse_reasoning_output(reasoning_output)
    final_answer = reasoning_parsed.get("final_answer", "")

    # 4. Score Update (per-agent: compare each agent's answer to gt, not final answer)
    def _norm(s: str) -> str:
        s = (s or "").strip().upper()
        for c in "ABCD":
            if c in s or f"({c})" in s:
                return f"({c})"
        return s

    if gt_answer and gt_answer.strip():
        gt_norm = _norm(gt_answer)
        for r in agent_results:
            agent_name = r.get("agent_name", "")
            ans = r.get("answer", "")
            correct = _norm(ans) == gt_norm
            score_manager.update(agent_name, pred_cat, correct)

    return {
        "predicted_category": pred_cat,
        "selected_agents": selected,
        "agent_results": agent_results,
        "final_answer": final_answer,
        "reasoning_justification": reasoning_parsed.get("justification", ""),
        "scores_updated": bool(gt_answer),
        "score_manager": score_manager,
    }
