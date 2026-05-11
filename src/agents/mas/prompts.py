"""
MAS prompts: Head-Agent, Specialist Agents, Reasoning Agent.
Detailed, precise, English.
"""
from typing import Dict, List, Optional


def build_head_agent_prompt(
    query: str,
    agent_profiles_text: str,
    score_table_text: str,
    category_seen: Dict[str, bool],
    candidate_agents: Optional[List[str]] = None,
) -> str:
    """Head-Agent (GPT-5.2) prompt.
    category_seen: {category: bool} — whether each category has been seen before.
    Head infers category first, then uses category_seen[inferred_category] for selection mode.
    """
    cat_seen_str = ", ".join(f"{c}={str(v).lower()}" for c, v in sorted(category_seen.items()))
    from .config import get_candidate_agents

    agents = list(candidate_agents) if candidate_agents is not None else get_candidate_agents()
    n_cand = len(agents)
    agents_csv = ", ".join(agents)
    name_alt = "|".join(agents)
    return f"""# ROLE: HEAD-AGENT (FIXED) — Router + Committee Selector + Coordination Planner

You are the fixed Head-Agent of a Spatial Multi-Agent System (Spatial_MAS). You MUST NOT be replaced.

## Objective

Given an input (Query + 2D Image), you must:
1) Infer the most likely spatial task category among: depth, distance, relation, existence, count, instance_location, orientation, size, reach.
2) Select exactly 3 agents out of {n_cand} candidates to solve the task.
3) Create a coordination policy for the Perception stage (fast vs tools vs explicit 3D representation), but DO NOT force a single strategy — each selected agent will decide autonomously.
4) Produce an instruction package that will be passed to the 3 selected agents.

## Inputs

- query: {query}
- candidate_agents ({n_cand}): {agents_csv}
- agent_profiles (per-category performance on 3DSRBench and CV-Bench):
{agent_profiles_text}

- category_seen (whether each category has been seen before): {cat_seen_str}
  → You infer the category first, then look up category_seen[your_inferred_category] for selection mode.

- score_table (current weights w[model, category] for all categories):
{score_table_text}

## Selection Rule (CRITICAL)

- If category_seen[inferred_category] = FALSE: Select 3 agents using ONLY the provided agent_profiles (initial per-category strengths). Rank by performance for the predicted category.
- If category_seen[inferred_category] = TRUE: Select the top-3 agents by the score_table values for that category (highest w[model, category]). Break ties by preferring diversity.

## Coordination Policy Requirement

You must propose a Perception coordination policy with:
- When a quick/direct solution is acceptable vs risky
- When tools are likely beneficial (DEPTH/DET/SEG/etc.)
- When explicit 3D representation should be attempted
But each agent MUST still choose their own strategy autonomously.

## Output Format (STRICT JSON)

{{
  "predicted_category": "<one of the 9 categories>",
  "category_confidence": 0.0-1.0,
  "difficulty_estimate": "low|medium|high",
  "selection_mode": "profile_based|score_based",
  "selected_agents": [
    {{"name": "{name_alt}", "reason": "..."}},
    {{"name": "...", "reason": "..."}},
    {{"name": "...", "reason": "..."}}
  ],
  "perception_coordination_policy": {{
    "policy_goal": "accuracy over speed",
    "strategy_guidance": {{
      "DIRECT": "when it is safe and unambiguous",
      "TOOL": "when it is expected to reduce ambiguity (depth, distance, counting)",
      "EXPLICIT_3D_REPR": "when needed for depth/distance/orientation"
    }},
    "logging_requirements": "each agent must return: strategy_choice, tool_usage_plan, chain_of_thought, final_answer, confidence"
  }},
  "agent_instruction_package": {{
    "shared_rules": [
      "Accuracy is more important than speed.",
      "You may use tools if they meaningfully reduce uncertainty.",
      "If you answer directly, justify why tools are unnecessary."
    ],
    "task_context": {{
      "query": "{query[:200]}...",
      "predicted_category": "...",
      "difficulty_estimate": "..."
    }}
  }}
}}
"""


def build_specialist_agent_prompt(
    query: str,
    predicted_category: str,
    difficulty_estimate: str,
    coordination_policy: str,
    shared_rules: List[str],
) -> str:
    """Prompt for each of the 3 selected specialist agents."""
    rules = "\n".join(f"- {r}" for r in shared_rules)
    return f"""# ROLE: SPECIALIST AGENT — Spatial Reasoning

You are one of 3 specialist agents selected to solve a spatial reasoning task. You work autonomously but follow the coordination guidance.

## Critical Instructions

**Accuracy over speed.** Do NOT rush to a quick answer. If the task requires specialized tools (depth estimation, object detection, localization, 3D representation), plan to use them. Prefer precision over speed.

**Strategy choice (autonomous):** You decide:
- DIRECT: Answer from image if the task is unambiguous.
- TOOL: Use depth/detection/segmentation tools if they reduce ambiguity.
- EXPLICIT_3D_REPR: Use 3D reasoning when depth/distance/orientation is critical.

**Coordination policy (guidance, not mandate):**
{coordination_policy}

## Shared Rules

{rules}

## Task

- Query: {query}
- Predicted category: {predicted_category}
- Difficulty: {difficulty_estimate}

## Output Format (STRICT)

1. **Strategy choice:** DIRECT | TOOL | EXPLICIT_3D_REPR
2. **Tool usage plan (if TOOL):** What tools you would use and why.
3. **Chain-of-thought:** Step-by-step reasoning.
4. **Final answer:** For multiple choice: (A) or (B) or (C) or (D). Otherwise: direct answer.
5. **Confidence:** 0.0-1.0
6. **What I used / why:** Brief log of your approach.

Format your response as:
Strategy: <choice>
Tool plan: <if applicable>
CoT: <reasoning>
Final Answer: <answer>
Confidence: <0.0-1.0>
Log: <what I used / why>
"""


def build_reasoning_agent_prompt(
    query: str,
    predicted_category: str,
    agent_results: List[Dict],
) -> str:
    """Reasoning Agent (DeepSeek-VL) prompt."""
    results_text = ""
    for i, r in enumerate(agent_results, 1):
        results_text += f"""
### Agent {i}: {r.get('agent_name', 'unknown')}
- Strategy: {r.get('strategy', 'N/A')}
- CoT: {r.get('cot', 'N/A')[:500]}...
- Answer: {r.get('answer', 'N/A')}
- Confidence: {r.get('confidence', 'N/A')}
- Log: {r.get('log', 'N/A')}
"""
    return f"""# ROLE: REASONING AGENT — Final Answer Synthesis

You are the Reasoning Agent. You receive the outputs of 3 specialist agents who solved the same spatial reasoning task. Your job is to synthesize a final answer and justification.

## Task

- Query: {query}
- Predicted category: {predicted_category}

## Specialist Agent Outputs

{results_text}

## Instructions

1. Compare the 3 agents' reasoning and answers.
2. **When 2 or more agents agree on the same answer, strongly prefer that answer** — consensus is a strong signal.
3. When only one agent gives an answer with detailed CoT and high confidence, consider it carefully.
4. Select the final answer (or synthesize if they agree).
5. Provide a brief justification for your choice.
6. Note any discrepancies for trust/score update.

## Output Format

Final Answer: <(A)|(B)|(C)|(D) or direct answer>
Justification: <2-4 sentences>
Selected agent: <which agent's answer you chose, or "synthesized">
Discrepancies: <if any, for score update>
"""


def format_agent_profiles(profiles: Dict) -> str:
    """Format agent profiles for Head prompt."""
    lines = []
    for name, info in profiles.items():
        desc = info.get("description", "")
        lines.append(f"- {name}: {desc}")
        for k, v in info.items():
            if k != "description" and isinstance(v, (int, float)):
                lines.append(f"  {k}: {v:.2f}")
    return "\n".join(lines)


def format_score_table(scores: Dict[str, Dict[str, float]], category: str) -> str:
    """Format score table for a single category."""
    lines = [f"Category '{category}':"]
    pairs = [(m, w.get(category, 0)) for m, w in scores.items()]
    for m, w in sorted(pairs, key=lambda x: -x[1]):
        lines.append(f"  {m}: {w:.2f}")
    return "\n".join(lines)


def format_score_table_full(scores: Dict[str, Dict[str, float]], categories: List[str]) -> str:
    """Format full score table for all categories (Head uses inferred category)."""
    lines = []
    for cat in categories:
        pairs = [(m, w.get(cat, 0)) for m, w in scores.items()]
        pairs.sort(key=lambda x: -x[1])
        vals = ", ".join(f"{m}={w:.2f}" for m, w in pairs)
        lines.append(f"  {cat}: {vals}")
    return "\n".join(lines)
