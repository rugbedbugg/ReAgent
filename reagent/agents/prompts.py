"""System prompts for the specialist evaluator agents.

Each agent judges one candidate route along one objective, reading only the
precomputed facts it is handed. The prompts share a common contract and add a
short objective-specific brief.
"""

from __future__ import annotations

import json

COMMON = """You are a specialist reviewer on a retrosynthesis planning team. You \
judge ONE candidate route along ONE objective, using ONLY the precomputed facts \
given to you.

Rules:
- Do not compute chemistry yourself; the facts are your only ground truth.
- Explain your score briefly, citing the fact that drove it.
- Output ONLY a JSON object: {"score": <float 0..1>, "rationale": "<short>"}. \
Higher score = better along this objective."""

OBJECTIVE_BRIEFS: dict[str, str] = {
    "feasibility": (
        "Objective: how likely each disconnection is to work.\n"
        "Decisive fact: min_policy_probability (weakest step's model confidence). "
        "Low probability means a weak route."
    ),
    "availability": (
        "Objective: are the leaf building blocks purchasable.\n"
        "Decisive fact: in_stock_fraction and not_in_stock. All in stock is best."
    ),
    "cost": (
        "Objective: procurement/synthesis cost (cost_proxy is a stand-in, lower is "
        "cheaper).\n"
        "Decisive fact: cost_proxy."
    ),
    "safety": (
        "Objective: hazardous/reactive functional groups in the route.\n"
        "Decisive fact: hazard_count and distinct_hazards. Fewer is safer."
    ),
    "sustainability": (
        "Objective: green-chemistry quality (proxies, no solvent data).\n"
        "Decisive fact: mean_step_atom_economy (higher better) and pmi_proxy "
        "(lower better)."
    ),
    "efficiency": (
        "Objective: route shape.\n"
        "Decisive fact: num_steps and convergence (>1.0 means convergent)."
    ),
}


def system_prompt(objective: str) -> str:
    return f"{COMMON}\n\n{OBJECTIVE_BRIEFS[objective]}"


def user_prompt(objective: str, facts: dict, target: str, num_steps: int) -> str:
    return (
        f"Target molecule: {target}\n"
        f"Route length: {num_steps} step(s)\n"
        f"Objective under review: {objective}\n"
        f"Precomputed facts:\n{json.dumps(facts, indent=2)}"
    )
