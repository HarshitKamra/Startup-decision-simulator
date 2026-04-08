from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    seed: int
    initial_cash: float
    initial_users: int
    initial_price: float
    competitor_price: float
    initial_churn_rate: float
    scenario_name: str = "saas_core"
    shock_schedule: List[Dict[str, Any]] = field(default_factory=list)


TASKS: Dict[str, TaskConfig] = {
    "support_turbulence": TaskConfig(
        task_id="support_turbulence",
        name="Customer Support Optimization",
        difficulty="easy",
        description=(
            "Improve customer sentiment by responding to user feedback and "
            "reducing frustration signals."
        ),
        max_steps=8,
        seed=11,
        initial_cash=18000.0,
        initial_users=90,
        initial_price=39.0,
        competitor_price=41.0,
        initial_churn_rate=0.15,
        scenario_name="support_turbulence",
        shock_schedule=[
            {"step": 2, "kind": "support_backlog", "severity": 0.8},
            {"step": 5, "kind": "word_of_mouth_bump", "severity": 0.6},
        ],
    ),
    "pricing_pressure": TaskConfig(
        task_id="pricing_pressure",
        name="Pricing Strategy",
        difficulty="medium",
        description=(
            "Balance pricing, churn, and revenue to improve profitability "
            "without losing too many users."
        ),
        max_steps=10,
        seed=17,
        initial_cash=24000.0,
        initial_users=120,
        initial_price=49.0,
        competitor_price=45.0,
        initial_churn_rate=0.13,
        scenario_name="pricing_pressure",
        shock_schedule=[
            {"step": 3, "kind": "competitor_discount", "severity": 0.9},
            {"step": 7, "kind": "cost_spike", "severity": 0.7},
        ],
    ),
    "runway_crisis": TaskConfig(
        task_id="runway_crisis",
        name="Startup Survival Scenario",
        difficulty="hard",
        description=(
            "Survive under tight cash constraints while trying to grow users "
            "and keep churn under control."
        ),
        max_steps=14,
        seed=23,
        initial_cash=9000.0,
        initial_users=75,
        initial_price=44.0,
        competitor_price=42.0,
        initial_churn_rate=0.18,
        scenario_name="runway_crisis",
        shock_schedule=[
            {"step": 2, "kind": "incident_outage", "severity": 0.7},
            {"step": 8, "kind": "investor_delay", "severity": 0.85},
            {"step": 11, "kind": "competitor_discount", "severity": 0.6},
        ],
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_id '{task_id}'. Valid task ids: {valid}")
    return TASKS[task_id]
