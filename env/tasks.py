from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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


TASKS: Dict[str, TaskConfig] = {
    "easy_support_optimization": TaskConfig(
        task_id="easy_support_optimization",
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
    ),
    "medium_pricing_strategy": TaskConfig(
        task_id="medium_pricing_strategy",
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
    ),
    "hard_startup_survival": TaskConfig(
        task_id="hard_startup_survival",
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
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_id '{task_id}'. Valid task ids: {valid}")
    return TASKS[task_id]
