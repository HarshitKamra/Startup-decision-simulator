from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from env.environment import StartupDecisionEnv
from env.models import Action
from env.tasks import get_task_config


def clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean_invalid_penalty(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    return sum(step["reward"]["invalid_or_bad_actions"] for step in trajectory) / len(trajectory)


def _terminal_failure(state: Dict[str, Any]) -> bool:
    return state["cash"] <= 0 or state["users"] <= 5


def run_episode(task_id: str, actions: Iterable[Dict[str, Any] | Action]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cfg = get_task_config(task_id)
    env = StartupDecisionEnv(
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        initial_cash=cfg.initial_cash,
        initial_users=cfg.initial_users,
        initial_price=cfg.initial_price,
        competitor_price=cfg.competitor_price,
        initial_churn_rate=cfg.initial_churn_rate,
        scenario_name=cfg.scenario_name,
        shock_schedule=cfg.shock_schedule,
    )

    obs = env.reset()
    first_sentiment = env._sentiment_score(obs.customer_feedback)
    trajectory: List[Dict[str, Any]] = []

    for action in actions:
        model_action = action if isinstance(action, Action) else Action(**action)
        obs, reward, done, info = env.step(model_action)
        trajectory.append(
            {
                "observation": obs.model_dump(),
                "reward": reward.model_dump(),
                "done": done,
                "info": info,
                "action": model_action.model_dump(),
            }
        )
        if done:
            break

    final_state = env.state()
    final_state["initial_sentiment"] = first_sentiment
    final_state["final_sentiment"] = env._sentiment_score()
    final_state["steps_taken"] = len(trajectory)
    final_state["sum_reward"] = sum(step["reward"]["total"] for step in trajectory)
    return final_state, trajectory


def grade_easy(actions: Iterable[Dict[str, Any] | Action]) -> float:
    """Support + sentiment + churn; penalize collapse, sloppy execution, support-only spam."""
    state, trajectory = run_episode("easy_support_optimization", actions)
    sentiment_delta = state["final_sentiment"] - state["initial_sentiment"]
    churn_component = clamp_01((0.26 - state["churn_rate"]) / 0.26)
    sentiment_component = clamp_01((sentiment_delta + 1.0) / 2.0)

    respond_share = 0.0
    if trajectory:
        respond_share = sum(1 for t in trajectory if t["action"].get("action_type") == "respond_to_feedback") / len(
            trajectory
        )
    # Slight bonus for balanced support usage (not never-respond, not only-respond).
    diversity = 0.06 if 0.12 <= respond_share <= 0.5 else 0.0

    score = 0.67 * sentiment_component + 0.29 * churn_component + diversity

    if _terminal_failure(state):
        score *= 0.42
    score -= 0.13 * clamp_01(_mean_invalid_penalty(trajectory))
    if sentiment_delta <= 0 and not _terminal_failure(state):
        score *= 0.85

    return round(clamp_01(score), 6)


def grade_medium(actions: Iterable[Dict[str, Any] | Action]) -> float:
    """Pareto-style: geometric blend rewards balanced profit, revenue, and retention."""
    cfg = get_task_config("medium_pricing_strategy")
    state, trajectory = run_episode("medium_pricing_strategy", actions)
    base_revenue = cfg.initial_users * cfg.initial_price
    profit_ratio = (state["cash"] - cfg.initial_cash) / max(1.0, cfg.initial_cash)
    revenue_ratio = (state["revenue"] - base_revenue) / max(1.0, base_revenue)
    churn_reduction = (cfg.initial_churn_rate - state["churn_rate"]) / max(0.01, cfg.initial_churn_rate)

    a = clamp_01((profit_ratio + 1.0) / 2.0)
    b = clamp_01((revenue_ratio + 1.0) / 2.0)
    c = clamp_01((churn_reduction + 1.0) / 2.0)
    eps = 0.06
    balanced = ((a + eps) * (b + eps) * (c + eps)) ** (1.0 / 3.0) - eps
    balanced = clamp_01(balanced)

    # Mild preference for simultaneous upside (not one-axis gaming).
    spread_penalty = abs(a - b) + abs(b - c) + abs(a - c)
    harmony = clamp_01(1.0 - spread_penalty / 3.0)

    score = 0.82 * balanced + 0.18 * harmony

    if _terminal_failure(state):
        score *= 0.52
    score -= 0.11 * clamp_01(_mean_invalid_penalty(trajectory))
    if state["churn_rate"] > cfg.initial_churn_rate + 0.02:
        score *= 0.88

    return round(clamp_01(score), 6)


def grade_hard(actions: Iterable[Dict[str, Any] | Action]) -> float:
    """Survival with partial credit; growth and churn must coexist with runway discipline."""
    cfg = get_task_config("hard_startup_survival")
    state, trajectory = run_episode("hard_startup_survival", actions)
    steps_ratio = state["steps_taken"] / max(1, cfg.max_steps)
    alive = state["cash"] > 0 and state["users"] > 5

    if state["steps_taken"] >= cfg.max_steps and alive:
        survival = 1.0
    elif alive:
        survival = 0.58 * (steps_ratio**1.25)
    else:
        survival = 0.0

    user_growth = (state["users"] - cfg.initial_users) / max(1, cfg.initial_users)
    churn_reduction = (cfg.initial_churn_rate - state["churn_rate"]) / max(0.01, cfg.initial_churn_rate)
    cash_efficiency = (state["cash"] - cfg.initial_cash) / max(1.0, cfg.initial_cash)

    runway_bonus = 0.0
    if alive and state["cash"] >= cfg.initial_cash * 0.85:
        runway_bonus = clamp_01((state["cash"] - cfg.initial_cash * 0.85) / max(1.0, cfg.initial_cash * 0.5)) * 0.06

    score = (
        0.48 * survival
        + 0.2 * clamp_01((user_growth + 1.0) / 2.0)
        + 0.2 * clamp_01((churn_reduction + 1.0) / 2.0)
        + 0.12 * clamp_01((cash_efficiency + 1.0) / 2.0)
        + runway_bonus
    )
    score -= 0.09 * clamp_01(_mean_invalid_penalty(trajectory))
    if not alive:
        score *= 0.72

    return round(clamp_01(score), 6)


def grade_task(task_id: str, actions: Iterable[Dict[str, Any] | Action]) -> float:
    if task_id == "easy_support_optimization":
        return grade_easy(actions)
    if task_id == "medium_pricing_strategy":
        return grade_medium(actions)
    if task_id == "hard_startup_survival":
        return grade_hard(actions)
    raise ValueError(f"Unknown task_id '{task_id}'")
