"""Deterministic heuristic policies for baselines, demos, and CI (no LLM required)."""

from __future__ import annotations

from typing import Any, Dict, List

RUBRIC_HINTS: Dict[str, str] = {
    "easy_support_optimization": (
        "Maximize sentiment_score improvement and churn reduction; use respond_to_feedback only when "
        "support_cooldown_steps==0; mix actions to avoid loop penalties."
    ),
    "medium_pricing_strategy": (
        "Balance cash, revenue, and churn—the grader uses a Pareto-style blend; avoid collapse and one-axis gaming."
    ),
    "hard_startup_survival": (
        "Survive to max_steps with users>5 and cash>0; protect runway_opex_steps; grow users and cut churn without reckless spend."
    ),
}


def heuristic_baseline_policy(
    task_id: str, observation: Dict[str, Any], action_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Strong deterministic policy aligned with observation signals and grader structure.
    Used when no API key is available and as an oracle fallback after LLM failures.
    """
    churn = observation["churn_rate"]
    cash = observation["cash"]
    competitor_price = observation["competitor_price"]
    sentiment = observation.get("sentiment_score", 0.0)
    support_cd = observation.get("support_cooldown_steps", 0)
    runway = observation.get("runway_opex_steps", cash / 600.0)
    feedback = " ".join(observation["customer_feedback"]).lower()
    requests = observation["feature_requests"]
    step = observation["step_count"]
    used_actions = [a.get("action_type", "") for a in action_history[-2:]]
    last_action = used_actions[-1] if used_actions else ""

    def avoid_repeat(candidate: Dict[str, Any]) -> Dict[str, Any]:
        if len(used_actions) >= 2 and used_actions[0] == used_actions[1] == candidate["action_type"]:
            return {"action_type": "adjust_price", "value": max(22.0, min(80.0, competitor_price - 1.0))}
        return candidate

    def can_respond() -> bool:
        return support_cd <= 0 and cash > 1100

    if task_id == "easy_support_optimization":
        crisis = churn > 0.14 or sentiment < -0.05 or any(
            w in feedback for w in ("too long", "unhappy", "bugs", "risk")
        )
        if crisis and can_respond() and last_action != "respond_to_feedback":
            return {"action_type": "respond_to_feedback", "payload": "triage_top_tickets"}
        if requests and cash > 5200 and churn > 0.09 and last_action != "add_feature":
            return {"action_type": "add_feature", "payload": requests[0]}
        if churn > 0.115 and cash > 2000 and last_action != "run_marketing":
            return {"action_type": "run_marketing", "value": 850.0}
        return avoid_repeat({"action_type": "adjust_price", "value": max(24.0, min(80.0, competitor_price - 0.8))})

    if task_id == "medium_pricing_strategy":
        if (churn > 0.15 or sentiment < -0.1) and can_respond() and last_action != "respond_to_feedback":
            return {"action_type": "respond_to_feedback", "payload": "retention_campaign"}
        if requests and churn > 0.10 and cash > 5600 and step % 3 == 1 and last_action != "add_feature":
            return {"action_type": "add_feature", "payload": requests[0]}
        if cash > 5200 and step % 4 == 0 and last_action != "run_marketing":
            return {"action_type": "run_marketing", "value": 1150.0}
        target = competitor_price - (1.2 if churn > 0.105 else 0.55)
        return avoid_repeat({"action_type": "adjust_price", "value": max(26.0, min(92.0, target))})

    if task_id == "hard_startup_survival":
        if (churn > 0.155 or sentiment < -0.12) and can_respond():
            return {"action_type": "respond_to_feedback", "payload": "critical_support"}
        if requests and cash > 5500 and churn > 0.12 and runway > 4.0 and step <= 7 and last_action != "add_feature":
            return {"action_type": "add_feature", "payload": requests[0]}
        if cash > 7500 and runway > 6.0 and step % 5 == 2 and last_action != "run_marketing":
            return {"action_type": "run_marketing", "value": 750.0}
        return avoid_repeat({"action_type": "adjust_price", "value": max(23.0, min(72.0, competitor_price - 0.5))})

    return avoid_repeat({"action_type": "adjust_price", "value": max(25.0, min(80.0, competitor_price - 0.5))})


def noop_policy(_task_id: str, _observation: Dict[str, Any], _history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deliberately weak baseline for score spread demos."""
    return {"action_type": "do_nothing"}
