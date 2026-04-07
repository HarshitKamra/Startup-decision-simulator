from __future__ import annotations

import copy
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from env.models import Action, Observation, Reward


DEFAULT_FEATURE_POOL = [
    "mobile_app",
    "analytics_dashboard",
    "slack_integration",
    "sso_login",
    "workflow_automation",
    "ai_assistant",
]

POSITIVE_FEEDBACK = [
    "Support response was quick and helpful.",
    "The product feels easier to use now.",
    "New workflow reduced manual effort.",
    "Pricing seems fair for the value.",
    "Great experience with onboarding.",
]

NEGATIVE_FEEDBACK = [
    "Too many bugs in key workflows.",
    "Price feels too high for small teams.",
    "Support response took too long.",
    "Missing important integrations.",
    "Churn risk: users are unhappy with reliability.",
]


class StartupDecisionEnv:
    """
    OpenEnv benchmark environment for startup strategy simulation.

    API surface:
    - reset() -> Observation
    - step(action: Action | dict) -> tuple[Observation, Reward, bool, dict]
    - state() -> dict[str, Any]
    """

    def __init__(
        self,
        seed: int = 7,
        max_steps: int = 12,
        initial_cash: float = 25000.0,
        initial_users: int = 120,
        initial_price: float = 49.0,
        competitor_price: float = 45.0,
        initial_churn_rate: float = 0.12,
        noise_scale: float = 0.04,
        scenario_name: str = "saas_core",
        shock_schedule: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.initial_cash = initial_cash
        self.initial_users = initial_users
        self.initial_price = initial_price
        self.initial_competitor_price = competitor_price
        self.initial_churn_rate = initial_churn_rate
        self.noise_scale = noise_scale
        self.scenario_name = scenario_name
        self.shock_schedule = list(shock_schedule or [])
        self._rng = random.Random(seed)
        self._delayed_effects: Deque[Tuple[int, Dict[str, float]]] = deque()
        self._feedback_skip_until_step: int = 0
        self._reset_internal_state()

    def _reset_internal_state(self) -> None:
        self.step_count = 0
        self.cash = float(self.initial_cash)
        self.users = int(self.initial_users)
        self.churn_rate = float(self.initial_churn_rate)
        self.price = float(self.initial_price)
        self.competitor_price = float(self.initial_competitor_price)
        self.revenue = self.users * self.price
        self.implemented_features: List[str] = []
        self.customer_feedback = self._sample_feedback(size=3)
        self.feature_requests = self._sample_feature_requests(size=3)
        self._delayed_effects.clear()
        self._feedback_skip_until_step = 0
        self._last_users = self.users
        self._last_revenue = self.revenue
        self._last_action_type = ""
        self._action_streak = 0
        self._marketing_streak = 0
        self._consecutive_support_response = 0
        self.current_alerts: List[str] = []
        self._shocks_by_step: Dict[int, List[Dict[str, Any]]] = {}
        for event in self.shock_schedule:
            step = int(event.get("step", -1))
            if step >= 0:
                self._shocks_by_step.setdefault(step, []).append(event)

    def _sample_feedback(self, size: int = 2) -> List[str]:
        feedback: List[str] = []
        for _ in range(size):
            bucket = NEGATIVE_FEEDBACK if self._rng.random() < 0.55 else POSITIVE_FEEDBACK
            feedback.append(self._rng.choice(bucket))
        return feedback

    def _sample_feature_requests(self, size: int = 2) -> List[str]:
        req_pool = [f for f in DEFAULT_FEATURE_POOL if f not in self.implemented_features]
        self._rng.shuffle(req_pool)
        return req_pool[: min(size, len(req_pool))]

    def _noise(self, factor: float = 1.0) -> float:
        return self._rng.uniform(-self.noise_scale, self.noise_scale) * factor

    def _sentiment_score(self, feedback: Optional[List[str]] = None) -> float:
        feedback_items = feedback if feedback is not None else self.customer_feedback
        if not feedback_items:
            return 0.0
        score = 0.0
        for line in feedback_items:
            lower = line.lower()
            if any(k in lower for k in ("quick", "helpful", "great", "easier", "fair", "reduced")):
                score += 1.0
            if any(k in lower for k in ("bugs", "high", "too long", "missing", "unhappy", "risk")):
                score -= 1.0
        return max(-1.0, min(1.0, score / max(1, len(feedback_items))))

    def _edge_penalty_crisis_idle(self) -> float:
        if self.churn_rate <= 0.16:
            return 0.0
        if self._sentiment_score() >= -0.15:
            return 0.0
        return 0.35

    def _apply_delayed_effects(self) -> None:
        retained: Deque[Tuple[int, Dict[str, float]]] = deque()
        while self._delayed_effects:
            remaining, effect = self._delayed_effects.popleft()
            if remaining <= 1:
                self.churn_rate = max(0.01, min(0.6, self.churn_rate + effect.get("churn_delta", 0.0)))
                user_gain = int(effect.get("user_gain", 0))
                self.users = max(0, self.users + user_gain)
            else:
                retained.append((remaining - 1, effect))
        self._delayed_effects = retained

    def reset(self) -> Observation:
        self._rng.seed(self.seed)
        self._reset_internal_state()
        return self._build_observation()

    def state(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "users": self.users,
            "churn_rate": self.churn_rate,
            "revenue": self.revenue,
            "price": self.price,
            "competitor_price": self.competitor_price,
            "customer_feedback": copy.deepcopy(self.customer_feedback),
            "feature_requests": copy.deepcopy(self.feature_requests),
            "implemented_features": copy.deepcopy(self.implemented_features),
            "scenario_name": self.scenario_name,
            "current_alerts": copy.deepcopy(self.current_alerts),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

    def _build_observation(self) -> Observation:
        steps_rem = max(0, self.max_steps - self.step_count)
        sup_cd = max(0, self._feedback_skip_until_step - self.step_count)
        opex = 600.0
        return Observation(
            cash=round(self.cash, 2),
            users=self.users,
            churn_rate=round(self.churn_rate, 4),
            revenue=round(self.revenue, 2),
            price=round(self.price, 2),
            customer_feedback=copy.deepcopy(self.customer_feedback),
            competitor_price=round(self.competitor_price, 2),
            feature_requests=copy.deepcopy(self.feature_requests),
            current_alerts=copy.deepcopy(self.current_alerts),
            step_count=self.step_count,
            max_steps=self.max_steps,
            sentiment_score=round(self._sentiment_score(), 4),
            price_vs_competitor=round(self.price / max(1e-6, self.competitor_price), 4),
            runway_opex_steps=round(self.cash / max(1.0, opex), 4),
            steps_remaining=steps_rem,
            support_cooldown_steps=sup_cd,
            implemented_feature_count=len(self.implemented_features),
        )

    def _apply_scenario_shocks(self) -> List[str]:
        triggered: List[str] = []
        for event in self._shocks_by_step.get(self.step_count, []):
            kind = str(event.get("kind", "generic_shock"))
            severity = max(0.0, min(1.0, float(event.get("severity", 0.5))))
            if kind == "competitor_discount":
                self.competitor_price = max(15.0, self.competitor_price - (1.0 + 3.0 * severity))
                self.churn_rate = min(0.6, self.churn_rate + 0.01 + 0.02 * severity)
                triggered.append("competitor_discount")
            elif kind == "incident_outage":
                self.churn_rate = min(0.6, self.churn_rate + 0.02 + 0.03 * severity)
                self.users = max(0, self.users - int(self.users * (0.02 + 0.03 * severity)))
                triggered.append("incident_outage")
            elif kind == "support_backlog":
                self.churn_rate = min(0.6, self.churn_rate + 0.015 + 0.02 * severity)
                triggered.append("support_backlog")
            elif kind == "cost_spike":
                self.cash -= 350.0 + 500.0 * severity
                triggered.append("cost_spike")
            elif kind == "investor_delay":
                self.cash -= 500.0 + 900.0 * severity
                self.churn_rate = min(0.6, self.churn_rate + 0.005 + 0.015 * severity)
                triggered.append("investor_delay")
            elif kind == "word_of_mouth_bump":
                self.users += int(3 + 8 * severity)
                self.churn_rate = max(0.01, self.churn_rate - (0.006 + 0.01 * severity))
                triggered.append("word_of_mouth_bump")
        if triggered:
            self.current_alerts = triggered
        else:
            self.current_alerts = []
        return triggered

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        action_invalid_schema = False
        if isinstance(action, dict):
            try:
                action = Action(**action)
            except Exception:
                action_invalid_schema = True
        if action_invalid_schema:
            action = Action(action_type="do_nothing")  # placeholder; branch skipped below

        if self.step_count >= self.max_steps:
            reward_obj = Reward(
                total=0.0,
                user_growth=0.0,
                revenue_change=0.0,
                churn_penalty=0.0,
                sentiment_score=self._sentiment_score(),
                invalid_or_bad_actions=1.0,
            )
            return self._build_observation(), reward_obj, True, {"reason": "episode_already_done"}

        # Edge: already bankrupt or no users — no meaningful action (terminal).
        if self.cash <= 0:
            reward_obj = self._compute_reward(invalid_or_bad_actions=1.0)
            return self._build_observation(), reward_obj, True, {"reason": "already_bankrupt", "terminal_reason": "bankrupt"}
        if self.users <= 0:
            reward_obj = self._compute_reward(invalid_or_bad_actions=1.0)
            return self._build_observation(), reward_obj, True, {"reason": "no_users", "terminal_reason": "user_collapse"}

        self._last_users = self.users
        self._last_revenue = self.revenue
        pre_step_cash = self.cash
        invalid_or_bad = 0.0
        action_note = ""
        loop_penalty = 0.0
        edge_breakdown: Dict[str, float] = {}
        triggered_shocks = self._apply_scenario_shocks()

        self._apply_delayed_effects()

        # Baseline operating cost to force strategy trade-offs.
        self.cash -= 600.0
        cash_after_opex = self.cash

        if action_invalid_schema:
            invalid_or_bad += 1.0
            action_note = "pydantic_validation_failed"
            edge_breakdown["invalid_action_schema"] = 1.0
            self._marketing_streak = 0
            self._consecutive_support_response = 0

        elif action.action_type == "adjust_price":
            target_price = float(action.value or self.price)
            price_diff = target_price - self.price
            rel_jump = abs(price_diff) / max(1e-6, self.price)
            if rel_jump > 0.35:
                shock = min(0.45, 0.15 + rel_jump * 0.4)
                invalid_or_bad += shock
                edge_breakdown["price_shock"] = shock
                self.churn_rate = min(0.6, self.churn_rate + 0.02 + rel_jump * 0.05)
                action_note = "price_shock_large_move"
            unclamped = target_price
            self.price = max(10.0, min(200.0, target_price))
            if unclamped < 10.0 or unclamped > 200.0:
                clamp_pen = 0.2
                invalid_or_bad += clamp_pen
                edge_breakdown["price_clamped"] = clamp_pen
                if not action_note:
                    action_note = "price_clamped_to_bounds"
            if self.price > 0 and self.competitor_price > 0:
                if self.price < self.competitor_price * 0.72:
                    pred = 0.22
                    invalid_or_bad += pred
                    edge_breakdown["predatory_pricing"] = pred
                    self.churn_rate = min(0.6, self.churn_rate + 0.012)
                    if action_note in ("price_shock_large_move", "price_clamped_to_bounds"):
                        action_note = f"{action_note}_predatory_undercut"
                    else:
                        action_note = "predatory_undercut"
            if price_diff > 0:
                self.churn_rate = min(0.6, self.churn_rate + 0.01 + abs(price_diff) / 500.0)
            else:
                self.churn_rate = max(0.01, self.churn_rate - 0.012 - abs(price_diff) / 600.0)
            if not action_note or action_note == "price_clamped_to_bounds":
                action_note = f"price_adjusted_to_{self.price:.1f}"
            self._marketing_streak = 0
            self._consecutive_support_response = 0

        elif action.action_type == "add_feature":
            feature_name = (action.payload or "").strip()
            feature_cost = 3500.0
            if not feature_name:
                invalid_or_bad += 1.0
                action_note = "feature_empty_name"
                self._marketing_streak = 0
                self._consecutive_support_response = 0
            elif self.cash < feature_cost:
                invalid_or_bad += 1.0
                action_note = "feature_rejected_low_cash"
                self._marketing_streak = 0
                self._consecutive_support_response = 0
            elif feature_name in self.implemented_features:
                invalid_or_bad += 0.6
                action_note = "feature_already_exists"
                self._marketing_streak = 0
                self._consecutive_support_response = 0
            else:
                in_pool = feature_name in DEFAULT_FEATURE_POOL
                in_requests = feature_name in self.feature_requests
                speculative = not in_pool and not in_requests
                if speculative:
                    spec = 0.28
                    invalid_or_bad += spec
                    edge_breakdown["speculative_feature"] = spec
                if len(self.implemented_features) >= 7:
                    bloat = 0.3
                    invalid_or_bad += bloat
                    edge_breakdown["feature_bloat"] = bloat
                self.cash -= feature_cost
                self.implemented_features.append(feature_name)
                self._delayed_effects.append(
                    (
                        2,
                        {
                            "churn_delta": -0.02 + self._noise(0.01),
                            "user_gain": 6 + int(8 * max(0.0, 0.5 - self.churn_rate)),
                        },
                    )
                )
                if feature_name in self.feature_requests:
                    self.churn_rate = max(0.01, self.churn_rate - 0.008)
                if self.churn_rate < 0.055 and not in_requests:
                    over = 0.18
                    invalid_or_bad += over
                    edge_breakdown["feature_when_healthy"] = over
                suffix = ""
                if speculative:
                    suffix = ";speculative_penalty"
                if self.churn_rate < 0.055 and not in_requests:
                    suffix += ";healthy_churn_misfire"
                action_note = f"feature_added_{feature_name}{suffix}"
                self._marketing_streak = 0
                self._consecutive_support_response = 0

        elif action.action_type == "run_marketing":
            budget = float(action.value or 0.0)
            if budget <= 0:
                invalid_or_bad += 0.5
                action_note = "invalid_marketing_budget"
                self._marketing_streak = 0
                self._consecutive_support_response = 0
            elif budget > self.cash:
                invalid_or_bad += 1.0
                action_note = "marketing_rejected_low_cash"
                self._marketing_streak = 0
                self._consecutive_support_response = 0
            else:
                if 0 < budget < 100.0:
                    micro = 0.22
                    invalid_or_bad += micro
                    edge_breakdown["micro_marketing_budget"] = micro
                    action_note = "micro_marketing_ineffective"
                if budget > 0.62 * max(1.0, cash_after_opex):
                    reck = 0.28
                    invalid_or_bad += reck
                    edge_breakdown["reckless_marketing_share"] = reck
                if self.users < 25 and budget > 800.0:
                    thin = 0.2
                    invalid_or_bad += thin
                    edge_breakdown["marketing_while_small_base"] = thin
                self._marketing_streak += 1
                if self._marketing_streak >= 3:
                    streak_pen = min(0.45, 0.12 * (self._marketing_streak - 2))
                    invalid_or_bad += streak_pen
                    edge_breakdown["consecutive_marketing_fatigue"] = streak_pen
                self.cash -= budget
                efficiency = 0.02 + self._noise(0.01)
                acquired = max(0, int((budget / max(10.0, self.price)) * efficiency * 10))
                self.users += acquired
                if budget > 3000:
                    # Aggressive campaigns attract low-fit users, slightly increasing churn.
                    self.churn_rate = min(0.6, self.churn_rate + 0.004 + self._noise(0.005))
                action_note = f"marketing_users_{acquired}"
                self._consecutive_support_response = 0

        elif action.action_type == "respond_to_feedback":
            self._marketing_streak = 0
            if self.step_count < self._feedback_skip_until_step:
                invalid_or_bad += 0.4
                action_note = "support_overused"
            else:
                support_cost = 900.0
                if self.cash < support_cost:
                    invalid_or_bad += 0.8
                    action_note = "support_rejected_low_cash"
                else:
                    if self.churn_rate < 0.048:
                        overkill = 0.2
                        invalid_or_bad += overkill
                        edge_breakdown["support_overkill_low_churn"] = overkill
                    self.cash -= support_cost
                    self.churn_rate = max(0.01, self.churn_rate - 0.015 + self._noise(0.005))
                    # Immediate sentiment bump; sustained effect comes from refreshed feedback.
                    self.customer_feedback = [
                        self._rng.choice(POSITIVE_FEEDBACK),
                        self._rng.choice(POSITIVE_FEEDBACK),
                        self._rng.choice(NEGATIVE_FEEDBACK) if self._rng.random() < 0.2 else self._rng.choice(POSITIVE_FEEDBACK),
                    ]
                    # Block respond on the immediate next step (step_count increments at end of this step).
                    self._feedback_skip_until_step = self.step_count + 2
                    action_note = "feedback_addressed"
                    self._consecutive_support_response += 1
                    if self._consecutive_support_response >= 4:
                        burn = 0.15 * (self._consecutive_support_response - 3)
                        burn = min(0.5, burn)
                        invalid_or_bad += burn
                        edge_breakdown["support_dependency"] = burn
            if action_note != "feedback_addressed":
                self._consecutive_support_response = 0

        elif action.action_type == "do_nothing":
            self.churn_rate = min(0.6, self.churn_rate + 0.01 + self._noise(0.01))
            invalid_or_bad += 0.2
            action_note = "idle_penalty"
            crisis = self._edge_penalty_crisis_idle()
            if crisis > 0:
                invalid_or_bad += crisis
                edge_breakdown["crisis_idle"] = crisis
            self._marketing_streak = 0
            self._consecutive_support_response = 0

        # Penalize repetitive loops that do not adapt strategy.
        if not action_invalid_schema:
            if action.action_type == self._last_action_type:
                self._action_streak += 1
            else:
                self._action_streak = 1
            self._last_action_type = action.action_type
            if self._action_streak >= 3:
                loop_penalty = min(0.8, 0.2 * (self._action_streak - 2))
                invalid_or_bad += loop_penalty
                edge_breakdown["action_loop"] = loop_penalty

        # Market dynamics and competitor movement.
        self.competitor_price = max(15.0, min(120.0, self.competitor_price + self._rng.uniform(-1.5, 1.5)))
        price_gap = self.price - self.competitor_price
        price_pressure = 0.005 * (1 if price_gap > 0 else -1)
        self.churn_rate = max(0.01, min(0.6, self.churn_rate + price_pressure + self._noise(0.005)))

        churned_users = int(self.users * self.churn_rate)
        self.users = max(0, self.users - churned_users)
        self.revenue = self.users * self.price
        self.cash += self.revenue

        if self.cash < 1800.0:
            runway = min(0.25, max(0.0, (1800.0 - self.cash) / 8000.0))
            if runway > 0:
                invalid_or_bad += runway
                edge_breakdown["low_runway_stress"] = runway

        if self.step_count % 2 == 1 or action.action_type == "respond_to_feedback":
            self.customer_feedback = self._sample_feedback(size=3)
            self.feature_requests = self._sample_feature_requests(size=3)

        reward_obj = self._compute_reward(invalid_or_bad_actions=invalid_or_bad)
        self.step_count += 1

        done = self.step_count >= self.max_steps or self.cash <= 0 or self.users <= 5
        info = {
            "action_note": action_note,
            "churned_users": churned_users,
            "price": round(self.price, 2),
            "competitor_price": round(self.competitor_price, 2),
            "cash_delta": round(self.cash - pre_step_cash, 2),
            "revenue_delta": round(self.revenue - self._last_revenue, 2),
            "user_delta": self.users - self._last_users,
            "sentiment": round(self._sentiment_score(), 4),
            "invalid_penalty": round(invalid_or_bad, 4),
            "loop_penalty": round(loop_penalty, 4),
            "action_streak": self._action_streak,
            "edge_penalties": edge_breakdown,
            "invalid_schema": action_invalid_schema,
            "triggered_shocks": triggered_shocks,
            "scenario_name": self.scenario_name,
        }
        if self.cash <= 0:
            info["terminal_reason"] = "bankrupt"
        elif self.users <= 5:
            info["terminal_reason"] = "user_collapse"
        elif self.step_count >= self.max_steps:
            info["terminal_reason"] = "max_steps_reached"

        return self._build_observation(), reward_obj, done, info

    def _compute_reward(self, invalid_or_bad_actions: float) -> Reward:
        user_growth = (self.users - self._last_users) / max(1, self._last_users)
        revenue_change = (self.revenue - self._last_revenue) / max(1.0, self._last_revenue)
        sentiment_score = self._sentiment_score()
        churn_penalty = self.churn_rate

        total = (
            0.3 * user_growth
            + 0.2 * revenue_change
            - 0.3 * churn_penalty
            + 0.2 * sentiment_score
            - 0.2 * invalid_or_bad_actions
        )
        return Reward(
            total=round(total, 6),
            user_growth=round(user_growth, 6),
            revenue_change=round(revenue_change, 6),
            churn_penalty=round(churn_penalty, 6),
            sentiment_score=round(sentiment_score, 6),
            invalid_or_bad_actions=round(invalid_or_bad_actions, 6),
        )
