from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, confloat, constr, model_validator


class ActionType(str, Enum):
    adjust_price = "adjust_price"
    add_feature = "add_feature"
    run_marketing = "run_marketing"
    respond_to_feedback = "respond_to_feedback"
    do_nothing = "do_nothing"


class Observation(BaseModel):
    cash: confloat(ge=0.0) = Field(..., description="Current cash reserves in USD.")
    users: int = Field(..., ge=0, description="Active paying users.")
    churn_rate: confloat(ge=0.0, le=1.0) = Field(..., description="Current churn rate.")
    revenue: confloat(ge=0.0) = Field(..., description="Revenue generated this step in USD.")
    price: confloat(gt=0.0) = Field(..., description="Current list price per user per billing period (USD).")
    customer_feedback: List[constr(min_length=1)] = Field(default_factory=list)
    competitor_price: confloat(gt=0.0) = Field(...)
    feature_requests: List[constr(min_length=1)] = Field(default_factory=list)
    current_alerts: List[constr(min_length=1)] = Field(
        default_factory=list,
        description="Active market/operational alerts injected by deterministic scenario events.",
    )
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    sentiment_score: confloat(ge=-1.0, le=1.0) = Field(
        ...,
        description="Normalized feedback sentiment in [-1,1]; aligns with reward and easy-task grader.",
    )
    price_vs_competitor: confloat(gt=0.0) = Field(
        ...,
        description="Ratio price / competitor_price (>1 means premium vs comp).",
    )
    runway_opex_steps: confloat(ge=0.0) = Field(
        ...,
        description="Approx. steps of baseline opex (600 USD/step) cash could cover; survival signal.",
    )
    steps_remaining: int = Field(..., ge=0, description="Episodes steps left including current (max_steps - step_count).")
    support_cooldown_steps: int = Field(
        ...,
        ge=0,
        description="Steps until respond_to_feedback avoids overuse penalty (0 = can respond).",
    )
    implemented_feature_count: int = Field(..., ge=0, description="Shipped features count (delayed effects may be queued).")


class Action(BaseModel):
    action_type: ActionType
    value: Optional[confloat()] = Field(
        default=None,
        description=(
            "Generic scalar parameter used by actions. "
            "For adjust_price it is the target price in USD. "
            "For run_marketing it is the budget in USD."
        ),
    )
    payload: Optional[constr(min_length=1)] = Field(
        default=None,
        description="Optional string payload used by add_feature/respond_to_feedback.",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        if self.action_type == ActionType.adjust_price:
            if self.value is None:
                raise ValueError("adjust_price requires value.")
            if self.value <= 0:
                raise ValueError("Price must be positive.")
        if self.action_type == ActionType.run_marketing:
            if self.value is None:
                raise ValueError("run_marketing requires value.")
            if self.value < 0:
                raise ValueError("Marketing budget must be non-negative.")
        if self.action_type in (ActionType.add_feature, ActionType.respond_to_feedback):
            if self.payload is None:
                raise ValueError(f"{self.action_type} requires non-empty payload.")
            stripped = self.payload.strip()
            if not stripped:
                raise ValueError(f"{self.action_type} requires non-empty payload.")
            self.payload = stripped
        return self


class Reward(BaseModel):
    total: float
    user_growth: float
    revenue_change: float
    churn_penalty: confloat(ge=0.0, le=1.0)
    sentiment_score: confloat(ge=-1.0, le=1.0)
    invalid_or_bad_actions: confloat(ge=0.0)

