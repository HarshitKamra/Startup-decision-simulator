from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ActionType = Literal[
    "adjust_price",
    "add_feature",
    "run_marketing",
    "respond_to_feedback",
    "do_nothing",
]


class Observation(BaseModel):
    cash: float = Field(..., description="Current cash reserves in USD.")
    users: int = Field(..., ge=0, description="Active paying users.")
    churn_rate: float = Field(..., ge=0.0, le=1.0, description="Current churn rate.")
    revenue: float = Field(..., ge=0.0, description="Revenue generated this step in USD.")
    price: float = Field(..., gt=0.0, description="Current list price per user per billing period (USD).")
    customer_feedback: List[str] = Field(default_factory=list)
    competitor_price: float = Field(..., gt=0.0)
    feature_requests: List[str] = Field(default_factory=list)
    current_alerts: List[str] = Field(
        default_factory=list,
        description="Active market/operational alerts injected by deterministic scenario events.",
    )
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Normalized feedback sentiment in [-1,1]; aligns with reward and easy-task grader.",
    )
    price_vs_competitor: float = Field(
        ...,
        gt=0.0,
        description="Ratio price / competitor_price (>1 means premium vs comp).",
    )
    runway_opex_steps: float = Field(
        ...,
        ge=0.0,
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
    value: Optional[float] = Field(
        default=None,
        description=(
            "Generic scalar parameter used by actions. "
            "For adjust_price it is the target price in USD. "
            "For run_marketing it is the budget in USD."
        ),
    )
    payload: Optional[str] = Field(
        default=None,
        description="Optional string payload used by add_feature/respond_to_feedback.",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        if self.action_type == "adjust_price":
            if self.value is None:
                raise ValueError("adjust_price requires value.")
            if self.value <= 0:
                raise ValueError("Price must be positive.")
        if self.action_type == "run_marketing":
            if self.value is None:
                raise ValueError("run_marketing requires value.")
            if self.value < 0:
                raise ValueError("Marketing budget must be non-negative.")
        if self.action_type in ("add_feature", "respond_to_feedback"):
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
    churn_penalty: float
    sentiment_score: float
    invalid_or_bad_actions: float
