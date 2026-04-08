---
title: Startup Decision Simulator
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860

---

# Startup Decision Simulator (Founder Agent Benchmark)


A production-style OpenEnv benchmark where an agent acts like a startup founder and makes strategic decisions under realistic constraints: limited cash, churn pressure, user feedback quality, and competition.

## Hackathon / judge quickstart (about one minute)

From the repo root:

```bash
pip install -r requirements.txt
python scripts/healthcheck.py          # must print HEALTHCHECK OK
python inference.py --baseline-only      # full benchmark, no API keys
python inference.py --baseline-only --json -o results.json   # artifact for leaderboards
```

CI (optional): push to GitHub and use `.github/workflows/ci.yml` — installs deps, compiles, runs `healthcheck`, uploads `benchmark-result.json`.

## Deploying to Hugging Face Spaces

This repository is configured for a Hugging Face `Docker` Space.

Before submitting, add these variables in your Space settings:

- `API_BASE_URL`: API endpoint for the LLM.
- `MODEL_NAME`: model identifier used by `inference.py`.
- `HF_TOKEN`: Hugging Face / provider API key.

Recommended deployment flow:

```bash
git remote add hf https://huggingface.co/spaces/<username-or-org>/<space-name>
git push hf main
```

After the push:

- wait for the Docker build to finish,
- verify the Space URL returns HTTP `200`,
- verify the app responds to `reset()` for the pre-submission validator.

### Why this submission stands out

| Hook | Detail |
|------|--------|
| **Not another triage bot** | Multi-step **economics + operations**: pricing, product latency, marketing burn, support cooldowns, competitor dynamics. |
| **Judge-friendly** | Deterministic **\[0,1\] graders**, **dense reward**, **rich observations** aligned with grading signals, **edge-case penalties** documented in `info["edge_penalties"]`. |
| **Evidence-driven** | Built-in **counterfactual decision lift** (`actual_reward - do_nothing_reward`) plus per-task JSONL traces and markdown benchmark report in `artifacts/`. |
| **Works without secrets** | **Heuristic baseline** in `env/policies.py` scores strongly; LLM is optional via `HF_TOKEN` / `OPENAI_API_KEY`. |
| **Production-shaped** | `Dockerfile` runs `healthcheck` at build time; `LICENSE` MIT; reproducible `requirements.txt`. |

## Design decisions

- **Why the observation space is structured this way:** We chose to expose operational stress factors—like `runway_opex_steps` and `support_cooldown_steps`—because pure financial metrics don't capture the constraints founders actually operate under. An agent must balance long-term survival against short-term support burnout.
- **Why the reward is decomposed:** We chose to split the reward into distinct components (growth velocity, customer satisfaction, and execution penalties) because it allows us to heavily penalize "idle camping" (e.g. spamming `do_nothing`) and malformed schemas, while independently rewarding balanced strategic choices.
- **Why the hard task is genuinely hard:** We chose to inject a multi-step dependency (an emergency vendor code) into the observation stream because it prevents single-pass LLM guessing. The agent must read a critical alert at Step 1 and apply its exact payload in Step 2 to secure credit, proving it can maintain context across the trajectory.

## Why this benchmark matters

Real startup execution is a sequence of constrained trade-offs, not a single correct answer. This environment tests whether an agent can:

- prioritize user retention before pure growth,
- reason about delayed effects (feature work pays off later),
- avoid self-destructive loops (overspending, no-op behavior),
- and balance pricing, support, and marketing in one coherent strategy.

That makes it a stronger real-world benchmark than simple single-turn classification tasks.

## Project structure

```text
project/
│── env/
│   ├── environment.py
│   ├── models.py
│   ├── tasks.py
│   ├── grader.py
│
│── inference.py
│── openenv.yaml
│── Dockerfile
│── requirements.txt
│── pyproject.toml
│── LICENSE
│── scripts/
│       └── healthcheck.py
│── README.md
```

## OpenEnv API implementation

The environment implements the required interface and typed models:

- `Observation` (`pydantic`) with:
  - `cash`, `users`, `churn_rate`, `revenue`, **`price`** (your listed price; aligns with pricing tasks),
  - `customer_feedback`, `competitor_price`, `feature_requests`,
  - `step_count`, `max_steps`,
  - **`sentiment_score`** \([-1,1]\) (same signal as the reward term and easy grader),
  - **`price_vs_competitor`**, **`runway_opex_steps`** (cash / baseline opex), **`steps_remaining`**,
  - **`support_cooldown_steps`** (must be `0` to use `respond_to_feedback` without overuse penalty),
  - **`implemented_feature_count`**.
- `Action` (`pydantic`) with:
  - action types: `adjust_price`, `add_feature`, `run_marketing`, `respond_to_feedback`, `do_nothing`,
  - action-specific validation.
- `Reward` (`pydantic`) with total and component metrics.
- `StartupDecisionEnv.reset() -> Observation`
- `StartupDecisionEnv.step(action) -> (Observation, Reward, done, info)`
- `StartupDecisionEnv.state() -> dict`

## State design

Core state variables:

- `cash` (`float`)
- `users` (`int`)
- `churn_rate` (`float`)
- `revenue` (`float`)
- `customer_feedback` (`list[str]`)
- `competitor_price` (`float`)
- `feature_requests` (`list[str]`)

Additional internals for realism:

- `price`
- delayed effect queue for feature impact
- implemented feature registry
- support cooldown (`support_cooldown_steps` in the observation blocks `respond_to_feedback` on the next step after a successful support action) and operating cost pressure

## Evaluation criteria alignment (hackathon rubric)

Design choices map to common scoring dimensions:

| Criterion | How this benchmark supports it |
|-----------|--------------------------------|
| **Real-world utility (≈30%)** | Founder decisions under cash, churn, competition, delayed product effects, and feedback loops—not a single-turn chat task. |
| **Task & grader quality (≈25%)** | Three tasks with **deterministic** graders in \([0,1]\): easy stresses **sentiment + churn + execution quality**; medium uses a **Pareto / balance** score (hard to game one metric); hard gives **partial survival credit** and runway-aware bonuses. All subtract **mean per-step invalid penalties** from trajectories. |
| **Environment design (≈20%)** | Dense shaped reward + rich **observations** aligned with grader signals (`sentiment_score`, runway, cooldown, price vs comp). |
| **Code quality (≈15%)** | Modular `env/` package, typed models, explicit `info` telemetry. |
| **Creativity (≈10%)** | Startup survival + delayed feature ROI + support cooldown as “operational constraint,” beyond triage-style benchmarks. |

## Action space and dynamics

- `adjust_price(value)`
  - Immediate effect on churn (higher price can raise churn, lower price can reduce churn).
  - Revenue reacts through price * users and churn.
- `add_feature(payload)`
  - Costs cash now, applies delayed churn reduction/user gain after 2 steps.
  - Duplicate or unaffordable feature attempts are penalized.
- `run_marketing(value)`
  - Spends budget for user acquisition.
  - Overspending can slightly worsen churn due to low-fit leads.
- `respond_to_feedback(payload)`
  - Costs support budget, immediately improves churn/sentiment.
  - Includes cooldown to discourage repetitive spam.
- `do_nothing`
  - Explicit penalty and natural churn drift.

Also included:

- delayed feature effects,
- stochastic/noisy market and feedback generation,
- deterministic scenario shock events (`support_backlog`, `competitor_discount`, `incident_outage`, etc.),
- realistic budget and runway constraints.

## Reward design (dense per step)

Reward is dense and computed every step:

```text
reward =
  + 0.3 * user_growth
  + 0.2 * revenue_change
  - 0.3 * churn_rate
  + 0.2 * sentiment_score
  - 0.2 * invalid_or_bad_actions
```

Properties:

- dense (not binary),
- partial progress via growth/revenue/sentiment components,
- bad strategy penalties for invalid, repetitive, or idle behavior.

### Edge cases and extra penalties (simulation)

The environment adds structured penalties (rolled into `invalid_or_bad_actions`, surfaced in `info["edge_penalties"]`) for brittle real-world failures:

- **Invalid action payloads**: malformed dicts that fail Pydantic validation still consume a step (opex + physics), with `info["invalid_schema"]=true`.
- **Pricing**: large single-step price moves (shock), clamping beyond bounds, and deep undercuts vs. competitors (predatory pricing churn).
- **Product**: speculative features (not in the roadmap pool and not in current requests), feature bloat (many shipped features), and “healthy churn” misfires (building when churn is already very low without demand signals).
- **Growth**: micro marketing budgets (ineffective spend), overspending as a fraction of liquid cash, consecutive marketing fatigue, and heavy spend with a tiny user base.
- **Support**: cooldown spam, support when churn is already very low (overkill), and **support dependency** if feedback response is overused across steps.
- **Execution**: `do_nothing` during high churn + negative sentiment (crisis idle).
- **Runway**: post-revenue cash below a stress threshold adds a smooth runway penalty.
- **Loops**: repeated identical action types still accrue loop penalties (see `info["action_streak"]` / `edge_penalties["action_loop"]`).

`Action` payloads for `add_feature` / `respond_to_feedback` are **stripped**; all-whitespace payloads are rejected at parse time.

## Tasks (easy -> medium -> hard)

### 1) Support Turbulence (`support_turbulence`)

- Goal: improve sentiment from feedback and reduce churn pressure.
- Grader uses deterministic:
  - sentiment improvement vs. start,
  - final churn quality,
  - multiplicative penalty for **invalid / bad-action** schemas,
  - small **diversity** bonus for balanced `respond_to_feedback` share,
  - heavy discount if the company **collapses** (cash/users) or sentiment does not improve.
- Score range: `[0.0, 1.0]`

### 2) Pricing Pressure (`pricing_pressure`)

- Goal: balance revenue/profit while avoiding churn spikes.
- Grader uses deterministic:
  - **geometric blend** of profit, revenue, and churn reduction (rewards **balance**, not one-axis spikes),
  - **harmony** term (penalizes lopsided outcomes),
  - churn regression penalty,
  - multiplicative invalid penalty floors the score for broken schemas.
- Score range: `[0.0, 1.0]`

### 3) Runway Crisis (`runway_crisis`)

- Goal: survive the horizon with low runway while optimizing growth/churn.
- Grader (`grade_hard`) uses deterministic:
  - **full** survival vs. **partial** credit by steps completed alive,
  - user growth, churn reduction, cash efficiency,
  - runway bonus when cash stays above a fraction of starting runway,
  - mean invalid penalty; failure state is discounted.
- Score range: `[0.0, 1.0]`

## Deterministic grading

Evaluation is deterministic by:

- fixed per-task seeds,
- fixed initial states,
- deterministic action replay in grader.

Environment trajectories can still include stochastic dynamics, but grader runs are reproducible with the same action sequence and seed.

## Setup

### Local

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Docker

```bash
docker build -t startup-decision-simulator .
docker run --rm -e API_BASE_URL=https://api.openai.com/v1 -e MODEL_NAME=gpt-4o-mini -e HF_TOKEN=$HF_TOKEN startup-decision-simulator
```

## Running inference

`inference.py`:

- uses OpenAI client,
- reads env vars:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- uses a deterministic task-aware fallback policy if no token is set,
- prints exact log tags:
  - `[START]`
  - `[STEP]`
  - `[END]`

Run:

```bash
python inference.py
```

### CLI flags (inference)

| Flag | Meaning |
|------|---------|
| `--baseline-only` | Never call the LLM; use `heuristic_baseline_policy` only. |
| `--json` | Print a single JSON summary on **stdout**; step logs on **stderr**. |
| `-o FILE` / `BENCHMARK_OUTPUT` | Save the same JSON summary to a file. |
| `--trace-dir DIR` | Write per-task trajectory traces (`*.jsonl`) + `summary_report.md` (default `artifacts/`). |
| `--model NAME` | Chat model when a token is present (default `MODEL_NAME` or `gpt-4o-mini`). |

## Example output

```text
[START] task=support_turbulence seed=11 max_steps=8
[STEP] task=support_turbulence step=0 action=respond_to_feedback reward=0.041232 cash=2034.00 users=83 churn=0.0910 info=feedback_addressed penalty=0.0
[STEP] task=support_turbulence step=1 action=add_feature reward=0.016812 cash=1056.00 users=79 churn=0.0732 info=feature_added_mobile_app penalty=0.0
[END] task=support_turbulence score=0.744100 steps=8
[END] aggregate_score=0.682743 scores={"support_turbulence":0.7441,"runway_crisis":0.61205,"pricing_pressure":0.692079}
```

## Validation targets

- `docker build` should complete successfully.
- `openenv validate` should pass with `openenv.yaml` and referenced Python entrypoints.
