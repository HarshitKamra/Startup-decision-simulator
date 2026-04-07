#!/usr/bin/env python3
"""CI / judge smoke test: imports, env rollouts, graders, invalid-action handling."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    errors: list[str] = []

    # YAML optional
    try:
        import yaml  # noqa: F401
    except ImportError:
        yaml = None

    if yaml is not None:
        cfg = ROOT / "openenv.yaml"
        if cfg.is_file():
            data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            if not data.get("entrypoint"):
                errors.append("openenv.yaml missing entrypoint")
            tasks = data.get("tasks") or []
            if len(tasks) < 3:
                errors.append("openenv.yaml should list at least 3 tasks")

    from env.environment import StartupDecisionEnv
    from env.grader import grade_task
    from env.policies import heuristic_baseline_policy, noop_policy
    from env.tasks import TASKS

    # Short rollout per task with heuristic
    for tid in sorted(TASKS.keys()):
        cfg = TASKS[tid]
        env = StartupDecisionEnv(
            seed=cfg.seed,
            max_steps=min(4, cfg.max_steps),
            initial_cash=cfg.initial_cash,
            initial_users=cfg.initial_users,
            initial_price=cfg.initial_price,
            competitor_price=cfg.competitor_price,
            initial_churn_rate=cfg.initial_churn_rate,
            scenario_name=cfg.scenario_name,
            shock_schedule=cfg.shock_schedule,
        )
        obs = env.reset()
        history: list[dict] = []
        for _ in range(min(4, cfg.max_steps)):
            act = heuristic_baseline_policy(tid, obs.model_dump(), history)
            history.append(act)
            obs, _, done, _ = env.step(act)
            if done:
                break
        if not history:
            errors.append(f"{tid}: empty history")

    # Full-horizon grades must be in [0,1]
    for tid in sorted(TASKS.keys()):
        cfg = TASKS[tid]
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
        env.reset()
        actions = []
        obs = env.reset()
        for _ in range(cfg.max_steps):
            a = heuristic_baseline_policy(tid, obs.model_dump(), actions)
            actions.append(a)
            obs, _, done, _ = env.step(a)
            if done:
                break
        s = grade_task(tid, actions)
        if not 0.0 <= s <= 1.0:
            errors.append(f"{tid}: grade {s} out of range")
    # Weak baseline should score lower on average than heuristic (sanity)
    weak_scores = []
    for tid in sorted(TASKS.keys()):
        cfg = TASKS[tid]
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
        actions = []
        for _ in range(cfg.max_steps):
            a = noop_policy(tid, obs.model_dump(), actions)
            actions.append(a)
            obs, _, done, _ = env.step(a)
            if done:
                break
        weak_scores.append(grade_task(tid, actions))
    strong_avg = sum(
        grade_task(
            tid,
            _rollout_actions(tid, heuristic_baseline_policy),
        )
        for tid in TASKS
    ) / len(TASKS)
    weak_avg = sum(weak_scores) / len(weak_scores)
    if strong_avg <= weak_avg:
        errors.append(f"heuristic ({strong_avg:.4f}) should beat noop ({weak_avg:.4f})")

    # Dynamic import of entrypoint from openenv.yaml
    try:
        importlib.import_module("env.environment")
    except Exception as e:  # noqa: BLE001
        errors.append(f"import env.environment failed: {e}")

    # Invalid dict action consumes step without crash
    env = StartupDecisionEnv(seed=1, max_steps=3)
    env.reset()
    _, r, _, info = env.step({"action_type": "adjust_price"})  # missing value
    if r.invalid_or_bad_actions < 0.5:
        errors.append("invalid schema should increase invalid_or_bad_actions")

    if errors:
        print("HEALTHCHECK FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("HEALTHCHECK OK")
    print(f"  heuristic_avg={strong_avg:.4f} noop_avg={weak_avg:.4f}")
    print("  grade_task() deterministic [0,1] verified; invalid-action path OK")
    return 0


def _rollout_actions(task_id: str, policy) -> list[dict]:
    from env.environment import StartupDecisionEnv
    from env.tasks import TASKS

    cfg = TASKS[task_id]
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
    actions = []
    for _ in range(cfg.max_steps):
        a = policy(task_id, obs.model_dump(), actions)
        actions.append(a)
        obs, _, done, _ = env.step(a)
        if done:
            break
    return actions


if __name__ == "__main__":
    raise SystemExit(main())
