from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, TextIO

from openai import OpenAI

from env.environment import StartupDecisionEnv
from env.grader import grade_task
from env.policies import RUBRIC_HINTS, heuristic_baseline_policy
from env.tasks import TASKS, get_task_config

LogFn = Callable[..., None]


def build_client() -> OpenAI | None:
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    token = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
    if not token:
        return None
    return OpenAI(base_url=base_url, api_key=token)


def get_action_from_model(
    client: OpenAI | None,
    model_name: str,
    task_id: str,
    observation: Dict[str, Any],
    step_index: int,
    action_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if client is None:
        return heuristic_baseline_policy(task_id, observation, action_history)

    prompt = (
        "You are controlling a startup simulation. Return ONLY minified JSON with keys "
        "action_type, and optional value/payload. Valid action_type values: "
        "adjust_price, add_feature, run_marketing, respond_to_feedback, do_nothing.\n"
        "Observation includes price, sentiment_score, price_vs_competitor, runway_opex_steps, "
        "steps_remaining, support_cooldown_steps (do not respond while >0), implemented_feature_count.\n"
        f"Objective={RUBRIC_HINTS.get(task_id, '')}\n"
        f"Task={task_id}\n"
        f"Step={step_index}\n"
        f"Observation={json.dumps(observation, separators=(',', ':'))}\n"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            top_p=1,
            messages=[
                {"role": "system", "content": "Return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception:
        return heuristic_baseline_policy(task_id, observation, action_history)


def run_task(
    task_id: str,
    client: OpenAI | None,
    model_name: str,
    log: LogFn,
) -> Dict[str, Any]:
    cfg = get_task_config(task_id)
    env = StartupDecisionEnv(
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        initial_cash=cfg.initial_cash,
        initial_users=cfg.initial_users,
        initial_price=cfg.initial_price,
        competitor_price=cfg.competitor_price,
        initial_churn_rate=cfg.initial_churn_rate,
    )
    obs = env.reset()
    action_history: List[Dict[str, Any]] = []
    sum_reward = 0.0

    log(f"[START] task={task_id} seed={cfg.seed} max_steps={cfg.max_steps}")
    for step in range(cfg.max_steps):
        obs_dict = obs.model_dump()
        action = get_action_from_model(client, model_name, task_id, obs_dict, step, action_history)
        action_history.append(action)
        obs, reward, done, info = env.step(action)
        sum_reward += reward.total
        log(
            f"[STEP] task={task_id} step={step} action={action.get('action_type')} "
            f"reward={reward.total:.6f} cash={obs.cash:.2f} users={obs.users} "
            f"churn={obs.churn_rate:.4f} info={info.get('action_note', '')} "
            f"penalty={info.get('invalid_penalty', 0.0)}"
        )
        if done:
            break

    score = grade_task(task_id, action_history)
    log(f"[END] task={task_id} score={score:.6f} steps={len(action_history)}")
    return {
        "task_id": task_id,
        "score": score,
        "steps": len(action_history),
        "sum_reward": round(sum_reward, 6),
        "seed": cfg.seed,
        "max_steps": cfg.max_steps,
        "llm_used": client is not None,
        "model_name": model_name if client else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Startup Decision Simulator — OpenEnv inference / benchmark")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"), help="Chat model when API key is set")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Force heuristic baseline (no LLM); reproducible scores without API",
    )
    parser.add_argument("--json", action="store_true", help="Write final summary JSON to stdout; logs go to stderr")
    parser.add_argument("-o", "--output", default=os.getenv("BENCHMARK_OUTPUT"), help="Also write JSON summary to this path")
    args = parser.parse_args()

    log: LogFn
    if args.json:

        def _log(*a: Any, **k: Any) -> None:
            print(*a, **k, file=sys.stderr)

        log = _log
    else:
        log = print

    model_name = args.model
    client: OpenAI | None = None if args.baseline_only else build_client()
    if client is None and not args.baseline_only:
        log("[INFO] No HF_TOKEN/OPENAI_API_KEY; using heuristic baseline policy.")

    rows: List[Dict[str, Any]] = []
    for task_id in sorted(TASKS.keys()):
        rows.append(run_task(task_id, client, model_name, log=log))

    scores = {r["task_id"]: r["score"] for r in rows}
    average = sum(scores.values()) / len(scores)
    log(f"[END] aggregate_score={average:.6f} scores={json.dumps(scores, sort_keys=True)}")

    summary = {
        "project": "startup-decision-simulator",
        "aggregate_score": round(average, 6),
        "per_task": rows,
        "mode": "heuristic" if client is None else "llm",
        "model_name": model_name if client else None,
    }
    json_str = json.dumps(summary, indent=2 if args.json else None, sort_keys=True)
    if args.json:
        print(json_str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str if args.json else json.dumps(summary, indent=2, sort_keys=True))
            f.write("\n")


if __name__ == "__main__":
    main()
