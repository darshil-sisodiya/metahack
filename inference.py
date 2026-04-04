"""Submission inference script for the OpenEnv distillation benchmark.

The required submission entry point is a root-level `inference.py` file that:

- uses the OpenAI-compatible client for LLM inference in submission mode
- reads API configuration from environment variables
- runs deterministic seeded rollouts across all tasks
- emits structured stdout logs using [START], [STEP], and [END] markers
- produces normalized 0.0-1.0 scores for each task and an overall score

The script is intended to run in LLM mode and will fail fast if the API
configuration is missing.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.grader import TASK_WEIGHTS, compute_episode_score
from app.models import Action, EnvironmentState, Observation
from app.runtime import OpenEnvRuntime
from app.tasks import get_all_tasks, get_task_by_name

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash").strip()
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
API_KEY = (HF_TOKEN.strip() if HF_TOKEN else "") or OPENAI_API_KEY or os.environ.get("API_KEY", "").strip()

OPENENV_EPISODES_PER_TASK = max(1, int(os.environ.get("OPENENV_EPISODES_PER_TASK", "1")))
OPENENV_BASE_SEED = int(os.environ.get("OPENENV_BASE_SEED", "42"))
OPENENV_REQUEST_TIMEOUT = float(os.environ.get("OPENENV_REQUEST_TIMEOUT", "60"))
API_CALL_DELAY = max(0.0, float(os.environ.get("API_CALL_DELAY", "0")))

AGENT_MODE = "llm"


def build_client() -> OpenAI | None:
    """Create the OpenAI-compatible client when running in LLM mode."""
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL must be set for inference.py.")
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME must be set for inference.py.")
    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN or OPENAI_API_KEY must be set for inference.py."
        )

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=OPENENV_REQUEST_TIMEOUT,
    )


CLIENT = build_client()

memory: dict[str, Any] = {
    "best_reward": -1e9,
    "best_action": None,
    "reward_history": [],
    "recent_steps": deque(maxlen=3),
}


def reset_memory() -> None:
    """Reset episode memory at the start of each episode."""
    memory["best_reward"] = -1e9
    memory["best_action"] = None
    memory["reward_history"] = []
    memory["recent_steps"] = deque(maxlen=3)


def compact_action(action: Action) -> dict[str, Any]:
    """Return a stable action payload for logs."""
    return {
        "steam_valve": round(action.steam_valve, 4),
        "reflux_ratio": round(action.reflux_ratio, 4),
        "feed_rate": round(action.feed_rate, 4),
        "vent": int(action.vent),
    }


def compact_observation(observation: Observation) -> dict[str, Any]:
    """Return a stable observation payload for logs."""
    return {
        "temperature": round(observation.temperature, 4),
        "pressure": round(observation.pressure, 4),
        "purity": round(observation.purity, 4),
        "flow_rate": round(observation.flow_rate, 4),
        "energy_usage": round(observation.energy_usage, 4),
        "time_step": int(observation.time_step),
    }


def compact_state(state: EnvironmentState) -> dict[str, Any]:
    """Return a stable state payload for prompts."""
    return {
        "active_task": state.active_task,
        "temperature": round(state.temperature, 4),
        "pressure": round(state.pressure, 4),
        "purity": round(state.purity, 4),
        "flow_rate": round(state.flow_rate, 4),
        "energy_usage": round(state.energy_usage, 4),
        "time_step": int(state.time_step),
        "hidden_instability": round(state.hidden_instability, 4),
        "cooling_failure": bool(state.cooling_failure),
        "pressure_spike": bool(state.pressure_spike),
        "prev_action": (
            compact_action(state.prev_action) if state.prev_action is not None else None
        ),
    }


def build_system_prompt(task_name: str, task_description: str) -> str:
    """Build a concise system prompt for the LLM controller."""
    return (
        "You are an AI agent controlling a chemical distillation column. "
        "Maximize reward by keeping Temperature near 100, Pressure safely below 2.5, "
        "and Purity above 60. Output ONLY valid JSON."
    )


def build_user_prompt(
    observation: Observation,
    state: EnvironmentState,
    last_reward: float | None,
) -> str:
    """Build a compact step prompt with static priority rules and rolling history."""
    # ── Previous action ────────────────────────────────────────────
    prev_steam = state.prev_action.steam_valve if state.prev_action is not None else 50.0
    prev_reflux = state.prev_action.reflux_ratio if state.prev_action is not None else 50.0
    prev_feed = state.prev_action.feed_rate if state.prev_action is not None else 50.0
    prev_vent = state.prev_action.vent if state.prev_action is not None else 0
    prev_reward = f"{last_reward:.4f}" if last_reward is not None else "N/A"

    # ── Best known action ──────────────────────────────────────────
    best_action = memory["best_action"]
    if best_action is not None:
        best_block = (
            f"* Steam: {best_action.steam_valve:.1f}  "
            f"Reflux: {best_action.reflux_ratio:.1f}  "
            f"Feed: {best_action.feed_rate:.1f}  "
            f"Vent: {best_action.vent}\n"
            f"* Best Reward: {memory['best_reward']:.4f}"
        )
    else:
        best_block = "* No data yet (first step)."

    # ── Rolling history ────────────────────────────────────────────
    recent = memory["recent_steps"]
    history_block = "\n".join(recent) if recent else "(no history yet)"

    return f"""CURRENT STATE:

* Temperature: {observation.temperature:.1f}
* Pressure: {observation.pressure:.2f}
* Purity: {observation.purity:.1f}
* Flow rate: {observation.flow_rate:.1f}

PREVIOUS ACTION:

* Steam: {prev_steam:.1f}
* Reflux: {prev_reflux:.1f}
* Feed: {prev_feed:.1f}
* Vent: {prev_vent}

PREVIOUS REWARD: {prev_reward}

BEST KNOWN ACTION:

{best_block}

RECENT HISTORY (last 3 steps):

{history_block}

CONTROL HIERARCHY (Follow strictly in order of priority):

PRIORITY 1: SAFETY (CRITICAL)
- Vent is your primary safety control. If Pressure > 2.0, "vent" MUST be 1.
- If Pressure > 2.0, you MUST also decrease "steam_valve".

PRIORITY 2: STABILITY
- Target Temperature is 90.
- If Temperature > 95, decrease "steam_valve" by at least 5.
- If Temperature < 85, increase "steam_valve" by at least 5.

PRIORITY 3: OPTIMIZATION
- Only when Pressure is safe and Temperature is stable, focus on Purity.
- If Purity < 60, increase "reflux_ratio".
- Do not let "feed_rate" drop below 20 unless Pressure is critical.

PRIORITY 4: ANTI-STAGNATION (MANDATORY)
- Do NOT output the exact same action as the PREVIOUS ACTION.
- You MUST change at least one variable (steam, reflux, or feed) by a minimum of 2.0 units every step to explore the environment.
- If the PREVIOUS REWARD decreased, reverse your previous action direction.

OUTPUT FORMAT (STRICT JSON ONLY):
{{{{
"steam_valve": number (0-100),
"reflux_ratio": number (0-100),
"feed_rate": number (0-100),
"vent": 0 or 1
}}}}

NO explanation. NO text. ONLY JSON."""

#TUSHAR
def parse_model_action(response_text: str) -> Action:
    """Parse the model response into a validated Action."""
    text = response_text.strip()

    try:
        return Action.model_validate_json(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return Action.model_validate_json(match.group(0))
        except Exception:
            pass

    raise ValueError("Model response was not valid strict JSON action output.")


def choose_action_with_llm(
    task_name: str,
    task_description: str,
    observation: Observation,
    state: EnvironmentState,
    last_reward: float | None,
) -> Action:
    """Query the configured LLM for the next action."""
    if CLIENT is None:
        raise RuntimeError("LLM mode requested without an initialized client.")

    if API_CALL_DELAY > 0:
        time.sleep(API_CALL_DELAY)

    try:
        response = CLIENT.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt(task_name, task_description)},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        observation=observation,
                        state=state,
                        last_reward=last_reward,
                    ),
                },
            ],
            temperature=0.4,
            top_p=0.9,
            response_format=Action,
        )
    except Exception as exc:
        print(
            f"Warning: LLM returned invalid action ({exc}). Using neutral fallback.",
            file=sys.stderr,
            flush=True,
        )
        return Action(steam_valve=50.0, reflux_ratio=50.0, feed_rate=50.0, vent=0)

    parsed = response.choices[0].message.parsed
    if parsed is None:
        print(
            "Warning: Model response did not contain a parsed Action. Using neutral fallback.",
            file=sys.stderr,
            flush=True,
        )
        return Action(steam_valve=50.0, reflux_ratio=50.0, feed_rate=50.0, vent=0)
    return parsed


def choose_action(
    task_name: str,
    task_description: str,
    observation: Observation,
    state: EnvironmentState,
    last_reward: float | None,
) -> Action:
    """Choose the next action using the configured agent mode."""
    return choose_action_with_llm(
        task_name=task_name,
        task_description=task_description,
        observation=observation,
        state=state,
        last_reward=last_reward,
    )


def run_episode(task_name: str, episode_index: int, seed: int) -> dict[str, Any]:
    """Run one deterministic episode and emit structured logs."""
    task = get_task_by_name(task_name)
    if task is None:
        raise ValueError(f"Unknown task '{task_name}'.")

    reset_memory()

    runtime = OpenEnvRuntime()
    observation = runtime.reset(task_name=task_name, seed=seed)
    last_reward: float | None = None
    cumulative_reward = 0.0
    reward_values: list[float] = []
    done = False
    step_count = 0
    info: dict[str, Any] = {
        "task_success": False,
        "task_failed": False,
        "failure_reason": "",
        "step_count": 0,
    }

    print(f"[START] task={task_name} env=openenv model={MODEL_NAME}", flush=True)

    while not done:
        state = runtime.state()
        action = choose_action(
            task_name=task_name,
            task_description=task.description,
            observation=observation,
            state=state,
            last_reward=last_reward,
        )
        observation, reward, done, info = runtime.step(action)
        memory["reward_history"].append(reward.value)

        # Append a compact summary to the rolling history window
        step_summary = (
            f"T:{observation.temperature:.1f} P:{observation.pressure:.2f} "
            f"Pur:{observation.purity:.1f} | "
            f"Act(S:{action.steam_valve:.1f} R:{action.reflux_ratio:.1f} "
            f"F:{action.feed_rate:.1f} V:{action.vent}) -> R:{reward.value:.3f}"
        )
        memory["recent_steps"].append(step_summary)

        if reward.value > memory["best_reward"]:
            memory["best_reward"] = reward.value
            memory["best_action"] = action

        cumulative_reward += reward.value
        reward_values.append(reward.value)
        last_reward = reward.value
        step_count += 1

        action_str = (
            f"steam={action.steam_valve:.4f},"
            f"reflux={action.reflux_ratio:.4f},"
            f"feed={action.feed_rate:.4f},"
            f"vent={int(action.vent)}"
        )
        done_str = str(bool(done)).lower()
        print(
            f"[STEP] step={step_count} action={action_str} "
            f"reward={reward.value:.2f} done={done_str} error=null",
            flush=True,
        )

    score = compute_episode_score(
        cumulative_reward=cumulative_reward,
        success=bool(info.get("task_success", False)),
        failed=bool(info.get("task_failed", False)),
        steps=step_count,
        max_steps=task.max_steps,
    )

    result = {
        "task": task_name,
        "episode": episode_index + 1,
        "seed": seed,
        "steps": step_count,
        "success": bool(info.get("task_success", False)),
        "failed": bool(info.get("task_failed", False)),
        "failure_reason": str(info.get("failure_reason", "")),
        "cumulative_reward": cumulative_reward,
        "score": score,
        "final_observation": compact_observation(observation),
    }

    success = str(bool(info.get("task_success", False))).lower()
    reward_list = ",".join(f"{reward_value:.2f}" for reward_value in reward_values)
    print(f"[END] success={success} steps={step_count} rewards={reward_list}", flush=True)
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode results into task scores and an overall score."""
    task_scores: dict[str, float] = {}

    for task_name in [task.name for task in get_all_tasks()]:
        task_results = [result for result in results if result["task"] == task_name]
        if not task_results:
            continue
        task_scores[task_name] = sum(result["score"] for result in task_results) / len(task_results)

    total_weight = sum(TASK_WEIGHTS[name] for name in task_scores)
    weighted_total = sum(task_scores[name] * TASK_WEIGHTS[name] for name in task_scores)
    overall_score = weighted_total / total_weight if total_weight > 0 else 0.0

    return {
        "kind": "summary",
        "mode": AGENT_MODE,
        "model": MODEL_NAME,
        "episodes_per_task": OPENENV_EPISODES_PER_TASK,
        "overall_score": round(overall_score, 6),
        "task_scores": {name: round(score, 6) for name, score in task_scores.items()},
    }

def agent_action(obs: Observation, state: EnvironmentState, last_reward: float | None):
    return choose_action(
        task_name="optimization",   # or pass dynamically later
        task_description="",
        observation=obs,
        state=state,
        last_reward=last_reward,
    )
def main() -> None:
    """Run the configured controller across all tasks."""
    all_results: list[dict[str, Any]] = []
    ordered_tasks = [task.name for task in get_all_tasks()]

    for task_index, task_name in enumerate(ordered_tasks):
        for episode_index in range(OPENENV_EPISODES_PER_TASK):
            seed = OPENENV_BASE_SEED + task_index * 10_000 + episode_index
            all_results.append(
                run_episode(
                    task_name=task_name,
                    episode_index=episode_index,
                    seed=seed,
                )
            )


if __name__ == "__main__":
    main()
