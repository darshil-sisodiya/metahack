"""Submission inference script for the OpenEnv distillation benchmark.

The required submission entry point is a root-level `inference.py` file that:

- uses the OpenAI-compatible client for LLM inference in submission mode
- reads API configuration from environment variables
- runs deterministic seeded rollouts across all tasks
- emits structured stdout logs using [START], [STEP], and [END] markers
- produces normalized 0.0-1.0 scores for each task and an overall score

For local offline smoke testing, the script also supports a deterministic
heuristic mode via `OPENENV_AGENT_MODE=heuristic`. Submission mode defaults
to `llm` and will fail fast if the API configuration is missing.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.grader import TASK_WEIGHTS, compute_episode_score
from app.models import Action, EnvironmentState, Observation
from app.runtime import OpenEnvRuntime
from app.tasks import get_all_tasks, get_task_by_name

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
API_KEY = HF_TOKEN or OPENAI_API_KEY or os.environ.get("API_KEY", "").strip()

OPENENV_AGENT_MODE = os.environ.get("OPENENV_AGENT_MODE", "llm").strip().lower()
OPENENV_EPISODES_PER_TASK = max(1, int(os.environ.get("OPENENV_EPISODES_PER_TASK", "1")))
OPENENV_BASE_SEED = int(os.environ.get("OPENENV_BASE_SEED", "42"))
OPENENV_REQUEST_TIMEOUT = float(os.environ.get("OPENENV_REQUEST_TIMEOUT", "60"))
API_CALL_DELAY = max(0.0, float(os.environ.get("API_CALL_DELAY", "0")))

SAFE_FALLBACK_ACTION = Action(
    steam_valve=50.0,
    reflux_ratio=50.0,
    feed_rate=50.0,
    vent=0,
)


def resolve_agent_mode() -> str:
    """Resolve the requested agent mode."""
    if OPENENV_AGENT_MODE not in {"llm", "heuristic"}:
        raise ValueError(
            "OPENENV_AGENT_MODE must be one of: llm or heuristic."
        )
    return OPENENV_AGENT_MODE


AGENT_MODE = resolve_agent_mode()


def build_client() -> OpenAI | None:
    """Create the OpenAI-compatible client when running in LLM mode."""
    if AGENT_MODE != "llm":
        return None
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL must be set when OPENENV_AGENT_MODE=llm.")
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME must be set when OPENENV_AGENT_MODE=llm.")
    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN or OPENAI_API_KEY must be set when OPENENV_AGENT_MODE=llm."
        )

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=OPENENV_REQUEST_TIMEOUT,
    )


CLIENT = build_client()


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
    """Build the fixed system prompt for the LLM controller."""
    return "\n".join(
        [
            "You are controlling a distillation column in a step-by-step simulation.",
            "Follow the user's control instructions exactly.",
            "Output strict JSON only.",
        ]
    )


def build_user_prompt(
    observation: Observation,
    state: EnvironmentState,
    last_reward: float | None,
) -> str:
    """Build the step prompt from the current observation and state."""
    return f"""You are controlling a distillation column in a step-by-step simulation.

You MUST adapt your actions based on the current state. Repeating the same action across steps is NOT allowed unless the state is unchanged.

CURRENT STATE:

* Temperature: {observation.temperature:.4f}
* Pressure: {observation.pressure:.4f}
* Purity: {observation.purity:.4f}
* Flow rate: {observation.flow_rate:.4f}

CONTROL OBJECTIVES:

* Keep temperature near 90
* Keep pressure below 2.5 (CRITICAL: vent if high)
* Increase purity above 60
* Maintain stable operation

CONTROL RULES (IMPORTANT):

* If pressure > 2.2 -> set vent = 1 and reduce steam
* If temperature > 95 -> decrease steam
* If temperature < 85 -> increase steam
* If purity < 55 -> increase reflux
* If purity > 65 -> reduce reflux
* If flow_rate too low -> increase feed
* If pressure rising -> reduce feed

ANTI-LAZY RULE:

* Do NOT repeat identical actions across steps if the state has changed
* Adjust at least one control variable every step

OUTPUT FORMAT (STRICT JSON ONLY):
{{
"steam_valve": number (0-100),
"reflux_ratio": number (0-100),
"feed_rate": number (0-100),
"vent": 0 or 1
}}

NO explanations. NO text. ONLY JSON."""


def parse_model_action(response_text: str, fallback_action: Action) -> Action:
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

    return fallback_action


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

    response_text = ""
    try:
        response = CLIENT.chat.completions.create(
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
            max_tokens=128,
            temperature=0.0,
        )
        response_text = response.choices[0].message.content or ""
    except Exception:
        return SAFE_FALLBACK_ACTION

    return parse_model_action(
        response_text=response_text,
        fallback_action=SAFE_FALLBACK_ACTION,
    )


def choose_action(
    task_name: str,
    task_description: str,
    observation: Observation,
    state: EnvironmentState,
    last_reward: float | None,
) -> Action:
    """Choose the next action using the configured agent mode."""
    if AGENT_MODE == "heuristic":
        # Explicit offline/debug mode only. Submission mode should use `llm`.
        steam_valve = max(30.0, min(70.0, 50.0 + (90.0 - observation.temperature)))
        vent = 1 if observation.pressure > 2.0 else 0
        return Action(
            steam_valve=steam_valve,
            reflux_ratio=50.0,
            feed_rate=50.0,
            vent=vent,
        )
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

    model_name = MODEL_NAME if AGENT_MODE == "llm" else "heuristic-baseline"
    print(f"[START] task={task_name} env=openenv model={model_name}", flush=True)

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
        "model": MODEL_NAME if AGENT_MODE == "llm" else "heuristic-baseline",
        "episodes_per_task": OPENENV_EPISODES_PER_TASK,
        "overall_score": round(overall_score, 6),
        "task_scores": {name: round(score, 6) for name, score in task_scores.items()},
    }


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
