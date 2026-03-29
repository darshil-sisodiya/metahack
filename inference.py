"""OpenEnv Inference Script — Hackathon Submission

Drives all three distillation-column tasks (stabilization, optimization,
emergency_control) with an LLM agent via the OpenAI-compatible chat API.

Required environment variables:
    API_BASE_URL  – OpenAI-compatible endpoint (e.g. vLLM / TGI)
    MODEL_NAME    – HuggingFace model identifier served at that endpoint
    HF_TOKEN      – HuggingFace token used as the API key

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import time
import logging
import os
import re
import sys
import textwrap
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Load .env before any os.environ.get() calls

from openai import OpenAI

from app.env import DistillationEnv
from app.models import Action, Observation
from app.tasks import get_all_tasks, get_task_by_name

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openenv-inference")

# ──────────────────────────────────────────────────────────────────────
# Configuration from environment
# ──────────────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", ""))

if not API_BASE_URL or not MODEL_NAME:
    log.warning(
        "API_BASE_URL and/or MODEL_NAME not set — "
        "falling back to defaults for local testing."
    )
    API_BASE_URL = API_BASE_URL or "http://localhost:8000/v1"
    MODEL_NAME = MODEL_NAME or "default-model"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)

# ──────────────────────────────────────────────────────────────────────
# Fallback action (safe neutral position)
# ──────────────────────────────────────────────────────────────────────
FALLBACK_ACTION = Action(
    steam_valve=50.0,
    reflux_ratio=50.0,
    feed_rate=50.0,
    vent=0,
)

# ──────────────────────────────────────────────────────────────────────
# System prompt — tailored to distillation-column control
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are an expert process-control AI operating a chemical distillation column.

    ── OBSERVATION SPACE (what you receive each step) ───────────────────
    • temperature  (float, °C)        — column temperature
    • pressure     (float, bar)       — system pressure
    • purity       (float, 0-100 %)   — product purity
    • flow_rate    (float, L/min)     — throughput flow rate
    • energy_usage (float, kWh)       — cumulative energy consumed
    • time_step    (int)              — current simulation step

    ── ACTION SPACE (what you must output) ──────────────────────────────
    • steam_valve   (float, 0–100)  — steam valve opening %
    • reflux_ratio  (float, 0–100)  — reflux ratio %
    • feed_rate     (float, 0–100)  — feed rate %
    • vent          (int,   0 or 1) — pressure-relief vent (0=closed, 1=open)

    ── SAFETY LIMITS ────────────────────────────────────────────────────
    • Pressure ≥ 2.8 bar → emergency shutdown (FAILURE)
    • Temperature ≥ 145 °C → emergency shutdown (FAILURE)
    • Keep pressure below ~2.0 bar and temperature near 85-115 °C.
    • Open the vent (vent=1) when pressure rises above ~1.8 bar.

    ── CONTROL STRATEGY GUIDELINES ──────────────────────────────────────
    • For STABILIZATION: keep temperature near 100 °C. Use steam_valve ~45-55.
    • For OPTIMIZATION: balance purity (raise reflux_ratio) with energy
      efficiency (moderate steam_valve). Keep flow_rate moderate (~50-60).
    • For EMERGENCY CONTROL: prioritise reducing pressure — open vent,
      lower steam_valve (< 40), and moderate feed_rate. Once pressure
      is below 1.5 bar, close vent and stabilise near 100 °C.

    ── OUTPUT FORMAT ────────────────────────────────────────────────────
    Respond with ONLY a JSON object on a single line. No explanation,
    no markdown fences, no extra text.

    Example:
    {"steam_valve": 50.0, "reflux_ratio": 60.0, "feed_rate": 55.0, "vent": 0}
""")


# ──────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────

def build_user_prompt(
    step: int,
    observation: Observation,
    task_name: str,
    history: list[dict[str, Any]] | None = None,
    reward_value: float | None = None,
) -> str:
    """Convert a simulation Observation into a readable user prompt."""
    lines = [
        f"Task: {task_name}",
        f"Step: {step}",
        "",
        "Current Observation:",
        f"  temperature  = {observation.temperature:.2f} °C",
        f"  pressure     = {observation.pressure:.3f} bar",
        f"  purity       = {observation.purity:.2f} %",
        f"  flow_rate    = {observation.flow_rate:.2f} L/min",
        f"  energy_usage = {observation.energy_usage:.2f} kWh",
        f"  time_step    = {observation.time_step}",
    ]

    if reward_value is not None:
        lines.append(f"\nLast reward: {reward_value:.4f}")

    # Include a summary of recent history so the LLM can reason about trends
    if history and len(history) >= 2:
        prev = history[-2]
        lines += [
            "",
            "Previous step observation:",
            f"  temperature  = {prev['temperature']:.2f} °C",
            f"  pressure     = {prev['pressure']:.3f} bar",
            f"  purity       = {prev['purity']:.2f} %",
            f"  flow_rate    = {prev['flow_rate']:.2f} L/min",
        ]

    lines.append("\nProvide your action as a JSON object:")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Action parsing
# ──────────────────────────────────────────────────────────────────────

def parse_model_action(response_text: str) -> Action:
    """Extract JSON from model response and parse into an Action.

    Tries multiple extraction strategies:
      1. Direct JSON parse of the full response
      2. Regex extraction of {...}
      3. Falls back to FALLBACK_ACTION on any failure
    """
    text = response_text.strip()

    # Strategy 1: direct parse (model followed instructions perfectly)
    try:
        data = json.loads(text)
        return _validate_action(data)
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 2: extract first JSON object via regex
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_action(data)
        except (json.JSONDecodeError, Exception):
            pass

    # Strategy 3: try to find JSON in markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1))
            return _validate_action(data)
        except (json.JSONDecodeError, Exception):
            pass

    log.warning("Failed to parse action from LLM response, using fallback.")
    log.debug("Raw response: %s", text[:300])
    return FALLBACK_ACTION


def _validate_action(data: dict) -> Action:
    """Validate and clamp raw dict into a safe Action."""
    return Action(
        steam_valve=max(0.0, min(100.0, float(data.get("steam_valve", 50.0)))),
        reflux_ratio=max(0.0, min(100.0, float(data.get("reflux_ratio", 50.0)))),
        feed_rate=max(0.0, min(100.0, float(data.get("feed_rate", 50.0)))),
        vent=1 if int(data.get("vent", 0)) == 1 else 0,
    )


# ──────────────────────────────────────────────────────────────────────
# LLM call helper
# ──────────────────────────────────────────────────────────────────────

# Delay between API calls (seconds). Set to 15 to stay under 5 RPM free-tier limits.
API_CALL_DELAY: int = 15


def call_llm(messages: list[dict[str, str]]) -> str:
    """Send messages to the OpenAI-compatible endpoint and return the text."""
    log.info("Waiting %ds to respect free-tier rate limits...", API_CALL_DELAY)
    time.sleep(API_CALL_DELAY)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=256,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────

def run_episode(task_name: str, seed: int = 42) -> dict[str, Any]:
    """Run a single episode on the named task using the LLM agent.

    Returns a dict with score-related info matching the grader schema.
    """
    task = get_task_by_name(task_name)
    if task is None:
        log.error("Unknown task: %s", task_name)
        return {"error": f"Unknown task: {task_name}"}

    env = DistillationEnv(seed=seed, max_steps=task.max_steps)
    obs = task.reset(env)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    history: list[dict[str, Any]] = [obs.model_dump()]
    cumulative_reward: float = 0.0
    reward_value: float | None = None
    done = False
    step = 0
    info: dict[str, Any] = {}

    log.info(
        "▶ Starting episode  task=%s  seed=%d  max_steps=%d",
        task_name, seed, task.max_steps,
    )

    while not done:
        # Build prompt
        user_prompt = build_user_prompt(
            step=step,
            observation=obs,
            task_name=task_name,
            history=history,
            reward_value=reward_value,
        )
        messages.append({"role": "user", "content": user_prompt})

        # Query LLM
        raw_response = call_llm(messages)
        action = parse_model_action(raw_response)

        # Keep assistant message for conversational context
        messages.append({"role": "assistant", "content": raw_response or "{}"})

        # Step the environment
        obs, reward, done, info = task.step(env, action)
        cumulative_reward += reward.value
        reward_value = reward.value
        history.append(obs.model_dump())
        step += 1

        # Log progress periodically
        if step % 10 == 0 or done:
            log.info(
                "  step=%3d  temp=%.1f  press=%.2f  purity=%.1f  reward=%.4f  cum=%.4f%s",
                step,
                obs.temperature,
                obs.pressure,
                obs.purity,
                reward.value,
                cumulative_reward,
                "  [DONE]" if done else "",
            )

        # Trim context window to prevent token overflow (keep system + last 10 exchanges)
        max_context_messages = 1 + 10 * 2  # system + 10 user/assistant pairs
        if len(messages) > max_context_messages:
            messages = [messages[0]] + messages[-(max_context_messages - 1):]

    success = bool(info.get("task_success", False))
    failed = bool(info.get("task_failed", False))
    failure_reason = str(info.get("failure_reason", ""))

    status = "✓ SUCCESS" if success else ("✗ FAILED" if failed else "— NEUTRAL")
    log.info(
        "■ Episode finished  task=%s  status=%s  steps=%d  cumulative_reward=%.4f%s",
        task_name,
        status,
        step,
        cumulative_reward,
        f"  reason={failure_reason}" if failure_reason else "",
    )

    return {
        "task": task_name,
        "seed": seed,
        "steps": step,
        "cumulative_reward": cumulative_reward,
        "success": success,
        "failed": failed,
        "failure_reason": failure_reason,
        "final_observation": obs.model_dump(),
    }


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the LLM agent across all three tasks and report results."""
    log.info("=" * 70)
    log.info("OpenEnv LLM Inference — Hackathon Submission")
    log.info("=" * 70)
    log.info("API_BASE_URL : %s", API_BASE_URL)
    log.info("MODEL_NAME   : %s", MODEL_NAME)
    log.info("HF_TOKEN     : %s", "***" if HF_TOKEN else "(not set)")
    log.info("")

    task_names = ["stabilization", "optimization", "emergency_control"]
    n_episodes = 3  # episodes per task (increase for full evaluation)
    all_results: dict[str, list[dict[str, Any]]] = {}

    for task_name in task_names:
        log.info("━" * 70)
        log.info("TASK: %s", task_name.upper())
        log.info("━" * 70)

        results: list[dict[str, Any]] = []
        for ep in range(n_episodes):
            seed = ep * 1000 + 42
            result = run_episode(task_name, seed=seed)
            results.append(result)

        all_results[task_name] = results

        # Summary
        successes = sum(1 for r in results if r.get("success"))
        failures = sum(1 for r in results if r.get("failed"))
        avg_reward = sum(r.get("cumulative_reward", 0) for r in results) / len(results)
        avg_steps = sum(r.get("steps", 0) for r in results) / len(results)

        log.info(
            "  Summary: %d/%d success, %d/%d failed, "
            "avg_reward=%.4f, avg_steps=%.1f",
            successes, n_episodes,
            failures, n_episodes,
            avg_reward,
            avg_steps,
        )

    # Final summary
    log.info("")
    log.info("=" * 70)
    log.info("FINAL RESULTS")
    log.info("=" * 70)
    for task_name, results in all_results.items():
        successes = sum(1 for r in results if r.get("success"))
        avg_reward = sum(r.get("cumulative_reward", 0) for r in results) / len(results)
        log.info(
            "  %-20s  success=%d/%d  avg_reward=%.4f",
            task_name, successes, n_episodes, avg_reward,
        )
    log.info("=" * 70)


if __name__ == "__main__":
    main()
