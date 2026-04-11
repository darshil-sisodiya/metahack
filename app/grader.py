"""OpenEnv - Grader Module.

Provides deterministic evaluation utilities for DistillationEnv tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.env import DistillationEnv
from app.models import Action, Observation
from app.tasks import BaseTask, get_all_tasks

# Weighted contribution of each task to the final overall score.
TASK_WEIGHTS: dict[str, float] = {
    "stabilization": 0.25,
    "optimization": 0.35,
    "emergency_control": 0.4,
}

SCORE_MIN = 0.1
SCORE_MAX = 0.99
SUBMISSION_ROUND_DIGITS = 6


def random_agent(obs: Observation) -> Action:
    """Return a simple deterministic baseline action.

    The baseline keeps all continuous controls neutral and opens the vent
    once pressure crosses a basic safety threshold.
    """
    return Action(
        steam_valve=50.0,
        reflux_ratio=50.0,
        feed_rate=50.0,
        vent=1 if obs.pressure > 2.2 else 0,
    )


def clamp_open_score(value: float) -> float:
    """Clamp a score into the strict-open range (0, 1)."""
    return max(SCORE_MIN, min(SCORE_MAX, float(value)))


def to_submission_score(value: float) -> float:
    """Return a rounded score guaranteed to stay strictly inside (0, 1)."""
    rounded = round(clamp_open_score(value), SUBMISSION_ROUND_DIGITS)
    if rounded <= 0.0:
        return 10 ** (-SUBMISSION_ROUND_DIGITS)
    if rounded >= 1.0:
        return 1.0 - (10 ** (-SUBMISSION_ROUND_DIGITS))
    return rounded


def validate_strict_open_scores(task_scores: dict[str, float], overall_score: float) -> None:
    """Raise when any emitted score violates strict-open validator constraints."""
    missing_tasks = [task_name for task_name in TASK_WEIGHTS if task_name not in task_scores]
    if missing_tasks:
        raise ValueError(f"Missing task scores for: {', '.join(sorted(missing_tasks))}")

    for task_name, score in task_scores.items():
        if not (0.0 < score < 1.0):
            raise ValueError(
                f"Task score for '{task_name}' must be strictly between 0 and 1. Got: {score}"
            )

    if not (0.0 < overall_score < 1.0):
        raise ValueError(
            f"Overall score must be strictly between 0 and 1. Got: {overall_score}"
        )


def _coerce_action(action: Action | dict[str, Any]) -> Action:
    """Convert agent output into a validated Action instance."""
    if isinstance(action, Action):
        return action
    return Action.model_validate(action)


def _compute_performance_score(cumulative_reward: float, max_steps: int) -> float:
    """Normalize episode reward into the 0.0-0.6 performance range.

    Reward is normalized by task horizon so episodes from tasks with different
    lengths remain comparable. The normalized value is then mapped from a
    clipped [-1, 1] band into [0.0, 0.6].
    """
    if max_steps <= 0:
        return 0.0001

    normalized_reward = cumulative_reward / max_steps
    clipped_reward = max(-1.0, min(1.0, normalized_reward))
    return ((clipped_reward + 1.0) / 2.0) * 0.6


def _compute_efficiency_bonus(success: bool, steps: int, max_steps: int) -> float:
    """Return an early-success bonus in the 0.0-0.1 range."""
    if not success or max_steps <= 1:
        return 0.0001

    remaining_steps = max(max_steps - steps, 0)
    return (remaining_steps / max_steps) * 0.1


def _compute_episode_score(
    cumulative_reward: float,
    success: bool,
    failed: bool,
    steps: int,
    max_steps: int,
) -> float:
    """Compute the final clamped episode score in the required 0.0-1.0 range."""
    performance_score = _compute_performance_score(cumulative_reward, max_steps)
    success_bonus = 0.3 if success else 0.0001
    efficiency_bonus = _compute_efficiency_bonus(success, steps, max_steps)

    score = performance_score + success_bonus + efficiency_bonus
    if failed:
        # Failed episodes still receive some partial credit for strong control.
        score = min(score, 0.3 + performance_score * 0.5)

    return clamp_open_score(score)


def compute_episode_score(
    cumulative_reward: float,
    success: bool,
    failed: bool,
    steps: int,
    max_steps: int,
) -> float:
    """Public scorer used by both the grader and the submission inference script."""
    return _compute_episode_score(
        cumulative_reward=cumulative_reward,
        success=success,
        failed=failed,
        steps=steps,
        max_steps=max_steps,
    )


def _run_episode(
    task: BaseTask,
    seed: int,
    agent_fn: Callable[[Observation], Action | dict[str, Any]],
) -> dict[str, Any]:
    """Run one seeded episode for a task using the provided policy."""
    env = DistillationEnv(seed=seed, max_steps=task.max_steps)
    obs = task.reset(env)

    cumulative_reward = 0.0
    done = False
    info: dict[str, Any] = {
        "task_success": False,
        "task_failed": False,
        "failure_reason": "",
        "step_count": 0,
    }

    while not done:
        action = _coerce_action(agent_fn(obs))
        obs, reward, done, info = task.step(env, action)
        cumulative_reward += reward.value

    score = compute_episode_score(
        cumulative_reward=cumulative_reward,
        success=bool(info.get("task_success", False)),
        failed=bool(info.get("task_failed", False)),
        steps=int(info.get("step_count", 0)),
        max_steps=task.max_steps,
    )

    return {
        "score": score,
        "success": bool(info.get("task_success", False)),
        "failed": bool(info.get("task_failed", False)),
        "steps": int(info.get("step_count", 0)),
        "final_obs": obs,
        "failure_reason": str(info.get("failure_reason", "")),
    }


def evaluate_episode(task: BaseTask, seed: int) -> dict[str, Any]:
    """Evaluate a single task episode with the built-in baseline policy.

    Args:
        task: Task instance to evaluate.
        seed: Deterministic seed for the environment and task rollout.

    Returns:
        Dictionary with the final score, success/failure flags, step count,
        final observation, and failure reason.
    """
    return _run_episode(task=task, seed=seed, agent_fn=random_agent)


def evaluate_all_tasks(
    agent_fn: Callable[[Observation], Action | dict[str, Any]],
    n_episodes: int = 20,
) -> dict[str, Any]:
    """Evaluate an agent across all tasks with deterministic seeded rollouts.

    Args:
        agent_fn: Policy callable mapping Observation -> Action.
        n_episodes: Number of seeded episodes to run per task.

    Returns:
        Dictionary containing weighted overall score, per-task average scores,
        and detailed per-task evaluation metrics.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    task_scores: dict[str, float] = {}
    details: dict[str, Any] = {}

    for task_index, task_template in enumerate(get_all_tasks()):
        task_name = task_template.name
        episode_results: list[dict[str, Any]] = []

        for episode_index in range(n_episodes):
            # Keep seeds deterministic and stable even if task ordering changes.
            seed = task_index * 10_000 + episode_index
            task_instance = type(task_template)()
            episode_results.append(_run_episode(task_instance, seed, agent_fn))

        avg_score = sum(result["score"] for result in episode_results) / n_episodes
        avg_score = clamp_open_score(avg_score)
        success_rate = sum(1 for result in episode_results if result["success"]) / n_episodes
        failure_rate = sum(1 for result in episode_results if result["failed"]) / n_episodes
        avg_steps = sum(result["steps"] for result in episode_results) / n_episodes

        task_scores[task_name] = avg_score
        details[task_name] = {
            "avg_score": avg_score,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "avg_steps": avg_steps,
            "episodes": episode_results,
        }

    weighted_total = sum(task_scores[name] * TASK_WEIGHTS[name] for name in task_scores)
    total_weight = sum(TASK_WEIGHTS[name] for name in task_scores)
    overall_score = weighted_total / total_weight if total_weight > 0 else 0.5
    overall_score = clamp_open_score(overall_score)

    return {
        "overall_score": overall_score,
        "task_scores": task_scores,
        "details": details,
    }
