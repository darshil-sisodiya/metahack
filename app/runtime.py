"""High-level OpenEnv runtime exposing the standard reset/step/state API."""

from __future__ import annotations

from typing import Any

from app.env import DistillationEnv
from app.models import Action, EnvironmentState, Observation, Reward
from app.tasks import BaseTask, get_all_tasks, get_task_by_name


class OpenEnvRuntime:
    """Task-aware runtime that wraps the base simulator with OpenEnv semantics."""

    def __init__(self, default_task_name: str = "optimization", default_seed: int = 42) -> None:
        self.default_task_name = default_task_name
        self.default_seed = default_seed
        self.env: DistillationEnv | None = None
        self.task: BaseTask | None = None
        self.task_name: str | None = None
        self.seed: int = default_seed

    def available_tasks(self) -> list[str]:
        """Return the list of supported task names."""
        return [task.name for task in get_all_tasks()]

    def reset(self, task_name: str | None = None, seed: int | None = None) -> Observation:
        """Reset the runtime and return the initial observation for the selected task."""
        resolved_task_name = task_name or self.default_task_name
        resolved_task = get_task_by_name(resolved_task_name)
        if resolved_task is None:
            raise ValueError(f"Unknown task '{resolved_task_name}'.")

        resolved_seed = self.default_seed if seed is None else seed
        self.task_name = resolved_task_name
        self.task = resolved_task
        self.seed = resolved_seed
        self.env = DistillationEnv(seed=resolved_seed, max_steps=resolved_task.max_steps)
        return resolved_task.reset(self.env)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Advance the selected task by one step."""
        if self.env is None or self.task is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        observation, reward, done, info = self.task.step(self.env, action)

        # Nuclear clamp: guarantee reward.value is strictly in (0, 1)
        reward.value = max(0.1, min(0.99, float(reward.value)))

        # Clamp any score-like values that may exist in the info dict
        for key in ("score", "task_score", "episode_score"):
            if key in info:
                info[key] = max(0.1, min(0.99, float(info[key])))

        return observation, reward, done, info

    def state(self) -> EnvironmentState:
        """Return the current runtime state as a typed model."""
        if self.env is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        raw_state = self.env.state()
        return EnvironmentState(
            active_task=self.task_name,
            temperature=float(raw_state["temperature"]),
            pressure=float(raw_state["pressure"]),
            purity=float(raw_state["purity"]),
            flow_rate=float(raw_state["flow_rate"]),
            energy_usage=float(raw_state["energy_usage"]),
            time_step=int(raw_state["time_step"]),
            hidden_instability=float(raw_state["hidden_instability"]),
            cooling_failure=bool(raw_state["cooling_failure"]),
            pressure_spike=bool(raw_state["pressure_spike"]),
            prev_action=(
                Action.model_validate(raw_state["prev_action"])
                if raw_state["prev_action"] is not None
                else None
            ),
        )
