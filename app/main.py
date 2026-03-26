"""OpenEnv API - Main FastAPI Application."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.env import DistillationEnv
from app.models import Action, Observation
from app.tasks import BaseTask, get_task_by_name


class ResetRequest(BaseModel):
    """Optional reset payload for selecting the active task."""

    task_name: str = "optimization"


class StepResponse(BaseModel):
    """Response payload returned by the `/step` endpoint."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


app = FastAPI(
    title="OpenEnv",
    description="OpenEnv Environment API",
    version="0.1.0",
)

# Single shared environment instance for the running server.
env: DistillationEnv | None = None
task: BaseTask | None = None


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint mirrors the health response for convenience."""
    return {"status": "ok"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
async def reset_env(request: ResetRequest | None = None) -> Observation:
    """Create a new seeded environment/task instance and return the first observation."""
    global env, task

    task_name = request.task_name if request is not None else "optimization"
    task = get_task_by_name(task_name)
    if task is None:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'.")

    env = DistillationEnv(seed=42)
    return task.reset(env)


@app.post("/step", response_model=StepResponse)
async def step_env(action: Action) -> StepResponse:
    """Advance the current environment by one step."""
    if env is None or task is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    observation, reward, done, info = task.step(env, action)
    return StepResponse(
        observation=observation,
        reward=reward.value,
        done=done,
        info=info,
    )
