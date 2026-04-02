"""OpenEnv API - Main FastAPI Application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Support running `uvicorn main:app` from inside the `app` directory by
# making the repository root importable before resolving `app.*` modules.
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from app.models import Action, EnvironmentState, Observation, Reward
from app.runtime import OpenEnvRuntime


class ResetRequest(BaseModel):
    """Optional reset payload for selecting the active task."""

    task_name: str = "optimization"
    seed: int = 42


class StepResponse(BaseModel):
    """Response payload returned by the `/step` endpoint."""

    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


app = FastAPI(
    title="OpenEnv",
    description="OpenEnv Environment API",
    version="0.1.0",
)

# Single shared runtime instance for the running server.
runtime = OpenEnvRuntime()


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
    task_name = request.task_name if request is not None else "optimization"
    seed = request.seed if request is not None else 42

    try:
        return runtime.reset(task_name=task_name, seed=seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
async def step_env(action: Action) -> StepResponse:
    """Advance the current environment by one step."""
    try:
        observation, reward, done, info = runtime.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=EnvironmentState)
@app.post("/state", response_model=EnvironmentState)
async def state_env() -> EnvironmentState:
    """Return the current environment state."""
    try:
        return runtime.state()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main():
    """Entry point for the OpenEnv validation script."""
    import uvicorn
    # 0.0.0.0 is strictly required for Hugging Face Spaces
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
