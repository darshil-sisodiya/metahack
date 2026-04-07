"""OpenEnv API - Main FastAPI Application."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure repo root is importable
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from app.models import Action, EnvironmentState, Observation, Reward
from app.runtime import OpenEnvRuntime


class ResetRequest(BaseModel):
    task_name: str = "optimization"
    seed: int = 42


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


app = FastAPI(
    title="OpenEnv",
    description="OpenEnv Environment API",
    version="0.1.0",
)

# Shared runtime
runtime = OpenEnvRuntime()


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
async def reset_env(request: ResetRequest | None = None) -> Observation:
    task_name = request.task_name if request else "optimization"
    seed = request.seed if request else 42

    try:
        return runtime.reset(task_name=task_name, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse)
async def step_env(action: Action) -> StepResponse:
    try:
        observation, reward, done, info = runtime.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=EnvironmentState)
@app.post("/state", response_model=EnvironmentState)
async def state_env() -> EnvironmentState:
    try:
        return runtime.state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
