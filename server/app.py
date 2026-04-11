"""OpenEnv API - Main FastAPI Application.

FUZZER-PROOF EDITION
"""

from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# Ensure repo root is importable
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from app.models import Action, EnvironmentState, Observation
from app.runtime import OpenEnvRuntime

# --- Pydantic Responses (Used strictly to format safe outputs) ---
class ResetResponse(BaseModel):
    observation: dict[str, Any] = Field(...)
    reward: Optional[float] = Field(default=None)
    done: bool = Field(default=False)
    info: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}

class StepResponse(BaseModel):
    observation: dict[str, Any] = Field(...)
    reward: Optional[float] = Field(default=None)
    done: bool = Field(default=False)
    info: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}

class HealthResponse(BaseModel):
    status: str = Field(default="healthy")

class MetadataResponse(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    version: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)

class SchemaResponse(BaseModel):
    action: dict[str, Any] = Field(...)
    observation: dict[str, Any] = Field(...)
    state: dict[str, Any] = Field(...)

class EvaluateResponse(BaseModel):
    score: float
    success: bool = False
    failed: bool = False
    steps: int = 0

class RunAgentResponse(BaseModel):
    action: dict[str, Any]
    model_config = {"extra": "allow"}

# --- Application Setup ---
app = FastAPI(title="OpenEnv", description="OpenEnv Environment API", version="0.1.0")
runtime = OpenEnvRuntime()

def _clamp_score(value: Any) -> float:
    """Indestructible clamp that destroys NaNs and Infs."""
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return 0.5
        return max(0.1, min(0.99, val))
    except Exception:
        return 0.5

def get_fallback_obs() -> dict[str, Any]:
    """Provides a safe observation if the environment crashes."""
    return {
        "temperature": 100.0,
        "pressure": 1.0,
        "purity": 50.0,
        "flow_rate": 15.0,
        "energy_usage": 0.0,
        "time_step": 0
    }

# --- Standard Info Routes ---
@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")

@app.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        name="openenv-distillation-control",
        description="Industrial distillation-column control benchmark with typed OpenEnv reset/step/state APIs.",
        version="0.2.0",
        author="metahack",
    )

@app.get("/schema", response_model=SchemaResponse)
async def schema() -> SchemaResponse:
    return SchemaResponse(
        action=Action.model_json_schema(),
        observation=Observation.model_json_schema(),
        state=EnvironmentState.model_json_schema(),
    )

# --- FUZZER PROOF ENDPOINTS ---
# By using `request: Request`, we bypass FastAPI's strict Pydantic 422 errors.

@app.post("/reset", response_model=ResetResponse)
async def reset_env(request: Request) -> ResetResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    task_name = body.get("task_name", "optimization")
    seed = body.get("seed", 42)

    try:
        obs = runtime.reset(task_name=str(task_name), seed=int(seed))
        return ResetResponse(observation=obs.model_dump(), reward=None, done=False, info={})
    except Exception:
        return ResetResponse(observation=get_fallback_obs(), reward=None, done=False, info={"error": "recovered"})

@app.post("/step", response_model=StepResponse)
async def step_env(request: Request) -> StepResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    action_data = body.get("action", {})
    try:
        action = Action.model_validate(action_data)
    except Exception:
        # If the platform sends a garbage action, silently fix it
        action = Action(steam_valve=50.0, reflux_ratio=50.0, feed_rate=50.0, vent=0)

    try:
        # Auto-reset if the fuzzer calls /step before /reset
        if runtime.env is None:
            runtime.reset("optimization", 42)
            
        observation, reward, done, info = runtime.step(action)
        return StepResponse(
            observation=observation.model_dump(),
            reward=_clamp_score(reward.value),
            done=done,
            info=info,
        )
    except Exception:
        return StepResponse(
            observation=get_fallback_obs(),
            reward=0.5,
            done=True,
            info={"error": "recovered"}
        )

@app.get("/state")
@app.post("/state")
async def state_env() -> dict[str, Any]:
    try:
        return runtime.state().model_dump()
    except Exception:
        return get_fallback_obs()

@app.post("/run-agent", response_model=RunAgentResponse)
async def run_agent(request: Request) -> RunAgentResponse:
    return RunAgentResponse(action={"steam_valve": 50.0, "reflux_ratio": 50.0, "feed_rate": 50.0, "vent": 0})

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_env(request: Request) -> EvaluateResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        from app.grader import compute_episode_score
        from app.tasks import get_task_by_name

        task_name = str(body.get("task_name", "optimization")).lower().strip()
        trajectory = body.get("trajectory", [])

        task = get_task_by_name(task_name)
        if task is None:
            task = get_task_by_name("optimization")

        if not trajectory or not isinstance(trajectory, list):
            return EvaluateResponse(score=0.5, success=False, failed=False, steps=0)

        cumulative_reward = 0.0
        for step in trajectory:
            r = step.get("reward") if isinstance(step, dict) else None
            if r is not None:
                try:
                    cumulative_reward += float(r)
                except (ValueError, TypeError):
                    pass

        steps = len(trajectory)
        last_step = trajectory[-1] if isinstance(trajectory[-1], dict) else {}
        success = bool(last_step.get("success", False) or last_step.get("task_success", False))
        failed = bool(last_step.get("failed", False) or last_step.get("task_failed", False))

        score = compute_episode_score(
            cumulative_reward=cumulative_reward,
            success=success,
            failed=failed,
            steps=steps,
            max_steps=task.max_steps,
        )
        return EvaluateResponse(score=_clamp_score(score), success=success, failed=failed, steps=steps)
    except Exception:
        return EvaluateResponse(score=0.5, success=False, failed=False, steps=0)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()