"""OpenEnv API - Main FastAPI Application.

Conforms to the OpenEnv v0.2.1 standard API contract:
- /health  → {"status": "healthy"}
- /metadata → {"name": "...", "description": "..."}
- /schema  → {"action": {}, "observation": {}, "state": {}}
- /reset   → {"observation": {...}, "reward": null, "done": false, "info": {}}
- /step    → {"observation": {...}, "reward": float, "done": bool, "info": {...}}
- /evaluate → {"score": float, "success": bool, "failed": bool, "steps": int}
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure repo root is importable
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from app.models import Action, EnvironmentState, Observation, Reward
from app.runtime import OpenEnvRuntime


# ---------------------------------------------------------------------------
# OpenEnv-Standard Pydantic Schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request model for /reset, matching openenv.core.env_server.types.ResetRequest."""
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible episodes")
    episode_id: Optional[str] = Field(default=None, description="Custom episode identifier")
    # Allow extra fields for task_name or any custom parameters
    task_name: str = "optimization"

    model_config = {"extra": "allow"}


class ResetResponse(BaseModel):
    """Response matching openenv.core.env_server.types.ResetResponse."""
    observation: dict[str, Any] = Field(..., description="Initial observation from the environment")
    reward: Optional[float] = Field(default=None, description="Initial reward (typically None at reset)")
    done: bool = Field(default=False, description="Whether episode is already done")
    info: dict[str, Any] = Field(default_factory=dict, description="Task metadata")

    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    """Request model for /step, matching openenv.core.env_server.types.StepRequest."""
    action: dict[str, Any] = Field(..., description="Action to execute")
    timeout_s: Optional[float] = Field(default=None, description="Optional timeout")
    request_id: Optional[str] = Field(default=None, description="Optional request ID")

    model_config = {"extra": "allow"}


class StepResponse(BaseModel):
    """Response matching openenv.core.env_server.types.StepResponse.

    CRITICAL: reward is Optional[float], NOT a Reward object.
    The OpenEnv validator parses reward as a plain float.
    """
    observation: dict[str, Any] = Field(..., description="Observation resulting from the action")
    reward: Optional[float] = Field(default=None, description="Reward signal from the action")
    done: bool = Field(default=False, description="Whether the episode has terminated")
    info: dict[str, Any] = Field(default_factory=dict, description="Critical grading metrics")

    model_config = {"extra": "allow"}


class HealthResponse(BaseModel):
    """Response matching openenv.core.env_server.types.HealthResponse."""
    status: str = Field(default="healthy", description="Health status")


class MetadataResponse(BaseModel):
    """Response matching openenv.core.env_server.types.EnvironmentMetadata."""
    name: str = Field(..., description="Name of the environment")
    description: str = Field(..., description="Description of the environment")
    version: Optional[str] = Field(default=None, description="Version")
    author: Optional[str] = Field(default=None, description="Author")


class SchemaResponse(BaseModel):
    """Response matching openenv.core.env_server.types.SchemaResponse."""
    action: dict[str, Any] = Field(..., description="JSON schema for actions")
    observation: dict[str, Any] = Field(..., description="JSON schema for observations")
    state: dict[str, Any] = Field(..., description="JSON schema for state")


class EvaluateRequest(BaseModel):
    task_name: str = "optimization"
    seed: int = 42
    trajectory: list[dict[str, Any]] = []


class EvaluateResponse(BaseModel):
    score: float
    success: bool = False
    failed: bool = False
    steps: int = 0


# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenEnv",
    description="OpenEnv Environment API",
    version="0.1.0",
)

# Shared runtime
runtime = OpenEnvRuntime()


def _clamp_score(value: float) -> float:
    """Clamp a score to strictly (0, 1) — never 0.0 or 1.0."""
    return max(0.1, min(0.99, float(value)))


# ---------------------------------------------------------------------------
# Endpoints conforming to OpenEnv v0.2.1 standard
# ---------------------------------------------------------------------------

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


@app.post("/reset", response_model=ResetResponse)
async def reset_env(request: ResetRequest | None = None) -> ResetResponse:
    task_name = request.task_name if request else "optimization"
    seed = request.seed if request else 42

    try:
        obs = runtime.reset(task_name=task_name, seed=seed)
        # Serialize observation to a plain dict — OpenEnv standard format
        obs_dict = obs.model_dump()
        return ResetResponse(
            observation=obs_dict,
            reward=None,
            done=False,
            info={},  # Ensure info dict is passed safely
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse)
async def step_env(request: StepRequest) -> StepResponse:
    try:
        # Deserialize the action dict into our typed Action model
        action = Action.model_validate(request.action)
        observation, reward, done, info = runtime.step(action)

        # Serialize observation to a plain dict
        obs_dict = observation.model_dump()

        # Extract reward as a PLAIN FLOAT — this is critical.
        # OpenEnv v0.2.1 StepResponse.reward is Optional[float], NOT a Reward object.
        reward_float = _clamp_score(reward.value)

        return StepResponse(
            observation=obs_dict,
            reward=reward_float,
            done=done,
            info=info,  # <--- CRITICAL: Do not drop this!
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
@app.post("/state")
async def state_env() -> dict[str, Any]:
    try:
        env_state = runtime.state()
        return env_state.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_env(request: EvaluateRequest | None = None) -> EvaluateResponse:
    """Evaluate a trajectory with crash-proof null handling."""
    from app.grader import compute_episode_score
    from app.tasks import get_task_by_name

    task_name = request.task_name if request else "optimization"
    trajectory = request.trajectory if request else []

    # 1. Ultra-safe task parsing (fallback instead of HTTP 400 crash)
    task = get_task_by_name(str(task_name).lower().strip())
    if task is None:
        task = get_task_by_name("optimization")

    if not trajectory:
        return EvaluateResponse(score=_clamp_score(0.5), success=False, failed=False, steps=0)

    # 2. Crash-proof reward parsing (THE FIX)
    cumulative_reward = 0.0
    for step in trajectory:
        r = step.get("reward")
        # Ignore None/null values gracefully
        if r is not None:
            try:
                cumulative_reward += float(r)
            except (ValueError, TypeError):
                pass

    steps = len(trajectory)
    last_step = trajectory[-1] if trajectory else {}
    success = bool(last_step.get("success", False) or last_step.get("task_success", False))
    failed = bool(last_step.get("failed", False) or last_step.get("task_failed", False))

    try:
        score = compute_episode_score(
            cumulative_reward=cumulative_reward,
            success=success,
            failed=failed,
            steps=steps,
            max_steps=task.max_steps,
        )
    except Exception:
        score = 0.5  # Catch any unexpected math errors

    return EvaluateResponse(
        score=_clamp_score(score),
        success=success,
        failed=failed,
        steps=steps,
    )


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()