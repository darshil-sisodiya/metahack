"""OpenEnv - Pydantic Models

Defines the core data structures for the OpenEnv environment:
- Observation: State representation from the environment
- Action: Agent's control inputs
- Reward: Feedback signal with component breakdown
"""

from typing import Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    Environment observation at a given time step.

    Represents the current state of the distillation process,
    including physical measurements and resource usage.
    """

    temperature: float = Field(
        ...,
        ge=0.0,
        description="Process temperature in Celsius",
        examples=[85.5],
    )
    pressure: float = Field(
        ...,
        ge=0.0,
        description="System pressure in bar",
        examples=[1.013],
    )
    purity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Product purity percentage (0-100)",
        examples=[95.2],
    )
    flow_rate: float = Field(
        ...,
        ge=0.0,
        description="Flow rate in L/min",
        examples=[10.5],
    )
    energy_usage: float = Field(
        ...,
        ge=0.0,
        description="Energy consumption in kWh",
        examples=[42.3],
    )
    time_step: int = Field(
        ...,
        ge=0,
        description="Current simulation time step",
        examples=[0],
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "temperature": 85.5,
            "pressure": 1.013,
            "purity": 95.2,
            "flow_rate": 10.5,
            "energy_usage": 42.3,
            "time_step": 0,
        }
    ]}}


class Action(BaseModel):
    """
    Agent action to control the environment.

    Defines the control inputs for the distillation process.
    All continuous values are normalized to 0-100 range.
    """

    steam_valve: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Steam valve opening percentage (0-100)",
        examples=[50.0],
    )
    reflux_ratio: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Reflux ratio percentage (0-100)",
        examples=[75.0],
    )
    feed_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Feed rate percentage (0-100)",
        examples=[60.0],
    )
    vent: Literal[0, 1] = Field(
        ...,
        description="Vent valve state: 0 = closed, 1 = open",
        examples=[0],
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "steam_valve": 50.0,
            "reflux_ratio": 75.0,
            "feed_rate": 60.0,
            "vent": 0,
        }
    ]}}


class Reward(BaseModel):
    """
    Reward signal from the environment.

    Contains the total reward value and a breakdown of
    individual reward components for interpretability.
    """

    value: float = Field(
        ...,
        description="Total reward value",
        examples=[0.85],
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward contributions by component",
        examples=[{"purity": 1.0, "energy_penalty": -0.1, "stability": 0.05}],
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "value": 0.85,
            "components": {
                "purity": 1.0,
                "energy_penalty": -0.1,
                "stability": 0.05,
            },
        }
    ]}}
