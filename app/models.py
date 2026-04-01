"""OpenEnv - Pydantic Models.

Defines the typed request/response payloads used by the OpenEnv runtime.
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


class EnvironmentState(BaseModel):
    """Full runtime state exposed through the standard `state()` API."""

    active_task: str | None = Field(
        default=None,
        description="Name of the currently selected task, if the environment has been reset.",
        examples=["optimization"],
    )
    temperature: float = Field(
        ...,
        ge=0.0,
        description="Process temperature in Celsius",
        examples=[88.2],
    )
    pressure: float = Field(
        ...,
        ge=0.0,
        description="System pressure in bar",
        examples=[1.24],
    )
    purity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Product purity percentage (0-100)",
        examples=[57.9],
    )
    flow_rate: float = Field(
        ...,
        ge=0.0,
        description="Flow rate in L/min",
        examples=[13.4],
    )
    energy_usage: float = Field(
        ...,
        ge=0.0,
        description="Cumulative energy usage",
        examples=[6.7],
    )
    time_step: int = Field(
        ...,
        ge=0,
        description="Current environment step",
        examples=[12],
    )
    hidden_instability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Latent instability factor used for fault injection",
        examples=[0.15],
    )
    cooling_failure: bool = Field(
        ...,
        description="Whether the cooling-failure fault is currently active",
        examples=[False],
    )
    pressure_spike: bool = Field(
        ...,
        description="Whether the pressure-spike fault is currently active",
        examples=[False],
    )
    prev_action: Action | None = Field(
        default=None,
        description="Most recent action applied by the controller",
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "active_task": "optimization",
            "temperature": 88.2,
            "pressure": 1.24,
            "purity": 57.9,
            "flow_rate": 13.4,
            "energy_usage": 6.7,
            "time_step": 12,
            "hidden_instability": 0.15,
            "cooling_failure": False,
            "pressure_spike": False,
            "prev_action": {
                "steam_valve": 44.0,
                "reflux_ratio": 52.0,
                "feed_rate": 58.0,
                "vent": 0,
            },
        }
    ]}}
