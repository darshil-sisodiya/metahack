"""OpenEnv - Environment Module

Implements the DistillationEnv class simulating a chemical distillation process.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.models import Action, Observation, Reward


@dataclass
class DistillationEnv:
    """
    Simulated distillation column environment.

    Models thermal dynamics, pressure behavior, purity changes,
    and energy consumption with realistic physics-inspired equations.
    """

    # Configuration
    seed: int = 42
    max_steps: int = 200

    # State variables
    temperature: float = field(default=80.0, init=False)
    pressure: float = field(default=1.0, init=False)
    purity: float = field(default=50.0, init=False)
    flow_rate: float = field(default=10.0, init=False)
    energy_usage: float = field(default=0.0, init=False)
    time_step: int = field(default=0, init=False)
    hidden_instability: float = field(default=0.0, init=False)

    # Internal state
    _prev_action: Action | None = field(default=None, init=False)
    # Seeded RNG for deterministic behavior: same seed always produces same sequence
    # This ensures reproducibility for debugging, testing, and scientific experiments
    _rng: np.random.Generator = field(default=None, init=False)
    _cooling_failure: bool = field(default=False, init=False)
    _pressure_spike: bool = field(default=False, init=False)

    # Physical bounds
    TEMP_MIN: float = field(default=20.0, init=False, repr=False)
    TEMP_MAX: float = field(default=150.0, init=False, repr=False)
    # Lower pressure minimum (0.1) allows smooth vent decay without hard clamping feel
    PRESSURE_MIN: float = field(default=0.1, init=False, repr=False)
    PRESSURE_MAX: float = field(default=5.0, init=False, repr=False)
    PURITY_MIN: float = field(default=0.0, init=False, repr=False)
    PURITY_MAX: float = field(default=100.0, init=False, repr=False)
    FLOW_MIN: float = field(default=0.0, init=False, repr=False)
    FLOW_MAX: float = field(default=50.0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize RNG and reset environment."""
        # Use numpy's modern Generator API with explicit seeding for determinism
        self._rng = np.random.default_rng(seed=self.seed)
        self.reset()

    def reset(self) -> Observation:
        """
        Reset environment to initial state.

        Returns:
            Initial observation.
        """
        # Re-seed RNG on reset to ensure deterministic episode starts
        self._rng = np.random.default_rng(seed=self.seed)

        # Realistic starting conditions with small initialization noise
        self.temperature = 80.0 + self._noise(2.0)
        self.pressure = 1.0 + self._noise(0.05)
        self.purity = 50.0 + self._noise(5.0)
        self.flow_rate = 10.0 + self._noise(1.0)
        self.energy_usage = 0.0
        self.time_step = 0
        self.hidden_instability = 0.0

        self._prev_action = None
        self._cooling_failure = False
        self._pressure_spike = False

        return self._get_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Apply action and advance simulation by one step.

        Args:
            action: Control inputs for this step.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        # Use delayed action for actuator response simulation
        effective_action = self._prev_action if self._prev_action else action
        self._prev_action = action

        # Apply dynamics
        self._update_temperature(effective_action)
        self._update_pressure(effective_action)
        self._update_purity(effective_action)
        self._update_flow_rate(effective_action)
        self._update_energy(effective_action)
        self._update_instability(effective_action)

        # Check for fault injection
        self._check_faults()

        # Add noise to state
        self._apply_noise()

        # Clamp values to valid bounds
        self._clamp_state()

        # Advance time
        self.time_step += 1

        # Check termination
        done = self._is_done()

        # Build info dict
        info = {
            "hidden_instability": self.hidden_instability,
            "cooling_failure": self._cooling_failure,
            "pressure_spike": self._pressure_spike,
        }

        # Placeholder reward
        reward = Reward(value=0.0, components={})

        return self._get_observation(), reward, done, info

    def state(self) -> dict[str, Any]:
        """
        Return full internal state for debugging/logging.

        Returns:
            Dictionary containing all state variables.
        """
        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "purity": self.purity,
            "flow_rate": self.flow_rate,
            "energy_usage": self.energy_usage,
            "time_step": self.time_step,
            "hidden_instability": self.hidden_instability,
            "cooling_failure": self._cooling_failure,
            "pressure_spike": self._pressure_spike,
            "prev_action": self._prev_action.model_dump() if self._prev_action else None,
        }

    def _get_observation(self) -> Observation:
        """Build observation from current state."""
        return Observation(
            temperature=self.temperature,
            pressure=self.pressure,
            purity=self.purity,
            flow_rate=self.flow_rate,
            energy_usage=self.energy_usage,
            time_step=self.time_step,
        )

    def _noise(self, scale: float) -> float:
        """
        Generate Gaussian noise with given standard deviation.

        Uses seeded numpy RNG for deterministic behavior.
        """
        return float(self._rng.normal(0.0, scale))

    def _update_temperature(self, action: Action) -> None:
        """
        Update temperature with thermal inertia.

        Temperature changes slowly toward a target based on steam valve.
        """
        # Target temperature based on steam valve (0-100 maps to 60-140°C)
        target_temp = 60.0 + (action.steam_valve / 100.0) * 80.0

        # Thermal inertia: slow approach to target (10% per step)
        inertia = 0.1
        self.temperature += inertia * (target_temp - self.temperature)

        # Cooling failure causes uncontrolled rise
        if self._cooling_failure:
            self.temperature += 2.0

    def _update_pressure(self, action: Action) -> None:
        """
        Update pressure with nonlinear dynamics.

        Pressure increases based on temperature and flow rate with nonlinear growth.
        Vent acts as a proportional release valve, directly reducing current pressure.
        """
        # --- Pressure increase calculation ---
        # Temperature contribution: nonlinear (squared) relationship
        # Higher temperatures cause exponentially more pressure buildup
        temp_normalized = self.temperature / 100.0
        temp_contribution = 0.08 * (temp_normalized ** 2)

        # Flow rate contribution: linear relationship
        # More flow = more vapor = more pressure
        flow_contribution = 0.02 * (self.flow_rate / self.FLOW_MAX)

        # Combined pressure increase per step
        pressure_increase = temp_contribution + flow_contribution

        # Pressure spike event adds sudden increase
        if self._pressure_spike:
            pressure_increase += 0.3

        # --- Vent mechanism (smooth proportional decay) ---
        # When vent is open, release pressure proportionally to current pressure.
        # This creates smooth exponential decay rather than a hard floor:
        # - At high pressure: large release (fast decay)
        # - At low pressure: small release (slow decay, approaches equilibrium smoothly)
        # No base release rate to avoid artificial floor behavior.
        vent_effect = 0.0
        if action.vent == 1:
            # Pure proportional release: 8% of current pressure per step
            # Results in smooth asymptotic decay toward equilibrium
            vent_effect = 0.08 * self.pressure

        # --- Apply pressure dynamics ---
        # New pressure = current + increase - vent release
        self.pressure = self.pressure + pressure_increase - vent_effect

        # Soft floor at 0.1 to prevent negative values while feeling natural
        # (physical systems have minimum atmospheric/ambient pressure)
        self.pressure = max(0.1, self.pressure)

    def _update_purity(self, action: Action) -> None:
        """
        Update purity based on operating conditions.

        Purity increases with reflux ratio but is affected by
        temperature (optimal range) and feed rate.
        """
        # Reflux ratio benefit (higher = better separation)
        reflux_benefit = (action.reflux_ratio / 100.0) * 2.0

        # Temperature effect: optimal range is 85-95°C
        temp_deviation = abs(self.temperature - 90.0)
        if temp_deviation < 5.0:
            temp_factor = 1.0
        elif temp_deviation < 15.0:
            temp_factor = 0.5
        else:
            temp_factor = -0.5  # Poor separation outside optimal range

        # Feed rate penalty (too high = less purity)
        feed_penalty = 0.0
        if action.feed_rate > 70.0:
            feed_penalty = (action.feed_rate - 70.0) / 30.0 * 1.5

        # Net purity change
        purity_delta = reflux_benefit * temp_factor - feed_penalty

        # Apply with some inertia
        self.purity += 0.2 * purity_delta

    def _update_flow_rate(self, action: Action) -> None:
        """
        Update flow rate based on feed rate action.

        Flow rate tracks feed rate with some lag.
        """
        # Target flow rate scaled from feed_rate action
        target_flow = (action.feed_rate / 100.0) * self.FLOW_MAX

        # Gradual adjustment
        self.flow_rate += 0.25 * (target_flow - self.flow_rate)

    def _update_energy(self, action: Action) -> None:
        """
        Update cumulative energy usage.

        Energy is proportional to steam valve and system stress.
        """
        # Base energy from steam valve
        steam_energy = (action.steam_valve / 100.0) * 0.5

        # Additional energy from high pressure (stress)
        stress_energy = max(0.0, (self.pressure - 2.0) * 0.2)

        # Vent uses some energy
        vent_energy = 0.05 if action.vent == 1 else 0.0

        # Accumulate
        self.energy_usage += steam_energy + stress_energy + vent_energy

    def _update_instability(self, action: Action) -> None:
        """
        Update hidden instability factor.

        Instability builds up with extreme operating conditions.
        """
        # High temperature increases instability
        if self.temperature > 120.0:
            self.hidden_instability += 0.1

        # High pressure increases instability
        if self.pressure > 3.0:
            self.hidden_instability += 0.15

        # Rapid changes increase instability
        if self._prev_action:
            valve_change = abs(action.steam_valve - self._prev_action.steam_valve)
            if valve_change > 30.0:
                self.hidden_instability += 0.05

        # Natural decay
        self.hidden_instability *= 0.95

        # Clamp
        self.hidden_instability = max(0.0, min(1.0, self.hidden_instability))

    def _check_faults(self) -> None:
        """
        Check for fault injection events based on instability.

        Faults are probabilistic based on hidden instability.
        """
        # Cooling failure: triggered by high instability
        if not self._cooling_failure and self.hidden_instability > 0.7:
            if self._rng.random() < 0.1:
                self._cooling_failure = True

        # Cooling failure recovers slowly
        if self._cooling_failure and self._rng.random() < 0.05:
            self._cooling_failure = False

        # Pressure spike: sudden event
        if not self._pressure_spike and self.hidden_instability > 0.5:
            if self._rng.random() < 0.05:
                self._pressure_spike = True

        # Pressure spike is brief
        if self._pressure_spike and self._rng.random() < 0.3:
            self._pressure_spike = False

    def _apply_noise(self) -> None:
        """
        Add small Gaussian noise to state variables.

        Noise is intentionally small to:
        - Simulate real-world sensor noise and process variability
        - Maintain system stability (no large jumps)
        - Keep behavior smooth and predictable

        All noise uses the seeded RNG for deterministic reproducibility.
        """
        # Temperature: small fluctuations from ambient/sensor noise
        self.temperature += self._rng.normal(0, 0.2)

        # Pressure: very small noise (pressure sensors are relatively accurate)
        self.pressure += self._rng.normal(0, 0.01)

        # Purity: small measurement noise
        self.purity += self._rng.normal(0, 0.1)

        # Flow rate: minor fluctuations
        self.flow_rate += self._rng.normal(0, 0.15)

    def _clamp_state(self) -> None:
        """Clamp all state variables to valid bounds."""
        self.temperature = max(self.TEMP_MIN, min(self.TEMP_MAX, self.temperature))
        self.pressure = max(self.PRESSURE_MIN, min(self.PRESSURE_MAX, self.pressure))
        self.purity = max(self.PURITY_MIN, min(self.PURITY_MAX, self.purity))
        self.flow_rate = max(self.FLOW_MIN, min(self.FLOW_MAX, self.flow_rate))
        self.energy_usage = max(0.0, self.energy_usage)

    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Max steps reached
        if self.time_step >= self.max_steps:
            return True

        # Critical failure: extreme pressure
        if self.pressure >= self.PRESSURE_MAX:
            return True

        # Critical failure: extreme temperature
        if self.temperature >= self.TEMP_MAX:
            return True

        return False
