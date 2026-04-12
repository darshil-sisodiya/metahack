from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.models import Action, Observation, Reward


@dataclass
class DistillationEnv:
    seed: int = 42
    max_steps: int = 200

    temperature: float = field(default=80.0, init=False)
    pressure: float = field(default=1.0, init=False)
    purity: float = field(default=50.0, init=False)
    flow_rate: float = field(default=10.0, init=False)
    energy_usage: float = field(default=0.0, init=False)
    time_step: int = field(default=0, init=False)
    hidden_instability: float = field(default=0.0, init=False)

    _prev_action: Action | None = field(default=None, init=False)
    _rng: np.random.Generator = field(default=None, init=False)
    _cooling_failure: bool = field(default=False, init=False)
    _pressure_spike: bool = field(default=False, init=False)

    TEMP_MIN: float = field(default=20.0, init=False, repr=False)
    TEMP_MAX: float = field(default=150.0, init=False, repr=False)
    PRESSURE_MIN: float = field(default=0.1, init=False, repr=False)
    PRESSURE_MAX: float = field(default=5.0, init=False, repr=False)
    PURITY_MIN: float = field(default=0.0, init=False, repr=False)
    PURITY_MAX: float = field(default=100.0, init=False, repr=False)
    FLOW_MIN: float = field(default=0.0, init=False, repr=False)
    FLOW_MAX: float = field(default=50.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(seed=self.seed)
        self.reset()

    def reset(self) -> Observation:
        self._rng = np.random.default_rng(seed=self.seed)

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
        effective_action = self._prev_action if self._prev_action else action
        self._prev_action = action

        self._update_temperature(effective_action)
        self._update_pressure(effective_action)
        self._update_purity(effective_action)
        self._update_flow_rate(effective_action)
        self._update_energy(effective_action)
        self._update_instability(effective_action)

        self._check_faults()
        self._apply_noise()
        self._clamp_state()

        self.time_step += 1
        done = self._is_done()

        info = {
            "hidden_instability": self.hidden_instability,
            "cooling_failure": self._cooling_failure,
            "pressure_spike": self._pressure_spike,
        }

        # ---------------- REWARD ----------------
        purity_score = self.purity / 100.0

        pressure_penalty = max(0.0, (self.pressure - 2.0) / 3.0)
        temp_penalty = abs(self.temperature - 90.0) / 60.0

        # FIXED: per-step energy penalty instead of cumulative doom
        avg_energy = self.energy_usage / (self.time_step + 1)
        energy_penalty = min(avg_energy / 10.0, 1.0)

        raw_reward = (
            purity_score
            - 0.3 * pressure_penalty
            - 0.2 * temp_penalty
            - 0.2 * energy_penalty
        )

        # Keep the environment reward continuous; strict-open sanitization
        # happens only when a public score is emitted.
        clipped = max(-1.0, min(1.0, raw_reward))

        reward = Reward(
            value=clipped,
            components={
                "purity": purity_score,
                "pressure_penalty": pressure_penalty,
                "temp_penalty": temp_penalty,
                "energy_penalty": energy_penalty,
            },
        )
        # ---------------------------------------

        return self._get_observation(), reward, done, info

    def state(self) -> dict[str, Any]:
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
        return Observation(
            temperature=self.temperature,
            pressure=self.pressure,
            purity=self.purity,
            flow_rate=self.flow_rate,
            energy_usage=self.energy_usage,
            time_step=self.time_step,
        )

    def _noise(self, scale: float) -> float:
        return float(self._rng.normal(0.0, scale))

    def _update_temperature(self, action: Action) -> None:
        target_temp = 60.0 + (action.steam_valve / 100.0) * 80.0
        self.temperature += 0.1 * (target_temp - self.temperature)

        if self._cooling_failure:
            self.temperature += 2.0

    def _update_pressure(self, action: Action) -> None:
        temp_contribution = 0.08 * ((self.temperature / 100.0) ** 2)
        flow_contribution = 0.02 * (self.flow_rate / self.FLOW_MAX)

        pressure_increase = temp_contribution + flow_contribution

        if self._pressure_spike:
            pressure_increase += 0.3

        vent_effect = 0.08 * self.pressure if action.vent == 1 else 0.0
        self.pressure = max(0.1, self.pressure + pressure_increase - vent_effect)

    def _update_purity(self, action: Action) -> None:
        reflux_benefit = (action.reflux_ratio / 100.0) * 2.0

        temp_deviation = abs(self.temperature - 90.0)
        if temp_deviation < 5.0:
            temp_factor = 1.0
        elif temp_deviation < 15.0:
            temp_factor = 0.5
        else:
            temp_factor = -0.5

        feed_penalty = 0.0
        if action.feed_rate > 70.0:
            feed_penalty = (action.feed_rate - 70.0) / 30.0 * 1.5

        purity_delta = reflux_benefit * temp_factor - feed_penalty
        self.purity += 0.2 * purity_delta

    def _update_flow_rate(self, action: Action) -> None:
        target_flow = (action.feed_rate / 100.0) * self.FLOW_MAX
        self.flow_rate += 0.25 * (target_flow - self.flow_rate)

    def _update_energy(self, action: Action) -> None:
        steam_energy = (action.steam_valve / 100.0) * 0.5
        stress_energy = max(0.0, (self.pressure - 2.0) * 0.2)
        vent_energy = 0.05 if action.vent == 1 else 0.0

        self.energy_usage += steam_energy + stress_energy + vent_energy

    def _update_instability(self, action: Action) -> None:
        if self.temperature > 120.0:
            self.hidden_instability += 0.1

        if self.pressure > 3.0:
            self.hidden_instability += 0.15

        if self._prev_action:
            if abs(action.steam_valve - self._prev_action.steam_valve) > 30.0:
                self.hidden_instability += 0.05

        self.hidden_instability *= 0.95
        self.hidden_instability = max(0.0, min(1.0, self.hidden_instability))

    def _check_faults(self) -> None:
        if not self._cooling_failure and self.hidden_instability > 0.7:
            if self._rng.random() < 0.1:
                self._cooling_failure = True

        if self._cooling_failure and self._rng.random() < 0.05:
            self._cooling_failure = False

        if not self._pressure_spike and self.hidden_instability > 0.5:
            if self._rng.random() < 0.05:
                self._pressure_spike = True

        if self._pressure_spike and self._rng.random() < 0.3:
            self._pressure_spike = False

    def _apply_noise(self) -> None:
        self.temperature += self._rng.normal(0, 0.2)
        self.pressure += self._rng.normal(0, 0.01)
        self.purity += self._rng.normal(0, 0.1)
        self.flow_rate += self._rng.normal(0, 0.15)

    def _clamp_state(self) -> None:
        self.temperature = max(self.TEMP_MIN, min(self.TEMP_MAX, self.temperature))
        self.pressure = max(self.PRESSURE_MIN, min(self.PRESSURE_MAX, self.pressure))
        self.purity = max(self.PURITY_MIN, min(self.PURITY_MAX, self.purity))
        self.flow_rate = max(self.FLOW_MIN, min(self.FLOW_MAX, self.flow_rate))
        self.energy_usage = max(0.0, self.energy_usage)

    def _is_done(self) -> bool:
        if self.time_step >= self.max_steps:
            return True
        if self.pressure >= self.PRESSURE_MAX:
            return True
        if self.temperature >= self.TEMP_MAX:
            return True
        return False
