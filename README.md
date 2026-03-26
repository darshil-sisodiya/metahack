# OpenEnv

OpenEnv is a lightweight process-control benchmark built around a simulated distillation column. The repo combines:

- A physics-inspired environment (`app/env.py`)
- Task wrappers with task-specific rewards, success criteria, and failure logic (`app/tasks.py`)
- A deterministic evaluation/grading layer (`app/grader.py`)
- A hand-tuned heuristic baseline (`app/baseline.py`)
- A minimal FastAPI shell (`app/main.py`)

This project is best understood as a control / RL playground, not a production plant model. It is useful for:

- Comparing simple control policies
- Testing reward and task design
- Measuring agent performance across multiple operating regimes
- Demonstrating how success criteria, delayed dynamics, and escalation rules change difficulty

## Executive Summary

At a high level, the loop is:

1. An agent receives an `Observation`
2. The agent returns an `Action`
3. A `Task` applies the action through `task.step(env, action)`
4. The underlying `DistillationEnv` advances the physical state
5. The task adds reward, success/failure logic, variance tracking, and escalation behavior
6. The grader converts the full episode into a 0-100 score

Important design rule:

- For task evaluation, always use `task.reset(env)` and `task.step(env, action)`, not raw `env.reset()` / `env.step()`

The task layer is where reward, success, failure, and instability escalation actually live.

## Current Project Status

What is implemented today:

- Deterministic environment seeding
- Three tasks: stabilization, optimization, emergency control
- Reward shaping and success/failure logic per task
- A grader that produces weighted 0-100 scores
- A runnable heuristic baseline
- Helper scripts for task difficulty and grader checks
- A minimal API health endpoint

What is not implemented yet:

- No ML training loop
- No policy learning or checkpointing
- No rich API endpoints beyond health check
- No real config system wired through `openenv.yaml`

## Repository Layout

| Path | Purpose | Notes |
| --- | --- | --- |
| `app/__init__.py` | Package marker | No runtime logic |
| `app/models.py` | Core Pydantic models | Defines `Observation`, `Action`, `Reward` |
| `app/env.py` | Base simulator | Implements all state dynamics, noise, faults, and hard bounds |
| `app/tasks.py` | Task layer | Adds task-specific reset, reward, success, failure, variance, and escalation |
| `app/grader.py` | Evaluation layer | Converts full episodes into scores and aggregates multi-task results |
| `app/baseline.py` | Heuristic baseline | Deterministic reference controller plus CLI entry point |
| `app/main.py` | FastAPI shell | Only exposes `/` health check currently |
| `test.py` | Random-policy task difficulty probe | Reports survival/success under random actions |
| `test_grader.py` | Ad hoc grader smoke test | Compares several simple agents |
| `requirements.txt` | Python dependencies | FastAPI, Pydantic, Uvicorn, NumPy |
| `Dockerfile` | Container image definition | Runs `uvicorn app.main:app` |
| `openenv.yaml` | Placeholder config file | Present, but not used by runtime code today |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn app.main:app --reload
```

Current API behavior:

- `GET /` returns `{"status": "ok"}`

### 3. Run the heuristic baseline

```bash
python -m app.baseline
```

### 4. Run task difficulty evaluation with a random policy

```bash
python test.py
```

### 5. Run grader smoke tests

```bash
python test_grader.py
```

### 6. Run with Docker

```bash
docker build -t openenv .
docker run -p 8000:8000 openenv
```

## Core Data Model

### Observation

Defined in `app/models.py`.

| Field | Meaning | Range / Unit |
| --- | --- | --- |
| `temperature` | Process temperature | `>= 0`, degrees C |
| `pressure` | System pressure | `>= 0`, bar |
| `purity` | Product purity | `0-100`, percent |
| `flow_rate` | Output flow | `>= 0`, L/min |
| `energy_usage` | Cumulative energy usage | `>= 0`, kWh-style running total |
| `time_step` | Current simulation step | integer, `>= 0` |

### Action

Defined in `app/models.py`.

| Field | Meaning | Allowed values | Primary effects |
| --- | --- | --- | --- |
| `steam_valve` | Heat input command | `0-100` | Drives temperature, energy, and indirectly pressure/purity |
| `reflux_ratio` | Separation quality command | `0-100` | Improves purity |
| `feed_rate` | Throughput command | `0-100` | Drives flow rate and can hurt purity if pushed too high |
| `vent` | Pressure release valve | `0` or `1` | Lowers pressure, slightly increases energy use |

### Reward

Defined in `app/models.py`.

| Field | Meaning |
| --- | --- |
| `value` | Scalar reward used by the grader |
| `components` | Named breakdown of reward terms for interpretability |

Important note:

- `DistillationEnv.step()` returns a placeholder reward of `0.0`
- Real rewards are computed in `Task.compute_reward(...)`

## Simulation Architecture

## Step Order

The environment update order matters a lot:

1. `env.step(action)` receives the current action
2. The environment applies a one-step actuator delay:
   - If there was a previous action, that previous action becomes the effective action
   - Otherwise the current action is used on the first step
3. State updates happen in this order:
   - temperature
   - pressure
   - purity
   - flow rate
   - energy usage
   - hidden instability
4. Faults may be injected
5. Small random noise is added
6. State is clamped to hard bounds
7. `time_step` increments
8. Environment-level termination is checked

Practical consequence:

- All agents are dealing with delayed dynamics
- Pressure uses the already-updated temperature, but the previous flow state
- Purity uses the updated temperature and the current effective action's feed setting
- Flow responds with lag, not instantly

## State Variables and Hard Bounds

These are enforced in `app/env.py`.

| Variable | Min | Max | Notes |
| --- | --- | --- | --- |
| Temperature | `20.0` | `150.0` | Hard physical clamp |
| Pressure | `0.1` | `5.0` | Soft floor plus hard ceiling |
| Purity | `0.0` | `100.0` | Percentage |
| Flow rate | `0.0` | `50.0` | Matches `FLOW_MAX` |
| Energy usage | `0.0` | No explicit upper cap | Only lower-clamped |

## Environment Dynamics In Detail

### 1. Temperature Dynamics

Temperature is controlled by `steam_valve`.

Effective target:

```text
target_temp = 60.0 + (steam_valve / 100.0) * 80.0
```

State update:

```text
temperature += 0.1 * (target_temp - temperature)
```

Interpretation:

- `steam_valve = 0` maps to a target of `60 C`
- `steam_valve = 50` maps to a target of `100 C`
- `steam_valve = 100` maps to a target of `140 C`
- The `0.1` multiplier means temperature only moves 10% of the remaining gap per step

Fault effect:

```text
if cooling_failure:
    temperature += 2.0
```

This makes cooling failures dangerous because they add heat on top of the normal controller response.

### 2. Pressure Dynamics

Pressure is driven by temperature and flow, and relieved by venting.

Pressure growth:

```text
temp_contribution = 0.08 * (temperature / 100.0)^2
flow_contribution = 0.02 * (flow_rate / 50.0)
pressure_increase = temp_contribution + flow_contribution
```

Optional spike fault:

```text
if pressure_spike:
    pressure_increase += 0.3
```

Vent relief:

```text
if vent == 1:
    vent_effect = 0.08 * pressure
else:
    vent_effect = 0.0
```

Final pressure update:

```text
pressure = max(0.1, pressure + pressure_increase - vent_effect)
```

Interpretation:

- Pressure rises faster at high temperatures because the temperature term is quadratic
- Higher flow adds pressure more gently
- Venting removes 8% of current pressure each step, so it behaves like smooth exponential decay rather than a hard reset

### 3. Purity Dynamics

Purity depends on reflux, temperature quality, and feed overloading.

Reflux benefit:

```text
reflux_benefit = (reflux_ratio / 100.0) * 2.0
```

Temperature quality around the `90 C` sweet spot:

```text
temp_deviation = abs(temperature - 90.0)

temp_factor = 1.0   if temp_deviation < 5.0
temp_factor = 0.5   if temp_deviation < 15.0
temp_factor = -0.5  otherwise
```

High-feed penalty:

```text
if feed_rate > 70.0:
    feed_penalty = ((feed_rate - 70.0) / 30.0) * 1.5
else:
    feed_penalty = 0.0
```

Purity update:

```text
purity_delta = reflux_benefit * temp_factor - feed_penalty
purity += 0.2 * purity_delta
```

Interpretation:

- Purity improves when reflux is high and temperature is near `90 C`
- Purity degrades when the column runs too hot or too cold
- Aggressive feed (`> 70`) directly hurts purity
- The `0.2` multiplier makes purity move gradually instead of instantly

### 4. Flow Rate Dynamics

Flow rate tracks feed rate with lag.

Target:

```text
target_flow = (feed_rate / 100.0) * 50.0
```

Update:

```text
flow_rate += 0.25 * (target_flow - flow_rate)
```

Interpretation:

- `feed_rate = 50` corresponds to target flow `25`
- Flow only closes 25% of the gap each step, so large feed changes take time to settle

### 5. Energy Usage

Energy accumulates rather than resets during an episode.

```text
steam_energy = (steam_valve / 100.0) * 0.5
stress_energy = max(0.0, (pressure - 2.0) * 0.2)
vent_energy = 0.05 if vent == 1 else 0.0

energy_usage += steam_energy + stress_energy + vent_energy
```

Interpretation:

- More steam always costs more energy
- High pressure adds stress-related energy cost
- Venting is not free

### 6. Hidden Instability

`hidden_instability` is not part of the observation, but it influences fault injection.

Build-up rules:

```text
if temperature > 120.0:
    hidden_instability += 0.1

if pressure > 3.0:
    hidden_instability += 0.15

if abs(current_steam - previous_steam) > 30.0:
    hidden_instability += 0.05
```

Decay and clamp:

```text
hidden_instability *= 0.95
hidden_instability = clip(hidden_instability, 0.0, 1.0)
```

Interpretation:

- Hot operation, high pressure, and abrupt steam changes all make the plant more fragile
- Instability does not explode forever because it decays and is capped at `1.0`

### 7. Fault Injection

Faults are probabilistic and use the seeded RNG.

Cooling failure:

```text
if hidden_instability > 0.7 and not cooling_failure:
    with probability 0.1: cooling_failure = True

if cooling_failure:
    with probability 0.05: cooling_failure = False
```

Pressure spike:

```text
if hidden_instability > 0.5 and not pressure_spike:
    with probability 0.05: pressure_spike = True

if pressure_spike:
    with probability 0.3: pressure_spike = False
```

Interpretation:

- Faults only become likely when the controller has already driven the plant into unstable conditions
- Because the RNG is seeded, the same policy and seed always produce the same fault sequence

### 8. Noise Model

Small Gaussian noise is added every step:

| Variable | Noise |
| --- | --- |
| Temperature | `N(0, 0.2)` |
| Pressure | `N(0, 0.01)` |
| Purity | `N(0, 0.1)` |
| Flow rate | `N(0, 0.15)` |

Purpose:

- Mimic sensor/process noise
- Prevent the environment from being perfectly clean
- Keep behavior realistic without creating huge jumps

### 9. Environment-Level Done Conditions

Raw environment termination occurs if:

- `time_step >= max_steps`
- `pressure >= 5.0`
- `temperature >= 150.0`

In practice, task-level termination usually happens earlier.

## Task Layer

The task layer in `app/tasks.py` is what turns the simulator into a benchmark.

Each task provides:

- Task-specific reset conditions
- Reward function
- Success condition
- Failure condition
- Instability thresholds
- Optional escalation logic

### Common Task Mechanics

All tasks inherit from `BaseTask`.

Shared behavior:

- Rolling windows of length `5` for temperature and pressure
- Variance computed with `np.var(...)`
- Common info keys added at each step:
  - `temperature_variance`
  - `pressure_variance`
  - `consecutive_instability_steps`
  - `task_failed`
  - `task_success`
  - `failure_reason`
  - `step_count`

Default early safety shutdowns:

- Pressure `>= 2.8`
- Temperature `>= 145.0`

Exception:

- `EmergencyControlTask` overrides early pressure failure to `3.0`

Task-level `done` becomes true if any of the following happens:

- Environment says done
- Task fails
- Task succeeds
- Task step count reaches `task.max_steps`

### Task 1: Stabilization

Goal:

- Hold the plant near a target temperature with low oscillation and safe pressure

Key config:

| Parameter | Value |
| --- | --- |
| `max_steps` | `50` |
| `target_temperature` | `100.0` |
| `temperature_variance_threshold` | `10.0` |
| `instability_temp_threshold` | `10.0` |
| `instability_pressure_threshold` | `0.15` |
| `temperature_min` | `50.0` |
| `temperature_max` | `140.0` |

Initial conditions:

- Temperature sampled uniformly in `[90, 110]`
- Pressure sampled uniformly in `[0.9, 1.3]`
- Purity fixed at `50`
- Flow rate fixed at `15`

Reward:

```text
temp_error = abs(temperature - 100.0)
temp_score = max(0, 1 - temp_error / 50)

pressure_score:
    if pressure < 2.0:
        1.0 - (pressure / 2.0) * 0.5
    elif pressure < early_failure_pressure:
        0.5 - (pressure - 2.0) * 0.5
    else:
        -1.0

variance_penalty only applies if temperature_variance > 10

reward = 0.6 * temp_score + 0.3 * pressure_score - 0.1 * variance_penalty
```

Success:

- Only checked at episode end
- `abs(temperature - 100.0) < 15.0`
- `temperature_variance < 10.0`

Failure:

- `temperature < 50`
- `temperature > 140`
- Plus shared early shutdowns

Escalation:

- None

Why it is easier:

- No instability escalation
- Clean success criterion
- Controller mainly needs to hold temperature steady

### Task 2: Optimization

Goal:

- Balance purity, throughput, and energy efficiency

Key config:

| Parameter | Value |
| --- | --- |
| `max_steps` | `75` |
| `max_energy_reference` | `100.0` |
| `instability_temp_threshold` | `25.0` |
| `instability_pressure_threshold` | `0.2` |
| `instability_escalation_steps` | `3` |
| `instability_escalation_rate` | `0.1` |
| `low_purity_threshold` | `45.0` |
| `low_purity_persistence_steps` | `30` |
| `energy_growth_penalty_threshold` | `2.0` |
| `target_purity` | `80.0` |
| `target_flow_rate` | `15.0` |
| `success_purity_threshold` | `57.5` |
| `success_flow_rate_threshold` | `12.0` |
| `success_temperature_variance_threshold` | `30.0` |

Initial conditions:

- Uses the default environment reset

Reward:

```text
purity_score = purity / 100.0
flow_score = min(flow_rate / 15.0, 1.0)
energy_penalty = energy_usage / 100.0

if energy_usage increased by more than 2.0 this step:
    energy_penalty *= 1.5

base_reward = 0.5 * purity_score + 0.3 * flow_score - 0.2 * energy_penalty

instability_penalty = 0.05 * consecutive_instability_steps

reward = base_reward - instability_penalty
```

Success:

- Only checked at episode end
- `purity >= 57.5`
- `flow_rate >= 12.0`
- `temperature_variance < 30.0`

Failure:

- Purity below `45.0` for more than `30` consecutive steps
- `consecutive_instability_steps >= 5`
- Plus shared early shutdowns

Escalation:

```text
if consecutive_instability_steps >= 3:
    pressure += 10% of current pressure every step
```

Important interpretation:

- Reward target flow is stricter than the success threshold
- Reward pushes for better purity/throughput/efficiency than the minimum success gate
- This is why a baseline can succeed often without being reward-optimal

### Task 3: Emergency Control

Goal:

- Recover the plant from an already elevated-pressure state

Key config:

| Parameter | Value |
| --- | --- |
| `max_steps` | `60` |
| `stability_temp_range` | `(85.0, 115.0)` |
| `stability_pressure_threshold` | `1.5` |
| `stability_temp_variance_threshold` | `15.0` |
| `stability_pressure_variance_threshold` | `0.15` |
| `stability_required_consecutive_steps` | `8` |
| `instability_temp_threshold` | `15.0` |
| `instability_pressure_threshold` | `0.15` |
| `instability_escalation_steps` | `5` |
| `instability_escalation_rate` | `0.04` |
| `cascading_pressure_threshold` | `2.6` |
| `cascading_pressure_rate` | `0.03` |
| `neutral_cooling_action` | `50.0` |
| `cooling_efficiency_reduction` | `0.8` |
| `pressure_growth_multiplier` | `1.0` |
| Early pressure failure override | `3.0` |

Initial conditions:

- Temperature sampled uniformly in `[90, 110]`
- Pressure sampled uniformly in `[1.35, 1.65]`
- Purity sampled uniformly in `[40, 60]`
- Flow fixed at `15`

Emergency-specific delayed dynamics:

- The base environment already has a one-step action delay
- Emergency task adds extra lag for cooling by buffering actions
- Temperature uses a steam command effectively delayed by two control decisions
- Reflux, feed, and vent still pass through the normal one-step delayed environment path

Cooling modification:

- Steam below `50` becomes less effective because of `cooling_efficiency_reduction = 0.8`
- This makes it harder to cool aggressively on command

Pressure modification:

- `pressure_growth_multiplier = 1.0`, so there is currently no extra multiplicative pressure growth beyond the normal pressure model
- Pressure is still made difficult through escalation and cascading penalties

Escalation:

```text
if consecutive_instability_steps >= 5:
    pressure += 4% of current pressure

if pressure > 2.6:
    pressure += additional 3% of current pressure
```

Stability definition:

- `85 <= temperature <= 115`
- `pressure < 1.5`
- `temperature_variance <= 15`
- `pressure_variance <= 0.15`

Success:

- Achieve the stability definition for at least `8` consecutive task steps

Reward:

```text
pressure_reduction_score = clamp((initial_pressure - pressure) / initial_pressure, 0, 1)
temperature_stability_score = max(0, 1 - abs(temperature - 100) / 40)
recovery_bonus = 1.0 if stable for >= 8 steps else 0.0
escalation_penalty = min(0.3, instability_escalation * 0.5)

reward = (
    0.4 * pressure_reduction_score
    + 0.4 * temperature_stability_score
    + 0.2 * recovery_bonus
    - escalation_penalty
)
```

Failure:

- `pressure >= 3.2`
- Timeout at final step if recovery was never achieved
- Early shutdown if pressure `>= 3.0`
- Early shutdown if temperature `>= 145`

Why it is hard:

- Starts near danger instead of near nominal
- Cooling response is delayed
- Pressure escalation compounds if the controller oscillates
- The controller must survive first, then recover, then stay stable

## Important Accuracy Note

Some inline comments/docstrings in `app/tasks.py` are older than the current tuned config. This README describes the effective runtime behavior in the code as it exists now.

Examples:

- Optimization currently succeeds at `purity >= 57.5`, not the older `55` text that still appears near one docstring
- Emergency control currently escalates after `5` unstable steps at `4%`, not the older `4-step / 6%` wording in stale comments
- `pressure_growth_multiplier` is currently `1.0`, so there is no extra pressure multiplier beyond the normal environment pressure model

## Grading System

The grading code lives in `app/grader.py`.

### `evaluate_episode(task, seed)`

Runs one episode using the built-in baseline from `app/grader.py`.

Returns:

```python
{
    "score": float,
    "success": bool,
    "failed": bool,
    "steps": int,
    "final_obs": Observation,
    "failure_reason": str,
}
```

### `evaluate_all_tasks(agent_fn, n_episodes=20)`

Runs the provided policy across all tasks and returns:

- `overall_score`
- `task_scores`
- `details` per task

### Scoring Formula

Performance score:

```text
normalized_reward = cumulative_reward / max_steps
clipped_reward = clip(normalized_reward, -1, 1)
performance_score = map clipped_reward from [-1, 1] to [0, 60]
```

Success bonus:

```text
success_bonus = 40 if task_success else 0
```

Efficiency bonus:

```text
efficiency_bonus = ((max_steps - steps) / max_steps) * 10
```

This only applies when the episode succeeds.

Failure cap:

```text
if failed:
    score = min(score, 30 + performance_score * 0.5)
```

Final clamp:

```text
score = clip(score, 0, 100)
```

### Task Weights

The overall multi-task score uses:

| Task | Weight |
| --- | --- |
| Stabilization | `0.25` |
| Optimization | `0.35` |
| Emergency control | `0.40` |

Interpretation:

- Emergency still matters most
- Optimization has significant influence
- Stabilization matters, but no longer dominates the full benchmark

### Determinism

The grading path is deterministic if the agent is deterministic.

Why:

- `DistillationEnv` is seeded
- `env.reset()` reseeds the RNG with the same seed
- Task reset logic uses the seeded RNG
- `evaluate_all_tasks(...)` chooses deterministic seeds:

```text
seed = task_index * 10000 + episode_index
```

Important caveat:

- `test_grader.py` contains a `random_agent` that uses global NumPy randomness and is therefore not deterministic unless you seed NumPy yourself

## Baseline Agent

The current baseline is in `app/baseline.py`.

This is a hand-tuned heuristic, not a learned model.

Design goals:

- Keep temperature near the purity-friendly zone
- Vent before pressure becomes dangerous
- Keep enough throughput to satisfy optimization
- Do not specifically solve the emergency task

### Baseline Logic

Temperature targeting:

- Default target temperature: `92 C`
- If pressure `> 1.7`, lower target to `88 C`
- If pressure `> 2.0`, lower target to `82 C`
- If purity is low (`< 55`) and pressure is not elevated, target `90 C`

Steam command:

```text
steam = clamp(50 + 1.2 * (target_temp - observed_temp), 30, 68)
```

Vent logic:

- Open vent if `pressure > 2.1`
- Also open vent if `pressure > 1.75` and `temperature > 98`

Reflux logic:

| Purity band | Reflux ratio |
| --- | --- |
| `< 52` | `48` |
| `< 57` | `34` |
| `< 62` | `24` |
| `>= 62` | `48` |

Feed logic:

| Condition | Feed rate |
| --- | --- |
| `pressure > 2.0` | `35` |
| `flow_rate < 12.0` | `72` |
| `flow_rate > 18.0` | `55` |
| Otherwise | `60` |

Interpretation:

- This controller is intentionally "good but not perfect"
- It keeps stabilization extremely strong
- It trades some purity margin for throughput in optimization
- It still lacks the memory and recovery-specific logic needed for emergency control

### Current Baseline Results

From `python -m app.baseline` on the current codebase:

| Metric | Value |
| --- | --- |
| Overall score | `65.48` |
| Stabilization score | `89.65` |
| Stabilization success | `100.0%` |
| Optimization score | `77.46` |
| Optimization success | `77.0%` |
| Emergency score | `39.89` |
| Emergency success | `0.0%` |

This is a useful reference point for future model-based agents.

## Helper Scripts

### `test.py`

Purpose:

- Measures task survivability and success rate under a seeded random policy

What it does:

- Builds a fresh task instance per episode
- Uses a seeded NumPy RNG to sample random actions
- Runs `200` episodes per task by default
- Prints survival, success, and failure breakdown

Current outputs on this codebase:

| Task | Survival | Success |
| --- | --- | --- |
| Stabilization | `93.0%` | `90.5%` |
| Optimization | `69.0%` | `25.0%` |
| Emergency control | `12.0%` | `12.0%` |

Use this script when you change task definitions and want to verify the difficulty profile.

### `test_grader.py`

Purpose:

- Quick smoke test for the grader with several agent styles

Agents included:

- `dumb_agent`
- `random_agent`
- `heuristic_agent`

Notes:

- Good for sanity-checking the scoring API
- Not a formal test suite
- The `random_agent` in this file is not deterministic unless NumPy is globally seeded

## FastAPI Layer

`app/main.py` currently only defines:

- `FastAPI(title="OpenEnv", version="0.1.0")`
- `GET /` health check

This means:

- The core of the project is the simulator and evaluator, not the web service
- If the team wants an HTTP evaluation service, the API layer is the place to extend

## Configuration and Packaging

### `requirements.txt`

Current dependencies:

- `fastapi`
- `pydantic`
- `uvicorn`
- `numpy`

### `Dockerfile`

The container:

- Starts from `python:3.10-slim`
- Installs requirements
- Copies the repo into `/app`
- Exposes port `8000`
- Runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### `openenv.yaml`

This file exists, but it is currently just a placeholder:

```yaml
name: openenv
version: "0.1.0"
```

The runtime does not consume it yet.

## How To Extend The Project

### Add a new task

1. Create a new `BaseTask` subclass in `app/tasks.py`
2. Implement:
   - `_apply_initial_conditions`
   - `compute_reward`
   - `check_success`
   - `check_failure`
3. Optionally override:
   - `_is_unstable`
   - `_apply_instability_escalation`
   - `step` if you need custom task-only dynamics
4. Add the task to `get_all_tasks()`
5. Add it to `get_task_by_name(...)`
6. Update documentation and evaluation expectations

### Add a new policy

1. Implement `agent_fn(obs) -> Action`
2. Run it through:

```python
from app.grader import evaluate_all_tasks

results = evaluate_all_tasks(agent_fn, n_episodes=20)
```

3. Compare:
   - overall score
   - task scores
   - per-task success rate
   - failure rate
   - average steps

### Add API endpoints

Good future additions:

- `/evaluate`
- `/tasks`
- `/baseline`
- `/rollout`

This would let the team expose the simulator as a service for other tools or frontends.

## Key Takeaways For The Team

- The project has a clean separation between physics (`env.py`), benchmark logic (`tasks.py`), scoring (`grader.py`), and controller baselines (`baseline.py`)
- The task layer is the real benchmark surface; raw environment stepping is not enough for evaluation
- Delayed dynamics are intentional and matter a lot for controller design
- Optimization is currently easier than a reward-optimal controller would suggest because its success thresholds are intentionally looser than its reward incentives
- Emergency control is the real hard case because of delayed cooling, elevated starting pressure, escalation, and recovery requirements
- The current heuristic baseline is a useful reference controller, but not a learned solution

If your team wants to evolve this repo next, the highest-leverage areas are:

- Add proper API evaluation endpoints
- Formalize tests beyond the current helper scripts
- Decide whether stale inline comments in `app/tasks.py` should be cleaned up to match the live config
- Add a training loop or policy optimization module if the goal is to benchmark learned agents instead of only heuristics
