---
title: OpenEnv Distillation Control
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /
tags:
  - openenv
---

# OpenEnv Distillation Control
Can a frozen 7B model survive a volatile chemical plant zero-shot?

We gave Qwen2.5-7B-Instruct the controls of a simulated industrial distillation column. No reflection loops. No GRPO. No active training. Just a strict JSON prompt, four continuous control valves, and a thermodynamic physics engine ready to explode.

Within a short episode horizon of pure zero-shot inference, it must balance thermodynamic energy, safely vent runaway pressure, and hit chemical purity targets.

This is OpenEnv Distillation Control: a mathematically grounded testbed where frozen LLMs enter an adversarial, coupled-physics environment that punishes unstable control.

**Meta x Scaler Hackathon Submission** | Built with OpenEnv v0.2.1 | Deployed on HF Spaces | Zero-shot inference with Qwen2.5-7B-Instruct

---

## The Story: Testing the Frozen Mind

### Act 1: The Cold Start (Stabilization)
Episode 1. The agent receives its first telemetry: "Temperature: 85.5 C, Pressure: 1.01 bar, Purity: 50.0%".

The system asks for a JSON payload. The `stabilization` task is relatively forgiving: the agent must hold the temperature near 100 C. By testing random continuous numbers, it learns that `steam_valve` drives the temperature, but it accidentally causes thermal runaway. The episode terminates early with a critical safety shutdown: `pressure >= 2.8` bar. Reward: `-1.0`.

### Act 2: Connecting the Physics (Optimization)
In the `optimization` task, the baseline difficulty spikes. The model must push `purity` above 57.5% and `flow_rate` over 12. It discovers that high `feed_rate` drops purity, and pushing `reflux_ratio` too high risks stagnation. It must discover the nonlinear physics that govern these variables through the environment's continuous reward signal, which actively applies an `energy_penalty` if the energy consumption curve grows too rapidly.

### Act 3: The Environment Fights Back (Emergency Control)
The ultimate zero-shot test. In `emergency_control`, the column starts artificially hot and pressurized (1.35 to 1.65 bar). Worse, there is stochastic fault injection: a 20% degradation in cooling efficiency.

Every step the agent survives without achieving stability, the pressure escalates. Oscillations cause compounding hidden instability. Random or static inputs cause cascading failure. It must systematically apply the `vent` valve while precisely backing off the steam. The heuristic baseline scores poorly on this task. Can a frozen LLM survive?

---

## Problem Statements Addressed
### Primary: Continuous Physics and Professional Tasks
The agent interacts with a rigorous physical simulation with continuous floating-point variables running on deterministic RNG, not discrete text states or mocked toggles.

- **Multi-step continuous workflows:** The LLM must output precise floats for `steam_valve`, `reflux_ratio`, and `feed_rate`, mapping `0.0` to `100.0%`.
- **Harsh safety bounds:** The environment aggressively terminates with `-1.0` reward if pressure or temperature cross critical physical limits.
- **Coupled variables:** It must deduce that increasing steam drives temperature, which directly causes nonlinear pressure buildups, which then requires emergency venting.

---

## How It Works

```text
+---------------------------- ZERO-SHOT INFERENCE LOOP ----------------------------+
|                                                                                  |
|  +-----------+   Observation   +------------+   Physics Engine                   |
|  | LLM Agent |<----------------| Tasks      |<-------------------------------+   |
|  |  (LLM)    |  T, P, Pur, F   | tasks.py   |                                |   |
|  |           |---------------->| Grader     |------------------------------->|   |
|  | Inference |   JSON Action   | grader.py  |  DistillationEnv (env.py)      |   |
|  +-----------+                 +------------+                                |   |
|                                                                                  |
+----------------------------------------------------------------------------------+
```

### The Loop
1. **Physics Engine** applies the previous continuous action through delayed actuators. It introduces deterministic noise and computes thermal inertia.
2. **Task Config** calculates a hidden instability factor. If the agent causes wild oscillations, the system artificially escalates the pressure.
3. **Agent** receives the telemetry and outputs a strict JSON payload deciding the next four control variables.
4. **Grader** computes a continuous reward based on distance to optimal states, action smoothness, and safety. The submission runner then blends average per-step reward with the raw episode score before emitting a final strict-open score.

---

## What Makes This Different
- **Zero-shot evaluation**: No training loops, no memory reflection buffers. This is a hard test of raw reasoning over continuous variables.
- **Continuous action space**: Not a text adventure. The agent is strictly bound to continuous controls like `48.5` and `61.2` against a Pydantic schema.
- **Actuator lag and hidden instability**: The environment has thermal mass; actions take time to manifest. Oscillating policies are explicitly tracked and punished.
- **Deterministic evaluation**: RNG is explicitly seeded with `np.random.default_rng`, making rollouts reproducible without removing stochastic difficulty.

---

## Architecture

```text
H100 GPU / API Client                               OpenEnv Distillation Engine
+--------------------------------+                  +---------------------------+
| FastAPI Server :7860           |                  | DistillationEnv           |
| - /reset                       |  JSON            | - Physics engine          |
| - /step                        |<---------------->| - Fault injection         |
| - /state                       |                  | - Task dynamics           |
| - /evaluate                    |                  |                           |
| - /run-agent                   |                  | Tasks                     |
|                                |                  | - stabilization           |
| inference.py                   |                  | - optimization            |
| - strict stdout logs           |                  | - emergency_control       |
| - blended final scoring        |                  |                           |
+--------------------------------+                  +---------------------------+
```

---

## Failure Types

| Type | What Triggers It | What Agent Must Do |
| --- | --- | --- |
| **Pressure Blowout** | Pressure >= 2.8 bar | Lower steam valve and turn `vent=1`. |
| **Thermal Runaway** | Temperature >= 145 C | Rapidly decrease `steam_valve`. |
| **Cascading Instability** | Oscillating actions cause compounding pressure | Maintain stable actions with low variance. |
| **Energy Waste** | High steam usage relative to flow rate | Dial back steam and optimize reflux cleanly. |

---

## Reward Signal
The reward function is continuous, bypassing typical sparse RL feedback mechanisms:
- **Purity and Flow (0.0 to 1.0):** Positive reward scaling with progression to optimization goals.
- **Temperature and Pressure Shaping:** Granular bonuses for operating inside favorable thermodynamic bands.
- **Variance Penalty (-0.0 to -0.2):** Penalizes jittery, oscillating control patterns that would stress actuators.
- **Energy Misuse Penalty:** If energy spikes too quickly relative to recent steps, the system applies a harsher penalty.
- **Critical Failure:** Safety limit breach triggers a fast `-1.0` and ends the episode instantly.

Internally, step rewards stay continuous and non-binary. For submission output, `inference.py` blends:

```text
0.7 * average_step_reward + 0.3 * raw_episode_score
```

Then the public score is sanitized only at the final emission stage so every reported task score remains strictly inside `(0, 1)`.

---

## Results
### Baseline Heuristic Agent
We wrote a deterministic threshold-based script (`random_agent` inside `grader.py`) to systematically map the baseline difficulty:

| Task | Score | Success Rate |
| --- | --- | --- |
| Stabilization | 0.812 | 100% |
| Optimization | 0.715 | 77% |
| **Emergency Control** | **0.414** | **0%** |

### Zero-Shot LLM Evaluation
Deploying Qwen2.5-7B-Instruct strictly zero-shot reveals the brutality of continuous physics control:
- **Hitting failure floors:** Fast zero-shot failure triggers `pressure_blowout` when the model treats `vent` like an analog dial instead of a discrete toggle.
- **Optimization limits:** Given reasonable temperature control, the LLM tends to push `reflux_ratio` upward while struggling to preserve flow against the energy penalty.
- **Emergency breakdown:** The degraded cooling efficiency and elevated starting pressure make `emergency_control` especially hard for frozen reasoning models without a reflection or training loop.

---

## Quick Start
```python
from app.models import Action
from app.runtime import OpenEnvRuntime

runtime = OpenEnvRuntime()
obs = runtime.reset(task_name="optimization", seed=42)
print(f"Start Temp: {obs.temperature:.1f} C")

action = Action(steam_valve=55.0, reflux_ratio=60.0, feed_rate=45.0, vent=0)
obs, reward, done, info = runtime.step(action)

print(f"Reward: {reward.value:.3f}")
```

## Inference Deployment
The inference loop and server run fluidly via `uvicorn`:

```bash
# Clone
git clone https://huggingface.co/spaces/Tushar-Projects/MetaHackathon
cd MetaHackathon
pip install -r requirements.txt

# Configure HF / OpenAI compatible API
cp .env.example .env

# Edit .env variables, then execute the evaluation
python inference.py
```

The submission runner emits structured stdout in the required three-line pattern only:

```text
[START] task=<task_name> env=openenv model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Rules enforced by the runner:
- `reward` and `rewards` are printed to 2 decimal places
- booleans are lowercase
- `score` on `[END]` is the exact same strict-open score stored internally for the episode
- stdout is limited to `[START]`, `[STEP]`, and `[END]` lines during evaluation

For local validation:

```bash
.venv\Scripts\openenv.exe validate
```

The FastAPI app also exposes `/run-agent` for validated multi-task rollouts.

## Configuration
`openenv.yaml` dictates thresholds across tasks:

| Variable | Description |
| --- | --- |
| `max_steps` | Task horizon before evaluation cutoff |
| `instability_temp_threshold` | Variance tolerance before pressure escalation |
| `cascading_pressure_rate` | Hard-task compounding penalty factor |

## Project Structure
```text
metahack/
|- inference.py              # Submission runner + strict [START]/[STEP]/[END] logs
|- server/
|  \- app.py                 # FastAPI server endpoints, including /run-agent
|- app/
|  |- env.py                 # Distillation physics and continuous environment reward
|  |- tasks.py               # Task objectives, shaping, failures, and escalation
|  |- grader.py              # Continuous episode scoring and task aggregation
|  |- scoring.py             # Strict-open public score sanitization helpers
|  |- runtime.py             # Task-aware reset/step/state runtime wrapper
|  |- baseline.py            # Deterministic heuristic baseline evaluator
|  |- config.py              # YAML fallback overrides
|  |- models.py              # Pydantic action/observation/reward schemas
|  \- __init__.py
|- tests/
|  |- test_scoring.py        # Score-range and logging regression tests
|  \- test_api.py            # API-level scoring checks
|- openenv.yaml              # OpenEnv metadata and task thresholds
|- Dockerfile                # Container build for deployment
|- requirements.txt          # Python runtime dependencies
|- pyproject.toml            # Package metadata
|- validate.sh               # Local submission validation helper
\- uv.lock                   # Locked dependency resolution
```
