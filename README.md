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
Can a frozen 7B model survive a volatile chemical plant — zero-shot?

We gave Qwen2.5-7B-Instruct the controls of a simulated industrial distillation column. No reflection loops. No GRPO. No active training. Just a strict JSON prompt, four continuous control valves, and a thermodynamic physics engine ready to explode.

Within a short episode horizon of pure zero-shot inference, it must balance thermodynamic energy, safely vent runaway pressure, and hit chemical purity targets.

This is OpenEnv Distillation Control — a ruthless, mathematically sound testing ground where frozen LLMs enter an adversarial, coupled-physics environment that actively punishes bad behavior.

**Meta × Scaler Hackathon Submission** | Built with OpenEnv v0.2.1 | Deployed live on HF Spaces | Zero-shot inference with Qwen2.5-7B-Instruct

---

## The Story: Testing the Frozen Mind

### Act 1: The Cold Start (Stabilization)
Episode 1. The agent receives its first telemetry: "Temperature: 85.5°C, Pressure: 1.01 bar, Purity: 50.0%".

The system asks for a JSON payload. The `Stabilization` task is relatively forgiving — the agent must hold the temperature near 100°C. By testing random continuous numbers, it learns that `steam_valve` drives the temperature, but it accidentally causes thermal runaway. The episode terminates early with a critical safety shutdown: `pressure >= 2.8` bar. Reward: -1.0.

### Act 2: Connecting the Physics (Optimization)
In the `Optimization` task, the baseline difficulty spikes. The model must push `purity` above 57.5% and `flow_rate` over 12. It discovers that high `feed_rate` drops purity, and pushing `reflux_ratio` too high risks stagnation. It must discover the nonlinear physics that govern these variables—purely through the environment's continuous reward signal, which actively applies an `energy_penalty` if the energy consumption curve grows too rapidly.

### Act 3: The Environment Fights Back (Emergency Control)
The ultimate zero-shot test. In `Emergency Control`, the column starts artificially hot and pressurized (1.35-1.65 bar). Worse, there is stochastic fault injection: a 20% degradation in cooling efficiency. 

Every step the agent survives without achieving stability, the pressure escalates. Oscillations cause compounding "hidden instability". Random or static inputs cause cascading failure. It must systematically apply the `vent` valve while precisely backing off the steam. The heuristic baseline (a hand-written agent) scores a 0% success rate on this task. Can a frozen LLM survive?

---

## Problem Statements Addressed
### Primary: Continuous Physics & Professional Tasks
The agent interacts with a rigorous physical simulation with continuous floating-point variables running on deterministic RNG, not discrete text states or mocked toggles. 

- **Multi-step continuous workflows:** The LLM must output precise floats for `steam_valve`, `reflux_ratio`, and `feed_rate`, mapping 0.0 to 100.0%.
- **Harsh Safety Bounds:** The environment aggressively terminates with -1.0 reward if pressure or temperature cross critical physical limits.
- **Coupled Variables:** It must deduce that increasing steam drives temperature, which directly causes non-linear pressure buildups, which then requires emergency venting.

---

## How It Works

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ZERO-SHOT INFERENCE LOOP                        │
│                                                                        │
│  ┌──────────────┐   Observation   ┌──────────────┐   Physics Engine    │
│  │  LLM Agent   │◄───────────────│    Tasks     │◄────────────────┐   │
│  │ (Qwen 7B)    │  T, P, Pur, F  │  (tasks.py)  │   Dynamics +    │   │
│  │              │────────────────►│              │   Fault Inject  │   │
│  │   Zero-Shot  │    JSON Action  │   Grader     │────────────────►│   │
│  │   Inference  │  S, R, F, Vent  │ (grader.py)  │ DistillationEnv │   │
│  └──────────────┘                 └──────────────┘    (env.py)     │   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### The Loop
1. **Physics Engine** applies the previous continuous action through delayed actuators. It introduces deterministic noise and computes thermal inertia. 
2. **Task Config** calculates a hidden instability factor. If the agent causes wild oscillations, the system artificially escalates the pressure. 
3. **Agent (Qwen2.5-7B-Instruct)** receives the telemetry and outputs a strict JSON payload deciding the next 4 control variables.
4. **Grader** computes a continuous reward based on distance to optimal states, applying deep penalties for excessive action-switching or wasted energy. 

---

## What Makes This Different
- **Zero-Shot Evaluation** — No training loops, no memory reflection buffers. This is a cruel testing ground for raw reasoning over continuous variables.
- **Continuous Action Space** — Not a text adventure. The agent is strictly bound to tweaking continuous floats like `48.5` and `61.2` against a Pydantic schema.
- **Actuator Lag & Hidden Instability** — The environment possesses thermal mass; actions take time to manifest. Reactionary, oscillating policies are explicitly tracked and punished by compounding pressure bounds.
- **Deterministic Evaluation** — RNG is explicitly seeded (`np.random.default_rng`), ensuring LLM benchmarking is highly reproducible without losing stochastic difficulty.

---

## Architecture

```text
H100 GPU (80GB)                                  OpenEnv Distillation Engine
┌──────────────────────────────────┐          ┌───────────────────────────┐
│                                  │          │                           │
│  FastAPI Server :7860            │  JSON    │  DistillationEnv          │
│  ├─ Environment (reset/step)     │◄────────►│   Physics Engine          │
│  ├─ YAML Config Loader           │          │                           │
│  ├─ Distillation Matrix          │          │  Task Curriculums         │
│  └─ Inference Client             │          │   Stabilization (Easy)    │
│                                  │          │   Optimization (Medium)   │
│  Agent: Qwen2.5-7B-Instruct      │          │   Emergency (Hard)        │
│  (Hugging Face API / Local)      │          │                           │
└──────────────────────────────────┘          └───────────────────────────┘
```

---

## Failure Types

| Type | What Triggers It | What Agent Must Do |
| --- | --- | --- |
| **Pressure Blowout** | Pressure >= 2.8 bar (Task configuration) | Lower steam valve, turn `vent: 1`. |
| **Thermal Runaway** | Temperature >= 145°C | Rapidly decrease `steam_valve`. |
| **Cascading Instability** | Oscillating actions causing compounding pressure | Maintain stable actions (low variance) across multiple steps. |
| **Energy Waste** | High steam usage relative to flow rate | Dial back steam and optimize reflux ratio cleanly. |

---

## Reward Signal
The reward function is **continuous**, bypassing typical sparse RL feedback mechanisms:
- **Purity & Flow (0.0 to 1.0)** — Direct positive reward scaling linearly with progression to optimization goals.
- **Temperature & Pressure Shaping** — Granular bonuses for resting inside optimal thermodynamic bounds without wild adjustments.
- **Variance Penalty (-0.0 to -0.2)** — Penalizes jittery, oscillating control patterns that would stress physical actuators.
- **Energy Misuse Penalty** — If cumulative energy spikes too quickly relative to past steps, the system incurs a harsh percentage-based penalty.
- **Critical Failure** — Terminating safety limit breach triggers a fast `-1.0` and ends the episode instantly.

This layered continuous calculation allows the `Grader` to evaluate success between exactly 0.0 and 1.0.

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
- **Hitting failure floors:** Fast zero-shot failure triggers `pressure_blowout` when the model treats `vent` like an analog dial rather than recognizing it as a discrete toggle.
- **Optimization limits:** Given reasonable temperature controls, the LLM will naturally maximize the `reflux_ratio` upwards while struggling to protect against flow loss, hitting a ceiling against the `Energy Penalty`.
- **Emergency Breakdown:** Validates that this benchmark—with degraded cooling efficiency and heavy starting pressure—is remarkably hard for frozen reasoning models to organically conquer without a reflection or training loop.

---

## Quick Start
```python
from app.models import Action, Observation
from app.runtime import OpenEnvRuntime

runtime = OpenEnvRuntime()
obs = runtime.reset(task_name="optimization", seed=42)
print(f"Start Temp: {obs.temperature:.1f}°C")

# The frozen LLM dictates this continuous action payload:
action = Action(steam_valve=55.0, reflux_ratio=60.0, feed_rate=45.0, vent=0)
obs, reward, done, info = runtime.step(action)

# The environment continuously grades performance:
print(f"Reward: {reward.value:.3f}")
```

## Inference Deployment
The inference loops and server run fluidly via `uvicorn`:
```bash
# Clone
git clone https://huggingface.co/spaces/Tushar-Projects/MetaHackathon
cd MetaHackathon
pip install -r requirements.txt

# Configure HF / OpenAI compatible API
cp .env.example .env
# Edit .env variables, then execute the evaluation:
python inference.py
```

## Configuration
`openenv.yaml` dictates thresholds seamlessly across Tasks:
| Variable | Description |
| --- | --- |
| `max_steps` | Task horizon before evaluation completes cutoff |
| `instability_temp_threshold` | Variance tolerance before pressure escalation initiates |
| `cascading_pressure_rate` | Hard task compounding penalty factor |

## Project Structure
```text
metahack/
├── inference.py              # Pure zero-shot LLM evaluation runner
├── server/
│   └── app.py                # FastAPI server (HF Spaces endpoint)
├── app/
│   ├── env.py                # Distillation physics + continuous dynamics
│   ├── tasks.py              # Task difficulty logic + reward calculators
│   ├── grader.py             # 0.0-1.0 deterministic evaluation + heuristic
│   ├── config.py             # YAML fallback overrides
│   └── models.py             # Pydantic Action/Observation schemas
├── openenv.yaml              # Escalation and failure thresholds
└── pyproject.toml            # Dependencies
```
