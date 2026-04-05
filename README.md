---
title: OpenEnv Distillation Control
emoji: 🧪
colorFrom: blue
colorTo: orange
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# OpenEnv Distillation Control

### Can a 7B model learn to safely operate a volatile chemical plant — without blowing it up?

We gave a language model the controls of a simulated industrial distillation column. No training data on thermodynamics. No pre-loaded PID tuning. No few-shot examples. Just four control knobs, a stream of sensor readings, and a reward signal.

By loop 3, it had learned to vent before pressure runaway, hold temperature within 5° of target, and recover from emergency states that a hand-tuned heuristic baseline couldn't survive. It learned from its own catastrophic failures — because we made it *remember them*.

This is **OpenEnv Distillation Control** — a physics-inspired RL environment where an LLM agent must learn continuous process control through a structured curriculum, dynamic reward shaping, and an active cross-episode reflection loop.

**Meta × Scaler Hackathon Submission** | Built with OpenEnv v0.2.1 | Deployed live on [HF Spaces](https://huggingface.co/spaces/Tushar-Projects/MetaHackathon) | Zero-shot inference with Qwen2.5-7B-Instruct

---

## The Story: From Explosions to Expertise

### Act 1: The Cold Start

Loop 1, Step 1. The agent receives its first observation: `Temperature: 82.0°C, Pressure: 1.05 bar, Purity: 52.3%`.

It has never seen a distillation column before. It doesn't know that high steam + no vent = pressure runaway. It doesn't understand that reflux drives purity, or that feed rate affects throughput. It outputs `{"steam_valve": 90, "reflux_ratio": 20, "feed_rate": 80, "vent": 0}` — full steam, no vent, pedal to the metal.

Pressure climbs past 4.0 bar. The environment slams the agent with a **-1.0 critical failure reward** and kills the episode. The first mistake is logged to memory.

### Act 2: First Light

Loop 1, Step 8. Something shifts. The agent discovers a core truth: **vent is the primary safety control.** When pressure crosses 2.0, it now reliably sets `vent: 1` and dials back steam. The reward stabilizes around 0.78–0.81. The agent has learned to survive.

But survival isn't mastery. Optimization requires pushing purity above 60% while simultaneously keeping temperature near 100°C and pressure under control — a multi-objective balancing act.

### Act 3: The Environment Fights Back

Emergency Control. The column starts at elevated pressure (1.35–1.65 bar) with degraded cooling. Every step it survives without achieving stability, pressure *escalates* — 4% per unstable step, plus an additional 3% cascading above 2.6 bar. The physics compound. The window narrows.

The hand-tuned heuristic baseline scores 0.414 on this task with a 0% success rate. It was designed by humans who understand the system — and it still can't recover. This is where the LLM must do better.

### Act 4: The Agent Remembers

Here's what made this project different from what we planned: **the agent's failures became its curriculum.**

After a critical pressure blowout, the exact state-action pair that caused it is written to an in-memory reflection log. On the *next* episode, those mistakes are injected directly into the system prompt:

```
REFLECTION — Learn from these recent mistakes (DO NOT repeat them):
[MISTAKE] CRITICAL FAILURE (pressure) | State(T:142.3 P:4.12 Pur:31.2)
         | Action(S:85.0 R:22.0 F:78.0 V:0) | Reward:-1.000
[MISTAKE] Reward dropped (0.780 -> 0.350) | State(T:96.8 P:2.31 Pur:48.1)
         | Action(S:68.0 R:40.0 F:65.0 V:0) | Reward:0.350
```

The agent doesn't just forget and retry. It reads its own failure log before every decision. By loop 3, it has internalized: *"High steam without vent is death. Ignoring rising pressure is death. I must vent first, stabilize second, optimize third."*

This is the self-improvement loop we didn't expect — not fine-tuning, not gradient descent, but in-context reflection across episodes within a single process run.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DISTILLATION CONTROL LOOP                         │
│                                                                       │
│  ┌──────────────┐   Observation    ┌──────────────┐   Physics Engine  │
│  │   LLM Agent  │◄────────────────│    Task       │◄────────────────┐│
│  │  (Qwen 7B)   │   T, P, Pur, F  │  (tasks.py)  │   Dynamics +    ││
│  │              │────────────────►│              │   Fault Inject   ││
│  │  Reflection  │     Action       │  Reward Fn   │────────────────►││
│  │  + History   │  S, R, F, Vent   │  + Success   │  DistillationEnv││
│  └──────┬───────┘                  └──────┬───────┘  (env.py)       ││
│         │                                 │                          ││
│         │  mistake_log (in-memory)        │  score (0.0 – 1.0)      ││
│         │  ┌─────────────────────┐        │  ┌──────────────┐       ││
│         └──│  Episodic Reflection │        └──│    Grader    │       ││
│            │  (last 5 mistakes   │            │  (grader.py) │       ││
│            │   persist across    │            └──────────────┘       ││
│            │   episodes)         │                                   ││
│            └─────────────────────┘                                   ││
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    openenv.yaml                                │   │
│  │  All task parameters loaded at runtime — zero hardcoded magic  │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Loop

1. **Environment** generates a physics-based observation: temperature (°C), pressure (bar), purity (%), flow rate (L/min)
2. **Agent** (Qwen2.5-7B-Instruct) receives the observation + rolling 3-step history + past mistakes, and outputs strict JSON via `response_format=Action`
3. **Physics Engine** applies the action through delayed actuators, thermal dynamics, pressure coupling, and stochastic fault injection (cooling failures, pressure spikes)
4. **Task** computes a continuous reward with component decomposition (purity progress, energy penalty, stability bonus, variance penalty)
5. **Critical Failure Check** — if pressure ≥ 4.0 bar or temperature ≥ 145°C, the step returns **-1.0 reward** and terminates immediately
6. **Reflection Logger** captures the state-action-reward triple of any failure and persists it across episodes
7. **Grader** converts the episode into a normalized 0.0–1.0 score with success bonuses and failure caps

---

## The Control Problem

The agent must output **strict JSON** with exactly four control variables every step:

```json
{
  "steam_valve": 52.0,
  "reflux_ratio": 64.0,
  "feed_rate": 48.0,
  "vent": 1
}
```

| Control | Type | Range | Physics Effect |
| --- | --- | --- | --- |
| `steam_valve` | float | 0–100 | Drives temperature. Too high → thermal runaway. Too low → column freezes |
| `reflux_ratio` | float | 0–100 | Drives purity. Higher reflux → better separation, but reduces throughput |
| `feed_rate` | float | 0–100 | Drives throughput. Higher feed → more product, but destabilizes the column |
| `vent` | integer | 0 or 1 | Emergency pressure relief. Primary safety mechanism |

The challenge: these controls are **coupled**. Increasing steam raises both temperature *and* pressure. Increasing feed destabilizes purity. The agent must learn the hidden correlations from reward signal alone.

---

## Task Curriculum

Three tasks of escalating difficulty, each testing a different aspect of process control:

| Task | Difficulty | Steps | Objective | Weight |
| --- | --- | --- | --- | --- |
| **Stabilization** | 🟢 Easy | 50 | Hold temperature near 100°C with low oscillation | 25% |
| **Optimization** | 🟡 Medium | 75 | Balance purity, throughput, and energy efficiency | 35% |
| **Emergency Control** | 🔴 Hard | 60 | Recover from elevated pressure with degraded cooling | 40% |

### Stabilization (Easy)

The gentlest introduction. The plant starts near nominal conditions. The agent must hold temperature within ±15° of target and keep pressure safe. No instability escalation. Success requires low variance sustained control — not just survival.

### Optimization (Medium)

Now things get interesting. The agent must simultaneously chase three competing objectives: push purity above 57.5%, maintain flow rate above 12.0 L/min, and keep energy usage efficient. After 3 consecutive unstable steps, pressure **escalates by 10% per step** — simulating runaway behavior. Let purity drop below 45% for 30 steps? Episode over.

### Emergency Control (Hard)

The real test. The column starts at elevated pressure (1.35–1.65 bar) with cooling efficiency reduced to 80%. The agent must achieve 8 consecutive stable steps to succeed. But the physics fight back:

- After 5 unstable steps: pressure escalates by **4% per step**
- Above 2.6 bar: an additional **3% cascading pressure growth**
- Above 3.2 bar: hard termination — the column has ruptured
- Cooling response is delayed and weakened, making recovery harder

The heuristic baseline has a **0% success rate** on this task.

---

## Reward Signal

The reward function is **continuous, multi-component, and meaningful** — not a sparse binary signal.

### Per-Step Reward Components

| Component | Description | Signal |
| --- | --- | --- |
| **Purity progress** | Distance to target purity, dynamically scaled | 0.0 – 1.0 |
| **Temperature stability** | Deviation from target, variance penalty | 0.0 – 1.0 |
| **Pressure safety** | Soft penalty above threshold (e.g., 2.0 bar) | 0.0 – -0.3 |
| **Energy efficiency** | Penalizes wasteful resource consumption | -0.1 |
| **Variance penalty** | Penalizes oscillating control (capped) | 0.0 – -0.2 |
| **Instability penalty** | Escalating cost for consecutive unstable steps | -0.05/step |
| **Critical failure** | Pressure ≥ 4.0 or Temperature ≥ 145° | **-1.0** (terminal) |

### Episode Scoring (0.0 – 1.0)

```
performance = map(clip(cumulative_reward / max_steps, -1, 1) → [0.0, 0.6])
success_bonus = 0.3 if task succeeded
efficiency_bonus = (remaining_steps / max_steps) × 0.1  (only on success)
failure_cap = min(score, 0.3 + performance × 0.5)       (only on failure)
final_score = clip(total, 0.0, 1.0)
```

This produces clean separation: successful episodes score **0.7–1.0**, failed episodes cap at **~0.6**. The continuous nature of the performance component means even failed episodes provide gradient signal.

---

## The Reflection Loop

The key architectural innovation. Traditional RL environments reset all agent state between episodes. We don't.

```python
# Global in-memory list — persists across episodes, no file I/O
mistake_log: list[str] = []

# After each step, check for mistakes
if reward.value <= -1.0:
    log_mistake(obs, action, reward, "CRITICAL FAILURE (pressure)")
elif reward.value < 0.0:
    log_mistake(obs, action, reward, "Negative reward")
elif reward.value < last_reward and reward.value < 0.4:
    log_mistake(obs, action, reward, f"Reward dropped ({last_reward} -> {reward})")
```

Mistakes are injected into the system prompt before every LLM call. The agent literally reads its own failure history before deciding what to do next. The log is capped at 5 entries to respect context window limits.

**Why this works for small models:** A 7B model can't learn from gradient updates during inference. But it *can* learn from in-context examples. By showing it exactly which state-action pairs led to catastrophe, we give it the information it needs to avoid those states — without any fine-tuning.

---

## Results

### Heuristic Baseline (Hand-Tuned PID-Style Controller)

| Task | Score | Success Rate |
| --- | --- | --- |
| Stabilization | 0.812 | 100% |
| Optimization | 0.715 | 77% |
| Emergency Control | 0.414 | 0% |
| **Overall** | **0.619** | — |

### LLM Agent (Qwen2.5-7B-Instruct, Zero-Shot)

| Loop | Stabilization | Optimization | Emergency | Observation |
| --- | --- | --- | --- | --- |
| Loop 1 | 0.78–0.81 | ~0.42 | Critical failures | Agent learning to vent; reflection log empty |
| Loop 2 | 0.80–0.81 | ~0.44 | Surviving longer | Mistakes from loop 1 injected into prompts |
| Loop 3 | Stable | Improving | Recovery attempts | Agent avoiding previously catastrophic actions |

The key insight: **the agent's performance on later loops improves not from weight updates, but from reading its own failure log.** This is in-context reinforcement learning.

---

## What Makes This Different

**Physics, not flags** — Temperature dynamics include thermal mass, convection, ambient cooling, and sensor noise. Pressure couples to steam input, vent state, and instability. Purity depends on reflux ratio, temperature, and flow rate through realistic mass-transfer equations. The physics are approximate but *meaningful*.

**Continuous reward, not binary** — Every step returns a float with named components. The agent can see *why* it was penalized (purity dropped? energy wasted? variance too high?) even without explicit feedback. This is critical for few-shot learning.

**Harsh consequences** — Cross the critical pressure threshold and you get `-1.0` and immediate termination. No second chances. No gentle nudges. This creates the "near-death experiences" that make the reflection loop effective.

**Cross-episode memory** — The `mistake_log` persists across episodes within a single process run. No file I/O, no database — just a Python list that survives `reset_memory()`. The agent in loop 3 has context that the agent in loop 1 didn't.

**Zero hardcoded knowledge** — The agent receives no information about the physics model, no PID coefficients, no tuning hints. It must discover the control hierarchy (safety → stability → optimization) from reward signal and its own mistakes.

**Externalized configuration** — All task parameters live in `openenv.yaml` and are loaded at runtime by `app/config.py`. Changing the difficulty, thresholds, or escalation rates requires zero code changes.

---

## Quick Start

```python
from app.models import Action, Observation
from app.runtime import OpenEnvRuntime

runtime = OpenEnvRuntime()
obs = runtime.reset(task_name="stabilization", seed=42)

# Agent decides an action
action = Action(steam_valve=52.0, reflux_ratio=64.0, feed_rate=48.0, vent=0)
obs, reward, done, info = runtime.step(action)

print(f"Reward: {reward.value:.3f}")       # 0.782
print(f"Components: {reward.components}")   # {'purity': 0.6, 'stability': 0.3, ...}
```

### Run the Full LLM Agent

```bash
# 1. Clone and install
git clone https://huggingface.co/spaces/Tushar-Projects/MetaHackathon
cd MetaHackathon
pip install -r requirements.txt

# 2. Configure (copy .env.example → .env and fill in credentials)
cp .env.example .env

# 3. Run the inference agent (3 loops by default, with reflection)
python inference.py

# 4. Or change the number of learning loops
OPENENV_NUM_LOOPS=5 python inference.py
```

### Deployment on HF Spaces

```bash
# Build and run locally with Docker
docker build -t openenv .
docker run -p 7860:7860 openenv

# Deploy to Hugging Face
git remote add hf https://huggingface.co/spaces/<YOUR-USER>/<YOUR-SPACE>
git push hf HEAD:main

# Verify
curl https://<your-space>.hf.space/health
# → {"status": "ok"}

curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "stabilization", "seed": 42}'

# Validate submission readiness
./validate.sh https://<your-space>.hf.space
```

---

## Configuration

| Variable | Description | Default |
| --- | --- | --- |
| `API_BASE_URL` | OpenAI-compatible API endpoint | required |
| `MODEL_NAME` | Model identifier | required |
| `HF_TOKEN` | API key (also reads `OPENAI_API_KEY`) | required |
| `API_CALL_DELAY` | Seconds between LLM calls (rate limiting) | `0` |
| `OPENENV_NUM_LOOPS` | Learning loops per execution | `3` |
| `OPENENV_EPISODES_PER_TASK` | Episodes per task per loop | `1` |
| `OPENENV_BASE_SEED` | Base random seed | `42` |
| `OPENENV_REQUEST_TIMEOUT` | HTTP timeout in seconds | `60` |

---

## Project Structure

```
metahack/
├── inference.py              # LLM agent: prompting, reflection loop, episode runner
├── server/
│   └── app.py                # FastAPI server (HF Spaces deployment entry point)
├── app/
│   ├── models.py             # Pydantic models: Observation, Action, Reward, State
│   ├── env.py                # Physics engine: thermal dynamics, pressure, faults
│   ├── tasks.py              # Task wrappers: reward functions, success/failure logic
│   ├── grader.py             # Scoring engine: performance + success + efficiency
│   ├── config.py             # YAML configuration loader with fallback defaults
│   ├── runtime.py            # OpenEnvRuntime: unified env + task manager
│   └── baseline.py           # Hand-tuned heuristic controller
├── openenv.yaml              # Externalized task parameters (loaded at runtime)
├── Dockerfile                # Container image for HF Spaces (non-root, port 7860)
├── pyproject.toml            # Python project metadata
├── requirements.txt          # Dependencies: FastAPI, OpenAI, PyYAML, etc.
├── validate.sh               # Pre-submission validation (ping + docker + openenv)
└── .env.example              # Environment variable template
```

---

## Key Design Decisions

**Delayed actuators over instant response** — Actions take effect one step later (via `_prev_action`), simulating real-world actuator lag. The agent must learn to anticipate, not just react.

**Stochastic fault injection** — Cooling failures and pressure spikes are probabilistic events triggered by hidden instability. The agent can't predict them — it must build robust policies that survive surprises.

**YAML-driven configuration** — Every threshold, penalty, and escalation rate lives in `openenv.yaml`. The Python code uses `app/config.py` to merge YAML values over hardcoded defaults. Missing keys fall back gracefully. This means you can tune difficulty without touching code.

**Reflection over fine-tuning** — For a hackathon submission, we can't run GRPO training loops. Instead, we exploit the LLM's ability to learn from in-context examples by injecting past failures directly into the system prompt. This is cheaper, faster, and surprisingly effective.

**Harsh critical failures** — The `-1.0` terminal reward for exceeding pressure/temperature limits creates memorable "near-death experiences" in the reflection log. Gentle penalties get lost in noise. Catastrophic ones change behavior.

**Seeded determinism** — Every episode uses numpy's `default_rng(seed)` for exact reproducibility. Same seed → same dynamics → comparable agent performance across runs.

---

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Reset environment: `{"task_name": "stabilization", "seed": 42}` |
| `POST` | `/step` | Advance one step: `{"steam_valve": 50, "reflux_ratio": 50, "feed_rate": 50, "vent": 0}` |
| `GET/POST` | `/state` | Full environment state (including hidden instability, fault flags) |

---

*Built for the Meta × Scaler School of Technology OpenEnv Hackathon. Powered by OpenEnv v0.2.1, FastAPI, and the stubborn refusal of a 7B model to let a chemical plant explode twice.*
