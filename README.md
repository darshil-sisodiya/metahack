---
title: OpenEnv Distillation Control
sdk: docker
app_port: 7860
pinned: false
colorFrom: blue
colorTo: green
tags:
  - openenv
  - industrial-control
  - fastapi
---

# OpenEnv Distillation Control

OpenEnv Distillation Control is a real-world industrial process-operations benchmark.
It simulates the task of operating a chemical distillation column: keeping the plant
stable, balancing throughput against purity and energy cost, and recovering from
dangerous pressure conditions.

The project exposes the standard OpenEnv interaction surface through typed
`reset()`, `step()`, and `state()` APIs, plus matching FastAPI endpoints for
containerized deployment.

## Why This Is A Real-World Task

Human operators and control engineers perform exactly this class of work:

- hold process variables near safe operating targets
- balance output quality, throughput, and energy use
- respond to escalating safety conditions under delayed dynamics
- recover equipment from abnormal pressure states without causing shutdowns

This benchmark is not a game and not a toy workflow. It models industrial
process control and safety decision-making.

## Environment Overview

The repo is organized into two layers:

- `app/env.py`: the physics-inspired process simulator
- `app/runtime.py`: the OpenEnv runtime that turns the simulator into a task-aware
  `reset/step/state` environment with typed rewards and state

Supporting files:

- `app/models.py`: typed Pydantic models for observation, action, reward, and state
- `app/tasks.py`: three graded tasks with increasing difficulty
- `app/grader.py`: deterministic 0.0-1.0 scoring
- `app/baseline.py`: deterministic heuristic baseline
- `app/main.py`: FastAPI app exposing the OpenEnv HTTP API
- `inference.py`: root submission script with structured stdout logging
- `openenv.yaml`: environment metadata and endpoint/task declarations

## API Surface

The environment supports both in-process Python usage and HTTP usage.

### Python API

```python
from app.models import Action
from app.runtime import OpenEnvRuntime

runtime = OpenEnvRuntime()
obs = runtime.reset(task_name="optimization", seed=42)
state = runtime.state()
obs, reward, done, info = runtime.step(
    Action(steam_valve=50, reflux_ratio=50, feed_rate=50, vent=0)
)
```

### HTTP API

- `GET /health` -> health check
- `POST /reset` -> returns initial `Observation`
- `POST /step` -> returns `StepResponse` with typed `Reward`
- `GET /state` and `POST /state` -> returns full typed `EnvironmentState`

Example:

```bash
curl -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d "{\"task_name\":\"optimization\",\"seed\":42}"

curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"steam_valve\":50,\"reflux_ratio\":50,\"feed_rate\":50,\"vent\":0}"

curl http://127.0.0.1:7860/state
```

## Observation, Action, Reward, And State

### Observation

The observation returned by `reset()` and `step()` contains:

- `temperature`
- `pressure`
- `purity`
- `flow_rate`
- `energy_usage`
- `time_step`

### Action

The controller must emit:

- `steam_valve`: float in `[0, 100]`
- `reflux_ratio`: float in `[0, 100]`
- `feed_rate`: float in `[0, 100]`
- `vent`: integer `0` or `1`

### Reward

`step()` returns a typed `Reward` model:

- `value`: scalar reward used by the grader
- `components`: reward breakdown for partial-progress signals and penalties

### State

`state()` returns the full typed `EnvironmentState`, including:

- the active task name
- current process variables
- latent instability and fault flags
- the previous action

## Tasks And Difficulty

The benchmark includes three tasks with explicit easy -> medium -> hard difficulty.

### 1. Stabilization

Difficulty: easy

Goal:

- keep temperature near 100 C
- keep pressure safely below shutdown thresholds
- finish with low oscillation

Why it is easy:

- no instability escalation
- forgiving initial conditions
- success is mostly about steady control

### 2. Optimization

Difficulty: medium

Goal:

- improve purity
- maintain usable throughput
- avoid wasting energy

Why it is medium:

- instability can escalate into pressure growth
- poor purity over many steps causes failure
- reward must balance multiple competing objectives

### 3. Emergency Control

Difficulty: hard

Goal:

- recover from elevated pressure
- handle delayed cooling behavior
- maintain enough stability to avoid runaway failure

Why it is hard:

- starts closer to dangerous operating conditions
- adds delayed and weakened cooling response
- requires deliberate vent use and recovery planning

## Reward Design

Rewards provide trajectory-level signal rather than only end-of-episode success.

- `Stabilization`: rewards target temperature and safe pressure, penalizes variance
- `Optimization`: rewards purity and throughput, penalizes energy use and sustained instability
- `Emergency Control`: rewards pressure reduction and recovery, penalizes escalation

Each task also includes interpretable reward components in the returned
`Reward.components` field so partial progress is visible at every step.

## Deterministic Grading

`app/grader.py` evaluates all tasks with deterministic seeds and returns normalized
scores in the required `0.0-1.0` range.

- `evaluate_episode(...)` returns a single-task normalized score
- `evaluate_all_tasks(...)` returns:
  - `overall_score`
  - `task_scores`
  - `details`

## Baseline Results

Current deterministic baseline results from `python -m app.baseline`:

| Metric | Score |
| --- | --- |
| Overall score | `0.619` |
| Stabilization | `0.812` |
| Optimization | `0.715` |
| Emergency control | `0.414` |

Success rates from the same run:

- stabilization: `100.0%`
- optimization: `77.0%`
- emergency_control: `0.0%`

These are intended as a reproducible reference point, not a learned upper bound.

## Inference Script

The required root-level submission script is `inference.py`.

Submission behavior:

- uses the OpenAI-compatible Python client in LLM mode
- reads environment variables from `.env` or the process environment
- runs deterministic seeded episodes on all three tasks
- emits strict stdout logs with one `[START]`, many `[STEP]`, and one `[END]` per episode
- adapts actions step by step from the current process state

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `OPENAI_API_KEY` (optional compatibility fallback)
- `OPENENV_AGENT_MODE` (`llm` or `heuristic`)
- `OPENENV_EPISODES_PER_TASK`
- `OPENENV_BASE_SEED`
- `API_CALL_DELAY`

Local offline smoke test:

```bash
$env:OPENENV_AGENT_MODE="heuristic"
python inference.py
```

Submission-mode run:

```bash
python inference.py
```

`inference.py` defaults to `OPENENV_AGENT_MODE=llm` and will fail fast if
`API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`/`OPENAI_API_KEY` are missing.

The default configuration is one deterministic episode per task so runtime stays
well within the 20-minute infrastructure limit on a modest CPU machine.

Episode log format:

```text
[START] task=<task_name> env=openenv model=<model_name>
[STEP] step=<n> action=steam=<val>,reflux=<val>,feed=<val>,vent=<val> reward=<0.00> done=<true|false> error=null
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

The LLM prompt is state-aware and instructs the model to:

- keep temperature near 90
- vent and reduce steam when pressure becomes unsafe
- raise reflux when purity is too low
- avoid repeating identical actions when the state has changed

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run The API Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Run The Baseline

```bash
python -m app.baseline
```

### Run The Task-Difficulty Probe

```bash
python test.py
```

## Docker

Build:

```bash
docker build -t openenv-distillation-control .
```

Run:

```bash
docker run -p 7860:7860 openenv-distillation-control
```

The container entrypoint serves the FastAPI app on port `7860`.

## Hugging Face Spaces

This repo is configured for a Docker Space:

- README front matter sets `sdk: docker`
- README metadata includes the `openenv` tag
- the app serves on `7860`, matching `app_port`
- `Dockerfile` runs the FastAPI server directly

To deploy:

1. Create a new Hugging Face Space using Docker SDK.
2. Push this repository contents to the Space.
3. Set the Space secrets for `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` if you
   want `inference.py` to call a remote LLM.
4. Confirm the deployed Space returns `200` from `/health` and responds to `/reset`.

## Validation Notes

Local checks completed on this repo:

- `/health`, `/reset`, `/step`, and `/state` respond correctly
- `/step` returns typed `Reward` payloads
- grading returns normalized `0.0-1.0` scores
- the three tasks are available as easy, medium, and hard
- `python -m app.baseline` runs successfully
- `python inference.py` runs successfully in deterministic heuristic mode with the required plain-text episode log format

## License

This project is provided as a benchmark/research environment for OpenEnv-style
agent evaluation.
