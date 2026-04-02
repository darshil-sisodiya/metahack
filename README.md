# OpenEnv — Distillation Column Control Benchmark

> **Meta × Scaler Hackathon Submission** — An LLM-driven agent that controls a simulated industrial distillation column across three progressively harder tasks.

OpenEnv is a lightweight process-control benchmark built around a physics-inspired distillation column simulator. An LLM agent receives real-time plant observations (temperature, pressure, purity, flow rate) and must return control actions (steam valve, reflux ratio, feed rate, vent) to keep the system stable, efficient, and safe.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Deployment on Hugging Face Spaces](#deployment-on-hugging-face-spaces)
- [Running the LLM Inference Script](#running-the-llm-inference-script)
- [Environment Variables](#environment-variables)
- [Repository Layout](#repository-layout)
- [Architecture Overview](#architecture-overview)
- [API Endpoints](#api-endpoints)
- [Tasks](#tasks)
- [Scoring System](#scoring-system)
- [Configuration via openenv.yaml](#configuration-via-openenvyaml)
- [Validation](#validation)
- [Helper Scripts](#helper-scripts)
- [Heuristic Baseline Results](#heuristic-baseline-results)

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for container builds and HuggingFace deployment)
- An OpenAI-compatible API key (HuggingFace, Gemini, OpenAI, or local vLLM/LM Studio)

### 1. Clone and install

```bash
git clone https://huggingface.co/spaces/Tushar-Projects/MetaHackathon
cd MetaHackathon

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your real credentials:

```env
API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME="gemini-2.5-flash"
HF_TOKEN="your-actual-api-key"
```

> **Security:** `.env` is listed in `.gitignore` — your secrets will never be committed.

### 3. Run the API server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

The server exposes the OpenEnv environment as a REST API at `http://localhost:7860`.

### 4. Run the LLM inference agent

```bash
python inference.py
```

This runs the LLM agent across all three tasks and emits structured `[START]`/`[STEP]`/`[END]` logs with per-task and overall scores.

### 5. Run the heuristic baseline (no LLM needed)

```bash
python -m app.baseline
```

---

## Deployment on Hugging Face Spaces

This project is deployed as a Docker-based HuggingFace Space. The Space serves the OpenEnv API so the hackathon evaluator can call `/reset`, `/step`, and `/state` remotely.

### Deploying to your own HF Space

**Step 1: Create a Docker Space on Hugging Face**

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and create a new Space with **SDK: Docker**.

**Step 2: Add the HF Space as a git remote**

```bash
git remote add hf https://huggingface.co/spaces/<YOUR-USERNAME>/<YOUR-SPACE-NAME>
```

**Step 3: Push your code**

```bash
git push hf HEAD:main
```

HuggingFace will automatically build the Docker image and start the server. Wait 2-3 minutes for the build to complete.

**Step 4: Verify the deployment**

```bash
# Health check
curl https://<your-space>.hf.space/health
# Expected: {"status":"ok"}

# Reset the environment
curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "stabilization", "seed": 42}'

# Step the environment
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"steam_valve": 50, "reflux_ratio": 50, "feed_rate": 50, "vent": 0}'
```

### Building and running with Docker locally

```bash
# Build the image
docker build -t openenv .

# Run the container
docker run -p 7860:7860 openenv
```

The server will be available at `http://localhost:7860`.

> **Note:** The Dockerfile creates a non-root `user` (UID 1000) as required by Hugging Face Spaces, and the `CMD` points to `server.app:app` on port `7860`.

---

## Running the LLM Inference Script

`inference.py` is the **mandatory hackathon submission entry point**. It uses the OpenAI-compatible Python client to drive an LLM agent through all three OpenEnv tasks.

### How it works

1. Loads environment variables from `.env` via `python-dotenv`
2. Initializes an OpenAI client pointed at the configured API endpoint
3. For each task, runs seeded episodes:
   - Builds a compact prompt from the current `Observation` (state + rolling 3-step history + static control hierarchy)
   - Sends it to the LLM via `CLIENT.beta.chat.completions.parse` with `response_format=Action`
   - Validates the response against Pydantic constraints (with graceful fallback on out-of-bounds values)
   - Steps the environment via the runtime and logs the result

### Running

```bash
# Make sure .env is configured, then:
python inference.py
```

### Output format

The script emits structured logs compatible with the hackathon evaluator:

```
[START] task=stabilization env=openenv model=gemini-2.5-flash
[STEP] step=1 action=steam=45.0000,reflux=55.0000,feed=50.0000,vent=0 reward=0.77 done=false error=null
[STEP] step=2 action=steam=42.0000,reflux=58.0000,feed=48.0000,vent=0 reward=0.78 done=false error=null
...
[END] success=true steps=50 rewards=0.77,0.78,...
```

### Rate limiting

Free-tier APIs enforce strict rate limits. The script has a configurable delay:

```env
# In your .env file:
API_CALL_DELAY="15"   # seconds between API calls (0 for paid tiers)
```

| Scenario | Recommended `API_CALL_DELAY` |
| --- | --- |
| **Free-tier Gemini** (5 RPM) | `15` |
| **Paid OpenAI / vLLM** | `0` |
| **HuggingFace Inference API** | `5`–`10` |

---

## Environment Variables

All configuration is via environment variables loaded from `.env`:

| Variable | Required | Description | Example |
| --- | --- | --- | --- |
| `API_BASE_URL` | Yes | OpenAI-compatible API endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `MODEL_NAME` | Yes | Model identifier | `gemini-2.5-flash` |
| `HF_TOKEN` | Yes | API key (also reads `OPENAI_API_KEY`, `API_KEY` as fallbacks) | `your-api-key` |
| `API_CALL_DELAY` | No | Seconds between LLM calls (default `0`) | `15` |
| `OPENENV_EPISODES_PER_TASK` | No | Episodes per task (default `1`) | `1` |
| `OPENENV_BASE_SEED` | No | Base random seed (default `42`) | `42` |
| `OPENENV_REQUEST_TIMEOUT` | No | HTTP timeout in seconds (default `60`) | `60` |

---

## Repository Layout

```
metahack/
├── inference.py              # Hackathon LLM inference entry point
├── server/
│   └── app.py                # FastAPI server (HF Spaces entry point)
├── app/
│   ├── __init__.py            # Package marker
│   ├── models.py              # Pydantic models: Observation, Action, Reward
│   ├── env.py                 # Physics simulator (DistillationEnv)
│   ├── tasks.py               # Task wrappers: reward, success, failure logic
│   ├── grader.py              # Scoring engine (0-100 scores)
│   ├── baseline.py            # Hand-tuned heuristic controller
│   ├── config.py              # YAML configuration loader
│   └── runtime.py             # OpenEnvRuntime (unified env+task manager)
├── openenv.yaml               # Task parameters (loaded at runtime)
├── Dockerfile                 # Container image for HF Spaces
├── pyproject.toml             # Python project metadata
├── requirements.txt           # pip dependencies
├── validate.sh                # Pre-submission validation script
├── .env.example               # Environment variable template
├── .gitignore                 # Protects .env from commits
├── test.py                    # Random-policy difficulty probe
└── test_grader.py             # Grader smoke test
```

---

## Architecture Overview

```
┌──────────────┐   Observation   ┌──────────────┐   task.step()   ┌──────────────┐
│   LLM Agent  │ ◄──────────── │    Task       │ ◄──────────── │  Distillation │
│ (inference.py│ ──────────────►│  (tasks.py)   │ ──────────────►│  Env (env.py) │
└──────────────┘    Action       └──────────────┘                └──────────────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │   Grader      │
                                │ (grader.py)   │
                                └──────────────┘
```

1. The agent receives an `Observation` (temperature, pressure, purity, flow rate, energy, time step)
2. The agent returns an `Action` (steam_valve, reflux_ratio, feed_rate, vent)
3. The `Task` applies the action through `task.step(env, action)`
4. The `DistillationEnv` advances the physical state with delayed dynamics, noise, and faults
5. The task computes reward, checks success/failure, and applies escalation logic
6. The `Grader` converts the full episode into a 0–100 score

---

## API Endpoints

The FastAPI server (`server/app.py`) exposes these endpoints:

| Method | Path | Description | Request Body | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/` | Root health check | — | `{"status": "ok"}` |
| `GET` | `/health` | Health check | — | `{"status": "ok"}` |
| `POST` | `/reset` | Reset environment with a task | `{"task_name": "stabilization", "seed": 42}` | `Observation` |
| `POST` | `/step` | Advance one step | `{"steam_valve": 50, "reflux_ratio": 50, "feed_rate": 50, "vent": 0}` | `{observation, reward, done, info}` |
| `GET/POST` | `/state` | Get current environment state | — | `EnvironmentState` |

---

## Tasks

The benchmark includes three tasks of increasing difficulty:

### Task 1: Stabilization (Easy)

- **Goal:** Hold the plant near 100°C with low oscillation and safe pressure
- **Max steps:** 50
- **Success:** `|temperature - 100| < 15` and `temperature_variance < 10` at episode end
- **Failure:** Temperature out of `[50, 140]` or safety shutdown

### Task 2: Optimization (Medium)

- **Goal:** Balance purity, throughput, and energy efficiency
- **Max steps:** 75
- **Success:** `purity >= 57.5`, `flow_rate >= 12`, and `temperature_variance < 30` at episode end
- **Failure:** Low purity for 30+ steps, 5+ instability steps, or safety shutdown
- **Escalation:** Pressure increases by 10% per step after 3 consecutive unstable steps

### Task 3: Emergency Control (Hard)

- **Goal:** Recover from an elevated-pressure starting state
- **Max steps:** 60
- **Success:** Maintain stability for 8+ consecutive steps
- **Failure:** Pressure ≥ 3.2, timeout without recovery, or safety shutdown
- **Escalation:** Pressure grows by 4% after 5 unstable steps, plus 3% cascading above pressure 2.6
- **Extra difficulty:** Delayed cooling response, reduced cooling effectiveness

### Task weights

| Task | Weight |
| --- | --- |
| Stabilization | `0.25` |
| Optimization | `0.35` |
| Emergency Control | `0.40` |

---

## Scoring System

Each episode produces a 0–100 score:

```
performance_score = map(clip(cumulative_reward / max_steps, -1, 1), [-1,1], [0,60])
success_bonus     = 40 if task_success else 0
efficiency_bonus  = ((max_steps - steps) / max_steps) * 10  (only on success)
failure_cap       = min(score, 30 + performance_score * 0.5) (only on failure)
final_score       = clip(total, 0, 100)
```

The overall score is a weighted average across all tasks.

---

## Configuration via openenv.yaml

All task parameters are stored in `openenv.yaml` and loaded at runtime by `app/config.py`. Any parameter present in the YAML overrides the Python-level default. Removing a key causes the code to fall back to its built-in default.

Example — change the stabilization target temperature:

```yaml
tasks:
  - name: stabilization
    parameters:
      target_temperature: 95.0    # was 100.0
      max_steps: 60               # was 50
```

No code changes required — restart the server and the new values take effect.

---

## Validation

A pre-submission validation script checks that your deployment is ready:

```bash
# Make the script executable (first time only)
chmod +x validate.sh

# Run validation against your HF Space
./validate.sh https://<your-space>.hf.space
```

The validator performs 3 checks:

1. **Ping test** — Calls `POST /reset` on your HF Space and expects HTTP 200
2. **Docker build** — Builds the Dockerfile locally to ensure it compiles
3. **OpenEnv validate** — Runs `openenv validate` to check the project structure

```
========================================
  OpenEnv Submission Validator
========================================
[18:14:17] PASSED -- HF Space is live and responds to /reset
[18:15:55] PASSED -- Docker build succeeded
[18:15:58] PASSED -- openenv validate passed

========================================
  All 3/3 checks passed!
  Your submission is ready to submit.
========================================
```

---

## Helper Scripts

### `test.py` — Task difficulty probe

Runs 200 episodes per task with seeded random actions to measure survivability:

```bash
python test.py
```

### `test_grader.py` — Grader smoke test

Compares a dumb agent, a random agent, and the heuristic agent through the grading pipeline:

```bash
python test_grader.py
```

### `app.baseline` — Heuristic baseline

Runs the hand-tuned heuristic controller with full scoring:

```bash
python -m app.baseline
```

---

## Heuristic Baseline Results

Reference scores from `python -m app.baseline`:

| Metric | Value |
| --- | --- |
| **Overall score** | `0.619` |
| Stabilization score | `0.812` |
| Stabilization success | `100.0%` |
| Optimization score | `0.715` |
| Optimization success | `77.0%` |
| Emergency score | `0.414` |
| Emergency success | `0.0%` |

This serves as a useful reference point for LLM-based agents.

---

## License

This project was built for the Meta × Scaler Hackathon.
