"""OpenEnv - Baseline Inference Script.

Runs a deterministic heuristic agent against the task grader.
"""

from __future__ import annotations
import json

from app.grader import evaluate_all_tasks
from app.models import Action, Observation


def baseline_agent(obs: Observation) -> Action:
    """Return a simple but task-aware deterministic heuristic action."""
    target_temp = 92.0
    if obs.pressure > 2.0:
        target_temp = 82.0
    elif obs.pressure > 1.7:
        target_temp = 88.0
    elif obs.purity < 55.0:
        target_temp = 90.0

    # Use a bounded proportional controller so steam adjustments stay smooth.
    steam = max(30.0, min(68.0, 50.0 + 1.2 * (target_temp - obs.temperature)))

    # Vent aggressively only when pressure is clearly trending unsafe.
    vent = 1 if obs.pressure > 2.1 or (obs.pressure > 1.75 and obs.temperature > 98.0) else 0

    # High reflux helps recover purity, then the controller gives some ground
    # back to throughput once the column is in an acceptable range.
    if obs.purity < 52.0:
        reflux = 48.0
    elif obs.purity < 57.0:
        reflux = 34.0
    elif obs.purity < 62.0:
        reflux = 24.0
    else:
        reflux = 48.0

    # Push throughput when the column can handle it, but still back off
    # under pressure to avoid runaway behavior.
    if obs.pressure > 2.0:
        feed = 35.0
    elif obs.flow_rate < 12.0:
        feed = 72.0
    elif obs.flow_rate > 18.0:
        feed = 55.0
    else:
        feed = 60.0

    return Action(
        steam_valve=steam,
        reflux_ratio=reflux,
        feed_rate=feed,
        vent=vent,
    )


def main() -> None:
    """Run baseline evaluation and print the STRICT JSON summary."""
    # REDUCED from 200 to 5 to prevent validator timeouts
    episodes = 5
    results = evaluate_all_tasks(baseline_agent, n_episodes=episodes)

    # Output EXACTLY the JSON format the platform parser expects
    summary = {
        "kind": "summary",
        "mode": "baseline",
        "model": "heuristic",
        "episodes_per_task": episodes,
        "overall_score": results["overall_score"],
        "task_scores": results["task_scores"]
    }
    
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()