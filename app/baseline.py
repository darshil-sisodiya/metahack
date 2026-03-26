"""OpenEnv - Baseline Inference Script.

Runs a deterministic heuristic agent against the task grader.
"""

from __future__ import annotations

from app.grader import evaluate_all_tasks
from app.models import Action, Observation


def baseline_agent(obs: Observation) -> Action:
    """Return a simple but task-aware deterministic heuristic action.

    The policy aims to keep the column near the purity-friendly temperature
    band while shedding pressure early enough to avoid instability.
    """
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
    """Run baseline evaluation and print a compact summary."""
    results = evaluate_all_tasks(baseline_agent, n_episodes=200)

    print("BASELINE EVALUATION")
    print(f"Overall Score: {results['overall_score']:.2f}")
    print("Per-Task Scores:")
    for task_name, score in results["task_scores"].items():
        print(f"  {task_name}: {score:.2f}")

    print("Success Rates:")
    for task_name, task_details in results["details"].items():
        success_rate = task_details["success_rate"] * 100.0
        print(f"  {task_name}: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
