import numpy as np

from app.grader import evaluate_all_tasks
from app.models import Action


# -----------------------------
# AGENTS TO TEST
# -----------------------------

def random_agent(obs):
    return Action(
        steam_valve=float(np.random.uniform(0, 100)),
        reflux_ratio=float(np.random.uniform(0, 100)),
        feed_rate=float(np.random.uniform(0, 100)),
        vent=int(np.random.randint(0, 2)),
    )


def dumb_agent(obs):
    # worst possible behavior
    return Action(
        steam_valve=100,
        reflux_ratio=0,
        feed_rate=100,
        vent=0,
    )



def heuristic_agent(obs):
    if obs.pressure > 2.3:
        vent = 1
        steam = 30   # reduce heat early
    else:
        vent = 0
        steam = 60

    return Action(
        steam_valve=steam,
        reflux_ratio=60,
        feed_rate=50,
        vent=vent,
    )

# -----------------------------
# RUN TESTS
# -----------------------------

def run_test(agent_fn, name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    results = evaluate_all_tasks(agent_fn, n_episodes=2000)

    print(f"\nOverall Score: {results['overall_score']:.3f}")

    print("\nPer Task Scores:")
    for task, score in results["task_scores"].items():
        print(f"  {task}: {score:.3f}")

    print("\nDetails:")
    for task, details in results["details"].items():
        print(f"\n{task}:")
        print(f"  Success Rate: {details['success_rate']:.2f}")
        print(f"  Failure Rate: {details['failure_rate']:.2f}")
        print(f"  Avg Steps: {details['avg_steps']:.2f}")


if __name__ == "__main__":
    run_test(dumb_agent, "DUMB AGENT (should be terrible)")
    run_test(random_agent, "RANDOM AGENT (baseline)")
    run_test(heuristic_agent, "HEURISTIC AGENT (should be best)")
