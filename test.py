import numpy as np

from app.env import DistillationEnv
from app.models import Action
from app.tasks import get_all_tasks


def random_action(rng):
    return Action(
        steam_valve=float(rng.uniform(0, 100)),
        reflux_ratio=float(rng.uniform(0, 100)),
        feed_rate=float(rng.uniform(0, 100)),
        vent=int(rng.integers(0, 2)),
    )


def run_episode(task, seed):
    env = DistillationEnv(seed=seed)
    rng = np.random.default_rng(seed + 1000)

    obs = task.reset(env)
    done = False

    while not done:
        action = random_action(rng)
        obs, reward, done, info = task.step(env, action)

    return info


def evaluate_task(task, n_episodes=200):
    survivals = 0
    successes = 0
    failures = {}

    for ep in range(n_episodes):
        task_instance = type(task)()  # fresh instance every run
        info = run_episode(task_instance, seed=ep)

        if info.get("task_success"):
            successes += 1
            survivals += 1
        elif not info.get("task_failed"):
            survivals += 1
        else:
            reason = info.get("failure_reason", "unknown").split(":")[0]
            failures[reason] = failures.get(reason, 0) + 1

    survival_rate = survivals / n_episodes * 100
    success_rate = successes / n_episodes * 100

    return survival_rate, success_rate, failures


def main():
    print("=" * 70)
    print("FINAL TASK DIFFICULTY EVALUATION")
    print("=" * 70)

    tasks = get_all_tasks()

    for task in tasks:
        print(f"\n{task.name.upper()}")

        survival, success, failures = evaluate_task(task, n_episodes=200)

        print(f"Survival Rate: {survival:.1f}%")
        print(f"Success Rate:  {success:.1f}%")

        if failures:
            print("Failure Breakdown:")
            for k, v in failures.items():
                print(f"  - {k}: {v}")



if __name__ == "__main__":
    main()