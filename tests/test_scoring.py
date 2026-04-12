from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app.grader import TASK_WEIGHTS, compute_episode_score, evaluate_all_tasks
from app.models import Action, Observation
from app.scoring import PUBLIC_SCORE_MAX, PUBLIC_SCORE_MIN, sanitize_public_score
from inference import run_episode, summarize_results


def neutral_agent(_: Observation) -> Action:
    return Action(steam_valve=50.0, reflux_ratio=50.0, feed_rate=50.0, vent=0)


class PublicScoringTests(unittest.TestCase):
    def assert_strict_open(self, value: float) -> None:
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)
        self.assertGreaterEqual(value, PUBLIC_SCORE_MIN)
        self.assertLessEqual(value, PUBLIC_SCORE_MAX)

    def test_sanitize_public_score_handles_invalid_inputs(self) -> None:
        cases = {
            0: PUBLIC_SCORE_MIN,
            1: PUBLIC_SCORE_MAX,
            -4.2: PUBLIC_SCORE_MIN,
            8.6: PUBLIC_SCORE_MAX,
            float("nan"): 0.5,
            float("inf"): 0.5,
            "abc": 0.5,
            None: 0.5,
        }

        for raw_value, expected in cases.items():
            with self.subTest(raw_value=raw_value):
                score = sanitize_public_score(raw_value)
                self.assertEqual(score, expected)
                self.assert_strict_open(score)

    def test_compute_episode_score_extremes_are_strictly_open(self) -> None:
        cases = [
            (-1e9, False, True, 1, 50),
            (1e9, True, False, 1, 50),
            (0.0, False, False, 0, 0),
            (1000.0, True, False, 50, 50),
        ]

        for case in cases:
            with self.subTest(case=case):
                score = compute_episode_score(*case)
                self.assert_strict_open(score)

    def test_evaluate_all_tasks_default_output_is_submission_safe(self) -> None:
        results = evaluate_all_tasks(neutral_agent, n_episodes=2)

        self.assertEqual(set(results["task_scores"]), set(TASK_WEIGHTS))
        self.assertNotIn("details", results)
        self.assert_strict_open(results["overall_score"])

        for score in results["task_scores"].values():
            self.assert_strict_open(score)

    def test_evaluate_all_tasks_details_use_counts_not_rates(self) -> None:
        results = evaluate_all_tasks(neutral_agent, n_episodes=2, include_details=True)

        for task_name, task_details in results["details"].items():
            with self.subTest(task_name=task_name):
                self.assertIn("success_count", task_details)
                self.assertIn("failure_count", task_details)
                self.assertIn("episode_count", task_details)
                self.assertNotIn("success_rate", task_details)
                self.assertNotIn("failure_rate", task_details)
                self.assert_strict_open(task_details["avg_score"])

    def test_summarize_results_requires_all_tasks(self) -> None:
        partial_results = [
            {"task": "stabilization", "score": 0.0},
            {"task": "optimization", "score": 1.0},
        ]

        with self.assertRaises(ValueError):
            summarize_results(partial_results, episodes_per_task=1)

    def test_summarize_results_sanitizes_boundary_scores(self) -> None:
        results = [
            {"task": "stabilization", "score": 0.0},
            {"task": "optimization", "score": 1.0},
            {"task": "emergency_control", "score": float("nan")},
        ]

        summary = summarize_results(results, episodes_per_task=1)

        self.assert_strict_open(summary["overall_score"])
        self.assertEqual(set(summary["task_scores"]), set(TASK_WEIGHTS))
        for score in summary["task_scores"].values():
            self.assert_strict_open(score)

    def test_run_episode_emits_required_log_shape(self) -> None:
        stdout = io.StringIO()
        fixed_action = Action(steam_valve=50.0, reflux_ratio=50.0, feed_rate=50.0, vent=0)
        with patch("inference.choose_action", return_value=fixed_action):
            with redirect_stdout(stdout):
                result = run_episode("stabilization", episode_index=0, seed=42)

        lines = [line for line in stdout.getvalue().splitlines() if line]
        self.assertGreater(len(lines), 2)
        self.assertTrue(lines[0].startswith("[START] task=stabilization env=openenv model="))
        self.assertTrue(all(line.startswith("[STEP] ") for line in lines[1:-1]))
        self.assertRegex(
            lines[-1],
            r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=.*$",
        )
        self.assertIn(f"score={result['score']:.2f}", lines[-1])


if __name__ == "__main__":
    unittest.main()
