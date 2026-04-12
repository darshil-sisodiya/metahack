from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.app import app


class ApiScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def assert_strict_open(self, value: float) -> None:
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_evaluate_endpoint_returns_safe_default_score(self) -> None:
        response = self.client.post("/evaluate", json={"task_name": "optimization", "trajectory": []})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assert_strict_open(payload["score"])

    def test_evaluate_endpoint_sanitizes_extreme_trajectory_rewards(self) -> None:
        trajectory = [
            {"reward": -1e9, "task_failed": True},
            {"reward": 1e9, "task_success": True},
        ]

        response = self.client.post(
            "/evaluate",
            json={"task_name": "optimization", "trajectory": trajectory},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assert_strict_open(payload["score"])

    def test_run_agent_endpoint_returns_validated_summary(self) -> None:
        fake_summary = {
            "kind": "summary",
            "mode": "llm",
            "model": "test-model",
            "episodes_per_task": 1,
            "overall_score": 0.55,
            "task_scores": {
                "stabilization": 0.51,
                "optimization": 0.52,
                "emergency_control": 0.53,
            },
        }

        with patch("inference.run_all_tasks", return_value=fake_summary):
            response = self.client.post("/run-agent", json={"episodes_per_task": 1, "base_seed": 42})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assert_strict_open(payload["overall_score"])
        for score in payload["task_scores"].values():
            self.assert_strict_open(score)


if __name__ == "__main__":
    unittest.main()
