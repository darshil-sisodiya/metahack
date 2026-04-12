"""Submission-safe scoring utilities.

Centralizes sanitization and validation for any score or reward value that may
be visible to external validators.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

PUBLIC_SCORE_MIN = 0.01
PUBLIC_SCORE_MAX = 0.99
PUBLIC_SCORE_FALLBACK = 0.5
PUBLIC_SCORE_DIGITS = 6


def sanitize_public_score(
    value: Any,
    *,
    digits: int = PUBLIC_SCORE_DIGITS,
    fallback: float = PUBLIC_SCORE_FALLBACK,
) -> float:
    """Return a rounded score guaranteed to stay strictly inside ``(0, 1)``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = fallback

    if not math.isfinite(numeric):
        numeric = fallback

    numeric = max(PUBLIC_SCORE_MIN, min(PUBLIC_SCORE_MAX, numeric))
    numeric = round(numeric, digits)
    numeric = max(PUBLIC_SCORE_MIN, min(PUBLIC_SCORE_MAX, numeric))

    if not (0.0 < numeric < 1.0):
        raise ValueError(f"Sanitized public score is not strictly open: {numeric}")

    return numeric


def sanitize_public_score_map(scores: Mapping[str, Any]) -> dict[str, float]:
    """Sanitize a mapping of task scores."""
    return {task_name: sanitize_public_score(score) for task_name, score in scores.items()}


def validate_strict_open_scores(
    task_scores: Mapping[str, Any],
    overall_score: Any,
    *,
    expected_tasks: Iterable[str],
) -> None:
    """Raise when a public score payload violates strict-open constraints."""
    expected_task_names = list(expected_tasks)
    missing_tasks = [task_name for task_name in expected_task_names if task_name not in task_scores]
    if missing_tasks:
        raise ValueError(f"Missing task scores for: {', '.join(sorted(missing_tasks))}")

    for task_name in expected_task_names:
        score = task_scores[task_name]
        numeric = float(score)
        if not math.isfinite(numeric) or not (0.0 < numeric < 1.0):
            raise ValueError(
                f"Task score for '{task_name}' must be strictly between 0 and 1. Got: {score}"
            )

    numeric_overall = float(overall_score)
    if not math.isfinite(numeric_overall) or not (0.0 < numeric_overall < 1.0):
        raise ValueError(
            f"Overall score must be strictly between 0 and 1. Got: {overall_score}"
        )
