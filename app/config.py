"""OpenEnv - Configuration Loader

Reads task parameters from ``openenv.yaml`` and exposes them to the rest
of the application.  If the YAML file is missing, malformed, or lacks a
particular task entry, callers receive an empty dict and continue with
the Python-level defaults defined in ``app/tasks.py``.

Usage::

    from app.config import get_task_config

    params = get_task_config("stabilization")
    # params == {"max_steps": 50, "target_temperature": 100.0, ...}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# Resolve the YAML path relative to the repo root (one level up from app/).
_DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "openenv.yaml"

# Module-level cache so the file is parsed at most once per process.
_cached_config: dict[str, Any] | None = None
_cached_path: Path | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Parse ``openenv.yaml`` and return the top-level mapping.

    The result is cached after the first successful call for a given
    *path*.  Pass a different *path* to force a re-read (useful in
    tests).

    Args:
        path: Absolute or relative path to the YAML file.
              Defaults to ``<repo_root>/openenv.yaml``.

    Returns:
        Parsed YAML as a Python dict, or an empty dict on any error.
    """
    global _cached_config, _cached_path  # noqa: PLW0603

    resolved = Path(path).resolve() if path else _DEFAULT_CONFIG_PATH

    if _cached_config is not None and _cached_path == resolved:
        return _cached_config

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            log.warning("openenv.yaml did not parse to a dict — ignoring.")
            data = {}
    except FileNotFoundError:
        log.info("openenv.yaml not found at %s — using Python defaults.", resolved)
        data = {}
    except yaml.YAMLError as exc:
        log.warning("Failed to parse openenv.yaml: %s — using Python defaults.", exc)
        data = {}

    _cached_config = data
    _cached_path = resolved
    return data


def get_task_config(task_name: str, *, path: str | Path | None = None) -> dict[str, Any]:
    """Return the ``parameters`` block for a specific task.

    Args:
        task_name: One of ``"stabilization"``, ``"optimization"``, or
                   ``"emergency_control"``.
        path:      Optional override for the YAML file location.

    Returns:
        A plain dict of parameter overrides.  Returns ``{}`` if the task
        is not found or the YAML is unavailable — the caller is expected
        to merge this over its own defaults.
    """
    config = load_config(path)
    tasks = config.get("tasks", [])

    if not isinstance(tasks, list):
        log.warning("'tasks' key in openenv.yaml is not a list — ignoring.")
        return {}

    for entry in tasks:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == task_name:
            params = entry.get("parameters", {})
            return dict(params) if isinstance(params, dict) else {}

    return {}


def reset_cache() -> None:
    """Clear the module-level config cache (for testing)."""
    global _cached_config, _cached_path  # noqa: PLW0603
    _cached_config = None
    _cached_path = None
