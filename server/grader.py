from __future__ import annotations

from typing import Any


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.01
    if score < 0.01:
        return 0.01
    if score > 0.99:
        return 0.99
    return score


def _extract_score_from_trajectory(trajectory: Any) -> float:
    if trajectory is None:
        return 0.01

    if isinstance(trajectory, (int, float)):
        return _clamp_score(trajectory)

    if isinstance(trajectory, dict):
        for key in ("score", "reward", "final_score", "final_reward"):
            if key in trajectory:
                return _clamp_score(trajectory.get(key))

        observation = trajectory.get("observation")
        if isinstance(observation, dict):
            for key in ("reward", "score"):
                if key in observation:
                    return _clamp_score(observation.get(key))

        steps = trajectory.get("steps")
        if isinstance(steps, list) and steps:
            last_step = steps[-1]
            if isinstance(last_step, dict):
                for key in ("reward", "score"):
                    if key in last_step:
                        return _clamp_score(last_step.get(key))

    if isinstance(trajectory, (list, tuple)) and trajectory:
        return _extract_score_from_trajectory(trajectory[-1])

    return 0.01


def grade_easy(trajectory: Any = None) -> float:
    return _extract_score_from_trajectory(trajectory)


def grade_medium(trajectory: Any = None) -> float:
    return _extract_score_from_trajectory(trajectory)


def grade_hard(trajectory: Any = None) -> float:
    return _extract_score_from_trajectory(trajectory)
