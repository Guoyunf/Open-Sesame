"""Primitive to pull the door handle and monitor joint effort."""

from typing import Any, List, Sequence, Tuple


def pull_handle(arm: Any, target_pose: Sequence[float]) -> float:
    """Move to ``target_pose`` and return joint3 effort.

    This mirrors ``main_open_door.py`` which commands an absolute pose using
    ``move_p`` rather than incremental deltas.

    Parameters
    ----------
    arm:
        Arm-like object with ``move_p`` and ``joint`` methods.
    target_pose:
        Absolute pose ``[x, y, z, r, p, y]`` to reach during the pull action.

    Returns
    -------
    float
        Effort of joint3 after motion. ``0.0`` if unavailable.
    """
    if arm is None:
        return 0.0

    if hasattr(arm, "move_p"):
        arm.move_p(list(target_pose))

    effort = 0.0
    if hasattr(arm, "joint"):
        _, eff = arm.joint()
        if len(eff) >= 3:
            effort = eff[2]
    return effort


def evaluate_joint3_effort(current: float, history: List[float]) -> bool:
    """Evaluate whether the door handle has been released.

    The rule is:
    * ``current > 0``  → error, the joint is being pulled away from the handle;
    * ``current < 0``  → normal operation;
    * ``current == 0`` → look back to the last non-zero effort.

    Parameters
    ----------
    current:
        Current effort measurement of joint3.
    history:
        List of previous effort measurements, most recent last.

    Returns
    -------
    bool
        ``True`` if an error is detected, ``False`` otherwise.
    """
    if current > 0:
        return True
    if current < 0:
        return False
    for value in reversed(history):
        if value != 0:
            return value > 0
    return False


def pull_handle_and_check(
    arm: Any, history: List[float], target_pose: Sequence[float]
) -> Tuple[bool, float]:
    """Perform the pull action and check for errors.

    Parameters
    ----------
    arm:
        Arm-like object.
    history:
        Mutable list storing past joint3 efforts.
    target_pose:
        Absolute pose for the pull motion.

    Returns
    -------
    Tuple[bool, float]
        ``(error_detected, current_effort)``.
    """
    effort = pull_handle(arm, target_pose)
    error = evaluate_joint3_effort(effort, history)
    history.append(effort)
    if len(history) > 50:
        history.pop(0)
    return error, effort
