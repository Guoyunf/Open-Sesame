"""Primitive to retreat the gripper along +Y."""

from typing import Any


def retreat_gripper(arm: Any, distance: float = 0.2) -> None:
    """Move the gripper ``distance`` meters in the +Y direction.

    Parameters
    ----------
    arm:
        Arm-like object providing ``pose`` and ``move_p`` methods.
    distance:
        Positive offset along the Y axis in meters.
    """
    if arm is None:
        return

    if hasattr(arm, "pose") and hasattr(arm, "move_p"):
        current = list(arm.pose())
        current[1] += abs(distance)
        arm.move_p(current)
