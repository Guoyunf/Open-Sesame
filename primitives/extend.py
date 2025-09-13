"""Primitive to extend the arm forward-left after releasing the door handle."""

from typing import Any, Sequence


def extend_forward_left(
    arm: Any, current_pose: Sequence[float], forward: float = 0.1, left: float = 0.1
) -> None:
    """Extend the arm from ``current_pose`` forward and left.

    Parameters
    ----------
    arm:
        Arm-like object providing ``move_p``.
    current_pose:
        Reference pose [x, y, z, roll, pitch, yaw].
    forward:
        Distance to move along the positive x-axis.
    left:
        Distance to move along the positive y-axis.
    """
    if arm is None or current_pose is None:
        return

    target = list(current_pose)
    target[0] += forward
    target[1] += left

    if hasattr(arm, "move_p"):
        arm.move_p(target)
