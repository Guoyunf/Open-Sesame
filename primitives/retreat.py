"""Primitives for retreating either the gripper or the base."""

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


def retreat_base(base: Any, time_s: float = 2.0, linear_velocity: float = 0.2) -> None:
    """Move the mobile base backward for ``time_s`` seconds.

    Parameters
    ----------
    base:
        Base-like object providing a ``move_T`` method.
    time_s:
        Duration of the backward motion in seconds.
    linear_velocity:
        Linear velocity used for the motion in meters per second.
    """
    if base is None:
        return

    if hasattr(base, "move_T"):
        base.move_T(-abs(time_s), linear_velocity=abs(linear_velocity))
