"""Primitives for retreating the robot."""

from typing import Any, Sequence


# Default "home" pose for resetting the arm.
HOME_POSITION = [
    0.21248655021190643,
    -0.2564840614795685,
    0.5075023174285889,
    1.512,
    -0.184,
    -1.422,
]


def retreat_gripper(arm: Any, home_position: Sequence[float] = HOME_POSITION) -> None:
    """Return the arm to the ``home_position``.

    Parameters
    ----------
    arm:
        Arm-like object providing a ``move_p`` method.
    home_position:
        Target pose representing the arm's "home" configuration.
    """
    if arm is None:
        return

    if hasattr(arm, "move_p"):
        arm.move_p(list(home_position))


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
