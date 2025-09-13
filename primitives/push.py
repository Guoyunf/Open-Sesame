"""Primitive to push the door open."""

from typing import Any, Sequence


def push_door(arm: Any, target_pose: Sequence[float]) -> None:
    """Move to ``target_pose`` using ``move_p``.

    Parameters
    ----------
    arm:
        Arm-like object.
    target_pose:
        Absolute pose for the push motion.
    """
    if arm is None:
        return

    if hasattr(arm, "move_p"):
        arm.move_p(list(target_pose))
