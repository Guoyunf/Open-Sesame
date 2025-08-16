"""Primitive to grasp the door handle."""

from typing import Any


def grasp_handle(arm: Any, close: bool = True) -> None:
    """Control the gripper to grasp or release the handle.

    Parameters
    ----------
    arm:
        Arm-like object providing ``close_gripper`` and ``open_gripper`` methods.
    close:
        If ``True`` close the gripper; otherwise open it.
    """
    if arm is None:
        return

    if close:
        if hasattr(arm, "close_gripper"):
            arm.close_gripper()
    else:
        if hasattr(arm, "open_gripper"):
            arm.open_gripper()
