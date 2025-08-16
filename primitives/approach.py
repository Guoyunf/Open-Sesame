"""Primitive to approach the door handle."""

from typing import Any, Sequence


def approach_handle(arm: Any, target_pose: Sequence[float]) -> None:
    """Move the arm close to the door handle.

    Parameters
    ----------
    arm:
        Arm-like object controlling the manipulator. It is expected to expose a
        ``pose`` method returning the current pose and a ``send_delta`` method
        for incremental motion commands.
    target_pose:
        Desired pose ``[x, y, z, roll, pitch, yaw]`` in the same frame as the
        arm pose.
    """
    if arm is None:
        return

    current = arm.pose() if hasattr(arm, "pose") else [0.0] * 6
    dx = target_pose[0] - current[0]
    dy = target_pose[1] - current[1]
    dz = target_pose[2] - current[2]
    dR = target_pose[3] - current[3]
    dP = target_pose[4] - current[4]
    dY = target_pose[5] - current[5]

    if hasattr(arm, "send_delta"):
        arm.send_delta(dx=dx, dy=dy, dz=dz, dR=dR, dP=dP, dY=dY)
