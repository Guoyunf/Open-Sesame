"""Primitive to approach the door handle using existing arm methods."""

from typing import Any, Sequence


def approach_handle(arm: Any, target_pose: Sequence[float]) -> None:
    """Open the gripper and move to the desired pose.

    This follows the approach used in ``main_open_door.py`` by calling the
    arm's high-level ``move_p`` method instead of incremental deltas.

    Parameters
    ----------
    arm:
        Arm-like object controlling the manipulator. It is expected to expose a
        ``control_gripper`` method to open the gripper and a ``move_p`` method
        to move to an absolute pose.
    target_pose:
        Desired pose ``[x, y, z, roll, pitch, yaw]`` in the same frame as the
        arm pose.
    """
    if arm is None:
        return

    if hasattr(arm, "control_gripper"):
        # Ensure gripper command completes before proceeding
        arm.control_gripper(open_value=0, wait=True)

    if hasattr(arm, "move_p"):
        arm.move_p(list(target_pose))
