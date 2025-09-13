"""Primitives for approaching the handle with optional force monitoring."""

from __future__ import annotations

import math
import threading
import time
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


class ForceRecorder(threading.Thread):
    """Continuously sample the external force magnitude."""

    def __init__(self, arm: Any, interval: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.arm = arm
        self.interval = interval
        self.data: list[float] = []
        self._stop_event = threading.Event()

    def run(self) -> None:  # pragma: no cover - requires hardware
        while not self._stop_event.is_set():
            if hasattr(self.arm, "joint"):
                try:
                    _, eff = self.arm.joint()
                    if eff:
                        mag = math.sqrt(sum(v * v for v in eff))
                        self.data.append(mag)
                except Exception:
                    pass
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


def approach_handle_and_check_force(
    arm: Any, target_pose: Sequence[float], threshold: float
) -> bool:
    """Approach the handle and flag if external force exceeds ``threshold``.

    The original movement logic is preserved while a background thread monitors
    the external force. If any sampled magnitude is above ``threshold`` the
    function returns ``True`` to indicate an error.
    """
    recorder = ForceRecorder(arm)
    recorder.start()
    try:
        approach_handle(arm, target_pose)
    finally:  # pragma: no branch - ensure recorder stops
        recorder.stop()
        recorder.join()

    return any(value > threshold for value in recorder.data)
