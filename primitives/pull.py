"""Primitive to pull the door handle and monitor joint effort."""

from typing import Any, List, Sequence, Tuple, Optional
import threading
import time


class Joint3EffortRecorder(threading.Thread):
    """Continuously sample joint3 effort in a background thread."""

    def __init__(self, arm: Any, interval: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.arm = arm
        self.interval = interval
        self.data: List[float] = []
        self._stop_event = threading.Event()

    def run(self) -> None:  # pragma: no cover - requires hardware
        while not self._stop_event.is_set():
            if hasattr(self.arm, "joint"):
                _, eff = self.arm.joint()
                if len(eff) >= 3:
                    self.data.append(eff[2])
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


def pull_handle(arm: Any, target_pose: Sequence[float]) -> None:
    """Move to ``target_pose`` using ``move_p``."""
    if arm is None:
        return
    if hasattr(arm, "move_p"):
        arm.move_p(list(target_pose))


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
    arm: Any,
    history: List[float],
    target_pose: Sequence[float],
    log_path: Optional[str] = None,
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
    recorder = Joint3EffortRecorder(arm)
    recorder.start()
    try:
        pull_handle(arm, target_pose)
    finally:  # pragma: no branch - ensure recorder stops
        recorder.stop()
        recorder.join()

    history.extend(recorder.data)
    current = history[-1] if history else 0.0
    error = evaluate_joint3_effort(current, history[:-1])

    if log_path is not None:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                for value in recorder.data:
                    f.write(f"{value}\n")
        except OSError:
            pass

    if len(history) > 200:
        del history[:-200]
    return error, current
