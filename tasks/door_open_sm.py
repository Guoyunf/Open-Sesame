"""Finite state machine for opening a door."""

from typing import Any, List, Sequence, Tuple, Callable, Optional

from primitives import (
    approach_handle_and_check_force,
    grasp_handle,
    retreat_gripper,
    retreat_base,
    pull_handle_and_check,
    push_door,
)


class DoorOpenStateMachine:
    """State machine coordinating primitives to open a door."""

    APPROACH = "approach"
    GRASP = "grasp"
    PULL = "pull"
    PUSH = "push"
    MOVE_BASE = "move_base"
    DONE = "done"
    ERROR = "error"

    def __init__(
        self,
        arm: Any,
        base: Any,
        max_attempts: int = 3,
        base_backoff_time: float = 2.0,
        base_backoff_velocity: float = 0.2,
        approach_force_threshold: float = 10.0,
    ):
        self.arm = arm
        self.base = base
        self.max_attempts = max_attempts
        self.base_backoff_time = base_backoff_time
        self.base_backoff_velocity = base_backoff_velocity
        self.approach_force_threshold = approach_force_threshold
        self.state = self.APPROACH
        self.joint3_history: List[float] = []

    def run(
        self,
        pose_fn: Callable[[], Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]]],
    ) -> str:
        """Run the state machine until completion or error.

        Parameters
        ----------
        pose_fn:
            Callable that returns ``(grasp_pose, pull_pose, push_pose)`` each
            time it is invoked. If detection fails it should return ``None``.

        Returns
        -------
        str
            Final state, either ``"done"`` or ``"error"``.
        """
        attempts = 0
        self.state = self.APPROACH
        self.joint3_history.clear()
        grasp_pose: Optional[Sequence[float]] = None
        pull_pose: Optional[Sequence[float]] = None
        push_pose: Optional[Sequence[float]] = None

        while attempts < self.max_attempts:
            if self.state == self.APPROACH:
                # Reset the arm before each detection attempt
                retreat_gripper(self.arm)
                poses = pose_fn()
                if poses is None:
                    return self.ERROR
                grasp_pose, pull_pose, push_pose = poses
                force_error = approach_handle_and_check_force(
                    self.arm, grasp_pose, self.approach_force_threshold
                )
                if force_error:
                    attempts += 1
                    continue
                self.state = self.GRASP

            elif self.state == self.GRASP:
                grasp_handle(self.arm)
                self.state = self.PULL

            elif self.state == self.PULL:
                log_path = f"joint3_effort_attempt{attempts}.txt"
                error, _ = pull_handle_and_check(
                    self.arm, self.joint3_history, pull_pose, log_path
                )
                if error:
                    retreat_gripper(self.arm)
                    attempts += 1
                    self.joint3_history.clear()
                    self.state = self.APPROACH
                else:
                    self.state = self.PUSH

            elif self.state == self.PUSH:
                if push_pose is not None:
                    push_door(self.arm, push_pose)
                self.state = self.MOVE_BASE

            elif self.state == self.MOVE_BASE:
                retreat_base(
                    self.base,
                    self.base_backoff_time,
                    self.base_backoff_velocity,
                )
                return self.DONE

        return self.ERROR
