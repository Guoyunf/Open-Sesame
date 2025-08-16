"""Finite state machine for opening a door."""

from typing import Any, List, Sequence

from primitives import (
    approach_handle,
    grasp_handle,
    retreat_gripper,
    retreat_base,
    pull_handle_and_check,
)


class DoorOpenStateMachine:
    """State machine coordinating primitives to open a door."""

    APPROACH = "approach"
    GRASP = "grasp"
    PULL = "pull"
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
        retry_backoff_distance: float = 0.2,
    ):
        self.arm = arm
        self.base = base
        self.max_attempts = max_attempts
        self.base_backoff_time = base_backoff_time
        self.base_backoff_velocity = base_backoff_velocity
        self.retry_backoff_distance = retry_backoff_distance
        self.state = self.APPROACH
        self.joint3_history: List[float] = []

    def run(self, grasp_pose: Sequence[float], pull_pose: Sequence[float]) -> str:
        """Run the state machine until completion or error.

        Parameters
        ----------
        grasp_pose:
            Pose used in the approach phase.
        pull_pose:
            Pose used for the downward pull after grasping.

        Returns
        -------
        str
            Final state, either ``"done"`` or ``"error"``.
        """
        attempts = 0
        self.state = self.APPROACH
        self.joint3_history.clear()

        while attempts < self.max_attempts:
            if self.state == self.APPROACH:
                approach_handle(self.arm, grasp_pose)
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
                    retreat_gripper(self.arm, self.retry_backoff_distance)
                    attempts += 1
                    self.joint3_history.clear()
                    self.state = self.APPROACH
                else:
                    self.state = self.MOVE_BASE

            elif self.state == self.MOVE_BASE:
                retreat_base(
                    self.base,
                    self.base_backoff_time,
                    self.base_backoff_velocity,
                )
                return self.DONE

        return self.ERROR
