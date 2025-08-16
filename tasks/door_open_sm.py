"""Finite state machine for opening a door."""

from typing import Any, List, Sequence

from primitives import (
    approach_handle,
    grasp_handle,
    move_base_backward,
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

    def __init__(self, arm: Any, base: Any, max_attempts: int = 3):
        self.arm = arm
        self.base = base
        self.max_attempts = max_attempts
        self.state = self.APPROACH
        self.joint3_history: List[float] = []

    def run(self, target_pose: Sequence[float]) -> str:
        """Run the state machine until completion or error.

        Parameters
        ----------
        target_pose:
            Pose used in the approach phase. The meaning of the values depends on
            the arm controller, but typically corresponds to ``[x, y, z, r, p, y]``.

        Returns
        -------
        str
            Final state, either ``"done"`` or ``"error"``.
        """
        attempts = 0
        while attempts < self.max_attempts:
            self.state = self.APPROACH
            self.joint3_history.clear()
            while self.state not in (self.DONE, self.ERROR):
                if self.state == self.APPROACH:
                    approach_handle(self.arm, target_pose)
                    self.state = self.GRASP
                elif self.state == self.GRASP:
                    grasp_handle(self.arm)
                    self.state = self.PULL
                elif self.state == self.PULL:
                    error, _ = pull_handle_and_check(self.arm, self.joint3_history)
                    if error:
                        self.state = self.ERROR
                    else:
                        self.state = self.MOVE_BASE
                elif self.state == self.MOVE_BASE:
                    move_base_backward(self.base)
                    self.state = self.DONE
            if self.state == self.DONE:
                return self.DONE
            attempts += 1
        return self.ERROR
