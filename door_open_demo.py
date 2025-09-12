#!/usr/bin/env python3
"""Runnable demo for opening a door using the task module.

This script initializes the robot components, provides placeholders for
navigation and post-opening actions, and invokes the high level door opening
routine.
"""

from camera import Camera
from arm_kinova import Arm
from dog_gs import RosBase
from tasks import open_door


def navigate_to_door(base: RosBase) -> None:
    """Navigate the base to the door. Placeholder for user implementation."""
    # TODO: Implement base navigation logic.
    pass


def after_open(base: RosBase, left_time: float = 1.0, forward_time: float = 1.0) -> None:
    """Strafe left then drive forward after the door is open."""
    base.strafe_T(T=left_time)
    base.move_T(T=forward_time)


def main() -> None:
    cam_base = Camera.init_from_yaml(cfg_path="cfg/cfg_cam.yaml")
    cam_arm = Camera.init_from_yaml(cfg_path="cfg/cfg_cam_arm.yaml")
    arm = Arm.init_from_yaml(cfg_path="cfg/cfg_arm_left.yaml")
    base = RosBase(linear_velocity=0.2, angular_velocity=0.5)

    navigate_to_door(base)
    result = open_door(cam_arm=cam_arm, cam_base=cam_base, arm=arm, base=base)
    print(f"Door opening finished with result: {result}")
    after_open(base)
    cam_base.disconnect()
    cam_arm.disconnect()


if __name__ == "__main__":
    main()
