"""High level task to detect a door handle and open the door."""

import math
from camera import Camera
from arm_kinova import Arm
from dog_gs import RosBase
from utils.lib_io import read_yaml_file

from .handle_detection import get_handle_coords_manual, get_handle_coords_model
from .door_open_sm import DoorOpenStateMachine


def _compute_pull_pose(grasp_pose, down_dist, toward_dist):
    """Compute pull pose based on handle orientation."""
    x, y, z, roll, pitch, yaw = grasp_pose
    new_x = x - toward_dist
    new_y = y
    new_z = z - down_dist
    return [new_x, new_y, new_z, roll, pitch, yaw]


def _compute_push_pose(pull_pose, forward_dist):
    """Compute push pose to further open the door."""
    x, y, z, roll, pitch, yaw = pull_pose
    new_x = x + forward_dist * math.cos(yaw)
    new_y = y + forward_dist * math.sin(yaw)
    return [new_x, new_y, z, roll, pitch, yaw]


def open_door(

    cam_arm: Camera = None,
    cam_base: Camera = None,
    arm: Arm = None,
    base: RosBase = None,
    cfg_path: str = "cfg/cfg_door_open.yaml",
    use_model: bool = False,
) -> str:
    """Run the full door opening procedure.

    Parameters
    ----------
    cfg_path:
        Path to the YAML configuration describing grasp orientation, pull and push
        distances, base retreat and retry settings.
        ``approach_force_threshold`` sets the maximum allowable force while
        moving the gripper to the handle.
    use_model:
        If ``True`` the vision model service is used for detection, otherwise
        manual clicking is performed.
    cam_arm, cam_base, arm, base:
        Optional pre-initialized robot components. If ``None`` they are created
        using default configuration files. The base-mounted camera is used for
        handle detection while the arm-mounted camera is initialised for
        potential future refinement.

    Returns
    -------
    str
        Final state returned by :class:`DoorOpenStateMachine`.
    """
    cfg = read_yaml_file(cfg_path)

    cam_arm_created = False
    cam_base_created = False
    arm_created = False
    base_created = False

    if cam_arm is None:
        cam_arm = Camera.init_from_yaml(cfg_path="cfg/cfg_cam_arm.yaml")
        cam_arm_created = True
    if cam_base is None:
        cam_base = Camera.init_from_yaml(cfg_path="cfg/cfg_cam.yaml")
        cam_base_created = True
    if arm is None:
        arm = Arm.init_from_yaml(cfg_path="cfg/cfg_arm_left.yaml")
        arm_created = True
    if base is None:
        base = RosBase(linear_velocity=0.2, angular_velocity=0.5)
        base_created = True

    def detect_and_plan():
        if use_model:
            x_cam, y_cam, z_cam = get_handle_coords_model(cam_base)
        else:
            x_cam, y_cam, z_cam = get_handle_coords_manual(cam_base)
        if x_cam is None:
            return None
        rx = cfg.grasp_orientation.roll
        ry = cfg.grasp_orientation.pitch
        rz = cfg.grasp_orientation.yaw
        grasp_cam = [x_cam, y_cam, z_cam, rx, ry, rz]
        grasp_base = arm.target2cam_xyzrpy_to_target2base_xyzrpy(xyzrpy_cam=grasp_cam)
        grasp_base[3:6] = [rx, ry, rz]
        pull_base = _compute_pull_pose(
            grasp_base,
            cfg.pull_down_distance,
            getattr(cfg, "pull_toward_hinge_distance", 0.05),
        )
        push_base = _compute_push_pose(
            pull_base,
            getattr(cfg, "push_forward_distance", 0.1),
        )
        return grasp_base, pull_base, push_base

    sm = DoorOpenStateMachine(
        arm,
        base,
        max_attempts=cfg.state_machine.max_attempts,
        base_backoff_time=cfg.base_backoff.time,
        base_backoff_velocity=cfg.base_backoff.linear_velocity,
        approach_force_threshold=getattr(cfg, "approach_force_threshold", 10.0),
    )
    result = sm.run(detect_and_plan)

    if cam_base_created:
        cam_base.disconnect()
    if cam_arm_created:
        cam_arm.disconnect()
    return result


__all__ = ["open_door"]
