"""High level task to detect a door handle and open the door."""

from typing import Optional

from camera import Camera
from arm_kinova import Arm
from dog_gs import RosBase
from utils.lib_io import read_yaml_file

from .handle_detection import get_handle_coords_manual, get_handle_coords_model
from .door_open_sm import DoorOpenStateMachine


def open_door(
    cfg_path: str = "cfg/cfg_door_open.yaml",
    use_model: bool = False,
    model: Optional[str] = None,
) -> str:
    """Run the full door opening procedure.

    Parameters
    ----------
    cfg_path:
        Path to the YAML configuration describing grasp orientation, pull distance,
        base motion and retry settings.
    use_model:
        If ``True`` the vision model path/name in ``model`` is used for detection,
        otherwise manual clicking is performed.
    model:
        Name or path of the detection model.

    Returns
    -------
    str
        Final state returned by :class:`DoorOpenStateMachine`.
    """
    cfg = read_yaml_file(cfg_path)

    cam = Camera.init_from_yaml(cfg_path="cfg/cfg_cam.yaml")
    arm = Arm.init_from_yaml(cfg_path="cfg/cfg_arm_left.yaml")
    base = RosBase(linear_velocity=cfg.base_move.linear_velocity, angular_velocity=0.5)

    if use_model:
        x_cam, y_cam, z_cam = get_handle_coords_model(model, cam)
    else:
        x_cam, y_cam, z_cam = get_handle_coords_manual(cam)

    if x_cam is None:
        cam.disconnect()
        return "error"

    rx = cfg.grasp_orientation.roll
    ry = cfg.grasp_orientation.pitch
    rz = cfg.grasp_orientation.yaw

    grasp_cam = [x_cam, y_cam, z_cam, rx, ry, rz]
    grasp_base = arm.target2cam_xyzrpy_to_target2base_xyzrpy(xyzrpy_cam=grasp_cam)
    grasp_base[3:6] = [rx, ry, rz]
    pull_base = list(grasp_base)
    pull_base[2] -= cfg.pull_down_distance

    sm = DoorOpenStateMachine(
        arm,
        base,
        max_attempts=cfg.state_machine.max_attempts,
        base_move_duration=cfg.base_move.duration,
        base_move_velocity=cfg.base_move.linear_velocity,
        retry_backoff_distance=cfg.retry_backoff.distance,
        retry_backoff_velocity=cfg.retry_backoff.linear_velocity,
    )
    result = sm.run(grasp_base, pull_base)

    cam.disconnect()
    return result


__all__ = ["open_door"]
