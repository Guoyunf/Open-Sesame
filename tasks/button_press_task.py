"""High level task for detecting and pressing a button."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Sequence, Tuple

from utils.lib_io import read_yaml_file

from .button_detection import get_button_coords_manual, get_button_coords_model

if TYPE_CHECKING:  # pragma: no cover - imported only for type hints
    from arm_kinova import Arm
    from camera import Camera


def _generate_press_sequence(
    target_pose: Sequence[float],
    approach_offset: float,
    press_distance: float,
    retreat_offset: float,
) -> List[Tuple[str, List[float]]]:
    """Create a sequence of labelled poses to approach, press and retreat."""

    base_pose = list(target_pose)
    base_pose[2] -= 0.035  # Slightly above the button
    base_pose[1] += 0.02  # Slightly to the side of the button
    base_pose[0] -= 0.025  # Slightly closer to the button
    sequence: List[Tuple[str, List[float]]] = []

    if approach_offset:
        approach_pose = base_pose.copy()
        approach_pose[1] += approach_offset
        sequence.append(("approach", approach_pose))

    sequence.append(("target", base_pose.copy()))

    if press_distance > 0:
        press_pose = base_pose.copy()
        press_pose[1] -= press_distance
        sequence.append(("press", press_pose))
        sequence.append(("release", base_pose.copy()))

    if retreat_offset:
        retreat_pose = base_pose.copy()
        retreat_pose[1] += retreat_offset
        sequence.append(("retreat", retreat_pose))

    return sequence


def _as_float(value: object, default: float = 0.0) -> float:
    """Safely convert ``value`` to ``float`` with a fallback."""

    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive programming
        return float(default)


def press_button(
    cfg_path: str = "cfg/cfg_button_press.yaml",
    use_model: bool = False,
    cam: "Camera" | None = None,
    arm: "Arm" | None = None,
) -> str:
    """Detect a button, move the arm to it and press along ``-Y``."""

    cfg = read_yaml_file(cfg_path)

    cam_created = False
    arm_created = False

    if cam is None:
        from camera import Camera as _Camera

        cam = _Camera.init_from_yaml(cfg_path="cfg/cfg_cam.yaml")
        cam_created = True
    if arm is None:
        from arm_kinova import Arm as _Arm

        arm = _Arm.init_from_yaml(cfg_path="cfg/cfg_arm_left.yaml")
        arm_created = True

    try:
        if use_model:
            x_cam, y_cam, z_cam = get_button_coords_model(cam)
        else:
            x_cam, y_cam, z_cam = get_button_coords_manual(cam)

        if x_cam is None or y_cam is None or z_cam is None:
            print("[ERROR] Button detection failed.")
            return "error"

        orientation = getattr(cfg, "grasp_orientation", None)
        rx = _as_float(getattr(orientation, "roll", 0.0))
        ry = _as_float(getattr(orientation, "pitch", 0.0))
        rz = _as_float(getattr(orientation, "yaw", 0.0))

        target_cam = [x_cam, y_cam, z_cam, rx, ry, rz]
        target_pose = list(
            arm.target2cam_xyzrpy_to_target2base_xyzrpy(xyzrpy_cam=target_cam)
        )
        target_pose[3:6] = [rx, ry, rz]

        approach_offset = _as_float(getattr(cfg, "approach_offset", 0.0))
        press_distance = abs(_as_float(getattr(cfg, "press_distance", 0.02)))
        retreat_cfg = getattr(cfg, "retreat_offset", None)
        retreat_offset = (
            _as_float(retreat_cfg)
            if retreat_cfg is not None
            else approach_offset
        )
        press_duration = _as_float(getattr(cfg, "press_duration", 0.5))

        if hasattr(arm, "close_gripper"):
            arm.close_gripper()

        sequence = _generate_press_sequence(
            target_pose, approach_offset, press_distance, retreat_offset
        )

        time.sleep(2.0)  # Wait a moment before starting
        for label, pose in sequence:
            print(f"Executing {label} pose: {pose}")
            arm.move_p(pose)
            if label == "press" and press_duration > 0:
                time.sleep(press_duration)

        return "success"

    except Exception as exc:  # pragma: no cover - hardware/runtime issues
        print(f"[ERROR] Failed to press the button: {exc}")
        return "error"

    finally:
        if cam_created:
            cam.disconnect()
        if arm_created and hasattr(arm, "open_gripper"):
            try:
                arm.open_gripper()
            except Exception:
                pass


__all__ = ["press_button"]
