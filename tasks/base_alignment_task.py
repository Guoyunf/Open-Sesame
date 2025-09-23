"""Task for keeping the robot base within a desired positional range."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Iterable, Tuple

if TYPE_CHECKING:  # pragma: no cover - for static typing only
    from arm_kinova import Arm
    from camera import Camera
    from dog_gs import RosBase


DetectionFn = Callable[["Camera"], Tuple[float | None, float | None, float | None]]


def _sorted_range(range_values: Iterable[float]) -> Tuple[float, float]:
    """Return the ``(min, max)`` pair for ``range_values``."""

    values = list(float(v) for v in range_values)
    if len(values) != 2:
        raise ValueError("Range must contain exactly two numeric bounds")
    lower, upper = sorted(values)
    return lower, upper


def maintain_base_position(
    *,
    cam_base: "Camera" | None = None,
    arm: "Arm" | None = None,
    base: "RosBase" | None = None,
    x_range: Tuple[float, float] = (-25.0, -5.0),
    y_range: Tuple[float, float] = (-110.0, -90.0),
    x_tolerance: float = 0.0,
    y_tolerance: float = 0.0,
    max_iterations: int = 10,
    detection_wait: float = 0.5,
    settle_time: float = 1.0,
    strafe_duration: float = 0.5,
    forward_duration: float = 0.5,
    strafe_velocity: float = 0.1,
    forward_velocity: float = 0.1,
    detect_fn: DetectionFn | None = None,
) -> Tuple[str, Tuple[float, float, float] | None]:
    """Align the quadruped base until the detected pose lies within bounds.

    Parameters
    ----------
    cam_base, arm, base:
        Optional pre-initialised hardware interfaces. When omitted the default
        configuration files are used to create them.
    x_range, y_range:
        Inclusive intervals that the detected base-frame ``x`` and ``y``
        coordinates must satisfy.
    x_tolerance, y_tolerance:
        Optional extra margin applied to the lower/upper bounds to avoid
        oscillations caused by perception noise.
    max_iterations:
        Maximum number of detection / adjustment cycles before aborting.
    detection_wait:
        Delay between iterations to allow the camera stream to update.
    settle_time:
        Waiting time after each movement command before taking another
        measurement.
    strafe_duration, forward_duration:
        Duration (in seconds) for lateral and forward/backward corrections.
    strafe_velocity, forward_velocity:
        Linear velocities (m/s) used for the respective movements.
    detect_fn:
        Override for the detection function. Defaults to
        :func:`target_detection.detect_top_left_black_center`.

    Returns
    -------
    tuple
        ``("success", pose)`` when the pose is within range, otherwise
        ``("failed", last_pose)``.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero")

    detect = detect_fn
    if detect is None:
        from . import target_detection as _target_detection

        detect = _target_detection.detect_top_left_black_center
    x_lower, x_upper = _sorted_range(x_range)
    y_lower, y_upper = _sorted_range(y_range)
    x_margin = abs(float(x_tolerance))
    y_margin = abs(float(y_tolerance))
    strafe_duration = abs(float(strafe_duration))
    forward_duration = abs(float(forward_duration))
    strafe_velocity = abs(float(strafe_velocity))
    forward_velocity = abs(float(forward_velocity))

    cam_created = False
    arm_created = False

    if cam_base is None:
        from camera import Camera as _Camera

        cam_base = _Camera.init_from_yaml(cfg_path="cfg/cfg_cam.yaml")
        cam_created = True
    if arm is None:
        from arm_kinova import Arm as _Arm

        arm = _Arm.init_from_yaml(cfg_path="cfg/cfg_arm_left.yaml")
        arm_created = True
    if base is None:
        from dog_gs import RosBase as _RosBase

        base = _RosBase(linear_velocity=forward_velocity or 0.2, angular_velocity=0.5)

    print("\n--- Starting Base Alignment Task ---")
    print(f"Desired x range: [{x_lower:.2f}, {x_upper:.2f}], y range: [{y_lower:.2f}, {y_upper:.2f}]")

    last_pose: Tuple[float, float, float] | None = None

    try:
        for iteration in range(1, max_iterations + 1):
            if detection_wait > 0:
                time.sleep(detection_wait)

            cam_x, cam_y, cam_z = detect(cam_base)
            if cam_x is None or cam_y is None or cam_z is None:
                print("[WARNING] Detection failed, retrying...")
                continue

            target_cam = [cam_x, cam_y, cam_z, 0.0, 0.0, 0.0]
            target_pose = list(
                arm.target2cam_xyzrpy_to_target2base_xyzrpy(xyzrpy_cam=target_cam)
            )
            pose = target_pose[0:3]
            x, y, z = (float(pose[0]), float(pose[1]), float(pose[2]))
            last_pose = (x, y, z)

            print(f"Iteration {iteration}: target pose in base frame: {pose}")

            moved = False

            if x < x_lower - x_margin:
                print(
                    f"[INFO] x={x:.2f} is less than lower bound {x_lower:.2f}. Strafing left to increase x."
                )
                base.strafe_T(strafe_duration, linear_velocity=strafe_velocity)
                moved = True
            elif x > x_upper + x_margin:
                print(
                    f"[INFO] x={x:.2f} is greater than upper bound {x_upper:.2f}. Strafing right to decrease x."
                )
                base.strafe_T(-strafe_duration, linear_velocity=strafe_velocity)
                moved = True

            if y < y_lower - y_margin:
                print(
                    f"[INFO] y={y:.2f} is less than lower bound {y_lower:.2f}. Moving forward to increase y."
                )
                base.move_T(forward_duration, linear_velocity=forward_velocity)
                moved = True
            elif y > y_upper + y_margin:
                print(
                    f"[INFO] y={y:.2f} is greater than upper bound {y_upper:.2f}. Moving backward to decrease y."
                )
                base.move_T(-forward_duration, linear_velocity=forward_velocity)
                moved = True

            if not moved:
                print(
                    f"[SUCCESS] Pose ({x:.2f}, {y:.2f}, {z:.2f}) is within desired range."
                )
                return "success", last_pose

            if settle_time > 0:
                time.sleep(settle_time)

        print("[ERROR] Failed to align base within the desired range.")
        return "failed", last_pose

    except Exception as exc:  # pragma: no cover - runtime/hardware issues
        print(f"[ERROR] Exception while maintaining base position: {exc}")
        return "failed", last_pose
    finally:
        if cam_created:
            cam_base.disconnect()
        if arm_created and hasattr(arm, "open_gripper"):
            try:
                arm.open_gripper()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


__all__ = ["maintain_base_position"]
