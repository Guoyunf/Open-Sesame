"""Button detection utilities using third-person camera images."""

from __future__ import annotations

from typing import Tuple

from camera import Camera

from .target_detection import get_target_coords_manual, get_target_coords_model


BUTTON_DETECTION_ENV = "BUTTON_DETECTION_HOST"
DEFAULT_BUTTON_DET_HOST = "http://127.0.0.1:18000"


def get_button_coords_manual(cam: Camera) -> Tuple[float, float, float]:
    """Capture an RGB-D frame and let the user select the button centre."""

    return get_target_coords_manual(cam, target_name="button")


def get_button_coords_model(
    cam: Camera, host: str | None = None, save_dir: str = "button_images"
) -> Tuple[float, float, float]:
    """Use a remote model to predict the button position in the camera frame."""

    return get_target_coords_model(
        cam,
        target_name="button",
        host=host,
        host_env_var=BUTTON_DETECTION_ENV,
        default_host=DEFAULT_BUTTON_DET_HOST,
        save_dir=save_dir,
    )


__all__ = ["get_button_coords_manual", "get_button_coords_model"]
