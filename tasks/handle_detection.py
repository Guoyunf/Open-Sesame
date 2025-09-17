"""Handle detection utilities using camera images."""

from __future__ import annotations

from typing import Tuple

from camera import Camera

from .target_detection import get_target_coords_manual, get_target_coords_model


HANDLE_DETECTION_ENV = "HANDLE_DETECTION_HOST"
DEFAULT_HANDLE_DET_HOST = "http://127.0.0.1:18000"


def get_handle_coords_manual(cam: Camera) -> Tuple[float, float, float]:
    """Capture an RGB-D frame and let the user click the door handle."""

    return get_target_coords_manual(cam, target_name="door handle")


def get_handle_coords_model(
    cam: Camera, host: str | None = None, save_dir: str = "handle_images"
) -> Tuple[float, float, float]:
    """Detect the door handle with a remote model and return camera-frame coordinates."""

    return get_target_coords_model(
        cam,
        target_name="door handle",
        host=host,
        host_env_var=HANDLE_DETECTION_ENV,
        default_host=DEFAULT_HANDLE_DET_HOST,
        save_dir=save_dir,
    )


__all__ = ["get_handle_coords_manual", "get_handle_coords_model"]
