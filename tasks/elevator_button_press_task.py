"""High level task for detecting and pressing an elevator button."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .button_press_task import _press_button_impl
from .elevator_button_detection import (
    get_elevator_button_coords_manual,
    get_elevator_button_coords_model,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for type hints
    from arm_kinova import Arm
    from camera import Camera


def press_elevator_button(
    button_label: str,
    *,
    cfg_path: str = "cfg/cfg_button_press.yaml",
    use_model: bool = False,
    cam: "Camera" | None = None,
    arm: "Arm" | None = None,
    host: str | None = None,
    save_dir: str = "elevator_button_images",
    sys_prompt: str | None = None,
    user_prompt_template: str | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """Press the specified elevator button using the arm."""

    def _manual_detector(camera: "Camera") -> Tuple[float, float, float]:
        return get_elevator_button_coords_manual(camera, button_label=button_label)

    def _model_detector(camera: "Camera") -> Tuple[float, float, float]:
        return get_elevator_button_coords_model(
            camera,
            button_label=button_label,
            host=host,
            save_dir=save_dir,
            sys_prompt=sys_prompt,
            user_prompt_template=user_prompt_template,
            max_new_tokens=max_new_tokens,
        )

    detector = _model_detector if use_model else _manual_detector
    return _press_button_impl(cfg_path, detector, cam=cam, arm=arm)


__all__ = ["press_elevator_button"]
