"""Elevator button detection utilities using third-person camera images."""

from __future__ import annotations

from typing import Tuple

from camera import Camera

from .target_detection import get_target_coords_manual, get_target_coords_model


ELEVATOR_BUTTON_DETECTION_ENV = "ELEVATOR_BUTTON_DETECTION_HOST"
DEFAULT_ELEVATOR_BUTTON_DET_HOST = "http://127.0.0.1:18000"

DEFAULT_ELEVATOR_BUTTON_SYS_PROMPT = (
    "You are a strict JSON generator.\n"
    "TASK: Return ONLY a valid JSON object with the pixel coordinates of the requested elevator button center.\n"
    'SCHEMA: {"x": <int>, "y": <int>}  (both must be integers)\n'
    "RULES:\n"
    " - Keys MUST be double-quoted.\n"
    " - Output MUST be a single JSON object on one line.\n"
    " - No extra text, no code fences, no lists, no explanations.\n"
    " - If uncertain, still return your best single-point integer guess.\n"
)

DEFAULT_ELEVATOR_BUTTON_USER_PROMPT_TEMPLATE = (
    "From the given image of an elevator control panel, locate the button labelled \"{label}\".\n"
    'Return ONLY one JSON object exactly like: {{"x": 123, "y": 456}}\n'
    "The label can be a digit, an arrow (up/down), an alarm bell, or door control symbol.\n"
    "Do NOT return arrays, ranges, confidence scores, or additional fields.\n"
)


def get_elevator_button_coords_manual(
    cam: Camera,
    button_label: str | None = None,
) -> Tuple[float, float, float]:
    """Capture an RGB-D frame and manually select an elevator button."""

    suffix = f" '{button_label}'" if button_label else ""
    return get_target_coords_manual(cam, target_name=f"elevator button{suffix}")


def get_elevator_button_coords_model(
    cam: Camera,
    button_label: str,
    *,
    host: str | None = None,
    save_dir: str = "elevator_button_images",
    sys_prompt: str | None = None,
    user_prompt_template: str | None = None,
    max_new_tokens: int | None = None,
) -> Tuple[float, float, float]:
    """Use a remote model to predict an elevator button position."""

    label_text = button_label or "the requested elevator button"
    target_name = "elevator button" if not button_label else f"elevator button '{button_label}'"
    prompt_template = user_prompt_template or DEFAULT_ELEVATOR_BUTTON_USER_PROMPT_TEMPLATE
    user_prompt = prompt_template.format(label=label_text)

    return get_target_coords_model(
        cam,
        target_name=target_name,
        host=host,
        host_env_var=ELEVATOR_BUTTON_DETECTION_ENV,
        default_host=DEFAULT_ELEVATOR_BUTTON_DET_HOST,
        save_dir=save_dir,
        sys_prompt=sys_prompt or DEFAULT_ELEVATOR_BUTTON_SYS_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
    )


__all__ = [
    "get_elevator_button_coords_manual",
    "get_elevator_button_coords_model",
    "DEFAULT_ELEVATOR_BUTTON_SYS_PROMPT",
    "DEFAULT_ELEVATOR_BUTTON_USER_PROMPT_TEMPLATE",
]
