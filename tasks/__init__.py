"""Public entry points for task-level helpers."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DoorOpenStateMachine",
    "open_door",
    "get_handle_coords_manual",
    "get_handle_coords_model",
    "get_button_coords_manual",
    "get_button_coords_model",
    "detect_top_left_black_center",
    "base_alignment_task",
    "maintain_base_position",
    "press_button",
    "auto_playback",
    "button_press_task",
]

_ATTR_TO_MODULE = {
    "DoorOpenStateMachine": "tasks.door_open_sm",
    "open_door": "tasks.door_open_task",
    "get_handle_coords_manual": "tasks.handle_detection",
    "get_handle_coords_model": "tasks.handle_detection",
    "get_button_coords_manual": "tasks.button_detection",
    "get_button_coords_model": "tasks.button_detection",
    "detect_top_left_black_center": "tasks.target_detection",
    "base_alignment_task": "tasks.base_alignment_task",
    "maintain_base_position": "tasks.base_alignment_task",
    "press_button": "tasks.button_press_task",
    "auto_playback": "tasks.auto_playback",
    "button_press_task": "tasks.button_press_task",
}


def __getattr__(name: str) -> Any:
    if name not in _ATTR_TO_MODULE:
        raise AttributeError(f"module 'tasks' has no attribute '{name}'")
    module = importlib.import_module(_ATTR_TO_MODULE[name])
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
