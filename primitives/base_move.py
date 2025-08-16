"""Primitive to move the mobile base."""

from typing import Any


def move_base_backward(base: Any, distance: float = 0.2) -> None:
    """Move the base backward to help open the door."""
    if base is None:
        return

    if hasattr(base, "move_by"):
        base.move_by(-distance, 0.0, 0.0)
    elif hasattr(base, "send_velocity"):
        base.send_velocity(-abs(distance), 0.0, 0.0)
