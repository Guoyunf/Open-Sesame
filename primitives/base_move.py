"""Primitive to move the mobile base."""

from typing import Any


def move_base_backward(base: Any, duration: float = 2.0, linear_velocity: float = 0.2) -> None:
    """Move the base backward using the existing ``move_T`` API."""
    if base is None:
        return

    if hasattr(base, "move_T"):
        base.move_T(-abs(duration), linear_velocity=linear_velocity)
