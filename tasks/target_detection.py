"""Generic helpers for selecting 3D targets from camera images."""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from typing import Tuple

import cv2
import requests

from camera import Camera


def _resolve_host(host: str | None, host_env_var: str | None, default_host: str) -> str:
    """Return the service host, honouring overrides and environment variables."""

    if host:
        return host
    if host_env_var:
        env_host = os.environ.get(host_env_var)
        if env_host:
            return env_host
    return default_host


def get_target_coords_manual(
    cam: Camera,
    target_name: str,
    *,
    wait_seconds: float = 2.0,
    window_name: str | None = None,
    depth_roi_radius: int = 15,
    depth_threshold: float = 0.05,
    valid_ratio_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Capture an image and let the user click on a target of interest."""

    print(f"\n--- Starting Manual {target_name.title()} Detection ---")
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f} seconds for the camera to stabilise...")
        time.sleep(wait_seconds)

    rgb_img, depth_img = cam.capture_rgbd()
    if rgb_img is None or depth_img is None:
        print("[ERROR] Failed to capture RGBD frame.")
        return None, None, None

    coords = {"x": -1, "y": -1, "X": None, "Y": None, "Z": None, "done": False}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f"\nPixel ({x},{y}) selected.")
        d_raw = cam.get_depth_point(x, y, depth_img)
        if d_raw == 0 or d_raw is None:
            print("Direct depth is zero, trying ROI average...")
            d_raw = cam.get_depth_roi(
                x,
                y,
                depth_img,
                radius=depth_roi_radius,
                depth_threshold=depth_threshold,
                valid_ratio_threshold=valid_ratio_threshold,
            )
        if d_raw is None:
            print(
                f"[WARNING] Could not determine a valid depth for pixel ({x},{y}). Please try again."
            )
            return

        X, Y, Z = cam.xy_depth_2_xyz(x, y, d_raw)
        print(
            "-> Pixel ({},{}) | Depth: {:.0f}mm | 3D Coords (X,Y,Z): ({:.4f}, {:.4f}, {:.4f}) m".format(
                x, y, d_raw, X, Y, Z
            )
        )
        coords.update({"x": x, "y": y, "X": X, "Y": Y, "Z": Z, "done": True})
        display_img = rgb_img.copy()
        cv2.circle(display_img, (x, y), 8, (0, 0, 255), 2)
        cv2.imshow(window_title, display_img)

    window_title = window_name or f"RGB Click to Annotate {target_name.title()}"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse)
    cv2.imshow(window_title, rgb_img)
    print(f"Please LEFT-CLICK on the {target_name}. Press ESC to quit.")
    while not coords["done"]:
        if cv2.waitKey(20) & 0xFF == 27:
            print("Annotation cancelled by user.")
            break
    cv2.destroyAllWindows()
    return coords["X"], coords["Y"], coords["Z"]


def get_target_coords_model(
    cam: Camera,
    target_name: str,
    *,
    host: str | None = None,
    host_env_var: str | None = None,
    default_host: str = "http://127.0.0.1:18000",
    save_dir: str = "target_images",
    endpoint: str = "predict_upload",
    request_timeout: int = 120,
    depth_roi_radius: int = 15,
    depth_threshold: float = 0.05,
    valid_ratio_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Detect a target with a remote vision model and return 3D coordinates."""

    print(f"\n--- Starting Model-Based {target_name.title()} Detection ---")

    rgb_img, depth_img = cam.capture_rgbd()
    if rgb_img is None or depth_img is None:
        print("[ERROR] Failed to capture RGBD frame.")
        return None, None, None

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(save_dir, f"{timestamp}_rgb.jpg")
    cv2.imwrite(raw_path, rgb_img)

    resolved_host = _resolve_host(host, host_env_var, default_host)
    endpoint = endpoint.lstrip("/")
    url = f"{resolved_host.rstrip('/')}/{endpoint}"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, rgb_img)
        with open(tmp_path, "rb") as f:
            response = requests.post(url, files={"file": f}, timeout=request_timeout)
        response.raise_for_status()
        result = response.json()
        x, y = result.get("x"), result.get("y")
    except Exception as ex:  # pragma: no cover - network or server failure
        print(f"[ERROR] Model inference failed: {ex}")
        return None, None, None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if x is None or y is None:
        print("[ERROR] Model failed to detect the target.")
        return None, None, None

    vis_img = rgb_img.copy()
    cv2.circle(vis_img, (int(x), int(y)), 8, (0, 0, 255), 2)
    vis_path = os.path.join(save_dir, f"{timestamp}_pred.jpg")
    cv2.imwrite(vis_path, vis_img)

    d_raw = cam.get_depth_point(x, y, depth_img)
    if d_raw == 0 or d_raw is None:
        d_raw = cam.get_depth_roi(
            x,
            y,
            depth_img,
            radius=depth_roi_radius,
            depth_threshold=depth_threshold,
            valid_ratio_threshold=valid_ratio_threshold,
        )
    if d_raw is None:
        print(f"[ERROR] Detected {target_name} at ({x},{y}), but depth is invalid.")
        return None, None, None

    X, Y, Z = cam.xy_depth_2_xyz(x, y, d_raw)
    print(
        "Model detected {} at pixel ({},{}) -> 3D Coords (X,Y,Z): ({:.4f}, {:.4f}, {:.4f}) m".format(
            target_name, x, y, X, Y, Z
        )
    )
    print(f"Saved raw image to {raw_path} and prediction to {vis_path}")
    return X, Y, Z


__all__ = ["get_target_coords_manual", "get_target_coords_model"]
