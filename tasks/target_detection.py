"""Generic helpers for selecting 3D targets from camera images."""

from __future__ import annotations

import os
import numpy as np
import tempfile
import time
from datetime import datetime
from typing import Tuple

import cv2
import requests

from camera import Camera


def _resolve_host(host: str | None, host_env_var: str | None, default_host: str) -> str:
    """Return the service host, honoring overrides and environment variables."""
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
    endpoint = endpoint.strip("/ ")
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
    except Exception as ex:  # pragma: no cover
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


def detect_top_left_black_center(
    cam: Camera,
    *,
    save_dir: str = "target_images",
    target_name: str = "top-left black region",
    wait_seconds: float = 0.0,
    threshold_value: int = 50,
    min_area: float = 10.0,
    depth_roi_radius: int = 15,
    depth_threshold: float = 0.05,
    valid_ratio_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Detect the top-left black region from an RGB frame, retrieve depth, and
    return 3D coordinates (X, Y, Z) in meters.

    The algorithm thresholds the grayscale image to isolate dark regions,
    performs basic morphology, finds external contours, selects the region
    whose bounding box is closest to the image origin (0,0), then looks up
    depth at the region center with ROI fallback and converts to XYZ.
    """
    print(f"\n--- Starting CV-Based {target_name.title()} Detection ---")
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f} seconds for the camera to stabilise...")
        time.sleep(wait_seconds)

    # Capture RGB-D
    rgb_img, depth_img = cam.capture_rgbd()
    if rgb_img is None or depth_img is None:
        print("[ERROR] Failed to capture RGBD frame.")
        return None, None, None

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(save_dir, f"{timestamp}_rgb.jpg")
    cv2.imwrite(raw_path, rgb_img)

    # Convert to grayscale and threshold (invert so dark -> 255)
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Morphology to denoise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If none found, progressively relax threshold
    if not contours:
        for lower in (30, 20, 10):
            _, binary = cv2.threshold(gray, lower, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                break

    if not contours:
        print("[ERROR] No dark region detected.")
        return None, None, None

    # Select the contour whose bounding box center is closest to (0,0)
    best = None
    best_dist = float("inf")
    best_center = (None, None)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        dist = (x ** 2 + y ** 2) ** 0.5  # distance of top-left corner to origin
        if dist < best_dist:
            best_dist = dist
            best = (x, y, w, h)
            best_center = (cx, cy)

    if best is None or best_center[0] is None:
        print("[ERROR] No valid dark region after filtering.")
        return None, None, None

    x_c, y_c = int(best_center[0]), int(best_center[1])

    # Visualize and save prediction overlay
    vis_img = rgb_img.copy()
    cv2.circle(vis_img, (x_c, y_c), 8, (0, 0, 255), 2)
    bx, by, bw, bh = best
    cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
    vis_path = os.path.join(save_dir, f"{timestamp}_cv_pred.jpg")
    cv2.imwrite(vis_path, vis_img)

    # Depth lookup with ROI fallback
    d_raw = cam.get_depth_point(x_c, y_c, depth_img)
    if d_raw == 0 or d_raw is None:
        d_raw = cam.get_depth_roi(
            x_c,
            y_c,
            depth_img,
            radius=depth_roi_radius,
            depth_threshold=depth_threshold,
            valid_ratio_threshold=valid_ratio_threshold,
        )
    if d_raw is None:
        print(f"[ERROR] Detected dark region at ({x_c},{y_c}), but depth is invalid.")
        return None, None, None

    # Pixel -> 3D using camera intrinsics/extrinsics
    X, Y, Z = cam.xy_depth_2_xyz(x_c, y_c, d_raw)
    print(
        "CV detected {} at pixel ({},{}) | Depth: {:.0f}mm -> 3D (X,Y,Z): ({:.4f}, {:.4f}, {:.4f}) m".format(
            target_name, x_c, y_c, d_raw, X, Y, Z
        )
    )
    print(f"Saved raw image to {raw_path} and CV prediction to {vis_path}")
    return X, Y, Z


def get_target_coords_cv():
    # Placeholder for future CV-based detectors with a uniform interface.
    pass


__all__ = [
    "get_target_coords_manual",
    "get_target_coords_model",
    "detect_top_left_black_center",
]
