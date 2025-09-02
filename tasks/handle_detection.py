"""Handle detection utilities using camera images."""

import os
import time
import tempfile
from typing import Tuple

import cv2
import requests

from camera import Camera


def get_handle_coords_manual(cam: Camera) -> Tuple[float, float, float]:
    """Capture an image and let user click the door handle.

    Returns 3D coordinates of the clicked pixel in the camera frame.
    """
    print("\n--- Starting Manual Handle Detection ---")
    print("Waiting 2 seconds for the camera to stabilize...")
    time.sleep(2)

    rgb_img, depth_img = cam.capture_rgbd()
    if rgb_img is None or depth_img is None:
        print("[ERROR] Failed to capture RGBD frame.")
        return None, None, None

    coords = {"x": -1, "y": -1, "X": None, "Y": None, "Z": None, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\nPixel ({x},{y}) selected.")
            d_raw = cam.get_depth_point(x, y, depth_img)
            if d_raw == 0 or d_raw is None:
                print("Direct depth is zero, trying ROI average...")
                d_raw = cam.get_depth_roi(
                    x,
                    y,
                    depth_img,
                    radius=15,
                    depth_threshold=0.05,
                    valid_ratio_threshold=0.5,
                )
            if d_raw is None:
                print(
                    f"[WARNING] Could not determine a valid depth for pixel ({x},{y}). Please try again."
                )
                return
            X, Y, Z = cam.xy_depth_2_xyz(x, y, d_raw)
            print(
                f"-> Pixel ({x},{y}) | Depth: {d_raw:.0f}mm | 3D Coords (X,Y,Z): ({X:.4f}, {Y:.4f}, {Z:.4f}) m"
            )
            coords.update({"x": x, "y": y, "X": X, "Y": Y, "Z": Z, "done": True})
            display_img = rgb_img.copy()
            cv2.circle(display_img, (x, y), 8, (0, 0, 255), 2)
            cv2.imshow("RGB Click to Annotate Handle", display_img)

    cv2.namedWindow("RGB Click to Annotate Handle", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("RGB Click to Annotate Handle", on_mouse)
    cv2.imshow("RGB Click to Annotate Handle", rgb_img)
    print("Please LEFT-CLICK on the door handle. Press ESC to quit.")
    while not coords["done"]:
        if cv2.waitKey(20) & 0xFF == 27:
            print("Annotation cancelled by user.")
            break
    cv2.destroyAllWindows()
    return coords["X"], coords["Y"], coords["Z"]


DEFAULT_HANDLE_DET_HOST = os.environ.get("HANDLE_DETECTION_HOST", "http://127.0.0.1:18000")


def get_handle_coords_model(cam: Camera, host: str | None = None) -> Tuple[float, float, float]:
    """Detect handle with a vision model and return coordinates in camera frame.

    Parameters
    ----------
    cam:
        Camera object used to capture an RGBD frame.
    host:
        Optional base URL of the handle-detection service. If ``None`` the value
        from the ``HANDLE_DETECTION_HOST`` environment variable is used, falling
        back to ``"http://127.0.0.1:18000"``.
    """
    print("\n--- Starting Model-Based Handle Detection ---")

    rgb_img, depth_img = cam.capture_rgbd()
    if rgb_img is None or depth_img is None:
        print("[ERROR] Failed to capture RGBD frame.")
        return None, None, None

    if host is None:
        host = DEFAULT_HANDLE_DET_HOST

    # Save RGB image to a temporary file and upload it to the inference service.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, rgb_img)
        url = f"{host.rstrip('/')}/predict_upload"
        with open(tmp_path, "rb") as f:
            response = requests.post(url, files={"file": f}, timeout=120)
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
        print("[ERROR] Model failed to detect the handle.")
        return None, None, None

    d_raw = cam.get_depth_point(x, y, depth_img)
    if d_raw == 0 or d_raw is None:
        d_raw = cam.get_depth_roi(x, y, depth_img)
    if d_raw is None:
        print(f"[ERROR] Detected handle at ({x},{y}), but depth is invalid.")
        return None, None, None

    X, Y, Z = cam.xy_depth_2_xyz(x, y, d_raw)
    print(
        f"Model detected handle at pixel ({x},{y}) -> 3D Coords (X,Y,Z): ({X:.4f}, {Y:.4f}, {Z:.4f}) m"
    )
    return X, Y, Z

__all__ = ["get_handle_coords_manual", "get_handle_coords_model"]
