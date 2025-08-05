#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_open_door.py
-----------------
This script orchestrates a robotic system (arm and base) to open a door.

The process is as follows:
1.  Initialize the camera, robotic arm, and mobile base.
2.  Capture an image of the scene containing the door.
3.  Detect the door handle's 3D coordinates in the camera's frame.
    - Option 1 (Default): Manual annotation by clicking on the displayed image.
    - Option 2 (Commented Out): Automatic detection using a pre-trained model.
4.  Transform the handle's coordinates from the camera frame to the robot's base frame.
5.  Execute the door-opening maneuver:
    a. Move the arm to an approach position near the handle.
    b. Close the gripper to grasp the handle.
    c. Pull the arm downwards to unlatch the door.
    d. Move the mobile base backward to pull the door open.
6.  Return the arm to a safe home position.
"""

import sys
import time
import cv2
import numpy as np

# --- Add project root to system path ---
# Ensures that custom modules like 'arm_kinova', 'dog_gs', etc., can be found.
root_dir = "./"
sys.path.append(root_dir)

# --- Import custom robot control classes ---
from camera import Camera
from arm_kinova import Arm
from dog_gs import RosBase

# from dtsam import DeticSAM # Uncomment when using model-based detection


# =============================================================================
# --- Option 1: Manual Handle Detection via Mouse Click ---
# =============================================================================
def get_handle_coords_manual(cam: Camera):
    """
    Captures an image, displays it, and waits for a user to click on the door handle.
    Calculates and returns the 3D coordinates of the selected point.

    Args:
        cam (Camera): An initialized Camera object.

    Returns:
        tuple: A tuple (X, Y, Z) representing the 3D coordinates in the camera frame (meters),
               or (None, None, None) if detection fails or is aborted.
    """
    print("\n--- Starting Manual Handle Detection ---")
    print("Waiting 2 seconds for the camera to stabilize...")
    time.sleep(2)

    rgb_img, depth_img = cam.capture_rgbd()
    if rgb_img is None or depth_img is None:
        print("[ERROR] Failed to capture RGBD frame.")
        return None, None, None

    # Store coordinates from the mouse callback
    coords = {"x": -1, "y": -1, "X": None, "Y": None, "Z": None, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\nPixel ({x},{y}) selected.")
            # 1) Attempt to get depth directly from the pixel
            d_raw = cam.get_depth_point(x, y, depth_img)

            # 2) If direct depth is invalid, try averaging a small region (ROI)
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

            # 3) Convert 2D pixel + depth to 3D coordinates
            X, Y, Z = cam.xy_depth_2_xyz(x, y, d_raw)
            print(
                f"-> Pixel ({x},{y}) | Depth: {d_raw:.0f}mm | 3D Coords (X,Y,Z): ({X:.4f}, {Y:.4f}, {Z:.4f}) m"
            )

            coords.update({"x": x, "y": y, "X": X, "Y": Y, "Z": Z, "done": True})

            # Draw a confirmation circle on the image
            display_img = rgb_img.copy()
            cv2.circle(display_img, (x, y), 8, (0, 0, 255), 2)
            cv2.imshow("RGB Click to Annotate Handle", display_img)

    # Setup and display the window for annotation
    cv2.namedWindow("RGB Click to Annotate Handle", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("RGB Click to Annotate Handle", on_mouse)
    cv2.imshow("RGB Click to Annotate Handle", rgb_img)

    print("Please LEFT-CLICK on the door handle. Press ESC to quit.")

    # Wait until a valid point is clicked or the user presses Esc
    while not coords["done"]:
        if cv2.waitKey(20) & 0xFF == 27:
            print("Annotation cancelled by user.")
            break

    cv2.destroyAllWindows()
    return coords["X"], coords["Y"], coords["Z"]


# =============================================================================
# --- Option 2: Automated Handle Detection via  QWen2.5 ---
# =============================================================================
def get_handle_coords_model(model: str, cam: Camera):
    """
    Captures an image and uses a detection model (e.g., Detic+SAM) to find the handle.
    Calculates and returns the 3D coordinates.

    Args:
        dtsam_model: An initialized detection model object.
        cam (Camera): An initialized Camera object.

    Returns:
        tuple: A tuple (X, Y, Z) of the handle's 3D coordinates (meters),
               or (None, None, None) if not found.
    """
    print("\n--- Starting Model-Based Handle Detection ---")
    rgb_img, depth_img = cam.capture_rgbd()
    if rgb_img is None or depth_img is None:
        print("[ERROR] Failed to capture RGBD frame.")
        return None, None, None

    x, y = 0, 0

    if x is None or y is None:
        print("[ERROR] Model failed to detect the handle.")
        return None, None, None

    # Once pixel is found, get depth and convert to 3D coords
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


# =============================================================================
# --- Main Execution Logic ---
# =============================================================================
def main():
    """
    Main function to run the complete door opening procedure.
    """
    # --- 1. Initialization ---
    print("Initializing robot components...")
    try:
        cam = Camera.init_from_yaml(cfg_path=f"{root_dir}/cfg/cfg_cam.yaml")
        arm = Arm.init_from_yaml(cfg_path=f"{root_dir}/cfg/cfg_arm_left.yaml")
        base = RosBase(linear_velocity=0.3, angular_velocity=0.5)
        # dtsam_model = DeticSAM(...) # Uncomment and initialize your model here for Option 2
    except Exception as e:
        print(f"[FATAL] Failed to initialize robot components: {e}")
        return

    # --- 2. Detect Handle ---
    # Use the manual method by default.
    x_cam, y_cam, z_cam = get_handle_coords_manual(cam)

    # To use the model-based detection, comment out the line above and uncomment the line below.
    # x_cam, y_cam, z_cam = get_handle_coords_model(dtsam_model, cam)

    if x_cam is None:
        print("Could not get handle coordinates. Aborting mission.")
        cam.disconnect()
        return

    # --- 3. Define Arm Poses ---
    print("\nCalculating arm target poses...")

    # Define the gripper's orientation (Roll, Pitch, Yaw) for grasping.
    # These values often require tuning for the specific handle type.
    rx_grasp = 1.512
    ry_grasp = -0.184
    rz_grasp = -1.422

    # Define the "pull down" position. We keep X/Y the same but move Z down.
    # The pull-down distance (e.g., 8 cm) may need adjustment.
    PULL_DOWN_DISTANCE = 0.08  # meters
    HOME_POSITION = [
        0.21248655021190643,
        -0.2564840614795685,
        0.5075023174285889,
        1.6500247716903687,
        1.11430025100708,
        0.12375058978796005,
    ]

    # The handle position becomes our "approach" and "grasp" target.
    xyzrpy_grasp_cam = [x_cam, y_cam, z_cam, rx_grasp, ry_grasp, rz_grasp]

    # The pull-down pose has a lower Z value in the camera frame.
    xyzrpy_pull_cam = [
        x_cam,
        y_cam,
        z_cam - PULL_DOWN_DISTANCE,
        rx_grasp,
        ry_grasp,
        rz_grasp,
    ]

    # --- 4. Coordinate Transformation & Pose Definition ---
    # Define the grasp pose in the camera frame first.
    xyzrpy_grasp_cam = [x_cam, y_cam, z_cam, rx_grasp, ry_grasp, rz_grasp]

    # a. Transform the grasp pose from the camera frame to the robot's base frame.
    print("Transforming grasp pose to base frame...")
    pos_grasp_base = arm.target2cam_xyzrpy_to_target2base_xyzrpy(
        xyzrpy_cam=xyzrpy_grasp_cam
    )

    # Ensure the desired gripper orientation is set correctly after transformation.
    pos_grasp_base[3:6] = rx_grasp, ry_grasp, rz_grasp
    print(f"Target Grasp Pose (Base Frame): {np.round(pos_grasp_base, 4)}")

    # b. Define the pull-down pose directly in the base frame.
    # This is done by subtracting from the Z-coordinate of the grasp pose.
    print(
        f"Calculating pull-down pose by moving {PULL_DOWN_DISTANCE}m down in the base frame..."
    )
    pos_pull_base = list(pos_grasp_base)  # Create a copy
    pos_pull_base[2] -= PULL_DOWN_DISTANCE  # Subtract from the Z-axis (index 2)

    print(f"Target Pull Pose (Base Frame):  {np.round(pos_pull_base, 4)}")
    time.sleep(1)

    # --- 5. Execute Door Opening Maneuver ---
    try:
        print("\n--- Starting Door Opening Maneuver ---")

        # a. Open gripper and move to grasp position
        print("Step 1: Opening gripper...")
        arm.control_gripper(open_value=0)  # Open wide
        time.sleep(1)

        print(f"Step 2: Moving to handle at {np.round(pos_grasp_base[:3], 4)}...")
        arm.move_p(pos_grasp_base)
        time.sleep(1.5)

        # b. Close gripper to grasp handle
        print("Step 3: Grasping handle...")
        arm.control_gripper(open_value=6000)  # Close to grasp
        time.sleep(1.5)  # Wait for grasp to be firm

        # Optional: Check if grasp was successful
        # if arm.get_gripper_grasp_return() != 2:
        #     print("[WARNING] Gripper did not confirm a successful grasp.")

        # c. Pull handle down
        print(f"Step 4: Pulling handle down to {np.round(pos_pull_base[:3], 4)}...")
        arm.move_p(pos_pull_base)
        time.sleep(1.0)

        # d. Move base backward to open the door
        print("Step 5: Moving robot base backward to open door...")
        # Move backward (negative linear velocity) for 2.5 seconds.
        base.move_T(-2, linear_velocity=0.2)
        print("Door opening maneuver complete.")

    except Exception as e:
        print(f"[ERROR] An error occurred during the arm/base maneuver: {e}")

    finally:
        # --- 6. Cleanup and Shutdown ---
        print("\n--- Mission Finished. Returning to home position. ---")
        # Optional: Add a 'home_position' and move the arm back
        arm.control_gripper(open_value=0)
        arm.move_p(HOME_POSITION)
        # For now, just release the gripper
        time.sleep(1)

        # Disconnect camera
        cam.disconnect()
        print("System shutdown.")


if __name__ == "__main__":
    main()
