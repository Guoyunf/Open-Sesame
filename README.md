# Open-Sesame

Collection of utilities for robotics experiments, including camera and arm
helpers as well as remote server management tools.

Documentation for the ROS1/ROS2 bridge setup is available at
[`docs/ros_bridge.md`](docs/ros_bridge.md).

## Door opening state machine

The `tasks` package provides a simple finite state machine that orchestrates
primitives in the `primitives` package to open a door. The pull primitive
monitors the effort of joint 3 to determine whether the door handle has been
accidentally released: a positive effort indicates the handle has been dragged
away, a negative value is considered normal, and a zero effort is resolved by
looking back at previous non‑zero measurements. During the pull, a background
thread continuously logs the joint effort to ``joint3_effort_attempt<i>.txt`` so
that the entire trajectory can be inspected later. If an error is detected
during the pull phase, the state machine automatically retries the entire
sequence up to three times before giving up. On each failed attempt, the base
first backs away by about 20 cm before re-approaching the handle.

A high-level helper `open_door` combines this state machine with camera-based
handle detection and coordinate transformation. The handle location can be
obtained either by manually clicking in the camera image or by plugging in a
vision model. Detected coordinates are converted from the camera frame to the
robot's base frame via `Arm.target2cam_xyzrpy_to_target2base_xyzrpy`, and all
poses are configured using values from `cfg/cfg_door_open.yaml`.
