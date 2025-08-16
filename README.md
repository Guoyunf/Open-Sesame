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
looking back at previous non‑zero measurements. If an error is detected during
the pull phase, the state machine automatically retries the entire sequence up
to three times before giving up.
