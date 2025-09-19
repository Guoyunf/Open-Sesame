#!/usr/bin/env python

"""High level controller for a quadruped base driven through ROS.

This module exposes the :class:`RosBase` class which publishes velocity
commands, consumes odometry feedback and optionally interfaces with the ROS
navigation stack.  Compared with the original socket based implementation the
class now offers closed loop helpers that rely on odometry to realise precise
translations and rotations.  The helpers use simple PID controllers allowing
the robot to reach positional and angular targets without tedious manual tuning
of open loop timings.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry

# Conditional import for keyboard control
if os.name == "nt":
    import msvcrt
else:
    import sys, tty, termios

# Imports for ROS Navigation Stack
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class RosBase(object):
    """
    A ROS-based class to control a robot's base.
    """

    def __init__(
        self,
        linear_velocity=0.2,
        angular_velocity=0.5,
        cmd_vel_topic="/nav/cmd_vel",
        odom_topic="/nav/leg_odom",
        move_base_action="/move_base",
        translation_scale: float = 1.0,
        lateral_scale: float = 1.0,
        rotation_scale: float = 1.0,
    ):
        """Initializes the ROS node, publishers, subscribers, and action client.

        Parameters
        ----------
        translation_scale, lateral_scale, rotation_scale:
            Optional multiplicative factors applied to raw odometry readings.
            They can be calibrated to correct systematic scale errors between
            commanded and measured motion.
        """
        # If the ROS node has not been initialized yet, initialize one.
        # This allows multiple classes to be instantiated safely in the same process.
        if not rospy.get_node_uri():
            # Use disable_signals=True for better integration in larger scripts
            rospy.init_node("ros_base_wrapper", anonymous=True, disable_signals=True)
            print("ROS Node 'ros_base_wrapper' has been initialized by RosBase class.")

        # Store parameters
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.control_rate_hz = 50.0
        self.translation_scale = float(translation_scale)
        self.lateral_scale = float(lateral_scale)
        self.rotation_scale = float(rotation_scale)
        self._last_motion_report: Optional[Dict[str, Any]] = None
        self.markers = {}  # To store named locations

        # State variables
        self.current_pose = Pose()
        self.odom_lock = threading.Lock()

        # ROS Publishers and Subscribers
        self.vel_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.odom_subscriber = rospy.Subscriber(
            odom_topic, Odometry, self._odom_callback
        )

        # ROS Action Client for Navigation
        self.move_base_client = actionlib.SimpleActionClient(
            move_base_action, MoveBaseAction
        )
        rospy.loginfo(f"Waiting for '{move_base_action}' action server...")
        # It's good practice to wait for the server to be available
        server_available = self.move_base_client.wait_for_server(rospy.Duration(5))
        if server_available:
            rospy.loginfo("Action server found.")
        else:
            rospy.logwarn(
                "Move base action server not available within timeout. "
                "Navigation helpers will retry on demand."
            )

        # Wait for the first odometry message to initialize pose
        rospy.loginfo("Waiting for initial odometry message...")
        while self.current_pose == Pose() and not rospy.is_shutdown():
            rospy.sleep(0.1)

        self.start_x, self.start_y, self.start_theta = self.get_location(if_p=False)
        rospy.loginfo("==========\nRosBase Controller Initialized\n==========")

    def __str__(self):
        return (
            f"[RosBase]: Default Linear Velocity: {self.linear_velocity} m/s, "
            f"Default Angular Velocity: {self.angular_velocity} rad/s"
        )

    def _odom_callback(self, msg):
        """
        Callback function for the odometry subscriber.
        Updates the robot's current pose thread-safely.
        """
        with self.odom_lock:
            self.current_pose = msg.pose.pose

    def _publish_cmd_vel(self, linear_x=0.0, angular_z=0.0, linear_y=0.0):
        """Helper function to publish a Twist message."""
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.linear.y = linear_y
        twist_msg.angular.z = angular_z
        self.vel_publisher.publish(twist_msg)

    def move_forward(self, vel=None):
        if vel is None:
            vel = self.linear_velocity
        self._publish_cmd_vel(abs(vel), 0)

    def move_back(self, vel=None):
        if vel is None:
            vel = self.linear_velocity
        self._publish_cmd_vel(-abs(vel), 0)

    def move_left(self, vel=None):
        if vel is None:
            vel = self.angular_velocity
        self._publish_cmd_vel(0, abs(vel))

    def move_right(self, vel=None):
        if vel is None:
            vel = self.angular_velocity
        self._publish_cmd_vel(0, -abs(vel))

    def strafe_left(self, vel=None):
        if vel is None:
            vel = self.linear_velocity
        self._publish_cmd_vel(0, 0, abs(vel))

    def strafe_right(self, vel=None):
        if vel is None:
            vel = self.linear_velocity
        self._publish_cmd_vel(0, 0, -abs(vel))

    def move_stop(self, if_p=False):
        self._publish_cmd_vel(0.0, 0.0, 0.0)
        if if_p:
            rospy.loginfo("[Base Stop]")

    def _store_motion_report(self, motion_type: str, report: Dict[str, Any]) -> None:
        """Cache telemetry about the last closed-loop motion."""

        cached = dict(report)
        cached["motion_type"] = motion_type
        cached["timestamp"] = rospy.get_time()
        self._last_motion_report = cached

    def get_last_motion_report(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the most recent motion telemetry."""

        if self._last_motion_report is None:
            return None
        return dict(self._last_motion_report)

    def update_odometry_scales(
        self,
        translation: Optional[float] = None,
        lateral: Optional[float] = None,
        rotation: Optional[float] = None,
    ) -> None:
        """Manually set scaling factors applied to odometry feedback."""

        if translation is not None:
            if translation <= 0:
                raise ValueError("translation scale must be positive")
            self.translation_scale = float(translation)
            rospy.loginfo("Updated forward odometry scale to %.4f", self.translation_scale)

        if lateral is not None:
            if lateral <= 0:
                raise ValueError("lateral scale must be positive")
            self.lateral_scale = float(lateral)
            rospy.loginfo("Updated lateral odometry scale to %.4f", self.lateral_scale)

        if rotation is not None:
            if rotation <= 0:
                raise ValueError("rotation scale must be positive")
            self.rotation_scale = float(rotation)
            rospy.loginfo("Updated rotational odometry scale to %.4f", self.rotation_scale)

    def _calibrate_scale(
        self,
        actual: float,
        odom: float,
        motion_type: str,
    ) -> float:
        """Helper computing |actual|/|odom| ensuring valid inputs."""

        odom_mag = abs(float(odom))
        if odom_mag <= 1e-6:
            raise ValueError("odometry distance must be non-zero for calibration")
        scale = abs(float(actual)) / odom_mag
        if scale <= 0:
            raise ValueError("calculated scale must be positive")
        rospy.loginfo(
            "Calibrated %s odometry scale: actual=%.4f, odom=%.4f, scale=%.4f",
            motion_type,
            actual,
            odom,
            scale,
        )
        return scale

    def calibrate_translation_scale(
        self, actual_distance: float, odom_distance: Optional[float] = None
    ) -> float:
        """Update the forward odometry scale using a ground-truth measurement."""

        if odom_distance is None:
            report = self.get_last_motion_report()
            if not report or report.get("motion_type") != "linear":
                raise ValueError(
                    "No previous linear motion report available; pass odom_distance explicitly."
                )
            odom_distance = report["odom_traveled"]
        scale = self._calibrate_scale(actual_distance, odom_distance, "linear")
        self.translation_scale = scale
        return scale

    def calibrate_lateral_scale(
        self, actual_distance: float, odom_distance: Optional[float] = None
    ) -> float:
        """Update the lateral odometry scale using a ground-truth measurement."""

        if odom_distance is None:
            report = self.get_last_motion_report()
            if not report or report.get("motion_type") != "lateral":
                raise ValueError(
                    "No previous lateral motion report available; pass odom_distance explicitly."
                )
            odom_distance = report["odom_traveled"]
        scale = self._calibrate_scale(actual_distance, odom_distance, "lateral")
        self.lateral_scale = scale
        return scale

    def calibrate_rotation_scale(
        self, actual_angle: float, odom_angle: Optional[float] = None
    ) -> float:
        """Update the angular odometry scale using a ground-truth measurement."""

        if odom_angle is None:
            report = self.get_last_motion_report()
            if not report or report.get("motion_type") != "angular":
                raise ValueError(
                    "No previous angular motion report available; pass odom_angle explicitly."
                )
            odom_angle = report["odom_traveled"]
        scale = self._calibrate_scale(actual_angle, odom_angle, "angular")
        self.rotation_scale = scale
        return scale

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap ``angle`` to the ``[-pi, pi]`` interval."""

        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _project_displacement(
        start_pose: Sequence[float], current_pose: Sequence[float]
    ) -> Tuple[float, float]:
        """Project displacement between ``start_pose`` and ``current_pose``.

        Parameters
        ----------
        start_pose:
            ``(x, y, theta)`` tuple describing the starting pose.
        current_pose:
            ``(x, y, theta)`` tuple describing the current pose.

        Returns
        -------
        Tuple[float, float]
            Forward and lateral displacement expressed in the robot body frame
            aligned with ``start_pose``.
        """

        sx, sy, stheta = start_pose
        cx, cy, _ = current_pose
        dx = cx - sx
        dy = cy - sy
        cos_theta = math.cos(stheta)
        sin_theta = math.sin(stheta)
        forward = cos_theta * dx + sin_theta * dy
        lateral = -sin_theta * dx + cos_theta * dy
        return forward, lateral

    @staticmethod
    def _coerce_location(location: Sequence[float]) -> Tuple[float, float, float]:
        """Ensure ``location`` is iterable with three numeric elements."""

        try:
            x, y, theta = location
        except (TypeError, ValueError):
            raise ValueError(
                "location must contain exactly three elements (x, y, theta)"
            ) from None
        return float(x), float(y), float(theta)

    def move_char(self, char, linear_velocity=None, angular_velocity=None):
        linear_vel = (
            linear_velocity if linear_velocity is not None else self.linear_velocity
        )
        angular_vel = (
            angular_velocity if angular_velocity is not None else self.angular_velocity
        )

        # Mapping for wasd and arrow keys
        if char in ("w", "H"):  # 'H' is up arrow on some systems
            self.move_forward(vel=linear_vel)
        elif char in ("s", "P"):  # 'P' is down arrow
            self.move_back(vel=linear_vel)
        elif char in ("a", "K"):  # 'K' is left arrow
            self.move_left(vel=angular_vel)
        elif char in ("d", "M"):  # 'M' is right arrow
            self.move_right(vel=angular_vel)
        elif char == "j":  # strafe left
            self.strafe_left(vel=linear_vel)
        elif char == "l":  # strafe right
            self.strafe_right(vel=linear_vel)
        elif char == "x":
            self.move_stop()
        else:
            pass  # Ignore other keys

    def move_T(self, T, linear_velocity=None, if_p=False):
        """Moves forward (T>0) or backward (T<0) for a specific duration T."""
        if linear_velocity is None:
            linear_velocity = self.linear_velocity

        rate = rospy.Rate(50)  # 50 Hz control loop
        start_time = rospy.get_time()

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < abs(T):
            vel = linear_velocity if T > 0 else -linear_velocity
            self.move_forward(vel) if T > 0 else self.move_back(abs(vel))
            if if_p:
                rospy.loginfo(f"[Time]: {rospy.get_time() - start_time:.2f}")
            rate.sleep()
        self.move_stop(if_p=True)

    def rotate_T(self, T, angular_velocity=None, if_p=False):
        """Rotates left (T<0) or right (T>0) for a specific duration T."""
        if angular_velocity is None:
            angular_velocity = self.angular_velocity

        rate = rospy.Rate(50)
        start_time = rospy.get_time()

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < abs(T):
            if T > 0:  # Positive T for right rotation
                self.move_right(angular_velocity)
            else:  # Negative T for left rotation
                self.move_left(angular_velocity)
            if if_p:
                rospy.loginfo(f"[Time]: {rospy.get_time() - start_time:.2f}")
            rate.sleep()
        self.move_stop(if_p=True)

    def strafe_T(self, T, linear_velocity=None, if_p=False):
        """Strafes left (T>0) or right (T<0) for duration ``T``."""
        if linear_velocity is None:
            linear_velocity = self.linear_velocity

        rate = rospy.Rate(50)
        start_time = rospy.get_time()

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < abs(T):
            vel = linear_velocity if T > 0 else -linear_velocity
            self.strafe_left(vel) if T > 0 else self.strafe_right(abs(vel))
            if if_p:
                rospy.loginfo(f"[Time]: {rospy.get_time() - start_time:.2f}")
            rate.sleep()
        self.move_stop(if_p=True)

    def move_distance(
        self,
        distance: float,
        max_linear_velocity: Optional[float] = None,
        tolerance: float = 0.01,
        kp: float = 1.2,
        ki: float = 0.0,
        kd: float = 0.05,
        integral_limit: float = 0.5,
        max_time: Optional[float] = None,
        log_progress: bool = False,
    ) -> dict:
        """Drive forward/backward for ``distance`` metres using odometry feedback.

        Positive distances move the robot forward while negative values command a
        backward motion.  The motion is controlled by a simple PID controller
        running at ``self.control_rate_hz``.

        Notes
        -----
        The raw odometry displacement is multiplied by
        :attr:`translation_scale` before being compared to ``distance``.  Tune
        this scale to compensate for systematic odometry bias measured during
        hardware experiments.

        Returns a dictionary containing telemetry of the manoeuvre.  The keys
        are ``reached`` (``bool``), ``traveled`` (``float``), ``error``
        (``float``) and ``duration`` (``float``).
        """

        if max_linear_velocity is None:
            max_linear_velocity = abs(self.linear_velocity)
        max_linear_velocity = abs(max_linear_velocity)
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        start_pose = self.get_location()
        start_time = rospy.get_time()
        last_time = start_time
        integral = 0.0
        previous_error = distance
        traveled_raw = 0.0
        traveled = 0.0
        reached = False
        rate = rospy.Rate(self.control_rate_hz)

        if max_time is None:
            # generous timeout: 3x the time needed at nominal velocity + 1s
            nominal = abs(distance) / max(max_linear_velocity, 1e-6)
            max_time = max(1.0, 3.0 * nominal)

        rospy.loginfo(
            "[RosBase] Closed-loop linear target=%.3f m (max vel %.2f m/s, scale %.3f)",
            distance,
            max_linear_velocity,
            self.translation_scale,
        )

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = max(1.0 / self.control_rate_hz, current_time - last_time)
            current_pose = self.get_location()
            traveled_raw, _ = self._project_displacement(start_pose, current_pose)
            traveled = traveled_raw * self.translation_scale
            error = distance - traveled

            if abs(error) <= tolerance:
                reached = True
                break

            integral += error * dt
            integral = max(-integral_limit, min(integral, integral_limit))
            derivative = (error - previous_error) / dt
            control = kp * error + ki * integral + kd * derivative
            control = max(-max_linear_velocity, min(control, max_linear_velocity))

            self._publish_cmd_vel(linear_x=control, angular_z=0.0)

            if log_progress:
                rospy.loginfo(
                    "[RosBase] distance remaining: %.3f m (odom %.3f m, scaled %.3f m), cmd %.3f m/s",
                    error,
                    traveled_raw,
                    traveled,
                    control,
                )

            previous_error = error
            last_time = current_time

            if current_time - start_time > max_time:
                rospy.logwarn("[RosBase] move_distance timed out after %.2f s", max_time)
                break

            rate.sleep()

        self.move_stop()

        duration = rospy.get_time() - start_time
        current_pose = self.get_location()
        traveled_raw, _ = self._project_displacement(start_pose, current_pose)
        final_traveled = traveled_raw * self.translation_scale
        final_error = distance - final_traveled
        rospy.loginfo(
            "[RosBase] Closed-loop linear motion finished (reached=%s, error=%.4f m, odom=%.4f m, scaled=%.4f m)",
            reached,
            final_error,
            traveled_raw,
            final_traveled,
        )
        result = {
            "reached": reached,
            "traveled": final_traveled,
            "odom_traveled": traveled_raw,
            "error": final_error,
            "duration": duration,
            "scale": self.translation_scale,
        }
        self._store_motion_report("linear", result)
        return result

    def strafe_distance(
        self,
        distance: float,
        max_linear_velocity: Optional[float] = None,
        tolerance: float = 0.01,
        kp: float = 1.2,
        ki: float = 0.0,
        kd: float = 0.05,
        integral_limit: float = 0.5,
        max_time: Optional[float] = None,
        log_progress: bool = False,
    ) -> dict:
        """Strafe left/right for ``distance`` metres using odometry feedback.

        Notes
        -----
        The raw lateral odometry displacement is multiplied by
        :attr:`lateral_scale` before being compared to ``distance``.
        """

        if max_linear_velocity is None:
            max_linear_velocity = abs(self.linear_velocity)
        max_linear_velocity = abs(max_linear_velocity)
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        start_pose = self.get_location()
        start_time = rospy.get_time()
        last_time = start_time
        integral = 0.0
        previous_error = distance
        lateral_raw = 0.0
        lateral = 0.0
        reached = False
        rate = rospy.Rate(self.control_rate_hz)

        if max_time is None:
            nominal = abs(distance) / max(max_linear_velocity, 1e-6)
            max_time = max(1.0, 3.0 * nominal)

        rospy.loginfo(
            "[RosBase] Closed-loop strafe target=%.3f m (max vel %.2f m/s, scale %.3f)",
            distance,
            max_linear_velocity,
            self.lateral_scale,
        )

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = max(1.0 / self.control_rate_hz, current_time - last_time)
            current_pose = self.get_location()
            _, lateral_raw = self._project_displacement(start_pose, current_pose)
            lateral = lateral_raw * self.lateral_scale
            error = distance - lateral

            if abs(error) <= tolerance:
                reached = True
                break

            integral += error * dt
            integral = max(-integral_limit, min(integral, integral_limit))
            derivative = (error - previous_error) / dt
            control = kp * error + ki * integral + kd * derivative
            control = max(-max_linear_velocity, min(control, max_linear_velocity))

            self._publish_cmd_vel(linear_x=0.0, angular_z=0.0, linear_y=control)

            if log_progress:
                rospy.loginfo(
                    "[RosBase] lateral remaining: %.3f m (odom %.3f m, scaled %.3f m), cmd %.3f m/s",
                    error,
                    lateral_raw,
                    lateral,
                    control,
                )

            previous_error = error
            last_time = current_time

            if current_time - start_time > max_time:
                rospy.logwarn("[RosBase] strafe_distance timed out after %.2f s", max_time)
                break

            rate.sleep()

        self.move_stop()

        duration = rospy.get_time() - start_time
        current_pose = self.get_location()
        _, lateral_raw = self._project_displacement(start_pose, current_pose)
        final_lateral = lateral_raw * self.lateral_scale
        final_error = distance - final_lateral
        rospy.loginfo(
            "[RosBase] Closed-loop strafe finished (reached=%s, error=%.4f m, odom=%.4f m, scaled=%.4f m)",
            reached,
            final_error,
            lateral_raw,
            final_lateral,
        )
        result = {
            "reached": reached,
            "traveled": final_lateral,
            "odom_traveled": lateral_raw,
            "error": final_error,
            "duration": duration,
            "scale": self.lateral_scale,
        }
        self._store_motion_report("lateral", result)
        return result

    def rotate_angle(
        self,
        angle: float,
        max_angular_velocity: Optional[float] = None,
        tolerance: float = math.radians(1.0),
        kp: float = 2.5,
        ki: float = 0.0,
        kd: float = 0.1,
        integral_limit: float = 1.0,
        max_time: Optional[float] = None,
        log_progress: bool = False,
    ) -> dict:
        """Rotate by ``angle`` radians using odometry feedback.

        Notes
        -----
        The yaw change reported by odometry is multiplied by
        :attr:`rotation_scale` before being compared to ``angle``.
        """

        if max_angular_velocity is None:
            max_angular_velocity = abs(self.angular_velocity)
        max_angular_velocity = abs(max_angular_velocity)
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        start_pose = self.get_location()
        start_yaw = start_pose[2]
        start_time = rospy.get_time()
        last_time = start_time
        integral = 0.0
        previous_error = self._normalize_angle(angle)
        yaw_raw = 0.0
        yaw = 0.0
        reached = False
        rate = rospy.Rate(self.control_rate_hz)

        if max_time is None:
            nominal = abs(angle) / max(max_angular_velocity, 1e-6)
            max_time = max(1.0, 3.0 * nominal)

        rospy.loginfo(
            "[RosBase] Closed-loop rotation target=%.2f deg (max vel %.2f rad/s, scale %.3f)",
            math.degrees(angle),
            max_angular_velocity,
            self.rotation_scale,
        )

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = max(1.0 / self.control_rate_hz, current_time - last_time)
            _, _, current_yaw = self.get_location()
            yaw_raw = self._normalize_angle(current_yaw - start_yaw)
            yaw = yaw_raw * self.rotation_scale
            error = self._normalize_angle(angle - yaw)

            if abs(error) <= tolerance:
                reached = True
                break

            integral += error * dt
            integral = max(-integral_limit, min(integral, integral_limit))
            derivative = (error - previous_error) / dt
            control = kp * error + ki * integral + kd * derivative
            control = max(-max_angular_velocity, min(control, max_angular_velocity))

            self._publish_cmd_vel(linear_x=0.0, angular_z=control)

            if log_progress:
                rospy.loginfo(
                    "[RosBase] angular err: %.3f rad (odom %.3f rad, scaled %.3f rad), cmd %.3f rad/s",
                    error,
                    yaw_raw,
                    yaw,
                    control,
                )

            previous_error = error
            last_time = current_time

            if current_time - start_time > max_time:
                rospy.logwarn("[RosBase] rotate_angle timed out after %.2f s", max_time)
                break

            rate.sleep()

        self.move_stop()

        duration = rospy.get_time() - start_time
        _, _, final_yaw_reading = self.get_location()
        final_yaw_raw = self._normalize_angle(final_yaw_reading - start_yaw)
        final_yaw = final_yaw_raw * self.rotation_scale
        final_error = self._normalize_angle(angle - final_yaw)
        rospy.loginfo(
            "[RosBase] Closed-loop rotation finished (reached=%s, error=%.4f rad, odom=%.4f rad, scaled=%.4f rad)",
            reached,
            final_error,
            final_yaw_raw,
            final_yaw,
        )
        result = {
            "reached": reached,
            "traveled": final_yaw,
            "odom_traveled": final_yaw_raw,
            "error": final_error,
            "duration": duration,
            "scale": self.rotation_scale,
        }
        self._store_motion_report("angular", result)
        return result

    def get_location(self, if_p=False):
        """
        Returns the current location [x, y, theta] from odometry.
        """
        with self.odom_lock:
            position = self.current_pose.position
            orientation_q = self.current_pose.orientation

        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        location = [position.x, position.y, yaw]
        if if_p:
            rospy.loginfo(
                f"Location: [x={location[0]:.3f}, y={location[1]:.3f}, theta={location[2]:.3f}]"
            )
        return location

    def _build_navigation_goal(
        self, location: Sequence[float], frame_id: str = "map"
    ) -> MoveBaseGoal:
        """Create a :class:`MoveBaseGoal` from ``location``."""

        x, y, theta = self._coerce_location(location)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame_id
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        return goal

    def navigate_to(
        self,
        location: Sequence[float],
        frame_id: str = "map",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[int]:
        """Send a goal to the navigation stack.

        Parameters
        ----------
        location:
            Target pose ``(x, y, theta)`` expressed in ``frame_id``.
        frame_id:
            Coordinate frame used for the goal.  ``"map"`` is the default when
            a localisation system is running.
        timeout:
            Maximum time to wait for the goal to finish.  ``None`` waits
            indefinitely.  When the timeout is exceeded the goal is cancelled.
        wait:
            If ``False`` the function returns immediately after sending the
            goal without waiting for completion.  In that case ``None`` is
            returned.

        Returns
        -------
        Optional[int]
            Resulting :class:`GoalStatus` value, or ``None`` if ``wait`` is
            ``False``.
        """

        if not self.move_base_client.wait_for_server(rospy.Duration(0.0)):
            rospy.loginfo("Waiting for move_base action server...")
            if not self.move_base_client.wait_for_server(rospy.Duration(5.0)):
                rospy.logerr("Move base action server is unavailable.")
                return GoalStatus.ABORTED

        x, y, theta = self._coerce_location(location)
        goal = self._build_navigation_goal((x, y, theta), frame_id)
        rospy.loginfo(
            "Sending navigation goal to x=%.2f, y=%.2f, theta=%.2f", x, y, theta
        )
        self.move_base_client.send_goal(goal)

        if not wait:
            return None

        if timeout is None:
            self.move_base_client.wait_for_result()
        else:
            finished = self.move_base_client.wait_for_result(rospy.Duration(timeout))
            if not finished:
                rospy.logwarn(
                    "Navigation goal timed out after %.2f s, cancelling...", timeout
                )
                self.move_base_client.cancel_goal()
                return GoalStatus.ABORTED

        state = self.move_base_client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation goal succeeded.")
        else:
            rospy.logwarn("Navigation goal finished with status %s", state)
        return state

    def move_location(
        self, location: Sequence[float], frame_id: str = "map", timeout: Optional[float] = None
    ) -> Optional[int]:
        """Backward compatible wrapper around :meth:`navigate_to`."""

        return self.navigate_to(location, frame_id=frame_id, timeout=timeout, wait=True)

    def navigate_waypoints(
        self,
        waypoints: Iterable[Sequence[float]],
        frame_id: str = "map",
        timeout: Optional[float] = None,
        stop_on_failure: bool = True,
    ) -> List[int]:
        """Navigate through a list of waypoints sequentially."""

        waypoint_list = [self._coerce_location(wp) for wp in waypoints]
        states: List[int] = []
        for index, waypoint in enumerate(waypoint_list, start=1):
            x, y, theta = waypoint
            rospy.loginfo(
                "Navigating to waypoint %d/%d: (%.3f, %.3f, %.3f)",
                index,
                len(waypoint_list),
                x,
                y,
                theta,
            )
            state = self.navigate_to(
                (x, y, theta), frame_id=frame_id, timeout=timeout, wait=True
            )
            states.append(state)
            if stop_on_failure and state != GoalStatus.SUCCEEDED:
                rospy.logwarn("Stopping waypoint navigation due to failure at index %d", index)
                break
        return states

    def cancel_navigation(self) -> None:
        """Cancel the current navigation goal if one is active."""

        rospy.loginfo("Cancelling current navigation goal...")
        self.move_base_client.cancel_goal()

    def get_navigation_state(self) -> int:
        """Return the current :class:`GoalStatus` of the navigation client."""

        return self.move_base_client.get_state()

    def is_navigation_active(self) -> bool:
        """Return ``True`` if a navigation goal is active or pending."""

        state = self.move_base_client.get_state()
        return state in (GoalStatus.ACTIVE, GoalStatus.PENDING)

    def insert_marker(self, marker_name):
        """Stores the robot's current location with a given name."""
        current_location = self.get_location()
        self.markers[marker_name] = current_location
        rospy.loginfo(f"Marker '{marker_name}' inserted at: {current_location}")

    def move_marker(self, marker_name):
        """Moves the robot to a previously stored marker's location."""
        if marker_name in self.markers:
            location = self.markers[marker_name]
            rospy.loginfo(f"Moving to marker '{marker_name}' at {location}")
            self.move_location(location)
        else:
            rospy.logerr(f"Marker '{marker_name}' not found.")

    def move_keyboard(self, interval=0.1):
        """Controls the robot using the keyboard."""
        rospy.loginfo("Starting keyboard control. Press 'x' to stop, Ctrl+C to exit.")

        # Set up getch for Linux or Windows
        if os.name != "nt":
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

        def getch():
            if os.name == "nt":
                return msvcrt.getch().decode("utf-8")
            else:
                try:
                    tty.setraw(sys.stdin.fileno())
                    char = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return char

        while not rospy.is_shutdown():
            char = getch()
            if char == "\x03":  # Ctrl+C
                break
            self.move_char(char)
            rospy.sleep(interval)

        self.move_stop(if_p=True)

    def shutdown(self):
        """Cleanly stops the robot and shuts down the node."""
        rospy.loginfo("Shutting down RosBase Controller...")
        self.move_stop()
        self.odom_subscriber.unregister()
        rospy.signal_shutdown("Shutdown requested.")


if __name__ == "__main__":
    try:
        # Initialize the base controller with fast settings
        base = RosBase(linear_velocity=0.5, angular_velocity=0.8)
        print(base)

        # Example Usage:
        # 1. Keyboard control (uncomment to use)
        # print("\n--- Starting Keyboard Control ---")
        # base.move_keyboard(interval=0.01)

        # 2. Closed-loop movement example using odometry
        print("\n--- Testing Closed-loop Movements ---")
        rospy.loginfo("Moving forward 0.5 m using odometry feedback...")
        base.move_distance(0.5, max_linear_velocity=0.3)
        rospy.sleep(1)  # Pause
        rospy.loginfo("Rotating left 90 degrees using odometry feedback...")
        base.rotate_angle(math.radians(90), max_angular_velocity=0.6)
        rospy.sleep(1)

        # # 3. Marker and Navigation example
        # # Assumes a navigation stack (like move_base) is running
        # print("\n--- Testing Navigation ---")
        # base.insert_marker("start_point")
        # rospy.sleep(1)

        # # Move to a specified location [x, y, theta]
        # # NOTE: This requires a running navigation stack (e.g., AMCL, move_base)
        # # and a map. The coordinates are relative to the 'map' frame by default.
        # rospy.loginfo("Sending goal to a new location (1.0, 0.5, 0.0)...")
        # base.move_location([1.0, 0.5, 0.0])
        # rospy.sleep(1)

        # rospy.loginfo("Returning to 'start_point' marker...")
        # base.move_marker("start_point")

    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted.")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        # Ensure the robot stops and node shuts down cleanly
        if "base" in locals() and isinstance(base, RosBase):
            base.shutdown()
