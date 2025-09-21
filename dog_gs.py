#!/usr/bin/env python

"""
Version: v2
Brief: This script provides a ROS-based interface to control a robot's base,
       replacing the original socket-based API. It uses ROS topics for movement
       control and odometry feedback, and the actionlib for navigation goals.
"""
import rospy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import tf
import math
import time
import os
import threading

try:  # Optional ROS 2 imports used for posture control commands
    import rclpy
    from rclpy.utilities import is_initialized as rclpy_is_initialized
except ImportError:  # pragma: no cover - ROS 2 not installed in some environments
    rclpy = None
    rclpy_is_initialized = None

try:  # RobotCommand service definition lives in the ROS 2 workspace
    from ysc_robot_msgs.srv import RobotCommand as RobotCommandSrv
except ImportError:  # pragma: no cover - service definition unavailable during tests
    RobotCommandSrv = None

# Conditional import for keyboard control
if os.name == "nt":
    import msvcrt
else:
    import sys, tty, termios

# Imports for ROS Navigation Stack
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def _make_robot_command_request():
    """Create an empty RobotCommand request instance."""

    if RobotCommandSrv is None:
        raise RuntimeError(
            "RobotCommand service support requires the message definitions to be installed."
        )

    request_cls = getattr(RobotCommandSrv, "Request", None)
    if request_cls is None:
        request_cls = getattr(RobotCommandSrv, "_request_class", None)

    if request_cls is None:  # pragma: no cover - depends on runtime bindings
        raise RuntimeError("RobotCommand request type is unavailable.")

    return request_cls()


def _set_robot_command_name(request, command_name):
    """Populate the command field on a RobotCommand request."""

    if hasattr(request, "comamnd_name"):
        setattr(request, "comamnd_name", command_name)
    elif hasattr(request, "command_name"):
        setattr(request, "command_name", command_name)
    else:  # pragma: no cover - defensive for unexpected interface
        raise AttributeError(
            "RobotCommand request does not define 'comamnd_name' or 'command_name'."
        )


def _call_robot_command_service_ros2(
    command_name,
    *,
    service_name="/nav/robot_command",
    timeout_sec=5.0,
):
    """Send a ``RobotCommand`` request via ROS 2."""

    if rclpy is None or RobotCommandSrv is None:
        raise RuntimeError(
            "RobotCommand service support requires the ROS 2 Python packages and "
            "the 'ysc_robot_msgs' interfaces to be installed."
        )

    needs_shutdown = False
    is_init = False
    if rclpy_is_initialized is not None:
        is_init = rclpy_is_initialized()
    else:  # pragma: no cover - depends on ROS 2 version at runtime
        try:
            is_init = rclpy.ok()
        except Exception:
            is_init = False

    if not is_init:
        rclpy.init(args=None)
        needs_shutdown = True

    node = rclpy.create_node("dog_gs_robot_command_client")

    try:
        client = node.create_client(RobotCommandSrv, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError(
                f"Service '{service_name}' unavailable after {timeout_sec} seconds."
            )

        request = _make_robot_command_request()
        _set_robot_command_name(request, command_name)

        future = client.call_async(request)

        end_time = time.monotonic() + timeout_sec if timeout_sec is not None else None
        while rclpy.ok() and not future.done():
            spin_timeout = 0.1
            if end_time is not None:
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    future.cancel()
                    raise TimeoutError(
                        f"Robot command '{command_name}' timed out after {timeout_sec} seconds."
                    )
                spin_timeout = min(spin_timeout, max(0.0, remaining))
            rclpy.spin_once(node, timeout_sec=spin_timeout)

        if future.cancelled():  # pragma: no cover - defensive
            raise RuntimeError("Robot command service call was cancelled.")

        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"Robot command service failed: {exception}")

        return future.result()
    finally:
        node.destroy_node()
        if needs_shutdown:
            rclpy.shutdown()


def _call_robot_command_service_ros1(
    command_name,
    *,
    service_name="/nav/robot_command",
    timeout_sec=5.0,
):
    """Send a ``RobotCommand`` request via ROS 1."""

    if RobotCommandSrv is None:
        raise RuntimeError(
            "RobotCommand service support requires the message definitions to be installed."
        )

    try:
        rospy.wait_for_service(service_name, timeout=timeout_sec)
    except rospy.ROSException as exc:
        raise TimeoutError(
            f"Service '{service_name}' unavailable after {timeout_sec} seconds."
        ) from exc

    client = rospy.ServiceProxy(service_name, RobotCommandSrv)
    request = _make_robot_command_request()
    _set_robot_command_name(request, command_name)

    try:
        return client(request)
    except rospy.ServiceException as exc:  # pragma: no cover - depends on ROS runtime
        raise RuntimeError(f"Robot command service failed via ROS 1: {exc}") from exc


def _call_robot_command_service(
    command_name,
    *,
    service_name="/nav/robot_command",
    timeout_sec=5.0,
    transport="auto",
):
    """Send a ``RobotCommand`` request using ROS 1, ROS 2 or both."""

    if transport not in {"auto", "ros1", "ros2"}:
        raise ValueError("transport must be 'auto', 'ros1', or 'ros2'.")

    transports_to_try = []
    if transport == "ros2":
        transports_to_try = ["ros2"]
    elif transport == "ros1":
        transports_to_try = ["ros1"]
    else:  # auto
        if rclpy is not None:
            transports_to_try.append("ros2")
        transports_to_try.append("ros1")

    last_exc = None
    for idx, mode in enumerate(transports_to_try):
        try:
            if mode == "ros2":
                return _call_robot_command_service_ros2(
                    command_name,
                    service_name=service_name,
                    timeout_sec=timeout_sec,
                )
            return _call_robot_command_service_ros1(
                command_name,
                service_name=service_name,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_exc = exc
            if idx < len(transports_to_try) - 1:
                next_mode = transports_to_try[idx + 1]
                rospy.logwarn(
                    "RobotCommand via ROS %s failed (%s); trying ROS %s instead.",
                    mode.upper(),
                    exc,
                    next_mode.upper(),
                )
            else:
                raise

    if last_exc is not None:  # pragma: no cover - defensive
        raise last_exc

    raise RuntimeError("No transport available for RobotCommand requests.")


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
    ):
        """
        Initializes the ROS node, publishers, subscribers, and action client.
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
        self.move_base_client.wait_for_server(rospy.Duration(5))
        rospy.loginfo("Action server found.")

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
        self._publish_cmd_vel(0, 0)
        if if_p:
            rospy.loginfo("[Base Stop]")

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

    def move_location(self, location, frame_id="map"):
        """
        Moves the robot to a specific location [x, y, theta] using the ROS navigation stack.
        """
        x, y, theta = location
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

        rospy.loginfo(
            f"Sending navigation goal to x={x:.2f}, y={y:.2f}, theta={theta:.2f}"
        )
        self.move_base_client.send_goal(goal)

        # Wait for the robot to reach the goal
        self.move_base_client.wait_for_result()

        state = self.move_base_client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation goal succeeded.")
        else:
            rospy.logwarn(f"Navigation failed with status: {state}")

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

    # ------------------------------------------------------------------ #
    # Robot dog posture control
    # ------------------------------------------------------------------ #
    def send_robot_command(
        self,
        command_name,
        *,
        service_name="/nav/robot_command",
        timeout_sec=5.0,
        transport="auto",
    ):
        """Send a low-level RobotCommand service request via ROS 1/ROS 2.

        Parameters
        ----------
        command_name:
            Name of the command to send to the quadruped.  ``"StandUpDown"``
            toggles between standing and lying postures on the Unitree robot.
        service_name:
            Target service name.  Defaults to ``"/nav/robot_command"`` as used
            in the ROS1/ROS2 bridge configuration.
        timeout_sec:
            Maximum amount of time to wait for the service to respond.
        transport:
            Communication transport to use.  ``"auto"`` (default) first attempts
            ROS 2 if available and falls back to ROS 1, ``"ros1"`` forces the
            ROS 1 service proxy, and ``"ros2"`` forces the ROS 2 client.

        Returns
        -------
        ysc_robot_msgs.srv.RobotCommand.Response
            The response returned by the service.

        Raises
        ------
        TimeoutError
            If the service call exceeds ``timeout_sec`` seconds.
        RuntimeError
            If the call fails or the requested transport is unavailable.
        """

        try:
            return _call_robot_command_service(
                command_name,
                service_name=service_name,
                timeout_sec=timeout_sec,
                transport=transport,
            )
        except TimeoutError as exc:
            rospy.logerr(
                f"Timed out while waiting for robot command '{command_name}': {exc}"
            )
            raise
        except Exception as exc:
            rospy.logerr(f"Failed to call robot command '{command_name}': {exc}")
            raise

    def lie_down(
        self,
        *,
        service_name="/nav/robot_command",
        timeout_sec=5.0,
        transport="auto",
    ):
        """Request the quadruped to assume a lying posture.

        This helper wraps the ``StandUpDown`` command exposed via the
        ``/nav/robot_command`` service, allowing callers to trigger the posture
        change directly from Python whether the service is accessed through
        ROS 2 or a ROS1/ROS2 bridge.

        Parameters
        ----------
        service_name:
            Target service name.  Defaults to ``"/nav/robot_command"``.
        timeout_sec:
            Maximum amount of time to wait for the service to respond.
        transport:
            Transport selector forwarded to :meth:`send_robot_command`.

        Returns
        -------
        ysc_robot_msgs.srv.RobotCommand.Response
            The response returned by the service call.
        """

        response = self.send_robot_command(
            "StandUpDown",
            service_name=service_name,
            timeout_sec=timeout_sec,
            transport=transport,
        )

        # Try to provide informative logging regardless of the specific
        # response schema implemented by the firmware.
        if response is None:
            rospy.loginfo("StandUpDown command sent; no response payload returned.")
        elif hasattr(response, "success"):
            rospy.loginfo(
                "StandUpDown command acknowledged with success=%s.", response.success
            )
        else:
            rospy.loginfo("StandUpDown command response: %s", response)

        return response


if __name__ == "__main__":
    try:
        # Initialize the base controller with fast settings
        base = RosBase(linear_velocity=0.5, angular_velocity=0.8)
        print(base)

        # Example Usage:
        # 1. Keyboard control (uncomment to use)
        # print("\n--- Starting Keyboard Control ---")
        # base.move_keyboard(interval=0.01)

        # 2. Timed movement example
        print("\n--- Testing Timed Movements ---")
        rospy.loginfo("Moving forward for 2 seconds...")
        base.move_T(-2, linear_velocity=0.2)
        rospy.sleep(1)  # Pause
        rospy.loginfo("Rotating right for 1.5 seconds...")
        base.rotate_T(1.5, angular_velocity=0.5)
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
