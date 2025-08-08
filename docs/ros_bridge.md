# ROS1/ROS2 Bridge Setup

The following commands launch the Kinova robot, start a ROS master and open
the ROS1/ROS2 bridge. Run the blocks in separate terminals as noted.

```bash
conda deactivate
source ~/catkin_ws/devel/setup.bash
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2n6s200
xhost +

# Terminal 1: ROS1 environment
source /opt/ros/noetic/setup.bash
roscore

# Terminal 2: bridge environment
source /opt/ros/noetic/setup.bash
source /opt/ros/galactic/setup.bash
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics
```

ros2 launch saturn_controller saturn_ros2.launch.py
