# ROS1/ROS2 Bridge Setup

The following commands launch the Kinova robot, start a ROS master and open
the ROS1/ROS2 bridge. Run the blocks in separate terminals as noted.


## in host
```bash
xhost +
```

## in kinova_noetic_intersense docker
```bash
# Terminal 1: ROS1 environmentx
# in base
source ~/catkin_ws/devel/setup.bash
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2n6s200
# Terminal 2: bridge environment
source /opt/ros/noetic/setup.bash
source /opt/ros/galactic/setup.bash
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics
```

## in saturn_ros2 docker
### start background service
```bash
(base) linux@linux-Victus-by-HP-Gaming-Laptop-16-r0xxx:~$ docker start saturn_ros2 
saturn_ros2
(base) linux@linux-Victus-by-HP-Gaming-Laptop-16-r0xxx:~$ docker exec -it saturn_ros2 

(base) linux@linux-Victus-by-HP-Gaming-Laptop-16-r0xxx:~$ ros2 launch saturn_controller saturn_ros2.launch.py
```