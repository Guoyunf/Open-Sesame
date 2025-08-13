# This dockerfile can be configured via --build-arg

ARG ROS_DISTRO=galactic
FROM ros:${ROS_DISTRO}-ros-base-focal

ARG ROS_DISTRO=galactic
ARG ROS2_WS=/root/saturn_ros2
WORKDIR $ROS2_WS

# Maintainer's Information
LABEL maintainer="lujw2@lenovo.com"

COPY docker/ros-archive-keyring.gpg /tmp/
RUN rm /etc/apt/sources.list.d/ros2-snapshots.list && \
    mv /tmp/ros-archive-keyring.gpg /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN sed -i 's/http:\/\/archive.ubuntu.com\//http:\/\/mirrors.aliyun.com\//g' /etc/apt/sources.list && \
    sed -i 's/http:\/\/ports.ubuntu.com\//http:\/\/mirrors.aliyun.com\//g' /etc/apt/sources.list && \
    apt-get update && \
    rosdep install -i --from-path ${ROS2_WS} --rosdistro ${ROS_DISTRO} -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

## copy ros2 pkgs to ws
COPY ./saturn_ros2/ ${ROS2_WS}/src/
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build && \
    rm -rf log build src

RUN echo "source /root/saturn_ros2/install/setup.bash " >> /root/.bashrc
## Set TimeZone
ENV LANG=C.UTF-8
ENV TimeZone=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TimeZone /etc/localtime && echo $TimeZone > /etc/timezone

# ADD docker/ros_entrypoint.sh /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
