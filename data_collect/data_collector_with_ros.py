#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Version: v1
Brief: 使用 Realsense 相机采集图像，同时通过 ROS 获取 Kinova 机械臂姿态并保存。
"""

import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg

# ------------------ 配置区 ------------------
ROBOT_PREFIX = "j2n6s200_"  # 请根据你的机械臂型号修改
DATA_FOLDER = "./data/arm_5/"  # 保存数据的目录
# -------------------------------------------

# 全局变量：当前机械臂位姿 [x, y, z, rx, ry, rz]
# 以及夹爪开合度（记录两指的平均值）
current_cartesian_command = [0.0] * 6
current_gripper = [0.0]


def set_current_cartesian_command(feedback):
    """
    回调函数，用于更新全局变量 current_cartesian_command。
    """
    global current_cartesian_command
    current_cartesian_command[0] = feedback.X
    current_cartesian_command[1] = feedback.Y
    current_cartesian_command[2] = feedback.Z
    current_cartesian_command[3] = feedback.ThetaX
    current_cartesian_command[4] = feedback.ThetaY
    current_cartesian_command[5] = feedback.ThetaZ


def set_current_gripper(feedback):
    """更新夹爪开合度"""
    current_gripper[0] = (feedback.finger1 + feedback.finger2) / 2.0


def init_ros_pose_listener():
    """
    初始化 ROS 订阅，获取 Kinova 当前位姿。
    """
    topic_pose = "/" + ROBOT_PREFIX + "driver/out/cartesian_command"
    rospy.loginfo("正在订阅机械臂位姿主题: " + topic_pose)
    rospy.Subscriber(
        topic_pose, kinova_msgs.msg.KinovaPose, set_current_cartesian_command
    )
    rospy.wait_for_message(topic_pose, kinova_msgs.msg.KinovaPose, timeout=5.0)
    rospy.loginfo("成功接收到初始位姿数据。")

    topic_grip = "/" + ROBOT_PREFIX + "driver/out/finger_position"
    rospy.loginfo("正在订阅夹爪状态主题: " + topic_grip)
    rospy.Subscriber(topic_grip, kinova_msgs.msg.FingerPosition, set_current_gripper)
    rospy.wait_for_message(topic_grip, kinova_msgs.msg.FingerPosition, timeout=5.0)
    rospy.loginfo("成功接收到初始夹爪数据。")


def data_collection(data_folder):
    """
    使用 Realsense 采集图像，并保存图像及对应姿态。
    """

    def callback(frame):
        nonlocal count
        scaling_factor = 1.0
        cv_img = cv2.resize(
            frame,
            None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("Capture_Video", cv_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            pose = current_cartesian_command.copy()
            grip_val = current_gripper[0]
            print(f"[INFO] Saved pose_{count}: {pose}  grip={grip_val:.1f}")

            # 保存姿态和夹爪
            with open(os.path.join(data_folder, "poses.txt"), "a+") as f:
                pose_line = ",".join([str(p) for p in pose] + [str(grip_val)]) + "\n"
                f.write(pose_line)

            # 保存图像
            image_path = os.path.join(data_folder, f"{count}.jpg")
            cv2.imwrite(image_path, cv_img)

            count += 1

    # 初始化 Realsense 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    count = 1
    rospy.loginfo("[INFO] 开始数据采集，按 's' 保存图像和姿态，Ctrl+C 或关闭窗口退出。")
    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            callback(color_image)
    except rospy.ROSInterruptException:
        rospy.loginfo("[INFO] 接收到 Ctrl+C，退出数据采集。")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("data_collector_node")

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    init_ros_pose_listener()
    data_collection(DATA_FOLDER)
