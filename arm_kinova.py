#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_kinova_plus.py
~~~~~~~~~~~~~~~~~~
Kinova JACO/MICO 机械臂二次封装。
此版本在原始代码基础上进行了增强：
1. 新增直接读取四元数姿态的功能。
2. 修正了初始化后立即读取状态可能为零的问题。

依赖:
    - rospy
    - actionlib
    - kinova_msgs
    - geometry_msgs
    - tf.transformations (在 Noetic 中自带)
"""

import os
import time
import yaml
import csv
import math
import threading
import numpy as np

import rospy
import actionlib
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point, Quaternion, PoseStamped  # 导入 PoseStamped 消息类型
from std_msgs.msg import Header
import kinova_msgs.msg as km

# 尝试从 tf.transformations 导入，这是 ROS1 的标准方式
try:
    from tf.transformations import quaternion_from_euler, euler_from_quaternion
# 如果失败，则尝试从 tf_conversions 导入，这在某些环境下需要
except ImportError:
    from tf_conversions import transformations as tfs
    quaternion_from_euler = tfs.quaternion_from_euler
    euler_from_quaternion = tfs.euler_from_quaternion

def euler_to_quaternion_zyx(roll, pitch, yaw):
    """
    一个标准的转换函数：将欧拉角(Roll, Pitch, Yaw)通过'zyx'内旋顺序转换为四元数。
    
    这个函数不包含任何针对特定机器人数据的重排逻辑，使其通用且可复用。
    
    Args:
        roll (float): 绕X轴的旋转角度（弧度）。
        pitch (float): 绕Y轴的旋转角度（弧度）。
        yaw (float): 绕Z轴的旋转角度（弧度）。
        
    Returns:
        numpy.ndarray: [x, y, z, w] 格式的四元数。
    """
    # 使用'zyx'旋转序列进行对象创建。
    # scikit-learn的 `from_euler` 期望的顺序就是 [roll, pitch, yaw]
    rotation_obj = Rotation.from_euler('zyx', [roll, pitch, yaw], degrees=False)
    
    # .as_quat() 返回 [x, y, z, w] 格式的四元数
    calculated_quaternion = rotation_obj.as_quat()
    
    return calculated_quaternion


# -------------------- ROS-层封装 -------------------- #

def _pose_action_address(prefix: str):
    """获取位姿控制 Action Server 的地址"""
    return f"/{prefix}driver/pose_action/tool_pose"


def _finger_action_address(prefix: str):
    """获取夹爪控制 Action Server 的地址"""
    return f"/{prefix}driver/fingers_action/finger_positions"


def _cartesian_euler_feedback_topic(prefix: str):
    """获取发布欧拉角位姿反馈的主题地址"""
    return f"/{prefix}driver/out/cartesian_command"

def _finger_position(prefix : str):
    """获取夹爪角度"""
    return f"/{prefix}driver/out/finger_position"

def _tool_pose_quaternion_feedback_topic(prefix: str):
    """获取发布四元数位姿反馈的主题地址"""
    return f"/{prefix}driver/out/tool_pose"


def _call_pose_action(prefix: str,
                      position_xyz,
                      quat_xyzw,
                      timeout: float = 30.0):
    """
    阻塞式发送笛卡尔位姿 action goal。
    """
    client = actionlib.SimpleActionClient(
        _pose_action_address(prefix), km.ArmPoseAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        raise RuntimeError("连接手臂 Action Server 失败，请检查 robot_prefix 是否正确、驱动是否已启动")
    goal = km.ArmPoseGoal()
    goal.pose.header = Header(frame_id=f"{prefix}link_base")
    goal.pose.pose.position = Point(*position_xyz)
    # 注意：ROS的Quaternion顺序是 (x,y,z,w)，但quaternion_from_euler返回的是(w,x,y,z)
    # 所以需要确认这里的输入顺序。kinova_msgs/ArmPose需要(x,y,z,w)
    goal.pose.pose.orientation = Quaternion(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])
    client.send_goal(goal)
    ok = client.wait_for_result(rospy.Duration(timeout))
    if not ok:
        client.cancel_all_goals()
        raise TimeoutError("等待手臂移动结束超时")
    return client.get_result()


def _call_finger_action(prefix: str,
                        turn_value: float,
                        timeout: float = 10.0):
    """
    阻塞式发送夹爪 action goal。双指爪 finger1/2 取相同值。
    """
    client = actionlib.SimpleActionClient(
        _finger_action_address(prefix), km.SetFingersPositionAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        raise RuntimeError("连接夹爪 Action Server 失败")
    goal = km.SetFingersPositionGoal()
    goal.fingers.finger1 = float(turn_value)
    goal.fingers.finger2 = float(turn_value)
    goal.fingers.finger3 = 0.0
    client.send_goal(goal)
    ok = client.wait_for_result(rospy.Duration(timeout))
    if not ok:
        client.cancel_all_goals()
        raise TimeoutError("等待夹爪动作结束超时")
    return client.get_result()


# -------------------- 与旧接口兼容的 Arm 类 -------------------- #

class Arm:
    """
    Kinova 机械臂的高级封装，使其看起来像你原来的 Arm 类。
    此版本同时支持读取欧拉角和四元数姿态。
    """

    @classmethod
    def init_from_yaml(cls, cfg_path: str, root_dir: str = "./"):
        """从YAML配置文件快速构造Arm实例"""
        full_cfg_path = os.path.join(root_dir, cfg_path)
        print(f"从 {full_cfg_path} 加载配置...")
        with open(full_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(root_dir=root_dir,
                   robot_prefix=cfg.get("robot_prefix", "j2n6s200_"),
                   cam2base_H_path=cfg.get("cam2base_H_path", "cfg/cam2base_H_left.csv"),
                   gripper_open=cfg.get("gripper_open", 0),
                   gripper_close=cfg.get("gripper_close", 6000))

    def __init__(self,
                 root_dir: str,
                 robot_prefix: str,
                 cam2base_H_path: str,
                 gripper_open: int,
                 gripper_close: int):
        """
        核心初始化方法。
        :param robot_prefix: ROS主题名字前缀, 如 'j2n6s200_'
        :param cam2base_H_path: 相机到基座的4×4齐次变换矩阵CSV文件路径
        """
        # 如果ROS节点尚未初始化，则初始化一个（允许在同一个进程里多实例化）
        if not rospy.get_node_uri():
            rospy.init_node("arm_kinova_wrapper", anonymous=True, disable_signals=True)
            print("ROS 节点 'arm_kinova_wrapper' 已初始化。")

        self.root_dir = root_dir
        self.prefix = robot_prefix
        self.cam2base_H = self._load_csv_matrix(os.path.join(root_dir, cam2base_H_path))
        self.gripper_open_val = gripper_open
        self.gripper_close_val = gripper_close
        
        # 创建一个线程锁来保护共享的姿态数据
        self._pose_lock = threading.Lock()

        # --- 订阅欧拉角姿态 ---
        self._current_pose_euler = [0.0] * 6  # [x, y, z, rx, ry, rz]
        rospy.Subscriber(_cartesian_euler_feedback_topic(self.prefix),
                         km.KinovaPose, self._feedback_euler_cb)
        print(f"已订阅欧拉角姿态主题: {_cartesian_euler_feedback_topic(self.prefix)}")

        # --- 新增：订阅四元数姿态 ---
        self._current_pose_quat = [0.0] * 7  # [x, y, z, qx, qy, qz, qw]
        rospy.Subscriber(_tool_pose_quaternion_feedback_topic(self.prefix),
                         PoseStamped, self._feedback_quat_cb)
        print(f"已订阅四元数姿态主题: {_tool_pose_quaternion_feedback_topic(self.prefix)}")

        # # --- new add  : finger action ---
        self._current_finger_action = [0.0] * 3
        rospy.Subscriber(_finger_position(self.prefix),
                         km.FingerPosition, self._feedback_finger_action)
        print(f"已订阅手指主题: {_finger_position(self.prefix)}")


    @staticmethod
    def _load_csv_matrix(path):
        """从CSV文件加载矩阵"""
        with open(path, newline="") as f:
            return np.array([[float(x) for x in row] for row in csv.reader(f)])

    def _feedback_euler_cb(self, msg: km.KinovaPose):
        """处理欧拉角姿态消息的回调函数"""
        with self._pose_lock:
            self._current_pose_euler = [msg.X, msg.Y, msg.Z,
                                        msg.ThetaX, msg.ThetaY, msg.ThetaZ]

    def _feedback_quat_cb(self, msg: PoseStamped):
        """处理四元数姿态消息的回调函数"""
        with self._pose_lock:
            p = msg.pose.position
            o = msg.pose.orientation
            self._current_pose_quat = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

    def _feedback_finger_action(self, msg : km.FingerPosition):
        """读取finger值"""
        with self._pose_lock:
            self._current_finger_action = [msg.finger1, msg.finger2, msg.finger3]
        
    def get_p(self, if_p=False):
        """
        获取机械臂末端的笛卡尔坐标 (使用欧拉角)。
        :return: [x, y, z, rx, ry, rz] (单位: m, rad)
        """
        with self._pose_lock:
            pose = self._current_pose_euler.copy()
        if if_p:
            # 使用 rospy.loginfo 来打印，这是ROS推荐的做法
            rospy.loginfo(f"[Arm INFO] 当前位姿 (欧拉角): {np.round(pose, 4).tolist()}")
        return pose

    def get_p_quat(self, if_p=False):
        """
        【新增】获取机械臂末端的笛卡尔坐标 (使用四元数)。
        :return: [x, y, z, qx, qy, qz, qw] (单位: m)
        """
        with self._pose_lock:
            pose = self._current_pose_quat.copy()
        if if_p:
            rospy.loginfo(f"[Arm INFO] 当前位姿 (四元数): {np.round(pose, 4).tolist()}")
        return pose

    def target2cam_xyzrpy_to_target2base_xyzrpy(self, xyzrpy_cam):
        """
        将 [x,y,z,rx,ry,rz] (弧度) 从相机坐标系变换到基坐标系。
        """
        cam2base = self.cam2base_H
        t2c_R = _eul2R(xyzrpy_cam[3:])
        t2c_t = np.array(xyzrpy_cam[:3]).reshape(3, 1)
        t2c_H = np.block([[t2c_R, t2c_t], [np.zeros((1, 3)), 1]])
        t2b_H = cam2base @ t2c_H
        xyz, rpy = _H_to_xyzrpy(t2b_H)
        return xyz + rpy

    def move_p(self, pos_rpy, block=True):
        """
        移动到指定的目标位姿。兼容旧接口。
        :param pos_rpy: 目标位姿 [x,y,z,rx,ry,rz] (m, rad)
        """
        if len(pos_rpy) != 6:
            raise ValueError("move_p 需要一个长度为 6 的列表 [x,y,z,rx,ry,rz]")
        position = pos_rpy[:3]
        # 注意: tf.transformations.quaternion_from_euler 的旋转顺序是 'sxyz'
        # roll, pitch, yaw -> x, y, z 轴旋转
        # 返回的是 (x, y, z, w)
        quat = euler_to_quaternion_zyx(pos_rpy[5], pos_rpy[4], pos_rpy[3])
        _call_pose_action(self.prefix, position, quat)
        # _call_pose_action 本身是阻塞的，所以不需要额外的等待
        return 0  # 与旧SDK保持一致, 0表示成功

    def control_gripper(self, open_value):
        """
        控制夹爪。
        :param open_value: 夹爪的开度值 (0~6000 左右, 0=张开, 6000=闭合)
        """
        _call_finger_action(self.prefix, open_value)
        return 0
    
    def gripper_value(self,if_p = False):
        with self._pose_lock:
            finger_pose = self._current_finger_action.copy()
        if if_p:
            # 使用 rospy.loginfo 来打印，这是ROS推荐的做法
            rospy.loginfo(f"[Arm INFO] 当前手指角度: {np.round(finger_pose, 2).tolist()}")
        return finger_pose

# -------------------- 辅助数学函数 -------------------- #

def _eul2R(rpy):
    """欧拉角到旋转矩阵的转换"""
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _H_to_xyzrpy(H):
    """齐次变换矩阵到 [xyz, rpy] 的转换"""
    xyz = H[:3, 3].flatten().tolist()
    R = H[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    return xyz, [rx, ry, rz]


# -------------------- 主程序入口：用于直接运行和测试 -------------------- #

if __name__ == "__main__":
    try:
        # --- 1. 初始化 ---
        # 假设配置文件在当前目录下的 cfg/cfg_arm_left.yaml
        # 如果您的文件在别处，请修改这里的路径
        arm = Arm.init_from_yaml(cfg_path="cfg/cfg_arm_left.yaml")

        # --- 2. 等待状态更新 ---
        # 关键步骤：给予 ROS 订阅者一点时间来接收第一次消息
        # 从而避免 get_p() 返回初始化的 [0,0,0,0,0,0]
        print("\n正在等待从机械臂接收到初始状态...")
        time.sleep(1.0) # 等待1秒，通常足够了

        # --- 3. 读取并打印当前位姿 ---
        print("\n--- 读取当前位姿 ---")
        # 使用老方法，获取欧拉角位姿
        current_pose_euler = arm.get_p(if_p=True)
        print(current_pose_euler)
        # 使用新方法，获取四元数位姿
        current_pose_quat = arm.get_p_quat(if_p=True)
        print(current_pose_quat)

        current_finger_action = arm.gripper_value(if_p=True)
        print(current_finger_action)

        # --- 4. 执行一个简单的动作 ---
        print("\n--- 执行一个简单的动作 ---")
        print("准备让机械臂在 Z 轴上移动 +3cm...")
        target_pose = arm.get_p() # 获取当前位姿作为基础
        if target_pose[2] < 0.01: # 简单的安全检查，防止姿态未正确获取
            print("错误：无法获取有效的当前位姿，取消移动。")
        else:
            target_pose[2] -= 0.0 # Z轴坐标增加3cm
            rospy.loginfo(f"目标位姿: {np.round(target_pose, 4).tolist()}")
            arm.move_p(target_pose)
            print("移动指令已发送。")
            
            # --- 5. 控制夹爪 ---
            print("\n--- 控制夹爪 ---")
            print("闭合夹爪...")
            arm.control_gripper(arm.gripper_close_val)
            time.sleep(2)
            print("张开夹爪...")
            arm.control_gripper(arm.gripper_open_val)
            print("夹爪操作完成。")

        print("\n测试脚本执行完毕。")

    except RuntimeError as e:
        print(f"\n运行时错误: {e}")
        print("请确保:")
        print("1. ROS master (roscore) 正在运行。")
        print("2. Kinova 驱动节点正在运行 (例如: roslaunch kinova_bringup kinova_robot.launch)。")
        print("3. 'robot_prefix' 与您启动的驱动匹配。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")