#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
arm_kinova_plus_keyboard.py  ‒  Kinova Gen2/Gen3 Keyboard Tele-op
键位对应 (见 UI 截图)：

╭─ Translation (1 cm) ──────────╮   ╭─ Orientation (5°) ─────────╮   ╭─ Gripper ───────╮
│  D : X-                       │   │  J : Roll +                 │   │  ↑ : Open 2f     │
│  A : X+                       │   │  L : Roll −                 │   │  ↓ : Close 2f    │
│  S : Y-                       │   │  K : Pitch +                │   │  → : Open 3f     │
│  W : Y+                       │   │  I : Pitch −                │   │  ← : Close 3f    │
│  E : Z-                       │   │  O : Yaw +                  │   ╰──────────────────╯
│  Q : Z+                       │   │  U : Yaw −                  │
╰───────────────────────────────╯   ╰─────────────────────────────╯

Others :  H = Home | space 打印位姿 | ESC 退出
"""

import sys, os, math, time, select, termios, tty, threading, yaml, rospy, actionlib, csv
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import JointState

# ─────────────────── 速度消息类型自动选择 ────────────────────
_vel_cls, _KINOVA_FIELDS = None, None
try:
    import kinova_msgs.msg as km
    _vel_cls = km.PoseVelocity
    _KINOVA_FIELDS = ('twist_linear_x','twist_linear_y','twist_linear_z',
                      'twist_angular_x','twist_angular_y','twist_angular_z')
except Exception:
    try:
        _vel_cls = km.CartesianVelocity
        _KINOVA_FIELDS = ('twist_linear_x','twist_linear_y','twist_linear_z',
                          'twist_angular_x','twist_angular_y','twist_angular_z')
    except Exception:
        from geometry_msgs.msg import Twist as _vel_cls
        _KINOVA_FIELDS = None
        print('[WARN] 未编译 kinova_msgs，降级 geometry_msgs/Twist')

# ─────────────────── 参数 ────────────────────
INCR_POS   = 0.01               # m
INCR_ROT   = math.radians(5)    # rad
GRIP_STEP  = 200
LOOP_HZ    = 100
HOLD_CYCLES= 5
FORCE_NO_VEL = False

HOME_POSE = [ 0.213, -0.256,  0.508,  1.651,  1.115,  0.122]
# ─────────────────── Topic 工具 ────────────────────
_ns = lambda p: f"/{p}driver"
_pose_action   = lambda p: f"{_ns(p)}/pose_action/tool_pose"
_finger_action = lambda p: f"{_ns(p)}/fingers_action/finger_positions"
_euler_fb      = lambda p: f"{_ns(p)}/out/cartesian_command"
_quat_fb       = lambda p: f"{_ns(p)}/out/tool_pose"
_finger_fb     = lambda p: f"{_ns(p)}/out/finger_position"
_vel_topic     = lambda p: f"{_ns(p)}/in/cartesian_velocity"
_joint_topic   = lambda p: f"{_ns(p)}/out/joint_state"

from utils.lib_math import euler_to_quaternion_zyx

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


# ─────────────────── Arm 封装 ────────────────────
class Arm:
    @classmethod
    def load(cls, cfg_path='cfg/cfg_arm_left.yaml'):
        cfg = yaml.safe_load(open(cfg_path,'r'))
        return cls(cfg.get('robot_prefix','j2n6s200_'),
                   cfg.get('gripper_open',0),
                   cfg.get('gripper_close',6000))

    @classmethod
    def init_from_yaml(cls, cfg_path='cfg/cfg_arm_left.yaml'):
        return cls.load(cfg_path)
    
    def __init__(self, prefix, g_open, g_close):
        if not rospy.get_node_uri():
            rospy.init_node('arm_keyboard', anonymous=True, disable_signals=True)
        self.prefix, self.g_open, self.g_close = prefix, g_open, g_close
        self.cam2base_H = self._load_csv_matrix("cfg/cam2base_H.csv")
        self._pose=[0.]*6; self._finger=[0.]*3
        self._joint_pos=[]; self._joint_eff=[]
        self._lock=threading.Lock()

        import kinova_msgs.msg as km
        rospy.Subscriber(_euler_fb(prefix), km.KinovaPose, self._cb_pose)
        rospy.Subscriber(_finger_fb(prefix), km.FingerPosition, self._cb_finger)
        rospy.Subscriber(_quat_fb(prefix), PoseStamped, lambda _:None)
        rospy.Subscriber(_joint_topic(prefix), JointState, self._cb_joint)

        self._pose_cli = actionlib.SimpleActionClient(_pose_action(prefix), km.ArmPoseAction)
        self._grip_cli = actionlib.SimpleActionClient(_finger_action(prefix), km.SetFingersPositionAction)
        self._pose_cli.wait_for_server(); self._grip_cli.wait_for_server()

        self._vel_pub=None; self._vel_mode=False
        if not FORCE_NO_VEL and _vel_cls:
            self._vel_pub = rospy.Publisher(_vel_topic(prefix), _vel_cls, queue_size=1)
            self._vel_mode = True
            rospy.loginfo(f"[arm] 速度模式 ON {_vel_topic(prefix)} [{_vel_cls.__name__}]")
        else:
            rospy.logwarn("[arm] 速度模式 OFF，回退增量目标")

    def _cb_pose(self, m):
        with self._lock:
            self._pose=[m.X,m.Y,m.Z,m.ThetaX,m.ThetaY,m.ThetaZ]
    def _cb_finger(self, m):
        with self._lock:
            self._finger=[m.finger1,m.finger2,m.finger3]
    def _cb_joint(self, m):
        with self._lock:
            self._joint_pos=list(m.position)
            self._joint_eff=list(m.effort)

    # 基本读接口
    def pose(self):  
        with self._lock: return self._pose.copy()
    def finger(self):
        with self._lock: return self._finger.copy()
    def joint(self):
        with self._lock: return self._joint_pos.copy(), self._joint_eff.copy()

    # 增量位姿
    def send_delta(self, dx=0,dy=0,dz=0, dR=0,dP=0,dY=0):
        import kinova_msgs.msg as km
        p=self.pose()
        tgt=[p[0]+dx,p[1]+dy,p[2]+dz,p[3]+dR,p[4]+dP,p[5]+dY]
        q=euler_to_quaternion_zyx(tgt[5],tgt[4],tgt[3])
        goal=km.ArmPoseGoal()
        goal.pose.header.frame_id=f"{self.prefix}link_base"
        goal.pose.pose.position=Point(*tgt[:3])
        goal.pose.pose.orientation=Quaternion(*q)
        self._pose_cli.send_goal(goal)

    def move_abs(self,xyzrpy,blocking=True,timeout=15.0):
        import kinova_msgs.msg as km
        q=euler_to_quaternion_zyx(xyzrpy[5],xyzrpy[4],xyzrpy[3])
        goal=km.ArmPoseGoal()
        goal.pose.header.frame_id=f"{self.prefix}link_base"
        goal.pose.pose.position=Point(*xyzrpy[:3])
        goal.pose.pose.orientation=Quaternion(*q)
        self._pose_cli.send_goal(goal)
        if blocking: self._pose_cli.wait_for_result(rospy.Duration(timeout))

    def move_p(self, xyzrpy):
        self.move_abs(xyzrpy)

    # 速度发布
    def send_velocity(self,vx,vy,vz,wx,wy,wz):
        if not self._vel_mode: return
        msg=_vel_cls()
        if _KINOVA_FIELDS:
            for fld,val in zip(_KINOVA_FIELDS,(vx,vy,vz,wx,wy,wz)):
                setattr(msg,fld,val)
        else:
            msg.linear.x, msg.linear.y, msg.linear.z = vx,vy,vz
            msg.angular.x, msg.angular.y, msg.angular.z = wx,wy,wz
        self._vel_pub.publish(msg)

    # 夹爪
    def set_gripper(self,val):
        import kinova_msgs.msg as km
        goal = km.SetFingersPositionGoal()
        goal.fingers.finger1 = goal.fingers.finger2 = val
        goal.fingers.finger3 = 0.0
        self._grip_cli.send_goal(goal)

    def control_gripper(self, open_value):
        self.set_gripper(open_value)

    def open_gripper(self):
        """Open the gripper to its configured open position."""
        self.set_gripper(self.g_open)

    def close_gripper(self):
        """Close the gripper to its configured closed position."""
        self.set_gripper(self.g_close)

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

    @staticmethod
    def _load_csv_matrix(path):
        """从CSV文件加载矩阵"""
        with open(path, newline="") as f:
            return np.array([[float(x) for x in row] for row in csv.reader(f)])

def _get_key():
    fd=sys.stdin.fileno(); old=termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch != '\x1b':
            return ch
        # 可能是 ESC 或方向键
        ch2 = sys.stdin.read(1)
        if ch2 != '[':
            return '\x1b'
        ch3 = sys.stdin.read(1)
        return '\x1b[' + ch3
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ─────────────────── 键位映射 ───────────────────
LINEAR = { 'd':(-INCR_POS,0,0), 'a':(+INCR_POS,0,0),
           's':(0,-INCR_POS,0), 'w':(0,+INCR_POS,0),
           'e':(0,0,-INCR_POS), 'q':(0,0,+INCR_POS)}
ANG    = { 'j':(+INCR_ROT,0,0), 'l':(-INCR_ROT,0,0),
           'k':(0,+INCR_ROT,0), 'i':(0,-INCR_ROT,0),
           'o':(0,0,+INCR_ROT), 'u':(0,0,-INCR_ROT)}
GRIP   = { '[':-GRIP_STEP, ']':+GRIP_STEP,   # 若用方向键则改 ANSI 识别
           '\x1b[A':+GRIP_STEP, '\x1b[B':-GRIP_STEP,  # ↑ ↓
           '\x1b[C':+GRIP_STEP, '\x1b[D':-GRIP_STEP } # → ←
HOME_KEY = 'h'

EXIT_KEY = '\x1b'

HELP_STR = __doc__.split('键位对应')[1].strip()

# ─────────────────── Tele-op 循环 ───────────────────
def teleop(arm:Arm):
    print(HELP_STR)
    rate = rospy.Rate(LOOP_HZ)
    fpos = arm.finger()[0]
    vx=vy=vz=wx=wy=wz=0.0; hold=0
    while not rospy.is_shutdown():
        k=_get_key()
        if not k: pass
        elif k == EXIT_KEY: break
        elif k==HOME_KEY:
            arm.move_abs(HOME_POSE)   # 简易回 Home
        elif k in LINEAR:
            dx,dy,dz=LINEAR[k]
            if arm._vel_mode:
                vx,vy,vz = dx*LOOP_HZ,dy*LOOP_HZ,dz*LOOP_HZ; hold=HOLD_CYCLES
            else: arm.send_delta(dx=dx,dy=dy,dz=dz)
        elif k in ANG:
            dR,dP,dY=ANG[k]
            if arm._vel_mode:
                wx,wy,wz = dR*LOOP_HZ,dP*LOOP_HZ,dY*LOOP_HZ; hold=HOLD_CYCLES
            else: arm.send_delta(dR=dR,dP=dP,dY=dY)
        elif k in GRIP:
            fpos=np.clip(fpos+GRIP[k],0,arm.g_close); arm.set_gripper(fpos)
        elif k==' ':
            print("Pose:",np.round(arm.pose(),3),"Grip:",arm.finger()[0])

        if arm._vel_mode:
            arm.send_velocity(vx,vy,vz,wx,wy,wz)
            if hold: hold-=1
            else: vx=vy=vz=wx=wy=wz=0.0
        rate.sleep()

# ─────────────────── main ───────────────────
if __name__ == '__main__':
    import argparse, signal
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='cfg/cfg_arm_left.yaml',
                    help='YAML config (default: cfg/cfg_arm_left.yaml)')
    ap.add_argument('--no-vel', action='store_true', help='禁用速度模式')
    args = ap.parse_args()
    FORCE_NO_VEL = args.no_vel

    arm = Arm.load(args.cfg)
    rospy.loginfo("等待姿态同步…"); time.sleep(1)
    teleop(arm)
