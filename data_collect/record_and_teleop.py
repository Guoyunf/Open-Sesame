#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_and_teleop.py  ‒  速度控制版
Enter  开始 / 停止一次 10 Hz 录制        Esc  退出
其它实时键：WASDQE · JLIKOU · 方向键（见 arm_kinova.py）
"""

import os, sys, time, threading, termios, tty, select, datetime
import numpy as np, cv2, pyrealsense2 as rs

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm_kinova import Arm, LINEAR, ANG, GRIP, HOME_KEY   # 复用封装

# ───────── 参数 ─────────
# HOME_POSE   = [0.213, -0.256, 0.508, 1.651, 1.115, 0.122]
HOME_POSE   = [0.212, -0.256, 0.506, 1.664, 1.115, -1.694]

LIN_VEL     = 0.8          # m/s  (≈4 cm/s)
ANG_VEL     = 0.30          # rad/s(≈17°/s)
SAVE_DIR    = 'data_record'; os.makedirs(SAVE_DIR, exist_ok=True)
REC_HZ      = 30
PUB_HZ      = 100           # 必须 100 Hz :contentReference[oaicite:1]{index=1}
CAM1_SN, CAM2_SN = "243122075526", "243222073031"

# ---- 非阻塞读取单键（含方向键） ----
def get_key(timeout=0.01):
    fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(fd)
        if not select.select([fd], [], [], timeout)[0]:
            return ''
        ch1 = sys.stdin.read(1)
        if ch1 != '\x1b': return ch1
        if select.select([fd], [], [], 0.001)[0] and sys.stdin.read(1) == '[':
            ch3 = sys.stdin.read(1)
            return '\x1b['+ch3           # ↑↓→←
        return '\x1b'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ---- 摄像头初始化 ----
def cam(sn):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
    pipe.start(cfg); return pipe
pipe1, pipe2 = cam(CAM1_SN), cam(CAM2_SN)

# ---- 10 Hz 记录线程 ----
class Recorder(threading.Thread):
    def __init__(self, tag, arm):
        super().__init__(daemon=True)
        self.tag, self.arm = tag, arm
        self.stop_evt = threading.Event()
        self.pose, self.grip = [], []
        self.root = os.path.join(SAVE_DIR, tag)
        for cam in ('cam1', 'cam2'):
            os.makedirs(os.path.join(self.root, cam, 'color'), exist_ok=True)
            os.makedirs(os.path.join(self.root, cam, 'depth'), exist_ok=True)
    def run(self):
        idx, period = 0, 1/REC_HZ
        while not self.stop_evt.is_set():
            t0 = time.time()
            self.pose.append(self.arm.pose())
            g = self.arm.finger(); self.grip.append([ (g[0]+g[1]) / 2 ])
            f1, f2 = pipe1.wait_for_frames(), pipe2.wait_for_frames()
            cv2.imwrite(os.path.join(self.root, 'cam1', 'color', f'{idx:06d}.png'),
                        np.asarray(f1.get_color_frame().get_data()))
            cv2.imwrite(os.path.join(self.root, 'cam1', 'depth', f'{idx:06d}.png'),
                        np.asarray(f1.get_depth_frame().get_data()))
            cv2.imwrite(os.path.join(self.root, 'cam2', 'color', f'{idx:06d}.png'),
                        np.asarray(f2.get_color_frame().get_data()))
            cv2.imwrite(os.path.join(self.root, 'cam2', 'depth', f'{idx:06d}.png'),
                        np.asarray(f2.get_depth_frame().get_data()))
            idx += 1; time.sleep(max(0, period-(time.time()-t0)))
    def stop(self):
        self.stop_evt.set(); self.join()
        np.save(os.path.join(self.root, 'pos.npy'),
                np.hstack((np.array(self.pose), np.array(self.grip))))
        print(f"[rec] 保存完毕 {self.tag}")

# ---- 主程序 ----
if __name__ == '__main__':
    arm = Arm.load(); time.sleep(1.0)         # 速度模式会自动检测并开启

    vx=vy=vz=wx=wy=wz = 0.0
    rec = None
    print("Enter 开始/结束录制 | Esc 退出 | 空格急停")

    pub_period = 1/PUB_HZ
    last_pub   = time.time()

    while True:
        k = get_key(0.002)
        if k == '\x1b':                 # Esc
            if rec: rec.stop()
            break
        elif k == '\r':                 # Enter
            if rec is None:
                print("...开始复位...", HOME_POSE)
                arm.move_abs(HOME_POSE); time.sleep(2)
                tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                rec = Recorder(tag, arm); rec.start()
                print("⚡ 开始录制…")
            else:
                rec.stop(); rec = None; print("■ 录制结束")
        elif k == ' ':
            vx=vy=vz=wx=wy=wz = 0.0     # 急停
        elif k == HOME_KEY:
            arm.move_abs(HOME_POSE)
        elif k in LINEAR:               # 线速度键
            dx,dy,dz = LINEAR[k]
            vx,vy,vz = np.sign([dx,dy,dz]) * LIN_VEL; wx=wy=wz=0
        elif k in ANG:                  # 角速度键
            dR,dP,dY = ANG[k]
            wx,wy,wz = np.sign([dR,dP,dY]) * ANG_VEL; vx=vy=vz=0
        elif k in GRIP:                 # 夹爪
            val = np.clip(arm.finger()[0] + GRIP[k], 0, arm.g_close); arm.set_gripper(val)
        elif k == '':
            pass
        else:
            print("Pose", np.round(arm.pose(),3), "Grip", arm.finger()[0])

        # ---- 100 Hz 发布速度 ----
        now = time.time()
        if now - last_pub >= pub_period:
            arm.send_velocity(vx,vy,vz,wx,wy,wz)  # PoseVelocity / CartesianVelocity :contentReference[oaicite:2]{index=2}
            last_pub = now
