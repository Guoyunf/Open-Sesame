#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_and_teleop.py
────────────────────
Enter     → 开始 / 停止一次 10 Hz 录制
Esc       → 退出程序
实时键位  → 来自 arm_kinova_plus_keyboard.py（WASDQE·JLIKOU·方向键）
"""

import os, sys, time, threading, termios, tty, select, datetime
import numpy as np, cv2, pyrealsense2 as rs

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm_kinova import Arm, LINEAR, ANG, GRIP, HOME_KEY

# -------- 参数 --------
HOME_POSE =  [ 0.213, -0.256,  0.508,  1.651,  1.115,  0.122]

SAVE_DIR = 'data_record'; os.makedirs(SAVE_DIR, exist_ok=True)
LOOP_HZ  = 10
CAM1_SN, CAM2_SN = "243122075526", "243222073031"

# -------- 实用：阻塞读键 --------
def read_key(timeout=0.05):
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if select.select([fd], [], [], timeout)[0]:
            ch1 = sys.stdin.read(1)
            if ch1 != '\x1b': return ch1
            # 可能是 ESC 或方向键
            if select.select([fd], [], [], 0.001)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == '[' and select.select([fd], [], [], 0.001)[0]:
                    ch3 = sys.stdin.read(1)
                    return '\x1b['+ch3        # ↑↓→←
            return '\x1b'
        return ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# -------- 摄像头 --------
def init_cam(sn):
    pipe,c = rs.pipeline(), rs.config()
    c.enable_device(sn)
    c.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
    c.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    pipe.start(c); return pipe
pipe1, pipe2 = init_cam(CAM1_SN), init_cam(CAM2_SN)

# -------- 记录线程 --------
class Recorder(threading.Thread):
    def __init__(self, tag, arm):
        super().__init__(daemon=True)
        self.tag, self.arm = tag, arm
        self.stop_evt = threading.Event()
        self.pose, self.grip = [], []
        self.base = os.path.join(SAVE_DIR, self.tag)
        os.makedirs(os.path.join(self.base, 'color'), exist_ok=True)
        os.makedirs(os.path.join(self.base, 'depth'), exist_ok=True)
    def run(self):
        idx, period = 0, 1/LOOP_HZ
        while not self.stop_evt.is_set():
            t0 = time.time()
            self.pose.append(self.arm.pose())
            g = self.arm.finger(); self.grip.append([(g[0]+g[1])/2])
            # 保存图像
            f1,f2 = pipe1.wait_for_frames(), pipe2.wait_for_frames()
            cv2.imwrite(os.path.join(self.base,'color',f"v1_{idx}.png"),
                        np.asarray(f1.get_color_frame().get_data()))
            cv2.imwrite(os.path.join(self.base,'depth',f"v1_{idx}.png"),
                        np.asarray(f1.get_depth_frame().get_data()))
            cv2.imwrite(os.path.join(self.base,'color',f"v2_{idx}.png"),
                        np.asarray(f2.get_color_frame().get_data()))
            cv2.imwrite(os.path.join(self.base,'depth',f"v2_{idx}.png"),
                        np.asarray(f2.get_depth_frame().get_data()))
            idx += 1
            dt = time.time()-t0
            if dt < period: time.sleep(period-dt)
    def stop(self):
        self.stop_evt.set(); self.join()
        np.save(os.path.join(self.base,'pos.npy'),
                np.hstack((np.array(self.pose), np.array(self.grip))))
        print(f"[rec] 保存完毕 {self.tag}")

# -------- 主程序 --------
if __name__=='__main__':
    arm = Arm.load()          # 默认 cfg/cfg_arm_left.yaml
    time.sleep(1.0)

    rec = None
    print("Enter 开始/结束录制 | Esc 退出 | 其余键实时控制")
    while True:
        k = read_key(0.05)
        if not k: continue
        if k == '\x1b':                # Esc
            if rec: rec.stop()
            break
        if k == '\r':                  # Enter
            if rec is None:
                print("...开始复位...")
                arm.move_abs(HOME_POSE); time.sleep(2)
                tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                rec=Recorder(tag,arm); rec.start()
                print("⚡ 开始录制…")
            else:
                rec.stop(); rec=None
                print("■ 录制结束")
            continue

        # ---- 实时遥控 ----
        if k == HOME_KEY:
            arm.move_abs(HOME_POSE)
        elif k in LINEAR:
            dx,dy,dz = LINEAR[k]; arm.send_delta(dx,dy,dz)
        elif k in ANG:
            dR,dP,dY = ANG[k]; arm.send_delta(dR=dR,dP=dP,dY=dY)
        elif k in GRIP:
            f = np.clip(arm.finger()[0] + GRIP[k], 0, arm.g_close)
            arm.set_gripper(f)
        elif k == ' ':
            print("Pose", np.round(arm.pose(),3), "Grip", arm.finger()[0])

    pipe1.stop(); pipe2.stop()
    print("退出")