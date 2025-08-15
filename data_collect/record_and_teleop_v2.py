#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_and_teleop.py – 速度控制 • 单次触发版
Enter  开始/结束 10 Hz 录制   Esc  退出
"""

import os, sys, time, threading, termios, tty, select, datetime
import numpy as np, cv2, pyrealsense2 as rs

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm_kinova import Arm, LINEAR, ANG, GRIP, HOME_KEY

# ───────── 常量 ─────────
HOME_POSE = [0.213,-0.256,0.508,1.651,1.115,0.122]
LIN_VEL   = 0.04        # m/s
ANG_VEL   = 0.30        # rad/s
PUB_HZ    = 100         # Kinova 官方要求 100 Hz:contentReference[oaicite:1]{index=1}
REC_HZ    = 10
SAVE_DIR  = 'data_record'; os.makedirs(SAVE_DIR, exist_ok=True)
CAM1_SN, CAM2_SN = "243122075526", "243222073031"

# ───────── 键盘读取 ─────────
def get_key(to=0.01):
    fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(fd)
        if not select.select([fd],[],[],to)[0]: return ''
        ch1 = sys.stdin.read(1)
        if ch1!='\x1b': return ch1
        if select.select([fd],[],[],0.001)[0] and sys.stdin.read(1)=='[':
            return '\x1b['+sys.stdin.read(1)
        return '\x1b'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ───────── 相机 ─────────
def init_cam(sn):
    p,c=rs.pipeline(),rs.config()
    c.enable_device(sn)
    c.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
    c.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    p.start(c); return p
pipe1, pipe2 = init_cam(CAM1_SN), init_cam(CAM2_SN)

# ───────── 10 Hz 记录线程 ─────────
class Recorder(threading.Thread):
    def __init__(self, tag, arm):
        super().__init__(daemon=True)
        self.tag, self.arm, self.stop_evt = tag, arm, threading.Event()
        self.pose, self.grip = [], []
        self.joint_pos, self.joint_eff = [], []
        self.root = os.path.join(SAVE_DIR, tag)
        for cam in ('cam1', 'cam2'):
            os.makedirs(os.path.join(self.root, cam, 'color'), exist_ok=True)
            os.makedirs(os.path.join(self.root, cam, 'depth'), exist_ok=True)
    def run(self):
        period = 1/REC_HZ; idx=0
        while not self.stop_evt.is_set():
            t0=time.time()
            self.pose.append(self.arm.pose())
            g=self.arm.finger(); self.grip.append([(g[0]+g[1])/2])
            jp, je = self.arm.joint(); self.joint_pos.append(jp); self.joint_eff.append(je)
            # image
            f1,f2=pipe1.wait_for_frames(),pipe2.wait_for_frames()
            cv2.imwrite(os.path.join(self.root,'cam1','color',f'{idx:06d}.png'),
                        np.asarray(f1.get_color_frame().get_data()))
            cv2.imwrite(os.path.join(self.root,'cam1','depth',f'{idx:06d}.png'),
                        np.asarray(f1.get_depth_frame().get_data()))
            cv2.imwrite(os.path.join(self.root,'cam2','color',f'{idx:06d}.png'),
                        np.asarray(f2.get_color_frame().get_data()))
            cv2.imwrite(os.path.join(self.root,'cam2','depth',f'{idx:06d}.png'),
                        np.asarray(f2.get_depth_frame().get_data()))
            idx+=1; time.sleep(max(0,period-(time.time()-t0)))
    def stop(self):
        self.stop_evt.set(); self.join()
        np.save(os.path.join(self.root, 'pos.npy'),
                np.hstack((np.array(self.pose),np.array(self.grip))))
        if self.joint_pos:
            np.save(os.path.join(self.root, 'joint.npy'),
                    np.hstack((np.array(self.joint_pos), np.array(self.joint_eff))))
        print(f"[rec] 保存完毕 {self.tag}")

# ───────── 主循环 ─────────
if __name__=='__main__':
    arm=Arm.load(); time.sleep(1)

    pub_dt=1/PUB_HZ; last_pub=time.time()
    vx=vy=vz=wx=wy=wz=0.0
    rec=None

    print("Enter 开始/结束录制 • Esc 退出 • 空格急停 • 单击键=瞬时速度")

    while True:
        k=get_key(0.005)

        # ----- 控制逻辑 -----
        if k=='\x1b':                                  # Esc
            if rec: rec.stop(); break
        elif k=='\r':                                  # Enter
            if rec is None:
                print("...开始复位...", HOME_POSE)
                arm.move_abs(HOME_POSE); time.sleep(2)
                tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                rec=Recorder(tag,arm); rec.start();  print("⚡ 开始录制…")
            else:
                rec.stop(); rec=None; print("■ 录制结束")
        elif k==' ':                                   # 急停
            vx=vy=vz=wx=wy=wz=0
        elif k==HOME_KEY:
            print("→ Move to Home"); arm.move_abs(HOME_POSE)
        elif k in LINEAR:                              # 线速度脉冲
            dx,dy,dz=LINEAR[k]; vx,vy,vz=np.sign([dx,dy,dz])*LIN_VEL; wx=wy=wz=0
        elif k in ANG:                                 # 角速度脉冲
            dR,dP,dY=ANG[k]; wx,wy,wz=np.sign([dR,dP,dY])*ANG_VEL; vx=vy=vz=0
        elif k in GRIP:                                # 夹爪
            arm.set_gripper(np.clip(arm.finger()[0]+GRIP[k],0,arm.g_close))
        elif k:                                        # 其它键打印
            print("Pose",np.round(arm.pose(),3),"Grip",arm.finger()[0])

        # ----- 发送一次速度脉冲 -----
        now=time.time()
        if now-last_pub>=pub_dt:
            arm.send_velocity(vx,vy,vz,wx,wy,wz)       # 发布速度:contentReference[oaicite:2]{index=2}
            last_pub=now
            # 只移动一个周期后立即清零
            vx=vy=vz=wx=wy=wz=0.0

    pipe1.stop(); pipe2.stop(); print("退出")
