#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_to_robot.py
──────────────────
把指定批次的 pos_<tag>.npy 逐点回放到 Kinova 机械臂。
同时复现录制时的夹爪开合。
"""

import glob, sys, os, re, time, argparse, numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm_kinova import Arm

DATA_DIR = 'data_record'
POS_RE   = re.compile(r'pos_(.+?)\.npy$')
HZ       = 10           # 回放频率 (与数据采集一致)
DT       = 1 / HZ

def pick_batch() -> str:
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'pos_*.npy')))
    if not files: raise FileNotFoundError("No pos_*.npy found, record first.")
    tags = [POS_RE.search(os.path.basename(f)).group(1) for f in files]
    print("Available batches:")
    for i, t in enumerate(tags): print(f"[{i}] {t}")
    idx = int(input(f"Select batch 0-{len(tags)-1}: "))
    return tags[idx]

def main(tag: str, dry_run: bool):
    path = os.path.join(DATA_DIR, f'pos_{tag}.npy')
    data = np.load(path)
    pose = data[:, :6]                 # X Y Z R P Y
    grip = data[:, 6] if data.shape[1] > 6 else None
    if grip is None:
        print(f"Loaded {pose.shape[0]} waypoints from {path} (no gripper data)")
    else:
        print(f"Loaded {pose.shape[0]} waypoints from {path} with gripper track")

    if dry_run:
        print("Dry-run only → no motion commands will be sent.")
        for i, p in enumerate(pose):
            line = (f"{i:4d}  x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f}  "
                    f"r={np.degrees(p[3]):.1f}°  p={np.degrees(p[4]):.1f}°  y={np.degrees(p[5]):.1f}°")
            if grip is not None:
                line += f"  g={grip[i]:.1f}"
            print(line)
        return

    arm = Arm.load()                   # 默认 cfg/cfg_arm_left.yaml
    time.sleep(1.0)

    print("Start replay…   (Ctrl+C to abort)")
    try:
        last_grip = None
        for i, p in enumerate(pose):
            t0 = time.time()
            arm.move_abs(p, blocking=False)    # 非阻塞发送，保持节奏
            if grip is not None:
                g_val = grip[i]
                if not np.isnan(g_val) and (last_grip is None or abs(g_val - last_grip) > 1e-3):
                    arm.set_gripper(float(g_val))
                    last_grip = g_val
            dt = time.time() - t0
            if dt < DT:
                time.sleep(DT - dt)
            if i % 50 == 0:
                print(f" sent {i+1}/{len(pose)}")
        print("✓ Replay finished.")
    except KeyboardInterrupt:
        print("\nUser aborted.")
    finally:
        # 可根据需要添加停止指令，例如急停 / 关闭力矩
        pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', help='batch tag to replay (default: interactive pick)')
    ap.add_argument('--dry', action='store_true', help='dry-run, only print waypoints')
    args = ap.parse_args()

    tag = args.tag or pick_batch()
    main(tag, args.dry)
