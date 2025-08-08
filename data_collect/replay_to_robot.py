#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_to_robot.py
──────────────────
把指定批次目录下的 pos.npy 逐点回放到 Kinova 机械臂
"""

import glob, sys, os, time, argparse, numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm_kinova import Arm

DATA_DIR = 'data_record'
HZ       = 30           # 录制时 10 Hz
DT       = 1 / HZ

def pick_batch() -> str:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*/pos.npy')))
    if not files: raise FileNotFoundError("No */pos.npy found, record first.")
    tags = [os.path.basename(os.path.dirname(f)) for f in files]
    print("Available batches:")
    for i, t in enumerate(tags): print(f"[{i}] {t}")
    idx = int(input(f"Select batch 0-{len(tags)-1}: "))
    return tags[idx]

def main(tag: str, dry_run: bool):
    path = os.path.join(DATA_DIR, tag, 'pos.npy')
    traj = np.load(path)[:, :6]        # X Y Z R P Y
    print(f"Loaded {traj.shape[0]} waypoints from {path}")

    if dry_run:
        print("Dry-run only → no motion commands will be sent.")
        for i, p in enumerate(traj):
            print(f"{i:4d}  x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f}  "
                  f"r={np.degrees(p[3]):.1f}°  p={np.degrees(p[4]):.1f}°  y={np.degrees(p[5]):.1f}°")
        return

    arm = Arm.load()                   # 默认 cfg/cfg_arm_left.yaml
    time.sleep(1.0)

    print("Start replay…   (Ctrl+C to abort)")
    try:
        for i, p in enumerate(traj):
            t0 = time.time()
            arm.move_abs(p, blocking=False)    # 非阻塞发送，保持节奏
            dt = time.time() - t0
            if dt < DT: time.sleep(DT - dt)
            if i % 50 == 0:
                print(f" sent {i+1}/{len(traj)}")
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
