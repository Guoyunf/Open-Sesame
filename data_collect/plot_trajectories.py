#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trajectories.py
--------------------
读取 data_record/pos_*.npy → 动画播放 + 最终保存一张静态图
"""

import glob, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------- 读取所有数据 --------
DATA_DIR = 'data_record'
files = sorted(glob.glob(os.path.join(DATA_DIR, 'pos_*.npy')))
if not files:
    raise FileNotFoundError("No pos_*.npy found in data_record/. Run the recorder first.")

trajectories = [np.load(f)[:, :3] for f in files]   # 仅取 X Y Z
max_len      = max(t.shape[0] for t in trajectories)

# -------- 画布设置 --------
fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection='3d')

# 坐标范围
all_xyz = np.vstack(trajectories)
margin  = 0.02
ax.set(
    xlim=(all_xyz[:,0].min()-margin, all_xyz[:,0].max()+margin),
    ylim=(all_xyz[:,1].min()-margin, all_xyz[:,1].max()+margin),
    zlim=(all_xyz[:,2].min()-margin, all_xyz[:,2].max()+margin),
    xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
    title='Kinova End-Effector Trajectories')

# 创建 Line 对象
lines = []
for tr in trajectories:
    ln, = ax.plot([], [], [], lw=2)
    lines.append((ln, tr))

# -------- 动画回调 --------
def update(frame):
    for ln, tr in lines:
        upto = min(frame, tr.shape[0]-1)
        ln.set_data(tr[:upto+1, 0], tr[:upto+1, 1])
        ln.set_3d_properties(tr[:upto+1, 2])
    return [ln for ln, _ in lines]

ani = FuncAnimation(fig, update, frames=max_len, interval=30, blit=True, repeat=True)

# -------- 预先绘制完整轨迹并保存 PNG --------
for ln, tr in lines:                           # 显示整条轨迹
    ln.set_data(tr[:,0], tr[:,1])
    ln.set_3d_properties(tr[:,2])
fig.savefig('trajectories_all.png', dpi=300)
print("Static figure saved to trajectories_all.png")

plt.tight_layout()
plt.show()

# 如需保存 mp4，取消下行注释：
# ani.save('trajectories_all.mp4', fps=30, dpi=200)
