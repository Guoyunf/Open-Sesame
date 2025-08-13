#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_trajectory_and_video.py  (show gripper pose axes)
彩色+深度同步播放，3-D 轨迹 + 姿态坐标系
"""
import glob, os, numpy as np, cv2, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R         # ⇐ 需要 scipy

DATA_DIR    = 'data_record'
SHOW_DEPTH  = True
SHOW_VIEW2  = True     # 同时显示第二视角
CMAP        = cv2.COLORMAP_JET
AXIS_LEN    = 0.03      # 三轴箭头长度 (m)

# ----- 扫描批次 -----
tags = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not tags:
    raise FileNotFoundError("no recordings found; run recorder first")
for i,t in enumerate(tags):
    print(f"[{i}] {t}")
tag = tags[int(input(f"Select batch 0-{len(tags)-1}: "))]

# ----- 载入位姿 (X Y Z R P Y) -----
pose6 = np.load(os.path.join(DATA_DIR, tag, 'pos.npy'))[:, :6]
xyz = pose6[:, :3]; rpy = pose6[:, 3:]
n_frames = xyz.shape[0]

# ----- 图片路径 -----
color_paths = sorted(glob.glob(os.path.join(DATA_DIR, tag, 'cam1/color/*.png')))
depth_paths = sorted(glob.glob(os.path.join(DATA_DIR, tag, 'cam1/depth/*.png'))) if SHOW_DEPTH else None

if SHOW_VIEW2:
    color_paths2 = sorted(glob.glob(os.path.join(DATA_DIR, tag, 'cam2/color/*.png')))
    depth_paths2 = (
        sorted(glob.glob(os.path.join(DATA_DIR, tag, 'cam2/depth/*.png')))
        if SHOW_DEPTH
        else None
    )
else:
    color_paths2 = depth_paths2 = None

# ---- 读取图像 ----
def read_color(paths, i):
    return cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)

def read_depth(paths, i):
    d16 = cv2.imread(paths[i], -1)
    lo, hi = np.percentile(d16[d16>0], (1, 99)) if np.any(d16) else (0, 1)
    d8  = cv2.convertScaleAbs(np.clip(d16, lo, hi), alpha=255/(hi - lo + 1e-3))
    return cv2.applyColorMap(d8, CMAP)[:, :, ::-1]

# ---- 画布 ----
# 图像视图按行排列时需要更高的画布
fig = plt.figure(figsize=(11, 5*(2 if SHOW_VIEW2 else 1)))
ax3d = fig.add_subplot(121, projection='3d')
axim = fig.add_subplot(122)
axim.axis('off')
fig.suptitle(f"Batch {tag}  |  frames={n_frames}")

# 轨迹边界
mrg=0.02; mins, maxs = xyz.min(0)-mrg, xyz.max(0)+mrg
ax3d.set(xlim=(mins[0],maxs[0]), ylim=(mins[1],maxs[1]), zlim=(mins[2],maxs[2]),
         xlabel='X', ylabel='Y', zlabel='Z')

line_traj, = ax3d.plot([],[],[], lw=1.5, color='tab:blue')

# --- 姿态三轴 (初始化为三条线段) ---
segments = np.zeros((3,2,3))   # 3 axes, 2 points each, 3 coords
axis_colors = ['r','g','b']
coll = Line3DCollection(segments, colors=axis_colors, linewidths=2)
ax3d.add_collection(coll)

# --- 图像 ---
if SHOW_DEPTH:
    rows = [np.hstack((read_color(color_paths,0), read_depth(depth_paths,0)))]
    if SHOW_VIEW2:
        rows.append(np.hstack((read_color(color_paths2,0), read_depth(depth_paths2,0))))
    init_img = np.vstack(rows)
else:
    rows = [read_color(color_paths,0)]
    if SHOW_VIEW2:
        rows.append(read_color(color_paths2,0))
    init_img = np.vstack(rows)
img_disp = axim.imshow(init_img)

# --- Slider & 交互 ---
ax_sld=plt.axes([0.15,0.03,0.7,0.03]); sld=Slider(ax_sld,'Frame',0,n_frames-1,valinit=0,valstep=1)
paused=False
fig.canvas.mpl_connect('button_press_event', lambda e:globals().__setitem__('paused', paused ^ (e.inaxes!=ax_sld)))

def update(frame):
    idx = int(sld.val) if paused else frame
    sld.set_val(idx)

    # 更新轨迹
    line_traj.set_data(xyz[:idx+1,0], xyz[:idx+1,1])
    line_traj.set_3d_properties(xyz[:idx+1,2])

    # 计算姿态旋转矩阵 → 三轴端点
    rot = R.from_euler('xyz', rpy[idx]).as_matrix()
    origin = xyz[idx]
    axes_pts = np.stack([origin,
                         origin + AXIS_LEN*rot[:,0],
                         origin + AXIS_LEN*rot[:,1],
                         origin + AXIS_LEN*rot[:,2]])
    segments = np.array([[origin, axes_pts[1]],
                         [origin, axes_pts[2]],
                         [origin, axes_pts[3]]])
    coll.set_segments(segments)

    # 图像
    if SHOW_DEPTH:
        rows = [np.hstack((read_color(color_paths, idx), read_depth(depth_paths, idx)))]
        if SHOW_VIEW2:
            rows.append(np.hstack((read_color(color_paths2, idx), read_depth(depth_paths2, idx))))
        img_disp.set_data(np.vstack(rows))
    else:
        rows = [read_color(color_paths, idx)]
        if SHOW_VIEW2:
            rows.append(read_color(color_paths2, idx))
        img_disp.set_data(np.vstack(rows))
    return line_traj, coll, img_disp

ani = FuncAnimation(fig, update, frames=n_frames, interval=33, blit=False, repeat=True)

plt.tight_layout(rect=[0,0.05,1,1])
plt.show()
