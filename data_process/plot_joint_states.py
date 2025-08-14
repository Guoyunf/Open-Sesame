#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_joint_states.py
--------------------
读取 ``data_record/<tag>/joint.npy`` 并与参考轨迹比较。
若末端关节差异超过阈值，则提示需要重试。
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_joint(path):
    data = np.load(os.path.join(path, 'joint.npy'))
    n = data.shape[1] // 2
    return data[:, :n], data[:, n:]


def plot_joint(pos, eff, label, ax):
    t = np.arange(pos.shape[0])
    for i in range(pos.shape[1]):
        ax[0].plot(t, pos[:, i], label=f'{label}_J{i+1}')
    for i in range(eff.shape[1]):
        ax[1].plot(t, eff[:, i], label=f'{label}_J{i+1}')
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('effort')
    ax[1].set_xlabel('frame')
    ax[0].legend(loc='upper right', fontsize='small')
    ax[1].legend(loc='upper right', fontsize='small')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference', help='成功示教的目录')
    ap.add_argument('attempt', help='本次尝试的目录')
    ap.add_argument('--threshold', type=float, default=0.1,
                    help='末端关节位置差阈值')
    args = ap.parse_args()

    ref_pos, ref_eff = load_joint(args.reference)
    att_pos, att_eff = load_joint(args.attempt)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plot_joint(ref_pos, ref_eff, 'ref', ax)
    plot_joint(att_pos, att_eff, 'att', ax)
    ax[0].set_title('Joint Position')
    ax[1].set_title('Joint Effort')
    plt.tight_layout()
    plt.show()

    diff = np.abs(ref_pos[-1] - att_pos[-1]).max()
    if diff > args.threshold:
        print(f'差异 {diff:.3f} > 阈值 {args.threshold}, 判定失败, 请重试')
    else:
        print(f'差异 {diff:.3f} <= 阈值 {args.threshold}, 判定成功')


if __name__ == '__main__':
    main()
