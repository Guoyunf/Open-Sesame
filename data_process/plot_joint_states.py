#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_joint_states.py
--------------------
读取 ``data_record/<tag>/joint.npy`` 并绘制关节位置和力矩轨迹。
可输入一个或多个目录进行对比展示。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='包含 joint.npy 的目录')
    ap.add_argument('--labels', nargs='*', help='对应目录的标签')
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.paths):
        ap.error('标签数量需与目录数量一致')

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i, path in enumerate(args.paths):
        pos, eff = load_joint(path)
        label = args.labels[i] if args.labels else os.path.basename(path.rstrip('/'))
        plot_joint(pos, eff, label, ax)

    ax[0].set_title('Joint Position')
    ax[1].set_title('Joint Effort')
    ax[0].legend(loc='upper right', fontsize='small')
    ax[1].legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
