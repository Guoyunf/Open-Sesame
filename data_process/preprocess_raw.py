#!/usr/bin/env python3
"""Preprocess recorded episodes into a replay buffer.

The recorder scripts store each episode under ``data_record/<tag>`` with the
following structure::

    pos.npy              # (N,7) array of [x,y,z,r,p,y,grip]
    cam1/color/000000.png
    cam1/depth/000000.png
    cam2/color/000000.png
    cam2/depth/000000.png

This script collects the pose data and camera-0 colour frames for each
episode, writes an ``mp4`` video for inspection and stores all robot poses and
next-step poses (as actions) in a Zarr replay buffer.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import zarr


@dataclass
class Episode:
    """Simple container for an episode."""

    frame_paths: List[str]
    poses: np.ndarray  # shape (N,6)


def create_video(frame_paths: List[str], out_dir: Path) -> None:
    """Create an ``mp4`` video from a list of frame paths."""

    sig = hashlib.md5("".join(frame_paths).encode()).hexdigest()
    list_file = Path(f"{sig}.txt")
    with list_file.open("w") as f:
        for p in frame_paths:
            f.write(f"file '{p}'\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "0.mp4"
    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vsync",
        "vfr",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)


def load_episode(ep_dir: Path) -> Episode:
    """Load pose array and frame paths from an episode directory."""

    poses = np.load(ep_dir / "pos.npy")[:, :6]
    frame_paths = sorted(glob.glob(str(ep_dir / "cam1/color/*.png")))
    n = min(len(frame_paths), len(poses))
    return Episode(frame_paths[:n], poses[:n])


def process_dataset(input_dir: Path, output_dir: Path) -> None:
    """Convert all episodes into a Zarr replay buffer and videos."""

    store = zarr.DirectoryStore(str(output_dir / "replay_buffer.zarr"))
    root = zarr.open(store, mode="w")
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")

    poses: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    timestamps: List[int] = []
    episode_ends: List[int] = []
    t = 0

    for idx, ep in enumerate(sorted(p for p in Path(input_dir).iterdir() if p.is_dir())):
        episode = load_episode(ep)
        create_video(episode.frame_paths, output_dir / "videos" / str(idx))

        poses.extend(episode.poses[:-1])
        actions.extend(episode.poses[1:])
        timestamps.extend(range(t, t + len(episode.poses) - 1))
        t += len(episode.poses) - 1
        episode_ends.append(len(poses))

    data_grp.create_dataset("action", data=np.array(actions), dtype="float64")
    data_grp.create_dataset("robot_pose", data=np.array(poses), dtype="float64")
    data_grp.create_dataset("timestamp", data=np.array(timestamps), dtype="float64")
    meta_grp.create_dataset("episode_ends", data=np.array(episode_ends), dtype="int64")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess recorded episodes")
    parser.add_argument("input_dir", type=str, help="Directory containing episode folders")
    parser.add_argument("output_dir", type=str, help="Directory to store processed dataset")
    args = parser.parse_args()
    process_dataset(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

