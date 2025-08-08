"""Utilities for preprocessing raw episode data.

This module loads low-dimensional robot state and camera frames from an
episode directory, aligns them on a uniform time base and writes the result
into a replay buffer stored as a Zarr dataset. It is a cleaned and
re-organised version of the original script.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, NewType, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from PIL import Image
from scipy.interpolate import interp1d

JPGFilePath = NewType("JPGFilePath", str)
PickleFilePath = NewType("PickleFilePath", str)
VideoFilePath = NewType("VideoFilePath", str)
OSPath = NewType("OSPath", str)

XYPose = Tuple[float, float]
XYSpeed = Tuple[float, float]
Timestamp = float
Pose = Tuple[float, float, float, float, float, float]


@dataclass
class Episode:
    """Container holding synchronised episode data."""

    camera_0_paths: List[JPGFilePath]
    poses: List[XYPose]
    speeds: List[XYSpeed]
    timestamps: List[Timestamp]


def md5(input_string: str) -> str:
    return hashlib.md5(input_string.encode("utf-8")).hexdigest()


def get_frame_timestamp(path: str) -> float:
    return float(Path(path).stem.split("_")[-1])


def get_episode_folders(base_dir: OSPath) -> List[Path]:
    return sorted(Path(base_dir).glob("*"))


def _load_pickle(path: PickleFilePath) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_gzip_pickle(path: PickleFilePath) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_sample(episode_path: OSPath) -> dict:
    pkl_gz = list(Path(episode_path).glob("*.pkl.gz"))
    if pkl_gz:
        return _load_gzip_pickle(PickleFilePath(str(pkl_gz[0])))
    pkl = list(Path(episode_path).glob("*.pickle"))
    return _load_pickle(PickleFilePath(str(pkl[0])))


def get_poses(episode_path: OSPath) -> Tuple[List[XYPose], List[Timestamp]]:
    sample = load_sample(episode_path)
    poses = [obs["pose"][0] for obs in sample["obs_buffer"]]
    timestamps = [obs["pose"][1] for obs in sample["obs_buffer"]]
    xy = [(p[0], p[1]) for p in poses]
    return xy, timestamps


def get_speeds(episode_path: OSPath) -> Tuple[List[XYSpeed], List[Timestamp]]:
    sample = load_sample(episode_path)
    speeds = [cmd[0] for cmd in sample["command_history"]]
    timestamps = [cmd[1] for cmd in sample["command_history"]]
    xy = [(s[0], s[1]) for s in speeds]
    if not xy:
        poses, ts = get_poses(episode_path)
        xy = [((u - x) / t, (v - y) / t) for (x, y), (u, v), t in zip(poses, poses[1:], ts)]
        timestamps = ts[:-1]
    return xy, timestamps


def get_camera_frame_paths(episode_path: OSPath, camera_id: Literal[0, 1]) -> List[JPGFilePath]:
    files = sorted(
        Path(episode_path).glob(f"realsense_{camera_id}_*.jpg"),
        key=lambda p: get_frame_timestamp(str(p)),
    )
    return [JPGFilePath(str(p)) for p in files]


def linear_interpolate_resampling(val, val_t, target_t):
    val_t = np.asarray(val_t)
    val = np.asarray(val)
    interpolator = interp1d(val_t, val, kind="linear", fill_value="extrapolate")
    return list(interpolator(target_t))


def replace_by_prev_resampling(val, val_t, target_t):
    s = pd.Series(val, index=val_t)
    t = pd.Index(target_t)
    return s.reindex(t, method="ffill").to_list()


def resample_2d(vec, vec_t, target_t, method: Literal["linear", "prev"] = "linear"):
    dim0 = [v[0] for v in vec]
    dim1 = [v[1] for v in vec]
    if method == "linear":
        return list(
            zip(
                linear_interpolate_resampling(dim0, vec_t, target_t),
                linear_interpolate_resampling(dim1, vec_t, target_t),
            )
        )
    if method == "prev":
        return list(
            zip(
                replace_by_prev_resampling(dim0, vec_t, target_t),
                replace_by_prev_resampling(dim1, vec_t, target_t),
            )
        )
    raise NotImplementedError


def generate_sample_timestamps(start_time: float, end_time: float, interval: float) -> List[float]:
    samples = [start_time]
    while (next_sample := samples[-1] + interval) <= end_time:
        samples.append(next_sample)
    return samples


def resample_episode(ep: Episode, target_freq_hz: float, acc_rate: float = 1.0) -> Episode:
    assert 0 < target_freq_hz < 30
    start_t, end_t = ep.timestamps[0], ep.timestamps[-1]
    target_ts = generate_sample_timestamps(start_t, end_t, 1.0 / target_freq_hz)
    poses = resample_2d(ep.poses, ep.timestamps, target_ts, "linear")
    speeds = resample_2d(ep.speeds, ep.timestamps, target_ts, "prev")
    if acc_rate != 1.0:
        base = target_ts[0]
        target_ts = [base + (t - base) / acc_rate for t in target_ts]
    return Episode(ep.camera_0_paths, poses, speeds, target_ts)


def load_episode(episode_path: OSPath, freq_hz: float = 10.0, acc_rate: float = 1.0) -> Episode:
    poses, poses_t = get_poses(episode_path)
    speeds, speeds_t = get_speeds(episode_path)
    cam_paths = get_camera_frame_paths(episode_path, 0)
    cam_ts = [get_frame_timestamp(p) for p in cam_paths]
    poses = resample_2d(poses, poses_t, cam_ts, "linear")
    speeds = resample_2d(speeds, speeds_t, cam_ts, "prev")
    return resample_episode(Episode(cam_paths, poses, speeds, cam_ts), freq_hz, acc_rate)


def create_video(frame_paths: List[JPGFilePath], output_dir: OSPath, rename: Optional[str] = None) -> int:
    sig = md5("".join(frame_paths))
    if rename is None:
        rename = sig
    input_list = Path(f"{sig}.txt")
    output_path = Path(output_dir) / f"{rename}.mp4"
    with input_list.open("w") as f:
        for p in frame_paths:
            f.write(f"file '{p}'\n")
    cmd = (
        f"ffmpeg -f concat -safe 0 -i {input_list} -vsync vfr -pix_fmt yuv420p -y {output_path}"
    )
    error = os.system(cmd)
    input_list.unlink(missing_ok=True)
    return error


def process_dataset(input_dir: OSPath, output_dir: OSPath, freq_hz: float = 10.0, acc_rate: float = 1.0) -> None:
    store = zarr.DirectoryStore(os.path.join(output_dir, "replay_buffer.zarr"))
    root = zarr.open(store, mode="w")
    episode_ends: List[int] = []
    timestamps: List[float] = []
    poses: List[Pose] = []
    actions: List[Pose] = []

    for idx, ep_dir in enumerate(get_episode_folders(input_dir)):
        episode = load_episode(OSPath(str(ep_dir)), freq_hz, acc_rate)
        video_dir = Path(output_dir) / "videos" / str(idx)
        video_dir.mkdir(parents=True, exist_ok=True)
        error = create_video(episode.camera_0_paths, OSPath(str(video_dir)), "0")
        assert error == 0, ep_dir

        ep_poses = episode.poses
        ep_ts = episode.timestamps
        poses.extend(ep_poses[:-1])
        actions.extend(ep_poses[1:])
        timestamps.extend(ep_ts[:-1])
        episode_ends.append(len(poses))

    data_grp = root.create_group("data")
    data_grp.create_dataset("action", data=np.array(actions), dtype="float64")
    data_grp.create_dataset("robot_pose", data=np.array(poses), dtype="float64")
    data_grp.create_dataset("timestamp", data=np.array(timestamps), dtype="float64")
    meta_grp = root.create_group("meta")
    meta_grp.create_dataset("episode_ends", data=np.array(episode_ends), dtype="int64")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw episodes")
    parser.add_argument("input_dir", type=str, help="Directory containing raw episodes")
    parser.add_argument("output_dir", type=str, help="Directory to store processed dataset")
    parser.add_argument("--freq", type=float, default=10.0, help="Target frequency in Hz")
    parser.add_argument("--acc", type=float, default=1.0, help="Acceleration rate")
    args = parser.parse_args()
    process_dataset(OSPath(args.input_dir), OSPath(args.output_dir), args.freq, args.acc)


if __name__ == "__main__":
    main()
