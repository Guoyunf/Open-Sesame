import hashlib
import pickle
import numpy as np
import os
import zarr
import gzip
import pandas as pd
from einops import rearrange
from pathlib import Path
from PIL import Image
from copy import deepcopy
from datetime import datetime, timedelta
from glob import glob
from scipy.interpolate import interp1d
from typing import (
    List, 
    Literal, 
    NewType, 
    Optional,
    TypedDict, 
    Tuple
)

Speed = Tuple[float, float, float, float, float, float]
Pose = Tuple[float, float, float, float, float, float] # 6
Joint = Tuple[float, float, float, float, float, float, float] # 7
XYPose = Tuple[float, float]
XYSpeed = Tuple[float, float]
Timestamp = float

class Obs(TypedDict):
    pose: Tuple[np.ndarray, float]
    joint: Tuple[np.ndarray, float]

class LowDimData(TypedDict):
    obs_buffer: List[Obs]
    command_history: List[Tuple[List[float], float]]


def md5(input_string: str) -> str:
    encoded_string = input_string.encode('utf-8')
    md5_hash = hashlib.md5()
    md5_hash.update(encoded_string)
    return md5_hash.hexdigest()

JPGFilePath = NewType('JPGFilePath', str)
PickleFilePath = NewType('PickleFilePath', str)
VideoFilePath = NewType('VideoFilePath', str)
OSPath = NewType('OSPath', str)

class Episode(TypedDict):
    camera_0_paths: List[JPGFilePath]
    poses: List[XYPose]
    speeds: List[XYSpeed]
    timestamps: List[float]
    

def align_and_upsample(
        target_timestamp: List[float], 
        val: np.ndarray, 
        val_t: np.ndarray, 
        threshold: float, 
        fill_with: float = 0.0
    ) -> np.ndarray:
    result: List[np.ndarray] = []
    for t in target_timestamp:
        matching_indices = np.where(np.abs(val_t - t) <= threshold)[0] 
        
        if len(matching_indices) > 0:
            print(abs(val_t[matching_indices] - t))
            result.append(val[matching_indices[0]])
        else:
            print("no avaialble frame for", t)
            result.append(np.ones_like(val[0]) * fill_with)
    return np.asarray(result)


def get_folder_timestamp(x: str) -> float:
    return float(os.path.basename(x).replace('_', '.'))


def get_frame_timestamp(x: str) -> float:
    return float(os.path.basename(x).replace('.jpg', '').split('_')[-1])


def get_episodes_base_folders() -> List[Path]:
    return list(
        glob(os.path.join('/mnt/d/jiajingkai/episode', '*'))
    )


def get_sorted_paths(episode_path: OSPath, pattern: str) -> List[str]:
    return sorted(
        list(
            glob(os.path.join(episode_path, pattern))
        ),
        key=get_folder_timestamp
    )


def get_camera_frame_paths(episode_path: OSPath, camera_id: Literal[0, 1]) -> List[JPGFilePath]:
    return sorted(
        list(
            glob(os.path.join(episode_path, f'realsense_{camera_id}_*.jpg'))
        ),
        key=get_frame_timestamp
    )


def get_gzip_pickle_path(episode_path: OSPath) -> PickleFilePath:
    pickle_file_paths = list(glob(os.path.join(episode_path, "*.pkl.gz")))
    assert len(pickle_file_paths) == 1, f"error when load {episode_path}"
    return pickle_file_paths[0]

def get_pickle_path(episode_path: OSPath) -> PickleFilePath:
    pickle_file_paths = list(glob(os.path.join(episode_path, "*.pickle")))
    assert len(pickle_file_paths) == 1, f"error when load {episode_path}"
    return pickle_file_paths[0]


def get_sample(episode_path: OSPath, use_gzip: bool = True) -> LowDimData:
    if use_gzip is True:
        return load_gzip_pickle(get_gzip_pickle_path(episode_path))
    return load_pickle(get_pickle_path(episode_path))

def load_pickle(pickle_path: PickleFilePath) -> LowDimData:
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def load_gzip_pickle(pickle_path: PickleFilePath) -> LowDimData:
    with gzip.open(pickle_path, 'rb') as f:
        return pickle.load(f)

def get_poses(episode_path: OSPath) -> Tuple[List[XYPose], List[Timestamp]]:
    sample = get_sample(episode_path)
    poses = [
        sample['obs_buffer'][i]['pose'][0] for i in range(len(sample['obs_buffer']))
    ]
    xy_pose =  [
        (pose[0], pose[1]) for pose in poses
    ]
    timestamps = [
        sample['obs_buffer'][i]['pose'][1] for i in range(len(sample['obs_buffer']))
    ] 
    return xy_pose, timestamps


def get_joints(episode_path: OSPath) -> List[Joint]:
    sample = get_sample(episode_path)
    joints = [
        sample['obs_buffer'][i]['joint'][0] for i in range(len(sample['obs_buffer']))
    ]
    joints = [
        (joint[0], joint[1], joint[2], joint[3], joint[4], joint[5]) for joint in joints
    ]
    timestamps = [
        sample['obs_buffer'][i]['joint'][1] for i in range(len(sample['obs_buffer']))
    ] 
    return joints, timestamps


def get_speeds(episode_path: OSPath) -> Tuple[List[XYSpeed], List[Timestamp]]:
    sample = get_sample(episode_path)
    speeds = [
        sample['command_history'][i][0] for i in range(len(sample['command_history']))
    ]
    timestamps = [
        sample['command_history'][i][1] for i in range(len(sample['command_history']))
    ] 
    xy_speeds = [
        (speed[0], speed[1]) for speed in speeds
    ]
    if len(xy_speeds) == 0:
        xy_poses, timestamps = get_poses(episode_path)
        xy_speeds = [
            ((u - x) / t, (v - y) / t) for (x, y), (u, v), t in zip(xy_poses, xy_poses[1:], timestamps)
        ]
        timestamps = timestamps[:-1]

    return xy_speeds, timestamps


def get_cam(episode_path: OSPath, camera_id: Literal[0, 1]) -> Tuple[List[np.ndarray], List[float]]:
    jpg_paths = get_camera_frame_paths(episode_path, camera_id)
    imgs = [
         np.array(Image.open(jpg_path)) for jpg_path in jpg_paths
    ]
    timestamps = [
        get_frame_timestamp(jpg_path) for jpg_path in jpg_paths
    ]
    return (imgs, timestamps)


def linear_interpolate_resampling(
        val: List[float], 
        val_t: List[float], 
        target_t: List[float]
    ) -> List[float]:
        val_t = np.asarray(val_t)
        val = np.asarray(val)
        interpolator = interp1d(val_t, val, kind='linear', fill_value='extrapolate')
        interpolated_values = interpolator(target_t)
        return list(interpolated_values)


def replace_by_prev_resampling(
        val: List[float], 
        val_t: List[float], 
        target_t: List[float]
    ) -> List[float]:
        s = pd.Series(val, index=val_t)
        t = pd.Index(target_t)
        s = s.reindex(t, method='ffill')
        return s.to_list()


def resample_2d(
        vec: List[Tuple[float, float]], 
        vec_t: List[float], 
        target_t: List[float], 
        method=Literal['linear', 'prev']
    ) -> List[Tuple[float, float]]:
    assert len(vec[0]) == 2
    dim_0 = list(map(lambda x: x[0], vec))
    dim_1 = list(map(lambda x: x[1], vec))
    if method == 'linear':
        return list(
            (new_0, new_1) for new_0, new_1 in zip(
                linear_interpolate_resampling(dim_0, vec_t, target_t),
                linear_interpolate_resampling(dim_1, vec_t, target_t)
            )
        )
    if method == 'prev':
        return list(
            (new_0, new_1) for new_0, new_1 in zip(
                replace_by_prev_resampling(dim_0, vec_t, target_t),
                replace_by_prev_resampling(dim_1, vec_t, target_t)
            )
        )
    raise NotImplementedError

def accelerate_episode(episode: Episode, accelerate_rate: float) -> Episode:
    start_time = episode['timestamps'][0]
    accelerated_timestamps = [
        start_time + (t - start_time) / accelerate_rate for t in episode['timestamps']
    ]
    return {
        "poses": episode["poses"],
        "speeds": episode["speeds"],
        "camera_0_paths": episode['camera_0_paths'],
        "timestamps": accelerated_timestamps
    }

def resample_episode(episode: Episode, target_freq_hz: float) -> Episode:
    assert len(episode['camera_0_paths']) == len(episode['poses'])
    assert len(episode['poses']) == len(episode['speeds']), f"{len(episode['speeds'])} != {len(episode['poses'])}"
    assert len(episode['speeds']) == len(episode['timestamps']), f"{len(episode['speeds'])} != {len(episode['timestamps'])}"
    assert 0 < target_freq_hz < 30
    timestamps = episode['timestamps']
    start_t = timestamps[0]
    end_t = timestamps[-1]
    target_timestamps = generate_sample_timestamps(start_t, end_t, 1.0 / target_freq_hz)
    return {
        "poses": resample_2d(episode['poses'], episode['timestamps'], target_timestamps, 'linear'),
        "speeds": resample_2d(episode['speeds'], episode['timestamps'], target_timestamps, 'prev'),
        "camera_0_paths": episode['camera_0_paths'],
        "timestamps": target_timestamps
    }



def get_episode(episode_path: OSPath, freq_hz: float = 10, acc_rate: float = 1.0) -> Episode:
    xy_poses, xy_poses_t = get_poses(episode_path) 
    xy_speeds, xy_speeds_t = get_speeds(episode_path)


    cam_0_paths = get_camera_frame_paths(episode_path, 0)
    cam0_t = [
        get_frame_timestamp(jpg_path) for jpg_path in cam_0_paths
    ]
    xy_poses = resample_2d(xy_poses, xy_poses_t, cam0_t, 'linear')
    xy_speeds = resample_2d(xy_speeds, xy_speeds_t, cam0_t, 'prev')
    episode = {
        "camera_0_paths": cam_0_paths,
        "poses": xy_poses,
        "speeds": xy_speeds,
        "timestamps": cam0_t,
    }
    return accelerate_episode(resample_episode(episode, freq_hz), acc_rate)


def generate_sample_timestamps(start_time: float, end_time: float, interval: float) -> list[float]:
    samples = [start_time]
    while True:
        next_sample = samples[-1] + interval
        if next_sample > end_time:
            break
        samples.append(next_sample)
    return samples




def create_and_get_video_paths(
        frame_paths: List[JPGFilePath], 
        output_dir: OSPath,
        rename: Optional[str] = None
    ) -> int:
    sig = md5(''.join(frame_paths))
    input_file_path: OSPath = f'{sig}.txt'
    if rename is None:
        rename = sig
    output_video_path: VideoFilePath = os.path.join(output_dir, f'{rename}.mp4')
    images_with_timestamps = [
        (frame_path, int(get_frame_timestamp(frame_path))) for frame_path in frame_paths
    ]
    with open(input_file_path, 'w+') as f:
        for i, (image_path, _) in enumerate(images_with_timestamps):
            f.write(f"file '{image_path}'\n")
            #if i < len(durations):
            #    f.write(f"duration {durations[i]}\n")
    command = (
        f'ffmpeg -f concat -safe 0 -i {input_file_path}'
        f' -vsync vfr -pix_fmt yuv420p -y {output_video_path}'
    )
    error_code = os.system(command)
    os.remove(os.path.join(f'{sig}.txt'))
    return error_code


def start(output_dir: OSPath):
    store = zarr.DirectoryStore(os.path.join(output_dir, 'replay_buffer.zarr'))
    root = zarr.open(store, mode='w')
    episode_ends: List[int] = []
    timestamps: List[float] = []
    poses: List[Pose] = []
    actions: List[Pose] = []
    for i, episode_path in enumerate(get_episodes_base_folders()):
        episode = get_episode(episode_path)
        video_parent_path = os.path.join(output_dir, 'videos', str(i))
        Path(video_parent_path).mkdir(parents=True, exist_ok=True)
        error = create_and_get_video_paths(episode['camera_0_paths'], video_parent_path, '0')
        assert error == 0, episode_path
        episode_poses = deepcopy(episode['poses'])
        episode_timestampe = episode['timestamps']
        poses.extend(
            episode_poses[:-1]
        )
        actions.extend(
            episode_poses[1:]
        )
        timestamps.extend(
            episode_timestampe[:-1]
        )
        assert len(poses) == len(actions)
        assert len(actions) == len(timestamps)
        if len(episode_ends) == 0:
            episode_end = len(episode_poses) - 1 
        else:
            episode_end = len(episode_poses) - 1 + episode_ends[-1]
        episode_ends.append(episode_end)
    assert len(set(timestamps)) == len(timestamps)
    data_group = root.create_group('data')
    data_group.create_dataset(
        'action', data=np.array(actions), dtype='float64'
    )
    data_group.create_dataset(
        'robot_pose', data=np.array(poses), dtype='float64'
    )
    data_group.create_dataset(
        'timestamp', data=np.array(timestamps), dtype='float64'
    )
    meta_group = root.create_group('meta')
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends), dtype='int64')
    root = zarr.open(store, mode='r')
    print(root.tree()) 
    print(root['data/timestamp'][:5])
    print(root['data/action'][:5])
    print(root['meta/episode_ends'][:])


def main(output_dir: OSPath):
    start(os.path.join(output_dir, 'push_cube_v2'))


def test():
    get_episode('/mnt/d/jiajingkai/episode/1725912649_39431882')


if __name__ == "__main__":
    start(os.path.join('/mnt/d/zarr', 'ur5_push_block_v6'))
