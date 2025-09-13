import time
import numpy as np
import cv2
import pyrealsense2 as rs
import os

# 定义保存目录
save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 相机序列号
CAM1_SN = "243122075526"  # 第三视角
CAM2_SN = "243222073031"  # 第一视角

# 初始化相机-第三视角
pipe1 = rs.pipeline()
cfg1 = rs.config()
cfg1.enable_device(CAM1_SN)
cfg1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
try:
    pipe1.start(cfg1)
except Exception as e:
    print(f"Failed to connect to camera 1 (ID: {CAM1_SN}): {e}")
    raise

# 初始化相机-第一视角
pipe2 = rs.pipeline()
cfg2 = rs.config()
cfg2.enable_device(CAM2_SN)
cfg2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
try:
    pipe2.start(cfg2)
except Exception as e:
    print(f"Failed to connect to camera 2 (ID: {CAM2_SN}): {e}")
    raise

# 等待相机稳定
print("Waiting for cameras to stabilize...")
time.sleep(2)  # 稳定等待时间可调整

# 拍摄和保存图像
for i in range(10):  # 拍10张图像
    # 获取第三视角相机的帧
    try:
        frames1 = pipe1.wait_for_frames(timeout_ms=5000)  # 设置超时时间为5000ms
    except RuntimeError:
        print(f"Failed to get frame from camera 1 at time {i}, retrying...")
        time.sleep(1)
        frames1 = pipe1.wait_for_frames(timeout_ms=5000)  # 尝试重新获取帧
    color_frame1 = frames1.get_color_frame()
    depth_frame1 = frames1.get_depth_frame()
    color_image1 = np.asanyarray(color_frame1.get_data())
    depth_image1 = np.asanyarray(depth_frame1.get_data())

    # 保存第三视角图像
    cv2.imwrite(os.path.join(save_dir, f"color_v1_{i}.png"), color_image1)
    cv2.imwrite(os.path.join(save_dir, f"depth_v1_{i}.png"), depth_image1)

    # 获取第一视角相机的帧
    try:
        frames2 = pipe2.wait_for_frames(timeout_ms=5000)  # 设置超时时间为5000ms
    except RuntimeError:
        print(f"Failed to get frame from camera 2 at time {i}, retrying...")
        time.sleep(1)
        frames2 = pipe2.wait_for_frames(timeout_ms=5000)  # 尝试重新获取帧
    color_frame2 = frames2.get_color_frame()
    depth_frame2 = frames2.get_depth_frame()
    color_image2 = np.asanyarray(color_frame2.get_data())
    depth_image2 = np.asanyarray(depth_frame2.get_data())

    # 保存第一视角图像
    cv2.imwrite(os.path.join(save_dir, f"color_v2_{i}.png"), color_image2)
    cv2.imwrite(os.path.join(save_dir, f"depth_v2_{i}.png"), depth_image2)

    print(f"Captured images {i+1}/10")

# 停止相机流
pipe1.stop()
pipe2.stop()

print("Test completed. All images saved.")
