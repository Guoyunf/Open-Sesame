import time
import numpy as np
import cv2
import pyrealsense2 as rs
import os

# 定义保存目录
save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 初始化相机（只使用一个）
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 等待相机稳定
print("Waiting for camera to stabilize...")
time.sleep(2)

# 拍摄和保存图像
for i in range(10):  # 拍10张图像
    try:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
    except RuntimeError:
        print(f"Failed to get frame at time {i}, retrying...")
        time.sleep(1)
        frames = pipeline.wait_for_frames(timeout_ms=5000)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print(f"Empty frame at {i}, skipping...")
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 深度图颜色映射（用于可视化）
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )

    # 保存图像
    cv2.imwrite(os.path.join(save_dir, f"color_{i}.png"), color_image)
    cv2.imwrite(os.path.join(save_dir, f"depth_{i}.png"), depth_image)
    cv2.imwrite(os.path.join(save_dir, f"depth_color_{i}.png"), depth_colormap)

    print(f"Captured image {i+1}/10")

# 停止相机流
pipeline.stop()
print("Test completed. All images saved.")
