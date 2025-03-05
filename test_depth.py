import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 激光雷达数据路径
lidar_data_path = 'E:/desktop/nuscene_mini/scene0752_00/lidar/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385092150099.pcd.bin'
# 图像数据路径
image_data_path = 'E:/desktop/nuscene_mini/scene0752_00/color/1.jpg'

# LIDAR_TOP 到 CAM_FRONT 的旋转矩阵和平移向量（已给定）
lidar_to_camera_rotation = np.array([
    [0.99989861, 0.01122026, -0.00876743],
    [-0.00936803, 0.05464516, -0.99846189],
    [-0.0107239, 0.99844279, 0.05474473]
])

lidar_to_camera_translation = np.array([-0.01775806, -3.31183671, 2.73615242])

# 获取相机内参矩阵
def get_camera_intrinsic():
    # 提供的相机内参
    camera_intrinsic = np.array([
        [1252.8131021185304, 0.0, 826.588114781398],
        [0.0, 1252.8131021185304, 469.9846626224581],
        [0.0, 0.0, 1.0]
    ])
    return camera_intrinsic

# 将激光雷达点云投影到图像上
def project_lidar_to_camera(lidar_points, camera_intrinsic, lidar_to_camera_rotation, lidar_to_camera_translation):
    # 将LIDAR点云从车辆坐标系转换到相机坐标系
    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # 同次坐标
    lidar_points_in_camera = lidar_to_camera_rotation @ lidar_points_hom[:, :3].T + lidar_to_camera_translation[:, None]

    # 将LIDAR点云从摄像头坐标系转换到相机图像坐标系
    image_points = camera_intrinsic @ lidar_points_in_camera
    image_points /= image_points[2, :]  # 归一化到平面

    return image_points.T

# 读取激光雷达点云
point_cloud = np.fromfile(lidar_data_path, dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring_number

# 提取坐标
lidar_points = point_cloud[:, :3]  # 提取 X, Y, Z 坐标

# 获取相机内参矩阵
camera_intrinsic = get_camera_intrinsic()

# 将激光雷达点云投影到相机图像平面
image_points = project_lidar_to_camera(lidar_points, camera_intrinsic, lidar_to_camera_rotation, lidar_to_camera_translation)

# 读取图像文件
image = cv2.imread(image_data_path, cv2.IMREAD_COLOR)
height, width, _ = image.shape

# 在同一张图上显示图像和点云
plt.figure(figsize=(10, 6))
plt.imshow(image)

# 在图像上叠加点云
# 投影后的点云坐标 image_points[:, 0] 为 x 坐标， image_points[:, 1] 为 y 坐标
# 使用图像坐标系中正确的坐标绘制点云
plt.scatter(image_points[:, 0], image_points[:, 1], c='red', s=1, alpha=0.5)  # 使用红色点云
plt.title('LiDAR Points Projected onto Camera Image')

# 只调用一次plt.show()来显示图像和点云
plt.show()
