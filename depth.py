import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# 载入NuScenes数据集
nusc = NuScenes(version='v1.0-mini', dataroot='E:/desktop/nuscene_mini', verbose=True)

# 获取第四个场景
scene = nusc.scene[3]  # 这是示例，您可以修改为您想要的场景索引

# 获取该场景中的所有样本（每个样本包含一个时间戳和多个传感器数据）
sample_tokens = scene['first_sample_token']

# 激光雷达数据路径
lidar_data_dir = 'E:/desktop/nuscene_mini/scene0059_00/lidar'
# 图像数据路径
image_data_dir = 'E:/desktop/nuscene_mini/scene0059_00/color'
# 深度图数据保存路径
depth_data_dir = 'E:/desktop/nuscene_mini/scene0059_00/depth'

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


# 获取样本的LIDAR和图像数据
while sample_tokens:
    sample = nusc.get('sample', sample_tokens)

    # 获取激光雷达（LIDAR_TOP）的数据token
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_filename = os.path.basename(lidar_data['filename'])  # 提取文件名
    print("lidar_filename:", lidar_filename)

    # 获取相应的CAM_FRONT图像数据的token
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cam_filename = os.path.basename(cam_data['filename'])  # 提取文件名
    print("cam_filename:", cam_filename)
    print("\n")

    # 读取图像文件
    image_path = os.path.join(image_data_dir, cam_filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    # 读取激光雷达数据
    lidar_path = os.path.join(lidar_data_dir, lidar_filename)
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring_number

    # 提取坐标
    lidar_points = point_cloud[:, :3]  # 提取 X, Y, Z 坐标

    # 获取相机内参矩阵
    camera_intrinsic = get_camera_intrinsic()

    # 将激光雷达点云投影到相机图像平面
    image_points = project_lidar_to_camera(lidar_points, camera_intrinsic, lidar_to_camera_rotation,
                                           lidar_to_camera_translation)

    # 创建空白深度图（初始化为0）
    depth_image = np.zeros((height, width), dtype=np.float32)

    # 将激光雷达点云投影到相机图像平面
    for pt in image_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < width and 0 <= y < height:
            depth_image[y, x] = pt[2]  # 使用Z坐标作为深度信息

    # 保存深度图
    depth_image_filename = cam_filename.replace('.jpg', '.png')
    depth_image_path = os.path.join(depth_data_dir, depth_image_filename)
    cv2.imwrite(depth_image_path, depth_image)

    print(f"Depth image saved: {depth_image_filename}")

    # 获取下一个样本token
    sample_tokens = sample['next']
