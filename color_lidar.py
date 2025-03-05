import os
import shutil
from nuscenes.nuscenes import NuScenes

# 加载数据集
nusc = NuScenes(version='v1.0-mini', dataroot='E:/desktop/nuscene_mini')

# 获取场景中的所有样本
scenes = nusc.scene
scene = scenes[3]  # 假设我们选择第一个场景
first_sample_token = scene['first_sample_token']
current_sample_token = first_sample_token
print("first_sample_token",first_sample_token)
sample_tokens = []
while current_sample_token != '':
    sample = nusc.get('sample', current_sample_token)
    sample_tokens.append(sample['token'])
    current_sample_token = sample['next']

# 目标路径
point_cloud_target_folder = r'E:\desktop\nuscene_mini\scene0059_00\lidar'
cam_front_target_folder = r'E:\desktop\nuscene_mini\scene0059_00\color'

# 确保目标文件夹存在，如果不存在则创建
os.makedirs(point_cloud_target_folder, exist_ok=True)
os.makedirs(cam_front_target_folder, exist_ok=True)

# 提取点云和CAM_FRONT图像路径，并执行复制操作
point_cloud_paths = []
cam_front_paths = []

for sample_token in sample_tokens:
    sample = nusc.get('sample', sample_token)

    # 获取该样本的所有传感器数据
    for sensor, data_token in sample['data'].items():
        # 获取点云数据 (LIDAR_TOP)
        if sensor == 'LIDAR_TOP':
            sample_data = nusc.get('sample_data', data_token)
            point_cloud_path = nusc.get_sample_data_path(data_token)
            point_cloud_paths.append(point_cloud_path)

        # 获取前置摄像头图像数据 (CAM_FRONT)
        elif sensor == 'CAM_FRONT':
            sample_data = nusc.get('sample_data', data_token)
            cam_front_path = nusc.get_sample_data_path(data_token)
            cam_front_paths.append(cam_front_path)

# 复制点云文件到目标文件夹
for pc_path in point_cloud_paths:
    # 获取文件名
    filename = os.path.basename(pc_path)
    # 目标路径
    target_path = os.path.join(point_cloud_target_folder, filename)
    # 执行复制
    shutil.copy(pc_path, target_path)
    print(f"复制点云文件: {pc_path} 到 {target_path}")

# 复制CAM_FRONT图像文件到目标文件夹
for img_path in cam_front_paths:
    # 获取文件名
    filename = os.path.basename(img_path)
    # 目标路径
    target_path = os.path.join(cam_front_target_folder, filename)
    # 执行复制
    shutil.copy(img_path, target_path)
    print(f"复制图像文件: {img_path} 到 {target_path}")
