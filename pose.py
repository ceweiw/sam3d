import numpy as np
import json
import os

# 手动指定 CAM_FRONT 到自车坐标系的旋转四元组和位移向量
camera_rotation = [
    0.5077241387638071,
    -0.4973392230703816,
    0.49837167536166627,
    -0.4964832014373754
]
camera_translation = [
    1.72200568478,
    0.00475453292289,
    1.49491291905
]

# 加载 nuscenes 数据集
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='E:/desktop/nuscene_mini')

# 获取第四个场景数据
scenes = nusc.scene
scene = scenes[3]  # 选择第四个场景
first_sample_token = scene['first_sample_token']
current_sample_token = first_sample_token

# 读取 ego_pose 数据
with open('E:/desktop/nuscene_mini/v1.0-mini/ego_pose.json', 'r') as f:
    ego_pose_data = json.load(f)

# 获取第四个场景的所有样本的token
sample_tokens = []
while current_sample_token != '':
    sample = nusc.get('sample', current_sample_token)
    sample_tokens.append(sample['token'])
    current_sample_token = sample['next']


# 获取 quaternion (四元组) 到旋转矩阵的函数
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return R


# 获取 CAM_FRONT 到自车坐标系的转换矩阵
def get_camera_to_vehicle_transform():
    R = quaternion_to_rotation_matrix(camera_rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = camera_translation
    return T


# 获取 ego_pose 转换矩阵
def get_ego_to_global_transform(timestamp):
    ego_pose = next(item for item in ego_pose_data if item['timestamp'] == timestamp)
    rotation = ego_pose['rotation']
    translation = ego_pose['translation']
    R = quaternion_to_rotation_matrix(rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


# 遍历每个样本，计算从 CAM_FRONT 到全局坐标系的转换
output_dir = 'E:/desktop/nuscene_mini/scene0059_00/pose'
os.makedirs(output_dir, exist_ok=True)

for sample_token in sample_tokens:
    sample = nusc.get('sample', sample_token)

    # 获取 CAM_FRONT 的传感器 token 和图像文件名
    cam_front_token = sample['data']['CAM_FRONT']
    cam_front_data = nusc.get('sample_data', cam_front_token)
    cam_front_filename = cam_front_data['filename']

    # 获取 CAM_FRONT 到自车坐标系的转换矩阵
    camera_to_vehicle_transform = get_camera_to_vehicle_transform()

    # 获取自车到全局坐标系的转换矩阵
    timestamp = cam_front_data['timestamp']
    ego_to_global_transform = get_ego_to_global_transform(timestamp)

    # 计算从 CAM_FRONT 到全局坐标系的转换矩阵
    camera_to_global_transform = np.dot(ego_to_global_transform, camera_to_vehicle_transform)

    # 保存转换矩阵为 txt 文件，文件名与 CAM_FRONT 图像文件名一致
    txt_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(cam_front_filename))[0] + '.txt')
    with open(txt_filename, 'w') as f:
        for row in camera_to_global_transform:
            f.write(' '.join([str(val) for val in row]) + '\n')

    print(f"Saved transformation matrix for {cam_front_filename}")

