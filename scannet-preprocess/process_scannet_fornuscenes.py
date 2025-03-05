from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import CameraToken

# 加载数据集
nusc = NuScenes(version='v1.0-mini', dataroot='path_to_your_nuscenes_data')
# 获取所有场景的列表
scenes = nusc.scene

# 假设你选择了第一个场景
scene = scenes[0]
# 获取场景中的第一个样本
sample = nusc.get('sample', scene['first_sample_token'])

# 获取样本的所有数据（摄像头、点云等）
sample_data_token = sample['data']['CAM_FRONT']  # CAM_FRONT 是前置摄像头的标识

# 获取摄像头图像信息
sample_data = nusc.get('sample_data', sample_data_token)

# 获取图像文件路径
image_path = nusc.get_sample_data_path(sample_data_token)
