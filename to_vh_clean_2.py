# import numpy as np
# from plyfile import PlyData, PlyElement
# from scipy.spatial import Delaunay
# import open3d as o3d
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
#
# # 加载 NuScenes 数据集
# nusc = NuScenes(version='v1.0-trainval', dataroot='/path/to/nuscenes', verbose=True)
#
# # 获取场景token
# scene_token = nusc.scene[0]['token']
# scene = nusc.get('scene', scene_token)
#
# # 获取场景的第一个样本
# first_sample_token = scene['first_sample_token']
# sample = nusc.get('sample', first_sample_token)
#
# # 获取该样本的 LiDAR 数据
# lidar_data = [s for s in nusc.sample_data if s['sensor_modality'] == 'lidar' and s['sample_token'] == sample['token']]
# lidar_point_cloud = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_data[0]['token']))
#
# # 提取 LiDAR 点云的 x, y, z 坐标
# points = lidar_point_cloud.points.T  # 转置
# xyz_points = points[:, :3]  # 只取 xyz 坐标
#
# # 使用 x, y 坐标进行 Delaunay 三角剖分生成面信息
# xy_points = xyz_points[:, :2]  # 只使用 x, y 坐标进行三角剖分
# triangulation = Delaunay(xy_points)  # 生成三角网格
#
# # 生成面信息：每个面是三角形的三个顶点索引
# faces = triangulation.simplices  # 返回三角形顶点的索引数组
#
# # 创建一个包含点云数据的PlyElement
# vertices = np.array([(x, y, z) for x, y, z in xyz_points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#
# # 创建面信息
# faces_element = np.array([(i, j, k) for i, j, k in faces], dtype=[('vertex_indices', 'i4', (3,))])
#
# # 创建 ply 文件数据结构
# vertex_element = PlyElement.describe(vertices, 'vertex')
# faces_element = PlyElement.describe(faces_element, 'face')
#
# # 保存 ply 文件
# ply_data = PlyData([vertex_element, faces_element])
# ply_data.write('output_scene_with_faces.ply')
#
# # 使用 Open3D 可视化点云
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz_points)
# o3d.visualization.draw_geometries([pcd])


#
# import numpy as np
#
# # 加载 npz 文件
# data = np.load('E:/desktop/nuScenes-panoptic-v1.0-mini/nuScenes-panoptic-v1.0-mini/panoptic/v1.0-mini/0ab9ec2730894df2b48df70d0d2e84a9_panoptic.npz')
# array1 = data['data']
# # 查看文件内容（所有数组的名字）
# np.savetxt('array1_output.txt', array1, fmt='%s')  # 使用适当的格式（如 %s）来保存
# print(data.files)
# print(array1)


# import os
# import cv2
#
# # 设置depth文件夹路径
# depth_folder = r"E:\desktop\nuscene_mini\scene0752_00\depth"
#
# # 获取该文件夹下所有的png文件
# depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.png')]
#
# # 遍历所有的png文件，读取并打印其(height, width)
# for depth_file in depth_files:
#     # 拼接完整的文件路径
#     depth_file_path = os.path.join(depth_folder, depth_file)
#
#     # 读取深度图像
#     depth_img = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
#
#     # 获取图像的高度和宽度
#     height, width = depth_img.shape
#
#     # 打印出文件名和对应的(height, width)
#     print(f"File: {depth_file}, Height: {height}, Width: {width}")

import os
import cv2

# 设置路径
color_dir = r'E:\desktop\nuscene_mini\scene0752_00\color'

# 获取所有 .jpg 文件
jpg_files = [f for f in os.listdir(color_dir) if f.endswith('.jpg')]

# 遍历所有 .jpg 文件并打印其尺寸
for jpg_file in jpg_files:
    # 获取完整的文件路径
    jpg_path = os.path.join(color_dir, jpg_file)

    # 读取图像
    image = cv2.imread(jpg_path)

    # 获取图像的尺寸
    height, width, _ = image.shape

    # 打印图像文件名及其尺寸
    print(f"File: {jpg_file}, Height: {height}, Width: {width}")

