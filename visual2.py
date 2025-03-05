import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# #这块是可视化一帧的点云信息出来的ply文件
# pcd = o3d.io.read_point_cloud('output_rgb_combined_149.ply')
# o3d.visualization.draw_geometries([pcd])

# file=r"E:\desktop\nuscenes_sam\output_pcd\scene0149_00_coords.ply"
# file=r"dense_ground.pcd"
# ply = o3d.io.read_point_cloud(file)
# o3d.visualization.draw_geometries([ply])  # 可视化

# #原始的颜色
# data_dict = torch.load(r'E:\desktop\nuscenes_sam\scene0354_00_before_seg_dict.pth')  #这个是原始的数据集  这个是模型的输入之一
# scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()
#
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(scene_coord)
#
# o3d.visualization.draw_geometries([pcd])



def visualize_point_cloud_with_groups(pth_file_path):
    # 加载 pth 文件
    data_dict = torch.load(pth_file_path)

    # 提取点云坐标
    scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()  # 点云坐标
    # --- 创建仅坐标的点云对象用于保存 ---
    pcd_coords = o3d.geometry.PointCloud()
    pcd_coords.points = o3d.utility.Vector3dVector(scene_coord)

    # 保存纯坐标的PLY文件
    # if save_path:
    #     # 自动添加后缀如果用户忘记
    #     if not save_path.endswith('.ply'):
    #         save_path += '.ply'
    #     o3d.io.write_point_cloud(save_path, pcd_coords)
    #     print(f"坐标点云已保存至: {save_path}")

    # 提取分组信息
    group_data = torch.tensor(data_dict["group"]).cpu().numpy()  # 分组信息

    # 打印分组信息（唯一分组编号）
    unique_groups = np.unique(group_data)
    # print(f"Unique groups in the data: {unique_groups}")

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_coord)

    # 使用 color map 为每个分组分配颜色
    num_groups = len(unique_groups)  # 获取分组数量
    print("num_groups", num_groups)

    # 使用一个对比度较强的色图（比如 coolwarm）
    color_map = cm.get_cmap("plasma", num_groups)  # 使用 "coolwarm" 色图 viridis  plasma
    norm = Normalize(vmin=np.min(group_data), vmax=np.max(group_data))
    sm = ScalarMappable(cmap=color_map, norm=norm)

    # 将每个点分配对应的颜色
    group_colors = sm.to_rgba(group_data)[:, :3]  # 获取 RGB 颜色，忽略透明度

    # 设置点云的颜色
    pcd.colors = o3d.utility.Vector3dVector(group_colors)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

# 例子：加载并可视化分组后的点云
pth_file = r'E:\desktop\nuscenes_sam\scene0149_00_before_seg_dict.pth'
# save_path = r'E:\desktop\nuscenes_sam\output_pcd\scene0149_00_coords.ply'
visualize_point_cloud_with_groups(pth_file)

