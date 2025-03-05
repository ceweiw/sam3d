import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def cluster_point_cloud(pth_file_path, eps=0.5, min_points=10):
    """
    加载 pth 文件，基于点云坐标进行 DBSCAN 聚类，
    更新数据字典中的 group 信息，并返回更新后的字典。

    参数：
        pth_file_path: 原始 pth 文件路径
        eps: DBSCAN 中邻域搜索的半径参数
        min_points: DBSCAN 中构成簇所需的最小点数
    """
    # 加载数据
    data_dict = torch.load(pth_file_path)
    scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()

    # 构造 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_coord)

    # 使用 DBSCAN 进行聚类
    print("Performing DBSCAN clustering ...")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    print("聚类结果：", np.unique(labels))

    # 更新数据字典中的 group 信息
    data_dict["group"] = labels
    return data_dict


def save_pth(data_dict, save_file_path):
    """
    将更新后的数据字典保存为新的 pth 文件。
    """
    torch.save(data_dict, save_file_path)
    print(f"保存更新后的数据到: {save_file_path}")


def visualize_point_cloud_with_groups(pth_file_path):
    """
    加载 pth 文件，并根据 group 信息为点云分配离散颜色后进行可视化。
    """
    # 加载 pth 文件
    data_dict = torch.load(pth_file_path)

    # 提取点云坐标和分组信息
    scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()
    group_data = torch.tensor(data_dict["group"]).cpu().numpy()

    # 获取所有唯一的分组
    unique_groups = np.unique(group_data)
    print("Unique groups:", unique_groups)

    # 为每个唯一分组分配一个离散颜色
    num_groups = len(unique_groups)
    color_map = cm.get_cmap("plasma", num_groups)
    group_color_dict = {group: color_map(i)[:3] for i, group in enumerate(unique_groups)}

    # 将每个点的分组信息映射到对应的颜色
    group_colors = np.array([group_color_dict[g] for g in group_data])

    # 创建 Open3D 点云对象，并设置点和对应的颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_coord)
    pcd.colors = o3d.utility.Vector3dVector(group_colors)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])


# ---------------------------
# 主流程

# 原始 pth 文件路径
original_pth_file = r'E:\desktop\nuscenes_sam\fixbm\scene0149_00_before_seg_dict_15.pth'
# 保存更新后（聚类后）的 pth 文件路径
save_pth_file = r'E:\desktop\nuscenes_sam\fixbm\scene0149_00_after_cluster.pth'

# 使用 DBSCAN 对点云进行聚类
clustered_data_dict = cluster_point_cloud(original_pth_file, eps=0.7, min_points=10)

# 保存更新后的数据字典
save_pth(clustered_data_dict, save_pth_file)

# 可视化聚类后的点云
visualize_point_cloud_with_groups(save_pth_file)

