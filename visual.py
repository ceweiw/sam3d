# import torch
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
#
#
# # #这块是可视化一帧的点云信息出来的ply文件
# # pcd = o3d.io.read_point_cloud('output_rgb_combined_149.ply')
# # o3d.visualization.draw_geometries([pcd])
#
# # file=r"E:\desktop\nuscenes_sam\output_pcd\scene0149_00_coords.ply"
# # file=r"dense_ground.pcd"
# # ply = o3d.io.read_point_cloud(file)
# # o3d.visualization.draw_geometries([ply])  # 可视化
#
# # #原始的颜色
# # data_dict = torch.load(r'E:\desktop\nuscenes_sam\scene0354_00_before_seg_dict.pth')  #这个是原始的数据集  这个是模型的输入之一
# # scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()
# #
# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(scene_coord)
# #
# # o3d.visualization.draw_geometries([pcd])
#
#
#
# def visualize_point_cloud_with_groups(pth_file_path):
#     # 加载 pth 文件
#     data_dict = torch.load(pth_file_path)
#
#     # 提取点云坐标
#     scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()  # 点云坐标
#     # --- 创建仅坐标的点云对象用于保存 ---
#     pcd_coords = o3d.geometry.PointCloud()
#     pcd_coords.points = o3d.utility.Vector3dVector(scene_coord)
#
#     # 保存纯坐标的PLY文件
#     # if save_path:
#     #     # 自动添加后缀如果用户忘记
#     #     if not save_path.endswith('.ply'):
#     #         save_path += '.ply'
#     #     o3d.io.write_point_cloud(save_path, pcd_coords)
#     #     print(f"坐标点云已保存至: {save_path}")
#
#     # 提取分组信息
#     group_data = torch.tensor(data_dict["group"]).cpu().numpy()  # 分组信息
#
#     # 打印分组信息（唯一分组编号）
#     unique_groups = np.unique(group_data)
#     # print(f"Unique groups in the data: {unique_groups}")
#
#     # 创建 Open3D 点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(scene_coord)
#
#     # 使用 color map 为每个分组分配颜色
#     num_groups = len(unique_groups)  # 获取分组数量
#     print("num_groups", num_groups)
#
#     # 使用一个对比度较强的色图（比如 coolwarm）
#     color_map = cm.get_cmap("plasma", num_groups)  # 使用 "coolwarm" 色图 viridis  plasma
#     norm = Normalize(vmin=np.min(group_data), vmax=np.max(group_data))
#     sm = ScalarMappable(cmap=color_map, norm=norm)
#
#     # 将每个点分配对应的颜色
#     group_colors = sm.to_rgba(group_data)[:, :3]  # 获取 RGB 颜色，忽略透明度
#
#     # 设置点云的颜色
#     pcd.colors = o3d.utility.Vector3dVector(group_colors)
#
#     # 可视化点云
#     o3d.visualization.draw_geometries([pcd])
#
# # 例子：加载并可视化分组后的点云
# pth_file = r'E:\desktop\nuscenes_sam\fixbm\scene0149_00_before_seg_dict.pth'
# # save_path = r'E:\desktop\nuscenes_sam\output_pcd\scene0149_00_coords.ply'
# visualize_point_cloud_with_groups(pth_file)
#


# import torch
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
# def visualize_point_cloud_with_groups(pth_file_path):
#     # 加载 pth 文件
#     data_dict = torch.load(pth_file_path)
#
#     # 提取点云坐标和分组信息
#     scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()
#     group_data = torch.tensor(data_dict["group"]).cpu().numpy()
#
#     # 过滤掉分组信息为 -1 的点
#     valid_mask = group_data != -1
#     scene_coord = scene_coord[valid_mask]
#     group_data = group_data[valid_mask]
#
#     # 获取所有有效的唯一分组
#     unique_groups = np.unique(group_data)
#     print("Unique groups:", unique_groups)
#
#     # 为每个唯一分组分配一个离散颜色
#     num_groups = len(unique_groups)
#     color_map = cm.get_cmap("plasma", num_groups)
#     group_color_dict = {group: color_map(i)[:3] for i, group in enumerate(unique_groups)}
#
#     # 将每个点的分组信息映射到对应的颜色
#     group_colors = np.array([group_color_dict[g] for g in group_data])
#
#     # 创建 Open3D 点云对象，并设置点和对应的颜色
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(scene_coord)
#     pcd.colors = o3d.utility.Vector3dVector(group_colors)
#
#     # 可视化点云
#     o3d.visualization.draw_geometries([pcd])
#
#
# # 例子：加载并可视化分组后的点云（不显示 group 为 -1 的点）
# pth_file = r'E:\desktop\nuscenes_sam\fixbm\scene0149_00_before_seg_dict——15.pth'
# visualize_point_cloud_with_groups(pth_file)


# import torch
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
# def visualize_point_cloud_with_groups(pth_file_path):
#     data_dict = torch.load(pth_file_path)
#
#     # 提取坐标和分组数据
#     scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()
#     group_data  = torch.tensor(data_dict["group"]).cpu().numpy()
#
#     # 过滤掉分组信息为 -1 的点
#     valid_mask  = group_data != -1
#     scene_coord = scene_coord[valid_mask]
#     group_data  = group_data[valid_mask]
#
#     unique_groups = np.unique(group_data)
#     print("Unique groups:", unique_groups)
#
#     # 定义颜色映射
#     num_groups = len(unique_groups)
#     color_map = cm.get_cmap("hsv", num_groups)
#
#     # 建立 group->color 的字典，方便可视化
#     group_color_dict = {}
#     for i, g in enumerate(unique_groups):
#         group_color_dict[g] = color_map(i)[:3]  # RGBA中的前3位
#
#     # 给每个点涂上对应的颜色
#     group_colors = np.array([group_color_dict[g] for g in group_data])
#
#     # 输出颜色 - 分组对照关系
#     print("Group -> Color:")
#     for g, c in group_color_dict.items():
#         print(f"  Group {g}: {c}")
#
#     # 使用 open3d 可视化
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(scene_coord)
#     pcd.colors = o3d.utility.Vector3dVector(group_colors)
#     o3d.visualization.draw_geometries([pcd])
#
#     # 如果还想在一个额外的 matplotlib 窗口中显示“分组-颜色”对照的 Legend，可以这样做
#     plt.figure()
#     for i, g in enumerate(unique_groups):
#         plt.scatter([], [], color=color_map(i)[:3], label=f"Group {g}")
#     plt.legend(loc='upper left')
#     plt.title("Group - Color Legend")
#     plt.show()
#
# # 使用示例
# pth_file = r'E:\desktop\nuscenes_sam\fixbm\scene0149_00_before_seg_dict——15.pth'
# visualize_point_cloud_with_groups(pth_file)




import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

def visualize_point_cloud_with_groups_randomized(pth_file_path):
    data_dict = torch.load(pth_file_path)

    # 提取坐标和分组数据
    scene_coord = torch.tensor(data_dict["coord"]).cpu().numpy()
    group_data  = torch.tensor(data_dict["group"]).cpu().numpy()
    color=torch.tensor(data_dict["color"]).cpu().numpy()
    print("scene_coord.shape",scene_coord.shape)
    print("group_data.shape", group_data.shape)
    print("color.shape", color.shape)
    # 过滤掉分组信息为 -1 的点
    valid_mask  = group_data != -1
    scene_coord = scene_coord[valid_mask]
    group_data  = group_data[valid_mask]

    unique_groups = np.unique(group_data)
    print("Unique groups:", unique_groups)

    # 随机打乱分组顺序
    group_list = list(unique_groups)
    random.shuffle(group_list)

    # 这里可以选一个离散的颜色表，比如 tab20、tab20b、tab20c 等等
    # 这些颜色表在 matplotlib 里往往是离散且相对分辨度较高的调色板
    cmap_name = "tab20"
    color_map = cm.get_cmap(cmap_name, len(group_list))

    group_color_dict = {}
    for i, g in enumerate(group_list):
        # 取 RGBA 的前 3 个通道作为 RGB
        group_color_dict[g] = color_map(i)[:3]

    # 根据分组给每个点赋颜色
    group_colors = np.array([group_color_dict[g] for g in group_data])

    print("Group -> Color:")
    for g, c in group_color_dict.items():
        print(f"  Group {g}: {c}")

    # 用 open3d 绘制点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_coord)
    pcd.colors = o3d.utility.Vector3dVector(group_colors)
    o3d.visualization.draw_geometries([pcd])

    # 如需一个 matplotlib 的图例：
    plt.figure()
    for i, g in enumerate(group_list):
        plt.scatter([], [], color=group_color_dict[g], label=f"Group {g}")
    plt.legend(loc='upper left')
    plt.title(f"Group-Color Legend ({cmap_name})")
    plt.show()

# 使用示例
pth_file = r"E:\desktop\nuscenes_sam\统一ground\scene0149_00_before_seg_dict.pth"
visualize_point_cloud_with_groups_randomized(pth_file)
