import numpy as np
import open3d as o3d
import torch
import os

'''
这个代码对seg_pcd函数调用get_pcd生成点云后进行可视化 主要是验证同一个物体在前后帧中的分组信息是否一致
经过这个代码验证 那么这里应该就是对于每一帧进行分割的时候都是不同的group_id 在后面进行融合的时候才会统一
（这个代码对于一个group_id会显示同一个颜色 但是在观察中 前后帧的树 并不是一个颜色）
另外我打印了每帧的分组信息 都是从-1开始的  那就进一步说明对于前后帧中同一个物体并不会有同一个分组id
还是得靠匹配来进行分组id的统一
'''

# 可视化多个点云帧，并根据分组信息着色
def visualize_multiple_frames_by_group(scene_name, save_path, num_frames=10):
    # 为了确保每个分组都有一个不同的颜色，我们可以为每个组分配一个随机颜色
    np.random.seed(42)  # 为了颜色的一致性，可以固定种子
    group_colors = np.random.rand(100, 3)  # 假设最多有100个分组

    # 创建一个字典来存储每一帧的 z 值范围
    z_ranges = {}

    for frame_idx in range(1, num_frames + 1):
        pcd_data_path = os.path.join(save_path, f"{scene_name}_frame_{frame_idx}.pth")
        if not os.path.exists(pcd_data_path):
            print(f"Frame {frame_idx} not found.")
            continue

        # 加载点云数据
        pcd_data = torch.load(pcd_data_path)
        coord = pcd_data["coord"]  # 点的坐标
        color = pcd_data["color"]  # 点的颜色
        group_ids = pcd_data["group"]  # 点的分组信息

        # 将坐标和颜色转为 numpy 数组
        pcd_points = np.array(coord)
        pcd_colors = np.array(color) / 255.0  # 将颜色归一化到 [0, 1] 范围

        # 根据分组信息为每个点分配颜色
        unique_groups = np.unique(group_ids)  # 获取所有唯一的分组 ID
        group_color_map = np.zeros_like(pcd_colors)  # 初始化一个数组来存储点的颜色

        for group_id in unique_groups:
            # 为每个分组分配一个颜色
            group_color = group_colors[group_id % 100]  # 使用 group_colors 中的颜色
            group_mask = (group_ids == group_id)  # 找到该分组对应的点
            group_color_map[group_mask] = group_color  # 为这些点赋予颜色

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(group_color_map)  # 使用基于分组的颜色

        # 可视化当前点云帧
        print(f"Visualizing frame {frame_idx}...")
        o3d.visualization.draw_geometries([pcd])

        # 计算 z 值的范围
        z_min = np.min(pcd_points[:, 2])  # z 值最小值
        z_max = np.max(pcd_points[:, 2])  # z 值最大值

        # 将 z 值范围存储到字典中
        z_ranges[frame_idx] = {"z_min": z_min, "z_max": z_max}

    # 返回所有帧的 z 值范围
    return z_ranges


# 可视化前 10 帧点云，按分组显示
z_ranges = visualize_multiple_frames_by_group('scene0149_00', r'E:\desktop\nuscenes_sam', num_frames=9)

# 打印 z 值范围字典
print("Z value ranges for each frame:")
print(z_ranges)