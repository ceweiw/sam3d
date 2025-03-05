import torch
import numpy as np
import os


def print_group_statistics(scene_name, save_path, num_frames=10):
    for frame_idx in range(num_frames):
        pcd_data_path = os.path.join(save_path, f"{scene_name}_frame_{frame_idx}.pth")

        if not os.path.exists(pcd_data_path):
            print(f"Frame {frame_idx} not found.")
            continue

        # 加载点云数据
        pcd_data = torch.load(pcd_data_path)
        group_ids = np.array(pcd_data["group"])  # 获取分组信息

        # 统计分组信息
        unique_groups, counts = np.unique(group_ids, return_counts=True)

        # 打印每个分组的 ID 和数量
        print(f"Statistics for frame {frame_idx}:")
        for group, count in zip(unique_groups, counts):
            print(f"Group ID: {group}, Count: {count}")
        print("-" * 50)


# 使用示例：
print_group_statistics('scene0149_00', r'E:\desktop\nuscenes_sam', num_frames=4)
