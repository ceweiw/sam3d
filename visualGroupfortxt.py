import numpy as np
import torch
import os

def save_pcd_to_txt(scene_name, save_path, num_frames=10):
    for frame_idx in range(num_frames):
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
        group_ids = np.array(group_ids)  # 分组信息

        # 创建一个文本文件来保存数据
        save_txt_path = os.path.join(save_path, f"{scene_name}_frame_{frame_idx}_pcd.txt")
        with open(save_txt_path, 'w') as f:
            # 写入每个点的坐标、颜色和分组 ID
            for i in range(len(pcd_points)):
                # 以“坐标X，坐标Y，坐标Z，颜色R，颜色G，颜色B，分组ID”格式保存每一行
                f.write(f"{pcd_points[i][0]:.6f} {pcd_points[i][1]:.6f} {pcd_points[i][2]:.6f} "
                        f"{pcd_colors[i][0]:.6f} {pcd_colors[i][1]:.6f} {pcd_colors[i][2]:.6f} "
                        f"{group_ids[i]}\n")

        print(f"Saved frame {frame_idx} point cloud data to {save_txt_path}.")

# 保存前 10 帧点云数据到文本文件
save_pcd_to_txt('scene0149_00', r'E:\desktop\nuscenes_sam', num_frames=10)
