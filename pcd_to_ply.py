# import numpy as np
# import cv2
# import open3d as o3d
# import matplotlib.pyplot as plt
# from os.path import join
# import os
#
# def get_pcd(name):
#     intrinsic_path = join('E:/desktop/nuscenes_sam/scene0149_00/intrinsics/intrinsic_depth.txt')
#     depth_intrinsic = np.loadtxt(intrinsic_path)
#
#     pose = join('E:/desktop/nuscenes_sam/scene0149_00/pose/', name + '.txt')
#     depth = join('E:/desktop/nuscenes_sam/scene0149_00/depth/', name + '.png')
#     color = join('E:/desktop/nuscenes_sam/scene0149_00/color/', name + '.jpg')
#
#     depth_img = cv2.imread(depth, -1)  # read 16bit grayscale image
#     mask = (depth_img != 0)
#     color_image = cv2.imread(color)
#
#     color_image = np.reshape(color_image[mask], [-1, 3])
#     colors = np.zeros_like(color_image)
#     colors[:, 0] = color_image[:, 2]
#     colors[:, 1] = color_image[:, 1]
#     colors[:, 2] = color_image[:, 0]
#
#     pose = np.loadtxt(pose)
#
#     depth_shift = 1000.0
#     x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
#                        np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
#     uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
#     uv_depth[:, :, 0] = x
#     uv_depth[:, :, 1] = y
#     uv_depth[:, :, 2] = depth_img / depth_shift
#     uv_depth = np.reshape(uv_depth, [-1, 3])
#     uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()
#
#     fx = depth_intrinsic[0, 0]
#     fy = depth_intrinsic[1, 1]
#     cx = depth_intrinsic[0, 2]
#     cy = depth_intrinsic[1, 2]
#     bx = depth_intrinsic[0, 3]
#     by = depth_intrinsic[1, 3]
#     n = uv_depth.shape[0]
#     points = np.ones((n, 4))
#     X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
#     Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
#     points[:, 0] = X
#     points[:, 1] = Y
#     points[:, 2] = uv_depth[:, 2]
#     points_world = np.dot(points, np.transpose(pose))
#     points_world = points_world[:, :3]
#
#     return points_world, colors
#
#
# def save_point_cloud(pcd, filename, save_dir):
#     """
#     保存点云到PLY文件（仅坐标）
#     参数：
#         pcd: Open3D点云对象
#         filename: 原始文件名（不带扩展名）
#         save_dir: 保存目录
#     """
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 创建仅包含坐标的点云对象
#     pcd_coords = o3d.geometry.PointCloud()
#     pcd_coords.points = pcd.points
#
#     # 保存PLY（不包含颜色和其他属性）
#     ply_path = join(save_dir, f"{filename}.ply")
#     o3d.io.write_point_cloud(ply_path, pcd_coords, write_ascii=False)
#     print(f"Saved Coordinate-only PLY to: {ply_path}")
#
#
# def visualize_point_clouds(filenames):
#     pcd_list = []
#     for filename in filenames:
#         points, colors = get_pcd(filename)
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#         pcd_list.append(pcd)
#
#         # 保存点云（仅坐标）
#         # if save_dir:
#         #     save_point_cloud(pcd, filename, save_dir)
#
#     # 可视化
#     o3d.visualization.draw_geometries(pcd_list)
#
#
# # 获取连续的10帧场景，并显示它们
# frame_ids = [str(i) for i in range(5, 9)]
# visualize_point_clouds(
#     filenames=frame_ids
# )
# # visualize_point_clouds(
# #     filenames=frame_ids,
# #     save_dir="E:/desktop/nuscenes_sam/output_pcd"  # 修改为你的保存路径
# # )


import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from os.path import join
import os


def get_pcd(name):
    intrinsic_path = join('E:/desktop/nuscenes_sam/scene0149_00/intrinsics/intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose_path = join('E:/desktop/nuscenes_sam/scene0149_00/pose/', name + '.txt')
    depth_path = join('E:/desktop/nuscenes_sam/scene0149_00/depth/', name + '.png')
    color_path = join('E:/desktop/nuscenes_sam/scene0149_00/color/', name + '.jpg')

    depth_img = cv2.imread(depth_path, -1)  # 读取16位灰度图
    mask = (depth_img != 0)
    color_image = cv2.imread(color_path)

    # 只保留深度有效处的颜色信息，并转换为RGB格式
    color_image = np.reshape(color_image[mask], [-1, 3])
    colors = np.zeros_like(color_image)
    colors[:, 0] = color_image[:, 2]
    colors[:, 1] = color_image[:, 1]
    colors[:, 2] = color_image[:, 0]

    pose = np.loadtxt(pose_path)

    depth_shift = 1000.0
    x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
                       np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x
    uv_depth[:, :, 1] = y
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()

    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]
    n = uv_depth.shape[0]
    points = np.ones((n, 4))
    X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 0] = X
    points[:, 1] = Y
    points[:, 2] = uv_depth[:, 2]
    points_world = np.dot(points, np.transpose(pose))
    points_world = points_world[:, :3]

    return points_world, colors


def save_point_cloud(pcd, filename, save_dir):
    """
    保存点云到PLY文件（仅坐标）
    参数：
        pcd: Open3D点云对象
        filename: 原始文件名（不带扩展名）
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 创建仅包含坐标的点云对象
    pcd_coords = o3d.geometry.PointCloud()
    pcd_coords.points = pcd.points

    # 保存PLY（不包含颜色和其他属性）
    ply_path = join(save_dir, f"{filename}.ply")
    o3d.io.write_point_cloud(ply_path, pcd_coords, write_ascii=False)
    print(f"Saved Coordinate-only PLY to: {ply_path}")


def visualize_point_clouds(filenames):
    pcd_list = []
    total_frames = len(filenames)
    # 使用 matplotlib 的 hsv colormap 生成不同的颜色
    cmap = plt.get_cmap("hsv")
    for i, filename in enumerate(filenames):
        points, original_colors = get_pcd(filename)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 根据帧的索引生成一个统一的颜色（RGB）
        color = np.array(cmap(i / total_frames)[:3])  # 舍弃 alpha 通道
        # 为当前点云中的所有点分配同一颜色
        colors = np.tile(color, (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd)

        # 如果需要保存点云（仅坐标），可以调用下面的函数
        # save_point_cloud(pcd, filename, "E:/desktop/nuscenes_sam/output_pcd")

    # 可视化所有帧的点云
    o3d.visualization.draw_geometries(pcd_list)


# 示例：获取连续的几帧（此处为示例，实际最多可显示41帧）
frame_ids = [str(i) for i in range(1,2)]
visualize_point_clouds(filenames=frame_ids)
