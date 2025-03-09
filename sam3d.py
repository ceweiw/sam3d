"""
Main Script

Author: Yunhan Yang (yhyang.myron@gmail.com)
"""

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
import multiprocessing as mp
import pointops
import random
import argparse

from segment_anything import build_sam, SamAutomaticMaskGenerator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image
from os.path import join
from util import *
from sklearn.neighbors import NearestNeighbors
import time
def pcd_ensemble(org_path, new_path, data_path, vis_path):
    new_pcd = torch.load(new_path)
    new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
    with open(org_path) as f:
        segments = json.load(f)
        org_pcd = np.array(segments['segIndices'])
    match_inds = [(i, i) for i in range(len(new_pcd))]
    new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
    data = torch.load(data_path)
    visualize_partition(data["coord"], new_group, vis_path)


def get_sam(image, mask_generator):
    masks = mask_generator.generate(image)
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1
    return group_ids

def ransac_plane_fitting(points, distance_threshold=0.2, iterations=1000):
    best_inlier_count = 0
    best_plane_params = None
    best_inliers = None

    for _ in range(iterations):
        # 随机选择3个点来拟合平面
        sample_indices = np.random.choice(points.shape[1], 3, replace=False)
        sample_points = points[:, sample_indices]

        # 计算平面方程 ax + by + cz + d = 0
        # 通过三个点计算平面的法向量
        v1 = sample_points[:, 1] - sample_points[:, 0]
        v2 = sample_points[:, 2] - sample_points[:, 0]
        normal = np.cross(v1, v2)  # 平面的法向量

        # 计算平面方程的参数
        a, b, c = normal
        d = -np.dot(normal, sample_points[:, 0])  # 平面方程的d

        # 计算所有点到平面的距离
        distances = np.abs(np.dot(normal, points[:3, :]) + d) / np.linalg.norm(normal)

        # 找到内点（距离小于阈值的点）
        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)

        # 如果找到更多的内点，更新最佳平面
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane_params = (a, b, c, d)
            best_inliers = inliers

    return best_plane_params, best_inliers


def filter_ground_points(points, plane_params, distance_threshold=0.2):
    a, b, c, d = plane_params
    distances = np.abs(np.dot(np.array([a, b, c]), points[:3, :]) + d) / np.linalg.norm([a, b, c])
    mask = distances < distance_threshold  # 小于阈值的点视为地面点
    ground_points = points[:, mask]
    return ground_points,mask


def densify_lidar_points(id_max,save_path,points, colors, group_ids, max_distance, densification_factor=2):
    filtered_points = points  # 传进来的这些是地面点
    print(f"filtered_points shape: {filtered_points.shape}")   #shape(3,N)

    # group_ids[group_ids!=-1]=id_max
    group_ids[group_ids!=-1]=id_max
    # 计算所有符合条件的点的密度（可以通过邻近点数目来估算）
    nn = NearestNeighbors(n_neighbors=4)  # 选择4个邻居来估算密度
    nn.fit(filtered_points.T)  # 训练模型
    distances_to_nearest_neighbors, _ = nn.kneighbors(filtered_points.T)  # 获取每个点到最近4个点的距离
    avg_distance = np.mean(distances_to_nearest_neighbors, axis=1)  # 计算每个点的平均距离

    # 根据距离调整稠密化的比例
    density_factors = densification_factor * (1 / (avg_distance + 1e-5))  # 越远的点密度因子越大
    densified_points = []
    densified_colors = []
    densified_group_ids = []

    # 对每个点根据密度因子进行稠密化
    for i, point in enumerate(filtered_points.T):
        # 通过密度因子来重复生成点
        num_new_points = int(density_factors[i]*0.1 )  # 生成新点的数量
        new_points = np.tile(point, (num_new_points, 1))  # 重复当前点
        densified_points.append(new_points)

        # 同时重复生成对应的颜色和分组ID
        densified_colors.append(np.tile(colors[:, i % colors.shape[1]], (num_new_points, 1)))
        densified_group_ids.append(np.tile(group_ids[i % len(group_ids)], num_new_points))

    densified_points = np.vstack(densified_points)  # 合并所有稠密化的点 (N,3)
    densified_colors = np.vstack(densified_colors)  # 合并颜色
    densified_group_ids = np.hstack(densified_group_ids)  # 合并分组ID

    if save_path is not None and scene_name is not None:
        pcd_data = {
            'coord': torch.tensor(densified_points, dtype=torch.float32),
            'colors': torch.tensor(densified_colors, dtype=torch.float32),
            'group': torch.tensor(densified_group_ids, dtype=torch.long),
        }
        # 使用时间戳生成唯一文件名
        timestamp = int(time.time())  # 获取当前时间戳
        file_name = f"{scene_name}_framedimian_{timestamp}.pth"
        # 保存文件
        torch.save(pcd_data, os.path.join(save_path, file_name))
        print(f"Saved densified point cloud to {os.path.join(save_path, file_name)}")
    return densified_points.T, densified_colors, densified_group_ids

def densificate_pc(save_path,points, colors, group_ids, max_distance=200.0, densification_factor=2):
    '''
    这个是地面点云稠密化的入口函数和一整个流程
    '''
    plane_params, inliers = ransac_plane_fitting(points)
    ground_points, ground_mask = filter_ground_points(points, plane_params)
    # print("points.shape", points.shape)  #shape(3,N)
    id_max= group_ids.max()+1
    ground_group_ids = group_ids[ground_mask]

    # 调用 densify_lidar_points(N,3) 来进行点云稠密化，并生成颜色和分组ID
    densified_points, densified_colors, densified_group_ids = densify_lidar_points(id_max,save_path,ground_points, colors, ground_group_ids, max_distance, densification_factor)
    # print("densified_points152",densified_points.shape)  #shape(3,N)
    # 合并地面点和非地面点
    non_ground_points = points[:, ~ground_mask]  # 非地面点
    non_ground_colors = colors[:, ~ground_mask]
    non_ground_group_ids = group_ids[~ground_mask]
    # print("non_ground_points.shape",non_ground_points.shape)  #(3,N)
    # 转置 non_ground_colors 使其与 densified_colors 在维度上匹配
    non_ground_colors = non_ground_colors.T  # 从 (3, N) 转为 (N, 3)
    # print("non_ground_colors",non_ground_colors.shape)  #shape(N,3)
    # 将地面点和非地面点合并
    all_points = np.hstack((densified_points, non_ground_points))  #shape(3,N)
    all_colors = np.vstack((densified_colors, non_ground_colors))
    all_group_ids = np.hstack((densified_group_ids, non_ground_group_ids))
    # print("all_points",all_points.shape)
    return all_points, all_colors, all_group_ids


def update_group_ids(id_max, points, colors, group_ids):
    """
    统一更新 group_ids，并保持输入输出的变量和维度一致
    """
    group_ids[group_ids!=-1]=id_max

    # 直接返回原始的 points, colors, 和 group_ids，保持维度一致性
    all_points = points  # 直接返回原始点云
    all_colors = colors # 保证 colors 的维度一致性 (3, N) 转为 (N, 3)
    all_group_ids = group_ids  # 返回修改后的 group_ids

    # print("all_points shape:", all_points.shape)  # 输出 all_points 维度
    # print("all_colors shape:", all_colors.shape)  # 输出 all_colors 维度
    # print("all_group_ids shape:", all_group_ids.shape)  # 输出 all_group_ids 维度
    return all_points, all_colors, all_group_ids

def make_ground_same(save_path, points, colors, group_ids, max_distance=200.0, densification_factor=2):
    """
    该函数实现点云数据的处理，统一 group_ids，并返回未变动的点云数据
    """
    plane_params, inliers = ransac_plane_fitting(points)
    ground_points, ground_mask = filter_ground_points(points, plane_params)
    print("points.shape", points.shape)  # shape(3,N)
    id_max = group_ids.max() + 1
    ground_group_ids = group_ids[ground_mask]  #(N,)
    print("194ground_group_ids.shape",ground_group_ids.shape)
    ground_color=colors[:,ground_mask]  #(3,N)
    ground_color=ground_color.T
    print("196ground_color.shape",ground_color.shape)
    ground_points=points[:,ground_mask]  #(3,N)
    print("198ground_points.shape",ground_points.shape)
    # 调用 update_group_ids 来进行统一分组ID
    ground_points, ground_color, ground_group_ids = update_group_ids(id_max, ground_points, ground_color, ground_group_ids)
    non_ground_group_ids = group_ids[~ground_mask] #(N,)
    print("202non_ground_group_ids.shape",non_ground_group_ids.shape)
    non_ground_colors = colors[:, ~ground_mask]  #(3,N)
    non_ground_colors=non_ground_colors.T
    print("204non_ground_colors.shape",non_ground_colors.shape)
    non_ground_points = points[:, ~ground_mask]  #(3,N)
    print("206non_ground_points.shape",non_ground_points.shape)
    # non_ground_colors=non_ground_colors.T

    all_points = np.hstack((ground_points, non_ground_points))  #shape(3,N)
    all_colors = np.vstack((ground_color, non_ground_colors))
    all_group_ids = np.hstack((ground_group_ids, non_ground_group_ids))
    # print("all_points.shape:", all_points.shape)  # shape(3, N)
    # print("all_colors.shape:", all_colors.shape)  # shape(N, 3)
    # print("all_group_ids.shape:", all_group_ids.shape)  # shape(N,)

    # 返回更新后的点云数据
    return all_points, all_colors, all_group_ids

def get_pcd(save_path,scene_name, color_name, rgb_path, mask_generator, save_2dmask_path):
    intrinsic_path = join(rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose = join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    depth = join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    color = join(rgb_path, scene_name, 'color', color_name)

    depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
    mask = (depth_img != 0)
    color_image = cv2.imread(color)
    # color_image = cv2.resize(color_image, (640, 480))

    save_2dmask_path = join(save_2dmask_path, scene_name)
    if mask_generator is not None:
        group_ids = get_sam(color_image, mask_generator)
        if not os.path.exists(save_2dmask_path):
            os.makedirs(save_2dmask_path)
        img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
        img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))   #这里保存了sam2d的分割掩码
    else:
        group_path = join(save_2dmask_path, color_name[0:-4] + '.png')
        img = Image.open(group_path)
        group_ids = np.array(img, dtype=np.int16)

    color_image = np.reshape(color_image[mask], [-1,3])
    group_ids = group_ids[mask]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]

    pose = np.loadtxt(pose)

    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()

    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(pose))   # Shape: (N, 4)

    # points_world = points_world[:, :3].T  #(3,N)
    # colors=colors.T #shape(3,N)
    # # densified_points, densified_colors, densified_group_ids = densificate_pc(save_path,points_world, colors, group_ids, max_distance=200.0)
    # densified_points, densified_colors, densified_group_ids = make_ground_same(save_path,points_world, colors, group_ids, max_distance=200.0)
    # densified_group_ids = num_to_natural(densified_group_ids)  #shape (N,)
    # save_dict = dict(coord=densified_points.T[:, :3], color=densified_colors, group=densified_group_ids)
    # print("217points_world.shape",densified_points.T[:, :3].shape)  #shape (N,3)
    # # print("218colors.shape",densified_colors.shape)  #shape (N,3)
    # # print("219group_ids.shape",densified_group_ids.T.shape)  #shape (N,)

    group_ids = num_to_natural(group_ids)  #shape(2980,)
    save_dict = dict(coord=points_world[:,:3], color=colors, group=group_ids)
    # # print("223points_world.shape",points_world.shape) #shape (N,4)
    # # print("224colors.shape",colors.shape)  #shape (N,3)
    # # print("225group_ids.shape",group_ids.shape)  #shape(N,)
    return save_dict


def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    print("231xyz.shape",xyz.shape)
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    # 获取输入点云的组信息
    group_0 = input_dict["group"] #pcd0
    group_1 = new_input_dict["group"]

    # 更新 group_1 中的组标识符，使其与 group_0 中的组标识符不会重复
    group_1[group_1 != -1] += group_0.max() + 1

    # 统计 group_0 和 group_1 中每个组的点数量
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # 计算重叠点的组映射关系
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]   #第一次调用 这里i会有n个点与之匹配
        group_j = group_0[j]

        # 如果 group_i 为 -1，说明该点还未分配组，直接将其分配为 group_j
        if group_i == -1:
            group_1[i] = group_0[j]
            continue

        # 如果 group_j 为 -1，跳过该点
        if group_j == -1:
            continue

        # 记录每对重叠点的组映射关系
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1   #这个里面存储的应该是与i对应的每个j的数量  ij理解为掩码

    # 更新 group_1 中的组信息
    for group_i, overlap_count in group_overlap.items():
        # 找到与 group_i 对应的重叠最多的 group_j
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]

        # 获取 group_0 和 group_1 中对应组的点数
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)

        # 如果重叠点的比例超过给定阈值，则合并两个组
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j

    return group_1


def cal_2_scenes(pcd_list, index, voxel_size, voxelize, ratio=0.5 ,th=50):
    # 如果索引列表只有一个元素，则返回该元素对应的点云
    if len(index) == 1:
        return(pcd_list[index[0]])

    # 获取索引对应的两个点云
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]

    # 使用给定的体素大小和阈值对两个点云进行体素化处理
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)

    # 如果 pcd0 或 pcd1 为 None，返回存在的点云数据
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # 计算两个点云的双向重叠区域
    #遍历pcd1为pcd0中的每一个点寻找最近点
    match_inds = get_matching_indices(pcd1, pcd0, 15 * voxel_size, 1)
    #pcd1的组id会偏移 更新pcd1的值
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds,ratio)  # BM过程

    match_inds = get_matching_indices(pcd0, pcd1, 15* voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group  # 将 pcd1 的新组信息加入到输入数据字典中
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds,ratio)  #

    # pcd0_new_group=input_dict_0["group"]

    # 合并两个点云的组信息
    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)  #

    # 合并两个点云的坐标和颜色信息
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)

    # 创建包含坐标、颜色和组信息的新字典
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    # 对合并后的点云进行体素化处理
    pcd_dict = voxelize(pcd_dict)

    return pcd_dict

###for save group info###
def save_pcd_list(pcd_list, save_path, scene_name):
    '''
    这个是用来保存中间帧的结果的
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存每一帧的点云数据
    for idx, pcd_dict in enumerate(pcd_list):
        pcd_data = {
            "coord": pcd_dict["coord"],  # 点的坐标
            "color": pcd_dict["color"],  # 点的颜色
            "group": pcd_dict["group"],  # 分组信息
        }
        torch.save(pcd_data, os.path.join(save_path, f"{scene_name}_frame_{idx}.pth"))


def extract_frames_from_seg_dict(seg_dict, frame_boundaries, num_frames=10):
    """
    这个是抽帧用的 根据之前保存的帧的范围从稠密化之后的去抽取
    """
    coords = np.array(seg_dict['coord'])  # 获取所有点的坐标
    frame_list = []  # 用于保存所有划分的帧数据

    # 对于每一帧的范围，进行坐标筛选
    for frame in frame_boundaries[:num_frames]:  # 如果frame_boundaries大于num_frames，取前num_frames个
        color_name, max_x, max_y, min_x, min_y = frame

        # 筛选出符合条件的点（不限制z值，只限制x和y范围）
        mask = (coords[:, 0] >= min_x) & (coords[:, 0] <= max_x) & (coords[:, 1] >= min_y) & (coords[:, 1] <= max_y)
        frame_coords = coords[mask]

        # 创建新的frame字典，保存相关信息
        frame_dict = seg_dict.copy()  # 保持原始字典的结构
        frame_dict['coord'] = frame_coords  # 更新坐标

        # 同样处理颜色和分组信息
        frame_dict['color'] = frame_dict['color'][mask]  # 保留颜色信息
        frame_dict['group'] = frame_dict['group'][mask]  # 保留分组信息

        # 将该帧添加到列表中
        frame_list.append(frame_dict)
    return frame_list


def cluster_point_cloud(data_dict, eps=0.5, min_points=10):
    """
    加载 pth 文件，基于点云坐标进行 DBSCAN 聚类，
    更新数据字典中的 group 信息，并返回更新后的字典。

    参数：
        pth_file_path: 原始 pth 文件路径
        eps: DBSCAN 中邻域搜索的半径参数
        min_points: DBSCAN 中构成簇所需的最小点数
    """
    # 加载数据
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


def seg_pcd(scene_name, rgb_path, data_path, save_path, mask_generator, voxel_size, voxelize, th, train_scenes, val_scenes, save_2dmask_path):
    print("scene_name：",scene_name, flush=True)
    if os.path.exists(join(save_path, scene_name + ".pth")):
        return
    #读取图像数据生成点云
    color_names = sorted(os.listdir(join(rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))
    pcd_list = []
    frame_boundaries = [] ####
    for color_name in color_names:
        print("color_namez：",color_name, flush=True)
        pcd_dict = get_pcd(save_path,scene_name, color_name, rgb_path, mask_generator, save_2dmask_path)
        if len(pcd_dict["coord"]) == 0:
            continue
        #这一部分是用来抽帧处理的
        # coords = np.array(pcd_dict["coord"])
        # max_x, max_y = np.max(coords, axis=0)[:2]
        # min_x, min_y = np.min(coords, axis=0)[:2]
        # frame_boundaries.append((color_name, max_x, max_y, min_x, min_y))

        pcd_dict = voxelize(pcd_dict)
        pcd_list.append(pcd_dict)

    # save_pcd_list(pcd_list, save_path, scene_name)  ########保存中间帧结果为pth文件

    # result = pcd_list[0]  # 将第一帧作为初始合并结果
    # for i in range(1, len(pcd_list)):
    #     print(f"Merging frame {i} into current result...", flush=True)
    #     # 将当前结果与下一帧组成一个列表，然后传入 cal_2_scenes 进行合并
    #     merged_frame = cal_2_scenes([result, pcd_list[i]], (0, 1), voxel_size=voxel_size, voxelize=voxelize,ratio)
    #     if merged_frame is not None:
    #         result = merged_frame
    #     else:
    #         print(f"Merge failed for frame {i}", flush=True)
    # seg_dict = result
    # #这三句是用来聚类的 并把聚类结果进行合并作为最后的结果
    # # clustered_data_dict = cluster_point_cloud(result, eps=0.7, min_points=10)
    # # final_res = cal_2_scenes([result, clustered_data_dict], (0, 1), voxel_size=voxel_size, voxelize=voxelize)
    # # seg_dict = final_res
    # seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

    #合并的过程
    i=0
    myratio=0.5
    while len(pcd_list) != 1:
        print("len(pcd_list)：",len(pcd_list), flush=True)
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):  #自定义函数 一一次拿出两帧数据
            pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize,ratio=myratio/(2**i)) #这里是BM过程
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
        i=i+1
    seg_dict = pcd_list[0] #这里是合并后的结果
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

    # densified_points, densified_colors, densified_group_ids = make_ground_same(save_path,seg_dict["coord"].T, seg_dict["color"].T, seg_dict["group"], max_distance=200.0)
    # densified_group_ids = num_to_natural(densified_group_ids)  #shape (N,)
    # seg_dict = dict(coord=densified_points.T[:, :3], color=densified_colors, group=densified_group_ids)
    #这里是对合并之后的一个完整的帧进行抽帧 达到稠密化的目的
    # frames = extract_frames_from_seg_dict(seg_dict, frame_boundaries, num_frames=len(frame_boundaries))

    torch.save(seg_dict, join(save_path, scene_name + "_before_seg_dict.pth"))

    #稠密化之后再进行合并 但是目前这种做法是没有意义的
    # pcd_list=frames
    # while len(pcd_list) != 1:
    #     print("len(pcd_list)：",len(pcd_list), flush=True)
    #     new_pcd_list = []
    #     for indice in pairwise_indices(len(pcd_list)):  #自定义函数 一一次拿出两帧数据
    #         pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize) #这里是BM过程
    #         if pcd_frame is not None:
    #             new_pcd_list.append(pcd_frame)
    #     pcd_list = new_pcd_list
    # seg_dict = pcd_list[0] #这里是合并后的结果
    # seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))
    # torch.save(seg_dict, join(save_path, scene_name + "_before_seg_dict.pth"))

    # #这上面是2d转3d的点云  下面是加在了原始的点云
    # if scene_name in train_scenes:
    #     scene_path = join(data_path, "train", scene_name + ".pth")
    # elif scene_name in val_scenes:
    #     scene_path = join(data_path, "val", scene_name + ".pth")
    # data_dict = torch.load(scene_path)
    # scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous() #原始的点云
    # new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    #
    # gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    # offset = torch.tensor(gen_coord.shape[0]).cuda()
    # gen_group = seg_dict["group"]
    # indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    # indices = indices.cpu().numpy()
    # group = gen_group[indices.reshape(-1)].astype(np.int16)
    # mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    # group[mask_dis] = -1
    # group = group.astype(np.int16)
    # torch.save(num_to_natural(group), join(save_path, scene_name + ".pth"))


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--rgb_path', type=str,default='/home/cbdes/zx/mydata/test1', help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='/home/cbdes/zx/processed_data', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, default='/home/cbdes/zx/SegmentAnything3D-main/pcd',help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='/home/cbdes/zx/SegmentAnything3D-main/sam2d', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--sam_checkpoint_path', type=str, default='/home/cbdes/zx/SegmentAnything3D-main/sam_vit_h_4b8939.pth', help='the path of checkpoint for SAM')
    parser.add_argument('--scannetv2_train_path', type=str, default='scannet-preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--voxel_size', default=0.05)
    parser.add_argument('--th', default=50, help='threshold of ignoring small groups to avoid noise pixel')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    print(args)
    # with open(args.scannetv2_train_path) as train_file:
    #     train_scenes = train_file.read().splitlines()
    # with open(args.scannetv2_val_path) as val_file:
    #     val_scenes = val_file.read().splitlines()
    train_scenes=1
    val_scenes=1
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))  #、、
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    scene_names = sorted(os.listdir(args.rgb_path))
    for scene_name in scene_names:
        print("scene_name：",scene_name)
        seg_pcd(scene_name, args.rgb_path, args.data_path, args.save_path, mask_generator, args.voxel_size, 
            voxelize, args.th, train_scenes, val_scenes, args.save_2dmask_path)
