import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def densify_ground(input_path, output_path, radial_step=0.3, angular_points=36):
    # Read point cloud and verify data structure
    cloud = o3d.io.read_point_cloud(input_path)
    points = np.asarray(cloud.points)
    print("Original point cloud dimensions:", points.shape)

    # Ensure (N,3) format (fix the check logic)
    if points.shape[1] != 3:  # Check if the second dimension is 3
        points = points.T

    # Correctly unpack center coordinates
    center = np.mean(points, axis=0)
    cx, cy, cz = center  # Correct unpacking of center coordinates

    # Corrected coordinate calculation
    xy = points[:, :2] - np.array([cx, cy])  # Now broadcasting shape should be (N, 2) - (1, 2)

    # Compute radial distances and identify the minimum and maximum radial distances
    radii = np.linalg.norm(xy, axis=1)
    r_min, r_max = np.min(radii), np.max(radii)

    # Prepare KNN search
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(points[:, :2])

    # Generate ring point cloud parameters
    new_radii = np.arange(r_min, r_max, radial_step)
    theta = np.linspace(0, 2 * np.pi, angular_points, endpoint=False)

    # Generate new point cloud coordinates
    new_points = []
    for r in new_radii:
        for t in theta:
            # Generate polar coordinates
            x = cx + r * np.cos(t)
            y = cy + r * np.sin(t)

            # Find nearest neighbor height (z)
            _, indices = knn.kneighbors([[x, y]])
            z = np.mean(points[indices, 2])

            new_points.append([x, y, z])

    # Combine original points with new points
    new_points = np.array(new_points)
    combined = np.vstack([points, new_points])

    # Save the result
    dense_cloud = o3d.geometry.PointCloud()
    dense_cloud.points = o3d.utility.Vector3dVector(combined)
    o3d.io.write_point_cloud(output_path, dense_cloud)

    print(f"Saved the densified point cloud to: {output_path}")

# Example usage
densify_ground("E:/desktop/nuscenes_sam/output_pcd/scene0149_00_coords.ply", "dense_ground.pcd")






import torch
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
    print("points.shape", points.shape)  #shape(3,N)
    id_max= group_ids.max()+1
    ground_group_ids = group_ids[ground_mask]

    # 调用 densify_lidar_points(N,3) 来进行点云稠密化，并生成颜色和分组ID
    densified_points, densified_colors, densified_group_ids = densify_lidar_points(id_max,save_path,ground_points, colors, ground_group_ids, max_distance, densification_factor)
    print("densified_points152",densified_points.shape)  #shape(3,N)
    # 合并地面点和非地面点
    non_ground_points = points[:, ~ground_mask]  # 非地面点
    non_ground_colors = colors[:, ~ground_mask]
    non_ground_group_ids = group_ids[~ground_mask]
    print("non_ground_points.shape",non_ground_points.shape)  #(3,N)
    # 转置 non_ground_colors 使其与 densified_colors 在维度上匹配
    non_ground_colors = non_ground_colors.T  # 从 (3, M) 转为 (M, 3)
    print("non_ground_colors",non_ground_colors.shape)  #shape(N,3)
    # 将地面点和非地面点合并
    all_points = np.hstack((densified_points, non_ground_points))  #shape(3,N)
    all_colors = np.vstack((densified_colors, non_ground_colors))
    all_group_ids = np.hstack((densified_group_ids, non_ground_group_ids))
    print("all_points",all_points.shape)
    return all_points, all_colors, all_group_ids