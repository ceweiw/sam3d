import numpy as np
import matplotlib.pyplot as plt

# 激光雷达数据路径
lidar_data_path = 'E:/desktop/nuscene_mini/scene0752_00/lidar/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385092150099.pcd.bin'

# LIDAR_TOP 到 CAM_FRONT 的旋转矩阵和平移向量（已给定）
lidar_to_camera_rotation = np.array([
    [0.99989861, 0.01122026, -0.00876743],
    [-0.00936803, 0.05464516, -0.99846189],
    [-0.0107239, 0.99844279, 0.05474473]
])

lidar_to_camera_translation = np.array([-0.01775806, -3.31183671, 2.73615242])

# 将激光雷达点云从激光雷达坐标系转换到相机坐标系
def transform_lidar_to_camera(lidar_points, lidar_to_camera_rotation, lidar_to_camera_translation):
    # 将LIDAR点云从车辆坐标系转换到相机坐标系
    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # 同次坐标
    lidar_points_in_camera = lidar_to_camera_rotation @ lidar_points_hom[:, :3].T + lidar_to_camera_translation[:, None]

    return lidar_points_in_camera.T

# 读取激光雷达点云
point_cloud = np.fromfile(lidar_data_path, dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring_number

# 提取坐标
lidar_points = point_cloud[:, :3]  # 提取 X, Y, Z 坐标

# 将激光雷达点云从激光雷达坐标系转换到相机坐标系
transformed_points = transform_lidar_to_camera(lidar_points, lidar_to_camera_rotation, lidar_to_camera_translation)

# 可视化转换后的点云（现在是3D）
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制变换后的点云
ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='blue', s=1, alpha=0.5)

# 设置标题和标签
ax.set_title('LiDAR Points Transformed to Camera Coordinate System (3D)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
