import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

# 定义四元数和位移
cam_front_rotation = [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754]
cam_front_translation = [1.72200568478, 0.00475453292289, 1.49491291905]

lidar_rotation = [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719]
lidar_translation = [0.985793, 0.0, 1.84019]

# 将四元数转换为旋转矩阵
R_cam_front = quat2mat(cam_front_rotation)
R_lidar = quat2mat(lidar_rotation)

# 计算 CAM_FRONT 到自车坐标系的逆变换
R_cam_front_inv = R_cam_front.T  # 旋转矩阵的逆就是其转置
t_cam_front_inv = -np.dot(R_cam_front_inv, cam_front_translation)  # 平移部分

# 计算 LIDAR_TOP 到 CAM_FRONT 的转换关系
R_lidar_to_camera = np.dot(R_cam_front_inv, R_lidar)
t_lidar_to_camera = np.dot(R_cam_front_inv, lidar_translation) - t_cam_front_inv

# 输出最终的转换关系
print("LIDAR_TOP 到 CAM_FRONT 的旋转矩阵：\n", R_lidar_to_camera)
print("LIDAR_TOP 到 CAM_FRONT 的平移向量：\n", t_lidar_to_camera)

# 将结果保存到txt文件
output_file = "LIDAR_to_CAMERA_transform.txt"
with open(output_file, 'w') as f:
    f.write("旋转矩阵 (3x3):\n")
    np.savetxt(f, R_lidar_to_camera)
    f.write("\n平移向量 (3x1):\n")
    np.savetxt(f, t_lidar_to_camera)
