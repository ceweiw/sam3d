# pip install nuscenes-devkit               # tingfeng1 需要包 nuscenes-devkit
# pip install numpy==1.22.0 matplotlib==3.5.3
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import numpy as np
import cv2
import os
import ast

'''
这是六个摄像头的结果 目前在目录里面已经生成了 可以看一下到底是干嘛用的  
需要从这个里面提取到点云的转换信息 如何让投射到图像上 然后再做一版
'''


np.set_printoptions(precision=3, suppress=True)

version = "mini"
dataroot = "E:/desktop/nuscene_mini"
nuscenes = NuScenes(version='v1.0-{}'.format(version), dataroot=dataroot, verbose=False)  # tingfeng2 实例化类NuScenes

cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
sample = nuscenes.sample[0]  # tingfeng3获取第一个样本


def to_matrix4x4_2(rotation, translation, inverse=True):
    output = np.eye(4)
    output[:3, :3] = rotation
    output[:3, 3] = translation

    if inverse:
        output = np.linalg.inv(output)
    return output


def to_matrix4x4(m):
    output = np.eye(4)
    output[:3, :3] = m
    return output


# tingfeng4 拿到样本后，获取Lidar的信息
# Lidar的点云是基于Lidar坐标系的
# 需要将lidar的坐标转换到ego，然后ego到global
lidar_sample_data = nuscenes.get('sample_data', sample['data']["LIDAR_TOP"])
lidar_file = os.path.join(nuscenes.dataroot, lidar_sample_data["filename"])
lidar_pointcloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
ego_pose = nuscenes.get('ego_pose', lidar_sample_data['ego_pose_token'])
ego_to_global = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']),
                               False)
lidar_sensor = nuscenes.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
lidar_to_ego = to_matrix4x4_2(Quaternion(lidar_sensor['rotation']).rotation_matrix,
                              np.array(lidar_sensor['translation']), False)
lidar_to_global = ego_to_global @ lidar_to_ego
lidar_points = np.concatenate([lidar_pointcloud[:, :3], np.ones((len(lidar_pointcloud), 1))], axis=1)
lidar_points = lidar_points @ lidar_to_global.T  # tingfeng5 点云转到global坐标系

def show_points(image,points,thickness=2):
    height, width, _ = image.shape
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 遍历点并绘制
    for (x, y) in points:
        # 检查点是否在图像范围内
        if 0 <= x < width and 0 <= y < height:
            rgb_value = image[y, x]
            # 绘制一个小圆圈表示该点
            cv2.circle(blank_image, (x, y), 5, (255, 255, 255), thickness)  # 白色圆圈

    # 显示结果图像
    cv2.imshow('RGB Values at Points', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# tingfeng6 获取每一个camera的信息
# 1. annotation标注是global坐标系的
# 2. 由于时间戳的缘故，每个camera/lidar/radar都有ego_pose用于修正差异
for camera in cameras:
    camera_sample_data = nuscenes.get('sample_data', sample['data'][camera])
    image_file = os.path.join(nuscenes.dataroot, camera_sample_data["filename"])
    ego_pose = nuscenes.get('ego_pose', camera_sample_data['ego_pose_token'])
    global_to_ego = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']))

    camera_sensor = nuscenes.get('calibrated_sensor', camera_sample_data['calibrated_sensor_token'])
    camera_intrinsic = to_matrix4x4(camera_sensor['camera_intrinsic'])
    ego_to_camera = to_matrix4x4_2(Quaternion(camera_sensor['rotation']).rotation_matrix,
                                   np.array(camera_sensor['translation']))

    global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego
    # tingfeng7 根据相机中得到的信息得到的三个矩阵，将global坐标系的坐标转换到image上
    image = cv2.imread(image_file)

    # # 矩形框不要
    # for annotation_token in sample['anns']:  # tingfeng8 标注的框的信息，在sample['anns']中，遍历标注框
    #     instance = nuscenes.get('sample_annotation', annotation_token)
    #     box = Box(instance['translation'], instance['size'], Quaternion(instance['rotation']))
    #     corners = np.ones((4, 8))
    #     corners[:3, :] = box.corners()
    #     corners = (global_to_image @ corners)[:3]  # tingfeng11 注意角点是global坐标，这里转换到image上
    #     corners[:2] /= corners[[2]]
    #     corners = corners.T.astype(int)  # tingfeng9 到此是计算框的角点
    #
    #     ix, iy = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 1, 2, 3, 0, 5, 6, 7, 4]
    #     for p0, p1 in zip(corners[ix], corners[iy]):
    #         if p0[2] <= 0 or p1[2] <= 0: continue
    #         cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (0, 255, 0), 2, 16)  # tingfeng10 到此是把角点划到image上

    # 获取点云
    image_based_points = lidar_points @ global_to_image.T  # tingfeng12 雷达的信息也投射到image上
    image_based_points[:, :2] /= image_based_points[:, [2]]
    show_points = image_based_points[image_based_points[:, 2] > 0][:, :2].astype(np.int32)

    # 黑色背景
    myimage=np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for x, y in show_points:
        try:
            c=image[y,x]
            c_tuple=tuple(map(lambda x:int(x),c))
            print(c_tuple)
            myimage=cv2.circle(myimage, (x, y), 3, c_tuple, -1, 16)  # # tingfeng13 lidar的信息画到image上
        except Exception as e:
            x=x
    cv2.imwrite(f"{camera}_balck.jpg", myimage)

    # # 正常背景
    # myimage2=image
    # for x, y in show_points:
    #     myimage2=cv2.circle(myimage2, (x, y), 3, (255,0,0), -1, 16)  # # tingfeng13 lidar的信息画到image上
    #     if(y >= myimage2.shape[1]):
    #         print(x,"  ",y)
    # cv2.imwrite(f"{camera}.jpg", myimage2)




