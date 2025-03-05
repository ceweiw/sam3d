import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Tuple, List, Iterable
import os
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from pyquaternion import Quaternion
import numpy as np
from PIL import Image as PILImage
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from typing import List, Tuple
import os.path as osp
from sklearn.neighbors import NearestNeighbors

def save_transformation(R, t,index, save_path):
    # Create the file path to save the transformation data
    file_path = os.path.join(save_path, f'{index}.txt')
    # Open the file in write mode
    with open(file_path, 'w') as f:
        # Write the 3x3 Rotation matrix R and 3x1 Translation vector t as a 4x4 matrix
        # Construct the 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t

        # Write the 4x4 matrix in the desired format
        for row in transformation_matrix:
            f.write('\t'.join([f'{val:.15f}' for val in row]) + '\n')

'''
ransac_plane_fitting、filter_ground_points、densify_lidar_points、densificate_pc是稠密化地面的
densified_points = densificate_pc(points)
new_points = densified_points   
'''
def filter_points_in_bbox(pc, x_min, x_max, y_min, y_max, z_min, z_max):
    # 筛选条件：点的 x, y, z 坐标是否超出边界框的范围
    mask = (pc[0, :] < x_min) | (pc[0, :] > x_max) | \
           (pc[1, :] < y_min) | (pc[1, :] > y_max) | \
           (pc[2, :] < z_min) | (pc[2, :] > z_max)

    # 使用mask筛选出符合条件的点
    filtered_pc = pc[:,mask]
    # inside_bbox_count = np.sum(~mask)  # ~mask表示在边界框内的点
    # print(f"Number of points inside the bounding box: {inside_bbox_count}")
    return filtered_pc


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


def densify_lidar_points(points, max_distance, densification_factor=2):
    filtered_points = points  # 保留这些点

    # 计算所有符合条件的点的密度（可以通过邻近点数目来估算）
    nn = NearestNeighbors(n_neighbors=4)  # 选择4个邻居来估算密度
    nn.fit(filtered_points.T)  # 训练模型
    distances_to_nearest_neighbors, _ = nn.kneighbors(filtered_points.T)  # 获取每个点到最近4个点的距离
    avg_distance = np.mean(distances_to_nearest_neighbors, axis=1)  # 计算每个点的平均距离

    # 根据距离调整稠密化的比例
    density_factors = densification_factor * (1 / (avg_distance + 1e-5))  # 越远的点密度因子越大

    densified_points = []

    # 对每个点根据密度因子进行稠密化
    for i, point in enumerate(filtered_points.T):
        # 通过密度因子来重复生成点
        num_new_points = int(density_factors[i] * 10)  # 生成新点的数量
        new_points = np.tile(point, (num_new_points, 1))  # 重复当前点
        densified_points.append(new_points)

    densified_points = np.vstack(densified_points)  # 合并所有稠密化的点
    print("densified_points.shape",densified_points.shape)
    return densified_points.T

def densificate_pc(points,max_distance=200.0,densification_factor=2):
    # 过滤出地面点云
    plane_params, inliers = ransac_plane_fitting(points)
    ground_points, ground_mask = filter_ground_points(points, plane_params)
    densified_points = densify_lidar_points(ground_points, max_distance, densification_factor)
    non_ground_points = points[:, ~ground_mask]  # 非地面点
    all_points = np.hstack((densified_points, non_ground_points))
    return all_points


def map_pointcloud_to_image(ann_list,base_path,index,pointsensor_token: str,camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False,
                            filter_lidarseg_labels: List = None,
                            lidarseg_preds_bin_path: str = None,
                            show_panoptic: bool = False) -> Tuple:
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        if show_lidarseg or show_panoptic:
            gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
            assert hasattr(nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

            # Ensure that lidar pointcloud is from a keyframe.
            assert pointsensor['is_key_frame'], \
                'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

            assert not render_intensity, 'Error: Invalid options selected. You can only select either ' \
                                         'render_intensity or show_lidarseg, not both.'

        pc = LidarPointCloud.from_file(pcl_path)  #得到的形状是3,N
    else:
        pc = RadarPointCloud.from_file(pcl_path)


    # im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))
    image_path = osp.join(nusc.dataroot, cam['filename'])
    im = Image.open(image_path)
    output_directory = base_path+'/color'
    # image_filename = os.path.basename(image_path)
    img_path = os.path.join(output_directory, f"{index}.jpg")
    im.save(img_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)  # 从pc到自车 提供的是calibrated_sensor获取的数据
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])  # point时刻的自车坐标
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    #去掉3d box
    for ann_token in ann_list:
        ann_rec = nusc.get('sample_annotation', ann_token)
        box = nusc.get_box(ann_rec['token'])

        corners = box.corners()

        # 计算x、y、z的最小/最大值
        x_min, x_max = np.min(corners[0, :]), np.max(corners[0, :])
        y_min, y_max = np.min(corners[1, :]), np.max(corners[1, :])
        z_min, z_max = np.min(corners[2, :]), np.max(corners[2, :])
        filtered_points = filter_points_in_bbox(pc.points, x_min, x_max, y_min, y_max, z_min, z_max)
        # print(f"Before filtering: {pc.points.shape[1]} points")
        pc.points = filtered_points
        # print(filtered_points.shape)
        # print(f"After filtering: {pc.points.shape[1]} points")


    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])  # cam时刻的自车坐标（全局坐标系下）
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    # 上面的是cam时刻的global to ego这里是ego to global
    R_ego_to_global = Quaternion(poserecord['rotation']).rotation_matrix
    t_ego_to_global = np.array(poserecord['translation'])

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    # 上面是ego to camera  这里是camera to ego
    R_cam_to_ego = Quaternion(cs_record['rotation']).rotation_matrix
    t_cam_to_ego = np.array(cs_record['translation'])

    # 旋转矩阵
    R_world_to_camera = R_ego_to_global @ R_cam_to_ego
    # 位移向量
    t_world_to_camera = t_ego_to_global - R_ego_to_global @ t_cam_to_ego
    pose_path = base_path+'/pose'
    save_transformation(R_world_to_camera, t_world_to_camera,index,  pose_path)

    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    elif show_lidarseg or show_panoptic:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']

        gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
        semantic_table = getattr(nusc, gt_from)

        if lidarseg_preds_bin_path:
            sample_token = nusc.get('sample_data', pointsensor_token)['sample_token']
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
        else:
            if len(semantic_table) > 0:  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                lidarseg_labels_filename = osp.join(nusc.dataroot,
                                                    nusc.get(gt_from, pointsensor_token)['filename'])
            else:
                lidarseg_labels_filename = None

        if lidarseg_labels_filename:
            # Paint each label in the pointcloud with a RGBA value.
            if show_lidarseg:
                coloring = paint_points_label(lidarseg_labels_filename,
                                              filter_lidarseg_labels,
                                              nusc.lidarseg_name2idx_mapping,
                                              nusc.colormap)
            else:
                coloring = paint_panop_points_label(lidarseg_labels_filename,
                                                    filter_lidarseg_labels,
                                                    nusc.lidarseg_name2idx_mapping,
                                                    nusc.colormap)

        else:
            coloring = depths
            print(f'Warning: There are no lidarseg labels in {nusc.version}. Points will be colored according '
                  f'to distance from the ego vehicle instead.')
    else:
        # Retrieve the color from the depth.
        coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    print("points.shape",points.shape)
    # print("points.shape",points.shape)
    coloring = coloring[mask]

    #
    '''
    add code to save depth img 
    '''
    depth_image = np.zeros((im.size[1], im.size[0]), dtype=np.float32)
    depth_image[points[1].astype(int), points[0].astype(int)] = depths[mask]
    # depth_image = np.flipud(depth_image)  # Flip the depth image to match the camera frame orientation
    # Ensure output directory exists
    depth_path = base_path+'\depth'
    os.makedirs(depth_path, exist_ok=True)
    # Save depth image
    depth_image_filename = osp.join(depth_path, f'{index}.png')
    depth_image_pil = PILImage.fromarray(np.uint16(depth_image * 1000))  # Scale to millimeters
    depth_image_pil.save(depth_image_filename)

    return points, coloring, im

def myrender_pointcloud_in_image(ann_list,base_path, sample_token: str,
                               index:int,
                               dot_size: int = 5,
                               pointsensor_channel: str = 'LIDAR_TOP',
                               camera_channel: str = 'CAM_FRONT',
                               out_path: str = None,
                               render_intensity: bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels: List = None,
                               ax: Axes = None,
                               show_lidarseg_legend: bool = False,
                               verbose: bool = True,
                               lidarseg_preds_bin_path: str = None,
                               show_panoptic: bool = False):
    if show_lidarseg:
        show_panoptic = False
    sample_record = nusc.get('sample', sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]

    points, coloring, im = map_pointcloud_to_image(ann_list,base_path,index,pointsensor_token, camera_token,
                                                        render_intensity=render_intensity,
                                                        show_lidarseg=show_lidarseg,
                                                        filter_lidarseg_labels=filter_lidarseg_labels,
                                                        lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                        show_panoptic=show_panoptic)

    # Init axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        if lidarseg_preds_bin_path:
            fig.canvas.set_window_title(sample_token + '(predictions)')
        else:
            fig.canvas.set_window_title(sample_token)
    else:  # Set title on if rendering as part of render_sample.
        ax.set_title(camera_channel)
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    ax.axis('off')

    # Produce a legend with the unique colors from the scatter.
    if pointsensor_channel == 'LIDAR_TOP' and (show_lidarseg or show_panoptic) and show_lidarseg_legend:
        # If user does not specify a filter, then set the filter to contain the classes present in the pointcloud
        # after it has been projected onto the image; this will allow displaying the legend only for classes which
        # are present in the image (instead of all the classes).
        if filter_lidarseg_labels is None:
            if show_lidarseg:
                # Since the labels are stored as class indices, we get the RGB colors from the
                # colormap in an array where the position of the RGB color corresponds to the index
                # of the class it represents.
                color_legend = colormap_to_colors(nusc.colormap, nusc.lidarseg_name2idx_mapping)
                filter_lidarseg_labels = get_labels_in_coloring(color_legend, coloring)
            else:
                # Only show legends for all stuff categories for panoptic.
                filter_lidarseg_labels = stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping))

        if filter_lidarseg_labels and show_panoptic:
            # Only show legends for filtered stuff categories for panoptic.
            stuff_labels = set(stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping)))
            filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

        create_lidarseg_legend(filter_lidarseg_labels, nusc.lidarseg_idx2name_mapping, nusc.colormap,
                               loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1.0))

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    # if verbose:
    #     plt.show()


nusc = NuScenes(version='v1.0-mini', dataroot='E:/desktop/nuscene_mini', verbose=True)
my_scene = nusc.scene[1]
first_sample_token = my_scene['first_sample_token']
current_sample_token = first_sample_token
base_path=r"E:\desktop\nuscenes_sam\scene0149_00"
index=1
while current_sample_token != '':
    # Get the current sample
    my_sample = nusc.get('sample', current_sample_token)

    ann_list = my_sample['anns']  # 取出该sample的所有sample_annotation_token

    # Call your rendering function for this sample, with the specified pointsensor channel
    myrender_pointcloud_in_image(ann_list,base_path,my_sample['token'], index, pointsensor_channel='LIDAR_TOP')

    # Get the next sample token
    current_sample_token = my_sample['next']
    index=index+1