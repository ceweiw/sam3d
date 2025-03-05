
from nuscenes.nuscenes import NuScenes

'''
这个是直接调用生成的 颜色表示深度信息   也可以从这里查看怎么投射到图像上
'''
nusc = NuScenes(version='v1.0-mini', dataroot='E:/desktop/nuscene_mini', verbose=True)
# nusc.list_scenes()
#
# my_scene = nusc.scene[0]
#
# first_sample_token = my_scene['first_sample_token']
#
# my_sample = nusc.get('sample', first_sample_token)
#
# nusc.list_sample(my_sample['token'])
#
# sensor = 'CAM_FRONT'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
#
# nusc.render_sample_data(cam_front_data['token'])
#
# my_annotation_token = my_sample['anns'][18]
# my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
#
# nusc.render_annotation(my_annotation_token)
#
# my_instance = nusc.instance[599]
#
# instance_token = my_instance['token']
# nusc.render_instance(instance_token)
my_scene = nusc.scene[0]
my_sample = nusc.sample[1]
scene_token = my_sample['scene_token']
# scene = nusc.get('scene', scene_token)
# nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
print("scene_token",scene_token)
#render_pointcloud_in_image在这个函数实现的 调用了map_pointcloud_to_image

#1、2  cc8c0bf57f984915a77078b10eb33198