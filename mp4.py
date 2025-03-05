from nuscenes.nuscenes import NuScenes
import cv2
from typing import Tuple, List, Iterable
import os.path as osp
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix


def get_color(nusc, category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """

    return nusc.colormap[category_name]

def render_scene_channel(nusc,
                         scene_token: str,
                         channel: str = 'CAM_FRONT',
                         freq: float = 10,
                         imsize: Tuple[float, float] = (640, 360),
                         out_path: str = None) -> None:
    """
    Renders a full scene for a particular camera channel.
    :param scene_token: Unique identifier of scene to render.
    :param channel: Channel to render.
    :param freq: Display frequency (Hz).
    :param imsize: Size of image to render. The larger the slower this will run.
    :param out_path: Optional path to write a video file of the rendered frames.
    """
    valid_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    assert imsize[0] / imsize[1] == 16 / 9, "Error: Aspect ratio should be 16/9."
    assert channel in valid_channels, 'Error: Input channel {} not valid.'.format(channel)

    if out_path is not None:
        assert osp.splitext(out_path)[-1] == '.avi' or osp.splitext(out_path)[-1] == '.mp4'

    # Get records from DB.
    scene_rec = nusc.get('scene', scene_token)
    sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', sample_rec['data'][channel])

    # Open CV init.
    name = '{}: {} (Space to pause, ESC to exit)'.format(scene_rec['name'], channel)
    cv2.namedWindow(name)
    cv2.moveWindow(name, 0, 0)

    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, freq, imsize)
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # out = cv2.VideoWriter(out_path, fourcc, freq, imsize)
    else:
        out = None

    has_more_frames = True
    while has_more_frames:

        # Get data from DB.
        impath, boxes, camera_intrinsic = nusc.get_sample_data(sd_rec['token'],
                                                                    box_vis_level=BoxVisibility.ANY)

        # Load and render.
        if not osp.exists(impath):
            raise Exception('Error: Missing image %s' % impath)
        im = cv2.imread(impath)
        for box in boxes:
            c = get_color(nusc,box.name)
            # box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Render.
        im = cv2.resize(im, imsize)
        cv2.imshow(name, im)
        if out_path is not None:
            out.write(im)

        key = cv2.waitKey(10)  # Images stored at approx 10 Hz, so wait 10 ms.
        if key == 32:  # If space is pressed, pause.
            key = cv2.waitKey()

        if key == 27:  # If ESC is pressed, exit.
            cv2.destroyAllWindows()
            break

        if not sd_rec['next'] == "":
            sd_rec = nusc.get('sample_data', sd_rec['next'])
        else:
            has_more_frames = False

    cv2.destroyAllWindows()
    if out_path is not None:
        out.release()

def myrender_scene_channel(nusc,scene_token: str, channel: str = 'CAM_FRONT', freq: float = 10,
                         imsize: Tuple[float, float] = (640, 360), out_path: str = None) -> None:
    render_scene_channel(nusc,scene_token, channel=channel, freq=freq, imsize=imsize, out_path=out_path)


nusc = NuScenes(version='v1.0-mini', dataroot='E:/desktop/nuscene_mini', verbose=True)
# print(nusc)
# print(nusc.list_scenes())
table_name, field, query = 'scene', 'name', 'scene-0553'
print("field2token", nusc.field2token(table_name, field, query))
my_scene_token = nusc.field2token(table_name, field, query)[0]
print("token", my_scene_token)
myrender_scene_channel(nusc,my_scene_token, 'CAM_FRONT', out_path=f"{query}.mp4")


