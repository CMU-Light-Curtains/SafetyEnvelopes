import concurrent.futures as futures
from dataclasses import dataclass

from easydict import EasyDict as edict
import io
import numpy as np
import os
from pathlib import Path
import pickle
import tqdm
from typing import Optional, NamedTuple
from PIL import Image
import fire
import skimage.io


class SynthiaVideoDataset:
    def __init__(self, split, cam=False, pc=False):
        """
        Format of video_infos: [(video_no, video_info)] where video_info = [{frame_info}]
        """
        video_infos_path = Path(os.environ["DATADIR"]) / "synthia" / f"synthia_video_infos_{split}.pkl"
        with open(video_infos_path, 'rb') as f:
            self.video_infos = pickle.load(f)

        self.cam = cam
        self.pc = pc

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, vid):
        """
        Args:
            vid: (int) this is the index of the video, from 0 to len(SynthiaVideoDataset) - 1
        """
        video_no, video_info = self.video_infos[vid]
        video = Video(vid, video_no, video_info, self.cam, self.pc)
        return video


########################################################################################################################
# region: Frame class
########################################################################################################################

@dataclass
class Frame:
    depth: Optional[np.ndarray] = None  # np.ndarray, dtype=np.float32, shape=(H, C)
    points: Optional[np.ndarray] = None  # np.ndarray, dtype=np.float32, shape=(N, 3) or (N, 6)
    cam: Optional[edict] = None  # image_str (str): image string, datatype (str): suffix type
    metadata: Optional[edict] = None  # fid (int): frame id, frame_no (str): frame number, datatype (str): datatype
    calib: Optional[dict] = None
    annos: Optional[dict] = None


########################################################################################################################
# region: Video class
########################################################################################################################
class Video:
    def __init__(self, vid, video_no, video_info, cam=False, pc=False):
        """
        Args:
            vid: (int) video index, from 0 to len(SynthiaVideoDataset) - 1
            video_no: (int) video number
            video_info: [{frame_info}], frame_info dictionary for every frame
            cam: (bool) whether to load camera image
            pc: (bool) whether to compute the point cloud
        """
        self.vid = vid
        self.video_no = video_no
        self.video_info = video_info
        self.cam = cam
        self.pc = pc
        self._MAX_DEPTH = 80.0  # only consider points within this depth

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, fid):
        """
        Args:
            fid: (int) this is the index of the frame, from 0 to len(Video) - 1
        """
        frame = Frame()

        frame_info = self.video_info[fid]  # frame info
        root_path = Path(os.environ["DATADIR"]) / "synthia"

        frame.metadata = edict(
            vid=self.vid,
            video_no=self.video_no,
            fid=fid,
            frame_no=frame_info["frame_no"],
            datatype=frame_info["image"]["image_shape"]
        )
        frame.calib = frame_info["calib"]

        # --------------------------------------------------------------------------------------------------------------
        # depth
        # --------------------------------------------------------------------------------------------------------------
        depth_path = Path(frame_info["depth"]["depth_path"])
        if not depth_path.is_absolute():
            depth_path = root_path / depth_path
        # synthia depth formula: "Depth = 5000 * (R + G*256 + B*256*256) / (256*256*256 - 1)"
        np_depth_image = np.array(Image.open(depth_path))  # (H, W, 4) dtype=np.uint8
        R, G, B = [np_depth_image[:, :, e].astype(np.int64) for e in range(3)]  # (H, W), dtype=np.int64
        np_depth_image = 5000 * (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)  # (H, W) dtype=np.float64
        np_depth_image = np_depth_image.astype(np.float32)  # (H, W) dtype=np.float32
        frame.depth = np_depth_image

        # --------------------------------------------------------------------------------------------------------------
        # cam
        # --------------------------------------------------------------------------------------------------------------
        if self.cam or self.pc:
            image_path = Path(frame_info['image']['image_path'])
            if not image_path.is_absolute():
                image_path = root_path / image_path
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            frame.cam = edict(image_str=image_str, datatype=image_path.suffix[1:])

        # --------------------------------------------------------------------------------------------------------------
        # points
        # --------------------------------------------------------------------------------------------------------------
        np_depth_image = np_depth_image[..., np.newaxis]  # (H, W, 1)
        if self.pc:
            # concatenate depth map with colors
            np_rgb_image = np.array(Image.open(io.BytesIO(image_str)))  # (H, W, 4)
            np_rgb_image = np_rgb_image[:, :, :3]  # (H, W, 3)
            np_depth_image = np.concatenate([np_depth_image, np_rgb_image], axis=2)  # (H, W, 4)

            # points in cam frame
            P2 = frame_info['calib']['P2']  # intrinsics matrix
            if P2.shape == (4, 4):
                P2 = P2[:3, :3]
            else:
                assert P2.shape == (3, 3)
            K = np_depth_image.shape[2] - 1
            app_np_depth_map = append_xyz_to_depth_map(np_depth_image, P2)  # (H, W, 3) or (H, W, 6)
            points = app_np_depth_map.reshape(-1, 3 + K)  # (N, 3 + K)

            # points within MAX_DEPTH
            points = points[points[:, 0] < self._MAX_DEPTH, :]  # (M, 3) or (M, 6)

            frame.points = points

        # --------------------------------------------------------------------------------------------------------------
        # annos
        # --------------------------------------------------------------------------------------------------------------
        annos = frame_info['annos']
        annos = self._remove_dontcare(annos)
        locs = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_boxes = box_camera_to_lidar(gt_boxes,
                                       r_rect=frame_info["calib"]["R0_rect"],
                                       velo2cam=frame_info["calib"]["Tr_velo_to_cam"]
                                       )
        # only center format is allowed. so we need to convert kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
        change_box3d_center_(gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])
        annos = {
            'names': gt_names,
            'boxes': gt_boxes,
            'boxes2d': annos["bbox"]
        }
        frame.annos = annos

        return frame

    @staticmethod
    def _remove_dontcare(annos):
        filtered_annos = {}
        relevant_annotation_indices = [i for i, x in enumerate(annos['name']) if x != "DontCare"]
        for key in annos.keys():
            filtered_annos[key] = (annos[key][relevant_annotation_indices])
        return filtered_annos


# endregion
########################################################################################################################
# region: Functions to create videoset files (data/SynthiaVideoSets) and load vid2metadata
########################################################################################################################

def create_videoset_files():
    """
    Each row in the vid2metadata file has the following columns:

    video_no  subdir                                      frame_nos
    0001      train/test5_10segs_weather...2018_12-47-37  000101 000102 ... 000280

    video_no  : video number (from 0 to TOTAL_VIDEOS_IN_SYNTHIA-1)
    subdir    : synthia subdirectory
    frame_nos : number of the subdirectory. files are named using this number.
    """
    root_path = Path(os.environ["DATADIR"]) / "synthia"
    videosets_dir = Path(__file__).resolve().parent / "SynthiaVideoSets"

    metadata, train_video_nos, test_video_nos = [], [], []

    train_dir = root_path / "train"
    test_dir = root_path / "test"
    for ttdir in [train_dir, test_dir]:
        for updir in sorted([e for e in ttdir.iterdir() if e.is_dir()]):
            for subdir in sorted([e for e in updir.iterdir() if e.is_dir()]):
                if not (subdir / "labels_kitti").is_dir():
                    continue  # empty data.
                frame_nos = []
                for imfile in sorted((subdir / "labels_kitti").iterdir()):
                    # Load all frames in the scene, whether they contain a Car or not.
                    frame_no = imfile.name.split('.')[0]
                    frame_nos.append(frame_no)
                metadata.append((subdir, frame_nos))

    with open(videosets_dir / "vid2metadata.txt", 'w') as f:
        for video_no, metadatum in enumerate(metadata):
            video_no = str(video_no).rjust(4, '0')
            subdir, frame_nos = metadatum
            subdir = str(subdir.relative_to(root_path))
            if subdir.startswith('train/'):
                train_video_nos.append(video_no)
            else:
                test_video_nos.append(video_no)
            rowtext = ' '.join([video_no, subdir] + frame_nos)
            print(rowtext, file=f)

    print(f'Synthia TRAIN: has {len(train_video_nos)} videos.')
    print(f'Synthia TEST : has {len(test_video_nos)} videos.')

    np.random.seed(0)

    with open(videosets_dir / "train.txt", 'w') as f:
        for video_no in np.random.permutation(train_video_nos):
            print(video_no, file=f)

    with open(videosets_dir / "test.txt", 'w') as f:
        for video_no in np.random.permutation(test_video_nos):
            print(video_no, file=f)


def load_vid2metadata():
    """
    Format:
    {
        video_no: {  // metadata
                    "subdir"  : subdirectory
                    "frame_nos": [list of frame_nos]
                  }
    }

    video_no: (int) video number
    subdirectory: (str) path to subdirectory
    frame_nos: list(int) list of frame_nos
    """
    VID2METADATA = {}
    vid2metadata_file = Path(__file__).resolve().parent / "SynthiaVideoSets" / "vid2metadata.txt"
    with open(vid2metadata_file, 'r') as f:
        lines = [line.rstrip().split() for line in f.readlines()]
        for line in lines:
            video_no, subdir, frame_nos = line[0], line[1], line[2:]
            VID2METADATA[video_no] = {'subdir': subdir, 'frame_nos': frame_nos}
    return VID2METADATA


# endregion
########################################################################################################################
# region: Functions to create video_infos pickle files in $DATASET/synthia
########################################################################################################################

def get_file_path(subdir,
                  frame_no,
                  info_type='image_2',
                  file_tail='.png',
                  relative_path=True,
                  exist_check=True):
    root_path = Path(os.environ["DATADIR"]) / "synthia"
    rel_file_path = f"{subdir}/{info_type}/{frame_no}{file_tail}"
    abs_file_path = root_path / rel_file_path
    if exist_check and not abs_file_path.exists():
        raise ValueError("file not exist: {}".format(abs_file_path))
    if relative_path:
        return str(rel_file_path)
    else:
        return str(abs_file_path)


def get_synthia_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'difficulty': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()

    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
        -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)

    # Adding difficulty to annotations.
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    dims = annotations['dimensions']  # lhw format
    bbox = annotations['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annotations['occluded']
    truncation = annotations['truncated']
    diff = []
    easy_mask = np.ones((len(dims),), dtype=np.bool)
    moderate_mask = np.ones((len(dims),), dtype=np.bool)
    hard_mask = np.ones((len(dims),), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annotations["difficulty"] = np.array(diff, np.int32)

    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_video_info(metadata,
                   extend_matrix=True,
                   num_worker=8,
                   relative_path=True,
                   with_imageshape=True):
    """ 
    Args:
        metadata: {
                    "subdir"  : subdirectory
                    "frame_nos": [list of frame_nos]
                  }
    Returns:
        video_info: list(frame_info), frame_info dictionary for every frame
    """
    subdir = metadata["subdir"]
    frame_nos = metadata["frame_nos"]
    root_path = Path(os.environ["DATADIR"]) / "synthia"

    def get_frame_info(frame_no):
        depth_path = get_file_path(subdir, frame_no, 'Depth', '.png', relative_path)
        image_path = get_file_path(subdir, frame_no, 'RGB', '.png', relative_path)
        label_path = get_file_path(subdir, frame_no, 'labels_kitti', '.txt', relative_path)
        calib_path = get_file_path(subdir, frame_no, 'calib_kitti', '.txt', relative_path=False)

        frame_info = {
            'frame_no': frame_no,
            'image': {
                'image_path': image_path,
                'image_shape': None
            },
            'depth': {
                'depth_path': depth_path
            },
            'annos': None,
            'calib': {

            }
        }

        # image shape
        img_path = frame_info['image']['image_path']
        if relative_path:
            img_path = str(root_path / img_path)
        frame_info['image']['image_shape'] = np.array(skimage.io.imread(img_path).shape[:2], dtype=np.int32)

        # annos
        if relative_path:
            label_path = str(root_path / label_path)
        annotations = get_synthia_label_anno(label_path)
        frame_info['annos'] = annotations

        # calib
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
        P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
        P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
        P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape([3, 4])
        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)
        R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect

        Tr_velo_to_cam = np.array([[0, -1,  0, 0],  # x <- -y
                                   [0,  0, -1, 0],  # y <- -z
                                   [1,  0,  0, 0],  # z <- +x
                                   [0,  0,  0, 1]], dtype=np.float32)
        frame_info["calib"]['P0'] = P0
        frame_info["calib"]['P1'] = P1
        frame_info["calib"]['P2'] = P2
        frame_info["calib"]['P3'] = P3
        frame_info["calib"]['R0_rect'] = rect_4x4
        frame_info["calib"]['Tr_velo_to_cam'] = Tr_velo_to_cam

        return frame_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        frame_infos = executor.map(get_frame_info, frame_nos)
    return list(frame_infos)


def create_video_infos_file(relative_path=True, split='train'):
    """
    Format of video_infos: [(video_no, video_info)] where video_info = [{frame_info}]
    """
    VID2METADATA = load_vid2metadata()
    videoset_file = Path(__file__).resolve().parent / "SynthiaVideoSets" / f"{split}.txt"

    root_path = Path(os.environ["DATADIR"]) / "synthia"
    video_infos_pkl_file = root_path / f'synthia_video_infos_{split}.pkl'

    with open(videoset_file, 'r') as f:
        video_nos = [line.rstrip() for line in f.readlines()]

    video_infos = []
    print("Generating video infos. This may take several minutes.")
    for video_no in tqdm.tqdm(video_nos):
        metadata = VID2METADATA[video_no]
        video_info = get_video_info(metadata, relative_path=True)
        video_infos.append((video_no, video_info))

    print(f"Synthia video infos file is saved to {video_infos_pkl_file}")
    with open(video_infos_pkl_file, 'wb') as f:
        pickle.dump(video_infos, f)


# endregion
########################################################################################################################
# region: Utility functions copied from pytorch.second repository
########################################################################################################################
def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)


########################################################################################################################
# region: Converting depth map to point cloud
########################################################################################################################
def append_xyz_to_depth_map(depth_map, cam_matrix):
    """
    Note: (1) Output point cloud is in CAMERA FRAME.
          Project points to VELO FRAME first using inverse of camera extrinsics.
          (2) Camera frame is as follows:
              - The focal point is (0, 0, 0).
              - +ve X and +ve Y axes go from left-to-right and top-to-bottom
                parallel the film respectively.
              - The imaging plane is placed parallel to the XY plane and 
                the +ve Z axis points towards the scene. 
    Args:
        depth_map: (np.ndarray, np.float32, shape=(H, W, 1 + K))
                   An image that contains at-least one channel for z-values.
                   It may contain K additional channels such as colors,
                   which will be appended to the point vector.
        cam_matrix: (np.ndarray, np.float32, shape=(3, 3)) camera intrinsics matrix.
    Returns:
        app_depth_map: (np.ndarray, np.float32, shape=(H, W, 3 + K)) points.
                       Axis 2: the first three channels correspond to x, y, z in the CAMERA frame.
                               The next K channels are copied over from depth_map (see description). 
    """
    H, W = depth_map.shape[:2]

    # Invert the steps of projection:
    #   1. Compute x, y in projected coordinates.
    #   2. Inverse the homographic division by z; multiply by z.
    #   3. Multiply by inverse of cam_matrix.
    y, x = np.mgrid[0:H, 0:W] + 0.5  # x and y are both (H, W)
    z = depth_map[:, :, 0]  # (H, W)
    x, y = x * z, y * z  # (H, W)
    xyz = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    xyz = xyz @ np.linalg.inv(cam_matrix).T  # (H, W, 3)
    # Now xyz are in the camera frame.

    app_depth_map = np.concatenate((xyz, depth_map[:, :, 1:]), axis=2)  # (H, W, 3 + K)
    return app_depth_map


########################################################################################################################


if __name__ == "__main__":
    fire.Fire()
