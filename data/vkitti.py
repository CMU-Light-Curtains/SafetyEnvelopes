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
import pandas as pd
from data.synthia import Frame, append_xyz_to_depth_map
from data.synthia import box_camera_to_lidar, change_box3d_center_

SCENE_IDS = ['0001', '0002', '0006', '0018', '0020']
SCENE_VARIATIONS = ['clone', \
                    '15-deg-left', \
                    '15-deg-right', \
                    '30-deg-left', \
                    '30-deg-right', \
                    'fog', \
                    'morning', \
                    'overcast', \
                    'rain', \
                    'sunset']

FOLDER_NAMES = {'depth': 'vkitti_1.3.1_depthgt', 'rgb': 'vkitti_1.3.1_rgb',
                'extrinsics': 'vkitti_1.3.1_extrinsicsgt', 'gt_folder': 'vkitti_1.3.1_motgt'}

# INTRINSICS: This is constant for all the collected data as mentioned in the Readme of the dataset
INTRINSICS = np.array([[725, 0, 620.5, 0],
                       [0, 725, 187.0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float32)

class VKittiVideoDataset:
    def __init__(self, split, cam=False, pc=False):
        """
        Format of video_infos: [(video_no, video_info)] where video_info = [{frame_info}]
        """
        video_infos_path = Path(os.environ['DATADIR'])/ "vkitti" / f"vkitti_video_infos_{split}.pkl"
        with open(video_infos_path, 'rb') as f:
            self.video_infos = pickle.load(f)
        
        self.cam = cam
        self.pc = pc

    def __len__(self):
        return len(self.video_infos)
    
    def __getitem__(self, vid):
        """
        Args:
            vid: (int) this is the index of the video, from 0 to len(VkittiVideoDataset) - 1
        """
        video_no, video_info = self.video_infos[vid]
        video = Video(vid, video_no, video_info, self.cam, self.pc)
        return video

########################################################################################################################
# region: Video class
########################################################################################################################
class Video:
    def __init__(self, vid, video_no, video_info, cam=False, pc=False):
        """
        Args:
            vid: (int) video index, from 0 to len(SynthiaVideoDataset) - 1
            video_no: (int) video number, this is the assigned video number in the metadata file
            video_info: [{frame_info}], frame_info dictionary for every frame
            cam: (bool) whether to load camera image
            pc: (bool) whether to compute the point cloud
        """
        self.vid = vid
        self.video_no = video_no
        self.video_info = video_info # list of dictionaries containing path and annotations and calib infos
        self.cam = cam
        self.pc = pc
        self._MAX_DEPTH = 80.0
    
    def __len__(self):
        return len(self.video_info)
    
    def __getitem__(self, fid):
        """
        Args:
            fid: (int) this is the index of the frame, from 0 to len(video)-1
        """
        frame = Frame()
        
        frame_info = self.video_info[fid] # dict
        root_path = Path(os.environ['DATADIR']) / 'vkitti'

        frame.metadata = edict(
            vid=self.vid,
            video_no=self.video_no,
            fid=fid,
            frame_no=frame_info["frame_no"],
            datatype=frame_info["image"]["image_shape"]
        )
        frame.calib = frame_info['calib']

        # --------------------------------------------------------------------------------------------------------------
        # depth
        # --------------------------------------------------------------------------------------------------------------
        depth_path = Path(frame_info['depth']['depth_path'])
        if not depth_path.is_absolute():
            depth_path = root_path / depth_path
        np_depth_image = np.array(Image.open(depth_path)) # (H, W) dtype=int32
        # Convert centimeters (int32) to meters.
        np_depth_image = np_depth_image.astype(np.float32) / 100  # (H, W) dtype=np.float32
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
            np_rgb_image = np.array(Image.open(io.BytesIO(image_str)))  # (H, W, 3)
            np_depth_image = np.concatenate([np_depth_image, np_rgb_image], axis=2)  # (H, W, 4)

            # points in cam frame
            P2 = frame_info['calib']['P2']  # intrinsics matrix
            if P2.shape == (4,4):
                P2 = P2[:3, :3]
            else:
                assert P2.shape == (3,3)
            K = np_depth_image.shape[2] - 1
            app_np_depth_map = append_xyz_to_depth_map(np_depth_image, P2) # (H, W, 3) or (H, W, 6)
            points = app_np_depth_map.reshape(-1, 3 + K)  # (N, 3 + K)

            # points in velo frame
            Tr_velo_to_cam = frame_info['calib']['Tr_velo_to_cam']  # extrinsics matrix
            Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
            xyz1_cam = np.hstack((points[:, :3], np.ones([len(points), 1], dtype=points.dtype)))  # (N, 4)
            xyz1_velo = xyz1_cam @ Tr_cam_to_velo.T  # (N, 4)
            points = np.hstack((xyz1_velo[:, :3], points[:, 3:]))  # (N, 3) or (N, 6)

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
        # change the gt boxes to velodyne coordinate system
        gt_boxes = box_camera_to_lidar(gt_boxes,
                                       r_rect=frame_info["calib"]["R0_rect"],
                                       velo2cam=frame_info["calib"]["Tr_velo_to_cam"]
                                       )
        change_box3d_center_(gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])

        # corners = center_to_corner_box3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, -1])
        # bbox_bounds = [get_bounds(c) for c in corners]
        # aabb = [o3d.geometry.AxisAlignedBoundingBox(b[0], b[1]) for b in bbox_bounds]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(frame.points[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(frame.points[:, 3:]/255.)
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.)
        # o3d.visualization.draw_geometries([pcd, coord]+aabb)

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


def load_vid2metadata_vkitti():
    """
    Format:
    {
        video_no: { "scene_variation" : variation of the scene defined above
                    "scene_id"  : id of the scene defined above
                    "frame_nos": [list of frame_nos]
                  }
    }

    video_no: (int) video number, this is the key
    values:
        scene_id: (str) id of the scene
        scene_variation: (str) variation of the scene
        frame_nos: list(int) list of frame_nos
    """
    VID2METADATA = {}
    vid2metadata_file = Path(__file__).resolve().parent / "VKittiVideoSets" / "vid2metadata.txt"
    with open(vid2metadata_file, 'r') as f:
        lines = [line.rstrip().split() for line in f.readlines()]
        for line in lines:
            video_no, scene_var, scene_id, frame_nos = line[0], line[1], line[2], line[3:]
            VID2METADATA[video_no] = dict(scene_id=scene_id, scene_variation=scene_var, frame_nos=frame_nos)
    
    num_keys = len(VID2METADATA.keys())
    print(f'number of videos are {num_keys}')
    return VID2METADATA

def get_file_path_vkitti(subdir,
                         frame_no,
                         info_type='image_2',
                         file_tail='.png',
                         relative_path=True,
                         exist_check=True):
    root_path = Path(os.environ['DATADIR']) / "vkitti"
    rel_file_path = f"{FOLDER_NAMES[info_type]}/{subdir}/{frame_no}{file_tail}"
    abs_file_path = root_path / rel_file_path
    if exist_check and not abs_file_path.exists():
        raise ValueError(f"file not exists: {abs_file_path}")
    if relative_path:
        return str(rel_file_path)
    else:
        return str(abs_file_path)
    
def get_label_path_vkitti(subdir, root_path, relative_path=True, exist_check=True):
    scene_id, scene_var = subdir.split('/')
    file_path = f'{FOLDER_NAMES["gt_folder"]}/{scene_id}_{scene_var}.txt'
    if exist_check and not (root_path / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(root_path / file_path)
    
def get_vkitti_label_anno(label_path, frame_id):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    content = pd.read_csv(label_path, sep=" ", index_col=False)
    content = content[content['frame'] == int(frame_id)]
    num_objects = len(content) - (content['label'] == 'DontCare').sum()
    annotations['name'] = content['label'].to_numpy().astype(np.str)
    num_gt = len(annotations['name'])
    annotations['truncated'] = content['truncr'].to_numpy()
    annotations['occluded'] = content['occluded'].to_numpy()
    annotations['alpha'] = content['alpha'].to_numpy()
    annotations['bbox'] = content[['l', 't', 'r', 'b']].to_numpy()
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = content[['l3d', 'h3d', 'w3d']].to_numpy()
    annotations['location'] = content[['x3d', 'y3d', 'z3d']].to_numpy()
    annotations['rotation_y'] = content['ry'].to_numpy()
    annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def add_dificulty_to_annos_vkitti(frame_info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = frame_info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
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
    annos["difficulty"] = np.array(diff, np.int32)
    return diff

def get_video_info_vkitti(metadata,
                          extend_matrix=True,
                          num_worker=8,
                          relative_path=True,
                          with_imageshape=True):
    """ 
    Args:
        metadata: {
                    "scene_id"  : id of the scene we are using
                    "scene_var" : variation of the scene id we are using
                    "frame_nos": [list of frame_nos]
                  }
    Returns:
        video_info: list(frame_info), frame_info dictionary for every frame
    """
    scene_id = metadata['scene_id']
    scene_var = metadata['scene_variation']
    frame_nos = metadata['frame_nos']
    root_path = Path(os.environ['DATADIR']) / "vkitti"
    subdir = f"{scene_id}/{scene_var}"
    print(subdir)

    def get_frame_info(frame_no):
        depth_path = get_file_path_vkitti(subdir, frame_no, 'depth', '.png', relative_path)
        rgb_path = get_file_path_vkitti(subdir, frame_no, 'rgb', '.png', relative_path)
        label_path = get_label_path_vkitti(subdir, root_path, relative_path)

        frame_info = {
            'frame_no': frame_no,
            'image': {
                'image_path': rgb_path,
                'image_shape': None,
            },
            'depth': {
                'depth_path': depth_path
            },
            'annos': None,
            'calib': {}
        }

        # image shape
        img_path = frame_info['image']['image_path']
        if relative_path:
            img_path = str(root_path / img_path)
        frame_info['image']['image_shape'] = np.array(skimage.io.imread(img_path).shape[:2], dtype=np.int32)

        # annos
        if relative_path:
            label_path = str(root_path / label_path)
            assert os.path.exists(label_path)

        annotations = get_vkitti_label_anno(label_path, frame_no)
        frame_info['annos'] = annotations
        if annotations is not None:
            # actually annotations are never None, this just return dictionary with empty fields
            add_dificulty_to_annos_vkitti(frame_info)

        # calib, its the same for all the files at least that is what I understand
        P0 = INTRINSICS.copy()
        P1 = INTRINSICS.copy()
        P2 = INTRINSICS.copy()
        P3 = INTRINSICS.copy()
        rect_4x4 = np.eye(4, dtype=np.float32) ## this is not given in the dataset but is done in previous project
        Tr_velo_to_cam = np.array([[0, -1,  0, 0],  # x <- -y
                                   [0,  0, -1, 0],  # y <- -z
                                   [1,  0,  0, 0],  # z <- +x
                                   [0,  0,  0, 1]], dtype=np.float32)
        
        frame_info['calib']['P0'] = P0
        frame_info["calib"]['P1'] = P1
        frame_info["calib"]['P2'] = P2
        frame_info["calib"]['P3'] = P3
        frame_info["calib"]['R0_rect'] = rect_4x4
        frame_info["calib"]['Tr_velo_to_cam'] = Tr_velo_to_cam

        return frame_info

    
    frame_infos = [get_frame_info(frame_no) for frame_no in frame_nos]
    return frame_infos


def create_video_infos_file_vkitti(relative_path=True, split='train'):
    """
    Format of video_infos: [(video_no, video_info)] where video_info = [{frame_info}]
    """
    VID2METADATA = load_vid2metadata_vkitti()
    # gives me for which videos should the video infos be created
    videoset_file = Path(__file__).resolve().parent / "VKittiVideoSets" / f"{split}.txt"

    root_path = Path(os.environ["DATADIR"]) / "vkitti"
    # path to pkl file where the video infos will be stored
    video_infos_pkl_file = root_path / f'vkitti_video_infos_{split}.pkl'

    with open(videoset_file, 'r') as f:
        video_nos = [line.rstrip() for line in f.readlines()]
    
    video_infos = list()
    print("Generating video infos. This may take several minutes")

    for video_no in tqdm.tqdm(video_nos):
        metadata = VID2METADATA[video_no]
        video_info = get_video_info_vkitti(metadata, relative_path=True)
        video_infos.append((video_no, video_info))
    
    print(f"Synthia video infos file is save to {video_infos_pkl_file}")
    with open(video_infos_pkl_file, 'wb') as f:
        pickle.dump(video_infos, f)

########################################################################################################################
# region: Generates the metadata file which contains information about videos to load
########################################################################################################################
def create_videoset_files_vkitti():
    """
    Each row in the vid2metadata file has the following columns:

    video_no  sc_var  sc_id        frame_nos
    0001      clone   0001   000101 000102 ... 000280

    video_no  : video number (from 0 to TOTAL_VIDEOS_IN_SYNTHIA-1)
    sc_id     : one of the 5 scene ids of vkitti
    sc_var    : one of the 10 variations present in vkitti
    frame_nos : number of the subdirectory. files are named using this number.
    """
    root_path = Path(os.environ['DATADIR']) / "vkitti"
    videosets_dir = Path(__file__).resolve().parent / "VKittiVideoSets"

    video_no = 0
    with open(videosets_dir / "vid2metadata.txt", "w") as f:
        for sc_var in SCENE_VARIATIONS:
            for sc_id in SCENE_IDS:
                to_write = str(video_no).rjust(4, '0')
                print(to_write)
                # get all the frame numbers in this video
                rgb_dir = root_path / FOLDER_NAMES['rgb_folder'] / sc_id / sc_var
                depth_dir = root_path / FOLDER_NAMES['depth_folder'] / sc_id / sc_var

                # check they contain the same number of files
                len_rgb = sum([1 for x in rgb_dir.iterdir()])
                len_depth = sum([1 for x in depth_dir.iterdir()])
                assert len_rgb == len_depth, "Number of files in depth should equal that in rgb"

                frame_nos = list()
                for imfile in sorted(rgb_dir.iterdir()):
                    frame_no = imfile.name.split('.')[0]
                    frame_nos.append(frame_no)
                # write the line in the file
                row_text = ' '.join([to_write, sc_var, sc_id] + frame_nos)
                print(row_text, file=f)
                video_no += 1

########################################################################################################################
# region: Utility
########################################################################################################################

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 0.5, 0.5),
                           axis=2):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def get_bounds(bbox_corners):
    """
    args:
        - bbox_corners: np.ndarray (8, 3)
    returns:
        - bounds: np.ndarray (2, 3)
    """
    min_bounds = np.min(bbox_corners, axis=0)
    max_bounds = np.max(bbox_corners, axis=0)
    bounds = np.stack((min_bounds, max_bounds))
    return bounds
    
if __name__ == '__main__':
    fire.Fire()
    # instance = VKittiVideoDataset(split='test_vkitti', cam=True, pc=True)
    # import ipdb; ipdb.set_trace()
    # video = instance[2]
    # print(video[0])
    # print(video[10])