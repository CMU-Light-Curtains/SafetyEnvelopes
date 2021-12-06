# This file is compatible with both Python 2 and 3

import base64
import cv2
import json
import numpy as np
from flask import Response
import time
import functools
from collections import deque


class Stream(deque):
    """
    A stream stores an output sequence stream of data. It inherits from deque.
    Stream contains oldest-newest data from left to right.
    Stream has a "capacity" -- if the stream is full, it will drop the oldest data.
    vizstream will monitor the stream and will send data to the browser whenever the
    stream is updated and its timestamp changes.
    """
    def __init__(self, capacity=1, fps=10):
        """
        Args:
            capacity: (int) maximum capacity of stream.
        """
        self.capacity = capacity
        self.fps = fps
        self.timestamp = 0
        super(Stream, self).__init__(maxlen=self.capacity)

    def publish(self, **kwargs):
        item = dict(data=kwargs, timestamp=self.timestamp)
        self.timestamp += 1
        self.append(item)
    
    def reset(self):
        self.clear()
        self.timestamp = 0


def vizstream(app, stream, astype):
    if astype == 'scene_cloud':
        url = '/api/stream_scene_cloud'
        data2msg = data2msg_scene_cloud
    elif astype == 'lc_curtain':
        url = '/api/stream_lc_curtain'
        data2msg = data2msg_lc_curtain
    # elif astype == 'camera_image':
    #     url = '/api/stream_camera_image'
    #     data2msg = data2msg_camera_image
    # elif astype == 'lidar_cloud':
    #     url = '/api/stream_lidar_cloud'
    #     data2msg = data2msg_lidar_cloud
    # elif astype == 'dt_boxes':
    #     url = '/api/stream_dt_boxes'
    #     data2msg = data2msg_dt_boxes
    # elif astype == 'entropy_map':
    #     url = '/api/stream_entropy_map'
    #     data2msg = data2msg_entropy_map
    # elif astype == 'arrows':
    #     url = '/api/stream_arrows'
    #     data2msg = data2msg_arrows
    else:
        raise Exception("astype={} not valid".format(astype))

    def generator():
        sent_timestamp = None
        while True:
            if len(stream) == 0:
                sent_timestamp = None
            elif sent_timestamp != stream[-1]["timestamp"]:
                sent_timestamp = stream[-1]["timestamp"]
                
                data = stream[-1]["data"]
                msg = data2msg(**data)
                yield "data:{}\n\n".format(msg)

            time.sleep(1.0 / stream.fps)

    @app.route(url, methods=['GET', 'POST'])
    @functools.wraps(data2msg)
    def route_fn():
        return Response(generator(), mimetype="text/event-stream")


########################################################################################################################
# region data2msg functions
########################################################################################################################

def data2msg_scene_cloud(scene_points, se_design_pts=None, downsample=False, int16_factor=100):
    """
    Args:
        scene_points (np.ndarray, dtype=float32, shape=(N, 6)): scene points
        se_design_points (Optional(np.ndarray, dtype=float32, shape=(C, 2))): design points of the safety envelope
    """
    # the next line downsamples the scene points. it selects one from every three points.
    if downsample:
        scene_points = scene_points[::3, :]

    # convert to int16
    scene_points = scene_points * int16_factor
    scene_points = scene_points.astype(np.int16)

    scene_pc_str = base64.b64encode(scene_points.tobytes()).decode("utf-8")
    send_dict = dict(scene_pc_str=scene_pc_str)

    if se_design_pts is not None:
        # convert to int16
        se_design_pts = se_design_pts * int16_factor
        se_design_pts = se_design_pts.astype(np.int16)

        se_pc_str = base64.b64encode(se_design_pts.tobytes()).decode("utf-8")
        send_dict["se_pc_str"] = se_pc_str

    json_str = json.dumps(send_dict)
    return json_str


# def data2msg_camera_image(data: Frame):
#     image_str = data.cam["image_str"]
#     image_dtype = data.cam["datatype"]
#     image_b64 = base64.b64encode(image_str).decode("utf-8")
#     image_b64 = f"data:image/{image_dtype};base64,{image_b64}"
#     return image_b64


# def data2msg_lidar_cloud(data, int16_factor=100):
#     points = data  # (N, 3)

#     # convert to int16
#     points = points * int16_factor
#     points = points.astype(np.int16)

#     pc_str = base64.b64encode(points.tobytes()).decode("utf-8")
#     return pc_str


def data2msg_lc_curtain(lc_image, lc_cloud, score=None, int16_factor=100):
    """
    Args:
        lc_image: light curtain image.
            - (np.ndarray, dtype=float32, shape=(H, C, 4)).
            - Axis 2 corresponds to (x, y, z, i):
                    - x : x in cam frame.
                    - y : y in cam frame.
                    - z : z in cam frame.
                    - i : intensity of LC cloud, lying in [0, 255].
        
        lc_cloud: light curtain point cloud.
            - (np.ndarray, dtype=float32, shape=(N, 4)).
            - Axis 2 corresponds to (x, y, z, i):
                    - x : x in cam frame.
                    - y : y in cam frame.
                    - z : z in cam frame.
                    - i : intensity of LC cloud, lying in [0, 1].
        score (Optional(float)): score to be displayed in kittiviewer
    """
    # boundary
    lc_image = lc_image[:, :, :3]  # (H, C, 3)
    ys = lc_image[:, :, 1]  # (H, C)
    ys[np.isnan(ys)] = 0  # replacing NaNs with zeros shouldn't affect the columnwise min or max of y
    top_inds = np.argmin(ys, axis=0)  # (C,)
    bot_inds = np.argmax(ys, axis=0)  # (C,)
    top_xyz = lc_image[top_inds, np.arange(len(top_inds)), :]  # (C, 3)
    bot_xyz = lc_image[bot_inds, np.arange(len(bot_inds)), :]  # (C, 3)
    boundary = np.stack([top_xyz, bot_xyz], axis=1)  # (C, 2, 3)
    mask = np.isfinite(boundary).all(axis=(1, 2))  # (C,)
    boundary = boundary[mask]  # (C', 2, 3)

    # intersection points
    isect_pts = lc_cloud[lc_cloud[:, 3] > 0.05]  # (N', 4)

    # convert to int16
    boundary = (boundary * int16_factor).astype(np.int16)
    isect_pts = (isect_pts * int16_factor).astype(np.int16)

    boundary_str = base64.b64encode(boundary.tobytes()).decode("utf-8")
    isect_pts_str = base64.b64encode(isect_pts.tobytes()).decode("utf-8")
    send_dict = dict(boundary=boundary_str, isect_pts=isect_pts_str, score=score)
    json_str = json.dumps(send_dict)
    return json_str


# def data2msg_dt_boxes(data):
#     dt_boxes = data["detections"]
#     json_str = json.dumps(dt_boxes)
#     return json_str


# def data2msg_entropy_map(data):
#     confidence_map = data["confidence_map"]
#     entropy_heatmap = _create_entropy_heatmap(confidence_map)

#     image_str = cv2.imencode('.png', entropy_heatmap)[1].tostring()
#     image_b64 = base64.b64encode(image_str).decode("utf-8")
#     image_b64 = f"data:image/png;base64,{image_b64}"
#     return image_b64


# def data2msg_arrows(data):
#     tails = list([float(e) for e in data["tails"].ravel()])
#     heads = list([float(e) for e in data["heads"].ravel()])
#     arrows = dict(tails=tails, heads=heads)
#     json_str = json.dumps(arrows)
#     return json_str


# endregion
########################################################################################################################
# region Helper functions
########################################################################################################################

# def _create_confidence_heatmap(confidence_map):
#     # Take the mean of confidences for the 0-degrees and 90-degrees anchors
#     conf_scores = confidence_map[:, :, 2:]  # (Y, X, K)
#     conf_scores = conf_scores.mean(axis=2)  # (Y, X)

#     # Rescale between 0 and 1.
#     # conf_scores = conf_scores - conf_scores.min()
#     # conf_scores = conf_scores / conf_scores.max()

#     heatmap = cv2.applyColorMap((conf_scores * 255).astype(np.uint8), cv2.COLORMAP_HOT)
#     return heatmap


# def _create_entropy_heatmap(confidence_map):
#     p = confidence_map[:, :, 2:]  # (Y, X, K)
#     p = p.clip(1e-5, 1-1e-5)  # (Y, X, K)
#     entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)  # (Y, X, K)
#     entropy = entropy.mean(axis=2)  # (Y, X)

#     heatmap = cv2.applyColorMap((entropy * 255).astype(np.uint8), cv2.COLORMAP_HOT)
#     return heatmap

# endregion
########################################################################################################################
