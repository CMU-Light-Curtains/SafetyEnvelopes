#!/usr/bin/python

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
import sys
import time

import rospy
import rospkg
import sensor_msgs.msg

devel_folder = rospkg.RosPack().get_path('params_lib').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-2]) + "/devel/lib/"
sys.path.append(devel_folder)
import params_lib_python
from utils import Stream, vizstream

import signal
def signal_handler(signal, frame):
    print("Ending")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

app = Flask("second")
CORS(app)

########################################################################################################################
# PARAMETERS
########################################################################################################################

LIDAR_CLOUD_FPS = 30
LC_CLOUD_FPS = 25
ROSPY_RATE = 30

# somehow the original image gets downsampled by a factor of 2 in both dimensions
H = 640 // 2
W = 512 // 2

########################################################################################################################

class ROSSub:
    def __init__(self):
        # streams
        self.lidar_cloud_stream = Stream(fps=LIDAR_CLOUD_FPS)
        self.lc_curtain_stream  = Stream(fps=LC_CLOUD_FPS)
        
        global app
        vizstream(app, self.lidar_cloud_stream, astype="scene_cloud")
        vizstream(app, self.lc_curtain_stream,  astype="lc_curtain")

        # initialize node
        rospy.init_node("kv_rossub_backend", anonymous=False)

        # subscriber
        self.lidar_sub = rospy.Subscriber("/ir_optical/points",
                                          sensor_msgs.msg.PointCloud2,
                                          self.lidar_points_callback,
                                          queue_size=1)
        
        self.lc_sub = rospy.Subscriber("/lc_wrapper/image_cloud",
                                       sensor_msgs.msg.PointCloud2,
                                       self.lc_points_callback,
                                       queue_size=1)

        self.rate = rospy.Rate(ROSPY_RATE) # 50hz
        time.sleep(1.0)

    def lidar_points_callback(self, lidar_cloud_msg):
        lidar_points = params_lib_python.convertPointCloud2toXYZ(lidar_cloud_msg)
        lidar_points = lidar_points[:, :3]
        self.lidar_cloud_stream.publish(scene_points=lidar_points, downsample=False)
    
    def lc_points_callback(self, lc_cloud_msg):
        lc_points = params_lib_python.convertPointCloud2toXYZRGB(lc_cloud_msg)  # (H * W, 6)
        lc_image = lc_points[:, :3].reshape([H, W, 3])  # (H, W, 3)
        lc_cloud = lc_points[np.isfinite(lc_points).all(axis=1)][:, [0, 1, 2, 4]]
        lc_cloud[:, 3] /= 255.0
        self.lc_curtain_stream.publish(lc_image=lc_image, lc_cloud=lc_cloud)

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def stop_simulation(self):
        raise NotImplementedError


ROSSUB = ROSSub()


@app.route('/api/run_simulation', methods=['GET', 'POST'])
def run_simulation():
    global ROSSUB
    instance = request.json
    response = {"status": "normal"}
    video_idx = instance["video_idx"]

    ROSSUB.run()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)

main()
