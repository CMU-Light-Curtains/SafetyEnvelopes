import zmq
import numpy as np
import scipy.ndimage
from typing import NoReturn, Optional, Tuple

from data.synthia import Frame
from devices.light_curtain import LightCurtain, LCReturn
from envs.env import Env
from lc_planner.config import LCConfig
import utils

ingr = utils.Ingredient("real_env")
ingr.add_config("config/real_env.yaml")


class RealEnv(Env):
    """
    Real-world environment that interfaces with the real light curtain device and LiDAR.
    """
    @ingr.capture
    def __init__(self,
                 use_random_curtain: bool,
                 random_curtain_updates_main_curtain: bool,
                 random_curtain_sampling: str,
                 random_curtain_spacing_power: float,
                 vertical_range: Tuple[float, float],
                 r_hit_intensity_thresh: int,
                 r_recession: float,
                 r_cache_file: str,
                 pp_smoothing: Optional[str],
                 tracking_rtol: float,
                 tracking_atol: float,
                 min_range: float,
                 max_range: float,
                 gt_min_filter_size: int,
                 baseline: dict,
                 debug: bool = False):

        # lc config
        self.lc_config = LCConfig(config_file='config/lc/real.yaml',
                                  default_config_file='LC-PLANNER/apis/py/lc_planner/defaults.yaml')

        # the next three lines establish a connection with the device-side zmq server
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

        # the next five lines request and receive thetas from the ZMQ server
        print("Attempting to get thetas ...")
        self.socket.send_multipart([b"GET_THETAS"], copy=False)
        message = self.socket.recv() # wait for the device to reply with thetas; this is a blocking call
        thetas = np.frombuffer(message, dtype=np.float32).reshape(self.lc_config.CAMERA_PARAMS["width"])  # (C,)
        print("Successfully received thetas!")

        # super init
        super().__init__(lc_config=self.lc_config,
                         thetas=thetas,
                         min_range=min_range,
                         max_range=max_range,
                         use_random_curtain=use_random_curtain,
                         random_curtain_updates_main_curtain=random_curtain_updates_main_curtain,
                         random_curtain_cache_file=r_cache_file,
                         random_curtain_sampling=random_curtain_sampling,
                         random_curtain_spacing_power=random_curtain_spacing_power,
                         vertical_range=vertical_range,
                         r_hit_intensity_thresh=r_hit_intensity_thresh,
                         r_recession=r_recession,
                         pp_smoothing=pp_smoothing,
                         tracking_rtol=tracking_rtol,
                         tracking_atol=tracking_atol,
                         baseline_config=baseline,
                         debug=debug)

        # the next few lines attempt to send random curtain metadata and receive an ACK
        print("Attempting to send random curtain metadata ...")
        self.socket.send_multipart([b"SEND_RC_METADATA", self.rand_curtain_gen.curtains,
                                   np.array([self._r_hit_intensity_thresh], dtype=np.float32)],
                                   copy=False)
        message = self.socket.recv()  # wait for the device to reply with ACK; this is a blocking call
        assert message == b'ACK'
        print("Successfully received ACK!")

        # parameter for min-filtering ground truth ranges obtained from LiDAR
        self._gt_min_filter_size = gt_min_filter_size

        # endregion
        ################################################################################################################

    ####################################################################################################################
    # region Env API functions
    ####################################################################################################################

    def reset(self,
              vid: Optional[int] = None,
              start: Optional[int] = None) -> np.ndarray:
        init_envelope: np.ndarray = self.env_current_gt_action()  # (C,)

        # the initial self.intensities is binary: 1 where objects exist (in front of max range) and 0 otherwise
        self.intensities = (init_envelope < self.max_range).astype(np.float32)  # (C,)

        return init_envelope

    # endregion
    ####################################################################################################################
    # region Env-specific helper functions for step()
    ####################################################################################################################

    def env_step_begin(self) -> NoReturn:
        pass  # real env does not need to do anything in the beginning
    
    def env_place_r_curtain(self) -> Tuple[np.ndarray, np.ndarray]:
        # get unified image of random curtains
        self.socket.send_multipart([b"R_CURTAIN", b""], copy=False)

        # wait for the return of this random curtain sent to image
        u_image = self.socket.recv()
        u_image = np.frombuffer(u_image, dtype=np.float32).reshape(640, 512, 4)

        # 2D mask of pixels where none of (x, y, z, i) are nan and i > threshold
        mask_2d = np.logical_not(np.isnan(u_image).any(axis=2))  # (H, C) (none of x, y, z, i) should be nan
        mask_2d = mask_2d & (u_image[:, :, 3] > self._r_hit_intensity_thresh)  # (H, C)

        xz = u_image[:, :, [0, 2]]  # (H, C, 2)
        ranges = np.linalg.norm(xz, axis=2)  # (H, C)
        ranges[~mask_2d] = np.inf  # replace pixels outside mask with infs before minimization
        row_inds = np.argmin(ranges, axis=0)  # (C,) pixel with point closest to sensor per column

        r_curtain = ranges[row_inds, np.arange(self.C)]  # (C,)
        r_hits = np.isfinite(r_curtain)  # (C,) columns having at least 1 pixel with intensity above threshold

        return r_curtain, r_hits

    def env_place_f_curtain(self,
                            f_curtain: np.ndarray,
                            score: Optional[float]) -> LCReturn:
        # sending forecasting curtain to the real device to image
        ranges = f_curtain.copy()
        self.socket.send_multipart([b"F_CURTAIN", ranges], copy=False)

        # wait for the return of this random curtain sent to image
        lc_image = self.socket.recv()
        lc_image = np.frombuffer(lc_image, dtype=np.float32).reshape(640, 512, 4)

        # Update curtain
        curtain, mask = LightCurtain._validate_curtain_using_lc_image(lc_image)
        ranges[mask] = curtain
        lc_ranges = ranges

        f_return: LCReturn = LCReturn(lc_ranges, lc_image, None, None, None, self._vertical_range)
        return f_return

    def env_current_gt_action(self) -> np.ndarray:
        # the next three lines get a depth map from server
        self.socket.send_multipart([b"LIDARDEPTH", b""])
        message = self.socket.recv()
        depth = np.frombuffer(message, dtype=np.float32).reshape(self.H, self.C)  # (H, C)

        # the next two lines wrap the depth map in a frame
        calib = dict(P2=self.lc_config.CAMERA_PARAMS["matrix"],
                     Tr_velo_to_cam=self.lc_config.TRANSFORMS['world_to_cam'])
        frame = Frame(depth=depth, calib=calib, annos={})

        # the next two lines augment the frame to compute the gt_action
        self.augment_frame_data(frame)
        gt_action = frame.annos["se_ranges"].copy()  # (C,)

        # the next line post-processes the ground truth ranges obtained from LiDAR by applying min-filtering
        gt_action = scipy.ndimage.minimum_filter1d(gt_action, size=self._gt_min_filter_size)  # (C,)

        return gt_action

    def env_end(self) -> bool:
        return False  # episodes in real env don't end

    # endregion
    ####################################################################################################################
