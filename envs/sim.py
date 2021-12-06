from flask import Flask
import numpy as np
import random
from typing import Optional, Tuple, NoReturn

from data.synthia import SynthiaVideoDataset, Frame
from devices.light_curtain import LightCurtain, LCReturn
from envs.env import Env
from kittiviewer.backend.utils import Stream, vizstream
import utils

ingr = utils.Ingredient("sim_env")
ingr.add_config("config/sim_env.yaml")


class SimEnv(Env):
    """
    Simulated environment using the SYNTHIA dataset.
    """
    @ingr.capture
    def __init__(self,
                 interval: int,
                 use_random_curtain: bool,
                 random_curtain_updates_main_curtain: bool,
                 random_curtain_sampling: str,
                 random_curtain_spacing_power: float,
                 vertical_range: Tuple[float, float],
                 r_hit_intensity_thresh: int,
                 r_recession: float,
                 pp_smoothing: Optional[str],
                 tracking_rtol: float,
                 tracking_atol: float,
                 baseline: dict,
                 split: str,
                 cam: bool = False,
                 pc: bool = False,
                 preload: bool = False,
                 progress: bool = False,
                 vis_app: Optional[Flask] = None,
                 debug: bool = False):

        # light curtain
        self.light_curtain = LightCurtain()

        # super init
        super().__init__(lc_config=self.light_curtain.lc_config,
                         thetas=self.light_curtain.thetas,
                         min_range=self.light_curtain.min_range,
                         max_range=self.light_curtain.max_range,
                         use_random_curtain=use_random_curtain,
                         random_curtain_updates_main_curtain=random_curtain_updates_main_curtain,
                         random_curtain_cache_file="cache/rand_curtains/sim_env.txt",
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

        self._interval = interval
        self._preload = preload
        self._progress = progress
        self._vis_app = vis_app

        # dataset
        self.dataset = SynthiaVideoDataset(split, cam, pc)

        # streams
        self.state_stream = Stream(capacity=1)
        self.lc_stream = Stream(capacity=1)

        # visualize streams
        if self._vis_app is not None:
            vizstream(self._vis_app, self.state_stream, astype="scene_cloud")
            vizstream(self._vis_app, self.lc_stream, astype="lc_curtain")

        # current video and frame id
        self.video = None
        self.frame = None

    ####################################################################################################################
    # region Env API functions
    ####################################################################################################################

    def reset(self,
              vid: Optional[int] = None,
              start: Optional[int] = None) -> np.ndarray:
        # the next two lines set the current video, and assert that the video is long enough
        self.video = self.dataset[vid]
        assert start + self._interval < len(self.video)  # at least one call to step() can be made

        # reset streams
        self.state_stream.reset()
        self.lc_stream.reset()

        # the next three lines compute, augment and set the current frame
        init_frame: Frame = self.video[start]
        self.augment_frame_data(init_frame)
        self.frame = init_frame

        init_envelope: np.ndarray = init_frame.annos["se_ranges"]  # (C,)
        
        # the initial self.intensities is binary: 1 where objects exist (in front of max range) and 0 otherwise
        self.intensities = (init_envelope < self.max_range).astype(np.float32)  # (C,)

        return init_envelope

    # endregion
    ####################################################################################################################
    # region Env-specific helper functions for step()
    ####################################################################################################################

    def env_step_begin(self) -> NoReturn:
        # visualize intermediate frames by publishing them
        fid = self.frame.metadata.fid
        if self._vis_app is not None:
            for fid_ in range(fid + 1, fid + self._interval):
                frame_ = self.video[fid_]
                self.augment_frame_data(frame_)
                self.state_stream.publish(scene_points=frame_.points,
                                          se_design_pts=frame_.annos["se_design_pts"],
                                          downsample=True)

        # take an environment step by updating current frame and publish it
        fid += self._interval
        self.frame = self.video[fid]
        self.augment_frame_data(self.frame)
        self.state_stream.publish(scene_points=self.frame.points,
                                  se_design_pts=self.frame.annos["se_design_pts"],
                                  downsample=True)

    def env_place_r_curtain(self) -> Tuple[np.ndarray, np.ndarray]:
        r_curtain = self.rand_curtain_gen.next()  # (C,)
        r_return: LCReturn = self.light_curtain.get_lc_return_from_state(self.frame, r_curtain, self._vertical_range)
        self.lc_stream.publish(lc_image=r_return.lc_image, lc_cloud=r_return.lc_cloud)

        r_curtain = r_return.lc_ranges.copy()  # (C,)
        r_hits = r_return.bev_hits(ithresh=self._r_hit_intensity_thresh)  # (C,)
        return r_curtain, r_hits

    def env_place_f_curtain(self,
                            f_curtain: np.ndarray,
                            score: Optional[float]) -> LCReturn:
        f_return: LCReturn = self.light_curtain.get_lc_return_from_state(self.frame, f_curtain, self._vertical_range)
        self.lc_stream.publish(lc_image=f_return.lc_image, lc_cloud=f_return.lc_cloud, score=score)
        return f_return

    def env_current_gt_action(self) -> np.ndarray:
        return self.frame.annos["se_ranges"].copy()  # (C,)

    def env_end(self) -> bool:
        fid = self.frame.metadata.fid
        return fid + self._interval >= len(self.video)

    # endregion
    ####################################################################################################################
    # region EpisodeIndexIterator
    ####################################################################################################################

    def train_ep_index_iterator(self):
        """
        An iterator over (vid, start) tuples.

        The tuples correspond to all possible starting points in all videos of the dataset, intended for training."
        """
        tuples = [(vid, start)
                  for vid in range(len(self.dataset))
                  for start in range(len(self.dataset[vid]) - self._interval)]

        i = 0
        while True:
            if i == len(tuples):
                i = 0  # reset to beginning, creating infinitely many rounds
            if i == 0:
                random.shuffle(tuples)  # shuffle at the beginning of every round
            yield tuples[i]
            i += 1

    def single_video_index_iterator(self, vid):
        """
        An iterator over (vid, start) tuples for a single video.

        It checks the current frame of the env where the previous episode had ended, and chooses the next frame as the
        starting frame of the next episode.
        """
        fid = 0

        while fid + self._interval < len(self.dataset[vid]):
            yield vid, fid

            # this is the last frame of the previous episode when it had ended
            prev_episode_frame: Frame = self.frame

            prev_episode_vid = prev_episode_frame.metadata["vid"]
            prev_episode_fid = prev_episode_frame.metadata["fid"]

            assert prev_episode_vid == vid, "Env must exclusively use the vid provided by the iterator"
            assert prev_episode_fid >= fid + self._interval, "Env cannot go backwards in the episode"

            # next fid
            fid = prev_episode_fid + 1

    def eval_ep_index_iterator(self):
        """
        An iterator over (vid, start) tuples.

        Iterates through a single video at a time, over all videos in the dataset. This is intended for evaluation.
        """
        for vid_ in range(len(self.dataset)):
            for vid, fid in self.single_video_index_iterator(vid_):
                yield vid, fid

    # endregion
    ####################################################################################################################
