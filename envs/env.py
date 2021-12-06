from abc import ABC, abstractmethod
import gym
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from setcpp import SmoothnessDPL1Cost, SmoothnessDPPairGridCost, SmoothnessGreedy
import tqdm
from typing import Optional, Tuple, NoReturn

from data.synthia import Frame, append_xyz_to_depth_map
from devices.light_curtain import LCReturn
from lc_planner.planner import PlannerRT
from lc_planner.config import LCConfig
import utils


########################################################################################################################
# region Base Env class
########################################################################################################################

class Env(ABC, gym.Env):
    """
    Base class for implementing environments for safety envelope tracking.
    This is intended to mimic the OpenAI Gym wrapper.
    """
    def __init__(self,
                 lc_config: LCConfig,
                 thetas: np.ndarray,
                 min_range: float,
                 max_range: float,
                 use_random_curtain: bool,
                 random_curtain_updates_main_curtain: bool,
                 random_curtain_cache_file: str,
                 random_curtain_sampling: str,
                 random_curtain_spacing_power: float,
                 vertical_range: Tuple[float, float],
                 r_hit_intensity_thresh: int,
                 r_recession: float,
                 pp_smoothing: Optional[str],
                 tracking_rtol: float,
                 tracking_atol: float,
                 baseline_config: dict,
                 debug: bool = False):

        assert len(thetas) == lc_config.CAMERA_PARAMS['width']
        self._lc_config = lc_config
        self._thetas = thetas
        self._min_range = min_range
        self._max_range = max_range

        self._use_random_curtain = use_random_curtain
        self._random_curtain_updates_main_curtain = random_curtain_updates_main_curtain
        self._random_curtain_sampling = random_curtain_sampling
        self._random_curtain_spacing_power = random_curtain_spacing_power
        self._vertical_range = vertical_range
        self._r_hit_intensity_thresh = r_hit_intensity_thresh
        self._r_recession = r_recession
        self._pp_smoothing = pp_smoothing
        self._rtol = tracking_rtol
        self._atol = tracking_atol
        self._debug = debug

        # config for handcrafted baseline policy
        self.baseline_config = baseline_config

        # options
        self._RANGES_PER_RAY_V2 = 1000  # 01.9cm apart

        # random curtain generator
        self.rand_curtain_gen = RandomCurtainGenerator(cache_file=random_curtain_cache_file)
        ranges = self.get_ranges(self.min_range, self.max_range, self._RANGES_PER_RAY_V2,
                                 self._random_curtain_spacing_power)
        if not self.rand_curtain_gen.has_curtains:
            plannerV2 = PlannerRT(self._lc_config, ranges, self.C, version=2)
            self.rand_curtain_gen.generate(planner=plannerV2, sampling=self._random_curtain_sampling)

        # smoothing
        self._SMOOTHNESS = 0.05
        smoothness_args = (self.C, self.min_range, self.max_range, self._SMOOTHNESS)
        self._smoothnessDPL1Cost = SmoothnessDPL1Cost(*smoothness_args)
        self._smoothnessDPPairGridCost = SmoothnessDPPairGridCost(*smoothness_args)
        self._smoothnessGreedy = SmoothnessGreedy(*smoothness_args)

        # stores the most recently obtained intensities.
        # this is used by heuristic_greedy smoothing to define a priority over camera rays.
        # this should be initialized in self.reset()
        self.intensities = None  # (C,)

        if self._debug:
            # visualize 20 random curtains
            for i in range(20):
                design_pts = plannerV2._planner.randomCurtainDiscrete(self._random_curtain_sampling)
                design_pts = np.array(design_pts, dtype=np.float32)  # (C, 3)
                plt.plot(design_pts[:, 0], design_pts[:, 1])

            plt.ylim(0, 20)
            plt.xlim(-7, 7)
            plt.title("power: {}, vel: {}, acc: {}".format(
                self._random_curtain_spacing_power,
                self._lc_config.LASER_PARAMS["max_omega"],
                self._lc_config.LASER_PARAMS["max_alpha"]), fontsize='xx-large')
            plt.tight_layout()
            plt.show()

    @property
    def thetas(self):
        return self._thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]

    @property
    def min_range(self):
        return self._min_range

    @property
    def max_range(self):
        return self._max_range

    @property
    def H(self):
        return self._lc_config.CAMERA_PARAMS['height']  # number of camera rows

    @property
    def C(self):
        return self._lc_config.CAMERA_PARAMS['width']  # number of camera columns

    @staticmethod
    def get_ranges(min_range, max_range, num_ranges, power):
        # generate numbers between 0 and 1
        unit_spacing = np.linspace(0, 1, num_ranges, dtype=np.float32)  # (R,)
        unit_spacing = np.power(unit_spacing, power)  # (R,)
        ranges = min_range + (max_range - min_range) * unit_spacing  # (R,)
        return ranges

    def safety_envelope(self,
                        frame: Frame) -> np.ndarray:
        """
        Computes ground truth safety envelope from the ground truth depth map in the frame.
        The safety envelope for each camera column is the smallest bev range value across all pixels in that column.

        Args:
            frame (Frame): frame containing ground truth depth.

        Returns:
            se_ranges: (np.ndarray, dtype=np.float32, shape=(C,)) the ranges of the ground truth safety envelope,
                       one per camera ray.
        """
        depth = frame.depth.copy()  # (H, C)

        # append x, y, z to depth
        P2 = frame.calib["P2"][:3, :3]  # (3, 3)
        cam_xyz = append_xyz_to_depth_map(depth[:, :, None], P2)  # (H, C, 3); axis 2 is (x, y, z) in cam frame

        cam_x, cam_y, cam_z = cam_xyz[:, :, 0], cam_xyz[:, :, 1], cam_xyz[:, :, 2]  # all are (H, C)
        bev_range = np.sqrt(np.square(cam_x) + np.square(cam_z))  # (H, C)  sqrt(x**2 + z**2)

        # we do not care about objects beyond "max_range"
        bev_range = bev_range.clip(max=self.max_range)  # (H, C)

        # pixels that are outside the vertical range are assumed to be infinitely far away
        # (note that cam_y points downwards)
        vrange_min, vrange_max = self._vertical_range
        outside_vrange_mask = (-cam_y < vrange_min) | (-cam_y > vrange_max)  # (H, C)
        bev_range[outside_vrange_mask] = self.max_range

        se_ranges = bev_range.min(axis=0)  # (C,)
        return se_ranges.astype(np.float32)

    def augment_frame_data(self, frame: Frame) -> NoReturn:
        """Compute the gt safety envelope and add it to the frame"""
        se_ranges = self.safety_envelope(frame)  # (C,)
        se_design_pts = utils.design_pts_from_ranges(se_ranges, self.thetas)  # (C, 2)
        frame.annos["se_ranges"] = se_ranges
        frame.annos["se_design_pts"] = se_design_pts

    ####################################################################################################################
    # region Env API functions
    ####################################################################################################################

    def reset(self,
              vid: Optional[int] = None,
              start: Optional[int] = None) -> np.ndarray:
        """Resets the state of the environment, returns the initial envelope and also initializes self.intensities.

        Args:
            vid (int): video id.
            start (int): start frame of video.

        Returns:
            init_envelope (np.ndarray, dtype=np.float32, shape=(C,)): the initial envelope.
        """
        raise NotImplementedError

    def step(self,
             action: Optional[np.ndarray],
             score: Optional[float] = None,
             get_gt: bool = False) -> Tuple[LCReturn, bool, dict]:
        """
        Compute the observations from the current step.
        This is derived by placing the light curtain computed from observations in the previous timestep,
        in the current frame.

        Args:
            action (np.ndarray, dtype=np.float32, shape=(C,)): Ranges of the light curtain.
                This is optional; if None, then the ground truth action will be used instead (for behavior cloning).
            score (Optional[float]): the score of the front curtain that needs to be published.
            get_gt (bool): whether to compute gt_action or not

        Returns:
            observation (LCReturn): agent's observation of the current environment. This is the return from the front
                                    light curtain. Always returns a valid observation, even when end=True.
            end (bool): is True for the last valid observation in the episode. No further calls to step() should be
                         made after a end=True has been returned.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
                         {
                            'gt_action' (optional):  (np.ndarray, dtype=float32, shape=(C,))
                                                     the light curtain placement that should be considered `ground
                                                     truth' for the previous timestep. This is what `action' should
                                                     ideally be equal to.
                            'ss_action' (optional): (np.ndarray, dtype=np.float32, shape=(C,))
                                                    partial ground truth self-supervision signal generated by random
                                                    light curtains. note that the mask is equal to
                                                    (ss_action < self.max_range).
                         }
        """
        self.env_step_begin()
        info = {}

        ################################################################################################################
        # region Random curtain
        ################################################################################################################

        if self._use_random_curtain:
            # place random curtain and move f_curtain to wherever hits are observed
            r_curtain, r_hits = self.env_place_r_curtain()

            # compute self-supervision signal
            # these are the ranges of the random curtain for those camera rays where a hit was observed.
            # rays that did not observe a hit are masked out.
            ss_action = r_curtain.copy()  # (C,)
            ss_action[~r_hits] = self.max_range
            info['ss_action'] = ss_action

        # endregion
        ################################################################################################################
        # region Pre-processing forecasting curtain
        ################################################################################################################

        if action is not None:
            # clip curtain between min and max range
            f_curtain = action.clip(min=self.min_range, max=self.max_range)  # (C,)

            if self._use_random_curtain and self._random_curtain_updates_main_curtain:
                # update f_curtain by moving it to locations where the random curtain observed returns
                # update only those locations where the random curtain detected objects *closer* than the main curtain
                r_update_mask = r_hits & (r_curtain < f_curtain)  # (C,)
                f_curtain[r_update_mask] = r_curtain[r_update_mask] - self._r_recession

                # since f_curtain is being updated, self.intensities must also be updated.
                # furthermore, the locations of random curtain hits should get the highest priority
                self.intensities[r_update_mask] = 1.1

        # endregion
        ################################################################################################################
        # region Smoothing forecasting curtain
        ################################################################################################################

        if action is not None:
            if self._pp_smoothing == "heuristic_global":
                # heuristic smoothing: difference between ranges on consecutive rays shouldn't exceed a threshold
                # global optimization: minimizes the sum of L1 differences across all rays using DP
                if self._use_random_curtain and self._random_curtain_updates_main_curtain:
                    # when using random curtains, the cost will be hierarchical:
                    # (sum of L1 costs over rays in r_update_mask, sum of L1 costs over rays outside r_update_mask)
                    # this priorities being close to the locations updated by r_curtain more than the other locations.

                    ranges = np.array(self._smoothnessDPPairGridCost.getRanges(), dtype=np.float32)  # (R,)
                    flat_cost = np.abs(ranges.reshape(-1, 1) - f_curtain)  # (R, C)

                    # hierarchical cost
                    # - (L1cost, 0): if on ray in r_update_mask
                    # - (0, L1cost): if on ray outside r_update_mask
                    pair_cost = np.zeros([len(ranges), self.C, 2], dtype=np.float32)  # (R, C, 2)
                    pair_cost[:, r_update_mask, 0] = flat_cost[:, r_update_mask]
                    pair_cost[:, ~r_update_mask, 1] = flat_cost[:, ~r_update_mask]

                    f_curtain = np.array(self._smoothnessDPPairGridCost.smoothedRanges(pair_cost), dtype=np.float32)  # (C,)
                else:
                    f_curtain = np.array(self._smoothnessDPL1Cost.smoothedRanges(f_curtain), dtype=np.float32)  # (C,)
            elif self._pp_smoothing == "heuristic_greedy":
                # heuristic smoothing: difference between ranges on consecutive rays shouldn't exceed a threshold
                # greedy optimization: greedily smoothes ranges while iterating over rays prioritized by largest weights
                f_curtain = np.array(self._smoothnessGreedy.smoothedRanges(f_curtain, self.intensities), dtype=np.float32)  # (C,)
            elif self._pp_smoothing == "planner_global":
                # create L1 cost function
                ranges = self.plannerV2.ranges  # (R,)
                cmap = -np.abs(ranges.reshape(-1, 1) - f_curtain)  # (R, C)
                design_pts = self.plannerV2.get_design_points(cmap)  # (C, 2)
                assert design_pts.shape == (self.plannerV2.num_camera_angles, 2)
                f_curtain = np.linalg.norm(design_pts, axis=1)  # (C,)
            else:
                raise Exception(f"env.pp_smoothing must be " +
                                "\"heuristic_global\" or \"heuristic_greedy\" or \"planner_global\"")

        # endregion
        ################################################################################################################
        # region GT-action and placing forecasting curtain
        ################################################################################################################

        if (action is None) and (get_gt == False):
            raise Exception("Must compute gt_action in behavior cloning")

        # the next line gets the ground truth action for the previous timestep
        # in the ideal policy, `action' should match this `gt_action'
        if get_gt:
            info['gt_action'] = self.env_current_gt_action()  # (C,)

        # if action is set to None (for eg. in behavior cloning), use the ground truth action instead
        if action is None:
            f_curtain = info['gt_action']

        # placing forecasting curtain
        obs: LCReturn = self.env_place_f_curtain(f_curtain, score=score)

        # the next line updates self.intensities
        self.intensities = obs.bev_intensities() / 255.0

        # the next line computes `end', which checks whether another env.step() call can be made
        end = self.env_end()

        time.sleep(0)  # interrupt, useful for RealEnv
        return obs, end, info

    def done(self,
             f_curtain: np.ndarray,
             se_ranges: np.ndarray) -> bool:
        """
        Whether the episode transitions to the terminal state or not.

        Done is true when the curtain has moved too far away from the safety envelope on any camera ray i.e.
        abs(f_curtain - se_ranges) > (atol + rtol * se_ranges) for any camera ray

        Args:
            f_curtain (np.ndarray, dtype=float32, shape=(C,)): curtain placement
            se_ranges (np.ndarray, dtype=float32, shape=(C,)): ground truth safety envelope.

        Returns:
            done (bool): whether f_curtain is too far away from se_ranges on any camera ray.
        """
        # the next line computes the mask over rays; only these rays should count towards termination
        mask = se_ranges < self.max_range  # (C,)
        f_curtain = f_curtain[mask]  # (C',)
        se_ranges = se_ranges[mask]  # (C',)
        # bad_rays = np.abs(f_curtain - se_ranges) > self._atol + self._rtol * se_ranges  # (C')
        # frac_bad_rays = bad_rays.sum() / mask.sum().clip(min=1)
        # return frac_bad_rays >= 0.5
        return np.any(np.abs(f_curtain - se_ranges) > self._atol + self._rtol * se_ranges)

    def render(self, mode='human'):
        pass

    # endregion
    ####################################################################################################################
    # region Env-specific helper functions for step()
    ####################################################################################################################

    @abstractmethod
    def env_step_begin(self) -> NoReturn:
        """
        Env-specific helper function for step().
        Any pre-processing that needs to be done at the start of the step() function.
        """
        raise NotImplementedError

    @abstractmethod
    def env_place_r_curtain(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Env-specific helper function for step().
        Places a random curtain and gets return.

        Returns:
            r_curtain (np.ndarray, dtype=float32, shape=(C,)): ranges of the unified random curtain
            r_hits (np.ndarray, dtype=bool, shape=(C,)): mask where hits were found in the unified random curtain
        """
        raise NotImplementedError

    @abstractmethod
    def env_place_f_curtain(self,
                            f_curtain: np.ndarray,
                            score: Optional[float]) -> LCReturn:
        """
        Env-specific helper function for step().
        Places a forecasting curtain and gets return.

        Args:
            f_curtain (np.ndarray, dtype=float32, shape=(C,)): ranges of the forecasting curtain
            score (Optional[float]): score from the previous timestep. SimEnv uses this to publish score to Kittiviewer.

        Returns:
            f_return (LCReturn): Forecasting curtain return.
        """
        raise NotImplementedError

    @abstractmethod
    def env_current_gt_action(self) -> np.ndarray:
        """
        Env-specific helper function for step().
        Computes the current gt_action.

        Returns:
            gt_action (np.ndarray, dtype=float32, shape=(C,)): current gt action
        """
        raise NotImplementedError

    @abstractmethod
    def env_end(self) -> bool:
        """Computes the end flag, which checks whether another env.step() call can be made"""
        raise NotImplementedError

    # endregion
    ####################################################################################################################
    # region Legacy helper functions
    ####################################################################################################################

    def _debug_visualize_curtains(self, f_curtain, r_curtain):
        design_pts = utils.design_pts_from_ranges(f_curtain, self.thetas)
        x, z = design_pts[:, 0], design_pts[:, 1]
        plt.plot(x, z, c='b')

        design_pts = utils.design_pts_from_ranges(r_curtain, self.thetas)
        x, z = design_pts[:, 0], design_pts[:, 1]
        plt.plot(x, z, c='r')

        plt.ylim(0, 21)
        plt.show()

    def _random_curtain(self,
                        r_type: str = "linear") -> np.ndarray:
        """Computes a random curtain across the entire scene
        
        Args:
            r_type (str): type of the random curtain. Options are (1) "uniform", (2) "linear".
        Returns:
            curtain (np.ndarray, dtype=np.float32, shape=(C,)): range per camera ray that may not correpsond to a
                     valid curtain.
        """
        limits_lo = np.ones(self.C, dtype=np.float32) * 0.5 * self.min_range  # (C,)
        limits_hi = np.ones(self.C, dtype=np.float32) * self.max_range  # (C,)

        if r_type == "uniform":
            curtain = np.random.uniform(low=limits_lo, high=limits_hi)  # (C,)
        elif r_type == "linear":
            curtain = np.sqrt(np.random.uniform(low=np.square(limits_lo), high=np.square(limits_hi)))  # (C,)
        else:
            raise Exception("r_type must be one of [uniform/linear]")

        return curtain

    # endregion
    ####################################################################################################################

# endregion
########################################################################################################################
# region Random curtain generator class
########################################################################################################################


class RandomCurtainGenerator:
    def __init__(self,
                 cache_file: str):
        self.curtains = None  # (N, C)
        self.ptr = 0

        self.cache_file = Path(cache_file)
        if self.cache_file.exists():
            self.load_from_cache_file()
    
    @property
    def has_curtains(self):
        return self.curtains is not None
    
    def load_from_cache_file(self):
        self.curtains = np.loadtxt(self.cache_file).astype(np.float32)  # (N, C)
        utils.cprint(f'Loaded {len(self.curtains)} random curtains from cache!', color='yellow')
    
    def generate(self,
                 planner: PlannerRT,
                 sampling: str,
                 num_curtains: int=1000):
        
        assert not self.cache_file.exists(), "Cannot generate curtains if cache file already exists"
        with open(self.cache_file, 'w') as f:
            for _ in tqdm.trange(num_curtains, desc='Creating random curtain cache ...'):
                while True:
                    curtain = self.generate_curtain_from_planner(planner, sampling)
                    # don't save degenerate curtains 90% of whose rays are behind 3m
                    if not ((curtain < 3.0).mean() > 0.9):
                        break
                print(' '.join([str(e) for e in curtain]), file=f)
        
        self.load_from_cache_file()
    
    @staticmethod
    def generate_curtain_from_planner(planner, sampling):
        r_curtain = planner._planner.randomCurtainDiscrete(sampling)
        r_curtain = np.array(r_curtain, dtype=np.float32)  # (C, 3)
        r_curtain = np.linalg.norm(r_curtain[:, :2], axis=1)  # (C,)
        return r_curtain
    
    def next(self):
        curtain = self.curtains[self.ptr]
        self.ptr += 1
        if self.ptr == len(self.curtains):
            self.ptr = 0
        return curtain  # (C,)

# endregion
########################################################################################################################
