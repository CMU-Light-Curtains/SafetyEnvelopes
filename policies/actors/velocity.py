from typing import Tuple, Optional
import numpy as np
import torch

from setcpp import SmoothnessDPPairGridCost
from policies.actors import Pi
from policies.actors.actor import Actor

from data.synthia import Frame
from devices.light_curtain import LCReturn


class VelocityActor(Actor):
    def __init__(self,
                 thetas: np.ndarray,
                 base_policy: Actor,
                 min_range: float,
                 max_range: float):
        self.base_policy = base_policy
        self.C = len(thetas)

        self.prev_intensity = np.zeros(self.C, dtype=np.float32)
        self.prev_delta = np.zeros(self.C, dtype=np.float32)

        # smoothing
        SMOOTHNESS = 0.05
        self._smoothnessDPPairGrid = SmoothnessDPPairGridCost(self.C,
                                                              min_range,
                                                              max_range,
                                                              SMOOTHNESS)
        self._ranges = np.array(self._smoothnessDPPairGrid.getRanges()).astype(np.float32)
        self.R = len(self._ranges)

        self.prev_pos_vals = None
        self.prev_pos_mask = None

    def init_action(self,
                    init_envelope: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """
        Args:
            init_envelope (np.ndarray, dtype=np.float32, shape=(C,)): the initial envelope provided by the environment.

        Returns:
            action (np.ndarray, dtype=np.float32, shape=(C,)): action, in terms of ranges for each camera ray.
            logp_a (Optional[np.ndarray]): log-probability of the sampled actions. None if sampling is deterministic.
            info (dict): auxiliary info generated by policy, such as a vector representation of the observation while
                         generating the action.
        """
        act, logp_a, info = self.base_policy.init_action(init_envelope)  # use base policy
        return act, None, {}

    def step(self,
             obs: LCReturn) -> Tuple[np.ndarray, Optional[np.ndarray], bool, dict]:
        """
        Args:
            obs (LCReturn): observations viz return from the front light curtain.

        Returns:
            act (np.ndarray, dtype=np.float32, shape=(C,)): sampled actions.
            logp_a (Optional[np.ndarray]): log-probability of the sampled actions. None if sampling is deterministic.
            control (bool): whether this policy had control of taking the action at this timestep.
                            - this will be used to determine whether to evaluate the policy at this frame.
                            - another eg. is that these are the timesteps when nn-based policies will run the network.
            info (dict): auxiliary info generated by policy, such as a vector representation of the observation while
                         generating the action.
        """
        ################################################################################################################
        # Compute bev ranges for positive and negative hypothesis from intensity
        ################################################################################################################

        lc_xyz = obs.lc_image[:, :, :3]  # (H, C, 3)
        lc_range = np.linalg.norm(lc_xyz, axis=2)  # (H, C)
        cam_ray_unit_vectors = lc_xyz / lc_range[:, :, None]  # (H, C, 3)

        # intensity
        # mask out pixels that are below GROUND_HEIGHT (note that cam_y points downwards)
        intensity_uint8 = obs.lc_image[:, :, 3]  # (H, C)
        intensity_uint8[-lc_xyz[:, :, 1] < obs.GROUND_HEIGHT_] = 0
        intensity = intensity_uint8 / 255.  # (H, C)

        # the next line computes the mask of pixels that are nan
        nan_mask_2d = np.isnan(lc_xyz).any(axis=2)  # (H, C)
        # the next line asserts that lc_thickness should be non-nan wherever lc_image is non-nan.
        assert np.isfinite(obs.lc_thickness)[~nan_mask_2d].all()
        intensity[nan_mask_2d] = 0  # (H, C), there are no more nans in the intensity image

        error = 0.5 * obs.lc_thickness * (1.0 - intensity)  # (H, C)

        pos_xyz = (lc_range + error)[..., None] * cam_ray_unit_vectors  # (H, C, 3)
        pos_bev_range = self._bev_range_from_xyz(pos_xyz)  # (H, C)

        neg_xyz = (lc_range - error)[..., None] * cam_ray_unit_vectors  # (H, C, 3)
        neg_bev_range = self._bev_range_from_xyz(neg_xyz)  # (H, C)

        # the next two lines compute the closest positive bev_range: columnwise min of pos_bev_range
        # we don't know the edges of the slab for pixels with nan thickness; set their bev_range to infinity
        pos_bev_range[nan_mask_2d] = np.inf
        pos_bev_range = pos_bev_range.min(axis=0)  # (C,)

        # the next two lines compute the closest negative bev_range: columnwise max of neg_bev_range
        # pixels that have nan values or zero-intensity should be excluded from the maximization
        neg_bev_range[nan_mask_2d] = -np.inf
        neg_bev_range = neg_bev_range.max(axis=0)  # (C,)

        # the next line computes the 1D mask of columns where all pixels are nan's
        # the output ranges for these are interpolated
        nan_mask_1d = nan_mask_2d.all(axis=0)  # (C,) -- columns where all pixels are nan's; interpolate these

        # the next line computes the 1D mask of columns where at least one pixel has a positive intensity
        # these columns have a higher priority to be imaged
        pos_mask_1d = (intensity > 0).any(axis=0)  # (C,)

        # the next two lines compute the maximum intensity per camera column
        intensity = intensity.max(axis=0)  # (C,)
        next_prev_intensity = intensity  # (C,) to save in self.prev_intensity

        # # the next lines compute the mask of columns that are preferred to be imaged.
        # pos_intensity_mask_1d = (intensity > 0).any(axis=0)  # (C,)
        # inds = np.where(pos_intensity_mask_1d)[0]  # (C',)

        ################################################################################################################
        # Algorithm to compute next LC placement
        ################################################################################################################

        if True:
            # take partials of quantities: subsets of all rays indexed by inds
            inds = np.where(pos_mask_1d)[0]
            pos_bev_range = pos_bev_range[inds]  # (C',)
            neg_bev_range = neg_bev_range[inds]  # (C',)
            # intensity = intensity[inds]  # (C',)
            # prev_intensity = self.prev_intensity[inds]  # (C',)

            # has_intensity_increased = intensity > prev_intensity  # (C',)
            # has_curtain_moved_forward = self.prev_delta[inds] > 0.  # (C',)

            # curtain should be moved forward in the next timestep if in the previous timestep, either
            #   (1) curtain moved forward and intensity increased, or
            #   (2) curtain moved backward and intensity decreased
            # move_curtain_forward = has_intensity_increased == has_curtain_moved_forward  # (C')

            # the next few lines computes the direction of curtain motion using the ground truth safety envelope, based on
            # which of the pos_bev_range and neg_bev_range the curtain is closer to
            se_ranges = obs.frame.annos["se_ranges"]  # (C,)
            se_ranges = se_ranges[inds]  # (C',)
            move_curtain_forward = np.abs(pos_bev_range - se_ranges) < np.abs(neg_bev_range - se_ranges)  # (C',)

            partial_curtain = np.where(move_curtain_forward, pos_bev_range, neg_bev_range)  # (C',)

            if len(inds) > 0:
                curtain = self._interpolated_curtain(partial_curtain, inds)
            else:
                curtain = obs.lc_ranges + self.base_policy._EXPANSION_F

            # enforce smoothness
            # curtain = np.array(enforce_smoothness(curtain, self.base_policy._smoothness), dtype=np.float32)  # (C,)

            # save current intensity and delta
            self.prev_intensity = next_prev_intensity  # (C,)
            self.prev_delta = curtain - obs.lc_ranges  # (C,)

        else:
            # the next few lines computes the direction of curtain motion using the ground truth safety envelope, based on
            # which of the pos_bev_range and neg_bev_range the curtain is closer to
            se_ranges = obs.frame.annos["se_ranges"]  # (C,)
            move_curtain_forward = np.abs(pos_bev_range - se_ranges) < np.abs(neg_bev_range - se_ranges)  # (C,)
            pos_vals = np.where(move_curtain_forward, pos_bev_range, neg_bev_range)  # (C,)

            cost_pair_grid = np.zeros([self.R, self.C, 2], dtype=np.float32)

            if self.prev_pos_vals is not None:
                # add velocity to those camera rays that have positive intensity in the curr and prev timesteps
                mask = self.prev_pos_mask & pos_mask_1d  # (C,)
                velocity = pos_vals - self.prev_pos_vals  # (C,)
                curtain = np.where(mask, pos_vals + velocity, pos_vals)  # (C,) these are the forecasted positions
                # curtain = pos_vals
            else:
                curtain = pos_vals

            # update prev pos
            self.prev_pos_vals = pos_vals
            self.prev_pos_mask = pos_mask_1d

            # smooth the curtain using an L1 cost:

            # (1) columns with positive intensities have high priority
            mask = pos_mask_1d  # (C,)
            cost_pair_grid[:, mask, 0] = np.abs(self._ranges.reshape(-1, 1) - curtain[mask])  # (R, C, 2)

            # (2): columns with zero intensity have low priority
            mask = ~(nan_mask_1d | pos_mask_1d)  # (C,)
            cost_pair_grid[:, mask, 1] = np.abs(self._ranges.reshape(-1, 1) - curtain[mask])  # (R, C, 2)

            curtain = np.array(self._smoothnessDPPairGrid.smoothedRanges(cost_pair_grid)).astype(np.float32)  # (C,)

        pi = Pi(actions=torch.from_numpy(curtain).unsqueeze(-1))  # batch_shape=(C,) event_shape=()
        info = dict(pi=pi)

        action, logp_a, control = curtain, None, True  # logp_a is None since this policy is deterministic
        return action, logp_a, control, info

    def forward(self,
                obs: torch.Tensor) -> torch.distributions.Distribution:
        """
        Args:
            obs (torch.Tensor): observation in torch tensors.

        Returns:
            pi (torch.distributions.Distribution): predicted action distribution of the policy.
        """
        raise NotImplementedError

    def evaluate_actions(self,
                         obs: torch.Tensor,
                         act: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs (torch.Tensor): observation in torch tensors.
            act (torch.Tensor): actions in torch tensors.

        Returns:
            logp_a (torch.Tensor): log probability of taking actions "act" by the actor under "obs", as a torch tensor.
        """
        raise NotImplementedError

    def reset(self):
        self.prev_intensity = np.zeros(self.C, dtype=np.float32)
        self.prev_delta = np.zeros(self.C, dtype=np.float32)

    @staticmethod
    def _bev_range_from_xyz(xyz: np.ndarray) -> np.ndarray:
        """
        Args:
            xyz (np.ndarray, dtype=float32, shape=(H, C, 3)): xyz coordinates in cam frame
                - Axis 2 corresponds to (x, y, z):
                - x : x in cam frame.
                - y : y in cam frame.
                - z : z in cam frame.

        Returns:
            bev_range (np.ndarray, dtype=float32, shape=(H, C)): bev ranges for every pixel.
        """
        x = xyz[:, :, 0]  # (H, C)
        z = xyz[:, :, 2]  # (H, C)
        bev_range = np.sqrt(np.square(x) + np.square(z))  # (H, C)
        return bev_range

    def _interpolated_curtain(self,
                              partial_curtain: np.ndarray,
                              inds: np.ndarray):
        """
        Creates an interpolated curtain, from a partial curtain at specified inds.

        Args:
            partial_curtain (np.ndarray, dtype=float32, shape=(C',)): partial curtain where C' < C
            inds (np.ndarray, dtype=int64, shape=(C')): indices of the partial curtain bev ranges. Each inds
                must lie between 0 and C-1.
        """
        assert len(inds) > 0  # there should be at least one element
        assert (inds == np.sort(inds)).all()  # assert that inds are in sorted order
        assert 0 <= inds.min() and inds.max() < self.C  # assert that the inds are valid

        inds = [0] + list(inds) + [self.C-1]
        vals = [partial_curtain[0]] + list(partial_curtain) + [partial_curtain[-1]]

        curtain = np.zeros([self.C], dtype=np.float32)  # (C,)
        for i in range(len(inds) - 1):
            ind_l, ind_r = inds[i], inds[i+1]
            val_l, val_r = vals[i], vals[i+1]
            interval_size = ind_r - ind_l + 1
            interval_vals = np.linspace(val_l, val_r, interval_size)
            curtain[ind_l:ind_r+1] = interval_vals

        return curtain
