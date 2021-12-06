from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as td

from data.synthia import Frame
from devices.light_curtain import LCReturn
from policies.actors import Pi
from policies.actors.actor import Actor
import utils

ingr = utils.Ingredient("ersnet")
ingr.add_config("config/networks/ersnet.yaml")


@utils.register_class("network")
class ERSNet(Actor):
    @ingr.capture
    def __init__(self,
                 thetas: np.ndarray,
                 base_policy: Actor,
                 num_bins: int,
                 min_range: float,
                 max_range: float,
                 learn_e: bool,
                 learn_r: bool,
                 learn_sigma: bool,
                 init_sigma: float,
                 history: int):
        """
        Args:
            thetas (np.ndarray, dtype=np.float32, shape=(C,)): thetas of the camera rays,
                in degrees and in increasing order in [-fov/2, fov/2]
            base_policy (Actor): a base policy that is used for the first action.
            num_bins (int): number of discrete bins (ranges) per camera ray used in inputs and outputs
            min_range (float): minimum range of the light curtain device
            max_range (float): maximum range of the light curtain device
            init_sigma (float): initialization of sigma as a fraction of the total span of the discrete ranges
            history (int): the number of previous observations the actor will use.
        """
        self.base_policy = base_policy

        self.R = num_bins
        self.C = len(thetas)
        self.min_range = min_range
        self.max_range = max_range
        self.bin_length = (self.max_range - self.min_range) / self.R
        utils.pprint(f"ERSNet: inputs/outputs are discretized into bins of length {self.bin_length * 100:.1f}cm", "red")

        self.bin_ls = np.linspace(self.min_range, self.max_range - self.bin_length, self.R, dtype=np.float32)  # (R,)
        self.bin_rs = self.bin_ls + self.bin_length  # (R,)
        self.bin_cs = 0.5 * (self.bin_ls + self.bin_rs)  # (R,)

        self.H = history
        self.history = utils.History(size=self.H)

        self.network = ERSNetwork(thetas, self.base_policy, self.bin_cs, self.H,
                                  learn_e, learn_r, learn_sigma, init_sigma)
        if torch.cuda.is_available():
            self.network = self.network.cuda()

    @dataclass
    class IR:
        i: np.ndarray  # 1D vector of LC intensities
        r: np.ndarray  # 1D vector of LC placement ranges

    @property
    def feat_dim(self) -> List[int]:
        return [2 * self.H, self.C]  # 2H features: intensity vector, placement (range) vector per timestep

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

    @torch.no_grad()
    def step(self,
             obs: LCReturn) -> Tuple[np.ndarray, Optional[np.ndarray], bool, dict]:
        """
        Args:
            obs (LCReturn): observations viz return from the front light curtain.

        Returns:
            act (np.ndarray, dtype=np.float32, shape=(C,)): sampled actions -- these are absolute, not residual.
            logp_a (Optional[np.ndarray]): log-probability of the sampled actions. None if sampling is deterministic.
            control (bool): whether this policy had control of taking the action at this timestep.
                            - this will be used to determine whether to evaluate the policy at this frame.
                            - another eg. is that these are the timesteps when nn-based policies will run the network.
            info (dict): auxiliary info generated by policy, such as a vector representation of the observation while
                         generating the action.
        """
        self.network.eval()

        # get placement and intensities from observation
        i = obs.bev_intensities() / 255.0  # (C,)
        r = obs.lc_ranges  # (C,)

        # create IR of current timestep and add to history
        ir = ERSNet.IR(i, r)
        self.history.add(ir)

        # create features from history
        inputs = sum([[ir.i, ir.r] for ir in self.history.get()], [])  # list of 2H vectors of shape (C,)
        feat = np.stack(inputs, axis=0)  # (2H, C)

        # action from network
        feat_tensor = torch.from_numpy(feat).unsqueeze(dim=0)  # (1, 2H, C)
        if torch.cuda.is_available():
            feat_tensor = feat_tensor.cuda()
        pi: Pi = self.forward(feat_tensor)  # batch_shape=(1, C) event_shape=()
        act, act_inds = pi.sample()  # both (1, C)
        logp_a = pi.log_prob(act_inds).squeeze(0).cpu().numpy()
        act = act.squeeze(0).cpu().numpy()  # (C,) dtype=float32
        act_inds = act_inds.squeeze(0).cpu().numpy()  # (C,) dtype=int64

        # test to check that init_baseline_params() produces same behavior as base policy
        # base_act, _, _, _ = self.base_policy.step(obs)
        # assert np.allclose(base_act, act)

        control = True
        info    = dict(
            features=feat,  # (2H, C)
            pi=pi,  # batch_shape=(1, C) event_shape=()
            action_inds=act_inds  # (C,) dtype=int64
        )
        return act, logp_a, control, info

    def forward(self,
                feats: torch.Tensor) -> Pi:
        """
        Args:
            feats (torch.Tensor, dtype=float32, shape=(B, 2H, C)): features.

        Returns:
            pi (Pi, batch_shape=(B, C), event_shape=()): action distribution.
        """
        pi: Pi = self.network(feats)  # batch_shape=(B, C) event_shape=()
        return pi

    def reset(self):
        super().reset()
        self.history.clear()

########################################################################################################################
# region Convolutional Neural Networks
########################################################################################################################


class ERSNetwork(nn.Module):
    def __init__(self,
                 thetas: np.ndarray,
                 base_policy: Actor,
                 discrete_ranges: np.ndarray,
                 H: int,
                 learn_e: bool,
                 learn_r: bool,
                 learn_sigma: bool,
                 init_sigma: float):
        """
        This superclass shall be inherited by other subclasses that implement the neural network.
        This superclass implements helper functions required to perform ERS.

        Args:
            thetas (np.ndarray, dtype=np.float32, shape=(C,)): thetas of the camera rays,
                in degrees and in increasing order in [-fov/2, fov/2]
            base_policy (Actor): a base policy that is used for the first action.
            discrete_ranges (np.ndarray, dtype=float32, shape=(R,)): the centers of the discretized bins
        """

        super().__init__()

        self.base_policy = base_policy
        self.discrete_ranges = discrete_ranges

        # torch tensor of discretized ranges
        self.discrete_ranges_torch_3d = torch.from_numpy(self.discrete_ranges[None, None])  # (1, 1, R)
        if torch.cuda.is_available():
            self.discrete_ranges_torch_3d = self.discrete_ranges_torch_3d.cuda()

        # pre-compute smoothness_deltas:
        # the maximum differences in ranges values between pairs of camera rays
        arange = np.arange(len(thetas), dtype=np.float32)  # (C,)
        self.smoothness_dist_matrix = \
            np.abs(arange.reshape(1, -1) - arange.reshape(-1, 1)) * self.base_policy._SMOOTHNESS  # (C, C)

        # smoothness distance matrix
        self.smoothness_dist_matrix = torch.from_numpy(self.smoothness_dist_matrix)  # (C, C)
        if torch.cuda.is_available():
            self.smoothness_dist_matrix = self.smoothness_dist_matrix.cuda()  # (C, C)
        self.smoothness_dist_matrix.unsqueeze_(dim=0)  # (1, C, C)

        ################################################################################################################
        # Backbone network
        ################################################################################################################
        dim = 2 * H

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        # self.bn1 = nn.BatchNorm1d(11)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        # self.bn2 = nn.BatchNorm1d(5)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)

        ################################################################################################################
        # Expansion
        ################################################################################################################

        init_e = self.base_policy._EXPANSION_F

        if learn_e:
            # the next two lines create and initialize a conv layer to predict expansion
            self.expansion_f = nn.Conv1d(dim, 1, kernel_size=5, padding=2)
            self._zero_weight_const_bias(self.expansion_f, bias=init_e)
        else:
            # the next lines creates a function that always outputs a fixed expansion
            self.expansion_f = self._single_output_function(value=init_e)

        ################################################################################################################
        # Recession
        ################################################################################################################

        init_r = self.base_policy._RECESSION_F

        if learn_r:
            # the next two lines create and initialize a conv layer to predict recession
            self.recession_f = nn.Conv1d(dim, 1, kernel_size=5, padding=2)
            self._zero_weight_const_bias(self.recession_f, bias=init_r)
        else:
            self.recession_f = self._single_output_function(value=init_r)

        ################################################################################################################
        # Sigma
        ################################################################################################################

        # initial sigma (original init_sigma is expressed as a fraction of the span of the discrete ranges).
        range_span = self.discrete_ranges.max() - self.discrete_ranges.min()
        init_sigma = init_sigma * range_span

        if learn_sigma:
            # the next two lines create and initialize a conv layer to predict sigma
            self.sigma = nn.Conv1d(dim, 1, kernel_size=5, padding=2)
            self._zero_weight_const_bias(self.sigma, bias=init_sigma)
        else:
            # the next lines creates a function that always outputs a fixed sigma
            self.sigma = self._single_output_function(value=init_sigma)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{self.__class__.__name__}: has {num_params} parameters")

    @staticmethod
    def _zero_weight_const_bias(conv, bias):
        conv.weight.data.zero_()  # set weight to zero
        conv.bias.data.fill_(bias)  # set bias to a fixed value

    @staticmethod
    def _single_output_function(value: float):
        """
        Returns a function that outputs a single value for all batches and camera rays.

        Args:
            value (float): the single value.

        Returns:
            fn (function):
                Args:
                    x (torch.Tensor, dtype=float32, shape=(B, N, C): input tensor
                Returns:
                    y (torch.Tensor, dtype=float32, shape=(B, 1, C): output tensor where all values are set to `value'
        """
        def fn(x: torch.Tensor) -> torch.Tensor:
            B, C = x.shape[0], x.shape[2]
            y = torch.full(size=[B, 1, C], fill_value=value, dtype=torch.float32, device=x.device)  # (B, 1, C)
            return y
        return fn

    def forward(self,
                feat: torch.Tensor) -> Pi:
        """
        Args:
            feat (torch.Tensor, dtype=float32, shape=(B, 2H, C)): batch-wise input features.

        Returns:
            pi (Pi, batch_shape=(B, C), event_shape=()): action distribution
        """
        x = feat  # (B, 3, C)

        # forward pass through the backbone network
        x = torch.relu(self.conv1(x))  # (B, 2H, C)
        x = torch.relu(self.conv2(x))  # (B, 2H, C)
        x = torch.relu(self.conv3(x))  # (B, 2H, C)

        # note that the three quantities below should be non-negative
        expansion_f = torch.abs(self.expansion_f(x).squeeze(1))  # (B, C)
        recession_f = torch.abs(self.recession_f(x).squeeze(1))  # (B, C)
        sigma = torch.abs(self.sigma(x).squeeze(1)).clamp(min=1e-6)  # (B, C)

        return self.ers_params_to_pi(feat, expansion_f, recession_f, sigma)

    def ers_params_to_pi(self,
                         feat: torch.Tensor,
                         expansion_f: torch.Tensor,
                         recession_f: torch.Tensor,
                         sigma: torch.Tensor):
        """
        Given expansion and recession parameters computed by the subclass, this function outputs the final Pi

        Args:
            feat (torch.Tensor, dtype=float32, shape=(B, 2H, C)): batch-wise input features.
            expansion_f (torch.Tensor, dtype=float32, shape=(B, C)): front curtain expansion parameters.
            recession_f (torch.Tensor, dtype=float32, shape=(B, C)): front curtain recession parameters.
            sigma (torch.Tensor, dtype=float32, shape=(B, C)): standard deviation of action for each ray.

        Returns:
            pi (Pi, batch_shape=(B, C), event_shape=()): action distribution
        """
        ################################################################################################################
        # Prepare input to ERS
        ################################################################################################################

        # curtain placement and intensities at the most recent timestep
        f_curtain = feat[:, -1, :]  # (B, C)
        f_intensities = feat[:, -2, :]  # (B, C)
        f_hits = (f_intensities >= self.base_policy._LC_INTENSITY_THRESH_F)  # (B, C) dtype=bool

        ################################################################################################################
        # Expand+Recede+Smooth frontier
        ################################################################################################################

        # Expansion
        f_curtain = f_curtain + ~f_hits * expansion_f  # (B, C)

        # Recession
        f_curtain = f_curtain -  f_hits * recession_f  # (B, C)

        # Smoothing
        f_curtain = f_curtain.unsqueeze(1)  # (B, 1, C)
        f_curtain = f_curtain + self.smoothness_dist_matrix  # (B, C, C)
        mu = f_curtain.min(dim=2)[0]  # (B, C)

        # the next line constructs a continuous independent Gaussian distributions on each camera ray
        gaussian = td.Normal(loc=mu.unsqueeze(-1), scale=sigma.unsqueeze(-1))  # BS=(B, C, 1) ES=()

        # we want to eventually output a categorical distribution. we will use the log probability density of the
        # continuous Gaussian at the discretized bins as logits of our categorical distribution!
        logits = gaussian.log_prob(self.discrete_ranges_torch_3d)  # (B, C, R)

        pi = Pi(actions=self.discrete_ranges_torch_3d, logits=logits)  # BS=(B, C) ES=()
        return pi

# endregion
########################################################################################################################
