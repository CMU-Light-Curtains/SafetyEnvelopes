from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np

import torch
import torch.nn as nn

from data.synthia import Frame
from devices.light_curtain import LCReturn
from policies.actors import Pi
from policies.actors.actor import Actor
import utils

ingr = utils.Ingredient("bevnet")
ingr.add_config("config/networks/bevnet.yaml")


@utils.register_class("network")
class BevNet(Actor):
    @ingr.capture
    def __init__(self,
                 thetas: np.ndarray,
                 base_policy: Actor,
                 num_bins: int,
                 min_range: float,
                 max_range: float,
                 history: int,
                 stochastic: bool):
        """
        Args:
            thetas (np.ndarray, dtype=np.float32, shape=(C,)): thetas of the camera rays,
                in degrees and in increasing order in [-fov/2, fov/2]
            base_policy (Actor): a base policy that the residual policy policy uses.
            num_bins (int): number of discrete bins (ranges) per camera ray used in inputs and outputs
            min_range (float): minimum range of the light curtain device
            max_range (float): maximum range of the light curtain device
            history (int): the number of previous observations the actor will use.
            stochastic (bool): whether the policy is stochastic, i.e. whether to sample actions or take the argmax
        """
        self.base_policy = base_policy
        self.stochastic = stochastic

        self.R = num_bins
        self.C = len(thetas)
        self.min_range = min_range
        self.max_range = max_range
        self.bin_length = (self.max_range - self.min_range) / self.R
        utils.pprint(f"BevNet: inputs/outputs are discretized into bins of length {self.bin_length * 100:.1f}cm", "red")

        self.bin_ls = np.linspace(self.min_range, self.max_range - self.bin_length, self.R, dtype=np.float32)  # (R,)
        self.bin_rs = self.bin_ls + self.bin_length  # (R,)
        self.bin_cs = 0.5 * (self.bin_ls + self.bin_rs)  # (R,)

        self.H = history
        self.history = utils.History(size=self.H)

        # discrete_ranges_torch_3d (torch.Tensor, device=cuda, dtype=float32, shape=(1, 1, R))
        discrete_ranges_torch_3d = torch.from_numpy(self.bin_cs[None, None].astype(np.float32))  # (1, 1, R)
        if torch.cuda.is_available():
            discrete_ranges_torch_3d = discrete_ranges_torch_3d.cuda()  # (1, 1, R)

        self.network = CNN(discrete_ranges_torch_3d, self.H)
        if torch.cuda.is_available():
            self.network = self.network.cuda()

    def ranges2bev(self,
                   ranges: np.ndarray,
                   values: Optional[np.ndarray] = None,
                   debug: bool = False) -> np.ndarray:
        """Converts ranges (C,) i.e. a range per camera ray to a 2D bev map (C, R)

        The 2D bev map will be initialized to all zeros, and only those cells that correspond to `ranges' will be
        assigned `values'. If `values' is None, 1 will be assigned to all the range cells.

        Args:
            ranges (np.ndarray, dtype=np.float32, shape=(C,): range per camera ray.
            values (Optional[np.ndarray, dtype=np.float32, shape=(C,)]): values to be placed
            debug (bool): whether to show debug visualizations or not

        Returns:
            bev (np.ndarray, dtype=np.float32, shape=(C, R): a 2D bev map, which is all zeros except those cells where
                                                             the range of each camera ray falls in.
        """
        # convert ranges to inds for each camera ray
        r_inds = ((ranges - self.min_range) / self.bin_length).astype(np.int).clip(min=0, max=self.R-1)  # (C,)

        bev = np.zeros((self.C, self.R), dtype=np.float32)  # (C, R)

        if values is not None:
            bev[np.arange(self.C), r_inds] = values
        else:
            bev[np.arange(self.C), r_inds] = 1

        if debug:
            import matplotlib.pyplot as plt
            plt.plot(ranges); plt.show()
            bev_im = np.flip(bev.T, axis=0)  # (R, C)
            plt.imshow(bev_im); plt.show()

        return bev

    @dataclass
    class IRA:
        i: np.ndarray  # bev map of LC intensities
        r: np.ndarray  # bev map of LC placement ranges
        a: np.ndarray  # bev map of base policy placement

    @property
    def feat_dim(self) -> List[int]:
        # 3H features: 1 intensity bev map, 1 LC placement bev map and 1 base policy placement bev map, per timestep
        return [3 * self.H, self.C, self.R]

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
            act (np.ndarray, dtype=float32, shape=(C,)): sampled actions -- these are absolute, not residual.
            logp_a (Optional[np.ndarray]): log-probability of the sampled actions. None if sampling is deterministic.
            control (bool): whether this policy had control of taking the action at this timestep.
                            - this will be used to determine whether to evaluate the policy at this frame.
                            - another eg. is that these are the timesteps when nn-based policies will run the network.
            info (dict): auxiliary info generated by policy, such as a vector representation of the observation while
                         generating the action.
        """
        self.network.eval()

        # convert obs to intensity and placement bev map
        intensities = obs.bev_intensities() / 255.0  # (C,)
        i = self.ranges2bev(obs.lc_ranges, values=intensities)  # (C, R)
        r = self.ranges2bev(obs.lc_ranges)  # (C, R)

        # get action that the base policy would have taken under this observation as input
        base_act, logp_a, _, _ = self.base_policy.step(obs)
        a = self.ranges2bev(base_act)  # (C, R)

        # create IRA of current timestep and add to history
        ira = BevNet.IRA(i, r, a)
        self.history.add(ira)

        # create features from history
        inputs = sum([[ira.i, ira.r, ira.a] for ira in self.history.get()], [])  # list of 3H vectors of shape (C, R)
        feat = np.stack(inputs, axis=0)  # (3H, C, R)

        # action from network
        feat_tensor = torch.from_numpy(feat).unsqueeze(dim=0)  # (1, 3H, C, R)
        if torch.cuda.is_available():
            feat_tensor = feat_tensor.cuda()
        pi: Pi = self.forward(feat_tensor)  # batch_shape=(1, C) event_shape=()

        # get actions from pi, either by sampling stochastically, or by deterministic argmax
        if self.stochastic:
            act, act_inds = pi.sample()  # both (1, C)
        else:
            act, act_inds = pi.argmax()  # both (1, C)

        logp_a = pi.log_prob(act_inds).squeeze(0).cpu().numpy()
        act = act.squeeze(0).cpu().numpy()  # (C,) dtype=float32
        act_inds = act_inds.squeeze(0).cpu().numpy()  # (C,) dtype=int64

        control = True
        info    = dict(
            features=feat,  # (3H, C, R)
            base_action=base_act,  # (C,)
            pi=pi,  # batch_shape=(1, C) event_shape=()
            action_inds=act_inds  # (C,) dtype=int64
        )
        return act, logp_a, control, info

    def forward(self,
                feats: torch.Tensor) -> Pi:
        """
        Args:
            feats (torch.Tensor, dtype=float32, shape=(B, 3H, C, R)): features.

        Returns:
            pi (Pi, batch_shape=(B, C), event_shape=()): action distribution.
        """
        pi: Pi = self.network(feats)  # batch_shape=(B, C) event_shape=()
        return pi

    def reset(self):
        super().reset()
        self.history.clear()


########################################################################################################################
# region UNet Parts
########################################################################################################################


class DoubleConv(nn.Module):
    # modified from https://github.com/milesial/Pytorch-UNet
    """(conv => BN => ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    # modified from https://github.com/milesial/Pytorch-UNet
    """Downsampling with MaxPool => DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((4, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    # modified from https://github.com/milesial/Pytorch-UNet
    """Upsampling => DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(4, 2), mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """x1 after up and conv, and x2 should be of the same size"""
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))

# endregion
########################################################################################################################
# region Convolutional Neural Network
########################################################################################################################


class CNN(nn.Module):
    """Performs 2D convolutions"""
    def __init__(self, discrete_ranges_torch_3d, H):
        super().__init__()

        S = 2  # scale to adjust the number of parameters

        self.in_c = DoubleConv(3*H, 4*S)
        self.down1 = Down(4*S, 8*S)
        self.down2 = Down(8*S, 8*S)
        self.up1 = Up(16*S, 4*S)
        self.up2 = Up(8*S,  2*S)
        self.out_c = OutConv(2*S + 3*H, 1)

        self.discrete_ranges_torch_3d = discrete_ranges_torch_3d

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{self.__class__.__name__}: has {num_params} parameters")

    def forward(self,
                feat: torch.Tensor) -> Pi:
        """
        Args:
            feat (torch.Tensor, dtype=np.float32, shape=(B, 3, C, R)): batch-wise input features.

        Returns:
            pi (Pi, batch_shape=(B, C), event_shape=()): action distribution
        """
        x0 = feat  # (B, 3H, C, R)

        x1 = self.in_c(x0)     # (B, 4S, C,    R)
        x2 = self.down1(x1)    # (B, 8S, C/4,  R/2)
        x3 = self.down2(x2)    # (B, 8S, C/16, R/4)
        x = self.up1(x3, x2)   # (B, 4S, C/4,  R/2)
        x = self.up2(x, x1)    # (B, 2S, C,    R)
        x = self.out_c(x, x0)  # (B, 1 , C,    R)

        x = x.squeeze(dim=1)  # (B, C, R)

        pi = Pi(actions=self.discrete_ranges_torch_3d, logits=x)  # batch_shape=(B, C), event_shape=()
        return pi

# endregion
########################################################################################################################
