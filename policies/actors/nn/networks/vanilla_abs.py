from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
import numpy as np

import torch
import torch.nn as nn

from data.synthia import Frame
from devices.light_curtain import LCReturn
from policies.actors import Pi
from policies.actors.actor import Actor
from policies.actors.nn.features.absolute import NNAbsoluteFeatures, NNAbsoluteFeaturizer, CurtainInfo
import utils


@utils.register_class("network")
class VanillaAbs(Actor):
    def __init__(self,
                 thetas: np.ndarray,
                 base_policy: Actor,
                 min_range: float,
                 max_range: float):
        """
        Args:
            thetas (np.ndarray, dtype=np.float32, shape=(C,)): thetas of the camera rays,
                in degrees and in increasing order in [-fov/2, fov/2]
            base_policy (Actor): a base policy that is used for the first action.
            min_range (float): minimum range of the light curtain device
            max_range (float): maximum range of the light curtain device
        """

        self.network = CNN()
        if torch.cuda.is_available():
            self.network = self.network.cuda()

        self.base_policy = base_policy

        self.featurizer = NNAbsoluteFeaturizer(thetas)

        self.history: List[VanillaAbs.PrevOC] = []

    @dataclass
    class PrevOC:
        o: LCReturn  # observation at time t-1
        c: CurtainInfo  # curtain info from observations time t-1

    @property
    def feat_dim(self) -> int:
        return self.featurizer.feat_dim

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

        # convert obs to cinfo and get base policy action
        # all these quantities are for current timestep
        cinfo = self.featurizer.obs_to_curtain_info(obs)
        oc = VanillaAbs.PrevOC(obs, cinfo)

        # first timestep, use base action.
        if len(self.history) == 0:
            act, logp_a, _, _ = self.base_policy.step(obs)
            self.history.append(oc)
            return act, None, False, {}

        cinfo_prev = self.history[0].c
        feat = self.featurizer.cinfos2feats(cinfo_prev, cinfo)
        self.history[0] = oc  # update history

        # action from network
        pi: Pi = self.forward(feat)  # (td.distribution, batch_shape=(1,), event_shape=(C,)
        act, _ = pi.sample()  # both (1, C)
        act: np.ndarray = act.squeeze(0).cpu().numpy()  # (C,)

        logp_a  = None
        control = True
        info    = dict(
            features=feat.to_numpy().squeeze(0),  # (F,)
            pi=pi  # batch_shape=(1, C) event_shape=()
        )
        return act, logp_a, control, info

    def forward(self,
                feats: Union[torch.Tensor, NNAbsoluteFeatures]) -> Pi:
        """
        Args:
            feats (torch.Tensor, dtype=float32, shape=(B, F)): features.

        Returns:
             pi (Pi, batch_shape=(B, C), event_shape=()): action distribution.
        """
        if type(feats) is not NNAbsoluteFeatures:
            feats = feats.cpu().numpy()
            feats: NNAbsoluteFeatures = NNAbsoluteFeatures.from_numpy(feats)  # NNAbsoluteFeatures

        pi: Pi = self.network(feats)  # batch_shape=(B, C) event_shape=()
        return pi

    def evaluate_actions(self,
                         feat: np.ndarray,
                         act : np.ndarray) -> torch.Tensor:
        """
        Args:
            feat (np.ndarray, dtype=np.float32, shape=(B, F)): batch of features.
            act (np.ndarray, dtype=np.float32, shape=(B, C)): batch of actions.

        Returns:
            logp_a (torch.Tensor): log probability of taking actions "act" by the actor under "obs", as a torch tensor.
        """
        feat = NNAbsoluteFeatures.from_numpy(feat)  # batch_size = (B,)
        pi = self.forward(feat)  # (td.Distribution, batch_shape=(B,) event_shape=(C,))
        act = torch.from_numpy(act)  # (B, C)
        if torch.cuda.is_available():
            act = act.cuda()
        logp_a = pi.log_prob(act)  # (B,)
        return logp_a

    def reset(self):
        super().reset()
        self.history.clear()

########################################################################################################################
# region Convolutional Neural Networks
########################################################################################################################


class CNN(nn.Module):
    """Performs 1D convolutions"""
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(21, 11, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(11)
        self.conv2 = nn.Conv1d(11,  5, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(5)

        self.mu    = nn.Conv1d(5,   1, kernel_size=5, padding=2)
        self.sigma = nn.Conv1d(5,   1, kernel_size=5, padding=2)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{self.__class__.__name__}: has {num_params} parameters")

    def forward(self,
                feat: NNAbsoluteFeatures) -> Pi:
        """
        Args:
            feat (NNAbsoluteFeatures): features with 2D tensors, where the first dimension is the batch

        Returns:
            pi (Pi, batch_shape=(B, C), event_shape=()): action distribution
        """
        B = feat.c1.i.shape[0]  # batch size

        def pad_zeros(x: np.ndarray,
                      left: bool):
            """
            Args:
                x (np.ndarray, dtype=np.float32, shape=(B, 2): 2D vector to pad zeros to
                left (bool): whether to pad zeros to the left or not (right).
            """
            assert x.ndim == 2 and x.shape[0] == B
            if left:
                return np.hstack([np.zeros([B, 1]).astype(x.dtype), x])
            else:
                return np.hstack([x, np.zeros([B, 1]).astype(x.dtype)])

        # horizontal edges
        eh1 = feat.c1[:, 1:] - feat.c1[:, :-1]
        eh2 = feat.c2[:, 1:] - feat.c2[:, :-1]
        # vertical edges
        ev = feat.c2 - feat.c1

        x = np.stack([
                      # intensities
                      feat.c1.i,
                      feat.c2.i,
                      # horizontal edges 1
                      pad_zeros(eh1.x, left=True ),
                      pad_zeros(eh1.x, left=False),
                      pad_zeros(eh1.z, left=True ),
                      pad_zeros(eh1.z, left=False),
                      pad_zeros(eh1.r, left=True ),
                      pad_zeros(eh1.r, left=False),
                      pad_zeros(eh1.t, left=True ),
                      pad_zeros(eh1.t, left=False),
                      # horizontal edges 2
                      pad_zeros(eh2.x, left=True),
                      pad_zeros(eh2.x, left=False),
                      pad_zeros(eh2.z, left=True),
                      pad_zeros(eh2.z, left=False),
                      pad_zeros(eh2.r, left=True),
                      pad_zeros(eh2.r, left=False),
                      pad_zeros(eh2.t, left=True),
                      pad_zeros(eh2.t, left=False),
                      # vertical edges
                      ev.x,
                      ev.z,
                      ev.r,
                      ], axis=1)  # (B, 21, C)

        x = torch.from_numpy(x)  # (B, 21, C)
        if torch.cuda.is_available():
            x = x.cuda()

        x = torch.relu(self.bn1(self.conv1(x)))  # (B, 11, C)
        x = torch.relu(self.bn2(self.conv2(x)))  # (B,  5, C)

        # mu: absolute
        mu = self.mu(x).squeeze(1)   # (B, C)

        c2_r = torch.from_numpy(feat.c2.r)
        if torch.cuda.is_available():
            c2_r = c2_r.cuda()

        mu = mu + c2_r  # (B, C)

        # sigma = 1e-5 + torch.relu(self.sigma(x)).squeeze(1)  # (B, C)
        sigma = 0.

        mu = mu.unsqueeze(-1)  # (B, C, 1)
        pi = Pi(actions=mu)
        return pi

# endregion
########################################################################################################################
