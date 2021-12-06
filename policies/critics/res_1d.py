from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np

import torch
import torch.nn as nn

from devices.light_curtain import LCReturn
from policies.actors import Pi
from policies.actors.actor import Actor
from policies.critics.critic import Critic
import utils


@utils.register_class("critic")
class Res1DCritic(Critic):
    def __init__(self,
                 thetas: np.ndarray,
                 raywise_loss_fn: Callable[[Pi, np.ndarray], np.ndarray],
                 base_policy: Actor):
        """
        Args:
            thetas (np.ndarray, dtype=float32, shape=(C,)): thetas of the camera rays, in degrees and in
                increasing order in [-fov/2, fov/2].
            raywise_loss_fn (Callable[[Pi, np.ndarray], np.ndarray]): the loss function whose value the critic is
                estimating.
                [Args:
                    pi (Pi, batch_shape=(C,) event_shape=()): action distribution.
                    gt_action (np.ndarray, dtype=float32, shape=(C,)): ground truth action.
                Returns:
                    raywise_loss (np.ndarray, dtype=float32, shape=(C,): loss for each camera ray.
                ]
            base_policy (Actor): a base policy that the residual policy policy uses.
        """
        self.raywise_loss_fn = raywise_loss_fn
        self.base_policy = base_policy
        self.C = len(thetas)

        self.history: List[Res1DCritic.PrevPIL] = []

        self.network = CNN()
        if torch.cuda.is_available():
            self.network = self.network.cuda()

    @dataclass
    class PrevPIL:
        p: np.ndarray  # placement of light curtain at time t-1
        i: np.ndarray  # intensity of light curtain at time t-1
        l: np.ndarray  # loss at time t-1

    @property
    def feat_dim(self) -> List[int]:
        return [8, self.C]  # 8 features: 2 x (placement + intensity + loss) + base action + GT action

    @torch.no_grad()
    def step(self,
             obs: LCReturn,
             gt_action: np.ndarray) -> Tuple[float, np.ndarray, bool]:
        """
        Args:
            obs (LCReturn): observations viz return from the front light curtain.
            gt_action (np.ndarray, dtype=float32, shape=(C,)): ranges of GT safety envelope.

        Returns:
            value (float): the value of the observation under the ground truth.
            feat (np.ndarray, dtype=float32, shape=(8, C): the input features used by the critic to compute this value.
            control (bool): these are the timesteps when nn-based critics have sufficient history to run the network.
                            When control is False, returned value is garbage.
        """
        self.network.eval()

        # get placement and intensities from observation
        p = obs.lc_ranges  # (C,)
        i = obs.bev_intensities() / 255.0  # (C,)
        pi = Pi(actions=torch.from_numpy(p).unsqueeze(-1))  # batch_shape=(C,) event_shape=()
        l = self.raywise_loss_fn(pi, gt_action)  # (C,)

        # get base policy action
        base_act, _, _, _ = self.base_policy.step(obs)  # (C,)

        # create pi of current timestep
        pil = Res1DCritic.PrevPIL(p, i, l)

        # cannot produce a value in the first timestep
        if len(self.history) == 0:
            self.history.append(pil)
            return 0., np.empty([], dtype=np.float32), False

        p_prev = self.history[0].p  # (C,)
        i_prev = self.history[0].i  # (C,)
        l_prev = self.history[0].l  # (C,)

        feat = np.stack([p_prev,
                         i_prev,
                         l_prev,
                         p,
                         i,
                         l,
                         base_act,
                         gt_action], axis=0)  # (8, C)
        self.history[0] = pil  # update history

        # value from network
        feat_tensor = torch.from_numpy(feat).unsqueeze(dim=0)  # (1, 8, C)
        if torch.cuda.is_available():
            feat_tensor = feat_tensor.cuda()
        value = self.forward(feat_tensor).item()

        return value, feat, True  # (8, C)

    def forward(self,
                feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats (torch.Tensor, dtype=np.float32, shape=(B, 8, C)): input features.

        Returns:
            value (torch.Tensor, device=cuda, dtype=float32, shape=(B,)): values for batch.
        """
        values = self.network(feats)  # (B,)
        return values

    def reset(self):
        super().reset()
        self.history.clear()


########################################################################################################################
# region Convolutional Neural Network
########################################################################################################################


class CNN(nn.Module):
    """Performs 1D convolutions"""
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(8, 4, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(4, 1, kernel_size=5, padding=2)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{self.__class__.__name__}: has {num_params} parameters")

    def forward(self,
                feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat (torch.Tensor, dtype=np.float32, shape=(B, 8, C)): input features.

        Returns:
            value (torch.Tensor, device=cuda, dtype=float32, shape=(B,)): values for batch.
        """
        x = feat  # (B, 8, C)

        # the next lines compute the loss of the most recent timestep
        l = feat[:, 5, :]  # (B, C) raywise loss * mask
        m = (torch.abs(l) > 1e-6).sum(dim=1).float().clamp(min=1)  # (B,) mask
        loss = l.sum(dim=1) / m  # (B,)

        x = torch.relu(self.conv1(x))  # (B, 4, C)
        x = torch.relu(self.conv2(x))  # (B, 1, C)
        x = x.mean(dim=-1).squeeze(dim=-1)  # (B,)

        # the final value is computed as a residual of the most recent loss
        return loss + x

# endregion
########################################################################################################################
