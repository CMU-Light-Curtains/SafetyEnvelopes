from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional, Tuple

from devices.light_curtain import LCReturn


class Critic(ABC):
    @abstractmethod
    def step(self,
             obs: LCReturn,
             gt_action: np.ndarray) -> Tuple[float, np.ndarray, bool]:
        """
        Args:
            obs (LCReturn): observations viz return from the front light curtain.
            gt_action (np.ndarray, dtype=float32, shape=(C,)): ranges of GT safety envelope.

        Returns:
            value (float): the value of the observation under the ground truth.
            feat (np.ndarray, dtype=float32, shape=(*F): the input features used by the critic to compute this value.
            control (bool): these are the timesteps when nn-based critics have sufficient history to run the network.
                            When control is False, returned value is garbage.
        """
        raise NotImplementedError

    def forward(self,
                feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats (torch.Tensor, dtype=np.float32, shape=(B, *F)): features of dimensions *F.
        
        Returns:
            value (torch.Tensor, device=cuda, dtype=float32, shape=(B,)): values for batch.
        """
        raise NotImplementedError

    def reset(self):
        """
        Generally, critics can maintain a history of observations from the environment, and values can be a function of
        the entire history. Reset is used to clear any such histories.
        """
        pass
