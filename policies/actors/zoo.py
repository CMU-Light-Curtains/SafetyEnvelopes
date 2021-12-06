from typing import Tuple, Optional
import numpy as np
import torch

from setcpp import enforce_smoothness
from policies.actors import Pi
from policies.actors.actor import Actor
from setcpp import SmoothnessDPL1Cost

from data.synthia import Frame
from devices.light_curtain import LCReturn
import utils

ingr = utils.Ingredient("zoo")
ingr.add_config("config/zoo.yaml")


class ZooActor(Actor):
    @ingr.capture
    def __init__(self,
                 thetas: np.ndarray,
                 min_range: float,
                 max_range: float,
                 smoothnessDPL1Cost: SmoothnessDPL1Cost,
                 kernel_size: int):
        """
        Args:
            thetas (np.ndarray, dtype=np.float32, shape=(C,)): thetas of the camera rays,
                in degrees and in increasing order in [-fov/2, fov/2]
            base_policy (Actor): a base policy that the residual policy policy uses.
            min_range (float): minimum range of the light curtain device
            max_range (float): maximum range of the light curtain device
        """
        # parameters
        self._EPSILON = 10  # epsilon for exploration
        self._ALPHA = 10.0  # alpha for gradient step
        self._MOMENTUM = 0.5  # update running gradient. this is the weight to the previous gradient
        self._SMOOTHNESS = 0.05

        self.C = len(thetas)
        self.K = kernel_size
        self.smoothnessDPL1Cost = smoothnessDPL1Cost

        # mode can be one of { "INIT" / "TIME" / "ESTM" / "STEP" }
        # - INIT: the first light curtain was placed at the location of the safety envelope
        # - TIME: place light curtain at the current location, to measure intensity derivative w.r.t. time
        # - ESTM: place light curtain with a random offset, to measure the ZO intensity derivative w.r.t. placement
        # - STEP: places light curtain at a step in the direction of the running gradient
        self.mode = None
        self.last_curtain = None
        self.last_intensity = None  # (C-K+1,)
        self.random_offset = None
        self.time_delta = None  # (C-K+1,)
        self.running_pos_gradient = None  # (C,)

        kernel = torch.ones([1, 1, self.K], dtype=torch.float32)  # (1, 1, K)

        self.conv1d = torch.nn.Conv1d(1, 1, kernel_size=self.K, bias=False)
        self.conv1d.weight = torch.nn.Parameter(kernel / self.K, requires_grad=False)

        self.deconv1d = torch.nn.ConvTranspose1d(1, 1, kernel_size=self.K, bias=False)
        self.deconv1d.weight = torch.nn.Parameter(kernel, requires_grad=False)

        self.reset()

    def _conv(self, x):
        """
        Args:
            x (np.ndarray, dtype=float32, shape=(C,)): input intensities

        Returns:
            y (np.ndarray, dtype=float32, shape=(C-K+1): output averaged intensities
        """
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, C)
        y = self.conv1d(x)  # (1, 1, C-K+1)
        y = y.squeeze(0).squeeze(0).numpy()  # (C-K+1,)
        return y

    def _deconv(self, y):
        """
        Args:
            y (np.ndarray, dtype=float32, shape=(C-K+1): input averaged intensities

        Returns:
            x (np.ndarray, dtype=float32, shape=(C,)): output intensities

        """
        y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)  # (1, 1, C-K+1)
        x = self.deconv1d(y)  # (1, 1, C)
        x = x.squeeze(0).squeeze(0).numpy()  # (C,)
        return x

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
        # return 14 * np.ones([self.C,], dtype=np.float32), None, {}  # place the first curtain at the initial envelope location (11, 12.5)
        return init_envelope, None, {}  # place the first curtain at the initial envelope location

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
        curtain = obs.lc_ranges.copy()  # (C,)
        intensity = obs.bev_intensities(avg=True) / 255.0  # (C,)  max intensities
        intensity = self._conv(intensity)  # (C-K+1,)

        if self.mode == "INIT":
            # first curtain was just placed at the safety envelope. estimate time gradient
            action = curtain
            self.mode = "TIME"
        elif self.mode == "TIME":
            # curtain was placed at the same location
            self.time_delta = intensity - self.last_intensity  # (C-K+1,) compute time gradient
            # import matplotlib.pyplot as plt
            # plt.plot(self.last_curtain, c='b')
            # plt.plot(curtain, c='r')
            # plt.ylim([0, 20])
            # plt.tight_layout()
            # plt.show()
            # plt.plot(self.last_intensity, c='b')
            # plt.plot(intensity, c='r')
            # plt.tight_layout()
            # plt.ylim([-0.05, 1.05])
            # plt.show()
            # assert (np.abs(self.time_gradient) < 1e-3).all()  # for static scenes
            self.random_offset = self._random_unit_vector()  # next action is at a random offset
            action = curtain + self._EPSILON * self.random_offset
            action = np.array(self.smoothnessDPL1Cost.smoothedRanges(action), dtype=np.float32)  # (C,)
            self.mode = "ESTM"
        elif self.mode == "ESTM":
            # the zero-order gradient estimate (Eqn 4 of https://arxiv.org/pdf/2006.06224.pdf)
            pos_delta = intensity - self.last_intensity  # (C-K+1,)
            pos_delta = pos_delta - self.time_delta  # (C-K+1,) subtract time delta
            pos_gradient = (self.C - self.K + 1) * self._deconv(pos_delta) / self._EPSILON * self.random_offset  # (C,)
            if self.running_pos_gradient is None:
                self.running_pos_gradient = pos_gradient
            else:
                self.running_pos_gradient = self._MOMENTUM * self.running_pos_gradient \
                                            + (1. - self._MOMENTUM) * pos_gradient
            action = self.last_curtain + self._ALPHA * self.running_pos_gradient  # action is along gradient step
            action = np.array(self.smoothnessDPL1Cost.smoothedRanges(action), dtype=np.float32)  # (C,)
            self.mode = "STEP"
        elif self.mode == "STEP":
            action = curtain  # place curtain at same location to estimate time gradient
            self.mode = "TIME"
        else:
            raise Exception("ZOO mode must be one of INIT/TIME/ESTM/STEP")

        self.last_curtain = curtain
        self.last_intensity = intensity

        pi = Pi(actions=torch.from_numpy(action[None, :, None]))  # batch_shape=(1, C) event_shape=()
        info = dict(pi=pi)

        logp_a, control = None, True  # logp_a is None since this policy is deterministic
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
        self.mode = "INIT"
        self.last_curtain = None
        self.last_intensity = None
        self.random_offset = None
        self.running_pos_gradient = None
        self.time_gradient = None

    def _random_unit_vector(self):
        ruv = np.random.randn(self.C)  # (C,)
        ruv = ruv / np.linalg.norm(ruv)  # (C,)
        return ruv
