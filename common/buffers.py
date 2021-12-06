from dataclasses import dataclass
from typing import Optional, Generator, Union, List, Tuple, NoReturn
import numpy as np
from gym import spaces
import scipy.signal
import torch

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize


########################################################################################################################
# region Imitation Buffer
########################################################################################################################


@dataclass
class ImitationBufferSamples:
    features: np.ndarray
    actions: np.ndarray
    demonstrations: np.ndarray


class ImitationBuffer(BaseBuffer):
    """
    Storage buffer for imitation learning. It can:

    1. Get batches even before the buffer is full.
    2. Add examples even after the buffer is full (according to FIFO).
    """
    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[torch.device, str] = "cpu",
                 ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=1)

        self.features, self.actions, self.demonstrations = None, None, None
        self.reset()

    def reset(self) -> None:
        self.features = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.demonstrations = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        super().reset()

    def add(self,
            feat: np.ndarray,
            act : np.ndarray,
            dem : np.ndarray) -> None:
        """
        :param feat: (np.ndarray) Features from observations
        :param act : (np.ndarray) Action
        :param dem : (np.ndarray) Demonstration
        """
        self.features[self.pos] = np.array(feat).copy()
        self.actions[self.pos] = np.array(act).copy()
        self.demonstrations[self.pos] = np.array(dem).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0  # FIFO: start overwriting oldest examples

    def num_samples(self):
        """Number of samples currently in the buffer"""
        return self.buffer_size if self.full else self.pos

    def inf_batches(self,
                    batch_size: Optional[int] = None) -> Generator[ImitationBufferSamples, None, None]:
        """Produces an infinite iterator of minibatches."""
        assert self.num_samples() >= 1  # buffer cannot be empty

        if batch_size is None:
            batch_size = self.num_samples()

        indices = np.random.permutation(self.num_samples())

        start_idx = 0
        while True:
            batch_indices = np.array([], dtype=np.int)

            while len(batch_indices) < batch_size:  # obtaining the set of indices for the current batch
                if start_idx == 0:
                    # new permutation
                    np.random.shuffle(indices)
                added_indices = indices[start_idx:start_idx + batch_size - len(batch_indices)]
                batch_indices = np.hstack([batch_indices, added_indices])
                start_idx = start_idx + len(added_indices)
                if start_idx >= len(indices):
                    start_idx = 0

            yield self._get_samples(batch_indices)

    def single_pass(self,
                    batch_size: Optional[int] = None) -> Generator[ImitationBufferSamples, None, None]:
        """Produces an iterator that cycle through all examples once"""
        assert self.num_samples() >= 1  # buffer should not be empty

        if batch_size is None:
            batch_size = self.num_samples()

        indices = np.random.permutation(self.num_samples())

        start_idx = 0
        while start_idx < len(indices):
            batch_inds = indices[start_idx:start_idx + batch_size]
            yield self._get_samples(batch_inds)
            start_idx += len(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> ImitationBufferSamples:
        feat: np.ndarray = self.features[batch_inds]
        act : np.ndarray = self.actions[batch_inds]
        dem : np.ndarray = self.demonstrations[batch_inds]
        return ImitationBufferSamples(feat, act, dem)
    
    def dump(self, save_dir, episode):
        # save the data in the npz file
        assert len(self.features) == self.buffer_size
        assert len(self.actions) == self.buffer_size
        assert len(self.demonstrations) == self.buffer_size

        # each feature, action and demonstration gets its own file
        for i, (f, a, d) in enumerate(zip(self.features, self.actions, self.demonstrations)):
            npy_path = save_dir / f'episode_{episode}_iter_{i}.npy'
            save_dict = dict(features=f, actions=a, demonstrations=d)
            np.save(npy_path, save_dict)
        self.reset()
    
    def load(self, feats: np.ndarray, acts: np.ndarray, demos: np.ndarray) -> NoReturn:
        """
        Args:
            feats: np.ndarray (T, *AF)
            acts: np.ndarray (T, C)
            demos: np.ndarray (T, C)
        return: NoReturn
        """
        # just to be extra sure
        assert len(feats) == len(demos)
        assert len(feats) == len(acts)

        self.features[range(len(feats))] = feats
        self.actions[range(len(acts))] = acts
        self.demonstrations[range(len(demos))] = demos

        # update the position pointer
        self.pos = len(feats)
        assert self.pos == self.buffer_size, "buffers were created such that the condition holds"
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0  # FIFO: start overwriting oldest examples

# endregion
########################################################################################################################
# region Rollout Buffer
########################################################################################################################

# copied and modified from two sources:
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
# https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo


@dataclass
class RolloutBufferSamples:
    actor_features: torch.Tensor
    critic_features: torch.Tensor
    action_inds: torch.Tensor
    gt_actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    costs: torch.Tensor
    returns: torch.Tensor


def discount_cumsum(x: np.ndarray,
                    discount: float) -> np.ndarray:
    """
    Copied from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py

    Args:
        x (np.array, dtype=float32, shape=(N,)): input vector
        discount (float): discount value

    Returns:
        y (np.ndarray, dtype=float32, shape=(N,)): transformed vector

    Example:
        input:
            [x0,
             x1,
             x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class RolloutBuffer(BaseBuffer):
    IntListOrTuple = Union[List[int], Tuple[int, ...]]

    def __init__(self,
                 buffer_size: int,
                 actor_feat_dim: IntListOrTuple,
                 critic_feat_dim: IntListOrTuple,
                 action_dim: IntListOrTuple,
                 device: Union[torch.device, str] = "cuda",  # GPU by default
                 gamma: float = 0.99):

        self.buffer_size = buffer_size
        self.actor_feat_dim = tuple(actor_feat_dim)
        self.critic_feat_dim = tuple(critic_feat_dim)
        self.action_dim = tuple(action_dim)
        self.pos = 0
        self.path_start_idx = 0
        self.full = False
        self.device = device
        self.n_envs = 1
        self.gamma = gamma

        self.actor_features, self.critic_features = None, None
        self.action_inds, self.gt_actions = None, None
        self.costs, self.returns, self.values, self.advantages, self.log_probs = None, None, None, None, None
        self.reset()

    def reset(self) -> None:
        self.actor_features = np.zeros((self.buffer_size,) + self.actor_feat_dim, dtype=np.float32)
        self.critic_features = np.zeros((self.buffer_size,) + self.critic_feat_dim, dtype=np.float32)
        self.action_inds = np.zeros((self.buffer_size,) + self.action_dim, dtype=np.int64)
        self.gt_actions = np.zeros((self.buffer_size,) + self.action_dim, dtype=np.float32)
        self.costs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)

        self.pos, self.path_start_idx = 0, 0
        self.full = False

    def add(self,
            actor_feat: np.ndarray,
            critic_feat: np.ndarray,
            action_inds: np.ndarray,
            gt_action: np.ndarray,
            cost: float,
            value: float,
            log_prob: float) -> NoReturn:
        """
        Args:
            actor_feat (np.ndarray, dtype=float32, shape=(*AF): input features for the actor
            critic_feat (np.ndarray, dtype=float32, shape=(*CF): input features for the critic
            action_inds (np.ndarray, dtype=int64, shape=(C,): indices of the categorical action for each cam ray.
            gt_action (np.ndarray, dtype=float32, shape=(C,)): ground truth actions.
            cost (float): cost
            value (float): value estimated by the critic under the current policy
            log_prob (np.ndarray, float32, shape=(,)): log probability of the action following the current policy.
        """
        self.actor_features[self.pos] = np.array(actor_feat).copy()
        self.critic_features[self.pos] = np.array(critic_feat).copy()
        self.action_inds[self.pos] = np.array(action_inds).copy()
        self.gt_actions[self.pos] = np.array(gt_action).copy()
        self.costs[self.pos] = cost
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def finish_path(self,
                    last_value: float) -> NoReturn:
        """
        Modified from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

        Call this at the end of a trajectory, or when one gets cut off by an epoch ending. This looks back in the buffer
        to where the trajectory started, and uses rewards and value estimates from the whole trajectory to compute
        advantage estimates, as well as compute the rewards-to-go for each state, to use as the targets for the value
        function.

        The "last_val" argument should be 0 if the trajectory ended because the agent reached a terminal state
        (absorbing state with 0 reward) and otherwise should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account for timesteps beyond the arbitrary episode
        horizon (or epoch cutoff).
        """
        # this asserts that the path length is non-zero i.e. self.add() has been called at least once for this path
        assert self.pos > self.path_start_idx

        path_slice = slice(self.path_start_idx, self.pos)
        path_costs = np.append(self.costs[path_slice], last_value)

        # the next line computes costs-to-go, to be targets for the value function
        path_returns = discount_cumsum(path_costs, discount=self.gamma)[:-1]
        self.returns[path_slice] = path_returns
        self.advantages[path_slice] = path_returns - self.values[path_slice]

        self.path_start_idx = self.pos

    def get(self,
            batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.actor_features[batch_inds],
            self.critic_features[batch_inds],
            self.action_inds[batch_inds],
            self.gt_actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.costs[batch_inds].flatten(),
            self.returns[batch_inds].flatten()
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


# endregion
########################################################################################################################
