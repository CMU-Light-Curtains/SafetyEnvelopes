from dataclasses import dataclass
import numpy as np
import torch
from typing import Tuple, Union
from policies.actors import Pi
import utils


@dataclass
class MetricHuber:
    sum_huber: float
    huber: float

    def str(self,
            total: bool):
        msg = ""
        if total:
            msg += f"| total huber loss: {self.sum_huber:7.4f} "
        msg += f"| avg huber loss: {self.huber:7.4f}"
        return msg

    def __str__(self):
        return self.str(total=False)


@utils.register_class("metric")
class Huber:
    """Performs ground truth evaluation of the huber loss"""
    def __init__(self,
                 min_range: float,
                 max_range: float,
                 thetas: np.ndarray,
                 delta: float = 0.1,
                 debug: bool = False):
        """
        Args:
            min_range (float): rays which have range lesser than this will not be included for evaluation.
            max_range (float): rays which have range greater than this will not be included for evaluation.
            thetas (np.ndarray, dtype=float32, shape=(C,)): camera angles.
            debug (bool): whether to debug or not
        """
        self._min_range = min_range
        self._max_range = max_range
        self._delta = delta
        self._debug = debug
        self._thetas = thetas

        self._store = {}

    def _torch_huber_raywise(self,
                             actions: torch.Tensor,
                             probs: torch.Tensor,
                             gt_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the huber loss. All inputs and outputs are torch tensors.

        Args:
            actions (torch.Tensor, dtype=float32, shape=(*L, C, R)): R actions per camera ray.
            probs (Optional[torch.Tensor, dtype=float32, shape=(*L, C, R)]): probability of taking each of the R actions.
                `probs' must sum to 1 along the last axis.
            gt_action (torch.Tensor, dtype=float32, shape=(*L, C,)): ground truth action.

        Returns:
            huber (torch.Tensor, dtype=float32, shape=(*L, C)): raywise huber loss.
            mask (torch.Tensor, dtype=float32, shape=(*L, C)): mask for huber loss.
        """
        assert torch.all(gt_action > 0)

        delta = self._delta

        ratio = actions / (gt_action + 1e-5).unsqueeze(dim=-1)  # (*L, C, R)
        error = torch.abs(ratio - 1.0)  # (B, C, R)

        # quadratic term
        qdr = torch.square(error) / (delta * (2.0 - delta))  # (*L, C, R)
        # linear term
        lin = (2.0 * error - delta) / (2.0 - delta)  # (*L, C, R)
        # combination of linear and quadratic terms
        huber = torch.where(error < delta, qdr, lin)  # (*L, C, R)

        if probs is None:
            assert huber.shape[-1] == 1, "probs can only be None if only one action provided per camera ray"
            huber = huber.squeeze(dim=-1)  # (*L, C)
        else:
            # take expectation of the huber loss w.r.t. probs
            huber = (huber * probs).sum(dim=-1)  # (*L, C)

        # masking: only the rays inside the mask contribute towards the mean huber loss
        mask = (self._min_range < gt_action) & (gt_action < self._max_range)

        return huber, mask

    def _torch_huber(self,
                     actions: torch.Tensor,
                     probs: torch.Tensor,
                     gt_action: torch.Tensor) -> torch.Tensor:
        """
        Computes the huber loss. All inputs and outputs are torch tensors.

        Args:
            actions (torch.Tensor, dtype=float32, shape=(*L, C, R)): R actions per camera ray.
            probs (Optional[torch.Tensor, dtype=float32, shape=(*L, C, R)]): probability of taking each of the R actions.
                `probs' must sum to 1 along the last axis.
            gt_action (torch.Tensor, dtype=float32, shape=(*L, C,)): ground truth action.

        Returns:
            huber (torch.Tensor, dtype=float32, shape=(*L)): huber loss, separate for every element in the batch.
        """
        huber, mask = self._torch_huber_raywise(actions, probs, gt_action)

        # huber is 0 (perfect) where the mask is empty
        huber = (huber * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (*L,)

        return huber

    ####################################################################################################################
    # region Loss API
    ####################################################################################################################

    def loss(self,
             pi: Pi,
             gt_action: Union[np.ndarray, torch.Tensor],
             reduce: bool = True,
             device: str = 'cuda') -> torch.Tensor:
        """
        Computes huber loss that can be used to backprop during training

        Args:
            pi (Pi, batch_shape=(*L, C) event_shape=()): action distribution.
            gt_action (np.ndarray/torch.Tensor, dtype=float32, shape=(*L, C)): ground truth action.
            reduce (bool): if True, returns the mean of the huber loss across the batch dimensions.
            device (str): which torch.Tensor device the computation should be done on.

        Returns:
            huber (torch.Tensor, dtype=float32, shape=(*L) or (,)): huber loss.
        """
        if type(gt_action) != torch.Tensor:
            gt_action = torch.from_numpy(gt_action)  # (*L, C)

        huber = self._torch_huber(pi.actions.to(device), pi.probs.to(device), gt_action.to(device))  # (*L)

        if reduce:
            huber = huber.mean()  # (,)

        return huber  # (*L) or (,)

    # endregion
    ####################################################################################################################
    # region Evaluation API
    ####################################################################################################################

    def reset(self):
        self._store.clear()

    def add(self,
            name: str,
            pi: Pi,
            gt_action: np.ndarray) -> float:
        """
        Args:
            name (str): name of the policy for which the metric is being computed
            pi (Pi, batch_shape=(1, C), event_shape=()): action probability distribution
            gt_action (np.ndarray, dtype=float32, shape=(C,): ground truth ranges (safety envelope).

        Returns:
            loss (float): huber loss of the input pi and gt_action; this is exactly what is added to the evaluation.
        """
        gt_action = gt_action.reshape(1, -1)  # (1, C)
        if name not in self._store:
            self._store[name] = []

        # use CPU torch tensors while computing metrics
        huber = self.loss(pi, gt_action, reduce=True, device='cpu').item()

        self._store[name].append(huber)
        return huber

    def metric(self,
               name: str) -> MetricHuber:
        """
        Args:
            name (str): name of policy

        Returns:
            MetricSER: metric of `name' policy.
        """
        if name not in self._store:
            raise Exception(f"Policy name {name} not found in Evaluator")

        huber_array = np.array(self._store[name])
        metric = MetricHuber(sum_huber=huber_array.sum(),
                             huber=huber_array.mean())

        return metric

    # endregion
    ####################################################################################################################
    # Elementwise Loss API
    ####################################################################################################################

    def raywise_loss(self,
                     pi: Pi,
                     gt_action: np.ndarray) -> np.ndarray:
        """
        Computes a huber loss separate for each camera ray (on CPU).
        This is useful as an input feature in critics.

        Args:
            pi (Pi, batch_shape=(C,) event_shape=()): action distribution.
            gt_action (np.ndarray, dtype=float32, shape=(C,)): ground truth action.

        Returns:
            huber (np.ndarray, dtype=float32, shape=(C,)): elementwise huber loss.
        """
        # convert to CPU torch tensors while computing metrics
        gt_action = torch.from_numpy(gt_action)  # (C,)
        huber, mask = self._torch_huber_raywise(pi.actions.cpu(), pi.probs.cpu(), gt_action)
        huber = (huber * mask).numpy()  # (C,)
        return huber

    # endregion
    ####################################################################################################################
