from dataclasses import dataclass
import numpy as np
import torch
from typing import Union, Dict
from policies.actors import Pi
import utils


@dataclass
class MetricStandard:
    rmse_linear: float
    rmse_log: float
    rmse_lsi: float
    abs_rel_diff: float
    sq_rel_diff: float
    threshold1: float
    threshold2: float
    threshold3: float

    def str(self):
        msg = ""
        msg += f"| avg rmse linear: {self.rmse_linear:7.4f} "
        msg += f"| avg rmse log: {self.rmse_log:7.4f} "
        msg += f"| avg rmse log scale invariant: {self.rmse_lsi:7.4f} "
        msg += f"| avg abs rel diff: {self.abs_rel_diff:7.4f} "
        msg += f"| avg sq rel diff: {self.sq_rel_diff:7.4f} "
        msg += f"| avg threshold1: {self.threshold1:7.4f} "
        msg += f"| avg threshold2: {self.threshold2:7.4f} "
        msg += f"| avg threshold3: {self.threshold3:7.4f}"
        return msg

    def __str__(self):
        return self.str()


@utils.register_class("metric")
class Standard:
    """Performs ground truth evaluation of standard metrics"""
    def __init__(self,
                 min_range: float,
                 max_range: float,
                 thetas: np.ndarray,
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
        self._thetas = thetas  # (C,)
        self._debug = debug

        self._cos_thetas = np.cos(np.deg2rad(self._thetas))  # (C,)
        self._cos_thetas = torch.from_numpy(self._cos_thetas)  # (C,)

        # various threshold values for the threshold metric
        self._thresh1 = 1.25
        self._thresh2 = 1.25**2
        self._thresh3 = 1.25**3

        self.metric_names = ['rmse_linear',
                             'rmse_log',
                             'rmse_lsi',
                             'abs_rel_diff',
                             'sq_rel_diff',
                             'threshold1',
                             'threshold2',
                             'threshold3']
        self._store = {}
    
    def _torch_standard(self,
                        actions: torch.Tensor,
                        probs: torch.Tensor,
                        gt_action: torch.Tensor) -> Dict:
        """
        Args:
            actions (torch.Tensor, dtype=float32, shape=(*L, C, R)): R actions per camera ray.
            probs (Optional[torch.Tensor, dtype=float32, shape=(*L, C, R)]): probability of taking each of the R actions.
                `probs' must sum to 1 along the last axis.
            gt_action (torch.Tensor, dtype=float32, shape=(*L, C,)): ground truth action.

        Returns:
            standard (dict): standard metrics.
        """
        assert torch.all(gt_action > 0)

        # convert gt_action and actions, which are ranges, to depths
        true_depths = gt_action * self._cos_thetas  # (*L, C)
        true_depths = true_depths.unsqueeze(-1)  # (*L, C, 1)
        pred_depths = actions * self._cos_thetas.unsqueeze(dim=-1)  # (*L, C, R)

        if probs is None:
            assert pred_depths.shape[-1] == 1, "probs can only be None if only one action provided per camera ray"
            probs = torch.ones_like(pred_depths)  # (*L, C, 1)

        # masking: only the rays inside the mask contribute towards the mean huber loss
        mask = (self._min_range < gt_action) & (gt_action < self._max_range)  # (*L, C)
        mask_sum = mask.sum(dim=-1).clamp(min=1)  # (*L)

        ################################################################################################################
        # RMSE Linear
        ################################################################################################################

        sq_diff = torch.square(pred_depths - true_depths)  # (*L, C, R)

        rmse_linear = (sq_diff * probs).sum(dim=-1)  # (*L, C)
        rmse_linear = (rmse_linear * mask).sum(dim=-1) / mask_sum  # (L*)
        rmse_linear = torch.sqrt(rmse_linear)  # (*L)

        ################################################################################################################
        # RMSE Log
        ################################################################################################################

        log_true_depths = torch.log(true_depths)  # (*L, C, 1)
        log_pred_depths = torch.log(pred_depths)  # (*L, C, R)
        diff_log = log_pred_depths - log_true_depths  # (*L, C, R)
        sq_diff_log = torch.square(diff_log)  # (*L, C, R)

        rmse_log = (sq_diff_log * probs).sum(dim=-1)  # (*L, C)
        rmse_log = (rmse_log * mask).sum(dim=-1) / mask_sum  # (*L)
        rmse_log = torch.sqrt(rmse_log)  # (*L)

        ################################################################################################################
        # RMSE Log Scale Invariant
        ################################################################################################################

        alpha = -diff_log  # (*L, C, R)
        alpha = (alpha * probs).sum(dim=-1)  # (*L, C)
        alpha = (alpha * mask).sum(dim=-1) / mask_sum  # (*L)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # (*L, 1, 1)

        rmse_lsi = torch.square(diff_log + alpha)  # (*L, C, R)
        rmse_lsi = (rmse_lsi * probs).sum(dim=-1)  # (*L, C)
        rmse_lsi = (rmse_lsi * mask).sum(dim=-1) / (2 * mask_sum)  # (*L)
        rmse_lsi = torch.sqrt(rmse_lsi)  # (*L)

        ################################################################################################################
        # Abs Rel Diff
        ################################################################################################################

        abs_diff = torch.abs(pred_depths - true_depths)  # (*L, C, R)
        abs_rel_diff = abs_diff / true_depths  # (*L, C, R)

        abs_rel_diff = (abs_rel_diff * probs).sum(dim=-1)  # (*L, C)
        abs_rel_diff = (abs_rel_diff * mask).sum(dim=-1) / mask_sum  # (*L)

        ################################################################################################################
        # Squared Rel Diff
        ################################################################################################################

        sq_rel_diff = sq_diff / true_depths  # (*L, C, R)

        sq_rel_diff = (sq_rel_diff * probs).sum(dim=-1)  # (*L, C)
        sq_rel_diff = (sq_rel_diff * mask).sum(dim=-1) / mask_sum  # (*L)

        ################################################################################################################
        # Threshold 1, 2, 3
        ################################################################################################################

        ratio1 = pred_depths / true_depths  # (*L, C, R)
        ratio2 = true_depths / pred_depths  # (*L, C, R)
        delta = torch.max(ratio1, ratio2)  # (*L, C, R)

        def compute_threshold_metric(thresh):
            threshold = delta < thresh  # (*L, C, R)
            threshold = (threshold * probs).sum(dim=-1)  # (*L, C)
            threshold = (threshold * mask).sum(dim=-1) / mask_sum  # (*L)
            return threshold

        threshold1 = compute_threshold_metric(self._thresh1)  # (*L)
        threshold2 = compute_threshold_metric(self._thresh2)  # (*L)
        threshold3 = compute_threshold_metric(self._thresh3)  # (*L)

        ret = dict(rmse_linear=rmse_linear,
                   rmse_log=rmse_log,
                   rmse_lsi=rmse_lsi,
                   abs_rel_diff=abs_rel_diff,
                   sq_rel_diff=sq_rel_diff,
                   threshold1=threshold1,
                   threshold2=threshold2,
                   threshold3=threshold3)

        return ret
    
    def reset(self):
        self._store.clear()

    def add(self,
            name: str,
            pi: Pi,
            gt_action: np.ndarray) -> Dict:
        """
        Args:
            name (str): name of the policy for which the metric is being computed
            pi (Pi, batch_shape=(1, C), event_shape=()): action probability distribution
            gt_action (np.ndarray, dtype=float32, shape=(C,): ground truth ranges (safety envelope).

        Returns:
            standard (dict): standard metrics.
        """
        gt_action = gt_action.reshape(1, -1)  # (1, C)
        if name not in self._store:
            self._store[name] = {metric_name: [] for metric_name in self.metric_names}

        # prepare inputs
        actions = pi.actions.to('cpu')  # (*L, C, R)
        probs = pi.probs.to('cpu')  # (*L, C, R)
        gt_action = torch.from_numpy(gt_action)  # (*L, C)

        standard = self._torch_standard(actions, probs, gt_action)

        # add metrics to store
        for metric_name in self.metric_names:
            metric_value = standard[metric_name].mean().item()
            self._store[name][metric_name].append(metric_value)

        return standard
    
    def metric(self,
               name: str) -> MetricStandard:
        """
        Args:
            name (str): name of policy

        Returns:
            MetricSER: metric of `name' policy.
        """
        if name not in self._store:
            raise Exception(f"Policy name {name} not found in Evaluator")

        kwargs = {}
        for metric_name in self.metric_names:
            kwargs[metric_name] = np.array(self._store[name][metric_name]).mean()

        metric = MetricStandard(**kwargs)
        return metric
