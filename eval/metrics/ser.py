from dataclasses import dataclass
import numpy as np
from typing import Optional, List

import utils
from data.synthia import Frame


@dataclass
class MetricSER:
    ser: float

    def __str__(self):
        return f"| squared-error-relative: {self.ser:02.4f} |"


class EvaluatorSER:
    """Performs ground truth evaluation of the squared error (relative)"""
    def __init__(self,
                 max_range: float,
                 debug: bool = False):
        """
        Args:
            max_range (float): rays which have range greater than this will not be included for evaluation.
            debug (bool): whether to debug or not
        """
        self._max_range = max_range
        self._debug = debug

        self._store = None

    def _create_store(self, n_policies):
        assert self._store is None, "Cannot create store without calling EvaluatorSER.reset()"
        self._store = [[] for i in range(n_policies)]

    def _compute_ser(self,
                     action: np.ndarray,
                     gt_action: np.ndarray):
        """
        Args:
            action (np.ndarray, dtype=np.float32, shape=(C,)): action.
            gt_action (np.ndarray, dtype=np.float32, shape=(C,)): ground truth action.

        Returns:
            ser (Optional[float]): the squared error (relative).
        """
        assert action.ndim == 1 and gt_action.ndim == 1
        assert np.all(gt_action > 0)
        ratio = action / (gt_action + 1e-5)  # (C,)
        ser = np.square(ratio - 1.0)  # (C,)
        # only the rays inside the mask contribute towards the mean ser
        mask = gt_action < self._max_range  # (C,)
        ser = ser.dot(mask) / mask.sum().clip(min=1)  # ser is 0 (perfect) where the mask is empty
        return ser

    ####################################################################################################################
    # region API
    ####################################################################################################################

    def reset(self):
        self._store = None

    def add(self,
            actions: List[np.ndarray],
            gt_action: Optional[np.ndarray] = None,
            frame: Optional[Frame] = None):
        """
        Args:
            actions (List[np.ndarray, dtype=float32, shape=(C,)]): an action (ranges) for each policy to evaluate.
            gt_action (Optional[np.ndarray, dtype=float32, shape=(C,)]: ground truth ranges (safety envelope).
            frame (Optional[Frame]): frame from GT state device.
        """
        if self._store is None:
            self._create_store(n_policies=len(actions))

        assert len(actions) == len(self._store), \
            f"Number of actions ({len(actions)}) does not match expected number ({len(self._store)}). " + \
            f"Use the same number of actions consistently."

        # get gt_action
        if gt_action is None:
            assert frame is not None, "Should provide either the ground truth action or the Frame"
            gt_action = frame.annos["se_ranges"]  # (C,)

        # accumulate hists
        for i, action in enumerate(actions):
            ser = self._compute_ser(action, gt_action)
            self._store[i].append(ser)

    def metrics(self,
                reset: bool = False) -> List[MetricSER]:
        """
        Args:
            reset (bool): whether to reset at the end or not. Equivalent to calling EvaluatorGT.reset() at the end.

        Returns:
            List[MetricSER]: metrics for each policy.
        """

        ret = []
        for list_ser in self._store:
            mean_ser = np.array(list_ser).mean()
            metrics = MetricSER(ser=mean_ser)
            ret.append(metrics)

        if reset:
            self.reset()

        return ret

    # region
    ####################################################################################################################
