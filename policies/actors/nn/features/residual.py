from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from devices.light_curtain import LCReturn
import utils


def _components_from_size(np_array: np.ndarray,
                          sizes: List[int]):
    sizes = np.array(sizes, dtype=np.int)
    ends = sizes.cumsum()
    starts = ends - sizes
    components = [np_array[:, start:end] for start, end in zip(starts, ends)]
    return components


@dataclass
class CurtainInfo:
    i: Optional[np.ndarray]  # intensity for each camera ray, shape = (B, C)
    x: Optional[np.ndarray]  # x coord for each camera ray,   shape = (B, C)
    z: Optional[np.ndarray]  # z coord for each camera ray,   shape = (B, C)
    r: Optional[np.ndarray]  # range for each camera ray,     shape = (B, C)
    t: Optional[np.ndarray]  # theta for each camera ray,     shape = (B, C)

    def __getitem__(self, s):
        return CurtainInfo(self.i[s],
                           self.x[s],
                           self.z[s],
                           self.r[s],
                           self.t[s])

    def __sub__(self, other):
        return CurtainInfo(None,  # difference of intensities doesn't make sense, shouldn't be used
                           self.x - other.x,
                           self.z - other.z,
                           self.r - other.r,
                           self.t - other.t)

    def to_numpy(self) -> np.ndarray:
        return np.hstack([
            self.i,
            self.x,
            self.z,
            self.r,
            self.t
        ]).astype(np.float32)  # (B, 5C)

    @staticmethod
    def from_numpy(np_array: np.ndarray) -> 'CurtainInfo':
        assert np_array.ndim == 2
        assert (np_array.shape[1]) % 5 == 0, f"CurtainInfo.from_numpy(): shape is {np_array.shape}"
        C = np_array.shape[1] // 5

        components = _components_from_size(np_array, sizes=[C, C, C, C, C])
        return CurtainInfo(*components)


@dataclass
class NNResidualFeatures:
    c1: CurtainInfo
    c2: CurtainInfo
    rb: np.ndarray  # ranges of base policy action

    def to_numpy(self) -> np.ndarray:
        return np.hstack([
            self.c1.to_numpy(),  # (B, 5C)
            self.c2.to_numpy(),  # (B, 5C)
            self.rb              # (B, C)
        ]).astype(np.float32)

    @staticmethod
    def from_numpy(np_array: np.ndarray) -> 'NNResidualFeatures':
        assert np_array.ndim == 2
        assert np_array.shape[1] % 11 == 0, f"NNResidualFeatures.from_numpy(): shape is {np_array.shape}"
        C = np_array.shape[1] // 11

        components = _components_from_size(np_array, sizes=[5*C, 5*C, C])

        c1 = CurtainInfo.from_numpy(components[0])
        c2 = CurtainInfo.from_numpy(components[1])
        rb = components[2]

        return NNResidualFeatures(c1, c2, rb)

    def feat_dim(C: int):
        return 11 * C


class NNResidualFeaturizer:
    def __init__(self,
                 thetas: np.ndarray):
        """
        Args:
            thetas (np.ndarray, dtype=np.float32, shape=(C,)): thetas of the camera rays (in degrees).
        """
        self._thetas = thetas

        # parameters
        self._LC_INTENSITY_THRESH = 200

    @property
    def C(self):
        return len(self._thetas)

    @property
    def feat_dim(self):
        return NNResidualFeatures.feat_dim(self.C)

    def obs_to_curtain_info(self,
                            lc_return: LCReturn) -> CurtainInfo:
        i = lc_return.bev_hits(ithresh=self._LC_INTENSITY_THRESH)  # (C,)
        r = lc_return.lc_ranges  # (C,)
        t = self._thetas
        design_pts = utils.design_pts_from_ranges(r, t)  # (C, 2)
        x, z = design_pts[:, 0], design_pts[:, 1]  # (C,) and (C,)

        i, x, z, r, t = [np.atleast_2d(e).astype(np.float32) for e in [i, x, z, r, t]]  # (1, C) each

        return CurtainInfo(i, x, z, r, t)

    @staticmethod
    def cinfos2feats(c1: CurtainInfo,
                     c2: CurtainInfo,
                     base_act: np.ndarray) -> NNResidualFeatures:
        """
        Computing the features for predicting actions at time t.
        NNResidualFeatures are extracted from curtain info at times t-1 and t-2.

        Args:
            c1 (CurtainInfo): info about the curtain placed at time t-2.
            c2 (CurtainInfo): info about the curtain placed at time t-1.
            base_act (np.ndarray, dtype=np.float32, shape=(C,)): action of the base policy at time t-1 using input2.
        """
        base_act = np.atleast_2d(base_act).astype(np.float32)  # (1, C)
        feats = NNResidualFeatures(c1, c2, base_act)
        return feats

    def cinfos2numpy(self,
                     c1: CurtainInfo,
                     c2: CurtainInfo,
                     base_act: np.ndarray) -> np.ndarray:
        feats = self.cinfos2feats(c1, c2, base_act)
        return feats.to_numpy()
