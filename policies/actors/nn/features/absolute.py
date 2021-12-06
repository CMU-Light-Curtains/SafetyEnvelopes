from dataclasses import dataclass
import numpy as np

from policies.actors.nn.features.residual import _components_from_size, CurtainInfo, NNResidualFeaturizer


@dataclass
class NNAbsoluteFeatures:
    c1: CurtainInfo
    c2: CurtainInfo

    def to_numpy(self) -> np.ndarray:
        return np.hstack([
            self.c1.to_numpy(),  # (B, 5C)
            self.c2.to_numpy(),  # (B, 5C)
        ]).astype(np.float32)

    @staticmethod
    def from_numpy(np_array: np.ndarray) -> 'NNAbsoluteFeatures':
        assert np_array.ndim == 2
        assert np_array.shape[1] % 10 == 0, f"NNResidualFeatures.from_numpy(): shape is {np_array.shape}"
        C = np_array.shape[1] // 10

        components = _components_from_size(np_array, sizes=[5*C, 5*C])

        c1 = CurtainInfo.from_numpy(components[0])
        c2 = CurtainInfo.from_numpy(components[1])

        return NNAbsoluteFeatures(c1, c2)

    def feat_dim(C: int):
        return 10 * C


class NNAbsoluteFeaturizer(NNResidualFeaturizer):
    @property
    def feat_dim(self):
        return NNAbsoluteFeatures.feat_dim(self.C)

    @staticmethod
    def cinfos2feats(c1: CurtainInfo,
                     c2: CurtainInfo) -> NNAbsoluteFeatures:
        """
        Computing the features for predicting actions at time t.
        NNResidualFeatures are extracted from curtain info at times t-1 and t-2.

        Args:
            cp1 (CurtainPairInfo): info about the curtain placed at time t-2.
            cp2 (CurtainPairInfo): info about the curtain placed at time t-1.
        """
        feats = NNAbsoluteFeatures(c1, c2)
        return feats

    def cinfos2numpy(self,
                     c1: CurtainInfo,
                     c2: CurtainInfo) -> np.ndarray:
        feats = self.cinfos2feats(c1, c2)
        return feats.to_numpy()
