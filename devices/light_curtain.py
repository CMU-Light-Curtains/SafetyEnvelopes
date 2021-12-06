from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
import lc_sim
from lc_planner.config import LCConfig

from data.synthia import append_xyz_to_depth_map, Frame
import utils

ingr = utils.Ingredient('lc_device')
ingr.add_config("config/lc_device.yaml")

########################################################################################################################
# region LCReturn class
########################################################################################################################


@dataclass
class LCReturn:
    ####################################################################################################################
    # Public members
    ####################################################################################################################
    """
    Ranges per camera ray of the light curtain that was actually placed by the light curtain device.
        - (np.ndarray, dtype=np.float32, shape=(C,))
    """
    lc_ranges: np.ndarray  # (np.ndarray, dtype=float32, shape=(C,))

    """
    Light curtain image.
        - (np.ndarray, dtype=float32, shape=(H, C, 4)).
        - Axis 2 corresponds to (x, y, z, i):
                - x : x in cam frame.
                - y : y in cam frame.
                - z : z in cam frame.
                - i : intensity of LC cloud, lying in [0, 255].
    """
    lc_image: np.ndarray  # (np.ndarray, dtype=float32, shape=(H, C, 4))

    """
    Light curtain point cloud.
        - (np.ndarray, dtype=float32, shape=(N, 4)).
        - Axis 2 corresponds to (x, y, z, i):
                - x : x in cam frame.
                - y : y in cam frame.
                - z : z in cam frame.
                - i : intensity of LC cloud, lying in [0, 1].
    """
    lc_cloud: np.ndarray  # (np.ndarray, dtype=float32, shape=(N, 4))

    """
    Light curtain thickness for each pixel.
        - (np.ndarray, dtype=float32, shape=(H, C)).
    """
    lc_thickness: np.ndarray  # (np.ndarray, dtype=float32, shape=(H, C))

    """
    Ground truth frame that generated this light curtain return.
    """
    frame: Frame

    ####################################################################################################################
    # Private members
    ####################################################################################################################

    # Light curtain return for height < vrange_min or height > vrange_max will be ignored.
    # Height in velo frame is +Z and cam frame is -Y.
    VERTICAL_RANGE: Tuple[float, float]

    ####################################################################################################################
    # Cached results to avoid duplicate computation
    ####################################################################################################################

    # cached result of the function call self.bev_intensities(avg=False)
    _cache_bev_max_intensities_: Optional[np.ndarray] = None  # (C,)

    ####################################################################################################################
    # Methods
    ####################################################################################################################

    def bev_intensities(self, avg=False) -> np.ndarray:
        """
        Returns a 1-D array of (maximum) columnwise intensities along the light curtain's placement
        
        *CAUTION*: the returned intensities lie in the range [0, 255].

        Returns:
            intensities (np.ndarray, dtype=np.bool, shape=(C,)): max/avg intensity along each column, lying in [0, 255].
            avg (bool): if True, takes the avg intensity across the column, else takes the max intensity.
        """
        # if cached result exists, simply return it
        if (self._cache_bev_max_intensities_ is not None) and (not avg):
            return self._cache_bev_max_intensities_.copy()

        intensities = self.lc_image[:, :, 3].copy()  # (H, C)

        # mask out NaN values
        intensities[np.isnan(self.lc_image).any(axis=2)] = 0  # (H, C)

        # mask out pixels that are outside VERTICAL_RANGE (note that cam_y points downwards)
        cam_y = self.lc_image[:, :, 1]  # (H, C)
        vrange_min, vrange_max = self.VERTICAL_RANGE
        outside_vrange_mask = (-cam_y < vrange_min) | (-cam_y > vrange_max)  # (H, C)
        intensities[outside_vrange_mask] = 0

        # final intensity is the maximum across camera columns
        if avg:
            intensities = intensities.mean(axis=0)  # (C,)
        else:
            intensities = intensities.max(axis=0)  # (C,)
            self._cache_bev_max_intensities_ = intensities  # store max intensities in cache before returning

        return intensities

    def bev_hits(self,
                 ithresh: float) -> np.ndarray:
        """
        Args:
            ithresh: (float) intensity threshold in [0, 255] above which returns will be considered as hits.

        Returns:
            hits (np.ndarray, dtype=np.bool, shape=(C,)): whether there is a hit or not for every camera column.
        """
        intensities = self.bev_intensities()  # (C,)
        hits = (intensities >= ithresh)  # (C,)

        return hits


# endregion
########################################################################################################################

class LightCurtain:
    @ingr.capture
    def __init__(self, min_range, max_range, debug=False):
        self.min_range = min_range
        self.max_range = max_range
        self._debug = debug

        self.lc_config = LCConfig(config_file='config/lc/synthia.yaml',
                                  default_config_file='LC-PLANNER/apis/py/lc_planner/defaults.yaml')
        self.lc_device = lc_sim.LCDevice(CAMERA_PARAMS=self.lc_config.CAMERA_PARAMS,
                                         LASER_PARAMS=self.lc_config.LASER_PARAMS)

        self._MAX_DEPTH = 80.0  # only consider points within this depth

    @property
    def thetas(self):
        return self.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]

    @staticmethod
    def _validate_curtain_using_lc_image(lc_image: np.ndarray):
        """Extract the actual valid curtain that was physically placed via the lc_image

        Args:
            lc_image (np.ndarray, dtype=float32, shape=(H, C, 4))):  output of LC device.
                        - Channels denote (x, y, z, intensity).
                        - Pixels that aren't a part of LC return will have NaNs in one of
                        the 4 channels.
                        - Intensity ranges from 0. to 255.
        Returns:
            curtain (np.ndarray, dtype=np.float32, shape=(C',)): range per camera ray that corresponds to a
                      valid curtain.
            mask (np.ndarray, dtype=np.bool, shape=(C,)): subset of the camera rays for which a return was observed.
        """
        mask_2d = np.logical_not(np.isnan(lc_image).any(axis=2))  # (H, C) (none of x, y, z, i) should be nan
        mask_1d = mask_2d.any(axis=0)  # (C,) at least one pixel along the camera column should be not-nan

        xz = lc_image[:, mask_1d, :][:, :, [0, 2]]  # (H, C', 2) each column has at least 1 non-nan pixel

        # take mean of xz across all non-nan pixels
        mask_2d = mask_2d[:, mask_1d]  # (H, C') mask of non-nan pixels
        xz[np.logical_not(mask_2d)] = 0.  # xz: (H, C', 2) replace all nan's with 0s before summing them
        xz_sum = xz.sum(axis=0)  # (C', 2)
        xz_counts = mask_2d.sum(axis=0)  # (C')
        xz_mean = xz_sum / xz_counts.reshape(-1, 1)  # (C', 2)

        curtain = np.linalg.norm(xz_mean, axis=1)  # (C',)

        return curtain, mask_1d

    def get_lc_return_from_state(self,
                                 frame: Frame,
                                 ranges: np.ndarray,
                                 vertical_range: Tuple[float, float]) -> LCReturn:
        """
        Args:
            frame (Frame): frame representing the state.
            ranges (np.ndarray, shape=(C,), dtype=np.float32): range per camera ray.
            vertical_range (Tuple[float, float]): minimum and maximum vertical ranges (+ve is upwards).

        Returns:
            lc_return (LCReturn): Light curtain return.
        """
        ranges = ranges.copy()
        design_pts = utils.design_pts_from_ranges(ranges, self.thetas)  # (C, 2)

        # The next line get lc_image and lc_thickness.
        lc_image, lc_thickness = \
            self.lc_device.get_return(frame.depth, design_pts, get_thickness=True)  # (H, C, 4) and (H, C)

        lc_cloud = lc_image.reshape(-1, 4)  # (N, 4)
        # Remove points which are NaNs.
        non_nan_mask = np.all(np.isfinite(lc_cloud), axis=1)
        lc_cloud = lc_cloud[non_nan_mask]  # (N, 4)
        # Rescale LC return to [0, 1].
        lc_cloud[:, 3] /= 255.  # (N, 4)

        if self._debug:
            self._debug_thickness(frame, lc_image, lc_thickness)

        # Update curtain
        curtain, mask = self._validate_curtain_using_lc_image(lc_image)
        ranges[mask] = curtain
        lc_ranges = ranges

        lc_return = LCReturn(lc_ranges, lc_image, lc_cloud, lc_thickness, frame, vertical_range)
        return lc_return

    @staticmethod
    def _debug_thickness(frame, lc_image, lc_thickness):
        # the next three lines assert that lc_thickness is non-nan wherever lc_image is non-nan.
        non_nan_mask_image = np.isfinite(lc_image).all(axis=2)  # (H, C)
        non_nan_mask_thick = np.isfinite(lc_thickness)  # (H, C)
        assert non_nan_mask_thick[non_nan_mask_image].all()

        # the next few lines verify the thickness model (assumed to be the linear falloff model)
        intensity = lc_image[:, :, 3] / 255.  # (H, C)
        lc_xyz = lc_image[:, :, :3]  # (H, C, 3)
        lc_range = np.linalg.norm(lc_xyz, axis=2)  # (H, C)

        # append x, y, z to depth
        P2 = frame.calib["P2"][:3, :3]  # (3, 3)
        gt_xyz = append_xyz_to_depth_map(frame.depth[:, :, None], P2)  # (H, C, 3); axis 2 is (x, y, z) in cam frame
        gt_range = np.linalg.norm(gt_xyz, axis=2)  # (H, C)

        error_from_intensity = 0.5 * lc_thickness * (1.0 - intensity)  # (H, C)
        error_from_gt = np.abs(gt_range - lc_range).clip(max=0.5 * lc_thickness)  # (H, C)

        # the next few lines plot the error images obtained from intensity and ground truth
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(error_from_gt)
        axes[0].set_title('Error from GT')
        axes[1].imshow(error_from_intensity)
        axes[1].set_title('Error from intensity')
        for ax in axes:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.show()

        assert np.allclose(error_from_gt[non_nan_mask_image], error_from_intensity[non_nan_mask_image],
                           rtol=0.0, atol=0.02)  # tolerance is 2cm (absolute)
