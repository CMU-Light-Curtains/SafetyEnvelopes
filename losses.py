import numpy as np
import torch
import torch.distributions as td


def loss_l2(pi: td.Distribution,
            dem: np.ndarray,
            max_range: float,
            max_safe_ratio: float) -> torch.Tensor:
    """
    Args:
        pi (td.Distribution, dtype=torch.float32, BS=(B,), ES=(C,)): distribution predicted by policy.
        dem (np.ndarray, dtype=np.float32, shape=(B, C)): demonstration.
        max_range (float): will not include loss for rays with this range or beyond.
        max_safe_ratio (float): maximum ratio after which the loss starts to increase sharply.

    Returns:
        loss (torch.Tensor, dtype=np.float32, shape=([])): L2 loss.
    """
    dem = torch.from_numpy(dem)  # (B, C)
    if torch.cuda.is_available():
        dem = dem.cuda()
    mask = dem < max_range  # (B, C)

    mu = pi.mean  # (B, C)
    loss = (mask * torch.square(mu - dem)).mean()
    return loss


def loss_smooth_clipped_ratio(pi: td.Distribution,
                              dem: np.ndarray,
                              max_range: float,
                              max_safe_ratio: float,
                              k: float = 25) -> torch.Tensor:
    """
    Args:
        pi (td.Distribution, dtype=torch.float32, BS=(B,), ES=(C,)): distribution predicted by policy.
        dem (np.ndarray, dtype=np.float32, shape=(B, C)): demonstration.
        max_range (float): will not include loss for rays with this range or beyond.
        max_safe_ratio (float): maximum ratio after which the loss starts to increase sharply.
        k (float): loss parameter that controls the smoothness of the loss for r > 1.

    Returns:
        loss (torch.Tensor, dtype=np.float32, shape=([])): smooth clipped ratio loss.
            - smooth_clipped_ratio_loss = min(r, exp(-k(r-1)) where r = predicted_range / gt_range.
            - ideally r >= 0. But if it is < 0, smooth_clipped_ratio_loss < 0. Hence we clamp it to be >= 0.
    """
    dem = torch.from_numpy(dem)  # (B, C)
    if torch.cuda.is_available():
        dem = dem.cuda()
    mask = dem < max_range  # (B, C)

    mu = pi.mean  # (B, C)

    ratio = mu / dem  # (B, C)
    lin_piece = ratio  # (B, C) increases linearly
    exp_piece = max_safe_ratio * torch.exp(-k * (ratio - max_safe_ratio))  # (B, C) decreases exponentially
    smooth_clipped_ratio = torch.min(lin_piece, exp_piece).clamp(min=0)  # (B, C)
    loss = -(mask * smooth_clipped_ratio).mean()
    return loss


def loss_squared_error_relative(pi: td.Distribution,
                                dem: np.ndarray,
                                max_range: float) -> torch.Tensor:
    """
    Args:
        pi (td.Distribution, dtype=torch.float32, BS=(B,), ES=(C,)): distribution predicted by policy.
        dem (np.ndarray, dtype=np.float32, shape=(B, C)): demonstration.
        max_range (float): will not include loss for rays with this range or beyond.

    Returns:
        loss (torch.Tensor, dtype=np.float32, shape=([])): squared error (relative) loss.
            - smooth_clipped_ratio_loss = min(r, exp(-k(r-1)) where r = predicted_range / gt_range.
            - ideally r >= 0. But if it is < 0, smooth_clipped_ratio_loss < 0. Hence we clamp it to be >= 0.
    """
    dem = torch.from_numpy(dem)  # (B, C)
    if torch.cuda.is_available():
        dem = dem.cuda()
    mask = dem < max_range  # (B, C)

    mu = pi.mean  # (B, C)

    ratio = mu / (dem + 1e-5)  # (B, C)
    ser = torch.square(ratio - 1)  # (B, C)
    ser = (ser * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B,) ser is 0 (perfect) where the mask is empty
    loss = ser.mean()

    return loss
