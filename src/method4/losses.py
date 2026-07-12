"""
losses.py — Combined Focal + Dice loss. Identical to Method 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred    = pred.clamp(1e-6, 1.0 - 1e-6)
        bce     = F.binary_cross_entropy(pred, target, reduction="none")
        p_t     = pred * target + (1.0 - pred) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f   = pred.view(-1)
        target_f = target.view(-1)
        intersection = (pred_f * target_f).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_f.sum() + target_f.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.focal   = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice    = DiceLoss()
        self.focal_w = focal_weight
        self.dice_w  = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_w * self.focal(pred, target) + \
               self.dice_w  * self.dice(pred, target)
