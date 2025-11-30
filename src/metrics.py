import json
import warnings
from typing import Optional, Tuple

import torch
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn.functional as F

from src.configs import TrainingConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_pandas(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, n_classes: int) -> float:
    """
    Fully vectorized computation of the average Dice over samples and classes (skip background=0).
    This removes the Python loop over samples by using boolean broadcasting.
    Note: memory usage ~ O(S * P * K) where S = n_samples, P = pixels per sample, K = NUM_CLASSES.
    """
    # transpose to get shape (n_samples, n_pixels)
    y_true = y_true_df.T.values  # shape (S, P)
    y_pred = y_pred_df.T.values  # shape (S, P)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape after transpose.")

    S, P = y_true.shape
    classes = np.arange(1, n_classes + 1)  # skip background 0, shape (K,)

    # boolean one-hot along classes: shape (S, P, K)
    gt_mask = (y_true[..., None] == classes)     # True where pixel belongs to class c in GT
    pred_mask = (y_pred[..., None] == classes)   # True where pixel belongs to class c in pred

    # intersection and sums per (sample, class)
    intersection = np.sum(pred_mask & gt_mask, axis=1).astype(np.float64)  # (S, K)
    sum_pred = np.sum(pred_mask, axis=1).astype(np.float64)               # (S, K)
    sum_gt   = np.sum(gt_mask, axis=1).astype(np.float64)                 # (S, K)

    denom = sum_pred + sum_gt  # (S, K)

    # dice per sample per class; where denom==0 => np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        dice_spc = np.where(denom == 0, np.nan, 2.0 * intersection / denom)  # (S, K)

    # average over samples (ignoring NaNs), then average over classes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cls_dices = np.nanmean(dice_spc, axis=0)  # (K,)

    return float(np.nanmean(cls_dices))

class SegmentationLoss:
    def __init__(self, train_cfg: TrainingConfig):
        self.train_cfg = train_cfg
        weight = get_class_weights().to(device) if train_cfg.use_labels_weight else None
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> dict[str, Tensor]:
        """
        Computes the weighted average of the cross entropy and dice loss of y pred.
        ### Returns:
        Dictionnary of loss_average, ce_loss and dice_loss
        """
        ce_loss = self.cross_entropy_loss(y_pred, y_true)
        base_d_loss = base_dice_loss(y_pred, y_true)
        invariant_d_loss = perm_invariant_dice_loss(y_pred, y_true, self.train_cfg.n_classes)
        loss = base_d_loss * self.train_cfg.dice_loss_weight \
            + ce_loss * self.train_cfg.cross_entropy_loss_weight \
            + invariant_d_loss * self.train_cfg.invariant_d_loss_weight
        return {
            "average_loss": loss,
            "cross_entropy_loss": ce_loss,
            "dice_loss": base_d_loss,
            "permuatation_invariant_dice_loss": invariant_d_loss,
        }

def base_dice_loss(pred: Tensor, target: Tensor, smooth: float=1e-7) -> Tensor:
    pred = torch.softmax(pred, dim=1)
    target_one_hot = (
        torch.nn.functional.one_hot(
            target,
            num_classes=pred.shape[1],
        )
        .permute(0, 3, 1, 2)
    )
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def get_class_weights() -> torch.Tensor:
    file_name = "./dataset/raw/annotated_labels.json"
    with open(file_name, 'r') as file :
        data = json.load(file)

    flattened_data = []
    for i in data :
        flattened_data += i

    labels, labels_count = np.unique(flattened_data, return_counts=True)
    labels_weights = labels_count / np.sum(labels_count)

    # adding the background
    labels_weights = np.insert(labels_weights, 0, 0.0001)
    class_weights = torch.from_numpy(labels_weights).type(torch.float32)
    return class_weights

# ---------- Sinkhorn (batched iterative scaling) ----------
def sinkhorn(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 50,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Batched Sinkhorn that returns a soft transport matrix P for each batch element.

    Args:
        cost: (B, N, M) cost matrix
        epsilon: entropy regularization
        n_iters: number of Sinkhorn iterations
        eps: tiny constant to avoid div-by-zero
    Returns:
        P: (B, N, M) soft transport matrices (rows/cols approximately normalized)
    """
    B, N, M = cost.shape
    # Kernel
    K = torch.exp(-cost / (epsilon + 1e-12))  # (B, N, M)

    # Uniform marginals
    a = torch.full((B, N), 1.0 / N, device=cost.device, dtype=K.dtype)
    b = torch.full((B, M), 1.0 / M, device=cost.device, dtype=K.dtype)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iters):
        K_v = torch.einsum('bnm,bm->bn', K, v)  # (B, N)
        u = a / (K_v + eps)
        K_t_u = torch.einsum('bnm,bn->bm', K, u)  # (B, M)
        v = b / (K_t_u + eps)

    P = (u.unsqueeze(2) * K) * v.unsqueeze(1)  # (B, N, M)
    return P

# ---------- Permutation-invariant Dice: scores + scalar loss ----------
def perm_invariant_dice_scores(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    smooth: float = 1e-7,
    detach_cost: bool = True,
) -> torch.Tensor:
    """
    Compute permutation-invariant Dice scores using batched Sinkhorn matching.
    Returns a (B, C_gt) tensor of Dice scores; GT-empty classes have value NaN.

    Args:
        logits: (B, C_pred, H, W) raw model outputs (float)
        target: (B, H, W) integer label maps (0..C_gt-1), background=0
        num_classes: if provided, treat GT channels count as this (including background).
                     otherwise inferred from target.max()+1
        epsilon: Sinkhorn regularization
        sinkhorn_iters: Sinkhorn iteration count
        smooth: small smoothing for Dice formula
        detach_cost: if True, detach cost before Sinkhorn (matching treated as non-differentiable).
                     if False, matching is differentiable (grad flows through the Sinkhorn ops).
    Returns:
        dice_scores: (B, C_gt) tensor with per-image-per-gt-class Dice; empty GT classes -> NaN
    """
    device = logits.device
    B, C_pred, H, W = logits.shape

    # infer C_gt
    if num_classes is None:
        C_gt = int(target.max().item()) + 1
    else:
        C_gt = int(num_classes)

    # soft predictions
    pred_probs = F.softmax(logits, dim=1)  # (B, C_pred, H, W)

    # one-hot GT (B, C_gt, H, W)
    gt_onehot = F.one_hot(target.long(), num_classes=C_gt).permute(0, 3, 1, 2).to(
        dtype=pred_probs.dtype, device=device
    )

    # pad preds and gt to K x K
    K = max(C_pred, C_gt)
    if K == C_pred and K == C_gt:
        pred_pad = pred_probs
        gt_pad = gt_onehot
    else:
        if C_pred < K:
            pad = torch.zeros((B, K - C_pred, H, W), dtype=pred_probs.dtype, device=device)
            pred_pad = torch.cat([pred_probs, pad], dim=1)
        else:
            pred_pad = pred_probs

        if C_gt < K:
            pad = torch.zeros((B, K - C_gt, H, W), dtype=gt_onehot.dtype, device=device)
            gt_pad = torch.cat([gt_onehot, pad], dim=1)
        else:
            gt_pad = gt_onehot

    # pairwise intersections S_{i,j} = sum_pixels pred_i * gt_j  -> (B, K, K)
    S = torch.einsum('bihw,bjhw->bij', pred_pad, gt_pad)  # float

    pred_area = pred_pad.sum(dim=(2, 3))  # (B, K)
    gt_area = gt_pad.sum(dim=(2, 3))      # (B, K)
    union = pred_area.unsqueeze(2) + gt_area.unsqueeze(1) - S  # (B, K, K)
    IoU = S / (union + 1e-12)

    cost = -IoU  # maximize IoU -> minimize cost

    if detach_cost:
        cost_for_sinkhorn = cost.detach()
    else:
        cost_for_sinkhorn = cost

    P = sinkhorn(cost_for_sinkhorn, epsilon=epsilon, n_iters=sinkhorn_iters)  # (B, K, K)

    # use only first C_pred rows and first C_gt cols
    P_sub = P[:, :C_pred, :C_gt]  # (B, C_pred, C_gt)

    # matched_pred for each gt channel j: sum_i P_{i,j} * pred_i
    matched_pred = torch.einsum('bij,bihw->bjhw', P_sub, pred_probs)  # (B, C_gt, H, W)

    # Dice per (B, C_gt)
    inter = torch.einsum('bchw,bchw->bc', matched_pred, gt_onehot)  # (B, C_gt)
    sum_pred = matched_pred.sum(dim=(2, 3))  # (B, C_gt)
    sum_gt = gt_onehot.sum(dim=(2, 3))       # (B, C_gt)
    denom = sum_pred + sum_gt

    dice = (2.0 * inter + smooth) / (denom + smooth)  # (B, C_gt)

    # Set entries for empty GT classes to NaN -> ignored in averages
    empty_mask = sum_gt == 0  # True where GT class has zero pixels
    if empty_mask.any():
        dice = dice.masked_fill(empty_mask, float('nan'))

    # Always include background (channel 0). We already included it by construction.

    return dice  # (B, C_gt) with NaNs for empty GT classes

def perm_invariant_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    smooth: float = 1e-7,
    detach_cost: bool = True,
) -> torch.Tensor:
    """
    Returns a scalar loss = 1 - mean(Dice) where mean is computed:
      - per-image mean across GT classes (ignoring NaNs for empty classes),
      - then mean across the batch (ignoring images with all-NaN).
    This mirrors "average per-image then average across images".
    """
    dice = perm_invariant_dice_scores(
        logits=logits,
        target=target,
        num_classes=num_classes,
        epsilon=epsilon,
        sinkhorn_iters=sinkhorn_iters,
        smooth=smooth,
        detach_cost=detach_cost,
    )  # (B, C_gt)

    # per-image mean (ignore NaNs)
    valid_mask = ~torch.isnan(dice)  # (B, C_gt)
    valid_count = valid_mask.sum(dim=1)  # (B,)
    # sum valid entries per image
    dice_sum = torch.where(valid_mask, dice, torch.tensor(0.0, device=dice.device, dtype=dice.dtype)).sum(dim=1)
    per_image_mean = torch.where(valid_count > 0, dice_sum / valid_count, torch.tensor(float('nan'), device=dice.device))

    # mean across images, ignoring NaNs
    valid_images = ~torch.isnan(per_image_mean)
    if valid_images.sum() == 0:
        mean_dice = torch.tensor(0.0, device=dice.device, dtype=dice.dtype)
    else:
        mean_dice = per_image_mean[valid_images].mean()

    return 1.0 - mean_dice
