import json
import warnings
from typing import Optional, Tuple

import torch
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn.functional as F

from src.configs import (
    TrainingConfig,
    N_CLASSES,
    EPSILON,
    DEVICE,
)
from scipy.optimize import linear_sum_assignment


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
        base_d_loss = torch_dice_loss(y_pred, y_true)
        loss = base_d_loss * self.train_cfg.dice_loss_weight \
            + ce_loss * self.train_cfg.cross_entropy_loss_weight
        return loss, {
            "average_loss": loss,
            "cross_entropy_loss": ce_loss,
            "dice_loss": base_d_loss,
        }

def torch_dice_loss(pred: Tensor, target: Tensor, smooth: float=1e-7) -> Tensor:
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

def one_hot_and_permute(x: Tensor) -> Tensor:
    return (
        torch.nn.functional.one_hot(
            x,
            num_classes=N_CLASSES,
        )
        .permute(0, 3, 1, 2)
    )

# @torch.no_grad # To make sure it's not used with a loss
# def torch_dice_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
#     y_pred_one_hot = one_hot_and_permute(y_pred.argmax(dim=1))
#     y_true_one_hot = one_hot_and_permute(y_true)
#     intersection = (y_pred_one_hot * y_true_one_hot).sum(dim=(2, 3))
#     union = y_pred_one_hot.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3))
#     # Use nan as some preds and/or masks channels could be all zeros
#     dice_scores = 2. * intersection / union
#     channel_wise_dice = torch.nanmean(dice_scores, dim=0)
#     dice_score = torch.nanmean(channel_wise_dice, dim=0)
#     return dice_score

@torch.no_grad
def torch_dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    y_pred: (B, C, H, W)  predicted probabilities or logits
    y_true: (B, H, W)     integer labels in [0, C-1]
    
    Computes:
      - Dice per sample per class (skip background=0)
      - ignore empty GT classes (nan)
      - average over samples (nanmean), then over classes (nanmean)
      - returns scalar tensor
    """
    B, C, H, W = y_pred.shape
    device = y_pred.device

    # Convert pred to discrete labels using argmax over channels
    # Equivalent to the pandas version which expects integer maps
    y_pred_labels = torch.argmax(y_pred, dim=1)    # (B, H, W)

    # Classes 1..C-1
    classes = torch.arange(1, C, device=device)    # (K,)
    K = classes.numel()

    # Build boolean masks: (B, H, W, K)
    # same as numpy broadcasting: (S, P, K)
    gt_mask = (y_true.unsqueeze(-1) == classes)           # bool(B,H,W,K)
    pred_mask = (y_pred_labels.unsqueeze(-1) == classes)  # bool(B,H,W,K)

    # Flatten spatial dims
    gt_mask = gt_mask.view(B, -1, K)       # (B, P, K)
    pred_mask = pred_mask.view(B, -1, K)   # (B, P, K)

    # Intersection and sums
    intersection = (gt_mask & pred_mask).sum(dim=1).float()  # (B, K)
    sum_pred = pred_mask.sum(dim=1).float()                  # (B, K)
    sum_gt   = gt_mask.sum(dim=1).float()                    # (B, K)

    denom = sum_pred + sum_gt                                # (B, K)

    # Dice per batch per class
    dice_spc = torch.where(denom == 0,
                           torch.nan,
                           2.0 * intersection / denom)       # (B, K)

    # nanmean over batch then over classes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cls_mean = torch.nanmean(dice_spc, dim=0)            # (K,)
        final_mean = torch.nanmean(cls_mean)                 # scalar

    return final_mean


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