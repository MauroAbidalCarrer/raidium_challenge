import warnings

import torch
import numpy as np
import pandas as pd
from torch import Tensor


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

def dice_loss(pred: Tensor, target: Tensor, smooth: float=1e-7) -> Tensor:
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