import json
import warnings
from typing import Optional, Tuple

import torch
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn.functional as F

from src.configs import TrainingConfig, N_CLASSES
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
        base_d_loss = base_dice_loss(y_pred, y_true)
        loss = base_d_loss * self.train_cfg.dice_loss_weight \
            + ce_loss * self.train_cfg.cross_entropy_loss_weight
        return loss, {
            "average_loss": loss,
            "cross_entropy_loss": ce_loss,
            "dice_loss": base_d_loss,
        }

class MyAwesomeLoss:
    def __init__(self, train_cfg: TrainingConfig, class_weights: Optional[Tensor]=None):
        self.train_cfg = train_cfg
        self.class_weights = class_weights
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # y_true_one_hot = one_hot_y_true(y_true)
        ce = torch.nn.functional.cross_entropy(
            y_pred,
            y_true.type(torch.long),
            # ignore_index=0,
            weight=self.class_weights,
        )
        return ce, {"my_awesome_loss": ce}
        
def my_awesome_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    ce = torch.nn.functional.cross_entropy(
        y_pred,
        y_true.type(torch.long),
        # ignore_index=0,
    )
    return ce, {"my_awesome_loss": ce}

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