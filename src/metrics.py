import json
import warnings

import torch
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn.functional as F

from src.configs import (
    TrainingConfig,
    N_CLASSES,
    DEVICE,
)

from src import configs as cfg


class SemiSupervisedLoss:
    """
    Computes a weighted average of mse loss of the images reconstruction, dice score and cross entropy.
    """
    def __init__(self, train_cfg: cfg.TrainingConfig):
        weight = get_class_weights().to(cfg.DEVICE)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.train_cfg = train_cfg

    def __call__(
            self,
            x: Tensor,
            x_hat: Tensor,
            mask: Tensor,
            y_true: Tensor,
        ) -> dict[str, Tensor]:
        # reconstruction loss
        y_true_is_mask = y_true > 0
        pixel_wise_rec_loss_weight = (y_true_is_mask * self.train_cfg.mask_rec_loss_weight ) + ~y_true_is_mask * 1
        pixel_wise_rec_loss_weight = pixel_wise_rec_loss_weight * mask[:, 0]
        pix_wise_rec_loss = F.mse_loss(x_hat[:, 0], x[:, 0], reduction="none") 
        pix_wise_rec_loss = pix_wise_rec_loss * pixel_wise_rec_loss_weight / self.train_cfg.mask_ratio
        rec_loss = pix_wise_rec_loss.mean()
        return {
            "loss": rec_loss,
            "rec_loss": rec_loss,
        }

def dice_pandas(y_true_df: np.ndarray, y_pred_df: np.ndarray) -> float:
    """
    Fully vectorized computation of the average Dice over samples and classes (skip background=0).
    This removes the Python loop over samples by using boolean broadcasting.
    Note: memory usage ~ O(S * P * K) where S = n_samples, P = pixels per sample, K = NUM_CLASSES.
    """
    # transpose to get shape (n_samples, n_pixels)
    y_true = y_true_df.T  # shape (S, P)
    y_pred = y_pred_df.T  # shape (S, P)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape after transpose.")

    S, P = y_true.shape
    classes = np.arange(1, N_CLASSES + 1)  # skip background 0, shape (K,)

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
        weight = get_class_weights().to(DEVICE)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
    
    def __call__(
            self,
            x: Tensor,
            x_hat: Tensor,
            mask: Tensor,
            y_true: Tensor,
        ) -> dict[str, Tensor]:
        """
        Computes the weighted average of the cross entropy and dice loss of y pred.
        ### Returns:
        Dictionnary of loss_average, ce_loss and dice_loss
        """
        y_true = y_true.long()
        y_pred = x_hat[:, 1:]
        ce_loss = self.cross_entropy_loss(y_pred, y_true)
        base_d_loss = torch_dice_loss(y_pred, y_true)
        loss = base_d_loss * self.train_cfg.dice_loss_weight \
            + ce_loss * self.train_cfg.cross_entropy_loss_weight
        return {
            "loss": loss,
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