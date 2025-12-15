import json
import warnings
from functools import partial
from typing import Optional, Any

import torch
import monai
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


class SelfSupervisedLoss:
    """
    Computes a weighted average of mse loss of the images reconstruction, dice score and cross entropy.
    """
    def __init__(self, train_cfg: cfg.TrainingConfig):
        weight = get_class_weights().to(cfg.DEVICE)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.train_cfg = train_cfg

    def __call__(self, forward_dict: dict[str, Any]) -> dict[str, Tensor]:
        # reconstruction loss
        x = forward_dict["x"]
        mask = forward_dict["mask"]
        x_hat = forward_dict["x_hat"]
        pix_wise_rec_loss = F.mse_loss(x_hat, x, reduction="none")
        pix_wise_rec_loss = pix_wise_rec_loss * mask[:, 0] / self.train_cfg.mask_ratio
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
        class_weights = get_class_weights().to(DEVICE)
        self.class_weights = class_weights
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        if self.train_cfg.use_cls_weights_for_dice:
            print("Going to apply class weights to dice loss.")
        print("dice loss method:", train_cfg.dice_loss)
        if train_cfg.dice_loss == "generalized_monai":
            self.dice_loss = monai.losses.GeneralizedDiceLoss(
                train_cfg.include_backgroud,
                softmax=True,
                to_onehot_y=True,
            )
        elif train_cfg.dice_loss == "monai":
            self.dice_loss = monai.losses.DiceLoss(
                train_cfg.include_backgroud,
                softmax=True,
                to_onehot_y=True,
            )
        elif train_cfg.dice_loss == "custom":
            self.dice_loss = partial(
                torch_dice_loss,
                class_weights=self.class_weights if self.train_cfg.use_cls_weights_for_dice else None,
                inclue_background=self.train_cfg.include_backgroud,
            )
        else:
            raise NotImplementedError(f"Dice loss '{train_cfg.dice_loss}' not implemented.")
    
    def __call__(self, forward_dict: dict[str, Any]) -> dict[str, Tensor]:
        """
        Computes the weighted average of the cross entropy and dice loss of y pred.
        ### Returns:
        Dictionnary of loss_average, ce_loss and dice_loss
        """
        y_true = forward_dict["y_true"].long()
        y_pred = forward_dict["y_pred"]
        ce_loss = self.cross_entropy_loss(y_pred, y_true)
        if "monai" in self.train_cfg.dice_loss:
            dice_loss = self.dice_loss(y_pred, y_true.unsqueeze(1))
        else:
            dice_loss = self.dice_loss(y_pred, y_true)
        loss = dice_loss * self.train_cfg.dice_loss_weight \
            + ce_loss * self.train_cfg.cross_entropy_loss_weight
        return {
            "loss": loss,
            "cross_entropy_loss": ce_loss,
            "dice_loss": dice_loss,
        }

def torch_dice_loss(
        y_pred: Tensor,
        y_true_one_hot: Tensor,
        class_weights: Optional[Tensor]=None,
        inclue_background: Optional[bool] = None,
        smooth: float=1e-7,
    ) -> Tensor:
    y_pred = torch.softmax(y_pred, dim=1)
    y_true_one_hot = (
        torch.nn.functional.one_hot(
            y_true_one_hot,
            num_classes=y_pred.shape[1],
        )
        .permute(0, 3, 1, 2)
    )
    if inclue_background is not None and not inclue_background:
        y_pred = y_pred[:, 1:]
        y_true_one_hot = y_true_one_hot[:, 1:]
    intersection = (y_pred * y_true_one_hot).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    if class_weights is not None:
        dice = dice * class_weights.unsqueeze(0)
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