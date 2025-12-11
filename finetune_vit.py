import os
import sys
import math
from tqdm import tqdm
from pathlib import Path
from typing import Any, overload
from collections import defaultdict

import torch
import wandb
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from src import (
    dataset,
    training,
    metrics,
    models,
)
from src import configs as cfg
from src.plotting import plt_sample


# TODO:
# - Add finetuning
# - Add submission creation and submit
# - Switch to one cycle lr


def main():
    if len(sys.argv) < 2:
        print("Please provide a path to a pretrained ViT checkpoint.")
        exit(1)
    chkpt_pth = sys.argv[1]
    checkpt_dict = torch.load(chkpt_pth, weights_only=False)
    train_cfg = cfg.TrainingConfig(
        n_epochs=600,
        batch_size=256,
        test_size=0.1,
        mask_ratio=0,
        start_lr=4e-4,
        cross_entropy_loss_weight=2,
        dice_loss_weight=0.1,
    )
    data_loaders = dataset.mk_data_loaders_for_finetuning(train_cfg)
    model_cfg = cfg.ModelConfig(**checkpt_dict["model_cfg"])
    model = models.MAE_ViT.from_config(model_cfg)
    model.load_state_dict(checkpt_dict["model"])
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.decoder.head.parameters():
        param.requires_grad_(True)
    criterion = metrics.SegmentationLoss(train_cfg)
    # criterion = ClassWeightedMSE(1e-4)
    optimizer = training.mk_optimizer(model, train_cfg)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    trainer = training.Trainer(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
        cfg.WandbConfig(["finetuning", "MAE"], "finetuning")
    )
    trainer.train_model(
        data_loaders,
        criterion,
        "checkpoints/finetuned/vit_epoch{epoch}.pt"
    )

class ClassWeightedMSE:
    def __init__(self, pixel_weight: float=0):
        class_weights = metrics.get_class_weights()
        class_weights = torch.cat((
            torch.ones(1) * pixel_weight,
            class_weights
        ))
        self.class_weights = class_weights.unsqueeze(0)
    
    def __call__(
            self,
            x: Tensor,
            x_hat: Tensor,
            mask: Tensor,
            y_true: Tensor,
        ) -> dict[str, Tensor]:
        one_hot_encoded_y_true = F.one_hot(y_true.long(), cfg.N_CLASSES)
        x = torch.cat(
            (x, one_hot_encoded_y_true.float().unsqueeze(1)),
            dim=1,
        )
        unreduced_mse = F.mse_loss(x_hat, x, reduction="none") # B, C, H, W
        img_reduced_mse = unreduced_mse.flatten(2).mean(dim=2) # B, C
        img_reduced_mse = img_reduced_mse * self.class_weights # B, C
        mse = img_reduced_mse.mean()
        return mse


if __name__ == "__main__":
    main()