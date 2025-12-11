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
    train_cfg = cfg.TRAIN_CONFIGS["finetuning"]
    data_loaders = dataset.mk_data_loaders_for_finetuning(train_cfg)
    model_cfg = cfg.ModelConfig(**checkpt_dict["model_cfg"])
    model = models.MAE_ViT.from_config(model_cfg)
    model.load_state_dict(checkpt_dict["model"])
    # for param in model.encoder.parameters():
    #     param.requires_grad_(False)
    # model.decoder = models.MAE_DecoderBF(256, 16, 256, 8, num_head=8, out_channels=56).to(cfg.DEVICE)
    # for param in model.decoder.head.parameters():
    #     param.requires_grad_(True)
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
        "checkpoints/finetuned/vit_epoch_{epoch}.pt"
    )


if __name__ == "__main__":
    main()
