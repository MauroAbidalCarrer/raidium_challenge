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
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

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
    if len(sys.argv) > 1:
        trainer = training.Trainer.from_checkpoint(sys.argv[1])
    else:
        trainer = mk_trainer_from_scratch()
    data_loaders = dataset.mk_semi_supervised_data_loaders(trainer.train_cfg)
    criterion = metrics.SemiSupervisedLoss(trainer.train_cfg)
    trainer.train_model(data_loaders, criterion, "checkpoints/pretrained/vit_epoch_{epoch}.pt")

def mk_trainer_from_scratch() -> training.Trainer:
    # setup configs
    train_cfg = cfg.TrainingConfig(
        max_lr=1e-3,
        n_epochs=5000,
        batch_size=64,
    )
    model_cfg = cfg.ModelConfig(
        n_encoder_heads=8,
        n_encoder_layers=4,
        compile=False,
    )
    # setup objects (no I didn't count the configs as objects...)
    model = models.MAE_ViT.from_config(model_cfg)
    optimizer = training.mk_optimizer(model, train_cfg)
    lr_scheduler = training.mk_lr_scheduler(train_cfg, optimizer)
    # start training
    return training.Trainer(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
        cfg.WandbConfig(["pretraining", "MAE"], "pretraining")
    )


if __name__ == "__main__":
    main()