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
    if len(sys.argv) <= 1:
        print("Please provide a path to a pretrained ViT checkpoint.")
    chkpt_pth = sys.argv[1]
    checkpt_dict = torch.load(chkpt_pth, weights_only=False)
    train_cfg = cfg.TrainingConfig(
        n_epochs=300,
        batch_size=64,
        test_size=0.1,
        mask_ratio=0,
    )
    data_loaders = mk_data_loaders_for_finetuning(train_cfg)
    model_cfg = cfg.ModelConfig(**checkpt_dict["model_cfg"])
    model = models.MAE_ViT.from_config(model_cfg)
    model.load_state_dict(checkpt_dict["model"])
    criterion = metrics.SegmentationLoss(train_cfg)
    optimizer = training.mk_optimizer(model, train_cfg)
    lr_scheduler = OneCycleLR(
        optimizer,
        train_cfg.max_lr,
        epochs=train_cfg.n_epochs,
        steps_per_epoch=len(data_loaders["train"]),
    )
    trainer = training.Trainer(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
    )
    trainer.train_model(data_loaders, criterion)

def mk_data_loaders_for_finetuning(
        train_cfg: cfg.TrainingConfig
    ) -> dict[str, DataLoader]:
    x_train, y_train, x_test = dataset.load_raw_dataset(cfg.DEVICE)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=train_cfg.test_size,
    )
    def mk_dl_from_tensors(*tensors: list[Tensor]) -> DataLoader:
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, train_cfg.batch_size)
    y_test_fill = torch.zeros(len(x_test), 256, 256, device=cfg.DEVICE)
    return {
        "train": mk_dl_from_tensors(x_train, y_train),
        "valid": mk_dl_from_tensors(x_valid, y_valid),
        "test": mk_dl_from_tensors(x_test, y_test_fill),
    }

def mk_trainer_from_scratch() -> training.Trainer:
    # setup configs
    train_cfg = cfg.TrainingConfig(
        max_lr=1e-3,
        n_epochs=500,
        batch_size=64,
    )
    return 


if __name__ == "__main__":
    main()