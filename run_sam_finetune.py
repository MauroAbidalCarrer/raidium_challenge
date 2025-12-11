import os
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
from scipy import ndimage
from torch import nn, Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from transformers import SamModel, SamProcessor
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


PRETRAINED_MODEL_ID = "facebook/sam-vit-base"
def main():
    dataset.mk_dataset(verbose=False)
    train_cfg = cfg.TrainingConfig(
        batch_size=16,
        n_epochs=10,
        test_size=0.1,
        cross_entropy_loss_weight=1,
        dice_loss_weight=2,
        start_lr=1e-5,
    )
    processor = transformers.SamProcessor.from_pretrained(PRETRAINED_MODEL_ID)
    model = transformers.SamModel.from_pretrained(PRETRAINED_MODEL_ID).to(cfg.DEVICE)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    optimizer = AdamW(
        model.mask_decoder.parameters(),
        lr=train_cfg.start_lr,
        weight_decay=0,
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    criterion = metrics.SegmentationLoss(train_cfg)
    data_loaders = dataset.mk_data_loaders_for_finetuning()
    trainer = training.Trainer(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
        cfg.WandbConfig(["SAM", "finetuning"], "SAM")
    )
    trainer.train_model(data_loaders, criterion, "checkpoints/SAM/sam_epoch_{epoch}.pt")


class SAMWrapper(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.wrapped_model = (
            SamModel.from_pretrained(model_id)
            .to(cfg.DEVICE)
        )
        self.processor = SamProcessor(model_id)
    
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = self.processor(batch["x"])
    
if __name__ == "__main__":
    main()