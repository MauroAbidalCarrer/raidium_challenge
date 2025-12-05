import os
import warnings
from tqdm import tqdm
from typing import (
    Callable,
    Dict,
    Tuple,
)

import torch
import wandb
import numpy as np
import pandas as pd
from torch import nn, Tensor
from monai.metrics import DiceMetric
from torchvision.tv_tensors import Mask
from torch.utils.data import DataLoader

from src.metrics import dice_pandas
from src.timing import time_to_run, print_time_dict
from src.configs import TrainingConfig, DatasetConfig, N_CLASSES, DEVICE


# Pro tip: Never fix warnings causes
warnings.filterwarnings("ignore", message="RandomErasing")
criterion_type = Callable[[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]
WANDB_LOG_COMMIT_INTERVAL = 100


# @torch.compile
def train_unet(
        model: nn.Module,
        dataset_cfg: DatasetConfig,
        train_cfg: TrainingConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: criterion_type,
        save_checkpoint: bool=True,
    ):
    wandb_init(train_cfg, dataset_cfg, model)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.starting_lr)
    model = model.to(DEVICE)
    step = 0
    training_samples_seen = 0
    for epoch in tqdm(range(train_cfg.n_epochs)):
        with time_to_run("train/total"):
            step, training_samples_seen = train_model_for_single_epoch(
                model,
                optimizer,
                train_loader,
                criterion,
                dataset_cfg,
                step,
                training_samples_seen,
            )
        with time_to_run("eval/total"):
            evaluate_model(
                model,
                valid_loader,
                criterion,
                train_cfg,
                step,
                training_samples_seen,
            )

        with time_to_run("other/save checkpoint"):
            if save_checkpoint:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch{epoch}.pth')

        print_time_dict()

    wandb.finish()

def wandb_init(train_cfg: TrainingConfig, dataset_cfg: DatasetConfig, model: nn.Module):
    wandb.init(
        project="raidium-challenge",
        config={
            **vars(train_cfg),
            **(vars(dataset_cfg)),
            **(vars(model.cfg) if hasattr(model, "cfg") else {}),
        },
    )

def train_model_for_single_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: criterion_type,
        dataset_cfg: DatasetConfig,
        step: int,
        training_samples_seen: int,
    ) -> tuple[int, int]:
    model.train()
    n_batches = len(train_loader)
    train_loader = iter(train_loader)
    for batch_idx in range(n_batches):
        with time_to_run("train/get batch"):
            x, y_true = next(train_loader)
        with time_to_run("train/to_device and to mask"):
            x = x.to(device=DEVICE)
            y_true = Mask(y_true.to(device=DEVICE))
        with time_to_run("train/data aug"):
            x, y_true = dataset_cfg.transform(x, y_true)
        with time_to_run("train/step"):
            model_step(model, x, y_true, optimizer, dataset_cfg, criterion)
        step += 1
        training_samples_seen += len(x)
    return step, training_samples_seen

@torch.compile
def model_step(model: nn.Module, x: Tensor, y_true: Tensor, optimizer: torch.optim.Optimizer, dataset_cfg: DatasetConfig, criterion: criterion_type):
    optimizer.zero_grad()
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        y_pred_logits = model(x)
        loss, losses = criterion(y_pred_logits, y_true)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

def evaluate_model(
        model: nn.Module,
        valid_loader: DataLoader,
        criterion: criterion_type,
        train_cfg: TrainingConfig,
        step: int,
        training_samples_seen: int,
    ):
    model.eval()
    predictions = []
    true_masks = []
    n_batches = len(valid_loader)
    valid_loader = iter(valid_loader)
    for batch_idx in range(n_batches):
        with time_to_run("eval/get bacth"):
            image, y_true = next(valid_loader)
        with time_to_run("eval/move to device"):
            image = image.to(device=DEVICE)
            y_true = y_true.to(device=DEVICE)
        with time_to_run("eval/forward pass"):
            y_pred_logits = model(image)
            loss, losses = criterion(y_pred_logits, y_true)
        with time_to_run("eval/move preds to CPU valid"):
            pred = torch.argmax(y_pred_logits, dim=1)
            true_masks.append(y_true.cpu().numpy().squeeze())
            predictions.append(pred.squeeze().cpu().numpy())
    with time_to_run("eval/pandas dice score"):
        predictions = np.concat(predictions).reshape(-1 , 256 * 256)
        valid = np.concat(true_masks).reshape(-1, 256 * 256)
        dice_score = dice_pandas(valid, predictions)
    with time_to_run("eval/wandb log"):
        wandb.log(
            data={
                **{"validation/" + k: l.item() for k, l in losses.items()},
                "validation/dice_score": dice_score,
                "training/samples_seen": training_samples_seen,
            },
            step=step,
        )
