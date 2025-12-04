import os
from time import time
from tqdm import tqdm
from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
)

import torch
import wandb
import numpy as np
import pandas as pd
from torch import nn, Tensor
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from src.metrics import dice_pandas, torch_dice_score
from src.timing import time_to_run, print_time_dict, time_dict
from src.configs import TrainingConfig, DatasetConfig, N_CLASSES


criterion_type = Callable[[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_LOG_COMMIT_INTERVAL = 10


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
    model = model.to(device)
    step = 0
    training_samples_seen = 0
    for epoch in tqdm(range(train_cfg.n_epochs)):
        with time_to_run("train/total"):
            step, training_samples_seen = train_model_for_single_epoch(
                model,
                optimizer,
                train_loader,
                criterion,
                train_cfg,
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
        train_cfg: TrainingConfig,
        step: int,
        training_samples_seen: int,
    ) -> tuple[int, int]:
    monai_dice_metric = DiceMetric(num_classes=N_CLASSES)
    model.train()
    total_loss = 0
    predictions = []
    true_masks = []
    n_batches = len(train_loader)
    train_loader = iter(train_loader)
    for batch_idx in range(n_batches):
        with time_to_run("train/get batch"):
            image, y_true = next(train_loader)
        with time_to_run("train/move to device"):
            image = image.to(device=device)
            y_true = y_true.to(device=device)
        model_step_start_time = time()
        with time_to_run("train/forward pass"):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                y_pred_logits = model(image)
                loss, losses = criterion(y_pred_logits, y_true)
        with time_to_run("train/backward pass"):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        time_to_perform_model_step = time() - model_step_start_time
        with time_to_run("train/add loss item to train total_loss"):
            total_loss += loss.item()
        with time_to_run("train/wandb log"):
            wandb.log(
                data=
                {
                    **{"training/" + k: l.item() for k, l in losses.items()},
                    "performance/time_to_perform_model_step": time_to_perform_model_step,
                    "training/samples_seen": training_samples_seen,
                },
                step=step,
                commit=step % WANDB_LOG_COMMIT_INTERVAL == 0,
            )
        step += 1
        training_samples_seen += len(image)
        with time_to_run("train/move preds & masks to cpu"):
            y_pred_argmax = torch.argmax(y_pred_logits, dim=1)
            true_masks.append(y_true.cpu().numpy().squeeze())
            predictions.append(y_pred_argmax.squeeze().cpu().numpy())
        with time_to_run("train/monai dice score"):
            with torch.no_grad():
                one_hot_y_pred = torch.nn.functional.one_hot(y_pred_argmax.long(), num_classes=N_CLASSES).permute(0, 3, 1, 2)
                one_hot_y_true = torch.nn.functional.one_hot(y_true.long(), num_classes=N_CLASSES).permute(0, 3, 1, 2)
                # print("y_true.shape", y_true.shape, "y_pred_argmax.shape", y_pred_argmax.shape, "image.shape", image.shape)
                monai_dice_metric(one_hot_y_pred, one_hot_y_true)
    with time_to_run("train/monai dice score agg"):
        monai_dice_score_agg = monai_dice_metric.aggregate()
    with time_to_run("train/monai dice score item"):
        monai_dice_score = monai_dice_score_agg.item()
    with time_to_run("train/monai dice score reset"):
        monai_dice_metric.reset()
    with time_to_run("train/torch_dice_score"):
        dice_score_torch = torch_dice_score(y_pred_logits, y_true)
    with time_to_run("train/dice_score_torch item"):
        dice_score_torch = dice_score_torch.item()
    with time_to_run("train/pandas dice score"):
        predictions = pd.DataFrame(np.concat(predictions).reshape(-1 , 256 * 256))
        valid = pd.DataFrame(np.concat(true_masks).reshape(-1, 256 * 256))
        dice_score = dice_pandas(valid, predictions, N_CLASSES)
    print("monai dice score:", monai_dice_score)
    print("pandas dice score:", dice_score)
    print("dice_score_torch:", dice_score_torch)
    print("abs dice_score_torch - manai dice score:", abs(dice_score_torch - monai_dice_score))
    print("abs pandas - manai dice score:", abs(dice_score - monai_dice_score))
    with time_to_run("train/wandb log dice score & samples seen"):
        wandb.log(
            data={
                "training/dice_score": dice_score,
                "training/samples_seen": training_samples_seen,
            },
            step=step,
        )
    return step, training_samples_seen

@torch.no_grad
def evaluate_model(
        model: nn.Module,
        valid_loader: DataLoader,
        criterion: criterion_type,
        train_cfg: TrainingConfig,
        step: int,
        training_samples_seen: int,
    ):
    model.eval()
    test_loss = 0
    predictions = []
    true_masks = []
    n_batches = len(valid_loader)
    valid_loader = iter(valid_loader)
    for batch_idx in range(n_batches):
        with time_to_run("eval/get bacth"):
            image, y_true = next(valid_loader)
        with time_to_run("eval/move to device"):
            image = image.to(device=device)
            y_true = y_true.to(device=device)
        with time_to_run("eval/forward pass"):
            y_pred_logits = model(image)
            loss, losses = criterion(y_pred_logits, y_true)
        with time_to_run("eval/add loss item to train total_loss"):
            test_loss += loss.item()
        with time_to_run("eval/move preds to CPU valid"):
            pred = torch.argmax(y_pred_logits, dim=1)
            true_masks.append(y_true.cpu().numpy().squeeze())
            predictions.append(pred.squeeze().cpu().numpy())
    with time_to_run("eval/pandas dice score"):
        predictions = pd.DataFrame(np.concat(predictions).reshape(-1 , 256 * 256))
        valid = pd.DataFrame(np.concat(true_masks).reshape(-1, 256 * 256))
        dice_score = dice_pandas(valid, predictions, N_CLASSES)
    with time_to_run("eval/wandb log"):
        wandb.log(
            data={
                **{"validation/" + k: l.item() for k, l in losses.items()},
                "validation/dice_score": dice_score,
                "training/samples_seen": training_samples_seen,
            },
            step=step,
        )

