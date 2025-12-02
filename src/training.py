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
from torch.utils.data import DataLoader

from src.plotting import plt_pred
from src.metrics import dice_pandas
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
        plt_preds: bool=False,
        x_test: Optional[Tensor]=None
    ):
    wandb_init(train_cfg, dataset_cfg, model)
    if plt_preds and x_test is None:
        print("plt_preds", plt_preds)
        print("x_test", x_test)
        raise ValueError("Did not provide a value for x_test when setting plt_preds to true.")

    torch.backends.cuda.matmul.fp32_precision = 'ieee'

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.starting_lr)
    model = model.to(device)
    step = 0
    training_samples_seen = 0
    for epoch in tqdm(range(train_cfg.n_epochs)):
        step, training_samples_seen = train_model_for_single_epoch(
            model,
            optimizer,
            train_loader,
            criterion,
            train_cfg,
            step,
            training_samples_seen,
        )
        evaluate_model(
            model,
            valid_loader,
            criterion,
            train_cfg,
            step,
            training_samples_seen,
        )

        if save_checkpoint:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch{epoch}.pth')

        if plt_preds:
            x_train, y_train = next(iter(train_loader))
            plt_pred(model, 0, x_train, y_train)
            x_valid, y_valid = next(iter(valid_loader))
            plt_pred(model, 0, x_valid, y_valid)
            plt_pred(model, 20, x_test)

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
    model.train()
    train_loss = 0
    predictions = []
    true_masks = []
    for (image, y_true) in train_loader:
        image = image.to(device=device)
        y_true = y_true.to(device=device)
        model_step_start_time = time()
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y_pred_logits = model(image)
            loss, losses = criterion(y_pred_logits, y_true)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        time_to_perform_model_step = time() - model_step_start_time
        train_loss += loss.item()
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
        pred = torch.argmax(y_pred_logits, dim=1)
        true_masks.append(y_true.cpu().numpy().squeeze())
        predictions.append(pred.squeeze().cpu().numpy())
    predictions = pd.DataFrame(np.concat(predictions).reshape(-1 , 256 * 256))
    valid = pd.DataFrame(np.concat(true_masks).reshape(-1, 256 * 256))
    wandb.log(
        data={
            "training/dice_score": dice_pandas(valid, predictions, N_CLASSES),
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

    for (image, y_true) in valid_loader:
        true_masks.append(y_true.cpu().numpy().squeeze())
        image = image.to(device=device)
        y_true = y_true.to(device=device)

        y_pred_logits = model(image)

        loss, losses = criterion(y_pred_logits, y_true)
        test_loss += loss.item()

        pred = torch.argmax(y_pred_logits, dim=1)
        predictions.append(pred.squeeze().cpu().numpy())

    predictions = pd.DataFrame(np.concat(predictions).reshape(-1 , 256 * 256))
    valid = pd.DataFrame(np.concat(true_masks).reshape(-1, 256 * 256))
    wandb.log(
        data={
            **{"validation/" + k: l.item() for k, l in losses.items()},
            "validation/dice_score": dice_pandas(valid, predictions, N_CLASSES),
            "training/samples_seen": training_samples_seen,
        },
        step=step,
    )

