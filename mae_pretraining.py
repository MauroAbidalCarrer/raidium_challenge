import math
from tqdm import tqdm
from typing import Any

import torch
import wandb
import plotly.express as px
from torch import nn, Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from torch.utils.data import TensorDataset, DataLoader

from mae.model import *
from src import configs as cfg
from src import dataset, training
from mae.utils import setup_seed


def main():
    max_lr=1e-3
    mask_ratio=0.75

    train_cfg = cfg.TrainingConfig(
        n_epochs=5000,
        batch_size=64,
    )

    dataset.mk_dataset(verbose=False)
    setup_seed(train_cfg.random_state)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'

    train_loader = dataset.mk_ssl_data_loader(train_cfg)

    model = MAE_ViT(
        image_size=256,
        mask_ratio=mask_ratio,
        patch_size=16,
        emb_dim=256,
        encoder_layer=8,
        encoder_head=8,
        decoder_head=8,
    ).to(cfg.DEVICE)
    model_cfg = cfg.ModelConfig()
    model_cfg.mask_ratio = 0.75
    model.cfg = model_cfg
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr * train_cfg.batch_size / 256,
        betas=(0.9, 0.95),
    )
    def lr_func(epoch: int) -> float:
        return min(
            (epoch + 1) / (train_cfg.n_epochs // 10 + 1e-8),
            0.5 * (math.cos(epoch / train_cfg.n_epochs * math.pi) + 1)
        )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    train_model(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
        train_loader,
        cfg.WandbConfig(["pretraining", "MAE"], "pretraining"),
    )

def train_model(
        model: torch.nn.Module,
        train_cfg: cfg.TrainingConfig,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        wandb_cfg: cfg.WandbConfig,
    ) -> dict[str, Tensor]:
    wandb_init(model, wandb_cfg, train_cfg)
    for epoch in tqdm(range(train_cfg.n_epochs)):
        epoch_dict = train_model_for_single_epoch(model, optimizer, lr_scheduler, train_loader)
        wandb.log(
            data={"training/" + k: v.item() for k, v in epoch_dict.items()},
            step=epoch,
        )

def train_model_for_single_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: torch.utils.data.DataLoader,
    ) -> dict[str, Tensor]:
    model.train()
    losses = []
    for (x,) in train_loader:
        x = dataset.preprocess_imgs(x)
        step_dict = perform_training_step(model, x, optimizer)
        losses.append(step_dict["loss"])
    lr_scheduler.step()
    avg_loss = sum(losses) / len(losses)
    return {"loss": avg_loss}

def wandb_init(
        model: nn.Module,
        wandb_cfg: cfg.WandbConfig,
        *configs: list[Any],
    ):
    cfg_vars = {}
    for cfg in configs:
        cfg_vars |= vars(cfg)
    wandb.init(
        project="raidium-challenge",
        config={
            **cfg_vars,
            "model_class": type(model).__class__,
        },
        **vars(wandb_cfg),
    )

def perform_training_step(model: torch.nn.Module, x: Tensor, otpimizer: torch.optim.Optimizer) -> dict[str, Tensor]:
    otpimizer.zero_grad()
    with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
        predicted_img, mask = model(x)
        loss = torch.mean((predicted_img - x) ** 2 * mask) / model.cfg.mask_ratio
    loss.backward()
    otpimizer.step()
    return {"loss": loss}

if __name__ == "__main__":
    main()