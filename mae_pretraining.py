import math
from tqdm import tqdm
from typing import Any

import torch
import wandb
from torch import nn, Tensor
from torch.optim import Optimizer, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from mae.model import *
from src import configs as cfg
from src import dataset, metrics
from mae.utils import setup_seed

# Add segmentation loss
# Add images
# Submit
# Add checkpoints

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
        out_channels=cfg.N_CLASSES + 1,
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

def mk_data_loaders(train_cfg: cfg.TrainingConfig) -> tuple[DataLoader, DataLoader]:
    x_train, y_train, x_test = dataset.load_raw_dataset()
    x_train, y_train, x_valid, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=train_cfg.test_size,
    )
    x_train = torch.cat((x_train, x_test))
    y_train = torch.cat((
        y_train,
        torch.zeros(x_test.shape[0], 256, 256, dtype=torch.unit8),
    ))
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)
    train_loader = DataLoader(train_dataset, train_cfg.batch_size)
    valid_loader = DataLoader(valid_dataset, train_cfg.batch_size)
    return train_loader, valid_loader

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

def perform_training_step(
        model: torch.nn.Module,
        x: Tensor,
        y_true: Tensor,
        otpimizer: torch.optim.Optimizer
    ) -> dict[str, Tensor]:
    otpimizer.zero_grad()
    with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
        predicted_img, mask = model(x)
        loss_dict = criterion(x, predicted_img, mask, y_true, model)
    loss_dict["loss"].backward()
    loss_norm = torch.optim.
    otpimizer.step()
    return {"loss": loss}

def criterion(
        x: Tensor,
        x_hat: Tensor,
        mask: Tensor,
        y_true: Tensor,
        model: nn.Module
    ) -> dict[str, Tensor]:
    reconstruction_loss = torch.mean((predicted_img - x) ** 2 * mask) / model.cfg.mask_ratio
    return {
        "reconstruction_loss": reconstruction_loss
    }

def reconstruct_img(predicted_img: Tensor, mask: Tensor, x: Tensor) -> Tensor:
    return predicted_img * mask + x * (1 - mask)

if __name__ == "__main__":
    main()