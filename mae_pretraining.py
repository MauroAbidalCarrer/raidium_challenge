import math
from tqdm import tqdm

import torch
import wandb
import plotly.express as px
from torch import nn, Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, Lambda
from torch.utils.data import TensorDataset, DataLoader

from mae.model import *
from src import dataset, training
from src import configs as cfg
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

    train_loader = dataset.mk_ssl_data_loader()

    model = MAE_ViT(
        image_size=256,
        mask_ratio=mask_ratio,
        patch_size=16,
        emb_dim=256,
        encoder_layer=8,
        encoder_head=8,
        decoder_head=8,
    ).to(cfg.DEVICE)
    model.cfg = dict(mask_ratio=0.75)
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
    train_model(model, train_cfg, optimizer, lr_scheduler, train_loader, train_loader)

def train_model(
        model: torch.nn.Module,
        train_cfg: cfg.TrainingConfig,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
    ) -> dict[str, Tensor]:
    training.wandb_init(
        train_cfg,
        cfg.DatasetConfig(0, None),
        model,
    )
    for e in tqdm(range(train_cfg.n_epochs)):
        epoch_dict = train_model_for_single_epoch(model, optimizer, lr_scheduler, train_loader)
        wandb.log(
            data={"pretraining/" + k: v.item() for k, v in epoch_dict.items()},
            step=e,
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
        losses.append(step_dict["loss"].item())
    lr_scheduler.step()
    avg_loss = sum(losses) / len(losses)
    return {"loss": avg_loss}

def perform_training_step(model: torch.nn.Module, x: Tensor, otpimizer: torch.optim.Optimizer) -> dict[str, Tensor]:
    otpimizer.zero_grad()
    with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
        predicted_img, mask = model(x)
        loss = torch.mean((predicted_img - x) ** 2 * mask) / model.cfg.mask_ratio
    loss.backward()
    otpimizer.step()
    return {"loss": loss}

@torch.no_grad
def wadnb_log_model_images(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
    ):
    model.eval()
    (x, ) = next(iter(data_loader))
    x = dataset.preprocess_imgs(x)
    N_IMGS_TO_PLT = 5
    x = x[:N_IMGS_TO_PLT]
    predicted_val_img, mask = model(x)
    predicted_val_img = predicted_val_img * mask + x * (1 - mask)
    img = torch.cat([x * (1 - mask), predicted_val_img, x], dim=0)
    img = img * dataset.STD + dataset.MEAN 
    np_imgs = (
        img
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )

if __name__ == "__main__":
    main()