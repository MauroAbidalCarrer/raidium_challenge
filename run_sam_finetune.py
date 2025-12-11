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
    model = SAMForSemanticSeg(PRETRAINED_MODEL_ID).to(cfg.DEVICE)
    models.print_model_params_count(model)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    optimizer = AdamW(
        model.seg_head.parameters(),
        lr=train_cfg.start_lr,
        weight_decay=0,
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    criterion = metrics.SegmentationLoss(train_cfg)
    data_loaders = dataset.mk_data_loaders_for_finetuning(train_cfg)
    trainer = training.Trainer(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
        cfg.WandbConfig(["SAM", "finetuning"], "SAM")
    )
    trainer.train_model(data_loaders, criterion, "checkpoints/SAM/sam_epoch_{epoch}.pt")

class SAMForSemanticSeg(nn.Module):
    def __init__(self, pretrained_model_id: str, num_classes=55):
        super().__init__()
        self.sam = SamModel.from_pretrained(pretrained_model_id)
        
        # Freeze the prompt encoders (optional)
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False

        # Replace SAM’s mask decoder with a semantic segmentation head
        embed_dim = self.sam.vision_encoder.config.output_channels  # usually 256

        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, batch: dict[str, Any]) -> Tensor:
        x: Tensor = batch["x"].repeat(1, 3, 1, 1)
        # x: (B, C, H, W)
        # SAM outputs (B, H/16 * W/16, C)
        x = F.interpolate(
            x, 
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False
        )

        image_embeddings = self.sam.vision_encoder(x).last_hidden_state

        B, C, H, W = image_embeddings.shape
        # H16 = W16 = int(HW**0.5)
        # feats = image_embeddings.transpose(1,2).reshape(B, C, H16, W16)

        logits = self.seg_head(image_embeddings)
        logits = F.interpolate(logits, size=batch["x"].shape[-2:], mode="bilinear", align_corners=False)
        return {"y_pred": logits}


class SAMWrapper(nn.Module):
    def __init__(self, model_id: str, img_size: int):
        super().__init__()
        # self.img_size = img_size
        # self.img_indices = torch.arange(img_size ** 2)
        self.wrapped_model = (
            SamModel.from_pretrained(model_id)
            .to(cfg.DEVICE)
        )
        self.processor = SamProcessor(model_id)
    
    def forward(self, batch_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_dict["x"] = self.processor(batch_dict["x"])
        batch_dict["bbox"] = get_bounding_box_batch(batch_dict["y_true"])
        model_output = self.wrapped_model(batch_dict)
        return model_output

def get_bounding_box_batch(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B, H, W) tensor, >0 indicates foreground
    returns: (B, 4) tensor of [x_min, y_min, x_max, y_max]
    """

    B, H, W = mask.shape

    # Boolean mask of shape (B, H, W)
    fg = mask > 0

    # Coordinate grids
    ys = torch.arange(H, device=mask.device).view(1, H, 1)  # (1, H, 1)
    xs = torch.arange(W, device=mask.device).view(1, 1, W)  # (1, 1, W)

    # Expand grids to (B, H, W)
    ys = ys.expand(B, H, W)
    xs = xs.expand(B, H, W)

    # Set non-FG positions to huge for min, and tiny for max
    big = torch.iinfo(torch.int32).max

    # Compute mins
    x_min = torch.where(fg, xs, big).amin(dim=(1, 2))
    y_min = torch.where(fg, ys, big).amin(dim=(1, 2))

    # Compute maxes
    x_max = torch.where(fg, xs, -1).amax(dim=(1, 2))
    y_max = torch.where(fg, ys, -1).amax(dim=(1, 2))

    # Handle empty-masks: when no fg, x_min==big
    empty = x_min == big

    # Normal jitter
    jitter_min = torch.randint(0, 20, (B,), device=mask.device)
    jitter_max = torch.randint(0, 20, (B,), device=mask.device)

    # Apply jitter, clamp within image bounds
    x_min = torch.clamp(x_min - jitter_min, min=0)
    y_min = torch.clamp(y_min - jitter_min, min=0)
    x_max = torch.clamp(x_max + jitter_max, max=W - 1)
    y_max = torch.clamp(y_max + jitter_max, max=H - 1)

    # For empty masks, return full image
    x_min = torch.where(empty, torch.zeros_like(x_min), x_min)
    y_min = torch.where(empty, torch.zeros_like(y_min), y_min)
    x_max = torch.where(empty, torch.full_like(x_max, W - 1), x_max)
    y_max = torch.where(empty, torch.full_like(y_max, H - 1), y_max)

    return torch.stack([x_min, y_min, x_max, y_max], dim=1).long()


if __name__ == "__main__":
    main()