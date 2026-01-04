
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
import numpy as np
from torch import nn
from datasets import load_dataset
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
import transformers
from transformers import (
    CONFIG_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForMaskedImageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from src import configs as cfg
from src import dataset, training, models, metrics


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=256, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())

class HFMaskedImageModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mask_generator,
        image_key: str = "x",
    ):
        super().__init__()
        self.model = model
        self.mask_generator = mask_generator
        self.image_key = image_key

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        batch:
            {
                "x": Tensor[B, C, H, W],
                ... (optional extra keys)
            }
        """

        pixel_values = batch[self.image_key]

        # Generate mask only during training
        if self.training:
            # HF expects (B, num_patches)
            bool_masked_pos = torch.stack(
                [self.mask_generator() for _ in range(pixel_values.shape[0])],
                dim=0,
            ).to(pixel_values.device)

            outputs = self.model(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
            )
        else:
            outputs = self.model(pixel_values=pixel_values)

        return outputs

def main():
    # check_min_version("4.57.0.dev0")
    # require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

    model_cfg = cfg.MODELS_CFGS["downscaled_swin_vit"]
    model = models.mk_model_from_cfg(model_cfg)
    mask_generator = MaskGenerator(
        input_size=model_cfg.constructor_kwargs["config"].image_size,
        mask_patch_size=32, # check in more details how this works
        model_patch_size=model_cfg.constructor_kwargs["config"].patch_size,
        mask_ratio=0.75,
    )
    model = HFMaskedImageModelWrapper(model, mask_generator)

    optim = training.mk_optimizer(model, cfg.OPTIM_CFGS["downscaled_vit_pretraining"])
    lr_scheduler = training.mk_lr_scheduler(cfg.TRAIN_CONFIGS["swin_pretraining"], optim)

    train_cfg = cfg.TRAIN_CONFIGS["swin_pretraining"]
    data_loaders = dataset.mk_ssl_loaders(train_cfg)

    wandb_run = training.wandb_init(
        model_cfg,
        train_cfg,
        tags=cfg.WANDB_RUN_TAGS["downscaled_swin_pretraining"],
        group="manual_training",
    )

    trainer = training.Trainer(
        model,
        cfg.TRAIN_CONFIGS["swin_pretraining"],
        optim,
        lr_scheduler,
        wandb_run,
    )
    trainer.train_model(
        data_loaders,
        metrics.SelfSupervisedLoss(train_cfg),
        "checkpoints/swin_MiM/{wandb_run_name}/swin_ssl_epoch_{epoch}.pt",
    )

if __name__ == "__main__":
    main()