
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
import numpy as np
from torch import nn, Tensor
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


def ssl_loss(forward_dict: dict[str, Any]) -> dict[str, Tensor]:
    return {
            "loss": forward_dict["loss"],
            "rec_l1_loss": forward_dict["loss"],
            "rec_l2_loss": forward_dict["loss"] ** 2,
    }

def main():
    model_cfg = cfg.MODELS_CFGS["downscaled_swin_vit"]
    model = models.mk_model_from_cfg(model_cfg)
    model.cfg = model_cfg

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
        ssl_loss,
        "checkpoints/swin_MiM/{wandb_run_name}/swin_ssl_epoch_{epoch}.pt",
    )

if __name__ == "__main__":
    main()
