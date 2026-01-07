
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


def main():
    model_cfg = cfg.MODELS_CFGS["downscaled_swin_vit"]
    optim_cfg = cfg.OPTIM_CFGS["downscaled_vit_pretraining"]
    train_cfg = cfg.TRAIN_CONFIGS["swin_pretraining"]

    print("Creating objects")
    model = models.mk_model_from_cfg(model_cfg)
    model.cfg = model_cfg
    optim = training.mk_optimizer(model, optim_cfg)
    lr_scheduler = training.mk_lr_scheduler(cfg.TRAIN_CONFIGS["swin_pretraining"], optim)
    data_loaders = dataset.mk_ssl_loaders(train_cfg)
    print("Creating wandb")
    wandb_run = training.wandb_init(
        model_cfg,
        train_cfg,
        optim_cfg,
        tags=cfg.WANDB_RUN_TAGS["downscaled_swin_pretraining"],
        group="manual_training",
    )

    print("Creating trainer")
    trainer = training.Trainer(
        model,
        cfg.TRAIN_CONFIGS["swin_pretraining"],
        optim,
        lr_scheduler,
        wandb_run,
    )
    print("Started training")
    trainer.train_model(
        data_loaders,
        metrics.ssl_loss,
        "checkpoints/swin_MiM/{wandb_run_name}/swin_ssl_epoch_{epoch}.pt",
    )

if __name__ == "__main__":
    main()
