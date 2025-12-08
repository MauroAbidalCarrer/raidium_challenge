from typing import Sequence
from dataclasses import dataclass, field

import torch
from torchvision.transforms import v2


N_CLASSES = 55
PIXEL_VALUE_CHANNEL_IDX = N_CLASSES
# The extra channel is for pixel value during ssl
N_MODEL_OUT_CHANNELS = N_CLASSES + 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class DatasetConfig:
    test_size: float = 0.02
    transform: v2.Transform = v2.Compose([
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
        v2.RandomAffine(degrees=(0, 0), translate=(0.1, 0.3), scale=(0.75, 1)),
    ])

@dataclass
class OptimizerConfig:
    starting_lr: float = 5e-4
    beta0: float = 0.9
    beta1: float = 0.999

@dataclass
class TrainingConfig:
    batch_size: int = 128
    n_epochs: int = 600
    cross_entropy_loss_weight: float = 1
    dice_loss_weight: float = 2
    rec_loss_weight: float = 1
    mask_ratio: float = 0.75
    random_state: int = 0
    optim_cfg: OptimizerConfig=field(default_factory=OptimizerConfig)
    max_lr: float=1e-3

@dataclass
class WandbConfig:
    tags: list[str]
    group: str

@dataclass
class ModelConfig:
    compile: bool = True