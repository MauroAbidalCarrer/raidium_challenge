from typing import (
    Optional,
    Callable,
    Dict,
    Tuple,
)
from typing import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torchvision.transforms import v2


N_CLASSES = 55
PIXEL_VALUE_CHANNEL_IDX = N_CLASSES
# The extra channel is for pixel value during ssl
N_MODEL_OUT_CHANNELS = N_CLASSES + 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_t = Callable[[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]


@dataclass
class OptimizerConfig:
    starting_lr: float = 5e-4
    beta0: float = 0.9
    beta1: float = 0.999

@dataclass
class TrainingConfig:
    n_epochs: int = 5000
    batch_size: int = 128
    test_size: float = 0.02
    # losses
    cross_entropy_loss_weight: float = 1
    dice_loss_weight: float = 2
    rec_loss_weight: float = 1
    mask_ratio: float = 0.75
    random_state: int = 0
    optim_cfg: OptimizerConfig=field(default_factory=OptimizerConfig)
    #TODO: Remove start lr and other optimizer values in train cfg and move them to appropriate cfg classes
    start_lr: float=2e-4
    max_lr: float=1e-3
    n_warmup_epochs: int = 50
    transform: v2.Transform = v2.Identity()
    use_cls_weights_for_dice: bool = False

TRAIN_CONFIGS = {
    "pretraining": TrainingConfig(
        max_lr=1e-3,
        n_warmup_epochs=50,
        batch_size=64,
        n_epochs=5000,
        transform=v2.RandomAffine(
            degrees=(-20, 20),
            translate=(0.1, 0.3),
            scale=(0.75, 2),
        )
    ),
    "finetuning": TrainingConfig(
        start_lr=2e-4,
        batch_size=64,
        n_epochs=600,
        dice_loss_weight = 2,
        cross_entropy_loss_weight= 1,
        mask_ratio = 0,
        transform=v2.Compose([
            v2.RandomErasing(ratio=(0.05, 0.15)),
            v2.RandomErasing(ratio=(0.05, 0.15)),
            v2.RandomErasing(ratio=(0.05, 0.15)),
            v2.RandomAffine(
                degrees=(-20, 20),
                translate=(0.1, 0.3),
                scale=(0.75, 2),
            ),
            v2.RandomErasing(ratio=(0.05, 0.15), value="random"),
            v2.RandomErasing(ratio=(0.05, 0.15), value="random"),
        ])
    ),
    "unet_training": TrainingConfig(
        start_lr=5e-4,
        batch_size=128,
        n_epochs=600,
        dice_loss_weight = 2,
        cross_entropy_loss_weight= 1,
        mask_ratio = 0,
        transform = v2.Compose([
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomAffine(degrees=(0, 0), translate=(0.1, 0.3), scale=(0.75, 1)),
        ]),
        optim_cfg=OptimizerConfig(starting_lr=5e-4)
    )
}

@dataclass
class WandbConfig:
    tags: list[str]
    group: str

@dataclass
class ModelConfig:
    architecutre: str
    constructor_kwargs: dict
    # n_encoder_layers: int = 8
    # n_encoder_heads:  int = 8
    # n_decoder_heads:  int = 8
    compile: bool = True

DFLT_MODELS_CFGS = {
    "unet": ModelConfig(
        architecutre="unet",
        constructor_kwargs={
            "channels": (64, 128, 256, 512),
            "num_res_units": 2,
            "act": ("leakyrelu", {"negative_slope": 0.01}),
            "norm": "instance",
        }
    ),
}

    # channels: Sequence[int] = field(default_factory=lambda: (64, 128, 256, 512))
    # strides: Sequence[int] = field(default_factory=lambda: (2, 2, 2))
    # kernel_size: int = 3
    # num_res_units: int = 2
    # act: tuple[str, dict] = field(
    #     default_factory=lambda: ("leakyrelu", {"negative_slope": 0.01})
    # )
    # norm: str = "instance"
    # dropout: float = 0.0
    # bias: bool = True
