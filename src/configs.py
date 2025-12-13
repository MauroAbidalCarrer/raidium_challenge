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
    # optim_cfg: OptimizerConfig=field(default_factory=OptimizerConfig)
    #TODO: Remove start lr and other optimizer values in train cfg and move them to appropriate cfg classes
    # start_lr: float=2e-4
    max_lr: float=1e-3
    n_warmup_epochs: int = 50
    transform: v2.Transform = v2.Identity()
    use_cls_weights_for_dice: bool = False
    use_cls_balanced_sampler: bool = False

TRAIN_CONFIGS = {
    "pretraining": TrainingConfig(
        # max_lr=1e-3,
        n_warmup_epochs=50,
        batch_size=64,
        n_epochs=5000,
        transform=v2.RandomAffine(
            degrees=(-20, 20),
            translate=(0.1, 0.3),
            scale=(0.75, 2),
        )
    ),
    "downscaled_pretraining": TrainingConfig(
        batch_size=32,
        n_epochs=5000,
        transform=v2.Compose([
            v2.RandomApply(torch.nn.ModuleList([v2.RandomResizedCrop(256, (0.3, 0.5)),])),
            v2.RandomAffine(
                degrees=(-10, 10),
                translate=(0.1, 0.3),
                scale=(0.75, 2),
            ),
        ]),
    ),
    "finetuning": TrainingConfig(
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
        ]),
        # optim_cfg=OptimizerConfig(OPTIMIZER_CFGS=2e-4)
    ),
    "unet_training": TrainingConfig(
        batch_size=128,
        n_epochs=2000,
        dice_loss_weight = 2,
        cross_entropy_loss_weight= 1,
        mask_ratio = 0,
        use_cls_balanced_sampler = True,
        transform = v2.Compose([
            v2.RandomApply(torch.nn.ModuleList([v2.RandomResizedCrop(256, (0.3, 0.5)),])),
            v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(9, sigma=5)])),
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
    )
}

OPTIM_CFGS = {
    "unet": OptimizerConfig(starting_lr=5e-4),
    "downscaled_vit_pretraining": OptimizerConfig(starting_lr=2e-4),
}

WANDB_RUN_TAGS = {
    "unet": ["segmentation", "unet"],
    "downscaled_pretraining": [
        "pretraining",
        "downscaled",
        "mae_vit",
        "ssl"
    ]
}

@dataclass
class ModelConfig:
    architecutre: str
    compile: bool = False
    constructor_kwargs: dict = field(default_factory=dict)
    downscaling: Optional[int] = None

MODELS_CFGS = {
    "downscaled_vit": ModelConfig(
        architecutre="mae_vit",
        downscaling=2,
    ),
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