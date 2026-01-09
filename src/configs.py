from typing import (
    Optional,
    Callable,
    Literal,
    Tuple,
    Dict,
)
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
CONFUSE_MAT_METRICS_NAMES = ["recall", "precision", "f1 score"]
X_MEAN: float = 14.0816
X_STD: float = 35.2164


@dataclass
class OptimizationConfig:
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
    random_state: int = 0
    n_warmup_epochs: int = 50
    transform: v2.Transform = v2.Identity()
    use_cls_weights_for_dice: bool = False
    sampling: Literal["weighted", "uniform", "shuffle"] = "shuffle"
    include_backgroud: Optional[bool] = True
    dice_loss: Literal["custom", "monai", "generalized_monai"] = "custom"
    max_loss_norm: float = 1
    checkpointing_interval: int  = 5
    eval_interval: int = 50

TRAIN_CONFIGS = {
    "unet_training": TrainingConfig(
        batch_size=32,
        n_epochs=600,
        dice_loss_weight = 2,
        cross_entropy_loss_weight= 1,
        sampling = "weighted",
        dice_loss = "custom",
        checkpointing_interval = 2,
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
    ),
    "swin_pretraining": TrainingConfig(
        n_epochs=2000,
        batch_size=32,
        transform=v2.RandomApply(torch.nn.ModuleList([v2.RandomResizedCrop(256, (0.3, 0.5)),])),
        eval_interval=10,
        checkpointing_interval=1,
        max_loss_norm=3.259898655506028, 
    )
}

OPTIM_CFGS = {
    "unet": OptimizationConfig(starting_lr=5e-4),
    "downscaled_swin_vit_pretraining": OptimizationConfig(
        # Lowest validation loss last hp tuning
        **{'beta0': 0.8751419558211533, 'beta1': 0.926601537931387, 'starting_lr': 0.0003003025846093607} 
    ),
    "finetuning": OptimizationConfig(starting_lr=5e-4),
}

WANDB_RUN_TAGS = {
    "unet": ["segmentation", "unet"],
    "downscaled_swin_pretraining": [
        "pretraining",
        "downscaled",
        "swin_vit",
        "MiM"
        "ssl"
    ],
    "donwscaled_swin_finetuning": [
        "segmentation",
        "finetuning",
        "downscaled",
        "swin_vit",
    ]
}

@dataclass
class ModelConfig:
    architecture: str
    compile: bool = False
    constructor_kwargs: dict = field(default_factory=dict)
    downscaling: Optional[int] = None
    up_scale_output: bool = False

MODELS_CFGS: dict[str, ModelConfig] = {
    "unet": ModelConfig(
        architecture="unet",
        constructor_kwargs={
            "channels": (64, 128, 256, 512, 512),
            "num_res_units": 2,
            "act": ("leakyrelu", {"negative_slope": 0.01}),
            "norm": "instance",
        }
    ),
    "downscaled_swin_vit": ModelConfig(
        architecture="hf_swin_vit",
        constructor_kwargs={
            # Lowest val loss from last hp run
            "patch_size": 4,
            "embed_dim": 128,
            "n_layers": 4,
            "per_layer_depth": 2,
            "mask_patch_size": 8,
            "mask_ratio": 0.75
        },
        downscaling=2,
        up_scale_output=False,
        compile=False,
    )
}
