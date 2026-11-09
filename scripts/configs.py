from pathlib import Path

import torch
from torchvision.transforms import v2

from raidium_challenge.config_classes import (
    TrainingConfig,
    ModelConfig,
    OptimizationConfig,
)


TRAIN_CONFIGS = {
    "unet_training": TrainingConfig(
        batch_size=32,
        n_epochs=600,
        dice_loss_weight = 2,
        cross_entropy_loss_weight= 1,
        sampling = "weighted",
        dice_loss = "custom",
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
        target_metric = "validation/dice_score",
        target_objective = "max",
        ckpt_dir = "artifacts/checkpoints/unet_supervised",
    ),
    "swin_pretraining": TrainingConfig(
        n_epochs=2000,
        batch_size=32,
        transform=v2.RandomApply(torch.nn.ModuleList([v2.RandomResizedCrop(256, (0.3, 0.5)),])),
        max_loss_norm=3.259898655506028, 
        target_metric = "validation/mae_loss",
        target_objective = "min",
        ckpt_dir = "cache/checkpoints/swin_pretraining",
    ),
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
    ],
    "fpn": [
        "segmentation",
        "downscaled",
        "swin_vit",
        "fpn"
    ],
}

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
