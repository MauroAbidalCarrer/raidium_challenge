from typing import Sequence
from dataclasses import dataclass, field

import torch
from torchvision.transforms import v2


N_CLASSES = 55
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
class TrainingConfig:
    batch_size: int = 64
    n_epochs: int = 600
    starting_lr: float = 5e-4
    cross_entropy_loss_weight: float = 1
    dice_loss_weight: float = 2
    invariant_d_loss_weight: float = 1
    use_labels_weight: bool = True
    random_state: int = 0

@dataclass
class ModelConfig:
    channels: Sequence[int] = field(default_factory=lambda: (64, 128, 256, 512))
    strides: Sequence[int] = field(default_factory=lambda: (2, 2, 2))
    kernel_size: int = 3
    num_res_units: int = 2
    act: tuple[str, dict] = field(
        default_factory=lambda: ("leakyrelu", {"negative_slope": 0.01})
    )
    norm: str = "instance"
    dropout: float = 0.0
    bias: bool = True
