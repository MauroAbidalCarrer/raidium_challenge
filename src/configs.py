from typing import Sequence
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int
    n_classes: int
    n_epochs: int
    starting_lr: float
    cross_entropy_loss_weight: float
    dice_loss_weight: float
    use_labels_weight: bool

@dataclass
class ModelConfig:
    channels: Sequence[int]       # encoder channels + bottleneck
    strides: Sequence[int]               # two encoder stages → two strides
    kernel_size: int
    num_res_units: int              # two residual blocks per stage
    act:tuple[str, dict]
    norm: str
    dropout: float
    bias: bool

