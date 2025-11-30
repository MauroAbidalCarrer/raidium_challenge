from typing import Sequence
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int=1
    n_classes: int=55
    n_epochs: int=100
    starting_lr: float=00.5
    cross_entropy_loss_weight: float=1
    dice_loss_weight: float=0
    invariant_d_loss_weight: float=True
    use_labels_weight: bool=1e-4

@dataclass
class ModelConfig:
    channels: Sequence[int]=(64, 128, 256),      # encoder channels + bottleneck
    strides: Sequence[int]=(2, 2),               # two encoder stages → two strides
    kernel_size: int=3,
    num_res_units: int=2,              # two residual blocks per stage
    act:tuple[str, dict]=("leakyrelu", {"negative_slope": 0.01}),
    norm: str="instance",
    dropout: float=0.0,
    bias: bool=True,

