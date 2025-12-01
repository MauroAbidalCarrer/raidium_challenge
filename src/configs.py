from typing import Sequence
from dataclasses import dataclass, field


N_CLASSES = 55

@dataclass
class TrainingConfig:
    batch_size: int = 1
    n_epochs: int = 600
    starting_lr: float = 5e-4
    cross_entropy_loss_weight: float = 1
    dice_loss_weight: float = 2
    invariant_d_loss_weight: float = 1
    use_labels_weight: bool = True
    test_size: float = 0.2
    random_state: int = 0

@dataclass
class ModelConfig:
    channels: Sequence[int] = field(default_factory=lambda: (64, 128, 256))
    strides: Sequence[int] = field(default_factory=lambda: (2, 2))
    kernel_size: int = 3
    num_res_units: int = 2
    act: tuple[str, dict] = field(
        default_factory=lambda: ("leakyrelu", {"negative_slope": 0.01})
    )
    norm: str = "instance"
    dropout: float = 0.0
    bias: bool = True
