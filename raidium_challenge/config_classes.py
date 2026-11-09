from pathlib import Path
from typing import (
    Optional,
    Callable,
    Literal,
    Tuple,
    Dict,
)
from operator import lt, gt
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
    target_metric: str = None
    target_objective: Literal["min", "max"] = None #: callable[[float, float], bool] = None
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
    # eval_interval: int = 50
    ckpt_dir: Path = None

    def __post_init__(self):
        assert self.target_metric is not None, "target metric in training config must be specified."
        assert self.target_objective in ('min', 'max'), f"target_objective must either be min or max, got {self.target_objective}."
        assert self.ckpt_dir is not None, "Please provide a checkpoint dir to save the checkpoints to."
        self.target_improved_fn = lt if self.target_objective == "min" else gt

@dataclass
class ModelConfig:
    architecture: str
    compile: bool = False
    constructor_kwargs: dict = field(default_factory=dict)
    downscaling: Optional[int] = None
    up_scale_output: bool = False
