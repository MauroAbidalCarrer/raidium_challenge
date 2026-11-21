from pathlib import Path
from typing import (
    Callable,
    Tuple,
    Dict,
)

import torch
from torch import Tensor


N_CLASSES = 55
X_STD: float = 35.2164
X_MEAN: float = 14.0816
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_t = Callable[[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]
RAW_DATASET_DIR = Path("artifacts/dataset/raw/")
PREPROCESSED_DATASET_DIR = Path("artifacts/dataset/preprocessed/")