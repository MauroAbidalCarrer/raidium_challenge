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
