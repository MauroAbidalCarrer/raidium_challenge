import torch
from torch import Tensor

from raidium_challenge.config_classes import N_CLASSES, DEVICE


@torch.no_grad()
def dice_enumerator(y_pred_argmax: Tensor, y_true: Tensor, n_classes=N_CLASSES) -> Tensor:
    """
    Computes the class wise denominator of the dice division
    Args:
        y_pred: y logits of shape [B, C, H, W]
        y_true: y ground truth of shape [B, H, W] where each value is the index of the class
    """
    pred_is_correct = y_pred_argmax == y_true
    return y_true[pred_is_correct].bincount(minlength=n_classes)

@torch.no_grad()
def dice_denominator(y_pred_argmax: Tensor, y_true: Tensor, n_classes: int=N_CLASSES) -> Tensor:
    y_pred_area = torch.bincount(y_pred_argmax.flatten(), minlength=n_classes)
    y_true_area = torch.bincount(y_true.flatten(), minlength=n_classes)
    return y_pred_area + y_true_area

class DiceScorer:
    def __init__(self, n_classes: int=N_CLASSES, device=DEVICE):
        self.n_classes = n_classes
        self.device = device
        self.reset_numerators()

    def reset_numerators(self):
        self.enumerator = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)
        self.denominator = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)
    
    @torch.no_grad()
    def step(self, y_pred: Tensor, y_true: Tensor):
        y_pred_argmax = y_pred.argmax(dim=1)
        self.enumerator += dice_enumerator(y_pred_argmax, y_true, self.n_classes) 
        self.denominator += dice_denominator(y_pred_argmax, y_true, self.n_classes) 

    def get_dice_score_per_class(self) -> Tensor:
        return 2 * self.enumerator / self.denominator

    def get_dice_score(self) -> Tensor:
        return torch.nanmean(self.get_dice_score_per_class())


if __name__ == "__main__":
    import torch.nn.functional as F

    from raidium_challenge import dataset

    x_train, y_train, x_test = dataset.load_raw_dataset("cpu")
    x_train, y_train = dataset.remove_samples_without_labels(x_train, y_train)
    BATCH_SIZE = 20
    x_train = x_train[:BATCH_SIZE]
    y_train = y_train[:BATCH_SIZE]
    y_pred_shape = (BATCH_SIZE, N_CLASSES, y_train.shape[1], y_train.shape[2])
    random_y_pred = torch.rand(y_pred_shape, dtype=torch.float32)
    dice_scorer = DiceScorer()
    dice_scorer.step(random_y_pred, y_train)
    dice_score_from_random_pred = dice_scorer.get_dice_score()
    print("dice_score_from_random_pred:", dice_score_from_random_pred)

    dice_scorer.reset_numerators()
    perfect_y_pred = F.one_hot(y_train.to(dtype=torch.long), N_CLASSES)
    perfect_y_pred = perfect_y_pred.movedim(3, 1)
    dice_scorer.step(perfect_y_pred, y_train)
    dice_score_from_perfect_pred = dice_scorer.get_dice_score()
    print("dice_score_from_perfect_pred:", dice_score_from_perfect_pred)

    dice_scorer.reset_numerators()
    perturebed_y_pred = perfect_y_pred + random_y_pred * 2
    dice_scorer.step(perturebed_y_pred, y_train)
    dice_score_perturebed_pred = dice_scorer.get_dice_score()
    print("dice_score_perturebed_pred:", dice_score_perturebed_pred)
