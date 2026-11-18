import torch
from torch import Tensor

from raidium_challenge.config_classes import N_CLASSES, DEVICE


@torch.no_grad()
def class_wise_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_bckg: bool = True,
    n_classes: int = N_CLASSES,
) -> Tensor:
    """
    Compute Dice independently for every image and class.

    Args:
        y_pred: Predicted class softmax probabilies, shape [B, C, H, W].
        y_true: Ground-truth class indices, shape [B, H, W].
        ignore_bckg: Whether to exclude class 0 from the result.
        n_classes: Total number of classes, INCLUDING background.

    Returns:
        Tensor of shape [B, C], where:
            B = batch size
            C = n_classes - int(ignore_bckg)

        Classes absent from both prediction and ground truth
        receive NaN, matching the competition metric.
    """
    y_pred = y_pred.argmax(1).flatten(1)
    y_true = y_true.flatten(1).long()
    correct = y_pred == y_true
    true_positives = count_by_class(y_true, correct, n_classes)
    pred_area = count_by_class(y_pred, torch.ones_like(y_pred), n_classes)
    true_area = count_by_class(y_true, torch.ones_like(y_true), n_classes)
    denominator = pred_area + true_area
    # Convert to float64 to avoid integer division and improve numerical precision.
    dice = 2.0 * true_positives.double() / denominator.double()
    dice = dice[:, int(ignore_bckg):] # remove background if needed
    return dice

def count_by_class(cls_idx: Tensor, values: Tensor, n_classes: int) -> Tensor:
    """B: batch size, P: n pixles in flatten imgs, C: n_classes

    Args:
        labels (Tensor): [B, P]
        values (Tensor): [B, P]

    Returns:
        Tensor: [B, C]
    """
    counts = torch.zeros(
        cls_idx.shape[0],
        n_classes,
        dtype=torch.int64,
        device=cls_idx.device,
    )
    # Add value to the counter of the corresponding class.
    counts.scatter_add_(
        dim=1,
        index=cls_idx,
        src=values.long(),
    )
    return counts


class DiceScorer:
    def __init__(self,
            n_classes: int=N_CLASSES,
            skip_background: bool=True,
            device=DEVICE,
        ):
        self.device = device
        self.n_classes = n_classes
        self.skip_bckg = skip_background
        self.reset()

    def reset(self):
        self.n_samples = 0
        # Memory inneficient but this will help us find which images have the worst/best scores
        # And also just easier to implement
        self.class_wise_sample_scores = [] 

    @torch.no_grad()
    def step(self, y_pred: Tensor, y_true: Tensor):
        dice_scores_update = class_wise_dice_score(y_pred, y_true, self.skip_bckg, self.n_classes)
        self.class_wise_sample_scores.append(dice_scores_update)

    @torch.no_grad()
    def get_dice_score_per_class(self) -> Tensor:
        return torch.cat(self.class_wise_sample_scores).nanmean(0)

    @torch.no_grad()
    def get_dice_score(self) -> Tensor:
        return self.get_dice_score_per_class().nanmean()


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

    dice_scorer.reset()
    perfect_y_pred = F.one_hot(y_train.to(dtype=torch.long), N_CLASSES)
    perfect_y_pred = perfect_y_pred.movedim(3, 1)
    dice_scorer.step(perfect_y_pred, y_train)
    dice_score_from_perfect_pred = dice_scorer.get_dice_score()
    print("dice_score_from_perfect_pred:", dice_score_from_perfect_pred)

    dice_scorer.reset()
    perturebed_y_pred = perfect_y_pred + random_y_pred * 2
    dice_scorer.step(perturebed_y_pred, y_train)
    dice_score_perturebed_pred = dice_scorer.get_dice_score()
    print("dice_score_perturebed_pred:", dice_score_perturebed_pred)
