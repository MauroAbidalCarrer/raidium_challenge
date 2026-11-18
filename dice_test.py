import torch
from torch import Tensor

from raidium_challenge.config_classes import N_CLASSES, DEVICE


# @torch.no_grad()
# def class_wise_dice_score(
#     y_pred_argmax: Tensor,
#     y_true: Tensor,
#     ignore_bckg=True,
#     n_classes=N_CLASSES,
# ) -> Tensor:
#     """Computes the dice score for each class of each image.

#     Args:
#         y_pred_argmax (Tensor): [B, H, W]
#         y_true (Tensor): [B, H, W]

#     Returns:
#         Tensor: [B, C]
#     """
#     pred_is_correct = y_pred_argmax == y_true
#     count_area  = lambda x: x.flatten().bincount(minlength=n_classes)
#     correct_area= count_area(y_true[pred_is_correct])
#     y_pred_area = count_area(y_pred_argmax)
#     y_true_area = count_area(y_true)
#     total_area  = y_pred_area + y_true_area
#     dice_score  = 2 * correct_area / total_area
#     dice_score  = dice_score[:, int(ignore_bckg):] # remove background
#     return dice_score


@torch.no_grad()
def class_wise_dice_score(
    y_pred_argmax: Tensor,
    y_true: Tensor,
    ignore_bckg: bool = True,
    n_classes: int = 55,
) -> Tensor:
    """
    Compute Dice independently for every image and class.

    Args:
        y_pred_argmax: Predicted class indices, shape [B, H, W].
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
    batch_size: int = y_true.shape[0]
    # [B, H, W] -> [B, P]
    # P = H * W
    pred: Tensor = y_pred_argmax.reshape(batch_size, -1).long()
    true: Tensor = y_true.reshape(batch_size, -1).long()
    # [B, P]
    correct: Tensor = pred == true

    def count_by_class(labels: Tensor, values: Tensor) -> Tensor:
        """B: batch size, P: n pixles in imgs, C: n_classes

        Args:
            labels (Tensor): [B, P]
            values (Tensor): [B, P]

        Returns:
            Tensor: [B, C]
        """
        counts: Tensor = torch.zeros(
            batch_size,
            n_classes,
            dtype=torch.int64,
            device=labels.device,
        )
        # For every pixel, add its value to the counter
        # corresponding to its class label.
        # labels: [B, P] -> class indices
        # values: [B, P] -> values to accumulate
        # counts: [B, C] -> output counters
        counts.scatter_add_(
            dim=1,
            index=labels,
            src=values.long(),
        )
        return counts
    # [B, P] -> [B, C]
    true_positives: Tensor = count_by_class(true, correct)
    # [B, P] -> [B, C]
    pred_area: Tensor = count_by_class(
        pred,
        torch.ones_like(pred),
    )
    # [B, P] -> [B, C]
    true_area: Tensor = count_by_class(
        true,
        torch.ones_like(true),
    )
    # [B, C]
    denominator: Tensor = pred_area + true_area
    # Convert to float64 to avoid integer division and
    # improve numerical precision.
    dice: Tensor = (
        2.0 * true_positives.double()
        / denominator.clamp_min(1).double()
    )
    # If both masks are empty for a class, Dice is undefined.
    # Match the competition notebook by assigning NaN.
    dice = dice.masked_fill(
        denominator == 0,
        float("nan"),
    )
    if ignore_bckg:
        # [B, C] -> [B, C - 1]
        dice = dice[:, 1:]
    return dice


class DiceScorer:
    # class_wise_dice_scores_per_image(skip background) numerator: [C] denominator: [1]
    # dataset_and_class_wise_dice_scores [C]
    # dataset_wise_dice_score [1]
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
        numerator_size = self.n_classes - int(self.skip_bckg)
        self.class_wise_sample_scores = []
        # self.class_wise_dice_score_accumulator = (
        #     torch.empty(
        #     numerator_size,
        #     dtype=torch.long,
        #     device=self.device,
        #     )
        #     .div(0) # Divide by zero to get a tensor full of NaNs
        # )
        # self.denominator = 0#torch.zeros(numerator_size, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def step(self, y_pred: Tensor, y_true: Tensor):
        dice_scores_update = class_wise_dice_score(
            y_pred.argmax(dim=1),
            y_true,
            self.skip_bckg,
            self.n_classes,
        )
        self.class_wise_sample_scores.append(dice_scores_update)

    @torch.no_grad()
    def get_dice_score_per_class(self) -> Tensor:
        return torch.cat(self.class_wise_dice_score_accumulator).nanmean(0) #2 * self.accumulator #/ self.denominator

    @torch.no_grad()
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
