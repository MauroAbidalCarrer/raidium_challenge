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
    import numpy as np
    import torch.nn.functional as F

    from raidium_challenge import dataset

    # ------------------------------------------------------------------
    # NumPy reference: mirrors the competition demo notebook.
    #
    # n_classes is the TOTAL number of labels, including background 0.
    # Therefore the foreground classes are 1, ..., n_classes - 1.
    # ------------------------------------------------------------------

    def dice_pandas_reference(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int,
    ) -> tuple[float, np.ndarray]:
        """
        Competition-style Dice reference.

        Args:
            y_true: [B, H, W] integer label maps.
            y_pred: [B, H, W] integer label maps.
            n_classes: Total classes, including background.

        Returns:
            final_score: scalar competition score.
            class_scores: [n_classes - 1], averaged over images.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )

        per_image_class_scores = []

        for image_idx in range(y_true.shape[0]):
            image_scores = []

            for class_idx in range(1, n_classes):  # skip background 0
                pred_mask = y_pred[image_idx] == class_idx  # [H, W]
                true_mask = y_true[image_idx] == class_idx  # [H, W]

                denominator = pred_mask.sum() + true_mask.sum()

                if denominator == 0:
                    image_scores.append(np.nan)
                else:
                    intersection = np.logical_and(
                        pred_mask, true_mask
                    ).sum()
                    image_scores.append(
                        2.0 * intersection / denominator
                    )

            per_image_class_scores.append(image_scores)

        # [B, n_classes - 1]
        per_image_class_scores = np.asarray(
            per_image_class_scores, dtype=np.float64
        )

        # First average over images for each class: [n_classes - 1]
        class_scores = np.nanmean(per_image_class_scores, axis=0)

        # Then average over classes: scalar
        final_score = float(np.nanmean(class_scores))

        return final_score, class_scores

    def score_with_torch(
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_classes: int,
        batch_size: int | None = None,
    ) -> tuple[float, torch.Tensor]:
        """
        Run DiceScorer, optionally splitting the input into batches.

        logits:  [B, C, H, W]
        targets: [B, H, W]
        """
        scorer = DiceScorer(
            n_classes=n_classes,
            skip_background=True,
            device=logits.device,
        )

        if batch_size is None:
            batch_size = logits.shape[0]

        for start in range(0, logits.shape[0], batch_size):
            end = start + batch_size
            scorer.step(logits[start:end], targets[start:end])

        class_scores = scorer.get_dice_score_per_class()
        final_score = scorer.get_dice_score()

        return float(final_score.cpu()), class_scores.cpu()

    def logits_from_labels(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Create unambiguous logits from labels [B, H, W] -> [B, C, H, W]."""
        return (
            F.one_hot(labels.long(), num_classes=n_classes)
            .permute(0, 3, 1, 2)
            .float() * 10.0
        )

    def compare_case(
        name: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_classes: int,
        batch_size: int | None = None,
    ) -> None:
        """Compare Torch implementation against the NumPy notebook reference."""
        pred_labels = logits.argmax(dim=1)

        expected_score, expected_classes = dice_pandas_reference(
            targets.cpu().numpy(),
            pred_labels.cpu().numpy(),
            n_classes,
        )

        actual_score, actual_classes = score_with_torch(
            logits, targets, n_classes, batch_size=batch_size
        )

        np.testing.assert_allclose(
            actual_classes.numpy(),
            expected_classes,
            rtol=1e-6,
            atol=1e-7,
            equal_nan=True,
            err_msg=f"{name}: per-class scores differ",
        )
        np.testing.assert_allclose(
            actual_score,
            expected_score,
            rtol=1e-6,
            atol=1e-7,
            equal_nan=True,
            err_msg=f"{name}: final score differs",
        )

        print(
            f"[PASS] {name:<38} "
            f"score={actual_score:.8f} "
            f"(batch_size={batch_size})"
        )

    def run_synthetic_tests(n_classes: int) -> None:
        print("\n=== Synthetic tests ===")

        generator = torch.Generator().manual_seed(12345)
        batch_size, height, width = 7, 32, 29

        # Case 1: perfect predictions.
        targets = torch.randint(
            0, n_classes, (batch_size, height, width),
            generator=generator,
        )
        compare_case(
            "perfect predictions",
            logits_from_labels(targets, n_classes),
            targets,
            n_classes,
        )

        # Case 2: random predictions.
        random_logits = torch.randn(
            batch_size, n_classes, height, width,
            generator=generator,
        )
        compare_case(
            "random predictions",
            random_logits,
            targets,
            n_classes,
        )

        # Case 3: deliberately corrupted predictions.
        corrupted = targets.clone()
        noise_mask = torch.rand(
            corrupted.shape, generator=generator
        ) < 0.25
        random_labels = torch.randint(
            0, n_classes, corrupted.shape, generator=generator
        )
        corrupted[noise_mask] = random_labels[noise_mask]
        compare_case(
            "25% randomly corrupted labels",
            logits_from_labels(corrupted, n_classes),
            targets,
            n_classes,
        )

        # Case 4: all-background ground truth and prediction.
        all_background = torch.zeros(
            3, height, width, dtype=torch.long
        )
        compare_case(
            "all-background perfect",
            logits_from_labels(all_background, n_classes),
            all_background,
            n_classes,
        )

        # Case 5: all-background ground truth, foreground-only prediction.
        foreground_prediction = torch.ones_like(all_background)
        compare_case(
            "background GT, foreground prediction",
            logits_from_labels(foreground_prediction, n_classes),
            all_background,
            n_classes,
        )

        # Case 6: a foreground class is present in GT but never predicted.
        gt_one_class = torch.zeros(2, height, width, dtype=torch.long)
        gt_one_class[:, : height // 2, :] = 1
        pred_background = torch.zeros_like(gt_one_class)
        compare_case(
            "foreground GT, missed completely",
            logits_from_labels(pred_background, n_classes),
            gt_one_class,
            n_classes,
        )

        # Case 7: a class is predicted but absent from GT.
        gt_background = torch.zeros(2, height, width, dtype=torch.long)
        pred_class = torch.full_like(gt_background, 2)
        compare_case(
            "false-positive-only class",
            logits_from_labels(pred_class, n_classes),
            gt_background,
            n_classes,
        )

        # Case 8: images have different classes present.
        # This checks that class counts are kept separate per image.
        different_classes = torch.zeros(
            3, height, width, dtype=torch.long
        )
        different_classes[0, :, : width // 2] = 1
        different_classes[1, :, : width // 2] = 2
        different_classes[2, :, : width // 2] = 3
        different_predictions = different_classes.clone()
        different_predictions[1, : height // 3, :] = 1
        compare_case(
            "different classes across images",
            logits_from_labels(different_predictions, n_classes),
            different_classes,
            n_classes,
        )

        # Case 9: one image at a time.
        compare_case(
            "single-image batch",
            random_logits[:1],
            targets[:1],
            n_classes,
        )

        # Case 10: same dataset split into several batch sizes.
        for split_size in (1, 2, 3, batch_size):
            compare_case(
                f"batch partition size {split_size}",
                random_logits,
                targets,
                n_classes,
                batch_size=split_size,
            )

    def run_real_dataset_tests(n_classes: int) -> None:
        print("\n=== Real dataset tests ===")

        x_train, y_train, _ = dataset.load_raw_dataset("cpu")
        x_train, y_train = dataset.remove_samples_without_labels(
            x_train, y_train
        )

        # Test more than one dataset size, where available.
        available = y_train.shape[0]
        sample_counts = sorted(set(
            count for count in (1, 2, 7, 20, 50, available)
            if count <= available
        ))

        for sample_count in sample_counts:
            targets = y_train[:sample_count].long()

            min_label = int(targets.min())
            max_label = int(targets.max())
            if min_label < 0 or max_label >= n_classes:
                raise ValueError(
                    "Dataset labels do not fit the configured class count: "
                    f"observed [{min_label}, {max_label}], "
                    f"expected [0, {n_classes - 1}]. "
                    "Check whether N_CLASSES includes the background class."
                )

            batch, height, width = targets.shape

            # Perfect prediction.
            compare_case(
                f"real data: {sample_count} perfect images",
                logits_from_labels(targets, n_classes),
                targets,
                n_classes,
            )

            # Random logits, deterministic for reproducibility.
            generator = torch.Generator().manual_seed(1000 + sample_count)
            random_logits = torch.randn(
                batch, n_classes, height, width,
                generator=generator,
            )
            compare_case(
                f"real data: {sample_count} random images",
                random_logits,
                targets,
                n_classes,
                batch_size=min(8, sample_count),
            )

            # Perturb a perfect prediction while keeping the target fixed.
            perturbed_logits = logits_from_labels(targets, n_classes)
            perturbed_logits += 12.0 * torch.rand(
                perturbed_logits.shape,
                generator=generator,
            )
            compare_case(
                f"real data: {sample_count} perturbed images",
                perturbed_logits,
                targets,
                n_classes,
                batch_size=min(8, sample_count),
            )

    print(f"Configured N_CLASSES (including background): {N_CLASSES}")
    print(f"Foreground classes tested: {N_CLASSES - 1}")

    # The competition demo uses 54 foreground classes + background = 55.
    if N_CLASSES != 55:
        print(
            "WARNING: The demo notebook specifies 55 total classes "
            "(background 0 + foreground 1..54), but config N_CLASSES "
            f"is {N_CLASSES}. Verify the project's class-count convention."
        )

    run_synthetic_tests(N_CLASSES)
    run_real_dataset_tests(N_CLASSES)

    print("\nAll Torch-vs-NumPy comparisons passed.")
