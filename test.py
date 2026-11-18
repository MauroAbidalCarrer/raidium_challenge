"""
Dice scorer correctness and performance benchmark.

Written by ChatGPT.

This script compares the project's Torch DiceScorer against the official
competition notebook's NumPy/pandas Dice implementation. It benchmarks
dataset sizes from 0 through 4,000 samples, with 224x224 images only, and
batch sizes 64 and 128. Torch runs are measured on CPU and, when available,
GPU. The NumPy/pandas reference runs on CPU.

The Torch scorer is imported from raidium_challenge.metrics. The reference
implementation below is copied from the official competition baseline
notebook so this test does not depend on metrics.dice_pandas, which may
eventually be removed.

[Official competition baseline notebook](https://colab.research.google.com/drive/1tjL_vct6dihVApl8oBVg6_6JFk-U7F5S#scrollTo=EZSMQF6uSJ4t)
"""

from __future__ import annotations

import gc
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from raidium_challenge.config_classes import N_CLASSES
from raidium_challenge.metrics import DiceScorer


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
BATCH_SIZES = (64, 128)
DATASET_SIZES = (
    0, 1, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    511, 512, 513, 999, 1000, 1024, 1500, 2048, 3000, 3999, 4000,
)
SEED = 20261118
WARMUP = True

# In this project, N_CLASSES is expected to mean foreground classes.
# The competition has 54 foreground classes plus background label 0.
TOTAL_CLASSES = N_CLASSES
NUM_CLASSES = N_CLASSES - 1  # foreground classes, excluding background
#TOTAL_CLASSES = N_CLASSES


# ---------------------------------------------------------------------------
# Official competition notebook reference implementation
# ---------------------------------------------------------------------------

#NUM_CLASSES = 54  # 55 total classes, background 0


def dice_image(prediction, ground_truth):
    intersection = np.sum(prediction * ground_truth)
    if np.sum(prediction) == 0 and np.sum(ground_truth) == 0:
        return np.nan
    return 2 * intersection / (np.sum(prediction) + np.sum(ground_truth))


def dice_multiclass(prediction, ground_truth):
    dices = []
    for i in range(1, N_CLASSES):  # skip background
        dices.append(dice_image(prediction == i, ground_truth == i))
    return np.array(dices)


def dice_pandas(y_true_df, y_pred_df):
    y_pred_df = y_pred_df.T
    y_true_df = y_true_df.T
    individual_dice = []
    for row_index in range(y_true_df.values.shape[0]):
        dices = dice_multiclass(
            y_true_df.values[row_index].ravel(),
            y_pred_df.values[row_index].ravel()
        )
        individual_dice.append(dices)
    final = np.stack(individual_dice)
    # First average over images for each class
    cls_dices = np.nanmean(final, axis=0)
    return float(np.nanmean(cls_dices))


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------

class PeakRSS:
    """Approximate peak additional process RSS, sampled while a function runs."""

    def __init__(self, interval=0.002):
        self.interval = interval
        self.peak = 0
        self.baseline = 0
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _rss():
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss
        except ImportError:
            return None

    def _sample(self):
        while not self._stop.is_set():
            current = self._rss()
            if current is not None:
                self.peak = max(self.peak, current)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline = self._rss()
        if self.baseline is None:
            return self
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        final = self._rss()
        if final is not None:
            self.peak = max(self.peak, final)

    @property
    def mib(self):
        if self.baseline is None:
            return None
        return max(0, self.peak - self.baseline) / (1024 ** 2)


def format_mib(value):
    return "n/a (install psutil)" if value is None else f"{value:,.1f} MiB"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Reproducible streaming data generation
# ---------------------------------------------------------------------------

def make_batch(start, count, device):
    """
    Generate a deterministic target and prediction batch.

    Labels are generated on CPU first so the exact same data is used for
    CPU, GPU, and NumPy runs. The returned logits use uint8 one-hot values:
    this is sufficient for argmax and avoids a large float32 logits tensor.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + start)

    shape = (count, IMAGE_SIZE, IMAGE_SIZE)
    targets_cpu = torch.randint(
        0, TOTAL_CLASSES, shape, generator=generator, dtype=torch.int64
    )

    # Make predictions mostly correct, with reproducible random corruption.
    predictions_cpu = targets_cpu.clone()
    noise = torch.rand(shape, generator=generator) < 0.15
    random_labels = torch.randint(
        0, TOTAL_CLASSES, shape, generator=generator, dtype=torch.int64
    )
    predictions_cpu[noise] = random_labels[noise]

    # The scorer consumes [B,C,H,W] logits/probabilities and takes argmax.
    logits_cpu = torch.nn.functional.one_hot(
        predictions_cpu, num_classes=TOTAL_CLASSES
    ).permute(0, 3, 1, 2).to(torch.uint8)

    return (
        logits_cpu.to(device, non_blocking=False),
        targets_cpu.to(device, non_blocking=False),
        predictions_cpu.numpy(),
        targets_cpu.numpy(),
    )


def batches(dataset_size, batch_size, device):
    for start in range(0, dataset_size, batch_size):
        count = min(batch_size, dataset_size - start)
        yield make_batch(start, count, device)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    seconds: float
    memory_mib: float | None
    score: float


def run_torch(dataset_size, batch_size, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    scorer = DiceScorer(
        n_classes=N_CLASSES,
        skip_background=True,
        device=device,
    )
    synchronize(device)
    start_time = time.perf_counter()

    if device.type == "cuda":
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        for logits, targets, _, _ in batches(dataset_size, batch_size, device):
            scorer.step(logits, targets)
        synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device) - baseline_allocated
        memory_mib = peak_bytes / (1024 ** 2)
    else:
        with PeakRSS() as memory:
            for logits, targets, _, _ in batches(dataset_size, batch_size, device):
                scorer.step(logits, targets)
        memory_mib = memory.mib

    synchronize(device)
    elapsed = time.perf_counter() - start_time

    if dataset_size == 0:
        score = float("nan")
    else:
        score = float(scorer.get_dice_score().detach().cpu())

    del scorer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return Measurement(elapsed, memory_mib, score)


def run_numpy(dataset_size, batch_size):
    gc.collect()

    # Keep only the per-image/per-class reference scores between batches.
    # dice_pandas itself is run on each batch; the final aggregation mirrors
    # the notebook's image-then-class averaging.
    all_batch_scores = []
    start_time = time.perf_counter()

    with PeakRSS() as memory:
        for start in range(0, dataset_size, batch_size):
            count = min(batch_size, dataset_size - start)
            _, _, predictions, targets = make_batch(start, count, torch.device("cpu"))

            # Official notebook expects [pixels, samples] DataFrames.
            # Suppress only NumPy's expected "Mean of empty slice" warnings
            # for classes absent from every image in a batch.
            y_true_df = pd.DataFrame(targets.reshape(count, -1).T)
            y_pred_df = pd.DataFrame(predictions.reshape(count, -1).T)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                batch_score = dice_pandas(y_true_df, y_pred_df)

            # Recreate the notebook's per-image class scores so aggregation
            # remains correct across batches (not an average of batch means).
            per_image = []
            for i in range(count):
                class_scores = dice_multiclass(
                    predictions[i].ravel(), targets[i].ravel()
                )
                per_image.append(class_scores)
            all_batch_scores.extend(per_image)

            del y_true_df, y_pred_df, predictions, targets, per_image
            _ = batch_score  # the full-dataset score is computed below

        if dataset_size:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                class_scores = np.nanmean(
                    np.asarray(all_batch_scores, dtype=np.float64), axis=0
                )
                score = float(np.nanmean(class_scores))
        else:
            score = float("nan")

    elapsed = time.perf_counter() - start_time
    return Measurement(elapsed, memory.mib, score)


def assert_scores_match(torch_score, numpy_score, context):
    if math.isnan(torch_score) and math.isnan(numpy_score):
        return
    if not math.isclose(torch_score, numpy_score, rel_tol=1e-5, abs_tol=1e-6):
        raise AssertionError(
            f"{context}: Torch score {torch_score} != NumPy score {numpy_score}"
        )


def main():
    if TOTAL_CLASSES != NUM_CLASSES + 1:
        raise ValueError(
            f"Expected N_CLASSES={NUM_CLASSES} foreground classes, so "
            f"TOTAL_CLASSES={NUM_CLASSES + 1}; got N_CLASSES={N_CLASSES}."
        )

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Foreground classes: {NUM_CLASSES}; total incl. background: {TOTAL_CLASSES}")
    print(f"Dataset sizes: {DATASET_SIZES}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Torch devices: {', '.join(map(str, devices))}")
    print("Memory is additional peak working memory; generated input batches are included.")
    print()

    header = (
        f"{'samples':>7} {'batch':>5} {'implementation':<14} "
        f"{'device':<6} {'time (s)':>11} {'peak extra':>14} {'score':>10}"
    )
    print(header)
    print("-" * len(header))

    for dataset_size in DATASET_SIZES:
        for batch_size in BATCH_SIZES:
            numpy_result = run_numpy(dataset_size, batch_size)

            for device in devices:
                torch_result = run_torch(dataset_size, batch_size, device)
                assert_scores_match(
                    torch_result.score,
                    numpy_result.score,
                    f"samples={dataset_size}, batch={batch_size}, device={device}",
                )

                print(
                    f"{dataset_size:7d} {batch_size:5d} {'Torch':<14} "
                    f"{str(device):<6} {torch_result.seconds:11.4f} "
                    f"{format_mib(torch_result.memory_mib):>14} "
                    f"{torch_result.score:10.6f}"
                )

            print(
                f"{dataset_size:7d} {batch_size:5d} {'NumPy/pandas':<14} "
                f"{'cpu':<6} {numpy_result.seconds:11.4f} "
                f"{format_mib(numpy_result.memory_mib):>14} "
                f"{numpy_result.score:10.6f}"
            )

    print("\nAll Torch-vs-official-notebook comparisons passed.")


if __name__ == "__main__":
    main()
