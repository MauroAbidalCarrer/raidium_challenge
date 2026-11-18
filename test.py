"""Compare raidium_challenge.metrics.DiceScorer with metrics.dice_pandas.

Run from the project root:
    python benchmark_dice_scorer.py

Assumes config_classes.N_CLASSES is the number of foreground classes, as in
the competition notebook (labels 1..N_CLASSES; label 0 is background).
Thus DiceScorer receives N_CLASSES + 1 total classes.

The project's dice_pandas expects [pixels, samples] arrays and transposes them.
"""
from __future__ import annotations

import gc
import math
import threading
import time
from typing import Callable

import numpy as np
import torch

from raidium_challenge.config_classes import N_CLASSES
from raidium_challenge.metrics import DiceScorer, dice_pandas

FOREGROUND_CLASSES = int(N_CLASSES)
TOTAL_CLASSES = FOREGROUND_CLASSES + 1
BATCH_SIZES = (64, 128)
HEIGHT = WIDTH = 64
REPEATS = 5
SEED = 20261118


def logits_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """[B,H,W] class IDs -> [B,C,H,W] logits with matching argmax."""
    b, h, w = labels.shape
    logits = torch.zeros(
        b, TOTAL_CLASSES, h, w, dtype=torch.float32, device=labels.device
    )
    logits.scatter_(1, labels.long().unsqueeze(1), 10.0)
    return logits


def numpy_score(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    """Call the project's dice_pandas with its [pixels,samples] layout."""
    y_true = targets.detach().cpu().numpy().reshape(targets.shape[0], -1).T
    y_pred = predictions.detach().cpu().numpy().reshape(predictions.shape[0], -1).T
    return float(dice_pandas(y_true, y_pred))


def torch_score(
    logits: torch.Tensor, targets: torch.Tensor, device: torch.device
) -> float:
    scorer = DiceScorer(
        n_classes=TOTAL_CLASSES,
        skip_background=True,
        device=device,
    )
    scorer.step(logits, targets)
    return float(scorer.get_dice_score().detach().cpu())


def sync(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(fn: Callable[[], object], device: torch.device | None) -> float:
    fn()  # warm-up
    sync(device)
    values = []
    for _ in range(REPEATS):
        gc.collect()
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        values.append((time.perf_counter() - start) * 1000)
    return float(np.median(values))


def peak_cpu_rss_mb(fn: Callable[[], object]) -> float | None:
    """Approximate additional process RSS; requires optional psutil."""
    try:
        import psutil
    except ImportError:
        fn()
        return None

    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except Exception:
                pass
            stop.wait(0.002)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        fn()
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        thread.join()
    return max(0.0, (peak[0] - baseline) / (1024**2))


def peak_cuda_allocated_mb(fn: Callable[[], object], device: torch.device) -> float:
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    fn()
    sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    return max(0.0, (peak - baseline) / (1024**2))


def make_cases(
    b: int, h: int, w: int, generator: torch.Generator
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Each case returns (ground_truth, predicted_labels), both [B,H,W]."""
    target = torch.randint(0, TOTAL_CLASSES, (b, h, w), generator=generator)
    random_pred = torch.randint(0, TOTAL_CLASSES, (b, h, w), generator=generator)

    noisy = target.clone()
    mask = torch.rand((b, h, w), generator=generator) < 0.20
    replacement = torch.randint(0, TOTAL_CLASSES, (b, h, w), generator=generator)
    noisy[mask] = replacement[mask]

    bg = torch.zeros((b, h, w), dtype=torch.long)
    fg_pred = torch.ones_like(bg)

    sparse_gt = torch.zeros_like(bg)
    sparse_pred = torch.zeros_like(bg)
    for i in range(b):
        cls = 1 + (i % FOREGROUND_CLASSES)
        wrong_cls = 1 + (cls % FOREGROUND_CLASSES)
        sparse_gt[i, :h // 2, :w // 2] = cls
        sparse_pred[i, :h // 4, :w // 2] = wrong_cls
        sparse_pred[i, h // 4:h // 2, :w // 2] = cls

    return {
        "perfect": (target, target.clone()),
        "random": (target, random_pred),
        "20_percent_noise": (target, noisy),
        "all_background": (bg, bg.clone()),
        "background_GT_foreground_pred": (bg, fg_pred),
        "sparse_classes_per_image": (sparse_gt, sparse_pred),
    }


def assert_same(a: float, b: float, context: str) -> None:
    if math.isnan(a) and math.isnan(b):
        return
    if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"{context}: Torch={a}, dice_pandas={b}")


def run_one(
    case: str,
    targets_cpu: torch.Tensor,
    predictions_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> None:
    # Device transfer and input construction are deliberately outside timings.
    targets = targets_cpu.to(device)
    predictions = predictions_cpu.to(device)
    logits = logits_from_labels(predictions)

    reference = numpy_score(targets_cpu, predictions_cpu)

    def run_torch() -> float:
        return torch_score(logits, targets, device)

    actual = run_torch()
    assert_same(actual, reference, f"{case}, B={batch_size}, {device}")

    if device.type == "cuda":
        torch_mem = peak_cuda_allocated_mb(run_torch, device)
    else:
        torch_mem = peak_cpu_rss_mb(run_torch)

    torch_time = timed_ms(run_torch, device)

    print(
        f"{case:<29} B={batch_size:<3} Torch/{device.type:<4} "
        f"{torch_time:9.2f} ms | "
        f"{'n/a' if torch_mem is None else f'{torch_mem:.1f} MB'}"
    )

    if device.type == "cpu":
        def run_numpy() -> float:
            return numpy_score(targets_cpu, predictions_cpu)

        numpy_mem = peak_cpu_rss_mb(run_numpy)
        numpy_time = timed_ms(run_numpy, None)
        print(
            f"{'':29} {'':6} NumPy/CPU   "
            f"{numpy_time:9.2f} ms | "
            f"{'n/a' if numpy_mem is None else f'{numpy_mem:.1f} MB'}"
        )


def main() -> None:
    print("DiceScorer vs raidium_challenge.metrics.dice_pandas")
    print(
        f"Foreground classes={FOREGROUND_CLASSES}, total classes={TOTAL_CLASSES}; "
        f"image={HEIGHT}x{WIDTH}; batches={BATCH_SIZES}; repeats={REPEATS}"
    )
    print("Memory = approximate additional peak RSS (CPU) or CUDA allocated memory.")
    print("Install psutil for CPU RSS measurements: pip install psutil")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available; skipping GPU Torch runs.")

    for b in BATCH_SIZES:
        generator = torch.Generator().manual_seed(SEED + b)
        cases = make_cases(b, HEIGHT, WIDTH, generator)
        print(f"\\n--- Batch size {b} ---")
        for case, (targets, predictions) in cases.items():
            for device in devices:
                run_one(case, targets, predictions, device, b)

    print("\\nAll Torch results matched dice_pandas.")


if __name__ == "__main__":
    main()
