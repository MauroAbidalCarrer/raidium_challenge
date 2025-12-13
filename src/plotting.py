from typing import Optional, Any

import torch
import numpy as np
import plotly.express as px
from torch import nn, Tensor
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from src import dataset


MAX_N_SAMPLES_TO_PLOT = 5

    
@torch.no_grad
def plt_seg_pred(
        model: nn.Module,
        batch: dict[str, Any],
        transform: Optional[v2.Transform] = v2.Identity(),
        ax_size: Optional[int] = 4,
    ):
    batch = dataset.preprocess_batch(batch)
    batch_has_y_true = "y_true" in batch
    if batch_has_y_true:
        batch["x"], batch["y_true"] = transform(batch["x"], batch["y_true"])
    else:
        batch["x"] = transform(batch["x"])
    model_output_dict = model(batch)
    batch["y_pred"] = torch.argmax(model_output_dict["y_pred"], dim=1)
    seg_masks_keys = ("y_pred", "y_true") if batch_has_y_true else ("y_pred")
    n_samples_to_plt = min(len(batch["x"]), MAX_N_SAMPLES_TO_PLOT)
    n_rows = 2 + int(batch_has_y_true)
    batch["x"] = batch["x"].detach().cpu().numpy().squeeze()
    batch["y_pred"] = batch["y_pred"].detach().cpu().numpy()
    if batch_has_y_true:
        batch["y_true"] = batch["y_true"].detach().cpu().numpy()
    fig, axes = plt.subplots(n_rows, n_samples_to_plt, squeeze=False)
    fig.set_size_inches(n_samples_to_plt * ax_size, n_rows * ax_size)
    for seg_mask_key in seg_masks_keys:
        batch[seg_mask_key] = batch[seg_mask_key][:n_samples_to_plt]
    for j in range(n_samples_to_plt):
        axes[0, j].imshow(
            batch["x"][j],
            cmap="gray",
        )
        imshow_seg(axes[1, j], batch["x"][j], batch["y_pred"][j])
        if batch_has_y_true:
            imshow_seg(axes[2, j], batch["x"][j], batch["y_true"][j])

def imshow_seg(ax, x: Tensor, seg: Tensor):
    ax.imshow(x, cmap="gray")
    seg_masked = np.ma.masked_where(seg == 0, seg)
    ax.imshow(seg_masked, cmap="tab20")

def plt_recon_imgs_matplotlib(
        model: nn.Module,
        batch: dict[str, Tensor],
        transform: Optional[v2.Transform] = v2.Identity(),
        n_imgs_to_plt: int=MAX_N_SAMPLES_TO_PLOT,
        ax_size: int = 5,
    ):
    """
    Matplotlib version of the reconstruction visualization.
    Shows reconstructed images and originals in one grid.
    """
    batch = dataset.preprocess_batch(batch)
    batch["x"] = transform(batch["x"])
    model_output = model(batch)
    x = batch["x"]
    mask = model_output["mask"][:, :1]
    x_hat = model_output["x_hat"]
    masked_x = x * (1 - mask)
    x_hat = x_hat * mask + x * (1 - mask)
    # B, C, H, W → B, H, W
    x = x.detach().cpu().numpy().squeeze()[:n_imgs_to_plt]
    masked_x = masked_x.detach().cpu().numpy().squeeze()[:n_imgs_to_plt]
    x_hat = x_hat.detach().cpu().numpy().squeeze()[:n_imgs_to_plt]
    fig, axes = plt.subplots(
        nrows=3,
        ncols=x.shape[0],
        figsize=(x.shape[0] * ax_size, 3 * ax_size),
        squeeze=False
    )
    for sample_i, (x_samp, masked_x_samp, x_hat_samp) in enumerate(zip(x, masked_x, x_hat)):
        axes[0, sample_i].imshow(x_samp, cmap="rainbow")
        axes[1, sample_i].imshow(masked_x_samp, cmap="rainbow")
        axes[2, sample_i].imshow(x_hat_samp, cmap="rainbow")
    plt.tight_layout()
    plt.show()


def plt_recon_imgs_px(model: nn.Module, loader: DataLoader):
    x, y_true = next(iter(loader))
    x = dataset.preprocess_imgs(x)[:MAX_N_SAMPLES_TO_PLOT]
    x_hat, mask = model(x, 0.65)
    x_hat_img = x_hat[:, :1]
    mask = mask[:, :1]
    x_hat_img = x_hat_img * mask + x * (1 - mask)
    img = torch.cat(
        [
            x * (1 - mask),
            x_hat_img,
            x,
        ],
        dim=0,
    )
    # img = img * dataset.STD + dataset.MEAN 
    np_imgs = (
        img
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )
    fig = px.imshow(
        np_imgs,
        facet_col=0,
        facet_col_wrap=MAX_N_SAMPLES_TO_PLOT,
        color_continuous_scale="rainbow",
    )
    fig.show()
