from typing import Optional

import torch
import numpy as np
from torch import nn, Tensor
import matplotlib.pyplot as plt

from src import dataset


N_IMGS_TO_PLT = 5


def plt_sample(slice_image: Tensor, *seg_masks):
    fig, axes = plt.subplots(1, 1 + len(seg_masks), squeeze=False)
    fig = fig.set_size_inches(10, 10)
    axes[0, 0].imshow(slice_image, cmap="gray")
    for seg_i, seg_mask in enumerate(seg_masks):
        axes[0, seg_i + 1].imshow(slice_image, cmap="gray")
        seg_masked = np.ma.masked_where(seg_mask == 0, seg_mask)
        axes[0, seg_i + 1].imshow(seg_masked, cmap="tab20")
    plt.axis("off")
    plt.show()
    
@torch.no_grad
def plt_pred(
        model: nn.Module,
        sample_idx: int,
        x: Tensor,
        y: Optional[Tensor] = None,
    ):
    sample = x[sample_idx:sample_idx + 1]
    model_device = next(model.parameters()).device
    y_pred_logits = model(sample.to(model_device))
    y_pred = torch.argmax(y_pred_logits, dim=1)
    if y is not None:
        plt_sample(
            sample.squeeze().cpu().numpy(),
            y_pred.squeeze().cpu().numpy(),
            y[sample_idx].cpu().numpy(),
        )
    else:
        plt_sample(
            sample.squeeze().cpu().numpy(),
            y_pred.squeeze().cpu().numpy(),
        )

@torch.no_grad
def plt_model_preds(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        plt_recon: bool=True,
    ):
    """Plots the model's reconstruction and segmentation."""
    model.eval()
    x, y_true = next(iter(data_loader))
    sample_has_seg = y_true.flatten(1).__ne__(0).any(dim=1)
    print("n samples with seg:", sample_has_seg.sum(), sample_has_seg.shape)
    if sample_has_seg.any():
        x = x[sample_has_seg]
        y_true = y_true[sample_has_seg]
    x = dataset.preprocess_imgs(x)
    x = x[:N_IMGS_TO_PLT]
    x_hat, mask = model(x)
    x_hat_img = x_hat[:, :1]
    mask = mask[:, :1]
    x_hat_img = x_hat_img * mask + x * (1 - mask)
    plt_seg_imgs(x, y_true, x_hat[:, 1:], mask)
    if plt_recon:
        plt_recon_imgs(x, x_hat_img, mask)

def plt_recon_imgs(
        x: Tensor,
        x_hat_img: Tensor,
        mask: Tensor,
        n_cols: int = 8,
        n_imgs_to_plt: int=N_IMGS_TO_PLT,
    ):
    """
    Matplotlib version of the reconstruction visualization.
    Shows reconstructed images and originals in one grid.
    """
    # Combine images along batch dimension
    img = torch.cat(
        [
            # x * (1 - mask),   # optional: masked input
            x_hat_img,
            x,
        ],
        dim=0,
    )
    # unnormalize
    img = img * dataset.STD + dataset.MEAN  
    # B, C, H, W → B, H, W
    np_imgs = img.detach().cpu().numpy().squeeze()
    n_imgs = np_imgs.shape[0]
    n_rows = (n_imgs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        2, n_imgs_to_plt,
        figsize=(n_cols * 2, n_rows * 2),
        squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        if i < n_imgs:
            ax.imshow(np_imgs[i], cmap="rainbow")
            ax.set_title(f"{i}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plt_seg_imgs(
        x: Tensor,
        y_true: Tensor,
        y_pred: Tensor,
        mask: Tensor,
        n_imgs_to_plt: int=N_IMGS_TO_PLT
    ):
    x = x * (1 - mask)
    for i in range(min(n_imgs_to_plt, len(x))):

        plt_sample(
            x[i].squeeze().detach().cpu().numpy(),
            y_pred[i].squeeze().detach().cpu().argmax(dim=0).numpy(),
            y_true[i].squeeze().detach().cpu().numpy(),
        )
