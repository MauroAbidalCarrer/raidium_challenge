from typing import Optional

import torch
import numpy as np
from torch import nn, Tensor
import matplotlib.pyplot as plt


def plt_sample(slice_image: Tensor, *seg_masks):
    fig, axes = plt.subplots(1, 1 + len(seg_masks))
    fig = fig.set_size_inches(15, 15)
    axes[0].imshow(slice_image, cmap="gray")
    for seg_i, seg_mask in enumerate(seg_masks):
        axes[seg_i + 1].imshow(slice_image, cmap="gray")
        seg_masked = np.ma.masked_where(seg_mask.reshape((256,256)) == 0, (seg_mask.reshape((256,256))))
        axes[seg_i + 1].imshow(seg_masked, cmap="tab20")
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
    print(sample.shape)
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
