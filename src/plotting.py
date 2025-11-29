from typing import Optional

import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt


def plot_slice_seg(slice_image, *seg_masks):
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
    image_test = x[sample_idx:sample_idx + 1]
    model_device = next(model.parameters()).device
    label_test = model(image_test.to(model_device))
    pred_test = torch.argmax(label_test, dim=1)
    if y is not None:
        plot_slice_seg(
            image_test.squeeze(),
            pred_test.squeeze().cpu().numpy(),
            y[sample_idx],
        )
    else:
        plot_slice_seg(
            image_test.squeeze(),
            pred_test.squeeze().cpu().numpy(),
        )
