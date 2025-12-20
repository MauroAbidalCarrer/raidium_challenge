import os
from sys import argv
from tqdm import tqdm
from IPython.display import HTML

import torch
import numpy as np
import plotly.express as px
from torch import nn, Tensor
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

from src import configs as cfg
from src import dataset, models


N_SAMPLES_TO_SHOW = 10
N_CHKPT_TO_PLT = -1
NO_ARGS_MSG = """
Error: Please provide either path to:
- a model checkpoints directory.
- a cached segmentaiton .npy file.
"""


def main():
    dataset.mk_dataset()
    train_cfg = cfg.TrainingConfig(batch_size=50)
    loaders = dataset.mk_segmentation_data_loaders(train_cfg)
    x, y_true = next(iter(loaders["train"]))
    batch = dataset.preprocess_batch({"x": x, "y_true": y_true})
    if len(argv) < 2:
        print(NO_ARGS_MSG)
        exit(1)
    elif argv[1].endswith((".pt", ".pth")):
        segs = mk_segs_over_training_array(batch, argv[1])
    elif argv[1].endswith(".npy"):
        print("Making animaiton from cached segmentation npy file.")
        segs: np.ndarray = np.load(argv[1])
    mk_anim_from_segs(batch, segs)

def mk_segs_over_training_array(batch: dict[str, Tensor], chkpt_pth: str) -> np.ndarray:
    segs = (
        mk_segs_preds_over_epochs(batch, chkpt_pth)
        .detach()
        .cpu()
        .numpy()
    )
    np.save("segs.npy", segs, allow_pickle=False)
    return segs

def mk_anim_from_segs(batch: dict[str, Tensor], segs: np.ndarray) -> FuncAnimation:
    anim = mk_anim(batch, segs)
    anim.save(
        "segmentation_animation.mp4",
        fps=30,
        dpi=150,
    )

def mk_segs_preds_over_epochs(batch: dict[str, Tensor], chkpt_directory: str) -> Tensor:
    chkpt_filenames = os.listdir(chkpt_directory)[:N_CHKPT_TO_PLT]
    segs_buffer = torch.empty(
        len(chkpt_filenames), len(batch["x"]), 256, 256,
        dtype=torch.uint8,
        device=cfg.DEVICE,
    )
    with torch.no_grad():
        for chkpt_idx, chkpt_filename in enumerate(tqdm(chkpt_filenames)):
            chkpt_pth = os.path.join(chkpt_directory, chkpt_filename)
            model = load_chkpt(chkpt_pth)
            with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
                segs_buffer[chkpt_idx] = (
                    model(batch)["y_pred"]
                    .argmax(dim=1)
                    .to(dtype=torch.uint8)
                )
    return segs_buffer

def load_chkpt(chkpt_pth: str) -> torch.nn.Module:
    chkpt = torch.load(chkpt_pth, weights_only=False)
    model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
    model = models.mk_model_from_cfg(model_cfg)
    model.load_state_dict(chkpt["model"])
    model = model.eval()
    return model

def mk_anim(batch: dict[str, Tensor], segs: np.ndarray) -> FuncAnimation:
    # For segmentation labels (categorical)
    seg_norm = Normalize(
        vmin=0,
        vmax=cfg.N_CLASSES - 1,   # important: categorical, fixed range
    )

    colored_seg_buffer = plt.cm.rainbow(seg_norm(segs))  # -> [B, H, W, 4]

    # For grayscale images
    # Adjust depending on your preprocessing
    x_np = batch["x"].cpu().numpy()  # [B, H, W, C] or [B, H, W]

    x_norm = Normalize(
        vmin=x_np.min(),
        vmax=x_np.max(),
    )

    colored_x = plt.cm.gray(x_norm(x_np))  # -> [B, H, W, 4]
    seg_mask = (segs == 0)[..., None]  # background mask
    test_img_buffer = np.where(
        seg_mask,
        colored_x[None, :, 0],
        colored_seg_buffer,
    )

    # Remove alpha channel
    test_img_buffer = test_img_buffer[..., :3]  # [model_step, B, H, W, 3]


    T, B, H, W, _ = test_img_buffer.shape
    # Normalize inputs for display
    x_norm = Normalize(vmin=x_np.min(), vmax=x_np.max())
    x_gray = plt.cm.gray(x_norm(x_np))[:, :, :, :3]  # [B, H, W, 3]

    fig, axes = plt.subplots(
        2, B,
        figsize=(3 * B, 6),
        squeeze=False
    )

    for j in range(B):
        axes[0, j].imshow(x_gray[j])
        axes[0, j].set_title(f"Input {j}")
        axes[0, j].axis("off")

    seg_ims = []
    for j in range(B):
        im = axes[1, j].imshow(test_img_buffer[0, j])
        axes[1, j].set_title(f"Pred {j}")
        axes[1, j].axis("off")
        seg_ims.append(im)

    def update(t):
        for j in range(B):
            seg_ims[j].set_data(test_img_buffer[t, j])
        return seg_ims

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=30,
        blit=False,
        repeat=True,
    )
    plt.tight_layout()

    return anim


if __name__ == "__main__":
    main()