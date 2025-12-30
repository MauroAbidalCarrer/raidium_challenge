import os
import argparse

import torch
import numpy as np
from torch import Tensor
from rich.progress import track
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

from src import configs as cfg
from src import dataset, models


MK_SEGS_TRACK_DESC = "Making segmentation preds over epochs"
N_SAMPLES_TO_SHOW = 10
N_CHKPT_TO_PLT = -1


def main():
    args = parse_args()
    dataset.mk_dataset()
    vid_cache_dir = args.vid_cache or "train_vid_cache"
    
    if args.checkpoints is not None:
        batch = mk_preprocessed_batch(50)
        segs = mk_segs_preds_over_epochs(batch, args.checkpoints)
        save_segs_and_batch(batch, segs)
    elif args.vid_cache is not None:
        print("Making animation from cached segmentation.")
        segs = np.load(f"{vid_cache_dir}/segs.npy")
        batch = torch.load(f"{vid_cache_dir}/batch.pt")
    anim = mk_anim(batch, segs)
    anim_path = os.path.join(vid_cache_dir, "segmentation_animation.mp4")
    os.makedirs(vid_cache_dir, exist_ok=True)
    anim.save(anim_path, fps=30, dpi=150)
    print(f"Saved animation at {anim_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation visualization")
    parser.add_argument(
        "--checkpoints",
        type=str,
        help="Path to model checkpoint (.pt or .pth)",
    )
    parser.add_argument(
        "--vid_cache",
        type=str,
        help="Path to cached video directory (contains segs.npy and batch.pt)",
    )
    args = parser.parse_args()
    if args.checkpoints is None and args.vid_cache is None:
        parser.error("You must specify either --checkpoints or --vid_cache")
    if args.checkpoints is not None and args.vid_cache is not None:
        parser.error("Please specify only one of --checkpoints or --vid_cache")
    return args

def mk_preprocessed_batch(batch_size: int) -> dict[str, Tensor]:
    train_cfg = cfg.TrainingConfig(batch_size=50)
    loaders = dataset.mk_segmentation_data_loaders(train_cfg)
    x, y_true = next(iter(loaders["train"]))
    batch = dataset.preprocess_batch({"x": x, "y_true": y_true})
    return batch

def save_segs_and_batch(batch: dict[str, Tensor], segs: np.ndarray):
    os.makedirs("video_cache", exist_ok=True)
    batch = {k: v.detach().cpu() for k, v in batch.items()}
    torch.save(batch, "video_cache/batch.pt")
    np.save("video_cache/segs.npy", segs, allow_pickle=False)
    print("Saved segmentations and batch in vid_cache directory.")

def mk_segs_preds_over_epochs(batch: dict[str, Tensor], chkpt_directory: str) -> Tensor:
    chkpt_filenames = os.listdir(chkpt_directory)[:N_CHKPT_TO_PLT]
    segs_buffer = torch.empty(
        len(chkpt_filenames), len(batch["x"]), 256, 256,
        dtype=torch.uint8,
        device=cfg.DEVICE,
    )
    with torch.no_grad():
        ckpt_it = track(chkpt_filenames, MK_SEGS_TRACK_DESC)
        for chkpt_idx, chkpt_filename in enumerate(ckpt_it):
            chkpt_pth = os.path.join(chkpt_directory, chkpt_filename)
            model = load_chkpt(chkpt_pth)
            with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
                segs_buffer[chkpt_idx] = (
                    model(batch)["y_pred"]
                    .argmax(dim=1)
                    .to(dtype=torch.uint8)
                )
    return (
        segs_buffer
        .detach()
        .cpu()
        .numpy()
    )

def load_chkpt(chkpt_pth: str) -> torch.nn.Module:
    chkpt = torch.load(chkpt_pth, weights_only=False)
    model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
    model = models.mk_model_from_cfg(model_cfg)
    model.load_state_dict(chkpt["model"])
    model = model.eval()
    return model

# This is a really shitty function
def mk_anim(
        batch: dict[str, Tensor],
        segs: np.ndarray,
        n_samples_to_plt: int = 6,
    ) -> FuncAnimation:
    # Keep the first n_samples_to_plt
    x = (
        batch["x"]
        .cpu()
        .numpy()
        [:n_samples_to_plt]
    )
    y_true = (
        batch["y_true"]
        .cpu()
        .numpy()
        [:n_samples_to_plt]
    )
    segs = segs[:, :n_samples_to_plt]
    # For segmentation labels (categorical)
    y_norm = Normalize(vmin=0, vmax=cfg.N_CLASSES - 1)
    x_norm = Normalize(x.min(), x.max())
    
    colored_y_pred = plt.cm.tab20(y_norm(segs), bytes=True)  # -> [B, H, W, 4]
    colored_y_true = plt.cm.tab20(y_norm(y_true), bytes=True)
    colored_x = plt.cm.gray(x_norm(x), bytes=True)  # -> [B, H, W, 4]
    # Put the seg on top of the 
    seg_mask = (segs == 0)[..., None]  # background mask
    y_pred_imgs = np.where(
        seg_mask,
        colored_x[None, :, 0],
        colored_y_pred,
    )
    seg_mask = (y_true == 0)[..., None]  # background mask
    y_true_imgs = np.where(
        seg_mask,
        colored_x[None, :, 0],
        colored_y_true,
    )

    # Remove alpha channel
    y_pred_imgs = y_pred_imgs[..., :3]  # [model_step, B, H, W, 3]

    n_frames, batch_size, *_ = y_pred_imgs.shape

    fig, axes = plt.subplots(
        3, batch_size,
        figsize=(3 * batch_size, 6),
        squeeze=False
    )

    seg_ims = []
    for j in range(batch_size):
        axes[0, j].imshow(colored_x[j, 0])
        axes[0, j].set_title(f"Input {j}")
        axes[0, j].axis("off")
        im = axes[1, j].imshow(y_pred_imgs[0, j])
        axes[1, j].set_title(f"Pred {j}")
        axes[1, j].axis("off")
        seg_ims.append(im)
        axes[2, j].imshow(y_true_imgs[0, j])
        axes[2, j].axis("off")

    def update(t):
        for j in range(batch_size):
            seg_ims[j].set_data(y_pred_imgs[t, j])
        return seg_ims

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=30,
        blit=False,
        repeat=True,
    )
    plt.tight_layout()

    return anim


if __name__ == "__main__":
    main()