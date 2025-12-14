import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Optional

import torch
import requests
import torchvision
import numpy as np
import pandas as pd
from torch import Tensor
from rich.progress import track
from torchvision import tv_tensors
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
    Sampler,
)

import src.configs as cfg
from src.configs import TrainingConfig, DEVICE
from sklearn.model_selection import train_test_split


def mk_segmentation_data_loaders(
        train_cfg: cfg.TrainingConfig
    ) -> dict[str, DataLoader]:
    x_train, y_train, x_test = load_raw_dataset(cfg.DEVICE)
    x_train, y_train = remove_samples_without_labels(x_train, y_train)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )
    if hasattr(train_cfg, "sampling"):
        print("sampling method:", train_cfg.sampling)
        if train_cfg.sampling == "uniform":
            train_dl_kwargs = {"batch_sampler": UniformBatchSampler(y_train, train_cfg)}
        elif train_cfg.sampling == "weighted":
            samples_weights = mk_samples_weights(y_train)
            train_sampler = WeightedRandomSampler(
                samples_weights,
                num_samples=len(samples_weights),
                replacement=True,
            )
            train_dl_kwargs = {"sampler": train_sampler}
        elif train_cfg.sampling == "shuffle":
            train_dl_kwargs = {"shuffle": True}
        else:
            raise ValueError(f"Unrecognized training config sampling: {train_cfg.sampling}")
    else:
        print("WARNING: No sampling attribute found in train config, using base data loader with shuffle=True.")
        train_dl_kwargs = {"shuffle": True}
    print("train_dl_kwargs:", train_dl_kwargs)
    y_test_fill = torch.zeros(len(x_test), 256, 256, device=cfg.DEVICE)
    return {
        "train": mk_dl_from_tensors(x_train, y_train, **train_dl_kwargs),
        "valid": mk_dl_from_tensors(x_valid, y_valid, batch_size=train_cfg.batch_size),
        "test":  mk_dl_from_tensors(x_test, y_test_fill, batch_size=train_cfg.batch_size),
    }

def mk_dl_from_tensors(*tensors: list[Tensor], **data_loader_kwargs) -> DataLoader:
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, **data_loader_kwargs)

def mk_samples_weights(y_train_with_labels: Tensor) -> Tensor:
    """Takes in x and y with removed samples without labels and returns per sample weights."""
    y_classes = cls_presence_mask(y_train_with_labels)
    class_counts = y_classes.sum(dim=0, keepdim=True)
    cls_weight = 1 / (class_counts + 1e-8)
    sample_weights = (y_classes * cls_weight).sum(dim=1)
    return sample_weights

class UniformBatchSampler(Sampler):
    def __init__(self, y_train: Tensor, train_cfg: cfg.TrainingConfig):
        assert cfg.N_CLASSES <= train_cfg.batch_size, "Batch size should be less or equal to n classes when using unfiorm sampler."
        super().__init__()
        self.train_cfg = train_cfg
        self.n_samples = y_train.shape[0]
        quotient = self.n_samples // self.train_cfg.batch_size
        remainder = self.n_samples % self.train_cfg.batch_size
        self.n_batches = quotient + min(1, remainder)
        self.init_samples_per_cls_idx(y_train)
        self.batch_idx_buff = torch.empty(
            self.train_cfg.batch_size,
            dtype=torch.long,
            device=cfg.DEVICE,
        )
        self.arange_buff = torch.arange(
            cfg.N_CLASSES,
            dtype=torch.long,
            device=cfg.DEVICE,
        )

    def init_samples_per_cls_idx(self, y_train: Tensor) -> Tensor:
        cls_mask = cls_presence_mask(y_train)
        samples_idx_per_classes_lst: list[Tensor] = [torch.nonzero(cls_present).squeeze() for cls_present in cls_mask.T]
        self.max_n_samples_in_cls = max(map(len, samples_idx_per_classes_lst))
        self.samples_per_classes_t = torch.empty(
            self.max_n_samples_in_cls, cfg.N_CLASSES,
            dtype=torch.long,
            device=cfg.DEVICE,
        )
        for cls_idx, cls_samples_idx in enumerate(samples_idx_per_classes_lst):
            quotient  = self.max_n_samples_in_cls // len(cls_samples_idx)
            remainder = self.max_n_samples_in_cls % len(cls_samples_idx)
            n_repeats = quotient + min(1, remainder)
            repeated_samples = cls_samples_idx.repeat(n_repeats)
            self.samples_per_classes_t[:, cls_idx] = repeated_samples[:self.max_n_samples_in_cls]

    def __iter__(self):
        for _ in range(self.n_batches):
            uniform_rand_samp_idx = torch.randint(
                low=0,
                high=self.max_n_samples_in_cls,
                size=(cfg.N_CLASSES, ),
                dtype=torch.long,
                device=cfg.DEVICE,
            )
            # Should use torch.gather 
            uniform_rand_samp_idx = self.samples_per_classes_t[uniform_rand_samp_idx, self.arange_buff]
            self.batch_idx_buff[:cfg.N_CLASSES] = uniform_rand_samp_idx
            # fill the rest with random classes
            n_remaining_samples = self.train_cfg.batch_size - cfg.N_CLASSES
            uniform_rand_samp_idx = torch.randint(
                low=0,
                high=self.max_n_samples_in_cls,
                size=(n_remaining_samples, ),
                dtype=torch.long,
                device=cfg.DEVICE,
            )
            uniform_rand_cls_idx = torch.randint(
                low=0,
                high=cfg.N_CLASSES,
                size=(n_remaining_samples, ),
                dtype=torch.long,
                device=cfg.DEVICE,
            )
            remaining_unifor_samp_idx = self.samples_per_classes_t[uniform_rand_samp_idx, uniform_rand_cls_idx]
            self.batch_idx_buff[cfg.N_CLASSES:] = remaining_unifor_samp_idx
            permute_idx = torch.randperm(len(self.batch_idx_buff))
            yield self.batch_idx_buff[permute_idx]

    def __len__(self) -> int:
        return self.n_batches

def cls_presence_mask(y_train: Tensor) -> Tensor:
    cls_presence = torch.empty(y_train.shape[0], cfg.N_CLASSES)
    for class_idx in range(0, cfg.N_CLASSES):
        cls_presence[:, class_idx] = (y_train == class_idx).any(dim=(1, 2))
    return cls_presence

def mk_semi_supervised_data_loaders(train_cfg: TrainingConfig) -> dict[str, DataLoader]:
    x_train, y_train, x_test = load_raw_dataset(DEVICE)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )
    x_train = torch.cat((x_train, x_test))
    y_test_zeros = torch.zeros(
        x_test.shape[0], 256, 256,
        dtype=torch.uint8,
        device=y_train.device,
    )

    y_train = torch.cat((y_train, y_test_zeros))
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)
    test_dataset  = TensorDataset(x_test, y_test_zeros)
    train_loader = DataLoader(train_dataset, train_cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, train_cfg.batch_size)
    test_loader = DataLoader(test_dataset, train_cfg.batch_size)
    return {
        "train": train_loader,
        "valid": valid_loader,
        "test":  test_loader,
    }

def preprocess_batch(batch_dict: dict[str, Any]) -> dict[str, Any]:
    x = batch_dict["x"]
    x = x.to(device=DEVICE, dtype=torch.float)
    batch_dict["x"] = (x - cfg.X_MEAN) / cfg.X_STD
    if "y_true" in batch_dict:
        batch_dict["y_true"] = (
            tv_tensors.Mask(batch_dict["y_true"])
            .to(device=cfg.DEVICE, dtype=torch.long)
        )
    return batch_dict

def load_raw_dataset(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    y_train: Tensor = torch.load("dataset/formatted/y-train.pt").to(device=device, dtype=torch.uint8)
    x_train = torch.load("dataset/formatted/x-train.pt").to(device=device, dtype=torch.uint8)
    x_test = torch.load("dataset/formatted/x-test.pt").to(device=device, dtype=torch.uint8)
    return x_train, y_train, x_test

def remove_samples_without_labels(x_train: Tensor, y_train: Tensor) -> tuple[Tensor, Tensor]:
    has_labels_mask = y_train.amax((1, 2)) != y_train.amin((1, 2))
    return x_train[has_labels_mask], y_train[has_labels_mask]

def mk_dataset(verbose: bool=True):
    if not os.path.isdir("dataset"):
        if verbose:
            print("No directoty 'dataset' found, creating dataset...", end="")
        download_raw_dataset()
        format_dataset()
        if verbose:
            print("done")
    elif verbose:
        print("'dataset' directory already present not doing anythin, if you want to recreate it please delete the directory.")

def format_dataset():
    shutil.rmtree("dataset/formatted", ignore_errors=True)
    os.makedirs("dataset/formatted", exist_ok=True)
    format_imgs_into_pt_file(Path("dataset/raw/x-train"))
    format_imgs_into_pt_file(Path("dataset/raw/x-test"))
    y_train_np: pd.DataFrame = (
        pd.read_csv('dataset/raw/y-train.csv', index_col=0)
        .values
        .astype("uint8")
        .T
        .reshape(-1, 256, 256)
    )
    torch.save(
        torch.from_numpy(y_train_np),
        "dataset/formatted/y-train.pt"
    )

def format_imgs_into_pt_file(imgs_parent_dir: Path):
    tensor = load_imgs_as_tensor(imgs_parent_dir)
    target_file = Path(
        *imgs_parent_dir.parts[:-2],
        "formatted",
        imgs_parent_dir.parts[-1] + ".pt"
    )
    torch.save(tensor, target_file)

def load_imgs_as_tensor(imgs_parent_dir: Path) -> Tensor:
    imsge_files = list(sorted(
        imgs_parent_dir.glob("*.png"),
        key=lambda filename: int(filename.name.rstrip(".png"))
    ))
    imgs = torch.empty(
        len(imsge_files), 1, 256, 256,
        dtype=torch.uint8,
    )
    for img_idx, image_file in enumerate(imsge_files):
        imgs[img_idx] = torchvision.io.decode_image(image_file)
    return imgs

def download_raw_dataset():
    shutil.rmtree("path/to/directory", ignore_errors=True)
    raw_pth = Path("dataset/raw")
    raw_pth.mkdir(parents=True, exist_ok=True)
    wget_and_unzip(
        "https://challengedata.ens.fr/media/public/train-images.zip",
        "dataset/raw/x-train",
    )
    wget_and_unzip(
        "https://challengedata.ens.fr/media/public/test-images.zip",
        "dataset/raw/x-test",
    )
    wget_to_file(
        "https://challengedata.ens.fr/media/public/annotated_labels.json",
        "dataset/raw/annotated_labels.json",
    )
    wget_to_file(
        "https://challengedata.ens.fr/media/public/label_Hnl61pT.csv",
        "dataset/raw/y-train.csv",
    )

def wget_and_unzip(url: str, path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    zip_path = path.with_suffix(".zip")

    # Download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Unzip into the target path
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)

    # Remove zip file
    zip_path.unlink()

    # Detect the subfolder created by the ZIP
    subfolders = [p for p in path.iterdir() if p.is_dir()]
    if len(subfolders) == 1:
        sub = subfolders[0]
        # Move everything inside that subfolder up one level
        for item in sub.iterdir():
            item.rename(path / item.name)
        # Remove the now-empty subfolder
        sub.rmdir()

def wget_to_file(url: str, path: str):
    answer = requests.get(url)
    with open(path, "wb") as f:
        f.write(answer.content)