import os
import shutil
import zipfile
from pathlib import Path

import torch
import requests
import torchvision
import numpy as np
import pandas as pd
from torch import Tensor
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


train_transform = A.Compose(
    [
        A.Affine((0.5, 2), 0.2, fill=0),
        A.CoarseDropout(num_holes_range=[1, 5], fill=0, p=0.75),
    ],
    additional_targets={"mask": "mask"},
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    download_raw_dataset()
    format_dataset()

def get_data_loaders(
        x_train: Tensor,
        y_train: Tensor,
        x_valid: Tensor,
        y_valid: Tensor,
        batch_size: int,
    ):
    train_ds = SegmentationDataset(
        images=x_train,
        masks=y_train,
        transform=train_transform,
    )

    valid_ds = SegmentationDataset(
        images=x_valid,
        masks=y_valid,
        transform=None,   # no augmentation for validation
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_batch_size = int(np.clip(batch_size * 2, 1, 64))
    valid_loader = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False)

    return train_loader, valid_loader

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images: Tensor, masks: Tensor, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].cpu().numpy()  # Albumentations expects numpy HWC
        mask = self.masks[idx].cpu().numpy()

        # If tensor is CHW convert to HWC
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # back to torch
        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()
        return img, mask


def load_preprocessed_dataset() -> tuple[Tensor, Tensor, Tensor]:
    """
    - Loads formatted dataset 
    - Standardizes the x train and test by their combined mean/std 
      Since we have access to the x test we can "leak" its stats into the training.
    - Removes the train samples without labels

    Returns:
        tuple[Tensor, Tensor, Tensor]: x_train, y_train, x_test
    """
    y_train: Tensor = torch.load("dataset/formatted/y-train.pt")
    x_train = torch.load("dataset/formatted/x-train.pt").type(torch.float32)
    x_test = torch.load("dataset/formatted/x-test.pt").type(torch.float32)
    x_train_n_test = torch.cat((x_train, x_test))
    mean, std = x_train_n_test.mean(), x_train_n_test.std()
    x_train = (x_train - mean) / (std + 1e-8)
    x_test = (x_test - mean) / (std + 1e-8)
    x_train, y_train = remove_samples_without_labels(x_train, y_train)

    return x_train, y_train, x_test

def remove_samples_without_labels(x_train: Tensor, y_train: Tensor) -> tuple[Tensor, Tensor]:
    has_labels_mask = y_train.amax((1, 2)) != y_train.amin((1, 2))
    return x_train[has_labels_mask], y_train[has_labels_mask]

def load_imgs_as_tensor(imgs_parent_dir: Path) -> Tensor:
    imsge_files = list(sorted(
        imgs_parent_dir.glob("*.png"),
        key=lambda filename: int(filename.name.rstrip(".png"))
    ))
    imgs = torch.empty(
        len(imsge_files),
        1,
        256,
        256,
        dtype=torch.uint8,
    )
    for img_idx, image_file in enumerate(imsge_files):
        imgs[img_idx, 0] = torchvision.io.decode_image(image_file)[0]
    return imgs

def format_imgs_into_pt_file(imgs_parent_dir: Path):
    tensor = load_imgs_as_tensor(imgs_parent_dir)
    target_file = Path(
        *imgs_parent_dir.parts[:-2],
        "formatted",
        imgs_parent_dir.parts[-1] + ".pt"
    )
    torch.save(tensor, target_file)

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

def download_raw_dataset():
    shutil.rmtree("path/to/directory", ignore_errors=True)
    raw_pth = Path("dataset/raw")
    raw_pth.mkdir(parents=True, exist_ok=True)
    get_and_unzip(
        "https://challengedata.ens.fr/media/public/train-images.zip",
        "dataset/raw/x-train",
    )
    get_and_unzip(
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

def get_and_unzip(url: str, path: str | Path):
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

if __name__ == "__main__":
    main()