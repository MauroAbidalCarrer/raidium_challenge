import shutil
import zipfile
from pathlib import Path

import requests


def mk_raw_dataset():
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
    get_and_write_to(
        "https://challengedata.ens.fr/media/public/annotated_labels.json",
        "dataset/raw/annotated_labels.json",
    )
    get_and_write_to(
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

def get_and_write_to(url: str, path: str):
    answer = requests.get(url)
    with open(path, "wb") as f:
        f.write(answer.content)

if __name__ == "__main__":
    mk_raw_dataset()