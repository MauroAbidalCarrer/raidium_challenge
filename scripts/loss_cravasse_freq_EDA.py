import os
import argparse

from pandas import DataFrame as DF
import pandas as pd

from hp_effects_EDA import EXP_SAVE_ROOT_DIR


EXP_NAME = "cls_wise_dice_score"


def main():
    records = read_all_experiments_records()
    cls_dice_scores = records.filter(
        regex="training/class*",
        axis="columns",
    )
    print(cls_dice_scores.groupby(level=0).corr())

def read_all_experiments_records() -> DF:
    EXP_DIR = EXP_SAVE_ROOT_DIR + EXP_NAME + "/"
    save_dirs_it = os.listdir(EXP_DIR)
    records_pth_it = map(lambda pth: EXP_DIR + pth + "/records.parquet", save_dirs_it)
    records_it = map(tmp_read_parquet, records_pth_it)
    records = pd.concat(records_it, keys=save_dirs_it)
    records.index.set_names(("exp", "epoch"), inplace=True)
    return records

def tmp_read_parquet(pth: str) -> DF:
    if os.path.exists(pth):
        return pd.read_parquet(pth)
    else:
        print(f"WARNING: Skipping {pth} bc it doesn't exist make sure this is normal.")
        return DF()

if __name__ == "__main__":
    main()
