import os
import argparse

from pandas import DataFrame as DF
import pandas as pd


# PTH_TO_EXPERIMENTS = "artifacts/unet_experiments/{grp_name}/{run_name}/"


def main():
    read_all_experiments_records()

def read_all_experiments_records() -> DF:
    from hp_effects_EDA import EXP_SAVE_ROOT_DIR
    print(os.listdir(EXP_SAVE_ROOT_DIR))
    # for exp in os.listdir(EXP_SAVE_ROOT_DIR):
    
if __name__ == "__main__":
    main()