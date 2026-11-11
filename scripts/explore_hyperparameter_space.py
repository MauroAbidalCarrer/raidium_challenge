import os
import sys

import torch
import pandas as pd

import configs as cfg
from raidium_challenge.config_classes import (
    ModelConfig,
    TrainingConfig,
)
from raidium_challenge import dataset, training, models, metrics

BASELINE_TRAINING_CFG = TrainingConfig(
    batch_size = 32,
    n_epochs = 100,
    test_size = 0.2,
    target_metric = "validation/dice_score",
    target_objective = "max",
    dice_loss_weight = 1,
    max_loss_norm = 1000, 
    ckpt_dir = "artifacts/experiments/unet_baseline/"
)
WANDB_TAGS = [
    "unet",
    "segmentation",
    "experiment",
    "baseline",
]
N_TRIALS = 3
RECORDS_SAVE_DIR = "artifacts/experiments/unet_baseline/records/"

def main():
    for trial_i in range(N_TRIALS):
        if len(sys.argv) == 1:
            wandb_run_name = "baseline_experiment"
        else:
            wandb_run_name = sys.argv[1]
        wandb_run_name += "_" + str(trial_i)
        print("Starting trial", wandb_run_name)
        train_cfg = BASELINE_TRAINING_CFG
        model_cfg = cfg.MODELS_CFGS["unet"]
        model = models.mk_model_from_cfg(model_cfg)
        omptimizer_cfg = cfg.OPTIM_CFGS["unet"]
        optimizer = training.mk_optimizer(model, omptimizer_cfg)

        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
        criterion = metrics.SegmentationLoss(train_cfg)
        data_loaders = dataset.mk_segmentation_data_loaders(train_cfg)
        wandb_run = training.wandb_init(
            model_cfg,
            train_cfg,
            omptimizer_cfg,
            tags=WANDB_TAGS,
            group="manual_training",
            run_name=wandb_run_name,
        )
        trainer = training.Trainer(
            model,
            train_cfg, 
            optimizer,
            lr_scheduler,
            wandb_run,
        )
        records = trainer.train_model(data_loaders, criterion)
        df = pd.DataFrame.from_records(records)
        print(df)
        os.makedirs(RECORDS_SAVE_DIR, exist_ok=True)
        df.to_parquet(RECORDS_SAVE_DIR + wandb_run_name + ".parquet")

if __name__ == "__main__":
    main()
