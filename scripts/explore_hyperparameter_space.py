import os
import itertools 
from copy import deepcopy
from collections.abc import Iterable

import torch
import pandas as pd

import configs as cfg
from raidium_challenge.config_classes import (
    OptimizationConfig,
    TrainingConfig,
)
from raidium_challenge import dataset, training, models, metrics


BASELINE_OPTIMIZER_CFG = OptimizationConfig(
    starting_lr = 5e-4,
    beta0 = 0.9,
    beta1 = 0.999,
)
BASELINE_TRAINING_CFG = TrainingConfig(
    batch_size = 32,
    n_epochs = 2,
    test_size = 0.2,
    target_metric = "validation/dice_score",
    target_objective = "max",
    dice_loss_weight = 1,
    max_loss_norm = 1000, 
    ckpt_dir = "artifacts/unet_experiments/{experiment_name}/"
)
experiments_modifiers = {
    "baseline": {},
    "highest_impact_hps": {
        "training_cfg": {
            "batch_size": iter((1, 2, 4, 8, 16, 32, 128)),
        },
        "optimizer_cfg": {
            "starting_lr": iter((1e-4, 5e-4, 5e-3, 1e-2, 5e-2)),
        }
    },
}
WANDB_TAGS = [
    "unet",
    "segmentation",
    "experiment",
    "baseline",
]
N_TRIALS = 1
RECORDS_SAVE_DIR = "artifacts/experiments/unet_baseline/records/"


def main():
    for experiment_name, modifiers in experiments_modifiers.items():
        for train_cfg, optimizer_cfg in iter_over_cfgs(modifiers):
            for trial_i in range(N_TRIALS):
                wandb_run_name = experiment_name + "_" + str(trial_i)
                print("Starting trial", wandb_run_name)
                print("Configs:", train_cfg, optimizer_cfg)
                model_cfg = cfg.MODELS_CFGS["unet"]
                model = models.mk_model_from_cfg(model_cfg)
                optimizer = training.mk_optimizer(model, optimizer_cfg)
                lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
                criterion = metrics.SegmentationLoss(train_cfg)
                data_loaders = dataset.mk_segmentation_data_loaders(train_cfg)
                wandb_run = training.wandb_init(
                    model_cfg,
                    train_cfg,
                    optimizer_cfg,
                    tags=WANDB_TAGS,
                    group="manual_training",
                    name=wandb_run_name,
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
                os.makedirs(RECORDS_SAVE_DIR, exist_ok=True)
                df.to_parquet(RECORDS_SAVE_DIR + wandb_run_name + ".parquet")

def iter_over_cfgs(modifiers: dict) -> Iterable[tuple[TrainingConfig, OptimizationConfig]]:
    ACCEPTED_KEYS = ("training_cfg", "optimizer_cfg")
    for k in modifiers.keys():
        if k not in ACCEPTED_KEYS:
            raise ValueError(f"Key {k} in modifiers is not expected, should only be one of {ACCEPTED_KEYS}.")
    return itertools.product(
        iter_over_single_cfg(BASELINE_TRAINING_CFG, modifiers.get("training_cfg", {})),
        iter_over_single_cfg(BASELINE_OPTIMIZER_CFG, modifiers.get("optimizer_cfg", {})),
    )

def iter_over_single_cfg(base_cfg, modifiers: dict) -> Iterable:
    # print(modifiers)
    if len(modifiers) == 0:
        yield deepcopy(base_cfg)
    for k, v in modifiers.items():
        print("is Iterable:", isinstance(v, Iterable))
        if isinstance(v, Iterable):
            for v_element in v:
                cfg = deepcopy(base_cfg)
                if not hasattr(cfg, k):
                    raise ValueError(f"Erronous key '{k}' for config {cfg.__class__.__name__}.")
                setattr(cfg, k, v_element)
                # print("setting", k, "to", v_element, "\n", cfg)
                yield cfg
        else:
            cfg = deepcopy(base_cfg)
            if not hasattr(cfg, k):
                raise ValueError(f"Erronous key '{k}' for config {cfg.__class__.__name__}.")
            setattr(cfg, k, v)
            yield cfg
    

if __name__ == "__main__":
    main()
    # for experiment_name, modifiers in experiments_modifiers.items():
    #     print(experiment_name, ":")
    #     for train_cfg, optimizer_cfg in iter_over_cfgs(modifiers):
    #         print(train_cfg)
    #         print(optimizer_cfg)
    #         print()
