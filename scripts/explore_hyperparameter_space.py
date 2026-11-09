import sys

import torch

import configs as cfg
from raidium_challenge.config_classes import (
    ModelfConfig,
    TrainingConfig,
)
from raidium_challenge import dataset, training, models, metrics

BASELINE_TRAINING_CFG = TrainingConfig(
    n_epochs = 100,
    test_size = 0.2,
    target_metric = "validation/dice_score",
    target_obejctive = "max",
    dice_loss_weight = 1,
    max_loss_norm = 1000, 
    chpt_dir = "artifacts/experiments/unet_baseline/"
)

def main():
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
        tags=cfg.WANDB_RUN_TAGS["unet"],
        group="manual_training",
    )
    trainer = training.Trainer(
        model,
        train_cfg, 
        optimizer,
        lr_scheduler,
        wandb_run,
    )
    trainer.train_model(data_loaders, criterion)

#def mk_training_conf() -> TrainingConfig:
#    return TrainingConfig(
#        batch_size
#    )

if __name__ == "__main__":
    main()
