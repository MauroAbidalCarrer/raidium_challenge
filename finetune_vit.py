import os
import sys

import torch
from rich.traceback import install as install_rich_traceback

from src import configs as cfg
from src import dataset, training, models, metrics


def main():
    install_rich_traceback(width=300, extra_lines=1)
    if len(sys.argv) < 2:
        print("ERROR: Please provide path to pretrained MAE ViT checkpoint.")
        exit(1)
    chkpt = torch.load(sys.argv[1], weights_only=False)
    optim_cfg = cfg.OPTIM_CFGS["finetuning"]
    train_cfg = cfg.TRAIN_CONFIGS["finetuning"]
    model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
    model = models.mk_model_from_cfg(model_cfg)
    model.load_state_dict(chkpt["model"])
    optimizer = training.mk_optimizer(model, optim_cfg)

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    criterion = metrics.SegmentationLoss(train_cfg)
    data_loaders = dataset.mk_segmentation_data_loaders(train_cfg)
    wandb_run = training.wandb_init(
        model_cfg,
        train_cfg,
        optim_cfg,
        tags=cfg.WANDB_RUN_TAGS["finetuning"],
        group="manual_training",
    )
    print(wandb_run.name)
    trainer = training.Trainer(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
        wandb_run,
    )
    models.print_params_count(model)
    trainer.train_model(
        data_loaders,
        criterion,
        "checkpoints/finetuning/down_scaled_vit/{wandb_run_name}/pretrained_vit_epoch_{epoch}.pt",
    )


if __name__ == "__main__":
    main()