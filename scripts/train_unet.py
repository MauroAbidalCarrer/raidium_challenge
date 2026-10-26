import sys

import torch

from src import configs as cfg
from src import dataset, training, models, metrics


def main():
    if len(sys.argv) > 1:
        chkpt = torch.load(sys.argv[1], weights_only=False)
        train_cfg = cfg.TrainingConfig(**chkpt["train_cfg"])
        model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
        model = models.mk_model_from_cfg(model_cfg)
        model.load_state_dict(chkpt["model"])
        model.cfg = model_cfg
        omptimizer_cfg = cfg.OPTIM_CFGS["unet"]
        optimizer = training.mk_optimizer(model, omptimizer_cfg)
        optimizer.load_state_dict(chkpt["optimizer"])
    else:
        train_cfg = cfg.TRAIN_CONFIGS["unet_training"]
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
    if len(sys.argv) > 1:
        print("setting epoch, step and training samples seen")
        n_btaches_per_epoch = len(data_loaders["train"])
        samples_seen = chkpt["training_samples_seen"]
        trainer.training_samples_seen = samples_seen
        n_samples_per_epoch = n_btaches_per_epoch * train_cfg.batch_size
        trainer.epoch = chkpt.get("epoch", int(samples_seen // n_samples_per_epoch))
        trainer.step = chkpt.get("step", int(samples_seen // train_cfg.batch_size))
    trainer.train_model(
        data_loaders,
        criterion,
        "checkpoints/unet/{wandb_run_name}/unet_epoch_{epoch}.pt",
    )


if __name__ == "__main__":
    main()