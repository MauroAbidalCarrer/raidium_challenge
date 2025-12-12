import sys

import torch
from torch import nn

from src import configs as cfg
from src import dataset, training, models, metrics


def main():
    dataset.mk_dataset(verbose=False)
    wandb_cfg = cfg.WandbConfig(["unet", "segmentation"], "unet")
    if len(sys.argv) > 1:
        chkpt = torch.load(sys.argv[1], weights_only=False)
        train_cfg = cfg.TrainingConfig(**chkpt["train_cfg"])
        model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
        model = models.mk_unet(model_cfg)
        model.load_state_dict(chkpt["model"])
        model.cfg = model_cfg
        omptimizer_cfg = cfg.OPTIMIZER_CFGS["unet"]
        optimizer = training.mk_optimizer(model, omptimizer_cfg)
        optimizer.load_state_dict(chkpt["optimizer"])
    else:
        train_cfg = cfg.TRAIN_CONFIGS["unet_training"]
        model_cfg = cfg.DFLT_MODELS_CFGS["unet"]
        model = models.mk_unet(model_cfg)
        omptimizer_cfg = cfg.OPTIMIZER_CFGS["unet"]
        optimizer = training.mk_optimizer(model, omptimizer_cfg)
    wandb_cfg.tags = wandb_cfg.tags + ["from_checkpoint"]

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    criterion = metrics.SegmentationLoss(train_cfg)
    data_loaders = dataset.mk_data_loaders_for_segmentation(train_cfg)
    trainer = training.Trainer(
        model,
        train_cfg, 
        optimizer,
        lr_scheduler,
        wandb_cfg
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
        "checkpoints/unet/unet_epoch_{epoch}.pt",
    )


if __name__ == "__main__":
    main()