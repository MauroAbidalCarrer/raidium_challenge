import torch
from torch import nn

from src import configs as cfg
from src import dataset, training, models, metrics


def main():
    dataset.mk_dataset(verbose=False)
    train_cfg = cfg.TRAIN_CONFIGS["unet_training"]
    wandb_cfg = cfg.WandbConfig(["unet", "segmentation"], "unet")
    model_cfg = cfg.DFLT_MODELS_CFGS["unet"]
    model = models.mk_unet(model_cfg)
    optimizer = training.mk_optimizer(model, train_cfg)
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
    trainer.train_model(
        data_loaders,
        criterion,
        "checkpoints/unet/unet_epoch_{epoch}.pt",
    )


if __name__ == "__main__":
    main()